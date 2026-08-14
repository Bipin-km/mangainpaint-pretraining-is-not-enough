"""
Axis A4 (contextual attention branch): MangaFillNet with an added
Contextual Attention module (Yu et al., CVPR 2018, "Generative Image
Inpainting with Contextual Attention") running in parallel with the
existing dilated-residual bottleneck, merged before decoding.

Motivation: both vanilla FFC and its UFFC variant leave a fixed,
content-independent periodic texture in every hole fill -- this is a
structural property of routing the bottleneck through a *global*
frequency-domain operator (any single erroneous frequency bin rings
identically across the whole image regardless of content; UFFC suppresses
vanilla FFC's specific ringing pattern but produces its own, still
content-independent, replacement pattern -- learned gates confirm the
correction is genuinely engaged, not inert). Contextual attention has no
such global transform: each hole location is reconstructed as a
similarity-weighted average of *actual* valid-region feature patches --
literally copying real texture rather than synthesizing it through a
learned operator that can imprint a fixed bias. Of the from-scratch
architecture variants explored (see paper §6.3), this is the one that
doesn't share FFC/UFFC's structural failure mode.

Reused unmodified from `model_scratch.py`: Enc/DilRes/Dec/OutHead, and
`FFCBlock` (kept as-is here, not the UFFC variant, so this test isolates
the attention addition against the plain-FFC baseline rather than UFFC).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mangainpaint.model_scratch import Enc, DilRes, Dec, OutHead, FFCBlock
from mangainpaint.losses import sobel_mag


class ContextualAttentionBlock(nn.Module):
    """Adapted from Yu et al., CVPR 2018. For every spatial location,
    finds the most similar patch(es) among *valid* (non-hole) background
    patches (cosine similarity via a conv-as-correlation trick) and
    reconstructs that location as their softmax-weighted average --
    restricted to copy only from genuinely valid content, unlike a
    learned global transform (FFC/UFFC) that has no such restriction and
    can imprint a fixed bias everywhere regardless of image content.

    Effectively parameter-free (no learned weights of its own -- the
    "kernel" at every step is the input features themselves), unlike
    FFC/UFFC's per-position/per-frequency learned corrections.

    Simplified vs. the original paper: no multi-scale patches. The
    attention-map spatial-propagation/consistency step (the paper's
    left-right + top-down smoothing conv, meant to reduce patchy seams in
    the copied texture) was originally dropped here too -- added back
    (see `_fuse` below) after an early run without it showed a fixed,
    content-independent honeycomb-lattice artifact, and a dilation-
    schedule ablation ruled out `DilRes` as the cause: changing its
    dilation schedule made no visible difference, pointing back at this
    block's own missing consistency step as the likely source
    (independent per-pixel patch-matching is a classic cause of
    tiled/seamed copies in patch-based texture synthesis generally).

    Per-sample loop (not batched): each sample's background patches
    differ, so they can't share a single conv call the way a normal
    conv's fixed weight can -- standard for this operation, matches how
    reference implementations of contextual attention are written.
    """
    def __init__(self, patch_size=3, softmax_scale=10.0, fuse_k=3, use_fuse=True):
        super().__init__()
        self.patch_size = patch_size
        self.softmax_scale = softmax_scale
        self.fuse_k = fuse_k
        # Toggle kept (not just a permanent hardwire) so checkpoints trained
        # before this consistency step was added can still be loaded and run
        # in their original configuration for diagnostic comparisons --
        # those checkpoints' weights were never trained with this step, so
        # applying it post-hoc would misrepresent what was actually reported
        # for that run.
        self.use_fuse = use_fuse
        self._divisor_cache = None
        self._divisor_hw = None
        # Fixed (non-learned) diagonal-identity kernel for the consistency
        # fuse below -- matches this class's "effectively parameter-free"
        # design (no learned weights of its own).
        self.register_buffer("fuse_weight", torch.eye(fuse_k).view(1, 1, fuse_k, fuse_k))

    def _overlap_divisor(self, H, W, device, dtype):
        """Reconstruction (see forward) scatters overlapping weighted
        patches back onto the HxW grid via conv_transpose2d; interior
        pixels receive patch_size^2 overlapping contributions and must
        be divided down, border pixels fewer. This divisor depends only
        on (H, W, patch_size) -- not on any batch content -- so it's
        computed once and cached (standard unfold/fold roundtrip-on-ones
        trick for counting overlaps)."""
        if (self._divisor_hw != (H, W) or self._divisor_cache is None
                or self._divisor_cache.device != device):
            p, pad = self.patch_size, self.patch_size // 2
            ones = torch.ones(1, 1, H, W, device=device, dtype=dtype)
            unf = F.unfold(ones, kernel_size=p, padding=pad, stride=1)
            div = F.fold(unf, output_size=(H, W), kernel_size=p, padding=pad, stride=1)
            self._divisor_cache = div.clamp_min(1.0)
            self._divisor_hw = (H, W)
        return self._divisor_cache

    def _fuse(self, sim, H, W):
        """Yu et al. 2018's spatial-consistency propagation ("left-right +
        top-down" smoothing, see class docstring). `sim` is (L, H, W) =
        (background-patch-index, query-row, query-col); since patches are
        extracted at every location with stride 1, the background grid and
        query grid are the same HxW, so L == H*W.

        Treats (bg-index, query-index) as the two axes of a single 2D
        "image" of shape (L, L) and runs a small diagonal-identity conv
        over it: this sums scores along the local diagonal where
        Δ(bg-index) ≈ Δ(query-index), i.e. rewards a query pixel and its
        neighbor for preferring background patches that are *also*
        neighbors -- spatially coherent copying instead of each pixel
        matching independently, the latter being a classic cause of
        patchy/tiled seams in patch-based texture synthesis.

        Two passes with a transpose in between (matching the reference
        algorithm): the first pass's diagonal is dominated by column
        (left-right) consistency since the flat index changes fastest
        along W; transposing H<->W before the second pass makes its
        diagonal dominated by row (top-down) consistency instead.

        Applied to the *raw* (pre-mask) similarity scores -- the hole
        mask's -1e4 fill happens after this, in `forward`, so it isn't
        diluted by this step's local summation.
        """
        k, pad = self.fuse_k, self.fuse_k // 2
        L = H * W
        x = sim.view(1, 1, L, L)
        x = F.conv2d(x, self.fuse_weight, padding=pad)
        x = x.view(H, W, H, W).permute(1, 0, 3, 2).contiguous()
        x = x.view(1, 1, L, L)
        x = F.conv2d(x, self.fuse_weight, padding=pad)
        x = x.view(W, H, W, H).permute(1, 0, 3, 2).contiguous()
        return x.view(L, H, W)

    def forward(self, x, mask):
        """x: (B,C,H,W) bottleneck features. mask: (B,1,Hi,Wi) original-res
        hole mask (1=hole); downsampled to x's resolution here."""
        B, C, H, W = x.shape
        p, pad = self.patch_size, self.patch_size // 2
        m = F.interpolate(mask.float(), size=(H, W), mode='nearest')  # (B,1,H,W)

        # FFT-adjacent numerical-stability pattern already used elsewhere in
        # this codebase (FFCBlock._freq, regional_stats_loss): force fp32 for
        # the whole patch-similarity/softmax/reconstruction chain rather than
        # risk fp16 overflow in the scaled-softmax exponentials.
        with torch.amp.autocast(x.device.type, enabled=False):
            x32 = x.float()
            divisor = self._overlap_divisor(H, W, x.device, torch.float32)
            outs = []
            for b in range(B):
                xb = x32[b:b + 1]                        # (1,C,H,W)
                valid_b = (m[b, 0].reshape(-1) < 0.5)     # (L,) True = valid bg location

                patches = F.unfold(xb, kernel_size=p, padding=pad, stride=1)  # (1, C*p*p, L)
                L = patches.shape[-1]
                patches = patches.view(C, p * p, L).permute(2, 0, 1).contiguous()  # (L, C, p*p)
                patches = patches.view(L, C, p, p)

                norm = patches.reshape(L, -1).norm(dim=1).clamp_min(1e-4).view(L, 1, 1, 1)
                patches_n = patches / norm

                sim = F.conv2d(xb, patches_n, padding=pad).squeeze(0)  # (L, H, W)
                if self.use_fuse:
                    sim = self._fuse(sim, H, W)
                sim = sim.masked_fill(~valid_b.view(L, 1, 1), -1e4)

                attn = F.softmax(sim.view(L, -1) * self.softmax_scale, dim=0).view(1, L, H, W)

                recon = F.conv_transpose2d(attn, patches, padding=pad)  # (1, C, H, W)
                recon = recon / divisor
                outs.append(recon)

            out = torch.cat(outs, dim=0)
        return out.to(x.dtype)


class NoiseInjection(nn.Module):
    """StyleGAN2-style per-channel learned noise injection (Karras et al.
    2020): a single-channel spatial noise map, broadcast across channels
    and scaled by a learned per-channel weight initialized at zero, so a
    freshly warm-started model is byte-identical to the un-noised one at
    step 0 -- a true ablation, not just "close." Motivation: the msxie92
    SIGGRAPH-2021 manga-inpainting reference (`SemanticInpaintingModel.
    forward`, `models.py:59`) injects explicit `randn` noise into its
    generator input; deterministic one-shot regression has no "smooth"
    category to collapse to on bitonal manga, so it commits to a fixed
    generic pattern instead -- real stochastic generation capacity is one
    axis never varied across the architecture/loss variants explored here
    (see paper §6.3)."""
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
        return x + self.weight * noise


class EdgeHint(nn.Module):
    """Zero-initialized additive side path injecting an explicit Sobel
    edge-magnitude hint into the encoder's first-stage features -- the
    generator-side counterpart to other msxie92-motivated ideas explored
    here (noise injection above, and a self-referential edge-conditioned
    discriminator, both rejected). Distinct from both: this doesn't add
    stochastic capacity or change what the discriminator sees, it gives G
    itself explicit, crisper access to real nearby line structure (computed
    from the *masked* input, so it's real edges outside the hole and ~zero
    inside it -- no GT leakage), the "structure/texture disentanglement"
    ingredient from msxie92, scoped down from their real ScreenVAE-based
    approach to a cheap, no-new-dependency Sobel proxy (`mangainpaint/
    losses.py`'s `sobel_mag`, already used for the existing `p2_w_edge`
    loss -- reused here as an *input* signal, not just a loss target). Same
    zero-init warm-start-safe pattern as `NoiseInjection`/`model_projected_d.
    py`'s `edge_proj`: an old checkpoint loads with only this module's own
    keys missing, byte-identical behavior at step 0."""
    def __init__(self, out_ch):
        super().__init__()
        self.proj = nn.Conv2d(1, out_ch, 3, 1, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, edge):
        return self.proj(edge)


class MangaFillNetAttnNoFFC(nn.Module):
    """Axis A5: the clean test of the contextual-attention hypothesis.

    Identical to `MangaFillNetAttn` except `f1`/`f2` (`FFCBlock`) are
    removed entirely -- the bottleneck is dilated-residual (`r1..r8`) +
    `ContextualAttentionBlock` only, with zero frequency-domain ops
    anywhere in the network. `MangaFillNetAttn` still showed the same
    fixed, content-independent periodic artifact as vanilla FFC/UFFC, just
    reshaped again -- traced to `f1`/`f2` still running upstream of both
    branches, so the attention branch's own residual/patch-bank input
    already carried the FFC-imprinted bias. This class is the actual test
    of the original claim: does contextual attention alone (no global
    frequency transform anywhere) avoid that structural bias, not just add
    capacity on top of a network that still has it.
    """
    def __init__(self, in_ch=2, base=32, ratio_g=0.5, dilations=(1, 2, 4, 8), fuse_k=3, use_fuse=True,
                 use_noise=False, use_edge_hint=False):
        super().__init__()
        b = base
        self.e1 = Enc(in_ch, b)
        self.e2 = Enc(b, b * 2)
        self.e3 = Enc(b * 2, b * 4)
        bch = b * 4

        # `dilations` default (1,2,4,8) shares a common factor of 2, the
        # textbook Hybrid-Dilated-Convolution "gridding" condition (Wang et
        # al. 2018) suspected of causing the honeycomb-lattice artifact
        # discussed above. Conv weight shapes are independent of dilation
        # (`nn.Conv2d(ch, ch, 3, ..., dilation=d)`'s kernel is always 3x3),
        # so any schedule here stays checkpoint-compatible for a
        # warm-started fine-tune test.
        d0, d1_, d2_, d3_ = dilations
        self.r1 = DilRes(bch, d0)
        self.r2 = DilRes(bch, d1_)
        self.r4 = DilRes(bch, d2_)
        self.r8 = DilRes(bch, d3_)
        self.attn = ContextualAttentionBlock(patch_size=3, softmax_scale=10.0, fuse_k=fuse_k, use_fuse=use_fuse)
        self.merge = nn.Conv2d(bch * 2, bch, 1)
        self.use_noise = use_noise
        # Only registered when actually used: an unconditionally-created
        # NoiseInjection whose forward call is merely skipped by the
        # `use_noise` flag leaves `noise.weight` permanently absent from the
        # autograd graph on every `use_noise=False` run -- harmless without
        # DDP, but a real crash under it (`RuntimeError: Expected to have
        # finished reduction...`): DDP's gradient bucketing expects every
        # registered parameter to participate in every backward pass unless
        # told otherwise. `use_noise=True` runs are unaffected either way.
        if use_noise:
            self.noise = NoiseInjection(bch)

        self.use_edge_hint = use_edge_hint
        # Same conditional-registration discipline as `noise` above: only
        # ever created when actually used. Channel count matches the
        # bottleneck (`bch`), not `b` -- see `forward`'s docstring note:
        # this now injects at the bottleneck, after `merge`, not right
        # after `e1`.
        if use_edge_hint:
            self.edge_hint = EdgeHint(bch)

        self.d1 = Dec(bch, b * 4, b * 2, hid_mult=4)
        self.d2 = Dec(b * 2, b * 2, b, hid_mult=4)
        self.d3 = Dec(b, b, b * 2, hid_mult=8)

        self.head = OutHead(b * 2)

    def forward(self, x):
        mask = x[:, 1:2]
        if self.use_edge_hint:
            # Computed from the *masked* input (channel 0): real edges
            # outside the hole, ~zero inside it (the masked region is a
            # flat fill) -- no GT leakage, same masking discipline as the
            # rest of the network's own inputs. Injected at the bottleneck
            # (below, after `merge`) rather than right after `e1` -- an
            # earlier attempt injecting post-`e1` was a genuine null even
            # forced to a substantial weight, most likely diluted by two
            # more downsampling conv stages before reaching the
            # bottleneck/decoder where the periodic-artifact mechanism
            # actually operates. Kept at input resolution here and
            # downsampled to bottleneck size below, same as the mask.
            edge = torch.tanh(sobel_mag(x[:, 0:1]))
        x, s1 = self.e1(x)
        x, s2 = self.e2(x)
        x, s3 = self.e3(x)

        dilres = self.r1(x, mask)
        dilres = self.r2(dilres, mask)
        dilres = self.r4(dilres, mask)
        dilres = self.r8(dilres, mask)

        attn_out = self.attn(x, mask) + x

        x = F.leaky_relu(self.merge(torch.cat([dilres, attn_out], dim=1)), 0.2, inplace=True)
        if self.use_edge_hint:
            edge_ds = F.interpolate(edge, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = x + self.edge_hint(edge_ds)
        if self.use_noise:
            x = self.noise(x)

        x = self.d1(x, s3, mask)
        x = self.d2(x, s2, mask)
        x = self.d3(x, s1, mask)
        return self.head(x)


class MangaFillNetAttn(nn.Module):
    """Same architecture as `model_scratch.MangaFillNet`, with an added
    `ContextualAttentionBlock` running in *parallel* with the existing
    dilated-residual stack (r1..r8), merged by a 1x1 conv before
    decoding -- mirrors the original paper's two-branch refinement
    design (dilated-conv branch for structure + attention branch for
    texture copying) rather than serializing the new block into the
    existing stack."""
    def __init__(self, in_ch=2, base=32, ratio_g=0.5):
        super().__init__()
        b = base
        self.e1 = Enc(in_ch, b)
        self.e2 = Enc(b, b * 2)
        self.e3 = Enc(b * 2, b * 4)
        bch = b * 4

        self.f1 = FFCBlock(bch, ratio_g)
        self.f2 = FFCBlock(bch, ratio_g)
        self.r1 = DilRes(bch, 1)
        self.r2 = DilRes(bch, 2)
        self.r4 = DilRes(bch, 4)
        self.r8 = DilRes(bch, 8)
        self.attn = ContextualAttentionBlock(patch_size=3, softmax_scale=10.0)
        self.merge = nn.Conv2d(bch * 2, bch, 1)

        self.d1 = Dec(bch, b * 4, b * 2, hid_mult=4)
        self.d2 = Dec(b * 2, b * 2, b, hid_mult=4)
        self.d3 = Dec(b, b, b * 2, hid_mult=8)

        self.head = OutHead(b * 2)

    def forward(self, x):
        mask = x[:, 1:2]
        x, s1 = self.e1(x)
        x, s2 = self.e2(x)
        x, s3 = self.e3(x)
        x = self.f1(x) + x
        x = self.f2(x) + x

        dilres = self.r1(x, mask)
        dilres = self.r2(dilres, mask)
        dilres = self.r4(dilres, mask)
        dilres = self.r8(dilres, mask)

        attn_out = self.attn(x, mask) + x  # residual, matches f1/f2's convention

        x = F.leaky_relu(self.merge(torch.cat([dilres, attn_out], dim=1)), 0.2, inplace=True)

        x = self.d1(x, s3, mask)
        x = self.d2(x, s2, mask)
        x = self.d3(x, s1, mask)
        return self.head(x)
