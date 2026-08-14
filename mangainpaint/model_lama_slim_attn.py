"""
Axis A7-attn: `LamaSlimG` with its FFC-ResNet bottleneck stack REPLACED by
real windowed self-attention (Swin-style), not added alongside it.

Motivation: a reviewer-anticipation question -- LaMa is claimed decisive
but never compared/fused against modern attention-based inpainting (MAT,
ZITS, TFill -- real learned QKV self-attention). This project's existing
"attention" experiments (`model_attn.py`'s contextual attention,
`model_linattn.py`'s linear attention) are NOT that -- they're a
patch-copy mechanism and a linear-complexity kernel approximation,
genuinely different from a transformer block. This file is the real thing.

**Why FFC is fully removed, not kept alongside attention.** This project
already ran this exact confound once: Axis A4 (`model_attn.py`
`MangaFillNetAttn`) added contextual attention *alongside* FFC, with FFC
upstream -- rejected specifically because the attention branch only ever
saw FFC's already-spectrally-mixed features, so no result was attributable
to attention alone. Axis A5 (`model_attn.py` `MangaFillNetAttnNoFFC`) fixed
this by removing FFC entirely. Repeating A4's mistake with a transformer
block here would waste a training run on an uninterpretable result.

Unlike `model_attn.py`'s DilRes-based backbone (where FFC is a separable
add-on), `LamaSlimG`'s entire bottleneck IS an `FFCResnetBlock` stack --
there's no non-FFC fallback to strip it back to. "Removing FFC" here means
replacing the whole bottleneck, which does cost `LamaSlimG`'s one domain
argument (spectral mixing as a natural basis for periodic screentone) --
the same trade Axis A5 already made deliberately, for the same reason
(clean isolation over architectural purity).

**Why windowed, not global, self-attention.** The bottleneck at this
project's fixed 384px input / ngf=32 / 3 downsamples is 48x48 = 2,304
tokens. Full global self-attention there costs ~1.36GB per layer just for
the fp32 attention-probability matrix (batch=8, 8 heads) -- prohibitive
stacked across 12 layers alongside a discriminator and GAN losses on a
single mid-range GPU.
This isn't a compromise for tractability alone: MAT and ZITS, the exact
papers a reviewer would cite, don't use brute-force global self-attention
at this resolution either -- they use windowed/local attention for the
same reason. So windowed attention (Swin, Liu et al. ICCV 2021) is the
MORE faithful comparison, not a cheaper stand-in.

**Design**: same conv stem shape/channel budget as `LamaSlimG`
(ngf=32, n_downsampling=3 -> 256ch @ 48x48 bottleneck, matching exactly so
this stays pluggable into `mangainpaint/distill.py`'s existing adapter/hint
plumbing without rework if a later fusion experiment wants it) but built
with PLAIN Conv2d/BatchNorm2d instead of FFC_BN_ACT -- confirmed via
`BIG_LAMA_GEN_CFG` that all but the last downsample layer already have
ratio_gin=ratio_gout=0 in the real LaMa config (i.e. FFC_BN_ACT there
already reduces to a plain conv, zero global-path channels), so this is a
faithful "same shape, spectral mixing removed" swap, not a redesign of the
parts that were never doing anything spectral. Bottleneck: a learned
absolute position embedding (ViT-style, not Swin's per-layer relative
position bias -- simpler, still standard, an explicit scoping choice) plus
12 Swin blocks alternating regular/shifted 8x8 windows (window=8 divides
48 evenly; shift=4 is the standard half-window shift). Output head reused
directly from `mangainpaint/model_scratch.py`'s `OutHead` (conv+InstanceNorm+
LeakyReLU, spectral-norm conv, tanh) -- this model has no pretrained
initialization to preserve (sliced-init already measured useless, see
`model_lama_slim.py`), so there's no reason to keep LaMa's 3-channel-then-
mean output convention; single-channel tanh matches every from-scratch
generator in this codebase.

**Mask-awareness, deliberately absent**: self-attention here pools over
ALL positions (hole and valid) uniformly, with no explicit valid-region
sourcing restriction (unlike `ContextualAttentionBlock`, which explicitly
restricts copy *sources* to valid patches -- a documented gotcha in this
project when a sibling module skipped it). This is intentional parity with
what's being replaced: `FFCResnetBlock`'s own global Fourier mixing pools
the whole spatial extent unconditionally too (FFT has no notion of
hole/valid), so this is matching the masking discipline of the block being
swapped out, not overlooking it.

Interface: `forward(x)` with x = [B,2,H,W] (masked grayscale + mask),
output [B,1,H,W] in [-1,1] -- drop-in with mangainpaint/trainer.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mangainpaint.model_scratch import OutHead

IMAGE_SIZE = 384  # fixed project-wide assumption (every generator in this
                   # codebase hardcodes this via its downsampling math)


def window_partition(x, window_size):
    """x: (B,H,W,C) -> (num_windows*B, window_size, window_size, C)."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows, window_size, H, W):
    """Inverse of window_partition."""
    B = windows.shape[0] // ((H // window_size) * (W // window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


def _shift_attn_mask(H, W, window_size, shift_size):
    """Standard Swin shifted-window attention mask: prevents a window
    (after the cyclic roll) from mixing tokens that weren't spatially
    adjacent before the shift. Returns (num_windows, N, N) additive mask
    (0 where allowed, -100 where forbidden)."""
    img_mask = torch.zeros(1, H, W, 1)
    h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1
    mask_windows = window_partition(img_mask, window_size).view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        """x: (num_windows*B, N, C). mask: (num_windows, N, N) or None."""
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(out)


class SwinBlock(nn.Module):
    """Pre-norm windowed-self-attention transformer block. `shift_size=0`
    -> regular (non-overlapping) windows; `shift_size=window_size//2` ->
    cyclically shifted windows (Swin's cross-window mixing mechanism,
    Liu et al. ICCV 2021). H/W fixed at construction time (this project's
    image_size is a global constant) so the shift mask is precomputed once
    as a buffer, not recomputed per forward call."""

    def __init__(self, dim, num_heads, H, W, window_size=8, shift_size=0, mlp_ratio=2.0):
        super().__init__()
        assert H % window_size == 0 and W % window_size == 0
        self.window_size = window_size
        self.shift_size = shift_size
        self.H, self.W = H, W
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        if shift_size > 0:
            self.register_buffer("attn_mask", _shift_attn_mask(H, W, window_size, shift_size),
                                 persistent=False)
        else:
            self.attn_mask = None

    def forward(self, x):
        """x: (B,C,H,W) -> (B,C,H,W)."""
        B, C, H, W = x.shape
        x_bhwc = x.permute(0, 2, 3, 1)
        xn = self.norm1(x_bhwc)

        if self.shift_size > 0:
            xn = torch.roll(xn, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        windows = window_partition(xn, self.window_size).view(-1, self.window_size ** 2, C)
        attn_out = self.attn(windows, mask=self.attn_mask)
        attn_out = attn_out.view(-1, self.window_size, self.window_size, C)
        xn = window_reverse(attn_out, self.window_size, H, W)

        if self.shift_size > 0:
            xn = torch.roll(xn, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x_bhwc = x_bhwc + xn
        x_bhwc = x_bhwc + self.mlp(self.norm2(x_bhwc))
        return x_bhwc.permute(0, 3, 1, 2)


class LamaSlimAttnG(nn.Module):
    def __init__(self, in_ch=2, ngf=32, n_downsampling=3, n_blocks=12,
                 window_size=8, num_heads=8, mlp_ratio=2.0,
                 expose_bottleneck=False):
        super().__init__()
        self.expose_bottleneck = expose_bottleneck

        # Plain conv stem, same channel progression as LamaSlimG's FFC stem
        # (2 -> 32 -> 64 -> 128 -> 256), confirmed a faithful match since
        # the real LaMa config's FFC_BN_ACT there already runs at
        # ratio_gin=ratio_gout=0 (i.e. zero global-path channels, a plain
        # conv in every way that matters) for all but the transition into
        # the resnet stack -- which is exactly the spectral mixing this
        # file removes.
        ch = ngf
        stem = [nn.ReflectionPad2d(3), nn.Conv2d(in_ch, ch, 7, 1, 0),
               nn.BatchNorm2d(ch), nn.ReLU(inplace=True)]
        for _ in range(n_downsampling):
            stem += [nn.Conv2d(ch, ch * 2, 3, 2, 1), nn.BatchNorm2d(ch * 2), nn.ReLU(inplace=True)]
            ch *= 2
        self.stem = nn.Sequential(*stem)
        self.bneck_ch = ch  # 256 @ ngf=32, n_downsampling=3

        H = W = IMAGE_SIZE // (2 ** n_downsampling)  # 48
        self.pos_embed = nn.Parameter(torch.zeros(1, self.bneck_ch, H, W))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        blocks = []
        for i in range(n_blocks):
            shift = 0 if i % 2 == 0 else window_size // 2
            blocks.append(SwinBlock(self.bneck_ch, num_heads, H, W,
                                    window_size=window_size, shift_size=shift,
                                    mlp_ratio=mlp_ratio))
        self.blocks = nn.ModuleList(blocks)

        up = []
        ch = self.bneck_ch
        for _ in range(n_downsampling):
            up += [nn.ConvTranspose2d(ch, ch // 2, 3, 2, 1, output_padding=1),
                  nn.BatchNorm2d(ch // 2), nn.ReLU(inplace=True)]
            ch //= 2
        self.up = nn.Sequential(*up)
        self.head = OutHead(ch)

        self.last_bottleneck = None
        if expose_bottleneck:
            self.distill_adapter = nn.Conv2d(self.bneck_ch, 512, 1)  # matches LamaSlimG's teacher-width convention

    def forward(self, x):
        h = self.stem(x)
        h = h + self.pos_embed
        for blk in self.blocks:
            h = blk(h)
        if self.expose_bottleneck:
            self.last_bottleneck = h
        h = self.up(h)
        return self.head(h)
