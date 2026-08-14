"""
Projected-GAN discriminator (Sauer et al., NeurIPS 2021, arXiv:2111.01007).

Replaces the from-scratch PatchD (`mangainpaint/model_scratch.SingleScaleD`)
used in earlier runs. The from-scratch PatchD is confirmed to be the likely
bottleneck: D-loss collapsing to ~0 / D-dominating flags across runs, on a
small (6.8k image) dataset with a bimodal {ink, paper} pixel distribution
that a from-scratch D can trivially overfit. Routing real/fake images through
a *frozen*, ImageNet-pretrained feature backbone before the discriminator
head means D can't overfit the raw pixel statistics — it has to work in a
fixed, generic feature space instead.

Simplification vs. the original paper/reference implementation
(`autonomousvision/projected-gan`): the reference code mixes two backbones
(EfficientNet + a ViT via `timm`). This uses a single torchvision
EfficientNet-B0 (no extra dependency, smaller compute footprint — relevant
for keeping the training affordable on modest single-GPU hardware and for
keeping the wider from-scratch experiment matrix, see paper §6.3, tractable
on shared cloud GPUs). Follow-up reproductions of Projected GAN commonly use
a single EfficientNet backbone successfully; this is a scope trade-off, not
a correctness shortcut.

Architecture, per the paper's CCM (cross-channel mixing) + CSM (cross-scale
mixing) recipe:
  1. Frozen EfficientNet-B0 (torchvision, ImageNet-pretrained) — feature
     extractor only, truncated after the deepest stage we use. BatchNorm
     stays in eval() permanently regardless of the owning D's train()/eval()
     calls, since frozen running stats (not batch stats) are the correct
     behavior for a frozen feature extractor.
  2. CCM: a *fixed, randomly initialized, non-trainable* 1x1 conv per stage,
     projecting each stage's channel count down to a common width. The
     paper's finding (and the reason this is cheap) is that random
     projections work about as well as learned ones here.
  3. CSM: a small *trainable* top-down feature pyramid (nearest-upsample +
     3x3 conv, coarse -> fine) that fuses multi-scale context — this and the
     per-scale patch heads are the only trainable parts of the module.
  4. Per-scale PatchHead: a small spectral-norm conv stack producing a
     logit map + intermediate features (for the feature-matching loss the
     training loop already computes), conditioned on the hole mask
     (downsampled to that scale and concatenated in) since the frozen
     backbone itself has no notion of the inpainting mask.

Drop-in API-compatible with `mangainpaint/model_scratch.SingleScaleD`:
`forward(img_or_comp, mask, return_feats=False)` and `refresh()`, so it plugs
into `mangainpaint/trainer.py`'s existing `model_fn(cfg) -> (G, D)` factory pattern
without any train-loop changes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from mangainpaint.losses import sobel_mag

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_backbone_input(x, size=None):
    """x: (B,1,H,W) in [-1,1] -> (B,3,size,size) ImageNet-normalized.

    Resizing before the backbone isn't just a speed lever: EfficientNet-B0's
    ImageNet-pretrained BatchNorm stats were calibrated around ~224-256px
    inputs, so feeding it our native 384px is a train/test resolution
    mismatch on top of being 2.25x more expensive (FLOPs scale with H*W).
    """
    if size is not None and x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)
    x01 = (x + 1.0) * 0.5
    x3 = x01.repeat(1, 3, 1, 1)
    mean = IMAGENET_MEAN.to(x.device, x.dtype)
    std = IMAGENET_STD.to(x.device, x.dtype)
    return (x3 - mean) / std


class FrozenEfficientNetBackbone(nn.Module):
    # (stage index in torchvision's `.features` Sequential, output channels)
    # channel counts are input-resolution-invariant; spatial sizes below were
    # measured directly on a 384x384 input for reference —
    # ProjectedD resizes to `backbone_input_size` (default 256) before this,
    # so actual spatial sizes at runtime are smaller (e.g. 64/16/8 @ 256px):
    #   idx 2 -> 24ch  @ stride 4  (96x96 @ 384px input)
    #   idx 4 -> 80ch  @ stride 16 (24x24 @ 384px input)
    #   idx 6 -> 192ch @ stride 32 (12x12 @ 384px input)
    STAGE_IDXS = (2, 4, 6)
    STAGE_CHANNELS = (24, 80, 192)

    def __init__(self):
        super().__init__()
        m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        last_idx = max(self.STAGE_IDXS)
        self.features = m.features[:last_idx + 1]
        for p in self.parameters():
            p.requires_grad_(False)
        self._frozen_eval = True
        self.eval()

    def train(self, mode=True):
        # Always eval: frozen backbone must use running BN stats, not batch
        # stats, regardless of the owning discriminator's train()/eval() calls.
        return super().train(False)

    def forward(self, x):
        # No torch.no_grad() here: G's adversarial loss needs gradients to
        # flow from D's output all the way back through this backbone to the
        # generated composite image. Only the *weights* are frozen
        # (requires_grad=False), not the activation graph.
        feats = []
        h = x
        for i, layer in enumerate(self.features):
            h = layer(h)
            if i in self.STAGE_IDXS:
                feats.append(h)
        return feats


class RandomProjection(nn.Module):
    """Fixed (non-trainable) random 1x1 channel projection — the paper's CCM."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.proj(x)


class CSM(nn.Module):
    """Trainable top-down (coarse -> fine) feature fusion — the paper's CSM."""
    def __init__(self, ch, n_scales):
        super().__init__()
        self.fuse = nn.ModuleList([nn.Conv2d(ch, ch, 3, 1, 1) for _ in range(n_scales - 1)])

    def forward(self, feats):
        """feats: list ordered fine -> coarse. Returns fused list, same order."""
        fused = [None] * len(feats)
        fused[-1] = feats[-1]
        for i in range(len(feats) - 2, -1, -1):
            up = F.interpolate(fused[i + 1], size=feats[i].shape[2:], mode='nearest')
            fused[i] = feats[i] + self.fuse[i](up)
        return fused


class PatchHead(nn.Module):
    """Small spectral-norm patch classifier on top of one fused CSM scale.

    `edge_ch` (0 by default, unchanged behavior): an optional structure-
    conditioning side path (see `ProjectedD`'s docstring for the
    motivation) added *after* `b1`, via its own zero-initialized conv --
    not concatenated into `b1`'s input the way `mask` is. This is a
    deliberate warm-start-compatibility trade-off (same pattern as
    `mangainpaint/model_attn.py`'s `NoiseInjection`): concatenating at the input
    would change `b1`'s weight shape and break every existing real
    checkpoint's partial load; a zero-init additive side path instead means
    an old checkpoint (edge_ch=0, or edge_ch>0 with `edge_proj` absent from
    its state dict) loads with the edge path silently inert, byte-identical
    to the pre-edge model at step 0.
    """
    def __init__(self, in_ch, base=32, edge_ch=0):
        super().__init__()
        sn = nn.utils.spectral_norm
        self.b1 = nn.Sequential(sn(nn.Conv2d(in_ch, base * 2, 3, 1, 1)),
                                nn.LeakyReLU(0.2, inplace=True))
        self.b2 = nn.Sequential(sn(nn.Conv2d(base * 2, base * 4, 3, 1, 1)),
                                nn.LeakyReLU(0.2, inplace=True))
        self.out = sn(nn.Conv2d(base * 4, 1, 3, 1, 1))

        self.edge_ch = edge_ch
        if edge_ch > 0:
            self.edge_proj = nn.Conv2d(edge_ch, base * 2, 3, 1, 1)
            nn.init.zeros_(self.edge_proj.weight)
            nn.init.zeros_(self.edge_proj.bias)

    def forward(self, x, edge=None):
        f1 = self.b1[0](x)
        if self.edge_ch > 0 and edge is not None:
            f1 = f1 + self.edge_proj(edge)
        f1 = self.b1[1](f1)
        f2 = self.b2(f1)
        logit = self.out(f2)
        return logit, [f1, f2]

    def reinit(self):
        def _reinit(module):
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, a=0.2, nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.b2.apply(_reinit)
        self.out.apply(_reinit)


class ProjectedD(nn.Module):
    """`edge_ch` (0 by default, unchanged behavior for every existing
    script): optional structure conditioning, see module-level notes below
    and `PatchHead`'s docstring. Motivation: every prior run conditions D
    only on *where* the hole is (`mask`), never on *what structure* should
    plausibly continue there -- one of only two levers in this project's
    from-scratch search never varied (the other, content-verification
    losses, already tried; see paper §6.3). msxie92's own `D_mg`
    (Xie et al., SIGGRAPH 2021, "Seamless Manga Inpainting with Semantics
    Awareness", Eq. 12) is architecturally a plain 5-layer PatchGAN (no
    heavier than this project's `ProjectedD`, which already routes through
    a frozen EfficientNet) but jointly conditioned on a separately
    *inpainted* line map + screentone map -- a real structure hint, not
    self-referential. This project has no line-inpainting stage to draw
    that hint from, so the honest, scoped stand-in used here is a Sobel
    edge-magnitude map computed directly from the same image D is judging
    (`sobel_mag`, `mangainpaint/losses.py`, already used for the generator's own
    edge loss): for the real branch that's real edges everywhere; for the
    fake branch, real edges outside the hole and *generated* edges inside
    it. Self-referential rather than a real external hint, so this tests
    whether giving D access to edge structure at all helps, not the full
    msxie92 mechanism -- logged as a scope trade-off, not a literal port.
    """
    def __init__(self, mask_ch=1, proj_ch=64, base=32, backbone_input_size=256, edge_ch=0):
        super().__init__()
        self.backbone = FrozenEfficientNetBackbone()
        self.backbone_input_size = backbone_input_size
        self.edge_ch = edge_ch
        in_chs = FrozenEfficientNetBackbone.STAGE_CHANNELS
        self.ccms = nn.ModuleList([RandomProjection(c, proj_ch) for c in in_chs])
        self.csm = CSM(proj_ch, len(in_chs))
        self.heads = nn.ModuleList([PatchHead(proj_ch + mask_ch, base, edge_ch=edge_ch) for _ in in_chs])

    def forward(self, img_or_comp, mask, return_feats=False):
        x3 = to_backbone_input(img_or_comp, size=self.backbone_input_size)
        stage_feats = self.backbone(x3)
        proj_feats = [ccm(f) for ccm, f in zip(self.ccms, stage_feats)]
        fused = self.csm(proj_feats)

        edge_full = None
        if self.edge_ch > 0:
            # Native resolution, single-channel -- computed once, downsampled
            # per scale below (bilinear: edge magnitude is continuous, unlike
            # the binary `mask` which uses nearest).
            edge_full = torch.tanh(sobel_mag(img_or_comp))

        logits, feats_out = [], []
        for head, ff in zip(self.heads, fused):
            m = F.interpolate(mask.float(), size=ff.shape[2:], mode='nearest')
            edge = None
            if edge_full is not None:
                edge = F.interpolate(edge_full, size=ff.shape[2:], mode='bilinear', align_corners=False)
            logit, feats = head(torch.cat([ff, m], dim=1), edge=edge)
            logits.append(logit)
            feats_out.extend(feats)

        target_hw = logits[-1].shape[2:]  # coarsest (smallest) scale
        pooled = [F.adaptive_avg_pool2d(l, target_hw) for l in logits]
        combined = torch.stack(pooled, dim=0).mean(0)

        if return_feats:
            return combined, feats_out
        return combined

    def refresh(self):
        for h in self.heads:
            h.reinit()
