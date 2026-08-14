"""
Fusion student (Axis A7, compaction follow-up): a *narrower*
LaMa-shaped FFC-ResNet with ONE linear-attention global-mixing pass added on
the bottleneck. The "cleverest fusion + compaction" cell.

Why this exists: the compaction decision, from the held-out test.csv
leaderboard after S2/S3 landed:

    teacher (51M)                     EdgeF1 0.4751
    S2 (8.78M, distill only)          0.4322
    S3 (8.78M, distill + external)    0.4312   (best sub-10M on selection_score)
    S1 (8.78M, control)               0.4201

The clean fact this design rests on: distillation is the real lever --
S1->S2 = +0.0121 EdgeF1, roughly 3x any architecture delta in the study and
the only signal outside the documented noise band. Bottleneck mechanism is
a much smaller lever than it looks (Section 6.1: three architecturally
distinct from-scratch generators span 0.424-0.430, narrower than run-to-run
scatter). So we do NOT compact *toward* attention. We compact the FFC
student under distillation -- the thing that actually works -- and test
whether a *cheap* global-mixing branch lets the expensive FFC width shrink
further than pure-FFC compaction can.

The compaction lever, with real measured param counts (`LamaSlimG`):

    ngf=32 n_blocks=12 -> 8.78M   (S1/S2/S3)
    ngf=24 n_blocks=12 -> 4.94M   (C1, pure-FFC compaction -- the clean control)
    ngf=20 n_blocks=12 -> 3.44M   (this file's FFC backbone)

FFC's spectral branch is the parameter-heavy part (75% of bottleneck
channels go through the FFT-domain conv). Linear attention (Katharopoulos et
al. 2020) is O(N) in tokens and parameter-light (4 1x1 convs). So the
hypothesis is a genuine architectural trade, not a bolt-on: replace some
expensive FFC *width* with one cheap global-mixing pass, and see whether the
~3.5M fusion matches the 4.94M pure-FFC C1 (and the 8.78M S2). If it does,
distillation + cheap attention compacts the deliverable to ~3.5M.

**One global pass, on the FINAL bottleneck, valid-source-masked.** Not
interleaved into every resblock (`LinAttnRes`, rejected as a full FFC
replacement, see `model_linattn.py`) and not a third
merged branch on encoder features. This is
the minimal, most-defensible form: after the narrow FFC-ResNet stack has
produced the bottleneck, one linear-attention residual gives every position a
single shot of whole-map context that a narrow FFC's spectral branch may be
too thin to supply. The source pool is restricted to valid (non-hole)
positions exactly as `ContextualAttentionBlock`/`LinAttnBranch` do -- deep
bottleneck positions inside a large hole are derived purely from the flat
fill input and carry no signal, so pooling them would dilute the global
memory (the same failure `model_lcg.py` was rejected for).

Interface identical to `LamaSlimG`: `forward(x)`, x=[B,2,H,W], out [B,1,H,W]
in [-1,1]. `expose_bottleneck=True` stashes the POST-linattn bottleneck for
distillation feature-matching (the student's final bottleneck representation
is the sensible thing to match against the teacher's) and builds the same
1x1 `distill_adapter` lifting the narrow width to the teacher's 512. The
whole net runs fp32 (autocast disabled in `forward`, inherited from
`LamaSlimG`) so `LinearAttention`'s own fp32 KV chain adds no new fp16 risk.
"""
import torch
import torch.nn.functional as F

from mangainpaint.model_lama_slim import LamaSlimG
from mangainpaint.model_linattn import LinearAttention


class LamaSlimFusG(LamaSlimG):
    """Narrow FFC-ResNet student + one bottleneck linear-attention pass.
    See module docstring. `linattn_heads` must divide `ngf * 8` (the
    bottleneck channel count): default ngf=20 -> 160 ch, 4 heads -> hd=40."""

    def __init__(self, ngf=20, n_blocks=12, linattn_heads=4, **kw):
        super().__init__(ngf=ngf, n_blocks=n_blocks, **kw)
        assert self.bneck_ch % linattn_heads == 0, (
            f"bottleneck {self.bneck_ch} not divisible by {linattn_heads} heads")
        self.linattn_heads = linattn_heads
        self.linattn = LinearAttention(self.bneck_ch, num_heads=linattn_heads)

    def _forward_net(self, inp4, mask):
        """Copy of LamaSlimG._forward_net with one linear-attention residual
        inserted on the bottleneck tuple after the FFC resblock stack and
        before the tail. The bottleneck exposed for distillation is the
        POST-linattn one."""
        m = self.net.model
        x = inp4
        for i in range(self.head_len):
            x = m[i](x)
        # x is the (x_l, x_g) FFC tuple at bottleneck resolution.

        if self.use_screenvae_hint:
            x = self._inject_hint(x, inp4, mask)

        for i in range(self.head_len, self.head_len + self.n_blocks):
            x = m[i](x)

        # ── Fusion: one valid-source-masked global linear-attention pass ──
        # ratio_gin/gout=0.75 for LaMa's resnet blocks, so both x_l and x_g
        # are real tensors here (cl = bneck_cl, cg = bneck_cg); concat order
        # [x_l, x_g] mirrors LamaSlimG's own expose code so the re-split is
        # coherent.
        x_l, x_g = x
        assert torch.is_tensor(x_l) and torch.is_tensor(x_g), (
            "fusion assumes both FFC bottleneck paths are tensors (ratio_g=0.75)")
        cat = torch.cat([x_l, x_g], dim=1)
        B, C, H, W = cat.shape
        valid = (F.interpolate(mask.float(), size=(H, W), mode="nearest") < 0.5)
        valid = valid.float().view(B, -1)          # (B, N), 1=valid/0=hole
        cat = cat + self.linattn(cat, valid_mask=valid)   # residual global mixing

        if self.expose_bottleneck:
            self.last_bottleneck = cat

        x_l, x_g = torch.split(cat, [self.bneck_cl, self.bneck_cg], dim=1)
        x = (x_l, x_g)

        for i in range(self.head_len + self.n_blocks, len(m)):
            x = m[i](x)
        return x
