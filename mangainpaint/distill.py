"""
Teacher->student distillation from the fine-tuned LaMa transfer model into
the slim FFC-ResNet student (mangainpaint/model_lama_slim.py).

Why this is a NEW hypothesis and not an already-closed-out one: an earlier
attempt to "distill from LaMa" was closed out because **zero-shot
big-lama collapses to near-blank white on manga** -- a useless teacher
(see paper §6.2). That objection doesn't apply to the *manga-fine-tuned*
LaMa teacher used here, which is the best content reconstructor in the
project on the real brush task (see paper's results table, "Fine-tuned
LaMa" row -- wins every metric and every ink-density stratum). Distilling
THAT is a different experiment entirely.

And it is the on-thesis one: big-lama is good at completing complex shapes
because it saw millions of images -- exactly the thing 6,788 manga pages
cannot teach a small net from scratch. Distillation is the mechanism that
moves that capability into a model small enough to ship (<= 10M), which is
what the project actually set out to build.

Four terms, all CFG-weighted at the trainer call site (same convention as
every other loss in mangainpaint/trainer.py -- the loss object itself is
unweighted):

- **output KD**: L1 between student and teacher outputs, hole-weighted
  (`hole_mult`) so the teacher's *fill* dominates and its reproduction of
  already-visible pixels doesn't drown the signal.
- **feature KD**: L1 between the student's bottleneck (lifted 256->512 by
  `LamaSlimG.distill_adapter`) and the teacher's bottleneck, at the same
  spatial size. Output KD alone tends to hand over the teacher's mean
  behaviour but not its internal structure; the feature term is what
  actually transfers the shape priors.
- **wavelet KD** (`wavelet_kd_loss`, Zhang et al. CVPR 2022, "Wavelet
  Knowledge Distillation"): L1
  between student and teacher outputs' HIGH-frequency 3-level Haar-DWT
  subbands only, skipping the low-frequency residual. The paper's own
  motivating measurement (their Fig 1) is that GANs -- especially small
  ones -- already match GT almost perfectly on low frequency and fail
  specifically on high frequency; distilling low-freq content wastes
  capacity on something the student already gets right. This argument is
  SHARPER for manga than for the paper's own domain (shoes/zebras): ink
  lines and screentone dot patterns ARE the high-frequency band -- a
  bitonal manga page has almost no meaningful low-frequency content at
  all, unlike a natural photo.
- **hole-region contrastive KD** (`hole_patch_contrastive_loss`, Zhang et
  al. BMVC 2023, "Region-aware Knowledge Distillation" / ReKo): patch-wise
  InfoNCE between student and teacher bottleneck features, restricted to
  hole positions -- student feature at a position is the query, the
  teacher's feature at the SAME position is the positive key, the
  teacher's features at every OTHER hole position (same image) are
  negatives, no memory bank (exactly ReKo's eq. 2/4). ReKo itself finds its
  "crucial regions" via an unsupervised parameter-free attention module
  (top-K by teacher-feature magnitude, because in general image-to-image
  translation there's no ground truth for "the region that matters"). We
  have something strictly better: the mask IS the ground-truth crucial
  region -- the hole is exactly where the student has to actually
  reconstruct content rather than trivially copy the input, so no
  attention module is needed here.

Both new terms exist because the naive L1-only recipe (output KD + feature
KD alone) is precisely the failure mode both papers open by describing:
"directly minimizing the distance between the generated images of students
and teachers does not improve, but sometimes harms, GAN performance"
(WKD's introduction). This distillation recipe (S2/S3 in the paper) is
what addresses that.

The teacher runs under `no_grad` in fp32 (same overflow reason as
LamaTransferG -- see its forward()), is frozen, and is pinned to eval() so
its BatchNorm running stats never drift.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

_LAMA_ROOT = os.path.join(os.path.dirname(__file__), "..", "external", "lama")
if _LAMA_ROOT not in sys.path:
    sys.path.insert(0, _LAMA_ROOT)

from saicinpainting.training.modules.ffc import FFCResNetGenerator  # noqa: E402

_CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "external", "lama", "big-lama")


def _haar_dwt(x):
    """Single-level 2D Haar DWT via fixed depthwise conv (differentiable,
    GPU-native, no extra dependency -- pywt is numpy-only and not usable
    inside an autograd graph). x: (B,1,H,W), H and W even.
    Returns (ll, lh, hl, hh), each (B,1,H/2,W/2). Orthonormal-normalized
    (0.5 factor) so magnitudes stay comparable across levels."""
    k = x.new_tensor([[[[0.5, 0.5], [0.5, 0.5]]],     # LL: average
                      [[[0.5, 0.5], [-0.5, -0.5]]],   # LH: row (vertical) gradient
                      [[[0.5, -0.5], [0.5, -0.5]]],   # HL: column (horizontal) gradient
                      [[[0.5, -0.5], [-0.5, 0.5]]]])  # HH: diagonal
    out = F.conv2d(x, k, stride=2)
    return out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4]


def _haar_dwt_3level(x):
    """3-level Haar DWT, recursing on the LL band each time (matches WKD's
    own Psi^L(x) = LL3, Psi^H(x) = {HL3,LH3,HH3,HL2,LH2,HH2,HL1,LH1,HH1}).
    Returns the 9 high-frequency subbands (LL3, the low-frequency residual,
    is deliberately NOT included -- see wavelet_kd_loss)."""
    highs = []
    cur = x
    for _ in range(3):
        if cur.shape[-2] % 2 or cur.shape[-1] % 2:
            cur = F.pad(cur, (0, cur.shape[-1] % 2, 0, cur.shape[-2] % 2), mode="reflect")
        ll, lh, hl, hh = _haar_dwt(cur)
        highs += [lh, hl, hh]
        cur = ll
    return highs


def wavelet_kd_loss(student_out, teacher_out, mask=None, hole_mult=4.0):
    """L1 between student/teacher outputs' high-frequency 3-level Haar-DWT
    subbands only. See the module docstring for the WKD citation and why
    this is a sharper fit for manga than for the paper's own domain.

    student_out/teacher_out: (B,1,H,W) in [-1,1]. `mask`, if given,
    hole-weights each subband (downsampled to that subband's resolution via
    nearest interpolation, same convention as every other mask-downsample
    in this codebase, e.g. LatentCompletionNet) -- the valid region already
    matches trivially between two models fed the same input, so weighting
    it up would dilute the signal the same way output KD's `hole_mult`
    already prevents for the plain-pixel term."""
    s_highs = _haar_dwt_3level(student_out)
    t_highs = _haar_dwt_3level(teacher_out)
    total = student_out.new_zeros(())
    for s_h, t_h in zip(s_highs, t_highs):
        if mask is not None:
            m = F.interpolate(mask.float(), size=s_h.shape[2:], mode="nearest")
            w = 1.0 + (hole_mult - 1.0) * m
            total = total + (w * (s_h - t_h).abs()).sum() / w.sum().clamp(min=1.0)
        else:
            total = total + (s_h - t_h).abs().mean()
    return total / len(s_highs)


def hole_patch_contrastive_loss(student_feat, teacher_feat, mask,
                                temperature=0.07, max_positions=256):
    """Patch-wise InfoNCE distillation (ReKo), restricted to hole positions.
    See the module docstring for the citation and why the mask replaces
    ReKo's own unsupervised attention module here.

    student_feat/teacher_feat: (B,C,h,w), SAME channel count and spatial
    size (reuses the exact tensors DistillLoss.forward's feat_kd term
    already produces -- student projected through `distill_adapter`,
    teacher's raw bottleneck). mask: (B,1,H,W) at full resolution, 1=hole.

    Per image: gather feature vectors at hole positions, L2-normalize,
    student=query / teacher-same-position=positive / teacher-other-hole-
    position=negative, InfoNCE via cross_entropy against the diagonal
    label (mirrors ReKo's Fig 2c exactly: Softmax(Query.Key) -> identity
    ground truth). No memory bank, matching the paper's own design.

    `max_positions`: hole positions are subsampled (random, per-image)
    above this cap -- purely to bound the O(N^2) similarity matrix; at the
    bottleneck's coarse resolution a heavily-masked page can have several
    hundred hole positions, and this runs every step alongside two GAN
    networks on a single mid-range GPU."""
    B, C, h, w = student_feat.shape
    mask_ds = F.interpolate(mask.float(), size=(h, w), mode="nearest")
    s_flat = student_feat.permute(0, 2, 3, 1).reshape(B, h * w, C)
    t_flat = teacher_feat.permute(0, 2, 3, 1).reshape(B, h * w, C)
    m_flat = mask_ds.reshape(B, h * w) > 0.5

    total = student_feat.new_zeros(())
    n_valid = 0
    for b in range(B):
        idx = m_flat[b].nonzero(as_tuple=True)[0]
        if idx.numel() < 2:
            continue  # need >=2 positions for a negative to exist
        if idx.numel() > max_positions:
            perm = torch.randperm(idx.numel(), device=idx.device)[:max_positions]
            idx = idx[perm]
        sv = F.normalize(s_flat[b, idx], dim=-1)
        tv = F.normalize(t_flat[b, idx], dim=-1)
        logits = sv @ tv.t() / temperature
        labels = torch.arange(idx.numel(), device=idx.device)
        total = total + F.cross_entropy(logits, labels)
        n_valid += 1
    if n_valid == 0:
        return student_feat.new_zeros(())
    return total / n_valid


def adaptive_gn_multipliers(distill_fn, task_loss, terms, gen, bneck, cfg):
    """Gradient-norm adaptive weighting for the KD terms -- a direct port of
    Moebius's `cal_adaptive_weights_type8` (Duan/Xu et al., ECCV 2026,
    from their released `train_distillation.py`) to
    this trainer: each KD term is rescaled per-recompute by
    ||grad(task)|| / ||grad(term)||, measured at a shared probe tensor, so
    every KD term's gradient pressure is normalized to the task loss's own
    instead of being hand-tuned. The returned multiplier composes with (does
    not replace) the static `p*_w_distill_*` weights, exactly as Moebius
    composes `out_weight_outkd * args.KD_loss_weight`.

    Adaptations to this codebase, and why:
    - **Probes are activations, not parameters.** Moebius probes two module
      weights (last featkd layer / out layer). Here `out`/`wavelet` probe
      the student's output `gen` and `feat`/`patchnce` probe the stashed
      bottleneck (`LamaSlimG.last_bottleneck`) -- architecture-independent,
      and each term is probed where its gradient actually enters the
      student. (A parameter probe at `distill_adapter` would be degenerate:
      the task loss has zero gradient there, the adapter exists only for the
      KD terms.)
    - **Recomputed every `distill_gn_every` steps with EMA smoothing**, not
      every step: each measurement costs one partial backward per loss
      (5 total). Moebius pays that every step; alongside a discriminator
      on modest GPU budgets we amortize it instead. Between recomputes the
      EMA value is applied.
    - **fp16-safe**: losses are pre-scaled by a constant before
      `autograd.grad` (cancels exactly in the ratio) so the partial backward
      can't underflow under autocast, per this project's standing
      fp16-gotcha discipline; norms are taken in fp32.
    - **DDP-consistent**: freshly measured ratios are all-reduce-averaged so
      every rank applies identical weights (a per-rank divergence would make
      ranks optimize different objectives).

    `terms`: dict {"out","feat","wavelet","patchnce"} -> unweighted loss
    tensors from `DistillLoss.forward`. `bneck` may be None (no
    `expose_bottleneck`) -- feat/patchnce then keep their EMA/1.0 value.
    State (step counter + EMA dict) lives on `distill_fn`, which persists
    across steps; returns a dict with all four keys, defaulting to 1.0.

    CFG keys (all optional): `distill_gn_every` (default 8),
    `distill_gn_ema` (default 0.9), `distill_gn_clamp` (default (0.05, 20)).
    Gated at the call site by `distill_adaptive_gn` -- nothing changes for
    any run that doesn't set it."""
    every = max(1, int(cfg.get("distill_gn_every", 8)))
    decay = float(cfg.get("distill_gn_ema", 0.9))
    lo, hi = cfg.get("distill_gn_clamp", (0.05, 20.0))

    state = getattr(distill_fn, "_gn_state", None)
    if state is None:
        state = {"step": 0, "ema": {k: 1.0 for k in terms}}
        distill_fn._gn_state = state
    step = state["step"]
    state["step"] = step + 1
    if step % every != 0:
        return dict(state["ema"])

    probes = {"gen": gen}
    if bneck is not None and torch.is_tensor(bneck) and bneck.requires_grad:
        probes["bneck"] = bneck
    probe_names = list(probes.keys())
    probe_list = [probes[n] for n in probe_names]
    SCALE = 1024.0  # cancels in the ratio; guards fp16 underflow in the partial backward

    def probe_norms(loss):
        if not (torch.is_tensor(loss) and loss.requires_grad):
            return {}
        grads = torch.autograd.grad(loss * SCALE, probe_list,
                                    retain_graph=True, allow_unused=True)
        return {n: (g.float().norm() if g is not None else None)
                for n, g in zip(probe_names, grads)}

    term_probe = {"out": "gen", "wavelet": "gen", "feat": "bneck", "patchnce": "bneck"}
    task_n = probe_norms(task_loss)
    fresh = {}
    for name, loss in terms.items():
        p = term_probe.get(name, "gen")
        tn = task_n.get(p)
        an = probe_norms(loss).get(p)
        if tn is None or an is None:
            continue
        tn_f, an_f = float(tn), float(an)
        if an_f <= 0.0 or tn_f <= 0.0:
            continue  # zero-constant term this step (e.g. no hole positions)
        fresh[name] = min(max(tn_f / an_f, lo), hi)

    if fresh and torch.distributed.is_available() and torch.distributed.is_initialized():
        keys = sorted(fresh)
        t = torch.tensor([fresh[k] for k in keys], device=gen.device, dtype=torch.float32)
        torch.distributed.all_reduce(t)
        t = t / torch.distributed.get_world_size()
        fresh = {k: float(v) for k, v in zip(keys, t)}

    for k, v in fresh.items():
        state["ema"][k] = decay * state["ema"][k] + (1.0 - decay) * v
    return dict(state["ema"])


class DistillLoss(nn.Module):
    """Frozen fine-tuned-LaMa teacher + the four KD terms.

    `teacher_ckpt` is a checkpoint saved by mangainpaint/trainer.py (a dict with a
    "G" key holding `LamaTransferG`'s state_dict, i.e. keys prefixed
    `net.`) -- e.g. `lama_transfer_brush_v1/checkpoints/best.pt`.

    `screenvae_weights_dir` (optional, default None -- existing callers
    unaffected): when set, a frozen ScreenVAE encoder is loaded and
    `svae_kd()` becomes available as a FIFTH KD term -- L1 between the
    ScreenVAE latents of the student's and the TEACHER's composited outputs,
    hole-focused. This is the manga-native analogue of Moebius's
    "distillation strictly within the latent space" (their teacher/student
    are aligned inside PixelHacker's VAE latent, never in pixel space):
    instead of a photo-domain VAE we align inside msxie92's
    screentone-disentangled latent, i.e. "make the student's screentone
    *as the teacher renders it*" -- distinct from S3's
    `ScreenVAEConsistencyLoss`, which matches the student's latent to the
    GROUND TRUTH's and carries no teacher signal at all.
    """

    def __init__(self, teacher_ckpt, hole_mult=4.0, lama_ckpt_dir=None,
                 patch_temperature=0.07, patch_max_positions=256,
                 screenvae_weights_dir=None):
        super().__init__()
        self.hole_mult = hole_mult
        self.patch_temperature = patch_temperature
        self.patch_max_positions = patch_max_positions

        # Teacher architecture = big-lama's, exactly (the checkpoint being
        # loaded IS a fine-tuned LamaTransferG). Uses the real config.yaml if
        # the big-lama dir is on disk, else the inlined copy -- so distilling
        # needs only the fine-tuned checkpoint, not the 392MB pretrained one.
        from mangainpaint.model_lama_slim import _gen_cfg
        gcfg = _gen_cfg(ngf=64, n_blocks=18, ckpt_dir=lama_ckpt_dir)
        self.net = FFCResNetGenerator(**gcfg)
        self.n_downsampling = gcfg["n_downsampling"]
        self.head_len = 2 + self.n_downsampling
        self.n_blocks = gcfg["n_blocks"]

        ckpt = torch.load(teacher_ckpt, map_location="cpu", weights_only=False)
        sd = ckpt["G"] if "G" in ckpt else ckpt
        # LamaTransferG wraps FFCResNetGenerator as `self.net`, so its saved
        # keys carry a `net.` prefix; strip it to load the bare generator.
        sd = {k[len("net."):]: v for k, v in sd.items() if k.startswith("net.")}
        missing, unexpected = self.net.load_state_dict(sd, strict=False)
        assert not missing and not unexpected, (
            f"teacher checkpoint mismatch: missing={missing} unexpected={unexpected}")

        for p in self.net.parameters():
            p.requires_grad_(False)
        self.net.eval()

        # Optional 5th term: ScreenVAE-latent KD (see class docstring).
        # Lazy import -- model_screenvae is heavy and most distill runs
        # don't use this.
        self.svae = None
        if screenvae_weights_dir is not None:
            from mangainpaint.model_screenvae import ScreenVAE
            self.svae = ScreenVAE(weights_dir=screenvae_weights_dir)
        self._last_t_out = None

    def train(self, mode=True):
        super().train(mode)
        self.net.eval()  # frozen teacher never leaves eval, whatever the parent does
        if self.svae is not None:
            self.svae.eval()  # ScreenVAE.train() already forces eval; explicit anyway
        return self

    @torch.no_grad()
    def _teacher_forward(self, x):
        """Returns (out_gray_[-1,1], bottleneck_512ch). Mirrors
        LamaTransferG.forward's input adaptation exactly."""
        masked, mask = x[:, 0:1], x[:, 1:2]
        img01 = (masked + 1) / 2
        inp4 = torch.cat([img01.repeat(1, 3, 1, 1), mask], dim=1)

        with torch.amp.autocast(x.device.type, enabled=False):
            h = inp4.float()
            m = self.net.model
            for i in range(self.head_len + self.n_blocks):
                h = m[i](h)
            x_l, x_g = h
            bneck = torch.cat([t for t in (x_l, x_g) if torch.is_tensor(t)], dim=1)
            for i in range(self.head_len + self.n_blocks, len(m)):
                h = m[i](h)
            out3 = h  # sigmoid, [0,1]

        gray = out3.mean(dim=1, keepdim=True) * 2 - 1
        return gray.to(x.dtype), bneck

    def forward(self, gen, x, mask, student):
        """gen: student output [B,1,H,W] in [-1,1]. x: the dataset's
        model_input [B,2,H,W]. mask: [B,1,H,W], 1=hole. student: the
        *unwrapped* LamaSlimG (built with expose_bottleneck=True), read for
        its stashed `last_bottleneck` + `distill_adapter`.

        Returns (out_kd, feat_kd, wavelet_kd, patch_kd), all unweighted --
        the caller applies cfg["p*_w_distill_{out,feat,wavelet,patchnce}"].
        Always computes all four (cheap relative to a full GAN step) so a
        weight of 0.0 is a genuine no-op at the call site, matching this
        codebase's convention for every other optional loss term."""
        t_out, t_bneck = self._teacher_forward(x)
        # Stashed (already grad-free, computed under no_grad) so svae_kd()
        # can reuse the teacher output from THIS step without paying a
        # second 51M-teacher forward.
        self._last_t_out = t_out

        # Hole-weighted L1 on the output. The teacher's *fill* is the signal;
        # weighting the hole up keeps the term from being dominated by both
        # models trivially agreeing on the (unmasked, identical) valid region.
        w = 1.0 + (self.hole_mult - 1.0) * mask
        out_kd = (w * (gen - t_out).abs()).sum() / w.sum().clamp(min=1.0)

        wavelet_kd = wavelet_kd_loss(gen, t_out, mask=mask, hole_mult=self.hole_mult)

        feat_kd = gen.new_zeros(())
        patch_kd = gen.new_zeros(())
        s_bneck = getattr(student, "last_bottleneck", None)
        if s_bneck is not None and hasattr(student, "distill_adapter"):
            with torch.amp.autocast(x.device.type, enabled=False):
                s_proj = student.distill_adapter(s_bneck.float())
                t_b = t_bneck.float()
                if s_proj.shape[2:] != t_b.shape[2:]:
                    t_b = F.interpolate(t_b, size=s_proj.shape[2:],
                                        mode="bilinear", align_corners=False)
                feat_kd = F.l1_loss(s_proj, t_b)
                patch_kd = hole_patch_contrastive_loss(
                    s_proj, t_b, mask, temperature=self.patch_temperature,
                    max_positions=self.patch_max_positions)
            feat_kd = feat_kd.to(gen.dtype)
            patch_kd = patch_kd.to(gen.dtype)

        return out_kd, feat_kd, wavelet_kd, patch_kd

    def svae_kd(self, gen, img, mask):
        """ScreenVAE-latent KD (5th term, optional -- see class docstring).
        L1 between the frozen ScreenVAE encoder's latents of the student's
        and the teacher's composited outputs, hole-focused (same masking
        convention as `ScreenVAEConsistencyLoss`, whose target is GT rather
        than the teacher).

        MUST be called after `forward()` in the same step -- it reuses the
        teacher output `forward()` stashed (`_last_t_out`), because the
        teacher pass is the expensive part and `mangainpaint/trainer.py` always
        calls `forward()` first in the same distill block. `gen`/`img` in
        [-1,1]; `mask` 1=hole. Gradient flows into the student only through
        the hole (the composite's valid region is GT on both sides)."""
        assert self.svae is not None, "DistillLoss built without screenvae_weights_dir"
        t_out = self._last_t_out
        assert t_out is not None, "svae_kd() must be called after forward() in the same step"
        comp_s = gen * mask + img * (1 - mask)
        comp_t = t_out * mask + img * (1 - mask)
        with torch.no_grad():
            t_lat = self.svae(comp_t)
        s_lat = self.svae(comp_s)
        diff = (s_lat - t_lat).abs()
        m = F.interpolate(mask.float(), size=diff.shape[2:], mode="nearest")
        denom = (m.sum() * diff.shape[1]).clamp_min(1.0)
        return (diff * m).sum() / denom
