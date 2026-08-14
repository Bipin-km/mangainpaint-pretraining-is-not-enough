"""
S2-GN -- S2's exact 4-term distillation recipe + Moebius-style GRADIENT-NORM
ADAPTIVE loss weighting. One variable changed vs. S2.

**Where this comes from.** Moebius (Duan/Xu et al., ECCV 2026,
arXiv:2606.19195 -- the strongest zero-shot baseline in this project's
paper) distills its 0.22B student from PixelHacker with an adaptive
weighting mechanism (`cal_adaptive_weights_type8` in their released
`train_distillation.py`): at each step, every auxiliary/KD loss is rescaled
by ||grad(task)|| / ||grad(aux)|| measured at probe layers, so no KD term
can out-shout or get drowned by the task loss regardless of its raw
magnitude. Their paper credits "dynamically balancing training via a
gradient norm adaptive loss weighting mechanism" as one of the three
ingredients that lets a 0.22B student absorb a much larger teacher.

**Why it plausibly helps HERE.** S2's four KD weights
(out 2.0/1.0, feat 1.0/0.5, wavelet 1.0/0.5, patchnce 0.5/0.25 across
P1/P2) were set once by measured-magnitude eyeballing and never tuned --
tuning them properly would cost a grid of training runs. The adaptive
mechanism replaces that grid with per-step measurement: each term's
gradient pressure at its entry point (output for out/wavelet KD,
bottleneck for feat/patchnce KD) is normalized to the task loss's own
pressure, THEN the static weights apply as relative preferences. If S2's
hand weights were already near-optimal this lands as a tie -- itself
informative (the recipe is robust); if they were mis-scaled, this is a
free correction.

**Implementation** (`mangainpaint/distill.py:adaptive_gn_multipliers`, gated by
`distill_adaptive_gn` -- OFF for every other run): activation probes
instead of Moebius's parameter probes (architecture-independent, and the
task loss has zero gradient at `distill_adapter`, so a parameter probe
would be degenerate for the feature terms); recomputed every
`distill_gn_every` steps with EMA smoothing instead of every step (each
measurement is 5 partial backwards -- Moebius pays that per-step, this
amortizes it for a modest compute budget); fp16-safe (losses pre-scaled by
a constant that cancels in the ratio, norms in fp32); DDP-consistent
(fresh ratios all-reduce-averaged so both ranks apply identical weights).
Multipliers are clamped to [0.05, 20] and logged once per epoch.

**Reading the result.** vs. S2 (EdgeF1 0.4322 / LPIPS 0.0206, held-out
brush): a win here is a second, independent Moebius-technique validation
for the paper (beyond citing it as a baseline); a tie retires the
"were the KD weights tuned?" reviewer question with a measured answer.

**TEACHER CHECKPOINT required**, same as S2: `distill_teacher_ckpt`
must point at the fine-tuned teacher's `best.pt` (produced by
`recipes/teacher_lama_finetune.py`).

Run:
    torchrun --standalone --nproc_per_node=<N> recipes/ablation_s2_gradnorm.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mangainpaint.trainer import run
from mangainpaint.model_lama_slim import LamaSlimG
from mangainpaint.model_projected_d import ProjectedD

# Set this to the fine-tuned teacher's checkpoint (produced by
# recipes/teacher_lama_finetune.py), or export DISTILL_TEACHER_CKPT instead.
TEACHER_CKPT = os.environ.get("DISTILL_TEACHER_CKPT",
                              "./checkpoints/teacher_lama_finetune/best.pt")

# Set these to your local Manga109-s / Manga109-segmentation roots, or
# export MANGA109_ROOT / MANGA109_SEG_ROOT instead.
MANGA109_ROOT = os.environ.get("MANGA109_ROOT", "./data/Manga109s")
MANGA109_SEG_ROOT = os.environ.get("MANGA109_SEG_ROOT", "./data/Manga109_segmentation")

CFG = {
    "root_dir":  MANGA109_ROOT,
    "train_csv": os.path.join(MANGA109_ROOT, "train.csv"),
    "val_csv":   os.path.join(MANGA109_ROOT, "val.csv"),
    "test_csv":  os.path.join(MANGA109_ROOT, "test.csv"),

    "image_size":  384,
    "batch_size":  8,
    "num_workers": None,
    "epochs":      30,
    "betas":       (0.5, 0.999),
    "grad_clip":   1.0,
    "show_every":  5,

    "gan_phase_start": 10,

    "p1_w_hole_rec": 5.0, "p1_w_ring_rec": 3.0, "p1_w_valid_id": 0.5,
    "p1_w_edge": 3.0, "p1_w_fft": 3.0, "p1_w_lpips": 0.5, "p1_w_ink": 2.0,

    "p2_w_hole_rec": 4.0, "p2_w_ring_rec": 2.0, "p2_w_valid_id": 0.5,
    "p2_w_edge": 2.0, "p2_w_fft": 2.0, "p2_w_lpips": 0.3, "p2_w_ink": 2.0,
    "p2_w_gan": 1.0, "p2_w_fm": 2.0, "p2_w_r1": 1.0, "p2_r1_every": 4,

    # ── Distillation: S2's recipe, unchanged ──
    "distill_teacher_ckpt": TEACHER_CKPT,
    "distill_hole_mult": 4.0,
    "distill_patch_temperature": 0.07,
    "distill_patch_max_positions": 256,
    "p1_w_distill_out": 2.0, "p1_w_distill_feat": 1.0,
    "p1_w_distill_wavelet": 1.0, "p1_w_distill_patchnce": 0.5,
    "p2_w_distill_out": 1.0, "p2_w_distill_feat": 0.5,
    "p2_w_distill_wavelet": 0.5, "p2_w_distill_patchnce": 0.25,

    # ── THE one change vs. S2: Moebius-style adaptive GN weighting ──
    # (see this file's docstring + mangainpaint/distill.py:adaptive_gn_multipliers)
    "distill_adaptive_gn": True,
    "distill_gn_every": 8,          # recompute cadence (5 partial backwards per recompute)
    "distill_gn_ema": 0.9,          # EMA decay on the measured ratios
    "distill_gn_clamp": (0.05, 20.0),

    "ink_threshold": 0.4, "ink_extra": 2.0,
    "d_refresh_every": 10,

    "lr_g": 4e-4, "lr_d": 5e-5,

    "mask_brush_w_min": 7, "mask_brush_w_max": 25,
    "mask_strokes_min": 1, "mask_strokes_max": 4,
    "mask_len_min": 20, "mask_len_max": 90,
    "mask_large_prob": 0.20, "mask_large_frac": 0.25,

    "seg_root": MANGA109_SEG_ROOT,
    "mask_balloon_prob": 0.0,
    "balloon_extra_stroke_prob": 0.3,

    "ring_radius": 5,

    "proj_ch": 64,
    "backbone_input_size": 256,

    "lpips_train_net": "squeeze",
    "lpips_eval_net":  "vgg",

    "use_compile":    False,
    "profile_timing": True,

    "hole_fill": "white",
    "ckpt_dir": "checkpoints",
    "vis_dir":  "vis",
    "resume":   None,
}

SLIM_NGF = 32
SLIM_NBLOCKS = 12
SLIM_INIT = "random"


def model_fn(cfg):
    # Identical to S2's model_fn: expose_bottleneck=True stashes the
    # bottleneck each forward (read by both the feat/patchnce KD terms AND
    # the GN probe) and is used on every step, so DDP's
    # find_unused_parameters=False stays valid.
    G = LamaSlimG(ngf=SLIM_NGF, n_blocks=SLIM_NBLOCKS, init_mode=SLIM_INIT,
                  expose_bottleneck=True, use_screenvae_hint=False)
    D = ProjectedD(mask_ch=1, proj_ch=cfg["proj_ch"], base=32,
                   backbone_input_size=cfg["backbone_input_size"])
    return G, D


if __name__ == "__main__":
    run(CFG, model_fn)
