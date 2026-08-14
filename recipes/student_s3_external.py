"""
S3 -- slim LaMa student + distillation (S2's full recipe) + ScreenVAE
screentone loss + resnet_pl (ADE20K perceptual loss). Cell 3 of 3 (Axis
A7). The FUSION cell -- every external information source this project has
validated, stacked on one 8.8M student.

Identical to `student_s2_distill.py` (same 4-term distillation
recipe, unchanged) plus two additions: `p*_w_screenvae_consistency` and
`p*_w_resnet_pl`. So S3 vs. S2 isolates two variables at once, deliberately
-- see "Why bundled, not one-variable-at-a-time" below.

**The thesis this cell tests.** Sorting the study's design choices by what
they actually *supply* splits them cleanly in two (Section 6.2):

- Null -- every change that only re-parameterizes information the network
  could already derive from its own masked input and mask: a different
  bottleneck operator, an edge map, a Fourier code, a self-referential
  statistic, a reweighting of terms already in the objective. None of these
  moved the primary metrics outside the noise band.
- Moved the needle -- every change that injects information from outside
  the 6,788-page training corpus: big-lama pretraining (Places365 shape
  priors), ScreenVAE (a learned screentone manifold fit on a separate
  corpus), and `resnet_pl` (ADE20K scene-parsing features).

S3 stacks the two external sources that can be applied to an 8.8M student
at training time only, on top of S2's distillation from a teacher that is
itself external. It is the deployment-side test of the same principle.

**Why bundled, not one-variable-at-a-time.** The framing for this
phase is explicit: the goal isn't beating LaMa or preserving ablation
purity, it's finding a fusion where each ingredient covers another's
limitations. S1->S2 stays a clean single-variable test (zero external info
-> the teacher), because that's the confound a reviewer will actually
press on. S3 is deliberately the "everything validated, stacked" cell --
if it wins big, a follow-up ablates ScreenVAE-only vs. resnet_pl-only
against S2 to attribute the gain; if it's flat, that itself is informative.
Table 1 reports the outcome: S3 trades nothing measurable on EdgeF1 against
S2 for a further 7.2% LPIPS improvement, making it the strongest sub-10M
model in the study.

**ScreenVAE enters as a LOSS, not as an architectural hint.** The hint form
(`MangaFillNetScreenVAE`-style: frozen ScreenVAE -> LatentCompletionNet ->
zero-init hint_proj) requires running ScreenVAE **at inference**, and
ScreenVAE is 17.54M params + LatentCompletionNet(48) another 5.44M -- a
hint-equipped student ships 8.8 + 5.4 + 17.5 = 31.8M and busts the 10M
budget worse than big-lama's 51M did. `ScreenVAEConsistencyLoss` (already in
mangainpaint/losses' orbit, mangainpaint/model_screenvae.py) re-encodes the composited
output through the frozen ScreenVAE and matches its latent to the GT's --
training-only, **zero inference cost**. Same budget rule as the teacher and
`resnet_pl`: everything external enters at training time and is thrown
away. What ships is still one 8.8M FFC-ResNet.

Note that S2-VAE (`ablation_s2_screenvae_kd.py`) applies the same encoder
in a different role -- comparing student to *teacher* in ScreenVAE latent
space -- and does not help, because it re-expresses a signal the student
already receives through four other terms. Here the loss is a consistency
term against *ground truth*, injecting a fact about the target the student
cannot otherwise see. The distinction is what the external component is
asked to supervise, not whether a pretrained network is present.

**Required assets**: the teacher checkpoint (see S2), the ScreenVAE
weights (see `mangainpaint/checkpoint_registry.py`'s `SCREENVAE_WEIGHTS`),
and the `resnet_pl` ADE20K weights (91MB, CSAILVision scene-parsing
checkpoint -- see `mangainpaint/model_resnet_pl.py`'s docstring for the
origin URL).

Run:
    torchrun --standalone --nproc_per_node=<N> recipes/student_s3_external.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mangainpaint.trainer import run
from mangainpaint.model_lama_slim import LamaSlimG
from mangainpaint.model_projected_d import ProjectedD
from mangainpaint.checkpoint_registry import SCREENVAE_WEIGHTS

# Set this to the fine-tuned teacher's checkpoint (produced by
# recipes/teacher_lama_finetune.py), or export DISTILL_TEACHER_CKPT instead.
TEACHER_CKPT = os.environ.get("DISTILL_TEACHER_CKPT",
                              "./checkpoints/teacher_lama_finetune/best.pt")
# ADE20K scene-parsing checkpoint (CSAILVision, same file real big-lama's
# own resnet_pl uses -- see mangainpaint/model_resnet_pl.py's docstring for
# the origin URL). Mirrors this repo's own local layout
# (mangainpaint/pretrained/ade20k/...).
RESNET_PL_WEIGHTS = "mangainpaint/pretrained/ade20k/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth"

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

    # ── Distillation (inherited from S2, unchanged -- see that file for the
    # WKD/ReKo citations and the measured-magnitude rationale for weights) ──
    "distill_teacher_ckpt": TEACHER_CKPT,
    "distill_hole_mult": 4.0,
    "distill_patch_temperature": 0.07,
    "distill_patch_max_positions": 256,
    "p1_w_distill_out": 2.0, "p1_w_distill_feat": 1.0,
    "p1_w_distill_wavelet": 1.0, "p1_w_distill_patchnce": 0.5,
    "p2_w_distill_out": 1.0, "p2_w_distill_feat": 0.5,
    "p2_w_distill_wavelet": 0.5, "p2_w_distill_patchnce": 0.25,

    # ── ScreenVAE screentone loss ──
    # Training-only: the frozen 17.5M ScreenVAE re-encodes the composited
    # output and its latent is matched to the GT's. Nothing extra ships.
    "screenvae_weights_dir": SCREENVAE_WEIGHTS,
    "p1_w_screenvae_consistency": 1.0,
    "p2_w_screenvae_consistency": 1.0,

    # ── resnet_pl (LaMa's real HRF perceptual loss, ADE20K-pretrained) ──
    # w=10 follows LaMa's own high-receptive-field perceptual term, which
    # this reimplements (see this file's docstring). Training-only, zero
    # inference cost -- same budget rule as the teacher and ScreenVAE above.
    "resnet_pl_weights_path": RESNET_PL_WEIGHTS,
    "resnet_pl_input_size": 256,
    "p1_w_resnet_pl": 10.0,
    "p2_w_resnet_pl": 10.0,

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
    # use_screenvae_hint stays False on purpose -- see the docstring: the
    # hint form would ship 31.8M. ScreenVAE is a training-only LOSS here.
    G = LamaSlimG(ngf=SLIM_NGF, n_blocks=SLIM_NBLOCKS, init_mode=SLIM_INIT,
                  expose_bottleneck=True, use_screenvae_hint=False)
    D = ProjectedD(mask_ch=1, proj_ch=cfg["proj_ch"], base=32,
                   backbone_input_size=cfg["backbone_input_size"])
    return G, D


if __name__ == "__main__":
    run(CFG, model_fn)
