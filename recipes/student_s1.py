"""
S1 -- slim LaMa student, NO distillation. The honest control.

Axis A7 (the on-thesis generator family). `LamaTransferG` (51.0M) wins the
real brush task decisively (see paper §6.2), but 51M is off-thesis: this
project is meant to produce a *small, efficient* grayscale-manga
inpainter. So LaMa becomes the teacher and the deliverable becomes an
<= 10M student. This is cell 1 of 3.

`LamaSlimG(ngf=32, n_blocks=12)` = **8.8M params** (vs. big-lama's 51.0M,
vs. the from-scratch pack's 2.7-2.8M), randomly initialized -- no pretrained
weights of any kind. (A channel-sliced init from big-lama was built and
measured first; it carries essentially no function across and loses to
random. See SLIM_INIT below and mangainpaint/model_lama_slim.py's docstring for
the numbers.)

**What this run answers**: how much of LaMa's win is the pretrained
*weights* vs. simply being a bigger, better-shaped network? S1 is a
LaMa-shaped FFC-ResNet at ~3x the from-scratch pack's size, trained from
scratch on manga with NO external information at all. If S1 already lands
near LaMa's 0.4757 brush EdgeF1, then the win was capacity/architecture and
the distillation in S2 has little left to buy; if S1 lands in the
from-scratch 0.42-0.44 band, then whatever S2 adds on top is attributable
entirely to the teacher. Either outcome is a real result, and a reviewer
WILL ask this question.

Config is `lama_transfer_brush_v1`'s, byte-identical except the generator
(`LamaSlimG` instead of `LamaTransferG`) -- same 30 epochs, same lr, same
loss weights, same pure-brush masking (`mask_balloon_prob=0`), same
ProjectedD. So S1 vs. that run isolates exactly one variable: model size.

Deliberately NOT included: `regional_stats_loss` (the project's one
validated loss win, currently inert at 0.0 in every run since it was
validated). Adding it here would put a second variable into the
size comparison; it's a one-flag follow-up on whichever of S1/S2/S3 wins.

External assets: only the LaMa **source** (`external/lama/saicinpainting`, a
git clone -- `FFCResNetGenerator` lives there). The 392MB big-lama
checkpoint is NOT needed: with a random init nothing reads it, and the
generator config is inlined in mangainpaint/model_lama_slim.py's
BIG_LAMA_GEN_CFG. So this run has no pretrained-checkpoint download at all.

Run:
    torchrun --standalone --nproc_per_node=<N> recipes/student_s1.py
Resume:
    Set CFG["resume"] = "checkpoints/last.pt" and re-run.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mangainpaint.trainer import run
from mangainpaint.model_lama_slim import LamaSlimG
from mangainpaint.model_projected_d import ProjectedD

# Set this to your local Manga109-s root, or export MANGA109_ROOT instead.
MANGA109_ROOT = os.environ.get("MANGA109_ROOT", "./data/Manga109s")

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

    "ink_threshold": 0.4, "ink_extra": 2.0,
    "d_refresh_every": 10,

    "lr_g": 4e-4, "lr_d": 5e-5,

    "mask_brush_w_min": 7, "mask_brush_w_max": 25,
    "mask_strokes_min": 1, "mask_strokes_max": 4,
    "mask_len_min": 20, "mask_len_max": 90,
    "mask_large_prob": 0.20, "mask_large_frac": 0.25,

    # Pure B1 brush masking -- the real task. Inert seg keys kept only to
    # stay CFG-shape-identical to the teacher fine-tune recipe.
    "seg_root": os.environ.get("MANGA109_SEG_ROOT", "./data/Manga109_segmentation"),
    "mask_balloon_prob": 0.0,
    "balloon_extra_stroke_prob": 0.3,

    "ring_radius": 5,  # required by trainer.py's make_ring() -- do not omit

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

# Student sizing. ngf=32/n_blocks=12 -> 8.8M, the most capacity inside the
# 10M budget (sweep in mangainpaint/model_lama_slim.py's docstring).
SLIM_NGF = 32
SLIM_NBLOCKS = 12
# MEASURED: channel-slicing a trained FFC-ResNet into a narrower one
# carries essentially no function -- step-0 hole-L1 on 32 val pages was
# 0.846 random / 0.941 sliced-from-raw-big-lama / 0.818
# sliced-from-fine-tuned-teacher, against the 51M teacher's own 0.108.
# Even the best slice is ~8x worse than its source and barely beats
# random, so first-k slicing (no importance ranking) destroys the learned
# channel structure. Weight surgery is NOT a transfer channel here;
# distillation is the only one. Random init also makes S1->S2 a clean
# one-variable ablation. See mangainpaint/model_lama_slim.py's docstring.
SLIM_INIT = "random"


def model_fn(cfg):
    G = LamaSlimG(ngf=SLIM_NGF, n_blocks=SLIM_NBLOCKS, init_mode=SLIM_INIT,
                  expose_bottleneck=False, use_screenvae_hint=False)
    D = ProjectedD(mask_ch=1, proj_ch=cfg["proj_ch"], base=32,
                   backbone_input_size=cfg["backbone_input_size"])
    return G, D


if __name__ == "__main__":
    run(CFG, model_fn)
