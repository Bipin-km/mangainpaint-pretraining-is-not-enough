"""
S1-attn -- LamaSlimG's FFC-ResNet bottleneck REPLACED by real windowed
self-attention (Swin-style). No distillation, no external information --
same control-cell shape as S1, testing a different backbone mechanism.

The reviewer question this answers: LaMa is claimed decisive, but this
project's existing "attention" experiments (`model_attn.py`'s contextual
attention, `model_linattn.py`'s linear attention) aren't real transformer
self-attention -- a patch-copy mechanism and a linear-complexity kernel
approximation, respectively. `mangainpaint/model_lama_slim_attn.py`'s
`LamaSlimAttnG` is the real thing: standard QKV self-attention in Swin-style
alternating regular/shifted 8x8 windows, replacing the FFC stack entirely
(not added alongside it -- see that file's docstring for why keeping FFC
upstream would repeat this project's own already-rejected Axis A4 confound,
and why windowed rather than global attention is the MORE faithful
comparison to what MAT/ZITS actually do at this resolution, not a cheaper
stand-in).

**Prediction, stated before running** (this project's organizing
principle -- external info wins, internal re-parameterization doesn't,
see paper §6.3): self-attention, like FFC/UFFC/ctx-attn/
linattn before it, re-mixes information already present in the input --
it doesn't inject anything external. Every internal-reparameterization
swap tried so far (7+ independent architecture axes) has landed in the
same null band on the real brush task. This is very likely another one.
Running it anyway converts "we predict this is null" into "we tested it
and it's null, consistent with our mechanism" -- a materially stronger
claim for the manuscript and any rebuttal.

7.70M params (vs. LamaSlimG's 8.78M, vs. big-lama's 51.0M) -- inside
budget with room to spare. Config is byte-identical to `lama_slim_s1`
otherwise: same 30 epochs, same loss weights, same pure-brush masking,
same ProjectedD. So this vs. `lama_slim_s1` isolates exactly one variable:
FFC-ResNet vs. windowed self-attention as the bottleneck mechanism, both
at random init, both with zero external information.

External assets: only the LaMa **source** (`external/lama/saicinpainting`)
for the trainer/dataset's shared imports -- `LamaSlimAttnG` itself imports
nothing from `external/lama`, only `mangainpaint/model_scratch.py`'s `OutHead`.

Run:
    torchrun --standalone --nproc_per_node=<N> recipes/student_s1_attn.py
Resume:
    Set CFG["resume"] = "checkpoints/last.pt" and re-run.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mangainpaint.trainer import run
from mangainpaint.model_lama_slim_attn import LamaSlimAttnG
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

    # Pure B1 brush masking, matching every S-cell in this axis.
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

WINDOW_SIZE = 8
NUM_HEADS = 8
MLP_RATIO = 2.0
N_BLOCKS = 12


def model_fn(cfg):
    G = LamaSlimAttnG(window_size=WINDOW_SIZE, num_heads=NUM_HEADS,
                      mlp_ratio=MLP_RATIO, n_blocks=N_BLOCKS,
                      expose_bottleneck=False)
    D = ProjectedD(mask_ch=1, proj_ch=cfg["proj_ch"], base=32,
                   backbone_input_size=cfg["backbone_input_size"])
    return G, D


if __name__ == "__main__":
    run(CFG, model_fn)
