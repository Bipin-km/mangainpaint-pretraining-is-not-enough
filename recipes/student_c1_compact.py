"""
C1 -- COMPACTED slim student (4.94M) + S2's exact distillation recipe.

The clean single-variable compaction test off S2. Everything is byte-for-byte
S2's config -- the four-term WKD+ReKo distillation from the fine-tuned 51M
LaMa teacher, all loss weights, masking, schedule -- with ONE thing changed:
the FFC-ResNet width drops from ngf=32 to ngf=24, taking the shipped student
from 8.78M to **4.94M (-44%)**. n_blocks stays 12.

Why: after S2/S3 landed, the clean fact was that distillation is the real
lever (S1->S2 = +0.0121 EdgeF1, the only signal outside this project's noise
band), while bottleneck mechanism is not (the from-scratch pack spans
0.424-0.430, narrower than run-to-run scatter, Section 6.1).
Distillation is *the* mechanism for compaction -- its whole
purpose is moving a big teacher's capability into a small student. So the
question this run answers: **how far can the student shrink while distillation
holds its performance?** If C1 (4.94M) matches S2 (8.78M) on held-out
test.csv EdgeF1/LPIPS, that's a 44%-smaller model at equal quality -- a
strict win and a standalone result.

`ngf`/`n_blocks` are set in CFG (not just module constants) so they persist
into the saved checkpoint's cfg and `mangainpaint/checkpoint_registry.build_generator`
rebuilds the right-sized model at eval time. S1/S2/S3 only worked without this
because ngf=32 happens to be the registry default; a compacted model must
carry its own width.

The `distill_adapter` auto-resizes: at ngf=24 the bottleneck is 192ch (vs
256 at ngf=32), and the adapter lifts 192->512 to match the teacher. Nothing
else in the recipe is width-sensitive.

**TEACHER CHECKPOINT.** `distill_teacher_ckpt` points at the same
fine-tuned teacher's best.pt that S2/S3 used (produced by
`recipes/teacher_lama_finetune.py`).

Run:
    torchrun --standalone --nproc_per_node=<N> recipes/student_c1_compact.py
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

    # ── Compaction: the ONLY change vs. S2 ──
    "ngf": 24, "n_blocks": 12,

    # ── Distillation (identical to S2) ──
    "distill_teacher_ckpt": TEACHER_CKPT,
    "distill_hole_mult": 4.0,
    "distill_patch_temperature": 0.07,
    "distill_patch_max_positions": 256,
    "p1_w_distill_out": 2.0, "p1_w_distill_feat": 1.0,
    "p1_w_distill_wavelet": 1.0, "p1_w_distill_patchnce": 0.5,
    "p2_w_distill_out": 1.0, "p2_w_distill_feat": 0.5,
    "p2_w_distill_wavelet": 0.5, "p2_w_distill_patchnce": 0.25,

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


def model_fn(cfg):
    G = LamaSlimG(ngf=cfg["ngf"], n_blocks=cfg["n_blocks"], init_mode="random",
                  expose_bottleneck=True, use_screenvae_hint=False)
    D = ProjectedD(mask_ch=1, proj_ch=cfg["proj_ch"], base=32,
                   backbone_input_size=cfg["backbone_input_size"])
    return G, D


if __name__ == "__main__":
    run(CFG, model_fn)
