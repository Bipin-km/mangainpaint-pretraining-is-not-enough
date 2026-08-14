"""
S2-svaeKD -- S2's exact 4-term distillation recipe + a FIFTH KD term:
ScreenVAE-LATENT knowledge distillation. One variable changed vs. S2.

**Where this comes from.** Moebius (Duan/Xu et al., ECCV 2026 -- the
strongest zero-shot baseline in this project's paper) distills its student
from PixelHacker "strictly within the latent space" of a frozen VAE --
teacher and student are aligned in a learned perceptual latent, never in
raw pixels, which their paper credits for bridging the capacity gap
without expensive pixel-space decoding. The manga-native analogue of that
latent is not a photo VAE but msxie92's ScreenVAE (SIGGRAPH 2020): a
frozen encoder whose 4-channel latent disentangles *screentone* -- the
exact structure this project's students systematically under-reconstruct.

**What the new term is** (`mangainpaint/distill.py:DistillLoss.svae_kd`, gated
by `p*_w_distill_svae` -- 0.0/absent for every other run): L1 between the
frozen ScreenVAE encoder's latents of the STUDENT's and the TEACHER's
composited outputs, hole-focused. Gradient flows through the frozen
encoder back into the student's hole fill only. The teacher output is
reused from the same step's 4-term forward -- no second teacher pass.

**Why this is NOT S3's ScreenVAE loss.** S3's `ScreenVAEConsistencyLoss`
matches the student's latent to the GROUND TRUTH's -- it is a
reconstruction loss in a better space, and carries zero teacher signal.
This term matches the student to the TEACHER in that same space: "render
your screentone the way the teacher renders it," including on ambiguous
holes where the teacher's (excellent) fill differs from GT and a
GT-target loss would fight the KD terms instead of agreeing with them.
S2 + this term stays a pure-distillation cell: every training signal
beyond the base recipe comes from the teacher.

**Reading the result.** vs. S2 (EdgeF1 0.4322 / LPIPS 0.0206, held-out
brush): a win validates latent-space KD as a second Moebius technique
worth adopting (and stacks naturally with S3's external losses in a
follow-up); a tie says pixel+feature+wavelet KD already saturates what
this teacher can transfer at 8.8M.

**Required assets**, same as S3 used: the teacher `best.pt` and the
ScreenVAE encoder weights (training-only; the shipped student stays
8.78M) -- see `mangainpaint/checkpoint_registry.py`'s `SCREENVAE_WEIGHTS`.

Run:
    torchrun --standalone --nproc_per_node=<N> recipes/ablation_s2_screenvae_kd.py
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

    # ── THE one change vs. S2: ScreenVAE-latent KD (5th term) ──
    # Weights follow S3's ScreenVAE-consistency precedent (1.0 both phases;
    # that term's magnitude is in the same regime, same encoder, same
    # hole-focused normalization -- see mangainpaint/distill.py:svae_kd).
    "distill_svae_weights_dir": SCREENVAE_WEIGHTS,
    "p1_w_distill_svae": 1.0,
    "p2_w_distill_svae": 1.0,

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
    # Identical to S2's model_fn (see that file for the DDP note).
    G = LamaSlimG(ngf=SLIM_NGF, n_blocks=SLIM_NBLOCKS, init_mode=SLIM_INIT,
                  expose_bottleneck=True, use_screenvae_hint=False)
    D = ProjectedD(mask_ch=1, proj_ch=cfg["proj_ch"], base=32,
                   backbone_input_size=cfg["backbone_input_size"])
    return G, D


if __name__ == "__main__":
    run(CFG, model_fn)
