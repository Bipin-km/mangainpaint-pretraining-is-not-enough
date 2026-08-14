"""
S2 -- slim LaMa student + DISTILLATION from the fine-tuned LaMa teacher.
The main result. Cell 2 of 3 (Axis A7).

Identical to `student_s1.py` except the two distillation loss
terms are switched on and the generator exposes its bottleneck. So S2 vs. S1
isolates exactly one variable: the teacher.

**Why distillation is a new hypothesis here.** An earlier attempt to
"distill from LaMa" was closed out because *zero-shot* big-lama collapses
to near-blank white on manga, i.e. a useless teacher (see paper §6.2).
That objection doesn't apply here: the teacher used is a
**manga-fine-tuned** LaMa that is the best content reconstructor in the
project on the real brush task (EdgeF1 0.4757, wins every metric and every
ink-density stratum). Distilling that is a different experiment.

And it is the on-thesis one: big-lama completes complex shapes well because
it saw millions of images -- exactly what 6,788 manga pages cannot teach a
small net from scratch. Distillation is the mechanism that moves that
capability into a model small enough to ship. What ships is still just the
8.8M student; the 51M teacher and the +0.13M `distill_adapter` are both
training-only and thrown away.

Four terms (mangainpaint/distill.py), i.e. "distillation done properly" rather
than the naive L1-only recipe both WKD's and ReKo's introductions open by
warning against ("directly minimizing the distance between the generated
images of students and teachers does not improve, but sometimes harms, GAN
performance"):
- `p*_w_distill_out`     -- hole-weighted L1 to the teacher's output.
- `p*_w_distill_feat`    -- L1 between the student's bottleneck (lifted
  256->512 by `distill_adapter`) and the teacher's. Output KD alone tends to
  transfer mean behaviour but not internal structure; the feature term is
  what actually moves the shape priors.
- `p*_w_distill_wavelet` -- **Wavelet Knowledge Distillation** (Zhang et
  al., CVPR 2022): L1 restricted
  to the HIGH-frequency 3-level Haar-DWT subbands of student vs. teacher
  output. The paper's own motivation (their Fig 1) is that GANs already
  match GT low frequency almost perfectly and fail specifically on high
  frequency -- for manga this argument is sharper than for their own domain
  (shoes/zebras): ink lines and screentone dots ARE the high-frequency
  band, a bitonal page has almost no meaningful low-frequency content.
- `p*_w_distill_patchnce` -- **Region-aware Knowledge Distillation / ReKo**
  (Zhang et al., BMVC 2023): patch-wise InfoNCE between student/teacher
  bottleneck features, restricted to hole positions (student=query, teacher-same-
  position=positive, teacher-other-hole-position=negative, no memory bank
  -- exactly the paper's eq. 2/4). ReKo finds its "crucial regions" via an
  unsupervised attention module because general image translation has no
  ground truth for "the region that matters"; we use the mask directly,
  strictly more reliable than their proxy.

Both WKD and ReKo were real, verified, "later-phase" distillation
techniques identified early in the project's research but deferred until
distillation was in scope. This recipe is that later phase.

Weights are deliberately modest relative to the reconstruction terms
(hole_rec 5.0/4.0): the GT is still the primary target and the teacher is a
*prior*, not an oracle -- it is itself only 0.4757 EdgeF1, so weighting it
too hard would cap the student at the teacher's own errors. Distillation is
also strongest in phase 1 (pre-GAN) and eased off in phase 2, so the
adversarial signal isn't fighting the teacher's smoother fills late in
training. `distill_patchnce`'s weight is set lower than `distill_feat`'s
because its InfoNCE range (0 to ln(max_positions)=~5.5) is naturally larger
than `feat_kd`'s raw L1 (measured ~7 at step 0, but denser/noisier
signal) -- unweighted magnitudes were checked once
(`d_out=0.46 d_feat=7.15 d_wav=0.45 d_patch=5.56` at random student init;
the cited papers report low weight-sensitivity, so this wasn't
independently re-tuned beyond keeping the four terms in a comparable range).

**TEACHER CHECKPOINT required.** `distill_teacher_ckpt` must point at the
fine-tuned teacher's `best.pt` (a trainer.py checkpoint whose "G" key
holds `LamaTransferG`'s state_dict), produced by
`recipes/teacher_lama_finetune.py`. That run must finish first.

Run:
    torchrun --standalone --nproc_per_node=<N> recipes/student_s2_distill.py
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

# Set this to your local Manga109-s / Manga109-segmentation roots, or
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

    # ── Distillation (the ONLY change vs. S1) ──
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

SLIM_NGF = 32
SLIM_NBLOCKS = 12
SLIM_INIT = "random"


def model_fn(cfg):
    # expose_bottleneck=True builds `distill_adapter` and stashes the
    # post-resblock bottleneck each forward, which mangainpaint/distill.py reads.
    # It is used on every step (both distill weights > 0), so DDP's
    # find_unused_parameters=False stays valid.
    G = LamaSlimG(ngf=SLIM_NGF, n_blocks=SLIM_NBLOCKS, init_mode=SLIM_INIT,
                  expose_bottleneck=True, use_screenvae_hint=False)
    D = ProjectedD(mask_ch=1, proj_ch=cfg["proj_ch"], base=32,
                   backbone_input_size=cfg["backbone_input_size"])
    return G, D


if __name__ == "__main__":
    run(CFG, model_fn)
