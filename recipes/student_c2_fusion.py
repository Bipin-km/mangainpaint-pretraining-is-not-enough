"""
C2 -- FUSION student (~3.5M): narrow FFC (ngf=20) + one bottleneck
linear-attention pass, + S2's exact distillation recipe.

The "cleverest fusion + compaction" cell. Where C1 shrinks the FFC width to
ngf=24 (4.94M) and stops, C2 shrinks further to ngf=20 (3.44M FFC backbone)
and adds one cheap O(N) linear-attention global-mixing pass on the bottleneck
(~0.10M) to test whether cheap global context can buy back what the narrower
FFC gives up. Shipped student ~3.54M -- **-60% vs S2's 8.78M**.

The hypothesis: FFC's spectral branch is the
parameter-heavy part (75% of bottleneck channels go through the FFT-domain
conv); linear attention (Katharopoulos et al. 2020) is parameter-light and
also mixes globally. So this is a genuine architectural trade, not a bolt-on:
replace some expensive FFC width with one cheap global pass. If C2 (~3.5M)
matches C1 (4.94M) and S2 (8.78M) on held-out test.csv, the fusion compacts
the deliverable to ~3.5M. See `mangainpaint/model_lama_slim_fus.py` for the full
design rationale (one global valid-source-masked pass on the FINAL
bottleneck, not interleaved per-block and not a third encoder branch -- both
of those forms were already tested and rejected/within-noise).

Note on the framing ("it's the added attention"): the held-out strata do NOT
show attention beating FFC. Section 6.1 reports the opposite -- swapping the
student's FFC bottleneck for windowed self-attention at matched budget
(S1-attn) is the worst slim variant in the study, -0.0132 EdgeF1 and lower in
every stratum. So C2 is NOT a bet that attention is a better bottleneck than
FFC; it keeps FFC (the
one op with a domain argument -- spectral mixing for periodic screentone) and
adds attention only as a *cheap width-substitute* for compaction. Distillation
remains the actual performance lever; both C1 and C2 carry it identically.

Everything except the generator is byte-for-byte S2 (same teacher, same
four-term distillation, same weights/masking/schedule), so C2-vs-C1 isolates
"narrow-FFC + linattn vs wider-FFC" at matched distillation.

`ngf`/`n_blocks`/`linattn_heads` are in CFG so they persist into the
checkpoint and `checkpoint_registry.build_generator("lama_slim_fus", cfg)`
rebuilds the right model at eval. The `distill_adapter` lifts the 160ch
bottleneck -> the teacher's 512; distillation feature-matches the POST-linattn
bottleneck (the student's final bottleneck representation).

**TEACHER CHECKPOINT.** Same fine-tuned teacher's best.pt S2/S3/C1 used.

Run:
    torchrun --standalone --nproc_per_node=<N> recipes/student_c2_fusion.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mangainpaint.trainer import run
from mangainpaint.model_lama_slim_fus import LamaSlimFusG
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

    # ── Compaction + fusion: the change vs. S2 ──
    "ngf": 20, "n_blocks": 12, "linattn_heads": 4,

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
    G = LamaSlimFusG(ngf=cfg["ngf"], n_blocks=cfg["n_blocks"],
                     linattn_heads=cfg["linattn_heads"], init_mode="random",
                     expose_bottleneck=True, use_screenvae_hint=False)
    D = ProjectedD(mask_ch=1, proj_ch=cfg["proj_ch"], base=32,
                   backbone_input_size=cfg["backbone_input_size"])
    return G, D


if __name__ == "__main__":
    run(CFG, model_fn)
