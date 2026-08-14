"""
CtxAttn-GAN: the paper's contextual-attention from-scratch variant
(2.70M parameters, EdgeF1 0.429). Identical to the reference recipe
`ffc_gan.py` in every respect except the generator, so this is a
single-variable architecture comparison.

The generator is `MangaFillNetAttnNoFFC` (mangainpaint/model_attn.py),
which *replaces* the spectral bottleneck rather than adding a branch
beside it: `f1`/`f2` (`FFCBlock`) are removed entirely, leaving
dilated-residual blocks plus a `ContextualAttentionBlock` and no
frequency-domain operator anywhere in the network. That is what the
paper means by "replace spectral operators entirely with contextual
attention", and it is why this variant is *smaller* than the 2.84M
FFC-GAN reference rather than larger.

Motivation: both vanilla FFC and UFFC leave a fixed, content-independent
periodic texture in every hole fill -- structural to routing the
bottleneck through a *global* frequency-domain operator. Contextual
Attention (Yu et al., CVPR 2018) has no such global transform: each hole
location is reconstructed as a similarity-weighted average of *actual*
valid-region feature patches. See `mangainpaint/model_attn.py`'s
docstring for the mechanism and the simplifications vs. the original
paper.

The attention block is more compute-heavy than FFC/UFFC (per-sample
patch-similarity/softmax/reconstruction), so this run takes longer
wall-clock than the plain-FFC reference recipe.

`use_fuse=False` and `p2_w_regional_stats=0.0` match the evaluated
checkpoint; both are deliberately inert so the architecture is the only
variable.

Run:
    torchrun --standalone --nproc_per_node=<N> recipes/ctxattn_gan.py
Resume:
    Set CFG["resume"] = "checkpoints/last.pt" and re-run.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mangainpaint.trainer import run
from mangainpaint.model_attn import MangaFillNetAttnNoFFC
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
    "epochs":      50,
    "betas":       (0.5, 0.999),
    "grad_clip":   1.0,
    "show_every":  5,

    "gan_phase_start": 10,

    "p1_w_hole_rec": 5.0, "p1_w_ring_rec": 3.0, "p1_w_valid_id": 0.5,
    "p1_w_edge": 3.0, "p1_w_fft": 3.0, "p1_w_lpips": 0.5, "p1_w_ink": 2.0,

    "p2_w_hole_rec": 4.0, "p2_w_ring_rec": 2.0, "p2_w_valid_id": 0.5,
    "p2_w_edge": 2.0, "p2_w_fft": 2.0, "p2_w_lpips": 0.3, "p2_w_ink": 2.0,
    "p2_w_gan": 1.0, "p2_w_fm": 2.0, "p2_w_r1": 1.0, "p2_r1_every": 4,
    "p2_w_regional_stats": 0.0,  # deliberately inert -- see module docstring

    "ink_threshold": 0.4, "ink_extra": 2.0,
    "d_refresh_every": 10,

    "lr_g": 4e-4, "lr_d": 5e-5,

    "mask_brush_w_min": 7, "mask_brush_w_max": 25,
    "mask_strokes_min": 1, "mask_strokes_max": 4,
    "mask_len_min": 20, "mask_len_max": 90,
    "mask_large_prob": 0.20, "mask_large_frac": 0.25,

    "base": 32, "ratio_g": 0.5, "ring_radius": 5,
    "dilations": (1, 2, 4, 8), "fuse_k": 3,

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
    G = MangaFillNetAttnNoFFC(in_ch=2, base=cfg["base"], ratio_g=cfg["ratio_g"],
                              dilations=cfg.get("dilations", (1, 2, 4, 8)),
                              fuse_k=cfg.get("fuse_k", 3), use_fuse=False)
    D = ProjectedD(mask_ch=1, proj_ch=cfg["proj_ch"], base=cfg["base"],
                   backbone_input_size=cfg["backbone_input_size"])
    return G, D


if __name__ == "__main__":
    run(CFG, model_fn)
