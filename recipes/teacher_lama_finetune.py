"""
LaMa-transfer BRUSH confirmation run -- the canonical, airtight version of
the paper's central finding (see §6.1/§6.2): fine-tuning a pretrained LaMa
checkpoint decisively wins the real (pure-brush, over-recoverable-art)
inpainting task over every from-scratch architecture (EdgeF1 0.4757 vs. a
0.42-0.44 from-scratch pack, all metrics, all strata). Earlier LaMa-transfer
runs (this recipe's ancestors) established two things needed to get a
clean number:

- Fully-unfrozen fine-tuning (`freeze_up_to=0`) is required -- partial-freeze
  variants (`freeze_up_to` 8/16) land at 0.330-0.398 on the brush eval,
  BELOW the from-scratch pack. Transfer only wins fully unfrozen.
- Brush-mask generation needs to exclude real dialogue text (no ground
  truth exists under printed text); an earlier bug didn't.

This run combines both: fully-unfrozen (`freeze_up_to=0`) LaMa-transfer
trained on 100% brush masks (`mask_balloon_prob=0`), post-fix. It is the
canonical LaMa number reported in the manuscript's results table
("Fine-tuned LaMa, teacher" row).

`seg_root`/`balloon_extra_stroke_prob` are left in CFG but inert -- with
`mask_balloon_prob=0`, `make_loaders` never builds the balloon cache and
`generate_balloon_mask` is never called (verified: mangainpaint/dataset.py's
make_loaders gates balloon-cache build on `mask_balloon_prob > 0`).

Run:
    torchrun --standalone --nproc_per_node=<N> recipes/teacher_lama_finetune.py
Resume:
    Set CFG["resume"] = "checkpoints/last.pt" and re-run.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mangainpaint.trainer import run
from mangainpaint.model_lama import LamaTransferG
from mangainpaint.model_projected_d import ProjectedD

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

    "ink_threshold": 0.4, "ink_extra": 2.0,
    "d_refresh_every": 10,

    "lr_g": 4e-4, "lr_d": 5e-5,

    "mask_brush_w_min": 7, "mask_brush_w_max": 25,
    "mask_strokes_min": 1, "mask_strokes_max": 4,
    "mask_len_min": 20, "mask_len_max": 90,
    "mask_large_prob": 0.20, "mask_large_frac": 0.25,

    # Pure B1 brush masking -- the whole point of this run.
    # seg_root/balloon_extra_stroke_prob below are inert with this at 0
    # (make_loaders won't build the balloon cache; generate_balloon_mask
    # never fires) -- kept only for CFG-shape consistency with the
    # balloon-trained recipes.
    "seg_root": MANGA109_SEG_ROOT,
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


def model_fn(cfg):
    # freeze_up_to=0 -- 100% unfrozen, the only freeze fraction that wins on
    # the real task (partial-freeze variants land below the from-scratch
    # pack; see the module docstring).
    G = LamaTransferG(freeze_up_to=0)
    D = ProjectedD(mask_ch=1, proj_ch=cfg["proj_ch"], base=32,
                   backbone_input_size=cfg["backbone_input_size"])
    return G, D


if __name__ == "__main__":
    run(CFG, model_fn)
