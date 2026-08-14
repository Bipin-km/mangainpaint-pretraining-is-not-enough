"""
UFFC-GAN: the paper's unbiased-Fourier-convolution from-scratch variant
(3.10M parameters, EdgeF1 0.424). Identical to the reference recipe
`ffc_gan.py` in every respect except the generator, so this is a
single-variable architecture comparison.

The generator is `MangaFillNetUFFC` (mangainpaint/model_uffc.py): the same
encoder-decoder as the FFC-GAN reference with `f1`/`f2` swapped from
`FFCBlock` to `UFFCBlock` (Chu et al., ICCV 2023). UFFC suppresses vanilla
FFC's specific ringing pattern but produces its own content-independent
replacement, which is the observation Section 6.1 reports: three
architecturally distinct from-scratch bottlenecks span 0.424-0.430, a range
narrower than the scatter expected between two runs of one configuration.

`image_size` is passed to the generator because `UFFCBlock`'s `loc_map`
parameter shape is fixed at construction from the bottleneck resolution
(image_size / 8, from the three stride-2 encoder blocks), not lazily
inferred. It must match `CFG["image_size"]`.

Full-corpus training on the 6,788-page train split under the common recipe:
50 epochs, adversarial phase from epoch 10, brush-stroke procedural masking
with annotated text regions excluded. No balloon-shaped masking and no
subset training -- every number the paper reports for this row comes from
this configuration.

Run:
    torchrun --standalone --nproc_per_node=<N> recipes/uffc_gan.py
Resume:
    Set CFG["resume"] = "checkpoints/last.pt" and re-run.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mangainpaint.trainer import run
from mangainpaint.model_uffc import MangaFillNetUFFC
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

    "ink_threshold": 0.4, "ink_extra": 2.0,
    "d_refresh_every": 10,

    "lr_g": 4e-4, "lr_d": 5e-5,

    "mask_brush_w_min": 7, "mask_brush_w_max": 25,
    "mask_strokes_min": 1, "mask_strokes_max": 4,
    "mask_len_min": 20, "mask_len_max": 90,
    "mask_large_prob": 0.20, "mask_large_frac": 0.25,

    # Brush-stroke masking only -- mask_balloon_prob omitted/0.

    "base": 32, "ratio_g": 0.5, "ring_radius": 5,  # ring_radius required by trainer.py's make_ring() -- do not omit

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
    G = MangaFillNetUFFC(in_ch=2, base=cfg["base"], ratio_g=cfg["ratio_g"],
                         image_size=cfg["image_size"])
    D = ProjectedD(mask_ch=1, proj_ch=cfg["proj_ch"], base=cfg["base"],
                   backbone_input_size=cfg["backbone_input_size"])
    return G, D


if __name__ == "__main__":
    run(CFG, model_fn)
