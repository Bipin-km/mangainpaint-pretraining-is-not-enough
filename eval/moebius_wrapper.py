"""
Moebius (Duan et al., ECCV 2026, arXiv:2606.19195) zero-shot inpainting
wrapper, in the same `forward(x)` contract every other generator in this
codebase uses (x=[B,2,H,W]: 1-channel masked grayscale in [-1,1] plus a
1-channel hole mask, output [B,1,H,W] in [-1,1]) so `mangainpaint.trainer
.evaluate` scores it with no new metric code, mirroring
`eval_sd15_zeroshot.py`'s `DiffusionZeroShotG` for the SD1.5 baseline.

Requires Moebius's own inference code (not shipped here -- it is a
separate research codebase with its own pinned dependency stack that
collides with this project's, so it is a local vendor directory rather
than a pip dependency): clone `https://github.com/hustvl/Moebius` to
`MOEBIUS_ROOT` (default `external/moebius`, override with the `MOEBIUS_ROOT`
environment variable) and fetch its weights per its own instructions
(`weight/vae/` from `hustvl/PixelHacker`'s `vae/` subfolder, since Moebius's
own repo does not bundle one, and `weight/Moebius/pretrained/` from
`hustvl/Moebius`). Run this under a venv built from Moebius's own
`requirements.txt`, not this project's.

Fixed-resolution architecture: Moebius's attention block condenses spatial
context into fixed-size linear matrices tied to its trained 512px / 64x64-
latent geometry, so it is run at native 512px and resized back to this
project's 384px for scoring, the same pattern `DiffusionZeroShotG` uses for
SD1.5.
"""
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

MOEBIUS_ROOT = os.environ.get(
    "MOEBIUS_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "external", "moebius"),
)  # <- your local clone of https://github.com/hustvl/Moebius, or export MOEBIUS_ROOT

MOEBIUS_SIZE = 512  # Moebius's native/fixed working resolution, not configurable
NUM_STEPS = 20
GUIDANCE_SCALE = 2.0  # Moebius's own recommended `--cfg 2.0` inference default
GEN_SEED = 42


def build_moebius_pipe(device):
    """Builds Moebius's `RemovalSDXLPipeline_BatchMode` from its own config
    and weights under `MOEBIUS_ROOT`. Imports Moebius's `infer.utils` lazily,
    since that module only exists once `MOEBIUS_ROOT` is populated."""
    if MOEBIUS_ROOT not in sys.path:
        sys.path.insert(0, MOEBIUS_ROOT)
    cwd = os.getcwd()
    os.chdir(MOEBIUS_ROOT)  # weight/vae, weight/Moebius/... paths in its yaml config are relative
    try:
        from infer.utils import build_pipeline
        args = SimpleNamespace(
            model_config=os.path.join(MOEBIUS_ROOT, "config", "model_cfg", "moebius.yaml"),
            model_weight=os.path.join(MOEBIUS_ROOT, "weight", "Moebius", "pretrained",
                                       "diffusion_pytorch_model.bin"),
            device=device,
        )
        pipe = build_pipeline(args)
    finally:
        os.chdir(cwd)
    return pipe


class MoebiusZeroShotG(nn.Module):
    """Wraps Moebius's pipeline in this codebase's shared `forward(x)`
    contract so `mangainpaint.trainer.evaluate` scores it with no new
    metric code, the same pattern `DiffusionZeroShotG` uses for SD1.5."""

    def __init__(self, pipe, size=MOEBIUS_SIZE, num_steps=NUM_STEPS,
                 guidance_scale=GUIDANCE_SCALE, seed=GEN_SEED):
        super().__init__()
        self.pipe = pipe
        self.size = size
        self.num_steps = num_steps
        self.guidance_scale = guidance_scale
        self.seed = seed

    @torch.no_grad()
    def forward(self, x):
        masked, mask = x[:, 0:1], x[:, 1:2]
        B, _, H, W = masked.shape
        device = masked.device

        img01 = ((masked + 1) / 2).clamp(0, 1)
        img_rgb = img01.repeat(1, 3, 1, 1)
        img_rgb_r = F.interpolate(img_rgb, size=(self.size, self.size),
                                   mode="bilinear", align_corners=False).clamp(0, 1)
        mask_r = F.interpolate(mask.float(), size=(self.size, self.size), mode="nearest")

        pil_imgs = [TF.to_pil_image(img_rgb_r[b].cpu()) for b in range(B)]
        pil_masks = [TF.to_pil_image(mask_r[b, 0].cpu()) for b in range(B)]

        torch.manual_seed(self.seed)
        out_pils = self.pipe(pil_imgs, pil_masks, image_size=self.size,
                              num_steps=self.num_steps, guidance_scale=self.guidance_scale)

        out = torch.stack([TF.to_tensor(im) for im in out_pils]).to(device=device, dtype=torch.float32)
        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        gray01 = 0.299 * out[:, 0:1] + 0.587 * out[:, 1:2] + 0.114 * out[:, 2:3]
        return (gray01.clamp(0, 1) * 2 - 1).clamp(-1, 1)
