"""
Zero-shot pretrained diffusion inpainting baseline on the REAL brush task.

Motivation: a reviewer-anticipation question -- LaMa is claimed decisive
but never compared against modern diffusion inpainting SOTA. This directly
parallels the project's own zero-shot big-lama test (which showed
near-blank-white collapse on manga) -- same logic, same value: a real,
cheap, honest data point for the manuscript's baseline table, whichever
way it lands. This model class is deliberately NOT edge-deployable
regardless of parameter count (iterative multi-step sampling, unlike every
generator this project ships), which is itself part of the argument, not a
disqualifier for running it as a comparison.

Model: `StableDiffusionInpaintPipeline`, zero-shot, NO fine-tuning --
inference only, nothing trained, nothing saved as a checkpoint.

**MODEL_ID**: `stable-diffusion-v1-5/stable-diffusion-inpainting`. The
original `runwayml/stable-diffusion-v1-5` org was taken down in 2024 over
an IP/legal dispute; a dedicated `stable-diffusion-v1-5` HF org now hosts
successor mirrors (including this inpainting checkpoint) specifically to
survive that -- a community-maintained mirror is less likely to carry
Stability AI's own license-gating than their own org's repos. If this
401s, set an `HF_TOKEN` environment variable (see the login cell below) --
add the token, re-run, no code edit needed -- or fall back to one of the
alternatives listed next to MODEL_ID below.

Domain adaptation, reusing conventions already established elsewhere in
this codebase rather than inventing new ones:
- grayscale [-1,1] -> [0,1] -> RGB via R=G=B replicate, the same trick
  `mangainpaint/model_lama.py`'s `LamaTransferG` uses for its own frozen
  photo-domain backbone.
- resize to SD's native working resolution (512px) for the pipeline call,
  resize the output back to this project's native 384px for scoring -- so
  these numbers are directly comparable to every other checkpoint's
  `eval_brush_batch.py` numbers (same MASK_SEED, num_workers=0,
  mask_balloon_prob=0, same test.csv rows, same metric functions).
- mask: this project's "1=hole" convention already matches diffusers' own
  (white=inpaint region) -- no inversion needed.
- output RGB -> grayscale via the same luminance formula
  `mangainpaint/checkpoint_registry.py`'s `PConvWrapper` uses (0.299/0.587/0.114).

Requires a GPU with enough VRAM for the SD1.5 pipeline in fp16 (a few GB);
CPU inference is impractically slow for this model class.

Subset deliberately small (150 pages) -- zero-shot diffusion inference is
comparatively slow (roughly 20-40s/image on a fast scheduler); the full
907-page test.csv would cost several hours for one baseline number.

Run:
    python eval/eval_sd15_zeroshot.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from mangainpaint.dataset import make_loaders
from mangainpaint.trainer import evaluate, LPIPS_AVAIL

try:
    import lpips as lpips_lib
except Exception:
    lpips_lib = None

HERE = os.path.dirname(os.path.abspath(__file__))
# Set these to your local Manga109-s / Manga109-segmentation roots, or
# export MANGA109_ROOT / MANGA109_SEG_ROOT instead.
ROOT_DIR = os.environ.get("MANGA109_ROOT", "./data/Manga109s")
SEG_ROOT = os.environ.get("MANGA109_SEG_ROOT", "./data/Manga109_segmentation")
OUT_JSON = os.path.join(HERE, "diffusion_zeroshot_results.json")

MASK_SEED = 1234    # identical to every other brush eval in this codebase -- same holes, comparable numbers
N_SUBSET = 150       # bump if time allows (see docstring for cost)
SD_SIZE = 512         # SD2-inpainting's native working resolution
NUM_STEPS = 20        # DPM++ 2M (fast scheduler) step count -- quality/speed tradeoff, not re-tuned
GUIDANCE_SCALE = 1.0  # near-unconditional fill (empty prompt) -- a real guidance value would push
                      # toward a specific text-conditioned look, wrong for a blind content fill
# Fallback chain, edit if this one 401s too:
#   1. "stable-diffusion-v1-5/stable-diffusion-inpainting"  <- current default
#   2. "sd-legacy/stable-diffusion-inpainting"               (another current mirror)
#   3. "stabilityai/stable-diffusion-2-inpainting"            (needs an HF_TOKEN with license access)
MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"
GEN_SEED = 42


class DiffusionZeroShotG(nn.Module):
    """Wraps a diffusers `StableDiffusionInpaintPipeline` in this codebase's
    shared `forward(x)` contract (x=[B,2,H,W]: 1ch masked grayscale [-1,1] +
    1ch mask, out [B,1,H,W] [-1,1]) so `mangainpaint.trainer.evaluate` scores it
    with zero new metric code -- the same function, same math, every other
    checkpoint in this project has been scored with."""

    def __init__(self, pipe, sd_size=SD_SIZE, num_steps=NUM_STEPS,
                guidance_scale=GUIDANCE_SCALE, seed=GEN_SEED):
        super().__init__()
        self.pipe = pipe
        self.sd_size = sd_size
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
        img_rgb_r = F.interpolate(img_rgb, size=(self.sd_size, self.sd_size),
                                  mode="bilinear", align_corners=False).clamp(0, 1)
        mask_r = F.interpolate(mask.float(), size=(self.sd_size, self.sd_size), mode="nearest")

        pil_imgs = [TF.to_pil_image(img_rgb_r[b].cpu()) for b in range(B)]
        pil_masks = [TF.to_pil_image(mask_r[b].cpu()) for b in range(B)]

        gen = torch.Generator(device=device if device.type == "cuda" else "cpu").manual_seed(self.seed)
        # Diffusion's own internal precision is pipeline-managed (fp16
        # weights loaded explicitly below); disabling ambient autocast here
        # avoids double-casting, same defensive pattern this codebase uses
        # around every other frozen pretrained-photo-domain backbone.
        with torch.amp.autocast(device.type, enabled=False):
            result = self.pipe(
                prompt=[""] * B, image=pil_imgs, mask_image=pil_masks,
                num_inference_steps=self.num_steps, guidance_scale=self.guidance_scale,
                height=self.sd_size, width=self.sd_size,
                generator=gen, output_type="pt",
            ).images

        # `output_type="pt"` returns a (B,3,H,W) tensor in modern diffusers;
        # defensive fallback to a PIL-list return for older/mismatched
        # versions, since this path is untested against the real package.
        if torch.is_tensor(result):
            out = result
        else:
            out = torch.stack([TF.to_tensor(im) for im in result])

        out = out.to(device=device, dtype=torch.float32)
        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        gray01 = 0.299 * out[:, 0:1] + 0.587 * out[:, 1:2] + 0.114 * out[:, 2:3]
        return (gray01.clamp(0, 1) * 2 - 1).clamp(-1, 1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "diffusion inference on CPU is impractically slow"
    print(f"device = {device}\nZero-shot diffusion inpainting baseline, {N_SUBSET}-image brush subset\n")

    from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler

    # Optional HF auth -- a no-op unless HF_TOKEN is set in the environment.
    # Needed only if MODEL_ID turns out to require a click-through
    # license/token (see the fallback chain above).
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
        print("Logged in to Hugging Face Hub via HF_TOKEN.")

    print(f"Loading {MODEL_ID} (fp16)...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()  # VRAM safety margin on a modest GPU, minor speed cost
    pipe.enable_vae_slicing()        # batched VAE decode memory safety (batch_size=4 below)

    G = DiffusionZeroShotG(pipe).to(device)

    cfg = {
        "root_dir":  ROOT_DIR,
        "train_csv": os.path.join(ROOT_DIR, "train.csv"),
        "val_csv":   os.path.join(ROOT_DIR, "val.csv"),
        "test_csv":  os.path.join(ROOT_DIR, "test.csv"),
        "seg_root":  SEG_ROOT,
        "image_size": 384,
        "batch_size": 4,
        "num_workers": 0,          # main-process mask RNG -> reproducible, same convention as every brush eval
        "mask_balloon_prob": 0.0,  # pure brush -- the real task
        "mask_perlin_prob": 0.0,
        "mask_brush_w_min": 7, "mask_brush_w_max": 25,
        "mask_strokes_min": 1, "mask_strokes_max": 4,
        "mask_len_min": 20, "mask_len_max": 90,
        "mask_large_prob": 0.20, "mask_large_frac": 0.25,
        "hole_fill": "white",
        "lpips_eval_net": "vgg",
    }

    print("Building caches + shared brush-masked test loader...")
    _, _, test_loader, _ = make_loaders(cfg, rank=0, world_size=1)

    # Subsample to N_SUBSET rows -- same precedent as the RISE run (full
    # 907-page test.csv would cost several hours at diffusion inference
    # speed for one baseline number). Mutate the dataset's frame BEFORE
    # building the final loader, not after -- make_loaders' own test_loader
    # is DistributedSampler-wrapped against the full length, so reusing it
    # post-mutation would index out of bounds; build a fresh plain loader.
    ds = test_loader.dataset
    if len(ds) > N_SUBSET:
        ds.data = ds.data.sample(n=N_SUBSET, random_state=MASK_SEED).reset_index(drop=True)
    print(f"Evaluating on {len(ds)} pages")
    test_loader = torch.utils.data.DataLoader(ds, batch_size=cfg["batch_size"],
                                              shuffle=False, num_workers=0)

    lpips_eval_fn = None
    if LPIPS_AVAIL and lpips_lib is not None:
        lpips_eval_fn = lpips_lib.LPIPS(net=cfg["lpips_eval_net"], verbose=False).to(device)
        for p in lpips_eval_fn.parameters():
            p.requires_grad_(False)

    np.random.seed(MASK_SEED)
    with torch.no_grad():
        r = evaluate(G, test_loader, lpips_eval_fn, device, rank=0, desc="diffusion-zeroshot")

    result = {
        "model_id": MODEL_ID, "n_images": len(ds), "sd_size": SD_SIZE,
        "num_steps": NUM_STEPS, "guidance_scale": GUIDANCE_SCALE,
        "psnr": r["psnr"], "ssim": r["ssim"], "grad_l1": r["grad_l1"],
        "edge_f1": r["edge_f1"], "lpips": r["lpips"],
        "strata": {k: {"n": v["n"], "edge_f1": v["edge_f1"], "lpips": v["lpips"]}
                  for k, v in r["strata"].items()},
    }
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== Zero-shot {MODEL_ID}, pure brush-stroke, {len(ds)} pages ===")
    print(f"EdgeF1={r['edge_f1']:.4f} LPIPS={r['lpips']:.4f} PSNR={r['psnr']:.2f} SSIM={r['ssim']:.4f}")
    print("--- reference (same protocol, full 907-page test.csv) ---")
    print("  lama_transfer_test_v4 (51M, fine-tuned): EdgeF1 0.4757")
    print("  from-scratch pack (2.7-2.8M):            EdgeF1 0.42-0.44")
    print("  pconv_baseline_v1 (worst real checkpoint): EdgeF1 0.2963")
    print(f"\nSaved {OUT_JSON}")


if __name__ == "__main__":
    main()
