"""
Held-out test.csv evaluation under brush-stroke masking, for exactly the
13 checkpoints reported in the paper: Table 1's 11 rows (the "scratch"
architectures, the S1/S1-attn/S2/S3/C1/C2 distillation ladder, and the
fine-tuned teacher) plus the two distillation-signal ablations of the
S2-GN/S2-VAE table. This is the script that produces the raw per-checkpoint
numbers both tables are built from. See README's recipe/run-id table for
how `CHECKPOINTS` below maps to `recipes/`.

Fairness: one shared test_loader with `mask_balloon_prob=0.0`,
`num_workers=0`, and `np.random.seed(...)` reset before each checkpoint's
pass -- so every checkpoint is scored on byte-identical brush-stroke holes
(the mask RNG lives in the main process at that config). At eval
(`is_train=False`) `_procedural_mask_np` always draws brush strokes, never
the large-box branch, so this is a clean brush-only distribution. Every
model here was trained on this same brush-only distribution (`train_masking`
is printed alongside each result to make that explicit), so there is no
in-distribution/out-of-distribution split to read apart.

Usage:
    python eval_brush_batch.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from mangainpaint.dataset import make_loaders
from mangainpaint.trainer import evaluate, LPIPS_AVAIL
from mangainpaint.checkpoint_registry import build_generator, load_generator_state_dict

try:
    import lpips as lpips_lib
except ImportError:
    lpips_lib = None

HERE = os.path.dirname(os.path.abspath(__file__))
# Set these to your local Manga109-s / Manga109-segmentation roots, or
# export MANGA109_ROOT / MANGA109_SEG_ROOT instead.
ROOT_DIR = os.environ.get("MANGA109_ROOT", "./data/Manga109s")
SEG_ROOT = os.environ.get("MANGA109_SEG_ROOT", "./data/Manga109_segmentation")
# Root under which each checkpoint below is expected at <CKPT_ROOT>/<name>/best.pt
# (the layout release checkpoints unpack into; see README's Checkpoints section).
CKPT_ROOT = os.environ.get("MANGA_CKPT_ROOT", os.path.join(HERE, "..", "checkpoints"))
OUT_JSON = os.path.join(HERE, "brush_eval_results.json")
MASK_SEED = 1234

# 13-checkpoint roster: exactly Table 1's 11 rows plus the S2-GN/S2-VAE
# distillation-signal ablations, named after the original run ids (see
# README's recipe/run-id table for how these map to the recipes shipped
# here). (exp_name, arch, short label).
CHECKPOINTS = [
    ("pconv_baseline_v1",      "pconv",           "PConv-UNet"),
    ("uffc_test_kaggle_v2",    "uffc",            "UFFC-GAN"),
    ("attn2_test_v2",          "attn_noffc",      "CtxAttn-GAN"),
    ("projected_d_test_v2",    "vanilla",         "FFC-GAN"),
    ("lama_slim_s1_attn",      "lama_slim_attn",  "S1-attn (windowed self-attn)"),
    ("lama_slim_s1",           "lama_slim",       "S1 (no distillation)"),
    ("lama_slim_c2_fusion",    "lama_slim_fus",   "C2 (+distill., 3.5M fusion)"),
    ("lama_slim_c1_compact",   "lama_slim",       "C1 (+distill., compact)"),
    ("lama_distill_s2",        "lama_slim",       "S2 (+distillation)"),
    ("lama_distill_s3",        "lama_slim",       "S3 (+external losses)"),
    ("lama_transfer_brush_v1", "lama",            "Fine-tuned LaMa (teacher)"),
    ("lama_distill_s2_gn",     "lama_slim",       "S2-GN (adaptive KD weights, ablation)"),
    ("lama_distill_s2_svaekd", "lama_slim",       "S2-VAE (+ScreenVAE-latent KD, ablation)"),
]

LOCAL_PATH_OVERRIDES = {
    "root_dir":  ROOT_DIR,
    "train_csv": os.path.join(ROOT_DIR, "train.csv"),
    "val_csv":   os.path.join(ROOT_DIR, "val.csv"),
    "test_csv":  os.path.join(ROOT_DIR, "test.csv"),
    "seg_root":  SEG_ROOT,
    "num_workers": 0,      # main-process mask RNG -> reproducible across checkpoints
    "batch_size": 4,
    "mask_balloon_prob": 0.0,   # <-- the whole point: pure brush strokes, no balloon
    "mask_perlin_prob": 0.0,    # is_train=False gates this anyway, belt-and-suspenders
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}\nPURE BRUSH-STROKE (Axis B1) held-out test.csv eval, all {len(CHECKPOINTS)} checkpoints\n")

    # Build the shared loader once, from a reference checkpoint's cfg (all
    # checkpoints share image_size=384/base=32; only masking + paths are
    # overridden, and masking is forced to brush for everyone).
    ref_ckpt = torch.load(os.path.join(CKPT_ROOT, CHECKPOINTS[0][0], "best.pt"),
                          map_location="cpu")
    cfg = dict(ref_ckpt.get("cfg", {}))
    cfg.update(LOCAL_PATH_OVERRIDES)

    print("Building caches + shared brush-masked test loader...")
    _, _, test_loader, _ = make_loaders(cfg, rank=0, world_size=1)

    lpips_eval_fn = None
    if LPIPS_AVAIL and lpips_lib is not None:
        lpips_eval_fn = lpips_lib.LPIPS(net=cfg.get("lpips_eval_net", "vgg"), verbose=False).to(device)
        for p in lpips_eval_fn.parameters():
            p.requires_grad_(False)

    results = {}
    for exp_name, arch, label in CHECKPOINTS:
        ckpt = torch.load(os.path.join(CKPT_ROOT, exp_name, "best.pt"),
                          map_location=device)
        train_balloon_prob = dict(ckpt.get("cfg", {})).get("mask_balloon_prob", 0.0)
        G = build_generator(arch, dict(ckpt.get("cfg", {})), device)
        missing, unexpected = load_generator_state_dict(G, arch, ckpt["G"], strict=False)
        G.eval()

        # Identical brush masks for every checkpoint: reset the RNG the
        # dataset draws from before each single-pass evaluate().
        np.random.seed(MASK_SEED)
        with torch.no_grad():
            r = evaluate(G, test_loader, lpips_eval_fn, device, rank=0, desc="test")

        train_kind = "B1" if train_balloon_prob == 0 else f"B2(p={train_balloon_prob})"
        results[exp_name] = {
            "arch": arch, "label": label, "train_masking": train_kind,
            "load_missing": len(missing), "load_unexpected": len(unexpected),
            "psnr": r["psnr"], "ssim": r["ssim"], "grad_l1": r["grad_l1"],
            "edge_f1": r["edge_f1"], "lpips": r["lpips"],
            "strata": {k: {"n": v["n"], "edge_f1": v["edge_f1"], "lpips": v["lpips"]}
                       for k, v in r["strata"].items()},
        }
        with open(OUT_JSON, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  {exp_name:<28} [{train_kind:<9}] EdgeF1={r['edge_f1']:.4f} "
              f"LPIPS={r['lpips']:.4f} PSNR={r['psnr']:.2f} SSIM={r['ssim']:.4f} "
              f"(load {len(missing)}/{len(unexpected)})")
        del G
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n=== PURE BRUSH-STROKE (B1) test.csv ranking, best EdgeF1 first ===")
    print(f"{'checkpoint':<28} {'train':<10} {'EdgeF1':>8} {'LPIPS':>8} {'PSNR':>7} {'SSIM':>7}")
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["edge_f1"]):
        print(f"{name:<28} {r['train_masking']:<10} {r['edge_f1']:>8.4f} "
              f"{r['lpips']:>8.4f} {r['psnr']:>7.2f} {r['ssim']:>7.4f}")
    print(f"\nSaved {OUT_JSON}")


if __name__ == "__main__":
    main()
