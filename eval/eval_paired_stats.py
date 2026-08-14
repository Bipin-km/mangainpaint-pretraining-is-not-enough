"""
Per-page scores + paired uncertainty for the main comparison.

The paper's cells are single training runs, so the honest question a
reviewer asks is: how much of a reported gap could be evaluation noise?
Every model is scored on the *same* frozen holes, which makes the
comparison paired -- so the sampling uncertainty of a between-model gap can
be quantified exactly, without retraining anything, by bootstrapping over
the 907 held-out pages.

This does NOT substitute for seed replication (it says nothing about
training-run variance). It bounds the other, separate source of noise, and
the paper says so explicitly.

Writes per-page metrics for every model plus paired bootstrap CIs and
Wilcoxon signed-rank tests for the contrasts the paper actually claims.

Run:  MANGA109_ROOT=/path/to/Manga109s python eval/eval_paired_stats.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, HERE)

import numpy as np
import torch

from mangainpaint.metrics import (hole_psnr, hole_ssim, hole_edge_f1, hole_ink_frac)
from mangainpaint.checkpoint_registry import build_generator, load_generator_state_dict
from mangainpaint.ddp_utils import call_g
from fixed_mask_protocol import FullPinnedDataset, PIN_FULL

try:
    import lpips as lpips_lib
except Exception:
    lpips_lib = None

# Root under which each checkpoint below is expected at <CKPT_ROOT>/<name>/best.pt
# (the layout release checkpoints unpack into; see README's Checkpoints section).
CKPT_ROOT = os.environ.get("MANGA_CKPT_ROOT", os.path.join(HERE, "..", "checkpoints"))
OUT_JSON = os.path.join(HERE, "paired_stats.json")
N_BOOT = 10000
RNG_SEED = 0

MODELS = [
    ("pconv_baseline_v1",      "pconv",        "PConv-UNet"),
    ("uffc_test_kaggle_v2",    "uffc",         "UFFC-GAN"),
    ("attn2_test_v2",          "attn_noffc",   "CtxAttn-GAN"),
    ("projected_d_test_v2",    "vanilla",      "FFC-GAN"),
    ("lama_slim_s1_attn",      "lama_slim_attn", "S1-attn"),
    ("lama_slim_s1",           "lama_slim",    "S1"),
    ("lama_distill_s2",        "lama_slim",    "S2"),
    ("lama_distill_s3",        "lama_slim",    "S3"),
    ("lama_transfer_brush_v1", "lama",         "Teacher"),
]

# (a, b, metric, higher_is_better) -- the gaps the paper claims.
# The S3-vs-FFC-GAN pair is included precisely because it comes out null on
# EdgeF1: the distilled student's advantage over the from-scratch reference
# is perceptual and computational, not structural, and the paper says so.
CONTRASTS = [
    ("lama_transfer_brush_v1", "projected_d_test_v2", "edge_f1", True),
    ("lama_transfer_brush_v1", "projected_d_test_v2", "lpips",   False),
    ("lama_distill_s2",        "lama_slim_s1",        "edge_f1", True),
    ("lama_distill_s2",        "lama_slim_s1",        "lpips",   False),
    ("lama_distill_s3",        "lama_distill_s2",     "lpips",   False),
    ("lama_distill_s3",        "lama_distill_s2",     "edge_f1", True),
    ("lama_distill_s3",        "projected_d_test_v2", "edge_f1", True),
    ("lama_distill_s3",        "projected_d_test_v2", "lpips",   False),
    ("lama_slim_s1",           "lama_slim_s1_attn",   "edge_f1", True),
    ("projected_d_test_v2",    "attn2_test_v2",       "edge_f1", True),
]


def ckpt_path(name):
    p = os.path.join(CKPT_ROOT, name, "best.pt")
    if os.path.exists(p):
        return p
    raise FileNotFoundError(p)


def score_model(G, loader, lpips_fn, device):
    import torch.nn.functional as F
    rows = {k: [] for k in ("psnr", "ssim", "edge_f1", "lpips", "ink_frac")}
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            mask = batch["mask"].to(device)
            with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
                out = call_g(G, batch["model_input"].to(device), batch, device)
            comp = out * mask + img * (1 - mask)
            if lpips_fn is not None:
                c = F.interpolate(comp, size=224, mode='bilinear', align_corners=False)
                i = F.interpolate(img,  size=224, mode='bilinear', align_corners=False)
                lp = lpips_fn(c.repeat(1, 3, 1, 1), i.repeat(1, 3, 1, 1)).view(-1).tolist()
            else:
                lp = [float('nan')] * img.size(0)
            for b in range(img.size(0)):
                p, t, m = comp[b:b+1], img[b:b+1], mask[b:b+1]
                rows["psnr"].append(hole_psnr(p, t, m))
                rows["ssim"].append(hole_ssim(p, t, m))
                rows["edge_f1"].append(hole_edge_f1(p, t, m))
                rows["ink_frac"].append(hole_ink_frac(t, m))
                rows["lpips"].append(lp[b])
    return rows


def paired_bootstrap(a, b, n_boot=N_BOOT, seed=RNG_SEED):
    """Percentile CI for mean(a) - mean(b) resampling *pages*, keeping the
    pairing (both models saw the identical hole on that page)."""
    d = np.asarray(a) - np.asarray(b)
    d = d[~np.isnan(d)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    # `--from-cache` recomputes the contrast table from the per-page scores
    # already stored in paired_stats.json. Adding or changing a contrast costs
    # no GPU time and cannot perturb the underlying measurements.
    from_cache = "--from-cache" in sys.argv
    if from_cache:
        prev = json.load(open(OUT_JSON))
        per_page = prev["per_page"]
        sha, n_pages = prev["mask_sha256"], prev["n_pages"]
        print(f"recomputing contrasts from cache: {n_pages} pages, "
              f"{len(per_page)} models, sha256={sha[:16]}...")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ds = FullPinnedDataset(PIN_FULL)
        loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
        print(f"device={device}  pages={len(ds)}  mask sha256={ds.sha[:16]}...")

        lpips_fn = None
        if lpips_lib is not None:
            lpips_fn = lpips_lib.LPIPS(net="vgg", verbose=False).to(device)
            for p in lpips_fn.parameters():
                p.requires_grad_(False)

        per_page = {}
        for name, arch, label in MODELS:
            ckpt = torch.load(ckpt_path(name), map_location=device, weights_only=False)
            G = build_generator(arch, dict(ckpt.get("cfg", {})), device)
            load_generator_state_dict(G, arch, ckpt["G"], strict=False)
            G.eval()
            rows = score_model(G, loader, lpips_fn, device)
            per_page[name] = rows
            print(f"  {label:<12} EdgeF1={np.nanmean(rows['edge_f1']):.4f} "
                  f"LPIPS={np.nanmean(rows['lpips']):.4f}")
            del G
            if device.type == "cuda":
                torch.cuda.empty_cache()
        sha, n_pages = ds.sha, len(ds)

    try:
        from scipy.stats import wilcoxon
    except Exception:
        wilcoxon = None

    contrasts = []
    for a, b, metric, higher in CONTRASTS:
        xa, xb = per_page[a][metric], per_page[b][metric]
        mean, lo, hi = paired_bootstrap(xa, xb)
        entry = {"a": a, "b": b, "metric": metric, "higher_is_better": higher,
                 "mean_diff": mean, "ci95": [lo, hi],
                 "excludes_zero": bool(lo > 0 or hi < 0)}
        if wilcoxon is not None:
            d = np.asarray(xa) - np.asarray(xb)
            d = d[~np.isnan(d)]
            entry["wilcoxon_p"] = float(wilcoxon(d).pvalue)
        contrasts.append(entry)
        star = "*" if entry["excludes_zero"] else " "
        print(f"{star} {a} - {b} [{metric}] = {mean:+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]"
              + (f"  p={entry.get('wilcoxon_p'):.2e}" if "wilcoxon_p" in entry else ""))

    with open(OUT_JSON, "w") as f:
        json.dump({"mask_sha256": sha, "n_pages": n_pages, "n_boot": N_BOOT,
                   "per_page": {k: {m: list(map(float, v)) for m, v in r.items()}
                                for k, r in per_page.items()},
                   "contrasts": contrasts}, f)
    print(f"\nSaved {OUT_JSON}")


if __name__ == "__main__":
    main()
