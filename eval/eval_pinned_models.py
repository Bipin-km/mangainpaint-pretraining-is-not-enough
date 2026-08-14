"""
Score our own checkpoints against the frozen mask file built by
`fixed_mask_protocol.py`, so the zero-shot comparison table is like-for-like
by construction (see that module's docstring for why the subset runs could
not simply reuse the full-split numbers).

Run:  python eval/eval_pinned_models.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, HERE)

import torch

from mangainpaint.trainer import evaluate, LPIPS_AVAIL
from mangainpaint.checkpoint_registry import build_generator, load_generator_state_dict
from fixed_mask_protocol import FixedMaskDataset, PIN_PATH

try:
    import lpips as lpips_lib
except Exception:
    lpips_lib = None

# Root under which each checkpoint below is expected at <CKPT_ROOT>/<name>/best.pt
# (the layout release checkpoints unpack into; see README's Checkpoints section).
CKPT_ROOT = os.environ.get("MANGA_CKPT_ROOT", os.path.join(HERE, "..", "checkpoints"))
OUT_JSON = os.path.join(HERE, "brush_eval_pinned_models.json")

# Named after the original run ids -- see README's recipe/run-id table for
# how these map to the recipes shipped here.
MODELS = [
    ("pconv_baseline_v1",      "pconv",       "PConv-UNet"),
    ("uffc_test_kaggle_v2",    "uffc",        "UFFC-GAN"),
    ("attn2_test_v2",          "attn_noffc",  "CtxAttn-GAN"),
    ("projected_d_test_v2",    "vanilla",     "FFC-GAN"),
    ("lama_slim_s1",           "lama_slim",   "S1 (no distillation)"),
    ("lama_distill_s2",        "lama_slim",   "S2 (+distillation)"),
    ("lama_distill_s3",        "lama_slim",   "S3 (+external losses)"),
    ("lama_transfer_brush_v1", "lama",        "Fine-tuned LaMa (teacher)"),
]


def ckpt_path(name):
    p = os.path.join(CKPT_ROOT, name, "best.pt")
    if os.path.exists(p):
        return p
    raise FileNotFoundError(p)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = FixedMaskDataset(PIN_PATH)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    print(f"device={device}  pinned pages={len(ds)}  sha256={ds.sha[:16]}...")

    lpips_eval_fn = None
    if LPIPS_AVAIL and lpips_lib is not None:
        lpips_eval_fn = lpips_lib.LPIPS(net="vgg", verbose=False).to(device)
        for p in lpips_eval_fn.parameters():
            p.requires_grad_(False)

    results = {"mask_sha256": ds.sha, "n_images": len(ds), "models": {}}
    for name, arch, label in MODELS:
        ckpt = torch.load(ckpt_path(name), map_location=device, weights_only=False)
        G = build_generator(arch, dict(ckpt.get("cfg", {})), device)
        load_generator_state_dict(G, arch, ckpt["G"], strict=False)
        G.eval()
        with torch.no_grad():
            r = evaluate(G, loader, lpips_eval_fn, device, rank=0, desc=name)
        results["models"][name] = {
            "label": label, "psnr": r["psnr"], "ssim": r["ssim"],
            "edge_f1": r["edge_f1"], "lpips": r["lpips"],
            "strata": {k: {"n": v["n"], "edge_f1": v["edge_f1"], "lpips": v["lpips"]}
                       for k, v in r["strata"].items()},
        }
        print(f"  {label:<28} EdgeF1={r['edge_f1']:.4f} LPIPS={r['lpips']:.4f} "
              f"PSNR={r['psnr']:.2f} SSIM={r['ssim']:.4f}")
        with open(OUT_JSON, "w") as f:
            json.dump(results, f, indent=2)
        del G
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"\nSaved {OUT_JSON}")


if __name__ == "__main__":
    main()
