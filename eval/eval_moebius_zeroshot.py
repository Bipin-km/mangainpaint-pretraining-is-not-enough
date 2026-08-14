"""
Zero-shot Moebius baseline scored against the *pinned* mask protocol
(`fixed_mask_protocol.py`), for like-for-like comparison against SD1.5 and
every model in `eval_pinned_models.py`.

Motivation: an earlier ad hoc Moebius run drew its own subset and re-derived
masks from the global RNG in its own venv, which doesn't reproduce
deterministically -- so its LPIPS wasn't comparable to any other row in the
paper, and LPIPS is exactly the metric the Moebius result is cited for.
Scoring against the pinned 150-page mask file fixes that.

NOTE: this script needs a separate venv built from Moebius's own
`requirements.txt` (it pins a different torch/diffusers stack than this
project's `pyproject.toml`). `moebius_wrapper.py` provides the
`MoebiusZeroShotG`/`build_moebius_pipe` pipeline wrapper; see its docstring
for cloning Moebius and fetching its weights under `MOEBIUS_ROOT`.

Run (from repo root, in a venv built from Moebius's own requirements):
    python eval/eval_moebius_zeroshot.py
"""
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, HERE)

import numpy as np
import torch

from mangainpaint.trainer import evaluate, LPIPS_AVAIL
from fixed_mask_protocol import FixedMaskDataset, PIN_PATH
from moebius_wrapper import MoebiusZeroShotG, build_moebius_pipe, MOEBIUS_SIZE, NUM_STEPS, GUIDANCE_SCALE

try:
    import lpips as lpips_lib
except Exception:
    lpips_lib = None

OUT_JSON = os.path.join(HERE, "moebius_zeroshot_pinned_results.json")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "Moebius inference on CPU is impractically slow"

    ds = FixedMaskDataset(PIN_PATH)
    print(f"device={device}  pinned masks={len(ds)}  sha256={ds.sha[:16]}...")

    pipe = build_moebius_pipe(str(device))
    G = MoebiusZeroShotG(pipe).to(device)

    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    lpips_eval_fn = None
    if LPIPS_AVAIL and lpips_lib is not None:
        lpips_eval_fn = lpips_lib.LPIPS(net="vgg", verbose=False).to(device)
        for p in lpips_eval_fn.parameters():
            p.requires_grad_(False)

    with torch.no_grad():
        r = evaluate(G, loader, lpips_eval_fn, device, rank=0, desc="moebius-pinned")

    result = {
        "model": "hustvl/Moebius (pretrained, general-purpose, zero-shot)",
        "protocol": "pinned", "mask_sha256": ds.sha, "n_images": len(ds),
        "moebius_size": MOEBIUS_SIZE, "num_steps": NUM_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "psnr": r["psnr"], "ssim": r["ssim"], "grad_l1": r["grad_l1"],
        "edge_f1": r["edge_f1"], "lpips": r["lpips"],
        "strata": {k: {"n": v["n"], "edge_f1": v["edge_f1"], "lpips": v["lpips"]}
                   for k, v in r["strata"].items()},
    }
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nEdgeF1={r['edge_f1']:.4f} LPIPS={r['lpips']:.4f} "
          f"PSNR={r['psnr']:.2f} SSIM={r['ssim']:.4f}")
    print("strata:", {k: (v["n"], round(v["edge_f1"], 4), round(v["lpips"], 4))
                      for k, v in r["strata"].items()})
    print(f"Saved {OUT_JSON}")


if __name__ == "__main__":
    main()
