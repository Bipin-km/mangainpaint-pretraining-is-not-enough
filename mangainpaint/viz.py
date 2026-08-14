"""Qualitative visualization + curve plotting, shared by every training run."""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mangainpaint.ddp_utils import unwrap, call_g
from mangainpaint.metrics import denorm

plt.rcParams.update({'font.family': 'DejaVu Serif', 'figure.dpi': 150, 'font.size': 8})


@torch.no_grad()
def visualize(G, batch, epoch, device, save_dir, max_n=4, phase=""):
    os.makedirs(save_dir, exist_ok=True)
    G.eval()
    img = batch["image"].to(device)
    mask = batch["mask"].to(device)
    masked = batch["masked_image"].to(device)
    with torch.amp.autocast('cuda'):
        out = call_g(unwrap(G), batch["model_input"].to(device), batch, device)
    comp = out * mask + img * (1 - mask)
    n = min(max_n, img.size(0))
    fig, axes = plt.subplots(n, 5, figsize=(13, 3 * n))
    if n == 1: axes = axes[None]
    titles = ["Original", "Mask", "Masked", "Generated", "Composite"]
    for i in range(n):
        panels = [
            denorm(img[i, 0]).cpu().numpy(),
            mask[i, 0].cpu().numpy(),
            denorm(masked[i, 0]).cpu().numpy(),
            denorm(out[i, 0]).cpu().numpy(),
            denorm(comp[i, 0]).cpu().numpy(),
        ]
        for j, arr in enumerate(panels):
            axes[i, j].imshow(arr, cmap='gray', vmin=0, vmax=1)
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(titles[j], fontsize=8)
    es = f"{epoch:04d}" if isinstance(epoch, int) else str(epoch)
    title_suffix = f" [{phase}]" if phase else ""
    plt.suptitle(f"Epoch {es}{title_suffix}", fontsize=9)
    plt.tight_layout()
    p = os.path.join(save_dir, f"epoch_{es}.png")
    plt.savefig(p, bbox_inches='tight'); plt.close()


def plot_history(hist, save_dir, lpips_avail=True, phase_boundary=None, refresh_marks=None):
    os.makedirs(save_dir, exist_ok=True)
    keys = [
        ("train_g", "train_d", "Losses", "loss"),
        ("val_psnr", None, "PSNR (up)", "dB"),
        ("val_ssim", None, "SSIM (up)", ""),
        ("val_grad_l1", None, "GradL1 (down)", ""),
        ("val_edge_f1", None, "EdgeF1 (up)", ""),
    ]
    if lpips_avail:
        keys.append(("val_lpips", None, "LPIPS (down)", ""))
    if hist.get("val_score"):
        keys.append(("val_score", None, "Selection score (up)", ""))
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3))
    for ax, (k1, k2, title, yl) in zip(axes, keys):
        if hist.get(k1):
            ax.plot(hist[k1], label=k1.split("_", 1)[-1])
        if k2 and hist.get(k2):
            d_vals = hist[k2]
            xs = [i for i, v in enumerate(d_vals) if v is not None and not np.isnan(v)]
            ys = [d_vals[i] for i in xs]
            if xs: ax.plot(xs, ys, label=k2.split("_", 1)[-1])
        if phase_boundary is not None:
            ax.axvline(phase_boundary - 0.5, color='red', linestyle='--', alpha=0.4, linewidth=0.8)
        if refresh_marks:
            for r in refresh_marks:
                ax.axvline(r - 0.5, color='orange', linestyle=':', alpha=0.4, linewidth=0.6)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("epoch")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    p = os.path.join(save_dir, "curves.png")
    plt.savefig(p, bbox_inches='tight'); plt.close()
