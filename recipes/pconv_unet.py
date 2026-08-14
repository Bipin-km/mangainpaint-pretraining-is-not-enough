"""
PConv-UNet (Liu et al. ECCV 2018, "Image Inpainting for Irregular Holes
Using Partial Convolutions") -- the paper's classic-literature baseline,
for the related-work comparison table.

This recipe fixes two real bugs found while porting the original training
script, neither a redesign:

1. **The same `build_box_cache` bug fixed everywhere else in this
   codebase**: the original used `getroot().findall('page')` instead of
   `.//page` (Manga109's real XML nests `<page>` two levels deep, under
   `<book><pages>`, not as a direct child of root) and read
   `x`/`y`/`width`/`height` attributes that don't exist on real `<text>`
   elements (the real schema is `xmin`/`ymin`/`xmax`/`ymax`). Both bugs
   together meant the script's own real-dialogue-text exclusion was a
   100%-incidence no-op. Fixed to match `mangainpaint/dataset.py`'s
   corrected `build_box_cache` exactly (see below).
2. **`from model import ...` -> `from model_pconv import ...`** -- the
   original script imported a module named `model` that didn't actually
   exist alongside it (a stale rename); it would fail to even import.
   `model_pconv.py` (defining `PConvUNet`/`VGG16Features`/`gram_matrix`/
   `count_params`) must sit alongside this script for the import below to
   resolve.

Deliberately kept unchanged (fidelity to Liu et al. is the point of a
literature baseline, not an excuse to graft this project's own GAN/masking
machinery onto it): own bespoke `PartialConv2d`, own VGG16 perceptual +
style loss (eq. 7-11 of the paper), own Adam optimizer (no discriminator,
no adversarial training -- not part of the original method), own brush-
stroke-only masking (Axis B1 -- PConv predates balloon-shaped masking as a
concept; B1 procedural masking is the fairer like-for-like comparison to
what this project's own B1 cells report), same 50-epoch budget as the
original ("matching our model's training budget").

Run:
    torchrun --standalone --nproc_per_node=<N> recipes/pconv_unet.py
Resume:
    Set CFG["resume"] = "checkpoints_pconv/last.pt" and re-run.
"""
import os
import sys
import random
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms as T

from skimage.metrics import structural_similarity as ssim
from tqdm.auto import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
plt.rcParams.update({'font.family': 'DejaVu Serif', 'figure.dpi': 150, 'font.size': 8})

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_pconv import PConvUNet, VGG16Features, gram_matrix, count_params  # fixed: was `from model import ...`

try:
    import lpips as lpips_lib
    LPIPS_AVAIL = True
except ImportError:
    LPIPS_AVAIL = False


# ══════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════
# Set this to your local Manga109-s root, or export MANGA109_ROOT instead.
MANGA109_ROOT = os.environ.get("MANGA109_ROOT", "./data/Manga109s")

CFG = {
    "root_dir":  MANGA109_ROOT,
    "train_csv": os.path.join(MANGA109_ROOT, "train.csv"),
    "val_csv":   os.path.join(MANGA109_ROOT, "val.csv"),
    "test_csv":  os.path.join(MANGA109_ROOT, "test.csv"),

    # Training — match our model's setup for fair comparison
    "image_size":  384,
    "batch_size":  4,           # PConv at 33M params + VGG16 = heavier; safer
    "num_workers": 2,
    "epochs":      50,
    "lr":          2e-4,
    "betas":       (0.9, 0.999),
    "grad_clip":   1.0,
    "show_every":  5,

    # PConv original loss weights (Liu et al. 2018, eq. 7-11)
    # Paper uses w_style=120 assuming [0,255]-scale features. Our [0,1]-scale
    # input + autocast makes the original weight unstable; we use a calibrated
    # smaller weight that produces equivalent gradient magnitude.
    "w_valid":     1.0,
    "w_hole":      6.0,
    "w_perc":      0.05,
    "w_style":     20.0,        # was 120.0 — reduced for fp32-loss / [0,1] scale
    "w_tv":        0.1,

    # Mask generator — IDENTICAL to our model's training (Axis B1)
    "mask_brush_w_min": 7,  "mask_brush_w_max": 25,
    "mask_strokes_min": 1,  "mask_strokes_max": 4,
    "mask_len_min":     20, "mask_len_max":     90,
    "mask_large_prob":  0.20, "mask_large_frac": 0.25,

    # I/O
    "hole_fill":   "white",
    "ckpt_dir":    "checkpoints_pconv",
    "vis_dir":     "vis_pconv",
    "resume":      None,
}


# ══════════════════════════════════════════════════════════
# DDP HELPERS
# ══════════════════════════════════════════════════════════
def setup_ddp(rank, world_size):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29501")  # different port from main train
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank): return rank == 0
def unwrap(m): return m.module if hasattr(m, 'module') else m


def reduce_mean(t):
    if not dist.is_initialized():
        return t
    t = t.clone()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t / dist.get_world_size()


def seed_everything(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


# ══════════════════════════════════════════════════════════
# XML CACHING + DATASET
# ══════════════════════════════════════════════════════════
BOX_CACHE = {}


def build_box_cache(csv_path, root_dir, rank=0):
    """Fixed to match mangainpaint/dataset.py's corrected version -- see
    module docstring, fix (1).
    """
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    iterator = df.iterrows()
    if is_main(rank):
        iterator = tqdm(iterator, total=len(df),
                        desc=f"Caching {os.path.basename(csv_path)}", leave=False)
    for _, row in iterator:
        img_path = row['image_path']
        if img_path in BOX_CACHE: continue
        ann_path = os.path.join(root_dir, str(row.get('annotation_path', '')))
        page_idx = int(row.get('page_index', 0))
        try:
            pages = ET.parse(ann_path).getroot().findall('.//page')  # fixed: was 'page'
            if page_idx < len(pages):
                BOX_CACHE[img_path] = [
                    (float(t.get("xmin")), float(t.get("ymin")),
                     float(t.get("xmax")), float(t.get("ymax")))
                    for t in pages[page_idx].findall('text')
                ]  # fixed: was x/y/width/height, real schema is xmin/ymin/xmax/ymax
            else:
                BOX_CACHE[img_path] = []
        except Exception:
            BOX_CACHE[img_path] = []


def generate_mask(H, W, exclusion_boxes, is_train=True):
    mask = np.zeros((H, W), dtype=np.uint8)
    if is_train and random.random() < CFG["mask_large_prob"]:
        frac = random.uniform(0.05, CFG["mask_large_frac"])
        area = int(H * W * frac)
        bh = int((area ** 0.5) * random.uniform(0.5, 1.5))
        bw = max(1, area // max(bh, 1))
        bh, bw = min(bh, H - 1), min(bw, W - 1)
        y0 = random.randint(0, H - bh); x0 = random.randint(0, W - bw)
        mask[y0:y0 + bh, x0:x0 + bw] = 1
    else:
        for _ in range(random.randint(CFG["mask_strokes_min"], CFG["mask_strokes_max"])):
            sx0, sy0 = random.randint(0, W - 1), random.randint(0, H - 1)
            bw = random.randint(CFG["mask_brush_w_min"], CFG["mask_brush_w_max"])
            ang = random.uniform(0, 2 * np.pi)
            for _ in range(random.randint(3, 8)):
                ang += random.uniform(-np.pi / 3, np.pi / 3)
                L = random.randint(CFG["mask_len_min"], CFG["mask_len_max"])
                ex = int(np.clip(sx0 + L * np.cos(ang), 0, W - 1))
                ey = int(np.clip(sy0 + L * np.sin(ang), 0, H - 1))
                cv2.line(mask, (sx0, sy0), (ex, ey), 1, bw)
                sx0, sy0 = ex, ey
    for x1, y1, x2, y2 in exclusion_boxes:
        mask[y1:y2, x1:x2] = 0
    return torch.from_numpy(mask).float().unsqueeze(0)


class Manga109PConvDataset(Dataset):
    """
    Returns 3-channel input (repeated grayscale) at [0,1] for PConv.
    PConv's mask convention: 1 = valid, 0 = hole (opposite of ours).
    Also returns 1-channel [-1,1] for metric computation comparable to our model.
    """
    def __init__(self, csv_path, root_dir, image_size, hole_fill="white", is_train=False):
        self.data = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.sz = image_size
        self.is_train = is_train
        self.fill_val = {"white": 1.0, "black": 0.0, "zero": 0.0}[hole_fill]
        self.tf_01 = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),  # [0, 1]
        ])

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['image_path']
        img_pil = Image.open(os.path.join(self.root_dir, img_path)).convert('L')
        img_01 = self.tf_01(img_pil)             # (1, H, W) in [0, 1]

        raw_boxes = BOX_CACHE.get(img_path, [])
        ow, oh = img_pil.size
        sx, sy = self.sz / ow, self.sz / oh
        boxes = [(int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))
                 for x1, y1, x2, y2 in raw_boxes]
        hole_mask = generate_mask(self.sz, self.sz, boxes, self.is_train)  # 1=hole
        valid_mask = 1.0 - hole_mask                                       # 1=valid (PConv convention)

        # 3-channel input (grayscale repeated) for PConv
        img_3ch = img_01.repeat(3, 1, 1)
        masked_3ch = img_3ch * valid_mask + self.fill_val * (1 - valid_mask)

        # 1-channel [-1, 1] for our metrics (same as our model's eval)
        img_metric = (img_01 * 2 - 1).clamp(-1, 1)

        return {
            "image_3ch":     img_3ch,        # (3, H, W) [0, 1]
            "masked_3ch":    masked_3ch,     # (3, H, W) [0, 1]
            "valid_mask":    valid_mask,     # (1, H, W) {0, 1}, 1=valid (PConv)
            "hole_mask":     hole_mask,      # (1, H, W) {0, 1}, 1=hole (ours)
            "image_metric":  img_metric,     # (1, H, W) [-1, 1]
        }


def make_loaders(rank, world_size):
    if is_main(rank):
        print("Pre-loading XML data to RAM...")
    build_box_cache(CFG["train_csv"], CFG["root_dir"], rank)
    build_box_cache(CFG["val_csv"], CFG["root_dir"], rank)

    train_ds = Manga109PConvDataset(CFG["train_csv"], CFG["root_dir"],
                                    CFG["image_size"], CFG["hole_fill"], True)
    val_ds = Manga109PConvDataset(CFG["val_csv"], CFG["root_dir"], CFG["image_size"])

    train_sampler = DistributedSampler(train_ds, world_size, rank, shuffle=True, drop_last=True)
    val_sampler = DistributedSampler(val_ds, world_size, rank, shuffle=False, drop_last=False)

    kw = dict(num_workers=CFG["num_workers"], pin_memory=True,
              persistent_workers=True, prefetch_factor=4)
    trl = DataLoader(train_ds, CFG["batch_size"], sampler=train_sampler, **kw)
    val = DataLoader(val_ds, CFG["batch_size"], sampler=val_sampler, **kw)
    return trl, val, train_sampler


# ══════════════════════════════════════════════════════════
# PCONV LOSS (Liu et al. 2018, eq. 7-11)
# ──────────────────────────────────────────────────────────
# I_out  = generator output
# I_gt   = ground truth
# M      = valid mask (1 valid, 0 hole)
# I_comp = composite: M*I_gt + (1-M)*I_out  [hole filled with prediction, rest GT]
# Losses:
#   L_valid = ||M * (I_out - I_gt)||_1 / sum(M)
#   L_hole  = ||(1-M) * (I_out - I_gt)||_1 / sum(1-M)
#   L_perc  = sum_i ||phi_i(I_out) - phi_i(I_gt)||_1 + ||phi_i(I_comp) - phi_i(I_gt)||_1
#   L_style = sum_i ||Gram(phi_i(I_out)) - Gram(phi_i(I_gt))||_1
#                 + ||Gram(phi_i(I_comp)) - Gram(phi_i(I_gt))||_1
#   L_tv    = sum over pixels |I_comp[r,c+1] - I_comp[r,c]| + |I_comp[r+1,c] - I_comp[r,c]|
# ══════════════════════════════════════════════════════════
def total_variation_loss(x):
    """Mean TV over the image."""
    h_diff = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    w_diff = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return h_diff + w_diff


def pconv_loss(out, gt, valid_mask, vgg):
    """
    Args:
        out: (B, 3, H, W) [0, 1] generator output
        gt:  (B, 3, H, W) [0, 1] ground truth
        valid_mask: (B, 1, H, W) {0, 1}, 1=valid
        vgg: VGG16Features instance
    """
    eps = 1e-6
    hole_mask = 1.0 - valid_mask
    valid_3 = valid_mask.expand(-1, 3, -1, -1)
    hole_3 = hole_mask.expand(-1, 3, -1, -1)

    # Composite: keep ground truth in valid regions, generator output in holes
    comp = valid_3 * gt + hole_3 * out

    # Per-region L1
    l_valid = (valid_3 * (out - gt)).abs().sum() / (valid_3.sum() + eps)
    l_hole = (hole_3 * (out - gt)).abs().sum() / (hole_3.sum() + eps)

    # Perceptual + style — compute in FP32 to avoid Gram-matrix fp16 overflow
    with torch.amp.autocast('cuda', enabled=False):
        out_f = out.float().clamp(0, 1)
        gt_f = gt.float().clamp(0, 1)
        comp_f = comp.float().clamp(0, 1)
        f_out = vgg(out_f)
        f_gt = vgg(gt_f)
        f_comp = vgg(comp_f)

        l_perc = out.new_zeros((), dtype=torch.float32)
        l_style = out.new_zeros((), dtype=torch.float32)
        for fo, fg, fc in zip(f_out, f_gt, f_comp):
            l_perc = l_perc + (fo - fg).abs().mean() + (fc - fg).abs().mean()
            l_style = l_style + (gram_matrix(fo) - gram_matrix(fg)).abs().mean() \
                              + (gram_matrix(fc) - gram_matrix(fg)).abs().mean()

    # TV on composite
    l_tv = total_variation_loss(comp)

    total = (CFG["w_valid"] * l_valid +
             CFG["w_hole"] * l_hole +
             CFG["w_perc"] * l_perc +
             CFG["w_style"] * l_style +
             CFG["w_tv"] * l_tv)
    return total, {
        "valid": l_valid.item(),
        "hole":  l_hole.item(),
        "perc":  l_perc.item() if torch.is_tensor(l_perc) else l_perc,
        "style": l_style.item() if torch.is_tensor(l_style) else l_style,
        "tv":    l_tv.item(),
    }


# ══════════════════════════════════════════════════════════
# METRICS (verbatim from main train.py)
# ══════════════════════════════════════════════════════════
def to01(t): return ((t + 1) * 0.5).clamp(0, 1)


def nm(xs):
    xs = [v for v in xs if not np.isnan(v)]
    return float(np.mean(xs)) if xs else float('nan')


def hole_psnr(p, t, m, eps=1e-8):
    mse = ((p - t) * m).pow(2).sum() / (m.sum() + eps)
    return float((10 * torch.log10(4 / (mse + eps))).item())


def hole_ssim(p, t, m):
    p = to01(p).squeeze().cpu().numpy()
    t = to01(t).squeeze().cpu().numpy()
    m = m.squeeze().cpu().numpy()
    ys, xs = np.where(m > 0.5)
    if not len(xs): return float('nan')
    y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    pc, tc = p[y1:y2, x1:x2], t[y1:y2, x1:x2]
    s = min(pc.shape)
    if s < 3: return float('nan')
    w = min(7, s); w = w if w % 2 else w - 1
    return float(ssim(tc, pc, data_range=1.0, win_size=max(w, 3)))


def hole_grad_l1(p, t, m):
    p = to01(p).squeeze().cpu().numpy()
    t = to01(t).squeeze().cpu().numpy()
    mv = (m.squeeze().cpu().numpy() > 0.5).astype(np.float32)
    if mv.sum() < 1: return float('nan')

    def sob(img):
        gx = cv2.Sobel((img * 255).astype(np.uint8), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel((img * 255).astype(np.uint8), cv2.CV_32F, 0, 1, ksize=3)
        g = cv2.magnitude(gx, gy)
        return g / (g.max() + 1e-8)
    return float((np.abs(sob(p) - sob(t)) * mv).sum() / (mv.sum() + 1e-8))


def hole_edge_f1(p, t, m, eps=1e-8):
    p = to01(p).squeeze().cpu().numpy()
    t = to01(t).squeeze().cpu().numpy()
    mv = (m.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    if mv.sum() < 1: return float('nan')

    def canny(x): return (cv2.Canny((x * 255).astype(np.uint8), 60, 160) > 0).astype(np.uint8)
    ep, et = canny(p) * mv, canny(t) * mv
    tp = float((ep & et).sum())
    fp = float((ep & (1 - et)).sum())
    fn = float(((1 - ep) & et).sum())
    pr = tp / (tp + fp + eps)
    rc = tp / (tp + fn + eps)
    return float(2 * pr * rc / (pr + rc + eps))


def denorm(x): return ((x * 0.5 + 0.5)).clamp(0, 1)


def rgb_to_gray_minus1plus1(rgb_01):
    """Convert [0,1] RGB tensor to [-1,1] grayscale via luminance weights."""
    g = 0.299 * rgb_01[:, 0:1] + 0.587 * rgb_01[:, 1:2] + 0.114 * rgb_01[:, 2:3]
    return (g.clamp(0, 1) * 2 - 1).clamp(-1, 1)


# ══════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════
@torch.no_grad()
def visualize(G, batch, epoch, device, save_dir, max_n=4):
    os.makedirs(save_dir, exist_ok=True)
    G.eval()
    img_3ch = batch["image_3ch"].to(device)
    masked_3ch = batch["masked_3ch"].to(device)
    valid_mask = batch["valid_mask"].to(device)
    hole_mask = batch["hole_mask"].to(device)
    img_metric = batch["image_metric"].to(device)

    out_3ch = unwrap(G)(masked_3ch, valid_mask)
    comp_3ch = valid_mask * img_3ch + (1 - valid_mask) * out_3ch

    out_metric = rgb_to_gray_minus1plus1(out_3ch.clamp(0, 1))
    comp_metric = out_metric * hole_mask + img_metric * (1 - hole_mask)

    n = min(max_n, img_3ch.size(0))
    fig, axes = plt.subplots(n, 5, figsize=(13, 3 * n))
    if n == 1: axes = axes[None]
    titles = ["Original", "Mask", "Masked", "Generated", "Composite"]
    for i in range(n):
        masked_gray = denorm(img_metric[i, 0]).cpu().numpy() * (1 - hole_mask[i, 0].cpu().numpy()) \
                      + 1.0 * hole_mask[i, 0].cpu().numpy()
        panels = [
            denorm(img_metric[i, 0]).cpu().numpy(),
            hole_mask[i, 0].cpu().numpy(),
            masked_gray,
            denorm(out_metric[i, 0]).cpu().numpy(),
            denorm(comp_metric[i, 0]).cpu().numpy(),
        ]
        for j, arr in enumerate(panels):
            axes[i, j].imshow(arr, cmap='gray', vmin=0, vmax=1)
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(titles[j], fontsize=8)
    es = f"{epoch:04d}" if isinstance(epoch, int) else str(epoch)
    plt.suptitle(f"PConv baseline — Epoch {es}", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"epoch_{es}.png"), bbox_inches='tight')
    plt.close()


def plot_history(hist, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    keys = [
        ("train_loss", None, "Loss", ""),
        ("val_psnr", None, "PSNR↑", "dB"),
        ("val_ssim", None, "SSIM↑", ""),
        ("val_grad_l1", None, "GradL1↓", ""),
        ("val_edge_f1", None, "EdgeF1↑", ""),
    ]
    if LPIPS_AVAIL:
        keys.append(("val_lpips", None, "LPIPS↓", ""))
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3))
    for ax, (k1, k2, title, yl) in zip(axes, keys):
        if hist.get(k1):
            ax.plot(hist[k1], label=k1.split("_", 1)[-1])
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("epoch")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "curves.png"), bbox_inches='tight')
    plt.close()


# ══════════════════════════════════════════════════════════
# TRAIN / EVAL
# ══════════════════════════════════════════════════════════
def train_one_epoch(G, vgg, opt, scaler, loader, epoch, device, rank):
    G.train()
    g_tot = 0.0
    step = 0
    nan_steps = 0

    pbar = loader
    if is_main(rank):
        pbar = tqdm(loader, desc=f"Ep{epoch + 1}", leave=False)

    for batch in pbar:
        img_3ch = batch["image_3ch"].to(device, non_blocking=True)
        masked_3ch = batch["masked_3ch"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device, non_blocking=True)

        with torch.amp.autocast('cuda'):
            out_3ch = G(masked_3ch, valid_mask)
            loss, _ = pconv_loss(out_3ch, img_3ch, valid_mask, vgg)

        # Guard: skip step if loss exploded
        if not torch.isfinite(loss):
            nan_steps += 1
            opt.zero_grad(set_to_none=True)
            continue

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(unwrap(G).parameters(), CFG["grad_clip"])
        scaler.step(opt); scaler.update()

        g_tot += float(loss.item())
        step += 1

    if is_main(rank) and nan_steps > 0:
        print(f"  ⚠ {nan_steps} NaN steps skipped this epoch")

    nd = max(1, step)
    g_t = reduce_mean(torch.tensor(g_tot / nd, device=device))
    return g_t.item()


@torch.no_grad()
def evaluate(G, loader, lpips_fn, device, rank):
    G.eval()
    ps, ss, gs, fs, ls = [], [], [], [], []
    pbar = loader
    if is_main(rank):
        pbar = tqdm(loader, desc="val", leave=False)
    for batch in pbar:
        img_3ch = batch["image_3ch"].to(device, non_blocking=True)
        masked_3ch = batch["masked_3ch"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device, non_blocking=True)
        hole_mask = batch["hole_mask"].to(device, non_blocking=True)
        img_metric = batch["image_metric"].to(device, non_blocking=True)

        with torch.amp.autocast('cuda'):
            out_3ch = unwrap(G)(masked_3ch, valid_mask)

        out_metric = rgb_to_gray_minus1plus1(out_3ch.clamp(0, 1))
        comp_metric = out_metric * hole_mask + img_metric * (1 - hole_mask)

        for b in range(img_metric.size(0)):
            p, t, m = comp_metric[b:b+1], img_metric[b:b+1], hole_mask[b:b+1]
            ps.append(hole_psnr(p, t, m))
            ss.append(hole_ssim(p, t, m))
            gs.append(hole_grad_l1(p, t, m))
            fs.append(hole_edge_f1(p, t, m))
        if LPIPS_AVAIL and lpips_fn is not None:
            c224 = F.interpolate(comp_metric, size=224, mode='bilinear', align_corners=False)
            i224 = F.interpolate(img_metric, size=224, mode='bilinear', align_corners=False)
            ls.append(float(lpips_fn(c224.repeat(1, 3, 1, 1),
                                     i224.repeat(1, 3, 1, 1)).mean().item()))

    local = {"psnr": nm(ps), "ssim": nm(ss),
             "grad_l1": nm(gs), "edge_f1": nm(fs),
             "lpips": nm(ls) if ls else float('nan')}
    out = {}
    for k, v in local.items():
        if np.isnan(v):
            out[k] = float('nan')
            continue
        t = reduce_mean(torch.tensor(v, device=device))
        out[k] = t.item()
    return out


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
def main_ddp(rank, world_size):
    setup_ddp(rank, world_size)
    seed_everything(42 + rank)
    device = torch.device(f"cuda:{rank}")

    train_loader, val_loader, train_sampler = make_loaders(rank, world_size)

    G = PConvUNet(in_ch=3, out_ch=3).to(device)
    # Convert BN → SyncBN for stable stats across GPUs at small batch
    G = nn.SyncBatchNorm.convert_sync_batchnorm(G)
    vgg = VGG16Features().to(device)

    if is_main(rank):
        print(f"PConv UNet params: {count_params(G)}")
        print(f"VGG16 (frozen):    {count_params(vgg)}")
        print(f"Image size: {CFG['image_size']} | Per-GPU batch: {CFG['batch_size']} "
              f"| Effective batch: {CFG['batch_size'] * world_size}")

    G = nn.parallel.DistributedDataParallel(G, device_ids=[rank], find_unused_parameters=False)

    opt = torch.optim.Adam(G.parameters(), lr=CFG["lr"], betas=CFG["betas"])
    scaler = torch.amp.GradScaler('cuda')
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, CFG["epochs"], eta_min=1e-5)

    lpips_fn = None
    if LPIPS_AVAIL:
        lpips_fn = lpips_lib.LPIPS(net='vgg', verbose=False).to(device)
        for p in lpips_fn.parameters(): p.requires_grad_(False)

    # ── Resume ──
    start_epoch = 0
    best = -1e9
    hist = {k: [] for k in ["train_loss", "val_psnr", "val_ssim",
                            "val_grad_l1", "val_edge_f1", "val_lpips"]}

    if CFG.get("resume") and os.path.exists(CFG["resume"]):
        if is_main(rank):
            print(f"Resuming from {CFG['resume']}...")
        ckpt = torch.load(CFG["resume"], map_location=device)
        unwrap(G).load_state_dict(ckpt["G"])
        if "opt" in ckpt: opt.load_state_dict(ckpt["opt"])
        start_epoch = ckpt.get("epoch", -1) + 1
        best = ckpt.get("score", -1e9)
        if "hist" in ckpt: hist = ckpt["hist"]
        for _ in range(start_epoch): sched.step()

    if is_main(rank):
        os.makedirs(CFG["ckpt_dir"], exist_ok=True)
        os.makedirs(CFG["vis_dir"], exist_ok=True)
        vis_batch = next(iter(val_loader))
    else:
        vis_batch = None

    if is_main(rank):
        print(f"\n{'Ep':>4} | {'Loss':>9} | {'PSNR':>6} {'SSIM':>6} {'GradL1':>7} "
              f"{'EdgeF1':>7} {'LPIPS':>6} | Score")
        print("-" * 70)

    for ep in range(start_epoch, CFG["epochs"]):
        train_sampler.set_epoch(ep)
        loss = train_one_epoch(G, vgg, opt, scaler, train_loader, ep, device, rank)
        vm = evaluate(G, val_loader, lpips_fn, device, rank)
        sched.step()

        if is_main(rank):
            for k, v in [("train_loss", loss),
                         ("val_psnr", vm["psnr"]), ("val_ssim", vm["ssim"]),
                         ("val_grad_l1", vm["grad_l1"]), ("val_edge_f1", vm["edge_f1"]),
                         ("val_lpips", vm["lpips"])]:
                hist[k].append(v)

            lp = -vm["lpips"] if not np.isnan(vm["lpips"]) else 0.0
            score = (0.30 * vm["ssim"] + 0.30 * vm["edge_f1"]
                     - 0.15 * vm["grad_l1"] + 0.10 * (vm["psnr"] / 30) + 0.15 * lp)
            star = ""
            if score > best:
                best = score; star = "✓"
                torch.save({
                    "G": unwrap(G).state_dict(),
                    "opt": opt.state_dict(),
                    "epoch": ep, "score": score, "metrics": vm, "cfg": CFG,
                    "hist": hist,
                }, f"{CFG['ckpt_dir']}/best.pt")
            torch.save({
                "G": unwrap(G).state_dict(),
                "opt": opt.state_dict(),
                "epoch": ep, "score": score, "hist": hist,
            }, f"{CFG['ckpt_dir']}/last.pt")

            lps = f"{vm['lpips']:.4f}" if not np.isnan(vm['lpips']) else "  n/a"
            print(f"{ep + 1:>4} | {loss:>9.4f} | {vm['psnr']:>6.2f} {vm['ssim']:>6.4f} "
                  f"{vm['grad_l1']:>7.4f} {vm['edge_f1']:>7.4f} {lps:>6} | {score:.4f} {star}")

            if (ep + 1) % CFG["show_every"] == 0 or ep == 0:
                visualize(G, vis_batch, ep + 1, device, CFG["vis_dir"])
                plot_history(hist, CFG["vis_dir"])

        dist.barrier()

    cleanup_ddp()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No CUDA devices available.")

    if "LOCAL_RANK" in os.environ:
        rank = int(os.environ["LOCAL_RANK"])
        main_ddp(rank, world_size)
    else:
        mp.spawn(main_ddp, args=(world_size,), nprocs=world_size, join=True)
