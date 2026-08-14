"""Hole-region evaluation metrics, shared by every training run."""
import numpy as np
import torch
import cv2
from skimage.metrics import structural_similarity as ssim


def to01(t): return ((t + 1) * 0.5).clamp(0, 1)


def denorm(x): return ((x * 0.5 + 0.5)).clamp(0, 1)


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


def hole_ink_frac(t, m, threshold=0.4):
    """Fraction of the hole (ground-truth side) that is ink, using the same
    definition as `losses.make_ink_weight_map`. Used to stratify eval metrics
    by how much real content a mask actually covers -- a mask landing entirely
    on blank paper is a much easier reconstruction than one covering dense
    ink/screentone, and pooling them into one number is what let a near-white
    mean-fill output look artificially strong in the legacy sweep.
    """
    denom = m.sum()
    if denom.item() < 1: return float('nan')
    is_ink = (to01(t) < threshold).float()
    return float(((is_ink * m).sum() / denom).item())


# (name, ink_frac lower bound inclusive, upper bound exclusive)
INK_STRATA = (("sparse", 0.0, 0.05), ("moderate", 0.05, 0.20), ("dense", 0.20, 1.01))


def bucket_by_ink(samples):
    """Split a list of per-sample metric dicts (each must have 'ink_frac')
    into the `INK_STRATA` buckets. Samples with nan ink_frac (degenerate/
    empty mask) are dropped rather than assigned anywhere."""
    out = {name: [] for name, _, _ in INK_STRATA}
    for s in samples:
        f = s["ink_frac"]
        if np.isnan(f): continue
        for name, lo, hi in INK_STRATA:
            if lo <= f < hi:
                out[name].append(s)
                break
    return out


# Rough "poor" reference LPIPS (worst logged value across this project's
# earliest from-scratch GAN attempts was 0.069) -- used only to rescale
# LPIPS onto a ~[0,1] range comparable to EdgeF1 for `selection_score`
# below. Metrics themselves are still reported raw.
LPIPS_REF = 0.10


def selection_score(strata):
    """Checkpoint-selection score, replacing the old PSNR/SSIM-heavy composite.

    Averages (EdgeF1, rescaled -LPIPS) across ink-density strata with equal
    weight *per stratum*, not per sample -- so a model can't win by being
    great at the mostly-blank-paper majority of holes while failing on the
    ink-dense minority. That was the exact failure mode (hole-region PSNR/SSIM
    dominated by blank-paper mean-fill) that made Model C look "best" in the
    legacy sweep despite being a visual failure. PSNR/SSIM are still computed
    and reported per stratum but deliberately excluded from selection.

    `strata`: dict as returned by pairing `bucket_by_ink` buckets with
    per-stratum aggregate metrics (each value a dict with at least
    'n', 'edge_f1', 'lpips' keys).
    """
    vals = []
    for s in strata.values():
        if s.get("n", 0) == 0: continue
        ef1, lp = s.get("edge_f1"), s.get("lpips")
        if ef1 is None or lp is None or np.isnan(ef1) or np.isnan(lp): continue
        lpips_score = max(0.0, 1.0 - lp / LPIPS_REF)
        vals.append(0.5 * ef1 + 0.5 * lpips_score)
    return float(np.mean(vals)) if vals else -1e9
