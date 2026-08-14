"""
Shared two-phase (G-only -> G+D) training loop, DDP-driven, CFG-configurable.

Consolidated here (rather than duplicated per training script) so every
run (LaMa-transfer generator, Projected-GAN D, procedural brush masking)
gets the following for free:

  1. Dataset pre-caching  -> mangainpaint/dataset.py (IMG_CACHE), not this file.
  2. Configurable worker count, auto-sized to cpu_count() // world_size.
  3. LPIPS split into a light training-loop net (`lpips_train_net`, default
     'squeeze') vs. the heavier eval-only net (`lpips_eval_net`, default
     'vgg', kept for metric comparability with every baseline reported in
     the paper, all of which use LPIPS-VGG).
  4. R1 penalty frequency (`p2_r1_every`) is just a CFG value here — bump it
     in the entry script's CFG to make R1 less frequent.
  5. Optional torch.compile via `use_compile` (off by default; DDP + AMP +
     the D-refresh reinit path haven't been stress-tested with compile yet).
  6. Optional lightweight epoch timing breakdown (data-wait vs. compute) via
     `profile_timing`, so wall-clock improvement can actually be measured
     instead of asserted. Off by default (adds sync overhead).

Callers construct their own (G, D) nn.Module pair via a `model_fn(cfg)`
factory (so this file stays agnostic to which generator/discriminator
architecture is plugged in — Axis A1 from-scratch vs. Axis A2 LaMa-transfer,
PatchD vs. Projected-GAN D) and call `run(cfg, model_fn)`.
"""
import os
import time

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from tqdm.auto import tqdm

try:
    import lpips as lpips_lib
    LPIPS_AVAIL = True
except ImportError:
    LPIPS_AVAIL = False

from mangainpaint.ddp_utils import (setup_ddp, cleanup_ddp, is_main, unwrap, call_g,
                              reduce_mean, reduce_mean_of, reduce_count,
                              broadcast_state, seed_everything)
from mangainpaint.dataset import make_loaders, build_vis_batch
from mangainpaint.model_resnet_pl import ResNetPL
from mangainpaint.model_screenvae import ScreenVAEConsistencyLoss, ScreenVAEPatchMatchLoss
from mangainpaint.losses import (charbonnier, make_ring, make_ink_weight_map,
                           charbonnier_weighted, sobel_mag, fft_mag_loss,
                           bitonal_commitment_loss, bitonal_commitment_loss_gated,
                           regional_stats_loss, ring_consistency_loss, patch_match_loss,
                           d_hinge, g_hinge, lazy_r1_penalty)
from mangainpaint.metrics import (hole_psnr, hole_ssim, hole_grad_l1, hole_edge_f1,
                            hole_ink_frac, bucket_by_ink, selection_score, denorm)
from mangainpaint.viz import visualize, plot_history


class EpochTimer:
    """Optional data-wait vs. compute breakdown. No-op unless enabled."""
    def __init__(self, enabled):
        self.enabled = enabled
        self.wait = 0.0
        self.compute = 0.0
        self._t = None

    def mark_fetch_start(self):
        if self.enabled:
            self._t = time.perf_counter()

    def mark_fetch_end(self):
        if self.enabled:
            now = time.perf_counter()
            self.wait += now - self._t
            self._t = now

    def mark_compute_end(self):
        if self.enabled:
            torch.cuda.synchronize()
            now = time.perf_counter()
            self.compute += now - self._t
            self._t = now


def _lpips_term(lpips_fn, comp, img):
    c224 = F.interpolate(comp, size=224, mode='bilinear', align_corners=False)
    i224 = F.interpolate(img, size=224, mode='bilinear', align_corners=False)
    return lpips_fn(c224.repeat(1, 3, 1, 1), i224.repeat(1, 3, 1, 1)).mean()


def _lpips_per_sample(lpips_fn, comp, img):
    """Same as `_lpips_term` but without the batch `.mean()` -- needed for
    per-sample ink-density stratification. One forward pass for the whole
    batch, not a python loop, so this doesn't add extra backbone compute."""
    c224 = F.interpolate(comp, size=224, mode='bilinear', align_corners=False)
    i224 = F.interpolate(img, size=224, mode='bilinear', align_corners=False)
    return lpips_fn(c224.repeat(1, 3, 1, 1), i224.repeat(1, 3, 1, 1)).view(-1)


# ══════════════════════════════════════════════════════════
# TRAIN — Phase 1 (G-only)
# ══════════════════════════════════════════════════════════
def train_phase1(G, opt_g, scaler_g, loader, lpips_train_fn, resnet_pl_fn,
                 screenvae_consistency_fn, screenvae_patch_match_fn, distill_fn,
                 cfg, epoch, device, rank, timer):
    G.train()
    g_tot = 0.0
    step = 0

    pbar = loader
    if is_main(rank):
        pbar = tqdm(loader, desc=f"Ep{epoch + 1} [P1]", leave=False)

    timer.mark_fetch_start()
    for batch in pbar:
        timer.mark_fetch_end()
        img = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        inp = batch["model_input"].to(device, non_blocking=True)
        ring = make_ring(mask, cfg["ring_radius"])
        focus = (mask + ring).clamp(0, 1)

        ink_w = make_ink_weight_map(img, mask,
                                    threshold=cfg["ink_threshold"],
                                    extra=cfg["ink_extra"])

        with torch.amp.autocast('cuda'):
            gen = call_g(G, inp, batch, device)

            hole = charbonnier(gen, img, mask)
            ring_loss = charbonnier(gen, img, ring)
            vid = charbonnier(gen, img, 1 - mask)
            edg = ((sobel_mag(gen) - sobel_mag(img)).abs() * focus).mean()
            fft_l = fft_mag_loss(gen * focus, img * focus)
            ink_loss = charbonnier_weighted(gen, img, mask, ink_w)
            bitonal = bitonal_commitment_loss(gen, mask)
            bitonal_g = bitonal_commitment_loss_gated(
                gen, img, mask, cfg.get("bitonal_gate_thresh", 0.05))
            rstats = regional_stats_loss(gen, img, mask, cfg.get("regional_stats_window", 9))
            ring_cons = ring_consistency_loss(gen, mask, ring)
            pmatch = (patch_match_loss(gen, img, mask) if cfg.get("p1_w_patch_match", 0.0) > 0
                     else gen.new_zeros(()))

            loss_g = (cfg["p1_w_hole_rec"] * hole +
                      cfg["p1_w_ring_rec"] * ring_loss +
                      cfg["p1_w_valid_id"] * vid +
                      cfg["p1_w_edge"] * edg +
                      cfg["p1_w_fft"] * fft_l +
                      cfg["p1_w_ink"] * ink_loss +
                      cfg.get("p1_w_bitonal", 0.0) * bitonal +
                      cfg.get("p1_w_bitonal_gated", 0.0) * bitonal_g +
                      cfg.get("p1_w_regional_stats", 0.0) * rstats +
                      cfg.get("p1_w_ring_consistency", 0.0) * ring_cons +
                      cfg.get("p1_w_patch_match", 0.0) * pmatch)

            if LPIPS_AVAIL and cfg["p1_w_lpips"] > 0 and lpips_train_fn is not None:
                comp = gen * mask + img * (1 - mask)
                lp = _lpips_term(lpips_train_fn, comp, img)
                loss_g = loss_g + cfg["p1_w_lpips"] * lp

            if resnet_pl_fn is not None and cfg.get("p1_w_resnet_pl", 0.0) > 0:
                rpl = resnet_pl_fn(gen, img)
                loss_g = loss_g + cfg["p1_w_resnet_pl"] * rpl

            if (screenvae_consistency_fn is not None and cfg.get("p1_w_screenvae_consistency", 0.0) > 0) or \
               (screenvae_patch_match_fn is not None and cfg.get("p1_w_screenvae_patch_match", 0.0) > 0):
                comp_sv = gen * mask + img * (1 - mask)

            if screenvae_consistency_fn is not None and cfg.get("p1_w_screenvae_consistency", 0.0) > 0:
                svc = screenvae_consistency_fn(comp_sv, img, mask)
                loss_g = loss_g + cfg["p1_w_screenvae_consistency"] * svc

            if screenvae_patch_match_fn is not None and cfg.get("p1_w_screenvae_patch_match", 0.0) > 0:
                svpm = screenvae_patch_match_fn(comp_sv, img, mask)
                loss_g = loss_g + cfg["p1_w_screenvae_patch_match"] * svpm

            if distill_fn is not None and any(cfg.get(k, 0.0) > 0 for k in (
                    "p1_w_distill_out", "p1_w_distill_feat",
                    "p1_w_distill_wavelet", "p1_w_distill_patchnce",
                    "p1_w_distill_svae")):
                d_out, d_feat, d_wav, d_patch = distill_fn(gen, inp, mask, unwrap(G))
                gn = {"out": 1.0, "feat": 1.0, "wavelet": 1.0, "patchnce": 1.0}
                if cfg.get("distill_adaptive_gn", False):
                    from mangainpaint.distill import adaptive_gn_multipliers
                    gn = adaptive_gn_multipliers(
                        distill_fn, loss_g,
                        {"out": d_out, "feat": d_feat, "wavelet": d_wav, "patchnce": d_patch},
                        gen, getattr(unwrap(G), "last_bottleneck", None), cfg)
                loss_g = (loss_g + cfg.get("p1_w_distill_out", 0.0) * gn["out"] * d_out
                          + cfg.get("p1_w_distill_feat", 0.0) * gn["feat"] * d_feat
                          + cfg.get("p1_w_distill_wavelet", 0.0) * gn["wavelet"] * d_wav
                          + cfg.get("p1_w_distill_patchnce", 0.0) * gn["patchnce"] * d_patch)
                if cfg.get("p1_w_distill_svae", 0.0) > 0:
                    d_svae = distill_fn.svae_kd(gen, img, mask)
                    loss_g = loss_g + cfg["p1_w_distill_svae"] * d_svae

        opt_g.zero_grad(set_to_none=True)
        scaler_g.scale(loss_g).backward()
        scaler_g.unscale_(opt_g)
        nn.utils.clip_grad_norm_(unwrap(G).parameters(), cfg["grad_clip"])
        scaler_g.step(opt_g); scaler_g.update()

        g_tot += float(loss_g.item())
        step += 1
        timer.mark_compute_end()
        timer.mark_fetch_start()

    nd = max(1, step)
    g_t = reduce_mean(torch.tensor(g_tot / nd, device=device))
    return g_t.item(), float('nan')


# ══════════════════════════════════════════════════════════
# TRAIN — Phase 2 (G+D)
# ══════════════════════════════════════════════════════════
def train_phase2(G, D, opt_g, opt_d, scaler_g, scaler_d, loader,
                 lpips_train_fn, resnet_pl_fn, screenvae_consistency_fn, screenvae_patch_match_fn,
                 distill_fn, cfg, epoch, device, rank, timer):
    G.train(); D.train()
    g_tot = d_tot = dr = df = 0.0
    step = 0

    pbar = loader
    if is_main(rank):
        pbar = tqdm(loader, desc=f"Ep{epoch + 1} [P2]", leave=False)

    timer.mark_fetch_start()
    for batch in pbar:
        timer.mark_fetch_end()
        img = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        inp = batch["model_input"].to(device, non_blocking=True)
        ring = make_ring(mask, cfg["ring_radius"])
        focus = (mask + ring).clamp(0, 1)
        ink_w = make_ink_weight_map(img, mask,
                                    threshold=cfg["ink_threshold"],
                                    extra=cfg["ink_extra"])

        # ── DISCRIMINATOR STEP ────────────────────────────
        with torch.amp.autocast('cuda'):
            with torch.no_grad():
                gen = call_g(G, inp, batch, device)
                comp = gen * mask + img * (1 - mask)
            z = torch.zeros_like(mask)
            real_logit, real_feats = D(img, z, return_feats=True)
            fake_logit = D(comp.detach(), mask)
            loss_d = d_hinge(real_logit, fake_logit)

        if step % cfg["p2_r1_every"] == 0:
            pen = lazy_r1_penalty(unwrap(D), img)
            loss_d = loss_d + cfg["p2_w_r1"] * pen * cfg["p2_r1_every"] * 0.5

        opt_d.zero_grad(set_to_none=True)
        scaler_d.scale(loss_d).backward()
        scaler_d.unscale_(opt_d)
        nn.utils.clip_grad_norm_(unwrap(D).parameters(), cfg["grad_clip"])
        scaler_d.step(opt_d); scaler_d.update()

        dr += float(real_logit.mean().detach().item())
        df += float(fake_logit.mean().detach().item())
        d_tot += float(loss_d.item())

        real_feats_cached = [f.detach() for f in real_feats]

        # ── GENERATOR STEP ────────────────────────────────
        with torch.amp.autocast('cuda'):
            gen = call_g(G, inp, batch, device)
            comp = gen * mask + img * (1 - mask)

            hole = charbonnier(gen, img, mask)
            ring_loss = charbonnier(gen, img, ring)
            vid = charbonnier(gen, img, 1 - mask)
            edg = ((sobel_mag(gen) - sobel_mag(img)).abs() * focus).mean()
            fft_l = fft_mag_loss(gen * focus, img * focus)
            ink_loss = charbonnier_weighted(gen, img, mask, ink_w)
            bitonal = bitonal_commitment_loss(gen, mask)
            bitonal_g = bitonal_commitment_loss_gated(
                gen, img, mask, cfg.get("bitonal_gate_thresh", 0.05))
            rstats = regional_stats_loss(gen, img, mask, cfg.get("regional_stats_window", 9))
            ring_cons = ring_consistency_loss(gen, mask, ring)
            pmatch = (patch_match_loss(gen, img, mask) if cfg.get("p2_w_patch_match", 0.0) > 0
                     else gen.new_zeros(()))

            loss_g = (cfg["p2_w_hole_rec"] * hole +
                      cfg["p2_w_ring_rec"] * ring_loss +
                      cfg["p2_w_valid_id"] * vid +
                      cfg["p2_w_edge"] * edg +
                      cfg["p2_w_fft"] * fft_l +
                      cfg["p2_w_ink"] * ink_loss +
                      cfg.get("p2_w_bitonal", 0.0) * bitonal +
                      cfg.get("p2_w_bitonal_gated", 0.0) * bitonal_g +
                      cfg.get("p2_w_regional_stats", 0.0) * rstats +
                      cfg.get("p2_w_ring_consistency", 0.0) * ring_cons +
                      cfg.get("p2_w_patch_match", 0.0) * pmatch)

            fake_logit_g, fake_feats = D(comp, mask, return_feats=True)
            gan = g_hinge(fake_logit_g)
            fm = sum((a - b).abs().mean() for a, b in zip(fake_feats, real_feats_cached))
            fm = fm / max(1, len(fake_feats))
            loss_g = loss_g + cfg["p2_w_gan"] * gan + cfg["p2_w_fm"] * fm

            if LPIPS_AVAIL and cfg["p2_w_lpips"] > 0 and lpips_train_fn is not None:
                lp = _lpips_term(lpips_train_fn, comp, img)
                loss_g = loss_g + cfg["p2_w_lpips"] * lp

            if resnet_pl_fn is not None and cfg.get("p2_w_resnet_pl", 0.0) > 0:
                rpl = resnet_pl_fn(gen, img)
                loss_g = loss_g + cfg["p2_w_resnet_pl"] * rpl

            if screenvae_consistency_fn is not None and cfg.get("p2_w_screenvae_consistency", 0.0) > 0:
                svc = screenvae_consistency_fn(comp, img, mask)  # `comp` already composited above
                loss_g = loss_g + cfg["p2_w_screenvae_consistency"] * svc

            if screenvae_patch_match_fn is not None and cfg.get("p2_w_screenvae_patch_match", 0.0) > 0:
                svpm = screenvae_patch_match_fn(comp, img, mask)
                loss_g = loss_g + cfg["p2_w_screenvae_patch_match"] * svpm

            if distill_fn is not None and any(cfg.get(k, 0.0) > 0 for k in (
                    "p2_w_distill_out", "p2_w_distill_feat",
                    "p2_w_distill_wavelet", "p2_w_distill_patchnce",
                    "p2_w_distill_svae")):
                d_out, d_feat, d_wav, d_patch = distill_fn(gen, inp, mask, unwrap(G))
                gn = {"out": 1.0, "feat": 1.0, "wavelet": 1.0, "patchnce": 1.0}
                if cfg.get("distill_adaptive_gn", False):
                    from mangainpaint.distill import adaptive_gn_multipliers
                    gn = adaptive_gn_multipliers(
                        distill_fn, loss_g,
                        {"out": d_out, "feat": d_feat, "wavelet": d_wav, "patchnce": d_patch},
                        gen, getattr(unwrap(G), "last_bottleneck", None), cfg)
                loss_g = (loss_g + cfg.get("p2_w_distill_out", 0.0) * gn["out"] * d_out
                          + cfg.get("p2_w_distill_feat", 0.0) * gn["feat"] * d_feat
                          + cfg.get("p2_w_distill_wavelet", 0.0) * gn["wavelet"] * d_wav
                          + cfg.get("p2_w_distill_patchnce", 0.0) * gn["patchnce"] * d_patch)
                if cfg.get("p2_w_distill_svae", 0.0) > 0:
                    d_svae = distill_fn.svae_kd(gen, img, mask)
                    loss_g = loss_g + cfg["p2_w_distill_svae"] * d_svae

        opt_g.zero_grad(set_to_none=True)
        scaler_g.scale(loss_g).backward()
        scaler_g.unscale_(opt_g)
        nn.utils.clip_grad_norm_(unwrap(G).parameters(), cfg["grad_clip"])
        scaler_g.step(opt_g); scaler_g.update()

        g_tot += float(loss_g.item())
        step += 1
        timer.mark_compute_end()
        timer.mark_fetch_start()

    nd = max(1, step)
    dr_t = reduce_mean(torch.tensor(dr / nd, device=device))
    df_t = reduce_mean(torch.tensor(df / nd, device=device))
    g_t = reduce_mean(torch.tensor(g_tot / nd, device=device))
    d_t = reduce_mean(torch.tensor(d_tot / nd, device=device))

    if is_main(rank):
        if d_t.item() < 0.1:
            ok = "collapsing (loss -> 0)"
        elif df_t.item() < -1.5:
            ok = "D dominating"
        elif dr_t.item() < 0.3:
            ok = "G dominating"
        elif abs(dr_t.item() - 1.0) < 0.4 and abs(df_t.item() + 1.0) < 0.5:
            ok = "healthy"
        else:
            ok = "ok"
        print(f"  D health: real={dr_t.item():+.3f} fake={df_t.item():+.3f} d_loss={d_t.item():.3f} [{ok}]")

    return g_t.item(), d_t.item()


@torch.no_grad()
def evaluate(G, loader, lpips_eval_fn, device, rank, desc="val", ink_threshold=0.4):
    """Hole-region metrics, both pooled (`overall`, for continuity with every
    legacy/past-session number) and stratified by mask ink-density
    (`strata`), plus the ink-density-aware `score` used for checkpoint
    selection. See `mangainpaint/metrics.py: selection_score` for why pooled
    PSNR/SSIM alone is a gameable selection criterion on manga's bimodal
    (mostly-blank-paper) pixel distribution.
    """
    G.eval()
    samples = []  # per-sample dicts, this rank's val shard only
    pbar = loader
    if is_main(rank):
        pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        img = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        with torch.amp.autocast('cuda'):
            inp = batch["model_input"].to(device, non_blocking=True)
            out = call_g(unwrap(G), inp, batch, device)
        comp = out * mask + img * (1 - mask)
        lp_b = None
        if LPIPS_AVAIL and lpips_eval_fn is not None:
            lp_b = _lpips_per_sample(lpips_eval_fn, comp, img).tolist()
        for b in range(img.size(0)):
            p, t, m = comp[b:b + 1], img[b:b + 1], mask[b:b + 1]
            samples.append({
                "ink_frac": hole_ink_frac(t, m, ink_threshold),
                "psnr": hole_psnr(p, t, m),
                "ssim": hole_ssim(p, t, m),
                "grad_l1": hole_grad_l1(p, t, m),
                "edge_f1": hole_edge_f1(p, t, m),
                "lpips": lp_b[b] if lp_b is not None else float('nan'),
            })

    metric_keys = ("psnr", "ssim", "grad_l1", "edge_f1", "lpips")

    overall = {}
    for k in metric_keys:
        mean, _ = reduce_mean_of([s[k] for s in samples], device)
        overall[k] = mean

    buckets = bucket_by_ink(samples)
    strata = {}
    for name, bucket in buckets.items():
        entry = {"n": reduce_count(len(bucket), device)}
        for k in metric_keys:
            mean, _ = reduce_mean_of([s[k] for s in bucket], device)
            entry[k] = mean
        strata[name] = entry

    overall["strata"] = strata
    overall["score"] = selection_score(strata)
    return overall


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
def main_ddp(rank, world_size, cfg, model_fn):
    setup_ddp(rank, world_size)
    # Seed comes from the recipe so that a replicate is a config change
    # rather than an edit to this file; 42 reproduces every published run.
    seed_everything(cfg.get("seed", 42) + rank)
    device = torch.device(f"cuda:{rank}")

    train_loader, val_loader, test_loader, train_sampler = make_loaders(cfg, rank, world_size)

    G, D = model_fn(cfg)
    G, D = G.to(device), D.to(device)

    if is_main(rank):
        n_g = sum(p.numel() for p in G.parameters() if p.requires_grad) / 1e6
        n_d = sum(p.numel() for p in D.parameters() if p.requires_grad) / 1e6
        print(f"G params: {n_g:.3f}M | D params: {n_d:.3f}M")
        print(f"Image size: {cfg['image_size']} | Per-GPU batch: {cfg['batch_size']} "
              f"| Effective batch: {cfg['batch_size'] * world_size}")
        print(f"Phase 1 (G-only): epochs 0..{cfg['gan_phase_start'] - 1}")
        print(f"Phase 2 (G+D):    epochs {cfg['gan_phase_start']}..{cfg['epochs'] - 1}")
        if cfg["d_refresh_every"] > 0:
            refresh_eps = list(range(
                cfg["gan_phase_start"] + cfg["d_refresh_every"],
                cfg["epochs"],
                cfg["d_refresh_every"]))
            print(f"D refresh at epochs: {refresh_eps}")

    if cfg.get("use_compile"):
        G = torch.compile(G)
        D = torch.compile(D)

    G = nn.parallel.DistributedDataParallel(G, device_ids=[rank], find_unused_parameters=False)
    D = nn.parallel.DistributedDataParallel(D, device_ids=[rank], find_unused_parameters=False)

    opt_g = torch.optim.Adam(G.parameters(), lr=cfg["lr_g"], betas=cfg["betas"])
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg["lr_d"], betas=cfg["betas"])

    scaler_g = torch.amp.GradScaler('cuda')
    scaler_d = torch.amp.GradScaler('cuda')

    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, cfg["epochs"], eta_min=1e-5)
    p2_epochs = max(1, cfg["epochs"] - cfg["gan_phase_start"])
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, p2_epochs, eta_min=5e-6)

    lpips_train_fn = lpips_eval_fn = None
    if LPIPS_AVAIL:
        train_net = cfg.get("lpips_train_net", "squeeze")
        eval_net = cfg.get("lpips_eval_net", "vgg")
        lpips_eval_fn = lpips_lib.LPIPS(net=eval_net, verbose=False).to(device)
        for p in lpips_eval_fn.parameters(): p.requires_grad_(False)
        if train_net == eval_net:
            lpips_train_fn = lpips_eval_fn
        else:
            lpips_train_fn = lpips_lib.LPIPS(net=train_net, verbose=False).to(device)
            for p in lpips_train_fn.parameters(): p.requires_grad_(False)

    # LaMa's real "High Receptive Field Perceptual Loss" (resnet_pl) --
    # optional, built only when a run's CFG actually turns it on (weight=1.0
    # here deliberately: this file's own cfg["p1_w_resnet_pl"]/
    # cfg["p2_w_resnet_pl"] are the single source of truth for the term's
    # weight, matching how every other loss term in this file is weighted
    # at the call site rather than inside the loss function itself).
    resnet_pl_fn = None
    if cfg.get("p1_w_resnet_pl", 0.0) > 0 or cfg.get("p2_w_resnet_pl", 0.0) > 0:
        resnet_pl_fn = ResNetPL(weight=1.0, weights_path=cfg["resnet_pl_weights_path"],
                                input_size=cfg.get("resnet_pl_input_size", 256)).to(device)

    # ScreenVAE re-encode consistency loss -- same optional/CFG-gated
    # pattern as resnet_pl_fn above.
    screenvae_consistency_fn = None
    if cfg.get("p1_w_screenvae_consistency", 0.0) > 0 or cfg.get("p2_w_screenvae_consistency", 0.0) > 0:
        screenvae_consistency_fn = ScreenVAEConsistencyLoss(
            weights_dir=cfg["screenvae_weights_dir"]).to(device)

    # ScreenVAE-latent patch-match loss -- same optional/CFG-gated pattern
    # as resnet_pl_fn/screenvae_consistency_fn above.
    screenvae_patch_match_fn = None
    if cfg.get("p1_w_screenvae_patch_match", 0.0) > 0 or cfg.get("p2_w_screenvae_patch_match", 0.0) > 0:
        screenvae_patch_match_fn = ScreenVAEPatchMatchLoss(
            weights_dir=cfg["screenvae_weights_dir"]).to(device)

    # Fine-tuned-LaMa teacher distillation -- same optional/CFG-gated
    # pattern as the three fns above. Imported lazily so runs that don't
    # distill never touch the LaMa source tree.
    distill_fn = None
    if any(cfg.get(k, 0.0) > 0 for k in (
            "p1_w_distill_out", "p1_w_distill_feat", "p1_w_distill_wavelet", "p1_w_distill_patchnce",
            "p2_w_distill_out", "p2_w_distill_feat", "p2_w_distill_wavelet", "p2_w_distill_patchnce",
            "p1_w_distill_svae", "p2_w_distill_svae")):
        from mangainpaint.distill import DistillLoss
        # distill_svae_weights_dir (optional): enables the 5th, ScreenVAE-
        # latent KD term (student-vs-TEACHER latents; distinct from
        # screenvae_weights_dir's consistency loss, which targets GT).
        distill_fn = DistillLoss(teacher_ckpt=cfg["distill_teacher_ckpt"],
                                 hole_mult=cfg.get("distill_hole_mult", 4.0),
                                 patch_temperature=cfg.get("distill_patch_temperature", 0.07),
                                 patch_max_positions=cfg.get("distill_patch_max_positions", 256),
                                 screenvae_weights_dir=cfg.get("distill_svae_weights_dir")).to(device)
        if is_main(rank):
            n_t = sum(p.numel() for p in distill_fn.parameters()) / 1e6
            print(f"Distill teacher loaded ({n_t:.1f}M, frozen) from {cfg['distill_teacher_ckpt']}"
                  + (" + ScreenVAE-latent KD" if distill_fn.svae is not None else "")
                  + (" + adaptive GN weighting" if cfg.get("distill_adaptive_gn", False) else ""))

    # ── Resume ──
    start_epoch = 0
    best = -1e9
    refresh_marks = []
    hist = {k: [] for k in ["train_g", "train_d", "val_psnr", "val_ssim",
                            "val_grad_l1", "val_edge_f1", "val_lpips", "val_score"]}

    if cfg.get("resume") and os.path.exists(cfg["resume"]):
        if is_main(rank):
            print(f"Resuming from {cfg['resume']}...")
        ckpt = torch.load(cfg["resume"], map_location=device)
        unwrap(G).load_state_dict(ckpt["G"])
        if "D" in ckpt: unwrap(D).load_state_dict(ckpt["D"])
        if "opt_g" in ckpt: opt_g.load_state_dict(ckpt["opt_g"])
        if "opt_d" in ckpt: opt_d.load_state_dict(ckpt["opt_d"])
        start_epoch = ckpt.get("epoch", -1) + 1
        best = ckpt.get("score", -1e9)
        if "hist" in ckpt:
            hist = ckpt["hist"]
            for k in ["train_g", "train_d", "val_psnr", "val_ssim",
                      "val_grad_l1", "val_edge_f1", "val_lpips", "val_score"]:
                hist.setdefault(k, [])  # older checkpoints predate val_score
        if "refresh_marks" in ckpt: refresh_marks = ckpt["refresh_marks"]
        for e in range(start_epoch):
            sched_g.step()
            if e >= cfg["gan_phase_start"]:
                sched_d.step()

    if is_main(rank):
        os.makedirs(cfg["ckpt_dir"], exist_ok=True)
        os.makedirs(cfg["vis_dir"], exist_ok=True)
        vis_batch = build_vis_batch(val_loader.dataset, cfg, n=4)
    else:
        vis_batch = None

    if is_main(rank):
        print(f"\n{'Ep':>4} | {'G-loss':>8} {'D-loss':>8} | "
              f"{'PSNR':>6} {'SSIM':>6} {'GradL1':>7} {'EdgeF1':>7} {'LPIPS':>6} | Score | Time")
        print("-" * 90)

    profile_timing = cfg.get("profile_timing", False)

    for ep in range(start_epoch, cfg["epochs"]):
        train_sampler.set_epoch(ep)
        in_phase2 = ep >= cfg["gan_phase_start"]
        timer = EpochTimer(profile_timing)
        ep_t0 = time.perf_counter()

        # ── D refresh check ──
        if (in_phase2
                and cfg["d_refresh_every"] > 0
                and ep > cfg["gan_phase_start"]
                and (ep - cfg["gan_phase_start"]) % cfg["d_refresh_every"] == 0):
            if is_main(rank):
                unwrap(D).refresh()
                refresh_marks.append(ep)
            broadcast_state(D, src=0)
            opt_d.state.clear()
            import torch.distributed as dist
            dist.barrier()
            if is_main(rank):
                print(f"  D refresh at epoch {ep + 1} (optimizer state cleared)")

        if in_phase2:
            tg, td = train_phase2(G, D, opt_g, opt_d, scaler_g, scaler_d, train_loader,
                                  lpips_train_fn, resnet_pl_fn, screenvae_consistency_fn,
                                  screenvae_patch_match_fn, distill_fn, cfg, ep, device, rank, timer)
        else:
            tg, td = train_phase1(G, opt_g, scaler_g, train_loader,
                                  lpips_train_fn, resnet_pl_fn, screenvae_consistency_fn,
                                  screenvae_patch_match_fn, distill_fn, cfg, ep, device, rank, timer)

        if (is_main(rank) and cfg.get("distill_adaptive_gn", False)
                and distill_fn is not None
                and getattr(distill_fn, "_gn_state", None) is not None):
            ema = distill_fn._gn_state["ema"]
            print("  GN multipliers (EMA): "
                  + " ".join(f"{k}={v:.3f}" for k, v in sorted(ema.items())))

        vm = evaluate(G, val_loader, lpips_eval_fn, device, rank,
                      ink_threshold=cfg["ink_threshold"])

        sched_g.step()
        if in_phase2:
            sched_d.step()

        ep_dt = time.perf_counter() - ep_t0

        if is_main(rank):
            for k, v in [("train_g", tg), ("train_d", td),
                         ("val_psnr", vm["psnr"]), ("val_ssim", vm["ssim"]),
                         ("val_grad_l1", vm["grad_l1"]), ("val_edge_f1", vm["edge_f1"]),
                         ("val_lpips", vm["lpips"]), ("val_score", vm["score"])]:
                hist[k].append(v)

            # Selection score is ink-density-stratified EdgeF1/LPIPS (see
            # mangainpaint/metrics.py: selection_score) -- NOT the old pooled
            # PSNR/SSIM-heavy composite, which rewarded matching the mostly-
            # blank-paper majority of a hole over reconstructing ink.
            score = vm["score"]
            star = ""
            if score > best:
                best = score; star = "*"
                torch.save({
                    "G": unwrap(G).state_dict(),
                    "D": unwrap(D).state_dict(),
                    "opt_g": opt_g.state_dict(),
                    "opt_d": opt_d.state_dict(),
                    "epoch": ep, "score": score, "metrics": vm, "cfg": cfg,
                    "hist": hist, "refresh_marks": refresh_marks,
                }, f"{cfg['ckpt_dir']}/best.pt")
            torch.save({
                "G": unwrap(G).state_dict(),
                "D": unwrap(D).state_dict(),
                "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict(),
                "epoch": ep, "score": score, "hist": hist,
                "refresh_marks": refresh_marks,
            }, f"{cfg['ckpt_dir']}/last.pt")

            phase_tag = "P2" if in_phase2 else "P1"
            td_str = f"{td:>8.4f}" if not np.isnan(td) else "    n/a "
            lps = f"{vm['lpips']:.4f}" if not np.isnan(vm['lpips']) else "  n/a"
            print(f"{ep + 1:>3}{phase_tag} | {tg:>8.4f} {td_str} | "
                  f"{vm['psnr']:>6.2f} {vm['ssim']:>6.4f} {vm['grad_l1']:>7.4f} "
                  f"{vm['edge_f1']:>7.4f} {lps:>6} | {score:.4f} {star} | {ep_dt:6.1f}s")
            if profile_timing:
                print(f"       timing: data-wait={timer.wait:6.1f}s  compute={timer.compute:6.1f}s")
            strata_str = " | ".join(
                f"{name}(n={s['n']}) EdgeF1={s['edge_f1']:.3f} LPIPS={s['lpips']:.4f}"
                if s['n'] > 0 else f"{name}(n=0)"
                for name, s in vm["strata"].items()
            )
            print(f"       by ink-density: {strata_str}")

            if (ep + 1) % cfg["show_every"] == 0 or ep == 0 or ep + 1 == cfg["gan_phase_start"]:
                visualize(G, vis_batch, ep + 1, device, cfg["vis_dir"], phase=phase_tag)
                plot_history(hist, cfg["vis_dir"], lpips_avail=LPIPS_AVAIL,
                             phase_boundary=cfg["gan_phase_start"],
                             refresh_marks=refresh_marks)

        import torch.distributed as dist
        dist.barrier()

    cleanup_ddp()


def run(cfg, model_fn):
    """Entry point: spawns one process per visible GPU (or runs single-GPU
    directly when launched under torchrun, via LOCAL_RANK)."""
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No CUDA devices available.")

    if "LOCAL_RANK" in os.environ:
        rank = int(os.environ["LOCAL_RANK"])
        main_ddp(rank, world_size, cfg, model_fn)
    else:
        mp.spawn(main_ddp, args=(world_size, cfg, model_fn), nprocs=world_size, join=True)
