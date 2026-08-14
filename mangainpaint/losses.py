"""Loss functions, shared by every training run."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def charbonnier(pred, target, mask=None, eps=1e-3):
    err = ((pred - target) ** 2 + eps ** 2) ** 0.5
    if mask is None: return err.mean()
    return (err * mask).sum() / (mask.sum() + 1.0)


def make_ring(mask, r=4):
    return (F.max_pool2d(mask, 2 * r + 1, 1, r) - mask).clamp(0, 1)


def make_ink_weight_map(target, mask, threshold=0.4, extra=2.0):
    """
    Build per-pixel weight map that emphasises dense-ink regions inside the hole.
    `target` is in [-1, 1]; convert to [0, 1] then threshold.
    Returns: weight map of shape (B, 1, H, W), values in [1.0, 1.0+extra].
    """
    target_01 = (target + 1.0) * 0.5     # [0,1] where 0=black/ink, 1=white/paper
    is_ink = (target_01 < threshold).float()
    # Only emphasise ink that lies inside the hole
    ink_in_hole = is_ink * mask
    return 1.0 + extra * ink_in_hole


def charbonnier_weighted(pred, target, base_mask, weight_map, eps=1e-3):
    """Charbonnier x per-pixel weights, normalised by total weight."""
    err = ((pred - target) ** 2 + eps ** 2) ** 0.5
    w = base_mask * weight_map
    return (err * w).sum() / (w.sum() + 1.0)


def sobel_mag(x):
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                      device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    return (F.conv2d(x, kx, padding=1) ** 2 + F.conv2d(x, ky, padding=1) ** 2 + 1e-6) ** 0.5


def fft_mag_loss(pred, target):
    with torch.amp.autocast(pred.device.type, enabled=False):
        Fp = torch.fft.rfft2(pred.float(), norm='ortho').abs()
        Ft = torch.fft.rfft2(target.float(), norm='ortho').abs()
    return (Fp - Ft).abs().mean()


def bitonal_commitment_loss(gen, mask):
    """Push hole pixels toward the nearest bitonal extreme (ink=-1 or
    paper=+1) instead of a blended gray, in the generator's [-1,1] output
    space. Targets the mean-seeking failure mode where reconstruction
    losses alone leave uncertain hole pixels sitting near 0.

    Content-blind (confirmed empirically): pushes every hole pixel toward
    an extreme with no signal for whether that's actually correct, so it
    fixes genuine blank-paper under-confidence but equally overwrites real
    screentone/graphic texture where the true answer is a mid-tone. See
    `bitonal_commitment_loss_gated` for a version that avoids this."""
    commit = torch.minimum((gen - 1.0).abs(), (gen + 1.0).abs())
    return (commit * mask).sum() / (mask.sum() + 1.0)


def bitonal_commitment_loss_gated(gen, target, mask, extreme_thresh=0.05):
    """Like `bitonal_commitment_loss`, but only penalizes hole pixels where
    the *ground truth* itself is already confidently near an extreme
    (paper or ink) -- pixels whose true content is a genuine mid-tone
    (screentone/graphic texture) are excluded from the push entirely,
    since committing them would be wrong, not a fix. Only usable at train
    time (needs `target`); the generator still has to produce the right
    behavior unconditionally at inference.

    `target` is in [-1,1]; `extreme_thresh` is a [0,1]-normalized margin
    (same convention as `make_ink_weight_map`'s `threshold`)."""
    target_01 = (target + 1.0) * 0.5
    confident_extreme = ((target_01 < extreme_thresh) |
                         (target_01 > 1.0 - extreme_thresh)).float()
    gate = confident_extreme * mask
    commit = torch.minimum((gen - 1.0).abs(), (gen + 1.0).abs())
    return (commit * gate).sum() / (gate.sum() + 1.0)


def regional_stats_loss(gen, target, mask, window=9):
    """Match local windowed mean AND variance of the prediction to the GT,
    inside the hole. Unlike `bitonal_commitment_loss`, this is content-aware
    by construction: a blank-paper region has local mean~1, variance~0, so
    matching GT statistics there still pushes the fill toward committing to
    white. A screentone/hatched region has a mid mean but *high* local
    variance, so matching GT statistics there demands the fill reproduce
    that variance instead of washing it out to gray -- the exact coupling
    that made `bitonal_commitment_loss` (blanket and GT-gated) fail: that
    loss could only push toward extremes, with no way to ask for
    "committed but textured".

    `gen`/`target` in [-1,1], `mask` in {0,1}. `window` must be odd.
    """
    pad = window // 2

    # d/dx sqrt(x) is infinite at x=0, so flat regions (common in bitonal
    # art -- exactly the blank-paper case this loss targets) produce
    # inf/NaN gradients through the variance sqrt below even with an eps
    # added pre-sqrt, IF that eps is small enough to underflow to exactly
    # 0 under fp16 autocast (fp16's smallest normal is ~6.1e-5; an eps of
    # 1e-8 silently rounds to 0.0 in fp16, reintroducing the exact
    # singularity it was meant to avoid). That NaN survives multiplication
    # by a zero loss weight (0 * inf = NaN in IEEE float), silently
    # corrupting every G gradient for the whole run regardless of whether
    # this loss is actually turned on -- found when a weight-rebalance
    # sweep showed byte-identical checkpoints across different loss-weight
    # values even after a first (fp32-only-verified) fix attempt. Fixed
    # properly by forcing this whole computation to fp32
    # (same pattern already used by `fft_mag_loss` above, for the same
    # class of AMP-precision numerical-stability reason), where a modest
    # eps survives.
    with torch.amp.autocast(gen.device.type, enabled=False):
        gen32, target32, mask32 = gen.float(), target.float(), mask.float()
        kernel = torch.ones(1, 1, window, window, device=gen32.device, dtype=gen32.dtype) / (window * window)

        def local_mean(x):
            return F.conv2d(x, kernel, padding=pad)

        def local_var(x, mean):
            return local_mean(x * x) - mean * mean

        gen_mean = local_mean(gen32)
        tgt_mean = local_mean(target32)
        gen_var = local_var(gen32, gen_mean).clamp_min(0)
        tgt_var = local_var(target32, tgt_mean).clamp_min(0)

        mean_err = (gen_mean - tgt_mean).abs()
        eps = 1e-6
        var_err = ((gen_var + eps).sqrt() - (tgt_var + eps).sqrt()).abs()

        err = mean_err + var_err
        return (err * mask32).sum() / (mask32.sum() + 1.0)


def _directional_energy(x, region, eps=1e-6):
    """3-bin (horizontal/vertical/diagonal) Sobel-family edge-energy split,
    pooled over `region` and normalized to sum to 1 -- a cheap, fp16-safe
    stand-in for a full gradient-orientation histogram (HOG-style), used
    by `ring_consistency_loss` below."""
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                      device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    kd = torch.tensor([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
                      device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1).pow(2)
    gy = F.conv2d(x, ky, padding=1).pow(2)
    gd = F.conv2d(x, kd, padding=1).pow(2)

    area = region.sum().clamp_min(1.0)
    eh = (gx * region).sum() / area
    ev = (gy * region).sum() / area
    ed = (gd * region).sum() / area
    tot = eh + ev + ed + eps
    return torch.stack([eh, ev, ed]) / tot


def ring_consistency_loss(gen, mask, ring, eps=1e-6):
    """Self-referential local-consistency loss: penalizes the generated
    hole's *aggregate* local statistics (brightness, contrast, edge
    orientation) for not matching the SAME image's own valid boundary
    ring -- computed entirely from `gen`, no ground truth needed for the
    comparison itself (the ring's own values are separately anchored to
    GT via `ring_rec` in the same training step, so it's a reliable style
    reference).

    Motivation: every existing loss term
    is either GT-referenced and mean-seeking under uncertainty
    (`hole_rec`/`ring_rec`/`regional_stats_loss`, all minimized by the
    conditional average when the true content is uncertain) or an
    unconditional realism signal that can't tell "generic but plausible
    manga texture" from "the actual right content for this panel" (the
    GAN loss against `ProjectedD`). Nothing forces the generated hole to
    resemble *this specific image's* own surrounding style. This loss is
    the first one that does, and unlike `regional_stats_loss` it has no
    positional correspondence between hole and ring, so it can only be
    satisfied by matching genuine aggregate style (not by copying
    GT-shaped local windows) -- and it works at real inference time too
    (self-referential, no GT dependency in the comparison itself), unlike
    every GT-referenced loss above.

    Deliberately scoped: unweighted global (not windowed/positional)
    mean/std, and a 3-bin directional-energy split instead of a
    continuous gradient-orientation histogram, to stay cheap and
    comparable in cost to `regional_stats_loss`.
    """
    with torch.amp.autocast(gen.device.type, enabled=False):
        gen32, mask32, ring32 = gen.float(), mask.float(), ring.float()

        def stats(region):
            area = region.sum().clamp_min(1.0)
            mean = (gen32 * region).sum() / area
            var = ((gen32 - mean) ** 2 * region).sum() / area
            std = (var + eps).sqrt()
            return mean, std

        hole_mean, hole_std = stats(mask32)
        ring_mean, ring_std = stats(ring32)

        hole_dir = _directional_energy(gen32, mask32, eps)
        ring_dir = _directional_energy(gen32, ring32, eps)

        loss = ((hole_mean - ring_mean).abs() + (hole_std - ring_std).abs() +
                (hole_dir - ring_dir).abs().sum())
    return loss


def _patch_match_core(query_full, candidate_full, mask, patch, stride,
                      hole_thresh, valid_thresh, min_patches):
    """Shared nearest-valid-patch matching core, factored out so
    `patch_match_loss` (raw pixel space, below) and
    `model_screenvae.ScreenVAEPatchMatchLoss` (screentone latent space)
    don't duplicate the unfold/cdist/argmin mechanics -- only the space
    the patches are drawn from differs between the two callers.
    `query_full`/`candidate_full`/`mask` are all `(B,C,H,W)`, same H/W
    (the mask is resized by the caller if its native resolution differs
    from the comparison space's, though for both current callers it
    already matches: raw pixels and `ScreenVAE`'s dense per-pixel latent
    are both native input resolution)."""
    with torch.amp.autocast(query_full.device.type, enabled=False):
        q32, c32, m32 = query_full.float(), candidate_full.float(), mask.float()
        B = q32.shape[0]
        total = q32.new_zeros(())
        n_contrib = 0

        for b in range(B):
            g_patches = F.unfold(q32[b:b + 1], kernel_size=patch, stride=stride)[0].t()   # L, C*p*p
            im_patches = F.unfold(c32[b:b + 1], kernel_size=patch, stride=stride)[0].t()  # L, C*p*p
            m_patches = F.unfold(m32[b:b + 1], kernel_size=patch, stride=stride)[0]       # p*p, L
            hole_frac = m_patches.mean(dim=0)  # L

            q_idx = (hole_frac > hole_thresh).nonzero(as_tuple=True)[0]
            c_idx = (hole_frac < valid_thresh).nonzero(as_tuple=True)[0]
            if q_idx.numel() < min_patches or c_idx.numel() < min_patches:
                continue

            query = g_patches[q_idx]         # Q, C*p*p
            candidates = im_patches[c_idx]   # Kc, C*p*p

            with torch.no_grad():
                dists = torch.cdist(query, candidates)  # Q, Kc
                nn_idx = dists.argmin(dim=1)             # Q

            matched = candidates[nn_idx]  # Q, C*p*p (no grad needed -- real image)
            per_patch = (query - matched).pow(2).mean(dim=1)  # Q
            total = total + per_patch.mean()
            n_contrib += 1

        if n_contrib == 0:
            return q32.new_zeros(())
        return total / n_contrib


def patch_match_loss(gen, img, mask, patch=7, stride=8,
                     hole_thresh=0.5, valid_thresh=0.02, min_patches=4):
    """Differentiable nearest-valid-patch matching loss (a PatchMatch/
    Contextual-Loss-style term), forcing generated hole content to
    resemble a *specific real patch* found elsewhere in the SAME image's
    own valid region -- not just match aggregate statistics the way
    `regional_stats_loss`/`ring_consistency_loss` do.

    Motivation: both prior loss-level attempts at this project's
    periodic-artifact problem were "moment-matching" losses
    (GT-positional or self-referential
    aggregate statistics), and both were satisfiable by a texture
    generator producing statistically-plausible-but-content-blind output
    -- neither can force *content verification*. This loss is different
    in kind: for each patch inside the generated hole, it finds the
    single nearest real patch among this image's own valid pixels (L2
    distance in raw pixel space) and pulls the generated patch directly
    toward that specific match. A generic, content-independent texture
    can only satisfy this if it happens to closely resemble some
    specific real patch actually present nearby -- a much stronger
    requirement than matching an aggregate mean/variance/direction
    histogram.

    Nearest-neighbor search is done under `no_grad` (standard practice
    for this loss family -- the match *index* is not something gradient
    descent can sensibly optimize through); the actual loss value is then
    recomputed with gradient flowing through the generator's own patch
    values only (the matched candidate comes from the real input image,
    not the generator, so it needs no gradient there).

    Scoped choices (logged, not correctness shortcuts): raw single-
    channel pixel-space patches (no VGG/perceptual feature extractor --
    manga's bitonal ink/paper structure is already strongly expressed in
    raw pixel space, and this avoids a new pretrained-network dependency
    for what is meant to be a first, cheap screen); computed per-sample
    in a small Python loop over the batch (batch sizes here are 2-8, so
    the loop overhead is negligible, and it trivially avoids any risk of
    one image's hole matching against a different image's content); a
    moderate 7x7 patch / stride-8 grid at native 384px resolution
    (~2300 patch positions per image) to keep the O(query x candidate)
    distance matrix cheap on a small GPU while still capturing real local
    ink/screentone texture (unlike a downsampled-then-patched version,
    which would blur out exactly the fine detail this loss needs to
    match against).
    """
    return _patch_match_core(gen, img, mask, patch, stride, hole_thresh, valid_thresh, min_patches)


def d_hinge(real, fake):
    return F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean()


def g_hinge(fake): return -fake.mean()


def lazy_r1_penalty(D, img):
    x = img[:8].detach().float().requires_grad_(True)
    z = torch.zeros_like(x[:, :1])
    logit = D(x, z)
    grad = torch.autograd.grad(logit.sum(), x, create_graph=True)[0]
    return grad.pow(2).view(x.size(0), -1).sum(1).mean()
