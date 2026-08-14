"""
Axis A3 (UFFC bottleneck): MangaFillNet with its vanilla FFCBlock swapped
for an adapted "Unbiased FFC" block (Chu et al., ICCV 2023, "Rethinking
Fast Fourier Convolution in Image Inpainting"), adapted from the reference
implementation (github.com/1911cty/Unbiased-Fast-Fourier-Convolution,
`FourierUnit_modified` in uffc.py).

Motivated by a real qualitative finding, not the literature alone: every
loss-recipe variant tried against the vanilla-FFC baseline (loss-weight
rebalance, regional-stats loss) left the exact same fixed,
image-independent diagonal hatch/basket-weave texture in every hole fill,
regardless of loss recipe. A periodic artifact that's identical across
unrelated images is the signature of "spectrum-shifting/ringing" -- a 1x1
conv in frequency space is a *global* spatial-domain operation, so a
single erroneous frequency bin rings across the entire image the same way
regardless of content. This is exactly the failure mode the UFFC paper
targets, which is the motivation to test it here.

Everything except the FFC block is reused unmodified from
`model_scratch.py` (Enc/DilRes/Dec/OutHead) so this is a single-variable
architecture swap, directly comparable against the existing Axis A1
(`MangaFillNet`) result.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mangainpaint.model_scratch import Enc, DilRes, Dec, OutHead


class FourierUnitUFFC(nn.Module):
    """Adapted from `FourierUnit_modified` (see module docstring for the
    reference repo). Simplified vs. the reference: dropped `groups`/`ffc3d`/
    `spatial_scale_factor`/`spectral_pos_encoding`/`use_se` (unused
    options in the reference's own ablations, not needed for a single
    bottleneck-resolution generator here). `in_channels == out_channels`
    throughout (matches how the reference is actually used -- the
    channel-count arithmetic on the second conv only works out under that
    constraint anyway, see module docstring math checked during porting).

    Three changes vs. this codebase's vanilla `FFCBlock._freq`, all from
    the reference:
    1. A learnable per-frequency-position map (`loc_map`) concatenated as
       an extra channel before the frequency-domain conv, so corrections
       can be position-dependent instead of globally uniform.
    2. A second conv applied after `fftshift` (DC centered) with a larger
       dilated receptive field -- lets a local kernel see across the
       low/high-frequency boundary, which it cannot do near the
       unshifted spectrum's corner-DC layout.
    3. Output is forced to preserve the input's mean (DC term) and
       clamped to the input's original dynamic range, directly
       suppressing the out-of-range spatial ringing/overshoot
       unconstrained frequency-domain convs can introduce.
    A learnable sigmoid gate (`lambda_base`) blends the corrected
    spectrum with the original so training can smoothly interpolate
    towards "do nothing" rather than being forced to use the correction
    from initialization.

    FFT ops forced to fp32 regardless of ambient autocast state -- same
    pattern already used by `model_scratch.FFCBlock._freq` and
    `external/lama`'s FFC fix, for the same reason: cuFFT/complex-to-fp16
    casting is unsafe at the non-power-of-2 feature sizes this generator
    actually runs at.
    """
    def __init__(self, channels, bottleneck_hw):
        super().__init__()
        self.channels = channels
        w_half = bottleneck_hw // 2 + 1
        self.loc_map = nn.Parameter(torch.rand(bottleneck_hw, w_half))
        self.lambda_base = nn.Parameter(torch.tensor(0.0))
        self.conv1 = nn.Conv2d(channels * 2 + 1, channels * 2, 1, bias=False)
        self.conv2 = nn.Conv2d(channels * 2 + 1, channels * 2, 3, padding=2, dilation=2, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        H, W = x.shape[-2:]
        with torch.amp.autocast(x.device.type, enabled=False):
            x32 = x.float()
            X = torch.fft.rfft2(x32, norm='ortho')
            ffted = torch.cat([X.real, X.imag], dim=1)  # (B, 2C, H, W//2+1)

            loc = self.loc_map.expand(x.size(0), 1, -1, -1)
            stage1 = self.conv1(torch.cat([ffted, loc], dim=1))
            stage1 = torch.fft.fftshift(stage1, dim=-2)
            stage1 = self.relu(stage1)

            loc_shift = torch.fft.fftshift(loc, dim=-2)
            stage2 = self.conv2(torch.cat([stage1, loc_shift], dim=1))
            stage2 = torch.fft.ifftshift(stage2, dim=-2)

            gate = torch.sigmoid(self.lambda_base)
            corrected = ffted * gate + stage2 * (1 - gate)

            r, i = corrected.chunk(2, dim=1)
            out = torch.fft.irfft2(torch.complex(r, i), s=(H, W), norm='ortho')

            # Ringing suppression (verbatim from the reference): re-center
            # on the input's own mean/DC term and clamp to its dynamic
            # range, both global (whole-tensor) reductions as in the
            # original -- not obviously "more correct" than a per-sample
            # reduction, but this is the published/verified behavior and
            # changing it would no longer be testing the real technique.
            out = out - out.mean() + x32.mean()
            out = out.clamp(x32.min() - 0.5, x32.max() + 0.5)
        return out.to(x.dtype)


class UFFCBlock(nn.Module):
    """Drop-in replacement for `model_scratch.FFCBlock`: identical
    local/global channel split and forward(x) signature, only the global
    path's frequency processing differs (see `FourierUnitUFFC`)."""
    def __init__(self, ch, ratio_g=0.5, bottleneck_hw=48):
        super().__init__()
        self.g = int(ch * ratio_g)
        self.l = ch - self.g
        if self.l > 0:
            self.lconv = nn.Conv2d(self.l, self.l, 3, 1, 1)
        if self.g > 0:
            self.freq = FourierUnitUFFC(self.g, bottleneck_hw)
        self.mix = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        if self.l == 0:
            return F.leaky_relu(self.mix(self.freq(x)), 0.2, inplace=True)
        lp, gp = x[:, :self.l], x[:, self.l:]
        lo = F.leaky_relu(self.lconv(lp), 0.2, inplace=True)
        go = self.freq(gp)
        return F.leaky_relu(self.mix(torch.cat([lo, go], dim=1)), 0.2, inplace=True)


class MangaFillNetUFFC(nn.Module):
    """Same architecture as `model_scratch.MangaFillNet`, with `f1`/`f2`
    swapped from `FFCBlock` to `UFFCBlock`. `image_size` must match the
    training config's `image_size` (bottleneck runs at image_size / 8,
    from the 3 stride-2 Enc blocks) since `UFFCBlock`'s `loc_map`
    parameter shape is fixed at construction, not lazily inferred."""
    def __init__(self, in_ch=2, base=32, ratio_g=0.5, image_size=384):
        super().__init__()
        b = base
        bottleneck_hw = image_size // 8
        self.e1 = Enc(in_ch, b)
        self.e2 = Enc(b, b * 2)
        self.e3 = Enc(b * 2, b * 4)
        bch = b * 4

        self.f1 = UFFCBlock(bch, ratio_g, bottleneck_hw)
        self.f2 = UFFCBlock(bch, ratio_g, bottleneck_hw)
        self.r1 = DilRes(bch, 1)
        self.r2 = DilRes(bch, 2)
        self.r4 = DilRes(bch, 4)
        self.r8 = DilRes(bch, 8)

        self.d1 = Dec(bch, b * 4, b * 2, hid_mult=4)
        self.d2 = Dec(b * 2, b * 2, b, hid_mult=4)
        self.d3 = Dec(b, b, b * 2, hid_mult=8)

        self.head = OutHead(b * 2)

    def forward(self, x):
        mask = x[:, 1:2]
        x, s1 = self.e1(x)
        x, s2 = self.e2(x)
        x, s3 = self.e3(x)
        x = self.f1(x) + x
        x = self.f2(x) + x
        x = self.r1(x, mask)
        x = self.r2(x, mask)
        x = self.r4(x, mask)
        x = self.r8(x, mask)
        x = self.d1(x, s3, mask)
        x = self.d2(x, s2, mask)
        x = self.d3(x, s1, mask)
        return self.head(x)
