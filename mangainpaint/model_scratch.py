"""
MangaFillNet v3 — wider final decoder + 2-stage output head.

Targets the dense-ink failure mode: the previous OutHead (single 3x3 conv)
couldn't synthesize the bimodal {ink, paper} distribution of manga.

Generator:    ~3.5M params (base=32, ratio_g=0.5)
Discriminator: ~0.7M params (single-scale PatchD, base=32)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────
# Mask-conditional SPADE normalization
# ──────────────────────────────────────────────────────────
class MaskSPADE(nn.Module):
    def __init__(self, ch, mask_ch=1, hid_mult=4):
        super().__init__()
        self.norm = nn.InstanceNorm2d(ch, affine=False)
        # hid_mult lets us bump capacity at the highest-res decoder layer
        hid = max(32, ch // 4) * hid_mult // 4
        self.shared = nn.Conv2d(mask_ch, hid, 3, 1, 1)
        self.gamma = nn.Conv2d(hid, ch, 3, 1, 1)
        self.beta = nn.Conv2d(hid, ch, 3, 1, 1)

    def forward(self, x, mask):
        m = F.interpolate(mask.float(), size=x.shape[2:], mode='nearest')
        h = F.leaky_relu(self.shared(m), 0.2, inplace=True)
        return self.norm(x) * (1 + self.gamma(h)) + self.beta(h)


# ──────────────────────────────────────────────────────────
# Fast Fourier Convolution block (global receptive field)
# ──────────────────────────────────────────────────────────
class FFCBlock(nn.Module):
    def __init__(self, ch, ratio_g=0.5):
        super().__init__()
        self.g = int(ch * ratio_g)
        self.l = ch - self.g
        if self.l > 0:
            self.lconv = nn.Conv2d(self.l, self.l, 3, 1, 1)
        if self.g > 0:
            fc = self.g * 2
            self.fc1 = nn.Conv2d(fc, fc, 1)
            self.fc2 = nn.Conv2d(fc, fc, 1)
        self.mix = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        if self.l == 0:
            return F.leaky_relu(self.mix(self._freq(x)), 0.2, inplace=True)
        lp, gp = x[:, :self.l], x[:, self.l:]
        lo = F.leaky_relu(self.lconv(lp), 0.2, inplace=True)
        go = self._freq(gp)
        return F.leaky_relu(self.mix(torch.cat([lo, go], dim=1)), 0.2, inplace=True)

    def _freq(self, g):
        B, C, H, W = g.shape
        with torch.amp.autocast(g.device.type, enabled=False):
            G = torch.fft.rfft2(g.float(), norm='ortho')
            ri = torch.cat([G.real, G.imag], dim=1)
        ri = F.leaky_relu(self.fc1(ri.to(g.dtype)), 0.2, inplace=True)
        ri = self.fc2(ri)
        with torch.amp.autocast(g.device.type, enabled=False):
            r, i = ri.float().chunk(2, dim=1)
            out = torch.fft.irfft2(torch.complex(r, i), s=(H, W), norm='ortho')
        return out.to(g.dtype)


# ──────────────────────────────────────────────────────────
# Dilated residual block with mask conditioning
# ──────────────────────────────────────────────────────────
class DilRes(nn.Module):
    def __init__(self, ch, d=1):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, 1, d, dilation=d)
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.n1 = MaskSPADE(ch)
        self.n2 = MaskSPADE(ch)

    def forward(self, x, mask):
        r = F.leaky_relu(self.n1(self.c1(x), mask), 0.2, inplace=True)
        return F.leaky_relu(x + self.n2(self.c2(r), mask), 0.2, inplace=True)


# ──────────────────────────────────────────────────────────
# Encoder
# ──────────────────────────────────────────────────────────
class Enc(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.c1 = nn.Conv2d(ic, oc, 3, 1, 1)
        self.c2 = nn.Conv2d(oc, oc, 3, 1, 1)

    def forward(self, x):
        x = F.leaky_relu(self.c1(x), 0.2, inplace=True)
        x = F.leaky_relu(self.c2(x), 0.2, inplace=True)
        return F.avg_pool2d(x, 2), x


# ──────────────────────────────────────────────────────────
# Decoder — accepts hid_mult to widen MaskSPADE at high-res
# ──────────────────────────────────────────────────────────
class Dec(nn.Module):
    def __init__(self, ic, sc, oc, hid_mult=4):
        super().__init__()
        self.c1 = nn.Conv2d(ic + sc, oc, 3, 1, 1)
        self.n1 = MaskSPADE(oc, hid_mult=hid_mult)
        self.c2 = nn.Conv2d(oc, oc, 3, 1, 1)
        self.n2 = MaskSPADE(oc, hid_mult=hid_mult)

    def forward(self, x, skip, mask):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if x.shape[2:] != skip.shape[2:]:
            x = x[:, :, :skip.shape[2], :skip.shape[3]]
        x = F.leaky_relu(self.n1(self.c1(torch.cat([x, skip], 1)), mask), 0.2, inplace=True)
        return F.leaky_relu(self.n2(self.c2(x), mask), 0.2, inplace=True)


# ──────────────────────────────────────────────────────────
# 2-stage output head with intermediate processing.
# Designed for bimodal {ink, paper} distribution synthesis.
# ──────────────────────────────────────────────────────────
class OutHead(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.norm = nn.InstanceNorm2d(ch, affine=True)
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(ch, 1, 3, 1, 1))

    def forward(self, x):
        x = F.leaky_relu(self.norm(self.conv1(x)), 0.2, inplace=True)
        return torch.tanh(self.conv2(x))


# ──────────────────────────────────────────────────────────
# Generator — wider final decoder for fine detail
# ──────────────────────────────────────────────────────────
class MangaFillNet(nn.Module):
    # dilations default (1, 2, 4, 8): every rate shares common factor 2 once
    # rate 1 is excluded -- the textbook Hybrid Dilated Convolution "gridding"
    # condition (Wang et al. 2018). Override with an HDC-safe schedule (e.g.
    # (1, 2, 5, 9)) to test whether that's the source of the persistent
    # periodic artifact seen across every bottleneck variant tried.
    def __init__(self, in_ch=2, base=32, ratio_g=0.5, dilations=(1, 2, 4, 8)):
        super().__init__()
        b = base
        # Encoder
        self.e1 = Enc(in_ch, b)
        self.e2 = Enc(b, b * 2)
        self.e3 = Enc(b * 2, b * 4)
        bch = b * 4

        # Bottleneck (global context via FFC + multi-scale dilation)
        self.f1 = FFCBlock(bch, ratio_g)
        self.f2 = FFCBlock(bch, ratio_g)
        self.r1 = DilRes(bch, dilations[0])
        self.r2 = DilRes(bch, dilations[1])
        self.r4 = DilRes(bch, dilations[2])
        self.r8 = DilRes(bch, dilations[3])

        # Decoder — final layer is WIDER (b*2 instead of b) for fine-detail synthesis
        self.d1 = Dec(bch, b * 4, b * 2, hid_mult=4)
        self.d2 = Dec(b * 2, b * 2, b, hid_mult=4)
        self.d3 = Dec(b, b, b * 2, hid_mult=8)   # widened: oc=b*2 (was b), hid_mult doubled

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


# ──────────────────────────────────────────────────────────
# Discriminator (unchanged from v2)
# ──────────────────────────────────────────────────────────
class PatchD(nn.Module):
    def __init__(self, in_ch=2, base=32):
        super().__init__()
        sn = nn.utils.spectral_norm

        def blk(ic, oc, norm=True, stride=2):
            layers = [sn(nn.Conv2d(ic, oc, 4, stride, 1))]
            if norm:
                layers.append(nn.InstanceNorm2d(oc, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.b1 = blk(in_ch, base, norm=False)
        self.b2 = blk(base, base * 2)
        self.b3 = blk(base * 2, base * 4)
        self.b4 = blk(base * 4, base * 8, stride=1)
        self.out = sn(nn.Conv2d(base * 8, 1, 3, 1, 1))

    def forward(self, x, return_feats=False):
        f = []
        for b in (self.b1, self.b2, self.b3, self.b4):
            x = b(x)
            f.append(x)
        return (self.out(x), f) if return_feats else self.out(x)

    def reinit_last_layers(self):
        """Re-initialize the last two blocks + output. Used for D-refresh."""
        sn = nn.utils.spectral_norm

        def _reinit(module):
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, a=0.2, nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.InstanceNorm2d) and module.affine:
                nn.init.ones_(module.weight); nn.init.zeros_(module.bias)

        self.b3.apply(_reinit)
        self.b4.apply(_reinit)
        self.out.apply(_reinit)


class SingleScaleD(nn.Module):
    """Single PatchD wrapped to keep the (img, mask) -> (logits, feats) API."""
    def __init__(self, in_ch=2, base=32):
        super().__init__()
        self.D = PatchD(in_ch, base)

    def forward(self, img_or_comp, mask, return_feats=False):
        inp = torch.cat([img_or_comp, mask], 1)
        if return_feats:
            logit, feats = self.D(inp, True)
            return logit, feats
        return self.D(inp)

    def refresh(self):
        self.D.reinit_last_layers()


def count_params(m):
    return f"{sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6:.3f}M"


def build_models(device, base=32, ratio_g=0.5):
    G = MangaFillNet(in_ch=2, base=base, ratio_g=ratio_g).to(device)
    D = SingleScaleD(in_ch=2, base=base).to(device)
    return G, D
