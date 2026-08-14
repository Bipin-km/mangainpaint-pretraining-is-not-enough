"""
Partial Convolution UNet — baseline reimplementation of
Liu et al. ECCV 2018: "Image Inpainting for Irregular Holes Using
Partial Convolutions" (https://arxiv.org/abs/1804.07723).

Faithful to the published architecture and original training recipe:
  - 7-level encoder/decoder UNet
  - All convolutions are PartialConv2d (mask-gated)
  - Loss: L1_hole + 6*L1_valid + 0.05*perceptual + 120*style + 0.1*TV
  - Adam, lr=2e-4

This is the trainable comparison baseline for our manga inpainting paper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ──────────────────────────────────────────────────────────
# Partial Convolution layer (Liu et al. 2018)
# ──────────────────────────────────────────────────────────
class PartialConv2d(nn.Conv2d):
    """
    Convolution where each output element is renormalized by the fraction
    of valid (mask=1) input elements in its receptive field.

    Mask convention: 1 = valid pixel (visible), 0 = hole.
    The mask is updated after each layer: any output position that saw
    at least one valid input becomes valid in the next mask.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, dilation=dilation, bias=bias)

        # Mask-update kernel: ones, applied per-channel in the mask domain
        self.register_buffer(
            'mask_kernel',
            torch.ones(self.out_channels, self.in_channels,
                       self.kernel_size[0], self.kernel_size[1])
        )
        self.window_size = self.in_channels * self.kernel_size[0] * self.kernel_size[1]

    def forward(self, x, mask):
        # mask is (B, 1, H, W); broadcast it to all input channels
        if mask.size(1) == 1 and x.size(1) > 1:
            mask_in = mask.expand(-1, x.size(1), -1, -1)
        else:
            mask_in = mask

        # Mask update + ratio MUST run in fp32 — fp16 overflows when mask is
        # sparse (mask_ratio = window_size / mask_out can exceed fp16's 65504
        # limit, then 0 * inf = NaN). Force fp32 here regardless of autocast.
        with torch.amp.autocast(x.device.type, enabled=False):
            mask_in_f = mask_in.float()
            with torch.no_grad():
                mask_out = F.conv2d(mask_in_f, self.mask_kernel.float(),
                                    bias=None, stride=self.stride,
                                    padding=self.padding, dilation=self.dilation)
                mask_ratio = self.window_size / (mask_out + 1e-8)
                mask_out = (mask_out > 0).float()
                mask_ratio = mask_ratio * mask_out

        # Standard conv on masked input — runs at autocast precision (fp16 OK)
        raw = super().forward(x * mask_in)

        # Apply normalization in fp32 to avoid fp16 overflow on the multiply
        with torch.amp.autocast(x.device.type, enabled=False):
            raw_f = raw.float()
            mask_ratio_f = mask_ratio.float()
            mask_out_f = mask_out.float()
            if self.bias is not None:
                bias_view = self.bias.float().view(1, -1, 1, 1)
                output = (raw_f - bias_view) * mask_ratio_f + bias_view
                output = output * mask_out_f
            else:
                output = raw_f * mask_ratio_f
            # Cast back to original dtype (matches autocast context)
            output = output.to(raw.dtype)

        # Output mask collapses back to single channel
        out_mask = mask_out_f[:, 0:1].to(raw.dtype)
        return output, out_mask


# ──────────────────────────────────────────────────────────
# Encoder/decoder blocks
# ──────────────────────────────────────────────────────────
class PConvBlock(nn.Module):
    """PartialConv → BN → ReLU (encoder) or LeakyReLU (decoder)."""
    def __init__(self, in_ch, out_ch, kernel_size, stride=1,
                 bn=True, activ='relu'):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = PartialConv2d(in_ch, out_ch, kernel_size,
                                  stride=stride, padding=padding,
                                  bias=not bn)
        self.bn = nn.BatchNorm2d(out_ch) if bn else None
        if activ == 'relu':
            self.activ = nn.ReLU(inplace=True)
        elif activ == 'leaky':
            self.activ = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.activ = None

    def forward(self, x, mask):
        x, mask = self.conv(x, mask)
        if self.bn is not None: x = self.bn(x)
        if self.activ is not None: x = self.activ(x)
        return x, mask


# ──────────────────────────────────────────────────────────
# PConv UNet
# ──────────────────────────────────────────────────────────
class PConvUNet(nn.Module):
    """
    7-level UNet (matches Liu et al. 2018, Table 4).
    Channels: 3 → 64 → 128 → 256 → 512 → 512 → 512 → 512 (bottleneck)
    Then mirrored decoder with skip connections.
    """
    def __init__(self, in_ch=3, out_ch=3):
        super().__init__()
        # Encoder
        self.enc1 = PConvBlock(in_ch, 64, kernel_size=7, stride=2, bn=False)
        self.enc2 = PConvBlock(64, 128, kernel_size=5, stride=2)
        self.enc3 = PConvBlock(128, 256, kernel_size=5, stride=2)
        self.enc4 = PConvBlock(256, 512, kernel_size=3, stride=2)
        self.enc5 = PConvBlock(512, 512, kernel_size=3, stride=2)
        self.enc6 = PConvBlock(512, 512, kernel_size=3, stride=2)
        self.enc7 = PConvBlock(512, 512, kernel_size=3, stride=2)

        # Decoder — concat with skip then 3x3 PartialConv
        self.dec7 = PConvBlock(512 + 512, 512, kernel_size=3, activ='leaky')
        self.dec6 = PConvBlock(512 + 512, 512, kernel_size=3, activ='leaky')
        self.dec5 = PConvBlock(512 + 512, 512, kernel_size=3, activ='leaky')
        self.dec4 = PConvBlock(512 + 256, 256, kernel_size=3, activ='leaky')
        self.dec3 = PConvBlock(256 + 128, 128, kernel_size=3, activ='leaky')
        self.dec2 = PConvBlock(128 + 64, 64, kernel_size=3, activ='leaky')
        self.dec1 = PConvBlock(64 + in_ch, out_ch, kernel_size=3,
                               bn=False, activ=None)

        # Kaiming init for stability — large UNet without it can NaN early
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x, mask):
        """
        x:    (B, 3, H, W)  in [-1, 1] (we pass repeat-channel grayscale)
        mask: (B, 1, H, W)  in {0, 1}, 1 = valid, 0 = hole
        Returns: (B, 3, H, W) inpainted image
        """
        # Encoder skip connections (input + each level's output)
        h1, m1 = self.enc1(x, mask)
        h2, m2 = self.enc2(h1, m1)
        h3, m3 = self.enc3(h2, m2)
        h4, m4 = self.enc4(h3, m3)
        h5, m5 = self.enc5(h4, m4)
        h6, m6 = self.enc6(h5, m5)
        h7, m7 = self.enc7(h6, m6)

        # Decoder: upsample, concat skip, partial-conv
        def up_concat(h, m, skip_h, skip_m, dec):
            h_up = F.interpolate(h, scale_factor=2, mode='nearest')
            m_up = F.interpolate(m, scale_factor=2, mode='nearest')
            # Align spatial dims if interpolation was off-by-one
            if h_up.shape[2:] != skip_h.shape[2:]:
                h_up = F.interpolate(h_up, size=skip_h.shape[2:], mode='nearest')
                m_up = F.interpolate(m_up, size=skip_m.shape[2:], mode='nearest')
            h_cat = torch.cat([h_up, skip_h], dim=1)
            m_cat = torch.cat([m_up, skip_m], dim=1)
            # Use only single-channel mask for the partial conv
            return dec(h_cat, m_cat[:, 0:1])

        h, m = up_concat(h7, m7, h6, m6, self.dec7)
        h, m = up_concat(h, m, h5, m5, self.dec6)
        h, m = up_concat(h, m, h4, m4, self.dec5)
        h, m = up_concat(h, m, h3, m3, self.dec4)
        h, m = up_concat(h, m, h2, m2, self.dec3)
        h, m = up_concat(h, m, h1, m1, self.dec2)
        h, m = up_concat(h, m, x, mask, self.dec1)

        return h


# ──────────────────────────────────────────────────────────
# VGG-16 feature extractor for perceptual + style losses
# ──────────────────────────────────────────────────────────
class VGG16Features(nn.Module):
    """Extract pool1, pool2, pool3 features (matches PConv paper)."""
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*[vgg[i] for i in range(5)])    # → relu1_2
        self.slice2 = nn.Sequential(*[vgg[i] for i in range(5, 10)])  # → relu2_2
        self.slice3 = nn.Sequential(*[vgg[i] for i in range(10, 17)]) # → relu3_3
        for p in self.parameters(): p.requires_grad_(False)
        # ImageNet normalisation
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        # x: (B, 3, H, W) in [0, 1]
        x = (x - self.mean) / self.std
        f1 = self.slice1(x)
        f2 = self.slice2(f1)
        f3 = self.slice3(f2)
        return f1, f2, f3


def gram_matrix(feat):
    """Gram matrix per sample, normalised by feature size."""
    B, C, H, W = feat.shape
    f = feat.view(B, C, -1)
    G = torch.bmm(f, f.transpose(1, 2)) / (C * H * W)
    return G


def count_params(m):
    return f"{sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6:.3f}M"
