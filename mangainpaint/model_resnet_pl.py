"""
LaMa's real training loss: `ResNetPL`, the "High Receptive Field Perceptual
Loss" (Suvorov et al., WACV 2022, "Resolution-robust Large Mask Inpainting
with Fourier Convolutions" -- the actual big-lama recipe, see
`external/lama/big-lama/config.yaml`'s `resnet_pl: {weight: 30}`, alongside
`perceptual.weight: 0` -- i.e. the *real* checkpoint uses this instead of a
plain VGG perceptual loss). Ported from
`external/lama/saicinpainting/training/losses/perceptual.py` +
`external/lama/models/ade20k/{resnet,base}.py`, minimized to only what
`ResNetPL` actually needs (the resnet50-dilated *encoder*; the reference
code also imports `ppm_deepsup` decoder machinery it never instantiates --
`ModelBuilder.get_encoder` only builds the encoder and uses the decoder
name solely to compose the pretrained-checkpoint's file path) so this file
has zero dependency on `external/lama`'s `segm_lib` package or its
`.mat`/`.csv` label-color assets.

Motivation: every other loss-recipe experiment run against the from-scratch
architectures only re-weighted the *existing* L1/adversarial/
feature-matching terms toward big-lama's own ratio, never actually
implemented big-lama's real perceptual term. Testing it is a loss-side
lever orthogonal to every architecture-bottleneck experiment (FFC/UFFC/
attention/linear attention) since it's a training loss, not a generator
change.

Pretrained weights: `mangainpaint/pretrained/ade20k/ade20k-resnet50dilated-ppm_deepsup/
encoder_epoch_20.pth` (CSAILVision/semantic-segmentation-pytorch, ADE20K
scene-parsing pretrained, same checkpoint used by real big-lama), fetched
from its origin (`http://sceneparsing.csail.mit.edu/model/
pytorch/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth`, the exact
URL in `external/lama/README.md`) -- verified to load cleanly (385
state-dict keys, matches this file's encoder 1:1 modulo harmless
`_tmp_running_mean/var` sync-batchnorm bookkeeping buffers the checkpoint
carries that this non-distributed encoder doesn't have, ignored via
`strict=False` exactly as the reference implementation does).

Same domain-range handling already established for every other pretrained
photo-domain backbone in this codebase (`model_projected_d.py`'s
`to_backbone_input`, `model_lama.py`'s LaMa-transfer generator): our images
are single-channel [-1,1]; ImageNet-pretrained backbones expect 3-channel
[0,1]-then-normalized input, so `to_resnet_pl_input` converts before the
frozen forward pass, same as the other two.
"""
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_resnet_pl_input(x, size=None):
    """x: (B,1,H,W) in [-1,1] -> (B,3,size,size) ImageNet-normalized in
    [0,1]-space. `size` mirrors `model_projected_d.py`'s
    `backbone_input_size` -- this encoder runs at stride-8 (`ResnetDilated`,
    dilate_scale=8), so its feature maps are 4x larger (H/8 x W/8) than a
    vanilla stride-32 resnet50 at the same input resolution; batch=4 @
    384px OOMs outright on a small GPU even with nothing else loaded, so
    downsizing before this frozen backbone (same lever ProjectedD already
    uses, same justification) is necessary for this to be affordable
    during real training, not just an optional speed tweak."""
    if size is not None and x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)
    x01 = (x + 1.0) * 0.5
    x3 = x01.repeat(1, 3, 1, 1)
    mean = IMAGENET_MEAN.to(x.device, x.dtype)
    std = IMAGENET_STD.to(x.device, x.dtype)
    return (x3 - mean) / std


# ──────────────────────────────────────────────────────────
# Deep-stem ResNet50 (3x conv3x3 stem instead of torchvision's single 7x7),
# verbatim architecture from external/lama/models/ade20k/resnet.py -- this
# specific stem is why torchvision's resnet50 can't be reused: the ADE20K
# pretrained checkpoint's state-dict keys/shapes only match this variant.
# ──────────────────────────────────────────────────────────
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class ResNet(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 128
        super().__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)


def resnet50_deepstem():
    return ResNet(Bottleneck, [3, 4, 6, 3])


class ResnetDilated(nn.Module):
    """Applies dilated (atrous) convolution to layer3/layer4 in-place so the
    encoder keeps a stride-8 output (vs. stride-32 for a vanilla resnet50) --
    the "High Receptive Field" in this loss's name: large receptive field
    (via dilation) at higher spatial resolution (vs. plain VGG perceptual
    loss's much smaller effective receptive field), per the paper's stated
    motivation for why this loss captures large-scale structure a VGG
    perceptual loss misses."""
    def __init__(self, orig_resnet, dilate_scale=8):
        super().__init__()
        if dilate_scale == 8:
            orig_resnet.layer3.apply(partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=2))

        self.conv1, self.bn1, self.relu1 = orig_resnet.conv1, orig_resnet.bn1, orig_resnet.relu1
        self.conv2, self.bn2, self.relu2 = orig_resnet.conv2, orig_resnet.bn2, orig_resnet.relu2
        self.conv3, self.bn3, self.relu3 = orig_resnet.conv3, orig_resnet.bn3, orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1, self.layer2 = orig_resnet.layer1, orig_resnet.layer2
        self.layer3, self.layer4 = orig_resnet.layer3, orig_resnet.layer4

    @staticmethod
    def _nostride_dilate(m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            elif m.kernel_size == (3, 3):
                m.dilation = (dilate, dilate)
                m.padding = (dilate, dilate)

    def forward(self, x):
        conv_out = []
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x); conv_out.append(x)
        x = self.layer2(x); conv_out.append(x)
        x = self.layer3(x); conv_out.append(x)
        x = self.layer4(x); conv_out.append(x)
        return conv_out


class ResNetPL(nn.Module):
    """Frozen ADE20K-pretrained resnet50-dilated encoder; loss is the sum of
    per-stage MSE between pred/target feature maps (layer1..layer4),
    matching `external/lama/saicinpainting/training/losses/perceptual.py`'s
    `ResNetPL.forward` exactly. `pred`/`target` here are this project's
    native [-1,1] single-channel tensors -- `to_resnet_pl_input` handles the
    domain conversion this loss's reference implementation doesn't need
    (LaMa's own pipeline already works in [0,1] RGB; the LaMa-transfer
    generator handles the same class of domain mismatch, see
    `model_lama.py`).
    """
    def __init__(self, weight=30.0, weights_path=None, dilate_scale=8, input_size=256):
        super().__init__()
        self.impl = ResnetDilated(resnet50_deepstem(), dilate_scale=dilate_scale)
        self.input_size = input_size
        if weights_path:
            sd = torch.load(weights_path, map_location="cpu")
            missing, unexpected = self.impl.load_state_dict(sd, strict=False)
            # Only the harmless sync-batchnorm bookkeeping buffers
            # (`_tmp_running_mean`/`_tmp_running_var`) should ever land in
            # `unexpected`; any real missing key means the architecture
            # ported above doesn't actually match this checkpoint.
            real_missing = [k for k in missing]
            if real_missing:
                raise RuntimeError(f"ResNetPL: missing keys loading {weights_path}: {real_missing}")
        self.impl.eval()
        for p in self.impl.parameters():
            p.requires_grad_(False)
        self.weight = weight

    def train(self, mode=True):
        # Always eval -- frozen backbone must use running BN stats, not
        # batch stats, same reasoning as model_projected_d.py's
        # FrozenEfficientNetBackbone.
        return super().train(False)

    def forward(self, pred, target):
        """pred/target: (B,1,H,W) in [-1,1]. No torch.no_grad() -- gradients
        must flow from this loss back into the generator's `pred`; only this
        module's own weights are frozen (requires_grad=False)."""
        pred3 = to_resnet_pl_input(pred, size=self.input_size)
        target3 = to_resnet_pl_input(target, size=self.input_size)
        pred_feats = self.impl(pred3)
        with torch.no_grad():
            target_feats = self.impl(target3)
        losses = [F.mse_loss(p, t) for p, t in zip(pred_feats, target_feats)]
        return torch.stack(losses).sum() * self.weight
