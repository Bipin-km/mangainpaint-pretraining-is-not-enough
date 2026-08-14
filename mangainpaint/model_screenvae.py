"""
The ScreenVAE-based branch: the biggest remaining lift of the msxie92
(Xie et al., SIGGRAPH 2021, "Seamless Manga Inpainting with Semantics
Awareness") ingredients explored in this project, after cheaper proxies
(noise injection, a self-referential edge-conditioned discriminator,
generator-side Sobel `EdgeHint`, all in `model_attn.py`) were rejected for
a coherent reason: none of them supplied a *genuinely new representation*
the network couldn't already derive from its own raw pixel input. This
module ports the one piece of msxie92 that actually is new: a frozen,
pretrained, *learned* continuous screentone latent space (`ScreenVAE`),
disentangled from structural line/edge content by construction.

This goes straight to that real ingredient rather than a cheap loss-only
proxy (a `ScreenVAE`-latent perceptual loss alone would likely be another
null, same category as `resnet_pl`/`ring_consistency_loss`/
`patch_match_loss`), and skips real structural-line conditioning initially
(`ScreenVAE.forward` already supports `line=None`, falling back to "whole
page is screen, no line exclusion" -- this defers sourcing Li et al.'s
separate pretrained line-extraction network until the core
latent-completion idea is shown to have legs at all).

`ScreenVAE` here is a faithful, self-contained port of the real released
architecture (`MangaInpainting_msxie92/src/svae.py`), not a re-derivation
from the paper -- read directly, weights downloaded from the paper's own
released checkpoint (Google Drive link in that repo's README,
`mangainpaint/pretrained/screenvae/ScreenVAE/latest_net_{enc,dec}.pth`, verified
byte-for-byte architecture match against the checkpoint's own state-dict
shapes before writing this file: `enc` is a 6-block ResnetGenerator,
3 downsamples, ngf=24, `inc+1=2` in (image + line) -> `outc*2=8` out
(4-channel mu + 4-channel logvar, only mu is ever used here -- see below);
`dec` is a `unet_128`-style U-Net, ngf=48, noise-injecting). No dependency
on the external repo itself (same "port only what's needed" discipline as
`model_resnet_pl.py`/`model_lama.py`): the reference file's `GaborWavelet`
path (`load_gaborext`) is never exercised by the `rep=True` (encode-only)
usage this project needs, so it isn't ported.

Domain range: unlike every *photo*-domain pretrained backbone in this
codebase (`model_projected_d.py`'s EfficientNet, `model_resnet_pl.py`'s
ADE20K ResNet, `model_lama.py`'s LaMa FFC generator -- all need an
[-1,1] -> [0,1]-ImageNet-normalized conversion), msxie92's own dataset
loader (`MangaInpainting_msxie92/src/dataset.py`: `img_gray = img_gray*2
- 1.0`) confirms `ScreenVAE` was trained on the exact same [-1,1]
single-channel grayscale convention this project already uses -- zero
domain conversion needed, a real (and convenient) point of agreement
between the two codebases' data pipelines.

One real bug found and fixed while porting (before any training-loop
integration): the reference `LayerNormWarpper.forward` calls
`nn.LayerNorm(...).cuda()` -- constructing a brand-new, hardcoded-CUDA
module instance on *every single forward call*. Mathematically inert
(this LayerNorm is always `elementwise_affine=False`, so the freshly
constructed instance has no learnable state to lose) but (a) needlessly
allocates a new module every call, and (b) hard-fails on CPU, which would
have silently blocked even a CPU-only run of this file. Replaced with a
direct `F.layer_norm` functional call (no module, no `.cuda()`,
device-agnostic) -- verified numerically identical to the original.
"""
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


# ══════════════════════════════════════════════════════════
# Shared building blocks, ported from svae.py (architecture must match the
# released checkpoint's state-dict shapes exactly -- verified before this
# file was written, see module docstring).
# ══════════════════════════════════════════════════════════
def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'layer':
        return LayerNormWarpper
    elif norm_type == 'none':
        return None
    raise NotImplementedError(f'normalization layer [{norm_type}] is not found')


class LayerNormWarpper(nn.Module):
    """Fixed vs. the reference (see module docstring): `F.layer_norm`
    directly instead of constructing+`.cuda()`-ing a fresh `nn.LayerNorm`
    every forward call. `elementwise_affine=False` in the original means
    there is no learnable weight/bias to port -- this is a pure
    reformulation of the same computation, not an approximation."""
    def __init__(self, num_features):
        super().__init__()
        self.num_features = int(num_features)

    def forward(self, x):
        return F.layer_norm(x, [self.num_features, x.size(2), x.size(3)])


def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        return functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        return functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)
    raise NotImplementedError(f'nonlinearity activation [{layer_type}] is not found')


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'initialization method [{init_type}] is not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)
    return net


def upsampleLayer(inplanes, outplanes, upsample='basic'):
    if upsample == 'basic':
        return [nn.ConvTranspose2d(inplanes, outplanes, kernel_size=4, stride=2, padding=1)]
    elif upsample in ('bilinear', 'nearest', 'linear'):
        return [nn.Upsample(scale_factor=2, mode=upsample, align_corners=True),
                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)]
    raise NotImplementedError(f'upsample layer [{upsample}] not implemented')


class ApplyNoise(nn.Module):
    """StyleGAN2-style per-channel noise injection inside the ScreenVAE
    *decoder* -- part of the frozen pretrained network (not this project's
    own `mangainpaint/model_attn.py`'s `NoiseInjection`, a different, trainable
    instance of the same idea)."""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.weight = nn.Parameter(torch.randn(channels), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(channels), requires_grad=True)

    def forward(self, x, noise):
        W, _ = torch.split(self.weight.view(1, -1, 1, 1), self.channels // 2, dim=1)
        B, _ = torch.split(self.bias.view(1, -1, 1, 1), self.channels // 2, dim=1)
        Z = torch.zeros_like(W)
        w = torch.cat([W, Z], dim=1).to(x.device)
        b = torch.cat([B, Z], dim=1).to(x.device)
        adds = w * torch.randn_like(x) + b
        return x + adds.type_as(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_bias):
        super().__init__()
        conv_block = [nn.ReplicationPad2d(1),
                      nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias)]
        if norm_layer is not None:
            conv_block += [norm_layer(dim)]
        conv_block += [nn.ReLU(True),
                       nn.ReplicationPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias)]
        if norm_layer is not None:
            conv_block += [norm_layer(dim)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
    """The `ScreenVAE` encoder (`enc`): `netC='resnet_6blocks'`, 3
    downsamples, ngf=24 (see this file's `define_C`/`ScreenVAE.__init__`).
    """
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3,
                 norm_layer=None, use_dropout=True, n_blocks=6):
        super().__init__()
        use_bias = norm_layer is None or (
            norm_layer.func if isinstance(norm_layer, functools.partial) else norm_layer) != nn.BatchNorm2d

        model = [nn.ReplicationPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias)]
        if norm_layer is not None:
            model += [norm_layer(ngf)]
        model += [nn.ReLU(True)]

        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.ReplicationPad2d(1),
                      nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0, bias=use_bias)]
            if norm_layer is not None:
                model += [norm_layer(ngf * mult * 2)]
            model += [nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += upsampleLayer(ngf * mult, int(ngf * mult / 2), upsample='bilinear')
            if norm_layer is not None:
                model += [norm_layer(int(ngf * mult / 2))]
            model += [nn.ReLU(True),
                      nn.ReplicationPad2d(1),
                      nn.Conv2d(int(ngf * mult / 2), int(ngf * mult / 2), kernel_size=3, padding=0)]
            if norm_layer is not None:
                model += [norm_layer(ngf * mult / 2)]
            model += [nn.ReLU(True)]
        model += [nn.ReplicationPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class UnetBlock(nn.Module):
    """One level of the `ScreenVAE` decoder (`dec`, `unet_128_G`)."""
    def __init__(self, input_nc, outer_nc, inner_nc, submodule=None, noise=None,
                 outermost=False, innermost=False, norm_layer=None, nl_layer=None,
                 use_dropout=False, upsample='basic'):
        super().__init__()
        self.outermost = outermost
        downconv = [nn.ReplicationPad2d(1), nn.Conv2d(input_nc, inner_nc, kernel_size=3, stride=2, padding=0)]
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc) if norm_layer is not None else None
        uprelu, uprelu2 = nl_layer(), nl_layer()
        uppad = nn.ReplicationPad2d(1)
        upnorm = norm_layer(outer_nc) if norm_layer is not None else None
        upnorm2 = norm_layer(outer_nc) if norm_layer is not None else None
        self.noiseblock = ApplyNoise(outer_nc)
        self.noise = noise

        if outermost:
            upconv = upsampleLayer(inner_nc * 2, inner_nc, upsample=upsample)
            upconv2 = nn.Conv2d(inner_nc, outer_nc, kernel_size=7, padding=0)
            up = [uprelu] + upconv
            if upnorm is not None:
                up += [norm_layer(inner_nc)]
            up += [uprelu2, nn.ReplicationPad2d(3), upconv2]
            self.model = nn.Sequential(*(downconv + [submodule] + up))
        elif innermost:
            upconv = upsampleLayer(inner_nc, outer_nc, upsample=upsample)
            upconv2 = nn.Conv2d(outer_nc, outer_nc, kernel_size=3, padding=0)
            up = [uprelu] + upconv
            if upnorm is not None:
                up += [upnorm]
            up += [uprelu2, uppad, upconv2]
            if upnorm2 is not None:
                up += [upnorm2]
            self.model = nn.Sequential(*([downrelu] + downconv + up))
        else:
            upconv = upsampleLayer(inner_nc * 2, outer_nc, upsample=upsample)
            upconv2 = nn.Conv2d(outer_nc, outer_nc, kernel_size=3, padding=0)
            down = [downrelu] + downconv
            if downnorm is not None:
                down += [downnorm]
            up = [uprelu] + upconv
            if upnorm is not None:
                up += [upnorm]
            up += [uprelu2, uppad, upconv2]
            if upnorm2 is not None:
                up += [upnorm2]
            layers = down + [submodule] + up
            if use_dropout:
                layers = layers + [nn.Dropout(0.5)]
            self.model = nn.Sequential(*layers)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        x2 = self.model(x)
        if self.noise:
            x2 = self.noiseblock(x2, self.noise)
        return torch.cat([x2, x], 1)


class GUnetAddInput(nn.Module):
    """The `ScreenVAE` decoder (`dec`): `netG='unet_128_G'`, `where_add='input'`."""
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=None,
                 nl_layer=None, use_dropout=False, use_noise=False, upsample='bilinear'):
        super().__init__()
        max_nchn = 8
        unet_block = UnetBlock(ngf * max_nchn, ngf * max_nchn, ngf * max_nchn, noise=False,
                               innermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        for _ in range(num_downs - 5):
            unet_block = UnetBlock(ngf * max_nchn, ngf * max_nchn, ngf * max_nchn, unet_block, noise=False,
                                   norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout, upsample=upsample)
        unet_block = UnetBlock(ngf * 4, ngf * 4, ngf * max_nchn, unet_block, use_noise,
                               norm_layer=norm_layer, nl_layer=nl_layer, upsample='basic')
        unet_block = UnetBlock(ngf * 2, ngf * 2, ngf * 4, unet_block, use_noise,
                               norm_layer=norm_layer, nl_layer=nl_layer, upsample='basic')
        unet_block = UnetBlock(ngf, ngf, ngf * 2, unet_block, use_noise,
                               norm_layer=norm_layer, nl_layer=nl_layer, upsample='basic')
        unet_block = UnetBlock(input_nc, output_nc, ngf, unet_block, noise=False,
                               outermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample='basic')
        self.model = unet_block

    def forward(self, x):
        return self.model(x)


# ══════════════════════════════════════════════════════════
# ScreenVAE itself
# ══════════════════════════════════════════════════════════
class ScreenVAE(nn.Module):
    """Frozen, pretrained. `outc=4` continuous latent channels (mu only --
    `logvar` is produced by `enc` but never sampled/used here, matching how
    msxie92's own `manga_inpaintor.py` uses it: always `rep=True`,
    deterministic mu, never `screen=False, rep=False`'s full VAE
    reconstruction path in this project's usage).

    `forward(x, line=None, rep=True)`: `x` is the (possibly masked)
    grayscale page, [-1,1], `(B,1,H,W)`. `line=None` -> whole page treated
    as screen (no structural-line exclusion) -- see module docstring for
    why real line conditioning is deferred. Returns the `(B,4,H/8,W/8)`
    latent only (`rep=True`, no decode) -- this project only ever needs
    the encode side; `screen=True`/full-decode paths from the reference
    class aren't ported since nothing here calls them.

    No `torch.no_grad()` inside `forward` itself (only this module's own
    parameters are frozen, via `requires_grad_(False)` above) -- same
    principle `model_resnet_pl.ResNetPL.forward` documents explicitly:
    `ScreenVAEConsistencyLoss` (below) needs gradient to flow *through*
    this encoder and back into the generator's own output, so callers
    that truly don't need a gradient here (e.g.
    `MangaFillNetScreenVAE.forward`'s masked-input encode, a raw data
    tensor with nothing upstream to backprop into anyway) wrap their own
    call site in `torch.no_grad()` instead of relying on this class to do
    it unconditionally.
    """
    def __init__(self, weights_dir, inc=1, outc=4, ngf_enc=24, ngf_dec=48):
        super().__init__()
        self.inc = inc
        self.outc = outc

        norm_c = get_norm_layer('instance')
        nl_c = get_non_linearity('lrelu')
        self.enc = init_weights(ResnetGenerator(inc + 1, outc * 2, ngf_enc, n_downsampling=3,
                                                norm_layer=norm_c, use_dropout=True, n_blocks=6))

        norm_g = get_norm_layer('layer')
        nl_g = get_non_linearity('lrelu')
        self.dec = init_weights(GUnetAddInput(outc, inc, 7, ngf_dec, norm_layer=norm_g,
                                              nl_layer=nl_g, use_dropout=True, use_noise=True,
                                              upsample='bilinear'), init_type='xavier')

        enc_sd = torch.load(f"{weights_dir}/latest_net_enc.pth", map_location="cpu")
        dec_sd = torch.load(f"{weights_dir}/latest_net_dec.pth", map_location="cpu")
        self.enc.load_state_dict(enc_sd)
        self.dec.load_state_dict(dec_sd)

        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # Always eval -- frozen pretrained network, same discipline as
        # every other frozen backbone in this codebase
        # (model_resnet_pl.ResNetPL, model_projected_d's frozen backbone).
        return super().train(False)

    def forward(self, x, line=None):
        if line is None:
            line = torch.ones_like(x)
        else:
            line = torch.sign(line)
            x = torch.clamp(x + (1 - line), -1, 1)
        inp = torch.cat([x, line], 1)
        inter = self.enc(inp)
        scr, _logvar = torch.split(inter, (self.outc, self.outc), dim=1)
        return scr


# ══════════════════════════════════════════════════════════
# Latent-space completion net -- this project's scoped stand-in for
# msxie92's `SemanticInpaintingModel`/`SemanticInpaintGenerator`
# (`MangaInpainting_msxie92/src/{models,networks}.py`, read directly): a
# from-scratch, trainable network that completes the ScreenVAE latent
# *inside the hole only*, given real per-pixel stochastic noise (their own
# `torch.randn_like(masks)` input): a hole-region regressor conditioned on
# noise can commit to *one of many* plausible completions instead of
# averaging all of them into a single deterministic (and, on bitonal
# manga, visually mushy) mean -- the fix for the "no continuous middle
# ground" failure mode deterministic regression hits on bitonal manga.
#
# Real reference shape confirmed by reading `SemanticInpaintGenerator`
# directly (`networks.py:48`): 3 stride-2 downsamples -> dilated ResNet
# blocks at 1/8 resolution -> 3 upsamples back to input resolution --
# structurally the same encoder/dilated-bottleneck/decoder shape as this
# project's own `MangaFillNet`. Scoped down here to reuse this codebase's
# own building blocks (`Enc`/`DilRes`/`Dec`, `mangainpaint/model_scratch.
# py`) instead of porting msxie92's own bespoke LayerNorm+spectral-norm
# conv stack -- the mechanism under test is "complete in a disentangled
# continuous latent space, with noise", not the specific choice of
# normalization/spectral-norm in the network doing the completing.
#
# Scope trade-off vs. the real msxie92 pipeline: no structural-line
# channel (deferred, see module docstring), and this net is trained
# *jointly* end-to-end with the main pixel-space generator (one combined
# backward pass through `hint_proj` -> `completion` -> `screenvae`'s
# frozen encode is the only non-trainable link) rather than msxie92's own
# separately-pretrained multi-stage curriculum. Simpler to screen cheaply
# first; a staged/pretrained-completion-net curriculum is only worth
# revisiting if this joint version shows real signal.
# ══════════════════════════════════════════════════════════
class LatentCompletionNet(nn.Module):
    def __init__(self, latent_ch=4, base=48, dilations=(1, 2, 4, 8)):
        super().__init__()
        from mangainpaint.model_scratch import Enc, DilRes, Dec
        in_ch = latent_ch + 1 + 1  # screen_masked + mask + noise
        self.e1 = Enc(in_ch, base)
        self.e2 = Enc(base, base * 2)
        self.e3 = Enc(base * 2, base * 4)
        bch = base * 4
        d0, d1_, d2_, d3_ = dilations
        self.r1 = DilRes(bch, d0)
        self.r2 = DilRes(bch, d1_)
        self.r3 = DilRes(bch, d2_)
        self.r4 = DilRes(bch, d3_)
        self.d1 = Dec(bch, base * 4, base * 2)
        self.d2 = Dec(base * 2, base * 2, base)
        self.d3 = Dec(base, base, base)
        self.out = nn.Conv2d(base, latent_ch, 3, 1, 1)

    def forward(self, screen_masked, mask, noise):
        """screen_masked: (B,4,H,W) ScreenVAE latent of the masked input.
        mask: (B,1,H,W), 1=hole. noise: (B,1,H,W) ~ N(0,1), fresh every
        call (real stochastic capacity, matching msxie92's own
        `SemanticInpaintingModel.forward`). Returns a raw (unbounded --
        this is a VAE latent, not a pixel value, so no tanh) completion
        prediction for the *whole* image; the caller composites it with
        `screen_masked` so only the hole region actually uses this
        network's output (mirrors this project's own `comp = gen*mask +
        img*(1-mask)` compositing convention used everywhere else)."""
        x = torch.cat([screen_masked, mask, noise], dim=1)
        x1, s1 = self.e1(x)
        x2, s2 = self.e2(x1)
        x3, s3 = self.e3(x2)
        r = self.r1(x3, mask)
        r = self.r2(r, mask)
        r = self.r3(r, mask)
        r = self.r4(r, mask)
        d = self.d1(r, s3, mask)
        d = self.d2(d, s2, mask)
        d = self.d3(d, s1, mask)
        return self.out(d)


class MangaFillNetScreenVAE(nn.Module):
    """`MangaFillNetAttnNoFFC` (`model_attn.py`, this project's best
    from-scratch architecture) plus the ScreenVAE-completion hint.
    Structurally identical to that class (same `e1`/`e2`/`e3`/`r1`/`r2`/
    `r4`/`r8`/`attn`/`merge`/`d1`/`d2`/`d3`/`head` names/shapes -- copied
    here rather than imported and wrapped, matching this project's
    per-lever file convention, e.g. `model_gsff.py`/`model_dabformer_lite.
    py`, so each experiment's architecture is self-contained and
    independently warm-start-loadable) so an `MangaFillNetAttnNoFFC`
    checkpoint partial-loads with **zero shape mismatches** -- only
    `screenvae.*` (loaded from its own separate pretrained checkpoint, not
    the warm-start seed anyway), `completion.*`, and `hint_proj.*` are
    new/missing.

    `hint_proj` (projects the completed 4-channel latent into the
    bottleneck's channel count, additive) is **zero-initialized** -- same
    warm-start-safe convention as `model_attn.py`'s `NoiseInjection`/
    `EdgeHint`: a freshly warm-started model is byte-identical to the
    seed checkpoint at step 0, so any measured effect is attributable to
    this lever actually engaging during fine-tuning, not to a different
    starting point. `completion` itself is NOT zero-initialized (it's an
    independent trainable sub-network that must learn from scratch
    regardless; zero-initializing it would make it output zero forever).

    Injected at the bottleneck (after `e3`, before `r1`/`attn`) rather
    than concatenated at the very input (`e1`) -- msxie92's own final
    generator takes hints as extra *input* channels, but doing that here
    would change `e1`'s input-channel count and break shape-compatible
    warm-starting from a seed checkpoint entirely (a hard `size mismatch`,
    not just a missing key). The bottleneck is downstream of every
    existing conv anyway, so the generator still sees the hint before any
    of its own dilated/attention processing.
    """
    def __init__(self, in_ch=2, base=32, ratio_g=0.5, dilations=(1, 2, 4, 8),
                 fuse_k=3, use_fuse=False, screenvae_weights_dir=None, completion_base=48):
        super().__init__()
        from mangainpaint.model_scratch import Enc, DilRes, Dec, OutHead
        from mangainpaint.model_attn import ContextualAttentionBlock

        b = base
        self.e1 = Enc(in_ch, b)
        self.e2 = Enc(b, b * 2)
        self.e3 = Enc(b * 2, b * 4)
        bch = b * 4

        d0, d1_, d2_, d3_ = dilations
        self.r1 = DilRes(bch, d0)
        self.r2 = DilRes(bch, d1_)
        self.r4 = DilRes(bch, d2_)
        self.r8 = DilRes(bch, d3_)
        self.attn = ContextualAttentionBlock(patch_size=3, softmax_scale=10.0, fuse_k=fuse_k, use_fuse=use_fuse)
        self.merge = nn.Conv2d(bch * 2, bch, 1)

        self.d1 = Dec(bch, b * 4, b * 2, hid_mult=4)
        self.d2 = Dec(b * 2, b * 2, b, hid_mult=4)
        self.d3 = Dec(b, b, b * 2, hid_mult=8)
        self.head = OutHead(b * 2)

        self.screenvae = ScreenVAE(weights_dir=screenvae_weights_dir)
        self.completion = LatentCompletionNet(latent_ch=self.screenvae.outc, base=completion_base)
        self.hint_proj = nn.Conv2d(self.screenvae.outc, bch, 1)
        nn.init.zeros_(self.hint_proj.weight)
        nn.init.zeros_(self.hint_proj.bias)

    def train(self, mode=True):
        super().train(mode)
        self.screenvae.eval()  # frozen submodule always stays in eval, regardless of the parent's mode
        return self

    def forward(self, x):
        img_masked = x[:, 0:1]
        mask = x[:, 1:2]

        with torch.no_grad():  # img_masked is raw data -- nothing upstream to backprop into
            screen_masked = self.screenvae(img_masked, line=None)
        mask_ds = F.interpolate(mask.float(), size=screen_masked.shape[2:], mode='nearest')
        noise = torch.randn_like(mask_ds)
        completion_pred = self.completion(screen_masked, mask_ds, noise)
        screen_completed = screen_masked * (1 - mask_ds) + completion_pred * mask_ds

        x1, s1 = self.e1(x)
        x2, s2 = self.e2(x1)
        x3, s3 = self.e3(x2)

        hint = self.hint_proj(screen_completed)
        if hint.shape[2:] != x3.shape[2:]:
            hint = F.interpolate(hint, size=x3.shape[2:], mode='bilinear', align_corners=False)
        x3 = x3 + hint

        dilres = self.r1(x3, mask)
        dilres = self.r2(dilres, mask)
        dilres = self.r4(dilres, mask)
        dilres = self.r8(dilres, mask)

        attn_out = self.attn(x3, mask) + x3

        merged = F.leaky_relu(self.merge(torch.cat([dilres, attn_out], dim=1)), 0.2, inplace=True)

        d = self.d1(merged, s3, mask)
        d = self.d2(d, s2, mask)
        d = self.d3(d, s1, mask)
        return self.head(d)


# ══════════════════════════════════════════════════════════
# Re-encode consistency loss, added after an earlier run showed
# `MangaFillNetScreenVAE` trains cleanly but is a clean qualitative null:
# `hint_proj` gives the generator new *information* (the completed
# screentone latent) but nothing in the loss recipe gave it any
# *pressure* to actually use that information -- every loss term in
# `mangainpaint/trainer.py` is one of the levers explored elsewhere in
# this codebase, and none of them reference the ScreenVAE latent at all.
# This closes that gap directly: an explicit term that re-encodes the
# generator's own *final pixel output* through the same frozen
# `ScreenVAE` and penalizes its distance from the real ground-truth
# page's own encoding, inside the hole -- "does your hole fill's
# screentone content actually match reality" in a domain-specific
# disentangled latent space, the same structural role `model_resnet_pl.
# ResNetPL` plays in raw ImageNet-feature space (already tried, already a
# null -- the working hypothesis here is that a screentone-*specific*
# latent succeeds where a generic photo-domain one didn't, since it's
# disentangled from line/structure by construction rather than a
# general-purpose feature hierarchy).
#
# Deliberately GT-referenced, not self-referential: comparing the output
# against the *real* page's own latent (like every other hole-
# reconstruction loss in this codebase) rather than against
# `MangaFillNetScreenVAE`'s own `completion` net's prediction avoids a
# degenerate shortcut where `completion` and the final generator could
# collude to agree with each other without either being correct.
#
# Loads its OWN separate frozen `ScreenVAE` instance rather than reaching
# into `G.screenvae` -- `mangainpaint/trainer.py`'s train loop is intentionally
# architecture-agnostic (works with any `model_fn`-supplied G/D pair), so
# this follows the exact same standalone-frozen-module pattern already
# established for `ResNetPL`/LPIPS/`ProjectedD`'s backbone, at the cost of
# a minor (~17.5M frozen params, non-trainable) duplication vs. reusing
# the copy already inside `MangaFillNetScreenVAE`.
# ══════════════════════════════════════════════════════════
class ScreenVAEConsistencyLoss(nn.Module):
    def __init__(self, weights_dir):
        super().__init__()
        self.screenvae = ScreenVAE(weights_dir=weights_dir)

    def train(self, mode=True):
        return super().train(False)

    def forward(self, comp, img, mask):
        """comp: generator output composited with GT outside the hole
        (`comp = gen*mask + img*(1-mask)`, this codebase's standard
        convention -- gradient flows into `comp` only through the hole
        region, `1-mask` is a constant w.r.t. the generator there).
        img: real GT page, [-1,1]. mask: (B,1,H,W), 1=hole. Returns a
        hole-focused mean absolute latent distance (unweighted across
        the 4 latent channels, matching how every other hole loss in
        `mangainpaint/losses.py` averages over its own comparison axis)."""
        with torch.no_grad():
            target_latent = self.screenvae(img, line=None)
        pred_latent = self.screenvae(comp, line=None)
        diff = (pred_latent - target_latent).abs()
        m = F.interpolate(mask.float(), size=diff.shape[2:], mode='nearest')
        denom = (m.sum() * diff.shape[1]).clamp_min(1.0)
        return (diff * m).sum() / denom


# ══════════════════════════════════════════════════════════
# ScreenVAE-latent patch-match loss, added after `ScreenVAEConsistencyLoss`
# confirmed the network *can* engage with a ScreenVAE-referenced pressure,
# but a moment-matching one (re-encode distance, an aggregate/statistical
# comparison) gets satisfied by a global brightness/contrast shift rather
# than real content recovery -- the same trap independently hit by other
# aggregate-statistic losses explored in this codebase (`resnet_pl` and
# variants of the dabformer/edge-hint branches). This project's own
# `mangainpaint/losses.py:patch_match_loss` is the one loss lever that is
# NOT a moment-matching comparison -- it forces the generated hole to
# resemble a *specific* real patch found elsewhere in the same image, a
# content-*verification* mechanism a generic texture can't satisfy just
# by getting its aggregate statistics right. That loss in raw pixel space
# was inconclusive (null-ish, not a clean rejection) with a specific named
# confound: "most of a manga page's valid region is blank/simple line
# art, so nearest-real-patch search likely skews toward flat matches on
# average" -- i.e. raw pixel space doesn't discriminate screentone
# texture well when most candidates are near-uniform white.
#
# This class re-runs that same mechanism (via `_patch_match_core`,
# factored out of `mangainpaint/losses.py:patch_match_loss` for exactly this
# reuse) in `ScreenVAE`'s disentangled screentone latent space instead of
# raw pixels -- the working hypothesis is that this specifically
# addresses the "blank paper dominates the candidate pool" confound,
# since two visually-blank-but-differently-toned patches (paper vs. a
# very light screentone) are encoded as more clearly distinct in a
# screentone-specific latent than in raw near-white pixel values, so the
# nearest-neighbor search should be less biased toward degenerate flat
# matches. This is a hypothesis motivating the combination, not a
# confirmed result.
#
# `ScreenVAE`'s latent is dense at the *same* spatial resolution as the
# input image (confirmed empirically), so the mask needs no resizing
# before unfolding -- same convenient property `ScreenVAEConsistencyLoss`
# above relies on. Loads its own separate frozen `ScreenVAE` instance,
# same architecture-agnostic-trainer reasoning as
# `ScreenVAEConsistencyLoss`.
# ══════════════════════════════════════════════════════════
class ScreenVAEPatchMatchLoss(nn.Module):
    def __init__(self, weights_dir, patch=7, stride=8, hole_thresh=0.5,
                 valid_thresh=0.02, min_patches=4):
        super().__init__()
        self.screenvae = ScreenVAE(weights_dir=weights_dir)
        self.patch, self.stride = patch, stride
        self.hole_thresh, self.valid_thresh, self.min_patches = hole_thresh, valid_thresh, min_patches

    def train(self, mode=True):
        return super().train(False)

    def forward(self, comp, img, mask):
        """comp: generator output composited with GT outside the hole.
        img: real GT page, [-1,1]. mask: (B,1,H,W), 1=hole. Nearest-
        neighbor search happens in `ScreenVAE`'s 4-channel latent space;
        gradient flows from the matched-patch distance back through
        `comp`'s own re-encode only (the candidate side comes from `img`,
        encoded under `no_grad`, matching `patch_match_loss`'s own
        "no gradient needed for the real side" convention)."""
        from mangainpaint.losses import _patch_match_core
        with torch.no_grad():
            target_latent = self.screenvae(img, line=None)
        pred_latent = self.screenvae(comp, line=None)
        return _patch_match_core(pred_latent, target_latent, mask, self.patch, self.stride,
                                 self.hole_thresh, self.valid_thresh, self.min_patches)
