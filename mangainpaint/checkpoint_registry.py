"""
Shared checkpoint -> generator dispatch, used by every eval-time script
in this project. A single dispatch table so architecture support (e.g.
UFFC vs. the LaMa-slim family) can't drift between eval scripts the way
it would if each one built its own copy. Covers exactly the architectures
behind Table 1's rows plus the two distillation-signal ablations
(README's recipe/run-id table) -- nothing beyond the reported roster.
"""
import os
import sys

import torch
import torch.nn as nn

from mangainpaint.model_scratch import MangaFillNet
from mangainpaint.model_attn import MangaFillNetAttnNoFFC
from mangainpaint.model_uffc import MangaFillNetUFFC
from mangainpaint.model_lama import LamaTransferG

HERE = os.path.dirname(os.path.abspath(__file__))
SCREENVAE_WEIGHTS = os.path.join(HERE, "pretrained", "screenvae", "ScreenVAE")
# `model_pconv.py` defines the standalone PConvUNet architecture (Liu et
# al. 2018, "Image Inpainting for Irregular Holes Using Partial
# Convolutions") that `recipes/pconv_unet.py`'s training loop also imports
# directly via a bare `from model_pconv import ...`; both expect it to
# live alongside the recipes rather than inside this package, since
# PConv's dataset/training loop predates and stays independent of this
# registry (see `PConvWrapper` below).
PCONV_ROOT = os.path.join(HERE, "..", "recipes")

ARCH_NAMES = ("vanilla", "attn_noffc", "uffc", "lama", "pconv",
              # Slim distillation students (S1 family): S1, S1-attn's own
              # variant below, S2, S3, C1 (narrower ngf), and the S2-GN/
              # S2-VAE distillation-signal ablations all share this one
              # architecture class -- they differ only in training signal,
              # not in generator shape.
              "lama_slim", "lama_slim_attn",
              # C2: narrow FFC + one bottleneck linear-attention pass,
              # distilled. See model_lama_slim_fus.py.
              "lama_slim_fus")


class PConvWrapper(nn.Module):
    """Bridges `model_pconv.PConvUNet`'s real interface
    (`forward(masked_3ch01, valid_mask)`, 3-channel [0,1], PConv's own
    "1=valid" mask convention) to this project's shared `forward(x)`
    contract (x = [B,2,H,W], 1ch masked-grayscale in [-1,1] with the hole
    already white-filled + 1ch mask, our "1=hole" convention) -- so PConv
    can reuse `mangainpaint.trainer.evaluate`/`make_loaders` and every
    eval script exactly like every other real checkpoint, instead of
    needing its own separate eval path (`recipes/pconv_unet.py` has its
    own self-contained training loop, since PConv's dataset/training loop
    predates this registry).

    Range/fill-convention note: `x[:,0:1]` is already hole-filled with
    1.0 ("white") in [-1,1] space by every dataset path in this project;
    converting to [0,1] via `(x+1)/2` lands the hole at exactly 1.0,
    matching `Manga109PConvDataset`'s own `fill_val=1.0` ("white")
    convention in [0,1] space -- no separate re-fill needed.
    """
    def __init__(self, pconv_net):
        super().__init__()
        self.pconv_net = pconv_net

    def forward(self, x):
        masked01 = (x[:, 0:1] + 1.0) / 2.0
        masked_3ch = masked01.repeat(1, 3, 1, 1)
        valid_mask = 1.0 - x[:, 1:2]
        out_3ch = self.pconv_net(masked_3ch, valid_mask).clamp(0, 1)
        # Same luminance formula as recipes/pconv_unet.py's own
        # `rgb_to_gray_minus1plus1` (not imported from there to avoid
        # pulling in that script's module-level DDP/training-only setup).
        gray01 = 0.299 * out_3ch[:, 0:1] + 0.587 * out_3ch[:, 1:2] + 0.114 * out_3ch[:, 2:3]
        return (gray01.clamp(0, 1) * 2 - 1).clamp(-1, 1)


def build_generator(arch, cfg, device):
    if arch == "vanilla":
        # `dilations` forwarded: some checkpoints use an HDC-safe (1,2,5,9)
        # schedule instead of the (1,2,4,8) default, and silently rebuilding
        # with the wrong one would load fine (same shapes -- dilation isn't
        # a weight) while evaluating a DIFFERENT network than was trained.
        G = MangaFillNet(in_ch=2, base=cfg.get("base", 32), ratio_g=cfg.get("ratio_g", 0.5),
                         dilations=cfg.get("dilations", (1, 2, 4, 8)))
    elif arch == "attn_noffc":
        G = MangaFillNetAttnNoFFC(in_ch=2, base=cfg.get("base", 32), ratio_g=cfg.get("ratio_g", 0.5),
                                  dilations=cfg.get("dilations", (1, 2, 4, 8)),
                                  fuse_k=cfg.get("fuse_k", 3), use_fuse=False)
    elif arch == "uffc":
        G = MangaFillNetUFFC(in_ch=2, base=cfg.get("base", 32), ratio_g=cfg.get("ratio_g", 0.5),
                             image_size=cfg.get("image_size", 384))
    elif arch == "lama":
        # freeze_up_to defaults to 0 -- matches what the teacher fine-tune
        # recipe actually passed (never went through `cfg`); doesn't affect
        # eval-mode correctness regardless since load_state_dict below
        # overwrites every weight with the real trained checkpoint anyway.
        G = LamaTransferG(freeze_up_to=cfg.get("freeze_up_to", 0) or 0)
    elif arch == "pconv":
        if PCONV_ROOT not in sys.path:
            sys.path.insert(0, PCONV_ROOT)
        from model_pconv import PConvUNet
        G = PConvWrapper(PConvUNet(in_ch=3, out_ch=3))
    elif arch == "lama_slim":
        from mangainpaint.model_lama_slim import LamaSlimG
        # init_mode doesn't matter for eval -- load_state_dict below
        # overwrites every weight with the real trained checkpoint anyway,
        # same reasoning as LamaTransferG's freeze_up_to note above.
        G = LamaSlimG(ngf=cfg.get("ngf", 32), n_blocks=cfg.get("n_blocks", 12),
                     init_mode="random", expose_bottleneck=False, use_screenvae_hint=False)
    elif arch == "lama_slim_fus":
        from mangainpaint.model_lama_slim_fus import LamaSlimFusG
        # ngf/n_blocks read from the saved cfg (the compact runs put them
        # there, unlike S1/S2/S3 which happened to match the ngf=32 default).
        G = LamaSlimFusG(ngf=cfg.get("ngf", 20), n_blocks=cfg.get("n_blocks", 12),
                        linattn_heads=cfg.get("linattn_heads", 4),
                        init_mode="random", expose_bottleneck=False,
                        use_screenvae_hint=False)
    elif arch == "lama_slim_attn":
        from mangainpaint.model_lama_slim_attn import LamaSlimAttnG
        G = LamaSlimAttnG(expose_bottleneck=False)
    else:
        raise ValueError(f"unknown arch {arch!r}, expected one of {ARCH_NAMES}")
    return G.to(device)


def load_generator_state_dict(G, arch, state_dict, strict=False):
    """Load a checkpoint's flat `G` state dict into a built generator.
    PConvWrapper adds a `.pconv_net` prefix that the checkpoint's state
    dict (saved from the raw PConvUNet, pre-dating this registry) doesn't
    have -- target the inner module for that one arch. Every other arch's
    checkpoint matches its own top-level module directly."""
    target = G.pconv_net if arch == "pconv" else G
    missing, unexpected = target.load_state_dict(state_dict, strict=strict)
    if strict and (missing or unexpected):
        raise RuntimeError(f"{len(missing)} missing, {len(unexpected)} unexpected keys")
    return missing, unexpected


def load_checkpoint_generator(ckpt_path, arch, device, strict=False):
    """Load a real checkpoint's generator. Returns (G, ckpt) -- callers
    that need `cfg`/`epoch`/`score` can read them off `ckpt` directly."""
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("cfg", {})
    G = build_generator(arch, cfg, device)
    load_generator_state_dict(G, arch, ckpt["G"], strict=strict)
    G.eval()
    return G, ckpt
