"""
Axis A2 generator: LaMa-transfer (Sauer-style GAN, but here just the
generator side) — loads the real `advimman/lama` `FFCResNetGenerator` class,
initialized from the pretrained `big-lama` checkpoint, with a configurable
frozen prefix. Confirmed to load with 0 missing/unexpected keys and to fit
comfortably in VRAM with a partial freeze.

Drop-in compatible with mangainpaint/model_scratch.MangaFillNet's interface
(`forward(x)` where x is the dataset's `model_input`: [B,2,H,W] = 1ch masked
grayscale + 1ch mask, output [B,1,H,W] grayscale in [-1,1]) so
mangainpaint/trainer.py needs zero changes to use this generator instead.
"""
import os
import sys

import torch
import torch.nn as nn
from omegaconf import OmegaConf

_LAMA_ROOT = os.path.join(os.path.dirname(__file__), "..", "external", "lama")
if _LAMA_ROOT not in sys.path:
    sys.path.insert(0, _LAMA_ROOT)

from saicinpainting.training.modules.ffc import FFCResNetGenerator  # noqa: E402

_CKPT_DIR = os.path.join(_LAMA_ROOT, "big-lama")


class LamaTransferG(nn.Module):
    """Real FFCResNetGenerator, pretrained-initialized, partially frozen.

    `freeze_up_to=16` freezes `self.net.model[0:16]` (reflection pad, init
    FFC block, 3 downsample blocks, first 11 of 18 FFC-ResNet blocks),
    leaving the last 7 resnet blocks + upsample + output head trainable
    (~39.6% of params) -- a defensible first-run default, not re-tuned.
    Full fine-tuning (`freeze_up_to=0`) is what the paper actually reports
    (see §6.2); this partial-freeze default is kept as the class default
    for backward compatibility with earlier recipes.
    """

    def __init__(self, freeze_up_to=16):
        super().__init__()
        self.freeze_up_to = freeze_up_to

        cfg = OmegaConf.load(os.path.join(_CKPT_DIR, "config.yaml"))
        gcfg = OmegaConf.to_container(cfg.generator, resolve=True)
        gcfg.pop("kind")
        self.net = FFCResNetGenerator(**gcfg)

        ckpt = torch.load(os.path.join(_CKPT_DIR, "models", "best.ckpt"),
                          map_location="cpu", weights_only=False)
        sd = {k[len("generator."):]: v for k, v in ckpt["state_dict"].items()
              if k.startswith("generator.")}
        missing, unexpected = self.net.load_state_dict(sd, strict=False)
        assert not missing and not unexpected, (
            f"big-lama checkpoint mismatch: missing={missing} unexpected={unexpected}")

        for i, m in enumerate(self.net.model):
            m.requires_grad_(i >= freeze_up_to)

    def train(self, mode=True):
        # Frozen submodules must keep using their pretrained BatchNorm
        # running stats, not live batch stats, regardless of the owning
        # module's train()/eval() calls -- otherwise their calibration
        # silently drifts over the run even though requires_grad=False
        # already stops weight updates. Same principle as
        # model_projected_d.PretrainedBackbone.train(), adapted for a
        # partial (not whole-model) freeze.
        #
        # The trainable tail's own BatchNorm layers deliberately stay in
        # train() (live batch stats), not eval(): tried forcing them to
        # eval() too (reusing their pretrained running stats) and it
        # produced NaN activations from the very first forward pass --
        # manga (bitonal ink/paper, duplicated to a degenerate R=G=B input)
        # is far enough outside LaMa's Places365-photo training
        # distribution that some pretrained running_var entries are too
        # small relative to our actual activations, blowing up under fp16.
        # Live batch stats renormalize to the real activation distribution
        # each step and don't have this failure mode.
        super().train(mode)
        if mode:
            for i, m in enumerate(self.net.model):
                if i < self.freeze_up_to:
                    m.eval()
        return self

    def forward(self, x):
        masked, mask = x[:, 0:1], x[:, 1:2]
        # LaMa's pretrained weights (incl. frozen BatchNorm running stats)
        # were calibrated on [0, 1]-range inputs (see
        # external/lama/saicinpainting/evaluation/data.py:load_image) --
        # mangainpaint/dataset.py normalizes to [-1, 1] instead, so rescale here
        # rather than feeding an out-of-distribution range through the
        # frozen prefix (verified: feeding [-1, 1] directly produced NaN
        # activations within the first forward pass on real data).
        img01 = (masked + 1) / 2
        img3 = img01.repeat(1, 3, 1, 1)
        inp4 = torch.cat([img3, mask], dim=1)
        # Force fp32 for the *entire* net, not just the raw FFT calls
        # ffc.py's FourierUnit already wraps in fp32 (avoids the cuFFT
        # crash) -- but at real 384px input, the 18 stacked FFC-ResNet
        # blocks all operate at the same 48x48 bottleneck, and images with
        # large constant-filled masked regions (our own masking scheme)
        # produce very large-magnitude frequency-domain values there. Once
        # cast back to fp16 and fed into a later *plain* Conv2d
        # (confirmed via layer-by-layer tracing: the break was in a
        # FFCResnetBlock's convg2l, not in the FFT itself), that overflows
        # to inf/nan under fp16 autocast. Confirmed via reproduction on the
        # exact failing real batch: fp16 -> NaN, fp32 -> clean. Running the
        # whole generator in fp32 costs some speed/memory vs fp16, but it
        # fits comfortably (~3GB @ batch=4) on a small GPU, and this is the
        # only fix confirmed to eliminate the failure rather than just move
        # it elsewhere.
        with torch.amp.autocast(x.device.type, enabled=False):
            out3 = self.net(inp4.float())  # sigmoid, [0, 1]
        out3 = out3.to(x.dtype)
        gray = out3.mean(dim=1, keepdim=True)
        return gray * 2 - 1  # back to mangainpaint/dataset.py's [-1, 1] convention
