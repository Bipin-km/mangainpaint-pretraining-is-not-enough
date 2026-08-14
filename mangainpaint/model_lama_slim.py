"""
Slim LaMa student (Axis A7) -- the on-thesis generator: a LaMa-shaped
FFC-ResNet at a size we can actually ship (<= 10M params), instead of
big-lama's 51.0M.

Motivation: `LamaTransferG` (51.0M) decisively wins the real brush task
(see paper's results table), but a 51M generator is off-thesis -- the
project's goal is a *small, efficient* grayscale-manga inpainter. So LaMa
becomes the **teacher**, not the product. This file is the student
backbone.

Sizing sweep (real param counts, `FFCResNetGenerator`):

    ngf=64 n_blocks=18 -> 51.0M   (big-lama, off-thesis)
    ngf=48 n_blocks=6  -> 10.7M
    ngf=32 n_blocks=12 ->  8.8M   <- DEFAULT
    ngf=32 n_blocks=9  ->  6.8M
    ngf=24 n_blocks=18 ->  7.2M
    ngf=16 n_blocks=18 ->  3.2M

Default ngf=32/n_blocks=12 (8.8M) -- the most capacity inside the 10M
budget while keeping a deep-enough FFC-ResNet stack. For reference the
from-scratch pack (`MangaFillNet`, `MangaFillNetAttnNoFFC`) is 2.7-2.8M.

Why keep FFC at all: screentone is *periodic*, and a spectral global-mixing
op is the natural basis for periodic texture -- consistent with this
project's own finding that removing FFC entirely
(`MangaFillNetLinAttnNoFFC`) LOST real texture-generating capacity. This is
the one architectural choice here with a domain argument behind it, not
LaMa cargo-culting.

Three things this class supports, one per planned run (S1/S2/S3):

- `init_mode` -- **"random" is the default and what every run uses. The
  "sliced" path is a MEASURED NEGATIVE RESULT, kept only so the finding is
  reproducible.** Channel-slicing a trained FFC-ResNet down to a narrower
  one (structured first-k slice + evenly-spaced resblock selection) carries
  essentially no function across. Step-0 hole-L1 on 32 val pages, slim
  student (8.78M):

      random init                              0.8461
      sliced <- raw big-lama (photo)           0.9412   (worse than random)
      sliced <- fine-tuned LaMa v4 (manga)     0.8179   (best, but barely)
      ---
      the 51M teacher itself                   0.1076   (for scale)

  Even the best sliced student is ~8x worse than the teacher it was sliced
  from and only marginally better than random -- first-k slicing has no
  importance ranking, so it destroys the learned channel structure. Weight
  surgery is therefore NOT a viable transfer channel here.

  **Consequence: distillation is the only real transfer channel**, which is
  also what makes the S1->S2 ablation clean (random init on both sides, so
  the teacher is the single variable).
- `expose_bottleneck=True` -- stashes the post-resblock bottleneck tensor
  on `self.last_bottleneck` so mangainpaint/distill.py can feature-match it
  against the teacher's. Also builds `distill_adapter`, a 1x1 conv lifting
  the student's bottleneck width (256) to the teacher's (512).
- `use_screenvae_hint=True` -- the frozen msxie92 ScreenVAE ->
  LatentCompletionNet -> zero-init `hint_proj` side path from
  mangainpaint/model_screenvae.py, injected additively at the bottleneck. Same
  zero-init warm-start-safe convention as `MangaFillNetScreenVAE`
  (byte-identical to a no-hint model at step 0).

  **⚠ OFF-THESIS, and NOT used by any of the S1/S2/S3 runs.** ScreenVAE is
  17.54M params and LatentCompletionNet(base=48) another 5.44M, and BOTH
  must run at inference to produce the hint -- so a hint-equipped student
  ships 8.8 + 5.4 + 17.5 = 31.8M, i.e. it busts the 10M budget harder than
  big-lama's 51M did in relative terms. Kept here only as a possible
  upper-bound/oracle cell. The way ScreenVAE actually enters these runs is
  as a **training-only loss** (`ScreenVAEConsistencyLoss`, S3), which costs
  nothing at inference.

**The budget rule this file exists to enforce**: everything external --
LaMa's shape priors, ScreenVAE's screentone manifold -- enters at TRAINING
time (teacher KD, auxiliary loss) and is thrown away afterwards. What ships
is one 8.8M FFC-ResNet. `distill_adapter` (+0.13M) is training-only too.

Interface is drop-in identical to `LamaTransferG` / `MangaFillNet`:
`forward(x)` with x = [B,2,H,W] (masked grayscale + mask), out [B,1,H,W]
in [-1,1]. mangainpaint/trainer.py needs no changes to use it.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

_LAMA_ROOT = os.path.join(os.path.dirname(__file__), "..", "external", "lama")
if _LAMA_ROOT not in sys.path:
    sys.path.insert(0, _LAMA_ROOT)

from saicinpainting.training.modules.ffc import FFCResNetGenerator  # noqa: E402

_CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "external", "lama", "big-lama")

# big-lama's own topology, needed to locate the teacher's resblocks when
# slicing and to size the distill adapter.
TEACHER_NGF = 64
TEACHER_NBLOCKS = 18


# big-lama's generator config, inlined. Transcribed verbatim from
# external/lama/big-lama/config.yaml (`generator:` block, with the
# interpolations resolved) so the slim student needs only the LaMa *source*
# (`saicinpainting`, a git clone) and NOT the 392MB pretrained checkpoint --
# which, with init_mode="random", nothing here reads. Keeping these values
# identical to the pretrained model's is what makes the student a true
# width/depth scaling of LaMa rather than a different architecture.
BIG_LAMA_GEN_CFG = {
    "input_nc": 4,
    "output_nc": 3,
    "ngf": 64,
    "n_downsampling": 3,
    "n_blocks": 18,
    "add_out_act": "sigmoid",
    "init_conv_kwargs":       {"ratio_gin": 0,    "ratio_gout": 0,    "enable_lfu": False},
    "downsample_conv_kwargs": {"ratio_gin": 0,    "ratio_gout": 0,    "enable_lfu": False},
    "resnet_conv_kwargs":     {"ratio_gin": 0.75, "ratio_gout": 0.75, "enable_lfu": False},
}


def _gen_cfg(ngf, n_blocks, ckpt_dir=None):
    """big-lama's generator config with ngf/n_blocks overridden.

    Reads the real config.yaml if the big-lama checkpoint dir happens to be
    present (keeps this honest if the upstream config ever changes), and
    falls back to the inlined BIG_LAMA_GEN_CFG otherwise -- so a random-init
    run works with no pretrained checkpoint on disk at all."""
    cfg_path = os.path.join(ckpt_dir or _CKPT_DIR, "config.yaml")
    if os.path.exists(cfg_path):
        g = OmegaConf.to_container(OmegaConf.load(cfg_path).generator, resolve=True)
        g.pop("kind")
    else:
        g = {k: (dict(v) if isinstance(v, dict) else v)
             for k, v in BIG_LAMA_GEN_CFG.items()}
    g["ngf"] = ngf
    g["n_blocks"] = n_blocks
    return g


def _load_teacher_sd(src=None, ckpt_dir=None):
    """Teacher weights to slice from, as a bare FFCResNetGenerator state_dict.

    `src=None` -> raw big-lama (`external/lama/big-lama/models/best.ckpt`,
    keys prefixed `generator.`).
    `src=<path>` -> a mangainpaint/trainer.py checkpoint of a fine-tuned
    `LamaTransferG` (keys under "G", prefixed `net.`) -- i.e. one of the
    `lama_transfer_*` runs.

    **Prefer the fine-tuned source.** Verified locally: slicing raw big-lama
    gives a WORSE step-0 hole-L1 than random init (0.959 vs 0.856) -- but
    that test can't tell "the slicing destroyed the function" apart from
    "raw big-lama is a bad zero-shot manga model," which this project already
    established independently (it collapses to near-blank white on manga).
    A manga-fine-tuned LaMa has none of that problem and is architecturally
    identical, so it is the strictly better thing to slice.
    """
    if src is None:
        ckpt = torch.load(os.path.join(ckpt_dir or _CKPT_DIR, "models", "best.ckpt"),
                          map_location="cpu", weights_only=False)
        return {k[len("generator."):]: v for k, v in ckpt["state_dict"].items()
                if k.startswith("generator.")}

    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    sd = ckpt["G"] if "G" in ckpt else ckpt
    out = {k[len("net."):]: v for k, v in sd.items() if k.startswith("net.")}
    assert out, f"no `net.*` keys in {src} -- is this a LamaTransferG checkpoint?"
    return out


def _remap_block_index(key, student_nblocks, head_len):
    """Rewrite a teacher `model.<i>.…` key to the student's block numbering.

    Layout is fixed by FFCResNetGenerator.__init__:
        model[0]              ReflectionPad2d
        model[1]              init FFC_BN_ACT
        model[2 : 2+n_down]   downsample FFC_BN_ACT   (n_down=3)
        model[head_len : head_len+n_blocks]   FFC resblocks
        model[head_len+n_blocks : ]           ConcatTupleLayer + upsample + head
    (head_len = 2 + n_downsampling = 5; this is also why LamaTransferG's
    documented `freeze_up_to=16` == "pad + init + 3 downsample + first 11 of
    18 resblocks" -- 5 + 11 = 16. Consistent, so the layout is confirmed.)

    The student keeps every non-resblock module at the same index. For the
    resblocks it selects `student_nblocks` of the teacher's
    TEACHER_NBLOCKS, **evenly spaced** rather than the first-k -- LaMa's
    later resblocks operate on progressively more refined bottleneck
    features, so taking a prefix would throw away the whole back half of the
    refinement chain. Returns None for teacher keys with no student slot.
    """
    parts = key.split(".")
    if parts[0] != "model":
        return key
    i = int(parts[1])

    if i < head_len:
        return key

    t_tail_start = head_len + TEACHER_NBLOCKS
    if i >= t_tail_start:
        # tail (ConcatTupleLayer, upsample, out head) -- shift by the block delta
        s_i = i - TEACHER_NBLOCKS + student_nblocks
        return ".".join(["model", str(s_i)] + parts[2:])

    # resblock: teacher block (i - head_len) -> student slot, if selected
    t_block = i - head_len
    if student_nblocks == 1:
        chosen = [0]
    else:
        chosen = [round(j * (TEACHER_NBLOCKS - 1) / (student_nblocks - 1))
                  for j in range(student_nblocks)]
    if t_block not in chosen:
        return None
    s_block = chosen.index(t_block)
    return ".".join(["model", str(head_len + s_block)] + parts[2:])


def sliced_init_(net, student_nblocks, head_len=5, ckpt_dir=None, verbose=True,
                 slice_from=None):
    """Structured-slice a teacher's weights into `net` (in place).

    `slice_from`: path to a fine-tuned `LamaTransferG` checkpoint, or None
    for raw big-lama. See `_load_teacher_sd` -- fine-tuned is strictly
    better and is what S1/S2/S3 use.

    Every teacher tensor is cropped to the student's shape along each
    dimension (`t[:s0, :s1, …]`). Cross-layer consistency holds because FFC
    keeps its local/global paths in *separate* submodules (convl2l, convl2g,
    convg2l, convg2g), each with its own in/out channel dims -- so a
    per-tensor first-k crop selects the same channel indices on both sides of
    every connection. (A naive first-k crop over a single fused local+global
    channel axis would NOT be coherent; that trap is avoided by FFC's own
    parameterization, not by luck.)

    No importance ranking is applied -- first-k, not top-k-by-L1-norm.
    That makes this a hypothesis to be tested: does it beat random init at
    step 0? (see the module docstring's step-0 hole-L1 table -- it does
    not, meaningfully). Returns (n_loaded, n_skipped, n_shape_dropped)."""
    sd = _load_teacher_sd(slice_from, ckpt_dir)
    own = net.state_dict()

    loaded = skipped = dropped = 0
    new_sd = {}
    for k, t in sd.items():
        s_key = _remap_block_index(k, student_nblocks, head_len)
        if s_key is None or s_key not in own:
            skipped += 1
            continue
        s = own[s_key]
        if t.dim() != s.dim():
            dropped += 1
            continue
        if any(sd_ > td_ for sd_, td_ in zip(s.shape, t.shape)):
            # student is WIDER than teacher on some axis -- can't crop into it
            dropped += 1
            continue
        sl = tuple(slice(0, d) for d in s.shape)
        new_sd[s_key] = t[sl].clone()
        loaded += 1

    missing = [k for k in own if k not in new_sd]
    net.load_state_dict(new_sd, strict=False)
    if verbose:
        srcname = os.path.basename(slice_from) if slice_from else "raw big-lama"
        print(f"[sliced_init] loaded {loaded} tensors from {srcname}, "
              f"skipped {skipped} (no student slot), dropped {dropped} (shape), "
              f"left random-init: {len(missing)}")
    return loaded, skipped, dropped


class LamaSlimG(nn.Module):
    """Slim FFC-ResNet student. See module docstring."""

    def __init__(self, ngf=32, n_blocks=12, init_mode="sliced", slice_from=None,
                 expose_bottleneck=False, use_screenvae_hint=False,
                 screenvae_weights_dir=None, completion_base=48,
                 ckpt_dir=None):
        super().__init__()
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.init_mode = init_mode
        self.expose_bottleneck = expose_bottleneck
        self.use_screenvae_hint = use_screenvae_hint

        gcfg = _gen_cfg(ngf, n_blocks, ckpt_dir)
        self.n_downsampling = gcfg["n_downsampling"]
        self.head_len = 2 + self.n_downsampling  # = 5
        self.net = FFCResNetGenerator(**gcfg)

        if init_mode == "sliced":
            sliced_init_(self.net, n_blocks, self.head_len, ckpt_dir,
                         slice_from=slice_from)
        elif init_mode != "random":
            raise ValueError(f"init_mode must be 'sliced' or 'random', got {init_mode!r}")

        # Bottleneck channel split, mirroring FFC's own arithmetic
        # (in_cg = int(C * ratio_gin); in_cl = C - in_cg).
        self.bneck_ch = ngf * (2 ** self.n_downsampling)          # 256 @ ngf=32
        ratio = gcfg["resnet_conv_kwargs"]["ratio_gin"]           # 0.75
        self.bneck_cg = int(self.bneck_ch * ratio)                # 192
        self.bneck_cl = self.bneck_ch - self.bneck_cg             # 64

        self.last_bottleneck = None
        if expose_bottleneck:
            teacher_bneck = TEACHER_NGF * (2 ** self.n_downsampling)  # 512
            self.distill_adapter = nn.Conv2d(self.bneck_ch, teacher_bneck, 1)

        if use_screenvae_hint:
            from mangainpaint.model_screenvae import ScreenVAE, LatentCompletionNet
            self.screenvae = ScreenVAE(weights_dir=screenvae_weights_dir)
            self.completion = LatentCompletionNet(latent_ch=self.screenvae.outc,
                                                  base=completion_base)
            # Zero-init, same warm-start-safe convention as
            # MangaFillNetScreenVAE.hint_proj: at step 0 this model is
            # byte-identical to the no-hint one, so any measured effect is
            # the lever actually engaging, not a different starting point.
            self.hint_proj = nn.Conv2d(self.screenvae.outc, self.bneck_ch, 1)
            nn.init.zeros_(self.hint_proj.weight)
            nn.init.zeros_(self.hint_proj.bias)

    def train(self, mode=True):
        super().train(mode)
        if self.use_screenvae_hint:
            self.screenvae.eval()  # frozen submodule, always eval
        return self

    def _forward_net(self, inp4, mask):
        """Run FFCResNetGenerator's Sequential by hand, so we can (a) inject
        the ScreenVAE hint into the bottleneck tuple and (b) stash the
        bottleneck for distillation. Semantically identical to
        `self.net(inp4)` when both are off."""
        m = self.net.model
        x = inp4
        for i in range(self.head_len):
            x = m[i](x)
        # x is now the (x_l, x_g) FFC tuple at bottleneck resolution.

        if self.use_screenvae_hint:
            x = self._inject_hint(x, inp4, mask)

        for i in range(self.head_len, self.head_len + self.n_blocks):
            x = m[i](x)

        if self.expose_bottleneck:
            x_l, x_g = x
            cat = torch.cat([t for t in (x_l, x_g) if torch.is_tensor(t)], dim=1)
            self.last_bottleneck = cat
            # Continue the decode path FROM the cat node (re-sliced; values
            # byte-identical, autograd graph topology is the only change) so
            # `last_bottleneck` sits ON the main graph rather than being a
            # dead-end branch only the KD terms touch. Needed by
            # `mangainpaint/distill.py:adaptive_gn_multipliers`, which probes the
            # TASK loss's gradient at this tensor -- as a dead-end stash that
            # gradient is None and the feat/patchnce multipliers silently
            # never update. No effect on any existing run's numbers: forward values and
            # full-backward parameter gradients are identical either way.
            if torch.is_tensor(x_l) and torch.is_tensor(x_g):
                x = (cat[:, :x_l.shape[1]], cat[:, x_l.shape[1]:])
            elif torch.is_tensor(x_l):
                x = (cat, x_g)
            else:
                x = (x_l, cat)

        for i in range(self.head_len + self.n_blocks, len(m)):
            x = m[i](x)
        return x

    def _inject_hint(self, x, inp4, mask):
        img_masked = inp4[:, 0:1] * 2 - 1  # ScreenVAE expects the [-1,1] convention
        with torch.no_grad():  # frozen; nothing upstream to backprop into
            screen_masked = self.screenvae(img_masked, line=None)
        mask_ds = F.interpolate(mask.float(), size=screen_masked.shape[2:], mode="nearest")
        noise = torch.randn_like(mask_ds)
        completion_pred = self.completion(screen_masked, mask_ds, noise)
        screen_completed = screen_masked * (1 - mask_ds) + completion_pred * mask_ds

        x_l, x_g = x
        ref = x_l if torch.is_tensor(x_l) else x_g
        hint = self.hint_proj(screen_completed)
        if hint.shape[2:] != ref.shape[2:]:
            hint = F.interpolate(hint, size=ref.shape[2:], mode="bilinear", align_corners=False)
        hint_l, hint_g = torch.split(hint, [self.bneck_cl, self.bneck_cg], dim=1)
        if torch.is_tensor(x_l):
            x_l = x_l + hint_l
        if torch.is_tensor(x_g):
            x_g = x_g + hint_g
        return x_l, x_g

    def forward(self, x):
        masked, mask = x[:, 0:1], x[:, 1:2]
        # Same [0,1]-range + 3-channel-repeat input adaptation as
        # LamaTransferG (a sliced init inherits big-lama's BatchNorm running
        # stats, so the calibrated input range matters here for the same
        # reason it does there).
        img01 = (masked + 1) / 2
        img3 = img01.repeat(1, 3, 1, 1)
        inp4 = torch.cat([img3, mask], dim=1)
        # Whole generator in fp32: the stacked FFC-ResNet blocks all run at
        # the same bottleneck resolution and, with large constant-filled
        # masked regions, produce very large frequency-domain magnitudes that
        # overflow fp16 in a *later plain conv* (not in the FFT itself).
        # Identical failure and identical fix as LamaTransferG -- see its
        # forward() for the full trace.
        with torch.amp.autocast(x.device.type, enabled=False):
            out3 = self._forward_net(inp4.float(), mask.float())  # sigmoid, [0,1]
        out3 = out3.to(x.dtype)
        gray = out3.mean(dim=1, keepdim=True)
        return gray * 2 - 1
