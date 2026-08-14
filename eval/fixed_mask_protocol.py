"""
Loader for the frozen evaluation protocols shipped in `release/protocol/`.

Every number in the paper is scored against one of two stored mask files, so
that "the same holes" is true by construction rather than by re-running a
documented procedure and trusting it to reproduce. (It did not, once: see the
main paper's zero-shot section.)

What ships and what does not
----------------------------
The `.npz` files carry the mask arrays, the page identifiers, the draw seed
and the SHA-256 digests. They do **not** carry Manga109-s page pixels, which
the corpus licence does not let us redistribute. This module is the other half:
it reads pages from your own licensed copy and reproduces the exact tensors the
paper scored, using the same decode path as training
(`PIL` grayscale -> bilinear resize to 384 -> `[-1,1]`).

    from fixed_mask_protocol import FixedMaskDataset
    ds = FixedMaskDataset("../protocol/fixed_eval_masks_150.npz",
                          root_dir="/path/to/Manga109s")

The mask digest is verified on every load, so a corrupted or substituted file
fails loudly rather than quietly changing the benchmark. Page pixels cannot be
digest-checked the same way -- they come from your copy -- but the resize path
is fixed, so a page that decodes differently is a corpus difference and not a
protocol difference.

Obtaining the corpus: Manga109/Manga109-s is distributed by its maintainers at
http://www.manga109.org/ under an academic-use agreement. `root_dir` is the
directory containing `images/`.
"""
import hashlib
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
PROTOCOL = os.path.abspath(os.path.join(HERE, "..", "protocol"))
PIN_PATH = os.path.join(PROTOCOL, "fixed_eval_masks_150.npz")
PIN_FULL = os.path.join(PROTOCOL, "fixed_eval_masks_full.npz")
IMAGE_SIZE = 384


def default_root():
    """Corpus location, for scripts that take no `root_dir` argument.

    The eval scripts construct a dataset with a file path only, so the corpus
    has to come from somewhere: set `MANGA109_ROOT` once and every script in
    this directory works unchanged.
    """
    return os.environ.get("MANGA109_ROOT")


def load_pinned(path=PIN_PATH, root_dir=None, image_size=IMAGE_SIZE):
    """(paths, masks, images, digest) for a frozen protocol file.

    `images` is reconstructed from `root_dir` unless the file carries pixels
    itself (the internal, unredistributable variant does; the released one
    does not).
    """
    d = np.load(path, allow_pickle=False)
    masks = d["masks"]
    digest = hashlib.sha256(masks.tobytes()).hexdigest()
    if digest != str(d["sha256"]):
        raise RuntimeError(f"{os.path.basename(path)}: mask digest mismatch -- "
                           "the file is corrupt or has been modified")

    if "images" in d:
        images = d["images"]
    elif "images_u8" in d:
        images = d["images_u8"].astype(np.float32) / 255.0 * 2.0 - 1.0
    else:
        root_dir = root_dir or default_root()
        if root_dir is None:
            raise ValueError(
                f"{os.path.basename(path)} carries no page pixels (by design: "
                "the corpus licence forbids redistributing them). Pass "
                "root_dir=<your Manga109-s directory>, or set MANGA109_ROOT.")
        images = _read_pages(d["paths"], root_dir, image_size)
    return d["paths"], masks, images, digest


def _read_pages(paths, root_dir, image_size):
    """Decode pages exactly as training does: grayscale, bilinear to
    `image_size`, scaled to [-1,1]. Any deviation here would silently make the
    reproduced numbers differ from the published ones, so this mirrors
    `mangainpaint.dataset.build_img_cache` + `_build_sample` line for line."""
    out = np.empty((len(paths), image_size, image_size), np.float32)
    missing = []
    for i, rel in enumerate(paths):
        p = os.path.join(root_dir, str(rel))
        if not os.path.exists(p):
            missing.append(str(rel))
            continue
        img = Image.open(p).convert("L").resize((image_size, image_size),
                                                Image.BILINEAR)
        arr = np.asarray(img, dtype=np.uint8).astype(np.float32)
        out[i] = arr / 255.0 * 2.0 - 1.0
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(paths)} pages not found under {root_dir!r}; "
            f"first missing: {missing[0]}. root_dir should be the directory "
            "that contains images/.")
    return out


class FixedMaskDataset(Dataset):
    """Serves the frozen (image, mask) pairs in the training sample format, so
    `mangainpaint.trainer.evaluate` scores against it unchanged. Consumes no
    RNG: iteration order and batch size cannot affect the holes."""

    def __init__(self, path=PIN_PATH, root_dir=None, hole_fill="white",
                 image_size=IMAGE_SIZE):
        self.paths, self.masks, self.images, self.sha = load_pinned(
            path, root_dir=root_dir, image_size=image_size)
        self.fill_val = {"white": 1.0, "black": -1.0, "zero": 0.0}[hole_fill]

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, i):
        img = torch.from_numpy(self.images[i]).unsqueeze(0)
        mask = torch.from_numpy(self.masks[i].astype(np.float32)).unsqueeze(0)
        masked = img * (1 - mask) + self.fill_val * mask
        return {"image": img, "mask": mask, "masked_image": masked,
                "is_balloon": torch.tensor(0.0),
                "model_input": torch.cat([masked, mask], 0)}


# The 907-page protocol is the same object with a different file; both names
# exist because the eval scripts import one or the other.
FullPinnedDataset = FixedMaskDataset


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else None
    for name, path in (("150-page", PIN_PATH), ("907-page", PIN_FULL)):
        paths, masks, images, sha = load_pinned(path, root_dir=root)
        print(f"{name}: {len(paths)} pages  digest {sha[:16]}...  "
              f"hole fraction {masks.mean():.4f}  "
              f"pixels {'from corpus' if root else 'embedded'}")
