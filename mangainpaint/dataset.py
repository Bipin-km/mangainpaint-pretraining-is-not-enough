"""
Manga109 dataset + masking, shared by every training run.

Speed fix: earlier per-experiment training scripts re-decoded
and re-resized every page from disk with PIL on every __getitem__ call, every
epoch, for 50 epochs. Since image_size is fixed and resize is deterministic,
that work only needs to happen once per (image_path, image_size) pair — this
module now pre-decodes+resizes into an in-RAM uint8 cache (IMG_CACHE) built
before training starts, so all 50 epochs read from RAM instead of disk.
DataLoader workers are forked from the process that builds the cache, so they
share it via copy-on-write without duplicating memory per worker.
"""
import json
import os
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
import pandas as pd
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from mangainpaint.ddp_utils import is_main

# ══════════════════════════════════════════════════════════
# XML BOX CACHE (text bbox exclusion zones)
# ══════════════════════════════════════════════════════════
BOX_CACHE = {}


def build_box_cache(csv_path, root_dir, rank=0):
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    iterator = df.iterrows()
    if is_main(rank):
        iterator = tqdm(iterator, total=len(df),
                        desc=f"Caching boxes {os.path.basename(csv_path)}", leave=False)
    for _, row in iterator:
        img_path = row['image_path']
        if img_path in BOX_CACHE: continue
        ann_path = os.path.join(root_dir, str(row.get('annotation_path', '')))
        page_idx = int(row.get('page_index', 0))
        try:
            pages = ET.parse(ann_path).getroot().findall('.//page')
            if page_idx < len(pages):
                BOX_CACHE[img_path] = [
                    (float(t.get("xmin")), float(t.get("ymin")),
                     float(t.get("xmax")), float(t.get("ymax")))
                    for t in pages[page_idx].findall('text')
                ]
            else:
                BOX_CACHE[img_path] = []
        except Exception:
            BOX_CACHE[img_path] = []


# ══════════════════════════════════════════════════════════
# BALLOON SEGMENTATION CACHE (Axis B2 -- real balloon-shaped masks from the
# CVPR2025 MS92/MangaSegmentation annotations, in addition to/blended with the
# brush-stroke masks below). Optional: only touched when a run's cfg sets
# `seg_root` + `mask_balloon_prob > 0`; `pycocotools` is only imported lazily
# here so B1-only runs never need it installed.
#
# Stores the raw COCO RLE dict per balloon, NOT a decoded+resized dense mask:
# an average page has ~11 balloon annotations, and eagerly decoding+resizing
# all of them to 384x384 uint8 for the full 6,788-image train set would cost
# ~11GB of RAM for masks that are each individually used only sometimes (a
# `__getitem__` call only ever needs 1-2 per sample). RLE is a few KB per
# mask; decode+resize happens lazily in `generate_balloon_mask` only for the
# 1-2 chosen indices, at a measured ~4ms/mask -- negligible against
# real per-epoch times of several minutes.
# ══════════════════════════════════════════════════════════
BALLOON_CACHE = {}          # image_path -> list of COCO RLE segmentation dicts
_SEG_BOOK_CACHE = {}        # normalized book name -> parsed {page_to_id, anns_by_image} (or None)


def _norm_book(name):
    """MS92/MangaSegmentation spells a couple of book names with an apostrophe
    (e.g. "That'sIzumiko") where our Manga109s XML/csv drop it
    ("ThatsIzumiko") -- match on the apostrophe-stripped name instead of the
    literal filename."""
    return name.replace("'", "")


def _load_seg_book(seg_root, book_norm, seg_books_norm):
    if book_norm in _SEG_BOOK_CACHE:
        return _SEG_BOOK_CACHE[book_norm]
    json_name = seg_books_norm.get(book_norm)
    if json_name is None:
        _SEG_BOOK_CACHE[book_norm] = None
        return None
    with open(os.path.join(seg_root, "jsons", f"{json_name}.json"), "r", encoding="utf8") as f:
        data = json.load(f)
    balloon_cat = next((c["id"] for c in data["categories"] if c["name"] == "balloon"), None)
    # Match by the page number parsed from the json's own file_name, not the
    # literal file_name string -- sidesteps the apostrophe-spelling mismatch
    # entirely since page numbering is otherwise identical between the two
    # Manga109 annotation sources.
    page_to_id = {}
    for im in data["images"]:
        stem = os.path.splitext(os.path.basename(im["file_name"]))[0]
        try:
            page_to_id[int(stem)] = im["id"]
        except ValueError:
            continue
    anns_by_image = defaultdict(list)
    if balloon_cat is not None:
        for a in data["annotations"]:
            if a["category_id"] == balloon_cat:
                anns_by_image[a["image_id"]].append(a)
    parsed = {"page_to_id": page_to_id, "anns_by_image": anns_by_image}
    _SEG_BOOK_CACHE[book_norm] = parsed
    return parsed


def build_balloon_cache(csv_path, seg_root, rank=0):
    if not seg_root or not os.path.exists(csv_path) or not os.path.exists(seg_root):
        return
    seg_books_norm = {_norm_book(f[:-5]): f[:-5]
                      for f in os.listdir(os.path.join(seg_root, "jsons"))}
    df = pd.read_csv(csv_path)
    iterator = df.iterrows()
    if is_main(rank):
        iterator = tqdm(iterator, total=len(df),
                        desc=f"Caching balloons {os.path.basename(csv_path)}", leave=False)
    for _, row in iterator:
        img_path = row["image_path"]
        if img_path in BALLOON_CACHE: continue
        book = img_path.split("/")[1]
        try:
            page_num = int(os.path.splitext(os.path.basename(img_path))[0])
        except ValueError:
            BALLOON_CACHE[img_path] = []
            continue
        parsed = _load_seg_book(seg_root, _norm_book(book), seg_books_norm)
        rles = []
        if parsed is not None:
            image_id = parsed["page_to_id"].get(page_num)
            if image_id is not None:
                rles = [a["segmentation"] for a in parsed["anns_by_image"].get(image_id, [])]
        BALLOON_CACHE[img_path] = rles


# ══════════════════════════════════════════════════════════
# IMAGE CACHE (pre-decoded + pre-resized, uint8, in RAM)
# ══════════════════════════════════════════════════════════
IMG_CACHE = {}


def build_img_cache(csv_path, root_dir, image_size, rank=0):
    """Decode + resize every page once and keep it in RAM as uint8.

    Also records the original (pre-resize) width/height, needed to rescale
    the XML text bboxes into the resized coordinate frame at __getitem__ time.
    """
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    iterator = df.iterrows()
    if is_main(rank):
        iterator = tqdm(iterator, total=len(df),
                        desc=f"Caching images {os.path.basename(csv_path)}", leave=False)
    for _, row in iterator:
        img_path = row['image_path']
        if img_path in IMG_CACHE: continue
        img_pil = Image.open(os.path.join(root_dir, img_path)).convert('L')
        ow, oh = img_pil.size
        resized = img_pil.resize((image_size, image_size), Image.BILINEAR)
        IMG_CACHE[img_path] = (np.asarray(resized, dtype=np.uint8).copy(), ow, oh)


def _procedural_mask_np(H, W, cfg, is_train):
    """Random large-box or brush-stroke mask (Axis B1), as a raw numpy array
    with no exclusion applied yet -- factored out of `generate_mask` so
    `generate_balloon_mask` (Axis B2) can blend in a stroke without
    duplicating exclusion-box logic."""
    mask = np.zeros((H, W), dtype=np.uint8)
    if is_train and np.random.random() < cfg["mask_large_prob"]:
        frac = np.random.uniform(0.05, cfg["mask_large_frac"])
        area = int(H * W * frac)
        bh = int((area ** 0.5) * np.random.uniform(0.5, 1.5))
        bw = max(1, area // max(bh, 1))
        bh, bw = min(bh, H - 1), min(bw, W - 1)
        y0 = np.random.randint(0, H - bh + 1); x0 = np.random.randint(0, W - bw + 1)
        mask[y0:y0 + bh, x0:x0 + bw] = 1
    else:
        for _ in range(np.random.randint(cfg["mask_strokes_min"], cfg["mask_strokes_max"] + 1)):
            sx0, sy0 = np.random.randint(0, W), np.random.randint(0, H)
            bw = np.random.randint(cfg["mask_brush_w_min"], cfg["mask_brush_w_max"] + 1)
            ang = np.random.uniform(0, 2 * np.pi)
            for _ in range(np.random.randint(3, 9)):
                ang += np.random.uniform(-np.pi / 3, np.pi / 3)
                L = np.random.randint(cfg["mask_len_min"], cfg["mask_len_max"] + 1)
                ex = int(np.clip(sx0 + L * np.cos(ang), 0, W - 1))
                ey = int(np.clip(sy0 + L * np.sin(ang), 0, H - 1))
                cv2.line(mask, (sx0, sy0), (ex, ey), 1, int(bw))
                sx0, sy0 = ex, ey
    return mask


def generate_mask(H, W, exclusion_boxes, cfg, is_train=True):
    mask = _procedural_mask_np(H, W, cfg, is_train)
    for x1, y1, x2, y2 in exclusion_boxes:
        mask[y1:y2, x1:x2] = 0
    return torch.from_numpy(mask).float().unsqueeze(0)


def generate_balloon_mask(balloon_rles, exclusion_boxes, H, W, cfg, is_train=True):
    """Axis B2: union of 1-2 real balloon segmentation masks for this page,
    optionally blended with one procedural brush stroke
    (`balloon_extra_stroke_prob`). Real text bboxes are carved out exactly
    like the B1 path -- masking a balloon's visible white background is
    valid (it's real, known paper), only the actual printed-glyph pixels have
    no ground truth underneath.

    `balloon_rles` are raw COCO RLE dicts (see `BALLOON_CACHE`) -- decoded
    and resized to HxW here, lazily, only for the 1-2 indices actually
    chosen (not all of a page's ~11 balloons), to keep this cheap enough to
    call every `__getitem__` without needing to cache dense masks in RAM.
    """
    import pycocotools.mask as maskUtils

    n = min(len(balloon_rles), np.random.randint(1, 3))
    idx = np.random.choice(len(balloon_rles), size=n, replace=False)
    mask = np.zeros((H, W), dtype=np.uint8)
    for i in idx:
        m = maskUtils.decode(balloon_rles[i])
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = np.maximum(mask, m)
    if np.random.random() < cfg.get("balloon_extra_stroke_prob", 0.3):
        mask = np.maximum(mask, _procedural_mask_np(H, W, cfg, is_train))
    for x1, y1, x2, y2 in exclusion_boxes:
        mask[y1:y2, x1:x2] = 0
    return torch.from_numpy(mask).float().unsqueeze(0)


# ══════════════════════════════════════════════════════════
# AXIS B3: Perlin-noise masking (RAD, Kim/Suh/Lee, arXiv:2412.09191,
# Sec. 3.2 "Generating Noise Schedules"). Real mechanism per the paper:
# thresholded Perlin noise as a surrogate for real inpainting-mask
# distributions -- the noise's spatial scale is sampled to mix fine and
# coarse structure in one mask, and the black/white threshold is sampled
# separately to control the overall masked area. Orthogonal to both B1
# (brush-stroke, jagged/thin) and B2 (real balloon shapes, smooth but
# fixed-vocabulary) -- Perlin masks are smooth AND irregular AND blobby
# at a randomizable scale, a distinct point in mask-shape-space.
#
# Implemented as multi-octave value noise (coarse-random-grid ->
# bilinear-upsample -> accumulate at decreasing amplitude per octave)
# rather than true gradient-based Perlin noise -- this is the standard
# practical substitute used by most inpainting-mask-generation codebases
# for exactly this purpose (e.g. free-form mask generators), avoids
# adding a new pip dependency (a compiled `noise` package, real risk on
# ephemeral cloud training environments) for a property (gradient continuity) that doesn't matter once
# the result is thresholded to binary, and produces the same qualitative
# "smooth multi-scale blob" character the paper describes.
# ══════════════════════════════════════════════════════════
def _value_noise_2d(H, W, base_res, octaves=4, persistence=0.5):
    """Fractal value noise: sum of `octaves` random grids, each generated
    at `base_res * 2**i` resolution and bilinear-upsampled to HxW, summed
    with amplitude decaying by `persistence` per octave (standard fBm
    construction). Returns float32 array normalized to [0, 1]."""
    noise = np.zeros((H, W), dtype=np.float32)
    amp, amp_total = 1.0, 0.0
    for i in range(octaves):
        res = max(2, int(base_res * (2 ** i)))
        grid = np.random.rand(res, res).astype(np.float32)
        layer = cv2.resize(grid, (W, H), interpolation=cv2.INTER_LINEAR)
        noise += amp * layer
        amp_total += amp
        amp *= persistence
    noise /= amp_total
    lo, hi = noise.min(), noise.max()
    return (noise - lo) / (hi - lo + 1e-8)


def generate_perlin_mask(H, W, exclusion_boxes, cfg, is_train=True):
    """Axis B3: thresholded multi-octave (pseudo-Perlin) noise mask.
    `perlin_base_res` (spatial scale) and the binarization threshold are
    both randomly sampled per call, matching the paper's own description
    ("spatial scales are uniformly sampled... black-and-white conversion
    threshold [is sampled] to control the overall area")."""
    base_res = np.random.randint(cfg.get("perlin_base_res_min", 2),
                                 cfg.get("perlin_base_res_max", 8) + 1)
    noise = _value_noise_2d(H, W, base_res, octaves=cfg.get("perlin_octaves", 4))
    # Threshold sampled so the resulting mask area lands in a plausible
    # inpainting range (same spirit as mask_large_frac's range for B1) --
    # thresholding a uniform-ish noise field at quantile `q` masks
    # approximately the top (1-q) fraction of the field.
    frac = np.random.uniform(cfg.get("perlin_area_min", 0.05),
                             cfg.get("perlin_area_max", 0.35))
    thresh = np.quantile(noise, 1.0 - frac)
    mask = (noise >= thresh).astype(np.uint8)
    for x1, y1, x2, y2 in exclusion_boxes:
        mask[y1:y2, x1:x2] = 0
    return torch.from_numpy(mask).float().unsqueeze(0)


class Manga109Dataset(Dataset):
    def __init__(self, csv_path, root_dir, image_size, cfg, hole_fill="white", is_train=False):
        self.data = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.sz = image_size
        self.cfg = cfg
        self.is_train = is_train
        self.fill_val = {"white": 1.0, "black": -1.0, "zero": 0.0}[hole_fill]

    def __len__(self): return len(self.data)

    def _build_sample(self, idx, force_balloon=None):
        """`force_balloon`: None = normal per-sample roll (`__getitem__`'s
        behavior); True/False = bypass the roll, used by `build_vis_batch`
        below to guarantee balloon-mask representation in the fixed
        qualitative-viz batch instead of leaving it to RNG luck."""
        row = self.data.iloc[idx]
        img_path = row['image_path']
        arr, ow, oh = IMG_CACHE[img_path]
        img = torch.from_numpy(arr).float().div_(255.0).sub_(0.5).div_(0.5).unsqueeze(0)
        raw_boxes = BOX_CACHE.get(img_path, [])
        sx, sy = self.sz / ow, self.sz / oh
        boxes = [(int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))
                 for x1, y1, x2, y2 in raw_boxes]

        balloon_masks = BALLOON_CACHE.get(img_path) or []
        use_balloon = (force_balloon if force_balloon is not None else
                       (bool(balloon_masks) and np.random.random() < self.cfg.get("mask_balloon_prob", 0.0)))
        applied_balloon = use_balloon and bool(balloon_masks)
        if applied_balloon:
            mask = generate_balloon_mask(balloon_masks, boxes, self.sz, self.sz, self.cfg, self.is_train)
        elif self.is_train and np.random.random() < self.cfg.get("mask_perlin_prob", 0.0):
            # train-only gate: unlike
            # mask_balloon_prob (deliberately un-gated -- every B2 run's
            # val_score is meant to reflect a real-balloon-inclusive eval
            # distribution, consistently across the whole B2 leaderboard),
            # Perlin masking (Axis B3) was designed as a train-time-only
            # augmentation, meant to be judged against the same standard
            # eval distribution every other run uses. Without this gate,
            # val/test samples also rolled Perlin-shaped holes, silently
            # giving perlin_test's own val_score a different eval mask mix
            # than every other run it was being compared against.
            mask = generate_perlin_mask(self.sz, self.sz, boxes, self.cfg, self.is_train)
        else:
            mask = generate_mask(self.sz, self.sz, boxes, self.cfg, self.is_train)
        masked = img * (1 - mask) + self.fill_val * mask
        return {
            "image": img,
            "mask": mask,
            "masked_image": masked,
            # Axis B2 (real balloon/object-semantic mask) vs everything
            # else (procedural brush-stroke or Perlin) -- the
            # foreground/background category signal for LCG-lite
            # (mangainpaint/model_lcg.py). Harmless extra field for every other
            # generator (unused unless `wants_category=True`).
            "is_balloon": torch.tensor(1.0 if applied_balloon else 0.0),
            "model_input": torch.cat([masked, mask], 0),
        }

    def __getitem__(self, idx):
        return self._build_sample(idx)


def build_vis_batch(val_ds, cfg, n=4):
    """Deterministic qualitative-viz batch. `trainer.py` visualizes one fixed
    batch every epoch (drawn once, before training starts) rather than a
    fresh one per epoch, so whatever mask type that batch happens to get is
    what shows up in every `vis/epoch_*.png` for the whole run. Left to the
    per-sample RNG roll (like plain `__getitem__`), a `mask_balloon_prob=0.5`
    run has a real (observed) chance of the fixed batch's few samples all
    rolling procedural brush strokes, hiding Axis B2's balloon masks from
    every qualitative figure despite them being used throughout actual
    training/eval. Sidesteps that by forcing real balloon masks (bypassing
    the roll) for the first half of the batch whenever `mask_balloon_prob >
    0` and the page has real balloon annotations, and forcing pure
    procedural masks for the rest -- so the panel always shows both mask
    types side by side for Axis B2 runs instead of leaving it to chance.
    """
    want_balloon = cfg.get("mask_balloon_prob", 0.0) > 0.0
    n_balloon = n // 2 if want_balloon else 0
    samples, balloon_taken = [], 0
    for idx in range(len(val_ds)):
        if len(samples) >= n:
            break
        img_path = val_ds.data.iloc[idx]['image_path']
        force = None
        if want_balloon:
            has_balloon = bool(BALLOON_CACHE.get(img_path))
            force = has_balloon and balloon_taken < n_balloon
            if force: balloon_taken += 1
        samples.append(val_ds._build_sample(idx, force_balloon=force))
    keys = samples[0].keys()
    return {k: torch.stack([s[k] for s in samples]) for k in keys}


def make_loaders(cfg, rank, world_size):
    if is_main(rank):
        print("Pre-loading XML boxes + decoding/resizing images to RAM...")
    for split in ("train_csv", "val_csv", "test_csv"):
        build_box_cache(cfg[split], cfg["root_dir"], rank)
        build_img_cache(cfg[split], cfg["root_dir"], cfg["image_size"], rank)
        if cfg.get("mask_balloon_prob", 0.0) > 0.0:
            build_balloon_cache(cfg[split], cfg.get("seg_root"), rank)

    train_ds = Manga109Dataset(cfg["train_csv"], cfg["root_dir"],
                               cfg["image_size"], cfg, cfg["hole_fill"], True)
    val_ds = Manga109Dataset(cfg["val_csv"], cfg["root_dir"], cfg["image_size"], cfg)
    test_ds = Manga109Dataset(cfg["test_csv"], cfg["root_dir"], cfg["image_size"], cfg)

    train_sampler = DistributedSampler(train_ds, world_size, rank, shuffle=True, drop_last=True)
    val_sampler = DistributedSampler(val_ds, world_size, rank, shuffle=False, drop_last=False)
    test_sampler = DistributedSampler(test_ds, world_size, rank, shuffle=False, drop_last=False)

    num_workers = cfg["num_workers"]
    if num_workers is None:
        num_workers = max(1, min(4, (os.cpu_count() or 2) // max(1, world_size)))

    kw = dict(num_workers=num_workers, pin_memory=True)
    if num_workers > 0:
        # IMG_CACHE/BOX_CACHE are built in this process and must reach workers
        # via copy-on-write fork — some Python builds default multiprocessing
        # to 'forkserver'/'spawn', which would re-import this module fresh in
        # each worker (empty caches -> KeyError). Force 'fork' explicitly.
        kw.update(persistent_workers=True, prefetch_factor=4,
                  multiprocessing_context="fork")
    trl = DataLoader(train_ds, cfg["batch_size"], sampler=train_sampler, **kw)
    val = DataLoader(val_ds, cfg["batch_size"], sampler=val_sampler, **kw)
    tel = DataLoader(test_ds, cfg["batch_size"], sampler=test_sampler, **kw)
    return trl, val, tel, train_sampler
