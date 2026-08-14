# Pretraining Is Not Enough

Transfer, distillation, and benchmark design for small-data manga inpainting.

A controlled comparison of the three routes to a manga inpainting model at a data
scale of a few thousand pages (training a purpose-built GAN from scratch, applying
a large pretrained model zero-shot, and fine-tuning a pretrained model), plus the
distillation recipe that compresses the winner into an 8.78M-parameter student.

Research code for a manuscript under review at *The Visual Computer*. Nothing is
published yet, so there is no citation or DOI to point at.

## Authors

**Bipin Kumar Marasini** · **Nitesh Kumar Sah** ·
**Ramesh Kathayat** · **Rajad Shakya** (supervisor)

## Results

All models trained by us under one shared recipe, one masking distribution and one
selection rule, then scored in a single pass over a 907-page, book-disjoint
held-out split of Manga109-s. EdgeF1 and LPIPS are primary; PSNR and SSIM are
reported for completeness and are gameable by a blank-paper fill.

| Model | Params (M) | EdgeF1 ↑ | LPIPS ↓ | PSNR | SSIM |
|---|---|---|---|---|---|
| PConv-UNet | 25.8 | 0.296 | 0.0305 | 15.55 | 0.926 |
| UFFC-GAN | 3.1 | 0.424 | 0.0215 | 16.94 | 0.945 |
| CtxAttn-GAN | 2.7 | 0.429 | 0.0219 | 17.15 | 0.946 |
| FFC-GAN | 2.8 | 0.430 | 0.0218 | 17.13 | 0.946 |
| S1 student, no distillation | 8.8 | 0.420 | 0.0226 | 17.02 | 0.942 |
| S2 student, + distillation | 8.8 | 0.432 | 0.0206 | 17.24 | 0.946 |
| S3 student, + external losses | 8.8 | 0.431 | 0.0191 | 16.99 | 0.943 |
| **Fine-tuned LaMa, teacher** | 51.0 | **0.475** | **0.0170** | **17.77** | **0.951** |

1. **Fine-tuned transfer wins decisively**, leading every metric and every
   ink-density stratum. Partial freezing was consistently worse, monotonically so
   as fewer layers were trainable.
2. **Pretraining alone is not the reason.** The same checkpoint applied zero-shot
   collapses to near-blank fills, and Stable Diffusion 1.5 (~860M) and Moebius
   (0.22B) both land below every purpose-trained model here.
3. **From-scratch architecture search does not close the gap.** Twelve fully
   trained variants produce a 0.42–0.43 band. Levers that only re-parameterize
   information already in the input are null at this data scale; every lever
   injecting information from outside the corpus helps.
4. **Distillation buys perceptual quality and inference cost, not structure.** The
   four-term recipe improves the student's EdgeF1 by 2.9% and LPIPS by 9.1% over an
   identically trained undistilled control at 3.2× lower latency than the teacher,
   but does not reliably beat a 2.8M from-scratch FFC-GAN on edge structure.

## The benchmark, and why it is a contribution

Most of any randomly placed hole on a manga page is blank paper, so a model that
fills every hole with confident near-white scores well on PSNR and SSIM while
reconstructing nothing. On identical holes, PSNR and SSIM rank a near-blank-fill
baseline *above* a modern diffusion inpainter that recovers 21% more edge
structure.

The protocol answers that with edge-based primary metrics, ink-density strata, and
frozen masks. EdgeF1 is the whole idea in a dozen lines: F1 between Canny edge
maps *inside the hole*, so a blank fill scores zero however close its pixel values
are ([`mangainpaint/metrics.py`](mangainpaint/metrics.py)):

```python
def hole_edge_f1(p, t, m, eps=1e-8):
    def canny(x): return (cv2.Canny((x * 255).astype(np.uint8), 60, 160) > 0).astype(np.uint8)
    ep, et = canny(p) * mv, canny(t) * mv          # mv = hole mask
    tp = float((ep & et).sum())
    fp = float((ep & (1 - et)).sum())
    fn = float(((1 - ep) & et).sum())
    pr, rc = tp / (tp + fp + eps), tp / (tp + fn + eps)
    return float(2 * pr * rc / (pr + rc + eps))
```

The other two measures:

- **Ink-density strata.** Every metric is also reported over three *fixed* cut
  points on the hole ink fraction (sparse `[0, 0.05)`, moderate `[0.05, 0.20)`,
  dense `[0.20, 1]`), so failure on content-rich regions cannot hide inside an
  average dominated by blank pages. Fixed, not quantiles, so a stratum label means
  the same thing on any subset.
- **Byte-identical masks.** Holes are drawn once from seed 1234 in a fixed page
  order and every checkpoint is scored against that draw. For the compute-bound
  zero-shot baselines the masks are serialized to a SHA-256-checked file and every
  model in that comparison is scored against that same file, so like-for-like
  holds by construction.

Annotated text regions are excluded from all mask generation: printed dialogue is
typeset over the artwork and no ground truth exists beneath it.

## Layout

```
mangainpaint/   the library: models, losses, distillation, dataset, trainer, metrics
recipes/        one script per model reported in the paper
eval/           the protocol: held-out eval, pinned-mask eval, paired statistics
protocol/       the frozen mask files and their checksums (no corpus imagery)
```

The four distillation terms live in `DistillLoss`
([`mangainpaint/distill.py`](mangainpaint/distill.py)): hole-weighted output L1
(κ=4), bottleneck feature L1 through a training-only 1×1 adapter (256→512),
three-level Haar high-frequency L1, and a ReKo hole-InfoNCE term (τ=0.07). The
teacher runs under `no_grad`; nothing external survives into inference.

Every recipe below trains the architecture at the parameter count its paper row
reports, under the shared recipe: 50 epochs from scratch, 30 for transfer and
students, adversarial phase from epoch 10, brush-stroke masking with annotated
text regions excluded. `eval/eval_brush_batch.py` scores exactly these run ids.

| Recipe | Paper row | Original run id |
|---|---|---|
| `pconv_unet.py` | PConv-UNet | `pconv_baseline_v1` |
| `uffc_gan.py` | UFFC-GAN | `uffc_test_kaggle_v2` |
| `ctxattn_gan.py` | CtxAttn-GAN | `attn2_test_v2` |
| `ffc_gan.py` | FFC-GAN | `projected_d_test_v2` |
| `teacher_lama_finetune.py` | Fine-tuned LaMa, teacher | `lama_transfer_brush_v1` |
| `student_s1.py` | S1, undistilled control | `lama_slim_s1` |
| `student_s1_attn.py` | S1-attn, windowed attention | `lama_slim_s1_attn` |
| `student_s2_distill.py` | S2, four-term distillation | `lama_distill_s2` |
| `student_s3_external.py` | S3, + training-only external losses | `lama_distill_s3` |
| `student_c1_compact.py`, `student_c2_fusion.py` | C1, C2 width-floor probes | `lama_slim_c1_compact`, `lama_slim_c2_fusion` |
| `ablation_s2_gradnorm.py`, `ablation_s2_screenvae_kd.py` | distillation-signal ablations | `lama_distill_s2_gn`, `lama_distill_s2_svaekd` |

`recipes/model_pconv.py` is not a recipe: it is the standalone `PConvUNet`
architecture, kept in `recipes/` because both `pconv_unet.py` and
`checkpoint_registry.py` import it by bare module name from that directory.

## Data

**Manga109-s is not redistributed here and cannot be.** Its licence permits wider
reuse than full Manga109 but still requires a request to the maintainers:
<http://www.manga109.org/en/download_s.html>

The split is book-level to prevent style leakage: 6,788 training pages (69 titles),
824 validation (9 titles), 907 held-out test (9 titles), grayscale at 384×384. Title
lists are in `mangainpaint/dataset.py`.

Pretrained weights the recipes expect (Big-LaMa, EfficientNet-B0, ScreenVAE, an
ADE20K ResNet-50, LPIPS backbones) are fetched from their original sources by
`mangainpaint/checkpoint_registry.py`. None are vendored.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                    # add [diffusion] for the zero-shot baselines
```

### The LaMa source is a required second step

The teacher, the students and `DistillLoss` all import `FFCResNetGenerator`
from `saicinpainting`, which is not on PyPI — it lives in the LaMa repository
and is loaded from `external/lama/` on `sys.path`. Without this step,
`mangainpaint.model_lama`, `model_lama_slim`, `model_lama_slim_fus`,
`distill` and `checkpoint_registry` all raise `ModuleNotFoundError`:

```bash
git clone https://github.com/advimman/lama.git external/lama
```

That is enough for every from-scratch recipe and for the whole student family,
which build their generators with `init_mode="random"` and never read the
pretrained weights. Only `recipes/teacher_lama_finetune.py` additionally needs
the pretrained checkpoint, which it expects at `external/lama/big-lama/`
(`config.yaml` and `models/best.ckpt`) — download it from the LaMa repository's
own instructions. `external/` is git-ignored; nothing here is vendored.

Python 3.10, PyTorch 2.x with CUDA. Training was done on single- and dual-GPU
sessions; `mangainpaint/ddp_utils.py` handles the multi-GPU case. The paper's
inference benchmarks are from one laptop RTX 3050 and its host CPU.

## Running

```bash
export MANGA109_ROOT=/path/to/Manga109s     # the directory containing images/

python recipes/teacher_lama_finetune.py     # train the teacher
python recipes/student_s3_external.py       # train the distilled student
python eval/eval_brush_batch.py             # score on the held-out split
python eval/eval_pinned_models.py           # score against the checksummed mask file
python eval/eval_paired_stats.py --from-cache   # paired bootstrap CIs, no GPU needed

python eval/fixed_mask_protocol.py "$MANGA109_ROOT"   # verify both protocols load
```

`MANGA109_ROOT` is what makes the frozen protocols usable without the corpus
being redistributed. `protocol/*.npz` carries the mask arrays, page
identifiers, draw seed and SHA-256 digests but no page pixels;
`eval/fixed_mask_protocol.py` reads the named pages from your own licensed
copy and rebuilds the evaluation tensors through the same decode path training
uses. The reconstruction is exact --- verified bit-for-bit against the internal
pixel-carrying files for both the 150-page and the 907-page protocol --- and
the mask digest is checked on every load, so a modified protocol file fails
loudly instead of quietly changing the benchmark.

**Checkpoints are not released.** Every reported model is trained on Manga109-s,
whose licence permits commercial use of the -s subset but forbids redistribution
to third parties; the teacher is additionally a fine-tune of a third-party
pretrained checkpoint (`big-lama`, Apache-2.0), which would carry its own
redistribution terms on top. Rather than resolve that per checkpoint, none are
shipped. The recipes reproduce every reported checkpoint from scratch against
your own licensed Manga109-s copy, and the frozen mask protocol (`protocol/`)
lets you verify you land on the same numbers this paper reports without ever
needing a checkpoint file.

## Licence

MIT for the code in this repository (see `LICENSE`). Third-party components
keep their own licence: LaMa (Apache-2.0), ScreenVAE, the Manga109-s corpus,
and the pretrained backbones (fetched at run time, not vendored — see Data
above). No model checkpoints are distributed here; see Data.

## Acknowledgements

Builds directly on LaMa (Suvorov et al.), Projected GANs (Sauer et al.), ScreenVAE
(Xie et al.), wavelet knowledge distillation (Zhang et al.), region-aware knowledge
distillation (Zhang et al.), and Manga109 (Matsui et al.; Aizawa et al.).
