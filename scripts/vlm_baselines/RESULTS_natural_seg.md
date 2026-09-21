# Natural-image segmentation: CLIP MaskCLIP reference vs PAL

CLIP as the segmentation reference point via **MaskCLIP** (canonical, ECCV'22): last
transformer block drops q-k attention and the FFN and uses the value projection only
(value → out_proj, no residual, no FFN), then ln_post + visual.proj per patch token.
Scale-matched to the PAL main table (dinov2-L) with **CLIP ViT-L/14**; at img_size 224
both give a 16×16 patch grid, so the spatial resolution is identical.

Protocol byte-matched to our pipeline `run_eval`: square Resize((224,224),BICUBIC),
per-patch L2-norm + cosine with text, bilinear upsample to GT, argmax, same confusion-matrix
mIoU, same DatasetSpec (VOC21 / Context60 / ADE151). Dataset classes + metric + templates
copied verbatim and GT-verified identical to the pipeline. Text = imagenet-80 template
ensemble (same as PAL's "ensemble" strategy). `script: maskclip_seg.py`.

Implementation note: the installed open_clip is `batch_first=True`; the dense forward uses
`visual._embeds()` and honors batch_first (CLS token reproduces `encode_image` at cosine=1.0).

## mIoU-fg (%), ensemble strategy

| dataset | PAL vitL (roberta) | MaskCLIP ViT-L/14 | Δ (PAL − CLIP) |
|---|---|---|---|
| VOC2012 (21) | **32.3** | 29.11 | +3.2 |
| Pascal Context (60) | **25.5** | 23.54 | +2.0 |
| ADE20K (151) | 13.8 | 13.76 | +0.04 (tie) |

PAL ≥ CLIP MaskCLIP on all three: clear wins on VOC and Context, a tie on the 151-class
ADE20K. Frozen DINOv2 spatial features + PAL alignment match or beat CLIP's native
web-scale text alignment on dense open-vocab segmentation.

## MaskCLIP full numbers (mIoU-fg / mIoU-all, %)

| dataset | raw | ensemble |
|---|---|---|
| VOC2012 | 26.37 / 25.77 | 29.11 / 29.41 |
| Pascal Context | 20.76 / 20.44 | 23.54 / 23.19 |
| ADE20K | 12.30 / 12.30 | 13.76 / 13.76 |

Notes:
- Pascal Context uses the all-trainval enumeration (10103 imgs, no val.txt) — same as the
  PAL pipeline, so PAL-vs-CLIP is apples-to-apples even though it is not the official
  Context-59 val split.
- Single-scale 224, no sliding window / key-smoothing — MaskCLIP under OUR seg protocol,
  the fair matched-protocol comparison to PAL (not MaskCLIP's paper number).
- CLIP is the single reference across all three natural-image tasks (zs-cls / retrieval /
  seg); BLIP was dropped from the paper to keep one clean upper bound (BLIP retrieval
  numbers archived in RESULTS_natural_retrieval.md).
