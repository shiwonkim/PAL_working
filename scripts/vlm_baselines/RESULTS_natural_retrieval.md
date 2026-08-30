# Natural-image retrieval UPPER BOUNDS (CLIP / BLIP)

Reviewer request: web-scale VLMs (CLIP/BLIP) as upper bounds on the natural-image
retrieval track. Same protocol as PAL `evaluate_retrieval` (per-caption rows,
drop_duplicates=false, I2T groups by image, T2I unique-image gallery). Helpers copied
verbatim from `src/evaluation/retrieval.py`. BLIP = **ITC head only** (no ITM rerank) →
same dual-encoder condition as CLIP/PAL. Env: `vlm_eval`. Sets: flickr30-test (1000 img /
5000 cap), coco_karpathy-test (5000 img / 25010 cap).

Reproduce:
    CUDA_VISIBLE_DEVICES=<gpu> python vlm_natural_retrieval.py both   # CLIP + BLIP-coco
    # BLIP-flickr (for coco zero-shot transfer): run_blip(dfs,'Salesforce/blip-itm-base-flickr',...)
CLIP uses force_quick_gelu=True (openai weights are QuickGELU; matches open_clip benchmark).

## Scale-matched to PAL main table

PAL cells: vitL = dinov2-L (main table), vitB = dinov2-B. CLIP/BLIP upper bounds are
matched by VISION scale: ViT-L/14 + BLIP-large for the vitL main table, ViT-B/16 +
BLIP-base for vitB.

### Retrieval (I2T R@1 / T2I R@1), zero-shot condition
| upper bound | scale | flickr30 | coco_karpathy |
|---|---|---|---|
| CLIP ViT-B/16 | B | 0.850 / 0.653 | 0.525 / 0.331 |
| CLIP ViT-L/14 | L | 0.874 / 0.679 | 0.563 / 0.365 |
| BLIP-base (zs transfer) | B | 0.919 / 0.815 | 0.629 / 0.462 |
| BLIP-large (zs transfer) | L | 0.942 / 0.851 | 0.702 / 0.552 |

BLIP zero-shot transfer = cross-checkpoint (flickr from *-coco model, coco from *-flickr
model), so neither cell is in-domain — directly comparable to CLIP zero-shot.

### Zero-shot classification (macro / balanced accuracy)
| upper bound | scale | stl10 | cifar100 | caltech101 | dtd | eurosat | MEAN |
|---|---|---|---|---|---|---|---|
| CLIP ViT-L/14 | L | 0.994 | 0.783 | 0.932 | 0.554 | 0.634 | **0.779** |
| CLIP ViT-B/16 | B | 0.983 | 0.682 | 0.897 | 0.447 | 0.545 | **0.711** |

BLIP has NO classification (original paper reports retrieval/captioning/VQA only) → CLIP-only.
Same loaders as the pipeline (torchvision 0.16.2, seed=42 subsets), verified label-aligned.

### Segmentation (MaskCLIP, canonical) — running, see RESULTS_natural_seg.md

## Full retrieval numbers (R@1 / R@5 / R@10)

| model | condition | dataset | I2T R@1/5/10 | T2I R@1/5/10 |
|---|---|---|---|---|
| CLIP ViT-B/16 | zero-shot | flickr30 | 0.850 / 0.972 / 0.987 | 0.653 / 0.878 / 0.928 |
| CLIP ViT-B/16 | zero-shot | coco_karpathy | 0.525 / 0.767 / 0.847 | 0.331 / 0.584 / 0.690 |
| CLIP ViT-L/14 | zero-shot | flickr30 | 0.874 / 0.983 / 0.998 | 0.679 / 0.898 / 0.942 |
| CLIP ViT-L/14 | zero-shot | coco_karpathy | 0.563 / 0.794 / 0.866 | 0.365 / 0.611 / 0.711 |
| BLIP-base (coco-ft) | zs transfer | flickr30 | 0.919 / 0.986 / 0.998 | 0.815 / 0.956 / 0.979 |
| BLIP-base (coco-ft) | in-domain | coco_karpathy | 0.736 / 0.921 / 0.963 | 0.569 / 0.819 / 0.889 |
| BLIP-base (flickr-ft) | in-domain | flickr30 | 0.941 / 0.999 / 1.000 | 0.861 / 0.977 / 0.990 |
| BLIP-base (flickr-ft) | zs transfer | coco_karpathy | 0.629 / 0.852 / 0.918 | 0.462 / 0.731 / 0.825 |
| BLIP-large (coco-ft) | zs transfer | flickr30 | 0.942 / 0.996 / 0.999 | 0.851 / 0.967 / 0.981 |
| BLIP-large (coco-ft) | in-domain | coco_karpathy | 0.765 / 0.934 / 0.968 | 0.596 / 0.830 / 0.898 |
| BLIP-large (flickr-ft) | in-domain | flickr30 | 0.949 / 0.995 / 0.999 | 0.876 / 0.977 / 0.989 |
| BLIP-large (flickr-ft) | zs transfer | coco_karpathy | 0.702 / 0.896 / 0.945 | 0.552 / 0.802 / 0.876 |

## Paper rows

**BLIP zero-shot transfer** (neither cell in-domain — cross-checkpoint, comparable to CLIP):
- flickr30 from coco-ft model: I2T 0.919 / T2I 0.815
- coco_karpathy from flickr-ft model: I2T 0.629 / T2I 0.462

**BLIP in-domain finetuned** (higher ceiling, appendix):
- flickr30 from flickr-ft: I2T 0.941 / T2I 0.861
- coco_karpathy from coco-ft: I2T 0.736 / T2I 0.569

Notes for the caption:
- BLIP's only public ITR checkpoints are retrieval-finetuned (coco / flickr); a
  pretrain-only "pure zero-shot" BLIP retrieval checkpoint is not published, hence the
  cross-transfer construction for the zero-shot row.
- BLIP is ITC-only here (no ITM cross-attention rerank), matching CLIP/PAL dual-encoder.
- BLIP's original paper (Li et al., ICML 2022) reports NO zero-shot image classification,
  so BLIP appears only in the retrieval upper-bound block; the classification ceiling is CLIP.
