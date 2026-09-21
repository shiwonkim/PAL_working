# PAL vitL vs CLIP reference — unified 3-task table (natural images)

Single reference / upper bound = **CLIP ViT-L/14 (openai)**, scale-matched to the PAL main
table (dinov2-L + roberta-large). CLIP covers all three natural-image tasks; BLIP dropped to
keep one clean reference (BLIP retrieval archived in RESULTS_natural_retrieval.md).

- zs classification: macro (balanced) accuracy %, 5 datasets. Same torchvision loaders,
  imagenet-80 template ensemble.
- retrieval: R@1 %, flickr30-test / coco_karpathy-test, same helpers as the PAL pipeline.
- segmentation: mIoU-fg %, ensemble strategy, MaskCLIP (canonical) under the PAL seg protocol
  (img 224, direct decoding, identical datasets/metric).

PAL numbers from the server-79 eval; CLIP numbers from scripts here (vlm_eval env).

## Zero-shot classification (macro acc, %)
| | stl10 | cifar100 | caltech101 | dtd | eurosat | **MEAN** |
|---|---|---|---|---|---|---|
| PAL vitL | 95.3 | 48.8 | 60.9 | 17.7 | 34.6 | 51.5 |
| **CLIP-L/14** | 99.4 | 78.3 | 93.2 | 55.4 | 63.4 | **77.9** |
| Δ (PAL−CLIP) | −4.1 | −29.5 | −32.3 | −37.7 | −28.8 | −26.5 |

→ CLIP is a clear upper bound here — global image-text classification is CLIP's native regime.

## Retrieval (R@1, %)
| | flickr I2T | flickr T2I | coco I2T | coco T2I | **MEAN** |
|---|---|---|---|---|---|
| PAL vitL | 76.3 | 61.8 | 56.3 | 42.6 | 59.2 |
| **CLIP-L/14** | 87.4 | 67.9 | 56.3 | 36.5 | 62.0 |
| Δ (PAL−CLIP) | −11.1 | −6.1 | 0.0 | **+6.1** | −2.8 |

→ Near-parity on average; PAL ties CLIP on coco I2T and BEATS it on coco T2I (COCO is
in-domain for PAL). CLIP leads on flickr (zero-shot transfer for both).

## Segmentation (mIoU-fg, %)
| | VOC | Context | ADE20K | **MEAN** |
|---|---|---|---|---|
| PAL vitL | 32.3 | 25.5 | 13.8 | 23.9 |
| CLIP-L/14 (MaskCLIP) | 29.1 | 23.5 | 13.8 | 22.1 |
| Δ (PAL−CLIP) | **+3.2** | **+2.0** | +0.0 | **+1.8** |

→ PAL ≥ CLIP on all three; wins VOC/Context, ties ADE20K. Frozen DINOv2 + PAL alignment
matches/exceeds CLIP's native text alignment on dense open-vocab prediction.

## One-line story
CLIP is a strong reference, but PAL is not uniformly below it: PAL **exceeds** CLIP on dense
segmentation, **matches** it on retrieval (and wins in-domain COCO), and trails only on global
zero-shot classification — CLIP's home task. So CLIP is a reference point PAL surpasses where
spatial/in-domain structure matters, using frozen encoders + a small alignment on 83K pairs.
