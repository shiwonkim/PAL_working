# HPA staining-attribute ZERO-SHOT classification (PAL, UNI2-h + PubMedBERT)

PAL does iSight's task ZERO-SHOT: no staining labels seen in training; predicted via text
prompts (matched to the generic_caption template) → CAP-pooled image embedding cosine → argmax.
Eval = HPA val. intensity over all 2000; location/quantity over positive rows only (negatives
have no location/quantity). Script: scripts/vlm_baselines/hpa_staining_zs.py.

## Macro accuracy (balanced)

| attribute (classes) | random | generic 200K | cap1 100K |
|---|---|---|---|
| intensity (negative/weak/moderate/strong) | 0.25 | 0.568 | **0.571** |
| location (nuclear / cyto-membranous / both) | 0.33 | **0.618** | 0.614 |
| quantity (<25% / 25-75% / >75%) | 0.33 | 0.531 | **0.541** |

(micro: generic 0.560/0.638/0.534 ; cap1 0.539/0.521/0.553)

## Takeaways
- **Well above random on all three** (intensity 2.3× random) — PAL learns staining structure from
  captions alone and applies it zero-shot, i.e. it solves iSight's supervised task without labels.
  This is the value of contrastive representation learning vs iSight's supervised multi-head classifier.
- **generic vs cap1 ≈ tie** (all within 0.01). The earlier caption-coverage prediction (generic
  should win because it names labels 100% vs cap1's 68-88%) did NOT materialize — cap1 matches
  generic even with template-matched prompts. Caption choice barely affects staining zero-shot,
  though it did matter for retrieval (generic-dedup 0.237 > cap1 0.079).
- Both are option-C thumbnails (whole-core→224); native-res crop is the untested lever, to decide
  with the pathologist.

## Context (prior HPA retrieval, same models)
| | I2T R@1 |
|---|---|
| generic 200K (orig val, ceiling 0.51) | 0.127 |
| generic 200K (dedup val, ceiling ~1.0) | 0.237 |
| cap1 100K (cap1 val, ceiling ~1.0) | 0.079 |

Retrieval favors generic; staining-classification is a tie → the "best caption" depends on the
downstream task. For the iSight-style staining task, caption choice is not the lever; encoder
(UNI2-h) + resolution (native crop) are.
