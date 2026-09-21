# HPA10M PAL: UNI2-h vs UNI v1 (option C whole-core→224, generic caption)

Same recipe/data/eval as the UNI v1 bring-up; only the vision encoder changed (UNI v1 ViT-L →
UNI2-h ViT-H, IHC-exposed in pretraining). Both token-level K=512, PubMedBERT text, generic_caption,
whole-core→224 thumbnails. UNI2-h trained with batch_size 2048 (4096 OOM'd — 265×1536 tokens);
early-stopped @ epoch 246 (run lunar-darkness-224), val CLIP loss ~3.87 (vs UNI v1 ~4.04).

## Val in-domain retrieval (2,000 pairs)

| metric | UNI v1 | UNI2-h | Δ |
|---|---|---|---|
| I2T R@1 | 0.109 | **0.127** | +0.018 |
| I2T R@5 | 0.358 | **0.385** | +0.027 |
| I2T R@10 | 0.496 | **0.530** | +0.034 |
| T2I R@1 | 0.108 | **0.116** | +0.008 |
| T2I R@5 | 0.351 | **0.362** | +0.011 |
| T2I R@10 | 0.496 | **0.520** | +0.024 |

UNI2-h beats UNI v1 on EVERY metric (I2T R@1 +16% relative), consistent with its lower val loss.
Evidence that UNI2-h (IHC-exposed encoder) learns better IHC image-text alignment.

## Caveats (don't over-read the magnitude)
- Improvement is real but modest — throttled by two known bottlenecks:
  1. Option C thumbnails (whole 3000px core → 224) discard cell-level detail, suppressing UNI2-h's
     patch-encoder strength.
  2. generic_caption degeneracy → retrieval R@1 ceiling ≈0.51 (51% unique captions; "no staining"
     negatives share identical text). R@10 0.53 is already near that ceiling.
- UNI2-h's real headroom likely shows with native-res crop (option A/B) + caption_1/2 (99% unique,
  incl. negatives) + an HPA-native zero-shot classification eval (staining intensity/location/tissue).

## Status
Bring-up (UNI v1) ✅ → UNI2-h swap ✅ (improvement confirmed). Next levers are modeling/eval
decisions to make WITH the pathologist: crop mode, caption variant, and the zs-classification
benchmark. Encoder integration + pipeline are done and validated.
