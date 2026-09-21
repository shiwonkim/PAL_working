# HPA10M PAL bring-up (UNI v1 + PubMedBERT, option C whole-core→224)

First end-to-end run validating the HPA data + PAL pipeline BEFORE swapping to UNI2-h.

- Data: 199,924 train / 2,000 val IHC pairs, extracted from the 200 downloaded WebDataset shards
  by `scripts/prepare_hpa10m.py --mode whole --size 256`; caption = generic_caption.
- Encoder: UNI v1 (ViT-L/16, 197 tokens = CLS+196, NO register tokens) + PubMedBERT (≤41 tokens).
- PAL: token-level, K=512, temp 0.05. Trained ~6h, early-stopped @ epoch 253 (best ~53),
  val CLIP loss ~4.04 (< random ln4096≈8.3 → learned alignment). Checkpoint: giddy-deluge-222.
- Feature caches keyed by dataset name (`...-hpa10m-...`) → PathCap artifacts untouched.

## Val in-domain retrieval (2,000 pairs)

| | R@1 | R@5 | R@10 |
|---|---|---|---|
| I2T | 0.109 | 0.358 | 0.496 |
| T2I | 0.108 | 0.351 | 0.496 |

~200× random (1/2000). BUT retrieval is a weak metric for HPA:

## Caption degeneracy (why R@1 looks low)
- 2,000 val captions → only **1,025 unique (51%)**, ~2 images/caption.
- Worst offenders are IDENTICAL "no staining" (negative) captions shared across many genes
  (e.g. lymphoma "no staining in tumor cells" × 30).
- **Theoretical R@1 ceiling ≈ 0.512** (identical captions are indistinguishable). Our 0.109 is
  ~21% of the achievable ceiling — real signal, understated by caption ambiguity.

## Takeaways
- Bring-up SUCCESS: full pipeline works, model learned meaningful IHC image-text alignment.
- Retrieval is NOT the right HPA eval (caption degeneracy). Use HPA-native classification
  (staining intensity / location / quantity / tissue) — pathologist to design.
- Next levers: (1) UNI2-h encoder (IHC-exposed); (2) native-res crop (option A/B) for cell
  detail; (3) caption choice (pathologist); (4) balance/drop the degenerate "no staining" negatives.
