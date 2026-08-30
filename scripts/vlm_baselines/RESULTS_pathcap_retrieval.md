# PathCap-val IN-DOMAIN retrieval: PAL vs pathology VLMs

Set: PathCap-val le128 (1,500 images, 1 caption each → clean 1:1). Identical protocol for all
(pipeline `retrieval_metrics_df` / `text_to_image_retrieval_metrics`). PAL = UNI+PubMedBERT
token-level K=512, trained on PathCap-train (n=200K) → PathCap-val is in-domain for PAL.

## R@1 / R@5 / R@10

| model | train data | I2T R@1/5/10 | T2I R@1/5/10 |
|---|---|---|---|
| **PAL** (n200K) | PathCap-train (in-domain) | 0.266 / 0.495 / 0.592 | 0.256 / 0.487 / 0.599 |
| **CONCH** | 1.17M path. figure-caption | **0.370 / 0.536 / 0.616** | **0.425 / 0.631 / 0.708** |
| QuiltNet | Quilt-1M | 0.030 / 0.101 / 0.156 | 0.023 / 0.085 / 0.131 |
| PLIP | OpenPath | 0.028 / 0.081 / 0.130 | 0.021 / 0.071 / 0.110 |

## Reading

- **CONCH > PAL**, even on PathCap-val. The "in-domain for PAL only" framing does NOT hold:
  CONCH was trained on 1.17M pathology **figure-caption** pairs from educational/PubMed sources
  — the SAME distribution as PathCap (PubMed/textbook figures). So PathCap is quasi-in-domain for
  CONCH too, and retrieval (global image-text matching) is exactly what a fully-contrastive VLM is
  built for. PAL (frozen UNI + frozen PubMedBERT + small alignment) is not optimized for it.
- **PAL ≈ 10× QuiltNet / PLIP.** Those transfer poorly to PathCap figure-captions (Quilt/OpenPath
  are a different pathology distribution). PAL's frozen-encoder alignment generalizes far better.

## Takeaway (strategy)

Retrieval is NOT PAL's headline — CONCH edges it. This confirms the earlier decision to report
**CRC zero-shot classification** as the pathology headline (PAL 0.819 vs CONCH 0.756 with
22-template + synonym macro acc), not retrieval. PAL's strengths: data efficiency, strong
zero-shot classification, dense grounding (localization), and working atop ANY frozen encoder —
not global contrastive retrieval, where a purpose-built VLM like CONCH wins.
