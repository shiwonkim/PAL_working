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

## UNI2-h × PathCap training runs (2026-08-31 / 09-01)

Same recipe as the UNI v1 run above (PathCap-train, full 220,318, layer 23/12, K=512,
pool_temperature 0.05, clip temp 0.05), swapping only the vision encoder — and, for the second
run, the batch size back to UNI v1's 4096. PubMedBERT text cache reused; UNI2-h image cache
extracted once (179.4 GB fp16, `(220318, 265, 1536)`, 2h49m).

| run | encoder | batch | patience | best val | best ep | stopped ep | wall |
|---|---|---|---|---|---|---|---|
| pathcap_n200000 | UNI v1 | 4096 | 200 | (log lost) | 32 | 232 | — |
| vocal-breeze-231 | UNI2-h | 2048 | 200 | 3.1533 | 40 | 240 | 24h15m |
| comic-haze-232 | UNI2-h | **4096** | **40** | 3.1530 | 38 | 78 | 7h49m |

- **Batch size does not matter here**: 2048 vs 4096 best val 3.1533 vs 3.1530 (Δ 0.0003), best
  epoch 40 vs 38. So the earlier UNI v1 (4096) vs UNI2-h (2048) comparison was not confounded by
  batch, and comic-haze-232 gives a like-for-like 4096 pairing with the UNI v1 run.
- **Batch 4096 fits a 45.6 GB card only when alone on it**: 45,121 / 45,634 MiB (98.9%, 513 MiB
  spare). Measured scaling of the alignment activations is linear at 7.0 GB per 1000 samples
  (image CAP 265×1536 + text CAP 128×768 + B×B loss), so 4096 ≈ 28.7 GB activations + ~15 GB
  fixed. Not a safe default on a shared GPU.
- **Patience 40 is plenty**: across five runs the longest gap between successive best-val
  improvements was 6 epochs (3 at batch 4096); best epochs were 23–46. Patience 200 spent
  ~200 idle epochs per run (24h → 7.8h for identical quality). Use 40 going forward.
- Loss curves: UNI2-h × PathCap is the healthiest of our runs — train–val gap 0.35 at best
  (HPA generic 0.27, genbal 0.68, cap1 1.67 with val rising), initial val 4.09 (HPA cap1 7.58).
  Absolute values are not comparable across datasets (PathCap captions 97% unique vs HPA
  generic 3%); the shape is.
- Retrieval numbers for the two UNI2-h checkpoints are **not yet run** (this doc's table above is
  UNI v1). Cache row order equals selection-CSV order (`shuffle=False`, sequential offset fill),
  so an IHC-vs-H&E subset breakdown can be indexed from the same cache with no re-extraction.
