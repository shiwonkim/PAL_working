# VLM baseline eval scripts (pathology rebuttal)

Standalone scripts to score / zero-shot-eval frozen pathology VLMs (CONCH,
QuiltNet, PLIP) and natural-image CLIP, for comparison against PAL. They are
**not** part of the `src` pipeline and run in the **`vlm_eval` conda env**
(open_clip / conch / transformers — NOT the `structure` env, which must keep
timm 0.9.16). Data paths are hardcoded to `~/STRUCTURE/data/...`.

## Zero-shot classification
- `vlm_official.py` — **preferred.** Re-evals all 4 models on CRC100K + SICAP
  with the **official prompt ensemble** (per-class synonyms x 22 templates;
  prototype = mean over all). CRC prompts = CONCH's `prompts/crc100k_prompts_all_per_class.json`;
  SICAP prompts = same 22 templates + CONCH-style Gleason synonyms (`prompts/sicap_prompts.json`).
- `conch_zeroshot.py` / `quiltnet_plip_zs.py` / `clip_zs.py` — earlier per-model
  runs with a simple 5-template H&E ensemble (`DATASETS_TO_TEMPLATES`). Kept for
  provenance; superseded by `vlm_official.py`.
- `conch_sicap.py` / `quiltnet_plip_sicap.py` — SICAP-only variants of the above.

## In-domain retrieval (Quilt-1M val)
- `vlm_retrieval.py` (CONCH, QuiltNet), `plip_retrieval.py` (PLIP; CLIP 77-token cap).

## CONCH-score data selection (Quilt-1M)
- `conch_score_pool.py` — CONCH image-text cosine over a metadata pool selection.
- `conch_score_alltrain.py` — CONCH-score the **entire** Quilt train split (image
  dedup), writes `conch_scores_alltrain.csv` (used to build the top-N selections).
- `build_top100k.py` — take top-100K by CONCH score -> shuffled training selection.

## prompts/
- `conch_crc100k_prompts.json` — CONCH official CRC100K prompts (9 classes x synonyms, 22 templates).
- `sicap_prompts.json` — SICAP Gleason synonyms (same 22 templates), constructed CONCH-style.
