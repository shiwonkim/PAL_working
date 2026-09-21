# NCT-CRC-HE zero-shot classification — PAL vs pathology foundation models

The paper's pathology headline. Balanced (macro) accuracy on the **CRC-VAL-HE-7K test
split** (7,180 tiles, 9 tissue classes, different patients from the 100K train split).

| method | model | balanced acc. (%) |
|---|---|---|
| Generic VLM | CLIP | 21.4 |
| Pathology foundation model | QuiltNet | 59.8 |
| Pathology foundation model | PLIP | 67.4 |
| Pathology foundation model | CONCH | 75.6 |
| Pathology foundation model | MUSK | 79.9 |
| **Post-hoc alignment** | **PAL (PathCap 200K)** | **81.9** |

## Protocol (identical for every row)
- Images: `CRC-VAL-HE-7K` via torchvision `ImageFolder`; class order ADI, BACK, DEB, LYM,
  MUC, MUS, NORM, STR, TUM (= `DATASETS_TO_CLASSES["crc100k"]`).
- Prompts: CONCH's official 22 histopathology templates x per-class synonyms
  (`prompts/conch_crc100k_prompts.json`, = `DATASETS_TO_SYNONYMS["crc100k"]`). Text
  prototype per class = mean of the L2-normalised (synonym x template) embeddings.
- Score = cosine to each prototype, argmax; report macro (balanced) accuracy.
- PAL: frozen UNI (ViT-L/16) + PubMedBERT, token-level CAP K=512, trained on the PathCap
  le128 selection (`configs/pal/uni_pubmedbert/token_k512_pathcap.yaml`), evaluated by
  `src/eval.py --zs crc100k --use_synonyms true`.

## Where each number comes from
- CLIP / QuiltNet / PLIP / CONCH: `vlm_official.py` (`vlm_eval` env).
- MUSK: `musk_zs.py` (own `musk` env — torch 2.0.1 / timm 0.9.8 — set up by
  `musk_env_setup.sh` against a clone of the MUSK repo; model `hf_hub:xiangjx/musk`).
- PAL: `src/eval.py` on the `pathcap_n200000` checkpoint (best epoch 32).

Naming note: "CRC100K" and "NCT-CRC-HE" are the same dataset; NCT-CRC-HE-100K is the train
split, CRC-VAL-HE-7K the test split. All numbers above are on the test split.
