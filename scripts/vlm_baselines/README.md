# Frozen-VLM baselines & CLIP reference

Standalone scripts that score off-the-shelf vision-language models with the same
protocols PAL is evaluated with, so they can sit next to PAL in the paper's tables.
They are **not** part of the `src` pipeline and **do not run in the `structure` env**.

## Environments — read this first

| env | torch | timm | open_clip | runs |
|---|---|---|---|---|
| `structure` | 2.1.2 | **0.9.16** | — | the PAL pipeline (`src/`). *Not used here.* |
| **`vlm_eval`** | 2.1.2 | 1.0.28 | 3.3.0 | every script below except `musk_zs.py` |
| **`musk`** | 2.0.1 | 0.9.8 | 2.32.0 | `musk_zs.py` only |

**Why separate envs:** `open_clip` pulls in timm ≥ 1.0, and the PAL pipeline's
`torch.fx` feature extractor only works with timm 0.9.16 — installing open_clip into
`structure` breaks PAL training/eval. MUSK in turn pins torch 2.0.1 / timm 0.9.8, which
neither of the other two can satisfy.

```bash
# vlm_eval  (CLIP / CONCH / QuiltNet / PLIP)
conda env create -f scripts/vlm_baselines/environment_vlm_eval.yml
conda activate vlm_eval

# musk
bash scripts/vlm_baselines/musk_env_setup.sh      # clones lilab-stanford/MUSK, builds `musk`
conda activate musk
```

CONCH (`MahmoodLab/CONCH`) and MUSK (`xiangjx/musk`) weights are gated on Hugging Face —
`huggingface-cli login` before running. Data paths inside the scripts are hardcoded to
`~/STRUCTURE/data/...`; edit the constants at the top of each file.

## Pathology zero-shot classification — NCT-CRC-HE (`RESULTS_crc_zeroshot.md`)

The paper's pathology table. Every model uses CONCH's official prompt ensemble
(per-class synonyms × 22 histopathology templates, prototype = mean over all), which is
also what PAL is scored with (`evaluation.use_synonyms: true`).

| script | env | models | note |
|---|---|---|---|
| `vlm_official.py` | vlm_eval | CLIP (ViT-B/16), QuiltNet, PLIP, CONCH | **preferred** — produced the table's CLIP / QuiltNet / PLIP / CONCH rows |
| `musk_zs.py` | musk | MUSK | produced the MUSK row |
| `conch_zeroshot.py`, `quiltnet_plip_zs.py`, `clip_zs.py` | vlm_eval | one model each | earlier runs with a simple 5-template ensemble; kept for provenance, superseded by `vlm_official.py` |

`prompts/conch_crc100k_prompts.json` — CONCH's official CRC100K prompts (9 classes ×
synonyms, 22 templates); the same list is mirrored in
`src/evaluation/zero_shot_metadata.py` (`DATASETS_TO_SYNONYMS["crc100k"]`) for PAL.

`vlm_official.py` and `clip_zs.py` also carry PCam / SICAP hooks from the same eval session;
those datasets are not part of this repo's benchmark.

## Natural-image CLIP reference (`RESULTS_natural_UNIFIED.md`)

CLIP **ViT-L/14** evaluated under our own pipeline's protocols so it can be quoted as an
upper bound next to PAL (ViT-L). All in `vlm_eval`.

| script | task | results |
|---|---|---|
| `vlm_natural_zs.py` | zero-shot: STL10, CIFAR100, Caltech101, DTD, EuroSAT | `RESULTS_natural_UNIFIED.md` |
| `vlm_natural_retrieval.py` | retrieval: Flickr30k, COCO (CLIP-B/L, BLIP) | `RESULTS_natural_retrieval.md` |
| `maskclip_seg.py` | segmentation: VOC / Pascal Context / ADE20K via the MaskCLIP value-projection trick on our seg protocol | `RESULTS_natural_seg.md` |

Note the two tables use different CLIP checkpoints: **ViT-L/14** for the natural-image
reference (scale-matched to PAL's ViT-L backbone), **ViT-B/16** for the pathology table
(`vlm_official.py`, where CLIP is the generic-VLM baseline).
