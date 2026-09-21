# Frozen-VLM baselines & CLIP reference

Standalone scripts that score off-the-shelf vision-language models with the same
protocols PAL is evaluated with, so they can sit next to PAL in the paper's tables.
They are **not** part of the `src` pipeline; the frozen-VLM ones **do not run in the
`structure` env** (the env table says which env each script needs).

## Environments — read this first

| env | torch | timm | open_clip | runs |
|---|---|---|---|---|
| `structure` | 2.1.2 | **0.9.16** | — | the PAL pipeline (`src/`), and the HPA10M scripts below that drive it (`hpa_*.py`) |
| **`vlm_eval`** | 2.1.2 | 1.0.28 | 3.3.0 | every frozen-VLM script (CLIP / CONCH / QuiltNet / PLIP / KEEP) |
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


---

# Follow-up (pathology branch) — beyond the paper

Everything below is post-NeurIPS follow-up work and lives only on the pathology branch.

## Additional CRC / SICAP baselines (`vlm_eval`)
| script | what |
|---|---|
| `keep_crc_zs.py` | KEEP (Astaxanthin/KEEP, 2026) on CRC100K, same 22-template + synonym protocol. `RESULTS_keep_crc.md` (KEEP 87.8 > PAL 81.9 > MUSK 79.9 > CONCH 75.6). Needs a one-line `**kwargs` patch to KEEP's cached `modeling_keep.py` under timm 1.0.28 — see the results note. |
| `conch_sicap.py`, `quiltnet_plip_sicap.py` | SICAPv2 Gleason zero-shot; `prompts/sicap_prompts.json` (CONCH-style synonyms, 22 templates). `vlm_official.py` also covers SICAP. |

## Quilt-1M selection & retrieval (`vlm_eval`)
| script | what |
|---|---|
| `conch_score_pool.py`, `conch_score_alltrain.py` | CONCH image-text cosine over a selection / the whole Quilt train split (image-deduped) → `conch_scores_alltrain.csv`. |
| `build_top100k.py` | top-100K by CONCH score → shuffled training selection (`token_k512_top100k.yaml`). |
| `vlm_retrieval.py` (CONCH, QuiltNet), `plip_retrieval.py` (PLIP, 77-token cap) | in-domain retrieval on the Quilt-1M val selection. |

## PathCap retrieval & grounding
| script | env | what |
|---|---|---|
| `vlm_pathcap_retrieval.py` | vlm_eval | CONCH / QuiltNet / PLIP on PathCap-val (1,500, 1:1). `RESULTS_pathcap_retrieval.md` — also holds the UNI2-h × PathCap training runs (batch 2048 vs 4096, patience-40 justification). |
| `../pathcap_localization.py`, `../pathcap_localization_panel.py` | structure | anchor-mediated CAP grounding heatmaps for CRC classes / captions; per-class best-margin tile panels. |

## HPA10M (IHC TMA cores) — `structure` env
| script | what |
|---|---|
| `download_hpa_train.py`, `../prepare_hpa10m.py` | fetch WebDataset shards; extract to flat `images_<mode><size>/` + selection CSVs carrying all three caption variants and the staining / tissue metadata (`--mode whole|crop --size 256`). |
| `hpa_staining_zs.py` | zero-shot staining intensity / location / quantity from template prompts (iSight's task, no labels). `RESULTS_hpa_staining_zs.md` |
| `hpa_tissue_zs.py` | 65-way tissue type + derived binary malignancy, zero-shot. |
| `hpa_staining_linearprobe.py` | vision-only 10K-label probe over the 5 tasks for UNI2-raw (CLS), UNI2 mean-patch (pooling control) and each PAL checkpoint. `RESULTS_hpa_staining_probe.md` is the 5-task master table vs iSight / CONCH-FT / PLIP-FT. |
| `RESULTS_hpa10m_bringup.md`, `RESULTS_hpa10m_uni2.md` | UNI v1 bring-up and the UNI v1 → UNI2-h swap. |
| `RESULTS_caption_comparison.md` + `fig_caption_comparison.png` | COCO vs PathCap vs HPA10M image–caption relationship: what the caption describes and whether it is visible at the fed resolution (why native-res crops are the next lever). |
