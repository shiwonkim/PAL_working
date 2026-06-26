# `src/` file usage across the extract → train → eval pipeline

> **As of 2026-06-26** (HEAD `9ebf0bb`). Regenerate after large refactors.

Which `src/` files the **feature-extraction → alignment-training → evaluation**
pipeline actually loads, and which it does not.

## How this was determined

Not grep guesswork. The four pipeline entry points were **executed** and the
modules they actually imported were captured from `sys.modules` (dynamic ground
truth):

- `extract` / `train` / combined → `python -m src.train_alignment` on the COCO
  ViT-S smoke config (`run()` from `src/train_alignment.py`).
- `eval` → `rerun_eval.py` with the firefly token checkpoint (zs/rt emptied to
  capture imports without the slow eval loop).

A purely static import-graph BFS is **insufficient** here: it misses (1) the
factory pattern — `src/alignment/__init__.py` calls `initialize_package_factory`
which `importlib`-loads every module under `src/alignment/` to run their
`@register` decorators — and (2) `__init__.py` side effects. The dynamic capture
catches both.

Entry-point note: `src/extract.py`, `src/train.py`, `rerun_eval.py` are entry
points — they load others but are not themselves imported by the run, so they
are listed as USED by definition.

---

## ✅ Used files (54 incl. 12 `__init__.py`; tables omit bare `__init__.py`)

### Entry points (CLIs)
| File | Stage | Role |
|---|---|---|
| `src/extract.py` | **extract** | `run(extract_only=True)` — encoders → cache, no training |
| `src/train.py` | **train** | `run(require_cached=True)` — cache only, no encoders |
| `src/train_alignment.py` | extract+train | shared setup (`run` / `load_dataset`) + combined run |
| `rerun_eval.py` | **eval** | load checkpoint → standalone retrieval + zero-shot |

### Core pipeline (all three stages)
| File | How it is used |
|---|---|
| `src/trainers/alignment_trainer.py` | the hub: `prepare_features` (data) + `_train_layer_pair` (train) + eval methods |
| `src/trainers/base_trainer.py` | Trainer base (device, wandb init, lr finder) |
| `src/utils/feature_store.py` | cache path / load (mmap) / extract / dedup — extract·train·eval |
| `src/utils/feature_spec.py` | centralises `token_level` policy (suffix / pool / layer) |
| `src/utils/checkpoint.py` | state_dict serialize/load (train saves, eval loads) |

### Alignment layers (factory dynamically registers all of `src/alignment/`)
| File | Note |
|---|---|
| `alignment_factory.py`, `base_alignment_layer.py` | factory + base |
| `pal.py` | **PAL layer — the one actually trained** |
| `linear_alignment_layer.py`, `mlp_alignment_layer.py`, `freeze_align.py`, `sail_star_mlp.py`, `cca_class.py` | registered but **alternative** layers — only used when config `alignment_layer_name` is not PAL |

### Loss / evaluation / models / alignment-measure
| File | Stage | Role |
|---|---|---|
| `src/loss/clip_loss.py`, `siglip_loss.py` | train | CLIP / SigLip loss |
| `src/evaluation/retrieval.py` | eval | retrieval metrics |
| `src/evaluation/zero_shot_classifier.py`, `consts.py` | eval | zero-shot classifier / templates |
| `src/measure_alignment.py` | train | layer-selection score (`compute_score`, mutual_knn) — trimmed to that one function |
| `src/models/text/models.py` | extract | LLM loader (`load_llm` / `load_tokenizer`) |

### Alternative trainers (config branch; imported on the extract/train path only)
| File | Note |
|---|---|
| `src/trainers/clip_eval_trainer.py`, `csa_trainer.py` | only when config `clip:true` / `cca:true`; PAL default is `AlignmentTrainer` |

### Data / utils / core
| File |
|---|
| `src/dataset_preparation/data_utils.py` (get_datasets / transforms) |
| `src/utils/`: `utils.py`, `metrics.py`, `base_factory.py`, `load_modules.py` |
| `src/core/src/datasets/`: `coco_dataset.py`, `flickr30k_dataset.py`, `image_text_dataset.py`, `base_dataset.py` |
| `src/core/src/optimizers/`: `lars.py`, `utils.py` · `src/core/src/utils/`: `loader.py`, `plotting.py`, `utils.py` |

---

## ❌ Not used by this pipeline

| Class | Files | Actual nature |
|---|---|---|
| **Dataset-prep scripts** (one-off, outside the pipeline) | `dataset_preparation/prepare_{aircraft,birdsnap,clevr,k700,kitti,memes,pets,resisc45,ucf101}.py`, `vissl_download.py` | one-off per-downstream-dataset preprocessing, run by hand when building eval data |
| **Separate eval entry points** (segmentation — a different task) | `evaluation/zero_shot_segmentation.py`, `zero_shot_patch_voting.py` | segmentation eval used by `scripts/batch2_eval/run_segmentation.sh`, separate from retrieval/zero-shot |
| **Separate training entry point** | `train_subset.py` | standalone subset-training script |
| **Dead util** | `src/utils/paths.py` | imported nowhere (static and dynamic) |

### Deleted 2026-06-26 (Platonic-benchmark legacy, never used by PAL)
`extract_features.py`, `extract_token_features.py`, `src/utils/alignment_utils.py`,
`src/models/tasks.py` were removed, and `src/measure_alignment.py` trimmed to
`compute_score`. They were the multi-model "Platonic Representation" extraction +
alignment-benchmark path (model zoo via `get_models`, ViT+conv), which the PAL
single-encoder-pair workflow never ran. `extract_token_features.py` was a thin
wrapper superseded by `extract.py` (= `prepare_features`). The live
`compute_score` (layer selection) is retained.

---

## Observations

1. The real pipeline core is **narrow**: ~42 substantive files (of 72), and half
   of those are core/utils. The refactored `alignment_trainer` / `feature_store`
   / `feature_spec` / `checkpoint` are the spine.
2. The five non-PAL alignment layers are **"registered but unused"** — the
   factory imports the whole directory, but only `pal.py` is used in forward.
3. The one clear **dead-code** candidate is `src/utils/paths.py` (reached by
   nothing, static or dynamic).
4. The other "unused" files are mostly **live for other purposes** (data prep,
   segmentation eval, legacy extractors, subset training) — not deletion targets,
   just "not part of this pipeline".
