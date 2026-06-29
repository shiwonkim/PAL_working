# `src/` file usage across the extract → train → eval pipeline

> **Paths updated 2026-06-29** for the directory restructure (HEAD `e8d4c38`).
> USED/UNUSED verdicts are from the 2026-06-26 dynamic capture (`9ebf0bb`);
> only paths/names and deletions changed since. Regenerate after large refactors.

Which files the **feature-extraction → alignment-training → evaluation**
pipeline actually loads, and which it does not.

## How this was determined

Not grep guesswork. The four pipeline entry points were **executed** and the
modules they actually imported were captured from `sys.modules` (dynamic ground
truth):

- `extract` / `train` / combined → `python -m src.training.train_alignment` on the COCO
  ViT-S smoke config (`run()` from `src/training/train_alignment.py`).
- `eval` → `rerun_eval.py` with the firefly token checkpoint (zs/rt emptied to
  capture imports without the slow eval loop).

A purely static import-graph BFS is **insufficient** here: it misses (1) the
factory pattern — `src/models/alignment/__init__.py` calls
`initialize_package_factory` which `importlib`-loads every module under
`src/models/alignment/` to run their `@register` decorators — and (2)
`__init__.py` side effects. The dynamic capture catches both.

Entry-point note: `src/extract_features.py`, `src/train.py`, `rerun_eval.py` are
entry points — they load others but are not themselves imported by the run, so
they are listed as USED by definition.

---

## ✅ Used files

### Entry points (CLIs)
| File | Stage | Role |
|---|---|---|
| `src/extract_features.py` | **extract** | `run(extract_only=True)` — encoders → cache, no training |
| `src/train.py` | **train** | `run(require_cached=True)` — cache only, no encoders |
| `src/training/train_alignment.py` | extract+train | shared setup (`run` / `load_dataset`) + combined run |
| `rerun_eval.py` | **eval** | load checkpoint → standalone retrieval + zero-shot |

(`run_pipeline.sh` at the repo root chains these three stages.)

### Core pipeline (all three stages)
| File | How it is used |
|---|---|
| `src/training/trainers/alignment_trainer.py` | the hub: `prepare_features` (data) + `_train_layer_pair` (train) + eval methods |
| `src/training/trainers/base_trainer.py` | Trainer base (device, wandb init, lr finder) |
| `src/features/feature_store.py` | cache path / load (mmap) / extract / dedup — extract·train·eval |
| `src/features/feature_spec.py` | centralises `token_level` policy (suffix / pool / layer) |
| `src/utils/checkpoint.py` | state_dict serialize/load (train saves, eval loads) |

### Alignment layers (factory dynamically registers all of `src/models/alignment/`)
| File | Note |
|---|---|
| `alignment_factory.py`, `base_alignment_layer.py` | factory + base |
| `pal.py` | **PAL layer — the one actually trained** |
| `linear_alignment_layer.py`, `mlp_alignment_layer.py`, `freeze_align.py`, `sail_star_mlp.py`, `cca_class.py` | registered but **alternative** layers — only used when config `alignment_layer_name` is not PAL |

### Loss / evaluation / encoders / alignment-measure
| File | Stage | Role |
|---|---|---|
| `src/training/loss/clip_loss.py`, `siglip_loss.py` | train | CLIP / SigLip loss |
| `src/evaluation/retrieval.py` | eval | retrieval metrics |
| `src/evaluation/zero_shot_classifier.py`, `consts.py` | eval | zero-shot classifier / templates |
| `src/utils/measure_alignment.py` | train | layer-selection score (`compute_score`, mutual_knn) — trimmed to that one function |
| `src/models/encoders/text_models.py` | extract | LLM loader (`load_llm` / `load_tokenizer`) |
| `src/models/encoders/vision_models.py` | extract | vision encoder loader (`load_lvm`) |

### Alternative trainers (config branch; imported on the extract/train path only)
| File | Note |
|---|---|
| `src/training/trainers/clip_eval_trainer.py`, `csa_trainer.py` | only when config `clip:true` / `cca:true`; PAL default is `AlignmentTrainer` |

### Datasets / utils / optim
| File |
|---|
| `src/datasets/`: `data_utils.py` (get_datasets / transforms), `coco_dataset.py`, `flickr30k_dataset.py`, `image_text_dataset.py`, `base_dataset.py` |
| `src/utils/`: `utils.py`, `metrics.py`, `base_factory.py`, `load_modules.py`, `loader.py`, `plotting.py`, `train_utils.py` |
| `src/training/optim/`: `optimizer.py`, `lars.py` |

---

## ❌ Not used by this pipeline (but live for other purposes)

| Class | Files | Actual nature |
|---|---|---|
| **Separate eval entry points** (segmentation — a different task) | `src/evaluation/zero_shot_segmentation.py`, `zero_shot_patch_voting.py` | segmentation eval, separate from retrieval/zero-shot |
| **Separate training entry point** | `src/training/train_subset.py` | standalone subset-training script |
| **Interpretability scripts** | `viz/*.py` | repo-root standalone figure/analysis scripts; import `src` as a library, not loaded by the pipeline |

### Removed during the cleanup (no longer in the tree)
- **Platonic-benchmark legacy** (2026-06-26): old `extract_features.py`,
  `extract_token_features.py`, `src/utils/alignment_utils.py`, `src/models/tasks.py`
  removed; `measure_alignment.py` trimmed to `compute_score`.
- **Dead util**: `src/utils/paths.py` (imported nowhere).
- **Dataset-prep scripts**: `dataset_preparation/prepare_*.py`, `vissl_download.py`
  (one-off downstream-dataset setup).
- **scripts/**: experiment launchers (`vit*/`), `batch2_eval/`, the checkpoint
  migration tooling (`migrate_checkpoints.py`, `verify_migration_roundtrip.py`,
  `verify_cache_suffix.py`, `verify_alignment_checkpoint.py`). `run_pipeline.sh`
  moved to the repo root.

---

## Observations

1. The real pipeline core is **narrow**: the refactored `alignment_trainer` /
   `feature_store` / `feature_spec` / `checkpoint` are the spine.
2. The five non-PAL alignment layers are **"registered but unused"** — the
   factory imports the whole directory, but only `pal.py` is used in forward.
3. The "unused" files are mostly **live for other purposes** (segmentation eval,
   subset training, interpretability viz) — not deletion targets, just "not part
   of this pipeline".
