# PAL — refactor working copy

**Read this first.** This repo (`PAL_working`) is a **code-only refactor copy** of the
PAL × STRUCTURE codebase. The goal here is to clean up and restructure the
code — **not** to run experiments. The original repo (`~/STRUCTURE`) keeps all data,
feature caches, checkpoints, and the frozen, revision-ready code.

## What this repo is

- A snapshot of the STRUCTURE codebase (tracked source only), imported for refactoring.
- **"PAL"** (Projection-free Anchor Learning) is the method name, formerly "BridgeAnchors"
  / "BA". The alignment layer is a single `PALAlignmentLayer` (`src/models/alignment/pal.py`) that
  serves both modes — CLS (2D input → cosine profile) and token (3D input → CAP); which
  mode runs is set by config `token_level`, not the class. Checkpoints are rebuilt by
  `class_name` through `AlignmentFactory` (`src/utils/checkpoint.py`), so directory moves /
  renames don't break loading.
- **No `data/`, `results/`, or checkpoints here.** To actually run training/eval, symlink
  them from the original repo:
  `cd ~/PAL_working && ln -s ~/STRUCTURE/data ~/STRUCTURE/results .`

## Context / reference docs (historical — refactor notes, NOT current)

Assorted notes kept from the refactor: `docs/pipeline_overview.md` (+ `.ko.md`),
`docs/STRUCTURE_context.md` (the original repo's old `CLAUDE.md`), `IMPLEMENTATION.md`,
`PROJECT_LOG.md`, `EXPERIMENTS.md`, `docs/laion_reimplementation_TODO.md`. Background only —
written during the refactor and **not kept up to date**; verify against the code before
relying on any of them.

## Environment

- Python 3.10, conda env `structure` (PyTorch 2.1.2+cu118, timm 0.9.16,
  transformers 4.45.2). Same env as the original repo.
- **Single-GPU** train/eval; the code just uses `torch.device("cuda")` — there is
  no GPU flag in the configs or CLIs. Pick the GPU at run time with
  `CUDA_VISIBLE_DEVICES`, e.g. `CUDA_VISIBLE_DEVICES=1 python -m src.train
  --config_path <cfg>`; it propagates into `run_pipeline.sh` and `src/eval.py`
  too. In Docker, `--gpus all` exposes the GPUs and `-e CUDA_VISIBLE_DEVICES=1`
  selects one.

## Refactor goals (the plan)

1. **Checkpoints → `state_dict`** ✅ *done* — `src/utils/checkpoint.py` serializes / loads
   the self-describing state_dict format. The legacy pickled-module support, the
   `CLASS_NAME_ALIASES` remapping, and the one-off `migrate_checkpoints.py` converter were
   removed once every in-use checkpoint was migrated; old pickles load with the original
   pre-refactor code.
2. **Rename classes to PAL** ✅ *done* — `BridgeAnchor*` → `PAL*`, `configs/ba` →
   `configs/pal`, with runtime class-name checks and checkpoint `class_name` strings updated.
3. **Extract a FeatureStore** abstraction ✅ *core done; LAION (3.3) deferred* —
   `src/features/feature_store.py` owns cache path building, extract-or-load (mmap), backbone
   builders, text-mask I/O, and image-dedup; `FeatureSpec.cache_suffix` centralises suffix
   construction. The pipeline is split into two thin CLIs: `src/train.py` (extract-or-load →
   train → checkpoint) and `src/eval.py` (load checkpoint → retrieval + zero-shot), chained by
   `run_pipeline.sh`. **Remaining: 3.3 LAION memory reimplementation** (virtual-concat + mmap +
   buffer-shuffle + prefetch) — see `docs/laion_reimplementation_TODO.md`; deferred on the
   `serverB` branch.
4. **Consolidate the CLS ↔ Token branching** ✅ *done* — `token_level` stays a config flag
   (not derivable from the layer class); its propagation is centralised in
   `src/features/feature_spec.py` (`FeatureSpec`), and the CLS/token PAL layers are merged into one
   `PALAlignmentLayer` that picks the path by input rank.
5. **Split the oversized `AlignmentTrainer`** ✅ *done* — `fit()` (~840 lines) split into
   `prepare_features` (eager: load / dedup / subsample / layer-select / slice or token-load one
   layer pair → `PreparedFeatures`) and `_train_layer_pair` (build + train + checkpoint), with
   `fit()` a thin orchestrator. Evaluation is separate (`src/eval.py`); training ends at the saved
   checkpoint. Single layer pair only (multi-pair sweeps raise).

Already done upstream: `cls_attn_prior` (an unused, never-enabled feature) removed from
the PAL-token layer and the trainer.

## Further cleanup (post-goals)

- **`models/encoders` → `models/backbones`** — the folder holds frozen feature-extractor loaders
  (`load_llm` for text, incl. decoder-only LLMs like Llama/Qwen; `load_lvm` for vision), so the
  architecture-neutral "backbones" fits better than "encoders".
- **`models/alignment`** — layer files/classes normalised: `pal.py`/`PALAlignmentLayer`,
  plus `linear.py`, `mlp.py`, `fa.py`/`FAAlignmentLayer`, `sail.py`/`SAILAlignmentLayer`,
  `csa.py`. All register via `AlignmentFactory`; `alignment` names the role, not an architecture.
- **`structure_reg`** — now requires 2D inputs (raises on 3D). Pooling tokens to 2D is each
  layer's `reduce_for_structure_reg` (base raises → token-level structure_reg needs a layer that
  overrides it). The trainer reduces only when structure_reg is active AND the input is 3D; CLS
  (2D) passes through. Same rule in `train()` and `validate()`.
- **`evaluation/`** — `consts.py` → `zero_shot_metadata.py` (zero-shot class names + prompt
  templates; retrieval doesn't use them). Removed the unused `zero_shot_patch_voting.py`;
  `zero_shot_segmentation.py` stays a standalone CLI (its VOC/ADE class constants live next to
  the IoU logic).
- **`src/visualization/`** (old root `viz/`) removed — standalone interpretability / paper-figure
  scripts, never imported by the pipeline; kept separately by the author.

## Conventions

- Commit refactors straight to `main`; keep it green by running the relevant smoke
  before each commit (no long-lived side branches).
- **Whenever touching layer classes or checkpoint save/load, verify loadability** —
  rebuild from a checkpoint and confirm forward output matches before/after the change.
- Update this file when a refactor changes structure.
