# PAL — refactor working copy

**Read this first.** This repo (`PAL_working`) is a **code-only refactor copy** of the
PAL × STRUCTURE codebase. The goal here is to clean up and restructure the
code — **not** to run experiments. The original repo (`~/STRUCTURE`) keeps all data,
feature caches, checkpoints, and the frozen, revision-ready code.

## What this repo is

- A snapshot of the STRUCTURE codebase (tracked source only), imported for refactoring.
- **"PAL"** (Projection-free Anchor Learning) is the method name, formerly "BridgeAnchors"
  / "BA". The alignment layer is a single `PALAlignmentLayer` (`src/alignment/pal.py`) that
  serves both modes — CLS (2D input → cosine profile) and token (3D input → CAP); which
  mode runs is set by config `token_level`, not the class. A `CLASS_NAME_ALIASES` map in
  `src/utils/checkpoint.py` keeps pre-rename / pre-merge checkpoints loading.
- **No `data/`, `results/`, or checkpoints here.** To actually run training/eval, symlink
  them from the original repo:
  `cd ~/PAL_working && ln -s ~/STRUCTURE/data ~/STRUCTURE/results .`

## Context / reference docs (background, NOT current instructions)

- **`docs/pipeline_overview.md`** (+ `.ko.md`) — **start here**: full training/inference
  architecture map with file:line refs and refactoring observations.
- `docs/STRUCTURE_context.md` — the original repo's old `CLAUDE.md`. Describes the prior
  experiment workflow/state (batches, server A/B, etc.) — **treat as history, not live
  rules**; most of it does not apply to this refactor repo.
- `IMPLEMENTATION.md` — design notes & per-file diffs from the original build.
- `PROJECT_LOG.md` / `EXPERIMENTS.md` — chronological work log & experiment ledger.
- `docs/laion_reimplementation_TODO.md` — deferred LAION memory work (ties into the
  FeatureStore refactor).

## Environment

- Python 3.10, conda env `structure` (PyTorch 2.1.2+cu118, timm 0.9.16,
  transformers 4.45.2). Same env as the original repo.

## Refactor goals (the plan)

1. **Checkpoints → `state_dict`** ✅ *done* — `src/utils/checkpoint.py` (serialize / load
   with legacy + alias handling) and `scripts/migrate_checkpoints.py` convert the old
   pickled-module checkpoints to a self-describing state_dict format.
2. **Rename classes to PAL** ✅ *done* — `BridgeAnchor*` → `PAL*`, `configs/ba` →
   `configs/pal`, with runtime class-name checks and checkpoint `class_name` strings updated.
3. **Extract a FeatureStore** abstraction ✅ *core done; LAION (3.3) deferred* —
   `src/utils/feature_store.py` owns cache path building, extract-or-load (mmap), encoder
   builders, text-mask I/O, and image-dedup; `FeatureSpec.cache_suffix` centralises suffix
   construction. Stage decoupling (extract → train → eval) via `require_cached` + the thin
   CLIs `src/{extract,train}.py` + `scripts/run_pipeline.sh`. **Remaining: 3.3 LAION memory
   reimplementation** (virtual-concat + mmap + buffer-shuffle + prefetch) — see
   `docs/laion_reimplementation_TODO.md`; deferred on the `serverB` branch.
4. **Consolidate the CLS ↔ Token branching** ✅ *done* — `token_level` stays a config flag
   (not derivable from the layer class); its propagation is centralised in
   `src/utils/feature_spec.py` (`FeatureSpec`), and the CLS/token PAL layers are merged into one
   `PALAlignmentLayer` that picks the path by input rank.
5. **Split the oversized `AlignmentTrainer`** ✅ *done* — `fit()` (~840 lines) split into
   `prepare_features` (eager: load / dedup / subsample / layer-select / slice or token-load one
   layer pair → `PreparedFeatures`) and `_train_layer_pair` (build + train + checkpoint + eval),
   with `fit()` a thin orchestrator. Single layer pair only (multi-pair sweeps raise); extraction
   is `prepare_features` with no training.

Already done upstream: `cls_attn_prior` (an unused, never-enabled feature) removed from
the PAL-token layer and the trainer.

## Conventions

- Refactor on branches; keep `main` green.
- **Whenever touching layer classes or checkpoint save/load, verify loadability** —
  rebuild from a checkpoint and confirm forward output matches before/after the change.
- Update the relevant doc when a refactor changes structure (this file,
  `docs/pipeline_overview.md`).
