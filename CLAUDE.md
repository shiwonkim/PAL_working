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
3. **Extract a FeatureStore** abstraction — cache load / mmap / dedup are currently
   inlined in `AlignmentTrainer.fit()`; this is also the right home for the LAION memory
   reimplementation (virtual-concat + mmap + buffer-shuffle + prefetch).
4. **Consolidate the CLS ↔ Token branching** duplicated across extraction / fit / eval.
5. **Split the oversized `AlignmentTrainer`** (~2900 lines, `fit()` ~800) into focused
   responsibilities.

Already done upstream: `cls_attn_prior` (an unused, never-enabled feature) removed from
the PAL-token layer and the trainer.

## Conventions

- Refactor on branches; keep `main` green.
- **Whenever touching layer classes or checkpoint save/load, verify loadability** —
  rebuild from a checkpoint and confirm forward output matches before/after the change.
- Update the relevant doc when a refactor changes structure (this file,
  `docs/pipeline_overview.md`).
