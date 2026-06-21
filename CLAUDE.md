# PAL — refactor working copy

**Read this first.** This repo (`PAL_working`) is a **code-only refactor copy** of the
BridgeAnchors × STRUCTURE codebase. The goal here is to clean up and restructure the
code — **not** to run experiments. The original repo (`~/STRUCTURE`) keeps all data,
feature caches, checkpoints, and the frozen, revision-ready code.

## What this repo is

- A snapshot of the STRUCTURE codebase (tracked source only), imported for refactoring.
- **"PAL"** is the method name (formerly "BridgeAnchors" / "BA"). The code still uses the
  old `BridgeAnchor*` class names — renaming them to PAL is part of the refactor.
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

1. **Checkpoints → `state_dict`** instead of pickled module objects, so class
   renames/restructuring don't break loading. Provide a migration path for the existing
   pickled checkpoints in `~/STRUCTURE` (they store whole modules via
   `torch.save(model)` / `torch.load(weights_only=False)`).
2. **Rename classes to PAL** (e.g. `BridgeAnchorTokenAlignmentLayer` → a PAL/CAP name).
3. **Extract a FeatureStore** abstraction — cache load / mmap / dedup are currently
   inlined in `AlignmentTrainer.fit()`; this is also the right home for the LAION memory
   reimplementation (virtual-concat + mmap + buffer-shuffle + prefetch).
4. **Consolidate the CLS ↔ Token branching** duplicated across extraction / fit / eval.
5. **Split the oversized `AlignmentTrainer`** (~2900 lines, `fit()` ~800) into focused
   responsibilities.

Already done upstream: `cls_attn_prior` (an unused, never-enabled feature) removed from
the BA-token layer and the trainer.

## Conventions

- Refactor on branches; keep `main` green.
- **Whenever touching layer classes or checkpoint save/load, verify loadability** —
  rebuild from a checkpoint and confirm forward output matches before/after the change.
- Update the relevant doc when a refactor changes structure (this file,
  `docs/pipeline_overview.md`).
