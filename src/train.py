"""Training stage CLI (goal 3.2).

Reads feature caches only and refuses to run encoders: ``run(...,
require_cached=True)``. A cache miss raises a clear error pointing at the
extraction stage instead of silently spinning up encoders. Pair with
``src/extract_features.py`` (writes caches) and ``rerun_eval.py`` (eval), chained by
``scripts/run_pipeline.sh``.

    python -m src.train --config_path <config.yaml>
"""

from src.train_alignment import build_arg_parser, run

if __name__ == "__main__":
    args = build_arg_parser(
        description="Train alignment from cached features (no encoders)."
    ).parse_args()
    run(config_path=args.config_path, wandb_notes=args.wandb_notes, require_cached=True)
