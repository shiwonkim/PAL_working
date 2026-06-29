"""Extraction stage CLI (goal 3.2).

Runs encoders to materialise the feature caches, then stops before training:
``run(..., extract_only=True)``. Pair with ``src/train.py`` (cache-only) and
``rerun_eval.py`` (eval), chained by ``run_pipeline.sh``.

    python -m src.extract_features --config_path <config.yaml>
"""

from src.train_alignment import build_arg_parser, run

if __name__ == "__main__":
    args = build_arg_parser(
        description="Extract feature caches (encoders → cache, no training)."
    ).parse_args()
    run(config_path=args.config_path, wandb_notes=args.wandb_notes, extract_only=True)
