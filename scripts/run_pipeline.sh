#!/usr/bin/env bash
# Stage pipeline (goal 3.2): extract -> train [-> eval], chained.
#
#   extract_features.py  encoders -> feature cache (extract_only)
#   train.py    cache only, no encoders (require_cached)
#   rerun_eval  optional; needs the trained checkpoint
#
# Usage:
#   scripts/run_pipeline.sh <config.yaml>
#   CKPT=<ckpt> LABEL=<label> ZS=<csv> RT=<csv> scripts/run_pipeline.sh <config.yaml>
#
# The eval stage is run only when CKPT is set, because rerun_eval.py keys off
# the checkpoint produced by the train run (its path is run-name dependent and
# not known until training finishes).
set -euo pipefail

CONFIG="${1:?usage: run_pipeline.sh <config.yaml>}"

echo "=== [1/3] extract: encoders -> cache ==="
python -m src.extract_features --config_path "$CONFIG"

echo "=== [2/3] train: cache only (require_cached) ==="
python -m src.train --config_path "$CONFIG"

if [[ -n "${CKPT:-}" ]]; then
  echo "=== [3/3] eval: ${CKPT} ==="
  python rerun_eval.py \
    --config_path "$CONFIG" \
    --ckpt "$CKPT" \
    --label "${LABEL:-pipeline}" \
    --zs "${ZS:-}" \
    --rt "${RT:-}"
else
  echo "=== [3/3] eval: skipped (set CKPT=<path> to enable) ==="
fi
