#!/usr/bin/env bash
# Train -> eval, chained.
#
#   src/train.py  features (cache or extract-on-miss) -> train alignment layers
#                 -> save checkpoint. No evaluation.
#   src/eval.py   load that checkpoint -> retrieval + zero-shot evaluation.
#
# Usage:
#   CKPT=<ckpt> run_pipeline.sh <config.yaml>
#   CKPT=<ckpt> LABEL=<label> ZS=<csv> RT=<csv> run_pipeline.sh <config.yaml>
#
# The eval stage needs the checkpoint train just produced; its path is
# run-name dependent and not known until training finishes, so pass it via CKPT.
set -euo pipefail

CONFIG="${1:?usage: run_pipeline.sh <config.yaml>}"

echo "=== [1/2] train: features -> alignment layers -> checkpoint ==="
python -m src.train --config_path "$CONFIG"

if [[ -n "${CKPT:-}" ]]; then
  echo "=== [2/2] eval: ${CKPT} ==="
  python -m src.eval \
    --config_path "$CONFIG" \
    --ckpt "$CKPT" \
    --label "${LABEL:-pipeline}" \
    --zs "${ZS:-}" \
    --rt "${RT:-}"
else
  echo "=== [2/2] eval: skipped (set CKPT=<path> to enable) ==="
fi
