#!/usr/bin/env bash
# Re-run the PAL data-scaling sweep with ONE OS process per sample size.
#
# Why per-process (instead of `train_subset --samples 10000,50000`):
#   train_subset's internal loop over --samples runs all sizes in a SINGLE
#   process. That (a) reuses one wandb run across sizes (base_trainer guards
#   `if wandb.run is None` and the loop never calls wandb.finish()), so every
#   size lands in the same results dir; and (b) does not fully release GPU
#   memory between sizes, which OOM'd the 10000 point on 2026-07-06.
#   A fresh process per size => fresh GPU + a new wandb run each time.
#
# WANDB_NAME=n<size> labels the run so the offline dir is
#   results/alignment-<llm>-<lvm>-n<size>/...   (not the shared "-None").
#
# Usage:
#   bash rerun_sweep.sh                         # defaults below
#   SAMPLES="10000 50000" GPU=1 bash rerun_sweep.sh
#   CONFIG=configs/pal/vitl_roberta/token_k512.yaml bash rerun_sweep.sh
set -u

cd /workspace/PAL

CONFIG="${CONFIG:-configs/pal/vitl_roberta/token_k512.yaml}"
GPU="${GPU:-1}"
SAMPLES="${SAMPLES:-10000 50000}"          # space-separated; ONE process each
LOGDIR="${LOGDIR:-/tmp/claude-0/-workspace-PAL/40fd14c2-e617-4b63-987d-66669eb1238f/scratchpad/sweep_rerun}"

mkdir -p "$LOGDIR"
echo "config=$CONFIG  gpu=$GPU  samples=[$SAMPLES]  logdir=$LOGDIR"

for n in $SAMPLES; do
    log="$LOGDIR/n${n}.log"
    echo "=== [$(date '+%F %T')] START sample n=$n  (GPU $GPU)  -> $log ==="
    WANDB_NAME="n${n}" WANDB_MODE=offline WANDB_SILENT=true \
    CUDA_VISIBLE_DEVICES="$GPU" \
        python -u -m src.training.train_subset \
            --config_path "$CONFIG" \
            --samples "$n" \
            > "$log" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "=== [$(date '+%F %T')] DONE  sample n=$n (rc=0) ==="
    else
        echo "=== [$(date '+%F %T')] FAILED sample n=$n (rc=$rc) — see $log; continuing ==="
    fi
done

echo "=== [$(date '+%F %T')] sweep rerun complete ==="
