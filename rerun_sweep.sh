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

cd /home/shiwon/PAL_working

# Use the project conda env (torch/loguru/timm live here, not the system python).
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate structure

CONFIG="${CONFIG:-configs/pal/uni_pubmedbert/token_k512.yaml}"
GPU="${GPU:-0}"
SAMPLES="${SAMPLES:-500 1000 5000 10000 20000 40000}"   # space-separated; ONE process each
LOGDIR="${LOGDIR:-$HOME/sweep_rerun_logs}"

mkdir -p "$LOGDIR"
echo "config=$CONFIG  gpu=$GPU  samples=[$SAMPLES]  logdir=$LOGDIR"

for n in $SAMPLES; do
    log="$LOGDIR/${TAG:-}n${n}.log"
    echo "=== [$(date '+%F %T')] START sample n=$n  (GPU $GPU)  -> $log ==="
    WANDB_NAME="${TAG:-}n${n}" WANDB_MODE=offline WANDB_SILENT=true \
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
