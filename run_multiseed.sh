#!/usr/bin/env bash
# Multi-seed training sweep — ONE OS process per seed, general `src.train`.
#
# Reusable across methods: run PAL first, then swap CONFIG for each baseline
# (linear / mlp / fa / sail / csa / clip). Nothing here is PAL-specific.
#
# Why one process per seed (not a loop inside one process):
#   A fresh process per seed => fresh GPU + a new wandb run each time, and full
#   GPU/RAM release between seeds (a single long-lived process can hold onto CUDA
#   memory and reuse one wandb run). Same lesson as rerun_sweep.sh.
#
# Seed reaches training via `--seed` -> config["random_state"] (weight init,
# subsample selection, batch shuffle). The checkpoint dir already embeds the seed
# ("(img, txt)_<score>_seed<N>/"), so seeds never overwrite each other on disk;
# WANDB_NAME=<tag>_seed<N> additionally labels each run in the wandb UI and names
# the results dir (results/alignment-<llm>-<lvm>-<tag>_seed<N>/...).
#
# Usage:
#   bash run_multiseed.sh                                  # PAL k512, seeds 42 43 44
#   SEEDS="42 43 44" GPU=1 bash run_multiseed.sh
#   CONFIG=configs/linear/....yaml TAG=linear bash run_multiseed.sh   # next baseline
set -u

cd /workspace/PAL

CONFIG="${CONFIG:-configs/pal/vitl_roberta/token_k512.yaml}"
SEEDS="${SEEDS:-42 43 44}"          # space-separated; ONE process each
GPU="${GPU:-1}"
# Label for wandb run / results dir; defaults to the config filename stem
# (e.g. token_k512). Override TAG to disambiguate baselines with the same stem.
TAG="${TAG:-$(basename "${CONFIG%.yaml}")}"
LOGDIR="${LOGDIR:-/workspace/PAL/logs/multiseed}"   # gitignored

mkdir -p "$LOGDIR"
echo "config=$CONFIG  seeds=[$SEEDS]  gpu=$GPU  tag=$TAG  logdir=$LOGDIR"

for s in $SEEDS; do
    name="${TAG}_seed${s}"
    log="$LOGDIR/${name}.log"
    echo "=== [$(date '+%F %T')] START $name  (GPU $GPU)  -> $log ==="
    WANDB_NAME="$name" WANDB_MODE=offline WANDB_SILENT=true \
    CUDA_VISIBLE_DEVICES="$GPU" \
        python -u -m src.train \
            --config_path "$CONFIG" \
            --seed "$s" \
            > "$log" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "=== [$(date '+%F %T')] DONE  $name (rc=0) ==="
    else
        echo "=== [$(date '+%F %T')] FAILED $name (rc=$rc) — see $log; continuing ==="
    fi
done

echo "=== [$(date '+%F %T')] multi-seed sweep complete for $TAG ==="
