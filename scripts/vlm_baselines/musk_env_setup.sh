#!/usr/bin/env bash
# conda env for musk_zs.py (MUSK, Nature 2025 — lilab-stanford/MUSK).
#
# MUSK pins torch 2.0.1 / timm 0.9.8, which is incompatible with both the PAL
# `structure` env (timm 0.9.16) and `vlm_eval` (timm 1.0.28), so it gets its own env.
# The MUSK repo is installed editable from a clone; MUSK_DIR defaults to ~/MUSK.
# Weights (hf_hub:xiangjx/musk) are gated on HF: `huggingface-cli login` first.
#
#   bash scripts/vlm_baselines/musk_env_setup.sh
#   conda activate musk
#   python scripts/vlm_baselines/musk_zs.py
set -euo pipefail
MUSK_DIR="${MUSK_DIR:-$HOME/MUSK}"
MUSK_COMMIT="714b666"   # commit the paper's MUSK number was produced with

source "$(conda info --base)/etc/profile.d/conda.sh"
if [ ! -d "$MUSK_DIR" ]; then
  git clone https://github.com/lilab-stanford/MUSK "$MUSK_DIR"
fi
git -C "$MUSK_DIR" checkout -q "$MUSK_COMMIT"

conda create -n musk python=3.10 -y 2>&1 | tail -2
conda activate musk
cd "$MUSK_DIR"
pip install -q -r requirements.txt 2>&1 | tail -5     # brings torch 2.0.1 / timm 0.9.8
pip install -q -e . 2>&1 | tail -3
pip install -q scikit-learn==1.7.2 sentencepiece==0.2.2 transformers==4.45.2
python -c "import torch,timm; print('torch',torch.__version__,'timm',timm.__version__)"
echo "MUSK_ENV_DONE"
