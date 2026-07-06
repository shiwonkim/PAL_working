# PAL — Projection-free Anchor Learning

<!-- TODO(paper): title / venue once available. -->

<!-- TODO(paper): badges — arXiv / OpenReview / project page.
[![Paper](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
-->

<!-- TODO(paper): author list. -->

---

## Overview

<!-- TODO(paper): overview paragraph + teaser figure. -->

PAL (Projection-free Anchor Learning) aligns pretrained unimodal foundation
models (vision + text) with **limited paired data** — tens of thousands of pairs
rather than the millions typically used — building on the STRUCTURE
regularization setting.

## Key Contributions

<!-- TODO(paper): fill in from the paper. -->

## Installation

### Docker (recommended)

A prebuilt environment image is on Docker Hub
([`shiwonkim/pal:v1`](https://hub.docker.com/r/shiwonkim/pal)) — Python 3.10 +
PyTorch 2.1.2+cu118 and all dependencies. Code and data are mounted at runtime.

```bash
git clone <this-repo> PAL_working && cd PAL_working
bash docker/launch.sh /path/to/PAL_working /path/to/data
```
See [`docker/README.md`](docker/README.md) for details.

### Local (conda / pip)

Python 3.10 with PyTorch 2.1.2+cu118 (install those first, or just use the Docker
image), then:
```bash
pip install -r docker/requirements.txt
```

## Data

Datasets live under `data/` (gitignored). See **[`docs/data_setup.md`](docs/data_setup.md)**
for the per-dataset layout (COCO / Flickr30k / the zero-shot classification sets).

## Quick Start

The pipeline is two entry points, optionally chained by `run_pipeline.sh`:

```bash
# train: features (cache, or extract on a miss) -> alignment layers -> checkpoint
python -m src.train --config_path configs/pal/vits_minilm/token_k512.yaml

# eval: load a checkpoint -> retrieval + zero-shot classification
python -m src.eval --config_path <config> --ckpt <checkpoint.pth> \
    --zs cifar100,stl10 --rt flickr30,coco_karpathy

# or both, chained (train then eval):
CKPT=<checkpoint.pth> bash run_pipeline.sh configs/pal/vits_minilm/token_k512.yaml
```

## Configuration

YAML configs live in `configs/`, one directory per alignment method:

- `configs/pal/` — **PAL** (the method); `configs/default.yaml` — shared base
- `configs/{linear,mlp,fa,sail,csa,clip}/` — baseline / comparison methods
- `configs/dryrun/` — small smoke configs

## Project Structure

```
PAL_working/
├── configs/                 # YAML configs (one dir per method)
├── docker/                  # environment image (Dockerfile, requirements, launch.sh)
├── docs/                    # data_setup.md + refactor notes
├── src/
│   ├── train.py, eval.py    # entry points
│   ├── models/
│   │   ├── alignment/       # alignment layers (pal, linear, mlp, fa, sail, csa)
│   │   └── backbones/       # frozen feature extractors (LLM / vision loaders)
│   ├── training/            # trainers, loss, optim
│   ├── features/            # FeatureStore (extract-or-load) + FeatureSpec
│   ├── datasets/            # dataset loading
│   ├── evaluation/          # retrieval + zero-shot (+ standalone segmentation CLI)
│   └── utils/
├── run_pipeline.sh          # train -> eval chain
└── data/, results/          # datasets / outputs (gitignored)
```

## Citation

<!-- TODO(paper): PAL citation once available. -->

This work builds on STRUCTURE:
```bibtex
@inproceedings{groger2025structure,
  title={With Limited Data for Multimodal Alignment, Let the {STRUCTURE} Guide You},
  author={Gr{\"o}ger, Fabian and Wen, Shuo and Le, Huyen and Brbic, Maria},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=IkvQqD7hk3}
}
```
