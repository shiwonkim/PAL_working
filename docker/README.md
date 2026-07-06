# PAL — Docker Environment

Environment-only image for the **PAL** (Projection-free Anchor Learning) project.
Ships Python 3.10, PyTorch 2.1.2+cu118, and the pipeline dependencies. The image
contains **no code or data** — both are mounted at runtime via volumes.

## Quick setup (new server)

```bash
bash docker/setup_server.sh /path/to/PAL_working /path/to/data
```
Pulls the image, runs a GPU sanity check, and drops you into a container shell.

## Build (from source)

```bash
cd docker && docker build -f Dockerfile -t pal:v1 .
```

## Run

### docker compose
```bash
cd docker && docker compose run --rm pal
```
If `data/` and `results/` live **outside** the repo root, edit `docker-compose.yml`
first (uncomment the explicit `data`/`results` mounts).

### docker run (manual)
```bash
docker run --gpus all -it --shm-size=16g \
  -v /path/to/PAL_working:/workspace/PAL \
  -v /path/to/data:/workspace/PAL/data \
  -e PYTHONPATH=/workspace/PAL \
  -w /workspace/PAL \
  pal:v1 bash
```

## Notes

- Base image `pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel`. CUDA 11.8 is
  forward-compatible with CUDA 12.x host drivers.
- Model weights (DINOv2, RoBERTa, …) are **not** baked into the image — they are
  downloaded on the first feature-extraction run and then cached to disk.
- `requirements.txt` pins only the **direct** dependencies; pip resolves the
  transitive ones. Visualization tools (matplotlib / seaborn / umap-learn) are
  kept for the interpretability scripts.
- **wandb**: the image ships `wandb>=0.22.3`, required for the current 86-char
  API keys (older wandb rejects them with a 40-character error). Set
  `WANDB_API_KEY` in the environment before running.
- Claude Code is pre-installed for interactive development.
