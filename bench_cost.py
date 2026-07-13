"""bench_cost.py — isolated cost microbenchmark for alignment layers.

Measures the *marginal* cost of each alignment method (PAL / FA / SAIL / linear /
mlp), holding the shared frozen backbone constant. It builds the REAL layer from
its config (so params/kwargs match training exactly) and runs forward / forward+
backward on synthetic inputs of realistic shape. Because FLOPs, latency and
memory depend only on tensor SHAPES (not values), synthetic random inputs give
the same numbers as real features — this is a controlled per-step measurement,
not a full training run.

What this DOES measure (fair, contention-free, reproducible):
  * params
  * FLOPs per sample                 [CPU or GPU — device-independent]
  * forward / fwd+bwd latency (ms)   [GPU required to be meaningful]
  * peak activation memory (GB)      [GPU required]
plus K-sweep (PAL) and T-sweep (token methods), and the alignment/backbone ratio.

What this does NOT measure:
  * total wall-clock training time to convergence. That depends on #epochs each
    method needs (early stopping) and MUST be run on an empty server, each method
    alone, same hardware — log-based timing is confounded by GPU/RAM/disk
    contention (observed: PAL 100->153 s/it when another job shared the box).

Usage:
  CUDA_VISIBLE_DEVICES=0 python bench_cost.py --device cuda --out cost.csv
  python bench_cost.py --device cpu                 # FLOPs/params only
  CUDA_VISIBLE_DEVICES=0 python bench_cost.py --device cuda \
      --k-sweep 128,256,512,1024 --t-sweep 65,197,257,577
"""

import argparse
import csv
import inspect
import time

import torch
import yaml

# Registers every @AlignmentFactory.register() layer as a side effect of import.
import src.models.alignment  # noqa: F401
from src.models.alignment.alignment_factory import AlignmentFactory
from src.utils.yaml_loader import Loader, merge_dicts

# name -> config path. Each config supplies alignment_layer_name / kwargs /
# token_level so the benched layer is byte-for-byte what training builds.
METHOD_CONFIGS = {
    "linear": "configs/linear/vitl_roberta/linear_d512_struct.yaml",
    "mlp": "configs/mlp/vitl_roberta/mlp_d512_struct.yaml",
    "fa": "configs/fa/vitl_roberta/fa_d512.yaml",
    "sail": "configs/sail/vitl_roberta/sail_star_concat.yaml",
    "pal": "configs/pal/vitl_roberta/token_k512.yaml",
}

# Reference backbone cost for the "alignment / backbone" ratio. DINOv2 ViT-L/14
# @224 is ~80 GFLOPs/image; override with --backbone-gflops if you measure it.
DEFAULT_BACKBONE_GFLOPS = 80.0


def load_method(path):
    """Return (layer_name, kwargs, token_level) from a training config."""
    with open(path) as fh:
        c = yaml.load(fh, Loader=Loader)
    c = merge_dicts(c.get("defaults", {}), c.get("overrides", {}))
    return (
        c["training"]["alignment_layer_name"],
        dict(c["training"].get("alignment_layer_kwargs", {})),
        bool(c["training"].get("token_level", False)),
    )


def build_layer(name, input_dim, kwargs):
    return AlignmentFactory.create(name, input_dim=input_dim, **kwargs)


def make_input(token_level, B, T, D, device, with_mask):
    """Synthetic input matching the trainer's forward contract."""
    if token_level:
        z = torch.randn(B, T, D, device=device)
        mask = None
        if with_mask:
            mask = torch.ones(B, T, dtype=torch.long, device=device)
        return z, mask
    return torch.randn(B, D, device=device), None


def call_layer(layer, z, mask):
    """Call forward, passing mask only if the signature accepts it."""
    if mask is not None and "mask" in inspect.signature(layer.forward).parameters:
        return layer(z, mask=mask)
    return layer(z)


def count_flops(layer, z, mask):
    """Per-call FLOPs via torch's flop counter (matmul counted as M*N*K MACs;
    multiply by 2 for true FLOPs — reported consistently, so ratios are exact)."""
    from torch.utils.flop_counter import FlopCounterMode

    fc = FlopCounterMode(display=False)
    with torch.no_grad(), fc:
        call_layer(layer, z, mask)
    return fc.get_total_flops()


def time_ms(fn, device, warmup, reps):
    """Average wall time (ms) of fn() over reps, after warmup, with proper sync."""
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(reps):
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        return sum(ts) / len(ts)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1e3


def peak_mem_gb(layer, z, mask, train):
    """Peak CUDA memory (GB) for one fwd (or fwd+bwd) step; None on CPU."""
    if z.device.type != "cuda":
        return None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    if train:
        layer.zero_grad(set_to_none=True)
        out = call_layer(layer, z, mask)
        out.float().pow(2).sum().backward()
    else:
        with torch.no_grad():
            call_layer(layer, z, mask)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e9


def bench_one(name, path, D, B, T_img, T_txt, device, warmup, reps, backbone_gflops):
    layer_name, kwargs, token_level = load_method(path)
    rows = []
    # Image path (no mask) and, for token methods, text path (masked).
    modalities = [("image", T_img, False)]
    if token_level:
        modalities.append(("text", T_txt, True))
    for modality, T, with_mask in modalities:
        layer = build_layer(layer_name, D, kwargs).to(device).eval()
        params = sum(p.numel() for p in layer.parameters())
        z, mask = make_input(token_level, B, T, D, device, with_mask)

        flops = count_flops(layer, z, mask) / max(B, 1)  # per sample

        fwd = time_ms(
            lambda: call_layer(layer, z, mask) if True else None, device, warmup, reps
        )
        # fwd counts full batch; report per-sample.
        fwd_ps = fwd / B

        def train_step():
            layer.zero_grad(set_to_none=True)
            out = call_layer(layer, z, mask)
            out.float().pow(2).sum().backward()

        layer.train()
        fb = time_ms(train_step, device, warmup, reps) / B
        layer.eval()

        mem_inf = peak_mem_gb(layer, z, mask, train=False)
        mem_tr = peak_mem_gb(layer, z, mask, train=True)

        rows.append(
            {
                "method": name,
                "layer": layer_name,
                "modality": modality,
                "token_level": token_level,
                "T": T if token_level else 1,
                "params": params,
                "gflops_per_sample": round(flops / 1e9, 4),
                "backbone_gflops": backbone_gflops,
                "align_vs_backbone_%": round(100 * (flops / 1e9) / backbone_gflops, 3),
                "fwd_ms_per_sample": round(fwd_ps, 5),
                "fwdbwd_ms_per_sample": round(fb, 5),
                "peak_mem_inf_gb": None if mem_inf is None else round(mem_inf, 4),
                "peak_mem_train_gb": None if mem_tr is None else round(mem_tr, 4),
            }
        )
    return rows


def bench_k_sweep(D, B, T_img, device, warmup, reps, ks):
    """PAL only: cost vs number of anchors K (image path, T=T_img)."""
    _, base_kwargs, _ = load_method(METHOD_CONFIGS["pal"])
    rows = []
    for k in ks:
        kwargs = dict(base_kwargs)
        kwargs.pop("dim_alignment", None)  # avoid overriding num_anchors
        kwargs["num_anchors"] = k
        layer = build_layer("PALAlignmentLayer", D, kwargs).to(device).eval()
        z, mask = make_input(True, B, T_img, D, device, with_mask=False)
        flops = count_flops(layer, z, mask) / B
        fwd = time_ms(lambda: call_layer(layer, z, mask), device, warmup, reps) / B
        mem = peak_mem_gb(layer, z, mask, train=False)
        rows.append(
            {
                "sweep": "K",
                "K": k,
                "T": T_img,
                "params": sum(p.numel() for p in layer.parameters()),
                "gflops_per_sample": round(flops / 1e9, 4),
                "fwd_ms_per_sample": round(fwd, 5),
                "peak_mem_inf_gb": None if mem is None else round(mem, 4),
            }
        )
    return rows


def bench_t_sweep(D, B, device, warmup, reps, ts, k):
    """PAL only: cost vs token count T (image path, K fixed)."""
    _, base_kwargs, _ = load_method(METHOD_CONFIGS["pal"])
    kwargs = dict(base_kwargs)
    kwargs.pop("dim_alignment", None)
    kwargs["num_anchors"] = k
    rows = []
    for t in ts:
        layer = build_layer("PALAlignmentLayer", D, kwargs).to(device).eval()
        z, mask = make_input(True, B, t, D, device, with_mask=False)
        flops = count_flops(layer, z, mask) / B
        fwd = time_ms(lambda: call_layer(layer, z, mask), device, warmup, reps) / B
        mem = peak_mem_gb(layer, z, mask, train=False)
        rows.append(
            {
                "sweep": "T",
                "K": k,
                "T": t,
                "gflops_per_sample": round(flops / 1e9, 4),
                "fwd_ms_per_sample": round(fwd, 5),
                "peak_mem_inf_gb": None if mem is None else round(mem, 4),
            }
        )
    return rows


def print_table(rows):
    if not rows:
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print(" | ".join(c.ljust(widths[c]) for c in cols))
    print("-+-".join("-" * widths[c] for c in cols))
    for r in rows:
        print(" | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dim", type=int, default=1024, help="feature dim (ViT-L / RoBERTa-large = 1024)")
    ap.add_argument("--batch", type=int, default=64, help="batch for latency/mem (per-sample reported)")
    ap.add_argument("--t-img", type=int, default=257, help="image tokens (ViT-L/14@224 = 257)")
    ap.add_argument("--t-txt", type=int, default=64, help="text tokens (typical caption length)")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--methods", default=",".join(METHOD_CONFIGS))
    ap.add_argument("--k-sweep", default="", help="e.g. 128,256,512,1024 (PAL)")
    ap.add_argument("--t-sweep", default="", help="e.g. 65,197,257,577 (PAL)")
    ap.add_argument("--backbone-gflops", type=float, default=DEFAULT_BACKBONE_GFLOPS)
    ap.add_argument("--out", default="cost.csv")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA unavailable -> falling back to CPU (latency/mem invalid).")
        args.device = "cpu"
    device = torch.device(args.device)
    if device.type == "cpu":
        print("[note] CPU run: params & FLOPs are valid; latency/memory are NOT "
              "representative of deployment. Re-run with --device cuda for those.\n")

    all_rows = []
    print("=== Per-method cost (per sample) ===")
    main_rows = []
    for name in args.methods.split(","):
        name = name.strip()
        if name not in METHOD_CONFIGS:
            print(f"[skip] unknown method: {name}")
            continue
        main_rows += bench_one(
            name, METHOD_CONFIGS[name], args.dim, args.batch,
            args.t_img, args.t_txt, device, args.warmup, args.reps,
            args.backbone_gflops,
        )
    print_table(main_rows)
    all_rows += main_rows

    if args.k_sweep:
        ks = [int(x) for x in args.k_sweep.split(",")]
        print("\n=== PAL K-sweep (image path) ===")
        krows = bench_k_sweep(args.dim, args.batch, args.t_img, device, args.warmup, args.reps, ks)
        print_table(krows)

    if args.t_sweep:
        ts = [int(x) for x in args.t_sweep.split(",")]
        print("\n=== PAL T-sweep (image path, K=512) ===")
        trows = bench_t_sweep(args.dim, args.batch, device, args.warmup, args.reps, ts, 512)
        print_table(trows)

    # CSV: main table (sweeps are printed; re-run with a single sweep to CSV them).
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(main_rows[0].keys()))
        w.writeheader()
        w.writerows(main_rows)
    print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
