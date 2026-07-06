"""Data size sweep: PAL vs best baseline retrieval R@1 across training sizes.

One script, three layouts (``--mode``) over the same source numbers:

- ``combined`` — one panel per dataset, I2T (solid) + T2I (dashed).
- ``split``    — one dataset as two panels: I2T | T2I.
- ``avg``      — one line per dataset, I2T/T2I averaged.

Datasets are selected with ``--datasets`` (default: all). Add a new dataset by
appending one entry to ``DATA`` and it works in every mode.

Usage:
    python src/visualization/plot_data_size_sweep.py --mode combined
    python src/visualization/plot_data_size_sweep.py --mode split --datasets flickr
    python src/visualization/plot_data_size_sweep.py --mode avg --datasets flickr,coco
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ── Data (single source of truth; R@1 %, per training size) ───────────
SIZES = ["1K", "5K", "10K", "50K", "82K"]

DATA = {
    "flickr": {
        "title": "Flickr30k",
        "ba_i2t": [16.9, 35.6, 44.6, 64.6, 69.6],
        "ba_t2i": [13.7, 26.7, 33.9, 50.1, 54.5],
        "fa_i2t": [13.5, 30.1, 37.6, 57.5, 61.3],
        "fa_t2i": [10.3, 23.7, 29.0, 44.5, 47.5],
    },
    "coco": {
        "title": "COCO Karpathy",
        "ba_i2t": [10.4, 24.0, 30.7, 47.2, 51.5],
        "ba_t2i": [8.5, 18.2, 23.2, 34.7, 38.6],
        "fa_i2t": [8.5, 19.6, 24.1, 40.6, 44.2],
        "fa_t2i": [6.9, 16.4, 20.0, 30.2, 32.5],
    },
}

# ── Style ─────────────────────────────────────────────────────────────
C_BA = "#264653"   # dark teal  — PAL (ours)
C_FA = "#E76F51"   # burnt sienna — best baseline
OUT_DIR = "drafts/figures"


def _style_axis(ax, title):
    ax.set_facecolor("white")
    ax.set_xticks(np.arange(len(SIZES)))
    ax.set_xticklabels(SIZES, fontsize=10)
    ax.set_xlabel("Training data size", fontsize=11)
    ax.set_title(title, fontsize=11, pad=6)
    ax.tick_params(axis="y", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 78)


def _legend(fig, entries):
    fig.legend(handles=entries, loc="upper center", bbox_to_anchor=(0.5, 1.12),
               ncol=len(entries), fontsize=9, framealpha=1.0, edgecolor="#cccccc",
               handlelength=1.8, handletextpad=0.5, columnspacing=1.5)


def plot_combined(datasets):
    """One panel per dataset; I2T solid + T2I dashed, PAL vs baseline."""
    x = np.arange(len(SIZES))
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.0 * len(datasets), 3.4),
                             sharey=True, squeeze=False)
    for ax, name in zip(axes[0], datasets):
        d = DATA[name]
        ax.plot(x, d["ba_i2t"], "o-", color=C_BA, linewidth=1.8, markersize=4.5, zorder=3)
        ax.plot(x, d["ba_t2i"], "o--", color=C_BA, linewidth=1.8, markersize=4.5, zorder=3)
        ax.plot(x, d["fa_i2t"], "s-", color=C_FA, linewidth=1.8, markersize=4.5, zorder=3)
        ax.plot(x, d["fa_t2i"], "s--", color=C_FA, linewidth=1.8, markersize=4.5, zorder=3)
        _style_axis(ax, d["title"])
    axes[0][0].set_ylabel("Retrieval R@1 (%)", fontsize=11)
    _legend(fig, [
        Line2D([0], [0], color=C_BA, lw=1.8, marker="o", markersize=4.5, label="PAL (ours)"),
        Line2D([0], [0], color=C_FA, lw=1.8, marker="s", markersize=4.5, label="Best baseline (ret.)"),
        Line2D([0], [0], color="gray", lw=1.2, linestyle="-", label="I2T"),
        Line2D([0], [0], color="gray", lw=1.2, linestyle="--", label="T2I"),
    ])
    return fig, "combined"


def plot_split(datasets):
    """One dataset as I2T | T2I panels (falls back to first dataset if many)."""
    name = datasets[0]
    d = DATA[name]
    x = np.arange(len(SIZES))
    fig, (ax_i, ax_t) = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for ax, ba, fa, sub in [(ax_i, d["ba_i2t"], d["fa_i2t"], "I2T"),
                            (ax_t, d["ba_t2i"], d["fa_t2i"], "T2I")]:
        ax.plot(x, ba, "o-", color=C_BA, linewidth=1.8, markersize=4.5, zorder=3)
        ax.plot(x, fa, "s-", color=C_FA, linewidth=1.8, markersize=4.5, zorder=3)
        _style_axis(ax, f"{d['title']} {sub}")
    ax_i.set_ylabel("Retrieval R@1 (%)", fontsize=11)
    _legend(fig, [
        Line2D([0], [0], color=C_BA, lw=1.8, marker="o", markersize=4.5, label="PAL (ours)"),
        Line2D([0], [0], color=C_FA, lw=1.8, marker="s", markersize=4.5, label="Best baseline (ret.)"),
    ])
    return fig, f"split_{name}"


def plot_avg(datasets):
    """One panel per dataset; I2T/T2I averaged into a single line each."""
    x = np.arange(len(SIZES))
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.0 * len(datasets), 3.4),
                             sharey=True, squeeze=False)
    for ax, name in zip(axes[0], datasets):
        d = DATA[name]
        ba = [(i + t) / 2 for i, t in zip(d["ba_i2t"], d["ba_t2i"])]
        fa = [(i + t) / 2 for i, t in zip(d["fa_i2t"], d["fa_t2i"])]
        ax.plot(x, ba, "o-", color=C_BA, linewidth=1.8, markersize=4.5, zorder=3)
        ax.plot(x, fa, "s-", color=C_FA, linewidth=1.8, markersize=4.5, zorder=3)
        _style_axis(ax, d["title"])
    axes[0][0].set_ylabel("Avg. Retrieval R@1 (%)", fontsize=11)
    _legend(fig, [
        Line2D([0], [0], color=C_BA, lw=1.8, marker="o", markersize=4.5, label="PAL (ours)"),
        Line2D([0], [0], color=C_FA, lw=1.8, marker="s", markersize=4.5, label="Best baseline (ret.)"),
    ])
    return fig, "avg"


MODES = {"combined": plot_combined, "split": plot_split, "avg": plot_avg}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=list(MODES), default="combined")
    p.add_argument("--datasets", default=",".join(DATA),
                   help="comma-separated subset of: " + ", ".join(DATA))
    p.add_argument("--out-dir", default=OUT_DIR)
    args = p.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    unknown = [d for d in datasets if d not in DATA]
    if unknown:
        p.error(f"unknown datasets {unknown}; available: {list(DATA)}")

    fig, tag = MODES[args.mode](datasets)
    fig.tight_layout(pad=0.4)
    fig.subplots_adjust(top=0.85)
    os.makedirs(args.out_dir, exist_ok=True)
    stem = f"{args.out_dir}/retrieval_data_size_sweep_{tag}"
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight", dpi=300,
                    facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved {stem}.{{pdf,png}}  (mode={args.mode}, datasets={datasets})")


if __name__ == "__main__":
    main()
