#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# LiteJam ablation comparison - Version B
# - No title
# - Legend INSIDE top-right without overlapping bars
# - Matplotlib only (no seaborn), single-axes figure
# - Times New Roman (fallbacks if unavailable)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def main():
    # Headless backends (uncomment if running on servers)
    # matplotlib.use("Agg")

    # ---------- Data ----------
    variants = [
        "No Phase/Frequency Feature",
        "No Dynamic Sparse Attention",
        "No Hierarchical Heads",
        "Full LiteJam",
    ]
    metrics = ["det_f1", "type_f1", "area_f1", "snr_r2", "bwd_r2"]

    data = {
        "No Phase/Frequency Feature":   [0.9887, 0.8129, 0.8053, 0.8056, 0.9698],
        "No Dynamic Sparse Attention":  [0.9931, 0.8730, 0.8274, 0.8041, 0.9512],
        "No Hierarchical Heads":        [0.9968, 0.9415, 0.8953, 0.8908, 0.9743],
        "Full LiteJam":                 [1.0000, 0.9621, 0.9183, 0.9727, 0.9839],
    }
    df = pd.DataFrame(data, index=metrics)

    # ---------- Style ----------
    plt.rcParams.update({
        # Times New Roman with graceful fallback
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 13,
        "axes.labelsize": 15,
        "axes.titlesize": 15,   # no title used, but keep consistent sizing
        "legend.fontsize": 12,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "figure.dpi": 240,
    })

    # ---------- Plot ----------
    fig, ax = plt.subplots(figsize=(10.5, 4.8))

    n_metrics = len(metrics)
    n_variants = len(variants)
    x = np.arange(n_metrics)
    bar_width = 0.18
    offsets = np.linspace(-bar_width*(n_variants-1)/2, bar_width*(n_variants-1)/2, n_variants)

    palette = ["#2a81ba", "#4f9fd3", "#7fbce3", "#b7d9f2"]

    for i, v in enumerate(variants):
        vals = df[v].values
        ax.bar(
            x + offsets[i],
            vals,
            width=bar_width,
            label=v,
            color=palette[i % len(palette)],
            edgecolor="white",
            linewidth=0.5,
        )

    # Axes labels & limits (no title)
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(["Interference Detection F1", "Interference Type F1", "Antenna Area F1", "Signal-to-Noise $R^2$", "Bandwidth $R^2$"], rotation=0)
    ax.set_ylim(0.75, 1.02)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 1.24),
        frameon=False,
        handlelength=1.2,
        columnspacing=0.9,
        ncol=2,
    )

    # Relative change vs Full for ablations
    full_idx = variants.index("Full LiteJam")

    for i, v in enumerate(variants[:-1]):  # exclude Full
        vals = df[v].values
        for j, val in enumerate(vals):
            ax.text(
                x[j] + offsets[i],
                val + 0.004,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    # Value labels for Full
    for j, val in enumerate(df["Full LiteJam"].values):
        ax.text(
            x[j] + offsets[full_idx],
            val + 0.006,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Tight layout with slight right margin for legend padding
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    # ---------- Save ----------
    out_dir = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "litejam_ablation_comparison_vB.png"
    fig.savefig(png_path, bbox_inches="tight", dpi=400)
    print(f"Saved: {png_path}")

if __name__ == "__main__":
    main()
