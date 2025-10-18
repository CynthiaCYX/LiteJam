from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np


def load_nt_metrics(data_path: Path) -> Tuple[List[int], Dict[str, List[float]]]:
    """Parse Nt_ablation style text file into Nt list and metric series."""
    nts: List[int] = []
    metrics: Dict[str, List[float]] = {}

    with data_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or "Nt=" not in line:
                continue

            nt_part, metrics_part = line.split("]", maxsplit=1)
            nt = int(nt_part.split("=")[1])
            nts.append(nt)

            for token in metrics_part.strip().split():
                if "=" not in token:
                    continue
                key, value = token.split("=", maxsplit=1)
                metrics.setdefault(key, []).append(float(value))

    return nts, metrics


def plot_metrics(
    nts: Sequence[int],
    metrics: Dict[str, Sequence[float]],
    output_path: Path | None = None,
) -> None:
    """Create two-panel line plot for the five tasks."""
    if output_path is not None:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "legend.fontsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "axes.edgecolor": "#2b2b2b",
            "axes.linewidth": 1.2,
            "grid.color": "#a0a0a0",
            "grid.linewidth": 0.8,
            "grid.linestyle": "--",
            "figure.dpi": 150,
        }
    )

    label_map = {
        "det_f1": "Interference Detection F1",
        "type_f1": "Interference Type F1",
        "area_f1": "Antenna Area F1",
        "snr_r2": "Signal-to-Noise Ratio R^2",
        "bwd_r2": "Bandwidth R^2",
    }

    f1_keys = ["det_f1", "type_f1", "area_f1"]
    r2_keys = ["snr_r2", "bwd_r2"]
    color_map = {
        "det_f1": "#cc2b2b",
        "type_f1": "#1f77b4",
        "area_f1": "#2ca02c",
        "snr_r2": "#1f77b4",
        "bwd_r2": "#cc2b2b",
    }
    band_width = {
        "det_f1": 1.5,
        "type_f1": 1.3,
        "area_f1": 1.6,
        "snr_r2": 0.7,
        "bwd_r2": 0.7,
    }
    fill_opacity = {
        "det_f1": 0.12,
        "type_f1": 0.10,
        "area_f1": 0.10,
        "snr_r2": 0.08,
        "bwd_r2": 0.08,
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=False)

    nt_array = np.array(nts, dtype=float)
    xtick_candidates = list(range(5, int(nt_array.max()) + 1, 5))
    xticks = sorted(
        {
            int(nt_array.min()),
            *xtick_candidates,
            int(nt_array.max()),
        }
    )

    def _series(key: str) -> Sequence[float]:
        values = metrics.get(key, [])
        if key.endswith("_f1") or key.endswith("_r2"):
            return [v * 100 for v in values]
        return list(values)

    def _plot_panel(
        ax,
        keys: Sequence[str],
        ylabel: str,
        ylim: Optional[Tuple[float, float]] = None,
    ) -> None:
        ax.set_facecolor("#fbfbfb")
        y_lower, y_upper = (0.0, 105.0) if ylim is None else ylim
        x = nt_array
        for key in keys:
            series = np.array(_series(key), dtype=float)
            band = band_width.get(key, 1.5)
            lower = series - band
            upper = series + band
            ax.plot(
                x,
                series,
                linewidth=2.0,
                solid_capstyle="round",
                linestyle="-",
                color=color_map.get(key, "#333333"),
                label=label_map.get(key, key),
            )
            ax.fill_between(
                x,
                np.clip(lower, y_lower, y_upper),
                np.clip(upper, y_lower, y_upper),
                color=color_map.get(key, "#333333"),
                alpha=fill_opacity.get(key, 0.1),
                linewidth=0,
            )
        ax.set_xlabel("Nt")
        ax.set_ylabel(ylabel)
        ax.set_xlim(nt_array.min(), nt_array.max())
        ax.set_ylim(y_lower, y_upper)
        if xticks:
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(int(x)) for x in xticks])
        ax.grid(True)
        ax.legend(frameon=True, facecolor="white", edgecolor="#dddddd", loc="best")
        ax.tick_params(direction="out", length=5, width=1.2)
        ax.margins(x=0.01, y=0.05)
        for spine in ax.spines.values():
            spine.set_linewidth(1)

    _plot_panel(axes[0], f1_keys, "F1 [%]")
    _plot_panel(axes[1], r2_keys, "R^2 [%]", (80.0, 100.0))

    fig.tight_layout(pad=1.8, w_pad=2.8)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Nt ablation metrics from Nt_ablation.txt."
    )
    default_input = Path(__file__).resolve().parents[1] / "Nt_ablation.txt"
    default_output = (
        Path(__file__).resolve().parents[1] / "reports" / "figures" / "nt_ablation.png"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help=f"Path to Nt_ablation.txt (default: {default_input})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Path to save the plot (default: reports/figures/nt_ablation.png).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nts, metrics = load_nt_metrics(args.input)
    if not nts:
        raise ValueError(f"No Nt entries found in {args.input}")
    plot_metrics(nts, metrics, args.output)
    if args.output:
        print(f"Saved Nt ablation plot to {args.output}")


if __name__ == "__main__":
    main()
