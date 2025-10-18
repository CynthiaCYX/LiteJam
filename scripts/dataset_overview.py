#!/usr/bin/env python3
"""Dataset statistics and publication-ready plots for LiteJam GNSS splits."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SPLITS = ("train", "val", "test")


@dataclass
class RunningStats:
    count: int = 0
    s1: float = 0.0
    s2: float = 0.0
    vmin: float = math.inf
    vmax: float = -math.inf
    values: list[float] = None  # type: ignore[assignment]

    def update(self, value: float) -> None:
        self.count += 1
        self.s1 += value
        self.s2 += value * value
        if value < self.vmin:
            self.vmin = value
        if value > self.vmax:
            self.vmax = value
        if self.values is None:
            self.values = []
        self.values.append(value)

    def mean(self) -> float:
        return self.s1 / self.count if self.count else math.nan

    def std(self) -> float:
        if self.count <= 1:
            return 0.0
        mean = self.mean()
        variance = max(self.s2 / self.count - mean * mean, 0.0)
        return math.sqrt(variance)

    def quantiles(self) -> tuple[float, float, float]:
        if not self.values:
            return (math.nan, math.nan, math.nan)
        data = sorted(self.values)
        median = statistics.median(data)
        lower = data[: len(data) // 2]
        upper = data[len(data) // 2 + (0 if len(data) % 2 == 0 else 1) :]
        q1 = statistics.median(lower) if lower else median
        q3 = statistics.median(upper) if upper else median
        return (q1, median, q3)


_BWD_SUFFIX = re.compile(r"BW([0-9]+(?:[._p][0-9]+)?)", re.IGNORECASE)
_BWD_PREFIX = re.compile(r"([0-9]+(?:[._p][0-9]+)?)BW", re.IGNORECASE)


def _extract_bandwidth(name: str) -> Optional[float]:
    if not name or name.lower() == "none":
        return None
    match = _BWD_SUFFIX.search(name)
    if not match:
        match = _BWD_PREFIX.search(name)
    if not match:
        return None
    token = match.group(1).replace("_", ".").replace("p", ".")
    try:
        return float(token)
    except ValueError:
        return None


def parse_line(raw: str) -> tuple[int, float, int, str]:
    parts = raw.split("\t")
    if len(parts) < 5:
        raise ValueError(f"Malformed split line: {raw}")
    return int(parts[1]), float(parts[2]), int(parts[3]), parts[4]


def summarise_split(path: Path) -> Dict[str, object]:
    total = 0
    det_pos = 0
    class_counts: Counter[int] = Counter()
    area_counts: Counter[int] = Counter()
    area_det: Dict[int, Counter[str]] = defaultdict(Counter)
    area_type_counts: Dict[int, Counter[int]] = defaultdict(Counter)
    subjammer_counts: Dict[int, Counter[str]] = defaultdict(Counter)
    amp_total = RunningStats()
    amp_per_class: Dict[int, RunningStats] = defaultdict(RunningStats)
    bandwidth_values: list[float] = []

    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            class_id, amplitude, area, subjammer = parse_line(raw)
            total += 1
            class_counts[class_id] += 1
            area_counts[area] += 1
            area_det[area]["pos" if class_id > 0 else "neg"] += 1

            amp_total.update(amplitude)
            amp_per_class[class_id].update(amplitude)

            if class_id > 0:
                det_pos += 1
                area_type_counts[area][class_id] += 1
                subjammer_counts[class_id][subjammer] += 1

            bw = _extract_bandwidth(subjammer)
            if bw is not None:
                bandwidth_values.append(bw)

    return {
        "total": total,
        "det_pos": det_pos,
        "det_neg": total - det_pos,
        "class_counts": class_counts,
        "area_counts": area_counts,
        "area_det": area_det,
        "area_type_counts": area_type_counts,
        "subjammer_counts": subjammer_counts,
        "amp_total": amp_total,
        "amp_per_class": amp_per_class,
        "bandwidth_values": bandwidth_values,
    }


def write_csv(path: Path, header: Iterable[str], rows: Iterable[Iterable[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def plot_bar(path: Path, labels: list[str], values: list[int], title: str, ylabel: str = "Samples") -> None:
    if not labels:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(6, len(labels) * 0.75), 4))
    plt.bar(labels, values, color="#4E79A7")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_histogram(path: Path, values: list[float], title: str, xlabel: str) -> None:
    if not values:
        return
    bins = max(10, min(60, int(math.sqrt(len(values)))))
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=bins, color="#59A14F", edgecolor="white")
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_top_subjammers(path: Path, subjammer_counts: Dict[int, Counter[str]], top_n: int = 12) -> None:
    items: list[tuple[str, int]] = []
    for class_id, counter in subjammer_counts.items():
        for name, count in counter.items():
            items.append((f"C{class_id}:{name}", count))
    items.sort(key=lambda x: x[1], reverse=True)
    labels = [label for label, _ in items[:top_n]]
    values = [value for _, value in items[:top_n]]
    plot_bar(path, labels, values, title=f"Top {top_n} subjammer occurrences", ylabel="Instances")


def merge_stats(target: RunningStats, source: RunningStats) -> None:
    target.count += source.count
    target.s1 += source.s1
    target.s2 += source.s2
    if source.count:
        target.vmin = min(target.vmin, source.vmin)
        target.vmax = max(target.vmax, source.vmax)
        if target.values is None:
            target.values = []
        if source.values:
            target.values.extend(source.values)


def process_group(root: Path, group: str, suffix: str, output_dir: Path) -> None:
    base = root / "info" / "splits" / group
    if not base.exists():
        raise FileNotFoundError(f"Group directory not found: {base}")

    per_split: Dict[str, Dict[str, object]] = {}
    for split in SPLITS:
        split_path = base / f"{split}_{suffix}.txt"
        if not split_path.exists():
            print(f"[warn] Missing split: {split_path}")
            continue
        stats = summarise_split(split_path)
        per_split[split] = stats
        amp_mean, amp_std, amp_min, amp_max = stats["amp_total"].mean(), stats["amp_total"].std(), stats["amp_total"].vmin, stats["amp_total"].vmax
        q1, median, q3 = stats["amp_total"].quantiles()
        print(
            f"[{group} - {split}] total={stats['total']} det_pos={stats['det_pos']} det_neg={stats['det_neg']} "
            f"| amplitude mean={amp_mean:.2f}+/-{amp_std:.2f} median={median:.2f} (Q1={q1:.2f}, Q3={q3:.2f}) range=[{amp_min:.2f}, {amp_max:.2f}]"
        )

    if not per_split:
        print(f"[warn] No splits processed for group {group}")
        return

    overall_total = sum(stats["total"] for stats in per_split.values())
    overall_det_pos = sum(stats["det_pos"] for stats in per_split.values())
    overall_det_neg = sum(stats["det_neg"] for stats in per_split.values())

    overall_amp = RunningStats()
    overall_class_counts: Counter[int] = Counter()
    overall_area_counts: Counter[int] = Counter()
    overall_area_det: Dict[int, Counter[str]] = defaultdict(Counter)
    overall_sub_counts: Dict[int, Counter[str]] = defaultdict(Counter)
    overall_amp_per_class: Dict[int, RunningStats] = defaultdict(RunningStats)
    overall_bandwidth: list[float] = []

    for stats in per_split.values():
        merge_stats(overall_amp, stats["amp_total"])
        overall_class_counts.update(stats["class_counts"])
        overall_area_counts.update(stats["area_counts"])
        for area, counter in stats["area_det"].items():
            overall_area_det[area]["pos"] += counter.get("pos", 0)
            overall_area_det[area]["neg"] += counter.get("neg", 0)
        for cid, counter in stats["subjammer_counts"].items():
            overall_sub_counts[cid].update(counter)
        for cid, rs in stats["amp_per_class"].items():
            merge_stats(overall_amp_per_class[cid], rs)
        overall_bandwidth.extend(stats.get("bandwidth_values", []))

    overall_mean, overall_std, overall_min, overall_max = overall_amp.mean(), overall_amp.std(), overall_amp.vmin, overall_amp.vmax
    overall_q1, overall_median, overall_q3 = overall_amp.quantiles()
    print(
        f"[{group} - overall] total={overall_total} det_pos={overall_det_pos} det_neg={overall_det_neg} "
        f"| amplitude mean={overall_mean:.2f}+/-{overall_std:.2f} median={overall_median:.2f} (Q1={overall_q1:.2f}, Q3={overall_q3:.2f}) range=[{overall_min:.2f}, {overall_max:.2f}]"
    )

    group_dir = output_dir / group
    group_dir.mkdir(parents=True, exist_ok=True)

    # Summary table with per-split and overall metrics
    metrics = [
        ("total_samples", lambda s: s["total"]),
        ("det_positive", lambda s: s["det_pos"]),
        ("det_negative", lambda s: s["det_neg"]),
        ("amplitude_mean", lambda s: s["amp_total"].mean()),
        ("amplitude_std", lambda s: s["amp_total"].std()),
        ("amplitude_q1", lambda s: s["amp_total"].quantiles()[0]),
        ("amplitude_median", lambda s: s["amp_total"].quantiles()[1]),
        ("amplitude_q3", lambda s: s["amp_total"].quantiles()[2]),
        ("amplitude_min", lambda s: s["amp_total"].vmin if s["amp_total"].count else math.nan),
        ("amplitude_max", lambda s: s["amp_total"].vmax if s["amp_total"].count else math.nan),
    ]

    summary_rows = []
    for name, extractor in metrics:
        row = [name]
        for split in SPLITS:
            stats = per_split.get(split)
            if stats is None:
                row.append("")
                continue
            value = extractor(stats)
            row.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        # overall
        overall_value = None
        if name == "total_samples":
            overall_value = overall_total
        elif name == "det_positive":
            overall_value = overall_det_pos
        elif name == "det_negative":
            overall_value = overall_det_neg
        elif name == "amplitude_mean":
            overall_value = overall_mean
        elif name == "amplitude_std":
            overall_value = overall_std
        elif name == "amplitude_q1":
            overall_value = overall_q1
        elif name == "amplitude_median":
            overall_value = overall_median
        elif name == "amplitude_q3":
            overall_value = overall_q3
        elif name == "amplitude_min":
            overall_value = overall_min
        elif name == "amplitude_max":
            overall_value = overall_max
        row.append(f"{overall_value:.4f}" if isinstance(overall_value, float) else str(overall_value))
        summary_rows.append(row)

    write_csv(group_dir / "summary.csv", ["metric", *SPLITS, "overall"], summary_rows)

    # Class distribution table
    class_rows = []
    for cid in sorted(overall_class_counts.keys()):
        split_counts = [per_split.get(split, {}).get("class_counts", Counter()).get(cid, 0) for split in SPLITS]
        total_count = overall_class_counts[cid]
        ratio = total_count / overall_total * 100.0 if overall_total else 0.0
        class_rows.append([cid, *split_counts, total_count, f"{ratio:.4f}"])
    write_csv(
        group_dir / "class_distribution.csv",
        ["class_id", *SPLITS, "total", "total_ratio"],
        class_rows,
    )

    # Area distribution table
    area_rows = []
    for area in sorted(overall_area_counts.keys()):
        split_counts = [per_split.get(split, {}).get("area_counts", Counter()).get(area, 0) for split in SPLITS]
        total_count = overall_area_counts[area]
        ratio = total_count / overall_total * 100.0 if overall_total else 0.0
        det_pos = overall_area_det[area].get("pos", 0)
        det_neg = overall_area_det[area].get("neg", 0)
        area_rows.append([area, *split_counts, total_count, f"{ratio:.4f}", det_pos, det_neg])
    write_csv(
        group_dir / "area_distribution.csv",
        ["area", *SPLITS, "total", "total_ratio", "det_pos", "det_neg"],
        area_rows,
    )

    # Amplitude per class overall
    amp_rows = []
    for cid in sorted(overall_amp_per_class.keys()):
        rs = overall_amp_per_class[cid]
        mean, std = rs.mean(), rs.std()
        q1, median, q3 = rs.quantiles()
        amp_rows.append([
            cid,
            rs.count,
            f"{mean:.4f}",
            f"{std:.4f}",
            f"{q1:.4f}",
            f"{median:.4f}",
            f"{q3:.4f}",
            f"{rs.vmin:.4f}" if rs.count else "nan",
            f"{rs.vmax:.4f}" if rs.count else "nan",
        ])
    write_csv(
        group_dir / "amplitude_per_class.csv",
        ["class_id", "count", "mean", "std", "q1", "median", "q3", "min", "max"],
        amp_rows,
    )

    # Subjammer counts overall
    sub_rows = []
    for cid, counter in sorted(overall_sub_counts.items()):
        sub_rows.append([cid, len(counter), "; ".join(f"{name}:{cnt}" for name, cnt in counter.most_common(10))])
    write_csv(group_dir / "subjammer_counts.csv", ["class_id", "unique_subjammer", "top_examples"], sub_rows)

    # Plots (overall)
    plot_bar(
        group_dir / "class_distribution_overall.png",
        [str(cid) for cid in sorted(overall_class_counts.keys())],
        [overall_class_counts[cid] for cid in sorted(overall_class_counts.keys())],
        title=f"Class distribution ({group}, overall)",
    )

    positive_overall = [(cid, overall_class_counts[cid]) for cid in sorted(overall_class_counts.keys()) if cid > 0]
    if positive_overall:
        plot_bar(
            group_dir / "positive_class_distribution_overall.png",
            [str(cid) for cid, _ in positive_overall],
            [count for _, count in positive_overall],
            title=f"Positive class distribution ({group}, overall)",
        )

    plot_bar(
        group_dir / "area_distribution_overall.png",
        [str(area) for area in sorted(overall_area_counts.keys())],
        [overall_area_counts[area] for area in sorted(overall_area_counts.keys())],
        title=f"Area distribution ({group}, overall)",
    )

    plot_histogram(
        group_dir / "amplitude_distribution_overall.png",
        overall_amp.values or [],
        title=f"Amplitude distribution ({group}, overall)",
        xlabel="Amplitude / SNR proxy (dB)",
    )

    if overall_bandwidth:
        plot_histogram(
            group_dir / "bandwidth_distribution_overall.png",
            overall_bandwidth,
            title=f"Bandwidth distribution ({group}, overall)",
            xlabel="Bandwidth (MHz)",
        )
        write_csv(
            group_dir / "bandwidth_values.csv",
            ["bandwidth_mhz"],
            [[f"{value:.6f}"] for value in sorted(overall_bandwidth)],
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LiteJam dataset overview")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/controlled_low_frequency/GNSS_datasets/dataset1"),
        help="Dataset root directory",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["jammer", "signal_to_noise", "bandwidth"],
        help="Split groups to analyse",
    )
    parser.add_argument(
        "--split-suffix",
        default="subjammer_dep",
        help="Split suffix (default: subjammer_dep)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/dataset_overview"),
        help="Directory used to write CSV/PNG outputs",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for group in args.groups:
        print(f"\n==== {group.upper()} ====")
        process_group(args.root, group, args.split_suffix, args.output_dir)

    print("\nDataset overview complete. Outputs saved to", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
