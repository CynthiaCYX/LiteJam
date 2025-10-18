#!/usr/bin/env python3
"""Efficient dataset statistics for the controlled low frequency GNSS corpus."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SPLITS = ("train", "val", "test")


@dataclass
class RunningStats:
    count: int = 0
    s1: float = 0.0
    s2: float = 0.0
    min_val: float = math.inf
    max_val: float = -math.inf

    def update(self, value: float) -> None:
        self.count += 1
        self.s1 += value
        self.s2 += value * value
        if value < self.min_val:
            self.min_val = value
        if value > self.max_val:
            self.max_val = value

    def mean(self) -> float:
        return self.s1 / self.count if self.count else math.nan

    def std(self) -> float:
        if self.count <= 1:
            return 0.0
        mean = self.mean()
        variance = max(self.s2 / self.count - mean * mean, 0.0)
        return math.sqrt(variance)


def parse_line(line: str, split: str) -> Dict[str, object]:
    parts = line.strip().split("\t")
    if len(parts) < 7:
        raise ValueError(f"Malformed split line ({split}): {line}")
    return {
        "class_id": int(parts[1]),
        "amplitude": float(parts[2]),
        "area": int(parts[3]),
        "subjammer": parts[4],
    }


def summarize_split(path: Path, split: str) -> Dict[str, object]:
    total = 0
    det_pos = 0
    class_counts: Counter[int] = Counter()
    area_counts: Counter[int] = Counter()
    area_det: Dict[int, Counter[str]] = defaultdict(Counter)
    area_type_counts: Dict[int, Counter[int]] = defaultdict(Counter)
    subjammer_per_type: Dict[int, Counter[str]] = defaultdict(Counter)
    amplitude_stats = RunningStats()
    amplitude_per_class: Dict[int, RunningStats] = defaultdict(RunningStats)

    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            total += 1
            row = parse_line(raw, split)
            class_id = row["class_id"]  # type: ignore[assignment]
            amplitude = row["amplitude"]  # type: ignore[assignment]
            area = row["area"]  # type: ignore[assignment]
            subjammer = row["subjammer"]  # type: ignore[assignment]

            amplitude_stats.update(amplitude)
            amplitude_per_class[class_id].update(amplitude)

            class_counts[class_id] += 1
            area_counts[area] += 1
            key = "pos" if class_id > 0 else "neg"
            area_det[area][key] += 1

            if class_id > 0:
                det_pos += 1
                area_type_counts[area][class_id] += 1
                subjammer_per_type[class_id][subjammer] += 1

    return {
        "total": total,
        "det_pos": det_pos,
        "det_neg": total - det_pos,
        "class_counts": class_counts,
        "area_counts": area_counts,
        "area_det": area_det,
        "area_type_counts": area_type_counts,
        "subjammer_per_type": subjammer_per_type,
        "amplitude": amplitude_stats,
        "amplitude_per_class": amplitude_per_class,
        "unique_subjammer_total": sum(len(counter) for counter in subjammer_per_type.values()),
    }


def write_csv(path: Path, header: Iterable[str], rows: Iterable[Iterable[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def plot_bar(path: Path, labels: Iterable[str], values: Iterable[int], title: str) -> None:
    labels = list(labels)
    values = list(values)
    if not labels:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(6, len(labels) * 0.8), 4))
    plt.bar(labels, values, color="#4E79A7")
    plt.ylabel("samples")
    plt.title(title)
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def describe_stats(stats: RunningStats) -> Dict[str, float]:
    return {
        "count": stats.count,
        "mean": stats.mean(),
        "std": stats.std(),
        "min": stats.min_val if stats.count else math.nan,
        "max": stats.max_val if stats.count else math.nan,
    }


def summarize_group(
    root: Path,
    group: str,
    suffix: str,
    output_dir: Optional[Path],
) -> Dict[str, Dict[str, object]]:
    base = root / "info" / "splits" / group
    results: Dict[str, Dict[str, object]] = {}
    for split in SPLITS:
        split_path = base / f"{split}_{suffix}.txt"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split file: {split_path}")
        stats = summarize_split(split_path, split)
        results[split] = stats

        total = stats["total"]
        print(f"[{group} - {split}] total={total} det_pos={stats['det_pos']} det_neg={stats['det_neg']}")

        class_counts: Counter[int] = stats["class_counts"]  # type: ignore[index]
        for class_id, count in sorted(class_counts.items()):
            ratio = (count / total * 100.0) if total else 0.0
            print(f"  class_id={class_id}: {count} ({ratio:.2f}%)")

        amp = describe_stats(stats["amplitude"])  # type: ignore[index]
        print(
            f"  snr mean={amp['mean']:.2f} std={amp['std']:.2f} min={amp['min']:.2f} max={amp['max']:.2f}"
        )

        sub_per_type: Dict[int, Counter[str]] = stats["subjammer_per_type"]  # type: ignore[index]
        if sub_per_type:
            print("  subjammer variety (top 3 per class):")
            for class_id in sorted(sub_per_type):
                counter = sub_per_type[class_id]
                print(
                    f"    class_id={class_id}: {len(counter)} unique -> "
                    + ", ".join(f"{sj}({cnt})" for sj, cnt in counter.most_common(3))
                )

        if output_dir is not None:
            split_dir = output_dir / group / split
            split_dir.mkdir(parents=True, exist_ok=True)

            write_csv(
                split_dir / "class_distribution.csv",
                ["class_id", "count", "ratio"],
                [
                    [cid, count, f"{(count / total * 100.0):.4f}" if total else "0.0"]
                    for cid, count in sorted(class_counts.items())
                ],
            )

            area_counts: Counter[int] = stats["area_counts"]  # type: ignore[index]
            area_det: Dict[int, Counter[str]] = stats["area_det"]  # type: ignore[index]
            write_csv(
                split_dir / "area_distribution.csv",
                ["area", "count", "ratio", "det_pos", "det_neg"],
                [
                    [
                        area,
                        count,
                        f"{(count / total * 100.0):.4f}" if total else "0.0",
                        area_det[area].get("pos", 0),
                        area_det[area].get("neg", 0),
                    ]
                    for area, count in sorted(area_counts.items())
                ],
            )

            area_type: Dict[int, Counter[int]] = stats["area_type_counts"]  # type: ignore[index]
            rows = []
            for area, counter in sorted(area_type.items()):
                total_area = sum(counter.values()) or 1
                for class_id, count in sorted(counter.items()):
                    rows.append([area, class_id, count, f"{(count / total_area * 100.0):.4f}"])
            write_csv(
                split_dir / "type_per_area.csv",
                ["area", "class_id", "count", "ratio"],
                rows,
            )

            amplitude_per_class: Dict[int, RunningStats] = stats["amplitude_per_class"]  # type: ignore[index]
            write_csv(
                split_dir / "amplitude_per_class.csv",
                ["class_id", "count", "mean", "std", "min", "max"],
                [
                    [
                        cid,
                        rs.count,
                        f"{rs.mean():.4f}",
                        f"{rs.std():.4f}",
                        f"{rs.min_val:.4f}" if rs.count else "nan",
                        f"{rs.max_val:.4f}" if rs.count else "nan",
                    ]
                    for cid, rs in sorted(amplitude_per_class.items())
                ],
            )

            sub_rows = []
            for cid, counter in sorted(sub_per_type.items()):
                sub_rows.append(
                    [
                        cid,
                        len(counter),
                        "; ".join(f"{sj}:{cnt}" for sj, cnt in counter.most_common()),
                    ]
                )
            write_csv(
                split_dir / "subjammer_counts.csv",
                ["class_id", "unique_subjammer", "counts"],
                sub_rows,
            )

            plot_bar(
                split_dir / "class_distribution.png",
                [str(cid) for cid in sorted(class_counts.keys())],
                [class_counts[cid] for cid in sorted(class_counts.keys())],
                title=f"Class distribution ({group}-{split})",
            )

            plot_bar(
                split_dir / "area_distribution.png",
                [str(area) for area in sorted(area_counts.keys())],
                [area_counts[area] for area in sorted(area_counts.keys())],
                title=f"Area distribution ({group}-{split})",
            )

            positive_items = [(cid, count) for cid, count in class_counts.items() if cid > 0]
            if positive_items:
                plot_bar(
                    split_dir / "type_positive_distribution.png",
                    [str(cid) for cid, _ in sorted(positive_items)],
                    [count for _, count in sorted(positive_items)],
                    title=f"Positive class distribution ({group}-{split})",
                )

    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dataset statistics overview")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/controlled_low_frequency/GNSS_datasets/dataset1"),
        help="Dataset root",
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
        help="Split filename suffix",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory for CSV/figure outputs",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    output_dir = args.output_dir.resolve() if args.output_dir else None

    summary_rows: List[List[object]] = []

    for group in args.groups:
        print(f"\n===== {group.upper()} =====")
        results = summarize_group(args.root, group, args.split_suffix, output_dir)

        for split, stats in results.items():
            amp_stats = describe_stats(stats["amplitude"])  # type: ignore[index]
            summary_rows.append(
                [
                    group,
                    split,
                    stats["total"],
                    stats["det_pos"],
                    stats["det_neg"],
                    amp_stats["mean"],
                    amp_stats["std"],
                    amp_stats["min"],
                    amp_stats["max"],
                    stats.get("unique_subjammer_total", 0),
                ]
            )

    if output_dir is not None and summary_rows:
        write_csv(
            output_dir / "split_summary.csv",
            [
                "group",
                "split",
                "samples",
                "det_pos",
                "det_neg",
                "snr_mean",
                "snr_std",
                "snr_min",
                "snr_max",
                "unique_subjammer",
            ],
            summary_rows,
        )

    print("\nSummary complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
