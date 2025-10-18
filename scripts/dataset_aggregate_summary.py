#!/usr/bin/env python3
"""Aggregate dataset statistics for GNSS controlled low frequency corpus.

This tool summarises:
- Total sample counts per split directory (jammer / signal_to_noise / bandwidth)
  for ``subjammer_dep`` and ``subjammer_indep`` suffixes, without separating
  train/val/test (counts are aggregated across the three splits).
- Global class distribution, SNR range, and per-class subjammer diversity based
  on ``info/all_file_labels.txt``.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import math
import re

SPLITS = ("train", "val", "test")
DEFAULT_GROUPS = ("jammer", "signal_to_noise", "bandwidth")
DEFAULT_SUFFIXES = ("subjammer_dep", "subjammer_indep")

_BW_SUFFIX = re.compile(r"BW([0-9]+(?:[._p][0-9]+)?)", re.IGNORECASE)
_BW_PREFIX = re.compile(r"([0-9]+(?:[._p][0-9]+)?)BW", re.IGNORECASE)


def _parse_bandwidth(subjammer: str) -> float | None:
    if not subjammer or subjammer.lower() == "none":
        return None
    match = _BW_SUFFIX.search(subjammer)
    if not match:
        match = _BW_PREFIX.search(subjammer)
        if not match:
            return None
    token = match.group(1).replace("_", ".").replace("p", ".")
    try:
        return float(token)
    except ValueError:
        return None


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


def aggregate_group_details(root: Path, group: str, suffix: str) -> Dict[str, object]:
    base = root / "info" / "splits" / group
    seen_paths = set()
    total_lines = 0
    class_counts: Counter[int] = Counter()
    subjammer_per_type: Dict[int, Counter[str]] = defaultdict(Counter)
    amplitude_counts: Counter[float] = Counter()
    amplitude_stats = RunningStats()
    bandwidth_counts: Counter[float] = Counter()
    bandwidth_stats = RunningStats()


    for split in SPLITS:
        split_path = base / f"{split}_{suffix}.txt"
        if not split_path.exists():
            continue
        with split_path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                total_lines += 1
                parts = raw.split("\t")
                rel_path = parts[0]
                seen_paths.add(rel_path)
                class_id = int(parts[1]) if len(parts) > 1 else -1
                class_counts[class_id] += 1
                if class_id > 0 and len(parts) > 4:
                    subjammer = parts[4]
                    subjammer_per_type[class_id][subjammer] += 1
                if group == "signal_to_noise" and len(parts) > 2:
                    amp = float(parts[2])
                    amplitude_stats.update(amp)
                    amplitude_counts[amp] += 1
                if group == "bandwidth" and len(parts) > 4:
                    bw = _parse_bandwidth(parts[4])
                    if bw is not None:
                        bandwidth_stats.update(bw)
                        bandwidth_counts[bw] += 1

    return {
        "total": total_lines,
        "unique": len(seen_paths),
        "class_counts": class_counts,
        "subjammer_per_type": subjammer_per_type,
        "amplitude_counts": amplitude_counts if group == "signal_to_noise" else None,
        "amplitude_stats": amplitude_stats if group == "signal_to_noise" else None,
        "bandwidth_counts": bandwidth_counts if group == "bandwidth" else None,
        "bandwidth_stats": bandwidth_stats if group == "bandwidth" else None,
    }


def parse_all_file_labels(path: Path) -> Dict[str, object]:
    class_counts: Counter[int] = Counter()
    subjammer_per_class: Dict[int, Counter[str]] = defaultdict(Counter)
    snr_stats = RunningStats()
    area_counts: Counter[int] = Counter()

    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split("\t")
            if len(parts) < 5:
                continue
            class_id = int(parts[1])
            amplitude = float(parts[2])
            area = int(parts[3])
            subjammer = parts[4]

            class_counts[class_id] += 1
            subjammer_per_class[class_id][subjammer] += 1
            snr_stats.update(amplitude)
            area_counts[area] += 1

    return {
        "class_counts": class_counts,
        "subjammer_per_class": subjammer_per_class,
        "snr": snr_stats,
        "area_counts": area_counts,
    }


def write_csv(path: Path, header: Iterable[str], rows: Iterable[Iterable[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate dataset statistics without split separation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/controlled_low_frequency/GNSS_datasets/dataset1"),
        help="Dataset root directory",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=list(DEFAULT_GROUPS),
        help="Split directories to scan",
    )
    parser.add_argument(
        "--suffixes",
        nargs="+",
        default=list(DEFAULT_SUFFIXES),
        help="Split suffixes to aggregate (e.g. subjammer_dep)",
    )
    parser.add_argument(
        "--labels-file",
        type=Path,
        default=None,
        help="Optional path to all_file_labels.txt (defaults to root/info/all_file_labels.txt)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to write CSV summaries",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve() if args.output_dir else None

    agg_rows = []
    suffix_totals: Dict[str, Dict[str, int]] = {}
    detailed: Dict[Tuple[str, str], Dict[str, object]] = {}
    print("=== Aggregated split counts ===")
    for group in args.groups:
        for suffix in args.suffixes:
            details = aggregate_group_details(args.root, group, suffix)
            total = details["total"]  # type: ignore[index]
            unique = details["unique"]  # type: ignore[index]
            agg_rows.append([group, suffix, total, unique])
            detailed[(group, suffix)] = details
            suffix_totals.setdefault(suffix, {"total": 0, "unique": 0})
            suffix_totals[suffix]["total"] += total
            suffix_totals[suffix]["unique"] += unique
            print(f"{group} | {suffix} -> total entries={total} unique samples={unique}")

    if output_dir is not None and agg_rows:
        write_csv(
            output_dir / "aggregate_split_counts.csv",
            ["group", "suffix", "total_entries", "unique_samples"],
            agg_rows,
        )
        if suffix_totals:
            write_csv(
                output_dir / "aggregate_suffix_summary.csv",
                ["suffix", "total_entries", "unique_samples"],
                [[suffix, stats["total"], stats["unique"]] for suffix, stats in suffix_totals.items()],
            )

        for (group, suffix), details in detailed.items():
            total = details["total"] or 1  # type: ignore[index]
            class_counts: Counter[int] = details["class_counts"]  # type: ignore[index]
            subjammer_per_type: Dict[int, Counter[str]] = details["subjammer_per_type"]  # type: ignore[index]

            group_dir = output_dir / "per_group" / group / suffix
            if group == "jammer":
                write_csv(
                    group_dir / "class_distribution.csv",
                    ["class_id", "count", "ratio"],
                    [
                        [cid, count, f"{count / total * 100.0:.4f}"]
                        for cid, count in sorted(class_counts.items())
                    ],
                )

                write_csv(
                    group_dir / "subjammer_counts.csv",
                    ["class_id", "unique_subjammer", "counts", "ratio"],
                    [
                        [
                            cid,
                            len(counter),
                            "; ".join(f"{sj}:{cnt}" for sj, cnt in counter.most_common()),
                            f"{sum(counter.values()) / total * 100.0:.4f}",
                        ]
                        for cid, counter in sorted(subjammer_per_type.items())
                    ],
                )

            if group == "signal_to_noise":
                amp_counts: Counter[float] = details["amplitude_counts"]  # type: ignore[index]
                amp_stats: RunningStats = details["amplitude_stats"]  # type: ignore[index]
                write_csv(
                    group_dir / "amplitude_distribution.csv",
                    ["amplitude", "count", "ratio"],
                    [
                        [amp, count, f"{count / total * 100.0:.4f}"]
                        for amp, count in sorted(amp_counts.items())
                    ],
                )
                write_csv(
                    group_dir / "amplitude_stats.csv",
                    ["count", "mean", "std", "min", "max"],
                    [[
                        amp_stats.count,
                        f"{amp_stats.mean():.4f}",
                        f"{amp_stats.std():.4f}",
                        f"{amp_stats.min_val:.4f}",
                        f"{amp_stats.max_val:.4f}",
                    ]],
                )

            if group == "bandwidth":
                bw_counts = details.get("bandwidth_counts")  # type: ignore[assignment]
                bw_stats = details.get("bandwidth_stats")  # type: ignore[assignment]
                if not isinstance(bw_counts, Counter) or not isinstance(bw_stats, RunningStats):
                    continue
                write_csv(
                    group_dir / "bandwidth_distribution.csv",
                    ["bandwidth", "count", "ratio"],
                    [
                        [bw, count, f"{count / total * 100.0:.4f}"]
                        for bw, count in sorted(bw_counts.items())
                    ],
                )
                write_csv(
                    group_dir / "bandwidth_stats.csv",
                    ["count", "mean", "std", "min", "max"],
                    [[
                        bw_stats.count,
                        f"{bw_stats.mean():.4f}",
                        f"{bw_stats.std():.4f}",
                        f"{bw_stats.min_val:.4f}",
                        f"{bw_stats.max_val:.4f}",
                    ]],
                )

    labels_path = args.labels_file or (args.root / "info" / "all_file_labels.txt")
    if labels_path.exists():
        print("\n=== all_file_labels statistics ===")
        stats = parse_all_file_labels(labels_path)
        class_counts: Counter[int] = stats["class_counts"]  # type: ignore[index]
        snr_stats: RunningStats = stats["snr"]  # type: ignore[index]
        subjammer_per_class: Dict[int, Counter[str]] = stats["subjammer_per_class"]  # type: ignore[index]
        area_counts: Counter[int] = stats["area_counts"]  # type: ignore[index]

        total = sum(class_counts.values()) or 1
        print(f"samples={total}")
        print(
            f"snr mean={snr_stats.mean():.2f} std={snr_stats.std():.2f} "
            f"min={snr_stats.min_val:.2f} max={snr_stats.max_val:.2f}"
        )
        print("class distribution:")
        for cid, count in sorted(class_counts.items()):
            print(f"  class_id={cid}: {count} ({count / total * 100.0:.2f}%)")

        print("antenna area distribution:")
        area_total = sum(area_counts.values()) or 1
        for area, count in sorted(area_counts.items()):
            print(f"  area={area}: {count} ({count / area_total * 100.0:.2f}%)")

        print("subjammer counts per class (top 5 shown):")
        for cid, counter in sorted(subjammer_per_class.items()):
            entries = ", ".join(f"{sj}({cnt})" for sj, cnt in counter.most_common(5))
            print(f"  class_id={cid}: {len(counter)} unique -> {entries}")

        if output_dir is not None:
            write_csv(
                output_dir / "all_file_labels_class_distribution.csv",
                ["class_id", "count", "ratio"],
                [
                    [cid, count, f"{count / total * 100.0:.4f}"]
                    for cid, count in sorted(class_counts.items())
                ],
            )

            write_csv(
                output_dir / "all_file_labels_area_distribution.csv",
                ["area", "count", "ratio"],
                [
                    [area, count, f"{count / area_total * 100.0:.4f}"]
                    for area, count in sorted(area_counts.items())
                ],
            )

            write_csv(
                output_dir / "all_file_labels_subjammer_counts.csv",
                ["class_id", "unique_subjammer", "counts"],
                [
                    [
                        cid,
                        len(counter),
                        "; ".join(f"{sj}:{cnt}" for sj, cnt in counter.most_common()),
                    ]
                    for cid, counter in sorted(subjammer_per_class.items())
                ],
            )

            write_csv(
                output_dir / "all_file_labels_snr_stats.csv",
                ["count", "mean", "std", "min", "max"],
                [[
                    snr_stats.count,
                    f"{snr_stats.mean():.4f}",
                    f"{snr_stats.std():.4f}",
                    f"{snr_stats.min_val:.4f}",
                    f"{snr_stats.max_val:.4f}",
                ]],
            )
    else:
        print(f"Warning: labels file not found: {labels_path}")

    print("\nAggregate summary complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
