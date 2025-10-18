#!/usr/bin/env python3
"""Generate subjammer-dependent and -independent bandwidth splits.

The script reads ``*_random.txt`` files, groups samples by the fifth column
(``subjammer``), and writes two new split variants:

- ``*_subjammer_dep.txt``: each subjammer is internally split into train/val/test
  with 64/16/20 ratios.
- ``*_subjammer_indep.txt``: subjammer IDs are exclusive per split; complete
  groups are assigned to train/val/test with ~65/15/20 sample ratios.

Both variants preserve the original text format. Use ``--root`` to change the
bandwidth directory and ``--seed`` for reproducibility.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


DEFAULT_BANDWIDTH_DIR = Path(
    "data/controlled_low_frequency/GNSS_datasets/dataset1/info/splits/bandwidth"
)
DEP_RATIOS = (0.64, 0.16, 0.20)
INDEP_RATIOS = (0.65, 0.15, 0.20)


@dataclass(frozen=True)
class Entry:
    columns: Tuple[str, ...]

    @property
    def subjammer(self) -> str:
        try:
            return self.columns[4]
        except IndexError as err:
            raise ValueError("Record is missing the subjammer field") from err

    def to_line(self) -> str:
        return "\t".join(self.columns)


def read_split(path: Path) -> List[Entry]:
    entries: List[Entry] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            entries.append(Entry(tuple(stripped.split("\t"))))
    return entries


def load_all_entries(bandwidth_dir: Path) -> List[Entry]:
    entries: List[Entry] = []
    for split in ("train", "val", "test"):
        path = bandwidth_dir / f"{split}_random.txt"
        if not path.exists():
            raise FileNotFoundError(f"Missing required split file: {path}")
        entries.extend(read_split(path))
    return entries


def proportional_counts(total: int, ratios: Sequence[float]) -> List[int]:
    if total <= 0:
        return [0] * len(ratios)
    norm = sum(ratios)
    raw = [r / norm * total for r in ratios]
    counts = [int(x) for x in raw]
    remainder = total - sum(counts)
    # Use fractional parts to distribute leftover samples
    fractional = [x - int(x) for x in raw]
    order = sorted(range(len(ratios)), key=lambda idx: fractional[idx], reverse=True)
    for i in range(remainder):
        counts[order[i % len(ratios)]] += 1
    return counts


def split_dep(entries: Iterable[Entry], rng: random.Random) -> Dict[str, List[Entry]]:
    by_subjammer: Dict[str, List[Entry]] = defaultdict(list)
    for entry in entries:
        by_subjammer[entry.subjammer].append(entry)

    result = {"train": [], "val": [], "test": []}
    for subjammer, rows in by_subjammer.items():
        rows_copy = rows[:]
        rng.shuffle(rows_copy)
        train_n, val_n, test_n = proportional_counts(len(rows_copy), DEP_RATIOS)
        buckets = [train_n, val_n, test_n]
        start = 0
        for split_name, count in zip(("train", "val", "test"), buckets):
            end = start + count
            result[split_name].extend(rows_copy[start:end])
            start = end
        # Guard against rounding issues by dropping leftovers into train
        if start < len(rows_copy):
            result["train"].extend(rows_copy[start:])
    return result


def assign_subjammers_indep(
    grouped: Dict[str, List[Entry]],
    rng: random.Random,
) -> Dict[str, str]:
    items = list(grouped.items())
    rng.shuffle(items)

    total_samples = sum(len(rows) for _, rows in items)
    targets = proportional_counts(total_samples, INDEP_RATIOS)
    targets = [max(1, t) for t in targets]

    split_order = ["train", "val", "test"]
    sample_counts = {name: 0 for name in split_order}
    assignment: Dict[str, str] = {}

    for subjammer, rows in items:
        deficits = [targets[i] - sample_counts[split] for i, split in enumerate(split_order)]
        if all(d <= 0 for d in deficits):
            split_name = min(split_order, key=lambda name: sample_counts[name])
        else:
            best_idx = max(range(len(deficits)), key=lambda idx: deficits[idx])
            split_name = split_order[best_idx]
        assignment[subjammer] = split_name
        sample_counts[split_name] += len(rows)

    # Ensure that each split owns at least one subjammer group
    for split_name in split_order:
        if any(dest == split_name for dest in assignment.values()):
            continue
        donor_split = max(split_order, key=lambda name: sample_counts[name])
        donor_candidates = [sub for sub, dest in assignment.items() if dest == donor_split]
        if not donor_candidates:
            continue
        donor_sub = max(donor_candidates, key=lambda sub: len(grouped[sub]))
        assignment[donor_sub] = split_name
        sample_counts[donor_split] -= len(grouped[donor_sub])
        sample_counts[split_name] += len(grouped[donor_sub])

    return assignment


def split_indep(entries: Iterable[Entry], rng: random.Random) -> Dict[str, List[Entry]]:
    by_subjammer: Dict[str, List[Entry]] = defaultdict(list)
    for entry in entries:
        by_subjammer[entry.subjammer].append(entry)
    assignment = assign_subjammers_indep(by_subjammer, rng)
    result = {"train": [], "val": [], "test": []}
    for subjammer, dest in assignment.items():
        result[dest].extend(by_subjammer[subjammer])
    return result


def write_split(
    bandwidth_dir: Path,
    prefix: str,
    split_entries: Dict[str, List[Entry]],
) -> None:
    for split_name, rows in split_entries.items():
        path = bandwidth_dir / f"{split_name}_{prefix}.txt"
        rows_sorted = sorted(rows, key=lambda e: e.columns[0])
        with path.open("w", encoding="utf-8") as fh:
            for entry in rows_sorted:
                fh.write(entry.to_line() + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate subjammer-dependent and -independent bandwidth splits"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_BANDWIDTH_DIR,
        help="Directory containing bandwidth splits (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20240924,
        help="Random seed for reproducible partitioning",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    bandwidth_dir = args.root
    if not bandwidth_dir.exists():
        parser.error(f"Directory does not exist: {bandwidth_dir}")
        return 2

    rng = random.Random(args.seed)
    entries = load_all_entries(bandwidth_dir)
    if not entries:
        parser.error("No samples found in *_random.txt files")
        return 2

    dep_split = split_dep(entries, rng)
    indep_split = split_indep(entries, rng)

    write_split(bandwidth_dir, "subjammer_dep", dep_split)
    write_split(bandwidth_dir, "subjammer_indep", indep_split)

    print("Generated bandwidth splits:")
    for name, split_map in {"dep": dep_split, "indep": indep_split}.items():
        counts = {split: len(rows) for split, rows in split_map.items()}
        total = sum(counts.values())
        ratios = {split: counts[split] / total for split in counts}
        metrics = ", ".join(
            f"{split}={counts[split]} ({ratios[split]:.2%})" for split in ("train", "val", "test")
        )
        print(f"  {name}: {metrics}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
