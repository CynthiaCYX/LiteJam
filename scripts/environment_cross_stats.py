#!/usr/bin/env python3
"""Summarise label combinations and bandwidth values for a single environment file.

Usage::

    python scripts/environment_cross_stats.py

Update ``ENVIRONMENT_ROOT`` or ``DEFAULT_ENVIRONMENT`` below to point at a
different dataset, or pass ``--environment`` on the command line to override the
default file.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Set, Tuple


# Dataset root; adjust this constant to point at a different location.
ENVIRONMENT_ROOT = Path("data/controlled_low_frequency/GNSS_datasets/dataset1/info/splits/environment_cross")

DEFAULT_ENVIRONMENT = "env_03.txt"


# Keep the regex logic consistent with ``data.controlled_low_frequency_loader``.
_BANDWIDTH_PATTERN_SUFFIX = re.compile(r"BW([0-9]+(?:[._p][0-9]+)?)", re.IGNORECASE)
_BANDWIDTH_PATTERN_PREFIX = re.compile(r"([0-9]+(?:[._p][0-9]+)?)BW", re.IGNORECASE)


def extract_bandwidth(subjammer: str) -> float | None:
    """Parse the bandwidth token from a ``subjammer`` string."""

    if not subjammer or subjammer.lower() == "none":
        return None
    match = _BANDWIDTH_PATTERN_SUFFIX.search(subjammer)
    if not match:
        match = _BANDWIDTH_PATTERN_PREFIX.search(subjammer)
        if not match:
            return None
    token = match.group(1).replace("_", ".").replace("p", ".")
    try:
        return float(token)
    except ValueError:
        return None


def iter_rows(path: Path) -> Iterable[Tuple[int, float, int, str]]:
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, 1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                raise ValueError(f"{path}: line {lineno} has fewer than 5 fields")
            try:
                class_id = int(parts[1])
            except ValueError as err:
                raise ValueError(f"{path}: line {lineno} has a non-integer class_id") from err
            try:
                amplitude = float(parts[2])
            except ValueError as err:
                raise ValueError(f"{path}: line {lineno} has a non-float amplitude") from err
            try:
                area = int(parts[3])
            except ValueError as err:
                raise ValueError(f"{path}: line {lineno} has a non-integer area") from err
            subjammer = parts[4]
            yield class_id, amplitude, area, subjammer


def analyze(path: Path) -> Tuple[int, Set[int], Set[float], Set[int], Set[float]]:
    total = 0
    class_ids: Set[int] = set()
    amplitudes: Set[float] = set()
    areas: Set[int] = set()
    bwd_values: Set[float] = set()

    for class_id, amplitude, area, subjammer in iter_rows(path):
        total += 1
        class_ids.add(class_id)
        amplitudes.add(amplitude)
        areas.add(area)
        bwd = extract_bandwidth(subjammer)
        if bwd is not None:
            bwd_values.add(bwd)

    return total, class_ids, amplitudes, areas, bwd_values


def resolve_env_path(env: str) -> Path:
    candidate = Path(env)
    if candidate.exists():
        return candidate
    filename = f"{env}.txt" if not env.endswith(".txt") else env
    full_path = ENVIRONMENT_ROOT / filename
    if full_path.exists():
        return full_path
    raise FileNotFoundError(f"Environment file not found: {env}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarise an environment_cross label file")
    parser.add_argument(
        "environment",
        nargs="?",
        default=DEFAULT_ENVIRONMENT,
        help="Environment file path or name (default: %(default)s)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        env_path = resolve_env_path(args.environment)
    except FileNotFoundError as err:
        parser.error(str(err))
        return 2

    total, class_ids, amplitudes, areas, bwd_values = analyze(env_path)
    print(f"Environment file: {env_path}")
    print(f"Sample count: {total}")

    print("\nUnique class_id values (column 2):")
    for value in sorted(class_ids):
        print(f"  {value}")

    print("\nUnique amplitude values (column 3):")
    for value in sorted(amplitudes):
        print(f"  {value}")

    print("\nUnique area values (column 4):")
    for value in sorted(areas):
        print(f"  {value}")

    print("\nUnique bandwidth values (derived from subjammer):")
    if bwd_values:
        for value in sorted(bwd_values):
            print(f"  {value}")
    else:
        print("  <no parseable bandwidth found>")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
