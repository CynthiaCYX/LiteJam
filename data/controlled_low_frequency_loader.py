"""Utility helpers for the controlled low frequency GNSS dataset.

The dataset is stored under ``data/controlled_low_frequency/GNSS_datasets`` and
ships split files in ``info/splits/<group>`` where every line has the format::

    relative_path\tclass_id\tamplitude\tarea\tsubjammer\tposition\tenvironment

Example usage::

    from data.controlled_low_frequency_loader import (
        build_jammer_datasets,
        ControlledLowFrequencyDataset,
    )

    root = "/root/controlled_low_frequency/GNSS_datasets/dataset1"
    train_ds, val_ds, test_ds = build_jammer_datasets(root)

    print(len(train_ds))
    sample, label = train_ds[0]
    print(sample.shape, label)
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

CHANNEL_STRATEGY_TO_CHANNELS = {
    "iqfp": 6,
    "iq": 2,
    "repeat": 3,
    "single": 1,
}


def infer_input_channels(channel_strategy: str) -> int:
    """Map channel strategy identifiers to the expected tensor channel dimension."""

    try:
        return CHANNEL_STRATEGY_TO_CHANNELS[channel_strategy]
    except KeyError as exc:
        raise ValueError(f"Unsupported channel strategy: {channel_strategy}") from exc


SNR_BINS = np.arange(-10, 12, 2)  # [-10, -8, ..., 10]

DEFAULT_MIN_VALUE = -182.77
DEFAULT_MAX_VALUE = -17.12
DEFAULT_SPLIT_GROUP = "jammer"
DEFAULT_SPLIT_SUFFIX = "subjammer_dep"

_BANDWIDTH_PATTERN_SUFFIX = re.compile(r"BW([0-9]+(?:[._p][0-9]+)?)", re.IGNORECASE)
_BANDWIDTH_PATTERN_PREFIX = re.compile(r"([0-9]+(?:[._p][0-9]+)?)BW", re.IGNORECASE)


@dataclass(frozen=True)
class SplitEntry:
    """Single record from a split file."""

    relative_path: str
    class_id: int
    amplitude: float
    area: int
    subjammer: str
    position: int
    environment: int
    orientation: Optional[int] = None
    fiot_flag: Optional[int] = None

    def resolve_path(self, root: Path) -> Path:
        """Return the absolute sample path underneath ``root``."""

        return root / self.relative_path


def parse_split_file(split_path: Path) -> List[SplitEntry]:
    """Parse a split file into ``SplitEntry`` objects."""

    entries: List[SplitEntry] = []
    with split_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split("\t")
            if len(parts) < 7:
                raise ValueError(
                    f"Malformed line in {split_path}: expected at least 7 fields, got {len(parts)}"
                )
            orientation = int(parts[7]) if len(parts) > 7 and parts[7] != "" else None
            fiot_flag = int(parts[8]) if len(parts) > 8 and parts[8] != "" else None
            entries.append(
                SplitEntry(
                    relative_path=parts[0],
                    class_id=int(parts[1]),
                    amplitude=float(parts[2]),
                    area=int(parts[3]),
                    subjammer=parts[4],
                    position=int(parts[5]),
                    environment=int(parts[6]),
                    orientation=orientation,
                    fiot_flag=fiot_flag,
                )
            )
    return entries


class ControlledLowFrequencyDataset(Dataset):
    """PyTorch ``Dataset`` wrapper around the controlled low frequency splits.

    The dataset maps the available metadata onto LiteJam's multi-task heads as
    follows:

    - ``det``: binary label derived from ``class_id > 0``
    - ``type``: categorical label built from the ``subjammer`` field
    - ``snr``: categorical label from ``amplitude`` buckets
    - ``area``: categorical label from the ``area`` field
    - ``bwd``: regression target derived from the ``subjammer`` filename (BW token)

    If you have more accurate metadata (e.g. true bandwidth values) you can
    adjust the mapping logic accordingly. Samples missing the ``BW`` token are
    skipped to match the original bandwidth extraction logic.
    """

    def __init__(
        self,
        root: str | Path,
        split: str | Path,
        *,
        split_group: str = DEFAULT_SPLIT_GROUP,
        split_suffix: str = DEFAULT_SPLIT_SUFFIX,
        min_value: float = DEFAULT_MIN_VALUE,
        max_value: float = DEFAULT_MAX_VALUE,
        channel_strategy: str = "iqfp",
        transform=None,
        label_maps: Optional[Dict[str, Any]] = None,
        normalize_bwd: bool = True,
        tasks: Optional[Iterable[str]] = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.split_group = split_group
        self.split_suffix = split_suffix
        self.min_value = min_value
        self.max_value = max_value
        self.channel_strategy = channel_strategy
        self.input_channels = infer_input_channels(self.channel_strategy)
        self.transform = transform
        self.normalize_bwd = normalize_bwd
        self.tasks = set(tasks) if tasks is not None else {"det", "type", "snr", "area", "bwd"}

        split_path = self._resolve_split_path(split)
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")

        raw_entries = parse_split_file(split_path)
        entries = self._filter_existing(raw_entries)
        entries = self._filter_bandwidth(entries)
        self.entries = entries
        if not self.entries:
            raise ValueError(f"No samples found for split {split_path}")

        if label_maps is None:
            self.label_maps = self._build_label_maps(self.entries)
        else:
            self.label_maps = label_maps
            self.normalize_bwd = self.label_maps.get("normalize_bwd", self.normalize_bwd)
            self._ensure_type_labels()
            self._validate_label_maps()

        self.snr_mean = float(self.label_maps.get("snr_mean", 0.0))
        self.snr_std = float(self.label_maps.get("snr_std", 1.0))
        if self.snr_std <= 0:
            self.snr_std = 1.0

    def _resolve_split_path(self, split: str | Path) -> Path:
        split_path = Path(split)
        if split_path.is_absolute() or split_path.suffix:
            return split_path
        filename = f"{split}_{self.split_suffix}.txt"
        return self.root / "info" / "splits" / self.split_group / filename

    def _filter_existing(self, entries: Iterable[SplitEntry]) -> List[SplitEntry]:
        filtered: List[SplitEntry] = []
        missing: List[Path] = []
        for entry in entries:
            sample_path = entry.resolve_path(self.root)
            if sample_path.exists():
                filtered.append(entry)
            else:
                missing.append(sample_path)
        if missing:
            missing_str = "\n  ".join(str(p) for p in missing[:5])
            print(
                "[ControlledLowFrequency] warning: missing files (showing up to 5):\n  "
                f"{missing_str}"
            )
            if len(missing) > 5:
                print(
                    f"[ControlledLowFrequency] ... and {len(missing) - 5} more missing samples"
                )
        return filtered

    def _filter_bandwidth(self, entries: Iterable[SplitEntry]) -> List[SplitEntry]:
        return list(entries)

    def _build_label_maps(self, entries: Iterable[SplitEntry]) -> Dict[str, Any]:
        type_labels = list(range(1, 7))
        type_to_idx = {cls: idx for idx, cls in enumerate(type_labels)}
        type_counts_raw = Counter(entry.class_id for entry in entries if entry.class_id > 0)
        type_counts_idx = {type_to_idx[cls]: type_counts_raw.get(cls, 0) for cls in type_labels}
        type_classes = len(type_labels)

        area_values = sorted({entry.area for entry in entries})
        area_to_idx = {value: idx for idx, value in enumerate(area_values)}
        area_counts_raw = Counter(entry.area for entry in entries)
        area_counts_full = {area: area_counts_raw.get(area, 0) for area in area_values}

        snr_values = np.array([entry.amplitude for entry in entries], dtype=np.float32)
        if snr_values.size:
            snr_mean = float(snr_values.mean())
            snr_std = float(snr_values.std() + 1e-6)
            snr_min = float(snr_values.min())
            snr_max = float(snr_values.max())
        else:
            snr_mean = 0.0
            snr_std = 1.0
            snr_min = 0.0
            snr_max = 0.0

        bwd_values = [
            bw
            for bw in (self._extract_bandwidth(entry.subjammer) for entry in entries)
            if bw is not None
        ]
        if bwd_values:
            bwd_min = float(np.min(bwd_values))
            bwd_max = float(np.max(bwd_values))
        else:
            bwd_min = 0.0
            bwd_max = 0.0

        return {
            "type_to_idx": type_to_idx,
            "area_to_idx": area_to_idx,
            "type_classes": type_classes,
            "area_classes": len(area_to_idx),
            "det_classes": 2,
            "bwd_min": bwd_min,
            "bwd_max": bwd_max,
            "normalize_bwd": self.normalize_bwd,
            "type_counts": type_counts_idx,
            "area_counts": area_counts_full,
            "snr_mean": snr_mean,
            "snr_std": snr_std,
            "snr_min": snr_min,
            "snr_max": snr_max,
            "type_labels": type_labels,
            "area_labels": area_values,
            "channel_strategy": self.channel_strategy,
            "input_channels": self.input_channels,
        }

    def _ensure_type_labels(self) -> None:
        mapping = self.label_maps.get("type_to_idx")
        if not mapping:
            return
        updated = dict(mapping)
        next_idx = max(updated.values(), default=-1) + 1
        changed = False
        for cls in range(1, 7):
            if cls not in updated:
                updated[cls] = next_idx
                next_idx += 1
                changed = True
        if not changed:
            self.label_maps.setdefault("type_labels", sorted(updated.keys()))
            return

        self.label_maps["type_to_idx"] = updated
        counts = {int(k): v for k, v in self.label_maps.get("type_counts", {}).items()}
        for cls, idx in updated.items():
            counts.setdefault(idx, 0)
        self.label_maps["type_counts"] = counts
        self.label_maps["type_classes"] = len(updated)
        self.label_maps["type_labels"] = sorted(updated.keys())

    def _validate_label_maps(self) -> None:
        expected_strategy = self.label_maps.get("channel_strategy")
        if expected_strategy and expected_strategy != self.channel_strategy:
            raise ValueError(
                f"Provided label_maps expect channel_strategy={expected_strategy}, "
                f"but dataset was initialised with channel_strategy={self.channel_strategy}"
            )
        expected_channels = self.label_maps.get("input_channels")
        if expected_channels and expected_channels != self.input_channels:
            raise ValueError(
                f"Provided label_maps expect input_channels={expected_channels}, "
                f"but dataset was initialised with input_channels={self.input_channels}"
            )
        valid_types = set(self.label_maps["type_to_idx"].keys())
        missing_types = [entry.class_id for entry in self.entries if entry.class_id > 0 and entry.class_id not in valid_types]
        if missing_types:
            raise ValueError(
                "Provided label_maps lack class ids: " + ", ".join(str(cid) for cid in sorted(set(missing_types)))
            )
        missing_area = [
            entry.area for entry in self.entries if entry.area not in self.label_maps["area_to_idx"]
        ]
        if missing_area:
            raise ValueError(
                "Provided label_maps lack area values: " + ", ".join(str(v) for v in sorted(set(missing_area)))
            )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        entry = self.entries[index]
        array = np.load(entry.resolve_path(self.root))
        is_complex = np.iscomplexobj(array)
        complex_array = np.asarray(array, dtype=np.complex64) if is_complex else None

        if self.channel_strategy == "repeat" or self.channel_strategy == "single":
            if complex_array is not None:
                magnitude = np.abs(complex_array).astype(np.float32)
            else:
                magnitude = np.asarray(array, dtype=np.float32)
            magnitude = (magnitude - self.min_value) / (self.max_value - self.min_value)
            magnitude = np.clip(magnitude, 0.0, 1.0)
            if self.channel_strategy == "repeat":
                tensor = torch.from_numpy(magnitude).unsqueeze(0).repeat(3, 1, 1)
            else:
                tensor = torch.from_numpy(magnitude).unsqueeze(0)
        elif self.channel_strategy == "iq":
            if complex_array is None:
                real_part = np.asarray(array, dtype=np.float32)
                imag_part = np.zeros_like(real_part)
            else:
                real_part = complex_array.real.astype(np.float32)
                imag_part = complex_array.imag.astype(np.float32)
            stacked = np.stack([real_part, imag_part], axis=0)
            tensor = torch.from_numpy(stacked)
        elif self.channel_strategy == "iqfp":
            if complex_array is None:
                real_part = np.asarray(array, dtype=np.float32)
                imag_part = np.zeros_like(real_part)
                magnitude = np.abs(real_part)
            else:
                real_part = complex_array.real.astype(np.float32)
                imag_part = complex_array.imag.astype(np.float32)
                magnitude = np.abs(complex_array).astype(np.float32)
            phase = np.angle(real_part + 1j * imag_part).astype(np.float32)
            phase_sin = np.sin(phase).astype(np.float32)
            phase_cos = np.cos(phase).astype(np.float32)
            log_magnitude = np.log1p(magnitude).astype(np.float32)
            stacked = np.stack(
                [real_part, imag_part, magnitude, log_magnitude, phase_sin, phase_cos],
                axis=0,
            )
            tensor = torch.from_numpy(stacked)
        else:
            raise ValueError(f"Unsupported channel strategy: {self.channel_strategy}")

        if self.transform is not None:
            tensor = self.transform(tensor)

        det_label = torch.tensor(1 if entry.class_id > 0 else 0, dtype=torch.long)

        if "type" in self.tasks and entry.class_id > 0:
            type_label_val = self.label_maps["type_to_idx"][entry.class_id]
        else:
            type_label_val = -1
        type_label = torch.tensor(type_label_val, dtype=torch.long)

        if "snr" in self.tasks:
            snr_value = (float(entry.amplitude) - self.snr_mean) / self.snr_std
            snr_mask_val = 1
        else:
            snr_value = 0.0
            snr_mask_val = 0
        snr_norm = torch.tensor(snr_value, dtype=torch.float32)

        if "area" in self.tasks:
            area_idx = self.label_maps["area_to_idx"].get(entry.area, -1)
        else:
            area_idx = -1
        area_label = torch.tensor(area_idx, dtype=torch.long)

        bwd_raw = self._extract_bandwidth(entry.subjammer)
        bwd_valid = bwd_raw is not None and "bwd" in self.tasks
        if not bwd_valid:
            bwd_raw = 0.0
        if self.normalize_bwd and bwd_valid and self.label_maps["bwd_max"] > self.label_maps["bwd_min"]:
            denom = self.label_maps["bwd_max"] - self.label_maps["bwd_min"]
            bwd_value = (bwd_raw - self.label_maps["bwd_min"]) / denom
        else:
            bwd_value = bwd_raw
        bwd_tensor = torch.tensor(bwd_value, dtype=torch.float32)

        targets = {
            "det": det_label,
            "type": type_label,
            "snr": snr_norm,
            "area": area_label,
            "bwd": bwd_tensor,
            "bwd_mask": torch.tensor(1 if bwd_valid else 0, dtype=torch.bool),
            "snr_mask": torch.tensor(snr_mask_val, dtype=torch.bool),
        }

        return tensor, targets

    @staticmethod
    def _extract_bandwidth(subjammer: str) -> Optional[float]:
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

    @staticmethod
    def _format_snr(value: float) -> str:
        return f"{int(round(value)):+d}"


def build_group_datasets(
    root: str | Path,
    *,
    split_group: str,
    split_suffix: str = DEFAULT_SPLIT_SUFFIX,
    tasks: Optional[Iterable[str]] = None,
    transforms: Optional[Dict[str, Callable[[torch.Tensor], torch.Tensor]]] = None,
    dataset_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[ControlledLowFrequencyDataset, ControlledLowFrequencyDataset, ControlledLowFrequencyDataset]:
    """Build train/val/test datasets for a specific split group and task subset."""

    dataset_kwargs = dict(dataset_kwargs or {})
    dataset_kwargs.setdefault("channel_strategy", "iqfp")
    transforms = transforms or {}

    train_ds = ControlledLowFrequencyDataset(
        root=root,
        split="train",
        split_group=split_group,
        split_suffix=split_suffix,
        transform=transforms.get("train"),
        tasks=tasks,
        **dataset_kwargs,
    )
    label_maps = train_ds.label_maps

    val_ds = ControlledLowFrequencyDataset(
        root=root,
        split="val",
        split_group=split_group,
        split_suffix=split_suffix,
        transform=transforms.get("val"),
        label_maps=label_maps,
        tasks=tasks,
        **dataset_kwargs,
    )
    test_ds = ControlledLowFrequencyDataset(
        root=root,
        split="test",
        split_group=split_group,
        split_suffix=split_suffix,
        transform=transforms.get("test"),
        label_maps=label_maps,
        tasks=tasks,
        **dataset_kwargs,
    )
    return train_ds, val_ds, test_ds


def build_jammer_datasets(
    root: str | Path,
    *,
    split_group: str = DEFAULT_SPLIT_GROUP,
    split_suffix: str = DEFAULT_SPLIT_SUFFIX,
    transforms: Optional[Dict[str, Callable[[torch.Tensor], torch.Tensor]]] = None,
    **dataset_kwargs,
) -> Tuple[ControlledLowFrequencyDataset, ControlledLowFrequencyDataset, ControlledLowFrequencyDataset]:
    return build_group_datasets(
        root,
        split_group=split_group,
        split_suffix=split_suffix,
        tasks=dataset_kwargs.pop("tasks", None),
        transforms=transforms,
        dataset_kwargs=dataset_kwargs,
    )


def main() -> None:
    """Small demo that prints dataset statistics."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Preview controlled low frequency GNSS splits"
    )
    default_root = (
        Path(__file__).resolve().parent
        / "controlled_low_frequency"
        / "GNSS_datasets"
        / "dataset1"
    )
    try:
        default_display = default_root.relative_to(Path.cwd())
    except ValueError:
        default_display = default_root
    parser.add_argument(
        "root",
        nargs="?",
        default=default_root,
        type=Path,
        help=(
            "Dataset root directory (default: "
            f"{default_display})"
        ),
    )
    parser.add_argument(
        "--split-group",
        default=DEFAULT_SPLIT_GROUP,
        help="Split directory under info/splits (default: jammer)",
    )
    parser.add_argument(
        "--split-suffix",
        default=DEFAULT_SPLIT_SUFFIX,
        help="Split filename suffix (default: subjammer_dep)",
    )
    args = parser.parse_args()

    datasets = build_jammer_datasets(
        root=args.root,
        split_group=args.split_group,
        split_suffix=args.split_suffix,
    )
    names = ["train", "val", "test"]
    for name, dataset in zip(names, datasets):
        tensor, label = dataset[0]
        print(
            f"{name}: {len(dataset):6d} samples | first tensor {tuple(tensor.shape)} | label {label}"
        )


if __name__ == "__main__":
    main()
