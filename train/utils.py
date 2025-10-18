"""Utility helpers shared across LiteJam training scripts."""
from __future__ import annotations

from typing import Dict

import torch


TASK_ORDER = ("det", "type", "snr", "area", "bwd")


def parse_task_weight_spec(spec: str, base: Dict[str, float]) -> Dict[str, float]:
    """Parse a comma/space separated weight spec into a new dict.

    Args:
        spec: String like ``"det=1.0,type=0.5 area=2"``. Empty/"auto" keeps defaults.
        base: Default weights to start from.
    """

    weights = dict(base)

    if not spec:
        return weights

    spec = spec.strip()
    if spec.lower() in {"auto", "default"}:
        return weights

    tokens = []
    for part in spec.replace(",", " ").split():
        part = part.strip()
        if part:
            tokens.append(part)

    for token in tokens:
        if "=" not in token:
            raise ValueError(f"Invalid task weight token '{token}' (expected key=value)")
        name, value = token.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Invalid task name in token '{token}'")
        try:
            parsed = float(value)
        except ValueError as exc:  # pragma: no cover - CLI validation path
            raise ValueError(f"Invalid weight value in token '{token}'") from exc
        weights[name] = parsed
    return weights


def format_task_weights(weights: Dict[str, float]) -> str:
    """Deterministically format task weights for logging."""

    ordered = []
    for task in TASK_ORDER:
        if task in weights:
            ordered.append(f"{task}={weights[task]:.4g}")
    for task, value in weights.items():
        if task not in TASK_ORDER:
            ordered.append(f"{task}={value:.4g}")
    return ", ".join(ordered)

def resolve_device(requested: str) -> str:
    """Resolve a device string, auto-detecting cuda/mps when requested."""

    choice = (requested or "auto").lower()
    if choice != "auto":
        return choice

    try:
        if torch.cuda.is_available():
            return "cuda"
        backends = getattr(torch, "backends", None)
        mps = getattr(backends, "mps", None)
        if mps is not None and callable(getattr(mps, "is_available", None)) and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


__all__ = ["TASK_ORDER", "parse_task_weight_spec", "format_task_weights", "resolve_device"]
