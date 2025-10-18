"""Metric utilities for LiteJam."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)


def classification_metrics(targets: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    """Compute accuracy, precision, recall and F1 score."""

    return {
        "acc": accuracy_score(targets, preds),
        "precision": precision_score(targets, preds, average="weighted", zero_division=0),
        "recall": recall_score(targets, preds, average="weighted", zero_division=0),
        "f1": f1_score(targets, preds, average="weighted", zero_division=0),
    }


def regression_metrics(targets: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    """Compute RMSE, MAE and R2."""

    diff = targets - preds
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = mean_absolute_error(targets, preds)
    r_two = r2_score(targets, preds)
    return {"rmse": rmse, "mae": mae, "r2": r_two}


def evaluate_outputs(
    targets: Dict[str, np.ndarray],
    preds: Dict[str, np.ndarray],
    mask: np.ndarray,
    task_masks: Dict[str, np.ndarray] | None = None,
) -> Dict[str, Dict[str, float]]:
    """Compute metrics for all tasks given numpy arrays."""

    results: Dict[str, Dict[str, float]] = {}
    results["det"] = classification_metrics(targets["det"], preds["det"])

    if mask.any():
        base_mask = mask.astype(bool)
        task_masks = task_masks or {}

        def active_mask(name: str) -> np.ndarray:
            extra = task_masks.get(name)
            if extra is None:
                return base_mask
            return base_mask & extra.astype(bool)

        type_active = active_mask("type")
        snr_active = active_mask("snr")
        area_active = active_mask("area")
        bwd_active = active_mask("bwd")

        if type_active.any():
            results["type"] = classification_metrics(targets["type"][type_active], preds["type"][type_active])
        else:
            results["type"] = {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

        if snr_active.any():
            results["snr"] = regression_metrics(targets["snr"][snr_active], preds["snr"][snr_active])
        else:
            results["snr"] = {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}

        if area_active.any():
            results["area"] = classification_metrics(targets["area"][area_active], preds["area"][area_active])
        else:
            results["area"] = {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

        if bwd_active.any():
            results["bwd"] = regression_metrics(targets["bwd"][bwd_active], preds["bwd"][bwd_active])
        else:
            results["bwd"] = {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
    else:
        results["type"] = {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        results["snr"] = {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
        results["area"] = {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        results["bwd"] = {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
    return results


__all__ = ["classification_metrics", "regression_metrics", "evaluate_outputs"]
