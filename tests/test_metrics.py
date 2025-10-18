"""Metric unit tests for LiteJam."""
from __future__ import annotations

import numpy as np

from train.metrics import classification_metrics, evaluate_outputs, regression_metrics


def test_classification_metrics_perfect() -> None:
    targets = np.array([0, 1, 1, 0])
    preds = np.array([0, 1, 1, 0])
    metrics = classification_metrics(targets, preds)
    assert metrics["acc"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["f1"] == 1.0


def test_regression_metrics_basic() -> None:
    targets = np.array([0.0, 1.0, 2.0])
    preds = np.array([0.1, 0.9, 1.8])
    metrics = regression_metrics(targets, preds)
    assert metrics["rmse"] >= 0.0
    assert metrics["mae"] >= 0.0


def test_evaluate_outputs_masking() -> None:
    targets = {
        "det": np.array([0, 1]),
        "type": np.array([0, 1]),
        "snr": np.array([0, 1]),
        "area": np.array([0, 1]),
        "bwd": np.array([0.0, 1.0]),
    }
    preds = {
        "det": np.array([0, 1]),
        "type": np.array([0, 1]),
        "snr": np.array([0, 1]),
        "area": np.array([0, 1]),
        "bwd": np.array([0.0, 1.0]),
    }
    mask = np.array([False, True])
    metrics = evaluate_outputs(targets, preds, mask)
    assert metrics["det"]["acc"] == 1.0
    assert metrics["type"]["acc"] == 1.0
    assert metrics["bwd"]["rmse"] == 0.0
