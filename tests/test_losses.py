"""Tests for LiteJam loss aggregation."""
from __future__ import annotations

import torch

from train.losses import LiteJamLoss


def make_outputs(batch: int) -> dict:
    return {
        "det": torch.randn(batch, 2, requires_grad=True),
        "type": torch.randn(batch, 6, requires_grad=True),
        "snr": torch.randn(batch, 10, requires_grad=True),
        "area": torch.randn(batch, 4, requires_grad=True),
        "bwd": torch.randn(batch, 1, requires_grad=True),
        "mask": torch.ones(batch, dtype=torch.bool),
    }


def test_losses_masked_zero() -> None:
    loss_fn = LiteJamLoss({"det": 1.0, "type": 1.0, "snr": 1.0, "area": 1.0, "bwd": 1.0})
    outputs = make_outputs(3)
    targets = {
        "det": torch.zeros(3, dtype=torch.long),
        "type": torch.zeros(3, dtype=torch.long),
        "snr": torch.zeros(3, dtype=torch.long),
        "area": torch.zeros(3, dtype=torch.long),
        "bwd": torch.zeros(3, dtype=torch.float32),
    }
    losses = loss_fn(outputs, targets)
    assert losses["type"].item() == 0.0
    assert losses["snr"].item() == 0.0
    assert losses["area"].item() == 0.0
    assert losses["bwd"].item() == 0.0


def test_losses_positive_when_unmasked() -> None:
    loss_fn = LiteJamLoss({"det": 1.0, "type": 1.0, "snr": 1.0, "area": 1.0, "bwd": 1.0})
    outputs = make_outputs(4)
    targets = {
        "det": torch.tensor([1, 1, 1, 1], dtype=torch.long),
        "type": torch.zeros(4, dtype=torch.long),
        "snr": torch.zeros(4, dtype=torch.long),
        "area": torch.zeros(4, dtype=torch.long),
        "bwd": torch.zeros(4, dtype=torch.float32),
    }
    losses = loss_fn(outputs, targets)
    assert losses["type"].item() > 0.0
    assert losses["snr"].item() > 0.0
    assert losses["area"].item() > 0.0
    assert losses["bwd"].item() >= 0.0
