"""Forward pass tests for LiteJam model."""
from __future__ import annotations

import torch

from models.litejam import HeadsConfig, LiteJamConfig, LiteJamModel


def test_forward_shapes() -> None:
    config = LiteJamConfig(
        d_model=64,
        n_heads=4,
        ffn_dim=128,
        dropout=0.0,
        top_k=8,
        num_transformer_layers=1,
        head_config=HeadsConfig(det_classes=2, type_classes=6, area_classes=4),
    )
    model = LiteJamModel(config)
    batch = torch.randn(2, 3, 1024, 128)
    det_labels = torch.tensor([0, 1])
    outputs = model(batch, det_labels=det_labels, use_pred_mask=False)

    assert outputs["det"].shape == (2, 2)
    assert outputs["type"].shape == (2, 6)
    assert outputs["snr"].shape == (2,)
    assert outputs["area"].shape == (2, 4)
    assert outputs["bwd"].shape == (2, 1)
    assert outputs["mask"].dtype == torch.bool
    assert torch.equal(outputs["mask"], det_labels.bool())


def test_forward_pred_mask() -> None:
    config = LiteJamConfig(
        d_model=64,
        n_heads=4,
        ffn_dim=128,
        dropout=0.0,
        top_k=8,
        num_transformer_layers=1,
        head_config=HeadsConfig(det_classes=2, type_classes=6, area_classes=4),
    )
    model = LiteJamModel(config)
    batch = torch.randn(2, 3, 1024, 128)
    outputs = model(batch, det_labels=None, use_pred_mask=True)
    assert outputs["mask"].shape == (2,)
