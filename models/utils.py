"""Utility helpers for LiteJam model implementations."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


def infer_backbone_feature_dim(backbone: nn.Module, in_channels: int) -> int:
    """Infer the flattened feature dimension produced by a timm backbone.

    The helper first trusts ``backbone.num_features`` when available, but will
    fall back to (or validate against) a cheap dummy forward pass to cover
    backbones where the attribute is out of sync with the actual output shape.
    """

    candidate = getattr(backbone, "num_features", None)
    candidate_dim = int(candidate) if candidate is not None else 0

    def _dummy_input() -> torch.Tensor:
        default_cfg: Dict[str, object] = getattr(backbone, "default_cfg", {}) or {}
        input_size = default_cfg.get("input_size")  # expected (C, H, W)
        height = width = 224  # sensible default for vision backbones
        if isinstance(input_size, (tuple, list)) and len(input_size) == 3:
            _, height, width = input_size
        elif isinstance(default_cfg.get("img_size"), (tuple, list)):
            size = default_cfg["img_size"]
            height, width = size[0], size[1]
        elif isinstance(default_cfg.get("img_size"), int):
            height = width = int(default_cfg["img_size"])
        dummy = torch.zeros(1, in_channels, height, width)
        first_param = next(backbone.parameters(), None)
        if first_param is not None:
            dummy = dummy.to(first_param.device).to(first_param.dtype)
        return dummy

    actual_dim = 0
    was_training = backbone.training
    try:
        if was_training:
            backbone.eval()
        with torch.no_grad():
            dummy = _dummy_input()
            output = backbone(dummy)
        if output.ndim < 2:
            raise RuntimeError("Backbone output must be at least 2D")
        actual_dim = int(output.shape[-1])
    except Exception:
        actual_dim = 0
    finally:
        backbone.train(was_training)

    if actual_dim and candidate_dim and actual_dim != candidate_dim:
        return actual_dim
    if actual_dim:
        return actual_dim
    if candidate_dim:
        return candidate_dim
    raise RuntimeError("Unable to infer feature dimension for backbone")


__all__ = ["infer_backbone_feature_dim"]
