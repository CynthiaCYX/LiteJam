"""Loss computation for LiteJam multi-task training."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor


class LiteJamLoss(nn.Module):
    """Aggregate loss across five tasks with optional adaptive weighting."""

    def __init__(
        self,
        weights: Dict[str, float],
        scheme: str = "static",
        class_weights: Dict[str, Tensor] | None = None,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.weights = weights
        self.scheme = scheme
        self.class_weights = class_weights or {}
        self.focal_gamma = focal_gamma
        self.mse = nn.MSELoss(reduction="mean")

    def _classification_loss(self, logits: Tensor, targets: Tensor, task: str) -> Tensor:
        weight_tensor = self.class_weights.get(task)
        if weight_tensor is not None:
            weight_tensor = weight_tensor.to(logits.device)

        if self.scheme == "focal":
            ce = nn.functional.cross_entropy(
                logits, targets, reduction="none", weight=weight_tensor
            )
            pt = torch.exp(-ce)
            loss = ((1 - pt) ** self.focal_gamma * ce).mean()
        else:
            loss = nn.functional.cross_entropy(
                logits, targets, reduction="mean", weight=weight_tensor
            )
        return loss

    def forward(self, outputs: Dict[str, Tensor], targets: Dict[str, Tensor]) -> Dict[str, Tensor]:
        det_logits = outputs["det"]
        det_targets = targets["det"]
        det_loss = self._classification_loss(det_logits, det_targets, "det") * self.weights["det"]

        mask = det_targets == 1
        results: Dict[str, Tensor] = {"det": det_loss}

        type_targets = targets["type"]
        valid_type = (type_targets >= 0) & mask

        if valid_type.any():
            type_loss = self._classification_loss(
                outputs["type"][valid_type], type_targets[valid_type], "type"
            ) * self.weights["type"]
        else:
            zero = det_loss.new_tensor(0.0)
            type_loss = zero

        snr_mask = targets.get("snr_mask")
        if snr_mask is not None:
            valid_snr = mask & snr_mask
        else:
            valid_snr = mask
        if valid_snr.any():
            snr_loss = self.mse(
                outputs["snr"][valid_snr], targets["snr"][valid_snr].float()
            ) * self.weights["snr"]
        else:
            snr_loss = det_loss.new_tensor(0.0)

        area_targets = targets["area"]
        valid_area = mask & (area_targets >= 0)
        if valid_area.any():
            area_loss = self._classification_loss(
                outputs["area"][valid_area], area_targets[valid_area], "area"
            ) * self.weights["area"]
        else:
            area_loss = det_loss.new_tensor(0.0)

        bwd_mask = targets.get("bwd_mask")
        if bwd_mask is not None:
            bwd_valid = mask & bwd_mask
        else:
            bwd_valid = mask

        if bwd_valid.any():
            bwd_loss = self.mse(
                outputs["bwd"][bwd_valid].squeeze(-1), targets["bwd"][bwd_valid].float()
            ) * self.weights["bwd"]
        else:
            bwd_loss = det_loss.new_tensor(0.0)

        results["type"] = type_loss
        results["snr"] = snr_loss
        results["area"] = area_loss
        results["bwd"] = bwd_loss

        total = det_loss + type_loss + snr_loss + area_loss + bwd_loss
        results["total"] = total
        return results


__all__ = ["LiteJamLoss"]
