"""MobileOne-S0 baseline for LiteJam multi-task benchmarking."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

try:
    import timm  # type: ignore[import]
except ImportError:  # pragma: no cover - optional dependency
    timm = None  # type: ignore[assignment]

from models.litejam import HeadsConfig


@dataclass
class MobileOneS0BaselineConfig:
    """Configuration for the MobileOne-S0 baseline."""

    in_channels: int = 6
    dropout: float = 0.2
    pretrained: bool = False
    freeze_backbone: bool = False
    head_config: Optional[HeadsConfig] = None


class MobileOneS0Baseline(nn.Module):
    """MobileOne-S0 backbone with task-specific heads."""

    def __init__(self, config: MobileOneS0BaselineConfig) -> None:
        super().__init__()
        self.config = config
        head_cfg = config.head_config or HeadsConfig()

        if timm is None:
            raise ImportError(
                "MobileOne baseline requires the 'timm' package. Install via `pip install timm`."
            )

        if config.pretrained and config.in_channels != 3:
            raise ValueError(
                "Cannot use pretrained weights with in_channels != 3 for MobileOne baseline"
            )

        backbone = timm.create_model(
            "mobileone_s0",
            pretrained=config.pretrained,
            in_chans=config.in_channels,
            num_classes=0,
            global_pool="avg",
        )
        self.backbone = backbone
        self.feature_dim = backbone.num_features

        self.dropout = nn.Dropout(config.dropout)
        self.det_head = nn.Linear(self.feature_dim, head_cfg.det_classes)
        self.type_head = nn.Linear(self.feature_dim, head_cfg.type_classes)
        self.snr_head = nn.Linear(self.feature_dim, 1)
        self.area_head = nn.Linear(self.feature_dim, head_cfg.area_classes)
        self.bwd_head = nn.Linear(self.feature_dim, 1)

        if config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(
        self,
        inputs: Tensor,
        det_labels: Optional[Tensor] = None,
        use_pred_mask: bool = False,
    ) -> Dict[str, Tensor]:
        features = self.backbone(inputs)
        features = self.dropout(features)

        det_logits = self.det_head(features)
        type_logits = self.type_head(features)
        snr_pred = self.snr_head(features).squeeze(-1)
        area_logits = self.area_head(features)
        bwd_pred = self.bwd_head(features)

        outputs: Dict[str, Tensor] = {
            "det": det_logits,
            "type": type_logits,
            "snr": snr_pred,
            "area": area_logits,
            "bwd": bwd_pred,
        }

        if det_labels is not None:
            outputs["mask"] = det_labels.bool()
        else:
            prob_det = torch.softmax(det_logits, dim=1)[:, 1]
            outputs["mask"] = prob_det > 0.5
        return outputs


__all__ = ["MobileOneS0BaselineConfig", "MobileOneS0Baseline"]
