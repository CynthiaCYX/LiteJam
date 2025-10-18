"""MobileNetV3-S baseline for LiteJam multi-task benchmarking."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights

from models.litejam import HeadsConfig


@dataclass
class MobileNetV3SBaselineConfig:
    """Configuration for the MobileNetV3-S baseline."""

    in_channels: int = 6
    dropout: float = 0.2
    pretrained: bool = False
    freeze_backbone: bool = False
    head_config: Optional[HeadsConfig] = None


class MobileNetV3SBaseline(nn.Module):
    """MobileNetV3-S backbone with task-specific heads."""

    def __init__(self, config: MobileNetV3SBaselineConfig) -> None:
        super().__init__()
        self.config = config
        head_cfg = config.head_config or HeadsConfig()

        if config.pretrained and config.in_channels != 3:
            raise ValueError(
                "Cannot use pretrained weights with in_channels != 3 for MobileNetV3-S baseline"
            )

        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if config.pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)

        first_conv = backbone.features[0][0]
        if config.in_channels != first_conv.in_channels:
            new_conv = nn.Conv2d(
                config.in_channels,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=False,
            )
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
            backbone.features[0][0] = new_conv

        self.feature_dim = backbone.classifier[0].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone

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


__all__ = ["MobileNetV3SBaselineConfig", "MobileNetV3SBaseline"]
