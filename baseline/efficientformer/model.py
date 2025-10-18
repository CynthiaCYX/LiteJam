"""EfficientFormer-L1 baseline for LiteJam multi-task benchmarking."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    import timm  # type: ignore[import]
except ImportError:  # pragma: no cover - optional dependency
    timm = None  # type: ignore[assignment]

from models.litejam import HeadsConfig
from models.utils import infer_backbone_feature_dim


def _div_ceil(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _reset_attention_bias(module: nn.Module, hw: Tuple[int, int]) -> None:
    if not hasattr(module, "attention_biases") or not hasattr(module, "attention_bias_idxs"):
        return

    target_tokens = hw[0] * hw[1]
    bias: torch.nn.Parameter = getattr(module, "attention_biases")  # type: ignore[assignment]
    if bias.shape[1] == target_tokens:
        return

    device = bias.device
    dtype = bias.dtype
    num_heads = bias.shape[0]

    new_bias = torch.zeros(num_heads, target_tokens, device=device, dtype=dtype)
    module.register_parameter("attention_biases", nn.Parameter(new_bias))

    grid_y = torch.arange(hw[0], device=device)
    grid_x = torch.arange(hw[1], device=device)
    pos = torch.stack(torch.meshgrid(grid_y, grid_x, indexing="ij")).flatten(1)
    rel_pos = (pos[..., :, None] - pos[..., None, :]).abs()
    rel_pos = (rel_pos[0] * hw[1]) + rel_pos[1]

    module._buffers.pop("attention_bias_idxs", None)
    module.register_buffer("attention_bias_idxs", rel_pos, persistent=False)

    cache = getattr(module, "attention_bias_cache", None)
    if isinstance(cache, dict):
        cache.clear()


@dataclass
class EfficientFormerL1BaselineConfig:
    """Configuration for the EfficientFormer-L1 baseline."""

    in_channels: int = 6
    dropout: float = 0.2
    pretrained: bool = False
    freeze_backbone: bool = False
    head_config: Optional[HeadsConfig] = None
    img_size: Union[int, Tuple[int, int]] = 224


class EfficientFormerL1Baseline(nn.Module):
    """EfficientFormer-L1 backbone with task-specific heads."""

    def __init__(self, config: EfficientFormerL1BaselineConfig) -> None:
        super().__init__()
        self.config = config
        head_cfg = config.head_config or HeadsConfig()

        if timm is None:
            raise ImportError(
                "EfficientFormer baseline requires the 'timm' package. Install via `pip install timm`."
            )

        if config.pretrained and config.in_channels != 3:
            raise ValueError(
                "Cannot use pretrained weights with in_channels != 3 for EfficientFormer baseline"
            )

        if isinstance(config.img_size, tuple):
            input_hw = tuple(int(dim) for dim in config.img_size)
        else:
            edge = int(config.img_size)
            input_hw = (edge, edge)

        img_size_arg = input_hw[0] if input_hw[0] == input_hw[1] else max(input_hw)

        backbone = timm.create_model(
            "efficientformer_l1",
            pretrained=config.pretrained,
            in_chans=config.in_channels,
            num_classes=0,
            global_pool="avg",
            img_size=img_size_arg,
        )
        default_cfg = dict(getattr(backbone, "default_cfg", {}) or {})
        default_cfg["input_size"] = (config.in_channels, input_hw[0], input_hw[1])
        default_cfg["img_size"] = input_hw
        backbone.default_cfg = default_cfg

        stem_stride = getattr(backbone.stem, "stride", 4)
        height = _div_ceil(input_hw[0], stem_stride)
        width = _div_ceil(input_hw[1], stem_stride)

        attn_hw = (height, width)
        for stage in backbone.stages:
            if not isinstance(stage.downsample, nn.Identity):
                height = _div_ceil(height, 2)
                width = _div_ceil(width, 2)
            if any(
                hasattr(block, "token_mixer")
                and hasattr(block.token_mixer, "attention_biases")
                for block in getattr(stage, "blocks", [])
            ):
                attn_hw = (height, width)
        for module in backbone.modules():
            _reset_attention_bias(module, attn_hw)

        self.input_hw = input_hw
        self.backbone = backbone
        self.feature_dim = infer_backbone_feature_dim(backbone, config.in_channels)

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
        if inputs.shape[-2:] != self.input_hw:
            inputs = F.interpolate(
                inputs,
                size=self.input_hw,
                mode="bilinear",
                align_corners=False,
            )

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


__all__ = ["EfficientFormerL1BaselineConfig", "EfficientFormerL1Baseline"]
