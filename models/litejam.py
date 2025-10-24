"""LiteJam model: backbone, transformer and multi-task heads in one module."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import warnings


def _sincos_2d_pos_embedding(dim: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    if dim % 4 != 0:
        raise ValueError("Positional embedding dimension must be divisible by 4")

    dim_quarter = dim // 4
    omega = torch.arange(dim_quarter, device=device, dtype=dtype)
    div_term = torch.pow(10000.0, omega / dim_quarter)

    y = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")

    pos_y = grid_y.unsqueeze(-1) / div_term
    pos_x = grid_x.unsqueeze(-1) / div_term

    pos = torch.cat(
        [torch.sin(pos_y), torch.cos(pos_y), torch.sin(pos_x), torch.cos(pos_x)],
        dim=-1,
    )
    return pos.permute(2, 0, 1)


@dataclass
class DSAConfig:
    """Configuration for Dynamic Sparse Attention."""

    d_model: int
    n_heads: int
    top_k: int
    dropout: float = 0.1
    threshold: Optional[float] = None
    min_k: int = 1


class DynamicSparseAttention(nn.Module):
    """Dynamic sparse attention using top-k token selection for keys and values."""

    def __init__(self, config: DSAConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.top_k = config.top_k
        self.dropout = nn.Dropout(config.dropout)
        self.threshold = config.threshold
        self.min_k = max(1, int(config.min_k))

        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)

        hidden = max(self.d_model // 4, 1)
        self.importance = nn.Sequential(
            nn.Linear(self.d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.n_heads),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        bsz, length, dim = tokens.shape
        if dim != self.d_model:
            raise ValueError(f"Expected embedding dim {self.d_model}, got {dim}")

        top_k = length if self.top_k <= 0 or self.top_k > length else self.top_k
        top_k = max(top_k, self.min_k)

        importance_scores = self.importance(tokens)  # (B, L, H)
        scores = importance_scores.permute(0, 2, 1)  # (B, H, L)
        topk_scores, topk_indices = torch.topk(scores, top_k, dim=-1)  # (B, H, K)

        keep_mask: Tensor | None
        if self.threshold is not None:
            keep_mask = topk_scores > self.threshold
            if keep_mask.size(-1) > 0:
                min_keep = min(self.min_k, keep_mask.size(-1))
                if min_keep > 0:
                    keep_mask[..., :min_keep] = True
        else:
            keep_mask = None

        q = self.q_proj(tokens)
        k = self.k_proj(tokens)
        v = self.v_proj(tokens)

        head_dim = self.d_model // self.n_heads
        if head_dim * self.n_heads != self.d_model:
            raise ValueError("d_model must be divisible by n_heads")

        q = q.view(bsz, length, self.n_heads, head_dim).permute(0, 2, 1, 3)  # (B, H, L, D)
        k = k.view(bsz, length, self.n_heads, head_dim).permute(0, 2, 1, 3)  # (B, H, L, D)
        v = v.view(bsz, length, self.n_heads, head_dim).permute(0, 2, 1, 3)  # (B, H, L, D)

        gather_idx = topk_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)  # (B, H, K, D)
        k_selected = torch.gather(k, dim=2, index=gather_idx)  # (B, H, K, D)
        v_selected = torch.gather(v, dim=2, index=gather_idx)  # (B, H, K, D)

        attn_logits = torch.matmul(q, k_selected.transpose(-2, -1)) / (head_dim**0.5)  # (B, H, L, K)
        if keep_mask is not None:
            attn_logits = attn_logits.masked_fill(~keep_mask.unsqueeze(2), float("-inf"))
        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v_selected)  # (B, H, L, D)
        out = out.permute(0, 2, 1, 3).contiguous().view(bsz, length, dim)
        return out


class TransformerBlock(nn.Module):
    """Transformer encoder block with Dynamic Sparse Attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float,
        top_k: int,
        threshold: Optional[float] = None,
        min_k: int = 1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = DynamicSparseAttention(
            DSAConfig(
                d_model=d_model,
                n_heads=n_heads,
                top_k=top_k,
                dropout=dropout,
                threshold=threshold,
                min_k=min_k,
            )
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        tokens = tokens + self.attn(self.norm1(tokens))
        tokens = tokens + self.ff(self.norm2(tokens))
        return tokens


class TransformerBlockDense(nn.Module):
    """Transformer block using standard multi-head attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        residual = tokens
        tokens = self.norm1(tokens)
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = residual + attn_out
        tokens = tokens + self.ff(self.norm2(tokens))
        return tokens


@dataclass
class HeadsConfig:
    det_classes: int = 2
    type_classes: int = 6
    area_classes: int = 4


class HierarchicalHeads(nn.Module):
    """Multi-task heads fed with task-specific high-level features."""

    def __init__(self, in_dim: int, config: HeadsConfig) -> None:
        super().__init__()
        self._task_names = ["det", "type", "snr", "area", "bwd"]
        self._proj_alias = {name: f"proj_{name}" for name in self._task_names}
        hidden = max(in_dim // 2, 128)

        def _branch() -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, in_dim),
            )

        self.projectors = nn.ModuleDict({alias: _branch() for alias in self._proj_alias.values()})

        self.det_head = nn.Linear(in_dim, config.det_classes)
        self.type_head = nn.Linear(in_dim, config.type_classes)
        self.snr_head = nn.Linear(in_dim, 1)
        self.area_head = nn.Linear(in_dim, config.area_classes)
        self.bwd_head = nn.Linear(in_dim, 1)

        self._head_modules = {
            name: nn.ModuleList([self.projectors[self._proj_alias[name]], getattr(self, f"{name}_head")])
            for name in self._task_names
        }

    def forward(
        self,
        features: Dict[str, Tensor],
        det_mask: Tensor | None = None,
        use_pred_mask: bool = False,
        mask_mode: str = "hard",
        mask_threshold: float = 0.5,
        tau: float = 1.0,
    ) -> Dict[str, Tensor]:
        if mask_mode not in {"soft", "hard"}:
            raise ValueError(f"Unsupported mask_mode '{mask_mode}'. Expected 'soft' or 'hard'.")
        tau = max(float(tau), 1e-6)

        refined: Dict[str, Tensor] = {}
        for name in self._task_names:
            branch = self.projectors[self._proj_alias[name]]
            refined[name] = features[name] + branch(features[name])

        logits_det = self.det_head(refined["det"])
        if logits_det.size(1) < 2:
            raise ValueError("Detection head must have at least two logits for mask computation.")
        prob_det = torch.softmax(logits_det, dim=1)[:, 1]
        margin = logits_det[:, 1] - logits_det[:, 0]
        soft_mask_pred = torch.sigmoid(margin / tau)

        type_logits = self.type_head(refined["type"])
        snr_logits = self.snr_head(refined["snr"]).squeeze(-1)
        area_logits = self.area_head(refined["area"])
        bwd_logits = self.bwd_head(refined["bwd"])  # keep (B, 1)

        outputs: Dict[str, Tensor] = {
            "det": logits_det,
            "type": type_logits,
            "snr": snr_logits,
            "area": area_logits,
            "bwd": bwd_logits,
        }

        if det_mask is not None:
            outputs["mask"] = det_mask.float() if mask_mode == "soft" else det_mask.bool()
        elif use_pred_mask:
            outputs["mask"] = soft_mask_pred if mask_mode == "soft" else (prob_det > mask_threshold)
        else:
            outputs["mask"] = soft_mask_pred if mask_mode == "soft" else (prob_det > mask_threshold)

        return outputs

    def freeze_heads(self, active: Dict[str, bool]) -> None:
        for name, module in self._head_modules.items():
            requires_grad = active.get(name, True)
            for submodule in module:
                for param in submodule.parameters():
                    param.requires_grad = requires_grad
                submodule.train(requires_grad)


class SimpleHeads(nn.Module):
    """Fallback heads without task-specific refinements."""

    def __init__(self, in_dim: int, config: HeadsConfig) -> None:
        super().__init__()
        self._task_names = ["det", "type", "snr", "area", "bwd"]
        self.det_head = nn.Linear(in_dim, config.det_classes)
        self.type_head = nn.Linear(in_dim, config.type_classes)
        self.snr_head = nn.Linear(in_dim, 1)
        self.area_head = nn.Linear(in_dim, config.area_classes)
        self.bwd_head = nn.Linear(in_dim, 1)
        self._head_modules = {
            "det": self.det_head,
            "type": self.type_head,
            "snr": self.snr_head,
            "area": self.area_head,
            "bwd": self.bwd_head,
        }

    def forward(
        self,
        features: Dict[str, Tensor],
        det_mask: Tensor | None = None,
        use_pred_mask: bool = False,
        mask_mode: str = "hard",
        mask_threshold: float = 0.5,
        tau: float = 1.0,
    ) -> Dict[str, Tensor]:
        if mask_mode not in {"soft", "hard"}:
            raise ValueError(f"Unsupported mask_mode '{mask_mode}'. Expected 'soft' or 'hard'.")
        tau = max(float(tau), 1e-6)

        logits_det = self.det_head(features["det"])
        if logits_det.size(1) < 2:
            raise ValueError("Detection head must have at least two logits for mask computation.")
        prob_det = torch.softmax(logits_det, dim=1)[:, 1]
        margin = logits_det[:, 1] - logits_det[:, 0]
        soft_mask_pred = torch.sigmoid(margin / tau)

        outputs: Dict[str, Tensor] = {
            "det": logits_det,
            "type": self.type_head(features["type"]),
            "snr": self.snr_head(features["snr"]).squeeze(-1),
            "area": self.area_head(features["area"]),
            "bwd": self.bwd_head(features["bwd"]),
        }

        if det_mask is not None:
            outputs["mask"] = det_mask.float() if mask_mode == "soft" else det_mask.bool()
        else:
            if use_pred_mask:
                outputs["mask"] = soft_mask_pred if mask_mode == "soft" else (prob_det > mask_threshold)
            else:
                outputs["mask"] = soft_mask_pred if mask_mode == "soft" else (prob_det > mask_threshold)

        return outputs

    def freeze_heads(self, active: Dict[str, bool]) -> None:
        for name, module in self._head_modules.items():
            requires_grad = active.get(name, True)
            for param in module.parameters():
                param.requires_grad = requires_grad
            module.train(requires_grad)


class SqueezeExcite(nn.Module):
    """Channel attention module for lightweight modulation."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        reduced = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced, kernel_size=1, bias=True),
            nn.SiLU(),
            nn.Conv2d(reduced, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        scale = self.fc(self.pool(inputs))
        return inputs * scale


class DualBranchBlock(nn.Module):
    """Dual-branch depthwise block for temporal and spectral sensitivity."""

    def __init__(self, channels: int, freq_kernel: int = 7, time_kernel: int = 7, dilation: int = 1) -> None:
        super().__init__()
        freq_padding = (0, freq_kernel // 2)
        time_padding = (dilation * (time_kernel // 2), 0)

        self.freq_branch = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(1, freq_kernel),
                padding=freq_padding,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.Hardswish(),
        )
        self.time_branch = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(time_kernel, 1),
                padding=time_padding,
                dilation=(dilation, 1),
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.Hardswish(),
        )
        self.mix = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Hardswish(),
        )
        self.se = SqueezeExcite(channels)

    def forward(self, inputs: Tensor) -> Tensor:
        freq_feat = self.freq_branch(inputs)
        time_feat = self.time_branch(inputs)
        fused = torch.cat([freq_feat, time_feat], dim=1)
        out = self.mix(fused)
        out = self.se(out)
        return inputs + out


class EncoderStage(nn.Module):
    """Hierarchical stage composed of dual-branch blocks."""

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int, stride: int) -> None:
        super().__init__()
        if stride > 1:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Hardswish(),
            )
        elif in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Hardswish(),
            )
        else:
            self.proj = nn.Identity()

        self.blocks = nn.ModuleList([DualBranchBlock(out_channels) for _ in range(num_blocks)])

    def forward(self, inputs: Tensor) -> Tensor:
        x = self.proj(inputs)
        for block in self.blocks:
            x = block(x)
        return x


class MultiScaleFusion(nn.Module):
    """Top-down feature pyramid fusion."""

    def __init__(self, in_channels: Sequence[int], fpn_channels: int, out_channels: int) -> None:
        super().__init__()
        self.lateral = nn.ModuleList(
            nn.Conv2d(ch, fpn_channels, kernel_size=1, bias=False) for ch in in_channels
        )
        self.smooth = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(fpn_channels),
                nn.Hardswish(),
            )
            for _ in in_channels
        )
        self.output = nn.Sequential(
            nn.Conv2d(fpn_channels * len(in_channels), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(),
        )

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        assert len(features) == len(self.lateral)
        top: Tensor | None = None
        fused: list[Tensor] = []
        for idx in reversed(range(len(features))):
            lateral = self.lateral[idx](features[idx])
            if top is not None:
                top = F.interpolate(top, size=lateral.shape[2:], mode="bilinear", align_corners=False)
                lateral = lateral + top
            top = lateral
            fused.append(self.smooth[idx](lateral))

        # fused order: low-resolution -> high-resolution. Align all to the smallest map.
        target_h, target_w = fused[0].shape[2:]
        resized = [fused[0]]
        for tensor in fused[1:]:
            resized.append(
                F.interpolate(tensor, size=(target_h, target_w), mode="bilinear", align_corners=False)
            )
        concat = torch.cat(resized, dim=1)
        return self.output(concat)


class GNSSEncoder(nn.Module):
    """Dual-branch multi-scale encoder tailored for GNSS interference."""

    def __init__(
        self,
        in_channels: int,
        stage_channels: Sequence[int],
        num_blocks: Sequence[int],
        fpn_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        if len(stage_channels) != len(num_blocks):
            raise ValueError("stage_channels and num_blocks must have the same length")

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stage_channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stage_channels[0]),
            nn.Hardswish(),
        )

        stages = []
        prev_channels = stage_channels[0]
        for idx, (out_channels_stage, blocks) in enumerate(zip(stage_channels, num_blocks)):
            stride = 1 if idx == 0 else 2
            stages.append(EncoderStage(prev_channels, out_channels_stage, blocks, stride))
            prev_channels = out_channels_stage
        self.stages = nn.ModuleList(stages)

        self.fpn = MultiScaleFusion(stage_channels, fpn_channels, out_channels)

    def forward(self, inputs: Tensor) -> Tensor:
        x = self.stem(inputs)
        features: list[Tensor] = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return self.fpn(features)


@dataclass
class LiteJamConfig:
    """Configuration for LiteJam model."""

    d_model: int = 384
    n_heads: int = 4
    ffn_dim: int = 192
    dropout: float = 0.1
    top_k: int = 32
    dsa_threshold: Optional[float] = None
    dsa_min_k: int = 1
    num_transformer_layers: int = 2
    head_config: HeadsConfig = None
    dropout_head: float = 0.3
    pretrained_path: Optional[str] = None
    preprocessor_in_channels: int = 6
    preprocessor_out_channels: int = 8
    preprocessor_stem_channels: int = 8
    encoder_channels: Sequence[int] = (32, 48, 64)
    encoder_blocks: Sequence[int] = (1, 1, 1)
    encoder_fpn_channels: int = 64
    use_sparse_attention: bool = True
    use_hierarchical_heads: bool = True


class LiteJamModel(nn.Module):
    """GNSS-tailored encoder with transformer and hierarchical heads."""

    def __init__(self, config: LiteJamConfig) -> None:
        super().__init__()
        self.config = config
        head_config = config.head_config or HeadsConfig()
        self.preprocessor = GNSSPreprocessor(
            in_channels=config.preprocessor_in_channels,
            out_channels=config.preprocessor_out_channels,
            stem_channels=config.preprocessor_stem_channels,
        )
        if config.pretrained_path:
            warnings.warn(
                "pretrained_path is ignored because LiteJam uses the custom GNSS encoder backbone.",
                stacklevel=2,
            )

        self.encoder = GNSSEncoder(
            in_channels=config.preprocessor_out_channels,
            stage_channels=config.encoder_channels,
            num_blocks=config.encoder_blocks,
            fpn_channels=config.encoder_fpn_channels,
            out_channels=config.d_model,
        )

        transformer_blocks: List[nn.Module] = []
        for _ in range(config.num_transformer_layers):
            if config.use_sparse_attention:
                block = TransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    ffn_dim=config.ffn_dim,
                    dropout=config.dropout,
                    top_k=config.top_k,
                    threshold=config.dsa_threshold,
                    min_k=config.dsa_min_k,
                )
            else:
                block = TransformerBlockDense(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    ffn_dim=config.ffn_dim,
                    dropout=config.dropout,
                )
            transformer_blocks.append(block)
        self.transformer = nn.ModuleList(transformer_blocks)
        self.dropout = nn.Dropout(config.dropout_head)
        self.task_names = ["det", "type", "snr", "area", "bwd"]
        self.pool_norm = nn.LayerNorm(config.d_model)
        self.task_attn = nn.ParameterDict(
            {
                f"attn_{name}": nn.Parameter(torch.randn(config.d_model))
                for name in self.task_names
            }
        )
        if config.use_hierarchical_heads:
            self.heads = HierarchicalHeads(config.d_model, head_config)
        else:
            self.heads = SimpleHeads(config.d_model, head_config)

    def forward(
        self,
        inputs: Tensor,
        det_labels: Optional[Tensor] = None,
        use_pred_mask: bool = False,
        mask_mode: str = "hard",
        mask_threshold: float = 0.5,
        tau: float = 1.0,
    ) -> Dict[str, Tensor]:
        inputs = self.preprocessor(inputs)
        features = self.encoder(inputs)
        pos = _sincos_2d_pos_embedding(
            dim=features.shape[1],
            height=features.shape[2],
            width=features.shape[3],
            device=features.device,
            dtype=features.dtype,
        )
        features = features + pos.unsqueeze(0)
        bsz, dim, height, width = features.shape
        tokens = features.view(bsz, dim, height * width).permute(0, 2, 1)

        for block in self.transformer:
            tokens = block(tokens)

        tokens_norm = self.pool_norm(tokens)
        task_features: Dict[str, Tensor] = {}
        for name in self.task_names:
            attn_vec = self.task_attn[f"attn_{name}"]
            scores = torch.matmul(tokens_norm, attn_vec)
            weights = torch.softmax(scores / math.sqrt(dim), dim=1)
            pooled = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
            task_features[name] = self.dropout(pooled)

        det_mask = det_labels if det_labels is not None else None
        return self.heads(
            task_features,
            det_mask,
            use_pred_mask=use_pred_mask,
            mask_mode=mask_mode,
            mask_threshold=mask_threshold,
            tau=tau,
        )

    def freeze_heads(self, active: Dict[str, bool]) -> None:
        self.heads.freeze_heads(active)


__all__ = [
    "DSAConfig",
    "DynamicSparseAttention",
    "TransformerBlock",
    "TransformerBlockDense",
    "HeadsConfig",
    "HierarchicalHeads",
    "SimpleHeads",
    "GNSSEncoder",
    "LiteJamConfig",
    "LiteJamModel",
    "GNSSPreprocessor",
]


class GNSSPreprocessor(nn.Module):
    """GNSS-aware stem with frequency alignment and multi-scale interference filters."""

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 8,
        stem_channels: int = 8,
    ) -> None:
        super().__init__()
        if in_channels < 2:
            raise ValueError("GNSSPreprocessor expects at least two channels (I/Q)")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stem_channels = stem_channels
        self.log_eps = 1e-4

        self.extra_channels = max(in_channels - 2, 0)

        # Base feature set: log power, magnitude, phase sin/cos, aligned I/Q (+ optional extras)
        self._base_channels = 6 + self.extra_channels

        # Depthwise filters emphasising pulse-like and sweep-like interference
        self.time_filter = nn.Conv2d(
            self._base_channels,
            self._base_channels,
            kernel_size=(5, 1),
            padding=(2, 0),
            groups=self._base_channels,
            bias=False,
        )
        self.freq_filter = nn.Conv2d(
            self._base_channels,
            self._base_channels,
            kernel_size=(1, 9),
            padding=(0, 4),
            groups=self._base_channels,
            bias=False,
        )
        self.chirp_filter = nn.Conv2d(
            self._base_channels,
            self._base_channels,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=self._base_channels,
            bias=False,
        )

        # Lightweight channel attention to rebalance I/Q vs. magnitude cues
        reduced = max(self._base_channels // 2, 1)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self._base_channels, reduced, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(reduced, self._base_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        # Multi-scale towers (coarse context + fine local detail)
        self.coarse_branch = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Conv2d(self._base_channels, stem_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.Hardswish(),
        )
        self.fine_branch = nn.Sequential(
            nn.Conv2d(self._base_channels, stem_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.Hardswish(),
            nn.Conv2d(stem_channels, stem_channels, kernel_size=3, padding=1, groups=stem_channels, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.Hardswish(),
        )

        fusion_in = self._base_channels * 4 + stem_channels * 2
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_in, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(),
        )

    def _build_alignment_grid(self, magnitude: Tensor) -> Tensor:
        """Compute sampling grid for residual frequency offset correction."""

        bsz, _, height, width = magnitude.shape
        freq_profile = magnitude.mean(dim=2, keepdim=False).squeeze(1) + 1e-6  # (N, W)
        weights = freq_profile / freq_profile.sum(dim=-1, keepdim=True)

        freq_positions = torch.linspace(
            -1.0,
            1.0,
            width,
            device=magnitude.device,
            dtype=magnitude.dtype,
        )
        offsets = (weights * freq_positions.unsqueeze(0)).sum(dim=-1)
        offsets = offsets.clamp(min=-0.9, max=0.9)

        base_x = freq_positions.view(1, 1, width)
        grid_x = base_x - offsets.view(bsz, 1, 1)
        grid_x = grid_x.expand(bsz, height, width)
        grid_x = torch.clamp(grid_x, -1.0, 1.0)

        time_positions = torch.linspace(
            -1.0,
            1.0,
            height,
            device=magnitude.device,
            dtype=magnitude.dtype,
        )
        grid_y = time_positions.view(1, height, 1).expand(bsz, height, width)

        return torch.stack((grid_x, grid_y), dim=-1)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {inputs.shape[1]}"
            )

        iq_inputs = inputs[:, :2]
        magnitude = torch.sqrt(iq_inputs[:, 0:1] ** 2 + iq_inputs[:, 1:2] ** 2 + self.log_eps)
        grid = self._build_alignment_grid(magnitude)

        aligned_iq = F.grid_sample(
            iq_inputs,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        aligned_aux = None
        if self.extra_channels:
            aux_inputs = inputs[:, 2:]
            aligned_aux = F.grid_sample(
                aux_inputs,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )

        i_aligned, q_aligned = aligned_iq[:, 0:1], aligned_iq[:, 1:2]

        magnitude = torch.sqrt(i_aligned**2 + q_aligned**2 + self.log_eps)
        log_power = torch.log1p(magnitude)
        phase = torch.atan2(q_aligned, i_aligned + 1e-6)
        phase_sin = torch.sin(phase)
        phase_cos = torch.cos(phase)

        components = [log_power, magnitude, phase_sin, phase_cos, i_aligned, q_aligned]
        if aligned_aux is not None:
            components.append(aligned_aux)
        base = torch.cat(components, dim=1)
        attn = self.channel_attention(base)
        base = base * attn

        pulse_feat = self.time_filter(base)
        sweep_feat = self.freq_filter(base)
        chirp_feat = self.chirp_filter(base)

        coarse = self.coarse_branch(base)
        coarse = F.interpolate(coarse, size=base.shape[2:], mode="bilinear", align_corners=False)

        fine = self.fine_branch(base)

        fused = torch.cat([base, pulse_feat, sweep_feat, chirp_feat, coarse, fine], dim=1)
        return self.fusion(fused)
