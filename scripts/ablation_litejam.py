#!/usr/bin/env python3
"""Run LiteJam module ablations to quantify the benefit of each component."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

# Ensure repository root is on the path when executed directly
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from data.controlled_low_frequency_loader import build_group_datasets, infer_input_channels
except ImportError:  # pragma: no cover - backward compatibility shim
    from data.controlled_low_frequency_loader import build_group_datasets

    def infer_input_channels(channel_strategy: str) -> int:
        mapping = {
            "iqfp": 6,
            "iq": 2,
            "repeat": 3,
            "single": 1,
        }
        if channel_strategy not in mapping:
            raise ValueError(f"Unsupported channel strategy: {channel_strategy}")
        return mapping[channel_strategy]
from models.litejam import HeadsConfig, LiteJamConfig, LiteJamModel
from scripts.train_litejam import (
    TASK_CONFIGS,
    compute_class_weights,
    compute_group_loss,
    evaluate_model_groups,
    format_task_weights,
    parse_task_weight_spec,
)
from train.losses import LiteJamLoss
from train.utils import resolve_device


DEFAULT_ABLATIONS = [
    {
        "name": "litejam_full",
        "label": "LiteJam",
        "config_overrides": {},
        "channel_strategy": None,
    },
    {
        "name": "dense_attention",
        "label": "No Dynamic Sparse Attention",
        "config_overrides": {"use_sparse_attention": False},
        "channel_strategy": None,
    },
    {
        "name": "simple_heads",
        "label": "No Hierarchical Heads",
        "config_overrides": {"use_hierarchical_heads": False},
        "channel_strategy": None,
    },
    {
        "name": "amplitude_only",
        "label": "Amplitude Channels Only",
        "config_overrides": {},
        "channel_strategy": "repeat",
    },
]


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(
    root: Path,
    batch_size: int,
    num_workers: int,
    normalize_bwd: bool,
    channel_strategy: str,
    groups: Iterable[str],
) -> Dict[str, Dict[str, object]]:
    loaders: Dict[str, Dict[str, object]] = {}
    dataset_kwargs = {"normalize_bwd": not normalize_bwd, "channel_strategy": channel_strategy}
    for name in groups:
        cfg = TASK_CONFIGS[name]
        train_ds, val_ds, test_ds = build_group_datasets(
            root=root,
            split_group=cfg["split_group"],
            split_suffix=cfg["split_suffix"],
            tasks=cfg["tasks"],
            dataset_kwargs=dataset_kwargs,
        )
        loaders[name] = {
            "tasks": set(cfg["tasks"]),
            "train": DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=default_collate,
            ),
            "val": DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=default_collate,
            ),
            "test": DataLoader(
                test_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=default_collate,
            ),
            "label_maps": train_ds.label_maps,
            "train_dataset": train_ds,
        }
    return loaders


def train_model(
    loaders: Dict[str, Dict[str, object]],
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    mask_warmup: int,
    loss_weights: Dict[str, float],
    model_cfg: LiteJamConfig,
) -> LiteJamModel:
    model = LiteJamModel(model_cfg).to(device)

    jammer_info = loaders["jammer"]
    class_weights = compute_class_weights(jammer_info["train_dataset"], model_cfg.head_config or HeadsConfig())
    criterion = LiteJamLoss(loss_weights, class_weights=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_metric = -float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None

    for epoch in range(1, epochs + 1):
        model.train()
        for group_name, info in loaders.items():
            tasks = info["tasks"]
            loader = info["train"]
            for inputs, targets in loader:
                inputs = inputs.to(device)
                targets = {k: v.to(device) if hasattr(v, "to") else v for k, v in targets.items()}

                optimizer.zero_grad()
                use_pred_mask = epoch > mask_warmup and "det" in tasks
                outputs = model(inputs, det_labels=targets.get("det"), use_pred_mask=use_pred_mask)
                losses = criterion(outputs, targets)
                group_loss = compute_group_loss(losses, tasks)
                if group_loss is None:
                    continue
                group_loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        model.eval()
        with torch.no_grad():
            val_metrics = evaluate_model_groups(
                model,
                loaders,
                device,
                use_pred_mask=True,
                split="val",
            )
        det_f1 = val_metrics["jammer"]["det"].get("f1", float("nan"))
        if not math.isnan(det_f1) and det_f1 > best_metric:
            best_metric = det_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"[Epoch {epoch:03d}] val_det_f1={det_f1:.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate(model: LiteJamModel, loaders: Dict[str, Dict[str, object]], device: torch.device) -> Dict[str, Dict[str, Dict[str, float]]]:
    model.eval()
    with torch.no_grad():
        metrics = evaluate_model_groups(
            model,
            loaders,
            device,
            use_pred_mask=True,
            split="test",
        )
    return metrics


def summarise_metrics(metrics: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, float]:
    return {
        "det_f1": float(metrics["jammer"]["det"].get("f1", float("nan"))),
        "type_f1": float(metrics["jammer"].get("type", {}).get("f1", float("nan"))),
        "area_f1": float(metrics["jammer"].get("area", {}).get("f1", float("nan"))),
        "snr_r2": float(metrics["signal_to_noise"].get("snr", {}).get("r2", float("nan"))),
        "bwd_r2": float(metrics["bandwidth"].get("bwd", {}).get("r2", float("nan"))),
    }


def ensure_device(device_str: str) -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[Warn] CUDA requested but unavailable; using CPU", file=sys.stderr)
        device_str = "cpu"
    if device_str == "mps":
        backend_mps = getattr(torch.backends, "mps", None)
        if backend_mps is None or not backend_mps.is_available():
            print("[Warn] MPS requested but unavailable; using CPU", file=sys.stderr)
            device_str = "cpu"
    return torch.device(device_str)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LiteJam module ablation runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/controlled_low_frequency/GNSS_datasets/dataset1"),
        help="Dataset root directory",
    )
    parser.add_argument("--device", default="auto", help="Torch device or 'auto'")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs per ablation")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping (<=0 disables)")
    parser.add_argument("--mask-warmup", type=int, default=10, help="Epochs before using predicted det mask")
    parser.add_argument("--task-loss-weights", type=str, default="det=1,type=1,snr=1,area=1,bwd=1")
    parser.add_argument("--channel-strategy", choices=["iqfp", "iq", "repeat", "single"], default="iqfp")
    parser.add_argument("--no-bwd-normalize", action="store_true", help="Disable bandwidth normalization")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--dropout-head", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=192)
    parser.add_argument(
        "--ablations",
        nargs="*",
        default=None,
        help="Optional subset of ablation names to run",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ablation_results"),
        help="Directory to store metrics",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def select_ablations(requested: Optional[List[str]]) -> List[Dict[str, object]]:
    if not requested:
        return DEFAULT_ABLATIONS
    mapping = {abl["name"]: abl for abl in DEFAULT_ABLATIONS}
    selected = []
    for name in requested:
        if name not in mapping:
            raise ValueError(f"Unknown ablation '{name}'. Available: {list(mapping)}")
        selected.append(mapping[name])
    return selected


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    seed_everything(args.seed)
    device = ensure_device(resolve_device(args.device))
    print(f"[Config] device: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    loss_base = {"det": 1.0, "type": 1.0, "snr": 1.0, "area": 1.0, "bwd": 1.0}
    loss_weights = parse_task_weight_spec(args.task_loss_weights, loss_base)
    print(f"[Config] task loss weights: {format_task_weights(loss_weights)}")

    groups = ["jammer", "signal_to_noise", "bandwidth"]
    ablations = select_ablations(args.ablations)

    summary: List[Dict[str, float]] = []

    for entry in ablations:
        name = entry["name"]
        label = entry["label"]
        overrides = dict(entry.get("config_overrides", {}))
        channel_override = entry.get("channel_strategy") or args.channel_strategy

        print(f"\n===== Ablation: {label} ({name}) =====")
        loaders = build_loaders(
            args.root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            normalize_bwd=args.no_bwd_normalize,
            channel_strategy=channel_override,
            groups=groups,
        )
        jammer_ds = loaders["jammer"]["train_dataset"]
        input_channels = getattr(
            jammer_ds,
            "input_channels",
            infer_input_channels(getattr(jammer_ds, "channel_strategy", "iqfp")),
        )

        head_cfg = HeadsConfig(
            det_classes=2,
            type_classes=len(loaders["jammer"]["label_maps"]["type_to_idx"]),
            area_classes=len(loaders["jammer"]["label_maps"]["area_to_idx"]),
        )

        cfg_kwargs = dict(
            d_model=args.d_model,
            n_heads=args.n_heads,
            ffn_dim=args.ffn_dim,
            dropout=args.dropout,
            top_k=args.top_k,
            num_transformer_layers=args.layers,
            head_config=head_cfg,
            dropout_head=args.dropout_head,
            use_sparse_attention=True,
            use_hierarchical_heads=True,
            preprocessor_in_channels=input_channels,
        )
        cfg_kwargs.update(overrides)
        model_cfg = LiteJamConfig(**cfg_kwargs)

        model = train_model(
            loaders,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            mask_warmup=args.mask_warmup,
            loss_weights=loss_weights,
            model_cfg=model_cfg,
        )

        metrics = evaluate(model, loaders, device)
        scores = summarise_metrics(metrics)
        scores["name"] = name
        scores["label"] = label
        summary.append(scores)

        print(
            " | ".join(
                [
                    f"det_f1={scores['det_f1']:.4f}",
                    f"type_f1={scores['type_f1']:.4f}",
                    f"area_f1={scores['area_f1']:.4f}",
                    f"snr_r2={scores['snr_r2']:.4f}",
                    f"bwd_r2={scores['bwd_r2']:.4f}",
                ]
            ),
            flush=True,
        )

        del loaders
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    csv_path = args.output_dir / "ablation_metrics.csv"
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write("name,label,det_f1,type_f1,area_f1,snr_r2,bwd_r2\n")
        for row in summary:
            fh.write(
                f"{row['name']},{row['label']},{row['det_f1']:.6f},{row['type_f1']:.6f},"
                f"{row['area_f1']:.6f},{row['snr_r2']:.6f},{row['bwd_r2']:.6f}\n"
            )

    json_path = args.output_dir / "ablation_metrics.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"[Result] Saved CSV to {csv_path}")
    print(f"[Result] Saved JSON to {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
