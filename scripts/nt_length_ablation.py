#!/usr/bin/env python3
"""LiteJam time-series length ablation (Nt = 1..34).

This script trains a LiteJam model for a range of temporal extents (Nt) by cropping
the spectrogram time axis and measuring performance for all five tasks. Models are
trained on the dependent (``subjammer_dep``) splits and evaluated on their
corresponding test sets. Classification tasks report F1 scores, while regression
tasks report R^2.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

# Allow running as a standalone script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.controlled_low_frequency_loader import build_group_datasets
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CropTimeTransform:
    """Slice the spectrogram along the last dimension to the desired Nt."""

    def __init__(self, length: int) -> None:
        if length <= 0:
            raise ValueError("length must be positive")
        self.length = length

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim < 2:
            return tensor
        target = min(self.length, tensor.shape[-1])
        return tensor[..., :target]


def build_loaders_for_nt(
    root: Path,
    crop_length: int,
    batch_size: int,
    num_workers: int,
    normalize_bwd: bool,
    channel_strategy: str,
    groups: Iterable[str],
) -> Dict[str, Dict[str, object]]:
    loaders: Dict[str, Dict[str, object]] = {}
    crop = CropTimeTransform(crop_length)
    transforms = {phase: crop for phase in ("train", "val", "test")}

    for name in groups:
        cfg = TASK_CONFIGS[name]
        dataset_kwargs = {
            "normalize_bwd": not normalize_bwd,
            "channel_strategy": channel_strategy,
        }
        train_ds, val_ds, test_ds = build_group_datasets(
            root=root,
            split_group=cfg["split_group"],
            split_suffix=cfg["split_suffix"],
            tasks=cfg["tasks"],
            transforms=transforms,
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


def train_for_nt(
    nt: int,
    loaders: Dict[str, Dict[str, object]],
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    mask_warmup: int,
    loss_weights: Dict[str, float],
    dropout: float,
    head_dropout: float,
    top_k: int,
    layers: int,
    d_model: int,
    n_heads: int,
    ffn_dim: int,
) -> Tuple[LiteJamModel, Dict[str, Dict[str, float]]]:
    jammer_info = loaders["jammer"]
    head_cfg = HeadsConfig(
        det_classes=2,
        type_classes=len(jammer_info["label_maps"]["type_to_idx"]),
        area_classes=len(jammer_info["label_maps"]["area_to_idx"]),
    )
    model_cfg = LiteJamConfig(
        d_model=d_model,
        n_heads=n_heads,
        ffn_dim=ffn_dim,
        dropout=dropout,
        dropout_head=head_dropout,
        top_k=top_k,
        num_transformer_layers=layers,
        head_config=head_cfg,
    )
    model = LiteJamModel(model_cfg).to(device)

    class_weights = compute_class_weights(jammer_info["train_dataset"], head_cfg)
    criterion = LiteJamLoss(loss_weights, class_weights=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state: Dict[str, torch.Tensor] | None = None
    best_metric = -float("inf")
    best_val_metrics: Dict[str, Dict[str, float]] = {}

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
            best_val_metrics = val_metrics

        print(
            f"[Nt={nt:02d}] epoch {epoch:03d}/{epochs} val_det_f1={det_f1:.4f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_val_metrics = evaluate_model_groups(
            model,
            loaders,
            device,
            use_pred_mask=True,
            split="val",
        )

    return model, best_val_metrics


def gather_test_metrics(
    model: LiteJamModel,
    loaders: Dict[str, Dict[str, object]],
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
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


def extract_task_scores(metrics: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, float]:
    return {
        "det_f1": float(metrics["jammer"]["det"].get("f1", float("nan"))),
        "type_f1": float(metrics["jammer"].get("type", {}).get("f1", float("nan"))),
        "area_f1": float(metrics["jammer"].get("area", {}).get("f1", float("nan"))),
        "snr_r2": float(metrics["signal_to_noise"].get("snr", {}).get("r2", float("nan"))),
        "bwd_r2": float(metrics["bandwidth"].get("bwd", {}).get("r2", float("nan"))),
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LiteJam Nt length ablation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/controlled_low_frequency/GNSS_datasets/dataset1"),
        help="Dataset root directory",
    )
    parser.add_argument("--nt-min", type=int, default=1, help="Minimum Nt to evaluate")
    parser.add_argument("--nt-max", type=int, default=34, help="Maximum Nt to evaluate")
    parser.add_argument("--device", default="auto", help="Torch device string or 'auto'")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs per Nt")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping (<=0 disables)")
    parser.add_argument("--mask-warmup", type=int, default=10, help="Epochs before enabling predicted det mask")
    parser.add_argument(
        "--task-loss-weights",
        type=str,
        default="det=1,type=1,snr=1,area=1,bwd=1",
        help="Override task loss weights",
    )
    parser.add_argument(
        "--channel-strategy",
        choices=["iqfp", "iq", "repeat", "single"],
        default="iqfp",
        help="Input representation",
    )
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
        "--output-dir",
        type=Path,
        default=Path("nt_ablation_results"),
        help="Directory to store metrics",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def ensure_device(device_str: str) -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[Warn] CUDA requested but not available; falling back to CPU", file=sys.stderr)
        device_str = "cpu"
    if device_str == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            print("[Warn] MPS requested but not available; falling back to CPU", file=sys.stderr)
            device_str = "cpu"
    return torch.device(device_str)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.nt_min <= 0 or args.nt_max < args.nt_min:
        raise ValueError("Invalid Nt range")

    seed_everything(args.seed)
    device = ensure_device(resolve_device(args.device))
    print(f"[Config] device: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    loss_base = {"det": 1.0, "type": 1.0, "snr": 1.0, "area": 1.0, "bwd": 1.0}
    loss_weights = parse_task_weight_spec(args.task_loss_weights, loss_base)
    print(f"[Config] task loss weights: {format_task_weights(loss_weights)}")

    groups = ["jammer", "signal_to_noise", "bandwidth"]

    results: List[Dict[str, float]] = []

    for nt in range(args.nt_min, args.nt_max + 1):
        print(f"\n===== Nt = {nt} =====")
        loaders = build_loaders_for_nt(
            args.root,
            crop_length=nt,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            normalize_bwd=args.no_bwd_normalize,
            channel_strategy=args.channel_strategy,
            groups=groups,
        )

        model, _ = train_for_nt(
            nt,
            loaders,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            mask_warmup=args.mask_warmup,
            loss_weights=loss_weights,
            dropout=args.dropout,
            head_dropout=args.dropout_head,
            top_k=args.top_k,
            layers=args.layers,
            d_model=args.d_model,
            n_heads=args.n_heads,
            ffn_dim=args.ffn_dim,
        )

        test_metrics = gather_test_metrics(model, loaders, device)
        scores = extract_task_scores(test_metrics)
        scores["Nt"] = nt
        results.append(scores)

        print(
            f"[Nt={nt}] det_f1={scores['det_f1']:.4f} type_f1={scores['type_f1']:.4f} "
            f"area_f1={scores['area_f1']:.4f} snr_r2={scores['snr_r2']:.4f} bwd_r2={scores['bwd_r2']:.4f}",
            flush=True,
        )

        del loaders
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Persist results
    csv_path = args.output_dir / "nt_ablation.csv"
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write("Nt,det_f1,type_f1,area_f1,snr_r2,bwd_r2\n")
        for item in results:
            fh.write(
                f"{item['Nt']},{item['det_f1']:.6f},{item['type_f1']:.6f},{item['area_f1']:.6f},"
                f"{item['snr_r2']:.6f},{item['bwd_r2']:.6f}\n"
            )

    json_path = args.output_dir / "nt_ablation.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print(f"[Result] Saved CSV to {csv_path}")
    print(f"[Result] Saved JSON to {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
