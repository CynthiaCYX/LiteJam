"""LiteJam multi-split training script.

This version supports training different task heads from dedicated data splits:
- jammer split: det/type/area
- signal_to_noise split: snr
- bandwidth split: bwd

Each split is iterated separately within an epoch, and losses are masked so that
only the relevant heads contribute updates for that batch.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from tqdm.auto import tqdm

# Allow running the script directly (python scripts/train_litejam.py)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from data.controlled_low_frequency_loader import (
        ControlledLowFrequencyDataset,
        build_group_datasets,
        infer_input_channels,
    )
except ImportError:  # pragma: no cover - backward compatibility shim
    from data.controlled_low_frequency_loader import ControlledLowFrequencyDataset, build_group_datasets

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
from train.losses import LiteJamLoss
from train.utils import format_task_weights, parse_task_weight_spec, resolve_device
from train.metrics import evaluate_outputs


PROGRESS_DISABLED = not sys.stderr.isatty()


def compute_overall_accuracy(metrics: Dict[str, Dict[str, Dict[str, float]]]) -> float:
    total = 0.0
    count = 0
    for group_metrics in metrics.values():
        for task_metrics in group_metrics.values():
            acc = task_metrics.get("acc")
            if acc is None or np.isnan(acc):
                continue
            total += acc
            count += 1
    if count == 0:
        return float("nan")
    return total / count


TASK_CONFIGS = {
    "jammer": {
        "split_group": "jammer",
        "split_suffix": "subjammer_dep",
        "tasks": {"det", "type", "area"},
    },
    "signal_to_noise": {
        "split_group": "signal_to_noise",
        "split_suffix": "subjammer_dep",
        "tasks": {"snr"},
    },
    "bandwidth": {
        "split_group": "bandwidth",
        "split_suffix": "subjammer_dep",
        "tasks": {"bwd"},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LiteJam with multi-split supervision")
    parser.add_argument("--root", type=Path, default=Path("data/controlled_low_frequency/GNSS_datasets/dataset1"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device (cpu/mps/cuda); 'auto' picks cuda>mps>cpu",
    )
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default="adamw")
    parser.add_argument("--scheduler", choices=["none", "plateau"], default="plateau")
    parser.add_argument("--plateau-patience", type=int, default=8)
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--warmup-epochs", type=int, default=3, help="Linear LR warmup epochs")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clip (0 to disable)")
    parser.add_argument(
        "--freeze-det-after",
        type=int,
        default=-1,
        help="Epoch after which to freeze detection head (-1 disables)",
    )
    parser.add_argument(
        "--freeze-det-duration",
        type=int,
        default=-1,
        help="Number of epochs to keep detection head frozen (-1 keeps frozen)",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=list(TASK_CONFIGS.keys()),
        choices=list(TASK_CONFIGS.keys()),
        help="Task groups to train (default: all)",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-bwd-normalize", action="store_true", help="Use raw MHz for bandwidth regression")
    parser.add_argument("--mask-warmup", type=int, default=10, help="Epochs before using predicted det mask")
    parser.add_argument(
        "--task-loss-weights",
        type=str,
        default="auto",
        help="Override task loss weights, e.g. 'det=1,type=1,area=2'.",
    )
    parser.add_argument(
        "--channel-strategy",
        choices=["iqfp", "iq", "repeat", "single"],
        default="iqfp",
        help="Input representation from dataset loader",
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Transformer block dropout probability")
    parser.add_argument("--head-dropout", type=float, default=0.3, help="Task head dropout probability")
    return parser.parse_args()


def _weights_from_counts(counts: np.ndarray) -> torch.Tensor:
    counts = counts.astype(np.float32)
    counts[counts <= 0] = 1.0
    inv = counts.sum() / counts
    inv = inv / inv.sum() * len(counts)
    return torch.tensor(inv, dtype=torch.float32)


def compute_class_weights(dataset, head_cfg: HeadsConfig) -> Dict[str, torch.Tensor]:
    det_counts = np.zeros(2, dtype=np.float32)
    type_counts = np.ones(head_cfg.type_classes, dtype=np.float32)
    area_counts = np.ones(head_cfg.area_classes, dtype=np.float32)

    type_map = dataset.label_maps.get("type_to_idx", {})
    area_map = dataset.label_maps.get("area_to_idx", {})
    area_counts_map = dataset.label_maps.get("area_counts", {})

    for entry in dataset.entries:
        if entry.class_id > 0:
            det_counts[1] += 1
            type_idx = type_map.get(entry.class_id)
            if type_idx is not None and type_idx < len(type_counts):
                type_counts[type_idx] += 1
        else:
            det_counts[0] += 1

        area_idx = area_map.get(entry.area)
        if area_idx is not None and area_idx < len(area_counts):
            area_counts[area_idx] += 1

    if det_counts[0] == 0:
        det_counts[0] = 1
    if det_counts[1] == 0:
        det_counts[1] = 1

    # Incorporate pre-computed area counts if available
    for area_value, idx in area_map.items():
        if idx < len(area_counts):
            area_counts[idx] = max(area_counts[idx], area_counts_map.get(area_value, 1))

    return {
        "det": _weights_from_counts(det_counts),
        "type": _weights_from_counts(type_counts),
        "area": _weights_from_counts(area_counts),
    }


def save_checkpoint(
    model: LiteJamModel,
    model_cfg: LiteJamConfig,
    head_cfg: HeadsConfig,
    label_maps_by_group: Dict[str, Dict[str, object]],
    optimizer: torch.optim.Optimizer,
    suffix: str,
    run_stamp: str,
) -> Path:
    save_dir = REPO_ROOT / "saved"
    save_dir.mkdir(exist_ok=True)
    if suffix in {"best", "good", "last"}:
        ckpt_path = save_dir / f"{run_stamp}_{suffix}.pt"
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        ckpt_path = save_dir / f"{run_stamp}_{stamp}_{suffix}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "litejam_config": asdict(model_cfg),
            "heads_config": asdict(head_cfg),
            "label_maps_by_group": label_maps_by_group,
            "optimizer_state": optimizer.state_dict(),
        },
        ckpt_path,
    )
    return ckpt_path


def compute_group_loss(losses: Dict[str, torch.Tensor], tasks: Set[str]) -> torch.Tensor:
    relevant = []
    for task in ("det", "type", "snr", "area", "bwd"):
        if task in tasks and task in losses:
            relevant.append(losses[task])
    if not relevant:
        return None  # type: ignore
    total = relevant[0]
    for loss in relevant[1:]:
        total = total + loss
    return total


def evaluate_model_groups(
    model: LiteJamModel,
    loaders: Dict[str, Dict[str, object]],
    device: torch.device,
    use_pred_mask: bool,
    *,
    split: str = "val",
) -> Dict[str, Dict[str, Dict[str, float]]]:
    metrics_by_group: Dict[str, Dict[str, Dict[str, float]]] = {}
    model.eval()

    with torch.no_grad():
        for name, info in loaders.items():
            if split not in info:
                raise KeyError(f"Split '{split}' not available for group '{name}'")
            loader: DataLoader = info[split]  # type: ignore[assignment]
            tasks: Set[str] = info["tasks"]  # type: ignore[assignment]
            label_maps: Dict[str, object] = info["label_maps"]  # type: ignore[assignment]

            det_targets: List[np.ndarray] = []
            type_targets: List[np.ndarray] = []
            snr_targets: List[np.ndarray] = []
            area_targets: List[np.ndarray] = []
            bwd_targets: List[np.ndarray] = []

            det_preds: List[np.ndarray] = []
            type_preds: List[np.ndarray] = []
            snr_preds: List[np.ndarray] = []
            area_preds: List[np.ndarray] = []
            bwd_preds: List[np.ndarray] = []

            det_masks: List[np.ndarray] = []
            snr_masks: List[np.ndarray] = []
            bwd_masks: List[np.ndarray] = []

            snr_mean = float(label_maps.get("snr_mean", 0.0))
            snr_std = float(label_maps.get("snr_std", 1.0)) or 1.0

            for inputs, targets in loader:
                inputs = inputs.to(device)
                targets = {k: v.to(device) if hasattr(v, "to") else v for k, v in targets.items()}
                outputs = model(
                    inputs,
                    det_labels=targets.get("det"),
                    use_pred_mask=use_pred_mask and "det" in tasks,
                )

                det_targets.append(targets["det"].cpu().numpy())
                det_preds.append(outputs["det"].argmax(dim=1).cpu().numpy())

                type_targets.append(targets["type"].cpu().numpy())
                type_preds.append(outputs["type"].argmax(dim=1).cpu().numpy())

                snr_targets.append((targets["snr"] * snr_std + snr_mean).cpu().numpy())
                snr_preds.append((outputs["snr"] * snr_std + snr_mean).cpu().numpy())

                area_targets.append(targets["area"].cpu().numpy())
                area_preds.append(outputs["area"].argmax(dim=1).cpu().numpy())

                bwd_targets.append(targets["bwd"].cpu().numpy())
                bwd_preds.append(outputs["bwd"].squeeze(-1).cpu().numpy())

                det_masks.append((targets["det"] == 1).cpu().numpy().astype(bool))
                snr_mask_tensor = targets.get("snr_mask", torch.ones_like(targets["det"], dtype=torch.bool))
                snr_masks.append(snr_mask_tensor.cpu().numpy().astype(bool))
                bwd_mask_tensor = targets.get("bwd_mask", torch.ones_like(targets["bwd"], dtype=torch.bool))
                bwd_masks.append(bwd_mask_tensor.cpu().numpy().astype(bool))

            targets_np = {
                "det": np.concatenate(det_targets),
                "type": np.concatenate(type_targets),
                "snr": np.concatenate(snr_targets),
                "area": np.concatenate(area_targets),
                "bwd": np.concatenate(bwd_targets),
            }
            preds_np = {
                "det": np.concatenate(det_preds),
                "type": np.concatenate(type_preds),
                "snr": np.concatenate(snr_preds),
                "area": np.concatenate(area_preds),
                "bwd": np.concatenate(bwd_preds),
            }
            mask_np = np.concatenate(det_masks)
            snr_mask_np = np.concatenate(snr_masks) if snr_masks else np.ones_like(mask_np, dtype=bool)
            bwd_mask_np = np.concatenate(bwd_masks) if bwd_masks else np.ones_like(mask_np, dtype=bool)
            type_mask_np = targets_np["type"] >= 0
            area_mask_np = targets_np["area"] >= 0

            metrics = evaluate_outputs(
                targets_np,
                preds_np,
                mask_np,
                task_masks={
                    "snr": snr_mask_np,
                    "bwd": bwd_mask_np,
                    "type": type_mask_np,
                    "area": area_mask_np,
                },
            )

            # Filter to relevant tasks for readability
            filtered = {"det": metrics["det"]}
            if "type" in tasks:
                filtered["type"] = metrics["type"]
            if "snr" in tasks:
                filtered["snr"] = metrics["snr"]
            if "area" in tasks:
                filtered["area"] = metrics["area"]
            if "bwd" in tasks:
                filtered["bwd"] = metrics["bwd"]

            metrics_by_group[name] = filtered

    return metrics_by_group


def compute_overall_accuracy(metrics: Dict[str, Dict[str, Dict[str, float]]]) -> float:
    total = 0.0
    count = 0
    for group_metrics in metrics.values():
        for task_metrics in group_metrics.values():
            acc = task_metrics.get("acc")
            if acc is None or np.isnan(acc):
                continue
            total += acc
            count += 1
    if count == 0:
        return float("nan")
    return total / count


def build_loaders(
    root: Path,
    groups: List[str],
    batch_size: int,
    num_workers: int,
    normalize_bwd: bool,
    channel_strategy: Optional[str],
) -> Dict[str, Dict[str, object]]:
    loaders: Dict[str, Dict[str, object]] = {}
    for name in groups:
        cfg = TASK_CONFIGS[name]
        dataset_kwargs = {"normalize_bwd": not normalize_bwd}
        if channel_strategy is not None:
            dataset_kwargs["channel_strategy"] = channel_strategy
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
        print(
            f"[Data] group={name} tasks={cfg['tasks']} | type={train_ds.label_maps['type_labels']} "
            f"snr_mean={train_ds.label_maps['snr_mean']:.2f}+/-{train_ds.label_maps['snr_std']:.2f} "
            f"area={train_ds.label_maps['area_labels']}"
        )
    return loaders


def main() -> None:
    args = parse_args()
    device_str = resolve_device(args.device)
    print(f"[Config] device: {device_str}")
    device = torch.device(device_str)

    group_loaders = build_loaders(
        root=args.root,
        groups=args.groups,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize_bwd=args.no_bwd_normalize,
        channel_strategy=args.channel_strategy,
    )
    if not group_loaders:
        raise RuntimeError("No task groups available for training")

    # Head configuration derived from overall label maps (jammer group defines det/type/area sizes)
    jammer_maps = group_loaders.get("jammer")
    if jammer_maps is None:
        raise RuntimeError("Jammer group is required to configure the shared heads")

    head_cfg = HeadsConfig(
        det_classes=2,
        type_classes=len(jammer_maps["label_maps"]["type_to_idx"]),
        area_classes=len(jammer_maps["label_maps"]["area_to_idx"]),
    )
    jammer_ds = jammer_maps["train_dataset"]
    input_channels = getattr(
        jammer_ds,
        "input_channels",
        infer_input_channels(getattr(jammer_ds, "channel_strategy", "iqfp")),
    )
    model_cfg = LiteJamConfig(
        head_config=head_cfg,
        dropout=args.dropout,
        dropout_head=args.head_dropout,
        preprocessor_in_channels=input_channels,
    )
    model = LiteJamModel(model_cfg).to(device)

    class_weights = compute_class_weights(jammer_maps["train_dataset"], head_cfg)
    base_loss_weights = {
        "det": 1.0,
        "type": 1.0,
        "snr": 1.0,
        "area": 1.0,
        "bwd": 1.0,
    }
    loss_weights = parse_task_weight_spec(args.task_loss_weights, base_loss_weights)
    print(f"[Config] task loss weights: {format_task_weights(loss_weights)}")
    criterion = LiteJamLoss(loss_weights, class_weights=class_weights)

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=args.plateau_patience,
            factor=args.plateau_factor,
        )
    else:
        scheduler = None

    label_maps_by_group = {name: info["label_maps"] for name, info in group_loaders.items()}

    run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    best_val_loss = float("inf")
    best_path: Path | None = None
    best_test_acc = float("-inf")
    good_path: Path | None = None
    freeze_active = False
    freeze_end_epoch: int | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_steps = 0
        det_correct = 0
        det_total = 0

        if args.warmup_epochs > 0 and epoch <= args.warmup_epochs:
            warmup_scale = epoch / max(1, args.warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr * warmup_scale
        elif scheduler is None:
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr

        for group_name, info in group_loaders.items():
            tasks = info["tasks"]
            loader = info["train"]
            progress = tqdm(
                loader,
                desc=f"Epoch {epoch:03d} [train-{group_name}]",
                leave=False,
                disable=PROGRESS_DISABLED,
            )
            for inputs, targets in progress:
                inputs = inputs.to(device)
                targets = {k: v.to(device) if hasattr(v, "to") else v for k, v in targets.items()}

                optimizer.zero_grad()
                use_pred_mask = epoch > args.mask_warmup and "det" in tasks
                outputs = model(inputs, det_labels=targets.get("det"), use_pred_mask=use_pred_mask)
                losses = criterion(outputs, targets)
                group_loss = compute_group_loss(losses, tasks)
                if group_loss is None:
                    continue
                group_loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

                train_loss_total += group_loss.item()
                train_steps += 1

                if "det" in tasks:
                    preds = outputs["det"].argmax(dim=1)
                    det_targets = targets["det"]
                    det_correct += (preds == det_targets).sum().item()
                    det_total += det_targets.numel()

                progress.set_postfix(loss=f"{group_loss.item():.4f}")

        avg_train_loss = train_loss_total / max(train_steps, 1)
        train_acc = det_correct / det_total if det_total else 0.0

        model.eval()
        val_loss_total = 0.0
        val_steps = 0
        with torch.no_grad():
            for group_name, info in group_loaders.items():
                tasks = info["tasks"]
                loader = info["val"]
                progress = tqdm(
                    loader,
                    desc=f"Epoch {epoch:03d} [val-{group_name}]",
                    leave=False,
                    disable=PROGRESS_DISABLED,
                )
                for inputs, targets in progress:
                    inputs = inputs.to(device)
                    targets = {k: v.to(device) if hasattr(v, "to") else v for k, v in targets.items()}
                    outputs = model(
                        inputs,
                        det_labels=targets.get("det"),
                        use_pred_mask=epoch > args.mask_warmup and "det" in tasks,
                    )
                    losses = criterion(outputs, targets)
                    group_loss = compute_group_loss(losses, tasks)
                    if group_loss is None:
                        continue
                    val_loss_total += group_loss.item()
                    val_steps += 1
                    progress.set_postfix(loss=f"{group_loss.item():.4f}")

        avg_val_loss = val_loss_total / max(val_steps, 1)
        if scheduler is not None:
            scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d} | lr={current_lr:.6f} | train_loss={avg_train_loss:.4f} "
            f"train_det_acc={train_acc:.3f} | val_loss={avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = save_checkpoint(
                model,
                model_cfg,
                head_cfg,
                label_maps_by_group,
                optimizer,
                suffix="best",
                run_stamp=run_stamp,
            )
            print(f"  -> Improved validation loss; best checkpoint updated at {best_path}")

        eval_metrics = evaluate_model_groups(
            model,
            group_loaders,
            device,
            use_pred_mask=epoch > args.mask_warmup,
        )
        for group_name, group_metrics in eval_metrics.items():
            metric_parts: List[str] = []
            for task, vals in group_metrics.items():
                if not isinstance(vals, dict):
                    continue
                if task in {"det", "type", "area"}:
                    metric_parts.append(f"{task}_acc={vals.get('acc', float('nan')):.4f}")
                elif task in {"snr", "bwd"}:
                    metric_parts.append(f"{task}_r2={vals.get('r2', float('nan')):.4f}")
            metric_str = ", ".join(metric_parts)
            print(f"    [{group_name}] {metric_str}")
        test_metrics = evaluate_model_groups(
            model,
            group_loaders,
            device,
            use_pred_mask=epoch > args.mask_warmup,
            split="test",
        )
        test_acc = compute_overall_accuracy(test_metrics)
        if not np.isnan(test_acc) and test_acc > best_test_acc:
            best_test_acc = test_acc
            good_path = save_checkpoint(
                model,
                model_cfg,
                head_cfg,
                label_maps_by_group,
                optimizer,
                suffix="good",
                run_stamp=run_stamp,
            )
            print(f"  -> Improved test accuracy ({test_acc:.4f}); good checkpoint updated at {good_path}")

        if args.freeze_det_after >= 0 and epoch >= args.freeze_det_after and not freeze_active:
            model.freeze_heads({"det": False})
            freeze_active = True
            if args.freeze_det_duration > 0:
                freeze_end_epoch = epoch + args.freeze_det_duration
            print("[Schedule] Detection head frozen to emphasise auxiliary tasks")

        if (
            freeze_active
            and args.freeze_det_duration > 0
            and freeze_end_epoch is not None
            and epoch >= freeze_end_epoch
        ):
            model.freeze_heads({"det": True})
            freeze_active = False
            freeze_end_epoch = None
            print("[Schedule] Detection head unfrozen")

    final_path = save_checkpoint(
        model,
        model_cfg,
        head_cfg,
        label_maps_by_group,
        optimizer,
        suffix="last",
        run_stamp=run_stamp,
    )
    print(f"Training complete. Final checkpoint saved to {final_path}")
    if best_path is not None:
        print(f"Best (val_loss) checkpoint: {best_path}")
        print(f"[Summary] best_val_loss={best_val_loss:.6f}")
    if good_path is not None:
        print(f"Best (test acc) checkpoint: {good_path}")
    if best_test_acc > float("-inf"):
        print(f"[Summary] best_test_acc={best_test_acc:.6f}")


if __name__ == "__main__":
    main()
