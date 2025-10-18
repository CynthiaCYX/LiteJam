"""MobileNetV3-S baseline training script matching LiteJam's multi-task setup."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.controlled_low_frequency_loader import build_group_datasets
from models.litejam import HeadsConfig
from baseline.mobilenetv3 import MobileNetV3SBaseline, MobileNetV3SBaselineConfig

try:
    from scripts.train_litejam import (
        TASK_CONFIGS,
        compute_class_weights,
        compute_group_loss,
        evaluate_model_groups,
        compute_overall_accuracy,
    )
except ImportError:
    from scripts.train_litejam import (  # type: ignore
        TASK_CONFIGS,
        compute_class_weights,
        compute_group_loss,
        evaluate_model_groups,
    )

    def compute_overall_accuracy(metrics: Dict[str, Dict[str, Dict[str, float]]]) -> float:
        total = 0.0
        count = 0
        for group_metrics in metrics.values():
            for task_metrics in group_metrics.values():
                acc = task_metrics.get("acc")
                if acc is None or np.isnan(acc):
                    continue
                total += float(acc)
                count += 1
        if count == 0:
            return float("nan")
        return total / count

    def compute_overall_accuracy(metrics: Dict[str, Dict[str, Dict[str, float]]]) -> float:
        import numpy as np

        total = 0.0
        count = 0
        for group_metrics in metrics.values():
            for task_metrics in group_metrics.values():
                acc = task_metrics.get("acc")
                if acc is None or np.isnan(acc):
                    continue
                total += float(acc)
                count += 1
        if count == 0:
            return float("nan")
        return total / count
from train.losses import LiteJamLoss
from train.utils import format_task_weights, parse_task_weight_spec, resolve_device


PROGRESS_DISABLED = not sys.stderr.isatty()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MobileNetV3-S baseline for LiteJam tasks")
    parser.add_argument("--root", type=Path, default=Path("data/controlled_low_frequency/GNSS_datasets/dataset1"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device (cpu/mps/cuda); 'auto' picks cuda>mps>cpu",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default="adamw")
    parser.add_argument("--scheduler", choices=["none", "plateau"], default="plateau")
    parser.add_argument("--plateau-patience", type=int, default=10)
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--warmup-epochs", type=int, default=3, help="Linear LR warmup epochs")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clip (0 disables)")
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
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights (requires 3-channel input)")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze MobileNet backbone weights")
    parser.add_argument(
        "--channel-strategy",
        choices=["iqfp", "iq", "repeat", "single"],
        default="iqfp",
        help="Input representation from dataset loader",
    )
    parser.add_argument(
        "--task-loss-weights",
        type=str,
        default="auto",
        help="Override task loss weights, e.g. 'det=1,type=1,area=2'.",
    )
    return parser.parse_args()


def build_loaders(
    root: Path,
    groups: List[str],
    batch_size: int,
    num_workers: int,
    *,
    channel_strategy: str,
    disable_bwd_normalize: bool,
) -> Dict[str, Dict[str, object]]:
    loaders: Dict[str, Dict[str, object]] = {}
    for name in groups:
        cfg = TASK_CONFIGS[name]
        dataset_kwargs = {
            "normalize_bwd": not disable_bwd_normalize,
            "channel_strategy": channel_strategy,
        }
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


def save_checkpoint(
    model: MobileNetV3SBaseline,
    model_cfg: MobileNetV3SBaselineConfig,
    head_cfg: HeadsConfig,
    label_maps_by_group: Dict[str, Dict[str, object]],
    optimizer: torch.optim.Optimizer,
    suffix: str,
    run_stamp: str,
) -> Path:
    save_dir = REPO_ROOT / "saved"
    save_dir.mkdir(exist_ok=True)
    if suffix in {"best", "good", "last"}:
        ckpt_path = save_dir / f"mobilenetv3_{run_stamp}_{suffix}.pt"
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        ckpt_path = save_dir / f"mobilenetv3_{run_stamp}_{stamp}_{suffix}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": asdict(model_cfg),
            "heads_config": asdict(head_cfg),
            "label_maps_by_group": label_maps_by_group,
            "optimizer_state": optimizer.state_dict(),
        },
        ckpt_path,
    )
    return ckpt_path


def main() -> None:
    args = parse_args()
    device_str = resolve_device(args.device)
    print(f"[Config] device: {device_str}")
    device = torch.device(device_str)

    loaders = build_loaders(
        root=args.root,
        groups=args.groups,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        channel_strategy=args.channel_strategy,
        disable_bwd_normalize=args.no_bwd_normalize,
    )
    if not loaders:
        raise RuntimeError("No task groups available for training")

    jammer_info = loaders.get("jammer")
    if jammer_info is None:
        raise RuntimeError("Jammer group is required to configure the shared heads")
    head_cfg = HeadsConfig(
        det_classes=2,
        type_classes=len(jammer_info["label_maps"]["type_to_idx"]),
        area_classes=len(jammer_info["label_maps"]["area_to_idx"]),
    )

    sample_tensor, _ = jammer_info["train_dataset"][0]
    in_channels = sample_tensor.shape[0]
    if args.pretrained and in_channels != 3:
        raise ValueError("Pretrained MobileNetV3-S requires 3 input channels; choose a different channel strategy")

    model_cfg = MobileNetV3SBaselineConfig(
        in_channels=in_channels,
        dropout=args.dropout,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        head_config=head_cfg,
    )
    model = MobileNetV3SBaseline(model_cfg).to(device)

    class_weights = compute_class_weights(jammer_info["train_dataset"], head_cfg)
    base_loss_weights = {"det": 1.0, "type": 1.0, "snr": 1.0, "area": 1.0, "bwd": 1.0}
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
            verbose=True,
        )
    else:
        scheduler = None

    label_maps_by_group = {name: info["label_maps"] for name, info in loaders.items()}
    run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    best_val_loss = float("inf")
    best_path: Path | None = None
    best_test_acc = float("-inf")
    good_path: Path | None = None

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

        for group_name, info in loaders.items():
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
            for group_name, info in loaders.items():
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
            loaders,
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
            loaders,
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
            print(f"  -> Improved test accuracy ({test_acc:.4f}); good checkpoint updated")

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
