"""Evaluate LiteJam checkpoints on task-specific test splits."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

# Allow running the script directly (python scripts/eval_litejam.py)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.controlled_low_frequency_loader import ControlledLowFrequencyDataset, build_group_datasets
from models.litejam import HeadsConfig, LiteJamConfig, LiteJamModel
from train.metrics import evaluate_outputs, regression_metrics
from train.utils import resolve_device


def compute_confusion_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Iterable[int],
) -> Dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    total = cm.sum()
    accuracy = float(np.trace(cm) / total) if total else 0.0
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f_one = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {
        "cm": cm,
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f_one,
    }


TASK_EVAL_CONFIGS = {
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
    parser = argparse.ArgumentParser(description="Evaluate LiteJam checkpoint on task-specific splits")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to saved/<timestamp>.pt checkpoint")
    parser.add_argument("--root", type=Path, default=Path("data/controlled_low_frequency/GNSS_datasets/dataset1"))
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device (cpu/mps/cuda); 'auto' picks cuda>mps>cpu",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--groups",
        nargs="+",
        default=list(TASK_EVAL_CONFIGS.keys()),
        choices=list(TASK_EVAL_CONFIGS.keys()),
        help="Task groups to evaluate (default: all)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("eval_results"))
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def load_checkpoint(path: Path) -> Dict[str, object]:
    try:
        return torch.load(path, map_location="cpu")
    except TypeError:
        return torch.load(path, map_location="cpu")


def evaluate_group(
    model: LiteJamModel,
    device: torch.device,
    root: Path,
    label_maps: Dict[str, object],
    group_name: str,
    cfg: Dict[str, object],
    batch_size: int,
    num_workers: int,
    output_dir: Path,
) -> None:
    tasks = set(cfg["tasks"])  # type: ignore[arg-type]
    test_ds = ControlledLowFrequencyDataset(
        root=root,
        split="test",
        split_group=cfg["split_group"],
        split_suffix=cfg["split_suffix"],
        label_maps=label_maps,
        tasks=tasks,
    )
    loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=default_collate,
    )

    (
        metrics,
        targets_np,
        preds_np,
        probs_np,
        mask,
        bwd_mask_np,
        snr_mask_np,
        type_mask_np,
        area_mask_np,
    ) = evaluate(model, loader, device, label_maps)

    group_dir = output_dir / group_name
    group_dir.mkdir(parents=True, exist_ok=True)

    task_print_order = ["det", "type", "snr", "area", "bwd"]
    print(f"\n=== [{group_name}] ===")
    for task in task_print_order:
        if task not in tasks:
            continue
        if task not in metrics:
            continue
        metric_line = ", ".join(f"{k}={v:.4f}" for k, v in metrics[task].items())
        print(f"- {task}: {metric_line}")

    # Confusion matrices for classification tasks
    for task in ("det", "type", "area"):
        if task not in tasks:
            continue
        if task == "det":
            labels = [0, 1]
            valid_mask = np.ones_like(mask, dtype=bool)
        elif task == "type":
            labels = [
                test_ds.label_maps["type_to_idx"][cls]
                for cls in sorted(test_ds.label_maps["type_to_idx"].keys())
            ]
            valid_mask = mask & type_mask_np
        else:
            labels = list(range(len(test_ds.label_maps["area_to_idx"])))
            valid_mask = mask & area_mask_np

        if not valid_mask.any():
            print(f"[{task}] No valid samples for confusion matrix.")
            continue

        summary = compute_confusion_summary(targets_np[task][valid_mask], preds_np[task][valid_mask], labels=labels)
        print(f"[{task}] Confusion matrix:\n{summary['cm']}")
        print(
            f"[{task}] accuracy={summary['accuracy']:.4f}, precision={summary['precision_macro']:.4f}, "
            f"recall={summary['recall_macro']:.4f}, f1={summary['f1_macro']:.4f}"
        )

    # Bandwidth regression results
    if "snr" in tasks and mask.size:
        snr_active = mask & snr_mask_np.astype(bool)
        if snr_active.any():
            snr_reg_metrics = regression_metrics(targets_np["snr"][snr_active], preds_np["snr"][snr_active])
            print(
                f"[snr] rmse={snr_reg_metrics['rmse']:.4f}, mae={snr_reg_metrics['mae']:.4f}, "
                f"r2={snr_reg_metrics['r2']:.4f}"
            )
        else:
            print("[snr] No valid samples for regression metrics.")

    if "bwd" in tasks and mask.size:
        valid_bwd = mask & bwd_mask_np.astype(bool)
        if valid_bwd.any():
            reg_metrics = regression_metrics(targets_np["bwd"][valid_bwd], preds_np["bwd"][valid_bwd])
            if reg_metrics["r2"] < 0:
                print(
                    f"[bwd] rmse={reg_metrics['rmse']:.4f}, mae={reg_metrics['mae']:.4f}, r2={reg_metrics['r2']:.4f}"
                    "  <-- warning: R2 < 0 (worse than predicting mean)."
                )
            else:
                print(
                    f"[bwd] rmse={reg_metrics['rmse']:.4f}, mae={reg_metrics['mae']:.4f}, r2={reg_metrics['r2']:.4f}"
                )
        else:
            print("[bwd] No valid samples for regression metrics.")

    # Save predictions for further analysis
    np.savez(
        group_dir / "predictions.npz",
        det_true=targets_np["det"],
        det_pred=preds_np["det"],
        type_true=targets_np["type"],
        type_pred=preds_np["type"],
        snr_true=targets_np["snr"],
        snr_pred=preds_np["snr"],
        area_true=targets_np["area"],
        area_pred=preds_np["area"],
        bwd_true=targets_np["bwd"],
        bwd_pred=preds_np["bwd"],
        det_mask=mask,
        bwd_mask=bwd_mask_np,
        snr_mask=snr_mask_np,
        type_mask=type_mask_np,
        area_mask=area_mask_np,
        det_prob=probs_np.get("det"),
        type_prob=probs_np.get("type"),
    )

    # AUC reporting
    if "det" in tasks:
        try:
            det_auc = roc_auc_score(targets_np["det"], probs_np["det"][:, 1])
            print(f"[det] AUC={det_auc:.4f}")
        except Exception:
            print("[det] AUC undefined (single class).")
    if "type" in tasks:
        valid_mask = mask & type_mask_np
        if valid_mask.sum() > 1:
            try:
                type_auc = roc_auc_score(
                    targets_np["type"][valid_mask],
                    probs_np["type"][valid_mask],
                    multi_class="ovr",
                    average="macro",
                )
                print(f"[type] AUC={type_auc:.4f}")
            except Exception:
                print("[type] AUC undefined (insufficient classes).")


def evaluate(
    model: LiteJamModel,
    loader: DataLoader,
    device: torch.device,
    label_maps: Dict[str, object],
):
    det_targets: List[np.ndarray] = []
    det_preds: List[np.ndarray] = []
    det_probs: List[np.ndarray] = []
    type_targets: List[np.ndarray] = []
    type_preds: List[np.ndarray] = []
    type_probs: List[np.ndarray] = []
    snr_targets: List[np.ndarray] = []
    snr_preds: List[np.ndarray] = []
    area_targets: List[np.ndarray] = []
    area_preds: List[np.ndarray] = []
    bwd_targets: List[np.ndarray] = []
    bwd_preds: List[np.ndarray] = []
    bwd_masks: List[np.ndarray] = []
    snr_masks: List[np.ndarray] = []

    snr_mean = float(label_maps.get("snr_mean", 0.0))
    snr_std = float(label_maps.get("snr_std", 1.0)) or 1.0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = {k: v.to(device) if hasattr(v, "to") else v for k, v in targets.items()}
            outputs = model(inputs, det_labels=targets.get("det"))

            det_targets.append(targets["det"].cpu().numpy())
            det_probs.append(torch.softmax(outputs["det"], dim=1).cpu().numpy())
            det_preds.append(outputs["det"].argmax(dim=1).cpu().numpy())

            type_targets.append(targets["type"].cpu().numpy())
            type_probs.append(torch.softmax(outputs["type"], dim=1).cpu().numpy())
            type_preds.append(outputs["type"].argmax(dim=1).cpu().numpy())

            snr_targets.append((targets["snr"] * snr_std + snr_mean).cpu().numpy())
            snr_preds.append((outputs["snr"] * snr_std + snr_mean).cpu().numpy())
            snr_mask_tensor = targets.get("snr_mask", torch.ones_like(targets["det"], dtype=torch.bool))
            snr_masks.append(snr_mask_tensor.cpu().numpy().astype(bool))

            area_targets.append(targets["area"].cpu().numpy())
            area_preds.append(outputs["area"].argmax(dim=1).cpu().numpy())

            bwd_targets.append(targets["bwd"].cpu().numpy())
            bwd_preds.append(outputs["bwd"].squeeze(-1).cpu().numpy())
            mask_tensor = targets.get("bwd_mask", torch.ones_like(targets["bwd"], dtype=torch.bool))
            bwd_masks.append(mask_tensor.cpu().numpy())

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
    probs_np = {
        "det": np.concatenate(det_probs),
        "type": np.concatenate(type_probs),
    }
    mask = targets_np["det"] == 1
    bwd_mask_np = np.concatenate(bwd_masks)
    snr_mask_np = np.concatenate(snr_masks) if snr_masks else np.ones_like(mask)

    type_mask_np = targets_np["type"] >= 0
    area_mask_np = targets_np["area"] >= 0

    metrics = evaluate_outputs(
        targets_np,
        preds_np,
        mask,
        task_masks={
            "snr": snr_mask_np,
            "bwd": bwd_mask_np,
            "type": type_mask_np,
            "area": area_mask_np,
        },
    )
    return metrics, targets_np, preds_np, probs_np, mask, bwd_mask_np, snr_mask_np, type_mask_np, area_mask_np


def main() -> None:
    args = parse_args()
    device_str = resolve_device(args.device)
    print(f"[Config] device: {device_str}")
    device = torch.device(device_str)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = load_checkpoint(args.checkpoint)
    model_cfg_data = dict(ckpt["litejam_config"])
    head_cfg_data = dict(ckpt["heads_config"])
    head_cfg_data.pop("snr_classes", None)
    head_cfg = HeadsConfig(**head_cfg_data)
    model_cfg_data["head_config"] = head_cfg
    model_cfg = LiteJamConfig(**model_cfg_data)
    model = LiteJamModel(model_cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    label_maps_by_group = ckpt.get("label_maps_by_group")
    if label_maps_by_group is None:
        default_maps = ckpt.get("label_maps")
        label_maps_by_group = {"jammer": default_maps}

    for group_name in args.groups:
        cfg = TASK_EVAL_CONFIGS[group_name]
        label_maps = label_maps_by_group.get(group_name)
        if label_maps is None:
            # Fallback to the first available label map
            label_maps = next(iter(label_maps_by_group.values()))
        evaluate_group(
            model,
            device,
            args.root,
            label_maps,
            group_name,
            cfg,
            args.batch_size,
            args.num_workers,
            output_dir,
        )


if __name__ == "__main__":
    main()
