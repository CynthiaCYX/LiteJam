"""Evaluate ShuffleNetV2 baseline checkpoints on LiteJam task splits."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.controlled_low_frequency_loader import ControlledLowFrequencyDataset
from models.litejam import HeadsConfig
from baseline.shufflenetv2 import ShuffleNetV2Baseline, ShuffleNetV2BaselineConfig
from scripts.eval_litejam import (
    TASK_EVAL_CONFIGS,
    compute_confusion_summary,
    evaluate as run_evaluation,
)
from train.metrics import regression_metrics
from train.utils import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ShuffleNetV2 baseline checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to saved/*.pt checkpoint")
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
        help="Task groups to evaluate",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("eval_results"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--channel-strategy",
        choices=["iqfp", "iq", "repeat", "single"],
        default=None,
        help="Override dataset channel strategy (defaults to checkpoint metadata)",
    )
    return parser.parse_args()


def load_checkpoint(path: Path) -> Dict[str, object]:
    try:
        return torch.load(path, map_location="cpu")
    except TypeError:
        return torch.load(path, map_location="cpu")


def evaluate_group(
    model: ShuffleNetV2Baseline,
    device: torch.device,
    root: Path,
    label_maps: Dict[str, object],
    group_name: str,
    cfg: Dict[str, object],
    batch_size: int,
    num_workers: int,
    output_dir: Path,
    channel_strategy_override: str | None,
) -> None:
    tasks = set(cfg["tasks"])  # type: ignore[arg-type]
    dataset_kwargs = {
        "channel_strategy": channel_strategy_override or label_maps.get("channel_strategy", "iqfp"),
        "normalize_bwd": label_maps.get("normalize_bwd", True),
    }
    test_ds = ControlledLowFrequencyDataset(
        root=root,
        split="test",
        split_group=cfg["split_group"],
        split_suffix=cfg["split_suffix"],
        label_maps=label_maps,
        tasks=tasks,
        **dataset_kwargs,
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
    ) = run_evaluation(model, loader, device, label_maps)

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

    if "snr" in tasks and mask.size:
        snr_active = mask & snr_mask_np.astype(bool)
        if snr_active.any():
            reg = regression_metrics(targets_np["snr"][snr_active], preds_np["snr"][snr_active])
            print(
                f"[snr] rmse={reg['rmse']:.4f}, mae={reg['mae']:.4f}, r2={reg['r2']:.4f}"
            )
        else:
            print("[snr] No valid samples for regression metrics.")

    if "bwd" in tasks and mask.size:
        valid_bwd = mask & bwd_mask_np.astype(bool)
        if valid_bwd.any():
            reg = regression_metrics(targets_np["bwd"][valid_bwd], preds_np["bwd"][valid_bwd])
            print(
                f"[bwd] rmse={reg['rmse']:.4f}, mae={reg['mae']:.4f}, r2={reg['r2']:.4f}"
            )
        else:
            print("[bwd] No valid samples for regression metrics.")

    np.savez(
        group_dir / "predictions.npz",
        targets=targets_np,
        preds=preds_np,
        probs=probs_np,
        det_mask=mask,
        bwd_mask=bwd_mask_np,
        snr_mask=snr_mask_np,
        type_mask=type_mask_np,
        area_mask=area_mask_np,
    )


def main() -> None:
    args = parse_args()
    device_str = resolve_device(args.device)
    print(f"[Config] device: {device_str}")
    device = torch.device(device_str)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = load_checkpoint(args.checkpoint)
    head_cfg = HeadsConfig(**ckpt["heads_config"])
    model_cfg_dict = dict(ckpt["model_config"])
    model_cfg_dict["head_config"] = head_cfg
    model_cfg = ShuffleNetV2BaselineConfig(**model_cfg_dict)
    model = ShuffleNetV2Baseline(model_cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    label_maps_by_group = ckpt.get("label_maps_by_group")
    if not label_maps_by_group:
        default_maps = ckpt.get("label_maps")
        if default_maps is None:
            raise RuntimeError("Checkpoint is missing label maps for evaluation")
        label_maps_by_group = {"jammer": default_maps}

    for group_name in args.groups:
        cfg = TASK_EVAL_CONFIGS[group_name]
        label_maps = label_maps_by_group.get(group_name)
        if label_maps is None:
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
            args.channel_strategy,
        )


if __name__ == "__main__":
    main()
