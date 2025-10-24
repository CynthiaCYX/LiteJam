"""Evaluation entry point for LiteJam."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from ..dataio.collate import default_collate
from ..dataio.dataset import LiteJamDataset
from ..models.litejam import HeadsConfig, LiteJamConfig, LiteJamModel
from .metrics import evaluate_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LiteJam model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output", type=str, default="./artifacts")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def select_device(config_device: str) -> torch.device:
    if config_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config_device)


def save_table(metrics: Dict[str, Dict[str, float]], out_dir: Path) -> None:
    headers = {
        "det": ["acc", "precision", "recall", "f1"],
        "type": ["acc", "precision", "recall", "f1"],
        "snr": ["acc", "precision", "recall", "f1"],
        "area": ["acc", "precision", "recall", "f1"],
        "bwd": ["rmse", "mae", "r2"],
    }

    csv_lines = ["task," + ",".join(headers["det"])]
    md_lines = ["| Task | Metric | Value |", "| --- | --- | --- |"]

    for task, values in metrics.items():
        for key in headers[task]:
            value = values.get(key, float("nan"))
            csv_lines.append(f"{task},{key},{value:.6f}")
            md_lines.append(f"| {task} | {key} | {value:.6f} |")

    (out_dir / "metrics.csv").write_text("\n".join(csv_lines), encoding="utf-8")
    (out_dir / "metrics.md").write_text("\n".join(md_lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = select_device(config.get("device", "auto"))

    data_cfg = config["data"]
    dataset = LiteJamDataset(
        root=data_cfg["root"],
        split=args.split,
        height=data_cfg.get("height", 1024),
        width=data_cfg.get("width", 128),
        augmentation=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=config["train"].get("batch_size", 32),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=default_collate,
    )

    head_cfg = HeadsConfig(
        det_classes=data_cfg["num_classes"]["det"],
        type_classes=data_cfg["num_classes"]["type"],
        area_classes=data_cfg["num_classes"]["area"],
    )
    model_cfg = config["model"]
    litejam_cfg = LiteJamConfig(
        d_model=model_cfg.get("d_model", 512),
        n_heads=model_cfg.get("n_heads", 4),
        ffn_dim=model_cfg.get("ffn", 256),
        dropout=model_cfg.get("dropout", 0.1),
        top_k=model_cfg.get("dsa_topk", 32),
        head_config=head_cfg,
    )
    model = LiteJamModel(litejam_cfg).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    all_targets: Dict[str, list] = {k: [] for k in ["det", "type", "snr", "area", "bwd"]}
    all_preds: Dict[str, list] = {k: [] for k in ["det", "type", "snr", "area", "bwd"]}
    det_masks: list = []
    snr_masks: list = []
    bwd_masks: list = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            targets = {k: v.to(device) for k, v in batch["targets"].items()}
            outputs = model(inputs, det_labels=targets["det"], use_pred_mask=False)

            det_mask = (targets["det"] == 1).cpu().numpy().astype(bool)
            det_masks.append(det_mask)
            snr_mask_tensor = targets.get("snr_mask", torch.ones_like(targets["det"], dtype=torch.bool))
            snr_masks.append(snr_mask_tensor.cpu().numpy().astype(bool))
            bwd_mask_tensor = targets.get("bwd_mask", torch.ones_like(targets["bwd"], dtype=torch.bool))
            bwd_masks.append(bwd_mask_tensor.cpu().numpy().astype(bool))
            for key in all_targets:
                all_targets[key].append(targets[key].cpu().numpy())
            all_preds["det"].append(outputs["det"].argmax(dim=1).cpu().numpy())
            all_preds["type"].append(outputs["type"].argmax(dim=1).cpu().numpy())
            all_preds["snr"].append(outputs["snr"].cpu().numpy())
            all_preds["area"].append(outputs["area"].argmax(dim=1).cpu().numpy())
            all_preds["bwd"].append(outputs["bwd"].squeeze(-1).cpu().numpy())

    targets_np = {k: np.concatenate(v, axis=0) for k, v in all_targets.items()}
    preds_np = {k: np.concatenate(v, axis=0) for k, v in all_preds.items()}
    mask_np = np.concatenate(det_masks, axis=0)
    snr_mask_np = np.concatenate(snr_masks, axis=0)
    bwd_mask_np = np.concatenate(bwd_masks, axis=0)
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
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_table(metrics, out_dir)

    # Optional confusion matrices for downstream tasks
    if mask_np.any():
        type_active = mask_np & type_mask_np
        area_active = mask_np & area_mask_np
        if type_active.any():
            cm_type = confusion_matrix(targets_np["type"][type_active], preds_np["type"][type_active])
            np.savetxt(out_dir / "confusion_type.csv", cm_type, fmt="%d", delimiter=",")
        if area_active.any():
            cm_area = confusion_matrix(targets_np["area"][area_active], preds_np["area"][area_active])
            np.savetxt(out_dir / "confusion_area.csv", cm_area, fmt="%d", delimiter=",")

    print("Evaluation complete. Metrics saved to", out_dir)


if __name__ == "__main__":
    main()
