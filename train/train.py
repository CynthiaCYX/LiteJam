"""Training entry point for LiteJam."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ..dataio.collate import default_collate
from ..dataio.dataset import LiteJamDataset
from ..models.litejam import HeadsConfig, LiteJamConfig, LiteJamModel
from .engine import LiteJamEngine
from .losses import LiteJamLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LiteJam model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--output", type=str, default="./artifacts", help="Directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(config_device: str) -> torch.device:
    if config_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config_device)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    set_seed(config.get("seed", 42))
    device = select_device(config.get("device", "auto"))
    print(f"Using device: {device}")

    data_cfg = config["data"]
    root = data_cfg["root"]
    height = data_cfg.get("height", 1024)
    width = data_cfg.get("width", 128)

    train_dataset = LiteJamDataset(root=root, split="train", height=height, width=width, augmentation=True)
    val_dataset = LiteJamDataset(root=root, split="val", height=height, width=width, augmentation=False)

    batch_size = config["train"].get("batch_size", 32)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=default_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=default_collate,
    )

    model_cfg = config["model"]
    head_cfg = HeadsConfig(
        det_classes=data_cfg["num_classes"]["det"],
        type_classes=data_cfg["num_classes"]["type"],
        area_classes=data_cfg["num_classes"]["area"],
    )
    litejam_cfg = LiteJamConfig(
        d_model=model_cfg.get("d_model", 512),
        n_heads=model_cfg.get("n_heads", 4),
        ffn_dim=model_cfg.get("ffn", 256),
        dropout=model_cfg.get("dropout", 0.1),
        top_k=model_cfg.get("dsa_topk", 32),
        head_config=head_cfg,
    )
    model = LiteJamModel(litejam_cfg).to(device)

    weights = config["train"]["loss_weights"]
    loss_fn = LiteJamLoss(weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["lr"], weight_decay=config["train"].get("weight_decay", 0.01))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    engine = LiteJamEngine(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=Path(args.output),
    )

    epochs = args.epochs or config["train"].get("epochs", 200)
    best_path = engine.train(train_loader, val_loader, epochs=epochs)
    print(f"Training complete. Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
