"""Training and evaluation loops for LiteJam."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.litejam import LiteJamModel
from .losses import LiteJamLoss
from .metrics import evaluate_outputs


@dataclass
class StepResult:
    loss: float
    metrics: Dict[str, Dict[str, float]]


class LiteJamEngine:
    """Co-ordinates training, validation and evaluation."""

    def __init__(
        self,
        model: LiteJamModel,
        loss_fn: LiteJamLoss,
        optimizer: Optimizer,
        scheduler: Optional[ReduceLROnPlateau],
        device: torch.device,
        save_dir: Path,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        mask_threshold: float = 0.5,
    ) -> Path:
        best_val = math.inf
        best_path = self.save_dir / "best.pt"
        warmup_epochs = 5
        tau_start = 1.5
        tau_final = 0.3
        det_best_f1 = -math.inf
        det_plateau_epochs = 0
        freeze_patience = 5
        det_frozen = False

        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}")
            if epoch <= warmup_epochs:
                forward_opts = {
                    "mask_mode": "soft",
                    "tau": tau_start,
                    "mask_threshold": mask_threshold,
                    "use_pred_mask": False,
                    "use_gt_mask": True,
                }
            elif not det_frozen:
                progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
                tau_epoch = max(tau_final, tau_start - (tau_start - tau_final) * progress)
                forward_opts = {
                    "mask_mode": "soft",
                    "tau": tau_epoch,
                    "mask_threshold": mask_threshold,
                    "use_pred_mask": True,
                    "use_gt_mask": False,
                }
            else:
                forward_opts = {
                    "mask_mode": "hard",
                    "tau": tau_final,
                    "mask_threshold": mask_threshold,
                    "use_pred_mask": True,
                    "use_gt_mask": False,
                }

            train_result = self._run_epoch(train_loader, train=True, forward_kwargs=forward_opts)
            val_result = self._run_epoch(val_loader, train=False, forward_kwargs=forward_opts)

            print(f"  Train loss: {train_result.loss:.4f}")
            print(f"  Val loss:   {val_result.loss:.4f}")

            if self.scheduler is not None:
                self.scheduler.step(val_result.loss)

            if val_result.loss < best_val:
                best_val = val_result.loss
                torch.save({
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                }, best_path)
                print(f"  Saved new best checkpoint to {best_path}")

            det_metrics = val_result.metrics.get("det", {})
            det_f1 = det_metrics.get("f1")
            if det_f1 is not None and not math.isnan(det_f1):
                if det_f1 > det_best_f1 + 1e-4:
                    det_best_f1 = det_f1
                    det_plateau_epochs = 0
                else:
                    det_plateau_epochs += 1
            if not det_frozen and epoch > warmup_epochs and det_plateau_epochs >= freeze_patience:
                self.model.freeze_heads({"det": False})
                det_frozen = True
                print("  Detection head frozen; switched to hard mask inference.")
        return best_path

    def _run_epoch(
        self,
        loader: DataLoader,
        train: bool,
        forward_kwargs: Optional[Dict[str, Any]] = None,
    ) -> StepResult:
        mode = "train" if train else "eval"
        self.model.train(mode == "train")
        epoch_loss = 0.0
        total_samples = 0

        all_targets: Dict[str, List[np.ndarray]] = {k: [] for k in ["det", "type", "snr", "area", "bwd"]}
        all_preds: Dict[str, List[np.ndarray]] = {k: [] for k in ["det", "type", "snr", "area", "bwd"]}
        det_mask_list: List[np.ndarray] = []
        snr_mask_list: List[np.ndarray] = []
        bwd_mask_list: List[np.ndarray] = []

        base_kwargs: Dict[str, Any] = dict(forward_kwargs or {})
        use_gt_mask = bool(base_kwargs.pop("use_gt_mask", False))

        iterator = tqdm(loader, desc=mode, leave=False)
        for batch in iterator:
            inputs, targets = self._move_batch(batch)
            with torch.set_grad_enabled(train):
                model_kwargs = dict(base_kwargs)
                if use_gt_mask:
                    model_kwargs["det_labels"] = targets["det"]
                outputs = self.model(inputs, **model_kwargs)
                loss_dict = self.loss_fn(outputs, targets)
                loss = loss_dict["total"]

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            batch_size = inputs.size(0)
            epoch_loss += loss.item() * batch_size
            total_samples += batch_size

            det_mask = (targets["det"] == 1).detach().cpu().numpy().astype(bool)
            det_mask_list.append(det_mask)

            snr_mask_tensor = targets.get("snr_mask", torch.ones_like(targets["det"], dtype=torch.bool))
            snr_mask_list.append(snr_mask_tensor.cpu().numpy().astype(bool))
            bwd_mask_tensor = targets.get("bwd_mask", torch.ones_like(targets["bwd"], dtype=torch.bool))
            bwd_mask_list.append(bwd_mask_tensor.cpu().numpy().astype(bool))

            all_targets["det"].append(targets["det"].cpu().numpy())
            all_targets["type"].append(targets["type"].cpu().numpy())
            all_targets["snr"].append(targets["snr"].cpu().numpy())
            all_targets["area"].append(targets["area"].cpu().numpy())
            all_targets["bwd"].append(targets["bwd"].cpu().numpy())

            all_preds["det"].append(outputs["det"].argmax(dim=1).detach().cpu().numpy())
            all_preds["type"].append(outputs["type"].argmax(dim=1).detach().cpu().numpy())
            all_preds["snr"].append(outputs["snr"].detach().cpu().numpy())
            all_preds["area"].append(outputs["area"].argmax(dim=1).detach().cpu().numpy())
            all_preds["bwd"].append(outputs["bwd"].squeeze(-1).detach().cpu().numpy())

        epoch_loss /= max(total_samples, 1)
        targets_np = {k: np.concatenate(v, axis=0) for k, v in all_targets.items()}
        preds_np = {k: np.concatenate(v, axis=0) for k, v in all_preds.items()}
        det_mask_np = np.concatenate(det_mask_list, axis=0)
        snr_mask_np = np.concatenate(snr_mask_list, axis=0)
        bwd_mask_np = np.concatenate(bwd_mask_list, axis=0)
        type_mask_np = targets_np["type"] >= 0
        area_mask_np = targets_np["area"] >= 0
        metrics = evaluate_outputs(
            targets=targets_np,
            preds=preds_np,
            mask=det_mask_np,
            task_masks={
                "snr": snr_mask_np,
                "area": area_mask_np,
                "type": type_mask_np,
                "bwd": bwd_mask_np,
            },
        )
        iterator.close()
        return StepResult(loss=epoch_loss, metrics=metrics)

    def _move_batch(self, batch: Dict[str, Dict[str, Tensor]]) -> Tuple[Tensor, Dict[str, Tensor]]:
        inputs = batch["inputs"].to(self.device, non_blocking=True)
        targets = {k: v.to(self.device, non_blocking=True) for k, v in batch["targets"].items()}
        return inputs, targets


__all__ = ["LiteJamEngine", "StepResult"]
