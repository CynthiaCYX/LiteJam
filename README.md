# LiteJam

LiteJam is a lightweight multi-task learning framework for GNSS interference
analysis. A custom GNSS encoder, dynamic sparse attention, and hierarchical task
heads deliver five predictions from a single forward pass: interference
detection, jammer type, antenna sector, SNR bin, and bandwidth regression.

---

## Features
- Joint optimisation of five GNSS interference tasks with masked multi-split supervision.
- Dynamic sparse attention transformer blocks with optional hierarchical heads.
- Controlled Low Frequency dataset tooling, statistics, and experiment recipes.
- Comprehensive ablations, efficiency profiling, and hyperparameter sweeps.
- Drop-in baselines (ResNet-18, MobileNetV3, ShuffleNetV2, GhostNet, EfficientFormer, MobileOne).

---

## Project Layout
- `baseline/` – baseline model definitions plus training/eval entry points.
- `data/` – Controlled Low Frequency dataset loader and helpers.
- `models/` – LiteJam encoder, transformer blocks, and task heads.
- `scripts/` – training, evaluation, analysis, profiling, and sweep utilities.
- `tests/` – unit tests covering forwards, losses, metrics.
- `train/` – shared training utilities (losses, metrics, helpers).

---

## Setup
```bash
pip install -r requirements.txt
```

Optional dependencies:
- `timm` for GhostNet, EfficientFormer, and MobileOne baselines.
- `fvcore` for FLOP counting in profiling experiments.

---

## Dataset Preparation

1. Download the Controlled Low Frequency corpus from  
   `https://gitlab.cc-asp.fraunhofer.de/darcy_gnss/controlled_low_frequency`
2. Extract it to `data/controlled_low_frequency/GNSS_datasets/dataset1`
   preserving the manifest hierarchy.
3. Each manifest line is tab-separated:  
   `relative_path  class_id  amplitude  area  subjammer  position  environment  [orientation]  [fiot_flag]`

Sanity check:
```python
from pathlib import Path
from data.controlled_low_frequency_loader import build_jammer_datasets

root = Path("data/controlled_low_frequency/GNSS_datasets/dataset1")
train_ds, val_ds, test_ds = build_jammer_datasets(root)
print("train samples:", len(train_ds))
sample, targets = train_ds[0]
print(sample.shape, targets)
```

Dataset reports:
```bash
python scripts/dataset_summary.py \
  --root data/controlled_low_frequency/GNSS_datasets/dataset1 \
  --output-dir reports/dataset_summary
```

Aggregate statistics:
```bash
python scripts/dataset_aggregate_summary.py \
  --root data/controlled_low_frequency/GNSS_datasets/dataset1 \
  --output-dir reports/dataset_aggregate
```

Environment label coverage:
```bash
python scripts/environment_cross_stats.py \
  info/splits/environment_cross/env_03.txt
```

---

## Training LiteJam
```bash
python scripts/train_litejam.py \
  --root data/controlled_low_frequency/GNSS_datasets/dataset1 \
  --groups jammer signal_to_noise bandwidth \
  --batch-size 128 \
  --num-workers 16 \
  --device auto \
  --epochs 200 \
  --lr 2e-3 \
  --mask-warmup 10
```

Key switches:
- `--channel-strategy {iqfp,iq,repeat,single}` to select the input representation.
- `--task-loss-weights det=1,type=1,area=1.5` to override automatic weighting.
- `--freeze-det-after` / `--freeze-det-duration` to freeze detection heads.
- `--no-bwd-normalize` to predict raw MHz bandwidth.

---

## Evaluating Checkpoints
```bash
python scripts/eval_litejam.py \
  --checkpoint saved/<timestamp>_best.pt \
  --root data/controlled_low_frequency/GNSS_datasets/dataset1 \
  --groups jammer signal_to_noise bandwidth \
  --batch-size 128 \
  --num-workers 16 \
  --device auto \
  --output-dir eval_results
```

The command logs per-task metrics (accuracy/F1, RMSE/R^2) and can export confusion
matrices when labels are available.

---

## Component Ablations
```bash
python scripts/ablation_litejam.py \
  --root data/controlled_low_frequency/GNSS_datasets/dataset1 \
  --epochs 200 \
  --batch-size 128 \
  --device auto \
  --output-dir ablation_results
```

Optional arguments such as `--ablations litejam_full simple_heads` restrict the
experiment set.

---

## Temporal Window Ablation (Nt)
```bash
python scripts/nt_length_ablation.py \
  --root data/controlled_low_frequency/GNSS_datasets/dataset1 \
  --epochs 200 \
  --batch-size 128 \
  --device auto \
  --output-dir nt_ablation_results
```

---

## Efficiency & Deployability Profiling
```bash
python scripts/profile_models.py \
  --models litejam resnet18 mobilenetv3 \
  --height 1024 \
  --width 128 \
  --batch-size 128 \
  --warmup 10 \
  --iters 200 \
  --output reports/efficiency_summary.txt
```

Parameters include `--device`, `--litejam-tasks`, and `--models` to tailor the
profiling run.

---

## Hyperparameter Sweep
```bash
python scripts/hparam_sweep.py \
  --models litejam ghostnet efficientformer mobileone mobilenetv3 resnet18 shufflenetv2 \
  --root data/controlled_low_frequency/GNSS_datasets/dataset1 \
  --channel-strategy iqfp \
  --epochs 40 \
  --batch-size 128 \
  --num-workers 16 \
  --trials 6 \
  --stream-logs
```

Custom search spaces can be supplied via `--search-space path/to/space.json`.
Results accumulate in `sweeps/results.jsonl`.

---

## Baseline Training & Evaluation

Example: ResNet-18
```bash
python -m baseline.resnet18.train \
  --root data/controlled_low_frequency/GNSS_datasets/dataset1 \
  --groups jammer signal_to_noise bandwidth \
  --batch-size 128 \
  --num-workers 16 \
  --device auto \
  --epochs 200 \
  --lr 2e-3 \
  --mask-warmup 10 \
  --optimizer adamw \
  --scheduler plateau

python -m baseline.resnet18.eval \
  --checkpoint saved/resnet18_<timestamp>_best.pt \
  --root data/controlled_low_frequency/GNSS_datasets/dataset1 \
  --groups jammer signal_to_noise bandwidth \
  --batch-size 128 \
  --num-workers 16 \
  --device auto
```

Swap `resnet18` for `mobilenetv3`, `shufflenetv2`, `ghostnet`, `efficientformer`,
or `mobileone` as needed (install `timm` beforehand for the latter three).

---

## Adjustable Hyperparameters
- CLI flags within `scripts/train_litejam.py` (channel strategy, loss weights, warmup, freeze schedules, etc.).
- LiteJam configuration options: embedding dimension (`d_model`), transformer blocks (`num_transformer_layers`), sparse top-k (`top_k`), dropout rates.
- `LiteJamModel.freeze_heads()` for staged head training and deployment-specific pruning.

---

## License

This project is licensed under the Apache License 2.0. See `LICENSE` for details.
