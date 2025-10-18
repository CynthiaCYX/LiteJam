"""Simple random-search hyperparameter sweep runner for LiteJam baselines.

The script keeps the search space identical across models to ensure fair
comparison. Each trial launches the corresponding baseline training script
with a sampled configuration and logs the summary metrics printed by the
training run.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

MODEL_MODULES: Dict[str, str] = {
    "litejam": "scripts.train_litejam",
    "ghostnet": "baseline.ghostnet.train",
    "efficientformer": "baseline.efficientformer.train",
    "mobileone": "baseline.mobileone.train",
    "mobilenetv3": "baseline.mobilenetv3.train",
    "resnet18": "baseline.resnet18.train",
    "shufflenetv2": "baseline.shufflenetv2.train",
}

DEFAULT_SEARCH_SPACE: Dict[str, Any] = {
    "lr": [5e-4, 1e-3, 2e-3],
    "weight_decay": [0.0, 5e-5, 1e-4],
    "dropout": [0.1, 0.2, 0.3],
    "optimizer": ["adamw", "adam"],
    "scheduler": ["plateau", "none"],
    "plateau_patience": [5, 10, 15],
    "mask_warmup": [5, 10, 15],
    "grad_clip": [0.0, 1.0],
    "task_loss_weights": [
        "auto",
        "det=1,type=1,area=1.5",
        "det=1,type=1,area=2",
    ],
}

SUMMARY_PATTERN = re.compile(r"^\\[Summary\\]\\s+(?P<name>[a-zA-Z0-9_]+)=(?P<value>[-+eE0-9\.]+)")


def load_search_space(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return DEFAULT_SEARCH_SPACE
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Search space JSON must define an object mapping")
    return data


def sample_value(spec: Any, rng: random.Random) -> Any:
    if isinstance(spec, list):
        if not spec:
            raise ValueError("Search space choice list cannot be empty")
        return rng.choice(spec)
    if isinstance(spec, dict):
        kind = spec.get("type", "choice")
        if kind == "choice":
            return sample_value(spec.get("values", []), rng)
        if kind == "uniform":
            low = float(spec["min"])
            high = float(spec["max"])
            value = rng.uniform(low, high)
            step = spec.get("step")
            if step:
                step = float(step)
                value = round((value - low) / step) * step + low
            precision = spec.get("precision")
            if precision is not None:
                value = round(value, int(precision))
            return value
        if kind == "log_uniform":
            low = float(spec["min"])
            high = float(spec["max"])
            if low <= 0 or high <= 0:
                raise ValueError("log_uniform bounds must be > 0")
            log_low = math.log(low)
            log_high = math.log(high)
            value = math.exp(rng.uniform(log_low, log_high))
            precision = spec.get("precision")
            if precision is not None:
                value = round(value, int(precision))
            return value
        raise ValueError(f"Unsupported search space type: {kind}")
    return spec


def sample_configuration(space: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    return {name: sample_value(spec, rng) for name, spec in space.items()}


def detect_device(requested: str) -> str:
    requested = (requested or "auto").lower()
    if requested != "auto":
        return requested

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda"
        backends = getattr(torch, "backends", None)
        mps = getattr(backends, "mps", None)
        if mps is not None and callable(getattr(mps, "is_available", None)) and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def build_base_args(args: argparse.Namespace, device: str) -> List[str]:
    cli: List[str] = [
        "--root",
        str(args.root),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
    ]
    if device:
        cli.extend(["--device", device])
    if args.channel_strategy:
        cli.extend(["--channel-strategy", args.channel_strategy])
    if args.groups:
        cli.extend(["--groups", *args.groups])
    if args.extra_args:
        cli.extend(args.extra_args)
    return cli


def configuration_to_args(config: Dict[str, Any]) -> List[str]:
    cli: List[str] = []
    for key, value in config.items():
        if value is None:
            continue
        if key == "task_loss_weights":
            cli.extend(["--task-loss-weights", str(value)])
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cli.append(flag)
            continue
        cli.extend([flag, str(value)])
    return cli


def run_trial(
    command: List[str],
    cwd: Path,
    dry_run: bool,
    *,
    stream_logs: bool,
    log_path: Optional[Path],
) -> Dict[str, Any]:
    print("[Sweep] Launching:", " ".join(command))
    sys.stdout.flush()
    if dry_run:
        return {"returncode": 0, "stdout": "", "stderr": ""}

    start = time.time()
    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_lines: List[str] = []
    log_file = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8")

    assert proc.stdout is not None
    for line in proc.stdout:
        if stream_logs:
            sys.stdout.write(line)
            sys.stdout.flush()
        log_lines.append(line)
        if log_file is not None:
            log_file.write(line)
            log_file.flush()
    proc.wait()
    duration = time.time() - start
    output = "".join(log_lines)
    if log_file is not None:
        log_file.close()
    return {
        "returncode": proc.returncode,
        "stdout": output,
        "stderr": "",
        "duration": duration,
    }


def parse_summary(stdout: str) -> Dict[str, float]:
    summaries: Dict[str, float] = {}
    for line in stdout.splitlines():
        match = SUMMARY_PATTERN.match(line.strip())
        if match:
            name = match.group("name")
            value = float(match.group("value"))
            summaries[name] = value
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Random-search hyperparameter sweep for LiteJam baselines")
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_MODULES.keys()), required=True)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root", type=Path, default=Path("data/controlled_low_frequency/GNSS_datasets/dataset1"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device to force (cuda/mps/cpu); default auto detects cuda>mps>cpu.",
    )
    parser.add_argument("--channel-strategy", default=None)
    parser.add_argument("--groups", nargs="*")
    parser.add_argument("--search-space", type=Path)
    parser.add_argument("--output", type=Path, default=Path("sweeps/results.jsonl"))
    parser.add_argument("--log-dir", type=Path, default=Path("sweeps/logs"), help="Directory to store per-trial logs")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, help="Additional args appended to every run")
    parser.add_argument("--stream-logs", action="store_true", help="Stream child stdout to console (default writes to log file only)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    search_space = load_search_space(args.search_space)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.log_dir is not None and not args.dry_run:
        args.log_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    resolved_device = detect_device(args.device)
    if args.device.lower() == "auto":
        print(f"[Sweep] auto-detected device: {resolved_device}")
    base_args = build_base_args(args, resolved_device)

    with args.output.open("a", encoding="utf-8") as sink:
        for model in args.models:
            module = MODEL_MODULES[model]
            model_rng = random.Random(rng.random())  # independent stream per model
            for trial in range(1, args.trials + 1):
                config = sample_configuration(search_space, model_rng)
                cli_args = base_args + configuration_to_args(config)
                command = [PYTHON, "-m", module, *cli_args]
                log_path = None
                if not args.dry_run and args.log_dir is not None:
                    log_path = args.log_dir / f"{model}_trial{trial:03d}.log"
                result = run_trial(
                    command,
                    REPO_ROOT,
                    args.dry_run,
                    stream_logs=args.stream_logs,
                    log_path=log_path,
                )
                summaries = parse_summary(result.get("stdout", ""))
                record = {
                    "model": model,
                    "module": module,
                    "trial": trial,
                    "config": config,
                    "returncode": result.get("returncode"),
                    "duration_sec": result.get("duration"),
                    "summary": summaries,
                }
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                sink.flush()
                if result.get("returncode") != 0:
                    print(f"[Sweep] Trial {trial} for {model} failed with code {result['returncode']}")
                else:
                    if not args.stream_logs and log_path is not None:
                        print(f"[Sweep] Trial {trial} for {model} log saved to {log_path}")
                    if summaries:
                        summary_str = ", ".join(f"{k}={v:.4f}" for k, v in summaries.items())
                        print(f"[Sweep] Trial {trial} for {model} summaries: {summary_str}")
                    else:
                        print(f"[Sweep] Trial {trial} for {model} completed (no summary parsed)")


if __name__ == "__main__":
    main()
