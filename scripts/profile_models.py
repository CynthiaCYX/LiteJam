#!/usr/bin/env python3
"""
Profile LiteJam and baseline models for efficiency & deployability metrics.

Metrics reported (per model):
  - Parameter count (millions)
  - Estimated model size from state_dict bytes (MB)
  - FLOPs for a single forward pass (GigaFLOPs)
  - Inference latency on CUDA (milliseconds, batch size 1)
  - Peak GPU memory during inference (MB)

Usage:
    python scripts/profile_models.py \
        --models litejam resnet18 mobilenetv3 \
        --height 1024 --width 128 --batch-size 1
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch

# Ensure project root is on sys.path when invoked as a standalone script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.litejam import HeadsConfig, LiteJamConfig, LiteJamModel  # noqa: E402

try:
    from fvcore.nn import FlopCountAnalysis  # type: ignore
except ImportError:  # pragma: no cover
    FlopCountAnalysis = None  # type: ignore

try:
    from thop import profile as thop_profile  # type: ignore
except ImportError:  # pragma: no cover
    thop_profile = None  # type: ignore


@dataclass
class ProfileResult:
    name: str
    params_m: float | None = None
    size_mb: float | None = None
    flops_g: float | None = None
    latency_ms: float | None = None
    peak_mem_mb: float | None = None
    notes: str | None = None


def build_litejam() -> torch.nn.Module:
    cfg = LiteJamConfig()
    if cfg.head_config is None:
        cfg.head_config = HeadsConfig()
    return LiteJamModel(cfg)


def build_resnet18() -> torch.nn.Module:
    from baseline.resnet18.model import ResNet18Baseline, ResNet18BaselineConfig

    return ResNet18Baseline(ResNet18BaselineConfig(head_config=HeadsConfig()))


def build_mobilenetv3() -> torch.nn.Module:
    from baseline.mobilenetv3.model import MobileNetV3SBaseline, MobileNetV3SBaselineConfig

    return MobileNetV3SBaseline(MobileNetV3SBaselineConfig(head_config=HeadsConfig()))


def build_shufflenetv2() -> torch.nn.Module:
    from baseline.shufflenetv2.model import ShuffleNetV2Baseline, ShuffleNetV2BaselineConfig

    return ShuffleNetV2Baseline(ShuffleNetV2BaselineConfig(head_config=HeadsConfig()))


def build_ghostnet() -> torch.nn.Module:
    from baseline.ghostnet.model import GhostNetBaseline, GhostNetBaselineConfig

    return GhostNetBaseline(GhostNetBaselineConfig(head_config=HeadsConfig()))


def build_mobileone() -> torch.nn.Module:
    from baseline.mobileone.model import MobileOneS0Baseline, MobileOneS0BaselineConfig

    return MobileOneS0Baseline(MobileOneS0BaselineConfig(head_config=HeadsConfig()))


def build_efficientformer() -> torch.nn.Module:
    from baseline.efficientformer.model import EfficientFormerL1Baseline, EfficientFormerL1BaselineConfig

    return EfficientFormerL1Baseline(EfficientFormerL1BaselineConfig(head_config=HeadsConfig()))


MODEL_BUILDERS: Dict[str, Callable[[], torch.nn.Module]] = {
    "litejam": build_litejam,
    "resnet18": build_resnet18,
    "mobilenetv3": build_mobilenetv3,
    "shufflenetv2": build_shufflenetv2,
    "ghostnet": build_ghostnet,
    "mobileone": build_mobileone,
    "efficientformer": build_efficientformer,
}


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def estimate_state_dict_size_bytes(model: torch.nn.Module) -> int:
    size = 0
    for tensor in model.state_dict().values():
        size += tensor.nelement() * tensor.element_size()
    return size


def compute_flops(model: torch.nn.Module, dummy: torch.Tensor) -> Tuple[Optional[float], Optional[str]]:
    notes: List[str] = []

    if FlopCountAnalysis is not None:
        try:
            flops = FlopCountAnalysis(model, dummy)
            total = float(flops.total())
            return total, None
        except Exception as exc:  # pragma: no cover - depends on fvcore internals
            notes.append(f"fvcore failed: {exc}")
    else:
        notes.append("fvcore not installed (pip install fvcore)")

    if thop_profile is not None:
        try:
            flops, _ = thop_profile(model, inputs=(dummy,), verbose=False)
            return float(flops), "computed with thop"
        except Exception as exc:  # pragma: no cover
            notes.append(f"thop failed: {exc}")
    elif thop_profile is None:
        notes.append("thop not installed (pip install thop)")

    return None, "; ".join(notes) if notes else None


def measure_latency_and_memory(
    model: torch.nn.Module,
    dummy: torch.Tensor,
    device: torch.device,
    warmup: int,
    iters: int,
) -> Tuple[Optional[float], Optional[float]]:
    if device.type != "cuda":
        return None, None

    if not torch.cuda.is_available():
        return None, None

    target_index = device.index
    if target_index is None:
        target_index = torch.cuda.current_device() if torch.cuda.is_initialized() else 0

    torch.cuda.set_device(target_index)
    actual_device = torch.device("cuda", target_index)

    model = model.to(actual_device)
    dummy = dummy.to(actual_device)
    model.eval()

    torch.cuda.empty_cache()

    with torch.inference_mode():
        for _ in range(max(warmup, 1)):
            _ = model(dummy)
        torch.cuda.synchronize(actual_device)
        torch.cuda.reset_peak_memory_stats(actual_device)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(max(iters, 1)):
            _ = model(dummy)
        end_event.record()
        torch.cuda.synchronize(actual_device)

    elapsed_ms = start_event.elapsed_time(end_event) / max(iters, 1)
    peak_bytes = torch.cuda.max_memory_allocated(actual_device)
    torch.cuda.empty_cache()

    return float(elapsed_ms), float(peak_bytes) / (1024 ** 2)


def profile_model(
    name: str,
    builder: Callable[[], torch.nn.Module],
    device: torch.device,
    dummy_shape: Tuple[int, int, int, int],
    warmup: int,
    iters: int,
) -> ProfileResult:
    try:
        model = builder()
    except Exception as exc:
        return ProfileResult(name=name, notes=f"Instantiation failed: {exc}")

    model.eval()

    params = count_parameters(model)
    size_bytes = estimate_state_dict_size_bytes(model)

    dummy_cpu = torch.randn(dummy_shape, dtype=torch.float32)

    flops_g = None
    notes: List[str] = []

    needs_resize = getattr(model, "input_hw", None)
    if needs_resize is not None:
        h, w = needs_resize if isinstance(needs_resize, (list, tuple)) else (needs_resize, needs_resize)
        dummy_for_flops = torch.randn(dummy_shape[0], dummy_shape[1], h, w, dtype=torch.float32)
    else:
        dummy_for_flops = dummy_cpu.clone()

    try:
        model_cpu = copy.deepcopy(model).cpu()
        flops_val, flop_note = compute_flops(model_cpu, dummy_for_flops)
        if flops_val is not None:
            flops_g = flops_val / 1e9
        if flop_note:
            notes.append(flop_note)
    except Exception as exc:  # pragma: no cover
        notes.append(f"FLOPs failed: {exc}")
    finally:
        del model_cpu  # type: ignore

    latency_ms = None
    peak_mem_mb = None

    if device.type == "cuda" and torch.cuda.is_available():
        dummy_for_latency = dummy_for_flops.clone()
        latency_ms, peak_mem_mb = measure_latency_and_memory(
            model=model,
            dummy=dummy_for_latency,
            device=device,
            warmup=warmup,
            iters=iters,
        )
        if latency_ms is None:
            notes.append("Latency measurement skipped (CUDA unavailable).")
    else:
        notes.append("Latency measurement skipped (no CUDA device).")

    result = ProfileResult(
        name=name,
        params_m=params / 1e6,
        size_mb=size_bytes / (1024 ** 2),
        flops_g=flops_g,
        latency_ms=latency_ms,
        peak_mem_mb=peak_mem_mb,
        notes="; ".join(notes) if notes else None,
    )

    # Ensure we free GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def format_results(results: Iterable[ProfileResult]) -> str:
    headers = ["Model", "Params [M]", "Model Size [MB]", "FLOPs [G]", "Latency [ms]", "Peak Mem [MB]", "Notes"]
    rows: List[List[str]] = [headers]
    for res in results:
        rows.append(
            [
                res.name,
                f"{res.params_m:.3f}" if res.params_m is not None else "n/a",
                f"{res.size_mb:.2f}" if res.size_mb is not None else "n/a",
                f"{res.flops_g:.2f}" if res.flops_g is not None else "n/a",
                f"{res.latency_ms:.3f}" if res.latency_ms is not None else "n/a",
                f"{res.peak_mem_mb:.1f}" if res.peak_mem_mb is not None else "n/a",
                res.notes or "",
            ]
        )

    col_widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]
    lines: List[str] = []
    for row in rows:
        line = " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))
        lines.append(line)
        if row is headers:
            lines.append("-+-".join("-" * col_widths[i] for i in range(len(headers))))
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile LiteJam and baseline models.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_BUILDERS.keys()),
        choices=list(MODEL_BUILDERS.keys()),
        help="Subset of models to profile (default: all).",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for profiling (default: 1).")
    parser.add_argument("--channels", type=int, default=6, help="Number of input channels (default: 6).")
    parser.add_argument("--height", type=int, default=1024, help="Input height (default: 1024).")
    parser.add_argument("--width", type=int, default=128, help="Input width (default: 128).")
    parser.add_argument("--warmup", type=int, default=10, help="Warm-up iterations before timing.")
    parser.add_argument("--iters", type=int, default=50, help="Iterations used for latency timing.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for latency measurement (default: cuda if available).",
    )
    parser.add_argument("--output", type=Path, help="Optional path to save the table (text file).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    dummy_shape = (args.batch_size, args.channels, args.height, args.width)

    results: List[ProfileResult] = []
    for name in args.models:
        builder = MODEL_BUILDERS[name]
        print(f"[INFO] Profiling {name}...")
        res = profile_model(
            name=name,
            builder=builder,
            device=device,
            dummy_shape=dummy_shape,
            warmup=args.warmup,
            iters=args.iters,
        )
        results.append(res)

    table = format_results(results)
    print("\n" + table)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(table)
        print(f"\n[INFO] Saved table to {args.output}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
