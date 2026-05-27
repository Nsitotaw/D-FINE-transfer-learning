"""
latency_benchmark.py
Measure D-FINE inference latency, throughput, and memory usage.

Metrics reported:
  - Mean / Std / P50 / P95 / P99 latency (ms) per image
  - Throughput (FPS)
  - GPU memory footprint (MB)
  - Parameter count (M)
  - FLOPs / MACs (GFLOPs)

Usage:
    python eval/latency_benchmark.py \
        --config configs/dfine/full_finetune/dfine_s_pascal_voc.yml \
        --checkpoint results/full_finetune/pascal_voc/best.pth \
        --input_size 640 \
        --batch_size 1 \
        --num_iters 200 \
        --output_dir results/latency
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict

import torch
import numpy as np

_here = Path(__file__).parent
for _candidate in [_here.parent / "D-FINE", _here.parent, Path(".")]:
    if (_candidate / "src" / "core").exists():
        sys.path.insert(0, str(_candidate))
        break


def measure_latency(
    model: torch.nn.Module,
    input_size: int = 640,
    batch_size: int = 1,
    num_iters: int = 200,
    warmup: int = 50,
    device: str = "cuda",
) -> Dict[str, float]:
    model = model.to(device).eval()
    dummy = torch.randn(batch_size, 3, input_size, input_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)

    if device == "cuda":
        torch.cuda.synchronize()

    latencies = []
    with torch.no_grad():
        for _ in range(num_iters):
            if device == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end   = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(dummy)
                end.record()
                torch.cuda.synchronize()
                latencies.append(start.elapsed_time(end))  # ms
            else:
                t0 = time.perf_counter()
                _ = model(dummy)
                latencies.append((time.perf_counter() - t0) * 1000)

    latencies = np.array(latencies)
    per_image = latencies / batch_size

    return {
        "mean_ms":   float(per_image.mean()),
        "std_ms":    float(per_image.std()),
        "p50_ms":    float(np.percentile(per_image, 50)),
        "p95_ms":    float(np.percentile(per_image, 95)),
        "p99_ms":    float(np.percentile(per_image, 99)),
        "min_ms":    float(per_image.min()),
        "max_ms":    float(per_image.max()),
        "fps":       float(1000.0 / per_image.mean()),
        "batch_fps": float(batch_size * 1000.0 / latencies.mean()),
    }


def measure_memory(
    model: torch.nn.Module,
    input_size: int = 640,
    batch_size: int = 1,
    device: str = "cuda",
) -> Dict[str, float]:
    if device != "cuda":
        return {"peak_mb": 0, "reserved_mb": 0}

    torch.cuda.reset_peak_memory_stats()
    model = model.to(device).eval()
    dummy = torch.randn(batch_size, 3, input_size, input_size, device=device)

    with torch.no_grad():
        _ = model(dummy)
    torch.cuda.synchronize()

    return {
        "peak_mb":     torch.cuda.max_memory_allocated() / 1e6,
        "reserved_mb": torch.cuda.max_memory_reserved() / 1e6,
    }


def measure_flops(model: torch.nn.Module, input_size: int = 640) -> Dict[str, float]:
    """Estimate FLOPs using fvcore or thop."""
    dummy = torch.randn(1, 3, input_size, input_size)
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model.cpu(), dummy)
        gflops = flops.total() / 1e9
        return {"GFLOPs": gflops}
    except ImportError:
        pass

    try:
        from thop import profile
        macs, params = profile(model.cpu(), inputs=(dummy,), verbose=False)
        return {"GFLOPs": macs * 2 / 1e9, "params_M": params / 1e6}
    except ImportError:
        pass

    # Fallback: just count params
    params = sum(p.numel() for p in model.parameters())
    return {"params_M": params / 1e6}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input_size", type=int, default=640)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_iters",  type=int, default=200)
    parser.add_argument("--warmup",     type=int, default=50)
    parser.add_argument("--output_dir", default="results/latency")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tag",        default="",
                        help="Label for this run (e.g. full_finetune_pascal_voc)")
    args = parser.parse_args()

    print(f"Loading model: {args.checkpoint}")
    try:
        from src.core import YAMLConfig
        cfg = YAMLConfig(args.config, tuning=args.checkpoint)
        model = cfg.model
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state = ckpt.get("ema", {}).get("module", ckpt.get("model", ckpt))
        model.load_state_dict(state, strict=False)
        model.eval()
    except Exception as e:
        print(f"Model load error: {e}")
        raise

    print(f"Measuring latency (input={args.input_size}, bs={args.batch_size}, "
          f"iters={args.num_iters}, device={args.device})...")

    latency = measure_latency(
        model,
        input_size=args.input_size,
        batch_size=args.batch_size,
        num_iters=args.num_iters,
        warmup=args.warmup,
        device=args.device,
    )

    memory = measure_memory(model, args.input_size, args.batch_size, args.device)
    flops  = measure_flops(model, args.input_size)

    params_total = sum(p.numel() for p in model.parameters()) / 1e6
    params_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    results = {
        "tag":            args.tag or Path(args.checkpoint).stem,
        "input_size":     args.input_size,
        "batch_size":     args.batch_size,
        "device":         args.device,
        "latency":        latency,
        "memory":         memory,
        "flops":          flops,
        "params_total_M": params_total,
        "params_train_M": params_train,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    tag = args.tag.replace(" ", "_") or Path(args.checkpoint).stem
    out_path = os.path.join(args.output_dir, f"latency_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Latency Results ===")
    print(f"  Mean    : {latency['mean_ms']:.2f} ms")
    print(f"  P50     : {latency['p50_ms']:.2f} ms")
    print(f"  P95     : {latency['p95_ms']:.2f} ms")
    print(f"  FPS     : {latency['fps']:.1f}")
    print(f"  GPU mem : {memory.get('peak_mb', 0):.0f} MB")
    print(f"  Params  : {params_total:.2f}M total")
    print(f"  GFLOPs  : {flops.get('GFLOPs', 'N/A')}")
    print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
