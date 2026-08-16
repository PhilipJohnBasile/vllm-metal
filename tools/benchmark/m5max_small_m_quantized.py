#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure M5-class small-M quantized matmul dispatch.

Speculative verification and batched decode exercise matrices with only a few
rows. This benchmark records the latency cliff across M=1..32 for configurable
Qwen-like shapes and quantization modes. Run it against both the vllm-metal
pinned MLX build and a candidate MLX revision before changing the ABI pin.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import mlx.core as mx


def _parse_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--m-values", type=_parse_ints, default=(1, 2, 4, 8, 12, 16, 32))
    parser.add_argument("--k-values", type=_parse_ints, default=(5120,))
    parser.add_argument("--n-values", type=_parse_ints, default=(13824, 248320))
    parser.add_argument("--bits", type=_parse_ints, default=(4, 8))
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--repetitions", type=int, default=40)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    args = parser.parse_args()
    if args.group_size <= 0 or args.warmup < 0 or args.repetitions <= 0 or args.trials <= 0:
        parser.error("invalid benchmark count/group size")
    return args


def _device_info() -> dict[str, Any]:
    try:
        info = dict(mx.device_info())
    except Exception as exc:
        info = {"error": repr(exc)}
    return {str(key): value for key, value in info.items()}


def _bench(
    *,
    m: int,
    k: int,
    n: int,
    bits: int,
    group_size: int,
    dtype: Any,
    warmup: int,
    repetitions: int,
    trials: int,
) -> dict[str, Any]:
    # Keep one quantized weight resident and vary only the input row count.
    key = mx.random.key((m * 1_000_003 + k * 101 + n + bits) & 0xFFFFFFFF)
    weight = (mx.random.normal((n, k), key=key) / (k**0.5)).astype(dtype)
    quantized = mx.quantize(weight, group_size=group_size, bits=bits)
    del weight
    x = (
        mx.random.normal((m, k), key=mx.random.split(key, 2)[1]) / (k**0.5)
    ).astype(dtype)
    mx.eval(x, *quantized)

    def run() -> None:
        output = mx.quantized_matmul(
            x,
            *quantized,
            transpose=True,
            group_size=group_size,
            bits=bits,
        )
        mx.eval(output)

    for _ in range(warmup):
        run()
    trial_us: list[float] = []
    for _ in range(trials):
        started = time.perf_counter()
        for _ in range(repetitions):
            run()
        trial_us.append((time.perf_counter() - started) * 1e6 / repetitions)
    return {
        "m": m,
        "k": k,
        "n": n,
        "bits": bits,
        "group_size": group_size,
        "median_us": statistics.median(trial_us),
        "min_us": min(trial_us),
        "max_us": max(trial_us),
        "trial_us": trial_us,
        "rows_per_second": m / (statistics.median(trial_us) / 1e6),
    }


def main() -> int:
    args = parse_args()
    dtype = mx.float16 if args.dtype == "float16" else mx.bfloat16
    results: list[dict[str, Any]] = []
    for bits in args.bits:
        for k in args.k_values:
            for n in args.n_values:
                for m in args.m_values:
                    result = _bench(
                        m=m,
                        k=k,
                        n=n,
                        bits=bits,
                        group_size=args.group_size,
                        dtype=dtype,
                        warmup=args.warmup,
                        repetitions=args.repetitions,
                        trials=args.trials,
                    )
                    results.append(result)
                    print("SMALL_M_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
                    mx.clear_cache()

    report = {
        "mlx_version": getattr(mx, "__version__", "unknown"),
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "device_info": _device_info(),
        },
        "config": {
            "m_values": list(args.m_values),
            "k_values": list(args.k_values),
            "n_values": list(args.n_values),
            "bits": list(args.bits),
            "group_size": args.group_size,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "trials": args.trials,
            "dtype": args.dtype,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
