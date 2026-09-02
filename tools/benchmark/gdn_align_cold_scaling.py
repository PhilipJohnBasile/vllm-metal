#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure hybrid GDN all-cold scaling as prefix-cache state accumulates.

This benchmark targets the unique-prompt regression documented by upstream
vllm-metal PR #584.  It compares separate cache-off and cache-on processes;
each process submits the same unique prompts, then measures a new cold probe
after progressively larger retained-prefix working sets.

The prompt marker differs inside the first scheduler hash unit.  Because vLLM
chains prefix hashes, every later block is cold as well even though the body is
shared.  The benchmark also records GDN state-cache capacity-growth events so
throughput can be correlated with the align-mode high-water pool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProbeResult:
    retained_prompt_target: int
    prompt_tokens: int
    output_tokens: int
    wall_time_s: float
    wall_output_tok_s: float
    ttft_ms: float | None
    decode_time_s: float | None
    decode_tok_s: float | None
    e2e_ms: float | None
    mlx_active_bytes: int | None
    mlx_cache_bytes: int | None
    mlx_peak_bytes: int | None
    output_sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("off", "on"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=1152)
    parser.add_argument("--probe-output-tokens", type=int, default=32)
    parser.add_argument("--stages", default="0,8,24")
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    args = parser.parse_args()
    args.stage_values = tuple(int(item) for item in args.stages.split(","))
    if not args.stage_values or args.stage_values[0] != 0:
        parser.error("--stages must start at zero")
    if sorted(set(args.stage_values)) != list(args.stage_values):
        parser.error("--stages must be strictly increasing")
    return args


def _memory_value(mx: Any, name: str) -> int | None:
    fn = getattr(mx, name, None)
    if fn is None:
        metal = getattr(mx, "metal", None)
        fn = getattr(metal, name, None) if metal is not None else None
    if fn is None:
        return None
    try:
        return int(fn())
    except Exception:
        return None


def _metric_times(output: Any) -> tuple[float | None, float | None, float | None]:
    metrics = getattr(output, "metrics", None)
    if metrics is None:
        return None, None, None
    arrival = getattr(metrics, "arrival_time", None)
    first = getattr(metrics, "first_token_time", None)
    finished = getattr(metrics, "finished_time", None)
    if not all(isinstance(value, (int, float)) for value in (arrival, first, finished)):
        return None, None, None
    return float(first - arrival), float(finished - first), float(finished - arrival)


def _build_prompt(tokenizer: Any, index: int, target_tokens: int) -> list[int]:
    marker = (
        f"Cold request identity {index:08d}; deterministic nonce "
        f"{index * 2654435761 & 0xFFFFFFFF:08x}. "
    )
    body = (
        "Apple Silicon long-context inference uses paged attention, hybrid "
        "GatedDeltaNet state, prefix caching, and quantized model weights. "
    )
    text = marker
    while len(tokenizer.encode(text, add_special_tokens=False)) < target_tokens + 16:
        text += body
    return tokenizer.encode(text, add_special_tokens=False)[:target_tokens]


def main() -> int:
    args = parse_args()
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_METAL_USE_PAGED_ATTENTION", "1")
    os.environ.setdefault("VLLM_METAL_MEMORY_FRACTION", "auto")
    os.environ.setdefault("VLLM_METAL_GDN_LAZY_KERNELS", "1")
    os.environ.setdefault("VLLM_METAL_DECODE_PIPELINE", "1")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")

    import mlx.core as mx
    from vllm import LLM, SamplingParams

    from vllm_metal.attention.caches.gdn_cache import GDNPagedStateCache

    capacity_events: list[dict[str, int]] = []
    original_ensure_capacity = GDNPagedStateCache.ensure_capacity

    def traced_ensure_capacity(self: Any, num_seqs: int) -> None:
        before = int(self.allocated_seqs)
        original_ensure_capacity(self, num_seqs)
        after = int(self.allocated_seqs)
        if after != before:
            capacity_events.append(
                {
                    "requested": int(num_seqs),
                    "before": before,
                    "after": after,
                    "max_seqs": int(self.max_seqs),
                }
            )

    GDNPagedStateCache.ensure_capacity = traced_ensure_capacity
    enable_prefix_caching = args.mode == "on"
    started = time.perf_counter()
    try:
        llm = LLM(
            model=args.model,
            max_model_len=max(args.prompt_tokens + args.probe_output_tokens + 64, 1536),
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            enable_chunked_prefill=True,
            enable_prefix_caching=enable_prefix_caching,
        )
        cache_config = llm.llm_engine.vllm_config.cache_config
        tokenizer = llm.get_tokenizer()

        populate_params = SamplingParams(
            temperature=0,
            max_tokens=1,
            ignore_eos=True,
        )
        probe_params = SamplingParams(
            temperature=0,
            max_tokens=args.probe_output_tokens,
            ignore_eos=True,
        )

        # Compile/warm the same path before timed work.  The warmup prompt is
        # unique and therefore cannot accidentally become a hit for a probe.
        llm.generate(
            [
                {
                    "prompt_token_ids": _build_prompt(
                        tokenizer, 900_000, args.prompt_tokens
                    )
                }
            ],
            populate_params,
        )

        results: list[ProbeResult] = []
        populated = 0
        next_prompt_index = 0
        for stage in args.stage_values:
            to_add = stage - populated
            while to_add > 0:
                batch_size = min(args.max_num_seqs, to_add)
                batch = [
                    {
                        "prompt_token_ids": _build_prompt(
                            tokenizer,
                            next_prompt_index + offset,
                            args.prompt_tokens,
                        )
                    }
                    for offset in range(batch_size)
                ]
                llm.generate(batch, populate_params)
                next_prompt_index += batch_size
                populated += batch_size
                to_add -= batch_size

            probe_ids = _build_prompt(
                tokenizer,
                1_000_000 + stage,
                args.prompt_tokens,
            )
            probe_started = time.perf_counter()
            output = llm.generate(
                [{"prompt_token_ids": probe_ids}],
                probe_params,
            )[0]
            wall_time = time.perf_counter() - probe_started
            generated_ids = list(output.outputs[0].token_ids)
            generated = len(generated_ids)
            output_sha256 = hashlib.sha256(
                json.dumps(generated_ids, separators=(",", ":")).encode()
            ).hexdigest()
            ttft, decode_time, e2e = _metric_times(output)
            decode_tok_s = None
            if decode_time is not None and decode_time > 0 and generated > 1:
                decode_tok_s = (generated - 1) / decode_time
            result = ProbeResult(
                retained_prompt_target=stage,
                prompt_tokens=len(probe_ids),
                output_tokens=generated,
                wall_time_s=wall_time,
                wall_output_tok_s=generated / wall_time,
                ttft_ms=ttft * 1000 if ttft is not None else None,
                decode_time_s=decode_time,
                decode_tok_s=decode_tok_s,
                e2e_ms=e2e * 1000 if e2e is not None else None,
                mlx_active_bytes=_memory_value(mx, "get_active_memory"),
                mlx_cache_bytes=_memory_value(mx, "get_cache_memory"),
                mlx_peak_bytes=_memory_value(mx, "get_peak_memory"),
                output_sha256=output_sha256,
            )
            results.append(result)
            print(
                "GDN_COLD_PROBE=" + json.dumps(asdict(result), sort_keys=True),
                flush=True,
            )
    finally:
        GDNPagedStateCache.ensure_capacity = original_ensure_capacity

    output = {
        "mode": args.mode,
        "enable_prefix_caching": enable_prefix_caching,
        "defer_decode_state": os.getenv(
            "VLLM_METAL_GDN_DEFER_DECODE_STATE", "0"
        )
        == "1",
        "model": args.model,
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "config": {
            "prompt_tokens": args.prompt_tokens,
            "probe_output_tokens": args.probe_output_tokens,
            "stages": list(args.stage_values),
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "block_size": int(cache_config.block_size),
            "mamba_block_size": int(cache_config.mamba_block_size),
            "mamba_cache_mode": str(cache_config.mamba_cache_mode),
        },
        "engine_start_and_run_s": time.perf_counter() - started,
        "capacity_events": capacity_events,
        "results": [asdict(item) for item in results],
        "median_wall_output_tok_s": statistics.median(
            item.wall_output_tok_s for item in results
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print("GDN_COLD_RESULT=" + json.dumps(output, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
