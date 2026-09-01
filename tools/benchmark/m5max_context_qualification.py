#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Qualify cold prefill, prefix reuse, and long-context decode on an M5 Max.

The client attaches to an already-running OpenAI-compatible vLLM server and
measures three distinct workloads at exact tokenizer lengths:

- ``cold``: unique prompts whose first cache block differs;
- ``exact_repeat``: an exact prompt is primed, then submitted again;
- ``shared_suffix``: a long block-aligned prefix is primed, then reused with
  request-specific suffixes.

Streaming responses provide TTFT independently from end-to-end latency. The
report records per-request token counts and hashes plus aggregate output
throughput. Run each server configuration in a fresh process so cache and MLX
allocator state do not leak between profiles.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import aiohttp
from transformers import AutoTokenizer


@dataclass(frozen=True)
class RequestResult:
    scenario: str
    context_tokens: int
    request_index: int
    prompt_tokens: int
    output_tokens: int
    ttft_ms: float
    e2e_ms: float
    output_tok_s_after_first: float | None
    output_sha256: str


@dataclass(frozen=True)
class AggregateResult:
    scenario: str
    context_tokens: int
    concurrency: int
    repetitions: int
    successful_requests: int
    total_prompt_tokens: int
    total_output_tokens: int
    batch_wall_time_s: float
    output_throughput_tok_s: float
    median_ttft_ms: float
    p90_ttft_ms: float
    median_e2e_ms: float
    median_per_request_output_tok_s: float | None
    requests: tuple[RequestResult, ...]


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _parse_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True, help="served model name")
    parser.add_argument(
        "--tokenizer",
        help="tokenizer path; defaults to --model",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", type=_parse_ints, default=(8192, 32768))
    parser.add_argument("--concurrency", type=_parse_ints, default=(1, 4))
    parser.add_argument("--scenarios", default="cold,exact_repeat,shared_suffix")
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=544)
    parser.add_argument("--shared-suffix-tokens", type=int, default=96)
    parser.add_argument("--timeout-s", type=float, default=1800)
    parser.add_argument("--profile-name", default="unnamed")
    args = parser.parse_args()
    args.scenario_values = tuple(
        item.strip() for item in args.scenarios.split(",") if item.strip()
    )
    supported = {"cold", "exact_repeat", "shared_suffix"}
    unknown = set(args.scenario_values) - supported
    if unknown:
        parser.error(f"unsupported scenarios: {sorted(unknown)}")
    if args.output_tokens <= 0 or args.repetitions <= 0:
        parser.error("output tokens and repetitions must be positive")
    if args.block_size <= 0:
        parser.error("block size must be positive")
    return args


class PromptFactory:
    def __init__(self, tokenizer: Any, *, block_size: int, suffix_tokens: int) -> None:
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.suffix_tokens = suffix_tokens
        self._body = (
            "Apple Silicon inference qualification measures first-token latency, "
            "long-context decode bandwidth, scheduler-owned prefix blocks, and "
            "hybrid GatedDeltaNet state without changing model semantics. "
        )
        self._body_ids = self._encode(self._body)
        if not self._body_ids:
            raise ValueError("qualification body tokenized to an empty sequence")

    def _encode(self, text: str) -> list[int]:
        return [
            int(token)
            for token in self.tokenizer.encode(text, add_special_tokens=False)
        ]

    def _fill(self, target: int, *, seed: int) -> list[int]:
        marker = self._encode(
            f"Qualification stream {seed:08d} deterministic marker. "
        )
        output = list(marker)
        while len(output) < target:
            needed = target - len(output)
            output.extend(self._body_ids[:needed])
        return output[:target]

    def cold(self, context_tokens: int, index: int) -> list[int]:
        # The unique marker is inside block zero, invalidating the complete hash
        # chain so every request is genuinely cold.
        return self._fill(context_tokens, seed=1_000_000 + index)

    def exact(self, context_tokens: int) -> list[int]:
        return self._fill(context_tokens, seed=2_000_000)

    def shared_prefix(self, context_tokens: int) -> tuple[list[int], int]:
        available = max(0, context_tokens - self.suffix_tokens)
        prefix_tokens = (available // self.block_size) * self.block_size
        if prefix_tokens <= 0:
            raise ValueError(
                f"context {context_tokens} is too short for one shared block"
            )
        return self._fill(prefix_tokens, seed=3_000_000), prefix_tokens

    def shared_request(self, context_tokens: int, index: int) -> tuple[list[int], int]:
        prefix, prefix_tokens = self.shared_prefix(context_tokens)
        suffix = self._fill(context_tokens - prefix_tokens, seed=4_000_000 + index)
        return prefix + suffix, prefix_tokens


async def _server_model(base_url: str, timeout: aiohttp.ClientTimeout) -> str:
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(base_url.rstrip("/") + "/v1/models") as response:
            response.raise_for_status()
            payload = await response.json()
    return str(payload["data"][0]["id"])


async def _stream_completion(
    *,
    session: aiohttp.ClientSession,
    endpoint: str,
    served_model: str,
    tokenizer: Any,
    prompt_ids: Sequence[int],
    max_tokens: int,
    scenario: str,
    request_index: int,
) -> RequestResult:
    prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
    # Decode/encode can normalize tokenizer-specific byte sequences. Record the
    # exact length the server will see rather than assuming the sliced id count.
    server_prompt_tokens = len(
        tokenizer.encode(prompt, add_special_tokens=False)
    )
    payload = {
        "model": served_model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
    }

    started = time.perf_counter()
    first_token_at: float | None = None
    pieces: list[str] = []
    usage_output_tokens: int | None = None
    buffer = b""

    async with session.post(endpoint, json=payload) as response:
        response.raise_for_status()
        async for chunk in response.content.iter_any():
            buffer += chunk
            while b"\n\n" in buffer:
                event, buffer = buffer.split(b"\n\n", 1)
                for raw_line in event.splitlines():
                    line = raw_line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        continue
                    message = json.loads(data)
                    usage = message.get("usage")
                    if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
                        usage_output_tokens = int(usage["completion_tokens"])
                    for choice in message.get("choices") or []:
                        text = choice.get("text") or ""
                        if text:
                            if first_token_at is None:
                                first_token_at = time.perf_counter()
                            pieces.append(text)

    finished = time.perf_counter()
    generated_text = "".join(pieces)
    output_tokens = usage_output_tokens
    if output_tokens is None:
        output_tokens = len(
            tokenizer.encode(generated_text, add_special_tokens=False)
        )
    if first_token_at is None:
        raise RuntimeError("stream completed without a generated token")

    decode_elapsed = finished - first_token_at
    after_first = None
    if output_tokens > 1 and decode_elapsed > 0:
        after_first = (output_tokens - 1) / decode_elapsed
    return RequestResult(
        scenario=scenario,
        context_tokens=len(prompt_ids),
        request_index=request_index,
        prompt_tokens=server_prompt_tokens,
        output_tokens=output_tokens,
        ttft_ms=(first_token_at - started) * 1000,
        e2e_ms=(finished - started) * 1000,
        output_tok_s_after_first=after_first,
        output_sha256=hashlib.sha256(generated_text.encode("utf-8")).hexdigest(),
    )


async def _prime(
    *,
    session: aiohttp.ClientSession,
    endpoint: str,
    served_model: str,
    tokenizer: Any,
    prompt_ids: Sequence[int],
) -> None:
    await _stream_completion(
        session=session,
        endpoint=endpoint,
        served_model=served_model,
        tokenizer=tokenizer,
        prompt_ids=prompt_ids,
        max_tokens=1,
        scenario="prime",
        request_index=-1,
    )


async def _run_group(
    *,
    scenario: str,
    context_tokens: int,
    concurrency: int,
    repetitions: int,
    output_tokens: int,
    prompt_factory: PromptFactory,
    tokenizer: Any,
    base_url: str,
    served_model: str,
    timeout_s: float,
) -> AggregateResult:
    endpoint = base_url.rstrip("/") + "/v1/completions"
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    connector = aiohttp.TCPConnector(limit=max(8, concurrency * 2))
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        if scenario == "exact_repeat":
            await _prime(
                session=session,
                endpoint=endpoint,
                served_model=served_model,
                tokenizer=tokenizer,
                prompt_ids=prompt_factory.exact(context_tokens),
            )
        elif scenario == "shared_suffix":
            prefix, prefix_tokens = prompt_factory.shared_prefix(context_tokens)
            prime_suffix = prompt_factory._fill(
                context_tokens - prefix_tokens,
                seed=3_500_000,
            )
            await _prime(
                session=session,
                endpoint=endpoint,
                served_model=served_model,
                tokenizer=tokenizer,
                prompt_ids=prefix + prime_suffix,
            )

        all_results: list[RequestResult] = []
        wall_started = time.perf_counter()
        total_requests = concurrency * repetitions
        for repetition in range(repetitions):
            tasks = []
            for lane in range(concurrency):
                index = repetition * concurrency + lane
                if scenario == "cold":
                    prompt_ids = prompt_factory.cold(context_tokens, index)
                elif scenario == "exact_repeat":
                    prompt_ids = prompt_factory.exact(context_tokens)
                else:
                    prompt_ids, _ = prompt_factory.shared_request(
                        context_tokens,
                        index,
                    )
                tasks.append(
                    _stream_completion(
                        session=session,
                        endpoint=endpoint,
                        served_model=served_model,
                        tokenizer=tokenizer,
                        prompt_ids=prompt_ids,
                        max_tokens=output_tokens,
                        scenario=scenario,
                        request_index=index,
                    )
                )
            all_results.extend(await asyncio.gather(*tasks))
        wall_time = time.perf_counter() - wall_started

    if len(all_results) != total_requests:
        raise RuntimeError(
            f"expected {total_requests} results, got {len(all_results)}"
        )
    per_request_speeds = [
        result.output_tok_s_after_first
        for result in all_results
        if result.output_tok_s_after_first is not None
    ]
    total_output = sum(result.output_tokens for result in all_results)
    aggregate = AggregateResult(
        scenario=scenario,
        context_tokens=context_tokens,
        concurrency=concurrency,
        repetitions=repetitions,
        successful_requests=len(all_results),
        total_prompt_tokens=sum(result.prompt_tokens for result in all_results),
        total_output_tokens=total_output,
        batch_wall_time_s=wall_time,
        output_throughput_tok_s=total_output / wall_time,
        median_ttft_ms=statistics.median(result.ttft_ms for result in all_results),
        p90_ttft_ms=_percentile(
            [result.ttft_ms for result in all_results], 0.90
        ),
        median_e2e_ms=statistics.median(result.e2e_ms for result in all_results),
        median_per_request_output_tok_s=(
            statistics.median(per_request_speeds) if per_request_speeds else None
        ),
        requests=tuple(all_results),
    )
    print("M5MAX_CONTEXT_RESULT=" + json.dumps(asdict(aggregate), sort_keys=True))
    return aggregate


async def async_main(args: argparse.Namespace) -> int:
    tokenizer_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
    )
    timeout = aiohttp.ClientTimeout(total=min(args.timeout_s, 300))
    served_model = await _server_model(args.base_url, timeout)
    prompt_factory = PromptFactory(
        tokenizer,
        block_size=args.block_size,
        suffix_tokens=args.shared_suffix_tokens,
    )

    results: list[AggregateResult] = []
    for context_tokens in args.contexts:
        for concurrency in args.concurrency:
            for scenario in args.scenario_values:
                results.append(
                    await _run_group(
                        scenario=scenario,
                        context_tokens=context_tokens,
                        concurrency=concurrency,
                        repetitions=args.repetitions,
                        output_tokens=args.output_tokens,
                        prompt_factory=prompt_factory,
                        tokenizer=tokenizer,
                        base_url=args.base_url,
                        served_model=served_model,
                        timeout_s=args.timeout_s,
                    )
                )

    report = {
        "profile_name": args.profile_name,
        "requested_model": args.model,
        "served_model": served_model,
        "tokenizer": tokenizer_path,
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "config": {
            "contexts": list(args.contexts),
            "concurrency": list(args.concurrency),
            "scenarios": list(args.scenario_values),
            "output_tokens": args.output_tokens,
            "repetitions": args.repetitions,
            "block_size": args.block_size,
            "shared_suffix_tokens": args.shared_suffix_tokens,
        },
        "results": [asdict(result) for result in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


def main() -> int:
    return asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
