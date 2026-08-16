#!/usr/bin/env python3
"""A/B benchmark the experimental copyless GDN verification path."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import signal
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import aiohttp
from transformers import AutoTokenizer
from vllm.benchmarks.lib.endpoint_request_func import (
    RequestFuncInput,
    RequestFuncOutput,
    async_request_openai_completions,
)
from vllm.benchmarks.serve import fetch_spec_decode_metrics

MODEL_MAX_LEN = 1280
PROMPT_TOKENS = 1152
OUTPUT_TOKENS = 64
REQUESTS = 8
CONCURRENCY = 4
REPEATS = 2
SPEC_CONFIG = '{"method":"mtp","num_speculative_tokens":1}'
PRODUCTION_FILES = (
    "vllm_metal/attention/impls/gdn_lazy.py",
    "vllm_metal/attention/impls/linear.py",
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def tail(path: Path, lines: int = 200) -> str:
    if not path.exists():
        return "<missing log>"
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


class Server:
    def __init__(self, model_dir: Path, port: int, log_path: Path) -> None:
        self.model_dir = model_dir
        self.port = port
        self.log_path = log_path
        self.process: subprocess.Popen[str] | None = None
        self.handle: Any = None

    def start(self) -> None:
        command = [
            "vllm",
            "serve",
            str(self.model_dir),
            "--served-model-name",
            "qwen35-mtp-copyless-ab",
            "--enable-prefix-caching",
            "--no-async-scheduling",
            "--max-model-len",
            str(MODEL_MAX_LEN),
            "--max-num-seqs",
            str(CONCURRENCY),
            "--max-num-batched-tokens",
            "4096",
            "--block-size",
            "16",
            "--enforce-eager",
            "--stream-interval",
            "1",
            "--speculative-config",
            SPEC_CONFIG,
            "--port",
            str(self.port),
        ]
        env = os.environ.copy()
        env.update(
            {
                "VLLM_METAL_GDN_LAZY_KERNELS": "1",
                "VLLM_METAL_DECODE_PIPELINE": "1",
                "VLLM_METAL_SPEC_VERIFY_WINDOW": "0",
            }
        )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.log_path.open("w")
        self.process = subprocess.Popen(
            command,
            stdout=self.handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True,
        )
        deadline = time.monotonic() + 300
        url = f"http://127.0.0.1:{self.port}/v1/models"
        while time.monotonic() < deadline:
            assert self.process is not None
            code = self.process.poll()
            if code is not None:
                raise RuntimeError(
                    f"server exited with code {code}\n{tail(self.log_path)}"
                )
            try:
                with urllib.request.urlopen(url, timeout=1) as response:
                    if response.status == 200:
                        return
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(2)
        raise RuntimeError(f"server readiness timed out\n{tail(self.log_path)}")

    def stop(self) -> None:
        process = self.process
        self.process = None
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)
        if self.handle is not None:
            self.handle.close()
            self.handle = None


class Benchmarker:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), trust_remote_code=True
        )
        seed = (
            "Apple Silicon speculative decoding uses a shared cached prefix "
            "and a trained multi-token prediction head. "
        )
        text = seed
        while (
            len(self.tokenizer.encode(text, add_special_tokens=False))
            < PROMPT_TOKENS + 16
        ):
            text += seed
        ids = self.tokenizer.encode(text, add_special_tokens=False)[:PROMPT_TOKENS]
        self.prompt = self.tokenizer.decode(ids, skip_special_tokens=False)
        self.prompt_len = len(
            self.tokenizer.encode(self.prompt, add_special_tokens=False)
        )

    async def run(self, label: str, port: int) -> dict[str, Any]:
        base = f"http://127.0.0.1:{port}"
        timeout = aiohttp.ClientTimeout(total=300)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(base + "/v1/models") as response:
                response.raise_for_status()
                model = (await response.json())["data"][0]["id"]

            async def one(tag: str) -> RequestFuncOutput:
                request = RequestFuncInput(
                    prompt=self.prompt,
                    api_url=base + "/v1/completions",
                    prompt_len=self.prompt_len,
                    output_len=OUTPUT_TOKENS,
                    model=model,
                    ignore_eos=True,
                    extra_body={"temperature": 0},
                )
                output = await asyncio.wait_for(
                    async_request_openai_completions(request, session), timeout=300
                )
                if not output.success:
                    raise RuntimeError(f"{label}/{tag}: {output.error}")
                return output

            await one("warmup")
            before = await fetch_spec_decode_metrics(base, session)
            repeat_rows: list[dict[str, Any]] = []
            output_hashes: list[str] = []
            all_outputs: list[RequestFuncOutput] = []
            for repeat in range(REPEATS):
                outputs: list[RequestFuncOutput] = []
                started = time.perf_counter()
                for offset in range(0, REQUESTS, CONCURRENCY):
                    outputs.extend(
                        await asyncio.gather(
                            *(
                                one(f"r{repeat + 1}-{offset + i + 1}")
                                for i in range(CONCURRENCY)
                            )
                        )
                    )
                elapsed = time.perf_counter() - started
                all_outputs.extend(outputs)
                output_tokens = sum(item.output_tokens for item in outputs)
                input_tokens = sum(item.prompt_len for item in outputs)
                payload = json.dumps(
                    [item.generated_text for item in outputs],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode()
                output_hashes.append(hashlib.sha256(payload).hexdigest())
                repeat_rows.append(
                    {
                        "repeat": repeat + 1,
                        "elapsed_s": elapsed,
                        "output_throughput_tok_s": output_tokens / elapsed,
                        "total_token_throughput_tok_s": (
                            input_tokens + output_tokens
                        )
                        / elapsed,
                        "mean_ttft_ms": statistics.mean(
                            item.ttft * 1000 for item in outputs
                        ),
                        "mean_e2e_ms": statistics.mean(
                            item.latency * 1000 for item in outputs
                        ),
                    }
                )
            after = await fetch_spec_decode_metrics(base, session)

        drafts = accepted = 0
        if before is not None and after is not None:
            drafts = after.num_draft_tokens - before.num_draft_tokens
            accepted = after.num_accepted_tokens - before.num_accepted_tokens
        rates = [row["output_throughput_tok_s"] for row in repeat_rows]
        result = {
            "label": label,
            "requests_per_repeat": REQUESTS,
            "repeats": REPEATS,
            "concurrency": CONCURRENCY,
            "prompt_tokens_per_request": self.prompt_len,
            "output_tokens_per_request": OUTPUT_TOKENS,
            "output_throughput_tok_s": statistics.median(rates),
            "output_throughput_cv": statistics.pstdev(rates)
            / statistics.mean(rates),
            "total_token_throughput_tok_s": statistics.median(
                row["total_token_throughput_tok_s"] for row in repeat_rows
            ),
            "mean_ttft_ms": statistics.median(
                row["mean_ttft_ms"] for row in repeat_rows
            ),
            "mean_e2e_ms": statistics.median(
                row["mean_e2e_ms"] for row in repeat_rows
            ),
            "mtp_draft_tokens": drafts,
            "mtp_accepted_tokens": accepted,
            "mtp_acceptance_rate": accepted / drafts if drafts else None,
            "output_hashes": output_hashes,
            "repeat_metrics": repeat_rows,
        }
        print("COPYLESS_AB_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
        return result


async def run_condition(
    benchmarker: Benchmarker,
    label: str,
    model_dir: Path,
    output_dir: Path,
    port: int,
) -> dict[str, Any]:
    server = Server(model_dir, port, output_dir / "logs" / f"{label}.log")
    try:
        server.start()
        result = await benchmarker.run(label, port)
        write_json(output_dir / f"{label}.json", result)
        return result
    finally:
        server.stop()


def run_command(command: list[str], log_path: Path | None = None) -> None:
    if log_path is None:
        subprocess.run(command, check=True)
        return
    with log_path.open("w") as handle:
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)


async def async_main(args: argparse.Namespace) -> int:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    benchmarker = Benchmarker(args.model_dir)

    control_start = await run_condition(
        benchmarker, "control_start", args.model_dir, output_dir, args.port
    )

    run_command(["python", "benchmarks/apply_copyless_gdn_state.py"])
    run_command(["python", "-m", "py_compile", *PRODUCTION_FILES])
    run_command(
        [
            "pytest",
            "-q",
            "tests/test_gdn_lazy_wrapper.py",
            "tests/attention/test_align_gdn_state_manager.py",
            "tests/test_qwen_native_mtp_proposer.py",
            "tests/test_qwen_mtp_paged_cache.py",
        ],
        output_dir / "pytest-copyless.log",
    )
    run_command(["git", "diff", "--", *PRODUCTION_FILES], output_dir / "copyless.patch")

    copyless = await run_condition(
        benchmarker, "copyless", args.model_dir, output_dir, args.port + 1
    )

    run_command(["git", "checkout", "--", *PRODUCTION_FILES])
    run_command(["python", "-m", "py_compile", *PRODUCTION_FILES])
    control_end = await run_condition(
        benchmarker, "control_end", args.model_dir, output_dir, args.port + 2
    )

    controls = [
        control_start["output_throughput_tok_s"],
        control_end["output_throughput_tok_s"],
    ]
    median_control = statistics.median(controls)
    speedup = copyless["output_throughput_tok_s"] / median_control
    drift = control_end["output_throughput_tok_s"] / control_start[
        "output_throughput_tok_s"
    ] - 1
    reference_hashes = control_start["output_hashes"]
    outputs_identical = (
        control_end["output_hashes"] == reference_hashes
        and copyless["output_hashes"] == reference_hashes
    )
    summary = {
        "median_control_output_tok_s": median_control,
        "copyless_speedup": speedup,
        "control_drift": drift,
        "outputs_identical": outputs_identical,
        "results": {
            "control_start": control_start,
            "copyless": copyless,
            "control_end": control_end,
        },
    }
    write_json(output_dir / "copyless-summary.json", summary)

    lines = [
        "# Qwen MTP copyless GDN state A/B",
        "",
        f"- Median control: {median_control:.3f} output tok/s",
        f"- Copyless speedup: {speedup:.3f}x",
        f"- Control drift: {100 * drift:+.1f}%",
        f"- Greedy outputs identical: **{outputs_identical}**",
        "",
        "| Case | Output tok/s | TTFT ms | E2E ms | Acceptance | CV |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, result in summary["results"].items():
        acceptance = result["mtp_acceptance_rate"]
        acceptance_text = "n/a" if acceptance is None else f"{100 * acceptance:.1f}%"
        lines.append(
            f"| {label} | {result['output_throughput_tok_s']:.3f} | "
            f"{result['mean_ttft_ms']:.1f} | {result['mean_e2e_ms']:.1f} | "
            f"{acceptance_text} | {100 * result['output_throughput_cv']:.1f}% |"
        )
    report = "\n".join(lines) + "\n"
    (output_dir / "copyless-summary.md").write_text(report)
    print(report)
    if not outputs_identical:
        raise RuntimeError("copyless patch changed deterministic greedy output")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8400)
    return parser.parse_args()


def main() -> int:
    return asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
