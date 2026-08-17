#!/usr/bin/env python3
"""Qualify native Qwen3.6-27B MTP serving on a physical M5 Max.

The harness compares two ordinary baseline launches with two native-MTP
launches, uses deterministic shared-prefix workloads, verifies exact output
parity, records speculative acceptance and lifecycle evidence, and evaluates a
predeclared positive-performance gate without hiding negative results.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import aiohttp
import mlx.core as mx
from mlx_lm.generate import mtp_generate_step
from mlx_lm.utils import load
from safetensors import safe_open
from transformers import AutoTokenizer
from vllm.benchmarks.lib.endpoint_request_func import (
    RequestFuncInput,
    RequestFuncOutput,
    async_request_openai_completions,
)
from vllm.benchmarks.serve import fetch_spec_decode_metrics


SPEC_CONFIG = '{"method":"mtp","num_speculative_tokens":1}'
SERVED_MODEL_NAME = "qwen36-m5max-mtp-qualification"


@dataclass(frozen=True)
class Profile:
    name: str
    mode: str


@dataclass(frozen=True)
class Workload:
    name: str
    prompt_tokens: int
    output_tokens: int
    concurrency: int
    requests: int


PROFILES: tuple[Profile, ...] = (
    Profile("baseline_first", "baseline"),
    Profile("mtp_first", "mtp"),
    Profile("baseline_second", "baseline"),
    Profile("mtp_second", "mtp"),
)

WORKLOADS: tuple[Workload, ...] = (
    Workload("interactive_c1", 512, 256, 1, 3),
    Workload("serving_c4", 1024, 128, 4, 8),
    Workload("long_prefix_c1", 8192, 128, 1, 2),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--request-timeout-s", type=float, default=900.0)
    parser.add_argument("--server-ready-timeout-s", type=float, default=1200.0)
    parser.add_argument("--min-geomean-speedup", type=float, default=1.05)
    parser.add_argument("--min-workload-speedup", type=float, default=0.95)
    parser.add_argument("--max-cv", type=float, default=0.15)
    parser.add_argument("--max-launch-drift", type=float, default=0.15)
    parser.add_argument("--allow-non-m5max", action="store_true")
    parser.add_argument("--require-speedup", action="store_true")
    parser.add_argument("--native-check-only", action="store_true")
    args = parser.parse_args()
    if args.repeats < 2 and not args.native_check_only:
        parser.error("--repeats must be at least 2 for a qualification run")
    if args.min_geomean_speedup <= 0:
        parser.error("--min-geomean-speedup must be positive")
    if not 0 < args.max_cv < 1:
        parser.error("--max-cv must be between 0 and 1")
    return args


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def tail(path: Path, lines: int = 300) -> str:
    if not path.exists():
        return "<log file was not created>"
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


def run_text(command: list[str]) -> str:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()


def hardware_info(allow_non_m5max: bool) -> dict[str, Any]:
    profiler = run_text(["system_profiler", "SPHardwareDataType"])
    chip_match = re.search(r"^\s*Chip:\s*(.+)$", profiler, re.MULTILINE)
    memory_match = re.search(r"^\s*Memory:\s*(.+)$", profiler, re.MULTILINE)
    chip = chip_match.group(1).strip() if chip_match else "unknown"
    memory = memory_match.group(1).strip() if memory_match else "unknown"
    result = {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "chip": chip,
        "memory": memory,
        "sw_vers": run_text(["sw_vers"]),
        "system_profiler": profiler,
        "is_m5_max": "M5 Max" in chip,
    }
    if platform.machine() != "arm64":
        raise RuntimeError(f"qualification requires arm64, got {platform.machine()}")
    if not result["is_m5_max"] and not allow_non_m5max:
        raise RuntimeError(
            f"qualification requires a physical Apple M5 Max, detected {chip!r}; "
            "use --allow-non-m5max only for harness debugging"
        )
    return result


def model_config(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing model config: {config_path}")
    return json.loads(config_path.read_text())


def mtp_config_depth(config: dict[str, Any]) -> int:
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        return int(text_config.get("mtp_num_hidden_layers", 0) or 0)
    return int(config.get("mtp_num_hidden_layers", 0) or 0)


def iter_weight_keys(model_dir: Path) -> tuple[list[str], list[str]]:
    index_path = model_dir / "model.safetensors.index.json"
    referenced_files: list[str] = []
    keys: list[str] = []
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map", {})
        if not isinstance(weight_map, dict):
            raise RuntimeError(f"invalid weight_map in {index_path}")
        keys.extend(str(key) for key in weight_map)
        referenced_files.extend(sorted({str(value) for value in weight_map.values()}))
        return keys, referenced_files

    files = sorted(model_dir.glob("*.safetensors"))
    for path in files:
        with safe_open(path, framework="numpy") as handle:
            keys.extend(str(key) for key in handle.keys())
        referenced_files.append(path.name)
    return keys, referenced_files


def inspect_model_contract(model_dir: Path) -> dict[str, Any]:
    model_dir = model_dir.expanduser().resolve()
    if not model_dir.is_dir():
        raise NotADirectoryError(model_dir)
    config = model_config(model_dir)
    keys, referenced_files = iter_weight_keys(model_dir)
    mtp_keys = sorted(key for key in keys if "mtp." in key)
    sidecar = model_dir / "mtp.safetensors"
    sidecar_referenced = sidecar.name in referenced_files
    depth = mtp_config_depth(config)
    result = {
        "model_dir": str(model_dir),
        "model_type": config.get("model_type"),
        "mtp_num_hidden_layers": depth,
        "weight_key_count": len(keys),
        "mtp_weight_key_count": len(mtp_keys),
        "mtp_weight_key_examples": mtp_keys[:20],
        "referenced_weight_files": referenced_files,
        "standalone_mtp_sidecar_present": sidecar.is_file(),
        "standalone_mtp_sidecar_referenced": sidecar_referenced,
        "config_sha256": sha256_text(json.dumps(config, sort_keys=True)),
    }
    if depth <= 0:
        raise RuntimeError(
            "config.json does not advertise mtp_num_hidden_layers > 0; "
            "convert the official checkpoint with the mlx-lm #1740 head"
        )
    if not mtp_keys:
        detail = ""
        if sidecar.is_file() and not sidecar_referenced:
            detail = (
                " A standalone mtp.safetensors sidecar exists but is not part of "
                "model.safetensors.index.json; standard mlx-lm loading will ignore it."
            )
        raise RuntimeError(
            "the standard model weight map contains no native mtp.* tensors." + detail
        )
    return result


def native_mtp_check(model_dir: Path, output_dir: Path) -> dict[str, Any]:
    model, tokenizer = load(str(model_dir))
    cache = model.make_mtp_cache()
    if not bool(getattr(model, "supports_mtp", False)):
        raise RuntimeError("loaded model does not advertise supports_mtp")
    if not cache:
        raise RuntimeError("loaded model returned an empty MTP cache")

    target_prompt_tokens = 256
    seed = (
        "Apple Silicon native multi-token prediction should preserve exact greedy "
        "generation while reducing target decode rounds. "
    )
    text = seed
    while len(tokenizer.encode(text)) < target_prompt_tokens + 16:
        text += seed
    prompt = mx.array(tokenizer.encode(text)[:target_prompt_tokens], dtype=mx.uint32)

    accepted = 0
    emitted = 0
    output_ids: list[int] = []
    started = time.perf_counter()
    for token, _logprobs, from_draft in mtp_generate_step(
        prompt,
        model,
        max_tokens=64,
    ):
        output_ids.append(int(token.item()) if hasattr(token, "item") else int(token))
        emitted += 1
        accepted += int(from_draft)
    mx.synchronize()
    elapsed = time.perf_counter() - started
    result = {
        "supports_mtp": True,
        "mtp_cache_entries": len(cache),
        "output_tokens": emitted,
        "accepted_draft_tokens": accepted,
        "accepted_output_fraction": accepted / emitted if emitted else None,
        "output_throughput_tok_s": emitted / elapsed,
        "elapsed_s": elapsed,
        "output_token_sha256": sha256_text(json.dumps(output_ids)),
    }
    write_json(output_dir / "native_mtp.json", result)
    print("NATIVE_MTP_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    return result


def native_mtp_check_isolated(model_dir: Path, output_dir: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model-dir",
        str(model_dir),
        "--output-dir",
        str(output_dir),
        "--native-check-only",
    ]
    subprocess.run(command, check=True)
    return json.loads((output_dir / "native_mtp.json").read_text())


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def wait_port_released(port: int, timeout_s: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not port_is_open(port):
            return True
        time.sleep(0.25)
    return not port_is_open(port)


class Server:
    def __init__(
        self,
        profile: Profile,
        model_dir: Path,
        max_model_len: int,
        max_num_batched_tokens: int,
        block_size: int,
        log_path: Path,
        ready_timeout_s: float,
    ) -> None:
        self.profile = profile
        self.model_dir = model_dir
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.block_size = block_size
        self.log_path = log_path
        self.ready_timeout_s = ready_timeout_s
        self.port = find_free_port()
        self.process: subprocess.Popen[str] | None = None
        self._log_handle: Any = None
        self.startup_s: float | None = None
        self.stop_evidence: dict[str, Any] = {}

    def start(self) -> None:
        executable = shutil.which("vllm")
        if executable is None:
            raise RuntimeError("vllm executable was not found on PATH")
        command = [
            executable,
            "serve",
            str(self.model_dir),
            "--served-model-name",
            SERVED_MODEL_NAME,
            "--enable-prefix-caching",
            "--no-async-scheduling",
            "--max-model-len",
            str(self.max_model_len),
            "--max-num-batched-tokens",
            str(self.max_num_batched_tokens),
            "--block-size",
            str(self.block_size),
            "--port",
            str(self.port),
        ]
        if self.profile.mode == "mtp":
            command.extend(["--speculative-config", SPEC_CONFIG])

        env = os.environ.copy()
        env.update(
            {
                "VLLM_METAL_USE_PAGED_ATTENTION": "1",
                "VLLM_METAL_MEMORY_FRACTION": env.get(
                    "VLLM_METAL_MEMORY_FRACTION", "auto"
                ),
                "VLLM_METAL_SPEC_VERIFY_WINDOW": "0",
                "VLLM_METAL_DECODE_PIPELINE": "1",
                "VLLM_METAL_GDN_LAZY_KERNELS": "1",
                "GLOO_SOCKET_IFNAME": env.get("GLOO_SOCKET_IFNAME", "lo0"),
            }
        )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("w")
        print(
            "SERVER_START="
            + json.dumps(
                {
                    "profile": asdict(self.profile),
                    "command": command,
                    "port": self.port,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        started = time.monotonic()
        self.process = subprocess.Popen(
            command,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True,
        )
        self._wait_ready()
        self.startup_s = time.monotonic() - started

    def _wait_ready(self) -> None:
        assert self.process is not None
        deadline = time.monotonic() + self.ready_timeout_s
        url = f"http://127.0.0.1:{self.port}/v1/models"
        last_error = ""
        while time.monotonic() < deadline:
            code = self.process.poll()
            if code is not None:
                raise RuntimeError(
                    f"server {self.profile.name} exited with code {code}\n"
                    + tail(self.log_path)
                )
            try:
                with urllib.request.urlopen(url, timeout=1.0) as response:
                    if response.status == 200:
                        return
            except (OSError, urllib.error.URLError) as exc:
                last_error = str(exc)
            time.sleep(2)
        raise RuntimeError(
            f"server {self.profile.name} readiness timed out: {last_error}\n"
            + tail(self.log_path)
        )

    def stop(self) -> dict[str, Any]:
        process = self.process
        self.process = None
        evidence = {
            "profile": self.profile.name,
            "port": self.port,
            "pid": process.pid if process is not None else None,
            "term_sent": False,
            "kill_sent": False,
            "exit_code": None,
            "port_released": False,
        }
        if process is not None:
            if process.poll() is None:
                evidence["term_sent"] = True
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    evidence["kill_sent"] = True
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=15)
            evidence["exit_code"] = process.returncode
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
        evidence["port_released"] = wait_port_released(self.port)
        self.stop_evidence = evidence
        return evidence


class ServingBenchmarker:
    def __init__(
        self,
        model_dir: Path,
        repeats: int,
        request_timeout_s: float,
    ) -> None:
        self.model_dir = model_dir
        self.repeats = repeats
        self.request_timeout_s = request_timeout_s
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
        )
        self._prompt_cache: dict[str, list[tuple[str, int]]] = {}

    def prompts_for(self, workload: Workload) -> list[tuple[str, int]]:
        cached = self._prompt_cache.get(workload.name)
        if cached is not None:
            return cached
        seed = (
            "Apple Silicon serving uses a stable shared prefix, paged cache ownership, "
            "and deterministic target verification. "
        )
        seed_ids = self.tokenizer.encode(seed, add_special_tokens=False)
        prompts: list[tuple[str, int]] = []
        for index in range(workload.requests):
            suffix = (
                f"\nRequest slot {index + 1}: explain why native speculative "
                "verification must preserve exact greedy output."
            )
            suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)
            prefix_target = max(1, workload.prompt_tokens - len(suffix_ids))
            repeated = (seed_ids * (prefix_target // len(seed_ids) + 2))[:prefix_target]
            ids = (repeated + suffix_ids)[: workload.prompt_tokens]
            prompt = self.tokenizer.decode(ids, skip_special_tokens=False)
            actual = len(self.tokenizer.encode(prompt, add_special_tokens=False))
            prompts.append((prompt, actual))
        self._prompt_cache[workload.name] = prompts
        return prompts

    async def run(
        self,
        profile: Profile,
        workload: Workload,
        port: int,
    ) -> dict[str, Any]:
        base_url = f"http://127.0.0.1:{port}"
        timeout = aiohttp.ClientTimeout(total=self.request_timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(base_url + "/v1/models") as response:
                response.raise_for_status()
                served_model = (await response.json())["data"][0]["id"]

            prompts = self.prompts_for(workload)

            async def one(request_index: int, tag: str) -> RequestFuncOutput:
                prompt, prompt_len = prompts[request_index]
                request = RequestFuncInput(
                    prompt=prompt,
                    api_url=base_url + "/v1/completions",
                    prompt_len=prompt_len,
                    output_len=workload.output_tokens,
                    model=served_model,
                    ignore_eos=True,
                    extra_body={"temperature": 0, "seed": 0},
                )
                last_error = ""
                for attempt in range(2):
                    try:
                        output = await asyncio.wait_for(
                            async_request_openai_completions(request, session),
                            timeout=self.request_timeout_s,
                        )
                    except TimeoutError as exc:
                        last_error = f"timed out: {exc}"
                    else:
                        if output.success:
                            print(
                                f"{profile.name}/{workload.name} {tag}: "
                                f"out={output.output_tokens} "
                                f"ttft_ms={output.ttft * 1000:.1f} "
                                f"e2e_s={output.latency:.2f}",
                                flush=True,
                            )
                            return output
                        last_error = output.error
                    if attempt == 0:
                        await asyncio.sleep(2)
                raise RuntimeError(
                    f"{profile.name}/{workload.name} {tag} failed: {last_error}"
                )

            await one(0, "warmup")
            metrics_before = await fetch_spec_decode_metrics(base_url, session)

            repeat_metrics: list[dict[str, Any]] = []
            output_hashes_by_repeat: list[list[str]] = []
            all_outputs: list[RequestFuncOutput] = []
            for repeat in range(self.repeats):
                outputs_by_index: list[RequestFuncOutput | None] = [
                    None for _ in range(workload.requests)
                ]
                started = time.perf_counter()
                for offset in range(0, workload.requests, workload.concurrency):
                    indices = list(
                        range(
                            offset,
                            min(offset + workload.concurrency, workload.requests),
                        )
                    )
                    wave = await asyncio.gather(
                        *(
                            one(index, f"r{repeat + 1}:{index + 1}/{workload.requests}")
                            for index in indices
                        )
                    )
                    for index, output in zip(indices, wave):
                        outputs_by_index[index] = output
                elapsed = time.perf_counter() - started
                outputs = [output for output in outputs_by_index if output is not None]
                if len(outputs) != workload.requests:
                    raise RuntimeError("a request result was lost during collection")
                all_outputs.extend(outputs)
                hashes = [sha256_text(output.generated_text) for output in outputs]
                output_hashes_by_repeat.append(hashes)
                total_input = sum(output.prompt_len for output in outputs)
                total_output = sum(output.output_tokens for output in outputs)
                repeat_metrics.append(
                    {
                        "repeat": repeat + 1,
                        "elapsed_s": elapsed,
                        "completed_output_tokens": total_output,
                        "output_throughput_tok_s": total_output / elapsed,
                        "total_token_throughput_tok_s": (total_input + total_output)
                        / elapsed,
                        "mean_ttft_ms": statistics.mean(
                            output.ttft * 1000 for output in outputs
                        ),
                        "median_ttft_ms": statistics.median(
                            output.ttft * 1000 for output in outputs
                        ),
                        "mean_e2e_ms": statistics.mean(
                            output.latency * 1000 for output in outputs
                        ),
                        "mean_tpot_ms": statistics.mean(
                            output.tpot * 1000 for output in outputs
                        ),
                        "output_hashes": hashes,
                    }
                )

            metrics_after = await fetch_spec_decode_metrics(base_url, session)
            draft_tokens = 0
            accepted_tokens = 0
            if metrics_before is not None and metrics_after is not None:
                draft_tokens = (
                    metrics_after.num_draft_tokens - metrics_before.num_draft_tokens
                )
                accepted_tokens = (
                    metrics_after.num_accepted_tokens
                    - metrics_before.num_accepted_tokens
                )

            output_rates = [
                metric["output_throughput_tok_s"] for metric in repeat_metrics
            ]
            first_hashes = output_hashes_by_repeat[0]
            repeat_output_parity = all(
                hashes == first_hashes for hashes in output_hashes_by_repeat[1:]
            )
            result = {
                "profile": profile.name,
                "mode": profile.mode,
                "workload": workload.name,
                "workload_config": asdict(workload),
                "repeats": self.repeats,
                "actual_prompt_tokens": [length for _prompt, length in prompts],
                "output_throughput_tok_s": statistics.median(output_rates),
                "output_throughput_min_tok_s": min(output_rates),
                "output_throughput_max_tok_s": max(output_rates),
                "output_throughput_cv": (
                    statistics.pstdev(output_rates) / statistics.mean(output_rates)
                    if len(output_rates) > 1
                    else 0.0
                ),
                "mean_ttft_ms": statistics.median(
                    metric["mean_ttft_ms"] for metric in repeat_metrics
                ),
                "median_ttft_ms": statistics.median(
                    output.ttft * 1000 for output in all_outputs
                ),
                "mean_e2e_ms": statistics.median(
                    metric["mean_e2e_ms"] for metric in repeat_metrics
                ),
                "mean_tpot_ms": statistics.median(
                    metric["mean_tpot_ms"] for metric in repeat_metrics
                ),
                "mtp_draft_tokens": draft_tokens,
                "mtp_accepted_tokens": accepted_tokens,
                "mtp_acceptance_rate": (
                    accepted_tokens / draft_tokens if draft_tokens else None
                ),
                "canonical_output_hashes": first_hashes,
                "repeat_output_parity": repeat_output_parity,
                "repeat_metrics": repeat_metrics,
            }
            print(
                "BENCHMARK_RESULT=" + json.dumps(result, sort_keys=True),
                flush=True,
            )
            return result


def relative_drift(first: float, second: float) -> float:
    return second / first - 1.0


def aggregate(
    output_dir: Path,
    rows: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    lifecycle: list[dict[str, Any]],
    native: dict[str, Any],
    hardware: dict[str, Any],
    contract: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str]:
    by_workload: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_workload[row["workload"]][row["profile"]] = row

    workload_summaries: dict[str, dict[str, Any]] = {}
    speedups: list[float] = []
    baseline_drifts: list[float] = []
    mtp_drifts: list[float] = []
    parity_failures: list[str] = []
    cv_failures: list[str] = []
    acceptance_failures: list[str] = []

    for workload in (item.name for item in WORKLOADS):
        items = by_workload.get(workload, {})
        required = {profile.name for profile in PROFILES}
        if set(items) != required:
            continue
        baseline_first = items["baseline_first"]
        baseline_second = items["baseline_second"]
        mtp_first = items["mtp_first"]
        mtp_second = items["mtp_second"]
        baseline_rate = statistics.median(
            [
                baseline_first["output_throughput_tok_s"],
                baseline_second["output_throughput_tok_s"],
            ]
        )
        mtp_rate = statistics.median(
            [
                mtp_first["output_throughput_tok_s"],
                mtp_second["output_throughput_tok_s"],
            ]
        )
        speedup = mtp_rate / baseline_rate
        baseline_drift = relative_drift(
            baseline_first["output_throughput_tok_s"],
            baseline_second["output_throughput_tok_s"],
        )
        mtp_drift = relative_drift(
            mtp_first["output_throughput_tok_s"],
            mtp_second["output_throughput_tok_s"],
        )
        baseline_drifts.append(baseline_drift)
        mtp_drifts.append(mtp_drift)
        speedups.append(speedup)

        reference_hashes = baseline_first["canonical_output_hashes"]
        output_parity = all(
            row["canonical_output_hashes"] == reference_hashes
            and row["repeat_output_parity"]
            for row in items.values()
        )
        if not output_parity:
            parity_failures.append(workload)

        for profile_name, row in items.items():
            if row["output_throughput_cv"] > args.max_cv:
                cv_failures.append(
                    f"{profile_name}/{workload}={row['output_throughput_cv']:.3f}"
                )
        acceptance_values = [
            mtp_first["mtp_acceptance_rate"],
            mtp_second["mtp_acceptance_rate"],
        ]
        valid_acceptance = [
            float(value) for value in acceptance_values if value is not None
        ]
        if len(valid_acceptance) != 2 or any(
            value <= 0 for value in valid_acceptance
        ):
            acceptance_failures.append(workload)

        workload_summaries[workload] = {
            "baseline_output_throughput_tok_s": baseline_rate,
            "mtp_output_throughput_tok_s": mtp_rate,
            "speedup": speedup,
            "baseline_launch_drift": baseline_drift,
            "mtp_launch_drift": mtp_drift,
            "baseline_mean_ttft_ms": statistics.median(
                [baseline_first["mean_ttft_ms"], baseline_second["mean_ttft_ms"]]
            ),
            "mtp_mean_ttft_ms": statistics.median(
                [mtp_first["mean_ttft_ms"], mtp_second["mean_ttft_ms"]]
            ),
            "mtp_acceptance_rate": (
                statistics.median(valid_acceptance) if valid_acceptance else 0.0
            ),
            "exact_output_parity": output_parity,
        }

    expected_rows = len(PROFILES) * len(WORKLOADS)
    clean_lifecycle = len(lifecycle) == len(PROFILES) and all(
        item.get("port_released") and not item.get("kill_sent") for item in lifecycle
    )
    complete = len(rows) == expected_rows and len(workload_summaries) == len(WORKLOADS)
    geomean_speedup = statistics.geometric_mean(speedups) if speedups else None
    minimum_speedup = min(speedups) if speedups else None
    drift_failures = [
        f"baseline:{value:+.3f}"
        for value in baseline_drifts
        if abs(value) > args.max_launch_drift
    ] + [
        f"mtp:{value:+.3f}"
        for value in mtp_drifts
        if abs(value) > args.max_launch_drift
    ]

    functional_checks = {
        "physical_m5_max": bool(hardware.get("is_m5_max")) or args.allow_non_m5max,
        "native_mlx_lm_mtp_check": bool(native.get("supports_mtp"))
        and int(native.get("accepted_draft_tokens", 0)) > 0,
        "complete_matrix": complete,
        "zero_errors": not errors,
        "exact_output_parity": not parity_failures,
        "repeat_output_parity": all(row["repeat_output_parity"] for row in rows),
        "positive_mtp_acceptance": not acceptance_failures,
        "clean_server_lifecycle": clean_lifecycle,
    }
    functional_pass = all(functional_checks.values())
    performance_checks = {
        "functional_pass": functional_pass,
        "geomean_speedup_at_least_threshold": geomean_speedup is not None
        and geomean_speedup >= args.min_geomean_speedup,
        "no_workload_regression_below_floor": minimum_speedup is not None
        and minimum_speedup >= args.min_workload_speedup,
        "throughput_cv_within_limit": not cv_failures,
        "launch_drift_within_limit": not drift_failures,
    }
    performance_claim_pass = all(performance_checks.values())

    summary = {
        "hardware": hardware,
        "model_contract": contract,
        "native_mlx_lm_mtp": native,
        "thresholds": {
            "min_geomean_speedup": args.min_geomean_speedup,
            "min_workload_speedup": args.min_workload_speedup,
            "max_cv": args.max_cv,
            "max_launch_drift": args.max_launch_drift,
        },
        "functional_checks": functional_checks,
        "performance_checks": performance_checks,
        "functional_pass": functional_pass,
        "performance_claim_pass": performance_claim_pass,
        "geomean_speedup": geomean_speedup,
        "minimum_workload_speedup": minimum_speedup,
        "workloads": workload_summaries,
        "parity_failures": parity_failures,
        "acceptance_failures": acceptance_failures,
        "cv_failures": cv_failures,
        "drift_failures": drift_failures,
        "errors": errors,
        "lifecycle": lifecycle,
        "rows": rows,
    }
    write_json(output_dir / "qualification_summary.json", summary)

    lines = [
        "# Qwen3.6-27B native MTP qualification on M5 Max",
        "",
        f"- Functional qualification: **{'PASS' if functional_pass else 'FAIL'}**",
        "- Positive performance claim gate: "
        f"**{'PASS' if performance_claim_pass else 'FAIL'}**",
        f"- Detected chip: `{hardware.get('chip')}`",
        "- Native MLX-LM MTP check: "
        f"`{native.get('output_throughput_tok_s', 0):.2f}` output tok/s, "
        f"accepted draft outputs `{native.get('accepted_draft_tokens', 0)}`",
        "",
        "## Matched serving results",
        "",
        "| Workload | Baseline tok/s | MTP tok/s | MTP / baseline | "
        "Acceptance | Baseline TTFT | MTP TTFT | Exact output |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for workload, item in workload_summaries.items():
        lines.append(
            f"| {workload} | {item['baseline_output_throughput_tok_s']:.2f} | "
            f"{item['mtp_output_throughput_tok_s']:.2f} | {item['speedup']:.3f}x | "
            f"{100 * item['mtp_acceptance_rate']:.1f}% | "
            f"{item['baseline_mean_ttft_ms']:.1f} ms | "
            f"{item['mtp_mean_ttft_ms']:.1f} ms | "
            f"{'yes' if item['exact_output_parity'] else 'NO'} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate gate",
            "",
            f"- Geometric-mean speedup: `{geomean_speedup:.3f}x`"
            if geomean_speedup is not None
            else "- Geometric-mean speedup: unavailable",
            f"- Minimum workload speedup: `{minimum_speedup:.3f}x`"
            if minimum_speedup is not None
            else "- Minimum workload speedup: unavailable",
            f"- Required geometric mean: `>= {args.min_geomean_speedup:.3f}x`",
            f"- Required workload floor: `>= {args.min_workload_speedup:.3f}x`",
            f"- Throughput CV limit: `<= {100 * args.max_cv:.1f}%`",
            f"- Launch drift limit: `<= {100 * args.max_launch_drift:.1f}%`",
            "",
            "A failed performance gate is an honest negative result, not a failed "
            "functional implementation. Only a passing gate supports a positive "
            "M5 Max acceleration claim.",
        ]
    )
    if errors or parity_failures or cv_failures or drift_failures:
        lines.extend(["", "## Failures", ""])
        for error in errors:
            lines.append(
                f"- `{error.get('profile')}/{error.get('workload')}`: "
                f"{str(error.get('error')).replace(chr(10), ' ')[:600]}"
            )
        for value in parity_failures:
            lines.append(f"- Output parity failed: `{value}`")
        for value in cv_failures:
            lines.append(f"- CV exceeded: `{value}`")
        for value in drift_failures:
            lines.append(f"- Launch drift exceeded: `{value}`")
    report = "\n".join(lines) + "\n"
    (output_dir / "qualification_summary.md").write_text(report)
    return summary, report


async def async_main(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    (output_dir / "results").mkdir(exist_ok=True)

    hardware = hardware_info(args.allow_non_m5max)
    contract = inspect_model_contract(model_dir)
    write_json(output_dir / "hardware.json", hardware)
    write_json(output_dir / "model_contract.json", contract)
    write_json(output_dir / "profiles.json", [asdict(item) for item in PROFILES])
    write_json(output_dir / "workloads.json", [asdict(item) for item in WORKLOADS])

    native = native_mtp_check_isolated(model_dir, output_dir)
    max_model_len = max(
        workload.prompt_tokens + workload.output_tokens + 256 for workload in WORKLOADS
    )
    benchmarker = ServingBenchmarker(model_dir, args.repeats, args.request_timeout_s)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    lifecycle: list[dict[str, Any]] = []

    for profile in PROFILES:
        log_path = output_dir / "logs" / f"{profile.name}.log"
        server = Server(
            profile,
            model_dir,
            max_model_len,
            args.max_num_batched_tokens,
            args.block_size,
            log_path,
            args.server_ready_timeout_s,
        )
        try:
            server.start()
            for workload in WORKLOADS:
                try:
                    result = await benchmarker.run(profile, workload, server.port)
                except Exception as exc:
                    errors.append(
                        {
                            "profile": profile.name,
                            "workload": workload.name,
                            "error": str(exc),
                        }
                    )
                    print(
                        f"::warning::{profile.name}/{workload.name} failed: {exc}",
                        flush=True,
                    )
                    continue
                result["server_startup_s"] = server.startup_s
                result["server_port"] = server.port
                rows.append(result)
                write_json(
                    output_dir / "results" / f"{profile.name}__{workload.name}.json",
                    result,
                )
        except Exception as exc:
            errors.append(
                {
                    "profile": profile.name,
                    "workload": "server",
                    "error": str(exc),
                }
            )
            print(f"::warning::{profile.name} server failed: {exc}", flush=True)
        finally:
            evidence = server.stop()
            lifecycle.append(evidence)
            write_json(
                output_dir / "results" / f"{profile.name}__lifecycle.json",
                evidence,
            )
            print(f"===== {profile.name} server log tail =====", flush=True)
            print(tail(log_path), flush=True)

    summary, report = aggregate(
        output_dir,
        rows,
        errors,
        lifecycle,
        native,
        hardware,
        contract,
        args,
    )
    print(report, flush=True)
    print(
        "FINAL_QUALIFICATION="
        + json.dumps(
            {
                "functional_pass": summary["functional_pass"],
                "performance_claim_pass": summary["performance_claim_pass"],
                "geomean_speedup": summary["geomean_speedup"],
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not summary["functional_pass"]:
        return 2
    if args.require_speedup and not summary["performance_claim_pass"]:
        return 3
    return 0


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.model_dir = args.model_dir.expanduser().resolve()
    if args.native_check_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        inspect_model_contract(args.model_dir)
        native_mtp_check(args.model_dir, args.output_dir)
        return 0
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
