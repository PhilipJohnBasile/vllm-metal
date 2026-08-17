#!/usr/bin/env python3
"""Lifecycle-instrumented smoke ladder for Qwen native MTP candidates.

The ladder launches the same deterministic workload under current MTP,
copyless-only, deferred-only, and copyless+deferred. Every launch receives a
fresh port, retained server log, explicit process classification, forced
process-group cleanup, and a post-stop port-release check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

SPEC_CONFIG = '{"method":"mtp","num_speculative_tokens":1}'


@dataclass(frozen=True)
class Arm:
    name: str
    copyless: bool
    deferred: bool


ARMS: tuple[Arm, ...] = (
    Arm("mtp_current", copyless=False, deferred=False),
    Arm("mtp_copyless", copyless=True, deferred=False),
    Arm("mtp_deferred", copyless=False, deferred=True),
    Arm("mtp_combo", copyless=True, deferred=True),
)


class SmokeFailure(RuntimeError):
    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--copyless-patch",
        type=Path,
        default=Path("benchmarks/patches/qwen_mtp_copyless_gdn.patch"),
    )
    parser.add_argument("--launch-repeats", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=1152)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--server-ready-timeout-s", type=float, default=720.0)
    parser.add_argument("--request-timeout-s", type=float, default=240.0)
    args = parser.parse_args()
    if args.launch_repeats < 1:
        parser.error("--launch-repeats must be at least 1")
    if args.output_tokens < 1 or args.warmup_tokens < 1:
        parser.error("token counts must be positive")
    return args


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def tail(path: Path, lines: int = 300) -> str:
    if not path.exists():
        return "<log file was not created>"
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


def run_git(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        check=check,
        capture_output=True,
        text=True,
    )


def patch_state(patch: Path) -> str:
    forward = run_git(
        ["apply", "--check", "--whitespace=nowarn", str(patch)],
        check=False,
    ).returncode == 0
    reverse = run_git(
        ["apply", "--reverse", "--check", "--whitespace=nowarn", str(patch)],
        check=False,
    ).returncode == 0
    if forward and not reverse:
        return "absent"
    if reverse and not forward:
        return "applied"
    raise SmokeFailure(
        "copyless_patch_state_unknown",
        f"cannot prove whether {patch} is applied (forward={forward}, reverse={reverse})",
    )


def set_copyless_patch(enabled: bool, patch: Path) -> None:
    state = patch_state(patch)
    if enabled and state == "absent":
        run_git(["apply", "--whitespace=nowarn", str(patch)])
    elif not enabled and state == "applied":
        run_git(["apply", "--reverse", "--whitespace=nowarn", str(patch)])
    expected = "applied" if enabled else "absent"
    actual = patch_state(patch)
    if actual != expected:
        raise SmokeFailure(
            "copyless_patch_transition_failed",
            f"wanted copyless patch {expected}, observed {actual}",
        )
    run_git(["diff", "--check"])


def allocate_fresh_port(used_ports: set[int]) -> int:
    for _ in range(100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
        if port not in used_ports:
            used_ports.add(port)
            return port
    raise SmokeFailure("port_allocation_failed", "could not allocate a fresh port")


def port_is_bindable(port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", port))
    except OSError:
        return False
    return True


def wait_for_port_release(port: int, timeout_s: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if port_is_bindable(port):
            return True
        time.sleep(0.25)
    return port_is_bindable(port)


def request_json(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout_s: float,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        method="GET" if payload is None else "POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = response.read().decode(errors="replace")
            if response.status != 200:
                raise SmokeFailure(
                    "http_status",
                    f"{url} returned HTTP {response.status}: {body[-2000:]}",
                )
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise SmokeFailure(
            "http_error",
            f"{url} returned HTTP {exc.code}: {body[-4000:]}",
        ) from exc
    except (OSError, urllib.error.URLError) as exc:
        raise SmokeFailure("connection_failure", f"{url}: {exc}") from exc


def scrape_spec_metrics(base_url: str, timeout_s: float) -> dict[str, float] | None:
    request = urllib.request.Request(base_url + "/metrics", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            text = response.read().decode(errors="replace")
    except (OSError, urllib.error.URLError):
        return None

    def total_for(suffix: str) -> float | None:
        values: list[float] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            metric = parts[0].split("{", 1)[0]
            if not metric.endswith(suffix):
                continue
            try:
                values.append(float(parts[1]))
            except ValueError:
                continue
        return sum(values) if values else None

    draft = total_for("spec_decode_num_draft_tokens_total")
    accepted = total_for("spec_decode_num_accepted_tokens_total")
    if draft is None or accepted is None:
        return None
    return {"draft_tokens": draft, "accepted_tokens": accepted}


def metric_delta(
    before: dict[str, float] | None,
    after: dict[str, float] | None,
) -> dict[str, float | None]:
    if before is None or after is None:
        return {
            "draft_tokens": None,
            "accepted_tokens": None,
            "acceptance_rate": None,
        }
    draft = after["draft_tokens"] - before["draft_tokens"]
    accepted = after["accepted_tokens"] - before["accepted_tokens"]
    return {
        "draft_tokens": draft,
        "accepted_tokens": accepted,
        "acceptance_rate": accepted / draft if draft > 0 else None,
    }


class Server:
    def __init__(
        self,
        *,
        arm: Arm,
        model_dir: Path,
        port: int,
        log_path: Path,
        ready_timeout_s: float,
    ) -> None:
        self.arm = arm
        self.model_dir = model_dir
        self.port = port
        self.log_path = log_path
        self.ready_timeout_s = ready_timeout_s
        self.process: subprocess.Popen[str] | None = None
        self.log_handle: Any = None
        self.command: list[str] = []
        self.ready_s: float | None = None

    def start(self) -> None:
        executable = shutil.which("vllm")
        if executable is None:
            raise SmokeFailure("missing_vllm", "vllm executable was not found")
        self.command = [
            executable,
            "serve",
            str(self.model_dir),
            "--served-model-name",
            "qwen35-mtp-smoke",
            "--enable-prefix-caching",
            "--no-async-scheduling",
            "--max-model-len",
            "1280",
            "--max-num-batched-tokens",
            "4096",
            "--max-num-seqs",
            "4",
            "--block-size",
            "16",
            "--speculative-config",
            SPEC_CONFIG,
            "--port",
            str(self.port),
        ]
        env = os.environ.copy()
        env.update(
            {
                "VLLM_METAL_SPEC_VERIFY_WINDOW": "0",
                "VLLM_METAL_DECODE_PIPELINE": "1",
                "VLLM_METAL_GDN_LAZY_KERNELS": "1",
                "VLLM_METAL_GDN_DEFER_DECODE_STATE": str(
                    int(self.arm.deferred)
                ),
            }
        )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = self.log_path.open("w")
        self.process = subprocess.Popen(
            self.command,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True,
        )
        started = time.monotonic()
        deadline = started + self.ready_timeout_s
        url = f"http://127.0.0.1:{self.port}/v1/models"
        last_error = ""
        while time.monotonic() < deadline:
            code = self.process.poll()
            if code is not None:
                raise SmokeFailure(
                    "early_exit_before_ready",
                    f"server exited with code {code} before readiness\n{tail(self.log_path)}",
                )
            try:
                request_json(url, timeout_s=1.0)
            except SmokeFailure as exc:
                last_error = str(exc)
            else:
                self.ready_s = time.monotonic() - started
                return
            time.sleep(1.0)
        raise SmokeFailure(
            "readiness_timeout",
            f"server readiness timed out: {last_error}\n{tail(self.log_path)}",
        )

    def assert_alive(self, phase: str) -> None:
        assert self.process is not None
        code = self.process.poll()
        if code is not None:
            raise SmokeFailure(
                f"early_exit_{phase}",
                f"server exited with code {code} during {phase}\n{tail(self.log_path)}",
            )

    def stop(self) -> dict[str, Any]:
        process = self.process
        self.process = None
        sent_sigterm = False
        sent_sigkill = False
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                sent_sigterm = True
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                    sent_sigkill = True
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)
        returncode = None if process is None else process.poll()
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None
        return {
            "pid": None if process is None else process.pid,
            "returncode": returncode,
            "sent_sigterm": sent_sigterm,
            "sent_sigkill": sent_sigkill,
            "terminated": process is None or returncode is not None,
            "port_released": wait_for_port_release(self.port),
        }


def build_prompt(model_dir: Path, target_tokens: int) -> tuple[str, int]:
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
    )
    seed = (
        "Apple Silicon speculative decoding uses a shared cached prefix and a "
        "trained multi-token prediction head. "
    )
    text = seed
    while len(tokenizer.encode(text, add_special_tokens=False)) < target_tokens + 16:
        text += seed
    token_ids = tokenizer.encode(text, add_special_tokens=False)[:target_tokens]
    prompt = tokenizer.decode(token_ids, skip_special_tokens=False)
    actual = len(tokenizer.encode(prompt, add_special_tokens=False))
    return prompt, actual


def run_completion(
    *,
    base_url: str,
    prompt: str,
    max_tokens: int,
    timeout_s: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    response = request_json(
        base_url + "/v1/completions",
        payload={
            "model": "qwen35-mtp-smoke",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "seed": 0,
        },
        timeout_s=timeout_s,
    )
    elapsed = time.perf_counter() - started
    text = str(response["choices"][0]["text"])
    usage = response.get("usage") or {}
    completion_tokens = int(usage.get("completion_tokens", max_tokens))
    return {
        "elapsed_s": elapsed,
        "completion_tokens": completion_tokens,
        "output_throughput_tok_s": completion_tokens / elapsed,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text_preview": text[:160],
        "finish_reason": response["choices"][0].get("finish_reason"),
    }


def run_launch(
    *,
    arm: Arm,
    repeat: int,
    model_dir: Path,
    output_dir: Path,
    prompt: str,
    prompt_tokens: int,
    output_tokens: int,
    warmup_tokens: int,
    request_timeout_s: float,
    server_ready_timeout_s: float,
    used_ports: set[int],
) -> dict[str, Any]:
    port = allocate_fresh_port(used_ports)
    log_path = output_dir / "logs" / f"r{repeat}-{arm.name}.log"
    server = Server(
        arm=arm,
        model_dir=model_dir,
        port=port,
        log_path=log_path,
        ready_timeout_s=server_ready_timeout_s,
    )
    record: dict[str, Any] = {
        "arm": asdict(arm),
        "repeat": repeat,
        "port": port,
        "log_path": str(log_path),
        "prompt_tokens": prompt_tokens,
        "status": "failed",
    }
    try:
        server.start()
        assert server.process is not None
        record.update(
            {
                "pid": server.process.pid,
                "command": server.command,
                "ready_s": server.ready_s,
            }
        )
        base_url = f"http://127.0.0.1:{port}"
        record["warmup"] = run_completion(
            base_url=base_url,
            prompt=prompt,
            max_tokens=warmup_tokens,
            timeout_s=request_timeout_s,
        )
        time.sleep(1.0)
        server.assert_alive("after_warmup")
        metrics_before = scrape_spec_metrics(base_url, request_timeout_s)
        record["measured"] = run_completion(
            base_url=base_url,
            prompt=prompt,
            max_tokens=output_tokens,
            timeout_s=request_timeout_s,
        )
        metrics_after = scrape_spec_metrics(base_url, request_timeout_s)
        record["spec_decode"] = metric_delta(metrics_before, metrics_after)
        time.sleep(1.0)
        server.assert_alive("after_measured_request")
        record["status"] = "success"
    except SmokeFailure as exc:
        record["error_kind"] = exc.kind
        record["error"] = str(exc)
        record["server_log_tail"] = tail(log_path)
    except Exception as exc:  # preserve unexpected diagnostics in the artifact
        record["error_kind"] = type(exc).__name__
        record["error"] = str(exc)
        record["server_log_tail"] = tail(log_path)
    finally:
        record["lifecycle"] = server.stop()
        if not record["lifecycle"]["port_released"]:
            record["status"] = "failed"
            record.setdefault("error_kind", "port_not_released")
            record.setdefault("error", f"port {port} remained occupied after stop")
    write_json(output_dir / "launches" / f"r{repeat}-{arm.name}.json", record)
    print("SMOKE_LAUNCH=" + json.dumps(record, sort_keys=True), flush=True)
    return record


def coefficient_of_variation(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    mean = statistics.mean(values)
    return statistics.pstdev(values) / mean if mean else None


def summarize(
    output_dir: Path,
    records: list[dict[str, Any]],
    launch_repeats: int,
) -> dict[str, Any]:
    by_arm: dict[str, list[dict[str, Any]]] = {arm.name: [] for arm in ARMS}
    for record in records:
        by_arm[record["arm"]["name"]].append(record)

    arms: dict[str, dict[str, Any]] = {}
    successful_hashes: set[str] = set()
    for arm in ARMS:
        rows = by_arm[arm.name]
        successes = [row for row in rows if row["status"] == "success"]
        rates = [
            float(row["measured"]["output_throughput_tok_s"])
            for row in successes
        ]
        acceptances = [
            float(row["spec_decode"]["acceptance_rate"])
            for row in successes
            if row.get("spec_decode", {}).get("acceptance_rate") is not None
        ]
        for row in successes:
            successful_hashes.add(row["measured"]["text_sha256"])
        arms[arm.name] = {
            "launches": len(rows),
            "successful_launches": len(successes),
            "median_output_throughput_tok_s": (
                statistics.median(rates) if rates else None
            ),
            "throughput_cv": coefficient_of_variation(rates),
            "median_acceptance_rate": (
                statistics.median(acceptances) if acceptances else None
            ),
            "ports_released": all(
                bool(row.get("lifecycle", {}).get("port_released")) for row in rows
            ),
        }

    expected_launches_ok = all(
        arms[arm.name]["successful_launches"] == launch_repeats for arm in ARMS
    )
    lifecycle_ok = all(arms[arm.name]["ports_released"] for arm in ARMS)
    correctness_ok = len(successful_hashes) == 1 and bool(successful_hashes)
    functional_passed = expected_launches_ok and lifecycle_ok and correctness_ok

    copyless_rate = arms["mtp_copyless"]["median_output_throughput_tok_s"]
    combo_rate = arms["mtp_combo"]["median_output_throughput_tok_s"]
    speedup = (
        combo_rate / copyless_rate
        if copyless_rate is not None and combo_rate is not None and copyless_rate > 0
        else None
    )
    copyless_accept = arms["mtp_copyless"]["median_acceptance_rate"]
    combo_accept = arms["mtp_combo"]["median_acceptance_rate"]
    acceptance_delta = (
        combo_accept - copyless_accept
        if copyless_accept is not None and combo_accept is not None
        else None
    )
    stable = all(
        arms[name]["throughput_cv"] is not None
        and arms[name]["throughput_cv"] <= 0.10
        for name in ("mtp_copyless", "mtp_combo")
    )
    acceptance_ok = acceptance_delta is not None and abs(acceptance_delta) <= 0.01
    promotion_ready = launch_repeats >= 3
    promotion_passed = bool(
        promotion_ready
        and functional_passed
        and speedup is not None
        and speedup >= 1.05
        and stable
        and acceptance_ok
    )
    if not functional_passed:
        decision = "fix_functional_or_lifecycle_failure"
    elif not promotion_ready:
        decision = "run_three_launch_promotion_gate"
    elif promotion_passed:
        decision = "promote_copyless_plus_deferred"
    else:
        decision = "keep_copyless_only"

    summary = {
        "launch_repeats": launch_repeats,
        "functional_passed": functional_passed,
        "correctness_ok": correctness_ok,
        "lifecycle_ok": lifecycle_ok,
        "output_hashes": sorted(successful_hashes),
        "combo_speedup_vs_copyless": speedup,
        "acceptance_delta_combo_minus_copyless": acceptance_delta,
        "stable_throughput": stable,
        "promotion_ready": promotion_ready,
        "promotion_passed": promotion_passed,
        "decision": decision,
        "arms": arms,
        "records": records,
    }
    write_json(output_dir / "summary.json", summary)

    lines = [
        "# Qwen MTP copyless + deferred smoke ladder",
        "",
        "| Arm | Launches | Successful | Median output tok/s | CV | Acceptance | Ports released |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for arm in ARMS:
        row = arms[arm.name]
        rate = row["median_output_throughput_tok_s"]
        cv = row["throughput_cv"]
        acceptance = row["median_acceptance_rate"]
        lines.append(
            f"| {arm.name} | {row['launches']} | {row['successful_launches']} | "
            f"{rate:.3f} | {cv:.3f} | "
            f"{acceptance:.1%} | {row['ports_released']} |"
            if rate is not None and cv is not None and acceptance is not None
            else f"| {arm.name} | {row['launches']} | {row['successful_launches']} | "
            f"{rate if rate is not None else 'n/a'} | "
            f"{cv if cv is not None else 'n/a'} | "
            f"{acceptance if acceptance is not None else 'n/a'} | "
            f"{row['ports_released']} |"
        )
    lines.extend(
        [
            "",
            f"- Functional gate: **{functional_passed}**",
            f"- Exact deterministic output: **{correctness_ok}**",
            f"- Combined / copyless throughput: **{speedup:.3f}x**"
            if speedup is not None
            else "- Combined / copyless throughput: **n/a**",
            f"- Acceptance delta: **{acceptance_delta:+.2%}**"
            if acceptance_delta is not None
            else "- Acceptance delta: **n/a**",
            f"- Promotion gate (>=1.05x, <=10% CV, <=1 point acceptance delta, 3/3 launches): **{promotion_passed}**",
            f"- Decision: **{decision}**",
            "",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines))
    print("SMOKE_SUMMARY=" + json.dumps(summary, sort_keys=True), flush=True)
    return summary


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt, prompt_tokens = build_prompt(args.model_dir, args.prompt_tokens)
    used_ports: set[int] = set()
    records: list[dict[str, Any]] = []
    cleanup_error: str | None = None
    try:
        set_copyless_patch(False, args.copyless_patch)
        for repeat in range(1, args.launch_repeats + 1):
            for arm in ARMS:
                try:
                    set_copyless_patch(arm.copyless, args.copyless_patch)
                except SmokeFailure as exc:
                    record = {
                        "arm": asdict(arm),
                        "repeat": repeat,
                        "status": "failed",
                        "error_kind": exc.kind,
                        "error": str(exc),
                        "lifecycle": {"port_released": True},
                    }
                    records.append(record)
                    continue
                records.append(
                    run_launch(
                        arm=arm,
                        repeat=repeat,
                        model_dir=args.model_dir,
                        output_dir=args.output_dir,
                        prompt=prompt,
                        prompt_tokens=prompt_tokens,
                        output_tokens=args.output_tokens,
                        warmup_tokens=args.warmup_tokens,
                        request_timeout_s=args.request_timeout_s,
                        server_ready_timeout_s=args.server_ready_timeout_s,
                        used_ports=used_ports,
                    )
                )
    finally:
        try:
            set_copyless_patch(False, args.copyless_patch)
        except Exception as exc:
            cleanup_error = str(exc)

    if cleanup_error is not None:
        records.append(
            {
                "arm": {"name": "harness_cleanup"},
                "repeat": 0,
                "status": "failed",
                "error_kind": "copyless_patch_cleanup_failed",
                "error": cleanup_error,
                "lifecycle": {"port_released": True},
            }
        )
    summary = summarize(args.output_dir, records, args.launch_repeats)
    return 0 if summary["functional_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
