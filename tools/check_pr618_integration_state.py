#!/usr/bin/env python3
"""Verify the staged PR #618 current-main integration state.

The checker enforces two invariants:

1. Files without upstream overlap remain byte-identical to the preserved #618
   blobs.
2. An overlap file may remain at the pinned current-main blob or advance to a
   new combined blob, but it may never be replaced wholesale by the old #618
   blob.

Use --require-resolved once all semantic merges are expected to be complete.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

CURRENT_MAIN = "211a1e4bc976cd6c0c961cad8da59d649ea9bd65"

CLEAN_TRANSPLANTS = {
    ".github/workflows/qwen-mtp-serving-bench.yml": "9a2a29f74a352b2a4d4a4ff0412ff478415c79e5",
    "benchmarks/qwen_mtp_speed_matrix.py": "9f21c7f3dec72548a52017b7336985b4b4c9c71a",
    "benchmarks/results/qwen_mtp_copyless_promotion_20260817.md": "c139f5a3a53bbbab893cb433e29d739d0d646564",
    "tests/attention/test_align_gdn_state_manager.py": "12bddf505a7ee8730882e9cfdcce438d74ede6b5",
    "tests/test_qwen_mtp_paged_cache.py": "4c23ae1c4f9c75696ae3ea577e247994cf29ab48",
    "tests/test_qwen_mtp_worker_budget.py": "cace88eaf011ff222363bb0127d32d863b264239",
    "tests/test_qwen_mtp_wrapper_metadata.py": "0a5ce010ccd1d433220b3afe7bd829b1e94765b8",
    "tests/test_qwen_native_mtp_proposer.py": "2781db17abf098a8a41863c486001f50e2656b93",
    "vllm_metal/attention/runtime/hybrid.py": "c04a9cf23181216586b317aa1ea672aba262fc42",
    "vllm_metal/attention/state/align.py": "437f20945fd6eb11152b6e43833306e3d8005912",
    "vllm_metal/v1/model_adapter.py": "85ae0637d34eeea9d59b24988bffcffeaaf542c6",
    "vllm_metal/v1/proposer.py": "7b3fc43a3020d273abfc27b46ce53b529562198f",
    "vllm_metal/v1/qwen_mtp_paged.py": "42cb88ca95612d6047fcf86e0053651de7f62612",
    "vllm_metal/v1/spec_decode.py": "a988cb84cf3b4962abb90a6b909e4a85852ff7a5",
}


@dataclass(frozen=True)
class Overlap:
    upstream: str
    pr: str


OVERLAPS = {
    "tests/test_gdn_lazy_wrapper.py": Overlap(
        "b9edddc333621e3d1ea5d25ac01d23fda0f48914",
        "65b14e3aa9b05f2e78cd7c7e6131afb8dbc93bbd",
    ),
    "tests/test_v1_model_runner_generate.py": Overlap(
        "c3658280884a14efefca7ffee8478918412c7c1e",
        "06103a3e5c5031643cf6a80de6bce68e5c4019e2",
    ),
    "vllm_metal/attention/context.py": Overlap(
        "f7616981ccd6b861299ec8500e7231978fa2f85c",
        "1ba4c7ae0ad2d602ed129d23d51a5f472dfa9da2",
    ),
    "vllm_metal/attention/impls/gdn_lazy.py": Overlap(
        "a65cfcda75308d27e4e16d0f9172ad3ff24c2da6",
        "566fcdfed0acaa4327204c20238fc2a2a640cca7",
    ),
    "vllm_metal/attention/impls/linear.py": Overlap(
        "50f8bb9cd6c60c591ab45556075e48c20a1c5c19",
        "97e110b955d64731b64582c14abb4ecfc94cca96",
    ),
    "vllm_metal/platform.py": Overlap(
        "cfaebbb7ed5c811c088a4008957a7643a6c6d079",
        "366a086541b800bca2e2d259058e741af6b17a8b",
    ),
    "vllm_metal/v1/cache_policy.py": Overlap(
        "1692a59dce58e94dad3443d1d1bbf00de27f767f",
        "72aed78f5f0c859f67e2b333864a22ced8169974",
    ),
    "vllm_metal/v1/model_runner.py": Overlap(
        "2affefda8c94c0311ea0cf90706be650b9822d67",
        "2c18a5b26a28e9c777590261876d2595e0184469",
    ),
}


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def blob_for(path: str) -> str | None:
    result = run("git", "rev-parse", f"HEAD:{path}", check=False)
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def has_conflict_markers() -> list[str]:
    result = run(
        "git",
        "grep",
        "-l",
        "^<<<<<<< ",
        "--",
        ":(exclude)docs/pr_618_merge_report_20260822.md",
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr.strip())
    return [line for line in result.stdout.splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-resolved",
        action="store_true",
        help="Fail while any overlap path still equals its current-main blob.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the state summary as JSON.",
    )
    args = parser.parse_args()

    failures: list[str] = []
    pending: list[str] = []
    resolved: list[str] = []
    states: dict[str, dict[str, str | None]] = {}

    ancestor = run(
        "git", "merge-base", "--is-ancestor", CURRENT_MAIN, "HEAD", check=False
    )
    if ancestor.returncode != 0:
        failures.append(f"HEAD does not descend from pinned current main {CURRENT_MAIN}")

    for path, expected in CLEAN_TRANSPLANTS.items():
        actual = blob_for(path)
        states[path] = {
            "kind": "clean_transplant",
            "actual": actual,
            "expected": expected,
        }
        if actual != expected:
            failures.append(
                f"clean transplant {path} is {actual or 'missing'}, expected {expected}"
            )

    for path, overlap in OVERLAPS.items():
        actual = blob_for(path)
        if actual is None:
            state = "missing"
            failures.append(f"overlap path {path} is missing")
        elif actual == overlap.upstream:
            state = "pending"
            pending.append(path)
        elif actual == overlap.pr:
            state = "blind_pr_replacement"
            failures.append(
                f"overlap path {path} equals the old PR blob; upstream work was lost"
            )
        else:
            state = "resolved_candidate"
            resolved.append(path)
        states[path] = {
            "kind": "semantic_overlap",
            "state": state,
            "actual": actual,
            "upstream": overlap.upstream,
            "pr": overlap.pr,
        }

    marker_files = has_conflict_markers()
    if marker_files:
        failures.append("tracked conflict markers remain in: " + ", ".join(marker_files))

    if args.require_resolved and pending:
        failures.append("pending semantic overlaps: " + ", ".join(pending))

    summary = {
        "current_main": CURRENT_MAIN,
        "clean_transplants": len(CLEAN_TRANSPLANTS),
        "pending": pending,
        "resolved_candidates": resolved,
        "failures": failures,
        "states": states,
    }

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"clean transplants: {len(CLEAN_TRANSPLANTS)}")
        print(f"pending semantic overlaps: {len(pending)}")
        for path in pending:
            print(f"  PENDING  {path}")
        for path in resolved:
            print(f"  RESOLVED? {path}")
        for failure in failures:
            print(f"  ERROR    {failure}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
