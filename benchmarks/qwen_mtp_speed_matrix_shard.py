#!/usr/bin/env python3
"""Run a selected subset of the curated Qwen MTP speed matrix.

The underlying benchmark remains the source of truth. This wrapper only
filters its profile list so GitHub-hosted runs can be sharded without changing
measurement or aggregation behavior.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def load_benchmark_module():
    source = Path(__file__).with_name("qwen_mtp_speed_matrix.py")
    spec = importlib.util.spec_from_file_location("qwen_mtp_speed_matrix", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load benchmark module from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = load_benchmark_module()
    raw = os.environ.get("QWEN_MTP_PROFILE_NAMES", "").strip()
    if not raw:
        raise SystemExit("QWEN_MTP_PROFILE_NAMES must contain a comma-separated profile list")

    requested = [name.strip() for name in raw.split(",") if name.strip()]
    known = {profile.name for profile in module.PROFILES}
    unknown = sorted(set(requested) - known)
    if unknown:
        raise SystemExit(f"unknown Qwen MTP profiles: {', '.join(unknown)}")

    requested_set = set(requested)
    selected = tuple(
        profile for profile in module.PROFILES if profile.name in requested_set
    )
    selected_names = [profile.name for profile in selected]
    missing_controls = {"baseline_ref", "baseline_repeat"} - set(selected_names)
    if missing_controls:
        raise SystemExit(
            "each shard must include matched cold/hot baselines; missing: "
            + ", ".join(sorted(missing_controls))
        )

    module.PROFILES = selected
    print("QWEN_MTP_SHARD_PROFILES=" + ",".join(selected_names), flush=True)
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
