#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Aggregate sharded Qwen MTP breakthrough-matrix artifacts."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}x"


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def main() -> int:
    args = parse_args()
    summaries: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(args.root.rglob("breakthrough_summary.json")):
        try:
            summaries.append((path.parent.name, json.loads(path.read_text())))
        except Exception:
            continue

    serving_rows: list[dict[str, Any]] = []
    pressure_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    for shard, summary in summaries:
        shard_record = {
            "shard": shard,
            "mode": summary.get("mode"),
            "best_complete_arm": summary.get("best_complete_arm"),
            "errors": len(summary.get("errors", [])),
        }
        shards.append(shard_record)
        for error in summary.get("errors", []):
            errors.append({**error, "shard": shard})
        for row in summary.get("rows", []):
            tagged = {**row, "shard": shard}
            if row.get("kind") == "pressure":
                pressure_rows.append(tagged)
            else:
                serving_rows.append(tagged)

    serving_workloads = sorted({row["workload"] for row in serving_rows})
    serving_by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in serving_rows:
        if row.get("arm_config", {}).get("mode") == "mtp":
            serving_by_arm[row["arm"]].append(row)

    serving_ranking: list[dict[str, Any]] = []
    serving_incomplete: list[dict[str, Any]] = []
    expected = set(serving_workloads)
    for arm, rows in serving_by_arm.items():
        completed = {row["workload"] for row in rows}
        speedups = [
            row["speedup_vs_matched_baseline"]
            for row in rows
            if row.get("speedup_vs_matched_baseline") is not None
        ]
        current_gains = [
            row["speedup_vs_mtp_current"]
            for row in rows
            if row.get("speedup_vs_mtp_current") is not None
        ]
        item = {
            "arm": arm,
            "complete": completed == expected,
            "workloads_completed": len(completed),
            "workloads_expected": len(expected),
            "geomean_speedup_vs_baseline": (
                statistics.geometric_mean(speedups) if speedups else None
            ),
            "minimum_speedup_vs_baseline": min(speedups) if speedups else None,
            "maximum_speedup_vs_baseline": max(speedups) if speedups else None,
            "geomean_gain_vs_mtp_current": (
                statistics.geometric_mean(current_gains) if current_gains else None
            ),
            "all_outputs_match_baseline": all(
                row.get("output_matches_baseline") is True for row in rows
            ),
            "beats_baseline_everywhere": bool(speedups)
            and all(value > 1.0 for value in speedups),
            "beats_current_everywhere": bool(current_gains)
            and all(value > 1.0 for value in current_gains),
        }
        (serving_ranking if item["complete"] else serving_incomplete).append(item)

    serving_ranking.sort(
        key=lambda item: item.get("geomean_speedup_vs_baseline") or 0.0,
        reverse=True,
    )
    serving_incomplete.sort(
        key=lambda item: (
            item["workloads_completed"],
            item.get("geomean_speedup_vs_baseline") or 0.0,
        ),
        reverse=True,
    )

    pressure_by_arm: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in pressure_rows:
        pressure_by_arm[row["arm"]][int(row["retained_prefixes"])] = row
    pressure_ranking: list[dict[str, Any]] = []
    for arm, stages in pressure_by_arm.items():
        if not arm.startswith("mtp_"):
            continue
        ordered = sorted(stages)
        speedups = [
            stages[stage].get("speedup_vs_matched_baseline") for stage in ordered
        ]
        speedups = [value for value in speedups if value is not None]
        current_gains = [
            stages[stage].get("speedup_vs_mtp_current") for stage in ordered
        ]
        current_gains = [value for value in current_gains if value is not None]
        stage0 = stages.get(0)
        stage24 = stages.get(24)
        pressure_ranking.append(
            {
                "arm": arm,
                "stages_completed": ordered,
                "geomean_speedup_vs_baseline": (
                    statistics.geometric_mean(speedups) if speedups else None
                ),
                "geomean_gain_vs_mtp_current": (
                    statistics.geometric_mean(current_gains) if current_gains else None
                ),
                "stage24_retention_vs_stage0": (
                    stage24["output_throughput_tok_s"]
                    / stage0["output_throughput_tok_s"]
                    if stage0 and stage24
                    else None
                ),
                "all_outputs_match_baseline": all(
                    stages[stage].get("output_matches_baseline") is True
                    for stage in ordered
                ),
            }
        )
    pressure_ranking.sort(
        key=lambda item: item.get("geomean_speedup_vs_baseline") or 0.0,
        reverse=True,
    )

    all_mtp_rows = [
        row
        for row in serving_rows + pressure_rows
        if row.get("arm_config", {}).get("mode") == "mtp"
    ]
    correctness_ok = bool(all_mtp_rows) and all(
        row.get("output_matches_baseline") is True for row in all_mtp_rows
    )
    deferred_complete = any(
        item["arm"] in {"mtp_deferred", "mtp_deferred_inproc"}
        for item in serving_ranking
    )

    combined = {
        "shards_found": len(summaries),
        "shards": shards,
        "serving_workloads": serving_workloads,
        "best_serving_arm": serving_ranking[0] if serving_ranking else None,
        "serving_ranking": serving_ranking,
        "serving_incomplete": serving_incomplete,
        "best_pressure_arm": pressure_ranking[0] if pressure_ranking else None,
        "pressure_ranking": pressure_ranking,
        "correctness_gate_passed": correctness_ok,
        "deferred_complete": deferred_complete,
        "errors": errors,
        "serving_rows": serving_rows,
        "pressure_rows": pressure_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "combined_summary.json", combined)

    lines = [
        "# Qwen native MTP breakthrough matrix — combined",
        "",
        f"- Shard reports: **{len(summaries)}**",
        f"- Serving workloads: **{len(serving_workloads)}**",
        f"- Correctness/hash gate: **{correctness_ok}**",
        f"- Recorded errors: **{len(errors)}**",
        "",
        "## Serving ranking",
        "",
        "| Rank | Arm | Geomean vs baseline | Worst | Best | vs current MTP | Beats baseline everywhere | Hash parity |",
        "|---:|---|---:|---:|---:|---:|:---:|:---:|",
    ]
    for index, item in enumerate(serving_ranking, 1):
        lines.append(
            f"| {index} | {item['arm']} | "
            f"{ratio(item['geomean_speedup_vs_baseline'])} | "
            f"{ratio(item['minimum_speedup_vs_baseline'])} | "
            f"{ratio(item['maximum_speedup_vs_baseline'])} | "
            f"{ratio(item['geomean_gain_vs_mtp_current'])} | "
            f"{item['beats_baseline_everywhere']} | "
            f"{item['all_outputs_match_baseline']} |"
        )
    if not serving_ranking:
        lines.append("| — | No complete serving arm | — | — | — | — | — | — |")

    lines.extend(
        [
            "",
            "## Best serving arm by workload",
            "",
            "| Workload | Arm | Output tok/s | vs baseline | vs current MTP | TTFT ms | Decode tok/s | Acceptance |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    by_workload: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in serving_rows:
        if row.get("arm_config", {}).get("mode") == "mtp":
            by_workload[row["workload"]].append(row)
    for workload in sorted(by_workload):
        row = max(
            by_workload[workload], key=lambda item: item["output_throughput_tok_s"]
        )
        lines.append(
            f"| {workload} | {row['arm']} | "
            f"{row['output_throughput_tok_s']:.2f} | "
            f"{ratio(row.get('speedup_vs_matched_baseline'))} | "
            f"{ratio(row.get('speedup_vs_mtp_current'))} | "
            f"{row['mean_ttft_ms']:.1f} | "
            f"{row.get('median_decode_tok_s_after_first') or 0.0:.2f} | "
            f"{percent(row.get('mtp_acceptance_rate'))} |"
        )

    lines.extend(
        [
            "",
            "## Retained-prefix pressure",
            "",
            "| Rank | Arm | Stages | Geomean vs baseline | vs current MTP | Stage-24 / stage-0 | Hash parity |",
            "|---:|---|---|---:|---:|---:|:---:|",
        ]
    )
    for index, item in enumerate(pressure_ranking, 1):
        lines.append(
            f"| {index} | {item['arm']} | {item['stages_completed']} | "
            f"{ratio(item['geomean_speedup_vs_baseline'])} | "
            f"{ratio(item['geomean_gain_vs_mtp_current'])} | "
            f"{ratio(item['stage24_retention_vs_stage0'])} | "
            f"{item['all_outputs_match_baseline']} |"
        )
    if not pressure_ranking:
        lines.append("| — | No pressure results | — | — | — | — | — |")

    if errors:
        lines.extend(["", "## Failed cases", ""])
        lines.extend(
            f"- `{item.get('shard')}/{item.get('arm')}/{item.get('workload')}`: {item.get('error')}"
            for item in errors
        )
    report = "\n".join(lines) + "\n"
    (args.output_dir / "combined_summary.md").write_text(report)
    print(report)
    return 0 if correctness_ok and deferred_complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
