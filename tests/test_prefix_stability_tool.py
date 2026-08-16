# SPDX-License-Identifier: Apache-2.0
"""Tests for tools/prefix_stability.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_TOOL = Path(__file__).parents[1] / "tools" / "prefix_stability.py"
_SPEC = importlib.util.spec_from_file_location("prefix_stability", _TOOL)
assert _SPEC is not None and _SPEC.loader is not None
prefix_stability = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prefix_stability)


def _request(*, request_id: str = "r1", tools_reversed: bool = False):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "path": {"type": "string"},
                    },
                },
            },
        },
    ]
    if tools_reversed:
        tools.reverse()
    return {
        "request_id": request_id,
        "messages": [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "Inspect the repository."},
        ],
        "tools": tools,
        "temperature": 0,
    }


def test_canonicalization_stabilizes_tool_and_mapping_order() -> None:
    left = _request(tools_reversed=False)
    right = _request(tools_reversed=True)
    # Raw requests differ only because the tool array was reordered.
    assert left != right
    assert prefix_stability.canonicalize(left) == prefix_stability.canonicalize(right)
    report = prefix_stability.analyze(left, right)
    assert report.canonical_equal
    assert report.stable_prefix_equal


def test_volatile_request_id_does_not_change_stable_fingerprint() -> None:
    report = prefix_stability.analyze(
        _request(request_id="first"),
        _request(request_id="second"),
    )
    assert not report.raw_equal
    assert not report.canonical_equal
    assert report.stable_prefix_equal
    assert report.stable_sha256_left == report.stable_sha256_right
    assert report.volatile_paths_left == ("$.request_id",)
    assert report.volatile_paths_right == ("$.request_id",)


def test_message_mutation_changes_stable_prefix_and_reports_path() -> None:
    left = _request()
    right = _request()
    right["messages"][0]["content"] = "You are a coding agent. Date: tomorrow."
    report = prefix_stability.analyze(left, right)
    assert not report.stable_prefix_equal
    assert report.first_stable_diff is not None
    assert report.first_stable_diff.path == "$.messages[0].content"


def test_token_comparison_reports_first_mismatch_and_reusable_blocks() -> None:
    result = prefix_stability.compare_token_ids(
        [1, 2, 3, 4, 5, 6, 7],
        [1, 2, 3, 4, 9, 6, 7],
        block_size=4,
    )
    assert result.common_prefix_tokens == 4
    assert result.first_mismatch_index == 4
    assert result.reusable_full_blocks == 1
    assert result.left_cacheable_tokens == 4
    assert result.right_cacheable_tokens == 4
    assert result.left_cacheable_sha256 == result.right_cacheable_sha256


def test_token_length_only_change_marks_end_as_first_mismatch() -> None:
    result = prefix_stability.compare_token_ids([1, 2, 3], [1, 2, 3, 4], block_size=2)
    assert result.common_prefix_tokens == 3
    assert result.first_mismatch_index == 3
    assert result.reusable_full_blocks == 1
