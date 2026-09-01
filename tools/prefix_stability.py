#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Analyze and canonicalize frontend requests for reusable-prefix stability.

Prefix caching is token-exact. Semantically equivalent requests still miss when
frontends reorder tool schemas, serialize dictionaries differently, inject a
clock/request id into the system prefix, or otherwise mutate bytes that land
before the reusable suffix.

This tool compares two OpenAI-style request JSON files and reports:

- the first structural difference;
- volatile-only fields that should move to the suffix or request metadata;
- a canonical stable-prefix fingerprint;
- optional tokenizer-level first mismatch and reusable block count;
- canonical JSON suitable for frontend regression fixtures.

Examples:

    python tools/prefix_stability.py turn1.json turn2.json
    python tools/prefix_stability.py turn1.json turn2.json \
        --model /path/to/Qwen3.8-27B --block-size 544
    python tools/prefix_stability.py request.json --emit-canonical canonical.json

The canonicalizer never reorders conversation messages or arbitrary arrays.
It sorts mapping keys and the top-level ``tools`` array by tool identity, because
those operations preserve request meaning while preventing common frontend
serialization drift. Volatile fields are excluded only from the *stable-prefix*
fingerprint; the canonical full request retains them unless ``--strip-volatile``
is explicitly requested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# These values should be transported out-of-band or appended in the changing
# suffix. Matching is case-insensitive and applies at any mapping depth.
DEFAULT_VOLATILE_KEYS = frozenset(
    {
        "current_date",
        "current_time",
        "generated_at",
        "nonce",
        "now",
        "request_id",
        "run_id",
        "session_id",
        "timestamp",
        "trace_id",
    }
)


@dataclass(frozen=True)
class StructuralDiff:
    path: str
    left: Any
    right: Any
    reason: str


@dataclass(frozen=True)
class TokenComparison:
    left_tokens: int
    right_tokens: int
    common_prefix_tokens: int
    first_mismatch_index: int | None
    reusable_full_blocks: int
    left_cacheable_tokens: int
    right_cacheable_tokens: int
    left_cacheable_sha256: str
    right_cacheable_sha256: str


@dataclass(frozen=True)
class PrefixStabilityReport:
    raw_equal: bool
    canonical_equal: bool
    stable_prefix_equal: bool
    raw_sha256_left: str
    raw_sha256_right: str
    canonical_sha256_left: str
    canonical_sha256_right: str
    stable_sha256_left: str
    stable_sha256_right: str
    first_raw_diff: StructuralDiff | None
    first_stable_diff: StructuralDiff | None
    volatile_paths_left: tuple[str, ...]
    volatile_paths_right: tuple[str, ...]
    token_comparison: TokenComparison | None


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _tool_identity(tool: Any) -> tuple[str, str]:
    if not isinstance(tool, dict):
        return ("", sha256_json(tool))
    function = tool.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        if isinstance(name, str):
            return (name, sha256_json(tool))
    name = tool.get("name")
    return (name if isinstance(name, str) else "", sha256_json(tool))


def canonicalize(value: Any, *, path: tuple[str, ...] = ()) -> Any:
    """Return deterministic request JSON without changing message order."""
    if isinstance(value, dict):
        return {
            key: canonicalize(value[key], path=(*path, key))
            for key in sorted(value)
        }
    if isinstance(value, list):
        items = [
            canonicalize(item, path=(*path, str(index)))
            for index, item in enumerate(value)
        ]
        if path and path[-1] == "tools":
            items.sort(key=_tool_identity)
        return items
    return value


def split_stable_and_volatile(
    value: Any,
    *,
    volatile_keys: frozenset[str] = DEFAULT_VOLATILE_KEYS,
    path: tuple[str, ...] = (),
) -> tuple[Any, dict[str, Any]]:
    """Remove volatile mapping fields from the stable view and record them."""
    volatile: dict[str, Any] = {}
    if isinstance(value, dict):
        stable: dict[str, Any] = {}
        for key in sorted(value):
            child_path = (*path, key)
            rendered = _render_path(child_path)
            if key.casefold() in volatile_keys:
                volatile[rendered] = value[key]
                continue
            child_stable, child_volatile = split_stable_and_volatile(
                value[key],
                volatile_keys=volatile_keys,
                path=child_path,
            )
            stable[key] = child_stable
            volatile.update(child_volatile)
        return stable, volatile
    if isinstance(value, list):
        stable_items: list[Any] = []
        for index, item in enumerate(value):
            child_stable, child_volatile = split_stable_and_volatile(
                item,
                volatile_keys=volatile_keys,
                path=(*path, str(index)),
            )
            stable_items.append(child_stable)
            volatile.update(child_volatile)
        if path and path[-1] == "tools":
            stable_items.sort(key=_tool_identity)
        return stable_items, volatile
    return value, volatile


def _render_path(path: Iterable[str]) -> str:
    rendered = "$"
    for part in path:
        if part.isdigit():
            rendered += f"[{part}]"
        else:
            rendered += f".{part}"
    return rendered


def first_structural_diff(
    left: Any,
    right: Any,
    *,
    path: tuple[str, ...] = (),
) -> StructuralDiff | None:
    if type(left) is not type(right):
        return StructuralDiff(
            _render_path(path), left, right, "type mismatch"
        )
    if isinstance(left, dict):
        left_keys = set(left)
        right_keys = set(right)
        if left_keys != right_keys:
            missing_left = sorted(right_keys - left_keys)
            missing_right = sorted(left_keys - right_keys)
            return StructuralDiff(
                _render_path(path),
                {"missing": missing_left},
                {"missing": missing_right},
                "mapping keys differ",
            )
        for key in sorted(left):
            diff = first_structural_diff(
                left[key], right[key], path=(*path, key)
            )
            if diff is not None:
                return diff
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return StructuralDiff(
                _render_path(path), len(left), len(right), "array lengths differ"
            )
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            diff = first_structural_diff(
                left_item,
                right_item,
                path=(*path, str(index)),
            )
            if diff is not None:
                return diff
        return None
    if left != right:
        return StructuralDiff(_render_path(path), left, right, "values differ")
    return None


def common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    for index, (left_token, right_token) in enumerate(
        zip(left, right, strict=False)
    ):
        if left_token != right_token:
            return index
    return min(len(left), len(right))


def _token_bytes(tokens: Sequence[int]) -> bytes:
    # Unsigned 32-bit token ids cover current tokenizer vocabularies and avoid
    # JSON formatting differences in fingerprints.
    return b"".join(struct.pack(">I", int(token)) for token in tokens)


def compare_token_ids(
    left: Sequence[int],
    right: Sequence[int],
    *,
    block_size: int,
) -> TokenComparison:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    common = common_prefix_length(left, right)
    first_mismatch = None
    if common < max(len(left), len(right)):
        first_mismatch = common
    reusable_blocks = common // block_size
    left_cacheable = (len(left) // block_size) * block_size
    right_cacheable = (len(right) // block_size) * block_size
    return TokenComparison(
        left_tokens=len(left),
        right_tokens=len(right),
        common_prefix_tokens=common,
        first_mismatch_index=first_mismatch,
        reusable_full_blocks=reusable_blocks,
        left_cacheable_tokens=left_cacheable,
        right_cacheable_tokens=right_cacheable,
        left_cacheable_sha256=hashlib.sha256(
            _token_bytes(left[:left_cacheable])
        ).hexdigest(),
        right_cacheable_sha256=hashlib.sha256(
            _token_bytes(right[:right_cacheable])
        ).hexdigest(),
    )


def _tokenize_request(payload: dict[str, Any], model: str) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise ValueError("token comparison requires a request with a messages array")
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
    }
    tools = payload.get("tools")
    if tools is not None:
        kwargs["tools"] = tools
    try:
        token_ids = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("tools", None)
        token_ids = tokenizer.apply_chat_template(messages, **kwargs)
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(token) for token in token_ids]


def analyze(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    model: str | None = None,
    block_size: int = 16,
    volatile_keys: frozenset[str] = DEFAULT_VOLATILE_KEYS,
) -> PrefixStabilityReport:
    canonical_left = canonicalize(left)
    canonical_right = canonicalize(right)
    stable_left, volatile_left = split_stable_and_volatile(
        canonical_left, volatile_keys=volatile_keys
    )
    stable_right, volatile_right = split_stable_and_volatile(
        canonical_right, volatile_keys=volatile_keys
    )

    token_comparison = None
    if model is not None:
        token_comparison = compare_token_ids(
            _tokenize_request(canonical_left, model),
            _tokenize_request(canonical_right, model),
            block_size=block_size,
        )

    return PrefixStabilityReport(
        raw_equal=left == right,
        canonical_equal=canonical_left == canonical_right,
        stable_prefix_equal=stable_left == stable_right,
        raw_sha256_left=sha256_json(left),
        raw_sha256_right=sha256_json(right),
        canonical_sha256_left=sha256_json(canonical_left),
        canonical_sha256_right=sha256_json(canonical_right),
        stable_sha256_left=sha256_json(stable_left),
        stable_sha256_right=sha256_json(stable_right),
        first_raw_diff=first_structural_diff(left, right),
        first_stable_diff=first_structural_diff(stable_left, stable_right),
        volatile_paths_left=tuple(sorted(volatile_left)),
        volatile_paths_right=tuple(sorted(volatile_right)),
        token_comparison=token_comparison,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path, nargs="?")
    parser.add_argument("--model", help="tokenizer/model path for token-level comparison")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--emit-canonical", type=Path)
    parser.add_argument(
        "--strip-volatile",
        action="store_true",
        help="when emitting canonical JSON, remove volatile fields",
    )
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main() -> int:
    args = parse_args()
    left = _load(args.left)

    canonical_left = canonicalize(left)
    if args.strip_volatile:
        canonical_left, _ = split_stable_and_volatile(canonical_left)
    if args.emit_canonical is not None:
        args.emit_canonical.write_text(
            json.dumps(canonical_left, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        )

    if args.right is None:
        stable, volatile = split_stable_and_volatile(canonicalize(left))
        print(
            json.dumps(
                {
                    "canonical_sha256": sha256_json(canonicalize(left)),
                    "stable_prefix_sha256": sha256_json(stable),
                    "volatile_paths": sorted(volatile),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    report = analyze(
        left,
        _load(args.right),
        model=args.model,
        block_size=args.block_size,
    )
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2, sort_keys=True))
    # Exit 2 means the stable prefix changed. Volatile-only or tool-order-only
    # drift exits successfully after canonicalization.
    return 0 if report.stable_prefix_equal else 2


if __name__ == "__main__":
    raise SystemExit(main())
