#!/usr/bin/env python3
"""Resolve the two known first-commit conflicts in the PR #618 rehearsal.

This is intentionally narrow. It applies only when Git has stopped while
replaying ``bf59b7c`` onto vLLM Metal main after #630/#634. Every conflict side
is checked for the expected semantic anchors before the worktree is modified.

Resolutions:

* ``gdn_lazy.py`` keeps #634's in-place row writer and #618's distinct
  speculative destination slots.
* ``cache_policy.py`` keeps #630's scheduler-managed draft-model specs and then
  prepends #618's Qwen MTP cache-only specs, preserving the MTP grouping
  invariant.
"""

from __future__ import annotations

from pathlib import Path


class ResolutionError(RuntimeError):
    pass


def _split_single_conflict(text: str) -> tuple[str, str, str, str]:
    start = text.find("<<<<<<< HEAD\n")
    if start < 0:
        raise ResolutionError("expected one conflict start marker")
    middle = text.find("=======\n", start)
    end = text.find(">>>>>>> ", middle)
    end_line = text.find("\n", end)
    if middle < 0 or end < 0 or end_line < 0:
        raise ResolutionError("incomplete conflict marker sequence")
    if text.find("<<<<<<< HEAD\n", start + 1) >= 0:
        raise ResolutionError("expected exactly one conflict block")

    prefix = text[:start]
    ours = text[start + len("<<<<<<< HEAD\n") : middle]
    theirs = text[middle + len("=======\n") : end]
    suffix = text[end_line + 1 :]
    return prefix, ours, theirs, suffix


def _resolve_gdn(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    prefix, ours, theirs, suffix = _split_single_conflict(text)

    expected_ours = (
        "        state_cache.write_conv_rows(cache_idx, conv_state_updates, "
        "slot_ids_arr)\n"
    )
    if ours != expected_ours:
        raise ResolutionError(f"unexpected upstream gdn side:\n{ours}")
    if "state_pool[write_slot_ids_arr] = conv_state_updates" not in theirs:
        raise ResolutionError("feature gdn side lost speculative destination slots")
    if "state_cache.store_conv_state(cache_idx, state_pool)" not in theirs:
        raise ResolutionError("feature gdn side lost stable-pool publication")

    resolved = (
        "        state_cache.write_conv_rows(\n"
        "            cache_idx, conv_state_updates, write_slot_ids_arr\n"
        "        )\n"
    )
    path.write_text(prefix + resolved + suffix, encoding="utf-8")


def _resolve_cache_policy(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    prefix, ours, theirs, suffix = _split_single_conflict(text)

    if "self._draft_layer_specs" not in ours:
        raise ResolutionError("upstream cache side lost scheduler-managed draft specs")
    if "metadata = self._qwen_mtp_metadata()" not in theirs:
        raise ResolutionError("feature cache side lost Qwen MTP metadata")
    if "specs = {**mtp_specs, **specs}" not in theirs:
        raise ResolutionError("feature cache side no longer prepends MTP specs")

    # Preserve both. Draft-model specs are appended first; when Qwen native MTP
    # is configured, its distinct cache-only specs are then prepended to the
    # complete target/draft map so vLLM's early grouping probe sees them first.
    resolved = ours + theirs
    path.write_text(prefix + resolved + suffix, encoding="utf-8")


def main() -> int:
    root = Path.cwd()
    gdn = root / "vllm_metal/attention/impls/gdn_lazy.py"
    cache = root / "vllm_metal/v1/cache_policy.py"

    _resolve_gdn(gdn)
    _resolve_cache_policy(cache)

    for path in (gdn, cache):
        text = path.read_text(encoding="utf-8")
        if any(marker in text for marker in ("<<<<<<<", "=======", ">>>>>>>")):
            raise ResolutionError(f"conflict marker remains in {path}")
        print(f"resolved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
