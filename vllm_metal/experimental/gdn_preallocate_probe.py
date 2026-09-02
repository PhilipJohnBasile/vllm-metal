# SPDX-License-Identifier: Apache-2.0
"""Experimental steady-state preallocation probe for align-mode GDN state.

Upstream prefix caching grows scheduler-indexed GDN checkpoint arrays
geometrically.  This probe allocates the already-budgeted maximum once before
the first state-planning step so the benchmark can distinguish:

- cost caused by repeated high-water growth / allocator churn; from
- cost caused by touching a large monolithic checkpoint array itself.

It is a diagnostic switch, not a proposed production default. Large Qwen
checkpoints may reserve substantial unified memory, so it remains opt-in via
``VLLM_METAL_GDN_PREALLOCATE_CHECKPOINTS=1``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

import mlx.core as mx

from vllm_metal.attention.state.align import AlignGDNStateManager

logger = logging.getLogger(__name__)

_ENV_NAME = "VLLM_METAL_GDN_PREALLOCATE_CHECKPOINTS"
_PATCHED = False
_ORIGINAL: Callable[..., Any] | None = None


def _enabled() -> bool:
    return os.getenv(_ENV_NAME, "0") == "1"


def _populate_step_context(
    self: AlignGDNStateManager,
    *,
    req_ids: list[str],
    ctx: Any,
    state_block_ids: list[list[list[int]]] | None = None,
    step_positions: list[tuple[int, int]] | None = None,
) -> None:
    if not getattr(self, "_gdn_preallocation_probe_complete", False):
        cache = self._state_cache
        cache.ensure_capacity(cache.max_seqs)
        arrays = cache.updated_state_arrays()
        if arrays:
            mx.eval(*arrays)
        self._gdn_preallocation_probe_complete = True
        logger.info(
            "Preallocated %d scheduler-indexed GDN checkpoint rows",
            cache.allocated_seqs,
        )
    assert _ORIGINAL is not None
    _ORIGINAL(
        self,
        req_ids=req_ids,
        ctx=ctx,
        state_block_ids=state_block_ids,
        step_positions=step_positions,
    )


def apply_gdn_preallocation_probe() -> bool:
    global _PATCHED, _ORIGINAL
    if _PATCHED or not _enabled():
        return _PATCHED
    _ORIGINAL = AlignGDNStateManager.populate_step_context
    AlignGDNStateManager.populate_step_context = _populate_step_context
    _PATCHED = True
    logger.info("Enabled GDN checkpoint preallocation probe")
    return True


def remove_gdn_preallocation_probe_for_tests() -> None:
    global _PATCHED, _ORIGINAL
    if not _PATCHED:
        return
    assert _ORIGINAL is not None
    AlignGDNStateManager.populate_step_context = _ORIGINAL
    _ORIGINAL = None
    _PATCHED = False
