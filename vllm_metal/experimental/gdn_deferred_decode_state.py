# SPDX-License-Identifier: Apache-2.0
"""Experimental compact GDN decode-state handoff.

The align-mode hybrid cache stores scheduler-owned checkpoints in arrays indexed
by scheduler block id. Ordinary one-token decode previously scattered every
layer's compact state update back into those high-water arrays after every
step. MLX represents that indexed write as a new full-array value, so the cost
grows with the highest resident block id even though only a handful of request
rows are active.

This experiment keeps decode updates compact while the active request/slot
ordering remains unchanged. The stable checkpoint pool is flushed only when it
is actually needed: a block transition, zero-init, scheduler copy, prefix
restore, speculative state chain, or completed block boundary. When a request
finishes inside a partial block, its compact row is discarded instead of being
written into a block that automatic prefix caching cannot reuse.

The patch is guarded by ``VLLM_METAL_GDN_DEFER_DECODE_STATE=1`` and is applied
from the plugin registration path. It is intentionally isolated here while
serving A/Bs qualify the design; once proven, the methods should be folded into
the owning modules rather than kept as a compatibility patch.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

import mlx.core as mx

from vllm_metal.attention.context import get_context
from vllm_metal.attention.impls import gdn_lazy as lazy_mod
from vllm_metal.attention.state.align import AlignGDNStateManager

logger = logging.getLogger(__name__)

_ENV_NAME = "VLLM_METAL_GDN_DEFER_DECODE_STATE"
_PATCHED = False
_ORIGINALS: dict[str, Callable[..., Any]] = {}
_REQUEST_SLOTS_ATTR = "_deferred_gdn_request_slots"


def _enabled() -> bool:
    return os.getenv(_ENV_NAME, "0") == "1"


def _has_speculative_state_chains() -> bool:
    """Return whether the active context needs per-token checkpoint writes."""
    ctx = get_context()
    if ctx is None or ctx.gdn_group_state_chains is None:
        return False
    return any(
        bool(chain)
        for group in ctx.gdn_group_state_chains
        for chain in group
    )


def _should_defer_decode_state() -> bool:
    return _enabled() and not _has_speculative_state_chains()


def _try_conv_decode(
    self: lazy_mod.GDNLazyKernels,
    mixed_qkv: mx.array,
    inner: Any,
    state_cache: Any,
    cache_idx: int,
    slot_ids: list[int],
) -> mx.array | None:
    """Run lazy conv decode while retaining compact state when safe."""
    num_requests = len(slot_ids)
    total_tokens = mixed_qkv.shape[1]
    if not (
        self._enabled
        and self._conv_kernel is not None
        and total_tokens == num_requests
    ):
        return None

    conv_dim = state_cache.conv_dim
    kernel_size = inner.conv_kernel_size
    state_view = state_cache.conv_state_for_decode(cache_idx, slot_ids)
    conv_state_in = state_view.state
    state_pool = state_cache.conv_states[cache_idx]
    weight = inner.conv1d.weight

    mixed_qkv_2d = mixed_qkv.reshape(num_requests, conv_dim)
    slot_ids_arr = state_view.cache_slot_ids
    state_slot_ids_arr = state_view.state_slot_ids

    grid_size = num_requests * conv_dim
    tg_size = min(256, grid_size)
    state_updates_shape = (num_requests, kernel_size - 1, conv_dim)

    conv_silu_out, conv_state_updates = self._conv_kernel(
        inputs=[
            mixed_qkv_2d,
            conv_state_in,
            weight,
            state_slot_ids_arr,
            num_requests,
        ],
        template=[
            ("T", mixed_qkv.dtype),
            ("StT", conv_state_in.dtype),
            ("CONV_DIM", conv_dim),
            ("KERNEL_SIZE", kernel_size),
        ],
        grid=(grid_size, 1, 1),
        threadgroup=(tg_size, 1, 1),
        output_shapes=[(num_requests, conv_dim), state_updates_shape],
        output_dtypes=[mixed_qkv.dtype, conv_state_in.dtype],
    )

    if _should_defer_decode_state():
        # A matching compact input was consumed by this launch. Replace it
        # directly instead of scattering the old compact value first.
        if state_view.uses_compact_state:
            state_cache.clear_pending_conv_state(cache_idx)
        state_cache.set_pending_conv_state(cache_idx, slot_ids, conv_state_updates)
    else:
        state_pool[slot_ids_arr] = conv_state_updates
        state_cache.store_conv_state(cache_idx, state_pool)
        if state_view.uses_compact_state:
            state_cache.clear_pending_conv_state(cache_idx)

    return conv_silu_out.reshape(1, total_tokens, conv_dim)


def _try_recurrent_decode(
    self: lazy_mod.GDNLazyKernels,
    request: lazy_mod.GDNRecurrentDecodeRequest,
) -> mx.array | None:
    """Run lazy recurrent decode while retaining compact state when safe."""
    total_tokens = request.total_tokens
    n_hk = request.num_key_heads
    n_hv = request.num_value_heads
    d_k = request.key_head_dim
    d_v = request.value_head_dim
    num_requests = len(request.slot_ids)
    recurrent_shape_supported = (
        d_k % lazy_mod._RECURRENT_SIMDGROUP_WIDTH == 0
        and d_k <= lazy_mod._RECURRENT_MAX_KEY_HEAD_DIM
    )
    if not (
        self._enabled
        and self._recurrent_decode_kernel is not None
        and total_tokens == num_requests
        and recurrent_shape_supported
    ):
        return None

    state_cache = request.state_cache
    state_view = state_cache.recurrent_state_for_decode(
        request.cache_idx, request.slot_ids
    )
    state_in = state_view.state
    state_pool = state_cache.recurrent_states[request.cache_idx]
    slot_ids_arr = state_view.cache_slot_ids
    state_slot_ids_arr = state_view.state_slot_ids

    y_out, state_updates = self._recurrent_decode_kernel(
        inputs=[
            request.q.reshape(total_tokens, n_hk, d_k),
            request.k.reshape(total_tokens, n_hk, d_k),
            request.v.reshape(total_tokens, n_hv, d_v),
            request.g.reshape(total_tokens, n_hv),
            request.beta.reshape(total_tokens, n_hv),
            state_in,
            state_slot_ids_arr,
            num_requests,
        ],
        template=[
            ("T", request.output_dtype),
            ("StT", mx.float32),
            ("Dk", d_k),
            ("Dv", d_v),
            ("Hk", n_hk),
            ("Hv", n_hv),
        ],
        grid=(
            lazy_mod._RECURRENT_SIMDGROUP_WIDTH,
            d_v,
            num_requests * n_hv,
        ),
        threadgroup=(
            lazy_mod._RECURRENT_SIMDGROUP_WIDTH,
            request.threadgroup_dv,
            1,
        ),
        output_shapes=[
            (total_tokens, n_hv, d_v),
            (num_requests, n_hv, d_v, d_k),
        ],
        output_dtypes=[request.output_dtype, mx.float32],
    )

    if _should_defer_decode_state():
        if state_view.uses_compact_state:
            state_cache.clear_pending_recurrent_state(request.cache_idx)
        state_cache.set_pending_recurrent_state(
            request.cache_idx,
            request.slot_ids,
            state_updates,
        )
    else:
        state_pool[slot_ids_arr] = state_updates
        state_cache.store_recurrent_state(request.cache_idx, state_pool)
        if state_view.uses_compact_state:
            state_cache.clear_pending_recurrent_state(request.cache_idx)

    return y_out


def _step_has_speculation(
    manager: AlignGDNStateManager,
    ctx: Any,
    step_positions: list[tuple[int, int]],
) -> bool:
    return any(
        manager._num_speculative_blocks > 0
        and req_idx < ctx.num_decode_requests
        and num_scheduled > 1
        for req_idx, (_num_computed, num_scheduled) in enumerate(step_positions)
    )


def _can_skip_initial_pool_flush(
    manager: AlignGDNStateManager,
    *,
    ctx: Any,
    state_block_ids: list[list[list[int]]] | None,
    step_positions: list[tuple[int, int]] | None,
    req_ids: list[str],
) -> bool:
    """Conservatively prove that this step stays in the same state slabs."""
    if not _enabled() or state_block_ids is None or step_positions is None:
        return False
    if not (len(state_block_ids) == len(step_positions) == len(req_ids)):
        return False
    if _step_has_speculation(manager, ctx, step_positions):
        return False

    high_water = 0
    for tables in state_block_ids:
        for row in tables:
            if row:
                high_water = max(high_water, max(row) + 1)
    if high_water > manager._state_cache.allocated_seqs:
        return False

    num_groups = len(state_block_ids[0]) if state_block_ids else 0
    for group in range(num_groups):
        for req_idx, (num_computed, num_scheduled) in enumerate(step_positions):
            if num_scheduled <= 0 or num_computed <= 0:
                return False
            row = state_block_ids[req_idx][group]
            src_idx = (num_computed - 1) // manager._block_size
            dst_idx = (
                num_computed + num_scheduled - 1
            ) // manager._block_size
            if max(src_idx, dst_idx) >= len(row):
                return False
            if row[src_idx] != row[dst_idx]:
                return False
    return True


def _record_request_slots(
    manager: AlignGDNStateManager,
    req_ids: list[str],
    ctx: Any,
) -> None:
    """Remember each active request's current scheduler slab per GDN group."""
    mappings = ctx.gdn_group_slot_mappings
    if mappings is None:
        return
    if any(len(group) != len(req_ids) for group in mappings):
        raise RuntimeError(
            "deferred GDN request-slot tracking received a malformed mapping"
        )
    registry: dict[str, tuple[int, ...]] = getattr(
        manager, _REQUEST_SLOTS_ATTR, {}
    )
    for req_idx, req_id in enumerate(req_ids):
        registry[req_id] = tuple(
            int(group[req_idx]) for group in mappings
        )
    setattr(manager, _REQUEST_SLOTS_ATTR, registry)


def _populate_step_context(
    self: AlignGDNStateManager,
    *,
    req_ids: list[str],
    ctx: Any,
    state_block_ids: list[list[list[int]]] | None = None,
    step_positions: list[tuple[int, int]] | None = None,
) -> None:
    """Call the stock planner, skipping only its provably redundant drain."""
    original = _ORIGINALS["align_populate_step_context"]

    # A scheduler-visible checkpoint is required at completed block boundaries.
    positions = step_positions or []
    self._deferred_gdn_flush_after_step = bool(
        _step_has_speculation(self, ctx, positions)
        or any(
            num_scheduled > 0
            and (num_computed + num_scheduled) % self._block_size == 0
            for num_computed, num_scheduled in positions
        )
    )

    if not _can_skip_initial_pool_flush(
        self,
        ctx=ctx,
        state_block_ids=state_block_ids,
        step_positions=step_positions,
        req_ids=req_ids,
    ):
        original(
            self,
            req_ids=req_ids,
            ctx=ctx,
            state_block_ids=state_block_ids,
            step_positions=step_positions,
        )
        _record_request_slots(self, req_ids, ctx)
        return

    # The stock planner's first operation is an unconditional pending-state
    # drain. For a pure same-block decode step that operation is unnecessary.
    # Suppress exactly that first call; any later call (for example from a
    # capacity growth path) retains stock behavior.
    cache = self._state_cache
    original_apply = cache.apply_pending_states
    had_instance_override = "apply_pending_states" in cache.__dict__
    previous_override = cache.__dict__.get("apply_pending_states")
    call_count = 0

    def skip_first_apply() -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return
        original_apply()

    cache.__dict__["apply_pending_states"] = skip_first_apply
    try:
        original(
            self,
            req_ids=req_ids,
            ctx=ctx,
            state_block_ids=state_block_ids,
            step_positions=step_positions,
        )
    finally:
        if had_instance_override:
            cache.__dict__["apply_pending_states"] = previous_override
        else:
            cache.__dict__.pop("apply_pending_states", None)
    _record_request_slots(self, req_ids, ctx)


def _materialize_pending_state(self: AlignGDNStateManager) -> None:
    """Keep compact state live except at an explicit checkpoint boundary."""
    if not _enabled():
        _ORIGINALS["align_materialize_pending_state"](self)
        return

    force = bool(getattr(self, "_deferred_gdn_force_flush", False))
    boundary = bool(getattr(self, "_deferred_gdn_flush_after_step", False))
    if not self._needs_materialize and not force:
        return

    if force or boundary:
        self._state_cache.apply_pending_states()
        arrays = self._state_cache.updated_state_arrays()
        if arrays:
            mx.eval(*arrays)

    self._needs_materialize = False
    self._deferred_gdn_force_flush = False
    self._deferred_gdn_flush_after_step = False


def _prune_pending_kind(
    *,
    states: list[mx.array | None],
    slot_lists: list[list[int] | None],
    layer_idx: int,
    released_slots: set[int],
) -> bool:
    """Remove released request rows from one compact per-layer state value."""
    state = states[layer_idx]
    slots = slot_lists[layer_idx]
    if state is None:
        return slots is None
    if slots is None or len(slots) != state.shape[0]:
        return False

    keep_indices = [
        index for index, slot in enumerate(slots) if slot not in released_slots
    ]
    if len(keep_indices) == len(slots):
        return True
    if not keep_indices:
        states[layer_idx] = None
        slot_lists[layer_idx] = None
        return True

    keep = mx.array(keep_indices, dtype=mx.int32)
    states[layer_idx] = state[keep]
    slot_lists[layer_idx] = [slots[index] for index in keep_indices]
    return True


def _prune_released_pending_rows(
    manager: AlignGDNStateManager,
    released_by_group: dict[int, set[int]],
) -> bool:
    """Discard non-cacheable compact rows without touching stable pools."""
    cache = manager._state_cache
    for group, released_slots in released_by_group.items():
        if not released_slots:
            continue
        for layer_idx in cache.layers_for_group_ordinal(group):
            conv_ok = _prune_pending_kind(
                states=cache.pending_conv_states,
                slot_lists=cache.pending_conv_slot_ids,
                layer_idx=layer_idx,
                released_slots=released_slots,
            )
            recurrent_ok = _prune_pending_kind(
                states=cache.pending_recurrent_states,
                slot_lists=cache.pending_recurrent_slot_ids,
                layer_idx=layer_idx,
                released_slots=released_slots,
            )
            if not (conv_ok and recurrent_ok):
                return False
    return True


def _release_requests(self: AlignGDNStateManager, req_ids: set[str]) -> None:
    """Drop released partial-block rows; preserve shared/active compact rows."""
    if not _enabled() or not req_ids:
        _ORIGINALS["align_release_requests"](self, req_ids)
        return

    registry: dict[str, tuple[int, ...]] = getattr(
        self, _REQUEST_SLOTS_ATTR, {}
    )
    released_mappings: list[tuple[int, ...]] = []
    missing_mapping = False
    for req_id in req_ids:
        mapping = registry.pop(req_id, None)
        if mapping is None:
            missing_mapping = True
        else:
            released_mappings.append(mapping)
    setattr(self, _REQUEST_SLOTS_ATTR, registry)

    if released_mappings:
        num_groups = len(released_mappings[0])
        if any(len(mapping) != num_groups for mapping in released_mappings):
            missing_mapping = True
        elif any(len(mapping) != num_groups for mapping in registry.values()):
            missing_mapping = True
        else:
            # Numeric block ids can overlap across scheduler groups, so prune
            # each group's layers independently. Do not remove a slot still
            # owned by another active request.
            remaining_by_group = {
                group: {mapping[group] for mapping in registry.values()}
                for group in range(num_groups)
            }
            released_by_group = {
                group: {
                    mapping[group] for mapping in released_mappings
                }
                - remaining_by_group[group]
                for group in range(num_groups)
            }
            if not _prune_released_pending_rows(self, released_by_group):
                missing_mapping = True

    if missing_mapping:
        # Unknown ownership is rare (for example lifecycle invalidation before
        # the request has ever reached a GDN forward). Retain the old safe
        # fallback instead of guessing which compact row may be recycled.
        self._deferred_gdn_force_flush = True

    _ORIGINALS["align_release_requests"](self, req_ids)


def apply_deferred_gdn_state_patch() -> bool:
    """Install the experiment when its environment switch is enabled."""
    global _PATCHED
    if _PATCHED or not _enabled():
        return _PATCHED

    _ORIGINALS.update(
        {
            "try_conv_decode": lazy_mod.GDNLazyKernels.try_conv_decode,
            "try_recurrent_decode": lazy_mod.GDNLazyKernels.try_recurrent_decode,
            "align_populate_step_context": AlignGDNStateManager.populate_step_context,
            "align_materialize_pending_state": AlignGDNStateManager.materialize_pending_state,
            "align_release_requests": AlignGDNStateManager.release_requests,
        }
    )
    lazy_mod.GDNLazyKernels.try_conv_decode = _try_conv_decode
    lazy_mod.GDNLazyKernels.try_recurrent_decode = _try_recurrent_decode
    AlignGDNStateManager.populate_step_context = _populate_step_context
    AlignGDNStateManager.materialize_pending_state = _materialize_pending_state
    AlignGDNStateManager.release_requests = _release_requests
    _PATCHED = True
    logger.info("Enabled compact deferred GDN decode state")
    return True


def remove_deferred_gdn_state_patch_for_tests() -> None:
    """Restore original methods so focused tests do not leak global patches."""
    global _PATCHED
    if not _PATCHED:
        return
    lazy_mod.GDNLazyKernels.try_conv_decode = _ORIGINALS["try_conv_decode"]
    lazy_mod.GDNLazyKernels.try_recurrent_decode = _ORIGINALS[
        "try_recurrent_decode"
    ]
    AlignGDNStateManager.populate_step_context = _ORIGINALS[
        "align_populate_step_context"
    ]
    AlignGDNStateManager.materialize_pending_state = _ORIGINALS[
        "align_materialize_pending_state"
    ]
    AlignGDNStateManager.release_requests = _ORIGINALS["align_release_requests"]
    _ORIGINALS.clear()
    _PATCHED = False
