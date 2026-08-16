# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the compact deferred GDN decode-state experiment."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from vllm_metal.attention.caches.gdn_cache import GDNPagedStateCache
from vllm_metal.attention.context import PagedAttentionContext, clear_context
from vllm_metal.attention.impls.gdn_lazy import (
    GDNLazyKernels,
    GDNRecurrentDecodeRequest,
)
from vllm_metal.attention.state.align import AlignGDNStateManager
from vllm_metal.experimental.gdn_deferred_decode_state import (
    apply_deferred_gdn_state_patch,
    remove_deferred_gdn_state_patch_for_tests,
)


@pytest.fixture(autouse=True)
def _install_patch(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("VLLM_METAL_GDN_DEFER_DECODE_STATE", "1")
    remove_deferred_gdn_state_patch_for_tests()
    assert apply_deferred_gdn_state_patch()
    yield
    clear_context()
    remove_deferred_gdn_state_patch_for_tests()


def _cache(*, initial_seqs: int = 8) -> GDNPagedStateCache:
    return GDNPagedStateCache(
        num_layers=1,
        max_seqs=16,
        conv_kernel_dim=2,
        conv_dim=4,
        num_v_heads=1,
        value_head_dim=4,
        key_head_dim=32,
        initial_seqs=initial_seqs,
        dtype=mx.float32,
    )


class _IncrementConvKernel:
    def __call__(self, *, inputs, output_shapes, output_dtypes, **_kwargs):
        state_in = inputs[1]
        state_slot_ids = inputs[3]
        updates = state_in[state_slot_ids] + 1
        output = mx.zeros(output_shapes[0], dtype=output_dtypes[0])
        return output, updates


class _IncrementRecurrentKernel:
    def __call__(self, *, inputs, output_shapes, output_dtypes, **_kwargs):
        state_in = inputs[5]
        state_slot_ids = inputs[6]
        updates = state_in[state_slot_ids] + 1
        output = mx.zeros(output_shapes[0], dtype=output_dtypes[0])
        return output, updates


def _lazy() -> GDNLazyKernels:
    unused = object()
    return GDNLazyKernels(
        enabled=True,
        conv_kernel=_IncrementConvKernel(),
        conv_prefill_kernel=unused,
        recurrent_decode_kernel=_IncrementRecurrentKernel(),
        recurrent_prefill_kernel=unused,
    )


def _two_request_manager() -> tuple[GDNPagedStateCache, AlignGDNStateManager]:
    cache = _cache()
    manager = AlignGDNStateManager(cache, block_size=4)
    ctx = PagedAttentionContext(
        slot_mapping=[],
        cu_seqlens=[0, 1, 2],
        num_decode_requests=2,
    )
    manager.populate_step_context(
        req_ids=["left", "right"],
        ctx=ctx,
        state_block_ids=[[[2, 3, 4]], [[5, 6, 7]]],
        step_positions=[(5, 1), (5, 1)],
    )
    assert ctx.gdn_group_slot_mappings == ([3, 6],)
    return cache, manager


def _set_two_pending_rows(cache: GDNPagedStateCache) -> None:
    cache.set_pending_conv_state(
        0,
        [3, 6],
        mx.array(
            [
                [[7.0, 7.0, 7.0, 7.0]],
                [[8.0, 8.0, 8.0, 8.0]],
            ],
            dtype=mx.float32,
        ),
    )
    cache.set_pending_recurrent_state(
        0,
        [3, 6],
        mx.concatenate(
            [
                mx.full((1, 1, 4, 32), 9, dtype=mx.float32),
                mx.full((1, 1, 4, 32), 10, dtype=mx.float32),
            ],
            axis=0,
        ),
    )


def test_decode_replaces_compact_conv_state_without_touching_pool() -> None:
    cache = _cache()
    pool = cache.conv_states[0]
    pool[3] = mx.full_like(pool[3], 2)
    cache.store_conv_state(0, pool)
    mx.eval(pool)

    lazy = _lazy()
    inner = SimpleNamespace(
        conv_kernel_size=2,
        conv1d=SimpleNamespace(weight=mx.ones((4, 1), dtype=mx.float32)),
    )
    token = mx.ones((1, 1, 4), dtype=mx.float32)

    assert lazy.try_conv_decode(token, inner, cache, 0, [3]) is not None
    assert cache.has_pending_conv_state(0)
    np.testing.assert_allclose(np.array(cache.conv_states[0][3]), 2)
    np.testing.assert_allclose(np.array(cache.pending_conv_state(0, [3])), 3)

    assert lazy.try_conv_decode(token, inner, cache, 0, [3]) is not None
    assert cache.has_pending_conv_state(0)
    np.testing.assert_allclose(np.array(cache.conv_states[0][3]), 2)
    np.testing.assert_allclose(np.array(cache.pending_conv_state(0, [3])), 4)

    cache.apply_pending_conv_state(0)
    mx.eval(cache.conv_states[0])
    assert not cache.has_pending_conv_state(0)
    np.testing.assert_allclose(np.array(cache.conv_states[0][3]), 4)


def test_decode_replaces_compact_recurrent_state_without_touching_pool() -> None:
    cache = _cache()
    pool = cache.recurrent_states[0]
    pool[3] = mx.full_like(pool[3], 2)
    cache.store_recurrent_state(0, pool)
    mx.eval(pool)

    lazy = _lazy()
    request = GDNRecurrentDecodeRequest(
        q=mx.zeros((1, 1, 1, 32), dtype=mx.float32),
        k=mx.zeros((1, 1, 1, 32), dtype=mx.float32),
        v=mx.zeros((1, 1, 1, 4), dtype=mx.float32),
        g=mx.zeros((1, 1, 1), dtype=mx.float32),
        beta=mx.zeros((1, 1, 1), dtype=mx.float32),
        state_cache=cache,
        cache_idx=0,
        slot_ids=[3],
        output_dtype=mx.float32,
    )

    assert lazy.try_recurrent_decode(request) is not None
    assert cache.has_pending_recurrent_state(0)
    np.testing.assert_allclose(np.array(cache.recurrent_states[0][3]), 2)
    np.testing.assert_allclose(np.array(cache.pending_recurrent_state(0, [3])), 3)

    assert lazy.try_recurrent_decode(request) is not None
    assert cache.has_pending_recurrent_state(0)
    np.testing.assert_allclose(np.array(cache.recurrent_states[0][3]), 2)
    np.testing.assert_allclose(np.array(cache.pending_recurrent_state(0, [3])), 4)

    cache.apply_pending_recurrent_state(0)
    mx.eval(cache.recurrent_states[0])
    assert not cache.has_pending_recurrent_state(0)
    np.testing.assert_allclose(np.array(cache.recurrent_states[0][3]), 4)


def test_align_materialize_keeps_compact_state_until_checkpoint_boundary() -> None:
    cache = _cache()
    manager = AlignGDNStateManager(cache, block_size=4)
    conv_update = mx.full((1, 1, 4), 7, dtype=mx.float32)
    recurrent_update = mx.full((1, 1, 4, 32), 9, dtype=mx.float32)
    cache.set_pending_conv_state(0, [3], conv_update)
    cache.set_pending_recurrent_state(0, [3], recurrent_update)

    manager._needs_materialize = True
    manager._deferred_gdn_flush_after_step = False
    manager.materialize_pending_state()
    assert cache.has_pending_conv_state(0)
    assert cache.has_pending_recurrent_state(0)
    np.testing.assert_allclose(np.array(cache.conv_states[0][3]), 0)
    np.testing.assert_allclose(np.array(cache.recurrent_states[0][3]), 0)

    manager._needs_materialize = True
    manager._deferred_gdn_flush_after_step = True
    manager.materialize_pending_state()
    assert not cache.has_pending_conv_state(0)
    assert not cache.has_pending_recurrent_state(0)
    np.testing.assert_allclose(np.array(cache.conv_states[0][3]), 7)
    np.testing.assert_allclose(np.array(cache.recurrent_states[0][3]), 9)


def test_same_block_align_step_does_not_drain_pending_state() -> None:
    cache = _cache()
    manager = AlignGDNStateManager(cache, block_size=4)
    cache.set_pending_conv_state(
        0, [3], mx.full((1, 1, 4), 7, dtype=mx.float32)
    )
    cache.set_pending_recurrent_state(
        0, [3], mx.full((1, 1, 4, 32), 9, dtype=mx.float32)
    )
    ctx = PagedAttentionContext(
        slot_mapping=[],
        cu_seqlens=[0, 1],
        num_decode_requests=1,
    )

    manager.populate_step_context(
        req_ids=["req"],
        ctx=ctx,
        state_block_ids=[[[2, 3, 4]]],
        step_positions=[(5, 1)],
    )

    assert cache.has_pending_conv_state(0)
    assert cache.has_pending_recurrent_state(0)
    assert ctx.gdn_group_slot_mappings == ([3],)
    assert manager._deferred_gdn_request_slots == {"req": (3,)}
    assert not manager._deferred_gdn_flush_after_step


def test_block_transition_flushes_pending_state_before_copy() -> None:
    cache = _cache()
    manager = AlignGDNStateManager(cache, block_size=4)
    cache.set_pending_conv_state(
        0, [3], mx.full((1, 1, 4), 7, dtype=mx.float32)
    )
    cache.set_pending_recurrent_state(
        0, [3], mx.full((1, 1, 4, 32), 9, dtype=mx.float32)
    )
    ctx = PagedAttentionContext(
        slot_mapping=[],
        cu_seqlens=[0, 1],
        num_decode_requests=1,
    )

    manager.populate_step_context(
        req_ids=["req"],
        ctx=ctx,
        state_block_ids=[[[2, 3, 4]]],
        step_positions=[(8, 1)],
    )
    mx.eval(cache.conv_states[0], cache.recurrent_states[0])

    assert not cache.has_pending_conv_state(0)
    assert not cache.has_pending_recurrent_state(0)
    assert ctx.gdn_group_slot_mappings == ([4],)
    assert manager._deferred_gdn_request_slots == {"req": (4,)}
    np.testing.assert_allclose(np.array(cache.conv_states[0][4]), 7)
    np.testing.assert_allclose(np.array(cache.recurrent_states[0][4]), 9)


def test_release_prunes_one_compact_row_without_touching_pool() -> None:
    cache, manager = _two_request_manager()
    _set_two_pending_rows(cache)

    manager.release_requests({"left"})

    assert manager._deferred_gdn_request_slots == {"right": (6,)}
    assert cache.pending_conv_slot_ids[0] == [6]
    assert cache.pending_recurrent_slot_ids[0] == [6]
    np.testing.assert_allclose(np.array(cache.pending_conv_state(0, [6])), 8)
    np.testing.assert_allclose(np.array(cache.pending_recurrent_state(0, [6])), 10)
    np.testing.assert_allclose(np.array(cache.conv_states[0][3]), 0)
    np.testing.assert_allclose(np.array(cache.conv_states[0][6]), 0)
    np.testing.assert_allclose(np.array(cache.recurrent_states[0][3]), 0)
    np.testing.assert_allclose(np.array(cache.recurrent_states[0][6]), 0)
    assert not getattr(manager, "_deferred_gdn_force_flush", False)


def test_release_prunes_all_compact_rows_without_pool_flush() -> None:
    cache, manager = _two_request_manager()
    _set_two_pending_rows(cache)

    manager.release_requests({"left", "right"})

    assert manager._deferred_gdn_request_slots == {}
    assert not cache.has_pending_conv_state(0)
    assert not cache.has_pending_recurrent_state(0)
    np.testing.assert_allclose(np.array(cache.conv_states[0][3]), 0)
    np.testing.assert_allclose(np.array(cache.conv_states[0][6]), 0)
    np.testing.assert_allclose(np.array(cache.recurrent_states[0][3]), 0)
    np.testing.assert_allclose(np.array(cache.recurrent_states[0][6]), 0)
    assert not getattr(manager, "_deferred_gdn_force_flush", False)


def test_release_keeps_row_still_owned_by_another_request() -> None:
    cache = _cache()
    manager = AlignGDNStateManager(cache, block_size=4)
    manager._deferred_gdn_request_slots = {
        "left": (3,),
        "right": (3,),
    }
    cache.set_pending_conv_state(
        0, [3], mx.full((1, 1, 4), 7, dtype=mx.float32)
    )
    cache.set_pending_recurrent_state(
        0, [3], mx.full((1, 1, 4, 32), 9, dtype=mx.float32)
    )

    manager.release_requests({"left"})

    assert manager._deferred_gdn_request_slots == {"right": (3,)}
    assert cache.has_pending_conv_state(0)
    assert cache.has_pending_recurrent_state(0)
    np.testing.assert_allclose(np.array(cache.pending_conv_state(0, [3])), 7)
    np.testing.assert_allclose(np.array(cache.pending_recurrent_state(0, [3])), 9)


def test_missing_release_mapping_retains_safe_force_flush_fallback() -> None:
    cache = _cache()
    manager = AlignGDNStateManager(cache, block_size=4)
    cache.set_pending_conv_state(
        0, [3], mx.full((1, 1, 4), 7, dtype=mx.float32)
    )
    cache.set_pending_recurrent_state(
        0, [3], mx.full((1, 1, 4, 32), 9, dtype=mx.float32)
    )

    manager.release_requests({"unknown"})

    assert manager._deferred_gdn_force_flush
    manager.materialize_pending_state()
    assert not cache.has_pending_conv_state(0)
    assert not cache.has_pending_recurrent_state(0)
    np.testing.assert_allclose(np.array(cache.conv_states[0][3]), 7)
    np.testing.assert_allclose(np.array(cache.recurrent_states[0][3]), 9)
