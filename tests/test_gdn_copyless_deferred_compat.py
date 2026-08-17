# SPDX-License-Identifier: Apache-2.0
"""Combined-contract tests for copyless plus deferred GDN state."""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from vllm_metal.attention.caches.gdn_cache import GDNPagedStateCache
from vllm_metal.attention.context import clear_context
from vllm_metal.attention.impls.gdn_lazy import (
    GDNLazyKernels,
    GDNRecurrentDecodeRequest,
)
from vllm_metal.experimental.gdn_copyless_deferred_compat import (
    apply_copyless_deferred_gdn_compat_patch,
    remove_copyless_deferred_gdn_compat_patch_for_tests,
)
from vllm_metal.experimental.gdn_deferred_decode_state import (
    apply_deferred_gdn_state_patch,
    remove_deferred_gdn_state_patch_for_tests,
)

_COPYLESS_AVAILABLE = (
    "write_slot_ids"
    in inspect.signature(GDNLazyKernels.try_conv_decode).parameters
)
pytestmark = pytest.mark.skipif(
    not _COPYLESS_AVAILABLE,
    reason="copyless GDN candidate patch is not applied",
)


@pytest.fixture(autouse=True)
def _install_combined_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    monkeypatch.setenv("VLLM_METAL_GDN_DEFER_DECODE_STATE", "1")
    remove_copyless_deferred_gdn_compat_patch_for_tests()
    remove_deferred_gdn_state_patch_for_tests()
    assert apply_deferred_gdn_state_patch()
    assert apply_copyless_deferred_gdn_compat_patch()
    yield
    clear_context()
    remove_copyless_deferred_gdn_compat_patch_for_tests()
    remove_deferred_gdn_state_patch_for_tests()


def _cache() -> GDNPagedStateCache:
    return GDNPagedStateCache(
        num_layers=1,
        max_seqs=16,
        conv_kernel_dim=2,
        conv_dim=4,
        num_v_heads=1,
        value_head_dim=4,
        key_head_dim=32,
        initial_seqs=8,
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


def test_cross_slot_conv_uses_copyless_destination_contract() -> None:
    cache = _cache()
    pool = cache.conv_states[0]
    pool[3] = mx.full_like(pool[3], 2)
    pool[4] = mx.full_like(pool[4], 9)
    cache.store_conv_state(0, pool)
    mx.eval(cache.conv_states[0])

    inner = SimpleNamespace(
        conv_kernel_size=2,
        conv1d=SimpleNamespace(weight=mx.ones((4, 1), dtype=mx.float32)),
    )
    token = mx.ones((1, 1, 4), dtype=mx.float32)

    output = _lazy().try_conv_decode(
        token,
        inner,
        cache,
        0,
        [3],
        write_slot_ids=[4],
    )
    assert output is not None
    mx.eval(cache.conv_states[0])
    assert not cache.has_pending_conv_state(0)
    np.testing.assert_allclose(np.array(cache.conv_states[0][3]), 2)
    np.testing.assert_allclose(np.array(cache.conv_states[0][4]), 3)


def test_cross_slot_recurrent_uses_copyless_destination_contract() -> None:
    cache = _cache()
    pool = cache.recurrent_states[0]
    pool[3] = mx.full_like(pool[3], 2)
    pool[4] = mx.full_like(pool[4], 9)
    cache.store_recurrent_state(0, pool)
    mx.eval(cache.recurrent_states[0])

    request = GDNRecurrentDecodeRequest(
        q=mx.zeros((1, 1, 1, 32), dtype=mx.float32),
        k=mx.zeros((1, 1, 1, 32), dtype=mx.float32),
        v=mx.zeros((1, 1, 1, 4), dtype=mx.float32),
        g=mx.zeros((1, 1, 1), dtype=mx.float32),
        beta=mx.zeros((1, 1, 1), dtype=mx.float32),
        state_cache=cache,
        cache_idx=0,
        slot_ids=[3],
        write_slot_ids=[4],
        output_dtype=mx.float32,
    )

    output = _lazy().try_recurrent_decode(request)
    assert output is not None
    mx.eval(cache.recurrent_states[0])
    assert not cache.has_pending_recurrent_state(0)
    np.testing.assert_allclose(np.array(cache.recurrent_states[0][3]), 2)
    np.testing.assert_allclose(np.array(cache.recurrent_states[0][4]), 3)


def test_same_slot_conv_preserves_deferred_compact_state() -> None:
    cache = _cache()
    pool = cache.conv_states[0]
    pool[3] = mx.full_like(pool[3], 2)
    cache.store_conv_state(0, pool)
    mx.eval(cache.conv_states[0])

    inner = SimpleNamespace(
        conv_kernel_size=2,
        conv1d=SimpleNamespace(weight=mx.ones((4, 1), dtype=mx.float32)),
    )
    token = mx.ones((1, 1, 4), dtype=mx.float32)

    output = _lazy().try_conv_decode(token, inner, cache, 0, [3])
    assert output is not None
    assert cache.has_pending_conv_state(0)
    np.testing.assert_allclose(np.array(cache.conv_states[0][3]), 2)
    np.testing.assert_allclose(np.array(cache.pending_conv_state(0, [3])), 3)
