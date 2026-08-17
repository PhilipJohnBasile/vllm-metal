# SPDX-License-Identifier: Apache-2.0
"""Bridge the copyless and deferred experimental GDN decode paths.

The copyless candidate extends the lazy GDN decode contract so state can be
read from the committed rollback slot and written directly to a speculative
checkpoint slot. The deferred-state experiment replaces those lazy methods to
retain ordinary one-token decode updates in compact request-local arrays.

When both experiments are enabled, explicit cross-slot writes must retain the
copyless contract while ordinary same-slot decode must retain the deferred
contract. This shim is installed only when the method saved by the deferred
patch advertises ``write_slot_ids``; otherwise it is a no-op.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from typing import Any

import mlx.core as mx

from vllm_metal.attention.impls import gdn_lazy as lazy_mod
from vllm_metal.experimental import gdn_deferred_decode_state as deferred_mod

logger = logging.getLogger(__name__)

_PATCHED = False
_DEFERRED_METHODS: dict[str, Callable[..., Any]] = {}


def _supports_write_slots(function: Callable[..., Any]) -> bool:
    try:
        return "write_slot_ids" in inspect.signature(function).parameters
    except (TypeError, ValueError):
        return False


def apply_copyless_deferred_gdn_compat_patch() -> bool:
    """Preserve both contracts when copyless and deferred state are combined."""
    global _PATCHED
    if _PATCHED:
        return True
    if not getattr(deferred_mod, "_PATCHED", False):
        return False

    originals = getattr(deferred_mod, "_ORIGINALS", {})
    copyless_conv = originals.get("try_conv_decode")
    copyless_recurrent = originals.get("try_recurrent_decode")
    if not callable(copyless_conv) or not callable(copyless_recurrent):
        raise RuntimeError(
            "deferred GDN patch did not retain its lazy-kernel originals"
        )
    if not _supports_write_slots(copyless_conv):
        logger.debug(
            "Copyless GDN destination contract is absent; compatibility bridge "
            "is not required"
        )
        return False

    deferred_conv = lazy_mod.GDNLazyKernels.try_conv_decode
    deferred_recurrent = lazy_mod.GDNLazyKernels.try_recurrent_decode
    _DEFERRED_METHODS.update(
        {
            "try_conv_decode": deferred_conv,
            "try_recurrent_decode": deferred_recurrent,
        }
    )

    def try_conv_decode(
        self: lazy_mod.GDNLazyKernels,
        mixed_qkv: mx.array,
        inner: Any,
        state_cache: Any,
        cache_idx: int,
        slot_ids: list[int],
        write_slot_ids: list[int] | None = None,
    ) -> mx.array | None:
        if write_slot_ids is not None:
            return copyless_conv(
                self,
                mixed_qkv,
                inner,
                state_cache,
                cache_idx,
                slot_ids,
                write_slot_ids=write_slot_ids,
            )
        return deferred_conv(
            self,
            mixed_qkv,
            inner,
            state_cache,
            cache_idx,
            slot_ids,
        )

    def try_recurrent_decode(
        self: lazy_mod.GDNLazyKernels,
        request: lazy_mod.GDNRecurrentDecodeRequest,
    ) -> mx.array | None:
        if getattr(request, "write_slot_ids", None) is not None:
            return copyless_recurrent(self, request)
        return deferred_recurrent(self, request)

    lazy_mod.GDNLazyKernels.try_conv_decode = try_conv_decode
    lazy_mod.GDNLazyKernels.try_recurrent_decode = try_recurrent_decode
    _PATCHED = True
    logger.info("Enabled copyless + deferred GDN compatibility bridge")
    return True


def remove_copyless_deferred_gdn_compat_patch_for_tests() -> None:
    """Restore the deferred methods before its own test cleanup runs."""
    global _PATCHED
    if not _PATCHED:
        return
    lazy_mod.GDNLazyKernels.try_conv_decode = _DEFERRED_METHODS[
        "try_conv_decode"
    ]
    lazy_mod.GDNLazyKernels.try_recurrent_decode = _DEFERRED_METHODS[
        "try_recurrent_decode"
    ]
    _DEFERRED_METHODS.clear()
    _PATCHED = False
