#!/usr/bin/env python3
"""Apply an experimental copyless one-draft GDN verification patch.

The ordinary one-draft verification path copies the full post-confirmed
convolution and recurrent state from rollback slots into speculative slots,
then advances the draft token in place. This experiment keeps the source state
in the rollback slots and scatters only the compact kernel updates into the
speculative destinations.

This helper is intentionally benchmark-only. It edits the checkout in place;
the benchmark workflow restores the two production files after the A/B run.
"""

from __future__ import annotations

from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def patch_lazy_kernels(path: Path) -> None:
    text = path.read_text()

    text = replace_once(
        text,
        '''class GDNRecurrentDecodeRequest(GDNRecurrentRequest):
    """Inputs for one lazy GDN recurrent decode attempt."""

    threadgroup_dv: int = 4
''',
        '''class GDNRecurrentDecodeRequest(GDNRecurrentRequest):
    """Inputs for one lazy GDN recurrent decode attempt."""

    threadgroup_dv: int = 4
    # Read from slot_ids while scattering compact updates into these slots.
    write_slot_ids: list[int] | None = None
''',
        "decode request destination slots",
    )

    text = replace_once(
        text,
        '''        cache_idx: int,
        slot_ids: list[int],
    ) -> mx.array | None:
        """Run the lazy GDN conv decode fast path, or return None."""
''',
        '''        cache_idx: int,
        slot_ids: list[int],
        write_slot_ids: list[int] | None = None,
    ) -> mx.array | None:
        """Run the lazy GDN conv decode fast path, or return None.

        ``slot_ids`` address source state. ``write_slot_ids`` can select
        different stable-cache destinations for the compact updates.
        """
''',
        "conv decode destination signature",
    )

    text = replace_once(
        text,
        '''        state_view = state_cache.conv_state_for_decode(cache_idx, slot_ids)
        conv_state_in = state_view.state
        state_pool = state_cache.conv_states[cache_idx]
        weight = inner.conv1d.weight

        mixed_qkv_2d = mixed_qkv.reshape(num_requests, conv_dim)
        slot_ids_arr = state_view.cache_slot_ids
        state_slot_ids_arr = state_view.state_slot_ids
''',
        '''        state_view = state_cache.conv_state_for_decode(cache_idx, slot_ids)
        destination_slots = slot_ids if write_slot_ids is None else write_slot_ids
        if len(destination_slots) != num_requests:
            raise RuntimeError("GDN conv decode destination count mismatch")
        state_cache.require_allocated_slots(destination_slots)
        if destination_slots != slot_ids and state_view.uses_compact_state:
            raise RuntimeError(
                "copyless GDN conv decode requires stable source state"
            )
        conv_state_in = state_view.state
        state_pool = state_cache.conv_states[cache_idx]
        weight = inner.conv1d.weight

        mixed_qkv_2d = mixed_qkv.reshape(num_requests, conv_dim)
        slot_ids_arr = state_view.cache_slot_ids
        write_slot_ids_arr = (
            slot_ids_arr
            if destination_slots == slot_ids
            else mx.array(destination_slots, dtype=mx.int32)
        )
        state_slot_ids_arr = state_view.state_slot_ids
''',
        "conv decode destination setup",
    )

    text = replace_once(
        text,
        "        state_pool[slot_ids_arr] = conv_state_updates\n",
        "        state_pool[write_slot_ids_arr] = conv_state_updates\n",
        "conv decode destination scatter",
    )

    text = replace_once(
        text,
        '''        state_view = state_cache.recurrent_state_for_decode(
            request.cache_idx, request.slot_ids
        )
        state_in = state_view.state
        state_pool = state_cache.recurrent_states[request.cache_idx]
        slot_ids_arr = state_view.cache_slot_ids
        state_slot_ids_arr = state_view.state_slot_ids
''',
        '''        state_view = state_cache.recurrent_state_for_decode(
            request.cache_idx, request.slot_ids
        )
        destination_slots = (
            request.slot_ids
            if request.write_slot_ids is None
            else request.write_slot_ids
        )
        if len(destination_slots) != num_requests:
            raise RuntimeError("GDN recurrent destination count mismatch")
        state_cache.require_allocated_slots(destination_slots)
        if destination_slots != request.slot_ids and state_view.uses_compact_state:
            raise RuntimeError(
                "copyless GDN recurrent decode requires stable source state"
            )
        state_in = state_view.state
        state_pool = state_cache.recurrent_states[request.cache_idx]
        slot_ids_arr = state_view.cache_slot_ids
        write_slot_ids_arr = (
            slot_ids_arr
            if destination_slots == request.slot_ids
            else mx.array(destination_slots, dtype=mx.int32)
        )
        state_slot_ids_arr = state_view.state_slot_ids
''',
        "recurrent decode destination setup",
    )

    text = replace_once(
        text,
        "        state_pool[slot_ids_arr] = state_updates\n",
        "        state_pool[write_slot_ids_arr] = state_updates\n",
        "recurrent decode destination scatter",
    )

    path.write_text(text)


def patch_linear_wrapper(path: Path) -> None:
    text = path.read_text()

    text = replace_once(
        text,
        '''        cache_idx = self._gdn_cache_idx
        pool = self._gdn_state_cache.conv_states[cache_idx]
        src = mx.array(rollback_slots, dtype=mx.int32)
        dst = mx.array(speculative_slots, dtype=mx.int32)
        pool[dst] = pool[src]
        self._gdn_state_cache.store_conv_state(cache_idx, pool)

        second = self._gdn_lazy.try_conv_decode(
            mixed_qkv[:, 1::2, :],
            self._inner,
            self._gdn_state_cache,
            cache_idx,
            speculative_slots,
        )
''',
        '''        if isinstance(self._gdn_lazy, GDNLazyKernels):
            second = self._gdn_lazy.try_conv_decode(
                mixed_qkv[:, 1::2, :],
                self._inner,
                self._gdn_state_cache,
                self._gdn_cache_idx,
                rollback_slots,
                write_slot_ids=speculative_slots,
            )
        else:
            # Preserve the old shape for test doubles and external wrappers.
            cache_idx = self._gdn_cache_idx
            pool = self._gdn_state_cache.conv_states[cache_idx]
            src = mx.array(rollback_slots, dtype=mx.int32)
            dst = mx.array(speculative_slots, dtype=mx.int32)
            pool[dst] = pool[src]
            self._gdn_state_cache.store_conv_state(cache_idx, pool)
            second = self._gdn_lazy.try_conv_decode(
                mixed_qkv[:, 1::2, :],
                self._inner,
                self._gdn_state_cache,
                cache_idx,
                speculative_slots,
            )
''',
        "copyless conv verification",
    )

    text = replace_once(
        text,
        '''        def request(rows: slice, slot_ids: list[int]) -> GDNRecurrentDecodeRequest:
            return GDNRecurrentDecodeRequest(
                q=q[:, rows],
                k=k[:, rows],
                v=v[:, rows],
                g=g[:, rows],
                beta=beta[:, rows],
                state_cache=self._gdn_state_cache,
                cache_idx=self._gdn_cache_idx,
                slot_ids=slot_ids,
                output_dtype=state.x.dtype,
                threadgroup_dv=self._recurrent_decode_threadgroup_dv(),
            )
''',
        '''        def request(
            rows: slice,
            slot_ids: list[int],
            write_slot_ids: list[int] | None = None,
        ) -> GDNRecurrentDecodeRequest:
            return GDNRecurrentDecodeRequest(
                q=q[:, rows],
                k=k[:, rows],
                v=v[:, rows],
                g=g[:, rows],
                beta=beta[:, rows],
                state_cache=self._gdn_state_cache,
                cache_idx=self._gdn_cache_idx,
                slot_ids=slot_ids,
                output_dtype=state.x.dtype,
                threadgroup_dv=self._recurrent_decode_threadgroup_dv(),
                write_slot_ids=write_slot_ids,
            )
''',
        "recurrent request destination slots",
    )

    text = replace_once(
        text,
        '''        cache_idx = self._gdn_cache_idx
        pool = self._gdn_state_cache.recurrent_states[cache_idx]
        src = mx.array(rollback_slots, dtype=mx.int32)
        dst = mx.array(speculative_slots, dtype=mx.int32)
        pool[dst] = pool[src]
        self._gdn_state_cache.store_recurrent_state(cache_idx, pool)

        second = self._gdn_lazy.try_recurrent_decode(
            request(slice(1, None, 2), speculative_slots)
        )
''',
        '''        if isinstance(self._gdn_lazy, GDNLazyKernels):
            second = self._gdn_lazy.try_recurrent_decode(
                request(
                    slice(1, None, 2),
                    rollback_slots,
                    write_slot_ids=speculative_slots,
                )
            )
        else:
            # Preserve the old shape for test doubles and external wrappers.
            cache_idx = self._gdn_cache_idx
            pool = self._gdn_state_cache.recurrent_states[cache_idx]
            src = mx.array(rollback_slots, dtype=mx.int32)
            dst = mx.array(speculative_slots, dtype=mx.int32)
            pool[dst] = pool[src]
            self._gdn_state_cache.store_recurrent_state(cache_idx, pool)
            second = self._gdn_lazy.try_recurrent_decode(
                request(slice(1, None, 2), speculative_slots)
            )
''',
        "copyless recurrent verification",
    )

    path.write_text(text)


def main() -> None:
    patch_lazy_kernels(Path("vllm_metal/attention/impls/gdn_lazy.py"))
    patch_linear_wrapper(Path("vllm_metal/attention/impls/linear.py"))
    print("Applied experimental copyless GDN verification patch")


if __name__ == "__main__":
    main()
