from __future__ import annotations

import textwrap
from pathlib import Path


def class_methods(source: str) -> str:
    return textwrap.indent(textwrap.dedent(source).lstrip("\n"), "    ")


def patch_linear() -> None:
    path = Path("vllm_metal/attention/impls/linear.py")
    text = path.read_text()

    conv_marker = '''    def _run_conv_state_chains(
        self, mixed_qkv: mx.array, state: _GDNForwardState
    ) -> mx.array:
'''
    if conv_marker not in text:
        raise RuntimeError("conv state-chain insertion point not found")

    conv_helpers = class_methods(
        '''
        def _one_draft_state_chain_slots(
            self, state: _GDNForwardState
        ) -> tuple[list[int], list[int]] | None:
            """Return rollback/final slots for a pure one-draft verify batch.

            A one-draft chain is ``[initial, after_confirmed, after_draft]``.
            Align mode intentionally aliases ``initial`` and ``after_confirmed``;
            the speculative slot is distinct. With merged verify windows every
            request contributes exactly two packed rows, so the two token
            positions can be run as two batch-wide decode launches.
            """
            chains = state.state_chains
            if (
                not self._lazy_kernels_enabled()
                or chains is None
                or not chains
                or state.num_decode_requests != state.num_requests
                or state.total_tokens != 2 * state.num_requests
            ):
                return None

            rollback_slots: list[int] = []
            speculative_slots: list[int] = []
            for req_idx, chain in enumerate(chains):
                start = state.cu_seqlens[req_idx]
                end = state.cu_seqlens[req_idx + 1]
                if (
                    end - start != 2
                    or len(chain) != 3
                    or chain[0] != chain[1]
                    or chain[2] == chain[1]
                ):
                    return None
                rollback_slots.append(chain[0])
                speculative_slots.append(chain[2])

            if (
                len(set(rollback_slots)) != len(rollback_slots)
                or len(set(speculative_slots)) != len(speculative_slots)
                or set(rollback_slots) & set(speculative_slots)
            ):
                return None
            return rollback_slots, speculative_slots

        def _try_run_conv_one_draft_state_chains(
            self, mixed_qkv: mx.array, state: _GDNForwardState
        ) -> mx.array | None:
            slots = self._one_draft_state_chain_slots(state)
            if slots is None:
                return None
            rollback_slots, speculative_slots = slots

            first = self._gdn_lazy.try_conv_decode(
                mixed_qkv[:, 0::2, :],
                self._inner,
                self._gdn_state_cache,
                self._gdn_cache_idx,
                rollback_slots,
            )
            if first is None:
                return None

            # The first launch leaves the exact post-confirmed state in the
            # rollback slots. Seed the speculative slots from that state, then
            # advance every draft token together. This matches mlx-lm's
            # n_confirmed=1 rollback semantics without per-request Python loops.
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
            if second is None:
                raise RuntimeError(
                    "one-draft GDN conv fast path became ineligible mid-step"
                )

            return mx.stack([first[0], second[0]], axis=1).reshape(
                1, state.total_tokens, self._inner.conv_dim
            )

        '''
    )
    text = text.replace(conv_marker, conv_helpers + conv_marker, 1)

    conv_old = '''        """Produce an observable conv-state checkpoint after every token.

        This is the correctness-first path for speculative verification. It is
        intentionally request/token sequential; ordinary decode and prefill
        remain on their existing fused/lazy kernels.
        """
        inner = self._inner
'''
    conv_new = '''        """Produce an observable conv-state checkpoint after every token.

        One-draft pure-decode batches use two fused batch-wide decode launches;
        wider or mixed verification shapes retain the correctness-first
        request/token-sequential fallback below.
        """
        fast_path = self._try_run_conv_one_draft_state_chains(mixed_qkv, state)
        if fast_path is not None:
            return fast_path

        inner = self._inner
'''
    if conv_old not in text:
        raise RuntimeError("conv state-chain body marker not found")
    text = text.replace(conv_old, conv_new, 1)

    recurrent_marker = '''    def _run_recurrent_state_chains(
        self,
        q: mx.array,
'''
    if recurrent_marker not in text:
        raise RuntimeError("recurrent state-chain insertion point not found")

    recurrent_helper = class_methods(
        '''
        def _try_run_recurrent_one_draft_state_chains(
            self,
            q: mx.array,
            k: mx.array,
            v: mx.array,
            g: mx.array,
            beta: mx.array,
            state: _GDNForwardState,
        ) -> mx.array | None:
            slots = self._one_draft_state_chain_slots(state)
            if slots is None:
                return None
            rollback_slots, speculative_slots = slots

            def request(rows: slice, slot_ids: list[int]) -> GDNRecurrentDecodeRequest:
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

            first = self._gdn_lazy.try_recurrent_decode(
                request(slice(0, None, 2), rollback_slots)
            )
            if first is None:
                return None

            cache_idx = self._gdn_cache_idx
            pool = self._gdn_state_cache.recurrent_states[cache_idx]
            src = mx.array(rollback_slots, dtype=mx.int32)
            dst = mx.array(speculative_slots, dtype=mx.int32)
            pool[dst] = pool[src]
            self._gdn_state_cache.store_recurrent_state(cache_idx, pool)

            second = self._gdn_lazy.try_recurrent_decode(
                request(slice(1, None, 2), speculative_slots)
            )
            if second is None:
                raise RuntimeError(
                    "one-draft GDN recurrent fast path became ineligible mid-step"
                )

            return mx.stack([first, second], axis=1).reshape(
                state.total_tokens, v.shape[2], v.shape[3]
            )

        '''
    )
    text = text.replace(recurrent_marker, recurrent_helper + recurrent_marker, 1)

    recurrent_old = '''        """Produce one recurrent-state snapshot per verification token."""
        state_cache = self._gdn_state_cache
'''
    recurrent_new = '''        """Produce one recurrent-state snapshot per verification token."""
        fast_path = self._try_run_recurrent_one_draft_state_chains(
            q, k, v, g, beta, state
        )
        if fast_path is not None:
            return fast_path

        state_cache = self._gdn_state_cache
'''
    if recurrent_old not in text:
        raise RuntimeError("recurrent state-chain body marker not found")
    text = text.replace(recurrent_old, recurrent_new, 1)
    path.write_text(text)


def patch_tests() -> None:
    path = Path("tests/test_gdn_lazy_wrapper.py")
    text = path.read_text().rstrip()
    test = class_methods(
        '''
        def test_one_draft_chain_uses_two_batch_wide_lazy_passes(self) -> None:
            class FakeLazy:
                enabled = True

                def __init__(self) -> None:
                    self.conv_slots: list[list[int]] = []
                    self.recurrent_slots: list[list[int]] = []

                def try_conv_decode(
                    self, mixed_qkv, _inner, _cache, _cache_idx, slot_ids
                ):
                    self.conv_slots.append(list(slot_ids))
                    return mixed_qkv

                def try_recurrent_decode(self, request):
                    self.recurrent_slots.append(list(request.slot_ids))
                    return mx.zeros(
                        (
                            request.total_tokens,
                            request.num_value_heads,
                            request.value_head_dim,
                        ),
                        dtype=request.output_dtype,
                    )

            inner = _TinyGDNInner()
            cache = _make_state_cache(
                max_seqs=4,
                conv_kernel_dim=inner.conv_kernel_size,
                conv_dim=inner.conv_dim,
                num_v_heads=inner.num_v_heads,
                value_head_dim=inner.head_v_dim,
                key_head_dim=inner.head_k_dim,
            )
            wrapper = GDNPagedAttentionWrapper(
                inner, layer_idx=0, cache_idx=0, state_cache=cache
            )
            fake = FakeLazy()
            object.__setattr__(wrapper, "_gdn_lazy", fake)
            state = attention_linear._GDNForwardState(
                x=mx.zeros((1, 4, inner.conv_dim), dtype=mx.float32),
                cu_seqlens=[0, 2, 4],
                num_requests=2,
                total_tokens=4,
                slot_ids=[1, 3],
                num_decode_requests=2,
                state_chains=[[0, 0, 1], [2, 2, 3]],
            )

            conv = wrapper._run_conv_state_chains(state.x, state)
            assert conv.shape == (1, 4, inner.conv_dim)
            assert fake.conv_slots == [[0, 2], [1, 3]]

            q = mx.zeros((1, 4, 1, 32), dtype=mx.float32)
            v = mx.zeros((1, 4, 1, 4), dtype=mx.float32)
            gates = mx.zeros((1, 4, 1), dtype=mx.float32)
            recurrent = wrapper._run_recurrent_state_chains(
                q, q, v, gates, gates, state
            )
            assert recurrent.shape == (4, 1, 4)
            assert fake.recurrent_slots == [[0, 2], [1, 3]]
        '''
    )
    anchor = "class TestGDNPagedAttentionWrapperLazyKernels:"
    if anchor not in text:
        raise RuntimeError("GDN test class anchor not found")
    text = text.replace(anchor, test + "\n\n\n" + anchor, 1)
    path.write_text(text + "\n")


def main() -> None:
    patch_linear()
    patch_tests()
    print("Applied one-draft batched GDN verification fast path")


if __name__ == "__main__":
    main()
