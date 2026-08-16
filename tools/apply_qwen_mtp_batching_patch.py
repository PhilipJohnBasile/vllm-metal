from __future__ import annotations

import textwrap
from pathlib import Path


def class_methods(source: str) -> str:
    return textwrap.indent(textwrap.dedent(source).lstrip("\n"), "    ")


def patch_qwen_mtp_paged() -> None:
    path = Path("vllm_metal/v1/qwen_mtp_paged.py")
    text = path.read_text()
    start = text.index("    def run_pairs(\n")
    end = text.index("    def copy_blocks(", start)
    replacement = class_methods(
        '''
        def _project_mtp_hidden(self, hidden: mx.array) -> mx.array:
            """Project only MTP rows whose vocabulary logits are consumed."""
            args = getattr(self.model, "args", None)
            inner_model = getattr(self.model, "model", None)
            embed_tokens = getattr(inner_model, "embed_tokens", None)
            if bool(getattr(args, "tie_word_embeddings", False)):
                if embed_tokens is None or not callable(
                    getattr(embed_tokens, "as_linear", None)
                ):
                    raise RuntimeError(
                        "Qwen MTP tied output projection is unavailable"
                    )
                return embed_tokens.as_linear(hidden)
            lm_head = getattr(self.model, "lm_head", None)
            if lm_head is None:
                raise RuntimeError("Qwen MTP model has no output projection")
            return lm_head(hidden)

        def run_pairs_batch(
            self,
            *,
            hidden_rows_batch: Sequence[mx.array],
            next_token_ids_batch: Sequence[Sequence[int]],
            block_ids_by_group_batch: Sequence[Sequence[Sequence[int]]],
            start_positions: Sequence[int],
            draft_request_indices: Sequence[int] | None = None,
        ) -> list[int]:
            """Advance independent MTP cache segments in one packed forward.

            Varlen scheduler metadata keeps requests isolated. Only segment ends
            that need a speculative token pay the shared vocabulary projection;
            prefix-maintenance rows update MTP KV without materializing logits.
            """
            self._require_ready()
            num_requests = len(hidden_rows_batch)
            if not (
                num_requests == len(next_token_ids_batch)
                == len(block_ids_by_group_batch)
                == len(start_positions)
            ):
                raise RuntimeError(
                    "Qwen MTP batched request metadata length mismatch"
                )
            if num_requests == 0:
                return []

            if draft_request_indices is None:
                draft_indices = list(range(num_requests))
            else:
                draft_indices = [int(index) for index in draft_request_indices]
            if len(set(draft_indices)) != len(draft_indices) or any(
                index < 0 or index >= num_requests for index in draft_indices
            ):
                raise RuntimeError("Qwen MTP draft request index is out of range")

            ordinal = self.mtp_group_ordinal
            assert self._mtp_block_size is not None
            prefill_requests: list[tuple[list[int], int, int]] = []
            hidden_parts: list[mx.array] = []
            flat_tokens: list[int] = []
            segment_end_rows: list[int] = []
            cursor = 0

            for hidden_rows, next_token_ids, block_ids_by_group, start_pos in zip(
                hidden_rows_batch,
                next_token_ids_batch,
                block_ids_by_group_batch,
                start_positions,
                strict=True,
            ):
                token_ids = [int(token) for token in next_token_ids]
                if hidden_rows.shape[0] != len(token_ids):
                    raise RuntimeError(
                        "Qwen MTP hidden/token pair count mismatch: "
                        f"{hidden_rows.shape[0]} != {len(token_ids)}"
                    )
                if not token_ids:
                    raise RuntimeError("Qwen MTP forward requires at least one pair")
                if ordinal >= len(block_ids_by_group):
                    raise RuntimeError(
                        "Qwen MTP request is missing its scheduler KV group"
                    )
                mtp_blocks = list(block_ids_by_group[ordinal])
                last_pos = int(start_pos) + len(token_ids) - 1
                if last_pos // self._mtp_block_size >= len(mtp_blocks):
                    raise RuntimeError("Qwen MTP scheduler block table is too short")

                prefill_requests.append(
                    (mtp_blocks, len(token_ids), int(start_pos))
                )
                hidden_parts.append(hidden_rows)
                flat_tokens.extend(token_ids)
                cursor += len(token_ids)
                segment_end_rows.append(cursor - 1)

            prepare_unified(
                decode_requests=[],
                prefill_requests=prefill_requests,
                block_size=self._mtp_block_size,
            )
            cache = self._require_mtp_cache()
            try:
                packed_hidden = (
                    hidden_parts[0]
                    if len(hidden_parts) == 1
                    else mx.concatenate(hidden_parts, axis=0)
                )
                mtp_hidden = self.mtp_module(
                    packed_hidden[None],
                    mx.array([flat_tokens], dtype=mx.uint32),
                    self.model.model.embed_tokens,
                    [None] * self.num_layers,
                )
                if draft_indices:
                    end_rows = mx.array(
                        [segment_end_rows[index] for index in draft_indices],
                        dtype=mx.int32,
                    )
                    selected_hidden = mtp_hidden[0, end_rows]
                    selected_logits = self._project_mtp_hidden(selected_hidden)
                    draft_ids = mx.argmax(selected_logits, axis=-1)
                    mx.eval(
                        draft_ids,
                        *cache.key_caches,
                        *cache.value_caches,
                    )
                    return [int(token) for token in draft_ids.tolist()]

                mx.eval(*cache.key_caches, *cache.value_caches)
                return []
            finally:
                clear_context()

        def run_pairs(
            self,
            *,
            hidden_rows: mx.array,
            next_token_ids: Sequence[int],
            block_ids_by_group: Sequence[Sequence[int]],
            start_pos: int,
        ) -> int:
            drafts = self.run_pairs_batch(
                hidden_rows_batch=[hidden_rows],
                next_token_ids_batch=[next_token_ids],
                block_ids_by_group_batch=[block_ids_by_group],
                start_positions=[start_pos],
            )
            if len(drafts) != 1:
                raise RuntimeError(
                    "Qwen MTP single-request draft result is missing"
                )
            return drafts[0]

        '''
    )
    path.write_text(text[:start] + replacement + text[end:])


def patch_hybrid_runtime() -> None:
    path = Path("vllm_metal/attention/runtime/hybrid.py")
    text = path.read_text()
    marker = "    def extend_forward_eval_outputs(self, outputs: list[mx.array]) -> None:\n"
    if marker not in text:
        raise RuntimeError("hybrid runtime insertion point not found")
    addition = class_methods(
        '''
        def qwen_mtp_run_pairs_batch(
            self,
            *,
            hidden_rows_batch: Sequence[mx.array],
            next_token_ids_batch: Sequence[Sequence[int]],
            block_ids_by_group_batch: Sequence[Sequence[Sequence[int]]],
            start_positions: Sequence[int],
            draft_request_indices: Sequence[int] | None = None,
        ) -> list[int]:
            if self._qwen_mtp_state is None:
                raise RuntimeError("Qwen MTP paged state is not installed")
            return self._qwen_mtp_state.run_pairs_batch(
                hidden_rows_batch=hidden_rows_batch,
                next_token_ids_batch=next_token_ids_batch,
                block_ids_by_group_batch=block_ids_by_group_batch,
                start_positions=start_positions,
                draft_request_indices=draft_request_indices,
            )

        '''
    )
    path.write_text(text.replace(marker, addition + marker, 1))


def patch_proposer() -> None:
    path = Path("vllm_metal/v1/proposer.py")
    text = path.read_text()
    class_start = text.index("class QwenNativeMTPProposer")
    start = text.index("    def _run_pairs(\n", class_start)
    end = text.index("    def _advance_prefill(", start)
    helpers = class_methods(
        '''
        def _run_pairs_batch(
            self,
            items: Sequence[
                tuple[
                    _QwenMTPRequestState,
                    mx.array,
                    Sequence[int],
                    Sequence[Sequence[int]],
                    int,
                ]
            ],
            *,
            draft_request_indices: Sequence[int] | None = None,
        ) -> list[int]:
            if not items:
                return []
            for state, hidden_rows, next_token_ids, _, start_pos in items:
                if hidden_rows.shape[0] != len(next_token_ids):
                    raise RuntimeError(
                        "Qwen MTP hidden/token pair count mismatch: "
                        f"{hidden_rows.shape[0]} != {len(next_token_ids)}"
                    )
                if state.next_mtp_position != start_pos:
                    raise RuntimeError(
                        "Qwen MTP logical position mismatch: "
                        f"cache expects {state.next_mtp_position}, "
                        f"caller supplied {start_pos}"
                    )

            drafts = self._runtime().qwen_mtp_run_pairs_batch(
                hidden_rows_batch=[item[1] for item in items],
                next_token_ids_batch=[item[2] for item in items],
                block_ids_by_group_batch=[item[3] for item in items],
                start_positions=[item[4] for item in items],
                draft_request_indices=draft_request_indices,
            )
            expected_drafts = (
                len(items)
                if draft_request_indices is None
                else len(draft_request_indices)
            )
            if len(drafts) != expected_drafts:
                raise RuntimeError(
                    "Qwen MTP batched draft result count mismatch: "
                    f"{len(drafts)} != {expected_drafts}"
                )
            for state, _, next_token_ids, _, _ in items:
                state.next_mtp_position += len(next_token_ids)
            return drafts

        def _run_pairs(
            self,
            state: _QwenMTPRequestState,
            hidden_rows: mx.array,
            next_token_ids: Sequence[int],
            block_ids_by_group: Sequence[Sequence[int]],
            *,
            start_pos: int,
        ) -> int:
            drafts = self._run_pairs_batch(
                [
                    (
                        state,
                        hidden_rows,
                        next_token_ids,
                        block_ids_by_group,
                        start_pos,
                    )
                ]
            )
            return drafts[0]

        '''
    )
    text = text[:start] + helpers + text[end:]

    class_start = text.index("class QwenNativeMTPProposer")
    propose_start = text.index(
        "    def propose(self, ctx: ProposeContext) -> DraftTokenIds | None:\n",
        class_start,
    )
    propose = class_methods(
        '''
        def propose(self, ctx: ProposeContext) -> DraftTokenIds | None:
            if ctx.num_speculative_tokens != 1:
                raise RuntimeError(
                    "Qwen native MTP proposer received a non-one-token runtime width"
                )
            self.release_requests(ctx.finished_req_ids)
            hidden = ctx.target_hidden_states
            if hidden is None:
                return None

            draft_req_ids: list[str] = []
            drafts: list[list[int]] = []
            first_stage: list[
                tuple[
                    _QwenMTPRequestState,
                    mx.array,
                    Sequence[int],
                    Sequence[Sequence[int]],
                    int,
                ]
            ] = []
            decode_indices: list[int] = []
            decode_req_ids: list[str] = []

            for (req_id, state), segment, sampled_ids in zip(
                ctx.decode_reqs,
                ctx.decode_segments,
                ctx.decode_token_ids,
                strict=True,
            ):
                if (
                    not sampled_ids
                    or not self._runner._spec_decode_controller.can_draft_greedy(
                        req_id, state
                    )
                ):
                    continue
                request_state = self._states.get(req_id)
                if request_state is None or req_id in self._prefix_hit_blocked:
                    continue
                if request_state.next_mtp_position != segment.cache_start_pos:
                    raise RuntimeError(
                        f"Qwen MTP request {req_id!r} expected target position "
                        f"{request_state.next_mtp_position}, "
                        f"got {segment.cache_start_pos}"
                    )
                count = len(sampled_ids)
                hidden_rows = hidden[
                    segment.start_row : segment.start_row + count
                ]
                decode_indices.append(len(first_stage))
                decode_req_ids.append(req_id)
                first_stage.append(
                    (
                        request_state,
                        hidden_rows,
                        sampled_ids,
                        segment.block_ids,
                        segment.cache_start_pos,
                    )
                )

            pending_hidden_updates: list[
                tuple[_QwenMTPRequestState, mx.array]
            ] = []
            final_prefills: list[
                tuple[str, _QwenMTPRequestState, int, Sequence[Sequence[int]]]
            ] = []

            for index, (prefill, sampled_token_id, result_mode) in enumerate(
                zip(
                    ctx.prefill_reqs,
                    ctx.prefill_token_ids,
                    ctx.prefill_result_modes,
                    strict=True,
                )
            ):
                req_id = prefill.req_id
                request_state = self._states.get(req_id)
                if request_state is None:
                    request_state = self._new_state(prefill)
                    if request_state is None:
                        continue
                    self._states[req_id] = request_state
                if req_id in self._prefix_hit_blocked:
                    continue

                expected_position = prefill.start_pos - int(
                    request_state.pending_hidden is not None
                )
                if request_state.next_mtp_position != expected_position:
                    raise RuntimeError(
                        f"Qwen MTP prefill {req_id!r} expected MTP position "
                        f"{expected_position}, found "
                        f"{request_state.next_mtp_position}"
                    )

                start = ctx.cu_seqlens[ctx.num_decode_segments + index]
                end = ctx.cu_seqlens[ctx.num_decode_segments + index + 1]
                hidden_rows = hidden[start:end]
                input_token_ids = prefill.token_ids
                if hidden_rows.shape[0] != len(input_token_ids):
                    raise RuntimeError(
                        "Qwen MTP prefill hidden/token length mismatch"
                    )
                if not input_token_ids:
                    continue

                pair_hidden: list[mx.array] = []
                pair_tokens: list[int] = []
                if request_state.pending_hidden is not None:
                    pair_hidden.append(request_state.pending_hidden)
                    pair_tokens.append(int(input_token_ids[0]))
                if len(input_token_ids) > 1:
                    pair_hidden.append(hidden_rows[:-1])
                    pair_tokens.extend(
                        int(token) for token in input_token_ids[1:]
                    )
                if pair_tokens:
                    first_stage.append(
                        (
                            request_state,
                            mx.concatenate(pair_hidden, axis=0),
                            pair_tokens,
                            prefill.block_ids,
                            request_state.next_mtp_position,
                        )
                    )
                pending_hidden_updates.append(
                    (request_state, hidden_rows[-1:])
                )

                if result_mode == "intermediate":
                    continue
                state = ctx.request_states.get(req_id)
                if (
                    state is None
                    or not self._runner._spec_decode_controller.can_draft_greedy(
                        req_id, state
                    )
                ):
                    continue
                final_prefills.append(
                    (
                        req_id,
                        request_state,
                        int(sampled_token_id),
                        prefill.block_ids,
                    )
                )

            decode_drafts = self._run_pairs_batch(
                first_stage,
                draft_request_indices=decode_indices,
            )
            for request_state, pending_hidden in pending_hidden_updates:
                request_state.pending_hidden = pending_hidden
            for req_id, draft in zip(
                decode_req_ids,
                decode_drafts,
                strict=True,
            ):
                draft_req_ids.append(req_id)
                drafts.append([draft])

            final_stage: list[
                tuple[
                    _QwenMTPRequestState,
                    mx.array,
                    Sequence[int],
                    Sequence[Sequence[int]],
                    int,
                ]
            ] = []
            final_req_ids: list[str] = []
            final_states: list[_QwenMTPRequestState] = []
            for req_id, request_state, sampled_token_id, block_ids in final_prefills:
                if request_state.pending_hidden is None:
                    raise RuntimeError(
                        "Qwen MTP final prefill has no boundary hidden state"
                    )
                final_req_ids.append(req_id)
                final_states.append(request_state)
                final_stage.append(
                    (
                        request_state,
                        request_state.pending_hidden,
                        [sampled_token_id],
                        block_ids,
                        request_state.next_mtp_position,
                    )
                )

            final_drafts = self._run_pairs_batch(final_stage)
            for request_state in final_states:
                request_state.pending_hidden = None
            for req_id, draft in zip(
                final_req_ids,
                final_drafts,
                strict=True,
            ):
                draft_req_ids.append(req_id)
                drafts.append([draft])

            if not drafts:
                return None
            return DraftTokenIds(
                req_ids=draft_req_ids,
                draft_token_ids=drafts,
            )
        '''
    )
    path.write_text(text[:propose_start] + propose)


def patch_tests() -> None:
    path = Path("tests/test_qwen_native_mtp_proposer.py")
    text = path.read_text()
    init_old = (
        "        self.boundary_reads: list[int] = []\n"
        "        self.fail_boundary = False\n"
    )
    init_new = (
        "        self.boundary_reads: list[int] = []\n"
        "        self.batch_calls: list[dict[str, object]] = []\n"
        "        self.fail_boundary = False\n"
    )
    if init_old not in text:
        raise RuntimeError("fake runtime init insertion point not found")
    text = text.replace(init_old, init_new, 1)

    controller_marker = "\n\nclass _Controller:"
    if controller_marker not in text:
        raise RuntimeError("fake runtime method insertion point not found")
    fake_batch = class_methods(
        '''
        def qwen_mtp_run_pairs_batch(
            self,
            *,
            hidden_rows_batch,
            next_token_ids_batch,
            block_ids_by_group_batch,
            start_positions,
            draft_request_indices=None,
        ):
            indices = (
                list(range(len(hidden_rows_batch)))
                if draft_request_indices is None
                else list(draft_request_indices)
            )
            self.batch_calls.append(
                {
                    "requests": len(hidden_rows_batch),
                    "draft_request_indices": indices,
                }
            )
            all_drafts = [
                self.qwen_mtp_run_pairs(
                    hidden_rows=hidden_rows,
                    next_token_ids=next_token_ids,
                    block_ids_by_group=block_ids_by_group,
                    start_pos=start_pos,
                )
                for hidden_rows, next_token_ids, block_ids_by_group, start_pos in zip(
                    hidden_rows_batch,
                    next_token_ids_batch,
                    block_ids_by_group_batch,
                    start_positions,
                    strict=True,
                )
            ]
            return [all_drafts[index] for index in indices]
        '''
    )
    text = text.replace(controller_marker, "\n" + fake_batch + controller_marker, 1)

    test_method = class_methods(
        '''
        def test_decode_requests_share_one_mtp_runtime_batch(self) -> None:
            runtime = _FakePagedRuntime()
            proposer = QwenNativeMTPProposer(_runner(runtime=runtime))
            state0 = SimpleNamespace(
                sampling_params=SamplingParams(temperature=0)
            )
            state1 = SimpleNamespace(
                sampling_params=SamplingParams(temperature=0)
            )

            proposer.propose(
                _ctx(
                    hidden=_hidden(1, 2, 11, 12),
                    prefill_reqs=[
                        _prefill("r0", [1, 2], 0),
                        _prefill("r1", [11, 12], 0),
                    ],
                    prefill_token_ids=[0, 0],
                    prefill_result_modes=["intermediate", "intermediate"],
                    request_states={"r0": state0, "r1": state1},
                    cu_seqlens=[0, 2, 4],
                )
            )
            assert runtime.batch_calls[-1] == {
                "requests": 2,
                "draft_request_indices": [],
            }

            proposer.propose(
                _ctx(
                    hidden=_hidden(3, 13),
                    prefill_reqs=[
                        _prefill("r0", [3], 2),
                        _prefill("r1", [13], 2),
                    ],
                    prefill_token_ids=[4, 14],
                    prefill_result_modes=["new_final", "new_final"],
                    request_states={"r0": state0, "r1": state1},
                    cu_seqlens=[0, 1, 2],
                )
            )

            runtime.batch_calls.clear()
            result = proposer.propose(
                _ctx(
                    hidden=_hidden(4, 14),
                    decode_reqs=[("r0", state0), ("r1", state1)],
                    decode_segments=[
                        PagedDecodeSegment(
                            req_id="r0",
                            input_token_ids=(4,),
                            start_row=0,
                            num_query_tokens=1,
                            draft_token_ids=(),
                            cache_start_pos=3,
                            block_ids=((2, 3, 4, 5), (20, 21, 22, 23)),
                        ),
                        PagedDecodeSegment(
                            req_id="r1",
                            input_token_ids=(14,),
                            start_row=1,
                            num_query_tokens=1,
                            draft_token_ids=(),
                            cache_start_pos=3,
                            block_ids=((6, 7, 8, 9), (24, 25, 26, 27)),
                        ),
                    ],
                    decode_token_ids=[[5], [15]],
                    request_states={"r0": state0, "r1": state1},
                    cu_seqlens=[0, 1, 2],
                )
            )
            assert result is not None
            assert result.req_ids == ["r0", "r1"]
            assert result.draft_token_ids == [[6], [16]]
            assert runtime.batch_calls == [
                {"requests": 2, "draft_request_indices": [0, 1]}
            ]
        '''
    )
    text = text.rstrip() + "\n\n" + test_method
    path.write_text(text)


def main() -> None:
    patch_qwen_mtp_paged()
    patch_hybrid_runtime()
    patch_proposer()
    patch_tests()
    print("Applied batched native Qwen MTP proposer optimization")


if __name__ == "__main__":
    main()
