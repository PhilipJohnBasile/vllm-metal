# SPDX-License-Identifier: Apache-2.0
"""Proposer seam for Metal speculative decoding.

The model runner owns a single :class:`MetalProposer` and drives drafting
through its uniform :meth:`MetalProposer.propose` call, mirroring vLLM's
polymorphic ``self.drafter``. Gemma4 MTP and draft-model speculative decoding
are interchangeable implementations; the runner holds no per-method knowledge.

The shared *verify* half stays in
:class:`vllm_metal.v1.spec_decode.SpeculativeDecodeController`
(``build_decode_segments`` + ``verify_greedy``); only the *propose* half is
polymorphic here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import mlx.core as mx
from vllm.v1.outputs import DraftTokenIds

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from vllm_metal.v1.model_runner import (
        MetalModelRunner,
        PrefillRequest,
        RequestState,
    )
    from vllm_metal.v1.spec_decode import PagedDecodeSegment


@dataclass(frozen=True, slots=True)
class ProposeContext:
    """Per-step state a proposer may consume to draft the next tokens.

    Carries everything computed during target sampling that a drafter needs.
    Long-lived collaborators (models, caches, the assistant runtime) are held
    by the proposer implementation itself, not here.
    """

    target_hidden_states: mx.array | None
    decode_reqs: Sequence[tuple[str, RequestState]]
    decode_segments: Sequence[PagedDecodeSegment]
    decode_token_ids: Sequence[Sequence[int]]
    prefill_reqs: Sequence[PrefillRequest]
    prefill_token_ids: Sequence[int]
    prefill_result_modes: Sequence[str]
    request_states: Mapping[str, RequestState]
    cu_seqlens: Sequence[int]
    num_decode_segments: int
    num_speculative_tokens: int
    # Request ids the scheduler finished this step. vLLM can hand a finished
    # id straight back out to a new request in the same step, so a proposer
    # that keeps its own per-request state must clear against this, not
    # against absence from request_states (which the new request repopulates
    # under the same id).
    finished_req_ids: set[str]


class MetalProposer(Protocol):
    """Uniform drafting seam."""

    def needs_target_hidden_states(
        self,
        decode_segments: Sequence[PagedDecodeSegment],
        *,
        has_final_prefill: bool,
    ) -> bool:
        """Whether the runner must collect target hidden states for this drafter."""
        ...

    def propose(self, ctx: ProposeContext) -> DraftTokenIds | None:
        """Return per-request draft tokens for the next step, or ``None``."""
        ...

    def release_requests(self, req_ids: set[str]) -> None:
        """Release any per-request drafter state for these evicted/preempted ids.

        Called from the runner's lifecycle reconcile on eviction, preemption, and
        resume. A proposer that pins a bounded per-request resource (draft cache
        blocks) must release it here rather than hold it while the request waits;
        a stateless proposer is a no-op.
        """
        ...


class Gemma4MTPProposer:
    """:class:`MetalProposer` backed by the in-model Gemma4 MTP assistant.

    The assistant is read lazily from the runner: cache setup replaces it with
    a KV-sharing-bound instance (see ``cache_policy.install_gemma4_mtp_kv_sharing``)
    after model load, so capturing it at construction time would pin the
    pre-sharing object.
    """

    def __init__(self, runner: MetalModelRunner) -> None:
        self._runner = runner

    def needs_target_hidden_states(
        self,
        decode_segments: Sequence[PagedDecodeSegment],
        *,
        has_final_prefill: bool,
    ) -> bool:
        # The assistant consumes the previous target step's hidden states for
        # decode and final-prefill rows; intermediate prefill chunks never
        # sample, so they cannot seed a draft.
        return bool(decode_segments) or has_final_prefill

    def release_requests(self, req_ids: set[str]) -> None:
        # The assistant reads the target's paged KV (released by the runtime);
        # the proposer holds no per-request state of its own.
        del req_ids

    def propose(self, ctx: ProposeContext) -> DraftTokenIds | None:
        if ctx.num_speculative_tokens <= 0:
            return None

        runner = self._runner
        assistant = runner._gemma4_mtp_assistant
        if (
            assistant is None
            or not assistant.forward_ready
            or ctx.target_hidden_states is None
        ):
            return None

        seeds = runner._spec_decode_controller.build_gemma4_mtp_draft_seeds(
            decode_reqs=ctx.decode_reqs,
            decode_segments=ctx.decode_segments,
            decode_token_ids=ctx.decode_token_ids,
            prefill_reqs=ctx.prefill_reqs,
            prefill_token_ids=ctx.prefill_token_ids,
            prefill_result_modes=ctx.prefill_result_modes,
            request_states=ctx.request_states,
            cu_seqlens=ctx.cu_seqlens,
            num_decode_segments=ctx.num_decode_segments,
        )
        if not seeds:
            return None

        input_ids = mx.array([[seed.token_id for seed in seeds]], dtype=mx.int32)
        target_input_embeddings = runner._target_input_embeddings(input_ids)
        draft_token_ids = assistant.propose_draft_token_ids(
            seeds=seeds,
            target_hidden_states=ctx.target_hidden_states,
            target_input_embeddings=target_input_embeddings,
        )
        if not draft_token_ids:
            return None

        return DraftTokenIds(
            req_ids=[seed.req_id for seed in seeds],
            draft_token_ids=draft_token_ids,
        )


@dataclass(slots=True)
class _QwenMTPRequestState:
    cache: list[Any]
    pending_hidden: mx.array | None = None


class QwenNativeMTPProposer:
    """One-token native Qwen MTP proposer using the mlx-lm model protocol.

    State is request-local in this phase. Scheduler prefix hits are refused
    until MTP-head KV is represented in scheduler-owned blocks.
    """

    def __init__(self, runner: MetalModelRunner) -> None:
        self._runner = runner
        spec = runner.vllm_config.speculative_config
        if spec is None or spec.method != "mtp":
            raise ValueError("QwenNativeMTPProposer requires method='mtp'")
        width = int(spec.num_speculative_tokens or 0)
        if width != 1:
            raise ValueError(
                "Qwen native MTP on Metal currently supports exactly one "
                f"speculative token; got {width}."
            )
        model = runner._forward_model
        if not bool(getattr(model, "supports_mtp", False)):
            raise ValueError(
                "method='mtp' requires native MTP weights and supports_mtp=True"
            )
        if not callable(getattr(model, "mtp_forward", None)) or not callable(
            getattr(model, "make_mtp_cache", None)
        ):
            raise ValueError(
                "native MTP model must expose mtp_forward() and make_mtp_cache()"
            )
        self._model = model
        self._states: dict[str, _QwenMTPRequestState] = {}
        self._prefix_hit_blocked: set[str] = set()

    def needs_target_hidden_states(
        self,
        decode_segments: Sequence[PagedDecodeSegment],
        *,
        has_final_prefill: bool,
    ) -> bool:
        del decode_segments, has_final_prefill
        return True

    def release_requests(self, req_ids: set[str]) -> None:
        for req_id in req_ids:
            self._states.pop(req_id, None)
            self._prefix_hit_blocked.discard(req_id)

    def _new_state(self) -> _QwenMTPRequestState:
        cache = list(self._model.make_mtp_cache())
        if not cache:
            raise RuntimeError("native Qwen MTP model returned an empty MTP cache")
        return _QwenMTPRequestState(cache=cache)

    def _run_pairs(
        self,
        state: _QwenMTPRequestState,
        hidden_rows: mx.array,
        next_token_ids: Sequence[int],
    ) -> int:
        if hidden_rows.shape[0] != len(next_token_ids):
            raise RuntimeError(
                "Qwen MTP hidden/token pair count mismatch: "
                f"{hidden_rows.shape[0]} != {len(next_token_ids)}"
            )
        if not next_token_ids:
            raise RuntimeError("Qwen MTP forward requires at least one pair")
        logits = self._model.mtp_forward(
            hidden_rows[None],
            mx.array([list(next_token_ids)], dtype=mx.uint32),
            state.cache,
        )
        mx.eval(logits)
        return int(mx.argmax(logits[0, -1], axis=-1).item())

    def _advance_prefill(
        self,
        state: _QwenMTPRequestState,
        hidden_rows: mx.array,
        input_token_ids: Sequence[int],
    ) -> None:
        if hidden_rows.shape[0] != len(input_token_ids):
            raise RuntimeError("Qwen MTP prefill hidden/token length mismatch")
        if not input_token_ids:
            return
        pair_hidden: list[mx.array] = []
        pair_tokens: list[int] = []
        if state.pending_hidden is not None:
            pair_hidden.append(state.pending_hidden)
            pair_tokens.append(int(input_token_ids[0]))
        if len(input_token_ids) > 1:
            pair_hidden.append(hidden_rows[:-1])
            pair_tokens.extend(int(token) for token in input_token_ids[1:])
        if pair_tokens:
            self._run_pairs(
                state,
                mx.concatenate(pair_hidden, axis=0),
                pair_tokens,
            )
        state.pending_hidden = hidden_rows[-1:]

    def _draft_after_prefill_sample(
        self,
        state: _QwenMTPRequestState,
        sampled_token_id: int,
    ) -> int:
        if state.pending_hidden is None:
            raise RuntimeError("Qwen MTP final prefill has no boundary hidden state")
        draft = self._run_pairs(
            state,
            state.pending_hidden,
            [sampled_token_id],
        )
        state.pending_hidden = None
        return draft

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
            count = len(sampled_ids)
            hidden_rows = hidden[segment.start_row : segment.start_row + count]
            draft = self._run_pairs(request_state, hidden_rows, sampled_ids)
            draft_req_ids.append(req_id)
            drafts.append([draft])

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
                if prefill.start_pos > 0:
                    self._prefix_hit_blocked.add(req_id)
                    continue
                request_state = self._new_state()
                self._states[req_id] = request_state
            if req_id in self._prefix_hit_blocked:
                continue

            start = ctx.cu_seqlens[ctx.num_decode_segments + index]
            end = ctx.cu_seqlens[ctx.num_decode_segments + index + 1]
            self._advance_prefill(
                request_state,
                hidden[start:end],
                prefill.token_ids,
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
            draft = self._draft_after_prefill_sample(
                request_state,
                int(sampled_token_id),
            )
            draft_req_ids.append(req_id)
            drafts.append([draft])

        if not drafts:
            return None
        return DraftTokenIds(req_ids=draft_req_ids, draft_token_ids=drafts)
