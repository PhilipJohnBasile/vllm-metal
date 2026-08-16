# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest
from vllm.sampling_params import SamplingParams

from vllm_metal.v1.model_adapter import DefaultModelAdapter
from vllm_metal.v1.proposer import ProposeContext, QwenNativeMTPProposer
from vllm_metal.v1.spec_decode import PagedDecodeSegment

VOCAB = 32


class _FakeMTPCache:
    def __init__(self) -> None:
        self.seen: list[int] = []


class _FakeNativeMTPModel:
    supports_mtp = True

    def __init__(self) -> None:
        self.created: list[_FakeMTPCache] = []

    def make_mtp_cache(self):
        cache = _FakeMTPCache()
        self.created.append(cache)
        return [cache]

    def mtp_forward(self, hidden, next_token_ids, cache):
        del hidden
        tokens = [int(token) for token in next_token_ids[0].tolist()]
        cache[0].seen.extend(tokens)
        predicted = (next_token_ids.astype(mx.int32) + 1) % VOCAB
        return mx.eye(VOCAB, dtype=mx.float32)[predicted] * 100.0

    def __call__(self, input_ids, *, cache=None, return_hidden=False):
        del cache
        hidden = mx.repeat(input_ids[..., None].astype(mx.float32), 4, axis=-1)
        target = (input_ids.astype(mx.int32) + 5) % VOCAB
        logits = mx.eye(VOCAB, dtype=mx.float32)[target] * 100.0
        if return_hidden:
            return logits, hidden
        return logits


class _Controller:
    @staticmethod
    def can_draft_greedy(req_id, state):
        del req_id, state
        return True


def _runner(model=None, width=1):
    model = model or _FakeNativeMTPModel()
    return SimpleNamespace(
        _forward_model=model,
        vllm_config=SimpleNamespace(
            speculative_config=SimpleNamespace(
                method="mtp",
                num_speculative_tokens=width,
            )
        ),
        _spec_decode_controller=_Controller(),
    )


def _ctx(
    *,
    hidden,
    decode_reqs=(),
    decode_segments=(),
    decode_token_ids=(),
    prefill_reqs=(),
    prefill_token_ids=(),
    prefill_result_modes=(),
    request_states=None,
    cu_seqlens=(0,),
    finished_req_ids=None,
):
    return ProposeContext(
        target_hidden_states=hidden,
        decode_reqs=decode_reqs,
        decode_segments=decode_segments,
        decode_token_ids=decode_token_ids,
        prefill_reqs=prefill_reqs,
        prefill_token_ids=prefill_token_ids,
        prefill_result_modes=prefill_result_modes,
        request_states=request_states or {},
        cu_seqlens=cu_seqlens,
        num_decode_segments=len(decode_segments),
        num_speculative_tokens=1,
        finished_req_ids=finished_req_ids or set(),
    )


def _hidden(*token_ids):
    values = mx.array(token_ids, dtype=mx.float32)
    return mx.repeat(values[:, None], 4, axis=-1)


class TestQwenTargetHiddenContract:
    def test_adapter_uses_model_logits_and_pre_norm_hidden(self) -> None:
        model = _FakeNativeMTPModel()
        output = DefaultModelAdapter().target_forward(
            model,
            mx.array([[2, 3]], dtype=mx.int32),
            collect_hidden_states=True,
        )
        assert mx.argmax(output.logits[0, 0]).item() == 7
        assert output.hidden_states is not None
        assert output.hidden_states.shape == (2, 4)
        assert output.hidden_states[0, 0].item() == 2.0


class TestQwenNativeMTPProposer:
    def test_requires_the_trained_one_token_width(self) -> None:
        with pytest.raises(ValueError, match="exactly one speculative token"):
            QwenNativeMTPProposer(_runner(width=3))

    def test_collects_hidden_states_for_intermediate_prefill(self) -> None:
        proposer = QwenNativeMTPProposer(_runner())
        assert proposer.needs_target_hidden_states([], has_final_prefill=False)

    def test_chunked_prefill_decode_and_release_are_transactional(self) -> None:
        model = _FakeNativeMTPModel()
        proposer = QwenNativeMTPProposer(_runner(model=model))
        sampling = SamplingParams(temperature=0)
        state = SimpleNamespace(sampling_params=sampling)

        intermediate = SimpleNamespace(req_id="r0", token_ids=[1, 2, 3], start_pos=0)
        result = proposer.propose(
            _ctx(
                hidden=_hidden(1, 2, 3),
                prefill_reqs=[intermediate],
                prefill_token_ids=[0],
                prefill_result_modes=["intermediate"],
                request_states={"r0": state},
                cu_seqlens=[0, 3],
            )
        )
        assert result is None
        assert model.created[0].seen == [2, 3]

        final = SimpleNamespace(req_id="r0", token_ids=[4, 5], start_pos=3)
        result = proposer.propose(
            _ctx(
                hidden=_hidden(4, 5),
                prefill_reqs=[final],
                prefill_token_ids=[6],
                prefill_result_modes=["new_final"],
                request_states={"r0": state},
                cu_seqlens=[0, 2],
            )
        )
        assert result is not None
        assert result.req_ids == ["r0"]
        assert result.draft_token_ids == [[7]]
        assert model.created[0].seen == [2, 3, 4, 5, 6]

        segment = PagedDecodeSegment(
            req_id="r0",
            input_token_ids=(6, 7),
            start_row=0,
            num_query_tokens=2,
            draft_token_ids=(7,),
            cache_start_pos=5,
            block_ids=((2, 3),),
        )
        result = proposer.propose(
            _ctx(
                hidden=_hidden(6, 7),
                decode_reqs=[("r0", state)],
                decode_segments=[segment],
                decode_token_ids=[[7, 8]],
                request_states={"r0": state},
                cu_seqlens=[0, 2],
            )
        )
        assert result is not None
        assert result.draft_token_ids == [[9]]
        assert model.created[0].seen == [2, 3, 4, 5, 6, 7, 8]

        proposer.release_requests({"r0"})
        assert "r0" not in proposer._states

    def test_scheduler_prefix_hit_is_not_adopted_without_mtp_kv(self) -> None:
        model = _FakeNativeMTPModel()
        proposer = QwenNativeMTPProposer(_runner(model=model))
        state = SimpleNamespace(sampling_params=SamplingParams(temperature=0))
        prefix_hit = SimpleNamespace(req_id="hit", token_ids=[20, 21], start_pos=100)
        result = proposer.propose(
            _ctx(
                hidden=_hidden(20, 21),
                prefill_reqs=[prefix_hit],
                prefill_token_ids=[22],
                prefill_result_modes=["cached_final"],
                request_states={"hit": state},
                cu_seqlens=[0, 2],
            )
        )
        assert result is None
        assert not model.created
        assert "hit" in proposer._prefix_hit_blocked
