# Qwen native MTP handoff — August 22, 2026

This document preserves the exact state and continuation procedure for
[vllm-project/vllm-metal#618](https://github.com/vllm-project/vllm-metal/pull/618)
before Philip John Basile's August 26 vacation.

It is a handoff record, not a claim that the current branch is ready to merge.
The live PR remains draft.

## Preserved revisions

- Downstream feature head: `5c1250161a9de28fa5fd734d922a0df604101eb6`
- Downstream archive branch:
  `PhilipJohnBasile/vllm-metal:archive-pr-618-pre-vacation-20260822`
- Closed MLX-LM #1740 source head:
  `bc1d11414c12372cdf76538e09ce1fbc54b7fc7b`
- MLX-LM archive branch:
  `PhilipJohnBasile/mlx-lm:archive-1740-final-20260822`
- Active upstream review path:
  [ml-explore/mlx-lm#990](https://github.com/ml-explore/mlx-lm/pull/990),
  or a review-driven successor or split PR

The dependency is satisfied only when reviewed native Qwen MTP support lands
on `mlx-lm/main`. Do not pin a temporary PR head or a synthetic merge ref.

## #1740 behavior that must survive

AirRunner offered to carry the following work into the live upstream path with
attribution to #1740 and `bc1d114`:

1. Transactional multi-turn prompt-cache reuse, including:
   - finalization at the output-length boundary;
   - finalization when a generator is closed;
   - fail-closed handling for populated target caches without compatible MTP
     boundary state;
   - cached-versus-uncached multi-turn parity; and
   - prompt-cache serialization round trips.
2. Accepted draft tokens must report the target/verifier log-probability
   distribution, not the draft-head distribution.
3. Fail-closed detection for checkpoints without a usable loaded MTP head.
4. Rejection of unsupported stateful `logits_processors`, while retaining
   parity for supported context-sensitive stateless processors.
5. Focused regression coverage for all of the above.

Source record:

- https://github.com/PhilipJohnBasile/mlx-lm/commit/bc1d11414c12372cdf76538e09ce1fbc54b7fc7b

At the inspected #990 head `e8ceecc`, accepted drafts still yielded `draft_lp`.
The transfer should not be considered complete until the landed implementation
uses the target/verifier distribution and tests the behavior.

## Upstream API decision required

The MLX-LM review requested a model-provided `make_draft_model` integrated with
`speculative_generate_step`, rather than coupling a model module to a separate
`mtp_generate_step` loop.

The current #618 implementation consumes these lower-level contracts:

- `supports_mtp`;
- `mtp_forward`;
- pre-final-norm hidden states from `return_hidden=True`; and
- direct access to the trained MTP module for scheduler-owned paged MTP KV.

After upstream lands, #618 must either consume the canonical landed API or add
a narrow documented adapter. Changing only the MLX-LM SHA is insufficient.

## Current downstream divergence

The #618 feature branch was built on
`150dd3292bd940f2eac1b3442ece21355d8ebf19` (#620). Current upstream `main`
at handoff time is `acce6140320fc90482b9fe80d3f4b9573c171595`, nine commits
ahead.

Eight #618 files overlap changes made upstream after its integration base:

| File | Upstream work to preserve | Resolution rule |
| --- | --- | --- |
| `vllm_metal/v1/cache_policy.py` | #630 scheduler-managed draft KV and memory accounting | Keep `_draft_layer_specs`, `_adopt_draft_scheduler_group`, draft scratch reservation, and combined target+draft accounting. Add Qwen's cache-only MTP group alongside them; count Qwen auxiliary storage exactly once. |
| `vllm_metal/v1/model_runner.py` | #630 request `num_computed_tokens`, scheduler-owned draft ingest and updated drafter construction | Retain the new draft-model path. Add `QwenNativeMTPProposer` as a separate native-MTP case, preserve hidden-state capture, and commit verifier-selected GDN state before request lengths and scheduler ownership advance. |
| `vllm_metal/attention/impls/gdn_lazy.py` | #632 removal of per-layer prefill materialization barriers; #634 in-place state row writes | Do not restore `mx.eval` barriers or direct indexed pool assignments. Route speculative destination updates through `write_conv_rows` and `write_recurrent_rows`. |
| `vllm_metal/attention/impls/linear.py` | #632 lazy prefill policy and #634 state-write contract | Preserve upstream lazy behavior while retaining #618's per-token speculative source/destination metadata and checkpoint production. |
| `vllm_metal/attention/context.py` | #623 NAX prefill routing and verification-window exclusions | Preserve NAX routing. Native MTP verification must continue taking the non-NAX verification path while ordinary eligible prefill may use NAX. |
| `vllm_metal/platform.py` | #633 UniProc loopback workaround | Retain the `VLLM_HOST_IP=127.0.0.1` compatibility setup and merge native-MTP validation around it; do not return or raise before the upstream validation order is complete. |
| `tests/test_gdn_lazy_wrapper.py` | #632 lazy/no-materialization expectations | Keep the upstream expectations and layer #618's speculative destination-state cases on top. |
| `tests/test_v1_model_runner_generate.py` | #630 request-state and scheduler-managed draft changes | Update fixtures for `num_computed_tokens` and current drafter setup before restoring Qwen MTP assertions. |

The highest-risk conflicts are `cache_policy.py`, `model_runner.py`,
`gdn_lazy.py`, and `linear.py`. Resolve those semantically; do not choose one
side wholesale.

## Rebase invariants

### Scheduler and cache policy

- Target SDPA, Mamba/GDN, ordinary draft-model KV, and native Qwen MTP KV must
  remain distinct scheduler concepts where required.
- #630's committed draft-model KV group and proposer-local lookahead scratch
  reserve must remain intact.
- Qwen MTP's dedicated cache-only group must be visible to the scheduler and
  assigned the same prefix lineage rules as the target hybrid groups.
- Worker planning and scheduler reporting must agree on block count.
- Qwen MTP KV plus boundary-hidden storage must be counted exactly once.
- TurboQuant KV must continue failing closed for the native Qwen MTP path until
  explicitly supported.

### GDN state

- Ordinary decode retains #620's compact pending recurrent state.
- Multi-request prefill retains #632's barrier-free lazy graph.
- Stable-pool writes retain #634's in-place row-scatter primitive.
- Speculative verification may read one slot and write a different checkpoint
  slot.
- Verifier-selected promotion must materialize before scheduler block reuse,
  eviction, preemption, or copy.
- Concurrency 2+ speculative convolution continues using the exact path until a
  separately validated fast path replaces it.

### Generation semantics

- The trained Qwen head remains limited to one speculative token unless a
  deeper path demonstrates a measured net benefit.
- Greedy output parity must remain exact in the qualified matrix.
- Stochastic acceptance must preserve the target distribution.
- Accepted-token log probabilities come from the target/verifier distribution.
- Prefix reuse must fail closed when MTP boundary metadata is absent or stale.

## Required pin updates after upstream landing

Update both downstream references to the exact commit present on
`ml-explore/mlx-lm:main`:

1. `pyproject.toml`
   - replace `254d153fdeb6f150edd4fc5a54f9828638481fa8`;
2. `.github/workflows/qwen-mtp-serving-bench.yml`
   - replace `ac6aaffd8fdfb8c8e713e17f155d83e3d72b0a0f`;
   - replace the temporary `PhilipJohnBasile/mlx-lm` install URL with
     `ml-explore/mlx-lm`.

## Recommended continuation order

1. Record the exact MLX-LM commit that landed on `main`.
2. Verify the #1740 carry-forward checklist against that commit and its tests.
3. Decide the #618 adapter for the final MLX-LM API.
4. Rebase the #618 feature commits onto the latest vLLM Metal `main`.
5. Resolve `cache_policy.py` and `model_runner.py` against #630 first.
6. Resolve `gdn_lazy.py` and `linear.py` against #632 and #634.
7. Resolve `attention/context.py` against #623 and `platform.py` against #633.
8. Merge the two overlapping test files, then restore the remaining #618 tests.
9. Update both MLX-LM pins and the benchmark repository URL.
10. Run focused tests, the repository-wide non-slow suite, and physical M5 Max
    qualification.
11. Squash to one signed-off commit and mark #618 ready only after all gates
    pass.

## Final validation matrix

At minimum, rerun:

- native Qwen MTP proposer, paged-cache, worker-budget, wrapper-metadata, and
  model-runner tests;
- current GDN lazy-kernel, in-place scatter, wrapper, and align-state tests;
- current scheduler-managed draft-cache tests from #630;
- NAX enabled and disabled prefill/continuation smoke tests;
- repository-wide Ruff, formatting, mypy, `git diff --check`, native extension
  and Metal shader builds, tensor bridge, and non-slow tests;
- physical M5 Max four-request exact-parity gates at 48 and 128 output tokens;
- clean cancellation, shutdown, and listener-release checks; and
- matched baseline versus native-MTP serving measurements without making a
  positive acceleration claim unless the complete downstream path wins.

## Vacation authorization

Philip may be off-grid for approximately two weeks beginning August 26, 2026.
AirRunner and maintainers may continue the upstream review and carry-forward
work without waiting for a response. The downstream PR must remain draft until
the landed upstream API is known and the rebase and validation gates above are
complete.
