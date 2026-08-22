# Qwen native MTP handoff addendum — August 22, 2026 13:28 UTC

This addendum records upstream movement after the original #618 handoff was
frozen. It does not modify the live #618 feature branch or change its readiness
state.

## New vLLM Metal main tip

`vllm-project/vllm-metal:main` advanced to:

- `211a1e4bc976cd6c0c961cad8da59d649ea9bd65`
- PR `#636`: skip re-ingesting accepted draft tokens whose KV the lookahead
  already wrote

The original #618 integration base remains:

- `150dd3292bd940f2eac1b3442ece21355d8ebf19` (`#620`)

The base-to-main distance is now **11 commits**.

## Effect on the #618 rebase plan

#636 does not add another direct path collision with the 22 files changed by
#618. Its implementation is concentrated in the separate draft-model proposer,
its tests, and speculative-decoding documentation.

It does strengthen the architectural direction established by #630:

- committed draft KV is scheduler managed;
- speculative lookahead KV has explicit validity and physical-block lineage;
- reuse must be validated against both committed tokens and scheduler block
  mappings;
- stale, rejected, scratch-block, or reallocated state falls back rather than
  being trusted.

Those invariants match #618's fail-closed scheduler-owned Qwen MTP goals, but
they should be implemented through the canonical landed proposer/cache
interfaces rather than by preserving the old branch structure mechanically.

## Updated semantic priorities

The highest-risk direct merges remain:

1. `vllm_metal/v1/cache_policy.py` and `vllm_metal/v1/model_runner.py`
   against scheduler-managed speculative-cache work from #630;
2. `vllm_metal/attention/impls/gdn_lazy.py` and
   `vllm_metal/attention/impls/linear.py` against #632 and #634;
3. `vllm_metal/attention/context.py` against #623;
4. `vllm_metal/platform.py` against #633.

#636 is an additional **design constraint**, not a new direct text conflict:
Qwen native MTP continuation must not invent a parallel cache-validity model
that bypasses the scheduler's committed-token and physical-block lineage.

## Rebase acceptance addition

After the landed MLX-LM API is known, the #618 rehearsal and final rebase should
also confirm:

- accepted speculative KV is reused only when token identity and block lineage
  remain valid;
- rejection, scratch-block placement, reallocation, preemption, cancellation,
  and request-id reuse all fail closed;
- no optimization can skip the final committed row needed to produce the next
  draft;
- the Qwen MTP cache group remains included in the scheduler and worker memory
  budget.

No rebase or pin update should target this addendum's timestamped main SHA as a
permanent dependency. The final integration must use the then-current upstream
main and rerun the full qualification.
