# PR #618 current-main integration report — August 22, 2026

This report replaces the earlier placeholder merge report. It records the exact
state of the isolated continuation branch for
[vllm-project/vllm-metal#618](https://github.com/vllm-project/vllm-metal/pull/618).

## Scope and terminology

This is an **exact change-set and blob-identity report**, not a claim that a
remote `git merge` produced eight textual conflicts. GitHub App-generated writes
do not emit the Actions events needed to run the temporary merge workflow, so
no fabricated merge transcript is presented.

The unresolved set below is the exact intersection of:

1. files changed by #618; and
2. files changed on upstream `main` after #618's original integration base.

Every file is pinned by its upstream and #618 Git blob IDs. Some may merge
textually without conflict, but all require semantic review because choosing one
side wholesale would discard live work.

## Revisions

- Original #618 integration base: `150dd3292bd940f2eac1b3442ece21355d8ebf19`
  (`#620`)
- Preserved #618 head: `5c1250161a9de28fa5fd734d922a0df604101eb6`
- Current upstream `main`: `211a1e4bc976cd6c0c961cad8da59d649ea9bd65`
  (`#636`)
- Upstream commits after the original integration base: **11**
- Isolated continuation branch:
  `PhilipJohnBasile/vllm-metal:integration/pr-618-main-20260822`
- Integration branch head immediately before this report:
  `14df9c57fe672ccf4ba1751d830036b5714fd9a0`

The live #618 branch was not rewritten.

## Progress

- **14 / 22 files transplanted** as exact #618 blobs onto current upstream
  `main`.
- **8 / 22 files remain pending semantic integration**.
- No dependency pin was changed.
- No readiness, mergeability, performance, or passing-CI claim is made for the
  continuation branch.

## Exact clean transplants

These paths were not modified upstream after `150dd329`, so the continuation
branch uses the exact blob from #618.

| Path | #618 blob |
|---|---|
| `.github/workflows/qwen-mtp-serving-bench.yml` | `9a2a29f74a352b2a4d4a4ff0412ff478415c79e5` |
| `benchmarks/qwen_mtp_speed_matrix.py` | `9f21c7f3dec72548a52017b7336985b4b4c9c71a` |
| `benchmarks/results/qwen_mtp_copyless_promotion_20260817.md` | `c139f5a3a53bbbab893cb433e29d739d0d646564` |
| `tests/attention/test_align_gdn_state_manager.py` | `12bddf505a7ee8730882e9cfdcce438d74ede6b5` |
| `tests/test_qwen_mtp_paged_cache.py` | `4c23ae1c4f9c75696ae3ea577e247994cf29ab48` |
| `tests/test_qwen_mtp_worker_budget.py` | `cace88eaf011ff222363bb0127d32d863b264239` |
| `tests/test_qwen_mtp_wrapper_metadata.py` | `0a5ce010ccd1d433220b3afe7bd829b1e94765b8` |
| `tests/test_qwen_native_mtp_proposer.py` | `2781db17abf098a8a41863c486001f50e2656b93` |
| `vllm_metal/attention/runtime/hybrid.py` | `c04a9cf23181216586b317aa1ea672aba262fc42` |
| `vllm_metal/attention/state/align.py` | `437f20945fd6eb11152b6e43833306e3d8005912` |
| `vllm_metal/v1/model_adapter.py` | `85ae0637d34eeea9d59b24988bffcffeaaf542c6` |
| `vllm_metal/v1/proposer.py` | `7b3fc43a3020d273abfc27b46ce53b529562198f` |
| `vllm_metal/v1/qwen_mtp_paged.py` | `42cb88ca95612d6047fcf86e0053651de7f62612` |
| `vllm_metal/v1/spec_decode.py` | `a988cb84cf3b4962abb90a6b909e4a85852ff7a5` |

## Pending semantic integration

The continuation branch intentionally retains the current-main blob for each
path below until a combined implementation is reviewed.

| Path | Current-main blob | #618 blob | Required preservation |
|---|---|---|---|
| `tests/test_gdn_lazy_wrapper.py` | `b9edddc333621e3d1ea5d25ac01d23fda0f48914` | `65b14e3aa9b05f2e78cd7c7e6131afb8dbc93bbd` | Keep upstream lazy-prefill/in-place-scatter expectations and add #618 speculative destination coverage. |
| `tests/test_v1_model_runner_generate.py` | `c3658280884a14efefca7ffee8478918412c7c1e` | `06103a3e5c5031643cf6a80de6bce68e5c4019e2` | Keep current draft-model and sampling regressions; add Qwen native-MTP hidden-state and commit-path coverage. |
| `vllm_metal/attention/context.py` | `f7616981ccd6b861299ec8500e7231978fa2f85c` | `1ba4c7ae0ad2d602ed129d23d51a5f472dfa9da2` | Preserve #623 NAX routing while retaining the verification-window layout required by hybrid speculative state checkpoints. |
| `vllm_metal/attention/impls/gdn_lazy.py` | `a65cfcda75308d27e4e16d0f9172ad3ff24c2da6` | `566fcdfed0acaa4327204c20238fc2a2a640cca7` | Preserve #632 lazy prefill and #634 in-place row scatter; route #618 source/destination writes through `write_conv_rows` / `write_recurrent_rows`, never direct pool assignment. |
| `vllm_metal/attention/impls/linear.py` | `50f8bb9cd6c60c591ab45556075e48c20a1c5c19` | `97e110b955d64731b64582c14abb4ecfc94cca96` | Preserve removal of per-layer materialization barriers while threading speculative source/destination metadata into the lazy kernels. |
| `vllm_metal/platform.py` | `cfaebbb7ed5c811c088a4008957a7643a6c6d079` | `366a086541b800bca2e2d259058e741af6b17a8b` | Keep #633 UniProc loopback behavior and add only the #618 native-Qwen-MTP configuration validation. |
| `vllm_metal/v1/cache_policy.py` | `1692a59dce58e94dad3443d1d1bbf00de27f767f` | `72aed78f5f0c859f67e2b333864a22ced8169974` | Extend #630's scheduler-managed cache design with the Qwen MTP cache-only group and auxiliary-byte accounting; do not create a parallel private policy or double-count blocks. |
| `vllm_metal/v1/model_runner.py` | `2affefda8c94c0311ea0cf90706be650b9822d67` | `2c18a5b26a28e9c777590261876d2595e0184469` | Preserve current draft-model lifecycle and sampling behavior; add `QwenNativeMTPProposer`, target hidden capture, and verifier-selected GDN state promotion through the existing proposer seam. |

## Upstream work that must not regress

The combined implementation must retain all current-main behavior, including:

- `#623`: M5 NAX prefill routing and its verification-window exclusions;
- `#630`: scheduler-managed and budgeted committed draft-model KV;
- `#632`: no per-layer GDN prefill materialization barriers;
- `#633`: loopback default for single-process rendezvous;
- `#634`: aliased in-place GDN row-scatter helpers; and
- `#636`: validated speculative-KV reuse for accepted draft-model tokens.

`#636` is not one of the eight direct file overlaps above, but its draft-model
semantics are part of current main and must remain undisturbed.

## Resolution order

1. Resolve `attention/context.py` and `platform.py` first. They are narrow and
   establish routing and startup invariants.
2. Resolve `gdn_lazy.py` and `linear.py` together, then merge
   `test_gdn_lazy_wrapper.py`. The implementation must have no new `mx.eval`
   barrier and no direct indexed writes to aliased state pools.
3. Resolve `cache_policy.py` before `model_runner.py`, because the runner must
   consume the final scheduler group and memory-accounting contract.
4. Merge `test_v1_model_runner_generate.py` only after the runtime API is
   settled.
5. Run the checker in `tools/check_pr618_integration_state.py` throughout the
   process. It rejects blind replacement of an overlap file with the old #618
   blob.

## Validation gates after all eight files are resolved

1. `python tools/check_pr618_integration_state.py --require-resolved`
2. `git diff --check`
3. Ruff lint and formatting
4. mypy
5. Focused Qwen MTP, GDN, cache-policy, model-runner, and tensor-bridge tests
6. Repository-wide non-slow suite
7. Physical M5 Max four-request parity at 48 and 128 generated tokens
8. Update both MLX-LM pins only after the reviewed implementation lands on
   `mlx-lm/main`

The integration branch remains a work branch. It is not a replacement for the
live draft PR until every gate above is satisfied.
