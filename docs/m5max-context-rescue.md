# M5 Max long-context rescue program

## Goal

Keep long-running Qwen3.5/3.6/3.8 coding-agent sessions fast on Apple Silicon as context grows, while preserving correctness and delivering the work as small, measured PRs.

This program covers six related problems:

1. first-time prefill of a unique prompt;
2. long-context KV memory-bandwidth cost;
3. cache-memory growth and eviction/restart recompute;
4. inefficient attention, verification, and quantized small-M kernels;
5. frontends mutating otherwise reusable prefixes;
6. the tooling and performance-maturity gap versus CUDA serving.

## Active investigation

The first profiler is now in-tree at `tools/benchmark/gdn_align_cold_scaling.py` and runs through `.github/workflows/gdn-align-cold-scaling.yml`.

It compares cache-off and cache-on in separate processes, submits deterministically unique prompts so no prefix hit can hide the cold-path cost, measures a new probe after progressively larger retained working sets, records TTFT/decode/wall throughput and MLX memory, and traces `GDNPagedStateCache.ensure_capacity()` growth. The purpose is to correlate the all-cold slowdown with the align-mode state pool before changing the ownership model.

## Evidence already in the ecosystem

- Hybrid Qwen prefix caching is functional, but upstream vllm-metal PR #584 measured cache-on all-cold unique-prompt throughput at only 0.38x cache-off at concurrency 1 and 0.80x at concurrency 8. Its CUDA control stayed near parity, identifying an MLX worker/state-layout problem rather than an inherent prefix-caching tax.
- Standard paged-attention split-KV decode in upstream PR #437 improved 8K single-stream output throughput by 17.5%.
- Shared-KV-read speculative verification in upstream PR #534 delivered roughly 37-45% TPOT gains at high concurrency.
- Intermediate chunk body-only prefill in upstream PR #592 improved measured prefill throughput by roughly 8-14%.
- Metal KV offload/persistence in upstream PR #530 demonstrated 8.3x restore-vs-recompute at 32B and 11.1x at 70B, but was closed pending a core vLLM failed-load fix. That blocker has now merged as vllm-project/vllm#49328.
- Native Qwen MTP plus hybrid prefix caching is being integrated in upstream PR #618. A copyless speculative-GDN A/B has already shown a correctness-preserving 6.1% improvement, but the full path still needs substantial overhead reduction.

## Priority order

### P0-A: eliminate the align-mode GDN high-water-pool decode tax

The current MLX state updates can touch or re-materialize the scheduler-indexed state pool through the highest block ID. The target design is:

- compact request-local active GDN state for the hot decode path;
- immutable scheduler-block checkpoints for prefix reuse;
- scatter active state into the checkpoint pool only at block boundaries, admission/eviction boundaries, or explicit snapshot points;
- no full-pool source-to-destination copy for speculative verification;
- scheduler-coherent pool growth and checkpoint persistence.

Acceptance gates:

- cache-on all-cold unique-prompt throughput at least 0.95x cache-off at concurrency 1 and 8;
- preserve the existing shared-prefix TTFT win;
- token/output parity across cold, prefix-hit, chunked-prefill, cancellation, preemption, and speculative accept/reject cases;
- memory use proportional to resident checkpoints plus active requests, not every transient high-water ID operation.

### P0-B: enforce a stable reusable-prefix contract

Add request diagnostics and a canonicalization contract so frontends keep dynamic data out of the reusable prefix:

- deterministic tool/schema ordering;
- stable chat-template serialization;
- dynamic timestamps, request IDs, counters, and volatile metadata moved into the suffix;
- prefix fingerprint returned or logged per request;
- cache-miss reason: cold, token mismatch index, evicted, incompatible cache group, or unsupported state;
- benchmark fixtures that deliberately mutate one frontend field at a time.

Acceptance gate: identical semantic requests from supported frontends produce the same reusable-prefix fingerprint and real cache hits.

### P0-C: accelerate first-time prefill

Build a Qwen-specific prefill profile before adding more machinery. Measure 8K, 32K, 64K, and 128K on dense 27B and a representative MoE across:

- chunk size and max batched tokens;
- quantization format;
- GDN recurrent/conv prefill;
- SDPA prefill;
- lm_head/body-only behavior;
- CPU metadata and synchronization fences.

Then target the dominant kernel with one focused PR. Candidate directions are fused or varlen GDN prefill, fewer materialization barriers, and sparse/attested prefill only when exact correctness and downstream cache ownership are preserved.

Acceptance gate: at least 1.25x cold-prefill improvement on the M5 Max 27B target without reducing decode throughput or cache capacity.

### P1-A: reduce long-KV and speculative-verification bandwidth

Existing split-KV and verify-window work removes repeated KV scans. The next likely hotspot is the target's quantized M=2 verification matmul and MTP-head small-M path.

Work:

- profile M=1 baseline decode versus M=2 verify by layer and operator;
- tune or add small-M quantized QMV/QMM kernels for M5 Max;
- fuse dequantization with matmul or attention where bandwidth wins are measurable;
- tune split-KV partition thresholds by GPU family and context length;
- retain the copyless GDN state optimization if a higher-repetition A/B confirms it.

Acceptance gate: MTP verification cost low enough that a dense 27B model with at least 75% acceptance is net faster than baseline at concurrency 1 and 4.

### P1-B: bound cache growth and add persistence

Rebase or revive the Metal KV-offload work now that vllm-project/vllm#49328 is merged, then extend it from a single uniform KV group to hybrid Qwen transactions:

- SDPA KV groups;
- GDN conv/recurrent checkpoint state;
- MTP-head KV;
- target boundary hidden state and committed length;
- atomic restore or clean miss, never partial admission;
- pageable-memory L2 and optional disk L3 with bounded retention and garbage collection.

Acceptance gate: an evicted or post-restart 32K+ Qwen prefix restores at least 5x faster than recompute with output parity against an equivalent in-process resume.

### P2: close the CUDA maturity gap

- permanent M5 Max benchmark runner and public dashboard;
- cold-prefill, hot-prefix, long-KV decode, MTP, memory-capacity, and multi-turn frontend suites;
- operator-level Metal counters plus end-to-end metrics;
- automatic per-model/hardware selection of block size, batch-token budget, split-KV threshold, TurboQuant KV mode, and MTP enable/disable;
- compact reproducible artifacts attached to every performance PR;
- release and pinning discipline so validated mlx-lm commits reach users without months of manual patching.

## PR sequence

1. Benchmark and trace the GDN align-mode high-water regression.
2. Compact active GDN state plus checkpoint-only scatter.
3. Prefix fingerprint/miss-reason contract and frontend canonicalization fixtures.
4. Revive Metal KV offload and add hybrid transaction groups.
5. M5-Max small-M quantized verify kernel.
6. Cold-prefill hotspot PR.
7. Hardware auto-tuner and continuous benchmark dashboard.

Every PR must carry before/after serving numbers, exact model/quant/hardware/commit metadata, correctness tests, and an explicit rollback switch where appropriate.

## Current dependencies

- ml-explore/mlx-lm#1740: native Qwen MTP lower layer.
- vllm-project/vllm-metal#618: Qwen MTP plus hybrid prefix transaction.
- vllm-project/vllm#49328: merged; unblocks revisiting Metal offload.
