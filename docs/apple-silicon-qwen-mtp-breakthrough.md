# Apple Silicon Qwen MTP: from correctness to acceleration

## Executive summary

Qwen3.5/3.6/3.8 checkpoints include a trained multi-token-prediction (MTP) head, but safely combining that head with hybrid attention, GatedDeltaNet state, paged serving, and reusable prefixes requires more than loading extra weights.

This work built the missing correctness path and then treated performance as an empirical systems problem rather than assuming speculative decoding would automatically be faster.

The first complete hosted serving matrix produced an important negative result: native MTP was correct and functional, but the downstream vLLM Metal proposal/verification/state path was slower than the matched prefix-cached baseline on every workload. That result narrowed the problem from “does Qwen MTP work?” to “which state-motion and orchestration costs dominate Metal serving?”

The next matrix tests two measured code-level interventions—copyless speculative state and deferred compact GDN state—alone and together, with deterministic-output gates and matched baselines.

## Why this problem matters

Long-running local agents are often limited less by peak model throughput than by three compounding costs:

1. repeated prefill of already-seen system, tool, repository, and conversation prefixes;
2. context-dependent KV bandwidth during decode;
3. speculative-serving overhead that can exceed the saved target-model work.

A useful Apple Silicon serving stack therefore needs all of the following at once:

- exact prefix reuse;
- transactional hybrid-cache state;
- low-overhead draft generation and verification;
- bounded state growth;
- hardware-aware small-batch kernels;
- reproducible evidence on real Apple hardware.

## Correctness foundation

The consolidated native-Qwen-MTP implementation provides:

- preservation and loading of the vendor-trained MTP head;
- exact GatedDeltaNet convolution/recurrent-state rollback;
- scheduler-owned speculative state;
- dedicated paged MTP KV state;
- target-boundary hidden-state tracking;
- atomic, fail-closed hybrid prefix-cache admission;
- support for chunked prefill, cancellation, continuation, and warm-prefix reuse.

The combined macOS/Metal validation completed with 1,804 passing tests and zero failures.

## First complete serving matrix

Run: `31966399394`

Configuration:

- model: Qwen3.5-0.8B, MLX affine 4-bit, group size 64;
- hardware: GitHub-hosted virtual Apple M1, 3 cores, 7 GB RAM;
- prefix caching enabled in both arms;
- two timed repetitions per workload/profile;
- matched cold/hot baseline launches;
- scheduler budget, block size, verification-window mode, decode pipeline, GDN path, concurrency, and prompt length varied.

The best complete MTP profile used a 4096-token scheduler budget, block size 16, lazy GDN kernels, and decode pipelining.

| Workload | Baseline output tok/s | Best MTP output tok/s | MTP / baseline | Acceptance |
|---|---:|---:|---:|---:|
| 1,152 prompt / 128 output / concurrency 1 | 30.27 | 10.56 | 0.349x | 29.6% |
| 1,152 prompt / 128 output / concurrency 4 | 56.64 | 18.23 | 0.322x | 28.6% |
| 1,152 prompt / 128 output / concurrency 8 | 61.70 | 18.11 | 0.293x | 28.4% |
| 1,152 prompt / 64 output / concurrency 16 | 79.61 | 15.71 | 0.197x | 28.7% |
| 8,192 prompt / 64 output / concurrency 1 | 14.40 | 6.60 | 0.459x | 42.2% |
| 8,192 prompt / 64 output / concurrency 4 | 36.42 | 9.59 | 0.263x | 42.2% |

The isolated native MLX-LM MTP generator reached 24.61 output tok/s, confirming that model conversion and the trained head were usable. The larger loss appeared in downstream serving orchestration, proposal/verification, and state ownership.

## Bottlenecks isolated

### 1. Full speculative-state copy

The original one-draft verification path copied completed convolution and recurrent GDN state from confirmed slots into speculative slots before advancing the draft.

A copyless A/B instead read confirmed state directly and scattered only the compact resulting draft state into the speculative destination.

Measured result:

- 48 focused tests passed;
- deterministic outputs were bit-identical;
- acceptance remained 29.2%;
- throughput improved from 13.354 to 14.168 output tok/s;
- gain: 6.1%.

This did not close the full gap, but it demonstrated that avoidable state motion was a real cost.

### 2. Scheduler-indexed high-water GDN pools

Under retained-prefix pressure, ordinary one-token decode repeatedly materialized updates into scheduler-block-indexed state arrays whose capacity followed the highest allocated block ID rather than active request count.

A deferred-state prototype kept hot request state compact and wrote scheduler-owned checkpoints only at correctness boundaries.

Two deterministic A/Bs reported geometric-mean improvements of approximately 2.22x and 2.13x under prefix-pressure workloads.

A full-pool-preallocation control was only 0.701x as fast as compact deferred state. That ruled out allocation growth as the sole problem: the steady-state cost of touching a large monolithic pool was itself harmful.

## Breakthrough matrix

The current matrix starts from the best scheduler configuration and compares:

1. matched non-MTP baseline;
2. current native MTP;
3. MTP plus deferred compact GDN state;
4. MTP plus copyless speculative state;
5. MTP plus copyless and deferred state;
6. combined state optimizations plus verification-window mode;
7. matched in-process baseline/combined arms with V1 multiprocessing disabled.

Workloads include short and 8K shared prefixes, concurrency 1/4/8/16, and retained-prefix pressure at 0/8/24 cached prefixes.

Every arm records:

- median output throughput over repeated runs;
- TTFT and end-to-end latency;
- post-first-token decode throughput;
- MTP draft and acceptance counters;
- process-tree memory;
- deterministic output hashes;
- cold/hot baseline drift.

No optimization is promoted unless it preserves output parity and improves the median beyond measured run-to-run drift.

## Real-M5 qualification

The hosted M1 runner is a correctness and screening environment, not a performance proxy for a MacBook Pro M5 Max 128 GB running dense Qwen3.8-27B.

The final qualification suite targets:

- 8K, 32K, 64K, and 128K contexts;
- concurrency 1 and 4 where memory permits;
- cache-off, prefix-cache, MTP, and verification-window profiles;
- cold unique prompts, exact repeats, and shared-prefix/changing-suffix traffic;
- small-M affine quantized matmul shapes from M=1 through M=32;
- stable output hashes and memory safety floors.

The promotion target is not merely “MTP executes.” It is:

- prefix caching does not materially penalize cold unique prompts;
- repeated-prefix TTFT improves at every context;
- dense-27B native MTP is faster than the matched synchronous baseline when acceptance is high enough;
- performance remains stable as the cache working set grows.

## Engineering principles demonstrated

- **Negative results are evidence.** The first matrix was published even though it showed a slowdown.
- **Correctness and performance gates are separate.** A green test suite does not imply an acceleration.
- **Matched baselines matter.** Every candidate is normalized against cold/hot controls on the same runner.
- **Optimize ownership before syntax.** The biggest gains came from changing where state lives and when it materializes, not from tuning more flags.
- **Hardware specificity is explicit.** Hosted-M1 findings screen ideas; M5 Max results decide promotion.
- **Fail closed.** Prefix and speculative state are reused only when the complete transactional state is compatible.

## Portfolio framing

This project is an applied Apple Silicon inference case study spanning:

- MLX model architecture integration;
- Metal kernel and memory-path analysis;
- vLLM scheduler/cache lifecycle design;
- speculative decoding correctness;
- automated performance experimentation;
- reproducible benchmark reporting;
- upstream open-source collaboration.

The strongest outcome is not a single benchmark number. It is a disciplined path from an incomplete model feature, through transactional correctness, through an honest negative serving result, to experimentally isolated architectural bottlenecks and a hardware-specific promotion plan.
