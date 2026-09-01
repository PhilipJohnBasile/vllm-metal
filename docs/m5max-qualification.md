# M5 Max long-context qualification

GitHub's hosted Apple runner is an M1-class machine with about 7 GB of memory.
It cannot represent Qwen3.8-27B behavior, and its GPU generation does not take
the same affine small-batch quantized-matmul paths as an M5 Max.

This qualification suite is therefore intended for a dedicated self-hosted
MacBook Pro M5 Max runner labeled:

```text
self-hosted, macOS, ARM64, m5max
```

## What it measures

For each fresh server profile and exact tokenizer context length:

- cold unique-prompt TTFT and throughput;
- exact-repeat prefix reuse;
- block-aligned shared-prefix plus changing suffix;
- concurrency 1 and 4 by default;
- long-context output throughput and post-first-token generation speed;
- baseline, prefix-cache-only, native MTP, and MTP verify-window profiles;
- small-M quantized matmul latency for M=1,2,4,8,12,16,32.

The default matrix uses 8K, 32K, 64K, and 128K prompts with 128 output tokens.
Each server profile runs in a fresh process so MLX allocator state and prefix
blocks cannot leak between arms.

## Why both hosted and M5 results are needed

The hosted M1 runner is useful for correctness, failure reproduction, and
low-memory stress. It is not an honest performance proxy for an M5 Max:

- MLX's `qmv_wide` small-batch quantized kernel is generation-gated for affine
  quantization and has measured M5 gains at M=2,4,8;
- the later M5-specific qmv dispatch-limit tuning merged after the MLX 0.32.0
  release currently pinned by vllm-metal;
- memory capacity changes scheduler block counts and high-water state behavior;
- M5 Max bandwidth and GPU occupancy alter split-KV thresholds.

Performance claims for Qwen3.8-27B must therefore include the real M5 Max arm.

## Running

1. Install a GitHub Actions runner on the M5 Max and attach the `m5max` label.
2. Keep the model on local storage. The workflow accepts an absolute path and
   does not upload weights.
3. Dispatch **M5 Max Context Qualification** with:
   - a Qwen3.5/3.8 MLX model that includes the trained MTP head;
   - the tokenizer path when separate;
   - the desired context list and maximum model length.

The workflow rebuilds vllm-metal's native extension against the exact pinned
MLX ABI, verifies the MTP head, starts each profile, records streaming TTFT and
throughput, runs the operator microbench, and uploads one result artifact.

## Promotion gates

- prefix-cache-only cold unique traffic must remain at least 0.95x cache-off;
- exact/shared prefix TTFT must improve materially at every tested context;
- native MTP must be net faster than the matched synchronous baseline on the
  dense 27B target when acceptance is at least 75%;
- no profile may change deterministic output hashes unexpectedly;
- any MLX ABI-pin change must rebuild native artifacts and compare operator and
  serving results against the prior pin;
- 128K tests must remain inside the configured memory safety floor.
