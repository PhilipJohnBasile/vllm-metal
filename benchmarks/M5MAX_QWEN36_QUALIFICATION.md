# Qwen3.6-27B native MTP qualification on M5 Max

This branch provides a physical-machine gate for the native Qwen MTP path in
`vllm-project/vllm-metal#618`. It is intentionally separate from the feature
PR so benchmark tooling and local evidence do not enlarge the production diff.

## Exact source stack

- vLLM Metal feature head: `bf59b7c4d912f3aca5f8217b0da1bfd1ed260294`
- MLX-LM native-MTP head: `a9fd7ef1032419a584ead9a38bdb66635f2d85c3`
- vLLM CPU wheel: `0.27.1+cpu` for CPython 3.12 / macOS arm64
- target checkpoint: `Qwen/Qwen3.6-27B`
- native speculative depth exposed by vLLM Metal: one token

Qwen3.6-27B is the official current 27B dense checkpoint. The earlier working
label “Qwen3.8-27B” should not be used for this run.

## Model requirement

The model directory must contain native `mtp.*` tensors in the ordinary MLX
weight map. A standalone engine-specific `mtp.safetensors` file is not enough
unless it is referenced by `model.safetensors.index.json` with keys matching
the MLX-LM Qwen MTP module.

The harness checks all of this before loading 27B weights and fails closed with
a specific diagnostic. In particular, the AutomatosX AX Engine packages keep
the target shards and MTP sidecar separate; standard MLX-LM ignores that
sidecar, so those directories are not accepted unchanged by this gate.

The reproducible path is to convert the official checkpoint with the exact
MLX-LM PR head. The runner can do that automatically.

## One-command run

From this branch on the physical M5 Max:

```bash
scripts/run_m5max_qwen36_qualification.sh \
  --model-dir "$HOME/Models/Qwen3.6-27B-6bit-native-mtp" \
  --prepare-official
```

The first run creates a dedicated Python 3.12 virtual environment, installs the
exact source stack, builds the native Metal extension and shader libraries,
downloads the official checkpoint, converts it to MLX affine 6-bit/group-64
while retaining MTP weights, and executes the qualification.

The official BF16 download plus converted output can require roughly 100 GiB
of temporary/free disk space. Subsequent runs can reuse both environment and
model:

```bash
scripts/run_m5max_qwen36_qualification.sh \
  --model-dir "$HOME/Models/Qwen3.6-27B-6bit-native-mtp" \
  --skip-install
```

To run an already completed compatible conversion, omit `--prepare-official`.
To smoke-test the harness on another arm64 Mac, add `--allow-non-m5max`; such a
run does not qualify an M5 Max performance claim.

## What is measured

Four independent server launches are interleaved to expose launch drift:

1. baseline first
2. native MTP first
3. baseline second
4. native MTP second

Each launch serves three deterministic workloads with prefix caching enabled:

| Workload | Prompt | Output | Concurrency | Requests/repeat |
|---|---:|---:|---:|---:|
| interactive | 512 | 256 | 1 | 3 |
| serving | 1,024 | 128 | 4 | 8 |
| long prefix/chunked prefill | 8,192 | 128 | 1 | 2 |

The default is three timed repetitions per workload and launch. Prompts share a
large stable prefix but carry request-specific suffixes. Temperature and seed
are fixed, EOS is ignored, and every arm must produce the same per-request text
hash.

The evidence bundle records:

- exact source, model-config, OS, chip, and memory identity;
- native MLX-LM MTP load/generation check;
- server command, PID, port, startup time, logs, exit status, and port release;
- output throughput, TTFT, TPOT, end-to-end latency, and run-to-run CV;
- proposed and accepted speculative tokens;
- repeat-level and baseline-versus-MTP output hashes;
- per-workload launch drift and matched speedup;
- machine-readable JSON plus a Markdown decision report.

## Gates

The **functional gate** requires:

- physical M5 Max identity;
- native MTP weights loaded and draft outputs accepted;
- all 12 workload/launch cells completed;
- zero request/server errors;
- exact greedy output parity across repeats and arms;
- positive MTP acceptance on every workload;
- clean process-group shutdown and released listeners without SIGKILL.

The separate **positive performance claim gate** additionally requires:

- geometric-mean MTP throughput at least `1.05x` baseline;
- no workload below `0.95x` baseline;
- throughput CV no greater than 15%;
- absolute first-to-second launch drift no greater than 15%.

A functional pass with a performance-gate failure remains valid negative
evidence. It must not be presented as MTP acceleration.

## Results and exit codes

By default, evidence is written under:

```text
~/vllm-metal-results/qwen36-mtp-m5max-<UTC timestamp>/
```

Important files:

- `qualification_summary.md`
- `qualification_summary.json`
- `run.log`
- `revisions-and-hardware.txt`
- `native_mtp.json`
- `logs/*.log`
- `results/*.json`

Exit codes:

- `0`: functional and positive-performance gates passed;
- `2`: functional qualification failed;
- `3`: functional gate passed, but the positive-performance gate failed.

Use `--no-require-speedup` when collecting exploratory evidence that should
return success on a functional pass even when the speed claim fails.

## After MLX-LM #1740 merges

The downstream feature branch must still replace its temporary old MLX-LM pin
with the actual merge commit and rerun the exact-head validation. The physical
M5 Max evidence remains useful, but it does not substitute for that final pin
and source-tree validation.
