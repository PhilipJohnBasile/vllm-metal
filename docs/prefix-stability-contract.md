# Reusable frontend prefix contract

Automatic prefix caching is token-exact. A frontend can preserve the semantic
meaning of a request while still destroying every cache hit by changing any
token before the reusable boundary.

## Stable prefix

Keep these values deterministic and in a fixed order across turns:

- system and developer instructions;
- chat-template version;
- tool names, descriptions, and JSON schemas;
- tool ordering;
- repository or workspace instructions that are still valid;
- static retrieval context;
- model and tokenizer revision;
- reasoning/template mode flags.

Serialize mapping keys deterministically. Sort tools by their stable identity
before applying the chat template. Do not rebuild equivalent schemas in a
non-deterministic language/runtime order.

## Changing suffix

Place volatile or request-specific values after the stable prefix or transport
them outside the model prompt:

- current date/time;
- request, run, trace, and session identifiers;
- random nonces;
- progress counters;
- changing health/status text;
- temporary filesystem paths when they are not part of the model task;
- per-turn user input and new tool results.

A timestamp at the beginning of a system message invalidates every subsequent
cache block. Moving the same timestamp to the final user suffix lets the static
system/tool/repository prefix remain reusable.

## Diagnostics

Use `tools/prefix_stability.py` to compare two frontend requests.

```bash
python tools/prefix_stability.py turn-1.json turn-2.json
```

The command exits successfully when canonicalization produces the same stable
prefix, including volatile-only and tool-order-only differences. It exits 2
when stable semantics or serialization changed.

For token-level evidence, provide the exact served tokenizer/model and scheduler
block size:

```bash
python tools/prefix_stability.py turn-1.json turn-2.json \
  --model /path/to/Qwen3.8-27B-MLX \
  --block-size 544
```

The report includes:

- full and stable request fingerprints;
- the first structural difference;
- volatile paths detected in each request;
- tokenizer-level first mismatch index;
- common prefix token count;
- number of complete cache blocks that remain reusable;
- block-aligned token fingerprints.

Generate a deterministic fixture for a frontend regression test:

```bash
python tools/prefix_stability.py request.json \
  --emit-canonical canonical-request.json
```

`--strip-volatile` removes known volatile fields from the emitted fixture. It
must not be used blindly for production requests: values needed by the model
should move to the changing suffix rather than disappear.

## Frontend acceptance gate

For every supported frontend, maintain a two-turn fixture that proves:

1. the stable request fingerprint is identical;
2. tool/schema order is identical after canonicalization;
3. the first changed token is at or after the intended suffix boundary;
4. the expected number of full scheduler blocks is reusable;
5. the serving engine records a real prefix-cache hit.

The final serving assertion is essential. Canonical JSON equality is only a
precondition; tokenizer/chat-template changes can still alter token IDs.
