# Pinned 8-bit validation provision check — 2026-10-01

Issue #4 remains open. No successful 8-bit full-forward/parity/performance run
is claimed. Q4, tiny-model regression coverage, and tokenizer checks do not meet
its acceptance criteria.

The target is `openjev/openjev-MLX` at
`a9dcc20aa827a6c7eae478f6ebb3b255bb135451`, with 8-bit affine weights. The cached
six shard sizes match the pinned Hub metadata. No download was needed and no
cached data was deleted. This checks presence and file sizes, not tensor checksums.

This Apple M3 Pro has 36 GiB unified memory. Metal reports a recommended working
set of 30,150,672,384 bytes (28.08 GiB). The checkpoint index reports
28,579,478,528 tensor bytes (26.62 GiB), leaving only 1.46 GiB below that
recommendation before output, activations, KV/recurrent state, and allocator
allocations. The documented 4 GiB planning reserve exceeds that headroom. The
reserve is a conservative heuristic, not a measured universal hardware minimum.
Even an eligible preflight cannot guarantee the full benchmark will fit.

The prior repository report records severe 8-bit full-forward memory pressure
and an early exit 137. We did not repeat that failure on this live desktop or
stop OpenClaw/Ollama services. The raw provision check records memory pressure,
storage, and Ollama residency at collection time. During that check this chat's
separate Q4 held-out evaluation was also running; its residency is temporary,
and is not the basis for the tensor-versus-working-set blocker. The available
writable storage was sufficient for the cached checkpoint, so ENOSPC is an older
download incident, not the current blocker. GitHub's repository runner inventory
returned zero registered self-hosted runners.

The existing Q4 result files were re-read at main `b8f03f7`, their SHA-256 hashes
retained, and their gates recomputed from the per-case recorded pass/fail values:
56/60 cases passed, four failed, maximum probability error 0.0294921994, zero
reported flips. This audit verifies the stored evidence, not a new Q4 performance
run. We have not filled missing old per-row logits by inventing data.

## Fresh Q4 reproduction of the failed workload

The `four_questions_shared_state` workload was rerun on the full pinned Q4
checkpoint, six variants under each order, one measured repetition per variant.
Both commands completed their six cases and exited **1** for failed parity.
The raw reports record clean source commit
`ec0c86dbcaf1e028227992c29ed9a9e90d91d0f0`, pinned model/device/version metadata,
all 48 measured question rows, original reference logits, per-option probability
errors, and flips. The recorded option errors were recomputed and matched.

| Order | Variant | Maximum probability error | Flips |
|---|---|---:|---:|
| state-first | selected_head_batch4 | 0.0294921994 | 0 |
| state-first | prefix_batch4_cold | 0.0109576136 | 0 |
| state-first | prefix_batch4_warm | 0.0109576136 | 0 |
| rubric-first | selected_head_batch4 | 0.0254556537 | 0 |

These reproduce all four prior failures at the unchanged 0.005 gate. This is a
report-retention check on one workload, not a new complete 60-case performance
study or substitute for the 8-bit run. Single-sample timings are retained but no
performance claim is promoted. Fusion was disabled.

[State-first per-row evidence](q4-parity-retention-state-first-2026-10-01.json) and
[rubric-first per-row evidence](q4-parity-retention-rubric-first-2026-10-01.json).

## Resume on an adequately provisioned Apple Silicon host

Use enough memory to cover the stored tensors, the explicit runtime reserve,
and other resident services; inspect observed pressure and peak memory too.
The preflight performs read-only cached-file/Hub-metadata checks and exits 2 if
blocked. It never fetches weights. Check storage before resuming a missing
checkpoint download, preserving existing caches.

```sh
uv sync --locked --extra dev
uv run python scripts/preflight_checkpoint.py \
  --output benchmarks/results/pinned-8bit-preflight.json
# Only when provisioned, validate tokenizer and then run the full pinned suite:
uv run python scripts/check_checkpoint.py
uv run python scripts/benchmark.py --repeats 3 \
  --output benchmarks/results/pinned-8bit-state-first.json
uv run python scripts/benchmark.py --rubric-first --repeats 3 \
  --output benchmarks/results/pinned-8bit-rubric-first.json
# Optional and separate; no default-path success is inherited from fusion:
uv run python scripts/benchmark.py --fuse-gate-up --repeats 3 \
  --output benchmarks/results/pinned-8bit-experimental-fusion.json
```

The benchmark now writes an incomplete report before loading, preserves each
measured repetition, and records every question's reference/actual logits,
conditional probabilities, option-wise errors, and decision flips. It retains
exception details and stage information on failure. Completed execution and a
passed gate are separate fields. The unchanged gate requires zero flips and
maximum probability error at most 0.005. A nonzero exit is a failed/unfinished
check, never evidence of parity. Timing samples exclude the added report writes.

The synthetic parity gate applies to candidate probabilities at temperature
0.85, before the separate binary noul calibration transform. The held-out quality
evaluator applies the official noul transform and serves a different purpose.

Only close #4 after the pinned **8-bit** runs complete and their retained results
support all acceptance criteria. Publish failing variants and workload-dependent
latencies honestly; a gate failure must never be promoted as a speedup of an
equivalent implementation. Fusion stays experimental.

Evidence: [raw preflight](pinned-8bit-preflight-2026-10-01.json),
[existing Q4 audit](existing-q4-audit-2026-10-01.json), and the
[prior Q4 report](q4-validation-2026-10-01.md).
