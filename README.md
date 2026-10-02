# mlx-decisions

**Prefill-first decision inference for OpenJEV on Apple Silicon.**

Return typed `choice`, `noul` (yes/no probability), and `score` answers without
generating JSON or a chain of thought. By default, preserves the official OpenJEV text prompt,
letter codebook, and calibration while specializing GPU execution.

## What is implemented

- **Selected head:** project only 52 option-token rows, instead of 248,320 vocabulary rows.
- **Last-position readout:** intermediate prefill runs the trunk only; project the final hidden row.
- **Shared-prefix cache:** bounded LRU; independent branches copy attention KV **and** recurrent conv/SSM state.
- **GPU batching:** exact-length question/request buckets, configurable row/token limits, chunked prefill.
- **Concurrent server:** bounded admission, short cross-request microbatch window, one model-owning GPU thread.
- **Experimental fusion:** optional quantized gate/up row fusion inspired by the Yukon mlxfast challenge.
- **Auditable benchmarks:** official full-forward baseline, ablations, cold/warm cache separation,
  raw-logit/probability parity, decision flips, batch counts, and HTTP p50/p95 latency.

Python + MLX-LM backend, not a Swift engine fork. Current model adapter: dense Qwen3.5,
defaulting to the official **8-bit OpenJEV-MLX** checkpoint. The official 4-bit
checkpoint can be selected explicitly; it is never silently substituted.

## Quick start

Apple Silicon, Python 3.12+, enough unified memory for roughly 27 GB of weights
plus runtime allocations. Dependencies are pinned in `uv.lock`.

```sh
uv sync --extra dev
uv run mlx-decisions download
uv run mlx-decisions decide benchmarks/example.json
uv run mlx-decisions serve --port 3000
```

The download is pinned to `openjev/openjev-MLX` revision
`a9dcc20aa827a6c7eae478f6ebb3b255bb135451`. Weights stay in the Hugging Face cache,
not in Git. `--model /local/checkpoint` is also supported (local provenance is
reported as unknown rather than claiming a Hub revision).

For a lower-memory Q4 run, select the official 4-bit checkpoint and its own revision:

```sh
hf download openjev/openjev-MLX-4bit \
  --revision 63aecab0abcba710f38270ddc1061fb2e3d8d143 --max-workers 1
uv run mlx-decisions --model openjev/openjev-MLX-4bit \
  --revision 63aecab0abcba710f38270ddc1061fb2e3d8d143 \
  --rubric-first decide benchmarks/example.json
```

Q4 is a different quantized checkpoint: parity against its own full forward does
not establish equivalence to the 8-bit checkpoint. Weights are loaded lazily and
materialized sequentially to reduce temporary load-time memory peaks.

```sh
curl http://127.0.0.1:3000/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @benchmarks/example.json
```

```python
from mlx_decisions import DecisionRequest
from mlx_decisions.engine import Engine

engine = Engine()  # Load once and reuse.
request = DecisionRequest(
    state="Customer was charged twice.",
    questions={"route": {
        "type": "choice", "instructions": "Which team should handle this?",
        "criteria": {"billing": None, "shipping": None, "technical": None},
    }},
)
print(engine.decide(request))
# One scheduler call, potentially several GPU batches:
results = engine.decide_many([request, request, request, request])
```

## Parallel query modes

For several questions or requests about the **same state**, the default prefix
mode reuses the state prefill. Equal-length tails run as GPU batches. Different
lengths get separate buckets, without recurrent-state-corrupting padding.

For **unrelated states**, disable prefix grouping to batch equal-length complete prompts:

```sh
uv run mlx-decisions --no-prefix-cache --batch-size 4 serve
```

### Experimental rubric-first flag

For the **same question/criteria across changing states**, opt into rubric-first:

```sh
uv run mlx-decisions --rubric-first decide benchmarks/example.json
uv run mlx-decisions --rubric-first serve --port 3000
```

Python: `Engine(rubric_first=True)`. The flag is server/engine-wide, not an HTTP
request field. CLI global flags go **before** `decide` or `serve`.

This renders **Question + Options → State → answer instruction**, caching the
identical leading rubric tokens across requests. Changed instructions or option
descriptions/order select a different cache prefix. Token-level matching handles
tokenizer boundaries; cached branches still copy both KV and recurrent state.
`--no-prefix-cache` disables reuse without changing the selected prompt order.
Results include `prompt_order` (`state-first` or `rubric-first`).

**State-first remains the default.** Rubric-first changes the training-time prompt
layout: general accuracy and probability calibration are unvalidated. The same readout
formulas are applied, but that does not establish equivalent calibration. Cached
tokens still participate in attention. A single `decide` invocation is a fresh
process; reuse across calls requires a running server or a persistent `Engine`.

Concurrent HTTP callers are combined within a 4 ms window. The GPU has one owner;
this is batched parallel math, not simultaneous independent model executions.
Do **not** run multiple Uvicorn model workers on one Mac. Read [the design tradeoffs](docs/research.md).

## API and limits

| Route | Behavior |
| --- | --- |
| `POST /v1/systemone` | OpenJEV-shaped `state`, `questions`, optional `model`; returns `answers`, logical token usage, execution and queue metrics |
| `GET /health` | Process liveness |
| `GET /ready` | Model loaded and worker alive; queue depth/capacity |
| `GET /v1/models` | Loaded model identity |

- Text/JSON/DOM only. Screenshots and image fields are rejected.
- Choice: 1–52 options; score: 2–52 ordered levels; noul: optional true/false descriptions.
- Up to 64 questions/request, 16,384 tokens/prompt, 1 MiB HTTP body.
- Requests beyond 52 choices are rejected; no hidden tournament approximation.
- Queue overload: 429 + `Retry-After`; invalid input: 422; deadline: 504.
- Default prefix cache: at most two entries / 512 MiB. Model and temporary branch
  allocations are **additional**, not included in that cache budget.
- Default max GPU batch 4 rows / 4,096 token positions; a longer single prompt is
  allowed and prefills in 256-token chunks. Exact-length bucketing can fragment batches.
- Scores are conditional on the provided options, **not** full-vocabulary probability
  mass or proof of correctness. Include an escape option when appropriate.
- `usage.input_tokens` is logical prompt usage; `performance.prefill_tokens` is
  actually scheduled token work. Group metrics are shared, not per-request attribution.

The CLI binds localhost only, has no authentication, and is intended for a trusted
single-user machine. Cache entries remain in process memory, never on disk. If exposing
it remotely, add authentication/admission controls outside this local server first.

## Verification and benchmarks

```sh
uv run pytest -q
uv run ruff check src tests scripts
uv run python scripts/check_checkpoint.py
uv run python scripts/benchmark.py --repeats 3
# Rubric-first cache parity plus separately reported drift from state-first:
uv run python scripts/benchmark.py --rubric-first --repeats 3 \
  --output benchmarks/results/rubric-first.json
# Explicit experimental arm, compared against the unfused official baseline:
uv run python scripts/benchmark.py --fuse-gate-up --repeats 3 \
  --output benchmarks/results/fused.json
# With the server running:
uv run python scripts/benchmark_http.py --concurrency 1 2 4 8 --requests 16
```

The HTTP benchmark retains every attempted request, including HTTP overload/deadline
errors and transport failures. Latency percentiles and `requests_per_second` use
successful responses only; `attempted_requests_per_second`, status/error counts,
and raw samples expose failed work separately. Reports mark incomplete setup or
measurement runs, and the command exits nonzero if any measured request fails.

`benchmark.py` fails if any argmax changes or the maximum conditional-probability
error exceeds 0.005. It records error magnitudes and every timing sample. The gate
is a small synthetic compatibility check, **not** a held-out accuracy evaluation.
The benchmark caps the process-wide MLX allocator cache at 512 MiB by default
(`--metal-cache-mib`); this is separate from model weights and the prefix KV cache,
and the selected limit is recorded in the result file. For Q4, pass the explicit
`--model` and `--revision` shown above to the benchmark too.
With `--rubric-first`, the parity gate compares cached/optimized execution against
an unfused full forward of the **same reordered prompt**. Changes versus the
official state-first prompt are reported separately in
`prompt_order_drift_vs_state_first`, not gated as cache errors. Speedups are then
relative to rubric-first full forward, not the official prompt. Use the
`four_queries_distinct_state` workload to exercise a shared rubric across states.
No headline performance claim is inherited from Yukon's generation leaderboard.

Full-model **Q4** results on Apple M3 Pro (36 GiB): **56/60 variant/workload cases
passed probability parity; four failed**, with no decision flips. The failures
are the multi-question batch-four variant in both prompt orders, plus the
state-first cold/warm prefix variants. Maximum probability error was 0.029492
against a 0.005 limit. Do not promote those variants as probability-equivalent.

For one rubric across four distinct states, repeated-call group latency was
**3.89 s state-first versus 1.61 s rubric-first** with the default two-entry cache.
For four queries sharing a long state, warm-prefix latency reversed:
**1.95 s state-first versus 19.05 s rubric-first**. Each is a three-run median.
Prompt reordering itself changed conditional probabilities by up to **0.116852**
in this small synthetic suite, without changing the winning choices. These are
not held-out accuracy/calibration results or evidence of Q4/8-bit equivalence.

See the [full Q4 validation report](benchmarks/results/q4-validation-2026-10-01.md),
[raw state-first samples](benchmarks/results/q4-state-first-2026-10-01.json), and
[raw rubric-first samples](benchmarks/results/q4-rubric-first-2026-10-01.json).
A separate [held-out Q4 comparison](benchmarks/results/heldout-q4-prompt-order-v1.md)
evaluated 72 new synthetic policy decisions with unfused full forward in both
orders. Both scored 72/72 with zero flips, but option probabilities changed by
up to 0.260214. Rubric-first improved pooled log loss/Brier while worsening the
binary noul metrics. This easy, small, templated set does not establish general
calibration, statistical superiority, or 8-bit equivalence. State-first stays
the default. The [frozen protocol](benchmarks/heldout/README.md) separates this
held-out set from prompt and threshold tuning and retains all paired samples.

Both checkpoints are now downloaded. The 8-bit full-forward run hit severe memory
pressure on this machine; no successful 8-bit timing/parity result is claimed.
The [initial development evidence](benchmarks/results/validation-2026-10-01.json)
records the earlier download blocker, not the current Q4 validation state.

## Research and license

[Research and adaptation matrix](docs/research.md) compares the official helper,
jevmlx, Nimble, vLLM, llama.cpp, and the current mlxfast challenge. It explains why
DFlash/MTP and Bonsai ternary kernels were not transplanted into a one-position,
8-bit decision scorer.

Code: **Apache-2.0**. OpenJEV weights: **CC BY-NC 4.0** (separate commercial license
required for commercial model use). See [third-party notices](THIRD_PARTY_NOTICES.md).
