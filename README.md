# mlx-decisions

**Prefill-first decision inference for OpenJEV on Apple Silicon.**

Return typed `choice`, `noul` (yes/no probability), and `score` answers without
generating JSON or a chain of thought. Preserves the official OpenJEV text prompt,
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
targeting the official **8-bit OpenJEV-MLX** checkpoint. The 4-bit model is not substituted.

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
uv run python scripts/benchmark.py --repeats 3
# Explicit experimental arm, compared against the unfused official baseline:
uv run python scripts/benchmark.py --fuse-gate-up --repeats 3 \
  --output benchmarks/results/fused.json
# With the server running:
uv run python scripts/benchmark_http.py --concurrency 1 2 4 8 --requests 16
```

`benchmark.py` fails if any argmax changes or the maximum conditional-probability
error exceeds 0.005. It records error magnitudes and every timing sample. The gate
is a small synthetic compatibility check, **not** a held-out accuracy evaluation.
No headline performance claim is inherited from Yukon's generation leaderboard.

Initial development checks: **28 passed** on Apple M3 Pro, including real Metal
projection/fusion and tiny hybrid-Qwen cache tests. Full 27B checkpoint benchmarking
is in progress; do not interpret the tiny-model tests as full-model evidence.
Results will be recorded under `benchmarks/results/`.

## Research and license

[Research and adaptation matrix](docs/research.md) compares the official helper,
jevmlx, Nimble, vLLM, llama.cpp, and the current mlxfast challenge. It explains why
DFlash/MTP and Bonsai ternary kernels were not transplanted into a one-position,
8-bit decision scorer.

Code: **Apache-2.0**. OpenJEV weights: **CC BY-NC 4.0** (separate commercial license
required for commercial model use). See [third-party notices](THIRD_PARTY_NOTICES.md).
