# Research and adaptation decisions — 2026-10-01

## Recommendation

Build a **prefill-first MLX decision backend**, retaining the official OpenJEV
prompt and calibration. Use MLX-LM's supported Qwen3.5 trunk, not a wholesale fork
of a speculative text-generation engine. Make score extraction, cache ownership,
and batching the specialized layer. Implemented in this repo.

## Options examined

| Option | Useful properties | Why not simply use it unchanged? |
| --- | --- | --- |
| [Official OpenJEV MLX helper](https://huggingface.co/openjev/openjev/blob/ac97900fd034fdd7e7e536f3d4c21b836cae0750/helper/shim_mlx.py) | Authoritative trained prompts, 8-bit text-only 27B checkpoint, calibrated choice/noul/score readout | Global lock; separate full forward per question; full vocabulary output; no shared prefix reuse. Best compatibility baseline. |
| [jevmlx](https://github.com/bnsd55/jevmlx/tree/7e0d746081b88412ccd7d84a5ffdcf9d61b36904) | MLX schema compiler, letter and label scorers, prefix/cache branching, bounded serial GPU server, constraints, parity tooling | Its own prompt protocol and model-dependent scores are not the OpenJEV training contract. Excellent architecture reference, not a drop-in calibrated OpenJEV implementation. |
| [Bespoke Nimble](https://github.com/bespokelabsai/nimble) | Trained decision model, MLX ParallelScorer with shared prefill; latest release advertises 8,192-token context and up to 255 options | Different codebook/prompt/temperature. Current README says the MLX runner cannot use quantized weights or raw LoRA adapters. Benchmark accuracy figures across different evaluation sets are not comparable. |
| [OpenJEV with vLLM](https://huggingface.co/openjev/openjev/blob/ac97900fd034fdd7e7e536f3d4c21b836cae0750/serve/SERVE.md) | Targeted logprobs, prefix caching, batched server, image lane on supported full model | Strong NVIDIA deployment alternative, not native Apple Metal. H100 timing is not an M3 latency prediction. |
| [OpenJEV GGUF / llama.cpp](https://huggingface.co/openjev/openjev-GGUF) | Portable Metal/CPU serving and several quantizations | Different quantization/runtime; not a backend for the requested MLX checkpoint. Requires separate accuracy/parity evaluation. |
| [Jev](https://docs.typesafe.ai/) | Hosted typed-decision API and client ecosystem | Hosted service, not an open custom MLX execution backend. |
| Generic MLX-LM generation or grammar-constrained JSON | Broad model support; useful when free text is necessary | Generating JSON pays decode costs and does not reproduce this model's calibrated one-position readout. |

The requested default is `openjev/openjev-MLX` at
`a9dcc20aa827a6c7eae478f6ebb3b255bb135451`, 8-bit affine/group 64, dense Qwen3.5,
64 layers (three linear-attention layers per full-attention layer), 248,320-token
vocabulary, 5,120 hidden width. Model card: https://huggingface.co/openjev/openjev-MLX.
A separate [4-bit release](https://huggingface.co/openjev/openjev-MLX-4bit) is available,
but is **not silently substituted** for the requested model.

## What the live Yukon challenge actually measures

[Live challenge](https://www.yukon.org/mlxfast), retrieved 2026-10-01, points to the
Ternary Bonsai 2 27B engine. Its score combines prefill^0.25 × decode^0.75 on a paired
M5 run. At retrieval the displayed top improvement was about 450%, using DFlash
speculation. This is a rapidly moving leaderboard, not a measured gain for OpenJEV.
Search-index snippets still called it a Qwen challenge: the live page and source
were preferred over those stale snippets.

Inspected pinned repositories:

- [Bonsai engine](https://github.com/Layr-Labs/mlxfast-bonsai2-27b-engine/tree/d038e704422d8fabf4fe0c9b05a366e362cb7d7e)
- [Challenge harness](https://github.com/Layr-Labs/mlxfast-challenge/tree/4ea72c3b28873fca23b12b6f33193a2eeb5042f8)

| Technique | Disposition in mlx-decisions | Reason / proof required |
| --- | --- | --- |
| Evaluation-only and last-position prefill seams | Implemented: trunk-only intermediate chunks, only final hidden row reaches head | Avoid allocating/projecting discarded sequence × vocabulary logits. Upstream Qwen35.swift includes these explicit seams. |
| Selected output projection | Implemented: gather 52 quantized letter rows once, retain packed codes/scales/biases | Extends narrowing from position to allowed vocabulary. Conditional softmax normalization cancels the full-vocabulary logsumexp. Test full-head parity; no legal-mass claim. |
| Fused gate/up projection | Implemented behind `--fuse-gate-up` | Adapt row concatenation from the challenge's quantized fusion tests to dense SwiGLU. Same stored values, fewer matrix-multiply dispatches; summation/shape rounding can differ. Disabled by default pending measured gains. |
| Shared state ownership / compact recurrent carries | Explicit KV **and** conv/SSM fork; bounded in-memory LRU | Qwen3.5 is hybrid recurrent, not KV-only. Forking only keys/values is incorrect. Current MLX-LM already makes conv tails contiguous; no redundant patch. |
| Exact-length buckets and single-owner GPU microbatches | Implemented across question rows and concurrent HTTP requests | No fake padding tokens enter the recurrence. GPU parallelism is batch dimensions, not multiple Python model threads. |
| DFlash / MTP / token speculation | Not ported | A decision needs first-position scores, not a generated sequence. A draft model adds work without avoiding the authoritative prefill. |
| Bonsai 2-bit ternary/Hadamard kernels | Not ported | OpenJEV is 8-bit affine, not a ternary packed checkpoint. Repacking/requantizing would change the model and invalidate the requested baseline. |
| M5-specific tuned Metal kernels / private Swift seams | Research-only | This test machine is M3 Pro. Requires an isolated port with matching tensor layouts, architecture checks, correctness tests and real-device timing. Copying shader code is not proof of benefit. |
| Broad `mx.compile` of mutable cache flow | Not enabled | Dynamic cache mutation and shape churn make naive compilation unsafe/unhelpful. MLX already uses compiled SwiGLU and optimized gated-delta kernels. Measure pure stable-shape kernels individually before adding more. |

## Query parallelism: what is and is not parallel

1. **Many questions, same state:** state-prefix prefill once, independent cache
   branches, same-length question tails in GPU batches.
2. **Many requests, same state:** scheduler coalesces within a short window; same
   prefix reused; repeated shapes batch. Responses preserve request/question order.
3. **Many unrelated states:** `--no-prefix-cache` allows exact-length full prompts
   to batch together. Prefix mode groups by exact prefix, so unrelated states are
   normally separate GPU groups. A scheduler group is not necessarily a GPU batch.
4. **CPU admission:** asynchronous HTTP, bounded queue, one dedicated model owner.
   Model evaluation remains serialized by GPU batches; multiple model processes
   would duplicate roughly 27 GB of weights and contend for unified memory.
5. **Multi-device:** not implemented. MLX distributed inference is a future option,
   but this repo makes no claim of cross-Mac parallelism or concurrent Metal streams.

Measure p50/p95 **request wall latency**, queue wait, aggregate requests/s,
decisions/s, batch sizes, and peak Metal bytes. Throughput divided by concurrency
is not an individual request latency. A lower latency at batch one may be preferable
to the largest batch for a live agent loop.

## Compatibility boundaries

- Text/JSON/DOM only; reject image/screenshot input.
- Choice 1..52, score 2..52, noul. More-than-52 tournament behavior is intentionally
  rejected rather than presenting an approximate multi-pass distribution as exact.
- Prompt max 16,384 tokens. Dictionary instructions use the trained `pyrepr` mode.
- Choice temperature 0.85; noul slope temperature 1.829074, zero bias; no permutations.
- Retain candidate order. No dropping labels, top-k truncation, or padding-based shortcut.
- Code license Apache-2.0; model license **CC BY-NC 4.0**. Commercial use of the
  model requires a separate license; this implementation does not change that.
- Initial synthetic benchmarks test execution parity/performance, not held-out task
  accuracy, robust calibration, order invariance, or malicious-state resistance.

See `THIRD_PARTY_NOTICES.md`, exact dependency lock, and `benchmarks/results/` for
implementation provenance and measured evidence.
