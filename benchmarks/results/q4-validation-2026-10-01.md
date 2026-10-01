# Q4 full-model validation — 2026-10-01

Model: `openjev/openjev-MLX-4bit` at `63aecab0abcba710f38270ddc1061fb2e3d8d143`. Apple M3 Pro, 36 GiB unified memory.

**38 regression tests passed; lint clean. Full-model probability gate: 56/60 cases passed, 4 failed.** Both benchmark commands completed every case and exited 1 because their parity gate failed. This is not an all-green validation.

The 60 cases are five workloads × six variants × two prompt orders, including ten reference-baseline cases. Each has three measured repetitions plus unmeasured warm-up. Model loading is excluded from timing.

Maximum recorded MLX peak: **16.44 GiB**. The CLI also completed a real Q4 rubric-first request covering choice, noul, and score answers using ordinary CLI allocator settings.

## Same rubric, four different states

| Prompt order | Cache condition | Group median | Prefill tokens | Prefix hits / misses | GPU batch sizes | Parity |
|---|---|---:|---:|---|---|---|
| state-first | cold | 3.929 s | 260 | 0 / 4 | [1, 1, 1, 1] | pass |
| state-first | warm | 3.887 s | 260 | 0 / 4 | [1, 1, 1, 1] | pass |
| rubric-first | cold | 1.935 s | 173 | 0 / 1 | [4] | pass |
| rubric-first | warm | 1.607 s | 144 | 1 / 0 | [4] | pass |

The default two-entry LRU cannot retain four distinct state prefixes: the state-first arm labeled warm still has zero actual hits. Rubric-first reuses one common rubric and batches all four tails. This comparison measures the combined order/grouping/cache behavior, not prompt reordering in isolation.

## Long shared state: the opposite workload

| Prompt order | Full forward | Cold prefix | Warm prefix | Warm parity |
|---|---:|---:|---:|---|
| state-first | 21.625 s | 7.005 s | 1.950 s | pass |
| rubric-first | 21.248 s | 20.026 s | 19.047 s | pass |

## Probability parity failures

Gate unchanged: zero choice flips and maximum absolute conditional-probability error ≤ 0.005 versus full forward of the **same prompt order and Q4 checkpoint**.

| Prompt order | Workload | Variant | Maximum probability error | Choice flips |
|---|---|---|---:|---:|
| state-first | four_questions_shared_state | selected_head_batch4 | 0.029492 | 0 |
| state-first | four_questions_shared_state | prefix_batch4_cold | 0.010958 | 0 |
| state-first | four_questions_shared_state | prefix_batch4_warm | 0.010958 | 0 |
| rubric-first | four_questions_shared_state | selected_head_batch4 | 0.025456 | 0 |

Do not promote these failing variants as probability-equivalent. All tested serial/no-prefix-cache variants passed this synthetic gate; that is not general accuracy or calibration validation.

## Prompt-order drift (not a cache error)

| Workload | Maximum probability change | Choice flips |
|---|---:|---:|
| single | 0.001611 | 0 |
| four_questions_shared_state | 0.116852 | 0 |
| four_queries_shared_state | 0.001611 | 0 |
| four_queries_distinct_state | 0.000707 | 0 |
| long_state_four_queries | 0.004202 | 0 |

These compare two different prompts using unfused full forward. Unchanged winners do not establish unchanged calibration. Rubrics in this supplied benchmark are short; there is no held-out evaluation of long real-world rubrics.

## Runtime changes and boundaries

- The unchanged 8-bit weights loaded in a sequential-materialization probe, but full-forward inference on this 36 GiB machine produced severe pressure; one early run exited 137 and later diagnostics were terminated. No 8-bit timing/parity success is claimed.
- Engine loading now uses lazy load and sequential tensor evaluation with the allocator cache temporarily disabled, restoring the prior setting even on failure. Stored quantization codes are not changed.
- Benchmark allocator cache is capped at 512 MiB by default and recorded in both reports. This is separate from the prefix KV/recurrent cache and model allocations. Both arms use the same cap.
- The rubric arm includes stage-progress logging added after the state arm started; scoring and timing paths are otherwise the same. Results were collected sequentially on a live desktop, not a thermally controlled laboratory.
- The 8-bit checkpoint remains the CLI default; Q4 selection is explicit. Both downloaded checkpoints were retained. No OS memory/security settings were changed.
- The run used then-uncommitted source at the recorded base commit; exact source fingerprints are in the validation JSON.

## Reproduction

```sh
uv run pytest -q
uv run ruff check src tests scripts
uv run python scripts/benchmark.py --model openjev/openjev-MLX-4bit \
  --revision 63aecab0abcba710f38270ddc1061fb2e3d8d143 --repeats 3 \
  --output benchmarks/results/q4-state-first-2026-10-01.json
uv run python scripts/benchmark.py --model openjev/openjev-MLX-4bit \
  --revision 63aecab0abcba710f38270ddc1061fb2e3d8d143 --rubric-first --repeats 3 \
  --output benchmarks/results/q4-rubric-first-2026-10-01.json
```

## Evidence

- [State-first raw samples](q4-state-first-2026-10-01.json)
- [Rubric-first raw samples](q4-rubric-first-2026-10-01.json)
- [Validation summary and source fingerprints](q4-validation-2026-10-01.json)
- [Q4 checkpoint/tokenizer contract](q4-checkpoint-contract.json)
- [Real CLI result](q4-cli-smoke-2026-10-01.json)
