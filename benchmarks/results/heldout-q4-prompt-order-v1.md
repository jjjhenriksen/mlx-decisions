# Held-out Q4 prompt-order comparison — 2026-10-01

**72/72 decisions correct in both orders; zero paired decision flips.** This is a small synthetic policy set with a ceiling effect, not a demonstration of general accuracy or calibration equivalence. State-first remains the default.

Model: `openjev/openjev-MLX-4bit@63aecab0abcba710f38270ddc1061fb2e3d8d143`. Apple M3 Pro, 36 GiB unified memory. Both arms use serial, unfused full forward, with no prefix reuse; arm order alternates per case. Peak process MLX allocation was 14.50 GiB. This checkpoint is explicitly Q4 and does not satisfy issue #4’s pinned 8-bit run requirement.

## Fixed evaluation method

The dataset and evaluator were committed at `0abb56a061599253769b07d9d8ec60d424844954` before inference; the run recorded a clean source tree. Dataset SHA-256: `14f45a9e599b66d62050517aae2e90d7d29c7a13dd298b3705a0ac15603d9f84`. All 72 cases are distinct from the existing performance workloads. There was no prompt, policy, threshold, or temperature tuning based on any held-out answer. The original temperatures (0.85 and noul 1.829074) were preserved. Labels were used only for scoring after inference. See the [frozen protocol](../heldout/README.md) and [dataset](../heldout/policy-decisions-v1.json).

## Paired results

Accuracy is argmax option accuracy; for noul it uses the fixed 0.5 threshold. Scores also retain their API expected level. Brier is the sum of squared option errors. ECE uses ten fixed equal-width top-probability bins; lower log loss, Brier, ECE, and ordinal error are better. The pooled figures combine different option counts; inspect each type.

| Scope | Order | Accuracy | Log loss | Brier | Top-label ECE |
|---|---|---:|---:|---:|---:|
| overall (72) | state-first | 100.0% | 0.007716 | 0.002036 | 0.007096 |
| overall (72) | rubric-first | 100.0% | 0.004227 | 0.000082 | 0.004206 |
| choice (24) | state-first | 100.0% | 0.013029 | 0.005787 | 0.011254 |
| choice (24) | rubric-first | 100.0% | 0.001639 | 0.000010 | 0.001636 |
| noul (24) | state-first | 100.0% | 0.006900 | 0.000097 | 0.006876 |
| noul (24) | rubric-first | 100.0% | 0.010199 | 0.000236 | 0.010139 |
| score (24) | state-first | 100.0% | 0.003218 | 0.000225 | 0.003159 |
| score (24) | rubric-first | 100.0% | 0.000843 | 0.000001 | 0.000843 |

Every one of the six families had 12/12 correct predictions under both orders. The raw report contains all per-family metrics and reliability-bin counts. On the 24 score cases, expected-level mean absolute error was 0.003659 for state-first and 0.000741 for rubric-first.

Rubric-minus-state pooled differences: accuracy 0.0000, log loss -0.003489, Brier -0.001954, and ECE -0.002890. These are descriptive paired differences, not a significance claim. Binary noul log loss and Brier worsened with rubric-first even though accuracy remained perfect.

Largest option-probability change: **0.260214**, on `support-routing-07`, with the same correct winner. State-first assigned the target probability 0.735829; rubric-first assigned 0.996043. The case reports both a late shipment and a late refund, explicitly says the customer is not locked out, and requires prioritizing the unresolved payment problem. This demonstrates why unchanged winners alone are insufficient to establish equivalent probability behavior.

## Limits and interpretation

These are freshly authored deterministic policy cases held out from this project’s prompt/threshold tuning. They are small and easy for this checkpoint: both arms reached the accuracy ceiling. Related cases share templates and are not independent samples. Labels are rule-derived without independent human adjudication. Exclusion from model pretraining cannot be verified. The suite has no broad real-world task sampling, human disagreement, distribution shift, or challenging error population.

Low ECE and proper-score values on an all-correct sample do not establish general calibration. Ten-bin ECE is especially sensitive to sample size and bin choice. No recalibration, confidence threshold selection, or statistical significance test was performed. These results neither establish Q4/8-bit equivalence nor warrant switching the state-first default. Broader independent task data would be needed for that decision.

Timings are single forward samples on a live desktop and are retained for audit, not promoted as comparative performance claims. The earlier Q4 cache-parity failures remain unchanged and separate. Fusion was not enabled in either quality arm.

## Evidence and reproduction

[Complete raw paired report](heldout-q4-prompt-order-v1.json) retains all 144 arms, labels, candidate logits, unrounded calibrated probabilities, public answers, prompt token hashes, execution samples, model revision, device, dependency versions, source commit, and dataset hash. Metrics were independently recomputed from the retained observations and matched the report exactly.

Local tests: 54 passed; Ruff clean. GitHub CI on the frozen evaluator commit: 54 passed, no skipped tests. CI checks exercise the harness and tiny-model regressions; the full Q4 evidence comes from the local paired run, not CI.

```sh
uv sync --locked --extra dev
uv run python scripts/evaluate_prompt_order.py \
  --model openjev/openjev-MLX-4bit \
  --revision 63aecab0abcba710f38270ddc1061fb2e3d8d143 \
  --output benchmarks/results/heldout-q4-prompt-order-v1.json
```
