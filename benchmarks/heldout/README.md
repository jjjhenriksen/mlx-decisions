# Frozen prompt-order evaluation v1

The 72 cases in `policy-decisions-v1.json` were authored before inference on
2026-10-01 for issue #5. They have no overlap with the supplied cache/performance
workloads. No prompts, policy wording, readout temperatures, or thresholds may be
adjusted based on these answers. Any later tuning needs a new held-out set.
Labels follow explicit policy rules; 24 cases each cover choice, binary noul,
and ordered score decisions, across six families including longer policy rubrics,
negation, boundary values, historical distractors, and conflicting priorities.

Run both prompt orders on every case with the same checkpoint, serial unfused
full forward. Alternate which order runs first. Disable prefix reuse and clear
caches between arms. The runner does not pass labels into the model. Freeze this
file, dataset, and evaluator in Git before running; the report retains the dataset
SHA-256, source commit, candidate logits, unrounded calibrated probabilities,
public API answers, prompt token hashes, labels, and timing for both arms.

Choice/score probabilities use the existing temperature 0.85. Binary noul also
uses the existing clipped log-odds transform at temperature 1.829074. Use argmax
for diagnostic decision accuracy (equivalent to a fixed 0.5 noul threshold).
The score API returns an expected ordinal level, so report its mean absolute
level error as well. No fitting or recalibration is performed.

Report accuracy, mean negative log likelihood, multiclass Brier score (sum of
squared errors across options, without dividing by option count), and top-label
ECE using ten fixed equal-width bins. The confidence in ECE is maximum option
probability, not the API's transformed `confidence` field. Include reliability
bin counts, per-type and per-family metrics, paired differences, and every flip.
Pooled Brier/ECE combine tasks with different numbers of options: consult the
per-type figures. ECE is noisy on a small sample and is not a calibration proof.

This is a small **synthetic held-out policy set**, held out from this project's
prompt/threshold tuning. It is not a representative task benchmark. Model-training
exclusion cannot be established. Labels lack independent adjudication. Cases
within a family share templates; do not treat 72 cases as 72 independent task
families or infer statistical significance. These limitations apply even at 100%
accuracy. Q4 results cannot establish 8-bit task quality or equivalence. State-first
stays the default; speed and these limited results alone do not justify changing it.

```sh
uv run python scripts/evaluate_prompt_order.py \
  --model openjev/openjev-MLX-4bit \
  --revision 63aecab0abcba710f38270ddc1061fb2e3d8d143 \
  --output benchmarks/results/heldout-q4-prompt-order-v1.json
```

Omit the model overrides only in a provisioned environment for the pinned 8-bit
checkpoint. A failed run retains completed pairs and any pending arm, marks
`complete: false`, and exits nonzero. It must not be used as a complete comparison.
