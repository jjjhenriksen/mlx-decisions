"""Fixed readouts and paired quality metrics; no fitted thresholds or temperatures."""

import math
from collections import defaultdict

from .protocol import NOUL_TEMPERATURE, TEMPERATURE, DecisionRequest


def load_cases(dataset):
    if dataset.get("split") != "held-out" or not dataset.get("cases"):
        raise ValueError("a nonempty held-out split is required")
    seen, result = set(), []
    for case in dataset["cases"]:
        if case["id"] in seen:
            raise ValueError("duplicate case id")
        seen.add(case["id"])
        request = DecisionRequest.model_validate(case["request"])
        if len(request.questions) != 1:
            raise ValueError("exactly one labeled question per case is required")
        question = next(iter(request.questions.values()))
        labels = [key for key, _ in question.options()]
        if case["label"] not in labels:
            raise ValueError("label is outside the option set")
        result.append((case, request, question))
    return result


def distribution(question, logits):
    """Unrounded option probabilities, including the official noul calibration."""
    if len(logits) != len(question.options()) or not all(math.isfinite(x) for x in logits):
        raise ValueError("missing or nonfinite candidate logits")
    scaled = [x / TEMPERATURE for x in logits]
    maximum = max(scaled)
    exp = [math.exp(x - maximum) for x in scaled]
    p = [x / sum(exp) for x in exp]
    if question.type == "noul":
        yes = min(max(p[0], 1e-4), 1 - 1e-4)
        logodds = math.log(yes / (1 - yes)) / NOUL_TEMPERATURE
        yes = 1 / (1 + math.exp(-logodds))
        p = [yes, 1 - yes]
    return p


def observation(case, question, logits):
    p = distribution(question, logits)
    labels = [key for key, _ in question.options()]
    target = labels.index(case["label"])
    winner = max(range(len(p)), key=p.__getitem__)
    result = {
        "probabilities": dict(zip(labels, p, strict=True)),
        "prediction": labels[winner],
        "correct": winner == target,
        "top_probability": p[winner],
        "negative_log_likelihood": -math.log(max(p[target], 1e-300)),
        "multiclass_brier": sum((x - (i == target)) ** 2 for i, x in enumerate(p)),
        "candidate_logits": list(logits),
        "api_answer": question.answer(logits),
    }
    if question.type == "score":
        # The API returns an expected level; argmax accuracy is a separate diagnostic.
        result["expected_level"] = sum(i * x for i, x in enumerate(p))
        result["absolute_level_error"] = abs(result["expected_level"] - target)
    return result


def metrics(observations, bins=10):
    if not observations:
        raise ValueError("metrics require observations")
    if bins < 1:
        raise ValueError("bins must be positive")
    n = len(observations)
    buckets = defaultdict(list)
    for obs in observations:
        buckets[min(bins - 1, int(obs["top_probability"] * bins))].append(obs)
    reliability = []
    ece = 0.0
    for i in range(bins):
        group = buckets[i]
        accuracy = sum(x["correct"] for x in group) / len(group) if group else None
        confidence = sum(x["top_probability"] for x in group) / len(group) if group else None
        if group:
            ece += len(group) / n * abs(accuracy - confidence)
        reliability.append(
            dict(
                lower=i / bins,
                upper=(i + 1) / bins,
                count=len(group),
                accuracy=accuracy,
                confidence=confidence,
            )
        )
    result = {
        "count": n,
        "accuracy": sum(x["correct"] for x in observations) / n,
        "negative_log_likelihood": sum(x["negative_log_likelihood"] for x in observations) / n,
        "multiclass_brier": sum(x["multiclass_brier"] for x in observations) / n,
        "top_label_ece": ece,
        "reliability_bins": reliability,
    }
    ordinal = [x["absolute_level_error"] for x in observations if "absolute_level_error" in x]
    if ordinal:
        result["score_mean_absolute_level_error"] = sum(ordinal) / len(ordinal)
    return result


def summarize(samples):
    result = {}
    for order in ("state-first", "rubric-first"):
        result[order] = {"overall": metrics([x[order] for x in samples])}
        for group_key in ("type", "family"):
            result[order][f"by_{group_key}"] = {
                key: metrics([x[order] for x in samples if x[group_key] == key])
                for key in sorted({x[group_key] for x in samples})
            }
    result["paired"] = {
        "count": len(samples),
        "decision_flips": sum(
            x["state-first"]["prediction"] != x["rubric-first"]["prediction"] for x in samples
        ),
        "state_only_correct": sum(
            x["state-first"]["correct"] and not x["rubric-first"]["correct"] for x in samples
        ),
        "rubric_only_correct": sum(
            x["rubric-first"]["correct"] and not x["state-first"]["correct"] for x in samples
        ),
        "max_abs_probability_change": max(
            abs(p - x["rubric-first"]["probabilities"][key])
            for x in samples
            for key, p in x["state-first"]["probabilities"].items()
        ),
        "rubric_minus_state": {
            metric: result["rubric-first"]["overall"][metric]
            - result["state-first"]["overall"][metric]
            for metric in (
                "accuracy",
                "negative_log_likelihood",
                "multiclass_brier",
                "top_label_ece",
            )
        },
    }
    return result
