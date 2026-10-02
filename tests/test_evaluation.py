import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx_decisions.evaluation import distribution, load_cases, metrics, observation, summarize
from mlx_decisions.protocol import Question

ROOT = Path(__file__).parents[1]


def fixture():
    return json.loads((ROOT / "benchmarks/heldout/policy-decisions-v1.json").read_text())


def test_heldout_integrity_and_class_coverage():
    cases = load_cases(fixture())
    assert len(cases) == 72
    for kind in ("choice", "noul", "score"):
        selected = [(c, q) for c, _, q in cases if q.type == kind]
        assert len(selected) == 24
        assert {c["label"] for c, q in selected} == {
            key for _, q in selected for key, _ in q.options()
        }


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda d: d.update(split="tuning"), "held-out"),
        (lambda d: d["cases"].append(d["cases"][0]), "duplicate"),
        (lambda d: d["cases"][0].update(label="unlisted"), "option set"),
    ],
)
def test_dataset_validation(mutation, match):
    data = fixture()
    mutation(data)
    with pytest.raises(ValueError, match=match):
        load_cases(data)


def test_calibrated_distribution_matches_public_readout():
    for kind in ("choice", "noul", "score"):
        q = Question(
            type=kind,
            instructions="Choose",
            criteria={"a": None, "b": None}
            if kind == "choice"
            else ["low", "high"]
            if kind == "score"
            else None,
        )
        for logits in ([0, 0], [1.7, -0.3], [-1000, 1000]):
            p = distribution(q, logits)
            answer = q.answer(logits)
            expected = (
                [answer["noul"], 1 - answer["noul"]]
                if kind == "noul"
                else list(answer["probabilities"].values())
            )
            assert p == pytest.approx(expected, abs=0.000051)
        with pytest.raises(ValueError, match="nonfinite"):
            distribution(q, [float("nan"), 0])


def test_quality_metrics_known_probabilities_and_ece_boundaries():
    q = Question(type="choice", instructions="Choose", criteria={"a": None, "b": None})
    case = {"label": "a"}
    obs = observation(case, q, [0, 0])
    result = metrics([obs])
    assert result["accuracy"] == 1
    assert result["negative_log_likelihood"] == pytest.approx(math.log(2))
    assert result["multiclass_brier"] == 0.5
    assert result["top_label_ece"] == 0.5
    confident = dict(obs, top_probability=1.0)
    assert metrics([confident])["reliability_bins"][-1]["count"] == 1
    with pytest.raises(ValueError):
        metrics([])


def test_paired_flips_and_ordinal_expected_error():
    q = Question(type="score", instructions="Rate", criteria=["low", "high"])
    a = observation({"label": "0"}, q, [1.0, 0])
    b = observation({"label": "0"}, q, [0, 1.0])
    result = summarize([dict(type="score", family="test", **{"state-first": a, "rubric-first": b})])
    assert result["paired"]["decision_flips"] == 1
    assert result["paired"]["state_only_correct"] == 1
    assert result["paired"]["rubric_minus_state"]["accuracy"] == -1
    assert (
        result["state-first"]["overall"]["score_mean_absolute_level_error"] == a["expected_level"]
    )


def evaluator():
    spec = importlib.util.spec_from_file_location(
        "evaluate_prompt_order", ROOT / "scripts/evaluate_prompt_order.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeEngine:
    rubric_first = False

    def __init__(self, fail=False):
        self.orders = []
        self.fail = fail

    def clear_cache(self):
        pass

    def prepare(self, requests):
        assert len(requests) == 1
        q = next(iter(requests[0].questions.values()))
        return [SimpleNamespace(key="decision", question=q, tokens=(1, 2, 3))]

    def raw_scores(self, rows, *, reference):
        assert reference is True
        self.orders.append(self.rubric_first)
        if self.fail and len(self.orders) == 2:
            raise RuntimeError("second arm failed")
        return {(0, "decision"): [0.0] * len(rows[0].question.options())}, {}


def test_runner_alternates_full_forward_and_retains_partial_pair(tmp_path):
    module = evaluator()
    cases = load_cases(fixture())[:2]
    engine = FakeEngine()
    report = {"complete": False, "samples": []}
    module.evaluate(cases, engine, report, tmp_path / "complete.json")
    assert engine.orders == [False, True, True, False]
    assert report["complete"]
    assert len(report["samples"]) == 2
    partial = {"complete": False, "samples": []}
    output = tmp_path / "partial.json"
    with pytest.raises(RuntimeError, match="second arm"):
        module.evaluate(cases, FakeEngine(fail=True), partial, output)
    saved = json.loads(output.read_text())
    assert not saved["complete"]
    assert saved["error"]["type"] == "RuntimeError"
    assert "state-first" in saved["pending_sample"]
    assert "rubric-first" not in saved["pending_sample"]
