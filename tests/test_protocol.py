import ast
import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from mlx_decisions.protocol import LETTERS, DecisionRequest, Question


def reference_namespace():
    # Compile only pure prompt/readout functions. Never import upstream's server
    # or construct its OpenAI client. The frozen original is retained for audit.
    source = Path(__file__).parent / "fixtures/openjev_shim.py"
    names = {
        "_instr",
        "_desc",
        "choice_confidence",
        "score_confidence",
        "answer_choice",
        "answer_score",
        "answer_noul",
        "_readout_once",
        "prefix_text",
    }
    nodes = [
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    import os

    scope = {
        "json": json,
        "math": math,
        "os": os,
        "BadQuestion": ValueError,
        "NOUL_T": 1.829074,
        "NOUL_BIAS": 0,
        "TEMP": 0.85,
        "LETTERS": LETTERS,
        "pad_prefix": lambda x: x,
        "TARGETED": True,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    return scope


@pytest.mark.parametrize(
    "data",
    [
        {
            "type": "choice",
            "instructions": "Which team?",
            "criteria": {
                "billing": None,
                "shipping": "physical parcels",
                "technical": {"topic": "login"},
            },
        },
        {
            "type": "score",
            "instructions": {"goal": "assess priority"},
            "criteria": ["low", "middle", "high"],
        },
        {"type": "noul", "instructions": "Is it urgent?"},
        {
            "type": "noul",
            "instructions": "Is it urgent?",
            "criteria": {"true": "must act now", "false": "can wait"},
        },
        {"type": "choice", "instructions": "Choose", "criteria": {"only": None}},
    ],
)
def test_exact_upstream_prompts_and_calibration(data, monkeypatch):
    monkeypatch.setenv("READOUT_INSTR_STYLE", "pyrepr")
    ns = reference_namespace()
    q = Question.model_validate(data)
    raw = [1.3 - i * 0.8 for i in range(len(q.options()))]
    seen = {}

    def distribution(state, instructions, opts):
        seen.update(state=state, instructions=instructions, opts=opts)
        e = [math.exp(x / 0.85) for x in raw]
        return [v / sum(e) for v in e], 100

    ns["_distribution"] = distribution
    expected, _ = ns["answer_" + q.type]("some state", data)
    assert q.answer(raw) == expected
    prompt_lines = "\n".join(f"[{LETTERS[i]}] {k}: {d}" for i, (k, d) in enumerate(seen["opts"]))
    expected_prompt = f"State:\nsome state\n\nQuestion: {seen['instructions']}\nOptions:\n{prompt_lines}\n\nAnswer with the letter of the best option only."
    assert q.prompt("some state") == expected_prompt
    assert q.prompt("some state", rubric_first=False) == expected_prompt
    rubric = f"Question: {seen['instructions']}\nOptions:\n{prompt_lines}"
    assert q.prompt("some state", rubric_first=True) == (
        f"{rubric}\n\nState:\nsome state\n\nAnswer with the letter of the best option only."
    )
    assert q.prefix("some state", rubric_first=True) == rubric
    assert q.prefix("some state") == "State:\nsome state"


def test_selected_logits_equal_full_logprob_readout():
    q = Question(type="choice", instructions="Choose", criteria={"x": None, "y": None})
    assert q.answer([8.0, 6.0]) == q.answer([-1.3, -3.3])
    with pytest.raises(ValueError):
        q.answer([float("nan"), 0])


@pytest.mark.parametrize(
    "q",
    [
        {"type": "choice", "instructions": "q", "criteria": {}},
        {"type": "choice", "instructions": "q", "criteria": {str(i): None for i in range(53)}},
        {"type": "score", "instructions": "q", "criteria": ["one"]},
        {"type": "noul", "instructions": "q", "criteria": ["bad"]},
        {"type": "noul", "instructions": "q", "criteria": {"yes": None}},
    ],
)
def test_invalid_questions_rejected(q):
    with pytest.raises(ValidationError):
        Question.model_validate(q)


def test_state_and_image_contract():
    q = {"x": {"type": "noul", "instructions": "q"}}
    r = DecisionRequest(state={"text": "你好", "count": 2}, questions=q)
    assert r.state_text() == '{"text": "你好", "count": 2}'
    with pytest.raises(ValidationError):
        DecisionRequest(state={"image": "anything"}, questions=q)
    with pytest.raises(ValidationError):
        DecisionRequest(state="x", questions={})


def test_frozen_upstream_fixture_digest():
    import hashlib

    source = Path(__file__).parent / "fixtures/openjev_shim.py"
    assert (
        hashlib.sha256(source.read_bytes()).hexdigest()
        == "81a22f1b1b8912a465059207ef9f60b7c6c16b4de6372305d867efbe38a1987a"
    )
