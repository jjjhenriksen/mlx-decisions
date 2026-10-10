import json
import sys
from types import SimpleNamespace

import pytest

from mlx_decisions.cli import main


@pytest.mark.parametrize("rubric_first", [False, True])
@pytest.mark.parametrize("command", ["decide", "serve"])
def test_prompt_order_flag_reaches_engine(rubric_first, command, monkeypatch, tmp_path):
    seen = {}

    def engine(model, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(decide=lambda request: {"answers": {}})

    monkeypatch.setitem(sys.modules, "mlx_decisions.engine", SimpleNamespace(Engine=engine))
    monkeypatch.setitem(
        sys.modules,
        "mlx_decisions.server",
        SimpleNamespace(create_app=lambda factory, **kwargs: factory()),
    )
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=lambda *a, **kw: None))
    argv = ["mlx-decisions", *(["--rubric-first"] if rubric_first else []), command]
    if command == "decide":
        path = tmp_path / "request.json"
        path.write_text(
            json.dumps(
                {
                    "state": "test",
                    "questions": {"q": {"type": "noul", "instructions": "True?"}},
                }
            )
        )
        argv.append(str(path))
    monkeypatch.setattr(sys, "argv", argv)
    main()
    assert seen["rubric_first"] is rubric_first
    assert seen["prefix_reuse"] is True


@pytest.mark.parametrize("rubric_first", [False, True])
def test_inspect_reports_prompts_and_option_order_without_engine(
    rubric_first, monkeypatch, tmp_path, capsys
):
    path = tmp_path / "request.json"
    path.write_text(json.dumps({
        "state": {"text": "A café invoice"},
        "questions": {
            "route": {"type": "choice", "instructions": "Choose a team", "criteria": {
                "billing": None, "support": "technical help",
            }},
            "priority": {"type": "score", "instructions": "Rate urgency", "criteria": [
                "low", "high",
            ]},
            "urgent": {"type": "noul", "instructions": "Must act now?"},
        },
    }), encoding="utf-8")
    monkeypatch.setitem(sys.modules, "mlx_decisions.engine", None)
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    monkeypatch.setattr(sys, "argv", [
        "mlx-decisions", *(["--rubric-first"] if rubric_first else []), "inspect", str(path),
    ])
    main()
    report = json.loads(capsys.readouterr().out)
    assert report["prompt_order"] == ("rubric-first" if rubric_first else "state-first")
    assert report["question_count"] == 3
    route = report["questions"]["route"]
    rubric = "Question: Choose a team\nOptions:\n[A] billing: \n[B] support: technical help"
    state = 'State:\n{"text": "A café invoice"}'
    blocks = [rubric, state] if rubric_first else [state, rubric]
    assert route["prompt"] == "\n\n".join([
        *blocks, "Answer with the letter of the best option only.",
    ])
    assert route["options"] == [
        {"letter": "A", "label": "billing", "description": ""},
        {"letter": "B", "label": "support", "description": "technical help"},
    ]
    assert [option["label"] for option in report["questions"]["priority"]["options"]] == ["0", "1"]
    assert [option["label"] for option in report["questions"]["urgent"]["options"]] == ["yes", "no"]


@pytest.mark.parametrize("content", [None, "{broken", '{"state":"x","questions":{}}'])
def test_inspect_rejects_unreadable_or_invalid_requests_before_engine(
    content, monkeypatch, tmp_path, capsys
):
    path = tmp_path / "request.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    monkeypatch.setitem(sys.modules, "mlx_decisions.engine", None)
    monkeypatch.setattr(sys, "argv", ["mlx-decisions", "inspect", str(path)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Cannot read decision request" in captured.err
