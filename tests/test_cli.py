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
