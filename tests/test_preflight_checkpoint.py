import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "preflight_checkpoint", Path(__file__).parents[1] / "scripts/preflight_checkpoint.py"
)
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)
GIB = 1024**3


def test_complete_weights_still_need_runtime_headroom():
    shards = [dict(size_matches=True)]
    blocked = preflight.assess(shards, 27 * GIB, 36 * GIB, 0, 28 * GIB, 4 * GIB)
    assert len(blocked) == 1
    assert "working set" in blocked[0]
    assert preflight.assess(shards, 27 * GIB, 36 * GIB, 0, 48 * GIB, 4 * GIB) == []


def test_missing_shards_and_unknown_resources_fail_closed():
    blockers = preflight.assess(
        [dict(size_matches=False)], 27 * GIB, 1 * GIB, 5 * GIB, None, 4 * GIB
    )
    assert any("shards" in b for b in blockers)
    assert any("storage" in b for b in blockers)
    assert any("unavailable" in b for b in blockers)
    assert preflight.assess([], None, 36 * GIB, 0, None, 4 * GIB)
