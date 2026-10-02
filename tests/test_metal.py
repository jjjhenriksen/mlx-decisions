import copy

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
if not mx.metal.is_available():
    pytest.skip("Metal GPU required", allow_module_level=True)

from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.qwen3_5 import Model, ModelArgs
from mlx_lm.models.qwen3_next import Qwen3NextMLP

from mlx_decisions.engine import Engine
from mlx_decisions.optimizations import FusedGateUp, SelectedHead, evaluate_cache, fork_cache
from mlx_decisions.protocol import DecisionRequest

pytestmark = pytest.mark.metal


class TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 256 for c in text]

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return self.encode("<user>" + messages[0]["content"] + "</user><assistant>")


def tiny_engine(quantized=False, *, rubric_first=False):
    mx.random.seed(42)
    args = ModelArgs(
        model_type="qwen3_5",
        text_config={
            "model_type": "qwen3_5_text",
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "vocab_size": 256,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 32,
            "full_attention_interval": 4,
        },
    )
    model = Model(args)
    if quantized:
        nn.quantize(model, group_size=64, bits=8)
    model.eval()
    mx.eval(model.parameters())
    engine = Engine.__new__(Engine)
    engine.model, engine.tokenizer = model, TinyTokenizer()
    engine.model_id, engine.revision = "tiny-test", None
    engine._configure(4, 4096, 2048, 64, 100 * 1024**2, 2, True, True, rubric_first)
    return engine


@pytest.mark.parametrize("bits", [None, 4, 8])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_selected_head_matches_full_projection(bits, dtype):
    mx.random.seed(9)
    head = nn.Linear(128, 256, bias=True)
    if bits:
        head = nn.QuantizedLinear.from_linear(head, group_size=64, bits=bits)
    head.set_dtype(dtype)
    selected = SelectedHead(head, [0, 2, 51, 120, 255])
    x = mx.random.normal((4, 128)).astype(dtype)
    actual, expected = selected(x), head(x)[:, [0, 2, 51, 120, 255]]
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=0.01 if dtype == mx.bfloat16 else 1e-5).item()


@pytest.mark.parametrize("bits", [None, 8])
def test_gate_up_fusion_parity(bits):
    mx.random.seed(5)
    mlp = Qwen3NextMLP(128, 256)
    if bits:
        nn.quantize(mlp, group_size=64, bits=bits)
    fused = FusedGateUp(mlp)
    for shape in [(1, 1, 128), (2, 17, 128)]:
        x = mx.random.normal(shape)
        actual, expected = fused(x), mlp(x)
        mx.eval(actual, expected)
        assert mx.allclose(actual, expected, atol=2e-5).item()


def requests():
    return [
        DecisionRequest(
            state="Shared state. " * 5,
            questions={
                "one": {
                    "type": "choice",
                    "instructions": "Pick x",
                    "criteria": {"a": None, "b": None},
                },
                "two": {
                    "type": "choice",
                    "instructions": "Pick y",
                    "criteria": {"a": None, "b": None},
                },
                "longer": {"type": "noul", "instructions": "Is the statement true?"},
            },
        ),
        DecisionRequest(
            state="Other state.",
            questions={
                "score": {"type": "score", "instructions": "Rate", "criteria": ["low", "high"]},
            },
        ),
    ]


@pytest.mark.parametrize("quantized", [False, True])
def test_hybrid_qwen_prefix_batch_and_cache_isolation(quantized):
    engine = tiny_engine(quantized)
    rows = engine.prepare(requests())
    reference, _ = engine.raw_scores(rows, reference=True)
    actual, metrics = engine.raw_scores(rows)
    assert 2 in metrics["batch_sizes"]
    assert metrics["prefix_misses"] == 2
    for key in reference:
        assert mx.allclose(mx.array(reference[key]), mx.array(actual[key]), atol=2e-4).item()
    # Repeated and reordered questions must not mutate a cached recurrent branch.
    warm, metrics = engine.raw_scores(list(reversed(rows)))
    assert metrics["prefix_hits"] == 2
    for key in actual:
        assert mx.allclose(mx.array(warm[key]), mx.array(actual[key]), atol=2e-4).item()
    engine.prefix_reuse = False
    fresh, _ = engine.raw_scores(rows)
    for key in reference:
        assert mx.allclose(mx.array(fresh[key]), mx.array(reference[key]), atol=2e-4).item()


def test_cache_forks_copy_kv_and_recurrent_states():
    engine = tiny_engine()
    cache = make_prompt_cache(engine.model)
    engine._prefill(tuple(range(40)), cache)
    saved = copy.deepcopy(cache)
    fork = fork_cache(cache, 2)
    mx.eval(engine.trunk(mx.array([[40, 41], [42, 43]]), cache=fork))
    evaluate_cache(fork)
    for original, snapshot in zip(cache, saved):
        from mlx.utils import tree_flatten

        for (_, a), (_, b) in zip(tree_flatten(original.state), tree_flatten(snapshot.state)):
            if isinstance(a, mx.array):
                assert mx.array_equal(a, b).item()
            else:
                assert a == b


def test_cache_eviction_and_prompt_limit():
    engine = tiny_engine()
    engine.prefix_cache_entries = 1
    engine.raw_scores(engine.prepare(requests()))
    assert len(engine._prefixes) == 1
    engine.max_prompt_tokens = 5
    with pytest.raises(ValueError, match="prompt tokens"):
        engine.prepare(requests())


@pytest.mark.parametrize("quantized", [False, True])
def test_rubric_first_reuses_across_states_and_isolates_rubrics(quantized):
    engine = tiny_engine(quantized, rubric_first=True)
    question = requests()[0].questions["one"]
    cases = [
        DecisionRequest(state=state, questions={"route": question})
        for state in ["State one", "State two", "A longer state"]
    ]
    rows = engine.prepare(cases)
    assert len({r.prefix for r in rows}) == 1
    assert len({r.tokens for r in rows}) == 3
    assert all(r.tokens[: len(r.prefix)] == r.prefix for r in rows)
    assert all(0 < len(r.prefix) < len(r.tokens) for r in rows)
    reference, _ = engine.raw_scores(rows, reference=True)
    actual, metrics = engine.raw_scores(rows)
    assert metrics["prefix_misses"] == 1
    assert 2 in metrics["batch_sizes"]
    for key in reference:
        assert mx.allclose(mx.array(reference[key]), mx.array(actual[key]), atol=2e-4).item()
    # New state on a later call must hit the rubric, without stale recurrent state.
    new = DecisionRequest(state="Entirely new state", questions={"route": question})
    new_rows = engine.prepare([new])
    new_reference, _ = engine.raw_scores(new_rows, reference=True)
    warm, metrics = engine.raw_scores(new_rows)
    assert metrics["prefix_hits"] == 1
    assert metrics["prefill_tokens"] == len(new_rows[0].tokens) - len(new_rows[0].prefix)
    assert mx.allclose(
        mx.array(new_reference[(0, "route")]), mx.array(warm[(0, "route")]), atol=2e-4
    ).item()
    # Either changed instructions or changed criteria must miss the old rubric.
    for update in [{"instructions": "Pick z"}, {"criteria": {"a": "new meaning", "b": None}}]:
        changed = DecisionRequest(
            state=new.state, questions={"route": question.model_copy(update=update)}
        )
        changed_rows = engine.prepare([changed])
        assert changed_rows[0].prefix != new_rows[0].prefix
        _, metrics = engine.raw_scores(changed_rows)
        assert metrics["prefix_hits"] == 0
        assert metrics["prefix_misses"] == 1
    engine.prefix_reuse = False
    fresh, metrics = engine.raw_scores(rows)
    assert metrics["prefix_hits"] == metrics["prefix_misses"] == 0
    for key in reference:
        assert mx.allclose(mx.array(reference[key]), mx.array(fresh[key]), atol=2e-4).item()
    assert engine.decide(new)["prompt_order"] == "rubric-first"
    engine.rubric_first = False
    assert engine.decide(new)["prompt_order"] == "state-first"


@pytest.mark.parametrize("rubric_first", [False, True])
def test_benchmark_distinguishes_order_drift_from_cache_parity(rubric_first, monkeypatch, tmp_path):
    import importlib.util
    import json
    import sys
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "benchmark", Path(__file__).parents[1] / "scripts/benchmark.py"
    )
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    monkeypatch.setattr(
        benchmark, "Engine", lambda *a, **kw: tiny_engine(rubric_first=kw["rubric_first"])
    )
    question = requests()[0].questions["one"]
    monkeypatch.setattr(
        benchmark,
        "workloads",
        lambda: {
            "shared_rubric": [
                DecisionRequest(state=s, questions={"q": question}) for s in ["first", "other"]
            ],
        },
    )
    output = tmp_path / "benchmark.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "--repeats",
            "1",
            "--output",
            str(output),
            *(["--rubric-first"] if rubric_first else []),
        ],
    )
    benchmark.main()
    report = json.loads(output.read_text())
    assert report["complete"] is True
    assert report["parity_passed"] is True
    assert report["prompt_order"] == ("rubric-first" if rubric_first else "state-first")
    expected = "rubric_first_full_forward" if rubric_first else "official_full_forward"
    assert report["results"][0]["variant"] == expected
    for result in report["results"]:
        assert len(result["raw_score_samples"]) == 1
        assert len(result["raw_score_samples"][0]) == 2
        assert result["parity"]["passed"]
        assert result["speedup_vs_full_forward"] > 0
        assert (result["prompt_order_drift_vs_state_first"] is not None) is rubric_first
        if rubric_first:
            assert result["speedup_vs_official"] is None


def test_prepared_execution_matches_direct_and_does_not_tokenize(monkeypatch):
    engine = tiny_engine()
    cases = requests()
    direct = engine.decide_many(cases)
    rows = engine.prepare(cases)
    engine.clear_cache()

    def unexpected_prepare(*args, **kwargs):
        raise AssertionError("prepared execution must not tokenize again")

    monkeypatch.setattr(engine, "prepare", unexpected_prepare)
    prepared = engine.decide_prepared(rows, request_count=len(cases))
    for actual, expected in zip(prepared, direct, strict=True):
        assert actual["answers"] == expected["answers"]
        assert actual["usage"] == expected["usage"]
        assert actual["performance"]["group_requests"] == len(cases)
    with pytest.raises(ValueError, match="every request index"):
        engine.decide_prepared(rows, request_count=len(cases) + 1)


def test_benchmark_retains_each_probability_error_and_flip():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        'benchmark', Path(__file__).parents[1] / 'scripts/benchmark.py'
    )
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    engine = tiny_engine()
    rows = engine.prepare(requests())[:1]
    row = rows[0]
    key = (row.request, row.key)
    samples = benchmark.raw_comparison({key: [2.0, 0.0]}, {key: [0.0, 2.0]}, rows)
    assert samples[0]['decision_flip']
    assert samples[0]['reference_logits'] == [2.0, 0.0]
    assert samples[0]['actual_decision'] == 'b'
    assert max(samples[0]['absolute_probability_errors']) > 0.005


def test_benchmark_marks_loader_failure_incomplete(monkeypatch, tmp_path):
    import importlib.util
    import json
    import sys
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        'benchmark', Path(__file__).parents[1] / 'scripts/benchmark.py'
    )
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    output = tmp_path / 'failed.json'
    monkeypatch.setattr(sys, 'argv', ['benchmark.py', '--output', str(output)])

    def fail(*args, **kwargs):
        raise RuntimeError('cannot load checkpoint')

    monkeypatch.setattr(benchmark, 'Engine', fail)
    with pytest.raises(RuntimeError, match='cannot load'):
        benchmark.main()
    report = json.loads(output.read_text())
    assert report['complete'] is False
    assert report['phase'] == 'loading'
    assert report['error']['type'] == 'RuntimeError'
    assert report['results'] == []


def test_benchmark_retains_measured_repeat_before_interruption(monkeypatch, tmp_path):
    import importlib.util
    import json
    import sys
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        'benchmark', Path(__file__).parents[1] / 'scripts/benchmark.py'
    )
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    engine = tiny_engine()
    original = engine.raw_scores
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:  # reference, warm-up, measured repeat, interrupted repeat
            raise RuntimeError('measurement interrupted')
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, 'raw_scores', interrupted)
    monkeypatch.setattr(benchmark, 'Engine', lambda *args, **kwargs: engine)
    monkeypatch.setattr(benchmark, 'workloads', lambda: {'single': requests()[:1]})
    output = tmp_path / 'interrupted.json'
    monkeypatch.setattr(sys, 'argv', ['benchmark.py', '--repeats', '2', '--output', str(output)])
    with pytest.raises(RuntimeError, match='measurement interrupted'):
        benchmark.main()
    report = json.loads(output.read_text())
    assert report['complete'] is False
    assert report['error']['type'] == 'RuntimeError'
    assert len(report['pending_case']['group_wall_ms_samples']) == 1
    assert len(report['pending_case']['raw_score_samples']) == 1
    assert report['results'] == []
