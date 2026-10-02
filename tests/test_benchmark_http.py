import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

spec = importlib.util.spec_from_file_location(
    "benchmark_http", Path(__file__).parents[1] / "scripts/benchmark_http.py"
)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
SUCCESS = {"queue_wait_ms": 1, "performance": {"group_requests": 1, "batch_sizes": [1]}}


def arguments(tmp_path, requests):
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"state": "fixture"}))
    return SimpleNamespace(
        request=request,
        output=tmp_path / "results.json",
        url="http://fake",
        distinct=True,
        concurrency=[3],
        requests=requests,
    )


def test_mixed_http_and_transport_outcomes_are_retained(tmp_path):
    args = arguments(tmp_path, 6)

    def server(request):
        if request.url.path == "/ready" or json.loads(request.content)["state"] == "fixture":
            return httpx.Response(200, json=SUCCESS)
        index = int(json.loads(request.content)["state"].split()[1][:-1])
        if index == 4:
            return httpx.Response(200, text="invalid JSON")
        if index == 5:
            raise httpx.ReadTimeout("fixture timeout", request=request)
        return httpx.Response([200, 429, 504, 200][index], json=SUCCESS)

    report = asyncio.run(benchmark.run(args, transport=httpx.MockTransport(server)))
    assert json.loads(args.output.read_text()) == report
    assert report["complete"]
    result = report["results"][0]
    assert result["attempted_requests"] == 6
    assert result["successful_requests"] == 2
    assert result["failed_requests"] == 4
    assert result["requests_per_second"] == pytest.approx(
        result["attempted_requests_per_second"] / 3
    )
    assert result["status_counts"] == {"200": 3, "429": 1, "504": 1, "None": 1}
    assert result["error_counts"] == {"http_status": 2, "invalid_response": 1, "timeout": 1}
    assert [s["request_index"] for s in result["samples"]] == list(range(6))
    assert all(s["wall_ms"] >= 0 for s in result["samples"])
    latencies = [s["wall_ms"] for s in result["samples"] if s["success"]]
    assert result["p50_request_ms"] == sum(latencies) / 2
    assert result["p95_request_ms"] == max(latencies)


def test_all_failed_requests_have_no_success_latency(tmp_path):
    args = arguments(tmp_path, 2)

    def server(request):
        if request.url.path == "/ready" or json.loads(request.content)["state"] == "fixture":
            return httpx.Response(200, json=SUCCESS)
        return httpx.Response(429)

    report = asyncio.run(benchmark.run(args, transport=httpx.MockTransport(server)))
    result = report["results"][0]
    assert result["failed_requests"] == 2
    assert result["requests_per_second"] == 0
    assert result["error_rate"] == 1
    assert result["p50_request_ms"] is result["p95_request_ms"] is None


def test_warmup_failure_writes_incomplete_report(tmp_path):
    args = arguments(tmp_path, 2)

    def server(request):
        return httpx.Response(200 if request.url.path == "/ready" else 504)

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(benchmark.run(args, transport=httpx.MockTransport(server)))
    report = json.loads(args.output.read_text())
    assert not report["complete"]
    assert report["phase"] == "warm-up"
    assert report["failure_type"] == "HTTPStatusError"
    assert report["results"] == []


def test_partial_measurement_survives_unexpected_failure(tmp_path):
    args = arguments(tmp_path, 2)

    def server(request):
        if request.url.path == "/ready" or json.loads(request.content)["state"] == "fixture":
            return httpx.Response(200, json=SUCCESS)
        if json.loads(request.content)["state"].startswith("Request 1:"):
            raise RuntimeError("fixture failure")
        return httpx.Response(200, json=SUCCESS)

    with pytest.raises(RuntimeError):
        asyncio.run(benchmark.run(args, transport=httpx.MockTransport(server)))
    report = json.loads(args.output.read_text())
    assert not report["complete"]
    result = report["results"][0]
    assert not result["complete"]
    assert result["successful_requests"] == result["failed_requests"] == 1
    assert result["error_counts"] == {"unexpected_error": 1}
    assert result["samples"][0]["success"]


def test_partial_report_drains_inflight_attempts_before_saving(tmp_path):
    args = arguments(tmp_path, 3)

    async def server(request):
        if request.url.path == "/ready" or json.loads(request.content)["state"] == "fixture":
            return httpx.Response(200, json=SUCCESS)
        state = json.loads(request.content)["state"]
        if state.startswith("Request 0:"):
            return httpx.Response(200, json=SUCCESS)
        if state.startswith("Request 1:"):
            await asyncio.sleep(0)
            raise RuntimeError("fixture failure")
        await asyncio.sleep(10)
        return httpx.Response(200, json=SUCCESS)

    with pytest.raises(RuntimeError):
        asyncio.run(benchmark.run(args, transport=httpx.MockTransport(server)))
    report = json.loads(args.output.read_text())
    result = report["results"][0]
    assert result["attempted_requests"] == 3
    assert result["successful_requests"] == 1
    assert result["error_counts"] == {"unexpected_error": 1, "interrupted": 1}
