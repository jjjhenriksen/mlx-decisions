import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from mlx_decisions.protocol import DecisionRequest
from mlx_decisions.server import Batcher, create_app

BODY = {"state": "a", "questions": {"x": {"type": "noul", "instructions": "true?"}}}


@dataclass(frozen=True)
class FakeRow:
    request: int
    state: str
    key: str


class FakeEngine:
    model_id = "fake"

    def __init__(self, delay=0):
        self.batches = []
        self.threads = set()
        self.delay = delay
        self.prepared_states = []
        self.executed_rows = []

    def prepare(self, requests):
        self.prepared_states.extend(r.state for r in requests)
        if any(r.state == "overlong" for r in requests):
            raise ValueError("too many prompt tokens")

        return [FakeRow(i, r.state, key) for i, r in enumerate(requests) for key in r.questions]

    def decide_prepared(self, rows, *, request_count):
        self.threads.add(threading.get_ident())
        self.batches.append(request_count)
        self.executed_rows.append(rows)
        time.sleep(self.delay)
        return [
            {"answers": {"x": {"type": "noul", "noul": 0.7}}, "state": r.state}
            for r in [next(row for row in rows if row.request == i) for i in range(request_count)]
        ]


def test_http_validation_health_and_body_limit():
    engine = FakeEngine()
    with TestClient(create_app(lambda: engine, max_body_bytes=512)) as client:
        assert client.get("/ready").status_code == 200
        assert client.get("/health").status_code == 200
        assert client.get("/v1/models").json()["data"][0]["id"] == "fake"
        response = client.post("/v1/systemone", json=BODY)
        assert response.status_code == 200
        assert response.json()["answers"]["x"]["noul"] == 0.7
        assert response.json()["request_wall_ms"] >= response.json()["queue_wait_ms"]
        assert client.post("/v1/systemone", json={}).status_code == 422
        assert client.post("/v1/systemone", content=b"x" * 513).status_code == 413


def test_concurrent_requests_coalesce_and_bad_request_isolated():
    async def scenario():
        engine = FakeEngine()
        app = create_app(lambda: engine, window_ms=15)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                replies = await asyncio.gather(
                    *[
                        client.post("/v1/systemone", json={**BODY, "state": state})
                        for state in ["a", "b", "overlong", "c"]
                    ]
                )
                assert [r.status_code for r in replies] == [200, 200, 422, 200]
                assert engine.batches == [3]
                assert len(engine.threads) == 1
                assert [r.json()["state"] for r in replies if r.status_code == 200] == [
                    "a",
                    "b",
                    "c",
                ]

    asyncio.run(scenario())


def test_queue_full_timeout_and_shutdown():
    async def scenario():
        executor = ThreadPoolExecutor(max_workers=1)
        worker = Batcher(FakeEngine(delay=0.05), executor, capacity=1, window_ms=30, timeout=0.01)
        # A non-running-but-live task makes admission deterministic for queue-full.
        worker.task = asyncio.create_task(asyncio.sleep(1))
        first = asyncio.create_task(worker.submit(DecisionRequest(**BODY)))
        await asyncio.sleep(0)
        with pytest.raises(HTTPException) as full:
            await worker.submit(DecisionRequest(**BODY))
        assert full.value.status_code == 429
        with pytest.raises(HTTPException) as deadline:
            await first
        assert deadline.value.status_code == 504
        worker.task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker.task
        worker.task = asyncio.create_task(worker.run())
        await worker.close()
        executor.shutdown()

    asyncio.run(scenario())


def test_shutdown_during_batch_window():
    async def scenario():
        executor = ThreadPoolExecutor(max_workers=1)
        worker = Batcher(FakeEngine(), executor, window_ms=40)
        worker.task = asyncio.create_task(worker.run())
        caller = asyncio.create_task(worker.submit(DecisionRequest(**BODY)))
        await asyncio.sleep(0.01)
        await worker.close()
        assert (await caller)["answers"]["x"]["noul"] == 0.7
        executor.shutdown()

    asyncio.run(scenario())


def test_unexpected_preparation_error_isolated_and_worker_survives():
    class FailingEngine(FakeEngine):
        def prepare(self, requests):
            if any(r.state == "broken" for r in requests):
                raise RuntimeError("private tokenizer details")
            return super().prepare(requests)

    async def scenario():
        engine = FailingEngine()
        app = create_app(lambda: engine, window_ms=15, timeout=0.5)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                replies = await asyncio.gather(
                    *[
                        client.post("/v1/systemone", json={**BODY, "state": state})
                        for state in ["a", "broken", "overlong", "b"]
                    ]
                )
                assert [r.status_code for r in replies] == [200, 500, 422, 200]
                assert replies[1].json() == {"detail": "inference failed"}
                assert engine.batches == [2]
                assert (await client.get("/ready")).status_code == 200
                later = await client.post("/v1/systemone", json={**BODY, "state": "later"})
                assert later.status_code == 200
                assert later.json()["state"] == "later"
                await asyncio.wait_for(app.state.batcher.queue.join(), timeout=0.5)
                assert not app.state.batcher.task.done()

    asyncio.run(scenario())


def test_preparation_reused_with_compact_request_and_question_indexes():
    async def scenario():
        engine = FakeEngine()
        app = create_app(lambda: engine, window_ms=15)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                bodies = [
                    {
                        **BODY,
                        "state": "first",
                        "questions": {"x": BODY["questions"]["x"], "y": BODY["questions"]["x"]},
                    },
                    {**BODY, "state": "overlong"},
                    {**BODY, "state": "last"},
                ]
                replies = await asyncio.gather(
                    *(client.post("/v1/systemone", json=body) for body in bodies)
                )
                assert [r.status_code for r in replies] == [200, 422, 200]
                assert engine.prepared_states == ["first", "overlong", "last"]
                assert [(r.request, r.state, r.key) for r in engine.executed_rows[0]] == [
                    (0, "first", "x"),
                    (0, "first", "y"),
                    (1, "last", "x"),
                ]
                assert [r.json()["state"] for r in replies if r.status_code == 200] == [
                    "first",
                    "last",
                ]
                assert len(engine.threads) == 1

    asyncio.run(scenario())
