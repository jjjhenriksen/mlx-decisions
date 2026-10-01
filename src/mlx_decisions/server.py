"""Bounded concurrent HTTP admission and cross-request microbatching."""

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Request
from pydantic import ValidationError

from .protocol import MODEL_ID, DecisionRequest


@dataclass
class Pending:
    request: DecisionRequest
    future: asyncio.Future
    admitted: float


class Batcher:
    def __init__(self, engine, executor, *, capacity=32, max_requests=8, window_ms=4, timeout=300):
        if min(capacity, max_requests, timeout) <= 0 or window_ms < 0:
            raise ValueError("invalid scheduler limits")
        self.engine, self.executor = engine, executor
        self.queue = asyncio.Queue(maxsize=capacity)
        self.max_requests, self.window = max_requests, window_ms / 1000
        self.timeout = timeout
        self.task = None
        self.closing = False

    async def submit(self, request):
        if self.closing or self.task is None or self.task.done():
            raise HTTPException(503, "inference worker unavailable")
        future = asyncio.get_running_loop().create_future()
        try:
            self.queue.put_nowait(Pending(request, future, time.perf_counter()))
        except asyncio.QueueFull:
            raise HTTPException(429, "inference queue full", headers={"Retry-After": "1"}) from None
        try:
            return await asyncio.wait_for(future, self.timeout)
        except TimeoutError:
            raise HTTPException(504, "decision deadline exceeded") from None

    def _evaluate(self, group):
        # A malformed/overlong request must not poison unrelated queued clients.
        good, outputs = [], [None] * len(group)
        for i, item in enumerate(group):
            try:
                self.engine.prepare([item.request])
                good.append((i, item.request))
            except ValueError as error:
                outputs[i] = error
        if good:
            try:
                results = self.engine.decide_many([request for _, request in good])
                for (i, _), result in zip(good, results, strict=True):
                    outputs[i] = result
            except Exception as error:
                for i, _ in good:
                    outputs[i] = error
        return outputs

    async def run(self):
        while True:
            item = await self.queue.get()
            if item is None:
                self.queue.task_done()
                return
            group = [item]
            stop_after_group = False
            if self.window:
                await asyncio.sleep(self.window)
            while len(group) < self.max_requests:
                try:
                    next_item = self.queue.get_nowait()
                    if next_item is None:
                        self.queue.task_done()
                        stop_after_group = True
                        break
                    group.append(next_item)
                except asyncio.QueueEmpty:
                    break
            live = [x for x in group if not x.future.done()]
            started = time.perf_counter()
            if live:
                outputs = await asyncio.get_running_loop().run_in_executor(
                    self.executor, self._evaluate, live
                )
                for pending, output in zip(live, outputs, strict=True):
                    if pending.future.done():
                        continue
                    if isinstance(output, Exception):
                        pending.future.set_exception(
                            HTTPException(
                                422 if isinstance(output, ValueError) else 500,
                                str(output)
                                if isinstance(output, ValueError)
                                else "inference failed",
                            )
                        )
                    else:
                        output["queue_wait_ms"] = (started - pending.admitted) * 1000
                        output["request_wall_ms"] = (time.perf_counter() - pending.admitted) * 1000
                        pending.future.set_result(output)
            for _ in group:
                self.queue.task_done()
            if stop_after_group:
                return

    async def close(self):
        self.closing = True
        # Remove waiting clients, but let in-flight GPU work finish cleanly.
        while not self.queue.empty():
            pending = self.queue.get_nowait()
            if not pending.future.done():
                pending.future.set_exception(HTTPException(503, "server shutting down"))
            self.queue.task_done()
        await self.queue.put(None)
        if self.task is not None:
            await self.task


def create_app(
    engine_factory,
    *,
    capacity=32,
    max_requests=8,
    window_ms=4,
    timeout=300,
    max_body_bytes=1024 * 1024,
):
    @asynccontextmanager
    async def lifespan(app):
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-owner")
        try:
            engine = await asyncio.get_running_loop().run_in_executor(executor, engine_factory)
            batcher = Batcher(
                engine,
                executor,
                capacity=capacity,
                max_requests=max_requests,
                window_ms=window_ms,
                timeout=timeout,
            )
            app.state.batcher = batcher
            batcher.task = asyncio.create_task(batcher.run())
            try:
                yield
            finally:
                await batcher.close()
        finally:
            executor.shutdown(wait=True)

    app = FastAPI(title="mlx-decisions", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "alive"}

    @app.get("/ready")
    async def ready():
        worker = getattr(app.state, "batcher", None)
        if worker is None or worker.closing or worker.task.done():
            raise HTTPException(503, "not ready")
        return {
            "status": "ready",
            "queue_depth": worker.queue.qsize(),
            "queue_capacity": worker.queue.maxsize,
        }

    @app.get("/v1/models")
    async def models():
        return {
            "data": [
                {"id": getattr(app.state.batcher.engine, "model_id", MODEL_ID), "object": "model"}
            ]
        }

    @app.post("/v1/systemone")
    async def decide(request: Request):
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > max_body_bytes:
                raise HTTPException(413, "request body too large")
        try:
            parsed = DecisionRequest.model_validate_json(bytes(body))
        except (ValidationError, json.JSONDecodeError):
            raise HTTPException(422, "invalid decision request; see /docs and README") from None
        return await app.state.batcher.submit(parsed)

    return app
