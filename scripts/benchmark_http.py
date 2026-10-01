"""Measure request latency separately from throughput against a running server."""

import argparse
import asyncio
import json
import math
import statistics
import time
from pathlib import Path

import httpx


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


async def run(args):
    body = json.loads(args.request.read_text())
    results = []
    async with httpx.AsyncClient(base_url=args.url, timeout=300) as client:
        ready = await client.get("/ready")
        ready.raise_for_status()
        # Model/kernel and cache warm-up is not part of measured requests.
        warm = await client.post("/v1/systemone", json=body)
        warm.raise_for_status()
        for concurrency in args.concurrency:
            semaphore = asyncio.Semaphore(concurrency)
            samples = []

            async def one(index):
                async with semaphore:
                    request = (
                        body
                        if not args.distinct
                        else {**body, "state": f"Request {index}: {body['state']}"}
                    )
                    start = time.perf_counter()
                    response = await client.post("/v1/systemone", json=request)
                    elapsed = (time.perf_counter() - start) * 1000
                    response.raise_for_status()
                    data = response.json()
                    samples.append(
                        {
                            "wall_ms": elapsed,
                            "queue_wait_ms": data["queue_wait_ms"],
                            "group_requests": data["performance"]["group_requests"],
                            "gpu_batch_sizes": data["performance"]["batch_sizes"],
                        }
                    )

            start = time.perf_counter()
            await asyncio.gather(*(one(i) for i in range(args.requests)))
            wall = time.perf_counter() - start
            latencies = [s["wall_ms"] for s in samples]
            results.append(
                {
                    "concurrency": concurrency,
                    "requests": args.requests,
                    "requests_per_second": args.requests / wall,
                    "p50_request_ms": statistics.median(latencies),
                    "p95_request_ms": percentile(latencies, 0.95),
                    "samples": samples,
                }
            )
            print(json.dumps({k: v for k, v in results[-1].items() if k != "samples"}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "kind": "HTTP end-to-end warm workload",
                "distinct_states": args.distinct,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:3000")
    parser.add_argument("--request", type=Path, default=Path("benchmarks/example.json"))
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--distinct", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/http.json"))
    args = parser.parse_args()
    if args.requests < 1 or not args.concurrency or min(args.concurrency) < 1:
        parser.error("requests and concurrency must be positive")
    asyncio.run(run(args))
