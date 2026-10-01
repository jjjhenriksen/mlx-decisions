"""Measure successful request latency and retain every attempted HTTP outcome."""

import argparse
import asyncio
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path

import httpx


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def summarize(result, wall):
    samples = result["samples"]
    successful = [s for s in samples if s["success"]]
    latencies = [s["wall_ms"] for s in successful]
    result.update(
        attempted_requests=len(samples),
        successful_requests=len(successful),
        failed_requests=len(samples) - len(successful),
        requests_per_second=len(successful) / wall,
        attempted_requests_per_second=len(samples) / wall,
        error_rate=(len(samples) - len(successful)) / len(samples) if samples else None,
        p50_request_ms=statistics.median(latencies) if latencies else None,
        p95_request_ms=percentile(latencies, 0.95) if latencies else None,
        status_counts=dict(Counter(str(s["status_code"]) for s in samples)),
        error_counts=dict(Counter(s["error_category"] for s in samples if not s["success"])),
    )
    samples.sort(key=lambda sample: sample["request_index"])


async def run(args, *, transport=None):
    report = {
        "kind": "HTTP end-to-end warm workload",
        "distinct_states": args.distinct,
        "complete": False,
        "phase": "request-loading",
        "results": [],
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    try:
        body = json.loads(args.request.read_text())
        async with httpx.AsyncClient(base_url=args.url, timeout=300, transport=transport) as client:
            report["phase"] = "readiness"
            ready = await client.get("/ready")
            ready.raise_for_status()
            # Model/kernel and cache warm-up is not part of measured requests.
            report["phase"] = "warm-up"
            warm = await client.post("/v1/systemone", json=body)
            warm.raise_for_status()
            report["phase"] = "measurement"
            for concurrency in args.concurrency:
                semaphore = asyncio.Semaphore(concurrency)
                result = {
                    "concurrency": concurrency,
                    "requests": args.requests,
                    "complete": False,
                    "samples": [],
                }
                report["results"].append(result)

                async def one(index):
                    async with semaphore:
                        request = (
                            body
                            if not args.distinct
                            else {**body, "state": f"Request {index}: {body['state']}"}
                        )
                        sample = {
                            "request_index": index,
                            "status_code": None,
                            "success": False,
                            "error_category": "interrupted",
                        }
                        start = time.perf_counter()
                        try:
                            response = await client.post("/v1/systemone", json=request)
                            sample["status_code"] = response.status_code
                            response.raise_for_status()
                            data = response.json()
                            sample.update(
                                queue_wait_ms=data["queue_wait_ms"],
                                group_requests=data["performance"]["group_requests"],
                                gpu_batch_sizes=data["performance"]["batch_sizes"],
                                success=True,
                                error_category=None,
                            )
                        except httpx.HTTPStatusError:
                            sample["error_category"] = "http_status"
                        except httpx.TimeoutException:
                            sample["error_category"] = "timeout"
                        except httpx.RequestError:
                            sample["error_category"] = "transport_error"
                        except (ValueError, KeyError, TypeError):
                            sample["error_category"] = "invalid_response"
                        except Exception:
                            sample["error_category"] = "unexpected_error"
                            raise
                        finally:
                            sample["wall_ms"] = (time.perf_counter() - start) * 1000
                            result["samples"].append(sample)

                start = time.perf_counter()
                try:
                    await asyncio.gather(*(one(i) for i in range(args.requests)))
                    result["complete"] = True
                finally:
                    summarize(result, time.perf_counter() - start)
                    save()
                print(json.dumps({k: v for k, v in result.items() if k != "samples"}), flush=True)
        report["complete"] = True
        report["phase"] = "finished"
    except Exception as error:
        report["failure_type"] = type(error).__name__
        raise
    finally:
        save()
    return report


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
    report = asyncio.run(run(args))
    raise SystemExit(1 if any(r["failed_requests"] for r in report["results"]) else 0)
