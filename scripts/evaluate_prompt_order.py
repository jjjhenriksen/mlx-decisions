#!/usr/bin/env python3
"""Paired held-out evaluation using serial, unfused full forward for both orders."""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

from mlx_decisions.evaluation import load_cases, observation, summarize
from mlx_decisions.protocol import MODEL_ID, MODEL_REVISION, NOUL_TEMPERATURE, TEMPERATURE


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def evaluate(cases, engine, report, output, synchronize=lambda: None):
    try:
        for index, (case, request, question) in enumerate(cases):
            sample = dict(
                id=case["id"], family=case["family"], type=question.type, label=case["label"]
            )
            # Alternate first arm to reduce systematic ordering effects. No warm cache.
            orders = (
                ("state-first", "rubric-first")
                if index % 2 == 0
                else ("rubric-first", "state-first")
            )
            for order in orders:
                engine.rubric_first = order == "rubric-first"
                engine.clear_cache()
                rows = engine.prepare([request])
                synchronize()
                start = time.perf_counter()
                scores, work = engine.raw_scores(rows, reference=True)
                synchronize()
                logits = scores[(0, rows[0].key)][: len(question.options())]
                sample[order] = observation(case, question, logits)
                sample[order].update(
                    forward_ms=(time.perf_counter() - start) * 1000,
                    prompt_tokens=len(rows[0].tokens),
                    prompt_token_sha256=hashlib.sha256(
                        json.dumps(rows[0].tokens).encode()
                    ).hexdigest(),
                    work=work,
                )
                # Preserve the first arm even if the second fails.
                report["pending_sample"] = sample
                save(output, report)
            report["samples"].append(sample)
            report.pop("pending_sample", None)
            save(output, report)
            print(
                json.dumps(
                    {
                        "event": "pair_completed",
                        "id": case["id"],
                        "completed": len(report["samples"]),
                    }
                ),
                flush=True,
            )
        report["metrics"] = summarize(report["samples"])
        report["complete"] = True
    except BaseException as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        save(output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("benchmarks/heldout/policy-decisions-v1.json")
    )
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.dataset.read_bytes()
    dataset = json.loads(raw)
    cases = load_cases(dataset)  # validate before loading weights
    report = {
        "schema_version": 1,
        "complete": False,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model,
        "revision": args.revision,
        "dataset": {
            "name": dataset["name"],
            "sha256": hashlib.sha256(raw).hexdigest(),
            "count": len(cases),
            "provenance": dataset["provenance"],
            "limitations": dataset["limitations"],
        },
        "method": "Serial unfused full forward; alternate first order per pair; no prefix reuse.",
        "fixed_readout": {
            "temperature": TEMPERATURE,
            "noul_temperature": NOUL_TEMPERATURE,
            "decision": "argmax, binary noul threshold 0.5",
            "ece_bins": 10,
        },
        "tuning": "None. Dataset and method frozen in Git before model inference.",
        "machine": platform.platform(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        "samples": [],
    }
    save(args.output, report)
    try:
        import mlx.core as mx

        from mlx_decisions.engine import Engine

        mx.set_cache_limit(512 * 1024**2)
        report["device"] = mx.device_info()
        report["versions"] = {name: version(name) for name in ("mlx", "mlx-lm", "transformers")}
        engine = Engine(
            args.model,
            revision=args.revision,
            selected_head=False,
            prefix_reuse=False,
            max_batch_size=1,
        )
        report["model"], report["revision"] = engine.model_id, engine.revision
        save(args.output, report)
        evaluate(cases, engine, report, args.output, mx.synchronize)
        report["peak_metal_bytes"] = mx.get_peak_memory()
        save(args.output, report)
    except BaseException as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        save(args.output, report)
        raise


if __name__ == "__main__":
    main()
