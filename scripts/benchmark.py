#!/usr/bin/env python3
"""A/B benchmark with synchronized GPU work and decision/probability gates.

Timings include tokenization + scoring + answer assembly, but exclude model
load. Cold clears prefix and allocator caches; warm explicitly primes the
same workload. Request latency is NOT throughput divided by concurrency.
"""

import argparse
import json
import math
import platform
import statistics
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx

from mlx_decisions.engine import Engine
from mlx_decisions.optimizations import fuse_gate_up
from mlx_decisions.protocol import MODEL_ID, MODEL_REVISION, TEMPERATURE, DecisionRequest


def workloads():
    q = {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": None, "shipping": None, "technical": None},
    }
    same = "Customer message: I was charged twice for my order last week and nobody has replied."
    questions = {
        "route": q,
        "anger": {"type": "noul", "instructions": "Is the customer angry?"},
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": ["low", "medium", "high"],
        },
        "refund": {
            "type": "choice",
            "instructions": "What action is most appropriate?",
            "criteria": {"refund duplicate": None, "reship parcel": None, "reset password": None},
        },
    }
    return {
        "single": [DecisionRequest(state=same, questions={"route": q})],
        "four_questions_shared_state": [DecisionRequest(state=same, questions=questions)],
        "four_queries_shared_state": [
            DecisionRequest(state=same, questions={"route": q}) for _ in range(4)
        ],
        "four_queries_distinct_state": [
            DecisionRequest(
                state=f"Ticket {i}: Customer was charged twice for an order.",
                questions={"route": q},
            )
            for i in range(4)
        ],
        "long_state_four_queries": [
            DecisionRequest(
                state=("Order history: a duplicate payment was recorded. " * 48) + same,
                questions={"route": q},
            )
            for _ in range(4)
        ],
    }


def probabilities(values):
    a = mx.array(values, mx.float32) / TEMPERATURE
    return mx.softmax(a).tolist()


def parity(reference, actual, rows):
    largest_logit, largest_probability, flips = 0.0, 0.0, 0
    for row in rows:
        key, count = (row.request, row.key), len(row.question.options())
        a, b = reference[key][:count], actual[key][:count]
        largest_logit = max(largest_logit, max(abs(x - y) for x, y in zip(a, b)))
        pa, pb = probabilities(a), probabilities(b)
        largest_probability = max(largest_probability, max(abs(x - y) for x, y in zip(pa, pb)))
        flips += max(range(count), key=a.__getitem__) != max(range(count), key=b.__getitem__)
    return {
        "max_abs_logit_error": largest_logit,
        "max_abs_probability_error": largest_probability,
        "decision_flips": flips,
    }


def raw_comparison(reference, actual, rows):
    samples = []
    for row in rows:
        key, count = (row.request, row.key), len(row.question.options())
        a, b = reference[key][:count], actual[key][:count]
        pa, pb = probabilities(a), probabilities(b)
        options = [key for key, _ in row.question.options()]
        first = max(range(count), key=a.__getitem__)
        second = max(range(count), key=b.__getitem__)
        samples.append(
            {
                "request_index": row.request,
                "question": row.key,
                "question_type": row.question.type,
                "options": options,
                "reference_logits": a,
                "actual_logits": b,
                "reference_conditional_probabilities": pa,
                "actual_conditional_probabilities": pb,
                "absolute_probability_errors": [abs(x - y) for x, y in zip(pa, pb, strict=True)],
                "reference_decision": options[first],
                "actual_decision": options[second],
                "decision_flip": first != second,
            }
        )
    return samples


def save_report(output, report):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--revision", default=MODEL_REVISION)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--output", type=Path, default=Path("benchmarks/results/local.json"))
    ap.add_argument("--probability-atol", type=float, default=0.005)
    ap.add_argument("--workload", choices=list(workloads()), action="append")
    ap.add_argument("--fuse-gate-up", action="store_true")
    ap.add_argument("--rubric-first", action="store_true", help="experimental prompt ordering")
    ap.add_argument(
        "--metal-cache-mib",
        type=int,
        default=512,
        help="process-wide MLX allocator cache budget (not the model or prefix cache)",
    )
    args = ap.parse_args()
    if args.repeats < 1:
        ap.error("--repeats must be positive")
    if args.metal_cache_mib < 0:
        ap.error("--metal-cache-mib must be nonnegative")
    if not math.isfinite(args.probability_atol) or args.probability_atol < 0:
        ap.error("--probability-atol must be finite and nonnegative")
    initial = {
        "model": args.model,
        "revision": args.revision,
        "complete": False,
        "phase": "loading",
        "results": [],
        "machine": platform.platform(),
        "device": mx.device_info(),
        "experimental_fusion": args.fuse_gate_up,
        "prompt_order": "rubric-first" if args.rubric_first else "state-first",
    }
    save_report(args.output, initial)
    mx.set_cache_limit(args.metal_cache_mib * 1024**2)
    started = time.perf_counter()
    try:
        engine = Engine(args.model, revision=args.revision, rubric_first=args.rubric_first)
    except BaseException as exc:
        initial["error"] = {"type": type(exc).__name__, "message": str(exc)}
        save_report(args.output, initial)
        raise
    loaded = time.perf_counter() - started
    print(json.dumps({"event": "model_loaded", "seconds": loaded}), flush=True)
    report = {
        "complete": False,
        "experimental_fusion": args.fuse_gate_up,
        "model": engine.model_id,
        "revision": engine.revision,
        "prompt_order": "rubric-first" if args.rubric_first else "state-first",
        "parity_reference": "unfused full forward with the selected prompt order",
        "machine": platform.platform(),
        "device": mx.device_info(),
        "load_seconds": loaded,
        "metal_allocator_cache_limit_bytes": args.metal_cache_mib * 1024**2,
        "versions": {
            name: version(name) for name in ["mlx", "mlx-lm", "transformers", "mlx-decisions"]
        },
        "repeats": args.repeats,
        "probability_atol": args.probability_atol,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "synthetic correctness/performance smoke, not task accuracy or calibration validation",
        "results": [],
    }
    try:
        report["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        report["git_dirty"] = bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True)
        )
    except subprocess.CalledProcessError:
        report["git_commit"] = None
    save_report(args.output, report)
    try:
        run_cases(args, engine, report)
    except BaseException as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        save_report(args.output, report)
        raise


def run_cases(args, engine, report):
    cases = workloads()
    if args.workload:
        cases = {k: cases[k] for k in args.workload}
    references = {}
    order_drift = {}
    # Reference is collected BEFORE optional fusion, so gate cannot compare a
    # mutated optimized model against itself.
    for name, requests in cases.items():
        report["phase"] = {"stage": "reference", "workload": name}
        save_report(args.output, report)
        print(json.dumps({"event": "reference_started", "workload": name}), flush=True)
        rows = engine.prepare(requests)
        references[name] = engine.raw_scores(rows, reference=True)[0]
        if args.rubric_first:
            engine.rubric_first = False
            try:
                state_first = engine.raw_scores(engine.prepare(requests), reference=True)[0]
            finally:
                engine.rubric_first = True
            # Different prompts need not agree. Report drift, never call it a
            # cache parity failure or evidence of held-out accuracy/calibration.
            order_drift[name] = parity(state_first, references[name], rows)
    baseline_label = "rubric_first_full_forward" if args.rubric_first else "official_full_forward"
    variants = [
        (baseline_label, False, False, 1, True),
        ("last_position_full_head", False, False, 1, False),
        ("selected_head_serial", True, False, 1, False),
        ("selected_head_batch4", True, False, 4, False),
        ("prefix_batch4_cold", True, True, 4, False),
        ("prefix_batch4_warm", True, True, 4, False),
    ]
    if args.fuse_gate_up:
        engine.fused_layers = fuse_gate_up(engine.model)
        variants = [
            ("fused_prefix_batch4_cold", True, True, 4, False),
            ("fused_prefix_batch4_warm", True, True, 4, False),
        ]
    for name, requests in cases.items():
        baseline_ms = None
        for label, selected, prefix, batch, reference in variants:
            print(
                json.dumps({"event": "case_started", "workload": name, "variant": label}),
                flush=True,
            )
            report["phase"] = {"stage": "measurement", "workload": name, "variant": label}
            save_report(args.output, report)
            engine.selected_head, engine.prefix_reuse, engine.max_batch_size = (
                selected,
                prefix,
                batch,
            )
            durations, checks, all_metrics, raw_samples = [], [], [], []
            engine.clear_cache()
            # Warm kernels for every shape/variant. This is outside reported times.
            engine.raw_scores(engine.prepare(requests), reference=reference)
            for _ in range(args.repeats):
                if not label.endswith("_warm"):
                    engine.clear_cache()
                mx.synchronize()
                mx.reset_peak_memory()
                start = time.perf_counter()
                rows = engine.prepare(requests)
                actual, metrics = engine.raw_scores(rows, reference=reference)
                answers = [
                    r.question.answer(actual[(r.request, r.key)][: len(r.question.options())])
                    for r in rows
                ]
                mx.synchronize()
                durations.append((time.perf_counter() - start) * 1000)
                checks.append(parity(references[name], actual, rows))
                raw_samples.append(raw_comparison(references[name], actual, rows))
                all_metrics.append(metrics)
                report["pending_case"] = {
                    "workload": name,
                    "variant": label,
                    "group_wall_ms_samples": durations,
                    "parity_samples": checks,
                    "raw_score_samples": raw_samples,
                    "work": all_metrics,
                }
                save_report(args.output, report)
            median = statistics.median(durations)
            if reference:
                baseline_ms = median
            gate = {
                "max_abs_logit_error": max(x["max_abs_logit_error"] for x in checks),
                "max_abs_probability_error": max(x["max_abs_probability_error"] for x in checks),
                "decision_flips": max(x["decision_flips"] for x in checks),
            }
            gate["passed"] = (
                gate["decision_flips"] == 0
                and gate["max_abs_probability_error"] <= args.probability_atol
            )
            result = {
                "workload": name,
                "variant": label,
                "request_count": len(requests),
                "question_count": len(rows),
                "raw_score_samples": raw_samples,
                "parity_samples": checks,
                "group_wall_ms_samples": durations,
                "median_group_wall_ms": median,
                "requests_per_second": 1000 * len(requests) / median,
                "decisions_per_second": 1000 * len(rows) / median,
                "speedup_vs_full_forward": baseline_ms / median if baseline_ms else None,
                "speedup_vs_official": (
                    baseline_ms / median if baseline_ms and not args.rubric_first else None
                ),
                "peak_metal_bytes_last_repeat": mx.get_peak_memory(),
                "parity": gate,
                "prompt_order_drift_vs_state_first": order_drift.get(name),
                "work": all_metrics,
                "answers": answers,
            }
            report["results"].append(result)
            report.pop("pending_case", None)
            save_report(args.output, report)
            print(
                json.dumps(
                    {
                        k: result[k]
                        for k in [
                            "workload",
                            "variant",
                            "median_group_wall_ms",
                            "speedup_vs_full_forward",
                            "parity",
                            "prompt_order_drift_vs_state_first",
                        ]
                    }
                ),
                flush=True,
            )
    report["complete"] = True
    report["phase"] = "finished"
    report["parity_passed"] = all(x["parity"]["passed"] for x in report["results"])
    save_report(args.output, report)
    if not report["parity_passed"]:
        raise SystemExit("PARITY FAILED: do not promote the failing variants")


if __name__ == "__main__":
    main()
