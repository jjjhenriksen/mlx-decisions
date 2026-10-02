#!/usr/bin/env python3
"""Read-only pinned checkpoint/resource check. Never downloads or loads weights.

A conservative planning check, not a guarantee that inference will fit. The
runtime reserve is explicit and does not replace observed peak-memory evidence.
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

MODEL = "openjev/openjev-MLX"
REVISION = "a9dcc20aa827a6c7eae478f6ebb3b255bb135451"


def command(args):
    try:
        result = subprocess.run(args, text=True, capture_output=True, check=False, timeout=15)
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def assess(shards, tensor_bytes, disk_free, missing_bytes, recommended, runtime_reserve):
    blockers = []
    if not shards:
        blockers.append("No checkpoint shards verified; cached metadata is incomplete.")
    if any(not s["size_matches"] for s in shards):
        blockers.append("Pinned checkpoint shards are absent or have unexpected sizes.")
    # Existing incomplete files are retained; the full missing-file size is a conservative
    # download budget, plus reserve for metadata/filesystem/runtime activity.
    if missing_bytes + 2 * 1024**3 > disk_free:
        blockers.append("Writable storage does not cover missing shards plus 2 GiB reserve.")
    if recommended is None or tensor_bytes is None:
        blockers.append("Metal working-set or checkpoint tensor size is unavailable.")
    elif tensor_bytes + runtime_reserve > recommended:
        blockers.append(
            "Tensor bytes plus runtime reserve exceed the recommended Metal working set."
        )
    return blockers


def inspect(snapshot, disk_path, runtime_reserve):
    report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": MODEL,
        "revision": REVISION,
        "snapshot": str(snapshot),
        "machine": platform.platform(),
        "machine_model": command(["sysctl", "-n", "hw.model"]),
        "physical_memory": command(["sysctl", "-n", "hw.memsize"]),
        "memory_pressure": command(["memory_pressure", "-Q"]),
        "vm_stat": command(["vm_stat"]),
        "ollama_residency": command(["ollama", "ps"]),
        "shards": [],
        "metadata_errors": [],
        "runtime_reserve_bytes": runtime_reserve,
        "runtime_reserve_basis": "Conservative 4 GiB default planning allowance for full-forward output, activations, KV/SSM state, allocator, and other runtime allocations; not measured peak evidence.",
        "fusion": "Not loaded or tested; experimental arm remains separate.",
        "verification_boundary": "Read-only cached shard size/provision check. No download, load, inference, or tensor checksum verification; ready means eligible for a run, not validated parity or performance.",
    }
    usage = shutil.disk_usage(disk_path)
    report["storage"] = dict(
        total_bytes=usage.total, used_bytes=usage.used, writable_free_bytes=usage.free
    )
    try:
        config = json.loads((snapshot / "config.json").read_text())
        report["quantization"] = config["quantization"]
        index = json.loads((snapshot / "model.safetensors.index.json").read_text())
        report["tensor_bytes"] = index["metadata"]["total_size"]
        # Hub API metadata identifies each pinned shard's expected size. It downloads
        # no checkpoint files and a network failure is a blocker, never success.
        from huggingface_hub import HfApi

        info = HfApi().model_info(MODEL, revision=REVISION, files_metadata=True, timeout=20)
        if info.sha != REVISION:
            raise ValueError("Hub revision differs from pinned revision")
        files = {f.rfilename: f for f in info.siblings}
        for name in sorted(set(index["weight_map"].values())):
            expected = files[name].size
            path = snapshot / name
            present = path.stat().st_size if path.is_file() else 0
            if expected is None:
                raise ValueError(f"No expected size for {name}")
            report["shards"].append(
                dict(
                    file=name,
                    expected_bytes=expected,
                    present_bytes=present,
                    size_matches=present == expected,
                )
            )
    except Exception as exc:
        report["metadata_errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    try:
        import mlx.core as mx

        report["device"] = mx.device_info()
    except Exception as exc:
        report["metadata_errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    missing = sum(s["expected_bytes"] for s in report["shards"] if not s["size_matches"])
    report["missing_download_budget_bytes"] = missing
    report["blockers"] = assess(
        report["shards"],
        report.get("tensor_bytes"),
        usage.free,
        missing,
        report.get("device", {}).get("max_recommended_working_set_size"),
        runtime_reserve,
    )
    if report["metadata_errors"]:
        report["blockers"].append("Metadata verification failed; inspect metadata_errors.")
    report["status"] = "blocked" if report["blockers"] else "eligible_for_run"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-reserve-gib", type=float, default=4)
    args = parser.parse_args()
    if args.runtime_reserve_gib < 0 or not args.runtime_reserve_gib < float("inf"):
        parser.error("runtime reserve must be finite and nonnegative")
    cache = Path(
        os.environ.get(
            "HF_HUB_CACHE",
            str(Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "hub"),
        )
    )
    snapshot = cache / "models--openjev--openjev-MLX" / "snapshots" / REVISION
    report = inspect(snapshot, Path.cwd(), int(args.runtime_reserve_gib * 1024**3))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {k: report[k] for k in ("status", "blockers", "missing_download_budget_bytes")},
            indent=2,
        )
    )
    return 2 if report["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
