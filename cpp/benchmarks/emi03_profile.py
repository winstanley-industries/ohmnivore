"""Opt-in phase diagnosis with a fresh ordinary/profile trajectory parity check."""

import argparse
import json
import os
from pathlib import Path
import platform
import time

from reference.emi01 import study
from reference.emi02 import qualification


def numerical_statistics(record):
    return {
        key: value
        for key, value in record.get("statistics", {}).items()
        if key not in {"diagnostic_profile", "elapsed_seconds"}
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--model-archive", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--candidate", default="reference")
    parser.add_argument("--corner", default="nominal")
    args = parser.parse_args()
    start = time.perf_counter()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = study.selected_manifest("emi01-v2")
    choices = [
        spec
        for spec in study.make_specs(manifest, {}, "q0", level=0)
        if spec["candidate"]["id"] == args.candidate
        and spec["corner"]["id"] == args.corner
    ]
    if len(choices) != 1:
        raise ValueError("unsupported_input: unknown frozen candidate/corner")
    spec = {**choices[0], "limits": manifest["limits"]}
    sources = qualification.identities()
    for name in ("cpp/benchmarks/emi03_profile.h", "cpp/benchmarks/emi03_profile.py"):
        sources[name] = study.sha((qualification.ROOT / name).read_bytes())
    binaries = {"ordinary": args.cpu.resolve(), "profile": args.profile.resolve()}
    invocation = {
        "schema": "emi03-cpu-profile-invocation-v1",
        "sources": sources,
        "binaries": {
            key: study.sha(path.read_bytes()) for key, path in binaries.items()
        },
        "model_archive_sha256": study.sha(args.model_archive.read_bytes()),
        "job": spec,
        "platform": platform.platform(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "timing_boundary": "runner start through raw close; metadata close excluded",
        "interpretation": "instrumented diagnostic; no complete-study speedup claim",
    }
    study.write_json(out / "invocation.json", invocation)
    records = {}
    for lane, binary in binaries.items():
        records[lane] = qualification.run_cpu(
            {**spec, "out": str(out / lane)}, binary, args.model_archive.resolve()
        )
        print(json.dumps({"lane": lane, "status": records[lane]["status"]}), flush=True)
    ordinary, profiled = records["ordinary"], records["profile"]
    valid = all(record["status"] in study.VALID for record in records.values())
    exact = valid and (
        ordinary["raw_sha256"] == profiled["raw_sha256"]
        and ordinary["metrics"] == profiled["metrics"]
        and numerical_statistics(ordinary) == numerical_statistics(profiled)
    )
    phases = profiled.get("statistics", {}).get("diagnostic_profile")
    summary = {
        "schema": "emi03-cpu-profile-v1",
        "complete": valid,
        "exact_trajectory_metrics_work_parity": exact,
        "profile": phases,
        "ordinary_process_wall_s": ordinary.get("process", {}).get("wall_s"),
        "profile_process_wall_s": profiled.get("process", {}).get("wall_s"),
        "invocation_wall_s": time.perf_counter() - start,
        "gpu_authorized": False,
        "performance_gate_evaluated": False,
    }
    study.write_json(out / "profile.json", summary)
    print(json.dumps(summary), flush=True)
    return 0 if exact and phases is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
