"""EMI-01 finite external CPU reference study. No production simulator imports."""

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import re
import resource
import shutil
import signal
import struct
import tempfile
import sys
import time
import zlib

from third_party.emi_python.runtime import load_numpy

np = load_numpy()
from reference.emi01 import adapter, circuits, metrics, signals  # noqa: E402

VALID = {"predicted_feasible", "predicted_infeasible"}
FAILURES = {
    "unsupported_input",
    "provenance_mismatch",
    "simulator_failure",
    "numerical_failure",
    "timeout",
    "resource_limit",
    "missing_output",
    "malformed_output",
    "non_finite",
    "unsettled",
    "accuracy_failure",
    "internal_failure",
}
HERE = Path(__file__).parent
MANIFESTS = {"emi01-v1": "manifest.json", "emi01-v2": "manifest-v2.json"}
# Bazel generates executable-specific stage-two bootstrap modules beside these
# files. Their names/bytes are launcher details, not study inputs. Bind only the
# explicit shipped source contract, identically from study, report, and tests.
SOURCE_FILES = (
    "reference/emi01/BUILD.bazel",
    "reference/emi01/CPU_GAPS.md",
    "reference/emi01/MODEL_PROVENANCE.md",
    "reference/emi01/README.md",
    "reference/emi01/adapter.py",
    "reference/emi01/circuits.py",
    "reference/emi01/manifest.json",
    "reference/emi01/manifest-v2.json",
    "reference/emi01/metrics.py",
    "reference/emi01/report.py",
    "reference/emi01/requirements.txt",
    "reference/emi01/signals.py",
    "reference/emi01/study.py",
    "third_party/emi_python/runtime.py",
    "third_party/emi_python/runtime_rules.bzl",
    "third_party/emi_python/BUILD.bazel",
    "third_party/emi_python/PROVENANCE.md",
    "third_party/ngspice/BUILD.bazel",
    "third_party/ngspice/PROVENANCE.md",
    "docs/adr/ADR-002-inverter-emi-design-study.md",
    "MODULE.bazel",
    "MODULE.bazel.lock",
    "BUILD.bazel",
    ".bazelrc",
    ".bazelversion",
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source_identities():
    """Hash exactly the source files shipped by every reference entry point."""
    try:
        return {
            name: sha((HERE.parent.parent / name).read_bytes()) for name in SOURCE_FILES
        }
    except OSError as exc:
        raise ValueError(
            "provenance_mismatch: frozen reference source is unavailable"
        ) from exc


def encoded(obj):
    return (json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def write_json(path, obj):
    path.write_bytes(encoded(obj))


def manifest_path(version="emi01-v1"):
    if version not in MANIFESTS:
        raise ValueError("unsupported_input: unknown reference version")
    return HERE / MANIFESTS[version]


def study_counts(m):
    """Counts follow the finite three-level protocol, not recorded results."""
    cases = len(m["candidates"]) * len(m["corners"])
    qualification = 3 * (1 + cases)
    total = qualification + cases * len(m["workers"]) * (m["warmups"] + m["samples"])
    if m["qualification_jobs"] != qualification or m["expected_jobs"] != total:
        raise ValueError("unsupported_input: manifest protocol counts disagree")
    return {"qualification": qualification, "total": total, "checks": 4 * (1 + cases)}


def load_manifest(path, version="emi01-v1"):
    data = path.read_bytes()
    m = json.loads(data)
    if data != manifest_path(version).read_bytes() or m.get("schema") != version:
        raise ValueError(
            "unsupported_input: only the selected frozen manifest bytes are supported"
        )
    study_counts(m)
    return m, sha(data)


def selected_manifest(version="emi01-v1"):
    return load_manifest(manifest_path(version), version)[0]


def check_elf(path):
    data = path.read_bytes()
    if data[:6] != b"\x7fELF\x02\x01":
        raise ValueError("provenance_mismatch: expected little-endian ELF64 ngspice")
    phoff = struct.unpack_from("<Q", data, 32)[0]
    size, count = struct.unpack_from("<HH", data, 54)
    if size < 56 or phoff + size * count > len(data):
        raise ValueError("provenance_mismatch: invalid ELF program headers")
    if any(
        struct.unpack_from("<I", data, phoff + i * size)[0] in [2, 3]
        for i in range(count)
    ):
        raise ValueError("provenance_mismatch: ngspice must be static")
    return sha(data)


def snapshot_oracle(source, directory):
    """Bind one executable for the invocation, unaffected by later Bazel rebuilds."""
    expected = check_elf(source)
    snapshot = directory / "ngspice"
    shutil.copyfile(source, snapshot)
    snapshot.chmod(0o500)
    if check_elf(snapshot) != expected:
        raise ValueError("provenance_mismatch: oracle changed while snapshotting")
    return snapshot, expected


def execute(binary, cwd, limits):
    """Single-thread worker forks only to set limits and exec the pinned oracle."""
    start = time.perf_counter()
    pid = os.fork()
    if pid == 0:
        try:
            os.chdir(cwd)
            fd = os.open("simulator.log", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            os.dup2(fd, 1)
            os.dup2(fd, 2)
            os.close(fd)
            for kind, value in [
                (resource.RLIMIT_CPU, limits["cpu_s"]),
                (resource.RLIMIT_AS, limits["address_bytes"]),
                (resource.RLIMIT_FSIZE, limits["file_bytes"]),
            ]:
                resource.setrlimit(kind, (value, value))
            os.execve(
                binary,
                [binary, "-n", "-b", "driver.cir"],
                {
                    "PATH": "",
                    "LC_ALL": "C",
                    "LANG": "C",
                    "TZ": "UTC",
                    "HOME": str(cwd),
                    "TMPDIR": str(cwd),
                },
            )
        except BaseException:
            os._exit(126)
    timed_out = False
    while True:
        got, status, usage = os.wait4(pid, os.WNOHANG)
        if got:
            break
        if time.perf_counter() - start > limits["wall_s"]:
            os.kill(pid, signal.SIGKILL)
            _, status, usage = os.wait4(pid, 0)
            timed_out = True
            break
        time.sleep(0.005)
    result = {
        "wall_s": time.perf_counter() - start,
        "user_s": usage.ru_utime,
        "system_s": usage.ru_stime,
        "peak_rss_kib": usage.ru_maxrss,
        "wait_status": status,
    }
    result["status"] = (
        "timeout"
        if timed_out
        else "resource_limit"
        if os.WIFSIGNALED(status)
        and os.WTERMSIG(status) in [signal.SIGXCPU, signal.SIGXFSZ, signal.SIGKILL]
        else "simulator_failure"
        if status
        else "ok"
    )
    return result


TELEMETRY = {
    "accepted_steps": "Accepted timepoints",
    "rejected_steps": "Rejected timepoints",
    "newton_iterations": "Transient iterations",
    "total_iterations": "Total iterations",
    "equations": "Circuit Equations",
    "nonzeros": "Circuit total non-zeroes",
    "analysis_s": "Total analysis time (seconds)",
    "load_s": "Matrix load time",
    "factor_s": "Matrix factor time",
    "solve_s": "Matrix solve time",
    "reorder_s": "Matrix reorder time",
    "truncation_s": "Transient trunc time",
    "netlist_loading_s": "Netlist loading time",
    "expansion_s": "Subckt and Param expansion time",
    "parsing_s": "Netlist parsing time",
}


def telemetry(log):
    result = {
        "device_evaluation_s": None,
        "assembly_only_s": None,
        "unavailable_reason": "ngspice reports combined device evaluation and matrix load only",
    }
    for name, label in TELEMETRY.items():
        found = re.findall(
            r"^" + re.escape(label) + r"\s*=\s*([0-9.Ee+\-]+)\s*$", log, re.M
        )
        if len(found) != 1:
            raise ValueError("missing_output: missing/duplicate telemetry " + label)
        value = float(found[0])
        if not math.isfinite(value) or value < 0:
            raise ValueError("non_finite: invalid telemetry")
        integral = name.endswith(("steps", "iterations")) or name in [
            "equations",
            "nonzeros",
        ]
        if integral and not value.is_integer():
            raise ValueError("malformed_output: fractional telemetry count " + name)
        result[name] = int(value) if integral else value
    return result


def store_raw(out, raw):
    header, payload = raw.split(b"Binary:\n", 1)
    header += b"Binary:\n"
    blobs = out / "blobs"
    blobs.mkdir(exist_ok=True)
    chunks = []
    written = 0
    for i in range(0, len(payload), 4 * 1024 * 1024):
        chunk = payload[i : i + 4 * 1024 * 1024]
        digest = sha(chunk)
        path = blobs / (digest + ".gz")
        if not path.exists():
            compressed = gzip.compress(chunk, compresslevel=1, mtime=0)
            temp = blobs / (digest + f".{os.getpid()}.tmp")
            temp.write_bytes(compressed)
            os.replace(temp, path)
            written += len(compressed)
        chunks.append({"sha256": digest, "bytes": len(chunk)})
    return header, {
        "raw_sha256": sha(raw),
        "payload_bytes": len(payload),
        "chunks": chunks,
        "new_compressed_bytes": written,
    }


def restore(out, record):
    directory = out / "jobs" / record["id"]

    def limited_read(path, limit):
        try:
            with path.open("rb") as stream:
                data = stream.read(limit + 1)
        except OSError as exc:
            raise ValueError("missing_output: raw artifact " + str(path)) from exc
        if len(data) > limit:
            raise ValueError("resource_limit: raw artifact exceeds frozen byte budget")
        return data

    try:
        index = json.loads(limited_read(directory / "raw.json", 64 * 1024))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("malformed_output: invalid raw index") from exc
    header = limited_read(directory / "raw.header", signals.MAX_HEADER_BYTES)
    chunk_limit = 4 * 1024 * 1024
    if (
        not isinstance(index, dict)
        or not isinstance(index.get("chunks"), list)
        or not 1
        <= len(index["chunks"])
        <= math.ceil(signals.MAX_RAW_BYTES / chunk_limit)
        or type(index.get("payload_bytes")) is not int
        or not 0 < index["payload_bytes"] <= signals.MAX_RAW_BYTES - len(header)
        or not re.fullmatch(r"[0-9a-f]{64}", str(index.get("raw_sha256")))
    ):
        raise ValueError("malformed_output: raw index schema or byte budget")
    for i, chunk in enumerate(index["chunks"]):
        if (
            not isinstance(chunk, dict)
            or type(chunk.get("bytes")) is not int
            or not 0 < chunk["bytes"] <= chunk_limit
            or (i < len(index["chunks"]) - 1 and chunk["bytes"] != chunk_limit)
            or not re.fullmatch(r"[0-9a-f]{64}", str(chunk.get("sha256")))
        ):
            raise ValueError("malformed_output: raw chunk identity or byte budget")
    if sum(c["bytes"] for c in index["chunks"]) != index["payload_bytes"]:
        raise ValueError("malformed_output: raw payload byte count mismatch")
    payload = []
    for chunk in index["chunks"]:
        path = out / "blobs" / (chunk["sha256"] + ".gz")
        try:
            if path.stat().st_size > chunk_limit + 64 * 1024:
                raise ValueError("resource_limit: compressed raw chunk exceeds budget")
            with gzip.open(path, "rb") as stream:
                data = stream.read(chunk_limit + 1)
        except (gzip.BadGzipFile, EOFError, zlib.error) as exc:
            raise ValueError("malformed_output: invalid gzip raw chunk") from exc
        except OSError as exc:
            raise ValueError("missing_output: raw chunk " + str(path)) from exc
        if len(data) > chunk_limit:
            raise ValueError("resource_limit: decompressed raw chunk exceeds budget")
        if len(data) != chunk["bytes"] or sha(data) != chunk["sha256"]:
            raise ValueError("provenance_mismatch: corrupted raw chunk")
        payload.append(data)
    raw = header + b"".join(payload)
    if sha(raw) != index["raw_sha256"]:
        raise ValueError("provenance_mismatch: corrupt raw reconstruction")
    return signals.parse_raw(
        raw,
        circuits.DPT_NAMES if record["fixture"] == "dpt" else circuits.STUDY_NAMES,
        16e-6 if record["fixture"] == "dpt" else 200e-6,
        record["max_step_s"],
    )


def run_job(spec):
    start = time.perf_counter()
    out = Path(spec["out"])
    dest = out / "jobs" / spec["id"]
    dest.mkdir(parents=True)
    record = {
        key: value
        for key, value in spec.items()
        if key not in ["out", "binary", "model", "limits", "submitted"]
    }
    record.update(
        {
            "status": "internal_failure",
            "queue_s": start - spec["submitted"],
            "worker_pid": os.getpid(),
            "worker_affinity": sorted(os.sched_getaffinity(0)),
            "attempts": 1,
        }
    )
    phases = {}
    work = None
    try:
        work = Path(tempfile.mkdtemp(prefix="emi01-"))
        shutil.copyfile(spec["model"], work / "model.lib")
        deck = (
            circuits.dpt(spec["max_step_s"], spec.get("reference_version", "emi01-v1"))
            if spec["fixture"] == "dpt"
            else circuits.ensemble(
                spec["candidate"],
                spec["corner"],
                spec["max_step_s"],
                spec.get("reference_version", "emi01-v1"),
            )
        )
        (work / "circuit.cir").write_text(deck)
        (dest / "circuit.cir").write_text(deck)
        (work / "driver.cir").write_text(
            circuits.driver(spec.get("reference_version", "emi01-v1"))
        )
        (dest / "driver.cir").write_text(
            circuits.driver(spec.get("reference_version", "emi01-v1"))
        )
        record["deck_sha256"] = sha(deck.encode())
        phases["preparation_s"] = time.perf_counter() - start
        process = execute(spec["binary"], work, spec["limits"])
        record["process"] = process
        phases["simulation_s"] = process["wall_s"]
        begin = time.perf_counter()
        log = (work / "simulator.log").read_bytes()
        (dest / "simulator.log").write_bytes(log)
        if process["status"] != "ok":
            raise ValueError(process["status"] + ": ngspice child failed")
        log_text = log.decode("utf-8", errors="replace")
        if re.search(r"\b(error|failed|aborted|timestep too small)\b", log_text, re.I):
            raise ValueError("numerical_failure: ngspice error log")
        if "ngspice-46 done" not in log_text:
            raise ValueError("provenance_mismatch: ngspice version/completion absent")
        if not (work / "waveform.raw").exists():
            raise ValueError("missing_output: no raw waveform")
        if (work / "waveform.raw").stat().st_size > spec["limits"]["file_bytes"]:
            raise ValueError("resource_limit: raw file too large")
        raw = (work / "waveform.raw").read_bytes()
        if not raw.startswith(
            ("Title: " + deck.splitlines()[0].lower() + "\n").encode()
        ):
            raise ValueError(
                "malformed_output: raw title does not bind candidate/corner"
            )
        store_begin = time.perf_counter()
        if b"Binary:\n" in raw:
            header, index = store_raw(out, raw)
            (dest / "raw.header").write_bytes(header)
            write_json(dest / "raw.json", index)
            record["raw_sha256"] = index["raw_sha256"]
        else:
            (dest / "invalid.raw.gz").write_bytes(gzip.compress(raw, mtime=0))
        phases["required_output_s"] = time.perf_counter() - store_begin
        names = circuits.DPT_NAMES if spec["fixture"] == "dpt" else circuits.STUDY_NAMES
        table = signals.parse_raw(
            raw,
            names,
            16e-6 if spec["fixture"] == "dpt" else 200e-6,
            spec["max_step_s"],
        )
        record["telemetry"] = telemetry(log_text)
        if record["telemetry"]["accepted_steps"] != len(table):
            raise ValueError(
                "malformed_output: raw point count disagrees with telemetry"
            )
        phases["validation_s"] = (
            time.perf_counter() - begin - phases["required_output_s"]
        )
        begin = time.perf_counter()
        if spec["fixture"] == "dpt":
            record["metrics"] = metrics.dpt(table, spec["sample_step_s"])
            record["status"] = (
                "qualified" if record["metrics"]["pass"] else "accuracy_failure"
            )
        else:
            record["metrics"], spectrum = metrics.evaluate(
                table, spec["candidate"], spec["corner"], spec["sample_step_s"]
            )
            record["status"] = record["metrics"]["status"]
            (dest / "spectra.f64").write_bytes(spectrum.astype("<f8").tobytes())
            write_json(
                dest / "spectra.json",
                {
                    "columns": [
                        "frequency_hz",
                        "a_rms_a",
                        "b_rms_a",
                        "cm_rms_a",
                        "dm_rms_a",
                    ],
                    "shape": list(spectrum.shape),
                    "dtype": "little-endian float64",
                    "sha256": sha((dest / "spectra.f64").read_bytes()),
                },
            )
        phases["metrics_s"] = time.perf_counter() - begin
        record["raw_points"] = len(table)
    except BaseException as exc:
        status = str(exc).split(":", 1)[0]
        record["status"] = status if status in FAILURES else "internal_failure"
        record["error"] = str(exc)[:1000]
    finally:
        if work is not None:
            if (work / "simulator.log").exists() and not (
                dest / "simulator.log"
            ).exists():
                shutil.copyfile(work / "simulator.log", dest / "simulator.log")
            if (
                (work / "waveform.raw").exists()
                and not (dest / "raw.json").exists()
                and not (dest / "invalid.raw.gz").exists()
            ):
                data = (work / "waveform.raw").read_bytes()
                if b"Binary:\n" in data:
                    header, index = store_raw(out, data)
                    (dest / "raw.header").write_bytes(header)
                    write_json(dest / "raw.json", index)
                else:
                    (dest / "invalid.raw.gz").write_bytes(gzip.compress(data, mtime=0))
            shutil.rmtree(work)
    record["phases"] = phases
    record["worker_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record["elapsed_s"] = time.perf_counter() - start
    record["completion_latency_s"] = record["queue_s"] + record["elapsed_s"]
    record["files"] = {
        p.name: sha(p.read_bytes()) for p in sorted(dest.iterdir()) if p.is_file()
    }
    record["elapsed_s"] = time.perf_counter() - start
    record["completion_latency_s"] = record["queue_s"] + record["elapsed_s"]
    write_json(dest / "result.json", record)
    record["elapsed_s"] = time.perf_counter() - start
    record["completion_latency_s"] = record["queue_s"] + record["elapsed_s"]
    write_json(dest / "result.json", record)
    return record


def make_specs(m, common, label, fixture="ensemble", level=2):
    specs = []
    cases = (
        [(None, None)]
        if fixture == "dpt"
        else [(c, k) for c in m["candidates"] for k in m["corners"]]
    )
    for c, k in cases:
        identity = f"{label}-{fixture}" + ("" if c is None else f"-{c['id']}-{k['id']}")
        steps = (
            m["dpt_max_steps_s"]
            if c is None
            else m["candidate_max_steps_s"].get(c["id"], m["ensemble_max_steps_s"])
        )
        specs.append(
            {
                **common,
                "id": identity,
                "reference_version": m["schema"],
                "fixture": fixture,
                "candidate": c,
                "corner": k,
                "level": level,
                "max_step_s": steps[level],
                "sample_step_s": m[f"{fixture}_sample_steps_s"][level],
            }
        )
    return specs


def reconcile(records, expected):
    ids = [r["id"] for r in records]
    if ids != expected or len(ids) != len(set(ids)):
        raise ValueError("missing_output: job identities/order/duplicates mismatch")
    allowed = VALID | FAILURES | {"qualified"}
    if any(r["status"] not in allowed for r in records):
        raise ValueError("malformed_output: unknown terminal status")
    return {
        "expected": len(expected),
        "terminal": len(records),
        "validated": sum(
            r["status"] in VALID or r["status"] == "qualified" for r in records
        ),
        "failures": sum(r["status"] in FAILURES for r in records),
    }


def batch(specs, workers, out, label, qualified=None, started_at=None, manifest=None):
    start = time.perf_counter() if started_at is None else started_at
    records = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = []
        for spec in specs:
            spec["submitted"] = time.perf_counter()
            futures.append(pool.submit(run_job, spec))
        for spec, future in zip(specs, futures):
            try:
                record = future.result()
            except BaseException as exc:
                record = {
                    **{
                        key: value
                        for key, value in spec.items()
                        if key not in ["out", "binary", "model", "limits", "submitted"]
                    },
                    "status": "internal_failure",
                    "error": str(exc),
                    "elapsed_s": 0,
                    "completion_latency_s": time.perf_counter() - spec["submitted"],
                    "files": {},
                }
                dest = out / "jobs" / spec["id"]
                dest.mkdir(parents=True, exist_ok=True)
                record["files"] = {
                    p.name: sha(p.read_bytes())
                    for p in sorted(dest.iterdir())
                    if p.is_file() and p.name != "result.json"
                }
                write_json(dest / "result.json", record)
            records.append(record)
            print(
                record["id"],
                record["status"],
                round(record.get("elapsed_s", 0), 3),
                flush=True,
            )
    totals = reconcile(records, [s["id"] for s in specs])
    latencies = sorted(r["elapsed_s"] for r in records)
    summary = {
        "id": label,
        "workers": workers,
        **totals,
        "job_median_s": float(np.median(latencies)),
        "job_p95_s": latencies[math.ceil(0.95 * len(latencies)) - 1],
        "job_ids": [r["id"] for r in records],
        "peak_child_rss_kib": max(
            r.get("process", {}).get("peak_rss_kib", 0) for r in records
        ),
    }
    if qualified is not None:
        summary["ranking"] = ranking(records, qualified, manifest)
        summary["lightest_feasible"] = next(
            (r["candidate"] for r in summary["ranking"] if r["predicted_feasible"]),
            None,
        )
        summary["time_to_lightest_feasible_s"] = (
            None
            if summary["lightest_feasible"] is None
            else max(r["completion_latency_s"] for r in records)
        )
    # One durable summary pass is charged; terminal timing snapshot serialization is administrative.
    write_json(out / (label + ".json"), summary)
    summary["wall_s"] = time.perf_counter() - start
    summary["validated_jobs_per_hour"] = totals["validated"] * 3600 / summary["wall_s"]
    summary["runner_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(out / (label + ".json"), summary)
    return records, summary


def dpt_compare(x, y, identity):
    check = {"id": identity, "pass": True, "metrics": {}}
    for key in [
        "peak_v",
        "on_edge_s",
        "off_edge_s",
        "on_half_s",
        "off_half_s",
        "on_energy_j",
        "off_energy_j",
    ]:
        if key.endswith("energy_j"):
            limit = 2e-6 + 0.05 * abs(y[key])
        elif key == "peak_v":
            limit = 2 + 0.02 * abs(y[key])
        elif key.endswith("half_s"):
            limit = 2e-9 + 0.1 * y[key.replace("half", "edge")]
        else:
            limit = 2e-9 + 0.1 * abs(y[key])
        error = abs(x[key] - y[key])
        check["metrics"][key] = {"difference": error, "limit": limit}
        check["pass"] &= error <= limit
    check["pass"] &= x["ringing"]["status"] == y["ringing"]["status"]
    if x["ringing"]["status"] == y["ringing"]["status"] == "measured":
        check["pass"] &= x["ringing"]["polarity"] == y["ringing"]["polarity"]
    for key, tol in [("frequency_hz", 0.1), ("log_decrement", 0.2)]:
        if x["ringing"]["status"] == y["ringing"]["status"] == "measured":
            error = abs(x["ringing"][key] - y["ringing"][key])
            limit = tol * abs(y["ringing"][key])
            check["metrics"][key] = {"difference": error, "limit": limit}
            check["pass"] &= error <= limit
    return check


def qualify(out, records, write=True, manifest=None):
    manifest = selected_manifest() if manifest is None else manifest
    expected_checks = study_counts(manifest)["checks"]
    checks = []
    by_id = {r["id"]: r for r in records}
    successful = all(r["status"] in VALID | {"qualified"} for r in records)
    for candidate in (c["id"] for c in manifest["candidates"]):
        for corner in (k["id"] for k in manifest["corners"]):
            selected = [
                by_id[f"q{level}-ensemble-{candidate}-{corner}"] for level in range(3)
            ]
            if any(r["status"] not in VALID for r in selected):
                continue
            previous = restore(out, selected[0])
            for level in [1, 2]:
                current = restore(out, selected[level])
                check = metrics.compare(
                    previous,
                    current,
                    circuits.STUDY_NAMES,
                    selected[level]["sample_step_s"],
                    selected[level]["corner"]["bus_v"],
                )
                checks.append(
                    {"id": f"{candidate}-{corner}-integration-{level}", **check}
                )
                previous = current
            for old_dt in manifest["ensemble_sample_steps_s"][:2]:
                check = metrics.output_sampling_compare(
                    previous,
                    circuits.STUDY_NAMES,
                    old_dt,
                    bus=selected[-1]["corner"]["bus_v"],
                )
                checks.append({"id": f"{candidate}-{corner}-output-{old_dt}", **check})
    ds = [by_id[f"q{level}-dpt"] for level in range(3)]
    for a, b in zip(ds, ds[1:]):
        if a["status"] == b["status"] == "qualified":
            checks.append(
                dpt_compare(a["metrics"], b["metrics"], a["id"] + "-" + b["id"])
            )
    if ds[-1]["status"] == "qualified":
        fine = restore(out, ds[-1])
        for dt in manifest["dpt_sample_steps_s"][:2]:
            checks.append(
                dpt_compare(
                    metrics.dpt(fine, dt), ds[-1]["metrics"], f"dpt-output-{dt}"
                )
            )
    result = {
        "pass": bool(
            successful
            and len(checks) == expected_checks
            and all(c["pass"] for c in checks)
        ),
        "expected_checks": expected_checks,
        "checks": checks,
    }
    if write:
        write_json(out / "qualification.json", result)
    return result


def ranking(records, qualified, manifest=None):
    frozen = selected_manifest() if manifest is None else manifest
    candidates = {c["id"]: c for c in frozen["candidates"]}
    corners = {c["id"]: c for c in frozen["corners"]}
    expected = [(c, k) for c in candidates for k in corners]
    pairs = [
        (r.get("candidate", {}).get("id"), r.get("corner", {}).get("id"))
        for r in records
    ]
    ids = [r.get("id") for r in records]
    identity_ok = len(ids) == len(set(ids)) and len(pairs) == len(set(pairs))
    identity_ok &= all(pair in expected for pair in pairs)
    if identity_ok:
        identity_ok = [expected.index(pair) for pair in pairs] == sorted(
            expected.index(pair) for pair in pairs
        )
        labels = set()
        for record, (candidate, corner) in zip(records, pairs):
            suffix = f"-ensemble-{candidate}-{corner}"
            identity_ok &= isinstance(record.get("id"), str) and record["id"].endswith(
                suffix
            )
            identity_ok &= (
                record["candidate"] == candidates[candidate]
                and record["corner"] == corners[corner]
            )
            if identity_ok:
                labels.add(record["id"][: -len(suffix)])
        identity_ok &= len(labels) <= 1
    result = []
    for name in candidates:
        group = [r for r in records if r.get("candidate", {}).get("id") == name]
        complete = identity_ok and [r["corner"]["id"] for r in group] == list(corners)
        result.append(
            {
                "candidate": name,
                "complete": complete,
                "predicted_feasible": bool(
                    qualified
                    and complete
                    and all(r["status"] == "predicted_feasible" for r in group)
                ),
                "mass_kg": circuits.design(candidates[name])["mass_kg"],
            }
        )
    return sorted(result, key=lambda row: (row["mass_kg"], row["candidate"]))


def reference_cases(records, manifest):
    """Fixture-role gates are separate from simulation status and refinement checks."""
    gates = manifest.get("reference_case_gates")
    if gates is None:
        return {"applicable": False, "pass": True, "checks": []}
    completeness = {
        r["candidate"]: r["complete"] for r in ranking(records, False, manifest)
    }
    checks = []
    physical_keys = {
        "device_peak_v",
        "device_peak_a",
        "capacitor_peak_v",
        "winding_rms_a",
        "loss_w",
        "dm_peak_t",
        "cm_peak_t",
        "damping_a_w",
        "damping_b_w",
    }
    for role in ("passing", "boundary", "failing"):
        candidate = gates[role + "_candidate"]
        group = [r for r in records if r.get("candidate", {}).get("id") == candidate]
        margins = []
        physical = bool(completeness.get(candidate))
        numerical = bool(completeness.get(candidate))
        rejected = bool(completeness.get(candidate))
        violations_consistent = bool(completeness.get(candidate))
        for record in group:
            measured = record.get("metrics", {})
            metrics.finite_metrics(measured)
            stress, limits = (
                measured.get("stress", {}),
                measured.get("stress_limits", {}),
            )
            settling = measured.get("settling", {})
            research = measured.get("research_margin_db", {})
            valid = (
                record.get("status") in VALID
                and set(stress) == set(limits) == physical_keys
                and set(settling) == {"a", "b"}
                and all(value.get("pass") is True for value in settling.values())
                and set(research) == {"a", "b", "cm", "dm"}
            )
            numerical &= valid
            physical &= valid and all(
                stress[key] <= limits[key] for key in physical_keys
            )
            rejected &= valid and record.get("status") == "predicted_infeasible"
            if valid:
                margins.extend(research.values())
            if role == "failing":
                expected_violations = (
                    [key for key in physical_keys if stress[key] > limits[key]]
                    + [
                        "research_mask_" + key
                        for key, margin in research.items()
                        if margin < manifest["required_margin_db"]
                    ]
                    if valid
                    else []
                )
                observed = measured.get("violations")
                violations_consistent &= (
                    bool(expected_violations)
                    and isinstance(observed, list)
                    and sorted(observed) == sorted(expected_violations)
                )
        worst = min(margins) if numerical and margins else None
        passes = bool(physical and worst is not None)
        if role == "passing":
            passes &= all(r["status"] == "predicted_feasible" for r in group)
            passes &= worst is not None and worst >= manifest["required_margin_db"]
        elif role == "boundary":
            low, high = gates["boundary_margin_db"]
            passes &= worst is not None and low <= worst <= high
        else:
            passes = bool(
                gates["failing_requires_all_corners"] is True
                and numerical
                and rejected
                and violations_consistent
            )
        checks.append(
            {
                "role": role,
                "candidate": candidate,
                "complete": bool(completeness.get(candidate)),
                "all_physical_screens": bool(physical),
                "minimum_margin_db": worst,
                "distance_from_feasibility_db": None
                if worst is None
                else worst - manifest["required_margin_db"],
                "pass": bool(passes),
            }
        )
        if role == "failing":
            checks[-1].update(
                all_valid_settled=bool(numerical),
                all_corners_predicted_infeasible=bool(rejected),
                violations_consistent=bool(violations_consistent),
            )
    return {
        "applicable": True,
        "pass": all(c["pass"] for c in checks),
        "checks": checks,
    }


def reference_case_groups(records, manifest, summaries):
    by_id = {r["id"]: r for r in records}
    fine = [r for r in records if r["id"].startswith("q2-ensemble-")]
    groups = [{"id": "qualification-finest", **reference_cases(fine, manifest)}]
    for summary in summaries[1:]:
        groups.append(
            {
                "id": summary["id"],
                **reference_cases([by_id[i] for i in summary["job_ids"]], manifest),
            }
        )
    return {"pass": all(group["pass"] for group in groups), "groups": groups}


def audit(out, reference_version=None):
    """Reconstruct the frozen schedule independently of claimed terminal counts."""

    def read(path):
        if not path.is_file():
            raise ValueError("missing_output: missing artifact " + str(path))
        return path.read_bytes()

    def document(path):
        try:
            value = json.loads(read(path))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("malformed_output: invalid JSON " + str(path)) from exc
        if not isinstance(value, dict):
            raise ValueError("malformed_output: expected JSON object " + str(path))
        metrics.finite_metrics(value)
        return value

    def verify_files(directory, hashes, required, excluded=()):
        if not isinstance(hashes, dict) or not required <= set(hashes):
            raise ValueError(
                "missing_output: required artifact omitted from hash index"
            )
        actual = {p.name for p in directory.iterdir() if p.is_file()} - set(excluded)
        if actual != set(hashes):
            raise ValueError("missing_output: artifact/hash coverage differs")
        for name, digest in hashes.items():
            if Path(name).name != name or not re.fullmatch(
                r"[0-9a-f]{64}", str(digest)
            ):
                raise ValueError("malformed_output: artifact name/hash")
            if sha(read(directory / name)) != digest:
                raise ValueError("provenance_mismatch: artifact hash " + name)

    def qualification_result(records, tables):
        """Rebuild refinement checks from audited raw tables without writing files."""
        checks = []
        by_id = {r["id"]: r for r in records}
        for candidate in frozen["candidates"]:
            for corner in frozen["corners"]:
                suffix = f"ensemble-{candidate['id']}-{corner['id']}"
                ids = [f"q{level}-{suffix}" for level in range(3)]
                if any(by_id[identity]["status"] not in VALID for identity in ids):
                    continue
                for level in (1, 2):
                    comparison = metrics.compare(
                        tables[ids[level - 1]],
                        tables[ids[level]],
                        circuits.STUDY_NAMES,
                        frozen["ensemble_sample_steps_s"][level],
                        corner["bus_v"],
                    )
                    checks.append(
                        {
                            "id": f"{candidate['id']}-{corner['id']}-integration-{level}",
                            **comparison,
                        }
                    )
                for dt in frozen["ensemble_sample_steps_s"][:2]:
                    comparison = metrics.output_sampling_compare(
                        tables[ids[-1]], circuits.STUDY_NAMES, dt, bus=corner["bus_v"]
                    )
                    checks.append(
                        {
                            "id": f"{candidate['id']}-{corner['id']}-output-{dt}",
                            **comparison,
                        }
                    )
        for level in (0, 1):
            first, second = f"q{level}-dpt", f"q{level + 1}-dpt"
            if by_id[first]["status"] == by_id[second]["status"] == "qualified":
                checks.append(
                    dpt_compare(
                        by_id[first]["metrics"],
                        by_id[second]["metrics"],
                        first + "-" + second,
                    )
                )
        if by_id["q2-dpt"]["status"] == "qualified":
            for dt in frozen["dpt_sample_steps_s"][:2]:
                checks.append(
                    dpt_compare(
                        metrics.dpt(tables["q2-dpt"], dt),
                        by_id["q2-dpt"]["metrics"],
                        f"dpt-output-{dt}",
                    )
                )
        return {
            "pass": bool(
                all(r["status"] in VALID | {"qualified"} for r in records)
                and len(checks) == counts_contract["checks"]
                and all(c["pass"] for c in checks)
            ),
            "expected_checks": counts_contract["checks"],
            "checks": checks,
        }

    try:
        terminal = document(out / "terminal.json")
        if (
            terminal.get("schema") not in MANIFESTS
            or type(terminal.get("qualification_only")) is not bool
        ):
            raise ValueError("malformed_output: terminal schema or invocation mode")
        version = terminal["schema"]
        if reference_version is not None and version != reference_version:
            raise ValueError(
                "provenance_mismatch: requested and recorded reference versions differ"
            )
        frozen_bytes = read(manifest_path(version))
        if read(out / "manifest.json") != frozen_bytes:
            raise ValueError(
                "provenance_mismatch: study manifest differs from frozen bytes"
            )
        frozen, _ = load_manifest(manifest_path(version), version)
        counts_contract = study_counts(frozen)
        mhash = sha(frozen_bytes)
        metadata = document(out / "metadata.json")
        if (
            metadata.get("schema") != version
            or metadata["manifest_sha256"] != mhash
            or metadata["source_sha256"].get("reference/emi01/" + MANIFESTS[version])
            != mhash
        ):
            raise ValueError("provenance_mismatch: manifest identity in metadata")
        if not re.fullmatch(r"[0-9a-f]{64}", str(metadata.get("ngspice_sha256"))):
            raise ValueError("provenance_mismatch: oracle executable identity")
        if version == "emi01-v2" and metadata.get("oracle_snapshot") != {
            "sha256": metadata.get("ngspice_sha256"),
            "mode": "0500",
            "private": True,
        }:
            raise ValueError("provenance_mismatch: private oracle snapshot identity")
        current_sources = source_identities()
        if metadata["source_sha256"] != current_sources:
            raise ValueError(
                "provenance_mismatch: invoking source files differ from recorded source identities"
            )
        for key, value in {
            "adapter_version": adapter.ADAPTER_VERSION,
            "archive_sha256": adapter.ARCHIVE_SHA256,
            "member_sha256": adapter.MEMBER_SHA256,
            "adapted_sha256": adapter.ADAPTED_SHA256,
            "archive_bytes": adapter.ARCHIVE_SIZE,
            "member_bytes": adapter.MEMBER_SIZE,
            "member": adapter.MEMBER,
        }.items():
            if metadata["model"].get(key) != value:
                raise ValueError("provenance_mismatch: model metadata " + key)
        expected = []
        groups = []

        def append_group(label, fixture, level):
            cases = (
                [(None, None)]
                if fixture == "dpt"
                else [(c, k) for c in frozen["candidates"] for k in frozen["corners"]]
            )
            for candidate, corner in cases:
                if candidate is None:
                    max_step = frozen["dpt_max_steps_s"][level]
                else:
                    grid = frozen["candidate_max_steps_s"].get(
                        candidate["id"], frozen["ensemble_max_steps_s"]
                    )
                    max_step = grid[level]
                suffix = (
                    "" if candidate is None else f"-{candidate['id']}-{corner['id']}"
                )
                expected.append(
                    {
                        "id": f"{label}-{fixture}{suffix}",
                        "reference_version": version,
                        "fixture": fixture,
                        "candidate": candidate,
                        "corner": corner,
                        "level": level,
                        "max_step_s": max_step,
                        "sample_step_s": frozen[f"{fixture}_sample_steps_s"][level],
                        "manifest_sha256": mhash,
                    }
                )

        for level in range(3):
            append_group(f"q{level}", "dpt", level)
            append_group(f"q{level}", "ensemble", level)
        groups.append(("qualification-study", 4, [s["id"] for s in expected]))
        if not terminal["qualification_only"]:
            for workers in frozen["workers"]:
                for sample in range(-frozen["warmups"], frozen["samples"]):
                    label = f"w{workers}-" + (
                        f"warmup{-sample}" if sample < 0 else f"sample{sample}"
                    )
                    begin = len(expected)
                    append_group(label, "ensemble", 2)
                    groups.append((label, workers, [s["id"] for s in expected[begin:]]))
        expected_ids = [s["id"] for s in expected]
        if terminal["job_ids"] != expected_ids or set(terminal["result_hashes"]) != set(
            expected_ids
        ):
            raise ValueError(
                "missing_output: terminal differs from frozen job schedule"
            )
        required = {"metadata.json", "manifest.json", "qualification.json"} | {
            label + ".json" for label, _, _ in groups
        }
        if version == "emi01-v2":
            required.add("reference-cases.json")
        verify_files(out, terminal["files"], required, excluded=("terminal.json",))
        if not (out / "jobs").is_dir():
            raise ValueError("missing_output: jobs directory")
        if sorted(p.name for p in (out / "jobs").iterdir()) != sorted(expected_ids):
            raise ValueError(
                "missing_output: jobs directory differs from frozen schedule"
            )
        records = []
        qualification_tables = {}
        recomputed_qualification = None
        for spec in expected:
            directory = out / "jobs" / spec["id"]
            p = directory / "result.json"
            record = document(p)
            if sha(read(p)) != terminal["result_hashes"][spec["id"]]:
                raise ValueError("provenance_mismatch: completion record hash")
            if any(record.get(key) != value for key, value in spec.items()):
                raise ValueError(
                    "provenance_mismatch: job input identity " + spec["id"]
                )
            status = record.get("status")
            allowed = FAILURES | ({"qualified"} if spec["fixture"] == "dpt" else VALID)
            if status not in allowed:
                raise ValueError(
                    "malformed_output: terminal status inconsistent with fixture"
                )
            required = set()
            if status in VALID | {"qualified"}:
                required = {
                    "circuit.cir",
                    "driver.cir",
                    "simulator.log",
                    "raw.header",
                    "raw.json",
                }
                if spec["fixture"] == "ensemble":
                    required |= {"spectra.json", "spectra.f64"}
            verify_files(
                directory, record["files"], required, excluded=("result.json",)
            )
            deck = (
                circuits.dpt(
                    spec["max_step_s"], spec.get("reference_version", "emi01-v1")
                )
                if spec["fixture"] == "dpt"
                else circuits.ensemble(
                    spec["candidate"],
                    spec["corner"],
                    spec["max_step_s"],
                    spec.get("reference_version", "emi01-v1"),
                )
            ).encode()
            if "circuit.cir" in record["files"] and (
                read(directory / "circuit.cir") != deck
                or record.get("deck_sha256") != sha(deck)
            ):
                raise ValueError("provenance_mismatch: generated deck identity")
            if (
                "driver.cir" in record["files"]
                and read(directory / "driver.cir")
                != circuits.driver(spec.get("reference_version", "emi01-v1")).encode()
            ):
                raise ValueError("provenance_mismatch: generated driver identity")
            if status in VALID | {"qualified"}:
                expected_title = b"Title: " + deck.splitlines()[0].lower() + b"\n"
                if not read(directory / "raw.header").startswith(expected_title):
                    raise ValueError(
                        "provenance_mismatch: raw title differs from candidate/corner deck"
                    )
                if (
                    record.get("process", {}).get("status") != "ok"
                    or record["process"].get("wait_status") != 0
                    or record.get("attempts") != 1
                ):
                    raise ValueError(
                        "malformed_output: valid result lacks successful single simulator attempt"
                    )
                log = read(directory / "simulator.log").decode(
                    "utf-8", errors="replace"
                )
                if re.search(
                    r"\b(error|failed|aborted|timestep too small)\b", log, re.I
                ):
                    raise ValueError(
                        "numerical_failure: claimed valid result has simulator error log"
                    )
                if "ngspice-46 done" not in log:
                    raise ValueError(
                        "provenance_mismatch: successful simulator completion absent"
                    )
                measured_telemetry = telemetry(log)
                if encoded(record.get("telemetry")) != encoded(measured_telemetry):
                    raise ValueError(
                        "provenance_mismatch: stored telemetry differs from log"
                    )
                try:
                    table = restore(out, record)
                except FileNotFoundError as exc:
                    raise ValueError(
                        "missing_output: missing raw waveform chunk"
                    ) from exc
                raw_index = document(directory / "raw.json")
                if (
                    record.get("raw_sha256") != raw_index["raw_sha256"]
                    or record.get("raw_points") != len(table)
                    or measured_telemetry["accepted_steps"] != len(table)
                ):
                    raise ValueError(
                        "malformed_output: raw identities/point counts disagree"
                    )
                if spec["fixture"] == "dpt":
                    recomputed_metrics = metrics.dpt(table, spec["sample_step_s"])
                    recomputed_status = (
                        "qualified"
                        if recomputed_metrics["pass"]
                        else "accuracy_failure"
                    )
                else:
                    recomputed_metrics, spectrum = metrics.evaluate(
                        table, spec["candidate"], spec["corner"], spec["sample_step_s"]
                    )
                    recomputed_status = recomputed_metrics["status"]
                    spectrum_bytes = read(directory / "spectra.f64")
                    spectrum_schema = {
                        "columns": [
                            "frequency_hz",
                            "a_rms_a",
                            "b_rms_a",
                            "cm_rms_a",
                            "dm_rms_a",
                        ],
                        "shape": list(spectrum.shape),
                        "dtype": "little-endian float64",
                        "sha256": sha(spectrum_bytes),
                    }
                    if (
                        document(directory / "spectra.json") != spectrum_schema
                        or len(spectrum_bytes) != spectrum.size * 8
                    ):
                        raise ValueError(
                            "malformed_output: stored spectral schema/length"
                        )
                    stored_spectrum = np.frombuffer(
                        spectrum_bytes, dtype="<f8"
                    ).reshape(spectrum.shape)
                    if not np.isfinite(stored_spectrum).all():
                        raise ValueError("non_finite: stored spectrum")
                    if not np.array_equal(stored_spectrum, spectrum):
                        raise ValueError(
                            "provenance_mismatch: spectrum differs from raw-waveform recomputation"
                        )
                if (
                    encoded(record.get("metrics")) != encoded(recomputed_metrics)
                    or status != recomputed_status
                ):
                    raise ValueError(
                        "provenance_mismatch: metrics/status differ from raw-waveform recomputation"
                    )
                if len(records) < counts_contract["qualification"]:
                    qualification_tables[spec["id"]] = table
                del table
            records.append(record)
            if len(records) == counts_contract["qualification"]:
                recomputed_qualification = qualification_result(
                    records, qualification_tables
                )
                qualification_tables.clear()
        counts = reconcile(records, expected_ids)
        if counts != terminal["counts"]:
            raise ValueError("malformed_output: terminal counters")
        summaries = []
        by_id = {r["id"]: r for r in records}
        for label, workers, identities in groups:
            summary = document(out / (label + ".json"))
            group_counts = reconcile([by_id[i] for i in identities], identities)
            if (
                summary.get("id") != label
                or summary.get("workers") != workers
                or summary.get("job_ids") != identities
                or any(summary.get(k) != v for k, v in group_counts.items())
            ):
                raise ValueError("malformed_output: study summary identity/counts")
            if label != "qualification-study":
                group_records = [by_id[identity] for identity in identities]
                recomputed_ranking = ranking(
                    group_records, recomputed_qualification["pass"], frozen
                )
                lightest = next(
                    (
                        r["candidate"]
                        for r in recomputed_ranking
                        if r["predicted_feasible"]
                    ),
                    None,
                )
                time_to_lightest = (
                    None
                    if lightest is None
                    else max(r["completion_latency_s"] for r in group_records)
                )
                if (
                    encoded(summary.get("ranking")) != encoded(recomputed_ranking)
                    or summary.get("lightest_feasible") != lightest
                    or summary.get("time_to_lightest_feasible_s") != time_to_lightest
                ):
                    raise ValueError(
                        "malformed_output: ranking/lightest result disagrees with audited jobs"
                    )
            summaries.append(summary)
        if summaries != terminal["study_summaries"]:
            raise ValueError("malformed_output: terminal study summaries")
        qualification = document(out / "qualification.json")
        if (
            type(qualification.get("pass")) is not bool
            or qualification["pass"] != terminal["qualification_pass"]
            or encoded(qualification) != encoded(recomputed_qualification)
        ):
            raise ValueError(
                "malformed_output: qualification differs from audited refinement checks"
            )
        if version == "emi01-v2":
            cases = reference_case_groups(records, frozen, summaries)
            if (
                document(out / "reference-cases.json") != cases
                or type(terminal.get("reference_case_pass")) is not bool
                or terminal["reference_case_pass"] != cases["pass"]
            ):
                raise ValueError(
                    "malformed_output: reference case gates differ from audited jobs"
                )
        return counts
    except (KeyError, TypeError, AttributeError, OSError) as exc:
        raise ValueError(
            "malformed_output: incomplete audit schema or artifact"
        ) from exc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ngspice", type=Path, required=True)
    parser.add_argument("--model-archive", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--reference-version", choices=MANIFESTS, default="emi01-v1")
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--qualification-only", action="store_true")
    parser.add_argument("--exploratory-probe", action="store_true")
    args = parser.parse_args()
    if args.audit:
        print(
            json.dumps(
                audit(args.audit.resolve(), args.reference_version), sort_keys=True
            )
        )
        return
    if args.out is None:
        parser.error("--out required")
    start = time.perf_counter()
    out = args.out.resolve()
    if out.exists():
        raise ValueError("unsupported_input: output directory must be new")
    out.mkdir(parents=True)
    m, mhash = load_manifest(
        manifest_path(args.reference_version), args.reference_version
    )
    with tempfile.TemporaryDirectory(prefix="emi01-model-") as temp:
        snapshot, binary_hash = snapshot_oracle(args.ngspice.resolve(), Path(temp))
        binary = str(snapshot)
        model = Path(temp) / "model.lib"
        model_info = adapter.adapt_archive(args.model_archive.resolve(), model)
        sources = source_identities()
        metadata = {
            "schema": m["schema"],
            "manifest_sha256": mhash,
            "source_sha256": sources,
            "ngspice_sha256": binary_hash,
            "oracle_snapshot": {"sha256": binary_hash, "mode": "0500", "private": True},
            "model": model_info,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "python_executable_sha256": sha(Path(sys.executable).read_bytes()),
            "native_runtime_sha256": {
                str(Path(line.split()[-1]).resolve()): sha(
                    Path(line.split()[-1]).read_bytes()
                )
                for line in Path("/proc/self/maps").read_text().splitlines()
                if "/" in line.split()[-1]
                and ".so" in line.split()[-1]
                and "bazel" in line.split()[-1]
            },
            "platform": platform.platform(),
            "cpuinfo": Path("/proc/cpuinfo").read_text(),
            "affinity": sorted(os.sched_getaffinity(0)),
            "resource_limits": m["limits"],
            "runtime_threads": 1,
            "setup_s": time.perf_counter() - start,
            "io_boundary": "closed files, no fsync; build/fetch excluded",
        }
        write_json(out / "metadata.json", metadata)
        shutil.copyfile(manifest_path(args.reference_version), out / "manifest.json")
        common = {
            "out": str(out),
            "binary": binary,
            "model": str(model),
            "limits": m["limits"],
            "manifest_sha256": mhash,
        }
        qualification_started = time.perf_counter()
        specs = []
        for level in range(3):
            specs += make_specs(m, common, f"q{level}", "dpt", level)
            specs += make_specs(m, common, f"q{level}", level=level)
        if args.exploratory_probe:
            batch(
                [
                    spec
                    for spec in specs
                    if spec["fixture"] == "dpt"
                    or (
                        spec["candidate"] == m["candidates"][0]
                        and spec["corner"] == m["corners"][0]
                    )
                ],
                4,
                out,
                "exploratory-probe",
            )
            return
        records, summ = batch(
            specs, 4, out, "qualification-study", started_at=qualification_started
        )
        qualification = qualify(out, records, manifest=m)
        summaries = [summ]
        if not args.qualification_only:
            for workers in m["workers"]:
                for sample in range(-m["warmups"], m["samples"]):
                    label = f"w{workers}-" + (
                        f"warmup{-sample}" if sample < 0 else f"sample{sample}"
                    )
                    batch_started = time.perf_counter()
                    batch_records, summary = batch(
                        make_specs(m, common, label),
                        workers,
                        out,
                        label,
                        qualification["pass"],
                        started_at=batch_started,
                        manifest=m,
                    )
                    records += batch_records
                    summaries.append(summary)
        expected = study_counts(m)[
            "qualification" if args.qualification_only else "total"
        ]
        cases = reference_case_groups(records, m, summaries)
        if m["schema"] == "emi01-v2":
            write_json(out / "reference-cases.json", cases)
        if len(records) != expected:
            raise ValueError("missing_output: invocation count")
        result = {
            "schema": m["schema"],
            "qualification_only": args.qualification_only,
            "job_ids": [r["id"] for r in records],
            "counts": reconcile(records, [r["id"] for r in records]),
            "qualification_pass": qualification["pass"],
            "reference_case_pass": cases["pass"],
            "study_summaries": summaries,
            "invocation_wall_s": time.perf_counter() - start,
            "result_hashes": {
                r["id"]: sha((out / "jobs" / r["id"] / "result.json").read_bytes())
                for r in records
            },
            "files": {
                p.name: sha(p.read_bytes())
                for p in sorted(out.iterdir())
                if p.is_file()
            },
        }
        write_json(out / "terminal.json", result)
    result["invocation_wall_s"] = time.perf_counter() - start
    write_json(out / "terminal.json", result)
    print(
        json.dumps(
            {
                "counts": result["counts"],
                "qualified": qualification["pass"],
                "reference_case_pass": cases["pass"],
                "wall_s": result["invocation_wall_s"],
            }
        ),
        flush=True,
    )
    if not qualification["pass"] or not cases["pass"] or result["counts"]["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
