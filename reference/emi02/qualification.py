"""EMI-02 CPU-versus-external qualification with unchanged EMI-01 measurements."""

import argparse
import concurrent.futures
import json
import math
import hashlib
import multiprocessing
import os
from pathlib import Path
import re
import resource
import signal
import shutil
import subprocess
import tempfile
import time

from reference.emi01 import adapter, circuits, metrics, signals, study
from reference.emi02 import importer

ROOT = Path(__file__).parents[2]
WORKERS = 4
EXECUTION_MODE = "concurrent-qualification-v1"
START_METHOD = "spawn"
SOURCE_FILES = (
    "reference/emi02/importer.py",
    "reference/emi02/qualification.py",
    "reference/emi02/BUILD.bazel",
    "reference/emi02/README.md",
    "cpp/BUILD.bazel",
    "docs/adr/ADR-003-emi02-cpu-qualification.md",
    "docs/adr/ADR-004-emi02-behavioral-expressions.md",
    "docs/adr/ADR-005-emi02-behavioral-transient.md",
    "docs/adr/ADR-006-emi02-model-import-qualification.md",
    "docs/adr/ADR-007-generic-transient-improvements.md",
)


def identities():
    result = study.source_identities()
    for name in SOURCE_FILES:
        path = ROOT / name
        if not path.is_file():
            raise ValueError("provenance_mismatch: EMI-02 source unavailable")
        result[name] = study.sha(path.read_bytes())
    for pattern in ("cpp/include/ohmnivore/*.h", "cpp/src/*.cc", "cpp/src/*.h"):
        for path in sorted(ROOT.glob(pattern)):
            result[str(path.relative_to(ROOT))] = study.sha(path.read_bytes())
    return result


def execute_cpu(binary, directory, limits):
    def child_limits():
        for kind, value in [
            (resource.RLIMIT_CPU, limits["cpu_s"]),
            (resource.RLIMIT_AS, limits["address_bytes"]),
            (resource.RLIMIT_FSIZE, limits["file_bytes"]),
        ]:
            resource.setrlimit(kind, (value, value))

    start = time.perf_counter()
    with (directory / "process.log").open("wb") as log:
        process = subprocess.Popen(
            [str(binary), "circuit.cir", "waveform.raw", "statistics.json"],
            cwd=directory,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={"PATH": "", "LC_ALL": "C"},
            start_new_session=True,
            preexec_fn=child_limits,
        )
        status = "ok"
        try:
            code = process.wait(timeout=limits["wall_s"])
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
            status = "timeout"
        if status == "ok" and code != 0:
            status = (
                "resource_limit"
                if code in {-signal.SIGXCPU, -signal.SIGXFSZ, -signal.SIGKILL}
                else "numerical_failure"
            )
    return {"status": status, "exit_code": code, "wall_s": time.perf_counter() - start}


def cpu_failure_status(log, process_status):
    """Classify typed diagnostics without retaining model-bearing message text."""
    if process_status in {"timeout", "resource_limit"}:
        return process_status, process_status
    categories = {
        "parse": "parse_failure",
        "unsupported": "unsupported_input",
        "compile": "compile_failure",
        "non-finite": "non_finite",
        "unsupported-size": "resource_limit",
        "singular": "numerical_failure",
        "non-convergence": "numerical_failure",
        "solution-validation": "numerical_failure",
        "factorization": "numerical_failure",
        "invalid-structure": "malformed_output",
        "io": "missing_output",
    }
    tags = re.findall(rb"(?m)^([a-z-]+)(?: error|:)", log)
    tag = tags[0].decode() if tags else "unclassified"
    return categories.get(tag, "numerical_failure"), tag


def validate_statistics(statistics, points, variables, unknowns, raw_bytes):
    counts = {
        "points": points,
        "variables": variables,
        "unknowns": unknowns,
        "raw_bytes": raw_bytes,
    }
    if (
        not isinstance(statistics, dict)
        or statistics.get("schema") != "emi02-cpu-v2"
        or statistics.get("status") != "complete"
        or any(
            type(statistics.get(key)) is not int or statistics[key] != value
            for key, value in counts.items()
        )
        or any(
            type(statistics.get(key)) is not int or statistics[key] < 0
            for key in ("attempts", "rejected_steps")
        )
        or statistics["attempts"] - statistics["rejected_steps"] != points - 1
        or statistics["rejected_steps"] > statistics["attempts"]
        or type(statistics.get("elapsed_seconds")) not in (int, float)
        or not math.isfinite(statistics["elapsed_seconds"])
        or statistics["elapsed_seconds"] < 0
    ):
        raise ValueError("malformed_output: CPU statistics disagree with waveform")
    counter_names = (
        "nonlinear_rejections",
        "derivative_history_error_estimates",
        "derivative_history_step_doubling_checks",
        "step_doubling_error_estimates",
        "derivative_history_fallback_entries",
        "derivative_history_fallback_recoveries",
    )
    solver_names = (
        "symbolic_analyses",
        "numeric_factorizations",
        "numeric_refactorizations",
        "numeric_refactorization_fallbacks",
        "numeric_reuses",
        "solves",
        "iterative_refinement_solves",
    )
    solver = statistics.get("transient_solver_statistics")
    if (
        statistics.get("behavioral_error_estimator") != "derivative-history-audited-v1"
        or statistics.get("behavioral_integration_method") != "trapezoidal"
        or any(
            type(statistics.get(k)) is not int or statistics[k] < 0
            for k in counter_names
        )
        or statistics["nonlinear_rejections"] > statistics["rejected_steps"]
        or statistics["derivative_history_fallback_recoveries"]
        > statistics["derivative_history_fallback_entries"]
        or statistics["derivative_history_fallback_entries"] > statistics["attempts"]
        or statistics["derivative_history_step_doubling_checks"]
        > statistics["step_doubling_error_estimates"]
        or statistics["derivative_history_error_estimates"]
        + statistics["step_doubling_error_estimates"]
        + statistics["nonlinear_rejections"]
        > statistics["attempts"]
        or not isinstance(solver, dict)
        or set(solver) != set(solver_names)
        or any(type(solver[k]) is not int or solver[k] < 0 for k in solver_names)
        or solver["iterative_refinement_solves"]
        > 4
        * (
            solver["numeric_factorizations"]
            + solver["numeric_refactorizations"]
            + solver["numeric_reuses"]
        )
    ):
        raise ValueError("malformed_output: CPU estimator or solver accounting")


def run_reference(spec):
    """Use unchanged external measurements, publish no model-bearing diagnostics."""
    out = Path(spec["out"])
    with tempfile.TemporaryDirectory(prefix="emi02-oracle-output-") as temporary:
        private = Path(temporary)
        record = study.run_job({**spec, "out": str(private)})
        directory = private / "jobs" / spec["id"]
        log = directory / "simulator.log"
        if log.is_file():
            diagnostics = log.read_bytes()
            record["diagnostic_log_sha256"] = study.sha(diagnostics)
            record["diagnostic_log_bytes"] = len(diagnostics)
            log.write_text(
                "EMI-02 external diagnostics retained by identity only\n"
                + "sha256="
                + record["diagnostic_log_sha256"]
                + "\nbytes="
                + str(len(diagnostics))
                + "\n"
            )
        if "error" in record:
            record["error"] = (
                record["status"] + ": external job did not complete the frozen contract"
            )
        if (directory / "raw.json").is_file():
            # The unchanged external runner archives failure output in finally,
            # where it may not yet have assigned the record's waveform identity.
            record.setdefault(
                "raw_sha256",
                read_json_bounded(directory / "raw.json")["raw_sha256"],
            )
        record["files"] = {
            path.name: study.sha(path.read_bytes())
            for path in sorted(directory.iterdir())
            if path.is_file() and path.name != "result.json"
        }
        study.write_json(directory / "result.json", record)
        blobs = out / "blobs"
        blobs.mkdir(exist_ok=True)
        if (private / "blobs").is_dir():
            for blob in sorted((private / "blobs").iterdir()):
                destination = blobs / blob.name
                if destination.exists():
                    if study.sha(destination.read_bytes()) != study.sha(
                        blob.read_bytes()
                    ):
                        raise ValueError(
                            "provenance_mismatch: external chunk collision"
                        )
                else:
                    temporary_blob = blobs / (blob.name + f".{os.getpid()}.tmp")
                    try:
                        shutil.copyfile(blob, temporary_blob)
                        os.replace(temporary_blob, destination)
                    finally:
                        temporary_blob.unlink(missing_ok=True)
        (out / "jobs").mkdir(exist_ok=True)
        shutil.copytree(directory, out / "jobs" / spec["id"])
    return record


def run_cpu(spec, binary, archive):
    out = Path(spec["out"])
    dest = out / "jobs" / spec["id"]
    dest.mkdir(parents=True)
    record = {
        key: value
        for key, value in spec.items()
        if key not in {"out", "binary", "model", "limits", "submitted"}
    }
    record.update(status="internal_failure", attempts=1)
    start = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="emi02-cpu-") as temporary:
            directory = Path(temporary)
            source = (
                circuits.dpt(spec["max_step_s"], "emi01-v2")
                if spec["fixture"] == "dpt"
                else circuits.ensemble(
                    spec["candidate"], spec["corner"], spec["max_step_s"], "emi01-v2"
                )
            )
            flat, provenance = importer.import_archive(archive, source)
            (directory / "circuit.cir").write_text(flat)
            study.write_json(dest / "import.json", provenance)
            record["deck_sha256"] = provenance["source_deck_sha256"]
            record["import"] = provenance
            process = execute_cpu(binary, directory, spec["limits"])
            record["process"] = process
            # Error messages can include vendor equation text. Retain their identity
            # and a bounded error category, never their proprietary contents.
            log = (directory / "process.log").read_bytes()
            record["process_log_sha256"] = study.sha(log)
            record["process_log_bytes"] = len(log)
            if process["status"] != "ok":
                status, record["solver_error"] = cpu_failure_status(
                    log, process["status"]
                )
                raise ValueError(status + ": CPU execution failed")
            path = directory / "waveform.raw"
            if not path.exists():
                raise ValueError("missing_output: no CPU waveform")
            if path.stat().st_size > spec["limits"]["file_bytes"]:
                raise ValueError("resource_limit: CPU output byte limit")
            raw = path.read_bytes()
            names = (
                circuits.DPT_NAMES if spec["fixture"] == "dpt" else circuits.STUDY_NAMES
            )
            table = signals.parse_raw(
                raw,
                names,
                16e-6 if spec["fixture"] == "dpt" else 200e-6,
                spec["max_step_s"],
            )
            if not raw.startswith(
                ("Title: " + source.splitlines()[0].lower() + "\n").encode()
            ):
                raise ValueError("malformed_output: CPU title does not bind job")
            header, index = study.store_raw(out, raw)
            (dest / "raw.header").write_bytes(header)
            study.write_json(dest / "raw.json", index)
            record["raw_sha256"] = index["raw_sha256"]
            record["raw_points"] = len(table)
            if not (directory / "statistics.json").is_file():
                raise ValueError("missing_output: mandatory CPU statistics absent")
            statistics = json.loads((directory / "statistics.json").read_text())
            validate_statistics(
                statistics, len(table), len(names), provenance["mna_unknowns"], len(raw)
            )
            record["statistics"] = statistics
            study.write_json(dest / "statistics.json", statistics)
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
                study.write_json(
                    dest / "spectra.json",
                    {
                        "shape": list(spectrum.shape),
                        "sha256": study.sha((dest / "spectra.f64").read_bytes()),
                    },
                )
    except (ValueError, OSError, KeyError) as exc:
        status = getattr(exc, "status", str(exc).split(":", 1)[0])
        record["status"] = (
            status
            if status in study.FAILURES | {"parse_failure", "compile_failure"}
            else "internal_failure"
        )
        # Deliberately generic: output failures must not publish model text.
        record["error"] = (
            record["status"] + ": CPU job did not complete the frozen contract"
        )
    record["elapsed_s"] = time.perf_counter() - start
    record["files"] = {
        path.name: study.sha(path.read_bytes())
        for path in sorted(dest.iterdir())
        if path.is_file()
    }
    study.write_json(dest / "result.json", record)
    return record


def source_deck(spec):
    return (
        circuits.dpt(spec["max_step_s"], "emi01-v2")
        if spec["fixture"] == "dpt"
        else circuits.ensemble(
            spec["candidate"], spec["corner"], spec["max_step_s"], "emi01-v2"
        )
    )


def execution_identity(invocation):
    return (
        isinstance(invocation, dict)
        and type(invocation.get("workers")) is int
        and invocation["workers"] == WORKERS
        and invocation.get("execution_mode") == EXECUTION_MODE
        and invocation.get("multiprocessing_start_method") == START_METHOD
    )


def run_jobs(specs, cpu_binary, archive, out):
    """Execute every engine/job once, collecting completion into frozen order."""
    if len(specs) not in (1, 30) or len({spec["id"] for spec in specs}) != len(specs):
        raise ValueError("malformed_output: invalid scheduled job identities")
    completed = {"cpu": {}, "reference": {}}
    progress = {"cpu": [], "reference": []}
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=WORKERS, mp_context=multiprocessing.get_context(START_METHOD)
    ) as pool:
        pending = {}
        for spec in specs:
            for lane in ("cpu", "reference"):
                job = {
                    **spec,
                    "out": str(out / lane),
                    "submitted": time.perf_counter(),
                }
                future = (
                    pool.submit(run_cpu, job, cpu_binary, archive)
                    if lane == "cpu"
                    else pool.submit(run_reference, job)
                )
                pending[future] = lane, spec["id"]
        for future in concurrent.futures.as_completed(pending):
            lane, expected = pending[future]
            # An infrastructure exception leaves progress incomplete and cannot
            # publish qualification. It does not cause a hidden retry.
            record = future.result()
            if (
                not isinstance(record, dict)
                or record.get("id") != expected
                or expected in completed[lane]
            ):
                raise ValueError("provenance_mismatch: scheduled result identity")
            completed[lane][expected] = record
            progress = {
                engine: [
                    completed[engine][spec["id"]]
                    for spec in specs
                    if spec["id"] in completed[engine]
                ]
                for engine in ("cpu", "reference")
            }
            study.write_json(out / "progress.json", progress)
            print(
                json.dumps(
                    {"job": expected, "engine": lane, "status": record.get("status")}
                ),
                flush=True,
            )
    if any(len(progress[lane]) != len(specs) for lane in ("cpu", "reference")):
        raise ValueError("malformed_output: incomplete scheduled job results")
    return progress["cpu"], progress["reference"]


def frozen_job_identity(record, spec):
    if spec is None:
        return False
    expected = {**spec, "deck_sha256": study.sha(source_deck(spec).encode())}
    return all(
        study.encoded(record.get(key)) == study.encoded(value)
        for key, value in expected.items()
    )


def validate_successful_process(record, cpu):
    process = record.get("process")
    exit_key = "exit_code" if cpu else "wait_status"
    prefix = "process_log" if cpu else "diagnostic_log"
    if (
        not isinstance(process, dict)
        or process.get("status") != "ok"
        or type(process.get(exit_key)) is not int
        or process[exit_key] != 0
        or type(process.get("wall_s")) not in (int, float)
        or not math.isfinite(process["wall_s"])
        or process["wall_s"] < 0
        or not re.fullmatch(r"[a-f0-9]{64}", str(record.get(prefix + "_sha256")))
        or type(record.get(prefix + "_bytes")) is not int
        or not 0 <= record[prefix + "_bytes"] <= signals.MAX_RAW_BYTES
    ):
        raise ValueError(
            "malformed_output: successful job has invalid process diagnostics"
        )


def compare_jobs(cpu_out, reference_out, cpu_records, reference_records):
    """Compare every level, never select only passing corners or switch cases."""
    checks = []
    manifest = study.selected_manifest("emi01-v2")
    specs = expected_specs(manifest)
    expected_ids = [spec["id"] for spec in specs]
    by_spec = {spec["id"]: spec for spec in specs}
    identity_valid = (
        [record["id"] for record in cpu_records] == expected_ids
        and [record["id"] for record in reference_records] == expected_ids
        and all(
            frozen_job_identity(record, by_spec.get(record["id"]))
            for record in cpu_records + reference_records
        )
    )
    reference = {record["id"]: record for record in reference_records}
    for cpu in cpu_records:
        expected = reference.get(cpu["id"])
        check = {"id": cpu["id"], "pass": False}
        if (
            expected is None
            or cpu["status"] not in study.VALID | {"qualified"}
            or expected["status"] not in study.VALID | {"qualified"}
        ):
            check["status"] = "missing_or_failed_job"
            checks.append(check)
            continue
        if not frozen_job_identity(
            cpu, by_spec.get(cpu["id"])
        ) or not frozen_job_identity(expected, by_spec.get(cpu["id"])):
            check["status"] = "provenance_mismatch"
            checks.append(check)
            continue
        try:
            validate_successful_process(cpu, True)
            validate_successful_process(expected, False)
        except ValueError:
            check["status"] = "malformed_output"
            checks.append(check)
            continue
        a, b = study.restore(cpu_out, cpu), study.restore(reference_out, expected)
        if cpu["fixture"] == "dpt":
            check.update(
                study.dpt_compare(cpu["metrics"], expected["metrics"], cpu["id"])
            )
            # Drain, gate, load and terminal-current comparison on the same fixed
            # independent observation grid, including initialization and settling.
            grid, _ = signals.resample(b[:, 0], b[:, 1], 0, 16e-6, cpu["sample_step_s"])
            waves = {}
            for index, name in enumerate(circuits.DPT_NAMES[1:], 1):
                x, y = (
                    study.np.interp(grid, a[:, 0], a[:, index]),
                    study.np.interp(grid, b[:, 0], b[:, index]),
                )
                absolute = 10.0 if name == "v(d)" else 0.5 if name == "v(g)" else 0.02
                compared = signals.waveform_comparison(
                    x,
                    y,
                    absolute=absolute,
                    relative=0 if name.startswith("v(") else 0.02,
                )
                waves[name] = compared
                check["pass"] &= compared["passed"]
            check["waveforms"] = waves
        else:
            check.update(
                metrics.compare(
                    a,
                    b,
                    circuits.STUDY_NAMES,
                    cpu["sample_step_s"],
                    cpu["corner"]["bus_v"],
                )
            )
            # Matching spectra alone cannot repair a differently classified corner.
            check["classification_agrees"] = cpu["status"] == expected["status"]
            check["pass"] &= check["classification_agrees"]
        check["pass"] = bool(check["pass"])
        checks.append(check)
    return {
        "pass": identity_valid
        and len(checks) == 30
        and all(check["pass"] for check in checks),
        "identity_valid": identity_valid,
        "expected_checks": 30,
        "checks": checks,
    }


def read_json_bounded(path, limit=8 * 1024 * 1024):
    try:
        with path.open("rb") as stream:
            data = stream.read(limit + 1)
        if len(data) > limit:
            raise ValueError("resource_limit: evidence JSON byte budget")

        def object_pairs(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("malformed_output: duplicate JSON key")
                result[key] = value
            return result

        result = json.loads(
            data,
            object_pairs_hook=object_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(
                ValueError("non_finite: JSON constant")
            ),
        )
        study.encoded(result)  # Reject overflowed JSON numeric literals as well.
        return result
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(
            "malformed_output: evidence JSON unavailable or malformed"
        ) from exc


def expected_specs(manifest):
    return [
        spec
        for level in range(3)
        for fixture in ("dpt", "ensemble")
        for spec in study.make_specs(manifest, {}, f"q{level}", fixture, level)
    ]


def verify_record_files(directory, record, cpu):
    allowed = (
        {
            "import.json",
            "raw.header",
            "raw.json",
            "statistics.json",
            "spectra.f64",
            "spectra.json",
        }
        if cpu
        else {
            "circuit.cir",
            "driver.cir",
            "simulator.log",
            "raw.header",
            "raw.json",
            "spectra.f64",
            "spectra.json",
            "invalid.raw.gz",
        }
    )
    files = record.get("files")
    if not isinstance(files, dict) or set(files) - allowed:
        raise ValueError("malformed_output: unknown evidence filename")
    actual = {path.name for path in directory.iterdir() if path.name != "result.json"}
    if actual != set(files):
        raise ValueError("malformed_output: missing or unrecorded evidence file")
    for name, expected in files.items():
        path = directory / name
        if (
            not re.fullmatch(r"[a-f0-9]{64}", str(expected))
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_size > signals.MAX_RAW_BYTES
        ):
            raise ValueError("resource_limit: invalid evidence file or size")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while data := stream.read(1024 * 1024):
                digest.update(data)
        if digest.hexdigest() != expected:
            raise ValueError("provenance_mismatch: evidence file hash changed")


def audit_records(out, records, manifest, archive, cpu):
    specs = expected_specs(manifest)
    if (
        not isinstance(records, list)
        or len(records) != len(specs)
        or any(not isinstance(record, dict) for record in records)
        or [record.get("id") for record in records] != [spec["id"] for spec in specs]
    ):
        raise ValueError("malformed_output: all thirty ordered unique jobs required")
    identity_keys = (
        "id",
        "fixture",
        "candidate",
        "corner",
        "level",
        "max_step_s",
        "sample_step_s",
        "reference_version",
    )
    for record, spec in zip(records, specs, strict=True):
        if (
            any(
                study.encoded(record.get(key)) != study.encoded(spec[key])
                for key in identity_keys
            )
            or type(record.get("attempts")) is not int
            or record["attempts"] != 1
        ):
            raise ValueError("provenance_mismatch: changed job identity or retries")
        directory = out / "jobs" / spec["id"]
        if (
            directory.is_symlink()
            or read_json_bounded(directory / "result.json") != record
        ):
            raise ValueError(
                "provenance_mismatch: result manifest differs from job record"
            )
        verify_record_files(directory, record, cpu)
        source = (
            circuits.dpt(spec["max_step_s"], "emi01-v2")
            if spec["fixture"] == "dpt"
            else circuits.ensemble(
                spec["candidate"], spec["corner"], spec["max_step_s"], "emi01-v2"
            )
        )
        if record.get("deck_sha256") != study.sha(source.encode()):
            raise ValueError("provenance_mismatch: changed source deck")
        successful = record.get("status") in study.VALID | {"qualified"}
        if not cpu:
            for name, content in (
                ("circuit.cir", source),
                ("driver.cir", circuits.driver("emi01-v2")),
            ):
                if (successful or name in record["files"]) and record["files"].get(
                    name
                ) != study.sha(content.encode()):
                    raise ValueError("provenance_mismatch: changed external deck")
        if cpu:
            _, provenance = importer.import_archive(archive, source)
            if (
                record.get("import") != provenance
                or read_json_bounded(directory / "import.json") != provenance
            ):
                raise ValueError(
                    "provenance_mismatch: changed model expansion identity"
                )
        if successful:
            validate_successful_process(record, cpu)
        if not successful:
            if record.get("status") not in study.FAILURES | {
                "parse_failure",
                "compile_failure",
            }:
                raise ValueError("malformed_output: unknown failure category")
            if "raw.json" not in record["files"]:
                continue
            try:
                table = study.restore(out, record)
            except ValueError as exc:
                # Incomplete/non-finite simulator output can be honest failure
                # evidence, but corrupt or missing archived chunks cannot.
                if str(exc).split(":", 1)[0] in {
                    "provenance_mismatch",
                    "missing_output",
                    "resource_limit",
                }:
                    raise
                continue
        else:
            table = study.restore(out, record)
        if (
            not (directory / "raw.header")
            .read_bytes()
            .startswith(("Title: " + source.splitlines()[0].lower() + "\n").encode())
        ):
            raise ValueError("provenance_mismatch: waveform title differs from job")
        raw_index = read_json_bounded(directory / "raw.json")
        if record.get("raw_sha256") != raw_index.get("raw_sha256") or (
            (successful or "raw_points" in record)
            and (
                type(record.get("raw_points")) is not int
                or record["raw_points"] != len(table)
            )
        ):
            raise ValueError("provenance_mismatch: changed waveform identity")
        if not successful and "metrics" not in record:
            continue
        if cpu:
            statistics = read_json_bounded(directory / "statistics.json")
            if record.get("statistics") != statistics:
                raise ValueError("provenance_mismatch: changed CPU statistics")
            validate_statistics(
                statistics,
                len(table),
                table.shape[1],
                record["import"]["mna_unknowns"],
                (directory / "raw.header").stat().st_size + raw_index["payload_bytes"],
            )
        if spec["fixture"] == "dpt":
            recomputed = metrics.dpt(table, spec["sample_step_s"])
            status = "qualified" if recomputed["pass"] else "accuracy_failure"
        else:
            recomputed, spectrum = metrics.evaluate(
                table, spec["candidate"], spec["corner"], spec["sample_step_s"]
            )
            status = recomputed["status"]
            if study.sha(spectrum.astype("<f8").tobytes()) != study.sha(
                (directory / "spectra.f64").read_bytes()
            ):
                raise ValueError(
                    "provenance_mismatch: spectrum recomputation disagrees"
                )
        if record.get("metrics") != recomputed or record["status"] != status:
            raise ValueError("provenance_mismatch: measurement recomputation disagrees")
    return records


def audit(out, archive, cpu_binary, ngspice_binary):
    """Recompute complete evidence from bounded raw artifacts, never repair it."""
    invocation = read_json_bounded(out / "invocation.json")
    result = read_json_bounded(out / "qualification.json")
    manifest, manifest_sha = study.load_manifest(
        study.manifest_path("emi01-v2"), "emi01-v2"
    )
    if (
        not isinstance(invocation, dict)
        or not isinstance(result, dict)
        or invocation.get("schema") != "emi02-qualification-v1"
        or invocation.get("probe_only") is not False
        or invocation.get("expected_cpu_jobs") != 30
        or invocation.get("expected_reference_jobs") != 30
        or invocation.get("manifest_sha256") != manifest_sha
        or invocation.get("reference_version") != "emi01-v2"
        or not execution_identity(invocation)
        or invocation.get("sources") != identities()
    ):
        raise ValueError("provenance_mismatch: invocation identity or completeness")
    if invocation.get("cpu_sha256") != study.sha(
        cpu_binary.read_bytes()
    ) or invocation.get("ngspice_sha256") != study.check_elf(ngspice_binary):
        raise ValueError("provenance_mismatch: simulator identity changed")
    with tempfile.TemporaryDirectory(prefix="emi02-audit-") as temporary:
        model = adapter.adapt_archive(archive, Path(temporary) / "model.lib")
    if invocation.get("model") != model:
        raise ValueError("provenance_mismatch: model identity changed")
    cpu_out, reference_out = out / "cpu", out / "reference"
    cpu = audit_records(cpu_out, result.get("cpu_jobs"), manifest, archive, True)
    reference = audit_records(
        reference_out, result.get("reference_jobs"), manifest, archive, False
    )
    cpu_refinement = study.qualify(cpu_out, cpu, write=False, manifest=manifest)
    reference_refinement = study.qualify(
        reference_out, reference, write=False, manifest=manifest
    )
    differential = compare_jobs(cpu_out, reference_out, cpu, reference)
    passed = bool(
        cpu_refinement["pass"] and reference_refinement["pass"] and differential["pass"]
    )
    finest = [
        record
        for record in cpu
        if record["fixture"] == "ensemble" and record["level"] == 2
    ]
    recomputed = {
        "schema": "emi02-qualification-v1",
        "pass": passed,
        "complete": True,
        "probe_only": False,
        "cpu_refinement": cpu_refinement,
        "reference_refinement": reference_refinement,
        "differential": differential,
        "ranking": study.ranking(finest, passed, manifest),
        "cpu_jobs": cpu,
        "reference_jobs": reference,
        "gpu_authorized": False,
    }
    if result != recomputed:
        raise ValueError(
            "provenance_mismatch: full qualification recomputation disagrees"
        )
    return {
        "audit_pass": True,
        "qualification_pass": passed,
        "cpu_jobs": len(cpu),
        "reference_jobs": len(reference),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", required=True)
    parser.add_argument("--ngspice", required=True)
    parser.add_argument("--model-archive", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out")
    mode.add_argument("--audit")
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Run the first DPT pair only; never produces qualification",
    )
    args = parser.parse_args()
    # Bazel launches inside runfiles; resolve inputs before entering scratch space.
    cpu_source, ngspice_source, archive = map(
        lambda p: Path(p).resolve(), [args.cpu, args.ngspice, args.model_archive]
    )
    if args.audit:
        if args.probe:
            parser.error("a probe cannot be audited as a complete qualification")
        print(
            json.dumps(
                audit(Path(args.audit).resolve(), archive, cpu_source, ngspice_source)
            ),
            flush=True,
        )
        return 0
    out = Path(args.out).resolve()
    if out.exists():
        parser.error(
            "output directory must be new; evidence cannot be selectively repaired"
        )
    out.mkdir(parents=True)
    manifest, manifest_sha = study.load_manifest(
        study.manifest_path("emi01-v2"), "emi01-v2"
    )
    initial_sources = identities()
    with tempfile.TemporaryDirectory(prefix="emi02-run-") as temporary:
        scratch = Path(temporary)
        oracle, oracle_sha = study.snapshot_oracle(ngspice_source, scratch)
        cpu = scratch / "emi02_runner"
        cpu_bytes = cpu_source.read_bytes()
        cpu.write_bytes(cpu_bytes)
        cpu.chmod(0o500)
        cpu_sha = study.sha(cpu_bytes)
        provenance = adapter.adapt_archive(archive, scratch / "model.lib")
        invocation = {
            "schema": "emi02-qualification-v1",
            "reference_version": "emi01-v2",
            "manifest_sha256": manifest_sha,
            "sources": initial_sources,
            "model": provenance,
            "cpu_sha256": cpu_sha,
            "ngspice_sha256": oracle_sha,
            "expected_cpu_jobs": 30,
            "expected_reference_jobs": 30,
            "probe_only": args.probe,
            "workers": WORKERS,
            "execution_mode": EXECUTION_MODE,
            "multiprocessing_start_method": START_METHOD,
        }
        study.write_json(out / "invocation.json", invocation)
        cpu_out, reference_out = out / "cpu", out / "reference"
        cpu_out.mkdir()
        reference_out.mkdir()
        specs = []
        common = {
            "limits": manifest["limits"],
            "binary": str(oracle),
            "model": str(scratch / "model.lib"),
        }
        for level in range(3):
            for fixture in ("dpt", "ensemble"):
                specs.extend(
                    study.make_specs(manifest, common, f"q{level}", fixture, level)
                )
        if args.probe:
            specs = specs[:1]
        records, external = run_jobs(specs, cpu, archive, out)
        cpu_qualification = (
            study.qualify(cpu_out, records, manifest=manifest)
            if not args.probe
            else {"pass": False, "reason": "probe_only"}
        )
        external_qualification = (
            study.qualify(reference_out, external, manifest=manifest)
            if not args.probe
            else {"pass": False, "reason": "probe_only"}
        )
        differential = compare_jobs(cpu_out, reference_out, records, external)
        complete = (
            not args.probe
            and len(records) == len(external) == 30
            and identities() == initial_sources
            and study.sha(cpu.read_bytes()) == cpu_sha
            and study.check_elf(oracle) == oracle_sha
        )
        qualified = bool(
            complete
            and cpu_qualification["pass"]
            and external_qualification["pass"]
            and differential["pass"]
        )
        finest = [
            record
            for record in records
            if record["fixture"] == "ensemble" and record["level"] == 2
        ]
        result = {
            "schema": "emi02-qualification-v1",
            "pass": qualified,
            "complete": complete,
            "probe_only": args.probe,
            "cpu_refinement": cpu_qualification,
            "reference_refinement": external_qualification,
            "differential": differential,
            "ranking": study.ranking(finest, qualified, manifest),
            "cpu_jobs": records,
            "reference_jobs": external,
            "gpu_authorized": False,
        }
        study.write_json(out / "qualification.json", result)
        print(
            json.dumps({"pass": qualified, "complete": complete, "output": str(out)}),
            flush=True,
        )
    return 0 if qualified else 1


if __name__ == "__main__":
    raise SystemExit(main())
