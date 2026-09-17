"""Independent exact-byte comparison of the frozen 30 qualification jobs.

This review helper uses only Python standard-library decoding and hashing. It
does not call reference scheduling, raw restoration, metric, or audit helpers.
The diagnostic is a numerical reproducibility check, never retained timing.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

if sys.flags.optimize:
    raise SystemExit(
        "Run this review helper without -O or PYTHONOPTIMIZE; its acceptance checks require assertions."
    )


def digest(data):
    return hashlib.sha256(data).hexdigest()


def document(path):
    return json.loads(path.read_bytes())


def stable(value):
    return json.dumps(
        value, sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode()


def checked_payload(directory, identity, terminal):
    job = directory / "jobs" / identity
    result_path = job / "result.json"
    assert digest(result_path.read_bytes()) == terminal["result_hashes"][identity]
    result = document(result_path)
    for name in ("raw.json", "raw.header", "circuit.cir", "driver.cir"):
        assert digest((job / name).read_bytes()) == result["files"][name]
    index = document(job / "raw.json")
    header = (job / "raw.header").read_bytes()
    header_lines = header.splitlines(keepends=True)
    assert sum(line.startswith(b"Date:") for line in header_lines) == 1
    header_without_date = b"".join(
        line for line in header_lines if not line.startswith(b"Date:")
    )
    payload_hash = hashlib.sha256()
    full_hash = hashlib.sha256(header)
    total = 0
    for chunk in index["chunks"]:
        with gzip.open(directory / "blobs" / (chunk["sha256"] + ".gz"), "rb") as stream:
            data = stream.read(4 * 1024 * 1024 + 1)
        assert len(data) == chunk["bytes"] <= 4 * 1024 * 1024
        assert digest(data) == chunk["sha256"]
        payload_hash.update(data)
        full_hash.update(data)
        total += len(data)
    assert total == index["payload_bytes"] <= 512 * 1024 * 1024
    assert full_hash.hexdigest() == index["raw_sha256"] == result["raw_sha256"]
    spectra = None
    if "spectra.f64" in result["files"]:
        spectra = digest((job / "spectra.f64").read_bytes())
        assert spectra == result["files"]["spectra.f64"]
    return result, {
        "raw_payload_sha256": payload_hash.hexdigest(),
        "raw_header_without_date_sha256": digest(header_without_date),
        "payload_bytes": total,
        "raw_points": result["raw_points"],
        "metrics_sha256": digest(stable(result["metrics"])),
        "spectra_f64_sha256": spectra,
        "result_sha256": digest(result_path.read_bytes()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retained", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    identities = []
    for level in range(3):
        identities.append(f"q{level}-dpt")
        for candidate in ("light", "medium", "heavy"):
            for corner in ("nominal", "fast_low_lc", "hot_high_c"):
                identities.append(f"q{level}-ensemble-{candidate}-{corner}")
    retained = document(args.retained / "terminal.json")
    diagnostic = document(args.diagnostic / "terminal.json")
    assert retained["job_ids"][:30] == diagnostic["job_ids"] == identities
    assert diagnostic["qualification_only"] is True
    assert diagnostic["counts"] == {
        "expected": 30,
        "terminal": 30,
        "validated": 30,
        "failures": 0,
    }
    assert retained["qualification_pass"] is diagnostic["qualification_pass"] is True
    old_metadata = document(args.retained / "metadata.json")
    new_metadata = document(args.diagnostic / "metadata.json")
    assert old_metadata["source_sha256"] == new_metadata["source_sha256"]
    assert old_metadata["manifest_sha256"] == new_metadata["manifest_sha256"]
    assert old_metadata["model"] == new_metadata["model"]
    rows = []
    for identity in identities:
        old, old_info = checked_payload(args.retained, identity, retained)
        new, new_info = checked_payload(args.diagnostic, identity, diagnostic)
        equal = all(
            old_info[key] == new_info[key]
            for key in (
                "raw_payload_sha256",
                "raw_header_without_date_sha256",
                "payload_bytes",
                "raw_points",
                "metrics_sha256",
                "spectra_f64_sha256",
            )
        )
        identity_equal = all(
            old[key] == new[key]
            for key in (
                "id",
                "fixture",
                "candidate",
                "corner",
                "level",
                "max_step_s",
                "sample_step_s",
                "manifest_sha256",
                "status",
            )
        )
        rows.append(
            {
                "id": identity,
                "status": new["status"],
                "exact_numerical_match": equal,
                "same_job_identity_and_status": identity_equal,
                "retained": old_info,
                "rebuilt": new_info,
            }
        )
        print(
            identity, "exact" if equal and identity_equal else "DIFFERENT", flush=True
        )
    qualification_equal = document(args.retained / "qualification.json") == document(
        args.diagnostic / "qualification.json"
    )
    result = {
        "schema": "emi01-rebuilt-reference-comparison-v1",
        "performance_evidence": False,
        "retained_terminal_sha256": digest(
            (args.retained / "terminal.json").read_bytes()
        ),
        "diagnostic_terminal_sha256": digest(
            (args.diagnostic / "terminal.json").read_bytes()
        ),
        "retained_ngspice_sha256": old_metadata["ngspice_sha256"],
        "rebuilt_ngspice_sha256": new_metadata["ngspice_sha256"],
        "diagnostic_counts": diagnostic["counts"],
        "qualification_pass": diagnostic["qualification_pass"],
        "qualification_checks_exact_equal": qualification_equal,
        "all_30_jobs_exact_numerical_match": all(
            r["exact_numerical_match"] and r["same_job_identity_and_status"]
            for r in rows
        ),
        "comparison_scope": "All original binary waveform payload bytes (header dates excluded), derived metrics, spectra and frozen qualification checks. CPU timing, OS resource usage and timestamp-bearing headers are not claimed identical.",
        "jobs": rows,
    }
    args.out.write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    assert result["all_30_jobs_exact_numerical_match"] and qualification_equal


if __name__ == "__main__":
    main()
