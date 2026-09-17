"""Independent hostile accounting fixtures for the finite EMI-01 study."""

import copy
import hashlib
import gzip
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from reference.emi01 import report, study


CANDIDATES = ("light", "medium", "heavy")
CORNERS = ("nominal", "fast_low_lc", "hot_high_c")
FAILURES = (
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
)


def digest(contents):
    return hashlib.sha256(contents).hexdigest()


def write_document(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def frozen():
    return json.loads((Path(__file__).parent / "manifest.json").read_bytes())


def candidate_records():
    manifest = frozen()
    return [
        {
            "id": f"w1-sample0-ensemble-{candidate['id']}-{corner['id']}",
            "candidate": candidate,
            "corner": corner,
            "status": "predicted_feasible",
            "metrics": {"mass_kg": 0.1},
        }
        for candidate in manifest["candidates"]
        for corner in manifest["corners"]
    ]


def audit_fixture(directory, qualification_only=True):
    """Author independent all-failure evidence; do not call scheduling/count helpers."""
    manifest_bytes = (Path(__file__).parent / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    mhash = digest(manifest_bytes)
    (directory / "manifest.json").write_bytes(manifest_bytes)
    metadata = {
        "manifest_sha256": mhash,
        "source_sha256": study.source_identities(),
        "model": {
            "adapter_version": "emi01-microchip-ngspice-v1",
            "archive_sha256": "de4a3acf222cbc7f6c1a4d1a2bc449dd31b5db2c6ed153920d797752c3831d51",
            "archive_bytes": 28318,
            "member": "1200V-SMA-SiC-MOSFET-SPICE-Models/MSCSMA120.lib",
            "member_sha256": "6e888e103977f539e62b64952797ded391d2a49c59738bfd53f5fbe4b1fc8df7",
            "member_bytes": 29805,
            "adapted_sha256": "17732ddd7ab5361073f8f23594e32158270af47d9b7188cf0ce773189cafcf96",
        },
    }
    write_document(directory / "metadata.json", metadata)
    write_document(
        directory / "qualification.json",
        {"pass": False, "expected_checks": 40, "checks": []},
    )
    records = []
    summaries = []
    schedule = [(f"q{level}", level) for level in (0, 1, 2)]
    if not qualification_only:
        schedule += [
            (f"w{workers}-{sample}", 2)
            for workers in (1, 4)
            for sample in ("warmup1", "sample0", "sample1", "sample2")
        ]
    for label, level in schedule:
        start = len(records)
        cases = [(None, None)] if label.startswith("q") else []
        cases += [(c, k) for c in manifest["candidates"] for k in manifest["corners"]]
        for candidate, corner in cases:
            fixture = "dpt" if candidate is None else "ensemble"
            steps = (
                manifest["dpt_max_steps_s"]
                if candidate is None
                else manifest["candidate_max_steps_s"].get(
                    candidate["id"], manifest["ensemble_max_steps_s"]
                )
            )
            suffix = "" if candidate is None else f"-{candidate['id']}-{corner['id']}"
            record = {
                "id": f"{label}-{fixture}{suffix}",
                "fixture": fixture,
                "candidate": candidate,
                "corner": corner,
                "level": level,
                "max_step_s": steps[level],
                "sample_step_s": manifest[f"{fixture}_sample_steps_s"][level],
                "manifest_sha256": mhash,
                "status": "timeout",
                "files": {},
                "error": "independently authored timeout fixture",
            }
            job = directory / "jobs" / record["id"]
            job.mkdir(parents=True)
            write_document(job / "result.json", record)
            records.append(record)
        if label == "q2" or label.startswith("w"):
            group = records[:30] if label == "q2" else records[start:]
            count = len(group)
            summary = {
                "id": "qualification-study" if label == "q2" else label,
                "workers": 4 if label == "q2" else int(label[1]),
                "expected": count,
                "terminal": count,
                "validated": 0,
                "failures": count,
                "job_ids": [r["id"] for r in group],
            }
            if label.startswith("w"):
                summary.update(
                    {
                        "ranking": [
                            {
                                "candidate": c["id"],
                                "complete": True,
                                "predicted_feasible": False,
                                "mass_kg": study.circuits.design(c)["mass_kg"],
                            }
                            for c in manifest["candidates"]
                        ],
                        "lightest_feasible": None,
                        "time_to_lightest_feasible_s": None,
                    }
                )
            summaries.append(summary)
            write_document(directory / (summary["id"] + ".json"), summary)
    count = 30 if qualification_only else 102
    assert len(records) == count
    terminal = {
        "schema": "emi01-v1",
        "qualification_only": qualification_only,
        "job_ids": [r["id"] for r in records],
        "counts": {
            "expected": count,
            "terminal": count,
            "validated": 0,
            "failures": count,
        },
        "qualification_pass": False,
        "study_summaries": summaries,
        "result_hashes": {
            r["id"]: digest((directory / "jobs" / r["id"] / "result.json").read_bytes())
            for r in records
        },
        "files": {
            p.name: digest(p.read_bytes()) for p in directory.iterdir() if p.is_file()
        },
    }
    write_document(directory / "terminal.json", terminal)
    return terminal


def add_valid_ensemble_fixture(directory, terminal):
    """Synthetic zero-current trace tests audit recomputation, not circuit physics."""
    identity = "q0-ensemble-light-nominal"
    job = directory / "jobs" / identity
    record = json.loads((job / "result.json").read_bytes())
    names = study.circuits.STUDY_NAMES
    points = math.ceil(200e-6 / record["max_step_s"]) + 1
    raw = study.np.zeros((points, len(names)), dtype="<f8")
    raw[:, 0] = study.np.linspace(0, 200e-6, points)
    raw[:, names.index("v(p)")] = 400
    header = (
        "Title: emi-01 v1 light nominal\nDate: frozen fixture\n"
        "Command: ngspice-46\nPlotname: Transient Analysis\nFlags: real\n"
        f"No. Variables: {len(names)}\nNo. Points: {points}\nVariables:\n"
    )
    for index, name in enumerate(names):
        unit = (
            "time" if index == 0 else "current" if name.startswith("i(") else "voltage"
        )
        header += f"\t{index}\t{name}\t{unit}\n"
    header = (header + "Binary:\n").encode()
    payload = raw.tobytes()
    blobs = directory / "blobs"
    blobs.mkdir(exist_ok=True)
    chunks = []
    for start in range(0, len(payload), 4 * 1024 * 1024):
        chunk = payload[start : start + 4 * 1024 * 1024]
        hashed = digest(chunk)
        (blobs / (hashed + ".gz")).write_bytes(gzip.compress(chunk, mtime=0))
        chunks.append({"sha256": hashed, "bytes": len(chunk)})
    (job / "raw.header").write_bytes(header)
    write_document(
        job / "raw.json",
        {
            "raw_sha256": digest(header + payload),
            "payload_bytes": len(payload),
            "chunks": chunks,
        },
    )
    measured = {
        "accepted_steps": points,
        "rejected_steps": 0,
        "newton_iterations": points - 1,
        "total_iterations": points,
        "equations": 10,
        "nonzeros": 20,
        "analysis_s": 1.0,
        "load_s": 0.5,
        "factor_s": 0.2,
        "solve_s": 0.1,
        "reorder_s": 0.01,
        "truncation_s": 0.1,
        "netlist_loading_s": 0.01,
        "expansion_s": 0.01,
        "parsing_s": 0.01,
        "device_evaluation_s": None,
        "assembly_only_s": None,
        "unavailable_reason": "ngspice reports combined device evaluation and matrix load only",
    }
    labels = (
        ("Accepted timepoints", points),
        ("Rejected timepoints", 0),
        ("Transient iterations", points - 1),
        ("Total iterations", points),
        ("Circuit Equations", 10),
        ("Circuit total non-zeroes", 20),
        ("Total analysis time (seconds)", 1.0),
        ("Matrix load time", 0.5),
        ("Matrix factor time", 0.2),
        ("Matrix solve time", 0.1),
        ("Matrix reorder time", 0.01),
        ("Transient trunc time", 0.1),
        ("Netlist loading time", 0.01),
        ("Subckt and Param expansion time", 0.01),
        ("Netlist parsing time", 0.01),
    )
    (job / "simulator.log").write_text(
        "\n".join(f"{label} = {value}" for label, value in labels)
        + "\nngspice-46 done\n"
    )
    deck = study.circuits.ensemble(
        record["candidate"], record["corner"], record["max_step_s"]
    )
    (job / "circuit.cir").write_text(deck)
    (job / "driver.cir").write_text(study.circuits.driver())
    computed, spectrum = study.metrics.evaluate(
        raw, record["candidate"], record["corner"], record["sample_step_s"]
    )
    assert computed["status"] == "predicted_feasible"
    (job / "spectra.f64").write_bytes(spectrum.astype("<f8").tobytes())
    write_document(
        job / "spectra.json",
        {
            "columns": ["frequency_hz", "a_rms_a", "b_rms_a", "cm_rms_a", "dm_rms_a"],
            "shape": list(spectrum.shape),
            "dtype": "little-endian float64",
            "sha256": digest((job / "spectra.f64").read_bytes()),
        },
    )
    record.update(
        {
            "status": "predicted_feasible",
            "metrics": computed,
            "attempts": 1,
            "process": {"status": "ok", "wait_status": 0},
            "telemetry": measured,
            "raw_sha256": digest(header + payload),
            "raw_points": len(raw),
            "deck_sha256": digest(deck.encode()),
            "files": {
                p.name: digest(p.read_bytes())
                for p in job.iterdir()
                if p.name != "result.json"
            },
        }
    )
    write_document(job / "result.json", record)
    terminal["result_hashes"][identity] = digest((job / "result.json").read_bytes())
    terminal["counts"].update(validated=1, failures=29)
    terminal["study_summaries"][0].update(validated=1, failures=29)
    path = directory / "qualification-study.json"
    write_document(path, terminal["study_summaries"][0])
    terminal["files"][path.name] = digest(path.read_bytes())
    write_document(directory / "terminal.json", terminal)
    return record


class ReconciliationTests(unittest.TestCase):
    def test_heavy_override_applies_to_every_corner_and_finest_measurement(self):
        manifest = frozen()
        for level in range(3):
            specs = study.make_specs(manifest, {}, f"q{level}", level=level)
            self.assertEqual(len(specs), 9)
            for spec in specs:
                expected = (
                    0.625e-9 if spec["candidate"]["id"] == "heavy" else 2.5e-9
                ) / (2**level)
                self.assertEqual(spec["max_step_s"], expected)
                self.assertEqual(spec["sample_step_s"], 5e-9 / (2**level))
            dpt = study.make_specs(
                manifest, {}, f"q{level}", fixture="dpt", level=level
            )
            self.assertEqual(dpt[0]["max_step_s"], 2e-9 / (2**level))
        measured = study.make_specs(manifest, {}, "w4-sample2")
        self.assertEqual([s["max_step_s"] for s in measured[-3:]], [0.15625e-9] * 3)

    def test_telemetry_counts_must_be_integral_before_conversion(self):
        valid = "\n".join(label + " = 1" for label in study.TELEMETRY.values())
        self.assertEqual(study.telemetry(valid)["accepted_steps"], 1)
        damaged = valid.replace("Accepted timepoints = 1", "Accepted timepoints = 1.5")
        with self.assertRaisesRegex(
            ValueError, "^malformed_output: fractional telemetry count"
        ):
            study.telemetry(damaged)

    def test_ringing_refinement_requires_same_measured_polarity(self):
        measured = {
            "peak_v": 410.0,
            "on_edge_s": 10e-9,
            "off_edge_s": 12e-9,
            "on_half_s": 12e-6,
            "off_half_s": 14e-6,
            "on_energy_j": 100e-6,
            "off_energy_j": 110e-6,
            "ringing": {
                "status": "measured",
                "polarity": "positive",
                "frequency_hz": 20e6,
                "log_decrement": 0.5,
            },
        }
        self.assertTrue(study.dpt_compare(measured, measured, "same")["pass"])
        changed = copy.deepcopy(measured)
        changed["ringing"]["polarity"] = "negative"
        self.assertFalse(study.dpt_compare(measured, changed, "polarity")["pass"])
        changed["ringing"] = {"status": "unresolved"}
        self.assertFalse(study.dpt_compare(measured, changed, "missing")["pass"])

    def test_every_explicit_failure_stays_out_of_validated_count(self):
        statuses = (
            "predicted_feasible",
            "predicted_infeasible",
            "qualified",
            *FAILURES,
        )
        records = [
            {"id": str(i), "status": status} for i, status in enumerate(statuses)
        ]
        actual = study.reconcile(records, [str(i) for i in range(15)])
        self.assertEqual(
            actual, {"expected": 15, "terminal": 15, "validated": 3, "failures": 12}
        )

    def test_missing_duplicate_unknown_and_reordered_identity(self):
        records = [{"id": name, "status": "timeout"} for name in ("a", "b", "c")]
        cases = (
            records[:-1],
            records + [records[-1]],
            records[::-1],
            [records[0], records[0], records[2]],
            [records[0], {"id": "unknown", "status": "timeout"}, records[2]],
        )
        for damaged in cases:
            with (
                self.subTest(records=damaged),
                self.assertRaisesRegex(ValueError, "^missing_output:"),
            ):
                study.reconcile(damaged, ["a", "b", "c"])

    def test_unknown_terminal_status(self):
        with self.assertRaisesRegex(ValueError, "^malformed_output:"):
            study.reconcile([{"id": "a", "status": "running"}], ["a"])


class RankingTests(unittest.TestCase):
    def test_all_required_corners_and_qualification(self):
        self.assertTrue(
            all(
                r["predicted_feasible"]
                for r in study.ranking(candidate_records(), True)
            )
        )
        self.assertFalse(
            any(
                r["predicted_feasible"]
                for r in study.ranking(candidate_records(), False)
            )
        )

    def test_each_failure_and_infeasible_corner_excludes_candidate(self):
        for status in (*FAILURES, "predicted_infeasible", "qualified", "unknown"):
            records = candidate_records()
            records[2]["status"] = status
            with self.subTest(status=status):
                ranked = study.ranking(records, True)
                self.assertFalse(ranked[0]["predicted_feasible"])
                self.assertTrue(ranked[1]["predicted_feasible"])

    def test_missing_corner_excludes_candidate(self):
        records = candidate_records()
        del records[2]
        result = study.ranking(records, True)
        self.assertFalse(result[0]["complete"])
        self.assertFalse(result[0]["predicted_feasible"])

    def test_duplicate_reordered_unknown_or_modified_inputs_never_rank(self):
        base = candidate_records()
        damaged = []
        duplicate = copy.deepcopy(base)
        duplicate[2] = copy.deepcopy(duplicate[1])
        damaged.append(duplicate)
        damaged.append([base[1], base[0], *base[2:]])
        unknown_corner = copy.deepcopy(base)
        unknown_corner[2]["corner"]["id"] = "unmeasured_corner"
        damaged.append(unknown_corner)
        unknown_candidate = copy.deepcopy(base)
        extra = copy.deepcopy(base[0])
        extra["candidate"]["id"] = "unfrozen"
        unknown_candidate.append(extra)
        damaged.append(unknown_candidate)
        wrong_value = copy.deepcopy(base)
        wrong_value[2]["corner"]["bus_v"] = 1
        damaged.append(wrong_value)
        wrong_identity = copy.deepcopy(base)
        wrong_identity[2]["id"] = wrong_identity[1]["id"]
        damaged.append(wrong_identity)
        mixed_batches = copy.deepcopy(base)
        mixed_batches[2]["id"] = mixed_batches[2]["id"].replace("sample0", "sample1")
        damaged.append(mixed_batches)
        for index, records in enumerate(damaged):
            with self.subTest(index=index):
                self.assertFalse(
                    any(r["predicted_feasible"] for r in study.ranking(records, True))
                )


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.terminal = audit_fixture(self.directory)

    def save_terminal(self):
        write_document(self.directory / "terminal.json", self.terminal)

    def mutate_record(self, index, mutate):
        identity = self.terminal["job_ids"][index]
        path = self.directory / "jobs" / identity / "result.json"
        record = json.loads(path.read_bytes())
        mutate(record)
        write_document(path, record)
        self.terminal["result_hashes"][identity] = digest(path.read_bytes())
        self.save_terminal()

    def test_complete_failure_invocation_is_auditable_but_not_validated(self):
        before = {
            str(p.relative_to(self.directory)): digest(p.read_bytes())
            for p in self.directory.rglob("*")
            if p.is_file()
        }
        self.assertEqual(
            study.audit(self.directory),
            {"expected": 30, "terminal": 30, "validated": 0, "failures": 30},
        )
        after = {
            str(p.relative_to(self.directory)): digest(p.read_bytes())
            for p in self.directory.rglob("*")
            if p.is_file()
        }
        self.assertEqual(after, before, "audit must not rewrite evidence")

    def test_complete_full_schedule_has_exactly_102_terminal_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            full = Path(directory)
            audit_fixture(full, qualification_only=False)
            self.assertEqual(
                study.audit(full),
                {"expected": 102, "terminal": 102, "validated": 0, "failures": 102},
            )

    def test_self_consistent_truncation_is_rejected_against_frozen_schedule(self):
        omitted = self.terminal["job_ids"].pop()
        del self.terminal["result_hashes"][omitted]
        self.terminal["counts"] = {
            "expected": 29,
            "terminal": 29,
            "validated": 0,
            "failures": 29,
        }
        shutil.rmtree(self.directory / "jobs" / omitted)
        self.save_terminal()
        with self.assertRaisesRegex(ValueError, "^missing_output:"):
            study.audit(self.directory)

    def test_reordered_or_duplicate_terminal_schedule(self):
        for ids in (
            self.terminal["job_ids"][::-1],
            self.terminal["job_ids"][:-1] + [self.terminal["job_ids"][0]],
        ):
            original = self.terminal["job_ids"]
            self.terminal["job_ids"] = ids
            self.save_terminal()
            with self.assertRaisesRegex(ValueError, "^missing_output:"):
                study.audit(self.directory)
            self.terminal["job_ids"] = original

    def test_missing_result_is_explicit_failure(self):
        (self.directory / "jobs" / self.terminal["job_ids"][0] / "result.json").unlink()
        with self.assertRaisesRegex(ValueError, "^missing_output:"):
            study.audit(self.directory)

    def test_unknown_status_cannot_be_rehashed_into_success(self):
        self.mutate_record(1, lambda record: record.update(status="complete"))
        with self.assertRaisesRegex(ValueError, "^malformed_output:"):
            study.audit(self.directory)

    def test_wrong_corner_values_cannot_be_rehashed_into_valid_identity(self):
        self.mutate_record(1, lambda record: record["corner"].update(bus_v=999))
        with self.assertRaisesRegex(ValueError, "^provenance_mismatch:"):
            study.audit(self.directory)

    def test_changed_fixture_status_is_not_an_ensemble_validation(self):
        self.mutate_record(1, lambda record: record.update(status="qualified"))
        with self.assertRaisesRegex(ValueError, "^malformed_output:"):
            study.audit(self.directory)

    def test_unindexed_artifact_is_not_silently_ignored(self):
        (
            self.directory / "jobs" / self.terminal["job_ids"][0] / "simulator.log"
        ).write_text("failed simulation")
        with self.assertRaisesRegex(ValueError, "^missing_output:"):
            study.audit(self.directory)

    def test_success_requires_all_raw_and_spectral_artifacts(self):
        self.mutate_record(1, lambda record: record.update(status="predicted_feasible"))
        with self.assertRaisesRegex(ValueError, "^missing_output:"):
            study.audit(self.directory)

    def test_changed_model_metadata_is_not_accepted_after_rehash(self):
        path = self.directory / "metadata.json"
        metadata = json.loads(path.read_bytes())
        metadata["model"]["adapted_sha256"] = "0" * 64
        write_document(path, metadata)
        self.terminal["files"][path.name] = digest(path.read_bytes())
        self.save_terminal()
        with self.assertRaisesRegex(ValueError, "^provenance_mismatch:"):
            study.audit(self.directory)

    def test_heavy_cannot_rehash_default_grid_into_frozen_override(self):
        # q0 heavy/nominal is seventh ensemble entry, after the DPT entry.
        self.assertEqual(self.terminal["job_ids"][7], "q0-ensemble-heavy-nominal")
        self.mutate_record(7, lambda record: record.update(max_step_s=2.5e-9))
        with self.assertRaisesRegex(
            ValueError, "^provenance_mismatch: job input identity"
        ):
            study.audit(self.directory)

    def test_changed_source_metadata_is_not_accepted_after_rehash(self):
        path = self.directory / "metadata.json"
        metadata = json.loads(path.read_bytes())
        for mutation in ("changed", "missing", "extra"):
            damaged = copy.deepcopy(metadata)
            if mutation == "changed":
                damaged["source_sha256"]["reference/emi01/circuits.py"] = "0" * 64
            elif mutation == "missing":
                del damaged["source_sha256"]["reference/emi01/circuits.py"]
            else:
                damaged["source_sha256"]["unknown.py"] = "0" * 64
            write_document(path, damaged)
            self.terminal["files"][path.name] = digest(path.read_bytes())
            self.save_terminal()
            with (
                self.subTest(mutation=mutation),
                self.assertRaisesRegex(
                    ValueError, "^provenance_mismatch: invoking source files"
                ),
            ):
                study.audit(self.directory)

    def test_report_audit_ignores_generated_launcher_source_names(self):
        expected = {
            "reference/emi01/BUILD.bazel",
            "reference/emi01/CPU_GAPS.md",
            "reference/emi01/MODEL_PROVENANCE.md",
            "reference/emi01/README.md",
            "reference/emi01/adapter.py",
            "reference/emi01/circuits.py",
            "reference/emi01/manifest.json",
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
        }
        self.assertEqual(set(study.SOURCE_FILES), expected)
        self.assertEqual(set(study.source_identities()), expected)
        with tempfile.TemporaryDirectory() as scratch:
            source = Path(scratch) / "source"
            source.mkdir()
            here = source / "reference" / "emi01"
            evidence = Path(scratch) / "evidence"
            evidence.mkdir()
            for name in expected:
                (source / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(study.HERE.parent.parent / name, source / name)
            bootstrap = here / "_study_stage2_bootstrap.py"
            bootstrap.write_text("# generated study launcher\n")
            with mock.patch.object(study, "HERE", here):
                audit_fixture(evidence)
                bootstrap.unlink()
                (here / "_report_stage2_bootstrap.py").write_text(
                    "# different report launcher\n"
                )
                (here / "study_test.py").write_text(
                    "# target-specific tests are not runtime inputs\n"
                )
                self.assertEqual(report.study.audit(evidence)["terminal"], 30)
                (here / "metrics.py").write_text("# changed declared source\n")
                with self.assertRaisesRegex(
                    ValueError, "^provenance_mismatch: invoking source files"
                ):
                    report.study.audit(evidence)

    def test_manifest_requires_exact_bytes(self):
        path = self.directory / "manifest.json"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "^provenance_mismatch:"):
            study.audit(self.directory)

    def test_qualification_mode_cannot_claim_full_study(self):
        self.terminal["qualification_only"] = False
        self.save_terminal()
        with self.assertRaisesRegex(ValueError, "^missing_output:"):
            study.audit(self.directory)

    def test_self_consistent_qualification_pass_fails_raw_accounting(self):
        path = self.directory / "qualification.json"
        write_document(
            path,
            {
                "pass": True,
                "expected_checks": 40,
                "checks": [{"id": str(i), "pass": True} for i in range(40)],
            },
        )
        self.terminal["qualification_pass"] = True
        self.terminal["files"][path.name] = digest(path.read_bytes())
        self.save_terminal()
        with self.assertRaisesRegex(ValueError, "^malformed_output: qualification"):
            study.audit(self.directory)

    def test_nonfinite_completion_is_rejected_even_after_rehash(self):
        identity = self.terminal["job_ids"][0]
        path = self.directory / "jobs" / identity / "result.json"
        record = json.loads(path.read_bytes())
        record["elapsed_s"] = float("nan")
        path.write_text(json.dumps(record))
        self.terminal["result_hashes"][identity] = digest(path.read_bytes())
        self.save_terminal()
        with self.assertRaisesRegex(ValueError, "^non_finite:"):
            study.audit(self.directory)

    def test_forged_ranking_cannot_promote_failed_corners(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            terminal = audit_fixture(out, qualification_only=False)
            summary = terminal["study_summaries"][1]
            summary["ranking"][0]["predicted_feasible"] = True
            summary["lightest_feasible"] = "light"
            path = out / (summary["id"] + ".json")
            write_document(path, summary)
            terminal["files"][path.name] = digest(path.read_bytes())
            write_document(out / "terminal.json", terminal)
            with self.assertRaisesRegex(ValueError, "^malformed_output: ranking"):
                study.audit(out)

    def test_raw_metrics_spectrum_recomputed_and_restored_once(self):
        add_valid_ensemble_fixture(self.directory, self.terminal)
        with mock.patch.object(study, "restore", wraps=study.restore) as restore:
            self.assertEqual(study.audit(self.directory)["validated"], 1)
            self.assertEqual(restore.call_count, 1)

    def test_forged_metrics_after_rehash_are_rejected(self):
        add_valid_ensemble_fixture(self.directory, self.terminal)
        self.mutate_record(
            1, lambda record: record["metrics"]["stress"].update(device_peak_v=0)
        )
        with self.assertRaisesRegex(ValueError, "^provenance_mismatch: metrics/status"):
            study.audit(self.directory)

    def test_nonzero_child_exit_cannot_claim_valid_result(self):
        add_valid_ensemble_fixture(self.directory, self.terminal)
        self.mutate_record(1, lambda record: record["process"].update(wait_status=256))
        with self.assertRaisesRegex(ValueError, "^malformed_output: valid result"):
            study.audit(self.directory)

    def test_forged_spectrum_after_rehash_is_rejected(self):
        record = add_valid_ensemble_fixture(self.directory, self.terminal)
        job = self.directory / "jobs" / record["id"]
        path = job / "spectra.f64"
        data = study.np.frombuffer(path.read_bytes(), dtype="<f8").copy()
        data[1] = 1e-6
        path.write_bytes(data.tobytes())
        schema = json.loads((job / "spectra.json").read_bytes())
        schema["sha256"] = digest(path.read_bytes())
        write_document(job / "spectra.json", schema)

        def rehash(current):
            for name in ("spectra.f64", "spectra.json"):
                current["files"][name] = digest((job / name).read_bytes())

        self.mutate_record(1, rehash)
        with self.assertRaisesRegex(
            ValueError, "^provenance_mismatch: spectrum differs"
        ):
            study.audit(self.directory)

    def test_logged_error_cannot_claim_valid_result(self):
        record = add_valid_ensemble_fixture(self.directory, self.terminal)
        path = self.directory / "jobs" / record["id"] / "simulator.log"
        path.write_bytes(path.read_bytes() + b"Error: singular matrix\n")
        self.mutate_record(
            1,
            lambda current: current["files"].update(
                {"simulator.log": digest(path.read_bytes())}
            ),
        )
        with self.assertRaisesRegex(ValueError, "^numerical_failure:"):
            study.audit(self.directory)

    def test_forged_deck_after_rehash_is_rejected(self):
        record = add_valid_ensemble_fixture(self.directory, self.terminal)
        path = self.directory / "jobs" / record["id"] / "circuit.cir"
        path.write_bytes(
            path.read_bytes().replace(b"Vbus bus 0 400", b"Vbus bus 0 440")
        )

        def rehash(current):
            current["deck_sha256"] = digest(path.read_bytes())
            current["files"]["circuit.cir"] = current["deck_sha256"]

        self.mutate_record(1, rehash)
        with self.assertRaisesRegex(ValueError, "^provenance_mismatch: generated deck"):
            study.audit(self.directory)

    def test_rehashed_raw_title_must_match_candidate_corner(self):
        record = add_valid_ensemble_fixture(self.directory, self.terminal)
        job = self.directory / "jobs" / record["id"]
        header = job / "raw.header"
        header.write_bytes(
            header.read_bytes().replace(b"light nominal", b"heavy nominal")
        )
        index = json.loads((job / "raw.json").read_bytes())
        payload = b"".join(
            gzip.decompress(
                (self.directory / "blobs" / (chunk["sha256"] + ".gz")).read_bytes()
            )
            for chunk in index["chunks"]
        )
        index["raw_sha256"] = digest(header.read_bytes() + payload)
        write_document(job / "raw.json", index)

        def rehash(current):
            current["raw_sha256"] = index["raw_sha256"]
            for name in ("raw.header", "raw.json"):
                current["files"][name] = digest((job / name).read_bytes())

        self.mutate_record(1, rehash)
        with self.assertRaisesRegex(ValueError, "^provenance_mismatch: raw title"):
            study.audit(self.directory)


class RawReconstructionLimitsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.job = self.directory / "jobs" / "job"
        self.job.mkdir(parents=True)
        (self.directory / "blobs").mkdir()
        (self.job / "raw.header").write_bytes(b"header")
        self.index = {
            "raw_sha256": "0" * 64,
            "payload_bytes": 1,
            "chunks": [{"sha256": "1" * 64, "bytes": 1}],
        }
        self.record = {"id": "job", "fixture": "dpt", "max_step_s": 1e-9}

    def restore(self):
        write_document(self.job / "raw.json", self.index)
        return study.restore(self.directory, self.record)

    def test_chunk_path_and_schema_checked_before_file_access(self):
        self.index["chunks"][0]["sha256"] = "../../escape"
        with self.assertRaisesRegex(ValueError, "^malformed_output: raw chunk"):
            self.restore()

    def test_total_and_chunk_budgets_checked_before_allocation(self):
        for mutation in ("total", "chunk", "count", "fraction"):
            original = copy.deepcopy(self.index)
            if mutation == "total":
                self.index["payload_bytes"] = study.signals.MAX_RAW_BYTES + 1
            elif mutation == "chunk":
                self.index["chunks"][0]["bytes"] = 4 * 1024 * 1024 + 1
            elif mutation == "count":
                self.index["chunks"] *= (
                    math.ceil(study.signals.MAX_RAW_BYTES / (4 * 1024 * 1024)) + 1
                )
            else:
                self.index["chunks"][0]["bytes"] = 1.5
            with (
                self.subTest(mutation=mutation),
                self.assertRaisesRegex(ValueError, "^malformed_output:"),
            ):
                self.restore()
            self.index = original

    def test_decompression_is_bounded_even_for_false_declared_size(self):
        blob = self.directory / "blobs" / ("1" * 64 + ".gz")
        blob.write_bytes(gzip.compress(b"x" * (4 * 1024 * 1024 + 1), mtime=0))
        with self.assertRaisesRegex(
            ValueError, "^resource_limit: decompressed raw chunk"
        ):
            self.restore()

    def test_missing_or_invalid_gzip_chunk_has_typed_failure(self):
        with self.assertRaisesRegex(ValueError, "^missing_output: raw chunk"):
            self.restore()
        (self.directory / "blobs" / ("1" * 64 + ".gz")).write_bytes(b"not gzip")
        with self.assertRaisesRegex(ValueError, "^malformed_output: invalid gzip"):
            self.restore()


class WorkerFailureTests(unittest.TestCase):
    def test_batch_timer_can_include_preceding_job_materialization(self):
        record = {"id": "job", "status": "timeout", "elapsed_s": 1}
        pool = mock.MagicMock()
        pool.__enter__.return_value = pool
        pool.submit.return_value = mock.Mock(**{"result.return_value": record})
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(
                study.concurrent.futures, "ProcessPoolExecutor", return_value=pool
            ),
            mock.patch.object(study.time, "perf_counter", return_value=12),
        ):
            _, summary = study.batch(
                [{"id": "job"}], 1, Path(directory), "timed", started_at=10
            )
        self.assertEqual(summary["wall_s"], 2)

    def test_lightest_proof_waits_for_slower_infeasible_lighter_candidates(self):
        records = candidate_records()
        for record in records:
            heavy = record["candidate"]["id"] == "heavy"
            record.update(
                status="predicted_feasible" if heavy else "predicted_infeasible",
                elapsed_s=2 if heavy else 9,
                completion_latency_s=3 if heavy else 12,
            )
        pool = mock.MagicMock()
        pool.__enter__.return_value = pool
        pool.submit.side_effect = [
            mock.Mock(**{"result.return_value": r}) for r in records
        ]
        specs = [{"id": r["id"]} for r in records]
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(
                study.concurrent.futures, "ProcessPoolExecutor", return_value=pool
            ),
        ):
            _, summary = study.batch(
                specs, 4, Path(directory), "injected", qualified=True
            )
        self.assertEqual(summary["lightest_feasible"], "heavy")
        self.assertEqual(summary["time_to_lightest_feasible_s"], 12)

    def test_failed_future_has_durable_failure_and_remains_in_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            specs = [{"id": f"job-{index}"} for index in range(2)]
            future = mock.Mock()
            future.result.side_effect = RuntimeError("injected worker termination")
            pool = mock.MagicMock()
            pool.__enter__.return_value = pool
            pool.submit.return_value = future
            with mock.patch.object(
                study.concurrent.futures, "ProcessPoolExecutor", return_value=pool
            ):
                records, summary = study.batch(specs, 1, out, "injected")
            self.assertEqual(
                [r["status"] for r in records], ["internal_failure", "internal_failure"]
            )
            self.assertEqual(summary["expected"], 2)
            self.assertEqual(summary["terminal"], 2)
            self.assertEqual(summary["validated"], 0)
            self.assertEqual(summary["failures"], 2)
            self.assertEqual(summary["validated_jobs_per_hour"], 0)
            for spec in specs:
                saved = json.loads(
                    (out / "jobs" / spec["id"] / "result.json").read_bytes()
                )
                self.assertEqual(saved["status"], "internal_failure")
                self.assertEqual(saved["id"], spec["id"])


if __name__ == "__main__":
    unittest.main()
