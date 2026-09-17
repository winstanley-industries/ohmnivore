"""Failure accounting cannot turn a missing mandatory job into feasibility."""

from pathlib import Path
import concurrent.futures
import unittest
from unittest import mock
import tempfile
import hashlib
import json
import math
import shutil
import struct

from reference.emi01 import circuits, study
from reference.emi02 import qualification


class QualificationTest(unittest.TestCase):
    def test_execution_mode_and_worker_count_are_frozen(self):
        valid = {
            "workers": 4,
            "execution_mode": "concurrent-qualification-v1",
            "multiprocessing_start_method": "spawn",
        }
        self.assertTrue(qualification.execution_identity(valid))
        for key, value in (
            ("workers", 1),
            ("workers", 8),
            ("workers", 4.0),
            ("workers", True),
            ("execution_mode", "benchmark"),
            ("multiprocessing_start_method", "fork"),
        ):
            self.assertFalse(qualification.execution_identity({**valid, key: value}))
        for invalid in (None, [], {}):
            self.assertFalse(qualification.execution_identity(invalid))

    def test_concurrent_completion_restores_all_frozen_jobs_without_retry(self):
        specs = qualification.expected_specs(study.selected_manifest("emi01-v2"))
        pending = []

        def submit(function, job, *unused):
            future = concurrent.futures.Future()
            failed = function is qualification.run_cpu and job["id"] == specs[0]["id"]
            future.set_result(
                {"id": job["id"], "status": "resource_limit" if failed else "qualified"}
            )
            pending.append(future)
            return future

        with (
            tempfile.TemporaryDirectory() as temp,
            mock.patch.object(concurrent.futures, "ProcessPoolExecutor") as factory,
            mock.patch.object(
                concurrent.futures,
                "as_completed",
                side_effect=lambda futures: reversed(list(futures)),
            ),
            mock.patch("builtins.print") as printed,
        ):
            pool = factory.return_value.__enter__.return_value
            pool.submit.side_effect = submit
            out = Path(temp)
            cpu, reference = qualification.run_jobs(
                specs, Path("/unused-cpu"), Path("/unused-archive"), out
            )
            self.assertEqual(factory.call_args.kwargs["max_workers"], 4)
            self.assertEqual(
                factory.call_args.kwargs["mp_context"].get_start_method(), "spawn"
            )
            self.assertEqual(pool.submit.call_count, 60)
            self.assertEqual(len(set(pending)), 60)
            for records in (cpu, reference):
                self.assertEqual(
                    [row["id"] for row in records], [s["id"] for s in specs]
                )
            self.assertEqual(cpu[0]["status"], "resource_limit")
            self.assertEqual(reference[0]["status"], "qualified")
            self.assertEqual(
                qualification.read_json_bounded(out / "progress.json"),
                {"cpu": cpu, "reference": reference},
            )
            self.assertEqual(printed.call_count, 60)
            self.assertTrue(
                all(call.kwargs.get("flush") for call in printed.call_args_list)
            )

    def test_pool_misrouted_result_or_infrastructure_failure_cannot_complete(self):
        spec = qualification.expected_specs(study.selected_manifest("emi01-v2"))[0]
        for result in (
            {"id": "wrong-job"},
            {"id": spec["id"]},
            RuntimeError("worker unavailable"),
        ):
            with (
                tempfile.TemporaryDirectory() as temp,
                mock.patch.object(concurrent.futures, "ProcessPoolExecutor") as factory,
                mock.patch.object(
                    concurrent.futures,
                    "as_completed",
                    side_effect=lambda futures: list(futures),
                ),
            ):
                future = concurrent.futures.Future()
                if isinstance(result, Exception):
                    future.set_exception(result)
                else:
                    future.set_result(result)
                pool = factory.return_value.__enter__.return_value
                pool.submit.return_value = future
                with self.assertRaises((ValueError, RuntimeError)):
                    qualification.run_jobs(
                        [spec], Path("/unused-cpu"), Path("/unused-archive"), Path(temp)
                    )
                self.assertFalse((Path(temp) / "qualification.json").exists())

    def test_mandatory_metadata_matches_actual_output(self):
        valid = {
            "schema": "emi02-cpu-v1",
            "status": "complete",
            "points": 10,
            "variables": 5,
            "unknowns": 80,
            "raw_bytes": 700,
            "attempts": 12,
            "rejected_steps": 3,
            "elapsed_seconds": 0.25,
        }
        qualification.validate_statistics(valid, 10, 5, 80, 700)
        faults = [
            ("status", "failed"),
            ("schema", "other"),
            ("points", 9),
            ("variables", 4),
            ("unknowns", 79),
            ("raw_bytes", 699),
            ("attempts", 8),
            ("rejected_steps", 13),
            ("rejected_steps", 2),
            ("elapsed_seconds", float("nan")),
            ("elapsed_seconds", -1),
            ("points", True),
        ]
        for key, value in faults:
            with (
                self.subTest(key=key, value=value),
                self.assertRaisesRegex(ValueError, "malformed_output"),
            ):
                qualification.validate_statistics({**valid, key: value}, 10, 5, 80, 700)
        for bad in [
            None,
            [],
            {},
            {key: value for key, value in valid.items() if key != "points"},
        ]:
            with self.assertRaisesRegex(ValueError, "malformed_output"):
                qualification.validate_statistics(bad, 10, 5, 80, 700)

    def test_resource_and_parser_failures_preserve_categories(self):
        self.assertEqual(
            qualification.cpu_failure_status(
                b"unsupported-size: local message", "numerical_failure"
            )[0],
            "resource_limit",
        )
        self.assertEqual(
            qualification.cpu_failure_status(
                b"parse: local model text", "numerical_failure"
            )[0],
            "parse_failure",
        )
        self.assertEqual(
            qualification.cpu_failure_status(
                b"non-finite: local model text", "numerical_failure"
            )[0],
            "non_finite",
        )
        self.assertEqual(
            qualification.cpu_failure_status(b"parse: local model text", "timeout")[0],
            "timeout",
        )

    def test_audit_rejects_unsafe_or_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "record.json"
            for text in ['{"a": 1, "a": 2}', '{"a": NaN}', '{"a": 1e999}', "not json"]:
                path.write_text(text)
                with self.assertRaises(ValueError):
                    qualification.read_json_bounded(path)
            path.write_text("{} " * 100)
            with self.assertRaisesRegex(ValueError, "resource_limit"):
                qualification.read_json_bounded(path, limit=16)
            path.write_text('{"a": 1}')
            self.assertEqual(qualification.read_json_bounded(path), {"a": 1})

    def test_audit_binds_files_and_rejects_path_substitution(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            path = directory / "import.json"
            path.write_bytes(b"{}")
            record = {"files": {"import.json": hashlib.sha256(b"{}").hexdigest()}}
            qualification.verify_record_files(directory, record, True)
            path.write_bytes(b"[]")
            with self.assertRaisesRegex(ValueError, "provenance_mismatch"):
                qualification.verify_record_files(directory, record, True)
            path.write_bytes(b"{}")
            for name in ["../outside", "unknown.json"]:
                with self.assertRaisesRegex(ValueError, "malformed_output"):
                    qualification.verify_record_files(
                        directory, {"files": {name: "a" * 64}}, True
                    )
            path.unlink()
            with self.assertRaisesRegex(ValueError, "malformed_output"):
                qualification.verify_record_files(directory, record, True)
            path.symlink_to(directory / "absent")
            with self.assertRaisesRegex(ValueError, "resource_limit"):
                qualification.verify_record_files(directory, record, True)

    def test_audit_rejects_missing_duplicate_and_reordered_jobs(self):
        manifest = study.selected_manifest("emi01-v2")
        specs = qualification.expected_specs(manifest)
        for records in [
            [],
            specs[:-1],
            [specs[0]] * 30,
            list(reversed(specs)),
            [None, *specs],
            [None, *specs[1:]],
        ]:
            with self.assertRaisesRegex(ValueError, "malformed_output"):
                qualification.audit_records(
                    Path("/missing"), records, manifest, Path("/no-archive"), True
                )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "invocation.json").write_text(
                json.dumps({"schema": "emi02-qualification-v1", "probe_only": True})
            )
            (root / "qualification.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "provenance_mismatch"):
                qualification.audit(
                    root, Path("/no-archive"), Path("/no-cpu"), Path("/no-ngspice")
                )

    def test_external_equation_diagnostics_never_enter_published_artifacts(self):
        sentinel = "private-model-sentinel-12345"
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "published"
            out.mkdir()

            def failed_oracle(spec):
                directory = Path(spec["out"]) / "jobs" / spec["id"]
                directory.mkdir(parents=True)
                (directory / "simulator.log").write_text(sentinel)
                (directory / "circuit.cir").write_text("public deck with include only")
                record = {
                    "id": spec["id"],
                    "status": "numerical_failure",
                    "error": sentinel,
                }
                study.write_json(directory / "result.json", record)
                return record

            with mock.patch.object(study, "run_job", side_effect=failed_oracle):
                record = qualification.run_reference({"out": str(out), "id": "q0-dpt"})
            self.assertEqual(record["status"], "numerical_failure")
            self.assertEqual(
                record["diagnostic_log_sha256"],
                hashlib.sha256(sentinel.encode()).hexdigest(),
            )
            for path in out.rglob("*"):
                if path.is_file():
                    self.assertNotIn(sentinel.encode(), path.read_bytes())

    def test_external_blob_is_never_visible_as_a_partial_copy(self):
        payload = b"synthetic numeric chunk for publication race"
        name = hashlib.sha256(payload).hexdigest() + ".gz"
        copyfile = shutil.copyfile
        observed = []
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "published"
            out.mkdir()

            def oracle(spec):
                private = Path(spec["out"])
                directory = private / "jobs" / spec["id"]
                directory.mkdir(parents=True)
                (private / "blobs").mkdir()
                (private / "blobs" / name).write_bytes(payload)
                return {"id": spec["id"], "status": "resource_limit"}

            def interrupted_copy(source, destination, *args, **kwargs):
                if Path(source).parent.name == "blobs":
                    Path(destination).write_bytes(payload[:5])
                    self.assertFalse((out / "blobs" / name).exists())
                    observed.append(Path(destination))
                    Path(destination).write_bytes(payload)
                    return destination
                return copyfile(source, destination, *args, **kwargs)

            with (
                mock.patch.object(study, "run_job", side_effect=oracle),
                mock.patch.object(shutil, "copyfile", side_effect=interrupted_copy),
            ):
                for identity in ("q0-dpt", "q1-dpt"):
                    qualification.run_reference({"out": str(out), "id": identity})
            self.assertEqual(len(observed), 1)
            self.assertFalse(observed[0].exists())
            self.assertEqual((out / "blobs" / name).read_bytes(), payload)

    def test_both_lanes_cannot_agree_on_nonfrozen_job_parameters(self):
        specs = qualification.expected_specs(study.selected_manifest("emi01-v2"))
        records = [
            {
                **spec,
                "deck_sha256": study.sha(qualification.source_deck(spec).encode()),
                "status": "qualified",
                "max_step_s": spec["max_step_s"] * 2,
            }
            for spec in specs
        ]
        with mock.patch.object(
            study,
            "restore",
            side_effect=AssertionError("wrong identity must not reach waveforms"),
        ):
            compared = qualification.compare_jobs(
                Path("/missing"), Path("/missing"), records, records
            )
        self.assertFalse(compared["identity_valid"])
        self.assertFalse(compared["pass"])
        self.assertEqual(len(compared["checks"]), 30)
        self.assertTrue(
            all(
                check["status"] == "provenance_mismatch" for check in compared["checks"]
            )
        )

    def test_failed_archived_waveform_still_binds_identity_and_title(self):
        manifest = study.selected_manifest("emi01-v2")
        records = [
            {
                **spec,
                "status": "resource_limit",
                "attempts": 1,
                "deck_sha256": study.sha(qualification.source_deck(spec).encode()),
                "files": {},
            }
            for spec in qualification.expected_specs(manifest)
        ]
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            for record in records:
                directory = out / "jobs" / record["id"]
                directory.mkdir(parents=True)
                study.write_json(directory / "result.json", record)
            first = records[0]
            directory = out / "jobs" / first["id"]
            title = qualification.source_deck(first).splitlines()[0].lower()
            points = math.ceil(16e-6 / first["max_step_s"]) + 1
            names = circuits.DPT_NAMES
            payload = b"".join(
                struct.pack("<5d", 16e-6 * i / (points - 1), 0, 0, 0, 0)
                for i in range(points)
            )

            def save_raw(raw_title):
                lines = [
                    "Title: " + raw_title,
                    "Plotname: Transient Analysis",
                    "Flags: real",
                    "No. Variables: " + str(len(names)),
                    "No. Points: " + str(points),
                    "Variables:",
                ]
                lines.extend(
                    f"{i} {name} "
                    + (
                        "time"
                        if i == 0
                        else "current"
                        if name.startswith("i(")
                        else "voltage"
                    )
                    for i, name in enumerate(names)
                )
                header, index = study.store_raw(
                    out, ("\n".join(lines) + "\nBinary:\n").encode() + payload
                )
                (directory / "raw.header").write_bytes(header)
                study.write_json(directory / "raw.json", index)
                first["files"] = {
                    name: study.sha((directory / name).read_bytes())
                    for name in ("raw.header", "raw.json")
                }
                first["raw_sha256"] = index["raw_sha256"]
                first["raw_points"] = points
                study.write_json(directory / "result.json", first)

            def audit_records():
                return qualification.audit_records(
                    out, records, manifest, Path("/unused-model"), False
                )

            save_raw(title)
            self.assertEqual(len(audit_records()), 30)
            for key, value in (("raw_sha256", "a" * 64), ("raw_points", points - 1)):
                save_raw(title)
                first[key] = value
                study.write_json(directory / "result.json", first)
                with (
                    self.subTest(key=key),
                    self.assertRaisesRegex(
                        ValueError, "provenance_mismatch: changed waveform identity"
                    ),
                ):
                    audit_records()
            save_raw("some-other-job")
            with self.assertRaisesRegex(ValueError, "waveform title differs"):
                audit_records()
            save_raw(title)
            (directory / "circuit.cir").write_text("wrong external deck")
            first["files"]["circuit.cir"] = study.sha(b"wrong external deck")
            study.write_json(directory / "result.json", first)
            with self.assertRaisesRegex(ValueError, "changed external deck"):
                audit_records()

    def test_success_claim_requires_successful_process_and_diagnostics(self):
        for cpu in (True, False):
            exit_key = "exit_code" if cpu else "wait_status"
            prefix = "process_log" if cpu else "diagnostic_log"
            record = {
                "process": {"status": "ok", exit_key: 0, "wall_s": 0.5},
                prefix + "_sha256": "a" * 64,
                prefix + "_bytes": 123,
            }
            qualification.validate_successful_process(record, cpu)
            for key, value in [
                ("status", "timeout"),
                (exit_key, 1),
                (exit_key, False),
                ("wall_s", -1),
                ("wall_s", float("nan")),
            ]:
                with (
                    self.subTest(cpu=cpu, key=key, value=value),
                    self.assertRaisesRegex(ValueError, "malformed_output"),
                ):
                    qualification.validate_successful_process(
                        {**record, "process": {**record["process"], key: value}}, cpu
                    )
            for suffix, value in [("_sha256", "bad"), ("_bytes", -1), ("_bytes", True)]:
                with self.assertRaisesRegex(ValueError, "malformed_output"):
                    qualification.validate_successful_process(
                        {**record, prefix + suffix: value}, cpu
                    )

    def test_all_behavioral_sources_are_bound_to_invocation(self):
        identities = qualification.identities()
        for name in (
            "cpp/src/nonlinear_internal.h",
            "cpp/src/expression.cc",
            "cpp/src/behavioral.cc",
            "cpp/src/nonlinear.cc",
            "cpp/src/transient.cc",
            "cpp/src/emi02_runner.cc",
            "cpp/include/ohmnivore/expression.h",
            "cpp/BUILD.bazel",
        ):
            self.assertIn(name, identities)
            self.assertEqual(len(identities[name]), 64)

    def test_all_frozen_jobs_and_refinements(self):
        manifest = study.selected_manifest("emi01-v2")
        specs = []
        for level in range(3):
            for fixture in ("dpt", "ensemble"):
                specs.extend(
                    study.make_specs(manifest, {}, f"q{level}", fixture, level)
                )
        self.assertEqual(len(specs), 30)
        self.assertEqual(len({spec["id"] for spec in specs}), 30)
        self.assertEqual(sum(spec["fixture"] == "dpt" for spec in specs), 3)
        for candidate in ("boundary", "reference"):
            finest = [
                s
                for s in specs
                if s["candidate"]
                and s["candidate"]["id"] == candidate
                and s["level"] == 2
            ]
            self.assertEqual(len(finest), 3)
            self.assertTrue(all(s["max_step_s"] == 1.5625e-10 for s in finest))

    def test_failed_or_missing_jobs_never_compare_as_success(self):
        root = Path("/nonexistent-emi02-test")
        failed = [{"id": f"job-{n}", "status": "numerical_failure"} for n in range(30)]
        for cpu, external in [
            ([], []),
            (failed, []),
            (failed, failed),
            (failed[:-1], failed),
            (failed, failed[:-1]),
        ]:
            result = qualification.compare_jobs(root, root, cpu, external)
            self.assertFalse(result["pass"])
            self.assertEqual(result["expected_checks"], 30)
            self.assertTrue(all(not row["pass"] for row in result["checks"]))

    def test_failed_corner_suppresses_candidate(self):
        manifest = study.selected_manifest("emi01-v2")
        records = study.make_specs(manifest, {}, "q2", "ensemble", 2)
        for record in records:
            record["status"] = "predicted_feasible"
        good = study.ranking(records, True, manifest)
        self.assertTrue(all(row["predicted_feasible"] for row in good))
        records[-1]["status"] = "missing_output"
        ranked = {
            row["candidate"]: row for row in study.ranking(records, True, manifest)
        }
        self.assertFalse(ranked["reference"]["predicted_feasible"])
        self.assertTrue(ranked["boundary"]["predicted_feasible"])
        self.assertFalse(
            any(
                row["predicted_feasible"]
                for row in study.ranking(records, False, manifest)
            )
        )

    def test_wrong_job_identity_fails_before_reading_waveforms(self):
        cpu = {
            "id": "q0-dpt",
            "status": "qualified",
            "fixture": "dpt",
            "deck_sha256": "a",
        }
        external = {**cpu, "deck_sha256": "b"}
        result = qualification.compare_jobs(
            Path("/missing"), Path("/missing"), [cpu], [external]
        )
        self.assertFalse(result["pass"])
        self.assertEqual(result["checks"][0]["status"], "provenance_mismatch")


if __name__ == "__main__":
    unittest.main()
