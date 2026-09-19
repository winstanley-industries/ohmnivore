"""Hostile EMI-03 evidence and frozen whole-study gate checks."""

import copy
import json
from pathlib import Path
import queue
import tempfile
import threading
import types
import unittest
from unittest import mock

from reference.emi01 import study, circuits
from reference.emi03 import ensemble


def raw_fixture(spec):
    names = circuits.DPT_NAMES
    count = round(16e-6 / spec["max_step_s"]) + 1
    table = study.np.zeros((count, len(names)))
    table[:, 0] = study.np.linspace(0, 16e-6, count)
    title = ensemble.qualification.source_deck(spec).splitlines()[0].lower()
    header = (
        f"Title: {title}\nPlotname: Transient Analysis\nFlags: real\n"
        f"No. Variables: {len(names)}\nNo. Points: {count}\nVariables:\n"
    )
    for index, name in enumerate(names):
        unit = (
            "time" if index == 0 else "current" if name.startswith("i(") else "voltage"
        )
        header += f"{index}\t{name}\t{unit}\n"
    return header.encode() + b"Binary:\n", table


def mode_fixture():
    return [
        {
            "engine": engine,
            "workers": workers,
            "size": size,
            "setup_s": 0.1,
            "peak_host_bytes": 1024,
            "resource_pass": True,
            "device_incremental_peak_bytes": 1024 if engine == "gpu" else 0,
            "peak_device_bytes_upper_bound": 1024 if engine == "gpu" else 0,
            "studies": [
                {
                    "pass": True,
                    "completed_jobs": size,
                    "expected_jobs": size,
                    "complete_study_s": 1.0 if engine == "gpu" else 3.0,
                }
                for _ in range(6)
            ],
        }
        for engine, counts in (
            ("cpu", ensemble.CPU_WORKERS),
            ("gpu", ensemble.GPU_WORKERS),
        )
        for workers in counts
        for size in (9, 36)
    ]


class EnsembleTest(unittest.TestCase):
    def test_exact_replica_major_workloads_bind_every_input(self):
        manifest = study.selected_manifest("emi01-v2")
        for count in (9, 36):
            specs = ensemble.replay_specs(manifest, count, "measured")
            self.assertEqual(len(specs), count)
            self.assertEqual(len({spec["id"] for spec in specs}), count)
            for replica in range(count // 9):
                group = specs[replica * 9 : (replica + 1) * 9]
                self.assertEqual([s["replica"] for s in group], [replica] * 9)
                expected = study.make_specs(manifest, {}, "q2", "ensemble", 2)
                for actual, reference in zip(group, expected):
                    self.assertEqual(actual["reference_id"], reference["id"])
                    self.assertEqual(
                        {k: actual[k] for k in reference if k != "id"},
                        {k: reference[k] for k in reference if k != "id"},
                    )
        for count in (0, 8, 18, 37, True, 9.0):
            with self.assertRaisesRegex(ValueError, "unsupported_input"):
                ensemble.replay_specs(manifest, count, "bad")

    def test_gpu_success_requires_actual_execution_zero_fallbacks_and_bounded_memory(
        self,
    ):
        valid = {
            "schema": "emi03-cuda-v1",
            "job_id": "/input.cir",
            "backend": "cuda-fp64",
            "gpu_fallbacks": 0,
            "peak_device_bytes": 4096,
            "successful_solves": 1,
            "expression_batches": 1,
            "maximum_device_bytes": ensemble.DEVICE_WORKER_BYTES,
            "current_device_bytes": 0,
            "outstanding_device_bytes": 0,
            "live_factorizations": 0,
            "cleanup_failures": 0,
            "allocation_failures": 0,
        }
        ensemble.validate_gpu_telemetry(valid)
        for key, value in (
            ("backend", "cpu"),
            ("gpu_fallbacks", 1),
            ("gpu_fallbacks", False),
            ("peak_device_bytes", 0),
            ("peak_device_bytes", ensemble.DEVICE_BYTES + 1),
            ("peak_device_bytes", ensemble.DEVICE_WORKER_BYTES + 1),
            ("peak_device_bytes", float("nan")),
            ("successful_solves", 0),
            ("successful_solves", True),
            ("expression_batches", 0),
            ("current_device_bytes", 1),
            ("outstanding_device_bytes", 1),
            ("live_factorizations", 1),
            ("cleanup_failures", 1),
            ("maximum_device_bytes", ensemble.DEVICE_BYTES),
            ("job_id", "relative"),
            ("schema", "other"),
        ):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                ensemble.validate_gpu_telemetry({**valid, key: value})

    def test_warmups_failures_missing_jobs_and_missing_modes_never_enter_speedup(self):
        valid = mode_fixture()
        result = ensemble.performance_gate(valid, 1, 1)
        self.assertTrue(result["pass"])
        self.assertFalse(result["dispatch_authorized"])
        corruptions = [
            lambda modes: modes.pop(),
            lambda modes: modes.append(copy.deepcopy(modes[0])),
            lambda modes: modes[0]["studies"].pop(),
            lambda modes: modes[0]["studies"][0].update({"pass": False}),
            lambda modes: modes[0]["studies"][3].update({"completed_jobs": 8}),
            lambda modes: modes[0].update({"peak_host_bytes": ensemble.HOST_BYTES + 1}),
        ]
        for corrupt in corruptions:
            modes = copy.deepcopy(valid)
            corrupt(modes)
            result = ensemble.performance_gate(modes, 1)
            self.assertFalse(result["pass"])
            self.assertEqual(result["comparisons"], [])

    def test_each_size_compares_to_fastest_cpu_and_cold_qualification_is_charged_once(
        self,
    ):
        modes = mode_fixture()
        result = ensemble.performance_gate(modes, 10, 1)
        self.assertFalse(result["pass"])
        self.assertTrue(
            all(row["median_speedup"] == 3 for row in result["comparisons"])
        )
        self.assertTrue(all(row["cold_speedup"] < 1 for row in result["comparisons"]))
        for mode in modes:
            if mode["engine"] == "cpu" and mode["workers"] == 16:
                for item in mode["studies"]:
                    item["complete_study_s"] = 1.1
        result = ensemble.performance_gate(modes, 1)
        self.assertFalse(result["pass"])
        self.assertTrue(
            all(row["cpu_median_workers"] == 16 for row in result["comparisons"])
        )

    def test_p95_is_empirical_nearest_rank_and_nonfinite_timings_fail_closed(self):
        self.assertEqual(ensemble.nearest_rank([1, 2, 3, 4, 99]), 99)
        for values in ([], [float("nan")], [float("inf")], [-1], [0], [True]):
            with self.assertRaises(ValueError):
                ensemble.nearest_rank(values)

    def test_crossed_physical_and_numerical_reference_identities_rejected(self):
        record = {
            "fixture": "dpt",
            "candidate": None,
            "corner": None,
            "level": 0,
            "max_step_s": 1e-9,
            "sample_step_s": 1e-9,
            "reference_version": "emi01-v2",
            "deck_sha256": "a",
            "import": {"model": "a"},
        }
        for key in ("fixture", "level", "max_step_s", "deck_sha256", "import"):
            with self.assertRaisesRegex(ValueError, "provenance_mismatch"):
                ensemble.compare_pair(None, record, None, {**record, key: "wrong"})

    def test_invalid_raw_and_typed_failures_retain_terminal_records_without_spectra(
        self,
    ):
        spec = ensemble.qualification.expected_specs(
            study.selected_manifest("emi01-v2")
        )[0]
        header, table = raw_fixture(spec)
        valid = header + table.astype("<f8").tobytes()
        nonfinite = table.copy()
        nonfinite[2, 1] = float("nan")
        reversed_time = table.copy()
        reversed_time[2, 0] = 0
        faults = [
            ("non_finite", header + nonfinite.astype("<f8").tobytes()),
            ("malformed_output", valid[:-8]),
            ("malformed_output", header + reversed_time.astype("<f8").tobytes()),
            ("provenance_mismatch", valid.replace(b"Title: ", b"Title: other ", 1)),
            ("missing_output", None),
            ("unsupported_input", "unsupported_input"),
            ("resource_limit", "resource_limit"),
        ]
        for status, raw in faults:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)

                def request(source, raw_path, stats_path, exhausted):
                    if isinstance(raw, str):
                        ensemble.fail(raw, "injected failure")
                    if raw is not None:
                        raw_path.write_bytes(raw)
                    return {"status": "ok", "exit_code": 0, "wall_s": 0.01}

                worker = types.SimpleNamespace(
                    request=request, limits={"file_bytes": 536870912}
                )
                available = queue.Queue()
                available.put(worker)
                pool = types.SimpleNamespace(
                    available=available, workers=[worker], exhausted=threading.Event()
                )
                with mock.patch.object(
                    ensemble.importer,
                    "import_archive",
                    return_value=("flat", {"mna_unknowns": 5}),
                ):
                    result = ensemble.run_job(
                        {**spec, "submitted": ensemble.time.perf_counter()},
                        "binary",
                        None,
                        directory,
                        pool,
                        False,
                        None,
                        None,
                    )
                self.assertEqual(result["status"], status)
                self.assertEqual(result["attempts"], 1)
                dest = directory / "jobs" / spec["id"]
                self.assertEqual(json.loads((dest / "result.json").read_text()), result)
                self.assertNotIn("metrics", result)
                self.assertFalse((dest / "spectra.f64").exists())
                self.assertEqual(
                    (dest / "invalid.raw.gz").exists(), isinstance(raw, bytes)
                )

    def test_gpu_address_space_is_not_confused_with_resident_memory(self):
        limits = {"file_bytes": 512, "address_bytes": 1024}
        with mock.patch.object(ensemble.resource, "setrlimit") as setting:
            ensemble._child_limits(limits, True)
            self.assertEqual(
                [call.args[0] for call in setting.call_args_list],
                [ensemble.resource.RLIMIT_FSIZE],
            )
        with mock.patch.object(ensemble.resource, "setrlimit") as setting:
            ensemble._child_limits(limits, False)
            self.assertEqual(
                [call.args[0] for call in setting.call_args_list],
                [ensemble.resource.RLIMIT_FSIZE, ensemble.resource.RLIMIT_AS],
            )

    def test_reply_at_boundary_rechecks_wall_cpu_exit_and_job_identity(self):
        source, raw, stats = map(Path, ("/input.cir", "/output.raw", "/stats.json"))
        good = {"status": "complete", "input": str(source), "exit_code": 0}
        cases = [
            ([0, 0.5, 1.1], [(0, 0)] * 3, good, "timeout"),
            ([0, 0.5, 0.6], [(0, 0), (0, 0), (111, 0)], good, "resource_limit"),
            (
                [0, 0.5, 0.6],
                [(0, 0)] * 3,
                {**good, "exit_code": True},
                "malformed_output",
            ),
            ([0, 0.5, 0.6], [(0, 0)] * 3, {**good, "exit_code": 1}, "malformed_output"),
            (
                [0, 0.5, 0.6],
                [(0, 0)] * 3,
                {**good, "input": "/other.cir"},
                "provenance_mismatch",
            ),
        ]
        for clocks, usages, reply, expected in cases:
            worker = ensemble.Worker.__new__(ensemble.Worker)
            worker.process = mock.Mock(pid=123)
            worker.owner = 0
            worker.thread_id = 123
            worker.shared = None
            worker.input = worker.process.stdin
            worker.output = worker.process.stdout
            worker.process.poll.return_value = None
            worker.limits = {"wall_s": 1, "cpu_s": 110}
            worker.log = mock.Mock()
            worker.selector = mock.Mock()
            worker.selector.select.return_value = [True]
            worker.buffer = b""
            worker.requests = 1
            worker.stop = mock.Mock()
            with (
                self.subTest(expected=expected),
                mock.patch.object(ensemble.time, "perf_counter", side_effect=clocks),
                mock.patch.object(ensemble, "process_usage", side_effect=usages),
                mock.patch.object(
                    ensemble.os, "read", return_value=json.dumps(reply).encode() + b"\n"
                ),
                self.assertRaisesRegex(ValueError, expected),
            ):
                worker._request(source, raw, stats, threading.Event())

    def test_worker_rejects_crossed_gpu_owner_replies_and_stops_shared_process(self):
        source, raw, stats = map(Path, ("/input.cir", "/output.raw", "/stats.json"))
        good = {
            "status": "complete",
            "input": str(source),
            "exit_code": 0,
            "owner": 3,
            "thread_id": 456,
        }
        for field, value in (
            ("owner", 4),
            ("owner", True),
            ("thread_id", 457),
            ("thread_id", True),
        ):
            worker = ensemble.Worker.__new__(ensemble.Worker)
            worker.process = mock.Mock(pid=123)
            worker.process.poll.return_value = None
            worker.shared = mock.Mock()
            worker.owner, worker.thread_id = 3, 456
            worker.input, worker.output = mock.Mock(), mock.Mock()
            worker.limits = {"wall_s": 120, "cpu_s": 110}
            worker.log, worker.selector = mock.Mock(), mock.Mock()
            worker.selector.select.return_value = [True]
            worker.buffer = b""
            with (
                self.subTest(field=field, value=value),
                mock.patch.object(ensemble, "process_usage", return_value=(0, 0)),
                mock.patch.object(
                    ensemble.os,
                    "read",
                    return_value=json.dumps({**good, field: value}).encode() + b"\n",
                ),
                self.assertRaisesRegex(ValueError, "crossed GPU owner"),
            ):
                worker._request(source, raw, stats, threading.Event())
            worker.shared.stop.assert_called_once()

    def test_worker_failure_preserves_request_identity_and_cost(self):
        worker = ensemble.Worker.__new__(ensemble.Worker)
        worker.process = mock.Mock(pid=123)
        worker.owner = 0
        worker.thread_id = 123
        worker.shared = None
        worker.process.poll.return_value = None
        worker.requests = 2
        worker._request = mock.Mock(side_effect=ValueError("unsupported_input: fault"))
        with (
            mock.patch.object(
                ensemble, "process_usage", side_effect=[(1.0, 0), (1.4, 0)]
            ),
            self.assertRaisesRegex(ValueError, "unsupported_input"),
        ):
            worker.request(
                Path("/input"), Path("/raw"), Path("/stats"), threading.Event()
            )
        self.assertEqual(worker.last_process["worker_pid"], 123)
        self.assertEqual(worker.last_process["worker_request"], 3)
        self.assertAlmostEqual(worker.last_process["cpu_s"], 0.4)
        self.assertGreaterEqual(worker.last_process["wall_s"], 0)
        self.assertEqual(worker.last_process["status"], "unsupported_input")

    def test_failed_cpu_differential_publishes_no_feasible_metrics_or_spectrum(self):
        manifest = study.selected_manifest("emi01-v2")
        spec = ensemble.qualification.expected_specs(manifest)[1]
        title = ensemble.qualification.source_deck(spec).splitlines()[0].lower()
        table = study.np.zeros((2, len(circuits.STUDY_NAMES)))
        raw = f"Title: {title}\nBinary:\n".encode() + table.astype("<f8").tobytes()
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)

            def request(source, raw_path, stats_path, exhausted):
                raw_path.write_bytes(raw)
                stats_path.write_text("{}")
                return {"status": "ok", "exit_code": 0, "wall_s": 0.01}

            worker = types.SimpleNamespace(
                request=request, limits={"file_bytes": 536870912}
            )
            available = queue.Queue()
            available.put(worker)
            pool = types.SimpleNamespace(
                available=available, workers=[worker], exhausted=threading.Event()
            )
            reference = {
                **spec,
                "status": "predicted_feasible",
                "raw_sha256": "reference",
            }
            with (
                mock.patch.object(
                    ensemble.importer,
                    "import_archive",
                    return_value=("flat", {"mna_unknowns": 5}),
                ),
                mock.patch.object(ensemble.signals, "parse_raw", return_value=table),
                mock.patch.object(ensemble.qualification, "validate_statistics"),
                mock.patch.object(
                    ensemble.metrics,
                    "evaluate",
                    return_value=({"status": "predicted_feasible"}, table),
                ),
                mock.patch.object(ensemble.study, "restore", return_value=table),
                mock.patch.object(
                    ensemble, "compare_pair", return_value={"pass": False}
                ),
            ):
                result = ensemble.run_job(
                    {**spec, "submitted": ensemble.time.perf_counter()},
                    "binary",
                    None,
                    directory,
                    pool,
                    False,
                    {spec["id"]: reference},
                    directory,
                )
            self.assertEqual(result["status"], "accuracy_failure")
            self.assertNotIn("metrics", result)
            self.assertEqual(result["rejected_metrics"]["status"], "predicted_feasible")
            self.assertFalse((directory / "jobs" / spec["id"] / "spectra.f64").exists())

    def test_audit_recomputes_native_peaks_residency_delta_and_request_sequence(self):
        records = [
            {
                "process": {
                    "worker_pid": 12,
                    "worker_request": 1,
                    "worker_owner": 0,
                    "worker_thread": 13,
                    "cpu_accounting": "shared-process upper bound",
                },
                "gpu": {"peak_device_bytes": 10},
            },
            {
                "process": {
                    "worker_pid": 12,
                    "worker_request": 2,
                    "worker_owner": 0,
                    "worker_thread": 13,
                    "cpu_accounting": "shared-process upper bound",
                },
                "gpu": {"peak_device_bytes": 20},
            },
        ]
        workers = [{"pid": 12, "owner": 0, "thread_id": 13, "requests": 2}]
        retained = {
            "peak_device_bytes_upper_bound": 20,
            "device_observations_bytes": [100, 120, 110],
            "device_samples": 3,
            "device_baseline_bytes": 100,
            "device_sampled_peak_bytes": 120,
            "device_incremental_peak_bytes": 20,
            "peak_host_bytes": 1024,
            "monitor_errors": 0,
            "resource_pass": True,
        }
        self.assertTrue(ensemble.audit_resources(retained, records, workers, True))
        for key, value in (
            ("peak_device_bytes_upper_bound", 0),
            ("device_incremental_peak_bytes", 0),
            ("device_baseline_bytes", 120),
            ("device_samples", 1),
            ("peak_host_bytes", ensemble.HOST_BYTES + 1),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                ensemble.audit_resources(
                    {**retained, key: value}, records, workers, True
                )
        records[1]["process"]["worker_request"] = 1
        with self.assertRaisesRegex(ValueError, "request completion"):
            ensemble.audit_resources(retained, records, workers, True)

    def test_shared_process_keeps_all_owner_peaks_and_rejects_crossed_identities(self):
        workers = [
            {
                "pid": 100,
                "owner": owner,
                "thread_id": 101 + owner,
                "cpu_affinity": [ensemble.CPU_AFFINITY[owner]],
                "requests": 2,
            }
            for owner in range(16)
        ]
        records = [
            {
                "process": {
                    "worker_pid": 100,
                    "worker_owner": owner,
                    "worker_thread": 101 + owner,
                    "worker_request": request,
                    "cpu_accounting": "shared-process upper bound",
                },
                "gpu": {"peak_device_bytes": 10 * request},
            }
            for owner in range(16)
            for request in (1, 2)
        ]
        retained = {
            "peak_device_bytes_upper_bound": 320,
            "device_observations_bytes": [100, 420],
            "device_samples": 2,
            "device_baseline_bytes": 100,
            "device_sampled_peak_bytes": 420,
            "device_incremental_peak_bytes": 320,
            "peak_host_bytes": 1024,
            "monitor_errors": 0,
            "resource_pass": True,
        }
        self.assertTrue(ensemble.audit_resources(retained, records, workers, True))
        ensemble.audit_worker_affinity(workers, True, 16)
        for key, value in (
            ("worker_owner", 1),
            ("worker_owner", True),
            ("worker_thread", 102),
            ("worker_thread", None),
            ("cpu_accounting", "process delta"),
        ):
            changed = copy.deepcopy(records)
            changed[0]["process"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                ensemble.audit_resources(retained, changed, workers, True)
        for key, value in (
            ("pid", 200),
            ("owner", 1),
            ("thread_id", 102),
            ("cpu_affinity", [ensemble.CPU_AFFINITY[1]]),
        ):
            changed = copy.deepcopy(workers)
            changed[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                ensemble.audit_worker_affinity(changed, True, 16)
        with self.assertRaisesRegex(ValueError, "allocation accounting"):
            ensemble.audit_resources(
                {**retained, "peak_device_bytes_upper_bound": 20},
                records,
                workers,
                True,
            )

    def test_resource_snapshot_is_immutable_while_monitor_continues(self):
        pool = types.SimpleNamespace(
            device_lock=threading.Lock(),
            exhausted=threading.Event(),
            monitor_errors=0,
            peak_host_bytes=1024,
            device_peaks={(100, 0): 20},
            device_baseline_bytes=100,
            device_sampled_peak_bytes=100,
            device_samples=1,
            device_observations=[100],
        )
        before = ensemble.pool_resources(pool)
        ensemble.record_device_sample(pool, 150)
        after = ensemble.pool_resources(pool)
        self.assertEqual(before["device_observations_bytes"], [100])
        self.assertEqual(before["device_samples"], 1)
        self.assertEqual(after["device_observations_bytes"], [100, 150])
        self.assertEqual(after["device_samples"], 2)
        self.assertEqual(after["device_incremental_peak_bytes"], 50)

    def test_whole_invocation_monitor_follows_worker_and_oracle_descendants(self):
        contents = {
            "/proc/100/status": "VmHWM:\t1 kB\nVmRSS:\t1 kB\n",
            "/proc/200/status": "VmHWM:\t2 kB\nVmRSS:\t1 kB\n",
            "/proc/300/status": "VmHWM:\t3 kB\nVmRSS:\t1 kB\n",
            "/proc/100/task/100/children": "200",
            "/proc/200/task/200/children": "",
            "/proc/200/task/201/children": "300",
            "/proc/300/task/300/children": "",
        }

        def tasks(path):
            pid = path.parts[-2]
            return [path / pid] + ([path / "201"] if pid == "200" else [])

        with (
            mock.patch.object(ensemble.threading, "Thread"),
            mock.patch.object(ensemble.os, "getpid", return_value=100),
            mock.patch.object(
                Path,
                "read_text",
                autospec=True,
                side_effect=lambda path: contents[str(path)],
            ),
            mock.patch.object(Path, "iterdir", autospec=True, side_effect=tasks),
        ):
            monitor = ensemble.InvocationHostMonitor()
            monitor._sample()
            retained = monitor.snapshot()
            self.assertEqual(retained["peak_host_bytes"], 6 * 1024)
            self.assertEqual(retained["peak_observations"][0]["pids"], [100, 200, 300])
            self.assertTrue(ensemble.audit_host_resources(retained))
            with mock.patch.object(ensemble, "HOST_BYTES", 5000):
                monitor._sample()
                self.assertFalse(monitor.snapshot()["resource_pass"])
                self.assertFalse(ensemble.audit_host_resources(monitor.snapshot()))
                with self.assertRaisesRegex(ValueError, "memory gate"):
                    ensemble.audit_host_resources(
                        {**monitor.snapshot(), "resource_pass": True}
                    )

    def test_legacy_oracle_wall_overrun_prevents_qualification(self):
        records = [
            {"id": str(index), "process": {"wall_s": 119.9}} for index in range(30)
        ]
        self.assertTrue(
            ensemble.oracle_budget_check(records, records, {"wall_s": 120})["pass"]
        )
        for value in (120.0001, float("nan"), None, True):
            altered = copy.deepcopy(records)
            altered[0]["process"]["wall_s"] = value
            self.assertFalse(
                ensemble.oracle_budget_check(altered, records, {"wall_s": 120})["pass"]
            )

    def test_failed_completion_hashes_retained_files_and_rewrites_exact_terminal(self):
        spec = ensemble.qualification.expected_specs(
            study.selected_manifest("emi01-v2")
        )[0]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            provenance = {"source_deck_sha256": "source", "flat_deck_sha256": "flat"}
            study.write_json(directory / "import.json", provenance)
            study.write_json(directory / "result.json", {"id": "wrong"})
            for status in ("internal_failure", "provenance_mismatch"):
                record = ensemble.failed_completion(
                    spec, directory, status, "binary", True
                )
                self.assertEqual(record["id"], spec["id"])
                self.assertEqual(record["status"], status)
                self.assertEqual(
                    record["files"],
                    {
                        "import.json": study.sha(
                            (directory / "import.json").read_bytes()
                        )
                    },
                )
                self.assertEqual(
                    ensemble.qualification.read_json_bounded(directory / "result.json"),
                    record,
                )

    def test_failed_gpu_telemetry_is_auditable_but_cannot_diverge_from_sidecar(self):
        spec = ensemble.qualification.expected_specs(
            study.selected_manifest("emi01-v2")
        )[0]
        source = ensemble.qualification.source_deck(spec)
        telemetry = {
            "schema": "emi03-cuda-v1",
            "backend": "cuda-fp64",
            "job_id": "/input.cir",
            "peak_device_bytes": 10,
        }
        provenance = {
            "source_deck_sha256": study.sha(source.encode()),
            "flat_deck_sha256": study.sha(b"flat"),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "jobs" / spec["id"]
            directory.mkdir(parents=True)
            study.write_json(directory / "import.json", provenance)
            study.write_json(directory / "gpu.json", telemetry)
            record = ensemble.failed_completion(
                spec, directory, "numerical_failure", "binary", True
            )
            with mock.patch.object(
                ensemble.importer, "import_archive", return_value=("flat", provenance)
            ):
                ensemble.audit_gpu_records(
                    root, [record], [spec], None, "binary", {}, None
                )
                record["gpu"]["peak_device_bytes"] = 0
                study.write_json(directory / "result.json", record)
                with self.assertRaisesRegex(ValueError, "telemetry identity"):
                    ensemble.audit_gpu_records(
                        root, [record], [spec], None, "binary", {}, None
                    )


if __name__ == "__main__":
    unittest.main()
