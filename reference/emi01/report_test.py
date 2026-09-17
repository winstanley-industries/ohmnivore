"""Independent scheduling and optimistic performance-budget arithmetic tests."""

import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from reference.emi01 import report


def complete_jobs():
    # Deliberately uneven lanes and different fixed costs distinguish a FIFO
    # counterfactual from dividing aggregate time by a speed or worker count.
    return [
        {
            "id": str(index),
            "status": "predicted_feasible",
            "elapsed_s": total,
            "phases": {"simulation_s": simulation},
        }
        for index, (total, simulation) in enumerate([(10, 8), (8, 4), (6, 2)])
    ]


class SchedulingTest(unittest.TestCase):
    def test_fifo_makespan_uses_whole_job_service(self):
        self.assertEqual(report.makespan([3, 5, 1, 4], 1), 13)
        self.assertEqual(report.makespan([3, 5, 1, 4], 2), 8)
        self.assertEqual(report.makespan([3, 5, 1, 4], 4), 5)
        self.assertEqual(report.makespan([], 2), 0)

    def test_invalid_schedule_cannot_be_budgeted(self):
        for workers in [0, -1, 1.5, True]:
            with self.subTest(workers=workers), self.assertRaises(ValueError):
                report.makespan([1], workers)
        for cost in [-1, math.inf, math.nan]:
            with self.subTest(cost=cost), self.assertRaises(ValueError):
                report.makespan([cost], 1)

    def test_empirical_p95_has_declared_nearest_rank_definition(self):
        self.assertEqual(report.nearest_rank(list(range(27, 0, -1)), 0.95), 26)
        self.assertEqual(report.nearest_rank(list(range(1, 10)), 0.95), 9)
        self.assertEqual(report.nearest_rank([5], 0.95), 5)
        with self.assertRaises(ValueError):
            report.nearest_rank([], 0.95)


class WholeSimulatorBudgetTest(unittest.TestCase):
    def test_failed_global_qualification_blocks_complete_job_budget(self):
        result = report.counterfactual_budget(
            {"id": "sample", "wall_s": 17}, complete_jobs(), 2, qualified=False
        )
        self.assertEqual(result["status"], "unavailable")
        self.assertNotIn("simulation_only_acceleration", result)

    def test_counterfactual_preserves_fixed_service_and_overhead(self):
        result = report.counterfactual_budget(
            {"id": "sample", "wall_s": 17}, complete_jobs(), 2
        )
        self.assertEqual(result["fifo_model_wall_s"], 14)
        self.assertEqual(result["baseline_residual_s"], 3)
        self.assertEqual(result["unmodeled_overhead_s"], 3)
        self.assertEqual(result["fixed_non_simulation_service_s"], 10)
        model = result["simulation_only_acceleration"]
        self.assertEqual(model["1"]["predicted_wall_s"], 17)
        self.assertEqual(model["2"]["predicted_wall_s"], 14)
        self.assertEqual(model["infinite"]["predicted_wall_s"], 9)
        self.assertAlmostEqual(model["infinite"]["predicted_speedup"], 17 / 9)
        self.assertIn("no GPU measurement", result["label"])
        self.assertIn("raw output", result["label"])

    def test_negative_calibration_residual_is_exposed(self):
        result = report.counterfactual_budget(
            {"id": "sample", "wall_s": 12}, complete_jobs(), 2
        )
        self.assertEqual(result["baseline_residual_s"], -2)
        self.assertEqual(result["unmodeled_overhead_s"], 0)
        self.assertEqual(
            result["calibration_status"], "fifo_model_exceeds_measured_wall"
        )
        self.assertEqual(
            result["simulation_only_acceleration"]["1"]["predicted_wall_s"], 14
        )

    def test_failed_or_incomplete_jobs_do_not_establish_accuracy_budget(self):
        for mutation in ["failed", "missing_phase"]:
            jobs = complete_jobs()
            if mutation == "failed":
                jobs[1]["status"] = "timeout"
            else:
                jobs[1]["phases"] = {}
            result = report.counterfactual_budget(
                {"id": "sample", "wall_s": 17}, jobs, 2
            )
            self.assertEqual(result["status"], "unavailable")
            self.assertEqual(result["job_ids"], ["1"])
            self.assertNotIn("simulation_only_acceleration", result)

    def test_invalid_measured_phase_does_not_produce_speedup(self):
        jobs = complete_jobs()
        for simulation in [11, -1, math.nan, math.inf]:
            damaged = copy.deepcopy(jobs)
            damaged[0]["phases"]["simulation_s"] = simulation
            with self.subTest(simulation=simulation), self.assertRaises(ValueError):
                report.counterfactual_budget({"id": "sample", "wall_s": 17}, damaged, 2)

    def test_unbounded_ideal_limit_is_json_safe_and_explicit(self):
        jobs = [
            {
                "id": "only",
                "status": "predicted_feasible",
                "elapsed_s": 2,
                "phases": {"simulation_s": 2},
            }
        ]
        result = report.counterfactual_budget({"id": "sample", "wall_s": 2}, jobs, 1)
        ideal = result["simulation_only_acceleration"]["infinite"]
        self.assertEqual(ideal["predicted_wall_s"], 0)
        self.assertIsNone(ideal["predicted_speedup"])
        self.assertTrue(ideal["unbounded_ideal_model"])
        json.dumps(result, allow_nan=False)


class LinearAlgebraBudgetTest(unittest.TestCase):
    def test_failed_global_qualification_blocks_linear_algebra_budget(self):
        result = report.linear_algebra_budget(
            {"analysis_s": 10, "factor_s": 2, "solve_s": 1}, qualified=False
        )
        self.assertEqual(result["status"], "unavailable")

    def test_nested_analysis_timer_gives_factor_solve_only_ceiling(self):
        result = report.linear_algebra_budget(
            {"analysis_s": 10, "factor_s": 2, "solve_s": 1}
        )
        self.assertEqual(result["factor_plus_solve_s"], 3)
        self.assertEqual(result["factor_plus_solve_fraction_of_analysis"], 0.3)
        self.assertAlmostEqual(
            result["optimistic_analysis_only_speedup_ceiling"], 10 / 7
        )
        self.assertIn("nested inside analysis", result["label"])
        self.assertIn("zero acceleration overhead", result["label"])

    def test_missing_or_inconsistent_rounded_timers_do_not_imply_speedup(self):
        for telemetry in [
            {},
            {"analysis_s": 10, "factor_s": None, "solve_s": 1},
            {"analysis_s": 0, "factor_s": 0, "solve_s": 0},
            {"analysis_s": 1, "factor_s": 0.8, "solve_s": 0.3},
        ]:
            with self.subTest(telemetry=telemetry):
                result = report.linear_algebra_budget(telemetry)
                self.assertEqual(result["status"], "unavailable")
                self.assertNotIn("optimistic_analysis_only_speedup_ceiling", result)

    def test_linear_algebra_ideal_infinity_is_explicit_json_safe(self):
        result = report.linear_algebra_budget(
            {"analysis_s": 10, "factor_s": 9, "solve_s": 1}
        )
        self.assertTrue(result["unbounded_ideal_model"])
        self.assertIsNone(result["optimistic_analysis_only_speedup_ceiling"])
        json.dumps(result, allow_nan=False)

    def test_nonfinite_telemetry_rejected(self):
        with self.assertRaisesRegex(ValueError, "^non_finite:"):
            report.linear_algebra_budget(
                {"analysis_s": math.inf, "factor_s": 1, "solve_s": 1}
            )


class PairedInvocationsTest(unittest.TestCase):
    def test_report_cli_requires_explicit_v2_and_writes_matching_schema(self):
        with tempfile.TemporaryDirectory() as scratch:
            directory = Path(scratch)
            paths = [directory / "one", directory / "two"]
            for index, path in enumerate(paths):
                path.mkdir()
                (path / "terminal.json").write_text(
                    json.dumps({"schema": "emi01-v2", "time": index})
                )
            output = directory / "report.json"
            arguments = [
                "report",
                "--run",
                str(paths[0]),
                "--run",
                str(paths[1]),
                "--out",
                str(output),
            ]
            with (
                mock.patch.object(sys, "argv", arguments),
                self.assertRaisesRegex(ValueError, "^provenance_mismatch: report"),
            ):
                report.main()
            with (
                mock.patch.object(
                    sys, "argv", arguments + ["--reference-version=emi01-v2"]
                ),
                mock.patch.object(
                    report, "summarize", return_value={"reference_version": "emi01-v2"}
                ),
                mock.patch("builtins.print"),
            ):
                report.main()
            result = json.loads(output.read_bytes())
            self.assertEqual(result["schema"], "emi01-cpu-budget-v2")
            self.assertEqual(result["reference_version"], "emi01-v2")
            self.assertEqual(len(result["runs"]), 2)

    def test_mixed_or_wrong_selected_version_rejected_before_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / "first", Path(directory) / "second"]
            for index, path in enumerate(paths):
                path.mkdir()
                (path / "terminal.json").write_text(
                    json.dumps({"schema": f"emi01-v{index + 1}", "time": index})
                )
            with mock.patch.object(report, "summarize") as summarize:
                with self.assertRaisesRegex(ValueError, "^provenance_mismatch: report"):
                    report.paired_summaries(paths)
                (paths[0] / "terminal.json").write_text(
                    json.dumps({"schema": "emi01-v2", "time": 0})
                )
                with self.assertRaisesRegex(ValueError, "^provenance_mismatch: report"):
                    report.paired_summaries(paths, "emi01-v1")
                summarize.assert_not_called()

    def test_same_resolved_path_is_rejected_before_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory)
            with (
                mock.patch.object(report, "summarize") as summarize,
                self.assertRaisesRegex(ValueError, "^unsupported_input: two distinct"),
            ):
                report.paired_summaries([first, first / "unused" / ".."])
            summarize.assert_not_called()

    def test_identical_terminal_copy_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / "first", Path(directory) / "copy"]
            for path in paths:
                path.mkdir()
                (path / "terminal.json").write_bytes(b'{"time":1}\n')
            with (
                mock.patch.object(report, "summarize") as summarize,
                self.assertRaisesRegex(
                    ValueError, "^provenance_mismatch: identical terminal"
                ),
            ):
                report.paired_summaries(paths)
            summarize.assert_not_called()

    def test_distinct_terminal_identities_are_bound_into_report(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / "first", Path(directory) / "second"]
            expected = []
            for index, path in enumerate(paths):
                path.mkdir()
                data = json.dumps({"schema": "emi01-v1", "time": index}).encode()
                (path / "terminal.json").write_bytes(data)
                expected.append(report.study.sha(data))
            with mock.patch.object(
                report, "summarize", return_value={"counts": {}}
            ) as summarize:
                result = report.paired_summaries(paths)
            self.assertEqual([r["terminal_sha256"] for r in result], expected)
            self.assertEqual(summarize.call_count, 2)


class VersionedBudgetTest(unittest.TestCase):
    def test_role_gate_failure_or_omission_blocks_otherwise_qualified_budget(self):
        terminal = {
            "schema": "emi01-v2",
            "qualification_pass": True,
            "counts": {"failures": 0},
        }
        for value in (None, False, True):
            if value is not None:
                terminal["reference_case_pass"] = value
            accepted = report.performance_accepted(terminal)
            result = report.counterfactual_budget(
                {"id": "sample", "wall_s": 17}, complete_jobs(), 2, qualified=accepted
            )
            self.assertEqual(
                result["status"], "counterfactual" if value is True else "unavailable"
            )

    def test_worst_bin_margin_and_classification_changes_are_reported(self):
        manifest = report.study.selected_manifest("emi01-v2")
        manifest["candidates"] = [manifest["candidates"][1]]
        manifest["corners"] = [manifest["corners"][1]]
        records = {}
        with tempfile.TemporaryDirectory() as scratch:
            directory = Path(scratch)
            for level, margin in enumerate((5.5, 6.0, 6.5)):
                identity = f"q{level}-ensemble-boundary-fast_low_lc"
                record = {
                    "id": identity,
                    "status": "predicted_feasible"
                    if margin >= 6
                    else "predicted_infeasible",
                    "metrics": {
                        "research_margin_db": {"a": 10, "b": 10, "cm": margin, "dm": 10}
                    },
                }
                records[identity] = record
                job = directory / "jobs" / identity
                job.mkdir(parents=True)
                spectrum = report.study.np.array(
                    [
                        [150e3, 1e-6, 1e-6, 1e-6, 1e-6],
                        [250e3, 1e-6, 1e-6, 1e-6 * 10 ** ((90 - margin) / 20), 1e-6],
                    ],
                    dtype="<f8",
                )
                spectrum.tofile(job / "spectra.f64")
            result = report.refinement_margins(directory, records, manifest)["boundary"]
        self.assertEqual(len(result["classification_changes"]), 1)
        self.assertEqual(result["classification_changes"][0]["to_level"], 1)
        for level, margin in zip(result["levels"], (5.5, 6, 6.5)):
            worst = level["worst"]
            self.assertEqual(worst["corner"], "fast_low_lc")
            self.assertEqual(worst["worst_observable"], "cm")
            self.assertEqual(worst["worst_frequency_hz"], 250e3)
            self.assertAlmostEqual(worst["minimum_margin_db"], margin, places=12)
            self.assertAlmostEqual(
                worst["distance_from_feasibility_db"], margin - 6, places=12
            )


class SummaryAccountingTest(unittest.TestCase):
    def test_complete_failed_jobs_report_counts_without_metrics_or_false_throughput(
        self,
    ):
        # This isolates aggregation after audit; study_test independently tests
        # the complete frozen schedule and raw numerical validation.
        manifest = json.loads((report.study.HERE / "manifest.json").read_bytes())
        summaries = []
        records = [{"id": "q2-dpt", "status": "timeout", "error": "injected"}]
        for workers in (1, 4):
            for sample in range(3):
                label = f"w{workers}-sample{sample}"
                members = []
                for candidate in manifest["candidates"]:
                    for corner in manifest["corners"]:
                        identity = f"{label}-ensemble-{candidate['id']}-{corner['id']}"
                        records.append(
                            {
                                "id": identity,
                                "status": "timeout",
                                "candidate": candidate,
                                "corner": corner,
                                "elapsed_s": 1,
                            }
                        )
                        members.append(identity)
                summaries.append(
                    {
                        "id": label,
                        "job_ids": members,
                        "expected": 9,
                        "terminal": 9,
                        "validated": 0,
                        "failures": 9,
                        "wall_s": 20,
                        "runner_peak_rss_kib": 100,
                        "validated_jobs_per_hour": 99999,
                    }
                )
        terminal = {
            "schema": "emi01-v1",
            "qualification_only": False,
            "counts": {"expected": 55, "terminal": 55, "validated": 0, "failures": 55},
            "qualification_pass": False,
            "job_ids": [r["id"] for r in records],
            "study_summaries": summaries,
            "invocation_wall_s": 125,
        }
        with tempfile.TemporaryDirectory() as scratch:
            directory = Path(scratch)
            (directory / "terminal.json").write_text(json.dumps(terminal))
            (directory / "metadata.json").write_text(json.dumps({"setup_s": 3}))
            for record in records:
                job = directory / "jobs" / record["id"]
                job.mkdir(parents=True)
                (job / "result.json").write_text(json.dumps(record))
            with mock.patch.object(report.study, "audit") as audit:
                result = report.summarize(directory)
                audit.assert_called_once_with(directory)
        self.assertEqual(result["one_time_setup_s"], 3)
        self.assertEqual(len(result["failures"]), 55)
        self.assertIsNone(result["dpt"]["metrics"])
        for mode in result["modes"].values():
            self.assertEqual(mode["counts"]["failures"], 27)
            self.assertEqual(mode["counts"]["terminal"], 27)
            self.assertEqual(mode["validated_jobs_per_hour"], 0)
            self.assertEqual(mode["phase_observed_jobs"]["simulation_s"], 0)
            self.assertEqual(mode["telemetry_jobs"], 0)
            self.assertEqual(mode["linear_algebra_budget"]["status"], "unavailable")
            self.assertIsNone(mode["peak_child_rss_kib"])
            self.assertTrue(
                all(b["status"] == "unavailable" for b in mode["counterfactual_budget"])
            )


if __name__ == "__main__":
    unittest.main()
