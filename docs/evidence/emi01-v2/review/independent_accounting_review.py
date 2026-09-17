"""Independent final accounting review; run from the repository root after timing.

Uses only the Python standard library. Does not decompress raw waveforms, run
simulations, or import production scheduling, metric, ranking, or audit helpers.
The canonical full raw audit and independent numerical review remain separate.
"""

import collections
import hashlib
import json
import math
import statistics
import struct
import subprocess
import sys
from pathlib import Path

if sys.flags.optimize:
    raise RuntimeError("Run this assertion-based review without Python optimization")
root = Path.cwd()
base = root / "docs/evidence/emi01-v2"
frozen = "8e45b4232741b82ec476fb6904b80f774eaf31d0"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def doc(p):
    return json.loads(p.read_bytes())


def near(a, b):
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-10), (a, b)


def mass(c):
    v = 0.010
    for k, n, w in [("dm", 2, 1), ("cm", 1, 2)]:
        ae, le, turns, mlt = c[k + "_geometry"]
        v += (
            1.2
            * n
            * (4800 * ae * 1e-6 * le * 0.001 + w * 8960 * turns * mlt * 0.001 * 4e-6)
        )
    for cap in (c["cx_f"], c["cy_f"], c["cy_f"]):
        v += 0.001 + 2 * (cap * 630**2 / 2) / 0.2 * 0.0012
    return v


m = doc(root / "reference/emi01/manifest-v2.json")
mh = digest((root / "reference/emi01/manifest-v2.json").read_bytes())
assert m["schema"] == "emi01-v2" and [c["id"] for c in m["candidates"]] == [
    "light",
    "boundary",
    "reference",
]
assert [c["id"] for c in m["corners"]] == ["nominal", "fast_low_lc", "hot_high_c"]
limits = {
    "capacitor_peak_v": 504,
    "cm_peak_t": 0.2,
    "damping_a_w": 12,
    "damping_b_w": 12,
    "device_peak_a": 50,
    "device_peak_v": 960,
    "dm_peak_t": 0.2,
    "loss_w": 25,
    "winding_rms_a": 12,
}
expected = []
groups = []
for level in range(3):
    expected.append((f"q{level}-dpt", "dpt", level, None, None))
    for c in m["candidates"]:
        for k in m["corners"]:
            expected.append(
                (f"q{level}-ensemble-{c['id']}-{k['id']}", "ensemble", level, c, k)
            )
groups.append(("qualification-study", 4, [e[0] for e in expected]))
for workers in (1, 4):
    for sample in ("warmup1", "sample0", "sample1", "sample2"):
        label = f"w{workers}-{sample}"
        added = []
        for c in m["candidates"]:
            for k in m["corners"]:
                jid = f"{label}-ensemble-{c['id']}-{k['id']}"
                expected.append((jid, "ensemble", 2, c, k))
                added.append(jid)
        groups.append((label, workers, added))
allrecords = []
allraw = []
output = {
    "review_method": "Independent standard-library schedule, hash, stored-spectrum, classification and timing arithmetic; no production helpers, waveform recomputation or simulation.",
    "implementation_sha256": digest(Path(__file__).read_bytes()),
    "frozen_source_commit": frozen,
    "manifest_sha256": mh,
    "invocation_protocol_sha256": digest(
        (base / "invocation-protocol.json").read_bytes()
    ),
    "runs": [],
}
for n in (1, 2):
    d = base / f"run-{n}"
    t = doc(d / "terminal.json")
    meta = doc(d / "metadata.json")
    records = {}
    raws = {}
    small = 0
    peaks = {}
    assert (
        t["schema"] == meta["schema"] == "emi01-v2" and t["qualification_only"] is False
    )
    assert t["qualification_pass"] is True and t["reference_case_pass"] is True
    assert t["counts"] == dict(expected=102, terminal=102, validated=102, failures=0)
    assert t["job_ids"] == [e[0] for e in expected] and sorted(t["job_ids"]) == sorted(
        p.name for p in (d / "jobs").iterdir()
    )
    assert set(t["result_hashes"]) == set(t["job_ids"])
    assert (d / "manifest.json").read_bytes() == (
        root / "reference/emi01/manifest-v2.json"
    ).read_bytes()
    assert meta["manifest_sha256"] == mh and len(meta["source_sha256"]) == 25
    for path, h in meta["source_sha256"].items():
        assert digest((root / path).read_bytes()) == h, (path, "working source")
        assert (
            digest(
                subprocess.check_output(["git", "show", f"{frozen}:{path}"], cwd=root)
            )
            == h
        ), (path, "frozen source")
    assert meta["resource_limits"] == m["limits"] and meta["runtime_threads"] == 1
    assert meta["oracle_snapshot"] == {
        "sha256": meta["ngspice_sha256"],
        "mode": "0500",
        "private": True,
    }
    assert len(meta["native_runtime_sha256"]) == 8
    assert (
        meta["model"]["archive_sha256"]
        == "de4a3acf222cbc7f6c1a4d1a2bc449dd31b5db2c6ed153920d797752c3831d51"
    )
    assert (
        meta["model"]["member_sha256"]
        == "6e888e103977f539e62b64952797ded391d2a49c59738bfd53f5fbe4b1fc8df7"
    )
    assert (
        meta["model"]["adapted_sha256"]
        == "17732ddd7ab5361073f8f23594e32158270af47d9b7188cf0ce773189cafcf96"
    )
    assert {p.name for p in d.iterdir() if p.is_file()} == set(t["files"]) | {
        "terminal.json"
    }
    for filename, h in t["files"].items():
        assert digest((d / filename).read_bytes()) == h, (n, filename)
    for jid, fixture, level, c, k in expected:
        jd = d / "jobs" / jid
        data = (jd / "result.json").read_bytes()
        assert digest(data) == t["result_hashes"][jid]
        r = json.loads(data)
        records[jid] = r
        assert (
            r["id"] == jid
            and r["fixture"] == fixture
            and r["reference_version"] == "emi01-v2"
            and r["level"] == level
        )
        assert (
            r["manifest_sha256"] == mh
            and r["candidate"] == c
            and r["corner"] == k
            and r["attempts"] == 1
        )
        assert (
            r["process"]["status"] == "ok"
            and r["process"]["wait_status"] == 0
            and "error" not in r
        )
        assert r["sample_step_s"] == m[fixture + "_sample_steps_s"][level]
        steps = (
            m["dpt_max_steps_s"]
            if c is None
            else m["candidate_max_steps_s"].get(c["id"], m["ensemble_max_steps_s"])
        )
        assert r["max_step_s"] == steps[level]
        for key in ("queue_s", "elapsed_s", "completion_latency_s"):
            assert math.isfinite(r[key]) and r[key] >= 0
        near(r["completion_latency_s"], r["queue_s"] + r["elapsed_s"])
        assert sum(r["phases"].values()) <= r["elapsed_s"] + 1e-6
        assert {p.name for p in jd.iterdir() if p.is_file()} == set(r["files"]) | {
            "result.json"
        }
        for filename, h in r["files"].items():
            assert digest((jd / filename).read_bytes()) == h, (n, jid, filename)
        small += len(r["files"])
        raws[jid] = doc(jd / "raw.json")
        assert (
            r["raw_sha256"] == raws[jid]["raw_sha256"]
            and r["deck_sha256"] == r["files"]["circuit.cir"]
        )
        assert r["raw_points"] <= m["limits"]["raw_points"]
        if fixture == "dpt":
            assert r["status"] == "qualified" and r["metrics"]["pass"] is True
            continue
        measured = r["metrics"]
        assert measured["stress_limits"] == limits and set(measured["stress"]) == set(
            limits
        )
        assert set(measured["research_margin_db"]) == {"a", "b", "cm", "dm"}
        assert all(
            x["pass"] is True and x["rms_difference_a"] <= x["limit_a"]
            for x in measured["settling"].values()
        )
        failures = [
            x for x, limit in limits.items() if measured["stress"][x] > limit
        ] + [
            "research_mask_" + x
            for x, v in measured["research_margin_db"].items()
            if v < 6
        ]
        assert sorted(failures) == sorted(measured["violations"])
        assert (
            r["status"]
            == measured["status"]
            == ("predicted_infeasible" if failures else "predicted_feasible")
        )
        near(mass(c), measured["mass_kg"])
        rows = list(struct.iter_unpack("<5d", (jd / "spectra.f64").read_bytes()))
        assert len(rows) == 986
        for i, row in enumerate(rows):
            near(row[0], 150000 + i * 10000)
            assert all(math.isfinite(v) and v >= 0 for v in row)
        for i, name in enumerate(("a", "b", "cm", "dm"), 1):
            near(
                90 - 20 * math.log10(max(max(row[i] for row in rows), 1e-15) / 1e-6),
                measured["research_margin_db"][name],
            )
        wi = max(range(len(rows)), key=lambda x: max(rows[x][1:]))
        col = max(range(1, 5), key=lambda x: rows[wi][x])
        peaks[jid] = {
            "frequency_hz": rows[wi][0],
            "observable": ("a", "b", "cm", "dm")[col - 1],
            "margin_db": min(measured["research_margin_db"].values()),
        }
    qualification = doc(d / "qualification.json")
    assert (
        qualification["pass"] is True
        and qualification["expected_checks"] == 40
        and len(qualification["checks"]) == 40
        and all(x["pass"] is True for x in qualification["checks"])
    )
    assert len(t["study_summaries"]) == 9
    for (label, workers, ids), summary in zip(groups, t["study_summaries"]):
        assert (
            summary == doc(d / (label + ".json"))
            and summary["id"] == label
            and summary["workers"] == workers
            and summary["job_ids"] == ids
        )
        assert {
            x: summary[x] for x in ("expected", "terminal", "validated", "failures")
        } == dict(expected=len(ids), terminal=len(ids), validated=len(ids), failures=0)
        rs = [records[i] for i in ids]
        near(summary["validated_jobs_per_hour"], len(rs) * 3600 / summary["wall_s"])
        near(summary["job_median_s"], statistics.median(r["elapsed_s"] for r in rs))
        near(
            summary["job_p95_s"],
            sorted(r["elapsed_s"] for r in rs)[math.ceil(0.95 * len(rs)) - 1],
        )
        assert summary["peak_child_rss_kib"] == max(
            r["process"]["peak_rss_kib"] for r in rs
        )
        if label == "qualification-study":
            continue
        ranked = []
        for c in m["candidates"]:
            corners = [r for r in rs if r["candidate"] == c]
            assert [r["corner"] for r in corners] == m["corners"]
            ranked.append(
                {
                    "candidate": c["id"],
                    "complete": True,
                    "predicted_feasible": all(
                        r["status"] == "predicted_feasible" for r in corners
                    ),
                    "mass_kg": corners[0]["metrics"]["mass_kg"],
                }
            )
        ranked.sort(key=lambda r: (r["mass_kg"], r["candidate"]))
        assert ranked == summary["ranking"]
        assert summary["lightest_feasible"] == "reference"
        near(
            summary["time_to_lightest_feasible_s"],
            max(r["completion_latency_s"] for r in rs),
        )
        for r in rs:
            q = records[f"q2-ensemble-{r['candidate']['id']}-{r['corner']['id']}"]
            assert (
                r["metrics"] == q["metrics"]
                and r["files"]["spectra.f64"] == q["files"]["spectra.f64"]
            )
            assert raws[r["id"]]["chunks"] == raws[q["id"]]["chunks"]
    roles = doc(d / "reference-cases.json")
    assert roles["pass"] is True and len(roles["groups"]) == 9
    for group in roles["groups"]:
        assert (
            group["pass"] is True
            and len(group["checks"]) == 3
            and all(
                x["pass"] is True and x["complete"] is True for x in group["checks"]
            )
        )
        for check in group["checks"]:
            cn = check["candidate"]
            q = [records[f"q2-ensemble-{cn}-{k['id']}"] for k in m["corners"]]
            worst = min(
                v for r in q for v in r["metrics"]["research_margin_db"].values()
            )
            near(check["minimum_margin_db"], worst)
            near(check["distance_from_feasibility_db"], worst - 6)
            if cn == "reference":
                assert (
                    all(r["status"] == "predicted_feasible" for r in q)
                    and check["all_physical_screens"] is True
                )
            elif cn == "boundary":
                assert (
                    5 <= worst <= 7
                    and worst < 6
                    and check["all_physical_screens"] is True
                    and all(
                        not any(
                            v > limits[key] for key, v in r["metrics"]["stress"].items()
                        )
                        for r in q
                    )
                )
            else:
                assert (
                    all(
                        r["status"] == "predicted_infeasible"
                        and r["metrics"]["violations"]
                        for r in q
                    )
                    and check["all_valid_settled"] is True
                    and check["all_corners_predicted_infeasible"] is True
                    and check["violations_consistent"] is True
                )
    assert (
        t["invocation_wall_s"]
        >= sum(s["wall_s"] for s in t["study_summaries"]) + meta["setup_s"]
    )
    modes = {}
    for w in (1, 4):
        ss = [s for s in t["study_summaries"] if s["id"].startswith(f"w{w}-sample")]
        rs = [records[i] for s in ss for i in s["job_ids"]]
        modes[str(w)] = {
            "study_median_s": statistics.median(s["wall_s"] for s in ss),
            "jobs_per_hour": statistics.median(
                s["validated_jobs_per_hour"] for s in ss
            ),
            "job_p95_s": sorted(r["elapsed_s"] for r in rs)[25],
            "lightest_median_s": statistics.median(
                s["time_to_lightest_feasible_s"] for s in ss
            ),
        }
    levels = {
        c["id"]: [
            {
                "level": lev,
                "worst": min(
                    (
                        {
                            "corner": k["id"],
                            **peaks[f"q{lev}-ensemble-{c['id']}-{k['id']}"],
                        }
                        for k in m["corners"]
                    ),
                    key=lambda r: r["margin_db"],
                ),
                "statuses": [
                    records[f"q{lev}-ensemble-{c['id']}-{k['id']}"]["status"]
                    for k in m["corners"]
                ],
            }
            for lev in range(3)
        ]
        for c in m["candidates"]
    }
    output["runs"].append(
        {
            "run": n,
            "metadata_sha256": digest((d / "metadata.json").read_bytes()),
            "terminal_sha256": digest((d / "terminal.json").read_bytes()),
            "small_job_artifacts_verified": small,
            "source_files_verified": len(meta["source_sha256"]),
            "status_counts": dict(
                collections.Counter(r["status"] for r in records.values())
            ),
            "qualification_checks": 40,
            "role_groups": 9,
            "invocation_wall_s": t["invocation_wall_s"],
            "setup_s": meta["setup_s"],
            "modes": modes,
            "refinement_levels": levels,
            "peak_child_rss_kib": max(
                r["process"]["peak_rss_kib"] for r in records.values()
            ),
            "peak_worker_rss_kib": max(
                r["worker_peak_rss_kib"] for r in records.values()
            ),
            "pass": True,
        }
    )
    allrecords.append(records)
    allraw.append(raws)
for jid in allrecords[0]:
    a, b = allrecords[0][jid], allrecords[1][jid]
    assert (
        a["metrics"] == b["metrics"]
        and a["status"] == b["status"]
        and a["candidate"] == b["candidate"]
        and a["corner"] == b["corner"]
    )
    assert allraw[0][jid]["chunks"] == allraw[1][jid]["chunks"]
    if a["fixture"] == "ensemble":
        assert a["files"]["spectra.f64"] == b["files"]["spectra.f64"]
meta1 = doc(base / "run-1/metadata.json")
meta2 = doc(base / "run-2/metadata.json")
for key in (
    "model",
    "source_sha256",
    "native_runtime_sha256",
    "python_executable_sha256",
    "ngspice_sha256",
    "manifest_sha256",
    "resource_limits",
    "oracle_snapshot",
):
    assert meta1[key] == meta2[key]
output["cross_run_exact_metrics_and_indexed_payload_matches"] = 102
output["within_run_finest_replay_matches_per_run"] = 72


def fifo(costs, workers):
    lanes = [0.0] * workers
    for cost in costs:
        lane = min(range(workers), key=lambda i: lanes[i])
        lanes[lane] += cost
    return max(lanes)


paired = doc(base / "cpu-budget.json")
assert (
    paired["schema"] == "emi01-cpu-budget-v2"
    and paired["reference_version"] == "emi01-v2"
)
assert len(paired["runs"]) == 2
output["cpu_budget_sha256"] = digest((base / "cpu-budget.json").read_bytes())
output["counterfactual_samples_checked"] = 0
for n, (report, records) in enumerate(zip(paired["runs"], allrecords), 1):
    terminal = doc(base / f"run-{n}/terminal.json")
    assert report["terminal_sha256"] == output["runs"][n - 1]["terminal_sha256"]
    assert (
        report["qualified"] is True
        and report["reference_case_pass"] is True
        and report["accepted"] is True
    )
    assert report["counts"] == terminal["counts"] and report["failures"] == []
    near(report["invocation_wall_s"], terminal["invocation_wall_s"])
    near(report["one_time_setup_s"], output["runs"][n - 1]["setup_s"])
    assert report["terminal_status_counts"] == output["runs"][n - 1]["status_counts"]
    assert report["reference_cases"] == doc(base / f"run-{n}/reference-cases.json")
    for c in m["candidates"]:
        name = c["id"]
        near(report["candidates"][name]["mass_kg"], mass(c))
        for corner in report["candidates"][name]["corners"]:
            r = records[f"w1-sample0-ensemble-{name}-{corner['id']}"]
            assert corner["metrics"] == r["metrics"] and corner["status"] == r["status"]
        assert report["refinement_margins"][name]["classification_changes"] == []
        for level, detail in enumerate(report["refinement_margins"][name]["levels"]):
            expected_worst = output["runs"][n - 1]["refinement_levels"][name][level][
                "worst"
            ]
            assert detail["worst"]["corner"] == expected_worst["corner"]
            assert detail["worst"]["worst_observable"] == expected_worst["observable"]
            near(detail["worst"]["worst_frequency_hz"], expected_worst["frequency_hz"])
            near(detail["worst"]["minimum_margin_db"], expected_worst["margin_db"])
    for workers in (1, 4):
        key = str(workers)
        mode = report["modes"][key]
        summaries = [
            v
            for v in terminal["study_summaries"]
            if v["id"].startswith(f"w{workers}-sample")
        ]
        jobs = [records[i] for v in summaries for i in v["job_ids"]]
        assert len(jobs) == 27 and len(summaries) == 3
        assert mode["counts"] == dict(
            expected=27, terminal=27, validated=27, failures=0
        )
        near(mode["study_median_s"], statistics.median(v["wall_s"] for v in summaries))
        near(
            mode["validated_jobs_per_hour"],
            statistics.median(32400 / v["wall_s"] for v in summaries),
        )
        near(mode["job_median_s"], statistics.median(v["elapsed_s"] for v in jobs))
        near(mode["job_p95_s"], sorted(v["elapsed_s"] for v in jobs)[25])
        service = sum(v["elapsed_s"] for v in jobs)
        near(mode["job_service_s"], service)
        for phase, total in mode["phase_service_totals_s"].items():
            near(total, sum(v["phases"].get(phase, 0) for v in jobs))
            near(mode["service_fractions"][phase], total / service)
        near(
            mode["service_fractions"]["unclassified_cleanup_hashing_fraction"],
            1 - sum(mode["phase_service_totals_s"].values()) / service,
        )
        for field, value in mode["telemetry_sum"].items():
            near(value, sum(v["telemetry"][field] for v in jobs))
        la = mode["linear_algebra_budget"]
        assert la["status"] == "counterfactual"
        fraction = (
            mode["telemetry_sum"]["factor_s"] + mode["telemetry_sum"]["solve_s"]
        ) / mode["telemetry_sum"]["analysis_s"]
        near(la["factor_plus_solve_fraction_of_analysis"], fraction)
        near(la["optimistic_analysis_only_speedup_ceiling"], 1 / (1 - fraction))
        assert len(mode["counterfactual_budget"]) == 3
        for summary, budget in zip(summaries, mode["counterfactual_budget"]):
            assert (
                budget["study"] == summary["id"]
                and budget["status"] == "counterfactual"
            )
            members = [records[i] for i in summary["job_ids"]]
            service = [v["elapsed_s"] for v in members]
            simulation = [v["phases"]["simulation_s"] for v in members]
            fixed = [a - b for a, b in zip(service, simulation)]
            baseline = fifo(service, workers)
            residual = summary["wall_s"] - baseline
            overhead = max(0, residual)
            near(budget["fifo_model_wall_s"], baseline)
            near(budget["baseline_residual_s"], residual)
            near(budget["unmodeled_overhead_s"], overhead)
            near(budget["fixed_non_simulation_service_s"], sum(fixed))
            for speed, value in [
                ("1", 1),
                ("2", 2),
                ("4", 4),
                ("10", 10),
                ("infinite", math.inf),
            ]:
                wall = overhead + fifo(
                    [a + b / value for a, b in zip(fixed, simulation)], workers
                )
                near(
                    budget["simulation_only_acceleration"][speed]["predicted_wall_s"],
                    wall,
                )
                near(
                    budget["simulation_only_acceleration"][speed]["predicted_speedup"],
                    summary["wall_s"] / wall,
                )
            output["counterfactual_samples_checked"] += 1
    for field, metric in [("process", "peak_rss_kib")]:
        assert report["modes"]["1"]["peak_child_rss_kib"] == max(
            r["process"]["peak_rss_kib"]
            for r in records.values()
            if r["id"].startswith("w1-sample")
        )
        assert report["modes"]["4"]["peak_child_rss_kib"] == max(
            r["process"]["peak_rss_kib"]
            for r in records.values()
            if r["id"].startswith("w4-sample")
        )

supplement_path = base / "delivery-supplement.json"
if supplement_path.exists():
    supplement = doc(supplement_path)
    assert supplement["cpu_budget_sha256"] == output["cpu_budget_sha256"]
    assert supplement["implementation_sha256"] == digest(
        (base / "review/delivery_supplement.py").read_bytes()
    )
    assert len(supplement["runs"]) == 2
    for n, item in enumerate(supplement["runs"], 1):
        assert (
            item["terminal_sha256"] == output["runs"][n - 1]["terminal_sha256"]
            and item["run"] == n
        )
        for workers, mode in item["modes"].items():
            jobs = [
                r
                for r in allrecords[n - 1].values()
                if r["id"].startswith(f"w{workers}-sample")
            ]
            for field in ("queue_s", "completion_latency_s"):
                values = sorted(r[field] for r in jobs)
                for k, v in dict(
                    min=min(values),
                    median=statistics.median(values),
                    p95=values[25],
                    max=max(values),
                ).items():
                    near(mode[field][k], v)
            report_mode = paired["runs"][n - 1]["modes"][workers]
            near(
                mode["cleanup_hashing_unclassified_s"],
                report_mode["job_service_s"]
                - sum(report_mode["phase_service_totals_s"].values()),
            )
            for speed, value in mode["median_counterfactual_wall_s"].items():
                near(
                    value,
                    statistics.median(
                        b["simulation_only_acceleration"][speed]["predicted_wall_s"]
                        for b in report_mode["counterfactual_budget"]
                    ),
                )
    output["delivery_supplement_sha256"] = digest(supplement_path.read_bytes())
    output["delivery_supplement_checked"] = True

output["pass"] = True
(base / "review/independent-accounting-review.json").write_text(
    json.dumps(output, indent=2) + "\n"
)
print(json.dumps(output, indent=2))
