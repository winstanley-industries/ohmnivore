"""Audited two-invocation CPU performance and finite-candidate report."""

import argparse
from collections import Counter
import heapq
import json
import math
from pathlib import Path
import statistics

from reference.emi01 import circuits, metrics, study


def makespan(costs, workers):
    """FIFO list scheduling of independent complete jobs on identical CPU lanes."""
    if type(workers) is not int or workers < 1:
        raise ValueError("unsupported_input: positive integer worker count required")
    lanes = [0.0] * workers
    for cost in costs:
        if not math.isfinite(cost) or cost < 0:
            raise ValueError("non_finite: job service must be finite and nonnegative")
        ready = heapq.heappop(lanes)
        heapq.heappush(lanes, ready + cost)
    return max(lanes)


def nearest_rank(values, percentile):
    if (
        not values
        or not 0 < percentile <= 1
        or any(not math.isfinite(v) or v < 0 for v in values)
    ):
        raise ValueError(
            "unsupported_input: finite nonnegative latency sample and percentile required"
        )
    return sorted(values)[math.ceil(percentile * len(values)) - 1]


def counterfactual_budget(sample, members, workers, qualified=True):
    """Change only the simulator-process phase; preserve all other measured service."""
    if not qualified:
        return {
            "study": sample["id"],
            "status": "unavailable",
            "reason": "reference qualification failed; no fixed-accuracy acceleration budget",
        }
    unavailable = [
        r["id"]
        for r in members
        if r.get("status") not in study.VALID
        or "simulation_s" not in r.get("phases", {})
    ]
    if unavailable:
        return {
            "study": sample["id"],
            "status": "unavailable",
            "reason": "failed or incomplete jobs cannot establish a fixed-accuracy acceleration budget",
            "job_ids": unavailable,
        }
    wall = sample["wall_s"]
    if not math.isfinite(wall) or wall <= 0 or not members:
        raise ValueError(
            "malformed_output: positive measured study time and jobs required"
        )
    service = [r["elapsed_s"] for r in members]
    simulator = [r["phases"]["simulation_s"] for r in members]
    baseline = makespan(service, workers)
    if any(
        not math.isfinite(sim) or sim < 0 or sim > total
        for sim, total in zip(simulator, service)
    ):
        raise ValueError(
            "malformed_output: simulator phase exceeds complete job service"
        )
    residual = wall - baseline
    overhead = max(0.0, residual)
    fixed = [total - sim for total, sim in zip(service, simulator)]
    predictions = {}
    for label, speed in [
        ("1", 1),
        ("2", 2),
        ("4", 4),
        ("10", 10),
        ("infinite", math.inf),
    ]:
        predicted = overhead + makespan(
            [other + sim / speed for other, sim in zip(fixed, simulator)], workers
        )
        predictions[label] = {
            "predicted_wall_s": predicted,
            "predicted_speedup": wall / predicted if predicted > 0 else None,
            "unbounded_ideal_model": predicted == 0,
        }
    return {
        "study": sample["id"],
        "status": "counterfactual",
        "fifo_model_wall_s": baseline,
        "measured_wall_s": wall,
        "baseline_residual_s": residual,
        "unmodeled_overhead_s": overhead,
        "calibration_status": "nonnegative_residual"
        if residual >= 0
        else "fifo_model_exceeds_measured_wall",
        "fixed_non_simulation_service_s": sum(fixed),
        "simulation_only_acceleration": predictions,
        "label": "Model only, no GPU measurement. Scale the complete simulator-process phase, including launch, netlist parsing, initialization, integration and raw output. Preserve other measured job service and nonnegative fitted scheduling/startup/summary overhead. A negative signed residual is exposed, not interpreted as negative overhead; speed=1 then does not reproduce measured wall time.",
    }


def linear_algebra_budget(telemetry, qualified=True):
    """Optimistic analysis-only Amdahl bound for infinite factor+solve acceleration."""
    if not qualified:
        return {
            "status": "unavailable",
            "reason": "reference qualification failed; no fixed-accuracy acceleration budget",
        }
    keys = ("analysis_s", "factor_s", "solve_s")
    if any(telemetry.get(key) is None for key in keys):
        return {
            "status": "unavailable",
            "reason": "analysis/factor/solve telemetry unavailable",
        }
    analysis, factor, solve = (telemetry[key] for key in keys)
    if any(not math.isfinite(v) or v < 0 for v in (analysis, factor, solve)):
        raise ValueError("non_finite: invalid linear algebra timing telemetry")
    linear = factor + solve
    if analysis <= 0 or linear > analysis:
        return {
            "status": "unavailable",
            "reason": "zero analysis duration or rounded component timers exceed analysis duration",
            "analysis_s": analysis,
            "factor_plus_solve_s": linear,
        }
    fraction = linear / analysis
    return {
        "status": "counterfactual",
        "analysis_s": analysis,
        "factor_plus_solve_s": linear,
        "factor_plus_solve_fraction_of_analysis": fraction,
        "optimistic_analysis_only_speedup_ceiling": 1 / (1 - fraction)
        if fraction < 1
        else None,
        "unbounded_ideal_model": fraction == 1,
        "label": "Optimistic Amdahl model for removing only measured matrix factorization+solve from ngspice analysis; zero acceleration overhead assumed. The remaining analysis work is serial. Factor/solve timers are nested inside analysis, not additional wall-time phases. Excludes process launch, external validation/output, scheduling, data transfer and GPU setup; not an end-to-end or measured GPU speedup.",
    }


def margin_detail(directory, record, manifest):
    """Locate the limiting retained spectral bin after the enclosing audit passes."""
    if record["status"] not in study.VALID:
        return {"status": record["status"], "minimum_margin_db": None}
    spectrum = study.np.fromfile(
        directory / "jobs" / record["id"] / "spectra.f64", dtype="<f8"
    ).reshape(-1, 5)
    row, column = study.np.unravel_index(
        study.np.argmax(spectrum[:, 1:]), spectrum[:, 1:].shape
    )
    margin = manifest["research_mask_dbua"] - float(
        study.signals.dbua(spectrum[row : row + 1, column + 1])[0]
    )
    return {
        "status": record["status"],
        "minimum_margin_db": margin,
        "distance_from_feasibility_db": margin - manifest["required_margin_db"],
        "worst_observable": ("a", "b", "cm", "dm")[column],
        "worst_frequency_hz": float(spectrum[row, 0]),
        "research_margin_db": record["metrics"]["research_margin_db"],
    }


def refinement_margins(directory, records, manifest):
    result = {}
    for candidate in manifest["candidates"]:
        levels = []
        for level in range(3):
            corners = [
                {
                    "corner": corner["id"],
                    **margin_detail(
                        directory,
                        records[f"q{level}-ensemble-{candidate['id']}-{corner['id']}"],
                        manifest,
                    ),
                }
                for corner in manifest["corners"]
            ]
            valid = [c for c in corners if c["minimum_margin_db"] is not None]
            levels.append(
                {
                    "level": level,
                    "corners": corners,
                    "worst": min(valid, key=lambda c: c["minimum_margin_db"])
                    if len(valid) == len(corners)
                    else None,
                }
            )
        result[candidate["id"]] = {
            "levels": levels,
            "classification_changes": [
                {
                    "corner": corner["id"],
                    "from_level": level - 1,
                    "to_level": level,
                    "from": levels[level - 1]["corners"][index]["status"],
                    "to": levels[level]["corners"][index]["status"],
                }
                for index, corner in enumerate(manifest["corners"])
                for level in (1, 2)
                if levels[level - 1]["corners"][index]["status"]
                != levels[level]["corners"][index]["status"]
            ],
        }
    return result


def performance_accepted(terminal):
    """A role-gate failure is neither a solver failure nor an accepted v2 budget."""
    role_pass = terminal.get("reference_case_pass", terminal["schema"] == "emi01-v1")
    return (
        terminal["qualification_pass"] is True
        and role_pass is True
        and terminal["counts"]["failures"] == 0
    )


def summarize(directory):
    study.audit(directory)
    terminal = json.loads((directory / "terminal.json").read_bytes())
    metadata = json.loads((directory / "metadata.json").read_bytes())
    manifest = study.selected_manifest(terminal["schema"])
    reference_case_pass = terminal.get("reference_case_pass", True)
    accepted = performance_accepted(terminal)
    if terminal["qualification_only"]:
        raise ValueError(
            "unsupported_input: performance report requires a complete study invocation"
        )
    records = {
        identity: json.loads(
            (directory / "jobs" / identity / "result.json").read_bytes()
        )
        for identity in terminal["job_ids"]
    }
    result = {
        "reference_version": manifest["schema"],
        "counts": terminal["counts"],
        "terminal_status_counts": dict(
            sorted(Counter(r["status"] for r in records.values()).items())
        ),
        "failures": [
            {"id": r["id"], "status": r["status"], "error": r.get("error")}
            for r in records.values()
            if r["status"] in study.FAILURES
        ],
        "qualified": terminal["qualification_pass"],
        "reference_case_pass": reference_case_pass,
        "accepted": accepted,
        "invocation_wall_s": terminal["invocation_wall_s"],
        "one_time_setup_s": metadata["setup_s"],
        "timing_boundary": "Batch study wall excludes once-per-invocation input verification, runtime/source verification and model adaptation, reported separately as one_time_setup_s. Complete invocation_wall_s includes that setup, qualification, warmups, all measured studies, and terminal completion records. Per-job preparation remains inside batch study wall.",
        "modes": {},
        "dpt": {
            "status": records["q2-dpt"]["status"],
            "metrics": records["q2-dpt"].get("metrics"),
        },
        "candidates": {},
    }
    for workers in manifest["workers"]:
        summaries = [
            x
            for x in terminal["study_summaries"]
            if x["id"].startswith(f"w{workers}-sample")
        ]
        jobs = [records[identity] for x in summaries for identity in x["job_ids"]]
        latency = sorted(x["elapsed_s"] for x in jobs)
        totals = {
            key: sum(x.get("phases", {}).get(key, 0) for x in jobs)
            for key in [
                "preparation_s",
                "simulation_s",
                "validation_s",
                "metrics_s",
                "required_output_s",
            ]
        }
        total_service = sum(x["elapsed_s"] for x in jobs)
        phases = {
            key: value / total_service if total_service > 0 else None
            for key, value in totals.items()
        }
        phases["unclassified_cleanup_hashing_fraction"] = (
            1 - sum(phases.values()) if total_service > 0 else None
        )
        budgets = [
            counterfactual_budget(
                sample,
                [records[i] for i in sample["job_ids"]],
                workers,
                qualified=accepted,
            )
            for sample in summaries
        ]
        available_telemetry = [x for x in jobs if "telemetry" in x]
        telemetry = {
            key: sum(x["telemetry"][key] for x in available_telemetry)
            if available_telemetry
            else None
            for key in [
                "analysis_s",
                "load_s",
                "factor_s",
                "solve_s",
                "accepted_steps",
                "rejected_steps",
                "newton_iterations",
            ]
        }
        result["modes"][str(workers)] = {
            "counts": {
                key: sum(sample[key] for sample in summaries)
                for key in ("expected", "terminal", "validated", "failures")
            },
            "terminal_status_counts": dict(
                sorted(Counter(r["status"] for r in jobs).items())
            ),
            "study_wall_s": [x["wall_s"] for x in summaries],
            "study_median_s": statistics.median(x["wall_s"] for x in summaries),
            "validated_jobs_per_hour": statistics.median(
                x["validated"] * 3600 / x["wall_s"] for x in summaries
            ),
            "job_median_s": statistics.median(latency),
            "job_p95_s": nearest_rank(latency, 0.95),
            "latency_label": f"Empirical nearest-rank P95 over {len(jobs)} measured jobs; {len(summaries)} studies do not estimate a population tail. Failed-job service may be unavailable/zero; inspect failure counts.",
            "phase_service_totals_s": totals,
            "phase_observed_jobs": {
                key: sum(key in x.get("phases", {}) for x in jobs) for key in totals
            },
            "service_fractions": phases,
            "job_service_s": total_service,
            "telemetry_sum": telemetry,
            "telemetry_jobs": len(available_telemetry),
            "telemetry_missing_job_ids": [
                r["id"] for r in jobs if "telemetry" not in r
            ],
            "linear_algebra_budget": linear_algebra_budget(
                telemetry, qualified=accepted
            ),
            "peak_child_rss_kib": max(
                (
                    x["process"]["peak_rss_kib"]
                    for x in jobs
                    if "peak_rss_kib" in x.get("process", {})
                ),
                default=None,
            ),
            "peak_worker_rss_kib": max(
                (x["worker_peak_rss_kib"] for x in jobs if "worker_peak_rss_kib" in x),
                default=None,
            ),
            "runner_peak_rss_kib": max(x["runner_peak_rss_kib"] for x in summaries),
            "counterfactual_budget": budgets,
        }
    for candidate in (c["id"] for c in manifest["candidates"]):
        jobs = [
            records[f"w1-sample0-ensemble-{candidate}-{corner}"]
            for corner in (k["id"] for k in manifest["corners"])
        ]
        result["candidates"][candidate] = {
            "mass_kg": circuits.design(jobs[0]["candidate"])["mass_kg"],
            "corners": [
                {
                    "id": j["corner"]["id"],
                    "status": j["status"],
                    "metrics": j.get("metrics"),
                }
                for j in jobs
            ],
        }
    if manifest["schema"] == "emi01-v2":
        result["reference_cases"] = json.loads(
            (directory / "reference-cases.json").read_bytes()
        )
        result["refinement_margins"] = refinement_margins(directory, records, manifest)
    return result


def paired_summaries(paths, reference_version=None):
    """Reject obvious reuse; distinct timing records are not cryptographic proof."""
    directories = [path.resolve() for path in paths]
    if len(directories) != 2 or len(set(directories)) != 2:
        raise ValueError("unsupported_input: two distinct invocation paths required")
    try:
        hashes = [
            study.sha((directory / "terminal.json").read_bytes())
            for directory in directories
        ]
    except OSError as exc:
        raise ValueError("missing_output: paired invocation terminal record") from exc
    if len(set(hashes)) != 2:
        raise ValueError(
            "provenance_mismatch: identical terminal records are not independent invocations"
        )
    versions = [
        json.loads((directory / "terminal.json").read_bytes()).get("schema")
        for directory in directories
    ]
    if (
        len(set(versions)) != 1
        or versions[0] not in study.MANIFESTS
        or (reference_version is not None and versions[0] != reference_version)
    ):
        raise ValueError(
            "provenance_mismatch: report requires the same selected reference version"
        )
    return [
        {**summarize(directory), "terminal_sha256": digest}
        for directory, digest in zip(directories, hashes)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, action="append")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--probe", type=Path)
    parser.add_argument(
        "--reference-version", choices=study.MANIFESTS, default="emi01-v1"
    )
    args = parser.parse_args()
    if args.probe:
        records = {
            p.parent.name: json.loads(p.read_bytes())
            for p in (args.probe / "jobs").glob("*/result.json")
        }
        result = {}
        arrays = []
        for level in range(3):
            r = records[f"q{level}-ensemble-light-nominal"]
            if r["status"] not in study.VALID:
                print(json.dumps({"failure": r}, indent=2))
                return
            arrays.append(study.restore(args.probe, r))
        for level in [1, 2]:
            result[f"integration{level}"] = metrics.compare(
                arrays[level - 1], arrays[level], circuits.STUDY_NAMES, 1.25e-9
            )
        for dt in [5e-9, 2.5e-9]:
            result[f"sampling{dt}"] = metrics.output_sampling_compare(
                arrays[-1], circuits.STUDY_NAMES, dt
            )
        print(json.dumps(result, indent=2))
        return
    if not args.run or len(args.run) != 2 or args.out is None:
        parser.error("two --run paths and --out required")
    result = {
        "schema": "emi01-cpu-budget-" + args.reference_version.split("-")[-1],
        "reference_version": args.reference_version,
        "runs": paired_summaries(args.run, args.reference_version),
        "independence_label": "Distinct invocation paths and terminal SHA256 records are required. This rejects accidental duplicate input/copies; independent execution provenance is not cryptographic proof of separate physical runs.",
        "budget_label": "Counterfactual FIFO model from measured complete-job service; no GPU implementation or GPU measurements. Infinite simulator-process acceleration retains validation, output and fitted scheduling overhead. Factor+solve-only Amdahl ceilings are a separate optimistic analysis-time model with nested timers.",
    }
    study.write_json(args.out, result)
    print(args.out)


if __name__ == "__main__":
    main()
