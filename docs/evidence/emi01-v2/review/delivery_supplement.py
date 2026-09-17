"""Derive queue/completion quantiles from already audited EMI-01 job records.

Run from the repository root after the canonical paired report succeeds.
This is reporting arithmetic, not a substitute for the full evidence audit.
"""

import hashlib
import json
import math
from pathlib import Path
import statistics


def validate_report(paired):
    if (
        paired.get("schema") != "emi01-cpu-budget-v2"
        or paired.get("reference_version") != "emi01-v2"
        or len(paired.get("runs", [])) != 2
    ):
        raise ValueError("two audited v2 report entries required")
    if len({run["terminal_sha256"] for run in paired["runs"]}) != 2:
        raise ValueError("distinct invocation identities required")


def validate_binding(run, terminal_bytes):
    if hashlib.sha256(terminal_bytes).hexdigest() != run.get("terminal_sha256"):
        raise ValueError("paired report and invocation terminal identity disagree")
    terminal = json.loads(terminal_bytes)
    if (
        terminal.get("schema") != "emi01-v2"
        or run.get("reference_version") != "emi01-v2"
        or run.get("accepted") is not True
        or terminal.get("qualification_pass") is not True
        or terminal.get("reference_case_pass") is not True
        or terminal.get("qualification_only") is not False
        or terminal.get("counts")
        != {"expected": 102, "terminal": 102, "validated": 102, "failures": 0}
        or set(run.get("modes", {})) != {"1", "4"}
    ):
        raise ValueError("accepted complete v2 invocation required")
    for mode in run["modes"].values():
        budgets = mode.get("counterfactual_budget", [])
        if len(budgets) != 3 or any(
            budget.get("status") != "counterfactual" for budget in budgets
        ):
            raise ValueError("three available counterfactual budgets required")
    return terminal


def main():
    base = Path("docs/evidence/emi01-v2")
    paired = json.loads((base / "cpu-budget.json").read_text())
    validate_report(paired)
    output = {
        "schema": "emi01-v2-delivery-supplement-v1",
        "source": "Audited terminal/job records and paired cpu-budget.json; empirical nearest-rank P95, 27 measured jobs per mode.",
        "cpu_budget_sha256": hashlib.sha256(
            (base / "cpu-budget.json").read_bytes()
        ).hexdigest(),
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "runs": [],
    }
    for number, run in enumerate(paired["runs"], 1):
        directory = base / f"run-{number}"
        terminal_bytes = (directory / "terminal.json").read_bytes()
        terminal = validate_binding(run, terminal_bytes)
        entry = {
            "run": number,
            "terminal_sha256": hashlib.sha256(
                (directory / "terminal.json").read_bytes()
            ).hexdigest(),
            "modes": {},
        }
        for workers, mode in run["modes"].items():
            identities = [
                identity
                for sample in terminal["study_summaries"]
                if sample["id"].startswith(f"w{workers}-sample")
                for identity in sample["job_ids"]
            ]
            records = []
            for identity in identities:
                data = (directory / "jobs" / identity / "result.json").read_bytes()
                if (
                    hashlib.sha256(data).hexdigest()
                    != terminal["result_hashes"][identity]
                ):
                    raise ValueError("job identity differs from audited terminal")
                records.append(json.loads(data))
            extra = {}
            for field in ["queue_s", "completion_latency_s"]:
                values = sorted(record[field] for record in records)
                extra[field] = {
                    "min": min(values),
                    "median": statistics.median(values),
                    "p95": values[math.ceil(0.95 * len(values)) - 1],
                    "max": max(values),
                }
            extra["cleanup_hashing_unclassified_s"] = mode["job_service_s"] - sum(
                mode["phase_service_totals_s"].values()
            )
            extra["median_counterfactual_wall_s"] = {
                speed: statistics.median(
                    budget["simulation_only_acceleration"][speed]["predicted_wall_s"]
                    for budget in mode["counterfactual_budget"]
                )
                for speed in ["2", "4", "10", "infinite"]
            }
            entry["modes"][workers] = extra
        output["runs"].append(entry)
    (base / "delivery-supplement.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
