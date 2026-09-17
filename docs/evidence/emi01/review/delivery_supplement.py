"""Derive queue/completion quantiles from already audited EMI-01 job records.

Run from the repository root after the canonical paired report succeeds.
This is reporting arithmetic, not a substitute for the full evidence audit.
"""

import hashlib
import json
import math
from pathlib import Path
import statistics

base = Path("docs/evidence/emi01")
paired = json.loads((base / "cpu-budget.json").read_text())
output = {
    "schema": "emi01-delivery-supplement-v1",
    "source": "Audited terminal/job records and paired cpu-budget.json; empirical nearest-rank P95, 27 measured jobs per mode.",
    "cpu_budget_sha256": hashlib.sha256(
        (base / "cpu-budget.json").read_bytes()
    ).hexdigest(),
    "runs": [],
}
for number, run in enumerate(paired["runs"], 1):
    directory = base / f"run-{number}"
    terminal = json.loads((directory / "terminal.json").read_text())
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
        records = [
            json.loads((directory / "jobs" / identity / "result.json").read_text())
            for identity in identities
        ]
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
