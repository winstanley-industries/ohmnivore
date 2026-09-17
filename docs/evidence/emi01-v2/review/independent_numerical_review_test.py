"""Focused independent-review checks; run from repository root outside timing."""

import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

if sys.flags.optimize:
    raise RuntimeError("Run these assertion-based checks without Python optimization")

path = Path("docs/evidence/emi01-v2/review/independent_numerical_review.py")
tree = ast.parse(path.read_text())
# Load only independent definitions; do not execute the retained-run entry point.
keep = []
for node in tree.body:
    if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
        keep.append(node)
    elif isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id in {"np", "STRESS_LIMITS"}
        for t in node.targets
    ):
        keep.append(node)
ns = {}
exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), ns)
ns["report"] = {"metric_errors": []}
manifest = json.loads(Path("reference/emi01/manifest-v2.json").read_text())
ns["contract"](manifest)
for field, value in [
    ("schema", "emi01-v1"),
    ("research_mask_dbua", 91),
    ("required_margin_db", 5),
    ("ensemble_sample_steps_s", [5e-9, 2.5e-9, 2e-9]),
]:
    bad = copy.deepcopy(manifest)
    bad[field] = value
    try:
        ns["contract"](bad)
    except AssertionError:
        pass
    else:
        raise AssertionError(field + " incorrectly accepted")
for field in ["id", "cx_f"]:
    bad = copy.deepcopy(manifest)
    bad["candidates"][1][field] = "renamed" if field == "id" else 90e-9
    try:
        ns["contract"](bad)
    except AssertionError:
        pass
    else:
        raise AssertionError(field + " incorrectly accepted")
for field, value in [
    ("failing_candidate", "boundary"),
    ("failing_requires_all_corners", False),
    ("failing_requires_all_corners", 1),
]:
    bad = copy.deepcopy(manifest)
    bad["reference_case_gates"][field] = value
    try:
        ns["contract"](bad)
    except AssertionError:
        pass
    else:
        raise AssertionError(field + " incorrectly accepted")
for field in ["failing_candidate", "failing_requires_all_corners"]:
    bad = copy.deepcopy(manifest)
    del bad["reference_case_gates"][field]
    try:
        ns["contract"](bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("missing " + field + " incorrectly accepted")
rows = [
    {
        "corner": corner,
        "settled": True,
        "status": "predicted_infeasible",
        "physical_violations": ["capacitor_peak_v"],
        "minimum_margin_db": -30,
        "worst_channel": "a",
        "worst_frequency_hz": 150000,
        "mass_kg": 0.2285867686,
    }
    for corner in ["nominal", "fast_low_lc", "hot_high_c"]
]
assert ns["role_check"]("light", 2, rows)["pass"]
for status in [
    "predicted_feasible",
    "numerical_failure",
    "timeout",
    "missing_output",
    "unsupported_input",
    "unsettled",
]:
    bad = copy.deepcopy(rows)
    bad[1]["status"] = status
    assert not ns["role_check"]("light", 2, bad)["pass"], status
bad = copy.deepcopy(rows)
bad[1]["settled"] = False
assert not ns["role_check"]("light", 2, bad)["pass"]
for bad in [rows[:-1], [rows[0], rows[0], rows[2]], list(reversed(rows))]:
    try:
        ns["role_check"]("light", 2, bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect control corner set accepted")
for index, expected in enumerate([0.2285867686, 3.8062284436, 3.8084193316]):
    ns["near"](
        "frozen-material-volume-mass",
        ns["design_mass"](manifest["candidates"][index]),
        expected,
    )
assert not ns["report"]["metric_errors"]
try:
    ns["near"]("nonfinite", float("nan"), 1)
except AssertionError:
    pass
else:
    raise AssertionError("nonfinite accepted")
data = []
metrics = []
for i, dt in enumerate(manifest["dpt_sample_steps_s"]):
    jid = f"q{i}-dpt"
    d = ns["raw"]("docs/evidence/emi01/run-1", jid)
    data.append(d)
    measured = ns["dpt_metrics"](d, dt)
    metrics.append(measured)
    assert measured["pass"]
    stored = json.loads(
        (Path("docs/evidence/emi01/run-1/jobs") / jid / "result.json").read_text()
    )["metrics"]
    for key in [
        "peak_v",
        "on_edge_s",
        "off_edge_s",
        "on_half_s",
        "off_half_s",
        "on_energy_j",
        "off_energy_j",
        "sample_step_s",
        "min_v",
        "peak_gate_v",
        "min_gate_v",
        "load_at_second_on_a",
    ]:
        ns["near"](jid + "-" + key, measured[key], stored[key])
    for key in ["frequency_hz", "log_decrement"]:
        ns["near"](jid + "-" + key, measured["ringing"][key], stored["ringing"][key])
assert not ns["report"]["metric_errors"], ns["report"]["metric_errors"]
checks = [
    ns["dpt_refinement"]("integration", metrics[i - 1], metrics[i]) for i in [1, 2]
]
checks += [
    ns["dpt_refinement"]("sampling", ns["dpt_metrics"](data[-1], dt), metrics[-1])
    for dt in manifest["dpt_sample_steps_s"][:2]
]
assert len(checks) == 4 and all(x["pass"] for x in checks)
for script in [
    path,
    Path("docs/evidence/emi01-v2/design/explore.py"),
    Path("docs/evidence/emi01-v2/design/fine-refinement-review.py"),
]:
    p = subprocess.run(
        [sys.executable, "-O", "-P", str(script), "--help"],
        text=True,
        capture_output=True,
    )
    assert p.returncode != 0 and "without Python optimization" in p.stderr, (
        script,
        p.returncode,
        p.stderr,
    )
diagnostic = Path("docs/evidence/emi01-v2/design/fine-refinement-review.py")
diagnostic_hash = hashlib.sha256(diagnostic.read_bytes()).hexdigest()
for candidate in ["anchor", "boundary"]:
    document = json.loads(
        Path(f"docs/evidence/emi01-v2/design/{candidate}-fine-review.json").read_text()
    )
    assert document["implementation_sha256"] == diagnostic_hash
    assert all(len(row["raw_sha256"]) == 64 for row in document["physical"])
    assert all(len(row["record_sha256"]) == 64 for row in document["physical"])
with tempfile.TemporaryDirectory(prefix="emi01-review-test-") as directory:
    output = Path(directory) / "should-not-exist.json"
    p = subprocess.run(
        [
            sys.executable,
            "-P",
            str(path),
            "docs/evidence/emi01/run-1",
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
    )
    assert p.returncode != 0 and "AssertionError" in p.stderr
    assert not output.exists()
print(
    "PASS: strict v2 contract; 11 hostile manifest mutations; failing control rejects six invalid statuses, unsettled output and three malformed corner sets; three mass calculations; nonfinite guard; DPT raw metrics and four refinements; three optimization guards; two design-report implementation bindings; v1 schema rejection"
)
