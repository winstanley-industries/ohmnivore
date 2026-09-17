"""Reject crossed or unavailable inputs before deriving delivery-only statistics."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "supplement", Path(__file__).with_name("delivery_supplement.py")
)
supplement = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supplement)


def fixture():
    terminal = {
        "schema": "emi01-v2",
        "qualification_pass": True,
        "reference_case_pass": True,
        "qualification_only": False,
        "counts": {"expected": 102, "terminal": 102, "validated": 102, "failures": 0},
    }
    data = json.dumps(terminal).encode()
    run = {
        "reference_version": "emi01-v2",
        "accepted": True,
        "terminal_sha256": hashlib.sha256(data).hexdigest(),
        "modes": {
            str(workers): {
                "counterfactual_budget": [
                    {"status": "counterfactual"} for _ in range(3)
                ]
            }
            for workers in (1, 4)
        },
    }
    return terminal, data, run


class BindingTest(unittest.TestCase):
    def test_valid_binding(self):
        terminal, data, run = fixture()
        self.assertEqual(supplement.validate_binding(run, data), terminal)

    def test_crossed_or_stale_identity(self):
        _, data, run = fixture()
        run["terminal_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "identity disagree"):
            supplement.validate_binding(run, data)
        _, data, run = fixture()
        with self.assertRaisesRegex(ValueError, "identity disagree"):
            supplement.validate_binding(run, data + b"\n")

    def test_unaccepted_or_unavailable(self):
        for change in ("accepted", "version", "budget", "budget_count"):
            _, data, run = fixture()
            if change == "accepted":
                run["accepted"] = False
            elif change == "version":
                run["reference_version"] = "emi01-v1"
            elif change == "budget":
                run["modes"]["1"]["counterfactual_budget"][0]["status"] = "unavailable"
            else:
                run["modes"]["4"]["counterfactual_budget"].pop()
            with self.subTest(change=change), self.assertRaises(ValueError):
                supplement.validate_binding(run, data)

    def test_failed_or_incomplete_terminal_even_with_matching_hash(self):
        for change in (
            "schema",
            "qualification_pass",
            "reference_case_pass",
            "qualification_only",
            "counts",
        ):
            terminal, _, run = fixture()
            if change == "schema":
                terminal[change] = "emi01-v1"
            elif change == "counts":
                terminal[change]["failures"] = 1
            else:
                terminal[change] = not terminal[change]
            data = json.dumps(terminal).encode()
            run["terminal_sha256"] = hashlib.sha256(data).hexdigest()
            with self.subTest(change=change), self.assertRaises(ValueError):
                supplement.validate_binding(run, data)

    def test_exact_distinct_pair(self):
        _, _, run = fixture()
        other = copy.deepcopy(run)
        other["terminal_sha256"] = "1" * 64
        good = {
            "schema": "emi01-cpu-budget-v2",
            "reference_version": "emi01-v2",
            "runs": [run, other],
        }
        supplement.validate_report(good)
        for change in ("schema", "reference_version", "duplicate", "count"):
            paired = copy.deepcopy(good)
            if change in ("schema", "reference_version"):
                paired[change] = "emi01-v1"
            elif change == "duplicate":
                paired["runs"][1] = paired["runs"][0]
            else:
                paired["runs"].pop()
            with self.subTest(change=change), self.assertRaises(ValueError):
                supplement.validate_report(paired)


if __name__ == "__main__":
    unittest.main()
