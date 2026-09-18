"""Independent import invariants; no vendor text is embedded in this test."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from reference.emi01 import circuits
from reference.emi02 import importer

ARCHIVE = None


class ParameterTest(unittest.TestCase):
    def test_forward_references_and_caller_override(self):
        values = importer.resolve_parameters(
            {"first": "{second*3}", "second": "{MDE/2}", "MDE": "1"},
            {"MDE": 8},
            {"MDE": "MDE"},
        )
        self.assertEqual(values, {"mde": 8, "second": 4, "first": 12})
        self.assertEqual(
            importer.resolve_parameters({"a": "01.000", "b": "exp(0)+3m"})["b"], 1.003
        )

    def test_strict_parameter_dialect(self):
        for text in [
            "1/0",
            "exp(10000)",
            "mystery(1)",
            "1**2",
            "1 garbage",
            "{2",
            "nan",
            "inf",
        ]:
            with self.subTest(text=text), self.assertRaises(importer.ImportError):
                importer.resolve_parameters({"a": text})
        for definitions in [{"a": "b", "b": "a"}, {"a": "missing"}]:
            with self.assertRaises(importer.ImportError):
                importer.resolve_parameters(definitions)
        with self.assertRaises(importer.ImportError):
            importer.resolve_parameters({"a": "1"}, overrides={"a": "a"})
        with self.assertRaises(importer.ImportError):
            importer.resolve_parameters({"a": "1"}, overrides={"wrong": "3"})

    def test_assignment_and_budget_guards(self):
        self.assertEqual(
            importer.assignments("A={ 3 + 2 } B=1e-3"), {"a": "{ 3 + 2 }", "b": "1e-3"}
        )
        for value in ["a=1 A=2", "a", "a=", "a={1", "a=1 garbage"]:
            with self.subTest(value=value), self.assertRaises(importer.ImportError):
                assignment = importer.assignments(value)
                importer.resolve_parameters(assignment)
        for expression in ["(" * 65 + "1" + ")" * 65, "+".join(["1"] * 300)]:
            with self.assertRaises(importer.ImportError) as raised:
                importer.resolve_parameters({"a": expression})
            self.assertEqual(raised.exception.status, "resource_limit")

    def test_numeric_and_scope_binding(self):
        for token, value in [("1G", 1e9), ("1m", 1e-3), ("4MEG", 4e6), ("-3", -3)]:
            self.assertEqual(importer.number(token), value)
        expression = importer.bind_expression(
            "{exp(A)+v(42,32)+i(Vsense)*1e-9}", lambda n: "n" + n, ["x1"], {"a": 2}
        )
        self.assertEqual(expression, "{exp((2))+v(n42,n32)+i(v_x1__sense)*1e-9}")


class ArchiveTest(unittest.TestCase):
    def setUp(self):
        if ARCHIVE is None:
            self.fail("--archive is required; model tests never skip")
        self.manifest = json.loads(
            (Path(circuits.__file__).parent / "manifest-v2.json").read_text()
        )

    def test_exact_archive_all_thirty_decks(self):
        for level in range(3):
            dpt = circuits.dpt(self.manifest["dpt_max_steps_s"][level], "emi01-v2")
            flat, record = importer.import_archive(ARCHIVE, dpt)
            self.assertEqual(
                (record["packages"], record["instances"], record["mna_unknowns"]),
                (2, 4, 80),
            )
            self.assertIn("Vgate drive 0 DC -3 PWL(0 -3", flat)
            self.assertEqual(
                record["flat_deck_sha256"], hashlib.sha256(flat.encode()).hexdigest()
            )
            for candidate in self.manifest["candidates"]:
                for corner in self.manifest["corners"]:
                    steps = self.manifest["candidate_max_steps_s"].get(
                        candidate["id"], self.manifest["ensemble_max_steps_s"]
                    )
                    flat, record = importer.import_archive(
                        ARCHIVE,
                        circuits.ensemble(candidate, corner, steps[level], "emi01-v2"),
                    )
                    self.assertEqual(
                        (
                            record["packages"],
                            record["instances"],
                            record["mna_unknowns"],
                        ),
                        (4, 8, 185),
                    )
                    self.assertIn("Vgah driveah a DC -3 PULSE(", flat)
                    self.assertIn("Vgal driveal 0 DC 20 PULSE(", flat)
                    self.assertIn("Kcm Lcma Lcmb 0.995", flat)
                    self.assertIn("Kh Lha Lhb 0.2", flat)
                    self.assertNotIn(".param", flat.lower())
                    self.assertNotIn(".include", flat.lower())
                    self.assertNotIn(".options", flat.lower())
                    self.assertEqual(flat.count(" I={"), 8)
                    self.assertEqual(flat.count(" VALUE={"), 52)
                    self.assertLess(
                        record["expression_tokens"], importer.MAX_TOTAL_NODES
                    )
                    self.assertEqual(record["ambient_temperature_c"], 27)
                    self.assertAlmostEqual(
                        record["vt_volts"], 0.02586491689545, delta=1e-15
                    )
                    self.assertIn("(0.025864916895449997)", flat)

    def test_model_and_deck_fail_closed(self):
        original = circuits.dpt(2e-9, "emi01-v2")
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "bad.zip"
            archive.write_bytes(b"not the frozen archive")
            with self.assertRaises(importer.ImportError) as raised:
                importer.import_archive(archive, "also malformed")
            self.assertEqual(raised.exception.status, "provenance_mismatch")
        for old, new, status in [
            ("MSC040SMA120B", "UNKNOWN", "unsupported_input"),
            ("MSC040SMA120B", "MSC040SMA120B PARAMS: TJ_C=28", "unsupported_input"),
            ("MSC040SMA120B", "MSC040SMA120B PARAMS: GM=2", "unsupported_input"),
            (".include model.lib", ".include /tmp/model.lib", "unsupported_input"),
            ("Vbus bus 0", "Vbus emi02_private 0", "unsupported_input"),
            (
                ".end",
                "Xthird d g 0 MSC040SMA120B\nXfourth d g 0 MSC040SMA120B\nXfifth d g 0 MSC040SMA120B\n.end",
                "resource_limit",
            ),
            (".end", "Vbus x 0 1\n.end", "compile_failure"),
            (".end", ".control\n.end", "unsupported_input"),
            (".end", ".end\nRextra 1 0 1", "parse_failure"),
            ("Xlo d g 0", "Xlo emi02_private g 0", "unsupported_input"),
            (
                "PWL(0 -3 1u -3 1.01u 20 10u 20 10.01u -3 12u -3 12.01u 20 14u 20 14.01u -3)",
                "PWL()",
                "parse_failure",
            ),
            (".end", ".include model.lib\n.end", "parse_failure"),
        ]:
            with (
                self.subTest(old=old, new=new),
                self.assertRaises(importer.ImportError) as raised,
            ):
                importer.import_archive(ARCHIVE, original.replace(old, new))
            self.assertEqual(raised.exception.status, status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive")
    args, remaining = parser.parse_known_args()
    ARCHIVE = args.archive
    unittest.main(argv=[__file__, *remaining])
