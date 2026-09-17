"""Exercise the actual CPU runner's publication and evidence boundary."""

import argparse
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import unittest


CPU = None
DC_DECK = """* EMI02 public process fixture
Vdrive drive 0 3
Eout out 0 VALUE={2*v(drive)}
Rload out 0 2k
.op
.save i(Eout) v(out) v(drive)
.end
"""


class RunnerProcessTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="emi02-runner-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.deck = self.directory / "input.cir"
        self.deck.write_text(DC_DECK)

    def invoke(self, output="output.raw", metadata="statistics.json"):
        return subprocess.run(
            [str(CPU), str(self.deck), str(output), str(metadata)],
            cwd=self.directory,
            capture_output=True,
            check=False,
            timeout=30,
            env={"PATH": "", "LC_ALL": "C"},
        )

    def assert_unpublished(self):
        for name in (
            "output.raw",
            "output.raw.partial",
            "statistics.json",
            "statistics.json.partial",
        ):
            self.assertFalse((self.directory / name).exists(), name)

    def read_completed(self):
        raw = (self.directory / "output.raw").read_bytes()
        header, payload = raw.split(b"Binary:\n", 1)
        points = int(re.search(rb"No\. Points: (\d+)\n", header)[1])
        variables = int(re.search(rb"No\. Variables: (\d+)\n", header)[1])
        self.assertEqual(len(payload), points * variables * 8)
        table = list(struct.iter_unpack("<" + "d" * variables, payload))
        self.assertTrue(all(math.isfinite(value) for row in table for value in row))
        statistics = json.loads((self.directory / "statistics.json").read_text())
        self.assertEqual(statistics["schema"], "emi02-cpu-v1")
        self.assertEqual(statistics["status"], "complete")
        self.assertEqual(statistics["points"], points)
        self.assertEqual(statistics["variables"], variables)
        self.assertEqual(statistics["raw_bytes"], len(raw))
        self.assertEqual(
            statistics["attempts"] - statistics["rejected_steps"], points - 1
        )
        self.assertGreaterEqual(statistics["elapsed_seconds"], 0)
        self.assertFalse((self.directory / "output.raw.partial").exists())
        self.assertFalse((self.directory / "statistics.json.partial").exists())
        return header, table, statistics

    def test_success_publishes_complete_dc_header_selected_order_and_statistics(self):
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        header, table, statistics = self.read_completed()
        self.assertIn(b"Title: emi02 public process fixture\n", header)
        self.assertIn(b"Plotname: Operating Point\n", header)
        self.assertIn(b"1\ti(eout)\tcurrent\n2\tv(out)\tvoltage\n", header)
        self.assertEqual(statistics["unknowns"], 4)
        self.assertEqual(statistics["attempts"], 0)
        self.assertEqual(table[0][0], 0)
        self.assertAlmostEqual(table[0][1], -0.003, delta=1e-10)
        self.assertEqual(table[0][2:], (6, 3))

    def test_transient_publishes_exact_terminal_time_and_step_accounting(self):
        self.deck.write_text(
            "* EMI02 public transient process fixture\n"
            "Vinput input 0 DC 0 PWL(0 0 1u 1 3u 1)\n"
            "Edrive drive 0 VALUE={v(input)}\nRcharge drive out 1k\nCstate out 0 1n\n"
            ".tran 10n 3u\n.save v(out) i(Vinput)\n.end\n"
        )
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        header, table, statistics = self.read_completed()
        self.assertIn(b"Plotname: Transient Analysis\n", header)
        self.assertEqual(table[0][0], 0)
        self.assertEqual(table[-1][0], 3e-6)
        self.assertGreater(statistics["attempts"], 0)
        for previous, row in zip(table, table[1:]):
            self.assertGreater(row[0], previous[0])
            self.assertLessEqual(row[0] - previous[0], 10e-9 * (1 + 1e-12))
        expected = 1 - (1 - math.exp(-1)) * math.exp(-2)
        self.assertAlmostEqual(table[-1][1], expected, delta=1e-4)

    def test_identical_normalized_and_cross_partial_paths_are_rejected(self):
        (self.directory / "alias").symlink_to(self.directory, target_is_directory=True)
        for output, metadata in (
            ("output.raw", "output.raw"),
            ("output.raw", "./output.raw"),
            ("output.raw", "alias/output.raw"),
            ("output.raw", "output.raw.partial"),
            ("statistics.json.partial", "statistics.json"),
            ("input.cir", "statistics.json"),
        ):
            with self.subTest(output=output, metadata=metadata):
                result = self.invoke(output, metadata)
                self.assertNotEqual(result.returncode, 0)
                self.assertTrue(result.stderr.startswith(b"io:"), result.stderr)
                self.assertEqual(self.deck.read_text(), DC_DECK)
                self.assert_unpublished()

    def test_every_preexisting_final_or_partial_file_is_preserved(self):
        for name in (
            "output.raw",
            "statistics.json",
            "output.raw.partial",
            "statistics.json.partial",
        ):
            with self.subTest(name=name):
                path = self.directory / name
                original = b"existing user bytes\x00\xff"
                path.write_bytes(original)
                result = self.invoke()
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(path.read_bytes(), original)
                path.unlink()
                self.assert_unpublished()

    def test_every_dangling_final_or_partial_symlink_is_preserved(self):
        target = self.directory / "must_not_be_created"
        for name in (
            "output.raw",
            "statistics.json",
            "output.raw.partial",
            "statistics.json.partial",
        ):
            with self.subTest(name=name):
                path = self.directory / name
                path.symlink_to(target)
                result = self.invoke()
                self.assertNotEqual(result.returncode, 0)
                self.assertTrue(path.is_symlink())
                self.assertEqual(path.readlink(), target)
                self.assertFalse(target.exists())
                path.unlink()
                self.assert_unpublished()

    def test_existing_partial_survives_filesystem_validation_exception(self):
        partial = self.directory / "output.raw.partial"
        partial.write_bytes(b"preserve me")
        result = self.invoke(metadata="x" * 300)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(partial.read_bytes(), b"preserve me")
        partial.unlink()
        self.assert_unpublished()

    def test_failed_solve_removes_its_partial_output(self):
        self.deck.write_text(
            "* EMI02 contradictory public voltage sources\n"
            "Vleft out 0 1\nVright out 0 2\n.op\n.end\n"
        )
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(result.stderr.startswith(b"singular:"), result.stderr)
        self.assert_unpublished()

    def test_metadata_creation_failure_removes_completed_raw_temporary(self):
        result = self.invoke(metadata="absent/statistics.json")
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(result.stderr.startswith(b"io:"), result.stderr)
        self.assert_unpublished()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", required=True)
    args, remaining = parser.parse_known_args()
    CPU = Path(args.cpu).resolve()
    unittest.main(argv=[__file__, *remaining])
