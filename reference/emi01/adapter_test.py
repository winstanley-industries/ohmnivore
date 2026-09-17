"""Mechanical provenance tests and independent ngspice capacitor fixtures."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import subprocess
import tempfile
import unittest

from reference.emi01 import adapter


MODEL_ARCHIVE: Path | None = None
NGSPICE: Path | None = None


def synthetic_sources() -> str:
    # Independently authored expressions, not extracted vendor model text.
    return (
        ".param TEMP=27\n"
        "Cgate 42 ng 2n\nVgd ng 23 0\n"
        "Fgd 42 23 VALUE = {i(Vgd)*(3+2*v(42,23))}\n"
        "Cdrain 42 nd 3n\nVds nd 44 0\n"
        "Fds 42 44 VALUE = {i(Vds)*(5+v(42,44))}\n"
    )


class AdapterTest(unittest.TestCase):
    def test_frozen_archive_and_member(self):
        self.assertIsNotNone(
            MODEL_ARCHIVE, "--archive is required; model tests never skip"
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "local-model.lib"
            record = adapter.adapt_archive(MODEL_ARCHIVE, output)
            self.assertEqual(
                record["adapted_sha256"],
                hashlib.sha256(output.read_bytes()).hexdigest(),
            )
            self.assertEqual(record["adapted_bytes"], 29136)
            self.assertNotIn(b"\r", output.read_bytes())
            self.assertNotIn(b"Fgd ", output.read_bytes())
            self.assertNotIn(b"Fds ", output.read_bytes())

    def test_member_mismatch(self):
        with self.assertRaises(adapter.ModelError) as result:
            adapter.adapt_bytes(b"not the qualified model")
        self.assertEqual(result.exception.status, "provenance_mismatch")

    def test_missing_truncated_and_modified_archives(self):
        self.assertIsNotNone(MODEL_ARCHIVE)
        original = MODEL_ARCHIVE.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.zip"
            output = Path(directory) / "must-not-exist.lib"
            cases = (None, original[:-1], bytes([original[0] ^ 1]) + original[1:])
            for contents in cases:
                if contents is not None:
                    path.write_bytes(contents)
                with self.assertRaises(adapter.ModelError) as result:
                    adapter.adapt_archive(path, output)
                self.assertEqual(result.exception.status, "provenance_mismatch")
                self.assertFalse(output.exists())

    def test_translation_preserves_expressions_and_terminals(self):
        original = (
            synthetic_sources().replace("\n", "\r\n")
            + "* TEMPERATURE is an unrelated identifier\r\n"
        )
        actual = adapter._translate_text(original, expected_temp_tokens=1)
        expected = (
            synthetic_sources()
            .replace("TEMP", "TJ_C")
            .replace("Fgd", "Bgd")
            .replace("Fds", "Bds")
            .replace("VALUE =", "I=")
        )
        self.assertEqual(
            actual, expected + "* TEMPERATURE is an unrelated identifier\n"
        )

    def test_unexpected_constructs_fail_closed(self):
        text = synthetic_sources()
        cases = (
            text.replace(".param TEMP=27", ".param WRONG=27"),
            text.replace("Fgd 42 23", "Fgd 23 42"),
            text.replace("Fds", "Fother"),
            text + "Fgd 42 23 VALUE = {0}\n",
            text + "Fextra 1 2 Vmeasure 1\n",
        )
        for malformed in cases:
            with self.subTest(malformed=malformed):
                with self.assertRaises(adapter.ModelError) as result:
                    adapter._translate_text(malformed, expected_temp_tokens=1)
                self.assertEqual(result.exception.status, "unsupported_input")

    def test_analytic_capacitance_sign_derivative_and_charge(self):
        self.assertIsNotNone(
            NGSPICE, "--ngspice is required; analytic tests never skip"
        )
        for sign in (1, -1):
            with (
                self.subTest(ramp_sign=sign),
                tempfile.TemporaryDirectory() as directory,
            ):
                scratch = Path(directory)
                sources = adapter._translate_text(
                    synthetic_sources(), expected_temp_tokens=1
                )
                deck = (
                    "EMI01 independent affine capacitance fixture\n"
                    f"Vdrive 42 0 PWL(0 0 1u {sign})\n"
                    "Vgate 23 0 0\nVsource 44 0 0\n"
                    + sources
                    + ".options reltol=1e-9 abstol=1e-14 vntol=1e-12 method=gear maxord=2\n"
                    ".control\nset wr_singlescale\nset numdgt=17\n"
                    "tran 1n 1u 0 1n\nwrdata currents.txt v(42) i(vgate) i(vsource)\n"
                    "quit\n.endc\n.end\n"
                )
                (scratch / "fixture.cir").write_text(deck)
                result = subprocess.run(
                    [str(NGSPICE), "-n", "-b", "fixture.cir"],
                    cwd=scratch,
                    env={
                        "PATH": "",
                        "HOME": str(scratch),
                        "TMPDIR": str(scratch),
                        "LC_ALL": "C",
                    },
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
                self.assertEqual(
                    result.returncode, 0, result.stderr.decode(errors="replace")
                )
                rows = [
                    tuple(map(float, line.split()))
                    for line in (scratch / "currents.txt").read_text().splitlines()
                ]
                self.assertTrue(
                    all(
                        len(row) == 4 and all(math.isfinite(value) for value in row)
                        for row in rows
                    )
                )
                interior = [row for row in rows if 5e-9 <= row[0] <= 995e-9]
                self.assertGreater(len(interior), 900)
                slope = sign * 1e6
                for time, voltage, gate_current, drain_current in interior:
                    self.assertAlmostEqual(voltage, slope * time, delta=1e-9)
                    # C0 and multiplier both contribute: Ceff = C0*(1+f(V)).
                    self.assertAlmostEqual(
                        gate_current, 2e-9 * (4 + 2 * voltage) * slope, delta=1e-9
                    )
                    self.assertAlmostEqual(
                        drain_current, 3e-9 * (6 + voltage) * slope, delta=1e-9
                    )
                first, last = interior[0], interior[-1]
                measured_derivative = (last[2] - first[2]) / (last[1] - first[1])
                self.assertAlmostEqual(measured_derivative, 4e-9 * slope, delta=1e-8)
                va, vb = first[1], last[1]
                expected_charge = (
                    2e-9 * (4 * (vb - va) + vb * vb - va * va),
                    3e-9 * (6 * (vb - va) + (vb * vb - va * va) / 2),
                )
                for column, expected in zip((2, 3), expected_charge):
                    measured = sum(
                        (right[0] - left[0]) * (left[column] + right[column]) / 2
                        for left, right in zip(interior, interior[1:])
                    )
                    self.assertAlmostEqual(measured, expected, delta=1e-14)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--ngspice", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    MODEL_ARCHIVE = args.archive.resolve()
    NGSPICE = args.ngspice.resolve()
    unittest.main(argv=[__file__, *remaining])
