"""Check local scope binding against ngspice's independent subcircuit expansion."""

import argparse
from pathlib import Path
import re
import subprocess
import struct
import math
import json
import tempfile
import unittest

from reference.emi01 import adapter
from reference.emi02 import importer

ARCHIVE = None
NGSPICE = None
CPU = None


def dc(binary, directory, deck):
    (directory / "circuit.cir").write_text(deck)
    (directory / "driver.cir").write_text(
        "EMI02 import binding test\n.control\nset ngbehavior=ps\nset numdgt=17\nsource circuit.cir\nop\nprint i(vdrain) i(vgate) v(d) v(g)\nquit\n.endc\n.end\n"
    )
    result = subprocess.run(
        [str(binary), "-n", "-b", "driver.cir"],
        cwd=directory,
        capture_output=True,
        timeout=20,
        check=False,
        env={"PATH": "", "LC_ALL": "C"},
    )
    output = result.stdout.decode(errors="replace") + result.stderr.decode(
        errors="replace"
    )
    if result.returncode or re.search(r"\b(error|failed|aborted)\b", output, re.I):
        raise AssertionError("ngspice DC binding comparison failed; output stays local")
    measured = dict(
        (name.lower(), float(value))
        for name, value in re.findall(
            r"^([iv]\([^)]+\))\s*=\s*([+eE.0-9-]+)\s*$", output, re.M
        )
    )
    if set(measured) != {"i(vdrain)", "i(vgate)", "v(d)", "v(g)"}:
        raise AssertionError("ngspice DC binding output schema mismatch")
    return measured


def cpu_dc(binary, directory, deck):
    (directory / "cpu.cir").write_text(deck)
    for name in ("cpu.raw", "cpu.json"):
        (directory / name).unlink(missing_ok=True)
    result = subprocess.run(
        [str(binary), "cpu.cir", "cpu.raw", "cpu.json"],
        cwd=directory,
        capture_output=True,
        timeout=20,
        check=False,
        env={"PATH": "", "LC_ALL": "C"},
    )
    if result.returncode:
        raise AssertionError(
            "CPU DC binding comparison failed; model-containing diagnostics stay local"
        )
    raw = (directory / "cpu.raw").read_bytes()
    header, data = raw.split(b"Binary:\n", 1)
    if b"Plotname: Operating Point\n" not in header or len(data) != 40:
        raise AssertionError("CPU DC waveform schema mismatch")
    values = struct.unpack("<5d", data)
    metadata = json.loads((directory / "cpu.json").read_text())
    if (
        values[0] != 0
        or not all(math.isfinite(value) for value in values)
        or metadata.get("points") != 1
        or metadata.get("status") != "complete"
    ):
        raise AssertionError("CPU DC waveform/metadata incomplete")
    return dict(zip(["i(vdrain)", "i(vgate)", "v(d)", "v(g)"], values[1:], strict=True))


class ImportOracleTest(unittest.TestCase):
    def test_exact_import_bias_grid(self):
        self.assertIsNotNone(ARCHIVE, "archive required")
        self.assertIsNotNone(NGSPICE, "ngspice required")
        self.assertIsNotNone(CPU, "CPU runner required")
        with tempfile.TemporaryDirectory(prefix="emi02-binding-") as temp:
            directory = Path(temp)
            adapter.adapt_archive(ARCHIVE, directory / "model.lib")
            for temperature in (27, 125):
                for gate in (-3, 2, 20):
                    for drain in (-1, 0.1, 10, 400):
                        with self.subTest(
                            temperature=temperature, gate=gate, drain=drain
                        ):
                            source = f"EMI02 DC binding\n.include model.lib\nVdrain d 0 {drain}\nVgate g 0 {gate}\nXdevice d g 0 MSC040SMA120B PARAMS: TJ_C={temperature}\n.op\n.save i(vdrain) i(vgate) v(d) v(g)\n.end\n"
                            flattened, _ = importer.import_archive(ARCHIVE, source)
                            # ps compatibility applies its if() frontend to included libraries.
                            # A wrapper in that library preserves the already-bound graph
                            # without providing any parameters or further model hierarchy.
                            cards = [
                                line
                                for line in flattened.splitlines()[1:]
                                if not line.startswith((".", "Vdrain ", "Vgate "))
                            ]
                            (directory / "flat.lib").write_text(
                                ".subckt imported d g\n"
                                + "\n".join(cards)
                                + "\n.ends\n"
                            )
                            wrapped = f"EMI02 flattened binding\n.include flat.lib\nVdrain d 0 {drain}\nVgate g 0 {gate}\nXwrapper d g imported\n.op\n.end\n"
                            expected, actual = (
                                dc(NGSPICE, directory, source),
                                dc(NGSPICE, directory, wrapped),
                            )
                            for name in expected:
                                self.assertAlmostEqual(
                                    actual[name],
                                    expected[name],
                                    delta=1e-9 + 1e-8 * abs(expected[name]),
                                )
                            actual_cpu = cpu_dc(CPU, directory, flattened)
                            for name in expected:
                                self.assertAlmostEqual(
                                    actual_cpu[name],
                                    expected[name],
                                    delta=1e-7 + 1e-6 * abs(expected[name]),
                                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--ngspice", required=True)
    parser.add_argument("--cpu", required=True)
    args, remaining = parser.parse_known_args()
    ARCHIVE, NGSPICE, CPU = (
        Path(args.archive).resolve(),
        Path(args.ngspice).resolve(),
        Path(args.cpu).resolve(),
    )
    unittest.main(argv=[__file__, *remaining])
