"""Original selected-model standalone current/charge and refinement comparison."""

import argparse
from pathlib import Path
import subprocess
import tempfile
import unittest

from third_party.emi_python.runtime import load_numpy

np = load_numpy()

from reference.emi01 import adapter, circuits, signals, study  # noqa: E402
from reference.emi02 import importer, qualification  # noqa: E402

ARCHIVE = None
NGSPICE = None
CPU = None
NAMES = ["time", "v(d)", "v(g)", "i(vdrain)", "i(vgate)"]
STOP = 80e-9


def source(fixture, temperature, step):
    ramp = "PWL(0 {initial} 10n {initial} 20n {peak} 40n {peak} 50n {initial} 80n {initial})"
    gate = ramp.format(initial=-3, peak=20) if fixture == "gate_charge" else "-3"
    drain = "400" if fixture == "gate_charge" else ramp.format(initial=0, peak=-3.5)
    return f"EMI02 {fixture} TJ {temperature}\n.include model.lib\nVdrain d 0 {drain}\nVgate g 0 {gate}\nXdevice d g 0 MSC040SMA120B PARAMS: TJ_C={temperature}\n{circuits.OPTIONS}\n.tran {step / 2:.17g} {STOP:.17g} 0 {step:.17g}\n.save {' '.join(NAMES[1:])}\n.end\n"


def execute(binary, directory, arguments):
    process = subprocess.run(
        [str(binary), *arguments],
        cwd=directory,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={"PATH": "", "LC_ALL": "C"},
        timeout=30,
        check=False,
    )
    if process.returncode:
        # Do not publish model-bearing diagnostics from parser failures.
        category = qualification.cpu_failure_status(
            process.stdout, "numerical_failure"
        )[1]
        raise AssertionError("selected-model dynamic execution failed: " + category)
    return process.stdout


def observe(raw, dt):
    grid, _ = signals.resample(raw[:, 0], raw[:, 1], 0, STOP, dt)
    grid = study.np.append(grid, STOP)
    values = study.np.column_stack(
        [
            study.np.interp(grid, raw[:, 0], raw[:, column])
            for column in range(1, len(NAMES))
        ]
    )
    charges = {}
    for lower, upper in [(0, 30e-9), (30e-9, STOP)]:
        times = study.np.concatenate(
            ([lower], grid[(grid > lower) & (grid < upper)], [upper])
        )
        for column, name in [(3, "drain"), (4, "gate")]:
            charges[(lower, name)] = float(
                study.np.trapezoid(
                    study.np.interp(times, raw[:, 0], raw[:, column]), times
                )
            )
    return values, charges


class SelectedDeviceDynamicTest(unittest.TestCase):
    def assert_agrees(self, actual, expected):
        for column, name in enumerate(NAMES[1:]):
            absolute = (
                0.002 if name == "i(vgate)" else 0.02 if name == "i(vdrain)" else 1e-6
            )
            relative = 0.02 if name.startswith("i(") else 1e-6
            compared = signals.waveform_comparison(
                actual[0][:, column],
                expected[0][:, column],
                absolute=absolute,
                relative=relative,
            )
            self.assertTrue(compared["passed"], (name, compared))
        for name, charge in expected[1].items():
            self.assertAlmostEqual(
                actual[1][name], charge, delta=0.2e-9 + 0.02 * abs(charge)
            )

    def test_current_charge_integration_and_observation_refinement(self):
        self.assertIsNotNone(ARCHIVE)
        self.assertIsNotNone(NGSPICE)
        self.assertIsNotNone(CPU)
        for fixture in ("gate_charge", "reverse_ramp"):
            for temperature in (27, 125):
                with self.subTest(fixture=fixture, temperature=temperature):
                    trajectories = {"cpu": [], "reference": []}
                    with tempfile.TemporaryDirectory(prefix="emi02-device-") as temp:
                        directory = Path(temp)
                        adapter.adapt_archive(ARCHIVE, directory / "model.lib")
                        for level, step in enumerate((0.5e-9, 0.25e-9, 0.125e-9)):
                            original = source(fixture, temperature, step)
                            (directory / "circuit.cir").write_text(original)
                            (directory / "driver.cir").write_text(
                                circuits.driver("emi01-v2")
                            )
                            log = execute(
                                NGSPICE, directory, ["-n", "-b", "driver.cir"]
                            )
                            self.assertIn(b"ngspice-46 done", log)
                            trajectories["reference"].append(
                                signals.parse_raw(
                                    (directory / "waveform.raw").read_bytes(),
                                    NAMES,
                                    STOP,
                                    step,
                                )
                            )
                            flat, provenance = importer.import_archive(
                                ARCHIVE, original
                            )
                            (directory / "cpu.cir").write_text(flat)
                            raw_path, metadata_path = (
                                f"cpu{level}.raw",
                                f"cpu{level}.json",
                            )
                            execute(
                                CPU, directory, ["cpu.cir", raw_path, metadata_path]
                            )
                            raw_bytes = (directory / raw_path).read_bytes()
                            raw = signals.parse_raw(raw_bytes, NAMES, STOP, step)
                            qualification.validate_statistics(
                                qualification.read_json_bounded(
                                    directory / metadata_path
                                ),
                                len(raw),
                                len(NAMES),
                                provenance["mna_unknowns"],
                                len(raw_bytes),
                            )
                            trajectories["cpu"].append(raw)
                            self.assert_agrees(
                                observe(raw, 0.05e-9),
                                observe(trajectories["reference"][-1], 0.05e-9),
                            )
                    for backend in trajectories:
                        runs = trajectories[backend]
                        for coarse, fine in zip(runs, runs[1:]):
                            self.assert_agrees(
                                observe(coarse, 0.05e-9), observe(fine, 0.05e-9)
                            )
                        coarse, fine = (
                            observe(runs[-1], 0.1e-9),
                            observe(runs[-1], 0.05e-9),
                        )
                        grids = []
                        for dt in (0.1e-9, 0.05e-9):
                            grid, _ = signals.resample(
                                runs[-1][:, 0], runs[-1][:, 1], 0, STOP, dt
                            )
                            grids.append(study.np.append(grid, STOP))
                        reconstructed = study.np.column_stack(
                            [
                                study.np.interp(
                                    grids[1], grids[0], coarse[0][:, column]
                                )
                                for column in range(len(NAMES) - 1)
                            ]
                        )
                        self.assert_agrees((reconstructed, coarse[1]), fine)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--ngspice", required=True)
    parser.add_argument("--cpu", required=True)
    args, remaining = parser.parse_known_args()
    ARCHIVE, NGSPICE, CPU = (
        Path(value).resolve() for value in (args.archive, args.ngspice, args.cpu)
    )
    unittest.main(argv=[__file__, *remaining])
