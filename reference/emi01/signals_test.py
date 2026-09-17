"""Independent analytic and hostile-output tests for EMI-01 measurements."""

import math
import struct
import unittest

from third_party.emi_python.runtime import load_numpy

np = load_numpy()

from reference.emi01 import signals  # noqa: E402


def raw_bytes(rows=None, names=("time", "v(out)", "i(vsense)"), *, points=None):
    """Build test data independently of the production parser."""
    if rows is None:
        rows = ((0, 1, 0), (0.5e-6, 2, 1), (1e-6, 3, 2))
    if points is None:
        points = len(rows)
    header = (
        "Title: independent unit fixture\nDate: frozen test\n"
        "Command: ngspice-46\nPlotname: Transient Analysis\nFlags: real\n"
        f"No. Variables: {len(names)}\nNo. Points: {points}\nVariables:\n"
    )
    for index, name in enumerate(names):
        unit = (
            "time"
            if index == 0
            else ("current" if name.startswith("i(") else "voltage")
        )
        header += f"\t{index}\t{name}\t{unit}\n"
    return (header + "Binary:\n").encode("ascii") + b"".join(
        struct.pack("<" + "d" * len(row), *row) for row in rows
    )


class RawTests(unittest.TestCase):
    def parse(self, data):
        return signals.parse_raw(data, ["time", "v(out)", "i(vsense)"], 1e-6, 0.5e-6)

    def test_valid_binary_and_owned_values(self):
        result = self.parse(raw_bytes())
        np.testing.assert_array_equal(result, [[0, 1, 0], [0.5e-6, 2, 1], [1e-6, 3, 2]])
        self.assertTrue(result.flags.owndata)
        self.assertEqual(result.dtype, np.dtype("float64"))

    def test_missing_output(self):
        with self.assertRaisesRegex(ValueError, "^missing_output:"):
            self.parse(b"")

    def test_malformed_and_truncated_outputs(self):
        raw = raw_bytes()
        corruptions = {
            "truncated binary": raw[:-1],
            "extra byte": raw + b"x",
            "multiple plots": raw + raw,
            "wrong count": raw_bytes(points=4),
            "negative count": raw.replace(b"No. Points: 3", b"No. Points: -3"),
            "gigantic count token": raw.replace(
                b"No. Points: 3", b"No. Points: " + b"9" * 5000
            ),
            "missing count": raw.replace(b"No. Points: 3\n", b""),
            "duplicate count": raw.replace(
                b"No. Points: 3\n", b"No. Points: 3\nNo. Points: 3\n"
            ),
            "complex": raw.replace(b"Flags: real", b"Flags: complex"),
            "nontransient": raw.replace(b"Transient Analysis", b"AC Analysis"),
            "ascii": raw.replace(b"Binary:", b"Values:"),
            "schema omitted": raw.replace(b"\t2\ti(vsense)\tcurrent\n", b""),
            "schema duplicated": raw_bytes(names=("time", "v(out)", "v(out)")),
            "schema reordered": raw_bytes(names=("time", "i(vsense)", "v(out)")),
            "wrong unit": raw.replace(b"i(vsense)\tcurrent", b"i(vsense)\tvoltage"),
            "wrong index": raw.replace(b"\t2\ti(vsense)", b"\t3\ti(vsense)"),
            "non-ASCII header": raw.replace(b"frozen test", b"\xffrozen test"),
            "huge header": b"Title: " + b"x" * signals.MAX_HEADER_BYTES + b"\n" + raw,
        }
        for label, malformed in corruptions.items():
            with (
                self.subTest(label=label),
                self.assertRaisesRegex(ValueError, "^malformed_output:"),
            ):
                self.parse(malformed)

    def test_point_budget_before_payload_allocation(self):
        with self.assertRaisesRegex(ValueError, "^resource_limit:"):
            self.parse(raw_bytes(points=signals.MAX_POINTS + 1))

    def test_raw_size_budget_before_parse(self):
        original = signals.MAX_RAW_BYTES
        try:
            signals.MAX_RAW_BYTES = 8
            with self.assertRaisesRegex(ValueError, "^resource_limit:"):
                self.parse(raw_bytes())
        finally:
            signals.MAX_RAW_BYTES = original

    def test_nonfinite(self):
        for invalid in (float("nan"), float("inf"), -float("inf")):
            for column in range(3):
                rows = [[0, 1, 0], [0.5e-6, 2, 1], [1e-6, 3, 2]]
                rows[1][column] = invalid
                with (
                    self.subTest(invalid=invalid, column=column),
                    self.assertRaisesRegex(ValueError, "^non_finite:"),
                ):
                    self.parse(raw_bytes(rows))

    def test_complete_and_monotonic_time(self):
        for times in (
            (1e-12, 0.5e-6, 1e-6),
            (0, 0.5e-6, 0.9e-6),
            (0, 0.5e-6, 1.1e-6),
            (0, 0, 1e-6),
            (0, -1e-9, 1e-6),
            (0, 0.6e-6, 1e-6),
        ):
            with (
                self.subTest(times=times),
                self.assertRaisesRegex(ValueError, "^malformed_output:"),
            ):
                self.parse(raw_bytes([[time, 1, 1] for time in times]))


class AnalyticSignalTests(unittest.TestCase):
    def test_cm_dm_independent_port_identities(self):
        # Outward port currents: equal = pure CM; opposing = pure DM.
        a = np.array([4.0, 4.0, 7.0, -3.0])
        b = np.array([4.0, -4.0, -1.0, 9.0])
        common, differential = signals.cm_dm(a, b)
        np.testing.assert_array_equal(common, [4, 0, 3, 3])
        np.testing.assert_array_equal(differential, [0, 4, 4, -6])
        np.testing.assert_array_equal(common + differential, a)
        np.testing.assert_array_equal(common - differential, b)
        np.testing.assert_array_equal(
            a * a + b * b, 2 * (common * common + differential * differential)
        )
        voltage_common, voltage_diff = signals.voltage_cm_dm(a, b)
        np.testing.assert_array_equal(voltage_common, [4, 0, 3, 3])
        np.testing.assert_array_equal(voltage_diff, [0, 8, 8, -12])

    def test_no_extrapolation_and_half_open_grid(self):
        grid, values = signals.resample([0, 0.4, 1], [0, 0.8, 2], 0, 1, 0.25)
        np.testing.assert_array_equal(grid, [0, 0.25, 0.5, 0.75])
        np.testing.assert_allclose(values, [0, 0.5, 1, 1.5], atol=1e-15)
        for start, stop in ((-0.25, 1), (0, 1.25)):
            with self.assertRaisesRegex(ValueError, "^missing_output:"):
                signals.resample([0, 1], [0, 1], start, stop, 0.25)

    def test_dc_has_no_ac_bins(self):
        time = np.arange(80001) * 1.25e-9 + 100e-6
        frequency, amplitude = signals.spectrum(time, np.full(len(time), 3.5))
        self.assertEqual(len(frequency), 986)
        np.testing.assert_allclose(frequency[[0, -1]], [150e3, 10e6], rtol=1e-15)
        np.testing.assert_array_equal(amplitude, np.zeros(len(amplitude)))
        np.testing.assert_array_equal(
            signals.dbua([0, 1e-6, 1e-5, 1]), [-180, 0, 20, 120]
        )

    def test_bin_center_sine_phase_and_hann_side_bins(self):
        time = np.arange(80001) * 1.25e-9 + 100e-6
        for phase in (0.0, math.pi / 7, math.pi / 2, math.pi):
            signal = 2.75 * np.sin(2 * math.pi * 1e6 * time + phase) + 8.0
            frequency, amplitude = signals.spectrum(time, signal)
            peak = int(np.argmin(abs(frequency - 1e6)))
            self.assertAlmostEqual(amplitude[peak], 2.75 / math.sqrt(2), places=12)
            self.assertAlmostEqual(
                amplitude[peak - 1], 2.75 / (2 * math.sqrt(2)), places=12
            )
            self.assertAlmostEqual(
                amplitude[peak + 1], 2.75 / (2 * math.sqrt(2)), places=12
            )
            offpeak = np.delete(amplitude, [peak - 1, peak, peak + 1])
            self.assertLess(float(np.max(offpeak)), 1e-12)

    def test_mixed_tones_and_independent_parseval(self):
        # Small integer-period fixture independently uses a direct DFT identity.
        count, dt = 1024, 1e-8
        time = np.arange(count + 1) * dt
        signal = 1.3 * np.sin(
            2 * math.pi * 17 * np.arange(count + 1) / count + 0.3
        ) + 0.6 * np.cos(2 * math.pi * 83 * np.arange(count + 1) / count + 0.8)
        frequency, amplitude = signals.spectrum(
            time,
            signal,
            0,
            count * dt,
            dt,
            frequency_min=1 / (count * dt),
            frequency_max=(count / 2 - 1) / (count * dt),
        )
        self.assertAlmostEqual(amplitude[16], 1.3 / math.sqrt(2), places=13)
        self.assertAlmostEqual(amplitude[82], 0.6 / math.sqrt(2), places=13)
        window = np.array(
            [0.5 - 0.5 * math.cos(2 * math.pi * n / count) for n in range(count)]
        )
        windowed = (signal[:-1] - sum(signal[:-1]) / count) * window
        # This fixture has no DC/Nyquist energy. One-sided coherent amplitude
        # squares sum to 4 times mean(windowed**2), by Parseval and sum(w)=N/2.
        expected = 4 * sum(float(value) ** 2 for value in windowed) / count
        self.assertAlmostEqual(float(sum(amplitude**2)), expected, places=12)
        self.assertAlmostEqual(
            float(sum(amplitude**2)), 1.5 * (1.3**2 + 0.6**2) / 2, places=12
        )
        self.assertEqual(len(frequency), count // 2 - 1)

    def test_nonuniform_linear_interpolation(self):
        grid, values = signals.resample(
            [0, 0.13, 0.61, 1], [1, 1.26, 2.22, 3], 0, 1, 0.125
        )
        np.testing.assert_allclose(values, 1 + 2 * grid, atol=1e-15)

    def test_typed_invalid_signal_inputs(self):
        operations = [
            lambda: signals.cm_dm([1], [1, 2]),
            lambda: signals.cm_dm(["bad"], [1]),
            lambda: signals.resample([0, 0, 1], [1, 2, 3], 0, 1, 0.25),
            lambda: signals.resample([0, 1], [0, 1], 0, 1, 0.3),
            lambda: signals.dbua([-1]),
            lambda: signals.spectrum(
                [0, 1], [0, 1], 0, 1, 0.25, frequency_min=1, frequency_max=2
            ),
        ]
        for operation in operations:
            with self.assertRaisesRegex(ValueError, "^unsupported_input:"):
                operation()
        for operation in (
            lambda: signals.cm_dm([float("nan")], [1]),
            lambda: signals.cm_dm([1e308], [1e308]),
            lambda: signals.waveform_comparison([1e308], [1e308]),
        ):
            with self.assertRaisesRegex(ValueError, "^non_finite:"):
                operation()

    def test_spectral_gate_checks_every_bin_and_both_sides_of_floor(self):
        self.assertTrue(
            signals.spectrum_comparison([1e-3, 1e-7], [1.02e-3, 1e-6])["passed"]
        )
        above_floor = signals.spectrum_comparison([1e-3, 2e-5], [1e-3, 1e-5])
        self.assertFalse(above_floor["passed"])
        self.assertEqual(above_floor["failed_bins"], 1)
        self.assertAlmostEqual(
            above_floor["max_high_difference_db"], 20 * math.log10(2)
        )
        self.assertFalse(signals.spectrum_comparison([0], [2e-6])["passed"])
        self.assertFalse(signals.spectrum_comparison([2e-6], [0])["passed"])
        self.assertTrue(signals.spectrum_comparison([0], [1e-6])["passed"])
        self.assertTrue(signals.spectrum_comparison([9.5e-6], [1e-5])["passed"])
        self.assertFalse(signals.spectrum_comparison([1e-5], [1.2e-5])["passed"])

    def test_waveform_gate_and_settling_tolerance(self):
        passed = signals.waveform_comparison([1.03, 1.03], [1, 1])
        self.assertTrue(passed["passed"])
        self.assertAlmostEqual(passed["limit"], 0.04)
        self.assertFalse(signals.waveform_comparison([1.05, 1.05], [1, 1])["passed"])
        self.assertFalse(
            signals.waveform_comparison([1.03], [1], absolute=0.01, relative=0.01)[
                "passed"
            ]
        )


if __name__ == "__main__":
    unittest.main()
