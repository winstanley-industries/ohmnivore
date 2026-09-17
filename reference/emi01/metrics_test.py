"""Independent discriminating tests of EMI-01 stress and qualification metrics."""

import json
import math
from pathlib import Path
import unittest

from third_party.emi_python.runtime import load_numpy

np = load_numpy()

from reference.emi01 import circuits, metrics  # noqa: E402


def constant_study():
    time = np.linspace(0, 200e-6, 201)
    values = {
        "time": time,
        "v(p)": 400,
        "v(a)": 100,
        "v(b)": 300,
        "v(fa)": 60,
        "v(fb)": 40,
        "v(xcap)": 59.8,
        "v(yca)": 57.8,
        "v(ycb)": 37.8,
        "v(la)": 50,
        "v(lr)": 10,
        "v(lb)": 0,
        "i(lload)": 1.9,
        "i(va)": 2,
        "i(vb)": -2,
        "i(lda)": 2,
        "i(ldb)": -2,
        "i(lcma)": 2,
        "i(lcmb)": -2,
    }
    return np.column_stack(
        [
            np.broadcast_to(values.get(name, 0), time.shape)
            for name in circuits.STUDY_NAMES
        ]
    )


def analytic_dpt():
    """Linear 40-ns ramps permit exact crossings and signed energy integrals."""
    time = np.arange(64001) * 0.25e-9
    drain = np.interp(
        time,
        [0, 12.02e-6, 12.06e-6, 14.02e-6, 14.06e-6, 16e-6],
        [400, 400, 0, 0, 400, 400],
    )
    # An overshoot away from the energy windows tests peak evaluation separately.
    drain += 20 * np.maximum(0, 1 - abs(time - 15e-6) / 10e-9)
    gate = np.full(time.shape, -3.0)
    gate[(time >= 12.06e-6) & (time < 14.02e-6)] = 20
    on = (time >= 12.02e-6) & (time <= 12.06e-6)
    gate[on] = 5 + 0.02 * drain[on]
    gate[(time >= 14.02e-6) & (time <= 14.06e-6)] = 2
    return np.column_stack(
        [time, drain, gate, np.full(time.shape, 20), np.full(time.shape, 2)]
    )


def ringing_dpt():
    raw = analytic_dpt()
    elapsed = raw[:, 0] - 14.07e-6
    active = elapsed >= 0
    raw[active, 1] += (
        12
        * np.exp(-elapsed[active] / 80e-9)
        * np.sin(2 * math.pi * 40e6 * elapsed[active])
    )
    return raw


class StressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        manifest = json.loads((Path(__file__).parent / "manifest.json").read_text())
        cls.candidate = manifest["candidates"][0]
        cls.corner = manifest["corners"][0]

    def test_independent_constant_current_loss_flux_and_rating(self):
        result, spectrum = metrics.evaluate(
            constant_study(), self.candidate, self.corner, 1.25e-9
        )
        stress = result["stress"]
        self.assertEqual(result["status"], "predicted_feasible")
        self.assertEqual(result["rms_cm_a"], 0)
        self.assertEqual(result["rms_dm_a"], 2)
        self.assertEqual(result["load_series_resistor_w"], 80)
        self.assertEqual(result["load_winding_loss_surrogate_w"], 1)
        self.assertAlmostEqual(result["rms_load_inductor_a"], 1.9)
        self.assertEqual(result["rms_vcm_v"], 50)
        self.assertEqual(result["rms_vdm_v"], 20)
        self.assertAlmostEqual(stress["capacitor_peak_v"], 57.8)
        self.assertEqual(stress["winding_rms_a"], 2)
        self.assertAlmostEqual(stress["dm_peak_t"], 0.01)
        self.assertAlmostEqual(stress["cm_peak_t"], 1 / 960)
        # Independent SI arithmetic: two windings of each kind, X ESR current
        # 1 A and Y branch currents .1 A with .2-ohm ESR +21.8-ohm damper.
        expected_copper = (
            2 * 4 * (1.724e-8 * 20 * 0.045 / 4e-6 + 1.724e-8 * 12 * 0.040 / 4e-6)
        )
        expected_total = (
            expected_copper + 0.2 * (1 + 0.01 + 0.01) + 21.8 * (0.01 + 0.01)
        )
        self.assertAlmostEqual(stress["loss_w"], expected_total, places=11)
        self.assertAlmostEqual(stress["damping_a_w"], 0.218, places=11)
        self.assertAlmostEqual(stress["damping_b_w"], 0.218, places=11)
        self.assertEqual(spectrum.shape, (986, 5))
        np.testing.assert_array_equal(spectrum[:, 1:], 0)

    def test_individual_damper_rating_is_not_hidden_by_total(self):
        raw = constant_study()
        raw[:, circuits.STUDY_NAMES.index("v(yca)")] = 60 - 0.75 * 22
        result, _ = metrics.evaluate(raw, self.candidate, self.corner, 1.25e-9)
        self.assertLess(result["stress"]["loss_w"], 25)
        self.assertGreater(result["stress"]["damping_a_w"], 12)
        self.assertIn("damping_a_w", result["violations"])
        self.assertEqual(result["status"], "predicted_infeasible")

    def test_stress_peak_includes_discarded_startup(self):
        raw = constant_study()
        raw[0, circuits.STUDY_NAMES.index("v(p)")] = 1200
        result, _ = metrics.evaluate(raw, self.candidate, self.corner, 1.25e-9)
        self.assertEqual(result["stress"]["device_peak_v"], 1100)
        self.assertIn("device_peak_v", result["violations"])
        self.assertEqual(result["status"], "predicted_infeasible")

    def test_unsettled_is_failure_even_with_a_complete_spectrum(self):
        raw = constant_study()
        raw[:, circuits.STUDY_NAMES.index("i(va)")] = raw[:, 0] * 100000
        result, spectrum = metrics.evaluate(raw, self.candidate, self.corner, 1.25e-9)
        self.assertEqual(result["status"], "unsettled")
        self.assertFalse(result["settling"]["a"]["pass"])
        self.assertEqual(spectrum.shape[0], 986)

    def test_derived_overflow_is_typed_failure(self):
        raw = constant_study()
        raw[:, circuits.STUDY_NAMES.index("i(lda)")] = 1e308
        with self.assertRaisesRegex(ValueError, "^non_finite:"):
            metrics.evaluate(raw, self.candidate, self.corner, 1.25e-9)


class RefinementTests(unittest.TestCase):
    def test_high_side_vds_cannot_escape_integration_gate(self):
        coarse = constant_study()
        fine = coarse.copy()
        fine[:, circuits.STUDY_NAMES.index("v(p)")] += 40
        result = metrics.compare(coarse, fine, circuits.STUDY_NAMES, 1.25e-9)
        self.assertFalse(result["pass"])
        self.assertFalse(result["waveforms"]["vds_ah"]["pass"])
        self.assertFalse(result["waveforms"]["vds_bh"]["pass"])
        self.assertTrue(result["waveforms"]["vds_al"]["pass"])
        self.assertTrue(result["waveforms"]["vds_bl"]["pass"])
        self.assertTrue(all(check["pass"] for check in result["spectra"].values()))

    def test_high_side_output_grid_error_is_independent_of_spectral_pass(self):
        time = np.arange(160001) * 1.25e-9
        raw = np.zeros((len(time), len(circuits.STUDY_NAMES)))
        raw[:, 0] = time
        raw[:, circuits.STUDY_NAMES.index("v(p)")] = 400 + 40 * np.sin(
            2 * math.pi * 100e6 * time
        )
        result = metrics.output_sampling_compare(raw, circuits.STUDY_NAMES, 5e-9)
        self.assertFalse(result["pass"])
        self.assertFalse(result["waveforms"]["vds_ah"]["pass"])
        self.assertTrue(all(check["pass"] for check in result["spectra"].values()))


class DoublePulseTests(unittest.TestCase):
    def test_analytic_crossings_and_signed_energy(self):
        result = metrics.dpt(analytic_dpt(), 0.25e-9)
        # Correct edges and energy alone do not qualify the frozen ringing fixture.
        self.assertFalse(result["pass"])
        self.assertAlmostEqual(result["on_edge_s"], 32e-9, delta=1e-15)
        self.assertAlmostEqual(result["off_edge_s"], 32e-9, delta=1e-15)
        self.assertAlmostEqual(result["on_half_s"], 12.04e-6, delta=1e-15)
        self.assertAlmostEqual(result["off_half_s"], 14.04e-6, delta=1e-15)
        self.assertAlmostEqual(result["on_energy_j"], 192e-6, delta=1e-12)
        self.assertAlmostEqual(result["off_energy_j"], 128e-6, delta=1e-12)
        self.assertAlmostEqual(
            result["gate_dynamics"]["on_edge_gate_drop_v"], 6.4, delta=1e-10
        )
        self.assertEqual(result["ringing"]["status"], "unresolved")

    def test_memoryless_gate_and_negative_energy_do_not_qualify(self):
        self.assertTrue(metrics.dpt(ringing_dpt())["pass"])
        raw = ringing_dpt()
        raw[:, circuits.DPT_NAMES.index("v(g)")] = 20
        self.assertFalse(metrics.dpt(raw)["pass"])
        raw = ringing_dpt()
        raw[:, circuits.DPT_NAMES.index("i(vdrain)")] *= -1
        result = metrics.dpt(raw)
        self.assertLess(result["on_energy_j"], 0)
        self.assertFalse(result["pass"])

    def test_declared_output_grid_changes_sampled_peak(self):
        raw = analytic_dpt()
        index = int(round(15.0005e-6 / 0.25e-9))
        raw[index, circuits.DPT_NAMES.index("v(d)")] = 480
        coarse = metrics.dpt(raw, 1e-9)
        fine = metrics.dpt(raw, 0.5e-9)
        self.assertLess(coarse["peak_v"], 421)
        self.assertGreater(fine["peak_v"], 479)

    def test_both_polarities_of_analytic_damped_ringing(self):
        time = 14e-6 + np.arange(1601) * 0.25e-9
        elapsed = time - 14e-6
        wave = 20 * np.exp(-elapsed / 200e-9) * np.sin(2 * math.pi * 20e6 * elapsed)
        outcomes = []
        for sign in [1, -1]:
            result = metrics.ringing(time, 400 + sign * wave)
            self.assertEqual(result["status"], "measured")
            self.assertAlmostEqual(result["frequency_hz"], 20e6, delta=1e5)
            self.assertAlmostEqual(result["log_decrement"], 0.25, delta=0.005)
            self.assertTrue(
                all(
                    x > y
                    for x, y in zip(
                        result["peak_amplitudes_v"], result["peak_amplitudes_v"][1:]
                    )
                )
            )
            outcomes.append(result["polarity"])
        self.assertEqual(set(outcomes), {"positive", "negative"})

    def test_growing_ring_cannot_claim_positive_damping(self):
        time = 14e-6 + np.arange(1601) * 0.25e-9
        elapsed = time - 14e-6
        wave = 20 * np.exp(elapsed / 200e-9) * np.sin(2 * math.pi * 20e6 * elapsed)
        self.assertEqual(metrics.ringing(time, 400 + wave)["status"], "unresolved")


if __name__ == "__main__":
    unittest.main()
