# EMI-01 v2 independent numerical review

Both retained invocations pass the independent numerical and fixture-role review. Their
recomputed numerical results are identical; only the terminal-record hash differs between
the two reports. The reference candidate passes every declared corner. The boundary candidate
passes all physical screens but misses the required research-mask reserve by 0.483381 dB in
the fast corner. The light control is numerically valid and predicted infeasible at every
corner. These classifications remain unchanged at all three refinement levels and in both
invocations.

The simulations bind source commit `8e45b4232741b82ec476fb6904b80f774eaf31d0` and manifest
SHA-256 `ec68ddb48866d8127fbdd0ecf411dfc326b6c7bc7929c10429e40edd9cd2c81f`.
This review ran after both timed invocations completed. It imports no production parsing,
measurement, qualification, ranking or selection helpers. It uses the declared Bazel Python
3.12.13 / NumPy 2.4.3 runtime, reconstructs and hashes raw waveform payloads, and independently
computes Fourier amplitudes, physical stress, loss, mass, classifications and refinement gates.

## Coverage and retained reports

Each invocation records 102 validated jobs and zero simulation failures. The reviewer checks
the exact ordered 102-job identity set, then numerically recomputes its 30 qualification jobs:
27 coupled ensemble solves and three double-pulse solves. The canonical invocation audit
separately covers all terminal records, including the 72 warmup/sample jobs, their provenance,
resource accounting and required outputs. This numerical review does not replace that audit
or establish execution attestation.

| Independent checks per invocation | Count | Outcome |
|---|---:|---|
| Qualification checks: 36 ensemble and four DPT comparisons | 40 | All pass |
| Spectral comparisons: four observables, integration and sampling | 144 | Zero failing bins |
| Conductor and all-four-switch Vds waveform comparisons | 216 | All pass |
| Final-two-period settling checks | 54 | All pass |
| Physical limits: nine fixed limits across 27 ensemble solves | 243 | Recomputed; candidate failures preserved |
| Mass calculations and per-corner classifications | 27 each | Zero discrepancies |
| Candidate-role checks at all three refinement levels | 9 | All pass |

Each spectral comparison covers all 986 bins from 150 kHz through 10 MHz. All four stored
spectra and margins are also recomputed for each ensemble solve. There are zero discrepancies
with recorded metrics, physical limits, statuses or stored spectra in either invocation.
Limits are constants in the independent reviewer, rather than values trusted from results.

- [Run 1 independent report](review/retained-run-1-numerical.json) and
  [completion log](review/retained-run-1-numerical.log).
- [Run 2 independent report](review/retained-run-2-numerical.json) and
  [completion log](review/retained-run-2-numerical.log).

## Integration, output sampling and settling

The light control uses maximum integration steps 2.5 / 1.25 / 0.625 ns. Boundary and reference
use 0.625 / 0.3125 / 0.15625 ns. Integration comparisons use a common output grid. Separate
sampling comparisons reconstruct the finest integration waveform on 5 and 2.5 ns grids and
compare with 1.25 ns. Neither phase alignment nor frequency-bin exclusion is used.

The unchanged spectral gate is 1 dB where either amplitude exceeds 10 uA, and 1 uA absolute
difference below that floor. Amplitudes are periodic-Hann, mean-removed, single-bin RMS
equivalents over [100,200) us, using average-current CM/DM normalization. The detector and
research mask remain those frozen in ADR-002.

| Maximum error across all candidates/corners, identical in both runs | Above-floor dB | Subfloor uA |
|---|---:|---:|
| First adjacent integration pair | 0.713180 | 0.748781 |
| Second adjacent integration pair | 0.158232 | 0.265922 |
| Same-waveform output sampling | 0.009113 | 0.013857 |

The largest absolute amplitude change anywhere is 42.356677 uA, for boundary/hot conductor A
at 2.60 MHz, from 2.973224 to 2.930867 mA. That bin is above the 10 uA floor and passes the
relative gate; the subfloor absolute limit does not apply to it.

Maximum conductor RMS waveform differences are 0.000329539 A for integration and
0.000241885 A for sampling. Their respective allowed limits are 0.229429 and 0.252855 A.
Maximum switch Vds RMS differences are 0.074635 V for integration and 1.350406 V for
sampling, each against a 10.8 V limit. All four devices are checked.

The largest fraction of a settling allowance occurs for reference/hot conductor A at the
coarsest level: 0.075289 A between the last two periods, against 0.095172 A allowed.
All 54 settling checks pass in each invocation. This is the declared finite-window screen,
not a general stability proof.

## Mass, limiting bins and classification stability

Mass is recomputed from the frozen core/copper volumes, 20% assembly allowance, capacitor
energy-density assumption and two 5 g damping resistors. It is a hypothetical physical design
model, not a supplier part mass or a manufacturability claim.

| Candidate | Filter mass kg | Finest worst corner/channel/bin | Research margin dB | Overall classification |
|---|---:|---|---:|---|
| Light control | 0.2285867686 | Hot / B / 150 kHz | -37.743435 | Infeasible at every corner |
| Boundary, X = 80 nF | 3.8062284436 | Fast / A / 150 kHz | 5.516619 | Infeasible at fast corner |
| Reference, X = 1 uF | 3.8084193316 | Fast / A / 2.85 MHz | 10.102246 | Feasible at every corner |

The required margin is 6 dB below the 90 dBuA research mask, giving an effective acceptance
threshold of 84 dBuA. Boundary's finest limiting bin is 84.483381 dBuA; reference's is
79.897754 dBuA. The reference is the only feasible member of this frozen three-candidate set.
It is not claimed to be a minimum-mass optimum. Boundary and reference share the same
inductors, geometry and Y capacitance; their capacitor-mass difference is 2.190888 g.

| Candidate | Level 0 worst margin dB | Level 1 worst margin dB | Level 2 worst margin dB |
|---|---:|---:|---:|
| Light | -37.743438 | -37.743436 | -37.743435 |
| Boundary | 5.516605 | 5.516606 | 5.516619 |
| Reference | 10.071196 | 10.095318 | 10.102246 |

Boundary's nominal and hot finest margins are 8.121852 and 10.681709 dB, both limited by
conductor A at 150 kHz. Reference's nominal and hot finest margins are 15.669638 dB
(B, 2.95 MHz) and 15.822837 dB (B, 2.65 MHz). All corner classifications remain unchanged
under refinement. The light role requires three valid infeasible results; timeout, missing
output, unsupported input, numerical failure or unsettled output cannot satisfy that role.

## Physical screens

The table gives maxima across all three corners and all three integration levels. Both
invocations reproduce these values. Device, capacitor and flux peaks include startup;
winding RMS and losses use the declared observation interval.

| Screen | Limit | Boundary maximum | Reference maximum |
|---|---:|---:|---:|
| Device absolute Vds, V | 960 | 646.691854 | 646.691879 |
| Device absolute current, A | 50 | 35.796742 | 36.499455 |
| Capacitor absolute voltage, V | 504 | 469.582238 | 463.893894 |
| Winding RMS current, A | 12 | 10.339038 | 10.483683 |
| Copper + ESR + damping loss, W | 25 | 22.335780 | 17.555877 |
| DM peak flux, T | 0.20 | 0.066184 | 0.070003 |
| CM peak flux, T | 0.20 | 0.072548 | 0.073001 |
| A damping resistor loss, W | 12 | 8.459073 | 3.478212 |
| B damping resistor loss, W | 12 | 3.477869 | 3.456021 |

Every boundary/reference corner passes these screens at every refinement. The light control
retains real violations, including capacitor voltage up to 1203.752533 V and CM flux up to
0.466832 T; its infeasibility is not a missing or failed simulation counted as useful coverage.

## Double-pulse qualification

All three DPT solves independently pass the switching, gate/drain observability and resolved
ringing gates. The two integration and two same-finest-waveform sampling comparisons also
pass the frozen edge/crossing, overshoot, signed-energy, ringing-frequency and decay gates.

| Quantity | Range across the three integration levels |
|---|---:|
| Peak low-side Vds | 440.079697–440.108411 V |
| Turn-on 10–90% Vds edge | 17.121128–17.184723 ns |
| Turn-off 10–90% Vds edge | 12.694125–12.731577 ns |
| Signed turn-on energy | 124.236614–124.239880 uJ |
| Signed turn-off energy | 43.581641–43.584043 uJ |
| Ringing frequency | 66.115702–66.666667 MHz |
| Ringing log decrement | 0.608457–0.609948 |
| Observed gate drop during turn-on Vds transition | 1.367858–1.470284 V |

Ringing uses actual first-three same-polarity, strictly decaying peaks above the declared
0.2 V residual floor. The diagnostic independently verifies peak values/times and gate
crossings against the retained records; it does not fabricate unresolved damping.

## Diagnostic correction, identity and reproduction

The first post-measurement review attempts reached JSON report generation but failed because
a NumPy boolean in an independently computed DPT refinement result was not serializable.
The correction explicitly converts the derived ringing frequency to Python `float` and
refinement pass values to Python `bool`. A focused regression now serializes the complete
DPT metrics/comparison structure. No calculation, comparison, gate, frozen source or retained
input/output changed. The complete independent reviews were rerun successfully after the fix.

- Initial diagnostic failure logs are preserved for
  [run 1](review/retained-run-1-initial-diagnostic-error.log) and
  [run 2](review/retained-run-2-initial-diagnostic-error.log).
- [Pre-evidence helper validation](review/helper-checks.json) retains its historical hashes.
- [Post-measurement validation and identity](review/helper-checks-final.json) binds the revised
  reviewer, [focused tests](review/independent_numerical_review_test.py),
  [test log](review/helper-checks-final.log), manifest and final numerical reports.

Final [reviewer](review/independent_numerical_review.py) SHA-256:
`bcf5da9b1dd6963612c0e7e0b6b9e2cc9b7f71858b8542f804770138278911c3`.
Pinned Ruff checks and the focused hostile-input, mass, DPT, optimization-guard, serialization
and report-binding tests pass. Both full reviewer entry-point executions exit successfully.

Outside a timed invocation, build the default runtime and run from the repository root:

```sh
bazel build //reference/emi01:study
emi01_review_runfiles="$PWD/bazel-bin/reference/emi01/study.runfiles"
RUNFILES_DIR="$emi01_review_runfiles" \
PYTHONPATH="$emi01_review_runfiles/_main:$emi01_review_runfiles/rules_python+:$emi01_review_runfiles/rules_python++pip+emi01_pypi_312_numpy_cp312_cp312_manylinux_2_27_x86_64_e7dd01a4/site-packages" \
"$emi01_review_runfiles/rules_python++python+python_3_12_x86_64-unknown-linux-gnu/bin/python3" -P \
  docs/evidence/emi01-v2/review/independent_numerical_review.py \
  docs/evidence/emi01-v2/run-1 --output /tmp/emi01-v2-run-1-review.json
```

Repeat with `run-2` and a distinct output path. Use the same runtime prefix with
`docs/evidence/emi01-v2/review/independent_numerical_review_test.py` for the focused checks.
No vendor model archive or new simulation is required for these reviews. Python optimization
is rejected because assertions enforce diagnostic checks.

The conclusions are numerical predictions for the finite coupled surrogate and declared
corners. Fixed-temperature devices, linear magnetics without core loss, lumped harnesses,
the unmeasured load-loss surrogate and hypothetical geometry/mass remain limitations. These
results establish neither hardware correlation, motor impedance accuracy, thermal feasibility,
aviation compliance, a statistical tolerance guarantee nor performance outside the measured
band. They authorize no production device, parser, transient or GPU semantics.
