# EMI-01 independent numerical review

This review concerns the final, bounded external ngspice reference fixture in ADR-002. Its
calculations were performed independently of the harness's parsing, spectral, stress and
qualification helpers, using hash-verified raw payloads and the pinned Python 3.12.13 / NumPy
2.4.3 runtime. Analytic tests separately cover CM/DM normalization, Hann tone amplitudes,
Parseval, signed switching energy, crossing times, loss and flux calculations.

The exploratory results below were collected before retained timing and are not CPU performance
measurements. Raw exploratory files remain local; the small diagnostic reports preserve the
findings. Retained invocation reviews are recorded separately below. No audits, builds or
diagnostic simulations should overlap timed invocations.

## Retained invocation reviews

After each invocation finished, the packaged independent script executed successfully against
its retained raw artifacts. Each review audited all 30 qualification jobs: 144 spectral
comparisons, 216 waveform checks and 54 settling checks, with zero failures and zero
recomputed-metric discrepancies. All independently recomputed numerical results agree exactly
between the two invocations. In each, the maximum above-floor spectral error was 0.555321 dB
and the maximum subfloor amplitude error was 0.273035 uA, within the unchanged 1 dB / 1 uA
gates. Both DPT integration and output-grid refinements pass. Neither diagnostic overlapped
retained simulation timing or modified a run directory.

- [Run 1 numerical report](review/retained-run-1-numerical.json)
- [Run 1 diagnostic log](review/retained-run-1-numerical.log)
- [Run 1 runtime/source identity check](review/run-1-runtime-identities.json), a separate check
  against current canonical artifacts and the frozen source commit, not execution attestation.
- [Run 2 numerical report](review/retained-run-2-numerical.json)
- [Run 2 diagnostic log](review/retained-run-2-numerical.log)

The numerical review covers the qualification subset. The separate canonical invocation audit
also passed all 102 expected terminal jobs in run 1, with 102 validated and zero failures.
The paired canonical audit separately establishes complete-invocation accounting for both runs.

## Final settings and independent findings

The final study has nine coupled jobs: `light`, `medium`, `heavy`, each at `nominal`,
`fast_low_lc`, and `hot_high_c`. Light and medium use maximum integration steps
2.5 / 1.25 / 0.625 ns; heavy uses 0.625 / 0.3125 / 0.15625 ns. All candidates retain output
spacings 5 / 2.5 / 1.25 ns. Integration comparisons use a shared output grid; output-grid
comparisons separately resample the same finest integration waveform. Every job retains the
original solver tolerances, 100–200 us observation interval, periodic Hann window and
150 kHz–10 MHz current band.

The spectral gate remains **1 dB per bin when either amplitude exceeds 20 dBuA (10 uA), and
1 uA absolute difference below that floor**. Conductor waveform RMS differences must remain
within 0.02 A + 2% of reference RMS, and all four switch Vds waveform RMS differences within
2 V + 2% of bus voltage. No time alignment or phase optimization is used. Settling compares
the final two 20 us periods against 0.01 A + 1% of reference RMS.

| Independent check | Result |
|---|---|
| Light/medium integration spectra: six jobs, two adjacent pairs, four observables | All 48 comparisons pass |
| Light/medium output-grid spectra: six finest waveforms, two spacings, four observables | All 48 comparisons pass |
| Light/medium conductor and four-Vds integration/output waveform checks | All 144 checks pass |
| Heavy integration spectra: three jobs, two adjacent pairs, four observables | All 24 comparisons pass |
| Heavy output-grid spectra on each finest waveform | All 24 comparisons pass |
| Heavy conductor and four-Vds integration/output waveform checks | All 72 checks pass |
| Original 30-job exploratory corpus: numerical metrics, stress and stored spectra recomputation | No discrepancies |
| Settling in that 30-job corpus: all 27 ensemble solves, two conductors | All 54 checks pass |

The heavy integration results use the final finer policy; they replace the failed coarse heavy
comparisons in the earlier 30-job exploration. Maximum errors below refer to the original
20 dBuA floor, across A, B, CM and DM:

| Heavy corner | 0.625 → 0.3125 ns: max dB / max subfloor uA | 0.3125 → 0.15625 ns: max dB / max subfloor uA |
|---|---:|---:|
| Nominal | 0.213211 / 0.140590 | 0.060287 / 0.011956 |
| Fast, low L/C | 0.348977 / 0.192338 | 0.046253 / 0.011892 |
| Hot, high C | 0.186618 / 0.085805 | 0.067282 / 0.053994 |

Heavy finest raw files are approximately 277.6–277.7 MB, within the revised 512 MiB file/parser
limit and unchanged two-million-point limit. Observed child peak RSS in these probes was
approximately 820,484 KiB; the successful probes enforced the 1 GiB address-space limit. Probe wall times are not
retained performance evidence because other engineering work was running concurrently.

## Double-pulse qualification

DPT maximum integration steps are 2 / 1 / 0.5 ns, with uniform metric grids of 1 / 0.5 / 0.25 ns.
The independent raw recomputation agrees with the stored edge, signed energy and ringing
measurements. Across the three integration levels:

| Quantity | Observed range |
|---|---:|
| Peak low-side Vds | 440.0797–440.1084 V |
| Second turn-on 10–90% Vds edge | 17.1211–17.1847 ns |
| Second turn-off 10–90% Vds edge | 12.6941–12.7316 ns |
| Signed second turn-on energy | 124.2366–124.2399 uJ |
| Signed second turn-off energy | 43.5816–43.5840 uJ |
| Ringing frequency | 66.1157–66.6667 MHz |
| Ringing log decrement | 0.608457–0.609948 |

The coarse grid's first three positive residual peaks occur 45 / 60 / 75 ns after the second
turn-off trigger, at 40.0797 / 18.6422 / 11.8364 V above the nominal 400 V bus. These are actual
strictly decaying waveform peaks. Both adjacent DPT integration comparisons and both
same-finest-waveform output-grid comparisons satisfy the frozen gates: edge/50%-crossing
changes <= 2 ns + 10% of edge duration, peak change <= 2 V + 2%, energy change <= 2 uJ + 5%,
frequency change <= 10%, and log-decrement change <= 20%. The gate/drain observability checks
also pass. Ringing is now required to resolve for this fixture; unavailable damping is not
substituted with an estimate.

## Findings resolved before retained measurements

1. **Near-lossless load resonance.** The initial ideal 100 uH load branch produced substantial
   timestep-dependent spectral structure near 5 MHz. Raising the convergence floor was rejected:
   independent analysis found milliamperes of error in meaningful bins, not merely subfloor noise.
   The final fixture explicitly assumes a 100-ohm parallel winding-loss surrogate. Independent
   light-nominal analysis then passed both original spectral gates; conductor RMS changed by
   about -0.676% and load-terminal differential RMS voltage by -0.298%. This changes the physical
   workload and is documented as an unmeasured load assumption, not a solver correction or a
   validated motor model. Its power is reported, and filter mass excludes the load.
2. **Bypassed commutation inductance.** The initial DC-link placement bypassed the declared
   20 nH switching-loop inductance. Placing the capacitor upstream of that inductance, and
   retaining the held-off device's external gate resistor, produces the independently resolved
   ringing above. Earlier missing ringing was never counted as a measured frequency or damping.
3. **Heavy-candidate CM refinement.** With the corrected physical fixture, the coarser heavy
   policy still failed the unchanged gate. At 1.25 → 0.625 ns, fast-corner CM at 7.42 MHz changed
   from 13.9953 to 12.1694 uA (1.21430 dB); hot-corner CM at 6.58 MHz changed from 37.4882 to
   42.1272 uA (1.01334 dB). The candidate-specific finer integration policy resolves these
   failures without weakening tolerances or excluding bins.
4. **Metric coverage.** Review added both high-side Vds convergence checks, separate waveform
   output-sampling checks, explicit gate/drain observability, both ringing polarities with
   strictly decaying peaks, signed energy tests, correct Y-damper power/rating checks, and
   rejection of non-finite derived metrics before feasibility classification.

The review establishes numerical consistency within the declared surrogate. It does not establish
hardware correlation, motor impedance accuracy, thermal feasibility, aviation compliance or a
feasible candidate. Accuracy and predicted physical/research-mask feasibility remain separate.

## Review artifacts and reproduction

- [Earlier complete 30-job diagnostic](review/exploratory-before-heavy-refinement.json): includes
  the failed coarse heavy pairs; `old_failed_bins` refers to the governing 20 dBuA / 1 uA gate.
  Its alternate `failed_bins` diagnostic used a proposed broader floor that was rejected.
- [Final heavy integration checks](review/exploratory-heavy-integration.json): `failed_bins`
  uses the unchanged governing gate.
- [Final heavy waveform and sampling checks](review/exploratory-heavy-waveform-sampling.json):
  its sampling rows retain the same historical `old_failed_bins` / alternate `failed_bins`
  distinction. Its `sampling_spectral_failed_bins` summary uses the governing gate.
- [Independent numerical recomputation script](review/independent_numerical_review.py).

See the [field-name sidecar](review/README.md) before interpreting the historical JSON reports.

The script contains no vendor model text and imports no production measurement helpers. It
reconstructs and hashes raw payloads, recomputes spectra, settling, stress, signed DPT energies,
crossings and ringing, and compares recorded numerical outputs. Its gate summary covers the
30 qualification jobs in a 30-job exploration or a complete 102-job invocation; the canonical
harness audit separately checks all terminal identities, accounting and provenance. The packaged
script was formatted and checked with the repository's pinned Ruff binary, and its entry point
was exercised successfully on both retained invocations after their respective timing ended.

Build once outside a timing interval, then use the declared Bazel runtime rather than ambient
Python/NumPy. From the repository root, substitute the invocation directory and output report:

```sh
bazel build //reference/emi01:study
emi01_review_runfiles="$PWD/bazel-bin/reference/emi01/study.runfiles"
RUNFILES_DIR="$emi01_review_runfiles" \
PYTHONPATH="$emi01_review_runfiles/_main:$emi01_review_runfiles/rules_python+:$emi01_review_runfiles/rules_python++pip+emi01_pypi_312_numpy_cp312_cp312_manylinux_2_27_x86_64_e7dd01a4/site-packages" \
"$emi01_review_runfiles/rules_python++python+python_3_12_x86_64-unknown-linux-gnu/bin/python3" -P \
  docs/evidence/emi01/review/independent_numerical_review.py \
  /absolute/path/to/invocation --output /absolute/path/to/review.json
```

The script does not need the proprietary model archive or a new simulation. Its implementation,
manifest and terminal hashes are recorded in each newly generated review report. Run the
canonical invocation audit as well; numerical agreement alone cannot certify missing or
misidentified jobs.
