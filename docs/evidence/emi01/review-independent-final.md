# EMI-01 fresh independent source review

Reviewed implementation: `bf1165fe92fb5f527bbddfcb53374956d62929a5`.
Baseline: `17519fc869a7088eccdd8fed53ca1a3758758972`.
Review date: 2026-09-16.

**Reviewer role and independence:** `independent_final_review`, a newly assigned
reviewer who did not author the reference harness, adapter, numerical routines,
fixtures, tests, contracts, or earlier numerical/accounting reviews. This reviewer
made no implementation changes. Other reviewers' prior authorship is disclosed in
their separate records; their numerical results are supporting evidence rather
than a claim that this reviewer independently reran them.

**Disposition:** no unresolved implementation blocker found in the reviewed scope.
This is a source and contract review, not a declaration that EMI-01 is complete.
Retained invocation 1 was running during review; neither two-invocation completion
nor retained qualification/performance success is asserted here. Final delivery
still requires both complete invocations, terminal audits, independent numerical
recomputation, canonical runtime-identity verification, and the measured report.

## Scope and method

Read AGENTS.md, ADR-001, the EMI-01 frozen section of ADR-002, relevant building
instructions, all implementation modules under `reference/emi01`, numerical and
adapter tests plus selected metric/accounting/report test cases and coverage,
Python runtime/build/provenance
files, model provenance, proposed CPU capability slices, and the future GPU
experiment contract. Traced qualification, full job scheduling, result status,
ranking, raw reconstruction, source identity and timing paths. Inspected the
three exploratory independent numerical-review JSON records as historical
diagnostics and read the separately authored independent numerical-review script.

Only source/text reads and Git diff inspection were performed during measurement.
No build, test, simulation, evidence audit, raw decompression, FFT, benchmark, or
large-file hashing was launched by this reviewer. Test coverage below means
inspection of the test cases, not a claim that this reviewer executed them.

`git diff --name-status 17519fc..bf1165f` and a zero-difference protected-path
inspection confirmed no changes to `cpp`, `cuda`, `src`, `tests`, Cargo files,
`acceptance`, or existing `docs/evidence` bytes. The ngspice build diff exposes
existing outputs/provenance and reorders attributes; its configure options remain
unchanged. New Python/runtime targets serve only the external reference harness.

## Findings and conclusions

| Area | Review conclusion |
|---|---|
| Electrical job boundary | Each job contains the complete four-device bridge, two filter conductors, coupled choke/harness windings, load, and chassis return. Gate drives use the appropriate local source reference. Port sensors use outward current orientation. No coupled electrical subsystem is scheduled independently. |
| Final fixture identity | Both fixtures place the DC-link capacitor at `p1`, before the explicit 20 nH commutation inductance. The held-off DPT gate retains its resistor. The lossy load is explicitly hypothetical `20 ohm + (100 uH || 100 ohm)`; its separate dissipation is reported and excluded from filter loss/mass. ADR, manifest and deck construction agree on the final candidate-specific integration grids. |
| Device translation | The adapter checks complete archive, member and adapted identities, restricts the temperature rename and two behavioral-current syntax changes, and preserves expressions and terminal directions. Independent affine-capacitance tests discriminate current sign, the baseline `1 + multiplier`, derivative, and integrated charge. These tests establish the translation mechanism, not independent vendor-model accuracy. |
| Raw and spectral processing | Raw parsing rejects malformed schema, non-finite values, truncation, incomplete or non-monotonic time, and excessive integration gaps. Half-open resampling is bracketed. Mean removal, periodic Hann normalization, one-sided sinusoidal RMS amplitudes, the 986-bin 150 kHz through 10 MHz band, and average-current CM/DM signs match the contract. The accepted comparison uses either-side amplitude above 10 uA for the 1 dB gate and at most 1 uA absolute difference below it. Analytic tests discriminate normalization, phase, side bins, Parseval, interpolation and floor boundaries. |
| Numerical qualification | All nine candidate/corner combinations receive three integration levels, with the heavy override applied to every corner. Both adjacent integration comparisons and finest-waveform output-grid comparisons are included. The study requires 40 qualification checks and successful input jobs. DPT checks use signed terminal energy, fixed event windows, loaded second turn-on, gate/drain coupling and three strictly decaying same-polarity ringing peaks; unresolved ringing cannot qualify. |
| Physical constraints | Device/capacitor peaks and magnetic flux include startup. RMS loss calculations use the observation interval and account for copper, capacitor ESR and both damping resistors. Individual damper limits supplement total loss. Missing, failed or unsettled corners cannot produce a feasible candidate; passed global qualification is also required. The hypothetical mass/rating model is not represented as a component catalogue or thermal certification. |
| Accounting and evidence | The expected schedule is exactly 30 qualification plus 72 warmup/measured jobs. The audit reconstructs identities independently of claimed counts, checks required files and raw hashes, regenerates decks, and recomputes successful metrics, spectra, qualification and ranking. Failed futures retain terminal records. The numerical recomputation within this audit shares harness routines and is therefore complemented by the separate independent numerical review. |
| Runtime diagnosis | Study wall time begins before materialization and includes worker startup/shutdown, scheduling, simulation, validation, required output and a complete summary write. Setup is separately recorded and included in invocation time. Process-phase and nested ngspice timers are distinguished. FIFO and factor/solve budgets are explicitly counterfactual, and failed qualification disables those accuracy-dependent budgets. No measured GPU acceleration is claimed. |
| Scope | CPU FP64/KLU production authority remains unchanged. Proposed EMI-02 CPU slices and the future native-FP64 GPU experiment are documents only. They preserve whole-job state/rollback, fresh CPU qualification, complete output/accounting, fixed resource and performance gates, and do not authorize kernels, dispatch, mixed precision, an optimizer or distributed work. |

## Limits requiring accurate final reporting

1. `time_to_lightest_feasible_s` conservatively waits for **all nine job
   completions**, even if a lighter feasible candidate could be established
   earlier. It is a complete-study proof-time bound, not the earliest discovery
   time of an optimized fixed-order search. If no candidate is feasible, it must
   remain null.
2. The harness audit checks source/model identities and retained artifact
   consistency. Simulator/interpreter/native-library hashes are recorded runtime
   observations; the audit does not itself compare those fields against the
   currently selected canonical binaries or attest execution. A separate final
   artifact-identity check must compare the retained invocations with the canonical
   Bazel outputs and each other. Neither hashes nor differing terminal records
   cryptographically prove independent physical execution.
3. Internal timing excludes interpreter/import startup before `main`, Bazel
   fetch/build, and physical `fsync`. The last timing-field rewrite is declared
   administrative. Service latency excludes queue wait, which is separately
   recorded. Process RSS maxima do not establish aggregate simultaneous peak
   system memory. Three complete-study samples support only the stated empirical
   observations, not a population-tail or GPU-throughput claim.
4. The historical exploratory JSON schema contains both rejected-proposal and
   original-gate statistics. In `exploratory-before-heavy-refinement.json` and
   `exploratory-heavy-waveform-sampling.json`, `old_failed_bins` evaluates the
   accepted 10 uA / 1 uA floor policy; `failed_bins` and associated floor maxima
   arose from the rejected 100 uA / 10 uA proposal. The former record's
   `original_spectral_failed_bins: 273` is a real negative result, not a pass.
   It must not be relabeled as retained qualification. The final independently
   authored review script uses the original accepted gates only.
5. This reviewer did not rerun vendor downloads, license-site inspection, or
   numerical computations. The model omits bipolar reverse-recovery charge and
   self-heating; hardware ringing, energy and EMI can therefore differ materially.
   The load, magnetic geometry, mass and rating screens are hypothetical bounded
   assumptions. The research reserve does not bound unknown model error, and no
   laboratory, aviation, thermal, manufacturing or full-system stability
   qualification follows.

The separate numerical, accounting and canonical-artifact checks after both
retained invocations must supply the evidence needed to close these pending
delivery gates. This review identifies no reason to change the frozen running
workload or relax its numerical thresholds.
