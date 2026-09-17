# EMI-01 v2: passing and near-boundary filter references

This follow-up strengthens the external CPU reference with a passing filter and a distinct
near-boundary candidate. It preserves the [original v1 results](emi01-results.md) and raw evidence.
Production C++/CUDA, Rust/Cargo, CPU FP64/KLU authority and the proposed EMI-02 slice boundaries
remain unchanged.

## Frozen design and acceptance

The [ADR-002 follow-up](adr/ADR-002-inverter-emi-design-study.md) and
[v2 manifest](../reference/emi01/manifest-v2.json) bind the exact cases. The original `light`
candidate remains the failing control. The two new cases share 330 uH DM inductors, 1 mH CM
windings, 47 nF Y capacitors and the same enlarged hypothetical magnetic geometry. `reference`
uses a 1 uF X capacitor; `boundary` uses 80 nF. The component mass model predicts approximately
3.80842 kg and 3.80623 kg respectively. Their 2.19 g mass difference is not a mass-optimization
result; the capacitor change isolates useful acceptance-boundary behavior.

The inverter, dynamic SiC model, load/harness/chassis, source modulation, all three operating
corners, 90 dBuA research mask, 6 dB reserve and all physical screens remain unchanged. Both new
cases use 0.625/0.3125/0.15625 ns integration refinements. The original spectral floor/tolerances,
output sampling, observation interval, settling checks and resource limits remain unchanged.

The [51 exploratory attempts](evidence/emi01-v2/design/README.md) include rejected designs and
failed coarse-grid comparisons. They support design selection only. Complete retained v2
qualification and timing are recorded separately; exploratory timings are not evidence samples.

## Retained qualification

Both independent invocations complete **102/102 validated jobs with zero failures**, and each
passes all 40 numerical qualification checks and every fixture-role gate. Each retains
55 predicted-feasible ensemble results, 44 predicted-infeasible results and three qualified
double-pulse results. Valid infeasible designs are numerical results, not simulation failures.
The finest-grid results are below; complete artifact audits and independent recomputation pass. The
required reserve is **6 dB** against the unchanged 90 dBuA research mask, over every declared
frequency bin, corner, CM/DM component and individual conductor.

| Candidate | Model mass (kg) | Worst CM margin (dB) | Worst DM margin (dB) | Worst overall margin (dB) | All-corner decision |
|---|---:|---:|---:|---:|---|
| Light control | 0.228587 | -24.333287 | -37.669423 | -37.743435 | Infeasible |
| Boundary | 3.806228 | 10.007124 | 5.697721 | 5.516619 | Infeasible: margin |
| Reference | 3.808419 | 10.102329 | 26.718922 | 10.102246 | Feasible |

The boundary's limiting case is conductor A at 150 kHz in `fast_low_lc`, 0.483381 dB below
acceptance. Its nominal/hot-corner minimum margins are 8.121852/10.681709 dB. The reference's
limiting case is conductor A at 2.85 MHz in `fast_low_lc`, 4.102246 dB above acceptance; its
nominal/hot margins are 15.669638/15.822837 dB. Both new candidates pass all declared physical
and settling screens at every corner. The light control fails both margin and physical screens.

| Candidate | Maximum capacitor voltage (V; limit 504) | Maximum CM flux (T; limit 0.2) | Maximum winding RMS (A; limit 12) | Maximum filter loss (W; limit 25) |
|---|---:|---:|---:|---:|
| Light control | 1203.752533 | 0.466832 | 13.113873 | 26.770469 |
| Boundary | 469.581676 | 0.072548 | 10.339038 | 22.335094 |
| Reference | 463.893095 | 0.073001 | 10.483682 | 17.555169 |

These maxima can occur at different corners. The remaining device-current, device-Vds, DM-flux
and individual Y-damper screens pass for both new candidates. The declared constraints do not
establish complete gate-rating, safe-operating-area, thermal or manufacturing qualification.
Only the reference is feasible within this three-candidate set; it is not a minimum-mass design
outside that finite comparison.

The unchanged double-pulse fixture gives 17.973 A at the second turn-on, 17.121/12.694 ns
turn-on/turn-off Vds edges, 440.108 V peak Vds, and 124.237/43.582 uJ signed terminal energies.
Measured ringing is 66.116 MHz with logarithmic decrement 0.608457. Its gate range remains
-3.330 to 24.263 V: the declared crossing/edge observability test passes, but the peak is not
an absolute gate-oxide rating qualification. Signed terminal energy includes recoverable
capacitive energy and is not relabeled intrinsic switching loss.

The limiting margins at the three integration levels are 5.516605/5.516606/5.516619 dB
for the boundary and 10.071196/10.095318/10.102246 dB for the reference. No classification
changes across those refinements. The general 1 dB spectral comparison tolerance is not
permission to cross the 6 dB acceptance threshold: future CPU/GPU qualification must also
preserve the final feasibility decision. The boundary fixture makes that distinction testable.

## Complete CPU study and acceleration budget

The [canonical paired report](evidence/emi01-v2/cpu-budget.json) first audits every raw artifact,
result, identity and terminal record in both invocations. The
[delivery supplement](evidence/emi01-v2/delivery-supplement.json) adds queue/completion quantiles.
The recorded host is an AMD Ryzen 9 9950X3D, 16 cores / 32 logical CPUs. Each mode uses persistent
Python workers with one-thread numerical libraries and a fresh private ngspice process per job.
Each job keeps the entire electrically coupled network together.

Whole-invocation wall times are **3705.178 s** and **3716.740 s**, including initial preparation,
all qualification simulations and comparisons, both warmups, all measured studies, validation,
spectra and closed required output. One-time setup is 0.315/0.301 s. Build/fetch and subsequent
audits are excluded; physical fsync is excluded. The separately retained outer command timings
are rounded to 1:01:45 and 1:01:57. There are zero failed or missing jobs in any denominator.

Each measured study contains nine jobs; each row contains three studies and 27 job observations.
Service latency starts when a worker begins a job; queue and submission-to-completion latency
are reported separately. P95 is empirical nearest-rank over those 27 jobs, not a population-tail
estimate. Throughput is the median of the three per-study rates, each nine validated jobs divided
by complete-study wall time.

| Invocation / workers | Complete-study samples (s) | Median study (s) | Validated jobs/hour | Service median / P95 (s) |
|---|---|---:|---:|---:|
| 1 / 1 | 613.538, 610.738, 611.737 | 611.737 | 52.964 | 76.179 / 77.406 |
| 1 / 4 | 207.357, 207.529, 208.284 | 207.529 | 156.123 | 79.644 / 81.454 |
| 2 / 1 | 614.216, 612.415, 614.289 | 614.216 | 52.750 | 76.521 / 77.056 |
| 2 / 4 | 208.169, 207.655, 206.806 | 207.655 | 156.028 | 78.321 / 80.650 |

The four-worker complete-study speedups are **2.948x** and **2.958x**. The qualified reference is
the only all-corner feasible design in this finite set. Its complete-set confirmation occurs at
the last result, slightly before summary output closes; exact time-to-lightest fields remain in
the study records. Because qualification already evaluated these inputs, this is repeated
finite-set evaluation, not discovery of a new design.

| Invocation / workers | Queue median / P95 (s) | Submission-to-completion median / P95 (s) |
|---|---:|---:|
| 1 / 1 | 229.789 / 535.355 | 306.236 / 611.700 |
| 1 / 4 | 50.526 / 131.922 | 131.900 / 207.485 |
| 2 / 1 | 230.737 / 537.197 | 307.416 / 614.180 |
| 2 / 4 | 50.912 / 131.073 | 131.055 / 207.608 |

The following are **summed worker service seconds across 27 measured jobs**, not additive parallel
study wall time. Simulation includes process launch, ngspice computation and native raw output;
required output below is subsequent closed harness output. Metrics includes spectral processing.
Cleanup/hashing is the measured service residual, not a separately instrumented phase.

| Invocation / workers | Preparation | Simulation | Validation | Metrics/spectra | Required output | Cleanup/hashing residual |
|---|---:|---:|---:|---:|---:|---:|
| 1 / 1 | 0.023 | 1814.044 | 5.528 | 7.650 | 7.311 | 0.736 |
| 1 / 4 | 0.028 | 1900.539 | 5.504 | 7.732 | 7.736 | 0.727 |
| 2 / 1 | 0.023 | 1819.301 | 5.293 | 7.636 | 7.298 | 0.695 |
| 2 / 4 | 0.026 | 1884.362 | 5.671 | 8.279 | 7.863 | 0.740 |

Simulation accounts for 98.816–98.870% of measured job service. Independent FIFO replay of the
observed service durations leaves complete-study residuals of 0.222–0.253 s serial and
0.136–0.146 s parallel. These residuals include scheduling/summary overhead and model error;
there is no separately measured pure scheduler CPU cost. Per-study signed residuals are retained.

All 27 jobs in each row expose ngspice telemetry: **29,298,210 accepted steps**, **1,897,026 rejected
steps**, and **64,398,246 Newton iterations** per row. ngspice's load timer combines device
loading/evaluation and assembly; their separate costs are unavailable. Integration-control,
individual device-evaluation counts, and separately instrumented assembly costs are unavailable.
The timers below are nested within analysis, not extra phases to add to simulation wall time.

| Invocation / workers | Analysis (s) | Combined load (s) | Factor (s) | Solve (s) | Factor+solve fraction | Ideal analysis-only ceiling |
|---|---:|---:|---:|---:|---:|---:|
| 1 / 1 | 1798.927 | 829.905 | 145.564 | 60.327 | 11.445% | 1.1292x |
| 1 / 4 | 1884.657 | 875.002 | 151.661 | 62.988 | 11.389% | 1.1285x |
| 2 / 1 | 1801.628 | 831.085 | 145.606 | 60.135 | 11.420% | 1.1289x |
| 2 / 4 | 1868.109 | 866.018 | 150.398 | 62.163 | 11.378% | 1.1284x |

Eliminating factorization and solve entirely would yield only about **1.13x analysis-only speedup**
before acceleration overhead. It cannot support the proposed 2x complete-study gate. The combined
load/device path and transient stepping are the larger opportunities; their acceleration is
unproven and first requires the bounded CPU semantics in the capability-gap report.

A separate optimistic model divides only each measured simulator-phase duration by 2/4/10 or
removes it, then replays ordered jobs with the same worker count and retains external service plus
the nonnegative scheduling residual. It adds no GPU transfer, setup or certification cost.
These are counterfactual median complete-study seconds, **not measured GPU performance**:

| Invocation / workers | Simulator 2x | Simulator 4x | Simulator 10x | Simulator removed |
|---|---:|---:|---:|---:|
| 1 / 1 | 309.454 | 158.312 | 67.629 | 7.284 |
| 1 / 4 | 105.007 | 53.746 | 22.990 | 2.485 |
| 2 / 1 | 310.707 | 158.953 | 67.900 | 7.198 |
| 2 / 4 | 105.090 | 53.807 | 23.037 | 2.504 |

Even ideal 2x simulator acceleration gives only about 1.98x complete-study speedup in the parallel
case before new GPU costs. Any future experiment must beat the fastest qualified persistent CPU
mode and charge all validation/output costs. The proposed native-FP64 workload, accuracy,
resource, 2x median / 1.5x empirical-P95 and cold-time gates are frozen in the
[future GPU contract](emi01-gpu-experiment-contract.md). GPU usefulness remains unestablished.

The maximum observed simulator RSS over each complete invocation is 821,764 KiB. The maximum
reported worker/runner RSS is 1,015,712 / 1,015,964 KiB for runs 1/2. These are per-process high-water
observations, not a measured simultaneous aggregate host peak. Enforced child limits remain 120 s
wall, 110 s CPU, 1 GiB address space, 512 MiB raw file and two million points, with zero retries.
Each invocation retains 3,684,904,459 bytes in 1,050 lossless raw chunks, plus decks, spectra,
logs and records. Repeated identical waveform payloads share storage, but every job was simulated
afresh; novel designs need not have the same compression/output cost.

## Independent checks and scope

[Independent numerical review](evidence/emi01-v2/review-numerical.md) recomputes all 30 qualification
jobs in each invocation: 144 spectral comparisons, 216 waveform checks, 54 settling checks,
40 qualification decisions and all nine fixture-role checks. Maximum integration spectral
differences are 0.713180 dB above the declared floor and 0.748781 uA below it; output-sampling
maxima are 0.009113 dB and 0.013857 uA. All gates pass with no numerical discrepancies.
A diagnostic JSON serialization correction was regression-tested after measurement; it changed
neither predicates nor frozen harness sources. Its initial error logs and corrected identities
are retained separately.

The [independent accounting report](evidence/emi01-v2/review/independent-accounting-review.json)
reconstructs both 102-job schedules, classifications, mass/ranking and CPU-budget arithmetic
without production helpers. The [runtime identity audit](evidence/emi01-v2/review/paired-runtime-identities.json)
confirms all 25 bound sources and eight native runtime artifacts against the retained observations.
This is artifact identity checking, not execution attestation.
[Canonical checks](evidence/emi01-v2/validation/checks.json) include lint/build/tests, ASan/UBSan,
locked dependencies, ngspice/AC acceptance, explicit CUDA tests and required CUDA/sanitizer analysis
rejections. Historical v1 full audits pass from the original source checkout.
[Final delivery review](evidence/emi01-v2/review-independent-delivery.md) covers provenance,
numerical evidence, reporting/accounting and the protected scope. No EMI-02 implementation,
production solver change, GPU kernel or dispatch is introduced.

## Reproduction

```sh
bazel test //reference/emi01:signals_test //reference/emi01:metrics_test \
  //reference/emi01:adapter_test //reference/emi01:study_test //reference/emi01:report_test
bazel run //reference/emi01:study -- --reference-version=emi01-v2 --out=/absolute/new/run-1
bazel run //reference/emi01:study -- --reference-version=emi01-v2 --out=/absolute/new/run-2
bazel run //reference/emi01:study -- --reference-version=emi01-v2 --audit=/absolute/new/run-1
bazel run //reference/emi01:study -- --reference-version=emi01-v2 --audit=/absolute/new/run-2
bazel run //reference/emi01:report -- --reference-version=emi01-v2 --run=/absolute/new/run-1 \
  --run=/absolute/new/run-2 --out=/absolute/new/cpu-budget.json
```

Each invocation contains 30 qualification jobs and 72 warmup/measured jobs: one warmup plus
three measured nine-job studies at one and four workers. Each job simulates the entire coupled
network. The runner records numerical qualification separately from v2 fixture-role acceptance;
both must pass. Infeasible designs remain valid numerical results, while a failed or missing
simulation can never become a feasible candidate. Ranking explicitly orders complete candidates
by computed mass, with identity breaking exact ties.
Fixture coverage also requires the unchanged light control to be numerically valid and
predicted infeasible at all three corners; a solver failure cannot satisfy that role.

Historical v1 audits use the detached source commit documented in
[the reference guide](../reference/emi01/README.md), preserving strict source identities. Current
source hashes do not retroactively replace the original recorded hashes.

## Interpretation limits

A passing reference means predicted passing under the declared finite research model and
screens. It is not aviation compliance, laboratory correlation or a qualified hardware filter.
The enlarged geometry retains assumed CM winding capacitance/coupling, omits DM self-capacitance,
and has no winding-window/fill, gap-fringing or core-loss validation. The load remains an assumed
lumped lossy network. These limitations constrain the physical meaning of the mass and margin.

The [future GPU experiment](emi01-gpu-experiment-contract.md) uses this exact versioned workload
and retains its fresh-CPU-validation, complete-study timing and zero-failure gates. The next CPU
proposal remains [EMI-02A: disjoint linear coupled-inductor pairs](../reference/emi01/CPU_GAPS.md).
No EMI-02 or GPU implementation is included in this follow-up.
