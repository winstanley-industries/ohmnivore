# EMI-01: coupled SiC inverter/filter CPU reference

This study evaluates a finite, coupled inverter/filter/load/chassis network using the public
Microchip MSC040SMA120B dynamic model and the existing hermetic ngspice 46 oracle. Its purpose
is a reproducible design-study baseline and an acceleration budget. It adds no production
Ohmnivore device, parser, transient, GPU or dispatch behavior.

The baseline is `17519fc`; the frozen harness is commit `bf1165f`. The contract is
[ADR-002](adr/ADR-002-inverter-emi-design-study.md), and the executable inputs are the
[versioned manifest](../reference/emi01/manifest.json). Original/adapted proprietary model bytes
are fetched locally and excluded from this repository; see the
[exact model/provenance audit](../reference/emi01/MODEL_PROVENANCE.md).

## Reproduction and evidence

```sh
bazel test //reference/emi01:adapter_test //reference/emi01:signals_test \
  //reference/emi01:metrics_test //reference/emi01:study_test \
  //reference/emi01:report_test //third_party/emi_python:runtime_test
bazel run //reference/emi01:study -- --out=/absolute/new/run-1
bazel run //reference/emi01:study -- --out=/absolute/new/run-2
bazel run //reference/emi01:study -- --audit=/absolute/new/run-1
bazel run //reference/emi01:study -- --audit=/absolute/new/run-2
bazel run //reference/emi01:report -- --run=/absolute/new/run-1 \
  --run=/absolute/new/run-2 --out=/absolute/new/cpu-budget.json
```

Each invocation contains 30 qualification jobs and 72 serial/parallel warmup/measured jobs:
one warmup plus three measured nine-job studies at one and four workers. Every simulation runs
afresh. Raw waveform chunks are losslessly compressed and content-addressed; repeated payloads
share storage, while each job retains its own header, deck, log, spectrum and completion record.
The required-output timer includes checking/creating those links and artifacts. Identical warm
repetitions can reuse existing compressed chunks; this is not a throughput claim for an arbitrary
stream of previously unseen designs. Source, input, raw and terminal identities are auditable.

The [validation logs](evidence/emi01/validation/checks.json) and
[final delivery checks](evidence/emi01/validation/delivery-checks.json) include canonical CPU/sanitizer,
lockfile, CUDA regression/linkage and explicit CUDA/sanitizer incompatibility checks. The
[independent delivery review](evidence/emi01/review-independent-delivery.md) closes the earlier
[source review](evidence/emi01/review-independent-final.md) and complements the
[numerical review](evidence/emi01/review-numerical.md),
[model/scope review](evidence/emi01/review-model-scope.md) and
[accounting review](evidence/emi01/review-accounting.md). Their exploratory records are clearly
separated from retained measurements. Historical AC evidence bytes remain unchanged; the new
AC regression log is separately labeled.

### Executable identity and rebuild boundary

Both retained invocations used the exact ngspice executable beginning `b2574df8`; the successful
paired artifact audit compared that file with the recorded identities before final sanitizer
rebuilds. A later default rebuild produced `afb6b348`, and the subsequent strict identity check
correctly failed. The existing ngspice build embeds absolute installation prefixes containing
Bazel sandbox numbers, so its executable bytes are **not reproducible across those build paths**,
despite the pinned source, toolchain and dependencies. This limitation is not hidden by rewriting
the retained metadata.

The [reproducibility investigation](evidence/emi01/review/ngspice-reproducibility.md) records all
three full hashes. The measured binary was recovered exactly from a preserved exploratory
`9a2b9464` copy by changing six ASCII sandbox-number digits; its full SHA-256 matches the retained
identity. The rebuilt binary's longer paths also change data layout, so a textual comparison was
not treated as a semantic proof. A separate fresh **30-job qualification-only diagnostic** with
the rebuilt canonical executable passed all 40 checks. Independent comparison found every raw
numerical payload byte, spectrum, metric and qualification result exactly equal to the retained
reference (timestamp-bearing headers are compared with only their Date line removed).

Thus the finite numerical study was reproduced across these builds; byte-identical executables
and historical performance on the rebuilt binary are not claimed. The original failed recheck,
exact recovery and bounded comparison remain auditable. Diagnostic timing is excluded from both
retained performance invocations. A new reproduction records its own executable identity and
must run qualification; source identity alone is not permission to reuse numerical acceptance.

## Retained numerical and physical results

Both independent invocations completed **102/102 validated jobs, zero failures and all 40
qualification checks**: 204 complete jobs overall. Each invocation has three qualified DPT
records and 99 predicted-infeasible ensemble records. The independently implemented review
recomputed 144 spectral comparisons, 216 waveform comparisons and 54 settling checks per
invocation, with no failed checks or metric discrepancies. Maximum spectral differences were
0.555321 dB above the original 10 uA floor and 0.273035 uA below it, within the unchanged
1 dB / 1 uA gates. The two invocations' numerical metrics agree exactly.

The finest DPT result, independently recomputed in both invocations, is:

| Quantity | Result |
|---|---:|
| Load current at second turn-on | 17.973 A |
| Second turn-on / turn-off Vds 10–90% edge | 17.121 / 12.694 ns |
| Peak Vds at 400 V bus | 440.108 V |
| Signed second turn-on / turn-off energy | 124.237 / 43.582 uJ |
| Ringing frequency / logarithmic decrement | 66.116 MHz / 0.608457 |
| Full observed gate-voltage range | -3.330 to 24.263 V |

Both adjacent integration refinements and finest-waveform output-grid refinements pass.
The gate observability criterion constrains voltage at the Vds 50% crossings and the turn-on
gate drop; it is not an absolute gate-oxide rating check. The 24.263 V peak is reported, not
hidden behind the crossing criterion. No gate-rating, parasitic-turn-on or safe-operating-area
qualification is claimed.

No candidate meets all three corners. Margins below are the minimum over all declared frequency
bins and corners; the required reserve is **+6 dB**, not zero. The overall column includes both
individual conductors as well as CM and DM. Mass is the explicit hypothetical physical design
model, not a purchased-component mass claim.

| Candidate | Mass (g) | Worst CM margin (dB) | Worst DM margin (dB) | Worst overall margin (dB) | Decision |
|---|---:|---:|---:|---:|---|
| Light | 228.587 | -24.333 | -37.669 | -37.743 | Infeasible |
| Medium | 403.746 | -13.769 | -20.242 | -23.363 | Infeasible |
| Heavy | 668.835 | 2.892 | -5.035 | -6.635 | Infeasible |

Physical screens independently reject these designs:

| Candidate | Maximum capacitor voltage (V; limit 504) | Maximum CM flux (T; limit 0.2) | Maximum winding RMS (A; limit 12) | Maximum filter loss (W; limit 25) |
|---|---:|---:|---:|---:|
| Light | 1203.753 | 0.466832 | 13.114 | 26.770 |
| Medium | 1102.884 | 0.479984 | 11.836 | 28.720 |
| Heavy | 807.488 | 0.345641 | 20.017 | 107.355 |

These maxima can occur at different corners. All device-current, device-Vds, DM-flux and
individual Y-damper screens pass; that does not override the failed constraints above. The
assumed parallel load-loss resistor dissipates 98.115–589.903 W across the nine jobs, separately
from filter loss. The large load dependence is a material modeling limitation. Lightest feasible
candidate and time to a lightest feasible candidate are both null; no successful design is claimed.

## Measured CPU performance

Measurements used the recorded AMD Ryzen 9 9950X3D, 16 cores / 32 logical CPUs, under WSL2,
with one-thread numerical libraries and one or four workers. The complete coupled circuit is
the unit of scheduling. See the [audited paired report](evidence/emi01/cpu-budget.json) and
[evidence index](evidence/emi01/README.md) for all individual samples and identities.

| Invocation / workers | Nine-job study wall samples (s) | Median study (s) | Validated jobs/hour | Job service median / P95 (s) |
|---|---|---:|---:|---:|
| Run 1 / 1 | 539.639, 541.444, 545.096 | 541.444 | 59.840 | 55.531 / 76.206 |
| Run 1 / 4 | 184.962, 182.499, 184.187 | 184.187 | 175.908 | 59.301 / 80.072 |
| Run 2 / 1 | 541.031, 537.079, 538.076 | 538.076 | 60.214 | 55.028 / 76.208 |
| Run 2 / 4 | 184.574, 186.220, 185.386 | 185.386 | 174.770 | 58.986 / 81.198 |

Four workers provide **2.940x and 2.902x** median complete-study speedup. P95 is empirical
nearest-rank over 27 measured jobs per mode, not a population-tail estimate. Service latency
excludes queueing. The [queue/completion supplement](evidence/emi01/delivery-supplement.json)
reports median/P95 submission-to-completion latency: 264.842/541.408 s and 264.757/538.039 s
serially, versus 107.319/184.142 s and 107.633/185.341 s with four workers. Complete-study wall
time includes worker startup/shutdown, scheduling and summary output.

Whole-invocation harness times were **3333.199 and 3338.480 s**, including qualification,
warmups and every measured study. One-time verification/adaptation took 0.298 and 0.299 s
inside those totals. Outer command wall times, including Bazel/interpreter startup, were
3333.58 and 3338.87 s. No competing build, benchmark or numerical audit ran during either
retained invocation. Subsequent audits, sanitizer checks and publication ran after measurement.

Observed maximum simulator RSS was **821,000 KiB** in each invocation. Maximum Python
worker/runner RSS was **1,010,104 / 1,014,552 KiB** respectively. These are individual-process
high-water marks, not aggregate concurrent peak memory. Each simulator enforced the frozen
1 GiB address-space limit. All runs stayed within the wall/CPU/file/point limits, with zero retries.

### Complete service breakdown and telemetry

Totals below sum the 27 measured jobs per mode; parallel service totals are not study wall time.

| Invocation / workers | Preparation (s) | Simulator process (s) | Validation (s) | Metrics/spectra (s) | Required output (s) | Other cleanup/hashing (s) |
|---|---:|---:|---:|---:|---:|---:|
| Run 1 / 1 | 0.022 | 1608.319 | 4.558 | 6.110 | 5.981 | 0.592 |
| Run 1 / 4 | 0.026 | 1687.803 | 4.718 | 6.502 | 6.513 | 0.611 |
| Run 2 / 1 | 0.022 | 1598.328 | 4.652 | 6.138 | 5.860 | 0.594 |
| Run 2 / 4 | 0.027 | 1697.314 | 4.772 | 6.459 | 6.536 | 0.607 |

The simulator phase is approximately **98.93% of complete job service**. It includes process
launch, parsing, initialization, integration and raw output; it is not a kernel timer. The
positive measured-wall-minus-FIFO-service residual was 0.194–0.201 s per serial study and
0.126–0.131 s per four-worker study. This is a fitted scheduling/startup/summary residual,
not an independently isolated scheduler timer; job ordering and final-wave imbalance remain
inside the FIFO model.

Each 27-job measured set reported **24,036,600 accepted steps, 3,982,212 rejected steps and
58,019,652 transient Newton iterations**, identically across modes and invocations. All 27
jobs per mode supplied the available timers:

| Invocation / workers | ngspice analysis (s) | Combined device/matrix load (s) | Factorization (s) | Solve (s) |
|---|---:|---:|---:|---:|
| Run 1 / 1 | 1596.443 | 748.997 | 128.907 | 54.104 |
| Run 1 / 4 | 1674.683 | 787.588 | 135.092 | 56.553 |
| Run 2 / 1 | 1587.308 | 744.798 | 129.164 | 54.000 |
| Run 2 / 4 | 1684.443 | 793.413 | 135.841 | 56.867 |

These ngspice timers are nested within simulation, not extra wall-time phases. Separate device
evaluation and assembly costs are **unavailable**; ngspice exposes their combined load timer.
No estimate is substituted for missing telemetry. ngspice uses its configured Sparse solver;
these percentages are not measurements of Ohmnivore's production KLU implementation.

## Acceleration decision and proposed budget

**Reject a factorization/solve-only acceleration plan for this workload.** Those operations
occupy only 11.44–11.54% of reported analysis time. Even removing them entirely with zero
overhead yields an optimistic **1.129–1.130x analysis-only ceiling**, before charging the rest
of the job. It cannot deliver the proposed 2x complete-study usefulness gate.

A broader device-evaluation/integration experiment has potential because the simulator process
dominates, but that is an opportunity bound, not a GPU feasibility demonstration. The combined
load timer is about 47% of analysis. Accurate separate profiling must precede implementation
decisions, especially after the CPU KLU path supports the selected model.

The frozen report also scales the **entire simulator-process phase**, preserving measured
non-simulation service, FIFO scheduling and nonnegative fitted overhead. Median predicted
study times across each invocation's three samples are:

| Invocation / workers | Simulator 2x faster (s) | 4x (s) | 10x (s) | Simulator cost removed (s) |
|---|---:|---:|---:|---:|
| Run 1 / 1 | 273.703 | 139.832 | 59.510 | 5.961 |
| Run 1 / 4 | 93.222 | 47.735 | 20.400 | 2.098 |
| Run 2 / 1 | 272.026 | 139.001 | 59.186 | 5.960 |
| Run 2 / 4 | 93.803 | 48.000 | 20.489 | 2.139 |

These counterfactuals assume no new transfer, GPU setup or CPU certification cost. They are
deliberately optimistic, and accelerating factorization alone does not instantiate them.
The complete retained invocation also spends time qualifying references and warming every mode;
warm replay throughput does not erase that serial/setup cost or prove faster novel-design search.

The [future GPU contract](emi01-gpu-experiment-contract.md) proposes exact 9-job and
36-job replay ensembles (four replicas of each input), native FP64, full CPU validation,
zero failures/fallbacks, fixed resource
limits, and comparison with the best qualified persistent CPU worker count. It requires at least
2x median and 1.5x empirical P95 complete-study speedup, with cold time no worse, in two new
independent invocations. CPU references are regenerated and charged per invocation. Once they
exist, these candidates' feasibility is already known: faster replay does not establish faster
discovery. No transient GPU implementation or dispatch is authorized by this study.

## Interpretation limits and next work

This is a numerical qualification of a typical-device surrogate, not laboratory correlation or
aviation compliance. The 90 dBuA mask, six-dB reserve, frequency grid and tone detector are
explicit research choices. The model omits bipolar reverse-recovery charge, avalanche validation,
self-heating and statistical device variation. Those omissions can change loss, ringing and EMI;
the assumed model reserve is not a measured uncertainty bound.

The lossy load is an assumed `20 ohm + (100 uH || 100 ohm)` network with explicit terminal,
harness and chassis capacitances. Its loss changes load power and is reported separately from
filter loss. It is not a measured motor impedance. Magnetics are linear, with hypothetical
geometry/mass and flux/rating constraints. A candidate that exceeds the flux limit is rejected;
its linear-model spectrum is not a prediction of a saturated physical core. Thermal feasibility,
core loss, distributed cable effects and manufacturability remain outside this reference.

The next proposed CPU slice is **EMI-02A: disjoint pairs of linear coupled inductors**. The
[separate bounded CPU contracts](../reference/emi01/CPU_GAPS.md) then address only the selected
behavioral expressions/Jacobians, dynamic state and bounded model import/initialization needed
by this reference. None is implemented here. The
[proposed future GPU experiment](emi01-gpu-experiment-contract.md) freezes its workload, accuracy,
resource and complete-study performance gates before implementation. CPU FP64/KLU remains
Ohmnivore's production authority; the old AC thresholds confer no transient dispatch authority.
