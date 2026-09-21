# ADR-002: Batched inverter output-filter design studies

- **Status:** Accepted direction; reference workload and implementation contracts pending
- **Date:** 2026-09-16
- **Scope:** Planning for combined common-mode and differential-mode filter optimization
- **Relationship:** Updates the follow-up priorities in ADR-001; preserves its implemented contracts

## Context

The target workflow is minimum-mass output-filter design for switching motor-drive inverters,
including realistic SiC device behavior and parasitic capacitance and inductance. Both common-mode
(CM) and differential-mode (DM) emissions matter. The intended benefit is evaluating many filter
designs and operating/tolerance corners concurrently while retaining agreement with trusted
simulation and, when available, laboratory measurements.

The user identified prior work associated with
[US10079493B2, Parallel modular converter architecture](https://patents.google.com/patent/US10079493B2/en).
The description includes inverter output EMI filters and load-specific filtering of combined
parallel-module outputs. The related
[US9647455B2, EMI filter systems and methods for parallel modular converters](https://patents.google.com/patent/US9647455B2/en)
provides additional public filter context. These references motivate a public surrogate workload;
they do not supply the original netlists, device models, mass data, measurements, or applicable
aviation test category. The initial study must not claim to reproduce that historical design.

GPU-02 and GPU-02S established correct native-FP64 CUDA execution but no qualifying end-to-end
speedup on their frozen synthetic linear-AC workloads. Persistent ownership already removed CUDA
startup from steady sessions. Those negative results remain valid. They do not measure an ensemble
of nonlinear switching simulations or establish its likely speedup.

## Decision and unit of work

Prioritize a bounded inverter output-filter design study before further synthetic-AC tuning or a
new GPU solver implementation. One job is:

```text
complete coupled inverter/filter/load configuration
    x filter candidate
    x operating and tolerance corner
```

All electrically connected inverter modules, mutual inductances, chassis returns, and loads remain
in the same job. Separate filter candidates and corners are independent jobs. Topology sharing is
an optimization opportunity, not permission to split coupled modules or reuse another candidate's
nonlinear state. A simulation's timesteps are sequentially dependent; independent AC points are
not a substitute for this workload.

The primary performance objective is more numerically validated candidate/corner evaluations per
hour at fixed accuracy. A valid result can predict either feasible or infeasible behavior. A solver
failure is neither. Report time to complete the fixed study, individual-job median/P95 latency,
failure counts, memory, and time to the lightest feasible candidate in a fixed search order.

For a finite candidate set, the design objective is:

```text
minimize M(candidate)
subject to E(candidate, corner, observable, frequency)
           <= Limit(observable, frequency) - Margin(observable, frequency)
for every required corner and evaluated emission observable/frequency,
plus component stress, loss, thermal, and stability constraints declared by the study.
```

Emission quantities, limits, and margins must use the same units and detector definition. CM/DM
diagnostics do not replace any conductor or port measurement required by the chosen test setup.
Feasible means predicted feasible over the declared finite domain, not certified or globally
optimal. Mass comes from a versioned component catalogue or explicit magnetic/component design
model with ratings and provenance; nominal L/C values alone do not determine mass.

## Staged work

### EMI-01: Reference workload, CPU study, and performance budget

The next deliverable is one reproducible public reference study. Begin with one fixed inverter and
combined CM/DM output-filter topology and a finite, ordered candidate/corner set. Use a half-bridge
double-pulse fixture to qualify device switching behavior, then a switching pulse train and defined
load/harness/chassis network to evaluate the filter. A double-pulse spectrum is not the complete
motor-drive emissions benchmark.

Select exact topology, voltage/current/frequency envelope, component/model versions, and candidate
values from documented public sources. The original engineering files are not a prerequisite.
Record assumptions explicitly; a simplified load is a development fixture until its relevant
impedance behavior is substantiated. Do not substitute an ideal switch for a realistic SiC model
and label the resulting runtime or spectrum representative.

The reference simulator runs outside the Ohmnivore production path. Prefer the existing
checksum-pinned, Bazel-built ngspice oracle if it supports the selected model exactly. Audit model
syntax, equations, redistribution terms, and simulator compatibility first. A public vendor model
is not automatically compatible with ngspice or with Ohmnivore. Any additional reference tool or
model translation needs recorded version, provenance, settings, and a bounded compatibility
decision; do not add an ambient simulator/toolchain dependency.

EMI-01 must produce:

1. A versioned manifest binding circuit/model bytes, topology, candidate and corner identities,
   component/mass data, simulation settings, required outputs, and measurement processing.
2. A CPU reference runner and raw switching waveforms, spectra, and per-job completion records.
   Every scheduled job must have a result or explicit failure; missing jobs cannot disappear from
   the denominator or be classified as passing the emission limit.
3. A performance breakdown covering initialization, accepted/rejected timesteps, Newton
   iterations, device evaluation, assembly, factorization/solve, validation, output, and spectral
   processing. Mark counters unavailable from the reference simulator rather than inventing them.
4. A capability-gap report and bounded implementation contracts for the Ohmnivore features the
   selected model actually requires.
5. A quantified acceleration budget and frozen numerical/performance gates before GPU
   implementation. A negative feasibility result is a valid stopping point.

Choose the device and a runnable CPU reference case before implementing a general optimizer or
bulk-porting transistor models. A failure to qualify the reference model leaves EMI-01 incomplete;
it is not a reason to silently simplify the physics or weaken acceptance.

### EMI-02: Required CPU semantics and differential qualification

Implement the gaps identified by EMI-01 as separate, ADR-first slices. CPU FP64/KLU remains the
Ohmnivore authority and supported no-GPU implementation. The existing NL-04 MOSFET DC plan is a
foundation only: its Level-1, memoryless DC scope is insufficient for SiC switching and EMI.
Retain its boundary unless an explicit successor contract replaces it.

Current C++ capabilities and likely gaps are:

| Capability | Current boundary | Required study decision |
|---|---|---|
| Linear RLC, sources, adaptive transient | Implemented | Reuse established signs, initialization, and numerical contracts |
| Diode and BJT models | Memoryless diode DC/transient and bounded BJT DC | Specify required reverse-conduction and dynamic behavior |
| SiC MOSFET switching | MOSFETs and device charge are absent | Select bounded current/charge equations or an audited model subset; verify derivatives and charge conservation |
| Common-mode choke | No mutual-inductance component in the C++ IR | Specify winding orientation, coupling/leakage, passivity, and any required nonlinear magnetics |
| Vendor model language | No general subcircuit/controlled-source/behavioral-model support | Admit only constructs required by the selected reference, with explicit unsupported failures |
| Harness, motor, and chassis | Explicit RLC can represent a bounded equivalent network | Establish impedance/return paths over the claimed frequency band; do not assume an ideal load is representative |
| EMI measurement and physical mass | No study evaluator | Implement the frozen observable, spectral, rating, and mass contracts |

Validate analytic current/charge/Jacobian and coupled-element cases independently. Compare complete
switching waveforms and derived metrics with the external reference, including edge timing,
overshoot, ringing frequency/damping, switching energy, and CM/DM spectra. Laboratory correlation
is a separate claim requiring identified measurements and the measurement-chain uncertainty.
Simulator agreement alone does not establish agreement with hardware.

### EMI-03: Native-FP64 GPU transient ensembles

Only after CPU semantics and the reference workload qualify, evaluate GPU execution of independent
candidate/corner jobs. Keep per-job state, identity, convergence, timestep acceptance, rollback,
and failure accounting. Group compatible sparse structures where beneficial, but measure the cost
of divergent timesteps and Newton iterations. Do not force all jobs onto the hardest candidate's
time grid merely to make batching convenient.

Profile-guided candidates include device evaluation/assembly, batches of linear solves, and keeping
iteration state on the device. The reference budget selects the first bounded experiment; this ADR
does not select cuDSS, a custom solver, or device-resident Newton as the future algorithm.

Compare against a persistent parallel CPU ensemble with private solver state, using the same jobs,
accuracy, output contract, and hardware resource envelope. CPU core usage and tuning must be
recorded. Comparison against the deterministic single-thread oracle alone is insufficient.
Retain full CPU differential qualification. Any future production acceptance policy that avoids
fresh CPU re-solving needs its own correctness contract; the GPU-02S candidate-runtime lane does
not provide one.

### EMI-04: Filter search and robustness

Begin with deterministic enumeration of the frozen candidate set. Add an optimizer only after the
evaluator is trustworthy. Compare search methods with the same evaluation budget and report mass,
emission margin, losses, and all declared constraints. Track the best feasible design found rather
than claiming a global optimum from a finite or heuristic search.

Linear frequency-domain screening may reject or prioritize candidates only under a validated
source/load approximation and recorded screening policy. Changing a filter can change inverter
switching behavior. Detailed coupled transient evaluation remains necessary for final candidate
qualification, and screening errors must be measured before screening can discard candidates.
Retain promising designs across mass and margin tradeoffs and evaluate the declared tolerance and
operating corners; a nominal pass alone is insufficient.

## Reference inputs and qualification gates

The direction is accepted; the following experiment inputs are deliberately not fabricated here.
EMI-01 freezes circuit, design, measurement, and numerical-accuracy inputs before qualifying CPU
reference results. It uses the CPU runtime diagnosis to set the performance contract, then freezes
that contract before GPU implementation or CPU/GPU performance comparisons:

| Contract | Required contents |
|---|---|
| Circuit/model | Exact topology and model bytes, parameters, coupling signs, parasitics, drive timing, initialization, temperature assumptions, and load/chassis network |
| Design domain | Candidate values/parts and order, corner set, correlations, mass/rating provenance, and explicit exclusions |
| Measurement | Ports, current directions, voltage reference, CM/DM normalization, units, measurement network, and the actual test procedure/category if a requirements claim is made |
| Spectrum | Settling criterion, observation duration, maximum step/output sampling, anti-alias/resampling method, window and normalization, frequency grid, bandwidth, detector, and startup treatment |
| Accuracy | Absolute/relative waveform and derived-metric tolerances, frequency-dependent spectral tolerances and floor, reference refinement/convergence checks, and lab uncertainty where available |
| Numerical failures | Typed unsupported/malformed/singular/non-convergent/non-finite/budget-exhausted outcomes, deterministic retry and timestep limits, and no publication of partial or invalid spectra |
| Performance | Fixed batch sizes, repetitions/warmups, median/P95 throughput or campaign-time gates, memory/resource limits, timing boundaries, and full job-count reconciliation |

Do not infer an aviation standard, revision, category, limit curve, bandwidth, or detector from the
word "aviation" or from the patents. If those inputs are unavailable, use an explicitly identified
research mask and label results as research comparisons. Preserve a measurable margin and account
for numerical/model uncertainty before classifying a design as predicted feasible.

The spectral contract must demonstrate convergence as timestep/output sampling is refined. Specify
both frequency resolution and the high-frequency bandwidth of interest. Resampling adaptive output
or applying an FFT does not by itself establish a valid emission measurement. CM/DM transformation
and return paths require analytic fixtures; an omitted chassis path cannot be interpreted as zero
common-mode emission.

## Evidence boundaries and acceptance gates

Use one manifest and stable job identities for serial CPU, parallel CPU, and later CUDA results.
Preserve raw data and source/model/toolchain/hardware identities. Report predicted infeasibility
separately from numerical failure, timeout, and unsupported input. A benchmark with missing or
failed mandatory jobs cannot pass by reporting throughput for its surviving subset.

The primary completed-study time includes candidate materialization, preparation, scheduling,
nonlinear simulation, all host/device coordination, required numerical validation, spectral
evaluation, and required result output. Publish cold and persistent boundaries separately and
charge setup explicitly. Detailed phase telemetry is diagnostic; it does not replace this total.
Any certification performed outside a diagnostic interval must also appear in a certified total.

Before the first GPU implementation, freeze hardware, job set, accuracy, resource limits, sampling,
and numeric speedup thresholds in its bounded contract. Require two independent complete evidence
invocations for a speedup claim. Optimizations shared by both backends must benefit the CPU
comparator too. The old AC thresholds belong to their frozen workloads and are not silently
reinterpreted as transient-ensemble thresholds.

The immediate planning change is complete when ADR-001, the roadmap, and README agree on this
direction and identify the unresolved reference inputs. EMI-01 itself is complete only when its
manifest, qualified CPU results, runtime diagnosis, capability gaps, and next implementation
contract are reviewable. Later GPU work requires those results and the CPU qualification gates.

## Preservation and exclusions

This ADR changes planning priority, not supported simulator behavior. It adds no device model,
mutual-inductance stamp, model-language construct, spectral evaluator, search engine, GPU kernel,
automatic dispatch, or compliance claim. Existing Phase 1--3C, GPU-01, GPU-02, and GPU-02S runtime
and validation contracts remain in force. Rust/Cargo remains a behavioral reference. Mixed
precision, multi-GPU/distributed solving, single-circuit domain decomposition, radiated-field
simulation, and full aircraft/system certification are outside the initial study.

Preserve all existing AC replay and evidence bytes. GPU-02S fingerprints include README, ADR-001,
and the roadmap, so this planning update changes that benchmark's input fingerprint even though
solver code is unchanged. Its checked-in August evidence remains historical evidence for the
recorded inputs, available at `e3fe7128d26b4158402ec48b48e7268c227ee508`, not a fresh measurement of
this documentation revision. A future claim for revised inputs must regenerate and audit its
complete evidence; do not relabel old rows or edit their identities.

## Public reference starting points

- [Wolfspeed power-module SPICE model guide](https://assets.wolfspeed.com/uploads/2023/10/Wolfspeed_PRD-07913_Power_Modules_SPICE_Models_User_Guide.pdf):
  candidate source for model documentation and double-pulse fixtures, subject to the compatibility
  and provenance checks above; no part or model has yet been selected.
- [Existing ngspice provenance](../../third_party/ngspice/PROVENANCE.md): hermetic external oracle
  infrastructure, presently qualified only for the listed bounded acceptance fixtures.
- [GPU-02S evidence](../evidence/gpu02s-persistent-session-summary-2026-08-16.md): existing negative
  result and preparation/validation timing diagnosis; not evidence about the proposed EMI study.

## EMI-01 v1 frozen reference contract (2026-09-16)

This section authorizes only external reference tooling, fixtures, tests and evidence. It was
written after exploratory compatibility probes and before the retained harness and qualification.
Exploration in `/tmp/emi01-exploration` is not qualification or performance evidence. The untouched
Microchip model fails ngspice's PSpice parser (the local `TEMP` name collides with `temper`, and
`F ... VALUE` is not accepted). Wolfspeed's download terms disallow redistribution; the inspected
Microchip library is also proprietary. Neither library is vendored. A checksum-pinned public
Microchip archive is fetched for local evaluation. No proprietary model text is embedded in the
published decks or evidence.

### Device, circuit and finite domain

Use Microchip **MSC040SMA120B**, 1200 V / 40 milliohm TO-247 SiC MOSFET, from `MSCSMA120.lib`,
build 2026-05-05, internal version 2026.5. Archive SHA-256 is
`de4a3acf222cbc7f6c1a4d1a2bc449dd31b5db2c6ed153920d797752c3831d51`; member SHA-256 is
`6e888e103977f539e62b64952797ded391d2a49c59738bfd53f5fbe4b1fc8df7`.
Use the existing static ngspice 46 oracle without simulator changes. A bounded mechanical adapter
alpha-renames the model's local `TEMP` identifier to `TJ_C` and changes exactly two `Fgd/Fds ...
VALUE=` sources into `Bgd/Bds ... I=` sources, preserving terminals, expressions, sensing sources,
and baseline capacitors. It normalizes line endings/encoding only. No current or charge equation
is refitted. Tests check the exact substitutions and independently verify effective Cgd/Cds
current, including the baseline capacitor's **1 + multiplier**, signs and integrated charge.

The vendor model includes gate/source package inductance, reverse conduction, Cgs and nonlinear
Miller/output capacitance. It omits bipolar reverse-recovery charge (turn-on energy may be low),
self-heating, avalanche qualification and statistical process characterization. These omissions
bound the reference; this is numerical qualification of a public typical-device surrogate, not a
validated hardware model. Junction temperature is fixed within each job.

The qualification fixture is a 400 V half bridge with a 200 microhenry charging inductor, 20 nH
bus inductance, 20 milliohm supply resistance, 10 microfarad local DC link, 5 milliohm inductor
resistance, 4.7 ohm external gate resistance, and the same SiC model for both switches. High-side
gate is held at -3 V. Low-side gate PWL transitions (-3 to +20 V) last 10 ns: on at 1 us, off at
10 us, on at 12 us, off at 14 us; stop 16 us. No UIC: ngspice DC operating point initializes the
fixture. Record low-side drain voltage, gate voltage, drain terminal current and load current.

Study jobs contain four SiC devices in a two-leg full bridge. Both legs switch at 50 kHz: high-side
A on from 1 us for 14.8 us and high-side B on from 1 us for 4.8 us; low-side gates complement them
with 200 ns deadtime and 10 ns source edges. This is a fixed positive modulation operating segment,
not a complete rotating-machine fundamental cycle. Each gate has 4.7 ohm external resistance.
A 20 milliohm supply feeds a 10 uF DC link; 20 nH separates that capacitor from the bridge.
Each switch node has 100 pF to chassis.
Each conductor crosses a series DM inductor and one CM-choke winding (dots at the inverter side,
positive mutual coefficient 0.995); equal outward currents reinforce CM flux. Each CM winding has
10 pF terminal capacitance; 5 pF couples the two output conductors. Filter X capacitance connects
the outputs and two Y capacitors connect each output to chassis, all with 0.2 ohm ESR. Each Y branch also has 21.8 ohm series damping (22 ohm total). Winding
resistances follow the physical design below. Harness: 0.1 ohm and 1 uH per conductor, coupling
0.2, 200 pF/conductor to chassis. Load: 20 ohm in series with (100 uH || 100 ohm) between conductors, with 1 nF from
each load terminal to chassis. Chassis returns through 1 ohm + 50 nH to DC-negative reference;
DC-positive and DC-negative each have 100 pF to chassis. There are no disconnected return paths.
These lumped harness/load and constant linear magnetics are declared development assumptions;
no motor impedance, magnetic-loss frequency curve, saturation waveform or cable transmission-line
correlation is claimed above or below the measured band.

Candidate order is `light`, `medium`, `heavy`, with per-conductor DM inductance 10/22/47 uH,
per-winding CM inductance 100/220/470 uH, X capacitance 47/100/220 nF and Y capacitance
1/2.2/4.7 nF. Corner order is `nominal`, `fast_low_lc`, `hot_high_c`: respectively
(bus V, fixed junction C, L scale, filter C scale, stray C scale, gate-R scale) =
(400,27,1,1,1,1), (440,27,0.9,0.9,1.2,0.8), (360,125,1.1,1.1,1.2,1.2).
These nine jobs are finite engineering sensitivity points, not a statistical tolerance envelope.
Each job is solved in full; electrical modules never become independent scheduling units.

### Physical mass, ratings and predicted feasibility

Mass is an explicit hypothetical component design model, not a supplier quotation. For each of
two DM cores, candidate `(Ae mm^2, le mm, turns, mean turn length mm)` is
(100,60,20,45), (160,75,26,55), (250,90,32,65). For the shared two-winding CM core it is
(80,60,12,40), (120,75,18,50), (180,90,24,60). Use copper area 4 mm^2, copper resistivity
1.724e-8 ohm m at 20 C, copper density 8960 kg/m^3 and assumed core density 4800 kg/m^3;
add 20% of magnetic/copper mass for bobbin, insulation and assembly. `R = rho*N*MLT/Awire`.
Core volume is `Ae*le`; CM copper has two windings. Target inductance determines effective
permeability `L*le/(mu0*N^2*Ae)`; DM uses a gap-dominated equivalent, `gap=mu0*N^2*Ae/L`.
The geometry is an equivalent-volume model, not a manufacturable core drawing or loss guarantee.
Capacitor mass is `1 g + 2*(C*630^2/2)/(0.2 J/cm^3)*1.2 g/cm^3` each (one X and two Y),
with 630 V rating. Magnetic/copper assembly and these capacitors define filter mass only.

Feasibility requires every corner's research-mask margin >= 6 dB (3 dB numerical reserve plus
3 dB explicitly assumed model reserve), device |Vds| <= 960 V, |Id| <= 50 A, capacitor
|V| <= 504 V, winding RMS current <= 12 A, copper+capacitor ESR+damping loss <= 25 W, and calculated
DM/CM peak flux <= 0.20 T. DM flux is `L*i/(N*Ae)`; each CM winding's flux linkage is
`L*(ia+k*ib)/(N*Ae)` (both windings checked). The flux constraint only gates use of the
linear approximation; it does not simulate saturation. Fixed-temperature semiconductors and no
core-loss model exclude total efficiency, thermal feasibility and full stability certification.
Periodic waveform settling below is the bounded stability screen. Failure of a physical or
research-mask constraint is predicted infeasibility, not solver failure. Unknown hardware/model
uncertainty can exceed the reserve; no aviation or laboratory compliance is inferred.

### Measurements, signal processing and accuracy gates

Current sensors at the filter-to-harness ports are positive from inverter toward load.
`Icm=(Ia+Ib)/2`, `Idm=(Ia-Ib)/2`, both amperes; chassis return is separately saved. Conductor
spectra, CM and DM are all required. `Vcm=(Va+Vb)/2` and `Vdm=Va-Vb`, chassis-referenced volts.
Never use sum-current normalization under an average-current label. Save switch voltages/currents,
filter winding currents, capacitor voltages/currents and load ports for stress and loss metrics.

Every study integrates 0..200 us from its DC operating point. Discard 0..100 us; observe
[100,200) us (five switching periods). Compare last two periods on the output-current grid:
RMS difference <= 0.01 A + 1% of RMS magnitude for each conductor. Otherwise `unsettled`, no
feasibility/ranking. Qualification repeats **every candidate/corner** at maximum integration
steps 2.5, 1.25 and 0.625 ns for light/medium and 0.625, 0.3125 and 0.15625 ns
for heavy. All use output-grid spacings 5, 2.5 and 1.25 ns. Double pulse repeats at
2, 1 and 0.5 ns max step, sampled at 1, 0.5 and 0.25 ns for metric comparison. Production study
measurements use the finest qualified level; they are external ngspice references, not production
Ohmnivore execution.

Adaptive output must start at zero, reach the exact stop (1e-15 s tolerance), be finite, strictly
increasing, have correct schema/point count and gaps no larger than the requested maximum step
plus 1e-15 s. Linear interpolation only between bracketing points; no extrapolation. The output
sampling frequency at the finest level is 800 MHz, Nyquist 400 MHz; spectra are restricted to
150 kHz..10 MHz. No decimation is used. Piecewise-linear reconstruction, oversampling and measured
sampling refinement establish the bounded spectral claim; there is no claim of an ideal analog
anti-alias filter or fidelity above 10 MHz. Separate same-waveform output-grid refinement from
integration refinement so two errors cannot be conflated.

Remove arithmetic mean, use a periodic Hann window `w[n]=(1-cos(2*pi*n/N))/2`. With an
unnormalized forward DFT, report positive-frequency sinusoidal RMS amplitude
`sqrt(2)*abs(DFT(w*x))/sum(w)` A and `20*log10(max(A,1e-15)/1e-6)` dBuA.
The 100 us observation yields a 10 kHz grid and Hann ENBW 15 kHz. Detector is the maximum
single-bin RMS-equivalent tone amplitude across the stated grid; there is no quasi-peak detector,
standard receiver bandwidth or arbitrary zero padding. The **research mask** is a constant
90 dBuA for all four current observables on this band, with the stated 6 dB reserve. Retain every
bin, not only the worst margin.

For each adjacent integration refinement and finest-waveform output sampling refinement require:
conductor RMS waveform difference <= 0.02 A + 2% of reference RMS; switch Vds RMS difference
<= 2 V + 2% of bus V; spectral-bin difference <= 1 dB wherever either amplitude is above
20 dBuA, absolute amplitude difference <= 1 microampere below that floor. DPT 10--90% Vds edge
durations and 50% times differ <= 2 ns + 10% of the corresponding reference edge duration;
peak Vds differs <= 2 V + 2%; signed terminal
switching energy (integral Vds*Id over fixed ±0.2 us windows around second turn-on/turn-off)
differs <= 2 uJ + 5%; ringing frequency differs <= 10%, damping/log-decrement <= 20% when
three same-polarity peaks are measurable above 0.2 V. Inspect both residual polarities relative
to the nominal 400 V bus in (14.04,14.4) us; the first three peaks of a polarity must strictly
decrease in amplitude. If both qualify, select the polarity with the earlier first peak. Otherwise
ringing telemetry is explicitly unresolved and this frozen DPT fixture fails qualification. DPT metrics use the declared uniform
output grid, with the 16 us endpoint included for interpolation and integration. DPT must exhibit
finite edges, loaded second turn-on >= 10 A, 0 < switching energy < 2 mJ, Vds peak between bus and
960 V, and observable gate/drain coupling: second-turn-on external gate voltage must decrease by
0.2..30 V between the 90% and 10% Vds crossings, and gate voltage at both second-event 50% Vds
crossings must lie in [-3,23] V. Retain gate samples at all crossings and full gate extrema.
These pre-retained observability gates recognize Miller/package dynamics; they do not claim a
flat plateau or measured hardware agreement.
Independent tests use analytic CM/DM, DC, bin-centred sine, phase, mixed-tone and Parseval cases.

### Failure, resource, timing and evidence contracts

Use `reltol=1e-5`, `abstol=1e-9 A`, `vntol=1e-7 V`, `chgtol=1e-16 C`, Gear order 2,
`itl1=300`, `itl4=100`. No automatic retry or fallback (retry limit zero). Each simulator child
has 120 s wall timeout, 110 s CPU limit, 1 GiB address-space limit, 512 MiB output-file limit.
Maximum accepted raw points 2,000,000; the parser applies its size budget before allocation.
Typed terminal failures: `unsupported_input`, `provenance_mismatch`, `simulator_failure`,
`numerical_failure`, `timeout`, `resource_limit`, `missing_output`, `malformed_output`,
`non_finite`, `unsettled`, `accuracy_failure`, `internal_failure`. Nonzero exit, logged simulation
error, truncation, malformed header, missing/duplicate vectors and partial end time all fail.
A complete numerical result alone can be `predicted_feasible` or `predicted_infeasible`.
Missing/failed corners exclude the entire candidate from feasible ranking.

Freeze **one warmup study plus three measured studies per mode**, serial (one worker) and bounded
parallel (four workers), per invocation; two independent invocations. Each study runs all nine
ordered jobs once. Warmups also retain terminal accounting. Qualification consists of the three
DPT refinements and all 27 ensemble refinements, per invocation. Thus each invocation expects
102 simulation jobs: 30 qualification + 72 warmup/measured. These practical counts estimate CPU
runtime; three study samples do not support population-tail or GPU speedup claims. Job P95 is
nearest rank, explicitly empirical. CPU affinity and hardware are recorded; no competing build
or benchmark should run during retained measurements.

Complete study wall time includes materialization, scheduling, process launch/simulator parsing
and initialization, solving, raw validation, spectral processing, compression/content hashes,
and durable required job outputs. Report job phase times, queue wait, completion latency,
validated throughput, median/P95, every failure, child peak RSS and runner RSS. Charge worker-pool
creation and terminal-summary output to study totals. Whole invocation additionally records setup,
qualification, warmups and summaries. File close completion is the I/O boundary; physical disk
flush and Bazel fetch/build are outside and explicitly labeled. Serial and parallel use identical
jobs, processing and output requirements. Store raw header plus losslessly compressed raw binary
payload in a SHA-256 content-addressed store; this preserves exact original raw bytes while
allowing repeat payload deduplication. Store spectra, deck, log and completion links per job.
An independent audit recomputes hashes, identities, counts, statuses and the terminal record.

Use ngspice's measured accepted/rejected points, total/transient iterations, netlist loading/
expansion/parsing, matrix-load, reorder, factor/solve and truncation times. Matrix load combines
device evaluation and assembly; separate costs are **unavailable**, not estimated. ngspice uses
its own Sparse solver in this reference; Ohmnivore's production KLU is unchanged. Profile-derived
acceleration budgets and proposed EMI-02/03 gates are a later documentation result of EMI-01,
frozen before any GPU implementation. No old AC speed threshold transfers to this experiment.

### Pre-evidence fixture correction

The first exploratory full-network run failed the unchanged settling gate: the undamped CM
resonance persisted into the observation interval. Before retaining qualification, each Y branch
receives 21.8 ohm explicit series damping in addition to 0.2 ohm ESR. Its dissipation is included
in the same 25 W limit. Mass adds two assumed 5 g / 15 W pulse-rated resistors; each dissipates at
most 12 W in the finite-domain RMS check. This is a documented engineering fixture revision, not
a relaxation of the settling or convergence criteria. Exploratory run-1 is not retained evidence.


### Final pre-evidence physical and numerical revision

Exploratory runs exposed a nearly lossless differential load-terminal mode near 5.14 MHz:
the two coupled harness windings have differential series inductance 1.6 uH and the two
1.2 nF terminal-to-chassis capacitors have differential capacitance 0.6 nF. The ideal 100 uH
load inductor isolates the series 20 ohms at that frequency. Its resonant spectral bins did
not converge at the original 10/5/2.5 ns steps, despite passing waveform RMS checks. Tightening
RELTOL alone did not resolve this; smaller absolute current tolerances caused timestep failure.
Neither those trials nor a proposed higher spectral floor is accepted as qualification.

Freeze the explicitly hypothetical passive load as
`Zload(s)=20 ohm + (s*100 uH || 100 ohm)`, retaining the declared harness/chassis capacitances.
The added 100-ohm winding-loss surrogate contributes about `8.98+j28.59 ohm` at 50 kHz and
approaches 100 ohms at high frequency. This deliberately changes load current and power; it
represents a bounded lossy load, not a measured motor or a fitted device model. Retain its RMS
power, the 20-ohm load power, and load-inductor current. Load dissipation is excluded from filter
loss/mass; no motor thermal claim follows. No production semantics are added.

In both fixtures connect the 10 uF DC-link capacitor to `p1`, after the 20-milliohm supply
resistance and **before** the 20 nH inductance. This places the declared inductance in the actual
commutation path between the bulk capacitor and bridge. The prior capacitor connection at `p`
bypassed that inductance at switching frequencies and under-excited DPT ringing. Keep a 4.7-ohm
external resistor on the held-off upper DPT gate as well as the pulsed lower gate. Repeated DPT
qualification must now resolve three strictly decaying peaks under the existing ringing rule;
unresolved ringing is an `accuracy_failure` for this frozen fixture.

Use the original declared solver options, unchanged 20 dBuA / 1 microampere spectral gate,
and the finer ensemble integration levels **2.5, 1.25, 0.625 ns**. Output sampling remains
5, 2.5, 1.25 ns; it is separately refined on the same finest integration waveform. DPT levels,
mask, physical constraints, finite identities, resource limits and evidence sample counts remain
as declared. These choices and all resulting source hashes precede both retained invocations.


The final all-corner exploratory audit isolated remaining integration sensitivity to the heavy
candidate's 6.5--7.5 MHz common-mode resonance. Freeze a candidate-specific integration policy:
`heavy` uses **0.625, 0.3125, 0.15625 ns**; `light` and `medium` retain **2.5, 1.25, 0.625 ns**.
This finite manifest override applies to every corner, and measured serial/parallel studies use
each candidate's finest qualified step. There is no data-dependent retry or dynamic policy.
All candidates keep the same 5/2.5/1.25 ns output grids and unchanged spectral acceptance gates.
The raw-file/parser limit is **512 MiB**, needed because heavy's finest uniform step alone
requires at least 1.28 million points with 27 saved vectors (about 277 MB). The two-million-point,
1 GiB oracle address-space, 110 s CPU and 120 s wall limits remain unchanged. This is a declared
accuracy/resource tradeoff, not a speedup or floor relaxation.

### EMI-01 follow-up: passing and boundary reference design protocol

The 2026-09-17 follow-up authorizes strengthening the external reference and merging its PR
after validation. The original three-candidate evidence remains historical and byte-preserved.
No EMI-02 production implementation is authorized.

Before selecting the additional finite cases, explicitly labeled exploratory simulations may
vary only filter inductance, capacitance, winding/core geometry and passive filter damping.
Keep the same SiC model, full coupled inverter/load/harness/chassis, three operating corners,
initialization, 200 us interval, current ports, spectral processing, 90 dBuA research mask,
6 dB reserve and all physical limits. No reduced bus, disconnected load, altered measurement
port, relaxed emission/stress threshold or ideal-switch substitute may create a pass.
Explore a bounded set of engineering sensitivity cases, retaining the attempted parameter
table and diagnostic outcomes separately from final qualification/performance evidence.

The retained follow-up must freeze a versioned exact candidate set before measurements, include
at least one candidate passing every corner and a distinct boundary candidate whose worst
research-mask margin lies in [5,7] dB while passing every physical screen. A failing-side boundary
is preferred to exercise rejection alongside acceptance. Both must satisfy unchanged numerical
qualification and settling gates; near-boundary means numerical predicted classification under
this assumed research model, not a claim robust to unmeasured hardware variation. Preserve a
clearly failing original case in the finite study. Record mass under the same physical design
model, with any changed damping resistance and mass accounted explicitly. Freeze exact component
values, identities, integration refinements and counts in a further contract entry before the
two complete retained follow-up invocations. Original results must not be relabeled as current.

#### Initial exact v2 contract, before harness extension and retained evidence

The explicit selector is `--reference-version=emi01-v2`; the default remains `emi01-v1`.
The old `manifest.json` bytes stay unchanged; `manifest-v2.json` names this finite replacement
study. Both source manifests are hash-bound. Every new terminal/metadata record, deck title,
and performance report identifies its selected version. Unknown versions, mixed-version report
pairs or noncanonical manifests fail closed. Historical v1 evidence is audited from detached
commit `c621e5cfcedf2de934c8046595dbb9f7aa59e607` against absolute evidence paths, preserving its
strict source-identity check; running new source code must not claim that old hashes still match.

The ordered candidates are `light` (unchanged v1 failing control), `boundary`, and `reference`.
The latter two share per-conductor DM inductance 330 uH, per-winding CM inductance 1 mH,
Y capacitance 47 nF, DM geometry `[1200,160,48,135]`, and CM geometry `[800,140,40,110]` in
the same `(Ae mm^2, le mm, turns, mean turn length mm)` convention. X capacitance is initially
85 nF for `boundary` and 1 uF for `reference`. Geometry changes feed the same mass, copper
resistance and flux equations; they do not establish manufacturability or core-loss behavior.
Y damping remains exactly 21.8 ohm plus 0.2 ohm ESR, with the original mass/rating and loss
accounting. Every other circuit element, source, model and corner is unchanged.

Use integration maxima 0.625/0.3125/0.15625 ns for both new cases and the original
2.5/1.25/0.625 ns for `light`. All output grids, spectral and waveform tolerances, DPT settings,
settling intervals, physical limits and resource budgets remain unchanged. Qualification is
three DPT plus 27 full-network refinement solves and 40 checks. Each full invocation then runs
one warmup plus three measured nine-job studies with one and four workers: 102 records total.
Retain two complete independent invocations. Exploratory design/refinement runs and their failed
gates are separate diagnostics; they cannot substitute for either retained invocation.

In addition to numerical qualification, final v2 acceptance requires `reference` to pass all
corners and `boundary` to pass every physical screen, with minimum research margin in inclusive
[5,7] dB over every corner, four current observables and 986 frequency bins. The predicted
feasibility threshold remains exactly >=6 dB. Report the worst corner, observable and frequency,
distance from 6 dB, margins at all refinements, and any classification changes. Prefer a boundary
below 6 dB with stable refinement classification; do not infer robustness from proximity alone.
Any necessary pre-evidence parameter/refinement revision must be recorded explicitly here before
retained measurements. No threshold or model changes may substitute for a valid result.

Rank only complete qualified candidates by computed mass, with candidate identity breaking an
exact mass tie; execution order remains the frozen manifest order. Failures or missing corners
exclude a candidate. A private read/execute-only snapshot of the checksum-verified canonical
ngspice executable is made inside invocation preparation, rehashed and used by all its workers.
Charge that preparation to the full invocation; record the actual binary identity. Do not rebuild
or run competing studies during either retained performance invocation. New budgets describe only
v2; v1 results remain tied to their original inputs and executable.

The pre-evidence 85 nF boundary probe gave 5.9838 dB in the fast corner at 0.625 ns, only
0.0162 dB below the 6 dB threshold. Freeze **80 nF** instead to target a failing-side boundary
with a clearer numerical separation, retaining 330 uH and every other declared value. This is
an explicit capacitor design change, not a moved mask or relaxed acceptance gate. Its exact
all-corner margins and classification at all refinements must be recorded before acceptance.

The enlarged geometry keeps the original assumed 10 pF per-CM-winding capacitance and 0.995
coupling; these parasitics are not derived from winding geometry. DM self-capacitance is omitted.
The equivalent DM gap is about 10.53 mm, with no fringing or winding-window/fill validation.
These limits remain explicit: a v2 pass is a numerically qualified result under the declared
hypothetical model and screens, not an optimized or manufacturable 3.8 kg hardware design.

Independent pre-evidence review clarified that v2 fixture-role acceptance must also enforce its
failing control: `light` remains numerically valid and predicted infeasible at **every** original
corner, as in v1. A missing/failed simulation cannot satisfy this role. An unexpected passing
control fails fixture coverage without rewriting its computed result or numerical status.
