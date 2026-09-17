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
