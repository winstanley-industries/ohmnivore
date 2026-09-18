# EMI-02 CPU implementation and qualification

EMI-02 implements and qualifies all four CPU slices on top of the merged EMI-01
reference (`82b7a6e`). The fresh [run-3](evidence/emi02/run-3/qualification.json)
passes every frozen CPU/reference, integration-refinement and observation-grid
gate. All sixty simulator jobs complete within the original resource limits.
The [independent artifact audit](evidence/emi02/audit-v3.json) also passes.

## Delivered scope

| Slice | Implementation | Independent checks |
|---|---|---|
| A | Disjoint signed mutual-inductor pairs in native DC, AC and transient MNA | Exact two-winding modes, energy, sign reversal, hostile inputs and ngspice fixtures |
| B | Bounded expressions, parameter binding and sparse analytic derivatives | Independent values/derivatives, lazy branches, domain and budget failures, pinned ngspice expression comparisons |
| C | Simultaneous E/G/B residuals and Jacobians, native reactive state, transactional BE/TRAP integration | Charge primitives, scalar physical equations, RK4 package dynamics, step-doubling/controller reconstruction and rejected-step isolation |
| D | Exact pinned model import, explicit source initialization, bounded runner and complete CPU/reference qualification harness | Model provenance and import failures, original-versus-flattened DC, CPU DC, standalone signed charge/current refinement, process/publication and evidence-audit failures |

The ordinary parser gains only coupled-inductor syntax. Behavioral sources and
the selected model use an explicit experimental API/runner. CPU FP64 SuiteSparse
KLU remains authoritative. Rust, CUDA implementations, old reference waveforms
and existing prepared-AC evidence are unchanged.

## Numerical review

Independent review exposed companion-matrix cancellation and convergence checks
that could hide small physical changes behind large common-mode voltages.
Behavioral Newton now checks capacitor voltage differences and inductor currents,
forms its affine RHS without subtracting a rounded linear residual, and always
performs a correction before accepting an initial guess. Bounded original-matrix
iterative refinement supplies an additional correction even when the ordinary
backward-error guards already pass. Ordinary solver policy and validation
thresholds are preserved.

Behavioral LTE uses capacitor voltage differences and inductor currents. The
model runner explicitly selects an audited derivative-history estimator for
TRAP; BE startup and recovery retain actual step doubling. Full/two-half audits
run before history is available, periodically, at guarded transitions and during
fallback. Sixteen consecutive accepted audit agreements are required to recover
from fallback. The public default retains step doubling, and native transient
semantics remain unchanged. Independent physical and forcing oracles check the
history formula, unequal timesteps, controller decisions and rejected-state
isolation.

Invocation-owned preparation reuses immutable sparse structure and bounded
assembly storage. Exact expression reuse preserves state/dependency identity,
full trial derivative checks and a fresh final original-system value/residual
check. The private sparse fast path reuses already-established input admission
and accepts an all-row backward-error certificate only where its conservative
bound proves the original guards pass; other cases use the complete checked
path. KLU factorization/rank checks and mandatory correction remain in place.
No floating-point compiler flags, physical tolerances or external gates changed.
Matched development binaries preserved output bits and work counts for these
storage/evaluation optimizations; the full study below qualifies the delivered
numerical policy.

See the [initial dynamics review](evidence/emi02/INDEPENDENT_DYNAMICS_REVIEW.md),
[resumed implementation review](evidence/emi02/INDEPENDENT_RESUMED_REVIEW.md), and
[ADR-003](adr/ADR-003-emi02-cpu-qualification.md),
[ADR-004](adr/ADR-004-emi02-behavioral-expressions.md),
[ADR-005](adr/ADR-005-emi02-behavioral-transient.md) and
[ADR-006](adr/ADR-006-emi02-model-import-qualification.md).

## Generic transient regression coverage

The [21-circuit corpus](transient-regression-corpus.md) covers bipolar RC, lightly
damped/underdamped/stiff RLC and nonlinear charge dynamics, with amplitude, time
scale and common-mode variation. Both error estimators pass all 42 fine-resolution
analytic lanes. Another 42 coarse lanes compare prepared execution with the fully
checked path bit for bit, including states, decisions and sparse-solver counters.
The coarse analytic stress lanes retain 26 global-error failures already present
in the baseline; they are not presented as validated performance cases.

An endpoint-scale experiment reduced steps modestly but increased accumulated RLC
error, so it was rejected. A companion scale cache was also rejected after a 2.6%
corpus slowdown. The retained change prepares and updates only C-bearing companion
entries, preserving immutable G-only entries, original arithmetic, finite checks,
failure recovery and all timestep/error policies. It uses no additional cache or
per-call bookkeeping. [ADR-007](adr/ADR-007-generic-transient-improvements.md)
defines the invariant and acceptance requirements.

The [matched benchmark evidence](evidence/transient-generic/v1/README.md) records
identical numerical/work fields on every lane and 18 timed repetitions per lane.
Fine-lane geometric-mean runtime is 0.3% slower; the worst median is 2.6% slower
with overlapping observed ranges. This supports no runtime speedup claim. The
smaller update plan preserves the qualified behavior, while the broader corpus
provides regression gates for future numerical changes. The light-filter step-count
gap remains unresolved. The rejected patches and all screening failures are retained.

## Model interpretation and limitations

The pinned ngspice 46 `ps` frontend reserves `vt`, replacing the selected model's
local token with ambient thermal voltage. The importer deliberately reproduces
the actual historical reference interpretation, including at both fixed junction
temperatures. This is not validation of the vendor-intended local threshold
equation; repairing the collision requires a different reference and new evidence.
Original and flattened model equations remain temporary and are not published.

The model's omissions include bipolar reverse recovery, self-heating, avalanche
qualification and hardware correlation. No intrinsic switching-loss, laboratory
EMI, certification, GPU speedup or dispatch claim follows from this CPU port.

## Canonical validation

All thirteen required command stages passed on the final implementation. Default
and lockfile-mode suites each passed all 38 targets. ASan and UBSan each passed
all 33 compatible targets, with five declared exclusions. Lint, the complete
build, CUDA smoke, explicit ngspice/expression acceptance, prepared-AC tests and
the prescribed replay benchmark passed. Direct CUDA-plus-ASan and CUDA-plus-UBSan
targets failed during analysis as required by their incompatibility declarations.
`git diff --check` passed.

Exact commands, exit codes and durations are in
[validation-v3/results.json](evidence/emi02/validation-v3/results.json), with
adjacent logs. These are correctness and build checks, not a new GPU performance
claim. The earlier validation records remain historical evidence for run-1 and run-2.

## Passing complete qualification

The invocation covers three DPT refinements and three refinements of every one
of the nine candidate/corner circuits, on both CPU KLU and independent ngspice.
Each simulator process retains the frozen 110 s CPU, 120 s wall, 1 GiB address
space and 512 MiB output limits. Four spawned workers execute all sixty jobs once;
no selective retry repairs an existing record. Every trajectory also satisfies
the two-million-point cap. Candidate identities, model/archive bytes, measurement
rules, physical tolerances and spectral/refinement gates are unchanged.

The [complete retained invocation](evidence/emi02/run-3/qualification.json)
reports `complete=true`, `pass=true`, and `gpu_authorized=false`.

| Gate | Result |
|---|---|
| CPU DPT, three integration levels | All three complete and pass external differential comparisons |
| CPU inverter, nine candidate/corner jobs at three levels | All 27 complete within the frozen resource limits |
| External ngspice trajectories | All 30 complete |
| CPU integration/observation refinement | All 40 frozen checks pass |
| External integration/observation refinement | All 40 frozen checks pass |
| Full CPU/reference differential and classification agreement | All 30 comparisons pass |
| Whole-candidate feasible ranking | Only `reference` is predicted feasible across all three corners; `light` and `boundary` are not |

Each engine reports three qualified DPT results, fifteen individually
predicted-feasible inverter results and twelve individually predicted-infeasible
inverter results. Predicted EMI infeasibility is a valid simulated result, not a
solver failure. The ranking applies to the frozen research mask and model only.

CPU inverter process wall times span 32.424–108.215 s. The recorded host is an
AMD Ryzen 9 9950X3D under WSL2, with the invocation restricted to logical CPUs
4, 6, 20 and 22. Exact command, CPU topology, affinity, temporary directory, exit
code and elapsed time are retained in
[run-3-execution.json](evidence/emi02/run-3-execution.json). This concurrent
qualification is not a fair persistent parallel CPU performance baseline or a
portable throughput claim. Source/model/binary identities are in
[invocation.json](evidence/emi02/run-3/invocation.json).

The [independent artifact audit](evidence/emi02/audit-v3.json) exits zero and
reports `audit_pass=true` and `qualification_pass=true`. It recomputes measurements,
spectra, refinement, all-level differential checks and ranking from retained
numeric artifacts while checking source, model, binary, job and file identities.
Its [execution record](evidence/emi02/audit-v3-execution.json) and adjacent log
retain the exact command, environment and result. The current follow-up has a
[local review](evidence/emi02/GENERIC_TRANSIENT_REVIEW.md) without subagents;
the independent reviews linked above describe the earlier implementation. [Run-3 parity](evidence/emi02/run-3-parity.json) checks
all thirty CPU trajectories, measurements, classifications and numerical work
counters against run-2: every compared field and waveform hash is identical.

## Historical invocations

[Run-1](evidence/emi02/run-1/qualification.json) remains unchanged. It accounts for
all required jobs but reports `pass=false`: its three CPU DPT jobs succeeded,
while all 27 CPU inverter jobs hit the CPU limit. The original
[audit](evidence/emi02/audit.json) confirms that failed invocation's integrity;
it does not qualify the resumed source.

[Run-2](evidence/emi02/run-2/qualification.json) remains the passing historical
qualification before the generic companion improvement. Run-3 is a new complete
invocation with fresh CPU and ngspice execution, its own fingerprints and every
mandatory gate.

EMI-02D is qualified for this bounded CPU workload. EMI-03 has not started; GPU
transient execution, speedup claims and automatic dispatch remain unauthorized.
