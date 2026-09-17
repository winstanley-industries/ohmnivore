# EMI-02 CPU implementation and qualification

EMI-02 implements all four CPU slices on top of the merged EMI-01 reference
(`82b7a6e`). **EMI-02D is not qualified:** all 27 complete CPU inverter runs
reached the frozen resource limit. The double-pulse comparisons and canonical
checks pass, but full inverter accuracy and CPU resource qualification remain open.

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

Behavioral LTE uses complete physical reactive coordinates. BE restarts at source
corners; TRAP uses one full step and two private half steps with the corresponding
second-order error estimate. Independent tests reconstruct these equations and
controller decisions. See the [independent dynamics review](evidence/emi02/INDEPENDENT_DYNAMICS_REVIEW.md)
and governing [ADR-003](adr/ADR-003-emi02-cpu-qualification.md),
[ADR-004](adr/ADR-004-emi02-behavioral-expressions.md),
[ADR-005](adr/ADR-005-emi02-behavioral-transient.md) and
[ADR-006](adr/ADR-006-emi02-model-import-qualification.md).

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
and lockfile-mode suites each passed all 33 targets. ASan and UBSan each passed
all 28 compatible targets, with five declared exclusions. Lint, the complete
build, CUDA smoke, explicit ngspice/expression acceptance, prepared-AC tests and
the prescribed replay benchmark passed. Direct CUDA-plus-ASan and CUDA-plus-UBSan
targets failed during analysis as required by their incompatibility declarations.
`git diff --check` passed.

Exact commands, exit codes and durations are in
[validation/results.json](evidence/emi02/validation/results.json), with adjacent
logs. These are correctness and build checks, not a new GPU performance claim.

## Retained qualification

The invocation covers three DPT refinements and three refinements of every one
of the nine candidate/corner circuits, on both CPU KLU and independent ngspice.
Each simulator process retains the frozen 110 s CPU, 120 s wall, 1 GiB address
space and 512 MiB output limits. Four spawned workers execute the sixty jobs;
this concurrent qualification run is not a fair CPU performance baseline.
Every mandatory failure remains in the result, and one failed corner suppresses
whole-candidate feasible ranking. The audit independently recomputes measurements,
refinement, differential checks and ranking from retained numeric artifacts while
checking source, model, binary, job and file identities.

The [complete retained invocation](evidence/emi02/run-1/qualification.json)
accounts for every job and reports `complete=true`, `pass=false`, and
`gpu_authorized=false`. Here, complete means the required job accounting is
complete; it does not mean failed CPU jobs produced full trajectories.

| Gate | Result |
|---|---|
| CPU DPT, three integration levels | All three complete and pass the external differential comparisons |
| CPU DPT refinement | Both adjacent integration comparisons and both observation-grid checks pass |
| CPU inverter, nine candidate/corner jobs at three levels | All 27 hit the 110 s CPU limit; process exit is `-9`, with measured wall times 110.024–110.045 s |
| External ngspice trajectories | All 30 complete: three qualified DPT, 15 individually predicted-feasible and 12 individually predicted-infeasible inverter results |
| External integration/observation refinement | All 40 frozen checks pass |
| Complete CPU/reference differential | Fails: three DPT comparisons pass; 27 comparisons lack successful CPU trajectories |
| Whole-candidate feasible ranking | Suppressed by failed qualification; CPU resource failures are not predicted EMI infeasibility |

The qualification command exits 1, correctly recording a failed gate. Exact
command and elapsed time are in [execution.json](evidence/emi02/execution.json).
Source/model/binary identities are in [invocation.json](evidence/emi02/run-1/invocation.json).
The [independent artifact audit](evidence/emi02/audit.json) exits 0 and reports
`audit_pass=true`, `qualification_pass=false`, with all 30 jobs per engine.
This verifies the integrity of the failed study, not successful qualification.
A separate review reconstructed the mandatory identities and integration/sample
settings from the manifest, checked one attempt per job, failure/ranking accounting,
source coverage and publication privacy, and found no remaining publication blocker.

The remaining acceptance work is to complete every CPU inverter trajectory within
the unchanged bounds and pass all frozen full-waveform/refinement comparisons.
The passing short fixtures do not establish that result. EMI-03 remains closed.
