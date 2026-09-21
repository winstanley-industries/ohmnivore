# EMI-03 transient ensemble experiment

**EMI-03 is incomplete.** Implementation tests and audits of failed experiments
do not satisfy acceptance. Completion requires passing all frozen accuracy,
resource, zero-failure, median/P95 speedup and cold-time gates in both independent
invocations. The draft PR remains work in progress.

Both complete invocations are **negative at the resource gate**. Their fresh
CPU/ngspice references pass, but the frozen four-worker CUDA configuration
exceeds the aggregate device-residency budget during startup. No CUDA numerical
result is accepted, and no performance observation or speedup is published.

Follow-up development replaces oversized expression-kernel thread-local arrays
with explicitly accounted device scratch sized to the actual programs. A bounded
four-worker DPT diagnostic observed 1,089 MiB incremental device residency, but
the workers then exhausted the per-job CPU-time budget. This is an intermediate
diagnostic, not a passing resource/qualification result. A shortened DPT diagnostic
attributes most execution time to the repeated GPU factor/solve calls and
expression transfers/synchronization. The complete-study targets remain unmet;
neither diagnostic qualifies the changed implementation. The
[diagnostic records and exact source patch](evidence/emi03/diagnostics/scratch-workspace/README.md)
retain this development step separately from the two original invocations.

This opt-in experiment implements [ADR-008](adr/ADR-008-emi03-transient-ensembles.md)
against the qualified EMI-02 CPU baseline at `734a9da`. CPU FP64/KLU remains the
correctness authority and the supported no-GPU implementation. The experiment
does not enable dispatch, alter the frozen research mask, repair the historical
vendor `vt` binding, or start EMI-04.

The separate CUDA worker batches native-FP64 behavioral expression evaluation and
analytic derivatives, and uses cuDSS for real sparse factorization, solving and
bounded refinement. The qualified host assembly, Newton and adaptive integration
controller remain in use. Four persistent GPU workers own independent complete
circuits and fresh numerical state. The comparator supports persistent CPU
workers at 1, 4 and 16 physical cores. Every invocation creates fresh CPU and
ngspice qualification trajectories before any replay timing can be admitted.

## Diagnostic basis

[The initial complete-job profile](evidence/emi03/profiles/initial/profile.json)
and [the nested expression profile](evidence/emi03/profiles/expressions/profile.json)
retain separate ordinary/profile executions, raw output and exact parity checks.
These are historical diagnostic snapshots taken before the final implementation;
their identities are preserved and are not relabeled as final GPU evidence.

The first `q0/reference/nominal` run attributes 49.49% of instrumented runner time
to assembly/evaluation and 33.98% to the complete real factor/solve pipeline. Even
eliminating the latter yields only a 1.515x counterfactual ceiling before transfers
or certification. This selected a combined expression/linear candidate instead
of a linear-only candidate. The nested expression clocks add approximately 4.62 s
of wall time, so their reported phase shares are not an uninstrumented speedup
bound. Neither profile establishes GPU acceleration.

## Resident follow-up candidate

A separate `//cuda:emi03_resident_worker` keeps integration, Newton iterations,
expression evaluation, sparse numeric factorization, refinement and accepted-state
accounting on the device. Private job state and bounded output chunks retain the
complete circuit. A barrier protects each output row before its shared index
advances; a wide-circuit regression covers this boundary. Exact Jacobian reuse is
confined to a private chunk and never bypasses validation of a changed matrix.

Pinned transfer staging and completion queries with a 250 us sleep avoid the
observed CPU busy wait. A fresh full reference/nominal q0 probe uses only 3.751 s
of child CPU time but still reaches the 120 s wall limit. Its fresh CPU reference
finishes in 33.279 s. A separate full frozen q0 DPT completes and passes CPU
waveform/switching checks, taking 12.783 s on GPU versus 0.597 s on CPU.

The [resident snapshot](evidence/emi03/diagnostics/resident/README.md) retains exact
source/binary identities, raw outputs, tests, scripts and terminal failure. Seven
resident tests and the existing 20 CUDA cases pass. This is development evidence,
not the complete thirty-job qualification or a passing throughput result. Device
execution time remains an acceptance blocker.

A later [complete resident trajectory](evidence/emi03/diagnostics/resident-full-reference/README.md)
passes the full q0 reference/nominal CPU waveform, spectral and physical-classification
comparison. It takes 348.195 s on GPU versus 33.010 s on CPU and remains a
`resource_limit` result. An extended diagnostic ceiling allowed completion without
changing the frozen 120 s gate. Source reconstruction, exact raw outputs, all
spectral arrays and metrics reproduce in its audit. The GPU emits 483,937 points
versus 461,672 for CPU; most additional points arise early in the trajectory.

The [exact-state expression-cache checkpoint](evidence/emi03/diagnostics/resident-expression-cache/README.md)
passes eight resident tests, the existing 20 CUDA cases, the exact-reduction test
and all eighteen canonical validation stages. It preserves the fresh accepted-state
value check. Its full frozen q0 DPT passes the complete waveform and switching
comparison at 9.624 s GPU versus 0.618 s CPU, with source and raw reconstruction
verified by the retained audit. The complete reference trajectory above predates
this cache and does not qualify the changed implementation. Full acceptance
remains outstanding.

The [three-level DPT checkpoint](evidence/emi03/diagnostics/resident-dpt-refinements/README.md)
adds ordered row/source lists, compact shared scratch, optional shared CSR indices
and warp-aligned expression operators. All three frozen DPT CPU/GPU comparisons
and all eight DPT integration/output-refinement checks pass with fresh CPU runs.
GPU q0/q1/q2 wall times are 9.686 / 12.190 / 19.602 s, versus 0.667 / 0.817 /
1.418 s CPU. Ten resident tests and all eighteen canonical validation stages pass.
The audit reconstructs the exact sources and
all six raw trajectories and reproduces every DPT metric and comparison.
This checkpoint has no fresh coupled/ngspice qualification, aggregate resource
pass or passing throughput result.

The [Jacobian-reuse candidate](evidence/emi03/diagnostics/resident-jacobian-reuse/README.md)
reuses validated factors only within bounded Newton iterations and verifies the
actual analytic Jacobian at acceptance. Its complete q0 reference/nominal physical
comparison passes, but GPU wall time is 286.349 s versus 32.947 s CPU. The GPU
result remains `resource_limit` against the unchanged 120 s gate. Formatting and
rebuilding reproduce the exact tested binary; both source snapshots are retained.
Fresh DPT q0/q1/q2 comparisons and all eight refinement checks pass at
8.583 / 11.639 / 14.544 s GPU versus 0.617 / 0.817 / 1.418 s CPU.
All eighteen canonical checks pass. This does not establish full qualification
or throughput.

The [software profiling diagnostics](evidence/emi03/diagnostics/resident-profiling/README.md)
use a newer checksum-pinned Nsight Systems without hardware counters. They locate
single-job cost inside the resident kernel and expose interference between GPU
contexts and allocation owners. A separate one-process ownership prototype reduces
four shortened-job time from 8.449 s to 2.318 s after replacing legacy device
allocation and reusing private job staging. These diagnostic observations do not
qualify the scheduling change or satisfy any complete-study performance gate.

The [owner-pool and synchronization checkpoint](evidence/emi03/diagnostics/resident-owner-pool/README.md)
adds sixteen private owner threads in one process and retains memory-lifetime and
shared-control race fixes found by Compute Sanitizer. Targeted memcheck, racecheck,
synccheck and initcheck pass with their exact coverage recorded. Fresh DPT q0/q1/q2
comparisons and all eight refinements pass through the real pool at GPU request
wall times 8.526 / 11.669 / 14.492 s. The three-job diagnostic observes 300 MiB
incremental device residency and passes its resource audit. These results do not
qualify the full coupled workload or satisfy its performance targets. Historical
profiling prototype results remain identified separately from this corrected code.

The [hardware-counter investigation](evidence/emi03/diagnostics/resident-hardware-counters/README.md)
now captures the full Nsight Compute set successfully. Its source-level samples
locate waits around the triangular solve and factorization levels, with no
spilling or bandwidth bottleneck. Three smaller thread blocks, two symbolic
ordering policies and ordered operand prefetch were rejected. An initial AMD
prefix improvement reverses in a longer coupled run and in full DPT diagnostics.
The retained audit reconstructs six source variants, 56 trajectories, 46 GPU
waveform comparisons and 16 DPT refinement checks. None supplies a complete-study
acceptance pass; the original resource and performance targets remain in force.

Three further [factor-dependency and triangular-row trials](evidence/emi03/diagnostics/resident-factor-dependencies/README.md)
also fail to improve execution time. Reusing provably unchanged factor entries is
neutral; every factorization level still contains changing entries. Cooperative
long-row reductions are 5.9% slower on the shortened case. Thirty GPU waveform
comparisons and all thirty resident test executions pass across the three trials,
but all changes are removed. These remain diagnostic results.

The harness now explicitly requests [32 CUDA work queues](evidence/emi03/diagnostics/resident-work-queues/README.md).
Sixteen independent owners previously shared CUDA's default eight queues. On
sixteen shortened circuit jobs, increasing the queue count reduces ordinary
median execution-only batch time from 5.877 to 2.741 s; a reversed-order repeat
gives 5.142 versus 2.762 s. Separate hardware captures show increased SM activity.
All 320 GPU short-window waveform comparisons and six pool resource audits pass.
The actual child environment is integration-tested and invocation-audited. These
results select the queue setting for further qualification; they do not satisfy
the full thirty-job qualification or complete-study performance gates.

Four [correction-equation diagnostics](evidence/emi03/diagnostics/resident-correction-equations/README.md)
remain unselected. Direct correction solving fails the unchanged componentwise
linear certification on nearly zero rows. A bounded retry using the original
affine equation passes fresh DPT comparisons and refinements, but gives no
consistent timing benefit. Compensated companion history also fails to improve
the short diagnostic. All four source variants, 37 completed trajectories,
29 GPU waveform comparisons, eight DPT refinements and both terminal failures
reconstruct. These experiments do not qualify a replacement numerical policy.

The [explicit shared-address candidate](evidence/emi03/diagnostics/resident-shared-addresses/README.md)
keeps the sparse factor and triangular arithmetic and validation rules while
reducing generic address operations. Its short ordinary median is 1.963 versus
2.086 s; a triangular-only variant also improves a longer identical trajectory
from 30.436 to 28.677 s. Larger timing differences with different internal step
counts are kept separate. The selected factor/triangular variant passes all three
fresh DPT CPU comparisons and eight refinements at 7.634 / 10.097 / 12.324 s GPU.
Seven targeted Compute Sanitizer checks and nineteen canonical stages pass.
All nine distinct 20 us candidate/corner prefixes also pass fresh CPU comparisons
in two GPU batches. The warm request batch takes 46.313 s versus 7.017 s for the
fresh CPU batch. Boundary/nominal has nearly equal attempt counts but takes
21.867 s GPU versus 3.489 s CPU, pointing to execution cost per attempt as the
main gap there. These request-only diagnostics do not establish frozen timings.
The audit reconstructs three source variants, 75 trajectories and 58 GPU waveform
comparisons. Full qualification and complete-study performance acceptance remain
outstanding.

Two [kernel screening experiments](evidence/emi03/diagnostics/resident-kernel-screening/README.md)
are rejected. Serial sparse triangular solving takes 4.149 s versus 2.285 s for
the corrected baseline on a shortened coupled case. Outlining the timestep
routine takes 2.369 s versus 2.285 s. All twenty GPU diagnostic outputs pass
short-window CPU waveform checks, but neither trial improves performance or
qualifies a new implementation. Both patches and their negative results remain
reconstructable; the working kernel retains neither change.

## Resource interpretation

Each GPU worker enforces a shared 256 MiB cap for explicit expression/solver and
cuDSS device allocations, and reports cleanup and executed-work counters. The
harness separately samples aggregate device residency relative to its pre-worker
baseline and recursively samples host-process high-water memory. Driver contexts
and kernel local-memory allocations are outside the exact allocator ledger;
sampling cannot establish their instantaneous peak. Both the exact allocation
accounting and observed resource gates must pass. No claim of an exact total
driver-memory peak is made.

The harness retains all thirty terminal GPU qualification records, including
failures and unavailable workers. Missing telemetry or exhausted workers close
the resource gate and are never repaired by selective retry. Failed qualification
prevents collection of the 9/36-job performance modes; surviving jobs cannot be
reported as a speedup.

## Retained qualification

| Observation | Invocation 1 | Invocation 2 |
|---|---:|---:|
| Fresh CPU jobs accepted | 30/30 | 30/30 |
| Fresh ngspice jobs accepted | 30/30 | 30/30 |
| CPU + ngspice refinement checks | 80/80 | 80/80 |
| Complete CPU/ngspice differential checks | 30/30 | 30/30 |
| GPU jobs accepted | 0/30 | 0/30 |
| GPU requests started / rejected before submission | 4 / 26 | 4 / 26 |
| Observed incremental device residency | 4,781 MiB | 4,781 MiB |
| Frozen device budget | 4,096 MiB | 4,096 MiB |
| Aggregate host high-water observation | 5.126 GiB | 5.099 GiB |
| CPU/external generation, comparison and refinement | 968.961 s | 973.062 s |
| Complete invocation wall time | 970.778 s | 974.982 s |

The host is the recorded Ryzen 9 9950X3D / RTX 5080 platform under WSL2. Driver,
clock observations, power policy, physical-core affinity, complete source hashes,
binary hashes and model identity are retained in each invocation manifest. Both
invocations use identical source, binary, model and frozen manifest identities,
with fresh worker processes and fresh CPU/ngspice execution in each invocation.

In each invocation, board-wide device residency increased from 3,348 to 8,129 MiB,
then returned to the original baseline after the workers were terminated. The
4,781 MiB observed increase exceeds the cap by 685 MiB. The sampler recorded no
errors. These observations support association with the workers, but do not
isolate per-process allocations or exclude unrelated device activity. Replicated
CUDA context/library initialization and implicit kernel storage are plausible
contributors, not separately measured causes. Static linkage does not establish
that the library's on-disk binary bytes occupy VRAM.

Each invocation retains all thirty GPU records with `resource_limit` status. The
first four requests were terminated after approximately 0.54–0.58 s; the other twenty-six
were rejected before submission. None reached normal end-of-job telemetry
publication. The retained native-allocation sum is therefore empty: its numeric
zero is **unavailable native telemetry**, not evidence of zero allocation or
verified explicit cleanup. No surviving subset enters a feasible ranking or a
performance calculation.

[Invocation 1 summary](evidence/emi03/run-1/summary.json),
[qualification](evidence/emi03/run-1/qualification.json),
[manifest](evidence/emi03/run-1/invocation.json) and
[reconstruction audit](evidence/emi03/run-1-audit.json) retain the first result.
[Invocation 2 summary](evidence/emi03/run-2/summary.json),
[qualification](evidence/emi03/run-2/qualification.json),
[manifest](evidence/emi03/run-2/invocation.json) and
[reconstruction audit](evidence/emi03/run-2-audit.json) retain the second result.
Each audit recomputes raw reconstruction, measurements, differential/refinement
checks, resource observations and complete terminal accounting. A passing audit
status validates the recorded negative result; it does not mean GPU qualification
passed.

## Implementation validation

The normal Bazel suite and lockfile-enforced suite pass all 40 test targets. The
ASAN and UBSAN suites each pass 35 targets with five declared incompatible targets
skipped. CUDA smoke, static linkage and all 19 new CUDA tests pass, including
expression values/derivatives, lazy branch behavior, singular zero-RHS rejection,
original-system refinement, allocation/readback failure injection, cleanup and a
complete behavioral transient with repeated fresh state. The ensemble harness has
sixteen hostile/unit tests and a real persistent CPU-worker protocol test.

Lint, broad build, ngspice acceptance, prepared-AC correctness/replay checks and
the prescribed optimized prepared-AC benchmark pass. Explicit CUDA+ASAN and
CUDA+UBSAN builds reject both the smoke target and EMI-03 worker during analysis,
as declared. Historical evidence, prepared-AC semantics and Rust source are kept
intact. [Validation commands](evidence/emi03/validation/checks.json),
[CUDA test output](evidence/emi03/validation/cuda.log) and the
[protected-scope audit](evidence/emi03/validation/scope.json) accompany the evidence.

The supported conclusion is that this frozen CUDA configuration is ineligible
under the declared resource envelope. Full-workload CUDA accuracy and the
2.0x median / 1.5x P95 / cold-time usefulness gates remain unestablished. A changed
worker/context configuration would be a new candidate requiring newly frozen
inputs and complete fresh qualification; these artifacts cannot qualify it.
