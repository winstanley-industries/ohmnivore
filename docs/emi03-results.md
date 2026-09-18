# EMI-03 transient ensemble experiment

Both complete invocations are **negative at the resource gate**. Their fresh
CPU/ngspice references pass, but the frozen four-worker CUDA configuration
exceeds the aggregate device-residency budget during startup. No CUDA numerical
result is accepted, and no performance observation or speedup is published.

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
