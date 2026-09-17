# C++/CUDA Roadmap: Inverter EMI Design Studies

This document records the agreed direction after the deterministic FP64 CPU Phase 3C path and
the negative GPU-02/GPU-02S performance experiments. The next priority is the combined common-mode
and differential-mode inverter output-filter design study in
[ADR-002](adr/ADR-002-inverter-emi-design-study.md). Each implementation slice still requires a
bounded contract in ADR-001 or a successor ADR before code changes begin.

GPU-01 is complete under its exact bounded contract in ADR-001. GPU-02 is complete only as the
explicit native-complex-FP64 CUDA correctness and crossover experiment contracted in ADR-001;
GPU-02S is the bounded persistent-session evidence follow-up and changes no production routing.
Ordinary CUDA execution and automatic dispatch remain unauthorized. EMI-01 is complete as a
qualified external CPU reference study. Its [v2 results and acceleration budget](emi01-v2-results.md)
include two complete independent invocations with an all-corner passing reference, a near-boundary
rejection and a failing control under the fixed research mask. [V1 results](emi01-results.md) remain
historical evidence for their original candidates. NL-04 and EMI-02 onward have not started. Completion
of an experiment does not itself authorize downstream work.

The CPU implementation remains the correctness authority and supported no-GPU path. CUDA results
remain untrusted until the CPU differential and independent validation gates accept them. Planning
an epic does not make its syntax, model, backend, performance, or dispatch behavior supported.

## Roadmap and dependencies

The objective is to find lower-mass inverter output filters with sufficient predicted EMI margin
by evaluating many independent filter candidates and operating/tolerance corners. Each simulation
contains its complete coupled inverter/filter/load/chassis network. The performance metric is
validated candidate/corner evaluations per hour against a fair parallel CPU baseline, with fixed
accuracy and complete study timing.

| Stage | Deliverable | Dependency |
|---|---|---|
| EMI-01 | Public reference circuit/model, combined CM/DM filter candidate set, qualified CPU study, runtime breakdown, and frozen experiment contract | Complete as an external reference, including passing and near-boundary research fixtures; no production switching support or hardware qualification claimed |
| EMI-02 | Only the missing CPU model, coupled-element, transient, and measurement semantics needed by the reference | Qualified EMI-01; separate [EMI-02A through EMI-02D proposals](../reference/emi01/CPU_GAPS.md), beginning with disjoint linear coupled-inductor pairs |
| EMI-03 | Native-FP64 GPU execution of independent transient jobs, compared with persistent parallel CPU execution | Qualified EMI-02 CPU path and frozen accuracy/performance gates |
| EMI-04 | Filter search and robustness evaluation with physical mass and emission-margin accounting | Trustworthy evaluator; CPU enumeration can start before GPU acceleration |

The first milestone is a reference study, not a production optimizer or a new CUDA algorithm.
Qualify switching behavior with a double-pulse fixture, then evaluate a pulse train through a
defined combined CM/DM filter and load/harness/chassis network. Freeze the circuit, model, finite
candidate/corner set, component mass/rating data, measurement ports, spectral processing, numerical
tolerances, and performance budget before GPU work. No original proprietary design files are
required: a documented public surrogate is the starting point.

The exact aviation procedure/category, limits, bandwidth, and detector have not been supplied.
Until they are defined, use a declared research mask and make no requirements-compliance claim.
Likewise, simulator agreement does not establish laboratory correlation. The full specification
and unresolved inputs are in ADR-002.

GPU-01 and GPU-02 remain completed linear-AC experiments. Their infrastructure and lessons can be
reused, but they do not establish nonlinear transient performance. NL-04 remains an unstarted,
bounded CPU MOSFET DC foundation; realistic SiC charge/capacitance, coupled CM-choke windings,
vendor-model constructs, and EMI evaluation require additional contracts. Further synthetic-AC
tuning is deferred in favor of the representative workload and its measured bottlenecks.

## NL-04: Deterministic FP64 CPU MOSFET DC authority

### Objective

Add deterministic nonlinear `.DC`/`.OP` analysis for a strict minimal NMOS/PMOS MOSFET subset on
the existing CPU FP64 Newton and production KLU path. The result becomes the correctness oracle for
later MOSFET and CMOS CUDA work while remaining a supported no-GPU implementation.

### Required contract work

- Amend ADR-001, or add a successor ADR, with the exact instance and `.MODEL` grammar, parameter
  defaults and bounds, terminal order, polarity, current signs, region-boundary policy, equations,
  derivatives, limiting, convergence, determinism, and typed failure semantics before editing the
  parser or solver.
- Make the bulk-terminal policy explicit. An admitted terminal must have modeled semantics; the C++
  path may not silently ignore a bulk terminal or another accepted field.
- Bound the initial model to the separately authorized Level-1 DC behavior. Charge storage,
  capacitances, body effect, subthreshold behavior, temperature dependence, and advanced device
  effects remain excluded unless the epic contract explicitly replaces this boundary.
- Reuse the Phase 3A/3C Newton, source-stepping, GMIN-stepping, residual-validation, finite-value,
  KLU backward-error, accepted-Jacobian, and deterministic ordering contracts. Do not add another
  production linear or nonlinear solver.
- Extend the immutable nonlinear CSR union pattern and reuse one KLU symbolic analysis across
  Newton and continuation points.

### Evidence and acceptance

- Strict parser, direct-IR, compiler, descriptor, malformed-input, and unsupported-input tests.
- Independent analytic current and Jacobian oracles covering NMOS and PMOS cutoff, linear, and
  saturation regions, including region boundaries, terminal aliases, and ground connections.
- Deterministic Newton and continuation traces, finite-value and magnitude guards, structural
  validation, and adversarial failure-path tests.
- Hermetic ngspice comparisons for a bounded set of bias circuits.
- A representative, reproducible corpus containing individual NMOS/PMOS bias cases and CMOS
  inverter/corner cases suitable for later GPU work. This corpus is evidence material, not a claim
  that CUDA dispatch exists.
- The canonical lint, build, test, sanitizer, lockfile, hermeticity, ngspice, and diff gates.

### Explicit non-goals

MOSFET transient or charge behavior, MOSFET AC/noise/temperature behavior, BJT or diode semantic
changes, CUDA circuit kernels or dispatch, mixed precision, distributed solving, and performance
claims are outside NL-04.

## GPU-01: Prepared workload and evidence foundation

**Status:** Completed as the CPU-only prepared-workload, replay, hostile-validation, and evidence
foundation. No GPU-02 or NL-04 work is included.

### Objective

Define the smallest backend-neutral prepared-batch boundary that can support an honest CPU-versus-
GPU decision. Preserve KLU as the correctness authority, supported implementation, and fallback.
GPU-01 must make the GPU performance claim falsifiable before GPU-02 selects or implements its
production-candidate algorithm.

### Prepared workload contract

- Introduce a semantic/backend boundary for a prepared family of solves without exposing CUDA
  types, device ownership, streams, or allocation policy to parsing, Circuit IR, MNA compilation,
  analysis orchestration, or result formatting.
- Represent immutable canonical sparse structure separately from ordered per-member matrix values,
  right-hand sides, frequency/corner/circuit identity, and result association.
- Provide a CPU KLU implementation of the contract that preserves existing ordering, complex AC
  semantics, symbolic reuse, validation, typed errors, and CSV-visible results.
- Keep ordinary CPU execution unchanged. A prepared or experimental backend must be explicit and
  cannot silently alter the supported no-GPU path.

### Performance thesis

GPU-01 must freeze a falsifiable hypothesis before CUDA implementation:

> Reusing immutable sparse structure across a sufficiently large batch of independent AC points,
> parameter corners, or circuits will amortize preparation, upload, launch, synchronization,
> readback, and validation costs enough for native-FP64 CUDA throughput to exceed a parallel CPU
> KLU batch baseline on the declared reference hardware and workload corpus.

The timing model is end to end:

```text
T_gpu = T_prepare + T_upload + T_device + T_sync + T_readback + T_validate
T_cpu = T_prepare + T_parallel_schedule + T_klu + T_validate
```

Record cold execution and prepared/reused execution separately. CPU FP64 single-thread KLU remains
the deterministic correctness authority, but the performance competitor must also include a fair
parallel host baseline that schedules independent KLU solves across available CPU cores. Comparing
CUDA only with the single-thread authority is not sufficient evidence of a useful speedup.

### Evidence design and exit criteria

- Freeze a versioned replay corpus covering declared matrix dimensions, nonzero counts, sparsity
  shapes, batch sizes, frequency ranges, value scales, and reuse counts. Tiny acceptance fixtures
  alone are insufficient for a performance claim.
- Record target CPU, GPU, driver, toolchain, build mode, warmup, sampling, synchronization, and
  raw-sample metadata. Report cold latency, prepared latency, throughput, tail latency, CPU memory,
  GPU memory, validation cost, and failure counts.
- Define the crossover and automatic-dispatch eligibility thresholds before GPU-02 implementation.
  The thresholds may vary by declared workload class but may not be chosen after seeing the CUDA
  result.
- Reuse the existing finite-result and backward-error validation on every returned solution. Add
  hostile-result tests that inject non-finite values, excessive residuals, reordered associations,
  missing results, duplicate results, and stale replay identities.
- Document and checksum-pin any CUDA library or algorithm candidate before it becomes a build
  input. System CUDA, nvcc, compilers, headers, or libraries remain forbidden.
- Complete GPU-01 without CUDA circuit kernels, automatic backend selection, or a speedup claim.
  Its success criterion is a reviewable contract and experiment capable of disproving the thesis.

### Explicit non-goals

GPU kernels, CUDA solver dispatch, new device or analysis semantics, MOSFET work, mixed precision,
single-circuit domain decomposition, MPI/NCCL/RAS, multi-node execution, and production speedup
claims are outside GPU-01.

## GPU-02: Native FP64 CUDA batched-AC vertical slice

**Status:** Completed as an opt-in cuDSS experiment with CPU-certified replay-v1 correctness and
two reproducible native-uniform-batch evidence invocations. Correcting the initial serial
per-member cuDSS call shape materially reduced factor/solve time, but the frozen end-to-end timing
gate was still not achieved, so no workload class is eligible for a later dispatch proposal.
Automatic selection remains unauthorized independently of the measured result.

### Objective

Implement one opt-in, end-to-end CUDA vertical slice for the existing linear AC contract, using the
GPU-01 prepared workload and evidence harness. Upload immutable sparse structure once and evaluate
independent batch members without changing frequency generation, MNA signs, ordering, CSV output,
or CPU behavior.

### Required implementation boundary

- Start with native `double` and complex FP64 behavior. Mixed precision requires a later,
  separately authorized evidence phase.
- Preserve deterministic input/result association for every frequency, corner, and circuit. A
  missing, duplicate, reordered, stale, or non-finite result is a typed failure.
- Validate every CUDA solution on the CPU with the authoritative matrix and right-hand side before
  accepting it. CUDA-reported convergence or status is not sufficient.
- Keep CPU KLU available and unchanged. The CUDA path remains explicit and experimental until all
  correctness and performance gates pass; failure cannot silently produce partial output.
- Check checksum-pinned CUDA dependencies, runtime linkage, architecture coverage, replay identity,
  and the declared CUDA/sanitizer incompatibility through explicit Bazel targets.

### Acceptance and dispatch gate

- Focused kernel/library tests, exact-small or analytic tests where applicable, CPU/CUDA
  differential tests across the frozen corpus, hostile-result tests, repeated replay tests, and
  the canonical CPU and CUDA validation gates must pass.
- Measure both cold and prepared end-to-end paths against the single-thread CPU authority and the
  parallel CPU KLU performance baseline. Include preparation, transfer, synchronization, readback,
  and CPU validation; kernel-only timing cannot establish eligibility.
- Evaluate each workload class against the crossover thresholds frozen by GPU-01 without weakening
  correctness, determinism, validation, or failure semantics. Passing the experimental threshold
  would only support a later ADR proposal; GPU-02 itself never authorizes automatic dispatch.
- If the declared crossover is not achieved, GPU-02 still succeeds as an experiment when it
  produces complete reproducible evidence. CUDA remains opt-in, and the negative result must guide
  the next architecture decision rather than being hidden by a narrower timing boundary.

### Explicit non-goals

Nonlinear device evaluation, Newton iteration, transient execution, MOSFET execution, semantic
changes to linear AC, mixed precision, automatic or production-default dispatch, distributed
solving, and single-circuit domain decomposition are outside GPU-02.

## GPU-02S: Persistent-session crossover evidence

**Status:** Completed as a bounded negative experiment. Neither fresh nor persistent sessions met
the declared crossover gates in either timing lane. It retains the GPU-02 executor configuration
and adds same-structure value/RHS refresh, a compiler-derived large-sweep session corpus, a fair
persistent CPU worker-pool comparator, and reproducible cold plus long-lived-process measurements.
It does not change the frozen GPU-02 replay-v1 evidence or gate.

GPU-02S answers the narrower Ohmnivore use-case question left open by fresh-process GPU-02:
whether multiple frequency/corner batches in one process can amortize CUDA context, structure, and
analysis cost. Its four cases are one permanently ineligible 64-point control plus 512-point and
256-point grids and a 2,048-point multi-source ring, each with four same-structure component/source
corners. Every batch is constructed through public `Circuit`, `CompileMna`, and
`PrepareLinearAcBatch` paths.

Both CPU and CUDA evidence use persistent owners. CPU workers retain private KLU symbolic state;
CUDA retains context, stream, cuDSS objects, allocations, canonical structure, and analysis while
refreshing complete FP64 values and RHS buffers. The `inline_certified` lane includes fresh KLU
certification. The evidence-only `candidate_runtime` lane measures residual validation separately
from mandatory fresh KLU certification; it is not an acceptance or dispatch policy. Fresh-child
session and same-process steady-session ratios retain the 1.25 median, 1.10 P95, and 2 GiB bounds.

GPU-02S completed with a negative crossover verdict. Neither lane authorizes ordinary
`SimulateAc`, CSV output, automatic dispatch, mixed precision, nonlinear or transient GPU work, or
any downstream phase.

## Later decision boundary

The accepted EMI direction is a workload decision, not a speedup claim. A nonlinear GPU proposal
requires the CPU model/transient authority selected through EMI-01/EMI-02, including NL-04 or an
explicit successor contract, and must account for the entire Newton loop, timestep control,
validation, and spectral evaluation. Independent jobs retain their own timestep/retry state;
different convergence histories and load imbalance must be measured. Offloading device evaluation
alone is not presumed to be faster.

Preserve the frozen AC corpora, thresholds, and evidence. Their recorded fingerprints identify
historical inputs; this documentation revision does not refresh the performance measurements.
New experiments require new complete evidence. Automatic dispatch, mixed precision, and
distributed/domain-decomposed solving remain outside the initial EMI study.
