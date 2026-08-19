# ADR-001: Migrate the Solver Core to C++20 and CUDA

- **Status:** Accepted
- **Date:** 2026-08-04
- **Baseline:** Rust/wgpu `main` at `c2189a651d4879211019e109b2136dee836a5c5d`

## Context

The Rust prototype established that Ohmnivore's core decomposition is sound:

```text
SPICE netlist -> Circuit IR -> MNA compiler -> analysis engine -> solver -> results
```

It also accumulated useful behavioral contracts: insertion-ordered nodes, explicit ground
handling, CSR matrices, standard MNA stamps, DC/AC/transient orchestration, nonlinear device
models, continuation methods, CSV output, and ngspice fixtures.

The wgpu implementation made portability constraints control the numerical design. Native FP64
was not a dependable backend contract, so nonlinear work introduced paired-FP32 double-single
arithmetic and thousands of lines of WGSL. Iterative reductions require frequent host readback,
runtime pipeline compilation adds fixed costs, and direct access to NVIDIA sparse-solver,
profiling, communication, and graph-execution facilities is unavailable.

The prototype also exposed product-contract gaps that must not be inherited silently:

- `--cpu` does not provide a CPU-only nonlinear path;
- nonlinear transient analysis is absent;
- multi-rank nonlinear solving is absent;
- the distributed path is not wired into the CLI; and
- benchmark and GPU-test paths can suppress failures instead of producing durable evidence.

Rust can call CUDA through FFI, but that would retain a language boundary through the most
performance-sensitive subsystem. The project now has a proven hermetic C++/CUDA workflow from
APGAR and is targeting NVIDIA GPUs rather than graphics-API portability.

## Decision

Ohmnivore will migrate to a C++20 core with a CUDA-first GPU backend and Bazel as the canonical
build interface.

The following contracts are mandatory:

1. **CPU FP64 authority.** A deterministic CPU implementation is the correctness oracle and
   supported no-GPU path. CUDA results are untrusted until checked against it.
2. **Semantic/backend separation.** Parsing, Circuit IR, MNA compilation, analysis orchestration,
   and result formatting cannot depend on CUDA types or device ownership.
3. **Native precision first.** CUDA solver work starts with `double`. Mixed precision is admitted
   only by accuracy and end-to-end benchmark evidence.
4. **Hermetic toolchains.** Builds may not discover system CUDA, nvcc, GCC, Clang, headers, or
   libraries. Bazel selects checksum-pinned LLVM and CUDA/GCC toolchains.
5. **Explicit unsupported behavior.** Unported SPICE constructs return typed errors; they are not
   approximated, ignored, or routed through incomplete implementations.
6. **Differential acceptance.** Every migrated analysis requires focused unit tests, CPU
   behavioral tests, ngspice comparisons where applicable, and CUDA/CPU differential tests before
   CUDA dispatch can become eligible.
7. **Evidence before distribution.** Independent circuit/frequency/corner batching is evaluated
   before single-circuit domain decomposition. Multi-node RAS is not ported by default.
8. **Legacy preservation.** Rust/wgpu remains available as a temporary behavioral reference until
   an explicit parity decision retires it.

CUDA is isolated behind solver interfaces so another compute backend can be evaluated later
without changing circuit semantics. Cross-platform GPU support is not a current requirement.

## Phase 1: Foundation and one vertical slice

Phase 1 proves the new toolchain and architecture with the smallest observable simulation:

```text
R/V/.DC netlist -> C++ Circuit IR -> CSR MNA -> CPU FP64 solve -> CSV
```

### Included

- Bazel 9.2.0 selected by Bazelisk;
- zero-sysroot LLVM for normal C++ builds;
- opt-in, checksum-pinned CUDA 13.0.2 and GCC 15.2.0 for `sm_120`;
- pinned clang-format, Ruff, ShellCheck, and Buildifier through `bazel lint`;
- typed `Result<T>` errors at library boundaries;
- resistors, independent DC voltage sources, `.DC`, `.OP`, `.END`, and `.PRINT` as a documented
  compatibility no-op because the CLI emits every solved variable;
- case-insensitive `GND` plus `0` as ground;
- engineering-number suffixes matching the legacy parser;
- insertion-ordered nodes and voltage-source branch variables;
- resistor, voltage-source, RHS, and `1e-12 S` GMIN MNA stamps;
- deterministic CSR construction;
- dense partial-pivoting FP64 reference solve, explicitly temporary for this bounded slice;
- legacy DC CSV column and current-sign conventions; and
- a live CUDA allocation, kernel, synchronization, copy-back, CPU-oracle, and runtime-linkage
  smoke test.

### Excluded

- bulk porting of the Rust parser or test suite;
- capacitors, inductors, current sources, semiconductor devices, models, AC, or transient;
- selection of the production sparse-direct CPU dependency;
- CUDA circuit kernels, BiCGSTAB, preconditioners, or mixed precision;
- MPI, NCCL, RAS, domain decomposition, or cluster scheduling;
- deletion, movement, or modification of the Rust implementation; and
- performance claims.

### Acceptance gates

```sh
bazel lint
bazel build //...
bazel test //...
bazel test --config=asan //...
bazel test --config=ubsan //...
bazel test --lockfile_mode=error //...
bazel test --config=cuda //:cuda_smoke_test
git diff --check
```

CUDA plus LLVM sanitizer configurations are rejected at Bazel analysis time. CPU sanitizer gates
and the CUDA smoke gate are separate claims.

## Phase 2A: Linear-device CPU semantic foundation

Phase 2A extends the CPU semantic layer without opening AC, transient, sparse-solver selection, or
CUDA circuit execution. It adds:

- capacitor, inductor, and independent DC current-source Circuit IR and parsing alongside the
  existing resistor and independent DC voltage source;
- strict bare-value and `DC value` source forms, with malformed and recognized-but-unsupported
  source forms reported by distinct typed errors;
- deterministic insertion-ordered non-ground nodes and a single component-order branch sequence
  shared by voltage sources and inductors;
- independent deterministic CSR `G` and `C` matrices plus `b_dc`;
- current-source RHS orientation, capacitor dynamic stamps with open-circuit DC behavior, and
  inductor incidence plus `-L` dynamic stamps with short-circuit DC behavior;
- `.DC`/`.OP` operating-point execution for RLCVI netlists through the existing dense FP64 CPU
  correctness path; and
- focused parser, exact-CSR, ordering, malformed-CSR, and analytic operating-point tests.

Phase 2A does not execute AC or transient analysis, admit AC/transient source forms, select a
production sparse-direct dependency, add nonlinear devices, add CUDA circuit kernels or solver
dispatch, introduce mixed precision, or add distributed solving. The Rust implementation remains
unchanged as a behavioral reference.

## Phase 2B: Deterministic linear AC CPU correctness path

Phase 2B extends only the linear FP64 CPU semantic layer. It adds:

- optional DC and typed AC magnitude/phase specifications for independent voltage and current
  sources, including bare DC, `DC value`, AC-only, and combined DC/AC forms;
- strict full-token source parsing, rejection of sources without either specification, and typed
  unsupported errors for recognized transient waveforms;
- validated `.AC DEC|OCT|LIN points start stop` analysis IR with positive frequencies,
  `stop > start`, positive logarithmic density, at least two LIN points, and a one-million-point
  generated-sweep limit;
- deterministic inclusive frequency generation: LIN has exactly the declared total count, and
  DEC/OCT have `ceil(points * log_base(stop/start)) + 1` points with the exact stop included once;
  grids that would collapse distinct requested points in FP64 are rejected;
- a deterministic complex AC right-hand side and canonical `A(omega) = G + j * omega * C`
  formation that merges independent canonical G/C CSR patterns;
- a temporary dense `std::complex<double>` partial-pivoting CPU correctness solve;
- CPU AC orchestration plus legacy-compatible magnitude/phase CSV ordering; and
- focused parser, exact matrix/RHS, frequency, malformed-solver, analytic RC/RL/RLC, CSV, and DC
  preservation tests.

Phase 2B does not add transient execution or waveform sources, nonlinear devices, a production
sparse-direct dependency, CUDA circuit kernels or solver dispatch, mixed precision, or distributed
solving. Analytic FP64 solutions provide acceptance for this slice. A separately checksum-pinned
hermetic ngspice harness remains required before any ngspice differential-acceptance claim.

## Phase 2C: Deterministic linear transient CPU correctness path

Phase 2C completes and differentially accepts the linear FP64 CPU subset. It adds:

- typed PULSE, SIN, PWL, and EXP source waveforms after optional DC and AC specifications, with
  exact token consumption, finite values, explicit defaults, strict arity, and validated time and
  waveform domains;
- validated `.TRAN tstep tstop [tstart] [UIC]` analysis IR;
- deterministic time-dependent voltage/current-source RHS construction using the existing branch
  and current-source signs;
- canonical independent-pattern backward-Euler and trapezoidal companion systems;
- deterministic scaled FP64 BE step-doubling and BE/TRAP local-error control bounded by a
  declared maximum step, adaptive minimum, attempt/point limits, and typed failure contracts;
- exact output-start, waveform-breakpoint, and stop-time landing;
- left-limit integration and zero-time reactive-state-preserving projection at discontinuous
  source edges;
- DC operating-point initialization by default and deterministic zero capacitor-voltage/zero
  inductor-current UIC constraints with typed inconsistency failures;
- legacy-compatible transient CSV schema, insertion order, signs, and C++ CSV escaping; and
- a checksum-pinned, Bazel-built ngspice 46 acceptance harness covering only representative linear
  RLCVI DC, AC, and transient fixtures under recorded comparison tolerances.

Phase 2C does not add nonlinear or nonlinear-transient execution, initial-condition syntax beyond
the UIC zero-reactive-state contract, a production sparse-direct dependency, CUDA circuit kernels
or solver dispatch, mixed precision, MPI/NCCL/RAS/domain decomposition, or performance claims.
The ngspice claim applies only to the recorded linear fixtures and deterministic comparison
contract.

## Phase 2D: Hermetic production sparse-direct FP64 CPU path

Phase 2D changes only the linear CPU solver implementation. It adds:

- SuiteSparse KLU 2.3.6 from the checksum-pinned SuiteSparse 7.12.3 source archive, built directly
  by Bazel with serial 32-bit-index KLU, BTF, AMD, COLAMD, and SuiteSparse_config C sources and no
  system sparse solver, BLAS/LAPACK, Fortran, OpenMP, CMake, or `PATH` discovery;
- strict deterministic conversion from canonical square CSR to KLU CSC, retaining explicit zeros
  and rejecting duplicates, unordered columns, malformed dimensions/offsets, non-finite values,
  and indexes or nonzero counts outside signed 32-bit representation;
- real and complex FP64 factorization with BTF enabled, AMD ordering, maximum-magnitude row
  scaling, full partial-pivot threshold, singular halt, and single-process/single-thread execution;
- reusable symbolic analysis and deterministic numeric refactorization for fixed sparsity patterns
  across AC points and transient companion, UIC, and discontinuity-projection systems, with a
  fresh pivoting numeric factorization under the same symbolic analysis whenever fixed-pivot
  refactorization fails solution validation;
- typed fail-closed structure, size, singular/rank, factorization, non-finite-result, and
  solution-validation failures with no dense fallback;
- finite-result validation with a row-equilibrated normwise backward-error bound of `1e-10` and a
  maximum rowwise componentwise guard of `1e-5` for mixed-unit MNA systems, so an unrelated
  large-unit variable cannot hide a grossly bad small-unit row;
- a test-only dense partial-pivoting exact-small oracle that production targets cannot link or
  dispatch to; and
- reproducible, non-gating selection evidence over a fixed circuit-representative real/complex
  matrix corpus.

Phase 2D preserves every Phase 2A/2B/2C parser, MNA, GMIN, sign, ordering, grid, timestep,
waveform, CSV, and bounded ngspice-acceptance contract. It does not add nonlinear devices,
Newton iteration, nonlinear transient behavior, new syntax, CUDA circuit solving, mixed precision,
distributed solving, new ngspice fixture scope, or performance/scalability claims. Detailed
selection, license, build, storage, numerical, determinism, validation, and benchmark evidence is
recorded in `third_party/suitesparse/PROVENANCE.md`.

## Phase 3A: Deterministic FP64 CPU diode DC foundation

Phase 3A adds only nonlinear DC operating-point analysis for the legacy-compatible Shockley-diode
subset. The production CPU FP64 path is the correctness oracle for later CUDA nonlinear work.

### Netlist and model contract

- A diode instance is exactly `Dname anode cathode modelname`, with no area, temperature,
  initial-condition, geometry, or trailing fields. Anode-to-cathode voltage and current are
  positive. A diode model name or reference is one or more ASCII letters, digits, or underscores
  (`[A-Za-z0-9_]+`); definitions and references outside that lexical grammar are malformed.
- A diode model is `.MODEL modelname D`, `.MODEL modelname D()`, or
  `.MODEL modelname D(IS=value N=value)`. Inside parentheses, one or both whitespace-separated
  `key=value` fields may appear in either order. Keys and model lookup are ASCII
  case-insensitive; the original spelling and insertion order remain observable. Commas,
  detached parentheses, nested parentheses, trailing text, empty fields, and every parameter
  except `IS` and `N` are rejected.
- `IS` is saturation current in amperes and defaults to `1e-14`; `N` is the dimensionless
  emission coefficient and defaults to `1.0`. Both must be finite in `(0, 1e100]`, and `N * VT`
  must remain positively representable. Duplicate parameters and duplicate model names under
  case-insensitive comparison are parse errors. Missing models, invalid direct IR, or a diode
  whose terminals identify the same node are typed compile failures.
- The fixed temperature policy is the legacy `VT = kT/q = 0.02585 V` at 300 K. Temperature
  syntax and temperature sweeps are not admitted.

### Compilation and sparse-structure contract

Node discovery, ground aliases, voltage-source/inductor branch discovery, source signs, GMIN, and
all result ordering remain the Phase 2A--2D contracts. Diode descriptors are emitted in component
insertion order and contain optional anode/cathode node indexes, FP64 `IS` and `N*VT`, plus the
resolved CSR value indexes for `aa`, `ac`, `ca`, and `cc` Jacobian entries. A directly supplied
descriptor must preserve the model bounds, including `0 < N*VT <= 2.585e98 V`.

The nonlinear conductance matrix is one immutable canonical CSR union of every linear `G` entry
and every possible diode Jacobian coordinate. Rows are canonical, columns are strictly
increasing, repeated coordinates from linear or multiple-device stamps are combined
deterministically, and explicit numerical zero is retained exactly when the coordinate is needed
by a diode descriptor. Linear-only compilation keeps the Phase 2A--2D zero-elision policy and
therefore its exact CSR and fast path. Matrix dimensions, nonzero counts, row offsets, columns,
and descriptor value indexes must fit the signed 32-bit KLU contract.

### Device, Newton, and validation contract

For `v = va - vc`, `q = v/(N*VT)`, Phase 3A evaluates
`i(v) = IS*expm1(q)` and `g(v) = IS*exp(q)/(N*VT)` in FP64. The exponential argument is clamped
to the closed interval `[-80, 80]`, matching the retained legacy nonlinear range: below `-80`
the reverse current and conductance use `exp(-80)`, and above `80` they use `exp(80)`. Inputs,
intermediates, and results must be finite and no larger than `1e100` in magnitude; violation is a
typed non-finite failure rather than saturation to an unreported value. The bound is checked for
base matrix/RHS data, exponential arguments, Newton deltas and sums, update/residual normalization,
limiter differences, ratios, scales, and node updates as well as stored results. A mathematically
positive conductance that underflows to zero in FP64 is also a typed evaluation failure.

With the linear system written `G*x = b`, each diode contributes `+i(v)` to its anode KCL row and
`-i(v)` to its cathode KCL row. Therefore the nonlinear residual is
`F(x) = G*x - b + S*i(S^T*x)` and the Jacobian is
`J(x) = G + S*g(S^T*x)*S^T`, giving the canonical `+g,-g,-g,+g` stamp at
`aa,ac,ca,cc`. Newton solves `J(x_k)*delta = -F(x_k)` and proposes
`x_{k+1} = x_k + delta`.

The direct and source-stepping initial guesses are the all-zero vector. Every later continuation
step uses the preceding accepted step, including the last accepted source point when GMIN stepping
begins. After every linear solve, diode junction changes are limited in descriptor order with the
SPICE3 PN-junction logarithmic rule and
`vcrit = N*VT*log((N*VT)/(sqrt(2)*IS))`; a two-non-ground-node limiter scales both endpoint
updates by the same deterministic factor. No branch variable is limited.

An iterate is accepted only when all of the following hold:

- every node-voltage update is at most `1e-9 V + 1e-6*max(|new|,|old|)`;
- every branch-current update is at most `1e-12 A + 1e-6*max(|new|,|old|)`; and
- recomputing `F` at the limited point gives a maximum row-normalized residual no greater than
  one, using `1e-12 A + 1e-6*row_scale` on node KCL rows and
  `1e-9 V + 1e-6*row_scale` on branch KVL rows. `row_scale` is the sum of the absolute linear
  row product, absolute right-hand side, and absolute diode-current contributions for that row.

The original requested system is recomputed independently for final residual acceptance. Before
any point is accepted, including an iteration-zero point whose residual already passes, its current
Jacobian is numerically factorized and zero-solved through KLU so numerical rank deficiency cannot
bypass validation. An iteration limit, non-finite or over-bound value, malformed descriptor,
singular/rank-deficient Jacobian, KLU factorization failure, sparse backward-error failure, or
nonlinear residual failure is typed and fail-closed.

### Continuation, KLU reuse, and determinism contract

The bounded strategy order is exact:

1. direct Newton on the original sources and existing production GMIN, at most 50 iterations;
2. source stepping at fixed scales `0.0, 0.1, ..., 1.0`, at most 30 Newton iterations per scale;
3. extra-GMIN stepping at fixed siemens values
   `1e-3, 1e-4, ..., 1e-12, 0`, at most 30 Newton iterations per nonzero value and 50 for the
   final zero-extra-GMIN point.

There are no adaptive subdivisions or unbounded retries. A failed step ends that strategy. Source
scale `1.0` and extra GMIN `0` are original-system solves; regardless of where convergence is
first reported, the final result is accepted only after original-source, zero-extra-GMIN residual
validation. Exhausting all three strategies returns typed non-convergence.

One KLU symbolic analysis is created for the immutable union pattern and reused across all Newton
iterations and continuation strategies. Numeric values may refactor on every iteration. The Phase
2D fixed-pivot refactorization, pivot-safe fresh-numeric retry under the same symbolic analysis,
and normwise plus rowwise backward-error validation remain mandatory. Sparse, Newton, limiting,
continuation, finite-result, and validation failures never dispatch to the dense exact-small test
oracle or another solver.

Device evaluation, stamp accumulation, limiting, residual evaluation, convergence reduction, and
continuation are single-threaded and performed in stable component/row/index order. Repeated runs
on one supported toolchain/platform must produce bitwise-identical direct, source-stepping, and
GMIN-stepping traces, solver statistics, and results. Phase 3A makes no cross-libm or cross-platform
bitwise or numerical-equivalence guarantee; independent scalar and ngspice comparisons use only
the explicit tolerances of their individual tests and acceptance fixtures.

### Excluded

Phase 3A does not add BJTs, MOSFETs, diode charge/capacitance, diode AC/noise/temperature behavior,
nonlinear transient analysis, new initial conditions or waveforms, CUDA circuit kernels or GPU
dispatch, mixed precision, distributed solving, new linear semantics, or general performance and
scalability claims. Netlists that request AC or transient execution with a diode are rejected as
unsupported. Rust remains unchanged as behavioral reference material.

## Phase 3B: Deterministic FP64 CPU memoryless-diode transient analysis

Phase 3B adds only nonlinear transient execution for the exact diode syntax, model bounds,
fixed-temperature Shockley evaluation, and insertion-order semantics accepted by Phase 3A. A diode
remains a memoryless conductance/current device: Phase 3B adds no diode charge, junction or diffusion
capacitance, transit time, or other hidden dynamic state. The production FP64 CPU path remains the
correctness authority.

### Nonlinear DAE and integration contract

For the Phase 2C state order and source vector, define the memoryless diode residual contribution
`d(x) = S*i(S^T*x)`. Phase 3B solves exactly

```text
G*x + C*dx/dt - b(t) + d(x) = 0.
```

The Phase 2C timestep schedule, hard maximum `tstep`, adaptive minimum `tstep/10000`, accepted-step
and attempt limits, output grid, waveform breakpoints, left-limit integration, right-limit
projection, and BE/TRAP method-selection rules remain unchanged. At a step from accepted state
`x_p` to `x_n` with size `h`, the nonlinear systems are:

```text
BE:   (G + C/h)*x_n - (b_n + C*x_p/h) + d(x_n) = 0
TRAP: (G + 2*C/h)*x_n
      - (b_n + b_p + (2*C/h - G)*x_p - d(x_p)) + d(x_n) = 0.
```

The `d(x_p)` history term is freshly evaluated from the accepted previous state in stable diode
order. At a discontinuous source edge, `b_n` for integration is the left limit. After the step is
accepted, the existing zero-time projection uses the right-limit source value while preserving
every capacitor voltage and inductor current. The projected state and right-limit source vector
become the history for the next step. There is no diode-history or charge term.

Every full BE step, both BE half steps, each TRAP candidate, each BE LTE comparison candidate, and
each nonlinear projection is an independently converged and residual-validated implicit solve.
The Phase 2C local-error formulas, tolerances, safety factor, adaptation clamp, hard-point landing,
and accepted higher-accuracy state are not changed.

### Initialization and discontinuity projection

Without `UIC`, transient initialization first obtains the Phase 3A nonlinear DC operating point
using the original DC sources and the exact direct/source-stepping/GMIN-stepping schedule. It then
performs the Phase 2C zero-time projection if the transient source value at zero differs, preserving
the DC capacitor voltages and inductor currents while satisfying the right-limit nonlinear
algebraic equations.

With `UIC`, capacitor voltages and inductor currents are exactly zero under the existing Phase 2C
constraint-selection and redundancy rules. The remaining independent algebraic equations include
the Phase 3A diode currents and are solved by the same bounded FP64 Newton/limiting path. A diode
contributes only to a selected physical node-KCL equation; it does not contribute to a row replaced
by a reactive-state constraint. Missing rank, inconsistent or redundant-but-conflicting
constraints, Newton exhaustion, or failed post-solve constraint/residual checks are typed failures.

Every waveform discontinuity projection applies the same rule to the accepted left-limit state.
Projection Newton is seeded by the state being projected, is bounded to 50 iterations, and has no
continuation or unbounded retry. Linear netlists continue to use the unchanged Phase 2C linear
projection path.

### Newton, timestep retry, sparse reuse, and validation

Each implicit transient point is seeded by its immediately preceding state: the accepted step
state for a full BE, TRAP, or LTE BE solve, and the first accepted half-step state for the second
BE half step. Device evaluation, PN-junction limiting, update tolerances, row-scaled nonlinear
residual criteria, finite `1e100` magnitude bound, and accepted-Jacobian zero solve are exactly the
Phase 3A contracts. The direct transient Newton budget is 50 iterations. Source stepping and GMIN
stepping remain DC operating-point strategies and are not applied to a transient companion
equation.

Only typed `kNonConvergence` from an implicit step is recoverable by the transient controller. The
attempt is recorded as a nonlinear-convergence rejection, the proposed step is exactly halved, and
the retry is a backward-Euler recovery step. Retrying stops at the existing attempt limit, adaptive
minimum, or FP64 time-representability boundary and then returns typed `kNonConvergence`. Singular
or rank-deficient Jacobians, invalid structure, unsupported size, allocation/factorization errors,
non-finite arithmetic, sparse backward-error failure, nonlinear residual-validation failure, and
projection failure propagate immediately and are never reclassified as timestep rejections.

Compilation and companion formation retain one immutable canonical CSR union containing every
linear `G` coordinate, every dynamic `C` coordinate, and every possible active diode Jacobian
coordinate. Exact numerical cancellations and diode-required zeros are retained. Descriptor value
indexes are deterministically remapped to that union. KLU symbolic analysis is reused for every
matrix with that pattern across BE, TRAP, LTE, timestep retry, and repeated accepted steps; numeric
values refactor through the Phase 2D pivot-safe policy. Projection patterns are likewise cached by
exact structure. No failure can dispatch to the dense exact-small oracle or another solver.

Before accepting any nonlinear implicit or projected state, Phase 3B freshly recomputes the
original requested equation and requires the Phase 3A normalized nonlinear residual bound in
addition to KLU's normwise and rowwise componentwise backward-error checks. Reactive constraints
and every retained algebraic equation are also rechecked independently after projection.

Device evaluation, residual and history construction, limiting, convergence reductions, LTE,
timestep adaptation, retry decisions, and sparse-pattern lookup execute single-threaded in stable
row/device/index order. Repeated runs on one supported toolchain/platform must produce
bitwise-identical output times, states, step/rejection traces, and solver statistics. Cross-libm or
cross-platform bitwise equality is not claimed; analytic and ngspice comparisons use only their
recorded tolerances.

### Preservation and excluded work

Netlists without diodes retain the Phase 2C/2D linear initialization, projection, companion solve,
LTE, timestep, ordering, sign, GMIN, CSV, and KLU fast paths without nonlinear dispatch. Linear DC,
diode DC, linear and diode-free AC, waveform parsing, and all previously accepted fixtures remain
regression gates. AC analysis containing a diode remains explicitly unsupported because Phase 3B
adds neither diode small-signal linearization nor charge.

Phase 3B does not add or approximate diode charge/capacitance, AC, noise, temperature dependence,
temperature sweeps, BJTs, MOSFETs, new source or initial-condition syntax, CUDA circuit kernels or
solver dispatch, mixed precision, MPI/NCCL/RAS/domain decomposition, or performance/scalability
claims. The Rust/wgpu source and tests remain unchanged as behavioral reference material.

## Phase 3C: Deterministic FP64 CPU BJT DC operating-point analysis

Phase 3C adds only deterministic nonlinear DC operating-point analysis for a strict minimal
legacy-compatible Ebers--Moll BJT subset. The Phase 3A FP64 Newton, PN-junction limiting,
continuation, nonlinear residual, accepted-Jacobian, fixed-pattern KLU, finite-value, and
determinism contracts remain authoritative. Phase 3C extends those contracts to BJT device
evaluation and stamping; it does not introduce another nonlinear solver or linear-solve path.

### Netlist and model contract

- A BJT instance is exactly `Qname collector base emitter modelname`, with no substrate, area,
  multiplicity, OFF, initial-condition, geometry, temperature, or trailing fields. Model names and
  references use `[A-Za-z0-9_]+`; lookup is ASCII case-insensitive, while original instance, node,
  and model spelling and insertion order remain observable. Terminal aliases such as the
  diode-connected `collector == base` form are valid, but all three terminals may not identify the
  same electrical node after treating `0` and case-insensitive `GND` as ground.
- A BJT model is `.MODEL modelname NPN`, `.MODEL modelname NPN()`,
  `.MODEL modelname PNP`, `.MODEL modelname PNP()`, or the corresponding form with one or more
  whitespace-separated `key=value` fields inside the attached parentheses. The only admitted keys
  are `IS`, `BF`, `BR`, `NF`, and `NR`; each may appear at most once and may appear in any order.
  Keys, model types, and lookup are ASCII case-insensitive. Commas, detached, nested, empty
  parameter fields, or unclosed parameter lists, trailing text, duplicate parameters, and all
  other BJT parameters are rejected. Duplicate model names, including collisions between diode and
  BJT definitions, are rejected under case-insensitive comparison.
- Defaults match the retained legacy subset: `IS=1e-16 A`, `BF=100`, `BR=1`, `NF=1`, and `NR=1`.
  Every parameter must be finite in `(0, 1e100]`; `NF*VT` and `NR*VT` must also remain positive,
  finite, and no greater than `2.585e98 V`. Direct IR with invalid model data, invalid identifiers,
  duplicate definitions, a missing or wrong-type model reference, or an all-terminal self
  connection fails with a typed compile or invalid-structure error.
- `NPN` has polarity `p=+1` and `PNP` has `p=-1`. The temperature policy is fixed at the Phase 3A
  `VT=0.02585 V` at 300 K. Temperature syntax, temperature sweeps, and temperature-dependent model
  parameters are not admitted.

### Ebers--Moll evaluation, residual, and Jacobian contract

For collector, base, and emitter voltages `Vc`, `Vb`, and `Ve`, define

```text
vbe = p*(Vb - Ve)                 vbc = p*(Vb - Vc)
IF  = IS*expm1(clamp(vbe/(NF*VT), -80, 80))
IR  = IS*expm1(clamp(vbc/(NR*VT), -80, 80))
Ic  = p*(BF/(BF+1)*IF - IR/(BR+1))
Ib  = p*(IF/(BF+1) + IR/(BR+1))
Ie  = -(Ic + Ib).
```

`Ic`, `Ib`, and `Ie` are currents leaving the corresponding collector, base, and emitter terminals
and are added to those node KCL residuals. The four junction derivatives are

```text
dIc/dVbe = BF/(BF+1)*IS*exp(vbe/(NF*VT))/(NF*VT)
dIc/dVbc = -IS*exp(vbc/(NR*VT))/((BR+1)*(NR*VT))
dIb/dVbe = IS*exp(vbe/(NF*VT))/((BF+1)*(NF*VT))
dIb/dVbc = IS*exp(vbc/(NR*VT))/((BR+1)*(NR*VT)).
```

The same closed `[-80,80]` exponential clamp applies to the derivative exponent. Expanding these
four values through `Vbe=Vb-Ve`, `Vbc=Vb-Vc`, and `Ie=-(Ic+Ib)` gives the full collector/base/emitter
3-by-3 Jacobian. NPN and PNP use the same physical-voltage derivatives because the two polarity
factors cancel. Aliased terminals combine residual and Jacobian contributions in stable row-major
terminal order.

Every input, exponent, exponential, current, derivative, residual, row scale, stamp accumulation,
limiter value, and Newton value must be finite and at most `1e100` in magnitude. Each mathematically
positive junction conductance must remain positively representable. Violations fail with typed
non-finite errors rather than saturating, skipping a terminal, or changing the model.

### Compilation, Newton, sparse reuse, and determinism

Compilation extends the immutable canonical nonlinear `G` union with all non-ground coordinates
in each BJT's collector/base/emitter 3-by-3 block. Exact zeros required by either a diode or BJT
descriptor are retained, aliases resolve deterministically to the same canonical value index, and
linear-only compilation preserves its exact zero-elision fast path. BJT descriptors are emitted in
BJT component insertion order and contain the three optional node indexes, polarity, the five
validated FP64 model values represented as `IS`, `BF`, `BR`, `NF*VT`, and `NR*VT`, and nine resolved
CSR value indexes. Descriptor dimensions and indexes remain within the signed 32-bit KLU contract.

At each Newton proposal, BJT descriptors are limited in insertion order, after diode descriptors.
Each BJT applies the exact Phase 3A logarithmic PN limiter first to polarity-adjusted `VBE` using
`IS` and `NF*VT`, then to polarity-adjusted `VBC` using `IS` and `NR*VT`; a limited junction scales
both participating non-ground node updates by the same deterministic factor. The Phase 3A update
tolerances, row-scaled residual tolerances, direct/source/GMIN schedule, iteration bounds, original
system final check, and accepted-Jacobian zero solve are unchanged. Mixed diode/BJT DC circuits use
stable diode-then-BJT device-type order and insertion order within each type for evaluation,
stamping, limiting, and reductions.

One KLU symbolic analysis of the complete diode/BJT union pattern is reused across all Newton and
continuation iterations. Numeric changes use the Phase 2D refactorization, pivot-safe fresh-numeric
retry, and backward-error validation; no failure can dispatch to the dense exact-small oracle.
Repeated runs on one supported toolchain/platform must produce bitwise-identical solutions,
iteration and attempt traces, and KLU statistics. Analytic and hermetic ngspice comparisons use
only their recorded tolerances and do not claim cross-libm or cross-platform bitwise equality.

### Preservation and excluded work

Netlists without BJTs retain all Phase 2A--3B parsing, matrix, solver, timestep, diode, source,
ordering, sign, GMIN, and CSV behavior. BJT DC results use the existing DC node/branch ordering and
CSV schema. AC or transient analysis of any circuit containing a BJT fails explicitly as
unsupported; there is no silent linearization, device omission, or diode substitution.

Phase 3C does not add BJT transient or charge storage, capacitances, transit time, Early effect,
high-current effects, area or multiplicity scaling, initial conditions, AC, noise, temperature,
temperature sweeps, MOSFETs, new source or analysis syntax, CUDA circuit kernels or dispatch, mixed
precision, MPI/NCCL/RAS/domain decomposition, or performance/scalability claims. Diode behavior,
including Phase 3B memoryless transient analysis, is unchanged. Rust/wgpu source and tests remain
unchanged as behavioral reference material.

## GPU-01: Prepared linear-AC workload and evidence foundation

GPU-01 adds only an explicit backend-neutral boundary for batches of the already-supported linear
FP64 AC solves and the CPU-only evidence needed to evaluate a later CUDA experiment. It changes no
netlist, MNA, frequency-grid, nonlinear-device, CSV, or ordinary simulation semantics. Production
KLU remains the only implementation, the deterministic correctness authority, and the supported
no-GPU path. This section is the complete GPU-01 implementation contract; it does not authorize
GPU-02 or NL-04.

### Prepared batch and stable-identity contract

The version-1 prepared workload begins with one already-compiled linear `MnaSystem` and one
validated `AcAnalysis`. Diode or BJT descriptors are rejected as unsupported; no device is
linearized or omitted. `GenerateAcFrequencies` remains the sole frequency-grid authority and
`FormAcMatrix` remains the sole numerical definition of
`A_k = G + j*2*pi*f_k*C`. Member order is exactly generated-frequency order. Preparing a batch
does not sort, deduplicate, approximate, or regenerate the frequencies.

The batch stores one immutable canonical sparse structure---the square dimension, CSR row offsets,
and strictly increasing per-row column indexes---separately from each member's ordered complex
value array and right-hand side. All members must have that exact structure and value/RHS lengths.
Each member carries the following self-contained observable identity tuple; `contract_version`
and `replay_id` are stored in every member as well as in the enclosing batch:

```text
(contract_version=1, replay_id, circuit_id, corner_id,
 ordinal, frequency_hz_bit_pattern, structure_fingerprint,
 member_content_fingerprint)
```

`ordinal` is the zero-based position in the batch and must be contiguous. `circuit_id`,
`corner_id`, and `replay_id` are nonempty ASCII identifiers supplied by the explicit caller;
GPU-01 uses `corner_id=nominal` for ordinary compiled circuits. Frequency identity uses the exact
FP64 bit pattern, not formatted decimal text or a tolerance. Structure, member, and ordered-batch
fingerprints use the version-1 byte serialization and two independently seeded 64-bit FNV-1a
lanes documented with the replay schema. They are deterministic content identities, not
cryptographic authentication. A result echoes the contract version, replay/batch/structure
identities, and every complete member identity in the original order. A solution without that
association metadata is not a prepared result.

No CUDA type, handle, stream, event, allocation, memory-space, library status, or ownership object
may enter parsing, Circuit IR, MNA compilation, AC orchestration, result formatting, or this data
contract. The only backend seam in GPU-01 is a narrow prepared-linear-AC batch executor accepting
the data above and returning associated solution vectors. It is not a universal solver, device,
analysis, or allocation abstraction.

### Validation, typed failures, and CPU fallback

Every prepared batch is revalidated before execution and every returned batch is validated on the
CPU against the authoritative per-member matrix and right-hand side. Acceptance first applies the
Phase 2D finite-result, row-equilibrated normwise `1e-10`, and rowwise componentwise `1e-5` bounds,
then independently runs fresh CPU KLU numeric certification for every member. KLU must accept the
structure and numeric system, and every returned component must agree with that CPU result within
`1e-12 + 1e-9*max(abs(x_backend),abs(x_klu))`. Thus a singular zero-RHS system cannot be accepted
merely because an arbitrary vector has zero residual. Backend success or convergence status is
never sufficient. The entire result batch is accepted or rejected; partial publication is
forbidden.

GPU-01 adds these distinct library-boundary failures:

- `kPreparedBatchMalformed`: invalid version-1 input serialization, empty/invalid identity text,
  noncanonical or inconsistent structure, invalid member dimension/value/RHS/frequency/order, a
  duplicate member identity, or a fingerprint that does not recompute exactly;
- `kPreparedBatchStale`: returned contract, replay, batch, structure, or member content identity
  belongs to another preparation generation;
- `kPreparedResultMissing`: at least one expected member result is absent;
- `kPreparedResultDuplicate`: a returned member identity or ordinal occurs more than once;
- `kPreparedResultReordered`: the complete expected result set is present but is not in prepared
  member order;
- `kPreparedResultAssociation`: a result names an unknown circuit, corner, frequency, ordinal, or
  member identity instead of the corresponding prepared member; and
- `kPreparedInvalidResult`: a returned solution has the wrong dimension, contains a non-finite
  component, fails either authoritative backward-error bound, cannot be certified by CPU KLU, or
  disagrees with the CPU KLU solution; and
- `kPreparedBackendFailure`: a preferred or CPU prepared executor throws or otherwise escapes its
  typed `StatusOr` boundary.

Validation is linear in batch size: returned ordinals index the canonical prepared member array;
no pairwise identity scan is permitted. Duplicate ordinals or complete identities are rejected
first. For each unique in-range ordinal, a matching base identity with changed content fingerprint
is stale; another base identity is an association failure. Only the complete expected set can
then be classified as reordered. Numerical checks follow metadata checks so corrupt association
cannot be hidden by a coincidentally valid vector. A stale batch envelope or member content
fingerprint is reported as stale before association. An invalid prepared input fails before
backend invocation and is never made valid by fallback.

The explicit execution API has two policies: fail closed, or discard-and-resolve the complete
batch with CPU KLU. CPU fallback is never implicit. The execution boundary translates exceptions,
including allocation failures, to `kPreparedBackendFailure`. Under the second policy, any
preferred-backend execution exception, typed failure, or result-acceptance failure discards every
preferred result, creates a fresh
single-thread CPU KLU execution, validates the complete CPU result again, and returns it with an
observable `used_cpu_fallback` flag. A CPU KLU failure propagates unchanged and has no further
fallback; an exception from CPU execution is reported as `kPreparedBackendFailure`. GPU-01
provides only the single-thread CPU KLU backend, so the fallback protocol can be
hostile-tested without claiming another production backend. `SimulateAc`, `SimulateAcToCsv`, and
`SimulateToCsv` continue to call their existing ordinary CPU KLU path and do not dispatch through
the prepared interface.

### Version-1 replay corpus

The checked-in `prepared-ac-replay-v1` corpus is generated deterministically from four distinct
MNA sparsity shapes: a path ladder, binary tree, rectangular grid, and ring with multiple voltage
source branches. Every case verifies its declared node count, branch count, dimension, stored
G/C-union count, sweep type, sweep parameters, and exact generated batch size before timing.
`g_series`, `g_shunt`, `c_series`, and `c_shunt` are primitive stamp scales; assembled diagonal
values are deterministic sums. Branch RHS values and connection nodes follow the version-1
builder rule and are included in member fingerprints.

| Case and class | Shape; nodes + branches | Dimension | Union nnz | Sweep; batch; range (Hz) | Stamp scales `(g_series,g_shunt,c_series,c_shunt)` | Reuses |
|---|---:|---:|---:|---:|---|---:|
| `ladder_s_65`, small/control | path; 64 + 1 | 65 | 192 | LIN; 16; `1e1`--`1e4` | `(1e-3,1e-12,1e-9,1e-12)` | 8 |
| `tree_m_257`, medium | binary tree; 256 + 1 | 257 | 768 | DEC 10; 61; `1`--`1e6` | `(1e-3,1e-9,1e-8,1e-12)` | 4 |
| `grid_l_1025`, large | 32x32 grid; 1024 + 1 | 1025 | 4994 | DEC 20; 121; `1e2`--`1e8` | `(1e-2,1e-12,1e-12,1e-15)` | 2 |
| `ring_multi_m_260`, medium/wide | ring; 256 + 4 | 260 | 776 | DEC 43; 517; `1e-3`--`1e9` | `(1e-6,1e-12,1e-9,1e-15)` | 2 |

The versioned manifest is the replay authority. Parsing is strict and rejects missing or extra
columns, unknown schema/class/topology/sweep values, duplicate case IDs, noncanonical numbers,
inconsistent node/branch/dimension/nonzero/batch declarations, invalid ranges/scales, and
unbounded batch or reuse counts. The implementation pins the exact v1 manifest fingerprint and
golden generated structure, ordered-batch, aggregate-member, and boundary-member identities; a
different internally valid manifest is not v1. Evidence emits the manifest fingerprint, builder
source fingerprint, every prepared structure/batch/member identity, and complete reproduction
metadata. Changing a case, builder rule, fingerprint serialization, or identity field requires a
new replay schema version; editing a v1 meaning in place is forbidden. These cases deliberately
extend beyond tiny acceptance fixtures and cover dimensions 65--1025, 192--4994 stored entries,
batches 16--517, a true twelve-decade logarithmic sweep, several sparsity shapes, multiple
conductance/dynamic scales, multiple source branches, and reuse counts 2--8. They are
representative decision inputs, not a universality claim.

### CPU evidence and timing boundaries

The deterministic correctness authority executes members serially in identity order through one
`SparseComplexFactorization`, reusing one symbolic analysis and applying the Phase 2D pivot-safe
numeric-refactor policy and validation. Repeated authority runs on one supported build/platform
must be bitwise identical in identities, solutions, and KLU statistics.

The performance competitor is isolated to the manual evidence binary. It schedules independent
members over `min(available_hardware_threads, batch_size)` host threads unless an explicit recorded
thread count is supplied. Each worker owns its own serial KLU symbolic/numeric state; KLU objects
are never shared concurrently. A worker's first assigned member performs that worker's symbolic
analysis and numeric solve; no uncounted priming solve is permitted. Every measured execution must
report exactly `batch_size` scheduled KLU solves, and a prepared sample exactly
`batch_size*reuse_count`. Validation separately reports the same number of CPU KLU certification
solves, so the actual total KLU solves are twice the accepted-member count and none are hidden.
Results are written to their original ordinal slots, CPU-certified, and compared numerically with
the deterministic single-thread authority under the same `1e-12 + 1e-9*scale` component bound.
Only repeated serial-authority runs require bitwise equality. This parallel scheduler is not
reachable from `SimulateAc` or another production target.

Both single-thread authority and parallel-host evidence record two boundaries:

```text
T_cold = T_prepare_one_batch + T_schedule_and_klu_one_batch + T_validate_one_batch
T_prepared(R) = T_prepare_one_batch
              + sum[1..R](T_schedule_and_klu_reuse + T_validate_reuse)
```

`T_prepare` starts with the compiled `MnaSystem` plus `AcAnalysis` and includes frequency
generation, every `FormAcMatrix`, immutable-structure verification, value/RHS materialization, and
all identities/fingerprints. It excludes netlist file I/O, parsing, compilation, and CSV because
those operations are identical and outside the prepared backend seam. `T_schedule_and_klu`
includes worker creation/join for the parallel comparator, work assignment, symbolic analysis for
each participating worker, numeric factor/refactor, triangular solve, KLU's built-in validation,
and ordered result collection. `T_validate` is the separate prepared-result association,
finite/backward-error, fresh CPU KLU certification, and differential acceptance pass. Nothing
overlaps across these reported phase boundaries; their sum is the reported end-to-end time.
Prepared throughput uses all `batch_size*R` accepted members and includes the one preparation
cost. Cold and prepared samples are never combined.

The frozen reference target is the AMD Ryzen 9 9950X3D 16-Core Processor (32 hardware threads), an
NVIDIA GeForce RTX 5080, CUDA driver API 13030, and CUDA runtime/toolkit 13.3. Exact observed CPU,
GPU, driver, kernel, and hardware-thread values are recorded; a CPU-only run records
`unavailable` rather than inferring absent GPU data. Each raw measurement runs in a fresh child
process so its `ru_maxrss` is that sample's child-process peak, not a process-lifetime maximum
shared by earlier samples. Fork/setup occurs outside `T_cold` and `T_prepared`.

Every evidence stream begins with schema/corpus/manifest identity, exact command, UTC start time,
implementation/source fingerprint, binary build identity, SuiteSparse/KLU/compiler/Bazel pins,
target and observed hardware, kernel, build mode, clock, and timing/memory definitions. Every
prepared member identity is recorded. Each raw sample additionally records case/class/topology,
mode, requested/participating host threads, warmups, sample index, dimension, nnz, batch, sweep,
frequency range, stamp scales, reuse count, preparation/schedule-plus-KLU/validation/total
nanoseconds, actual KLU solve count, accepted-member throughput, child-process peak resident bytes,
GPU bytes (zero for GPU-01), and failure count. Summaries report min, P25, median, P75, P95, and max
total latency plus median throughput. Quantiles use the nearest-rank rule
`sorted[ceil(p*N)-1]`; for nine canonical repetitions P95 is therefore the maximum. A mandatory
terminal completion record gives expected and observed identity/sample/summary counts, zero
failures, UTC completion time, and a fingerprint of every preceding output record so truncated
output cannot appear complete. Evidence runs fail if a sample has a solve-count, backend,
association, validation, differential, metadata, or completion failure; there is no timing
pass/fail threshold in GPU-01.

### Frozen GPU crossover hypothesis and pre-CUDA dispatch gate

GPU-02 will test the falsifiable hypothesis that immutable-structure reuse across sufficiently
many independent FP64 AC members amortizes preparation, upload, launch, synchronization, readback,
and mandatory CPU validation enough to beat the parallel-host KLU boundary above on the declared
reference CPU plus RTX 5080. GPU time must be measured as

```text
T_gpu = T_prepare + T_upload + T_device + T_sync + T_readback + T_validate.
```

Before any automatic CUDA dispatch can be proposed, GPU-02 must use the unchanged v1 corpus, at
least 3 warmups and 20 raw samples per mode, identical build mode and reference host, zero failures,
and exact CPU acceptance. The small/control class is permanently ineligible in GPU-02 regardless
of timing. For each other class independently, the first declared batch size is a crossover only
when both of these end-to-end prepared/reused ratios hold against the parallel CPU comparator:

```text
parallel_cpu_median / gpu_median >= 1.25
parallel_cpu_P95    / gpu_P95    >= 1.10
```

The same class must reproduce both bounds in two complete benchmark invocations, and its cold GPU
median may not exceed `1.10 * parallel_cpu_cold_median`. A qualifying class also requires GPU peak
memory no greater than 2 GiB for the batch. Automatic selection remains unauthorized in GPU-02;
these numbers only define eligibility for a later reviewed dispatch change. Classes below the
first qualifying declared batch, any unrepresented dimension/nnz/frequency/value-scale range, and
all validation or metadata failures remain on CPU KLU. The thresholds may not be narrowed or
reinterpreted after CUDA results are observed without a new ADR decision and fresh evidence.

### Preservation and exclusions

GPU-01 adds no CUDA kernel, CUDA solver library, device allocation, upload/readback path, CUDA
dispatch, automatic backend selection, or GPU speedup claim. It adds no MOSFET or other device
semantics, nonlinear work, transient GPU work, mixed precision, MPI/NCCL/RAS, domain decomposition,
multi-node execution, or production parallel-CPU path. Phase 1--3C parser, compiler, MNA signs,
ordering, GMIN, frequency generation, symbolic reuse, DC/AC/transient/nonlinear execution, typed
failures, validation, and CSV contracts remain unchanged. Rust/wgpu remains immutable reference
material.

## GPU-02: Opt-in native-FP64 CUDA prepared linear AC

GPU-02 implements only one explicit CUDA executor behind the GPU-01 prepared-linear-AC seam. It
is a reproducible correctness and crossover experiment, not a production-dispatch decision. CPU
KLU remains unchanged, is the only supported no-GPU implementation, is the correctness authority,
and is the explicit whole-batch fallback. Ordinary `SimulateAc`, `SimulateAcToCsv`,
`SimulateToCsv`, parsing, compilation, result formatting, and production execution never construct
or select the GPU-02 executor.

### Algorithm, hermetic inputs, platform, and linkage

The sole CUDA sparse algorithm is NVIDIA cuDSS 0.8.0.10 general complex sparse direct
factorization and solve. Each matrix is `CUDSS_MTYPE_GENERAL`, the indexes are signed 32-bit CSR,
the data type is `CUDSS_C_64F`, the matrix view is full, and there is one complex-FP64 right-hand
side per batch member. cuDSS performs general complex LDU factorization. The prepared batch is one
cuDSS uniform batch: all members share the one canonical CSR structure while their values, RHS, and
solution vectors occupy contiguous member-major buffers. `CUDSS_CONFIG_UBATCH_SIZE` is the complete
prepared member count and `CUDSS_CONFIG_UBATCH_INDEX` remains at its `-1` default, so each numeric
phase processes the complete ordered batch in one library call. Serial per-member library calls or
`UBATCH_SIZE=1` are not a valid GPU-02 performance implementation because they leave the selected
library's native same-pattern batch parallelism unused.
GPU-02 uses one GPU, disables hybrid host execution, disables matching/scaling, requests one
native-FP64 iterative-refinement step, and uses the default reordering and factorization with the
automatic local-block pivot policy. These choices are the fixed GPU-02 execution configuration;
the preserved gate evidence does not claim that alternative cuDSS tuning combinations are
supported or globally optimal. GPU-01 requires bitwise repeatability only from the serial CPU
authority; every CUDA result remains subject to residual, fresh-KLU, and componentwise CPU
certification.
All factorization, refinement, and solve arithmetic remains native complex FP64. GPU-02 does not
call cuSOLVER, cuSPARSE factor APIs, a hand-written sparse factorization, or any real-valued,
split-complex, tensor-core, TF32, FP32, or mixed-precision path.

The only new vendor archive is the NVIDIA CUDA 13 Linux x86-64 cuDSS archive below. Bazel fetches
it directly; the URL, byte size, extraction prefix, SHA-256 digest, license, redistributable
manifest, and NVIDIA provenance are recorded in `third_party/cudss/PROVENANCE.md`.

```text
cuDSS version: 0.8.0.10
archive: libcudss-linux-x86_64-0.8.0.10_cuda13-archive.tar.xz
size: 157058452 bytes
sha256: ba18f5fd80dcbbe905d158caac5b3061d848442bb5abd477b5f296b4257a4937
redistributable manifest: https://developer.download.nvidia.com/compute/cudss/redist/redistrib_0.8.0.json
```

Only `libcudss_static.a`, `libcublas_static.a`, `libcublasLt_static.a`, and the pinned `culibos`
archive are linked for the cuDSS/cuBLAS graph. They come from the checksum-pinned cuDSS archive and
the cuBLAS component of the existing CUDA 13.0.2 redist toolchain; the CUDA runtime remains the
statically linked runtime selected by the CUDA build configuration. The final ELF `DT_NEEDED`
allowlist contains only declared glibc host-ABI components and rejects dynamic cudart, cuDSS,
cuBLAS, cuSOLVER, cuSPARSE, BLAS, libstdc++, libgcc, or any other undeclared library. The build must
not load an ambient CUDA toolkit or compiler. The host glibc and NVIDIA driver are platform ABIs,
not build inputs. The supported build platform is Linux x86-64 with the repository's pinned nvcc
and GCC 15.2.0 CUDA host toolchain; the device code retains the repository's `compute_120` PTX plus
`sm_120` cubin policy.
Execution additionally requires a driver and GPU that accept that code. CUDA/cudss headers,
libraries, handles, streams, events, allocations, and device ownership are confined to the CUDA
package and are absent from the backend-neutral GPU-01 interface.

### Immutable preparation, upload, execution, and lifetime

GPU-02 consumes a batch only after the unchanged GPU-01 version-1 validator accepts it. Dimensions,
row offsets, and column indexes must fit signed 32-bit cuDSS indexes without narrowing. The
executor owns one preparation generation at a time and keys it by the complete recomputed
contract/replay/batch/structure/member identity envelope. On first execution of a generation it:

1. creates one CUDA stream, timing events, cuDSS handle/configuration/data objects, and matrix
   wrappers owned by that executor;
2. converts and uploads the canonical CSR row offsets and column indexes exactly once;
3. packs and uploads every member's complex-FP64 values and RHS in unchanged member order; and
4. binds one cuDSS general-complex matrix view, one RHS, and one solution workspace and performs
   analysis once for the immutable structure.

No structure is copied or analyzed again while that exact generation remains prepared. The first
execution performs one uniform-batch numeric factorization call and subsequent executions perform
one uniform-batch numeric refactorization call. Every execution then performs one uniform-batch
solve call, synchronizes, and reads the contiguous member-major complex-FP64 solutions back in the
unchanged ordinal order. Per-member values and RHS are immutable device inputs for the generation;
repeated prepared executions reuse them as well as the structure and analysis. Logical
factorization/solve counters count accepted member systems, while separate library-call counters
must prove exactly one factor/refactor call and one solve call per complete batch execution. A
different envelope first destroys all matrix wrappers, cuDSS data/config/handle objects, events,
stream-owned buffers, and the stream, then creates a new generation. Destruction synchronizes
before release. The CUDA-specific owner-thread `Release` operation is the observable teardown
boundary: every CUDA/cuDSS cleanup status and a nonzero custom-allocator balance is a typed backend
failure. The destructor is a no-throw last resort; evidence and tests must call `Release`. Objects
are move-disabled, have one host-thread owner, and never share a CUDA context resource with an
ordinary simulator call.

### Synchronization, errors, acceptance, and fallback

Upload completes on the executor stream before analysis. Analysis, uniform-batch factorization or
refactorization, and uniform-batch solve are submitted in that order. GPU-02 synchronizes and
queries the independent cuDSS data-info status after each complete batch phase, treats any nonzero
CUDA or cuDSS status as failure, synchronizes before reading results, and synchronizes again before
reporting final timing and memory. A CUDA launch/status, cuDSS return
status, cuDSS asynchronous data-info status, allocation failure, event failure, stream failure,
device loss, readback failure, or exception becomes `kPreparedBackendFailure`. No library-reported
success, convergence, or solution count bypasses host validation. Failure releases or invalidates
the preparation generation; a later call must prepare it again.

The unchanged GPU-01 validator then checks, in order, the complete envelope and ordered member
associations; dimensions and finiteness; authoritative matrix/RHS normwise and componentwise
residual bounds; a fresh CPU KLU factorization and solve for each member; and the componentwise
`1e-12 + 1e-9*max(abs(x_cuda),abs(x_klu))` differential bound. Missing, duplicate, reordered,
partial, stale, incorrectly associated, non-finite, wrong-sized, or numerically hostile output
rejects the whole batch with the existing distinct typed failure. A CUDA result is never partly
published.

The caller chooses the existing fail-closed or explicit CPU-fallback policy. Fallback discards the
entire CUDA result and reruns every member through fresh CPU KLU, followed by the complete CPU
acceptance pass and observable `used_cpu_fallback=true`. GPU-02 neither retries a failed CUDA
member individually nor silently falls back. The CUDA-specific test seam may inject allocation,
library, and partial-result failures; allocation and status faults fire only after live preparation
work has begun and must invalidate that generation. The seam cannot weaken the production
validation boundary.

### Timing, synchronization, and memory accounting

All evidence phases are non-overlapping steady-clock intervals. Each complete backend `Execute`
wall is measured externally. Context initialization, CUDA/cuDSS object setup, host conversion and
packing, upload, matrix-wrapper setup, analysis submit/sync, factor/solve submit/sync, readback,
data-info status, memory queries, and result construction/association are classified subintervals.
`T_executor_overhead` is the exact nonnegative remainder of that complete wall and therefore
charges prepared-envelope validation, exception/fault checks, and timing bookkeeping rather than
dropping them. CUDA events record device elapsed telemetry but are not added to host intervals and
cannot replace the end-to-end total. Upload and readback use the executor stream and end at a
stream synchronization. Submit intervals include the host calls that invoke
analysis/factorization/solve; sync intervals include the corresponding waits and therefore device
execution. `T_validate` begins only after the complete readback and contains all fresh KLU
certification and differential work. For execution `i`:

```text
T_execute_i = T_context_setup_i + T_library_setup_i + T_host_pack_i
            + T_upload_i + T_matrix_setup_i
            + T_analysis_submit_i + T_analysis_sync_i
            + T_factor_solve_submit_i + T_factor_solve_sync_i
            + T_readback_i + T_phase_status_i + T_memory_accounting_i
            + T_result_assembly_i + T_executor_overhead_i

T_gpu_cold = T_prepare + T_executor_create + T_execute_1 + T_validate_1

T_gpu_prepared(R) = T_prepare + T_executor_create
                  + sum[1..R](T_execute_i + T_validate_i)
```

The one batch preparation, one upload, and one analysis are charged in both boundaries. Cold uses
`R=1`; prepared uses the replay-v1 reuse count. Parsing, netlist file I/O, compilation, evidence
process creation, and CSV remain outside every comparator exactly as in GPU-01. The CUDA executor
performs no KLU work. Every GPU sample separately reports CUDA factor/solve count, accepted member
count, validation KLU solve count, and zero hidden/priming KLU solves. Serial-authority and
parallel-host samples retain their GPU-01 boundaries and solve accounting. Owner-thread release
latency and post-release allocation balance are reported separately; release/destructor latency is
excluded symmetrically from both GPU and CPU comparator totals.

Device batch memory is the conservative maximum of (a) controlled structure/value/RHS/solution
buffers plus the greater of cuDSS's post-analysis peak-device estimate and the observed peak from
its accounted custom allocator, and (b) the observed decrease in `cudaMemGetInfo` free bytes
from the synchronized pre-generation baseline. Incremental stream/event/library residency after
primary-context initialization is reported separately; the CUDA runtime cannot observe a
pre-context free-memory baseline, so that field is not represented as complete primary-context
residency and is outside the batch gate. Host sample memory remains isolated-child `ru_maxrss`.
Allocation counters must return to zero at explicit generation teardown. The gate uses the maximum
conservative batch peak observed across both cold and prepared GPU modes, not a payload estimate or
an after-the-fact resident value.

### Reproducible evidence and frozen verdict

GPU-02 uses the unmodified `prepared-ac-replay-v1` manifest, meanings, identities, and GPU-01
thresholds. One canonical invocation contains all four cases and all six modes: deterministic
single-thread KLU cold/prepared, fair parallel-host KLU cold/prepared, and CUDA cold/prepared. It
uses at least three warmups and twenty retained raw samples for every case/mode. Two independent
complete invocations are preserved verbatim under `docs/evidence/`; neither may reuse raw samples
or omit a failing sample.

Each stream records all GPU-01 identity and completion fields plus the exact source and binary
fingerprints, cuDSS archive/version/hash, CUDA and cuBLAS package pins, compiler/Bazel/build flags,
CPU/kernel/hardware threads, GPU name/UUID/compute capability, driver/runtime versions, device
work and synchronization boundaries, solve and validation counts, throughput, latency quantiles,
peak CPU and GPU memory, CUDA/cudss/validation failures, and terminal completeness fingerprint.
Every raw CUDA sample reports all component phase times whose non-overlapping sum equals total.
The output fails closed on an identity, count, timing-sum, allocation-balance, validation,
differential, CUDA, cuDSS, memory, sample, summary, or completion mismatch.

For each invocation and each non-control class, the recorded verdict evaluates without adjustment:

```text
parallel_cpu_prepared_median / gpu_prepared_median >= 1.25
parallel_cpu_prepared_P95    / gpu_prepared_P95    >= 1.10
gpu_cold_median <= 1.10 * parallel_cpu_cold_median
gpu_peak_batch_memory <= 2147483648 bytes
```

The same class is eligible only if every bound passes in both complete invocations with zero
failures. `small_control` is always ineligible. A negative timing or memory result is a valid
GPU-02 experimental outcome and must be reported without changing the corpus, threshold, timing
boundary, or validation. Automatic CUDA selection remains unauthorized even for an eligible
class; a later ADR and reviewed implementation would be required. GPU-02 makes no production
speedup or generality claim.

### GPU-02 observed outcome

The first complete diagnostic used serial per-member cuDSS calls and its measured stream is
retained only for before/after comparison. Performance debugging identified that call shape as a
material defect:
the selected library provides native same-pattern uniform batching, while the diagnostic forced
`UBATCH_SIZE=1`. The final executor uses the complete member count and reduces prepared CUDA-event
factor/solve medians by 2.5x--22.8x across the four frozen cases. The two final evidence streams
share source fingerprint `v1-404c729f9594460a6dcd5497e7ef8823`, binary fingerprint
`v1-08f2e8cfef866e2d63c2334550bf8cba`, exact solve/call accounting, and zero failures.

The correction did not make a class eligible. Across the two final runs, prepared median ratios
were 0.0527--0.0538 (small/control), 0.1610--0.1611 (medium), 0.5027--0.5028 (large), and
0.4122--0.4134 (medium/wide); cold ratios were 116.89--118.87, 18.15--18.18, 2.718--2.739, and
3.517--3.535 respectively. All P95 timing bounds failed. Peak GPU batch memory was 32--64 MiB, so
every memory bound passed. The exact values, whole-file hashes, completion fingerprints, and raw
phase records are preserved in
`docs/evidence/gpu02-native-fp64-cuda-uniform-batch-summary-2026-08-08.md` and its two referenced
CSV streams.

The remaining loss is not attributed solely to sparse numeric work. Each frozen fresh-child sample
charges roughly 215--220 ms of CUDA primary-context initialization, and complete CPU certification
costs roughly 142 ms for the prepared wide ring and 226 ms for the prepared grid. Assuming all GPU
execution, setup, transfer, and readback time were zero, preparation plus mandatory validation
alone limits the observed median ratio to about 1.14 for the grid and 1.15 for the ring, below the
frozen 1.25 threshold. This evaluates rather than redefines the gate. It does not authorize hiding
context startup, removing CPU certification, altering replay reuse, or starting automatic
dispatch.

## GPU-02S: Persistent-session linear-AC crossover evidence

GPU-02S is a bounded evidence follow-up to the completed GPU-02 experiment. It answers whether
native-FP64 CUDA is useful for Ohmnivore's repeated linear-AC use case: one long-lived process,
one CUDA primary context, and multiple same-topology frequency/corner batches whose matrix values
and right-hand sides change while the canonical sparse structure remains fixed. GPU-02S does not
reinterpret, replace, or delete `prepared-ac-replay-v1`, either preserved GPU-02 invocation, or any
frozen GPU-02 threshold or verdict. It does not authorize ordinary simulator routing, automatic
dispatch, a production acceptance policy, or a general speedup claim.

### Session workload and immutable-structure lifetime

The separately versioned `prepared-ac-session-v1` manifest describes compiler-derived linear RLC
circuits, not arbitrary sparse matrices. Each declared case fixes its circuit topology, dimensions,
canonical union nonzeros, frequency grid, corner count, component scales, and deterministic corner
transform. The generator constructs the public `Circuit` IR and calls the authoritative
`CompileMna` and `PrepareLinearAcBatch` paths. Parsing and netlist file I/O remain excluded because
they are backend-independent and no large external/customer netlist is represented. Evidence must
therefore say `compiler-derived synthetic session envelope`, not `customer workload` or
`production representative`.

A session owns one backend instance. A topology generation is identified by the complete validated
prepared structure fingerprint and uniform-batch member count. Within a topology generation:

1. canonical row offsets and column indices are converted and uploaded once;
2. the CUDA stream, events, cuDSS handle/configuration/data objects, matrix wrappers, and allocated
   structure/value/RHS/solution buffers remain owned by the executor;
3. cuDSS analysis runs once for the immutable sparse structure;
4. each later corner batch must have the same structure fingerprint, dimensions, nonzeros, member
   count, and buffer sizes but a newly validated batch/member/content envelope;
5. every later corner repacks and uploads all complex-FP64 matrix values and RHS values, preserves
   the structure and analysis, performs exactly one uniform-batch numeric refactorization and one
   solve, synchronizes, reads back, and associates the complete new result envelope; and
6. any structure, member-count, size, validation, upload, CUDA, cuDSS, allocation, or cleanup
   mismatch fails closed. A different topology takes the existing full-generation teardown path.

The executor exposes cumulative structure preparations/uploads/analyses separately from value/RHS
refreshes. A successful same-structure refresh must increase the refresh counter without increasing
the structure-upload or analysis counter. An exact repeated batch performs neither upload. CUDA
types and ownership remain confined to the CUDA implementation. The CPU KLU authority and fallback
are unchanged.

### Fair persistent CPU comparator

The session performance competitor is benchmark-only. It creates one fixed worker pool before the
timed session, gives each worker private KLU state, and caches symbolic analysis by the validated
structure fingerprint rather than by matrix values or RHS. A same-structure corner performs fresh
numeric factor/refactor and solve work without reconstructing worker threads or symbolic state. A
new structure invalidates every worker's symbolic state. Work assignment and result slots retain
member order, and solve/analysis counts must reconcile exactly. This comparator remains unreachable
from ordinary simulation.

### Correctness qualification and two timing lanes

Every CUDA result in every lane must pass the complete GPU-01 association, dimension, finiteness,
residual/backward-error, fresh-KLU, and componentwise differential checks before the evidence run
can complete. CUDA status, cuDSS status, or a residual-only result never becomes an accepted
Ohmnivore result.

GPU-02S reports two non-overlapping interpretations of the same correctly certified executions:

- `inline_certified` includes complete fresh KLU certification before the result is released and
  answers the performance of the current experimental acceptance contract; and
- `candidate_runtime` times backend execution plus the GPU-01 envelope/association/dimension/
  finiteness/residual/backward-error checks, then performs the mandatory fresh KLU certification
  outside that candidate interval. A later certification failure invalidates the complete sample
  and evidence stream. This lane isolates technical solver potential only; it is not a production
  acceptance policy and cannot support dispatch without a later ADR.

The evidence-only residual validator is named and documented as such and is exposed only by the
test-only prepared-session support target, not the public `//cpp:core` API. It is not called by
`ExecutePreparedAcBatch`, `SimulateAc`, CSV output, or fallback, and must reject every hostile
envelope, association, dimension, non-finite, and excessive-residual class that can be rejected
without an independent solve. Only full `ValidatePreparedAcBatchResult` remains an acceptance
boundary. The public CPU KLU backend retains GPU-01's exact-batch symbolic-cache key; the
structure-keyed cache exists only inside the benchmark-only persistent comparator.

### Session timing, sampling, and crossover reporting

For one topology with `C` declared corner batches, a fresh process records:

```text
T_gpu_session_cold = T_prepare_all
                   + T_executor_create + T_context_once + T_library_once
                   + T_structure_upload_once + T_analysis_once
                   + sum[1..C](T_values_rhs_upload + T_factor_solve
                              + T_sync + T_readback + T_runtime_validate)

T_cpu_session = T_prepare_all + T_worker_pool_create
              + T_symbolic_once_per_worker
              + sum[1..C](T_numeric_solve + T_runtime_validate)
```

`inline_certified` substitutes complete fresh-KLU validation for
`T_runtime_validate`. `candidate_runtime` records the excluded certification interval and solve
count separately; it may not hide that work in a total labeled certified. Process creation,
parsing, file I/O, evidence serialization, and explicit final teardown remain outside both CPU and
GPU intervals. Teardown latency and allocation balance are reported separately.

Each complete invocation records at least three untimed warmup sessions and twenty raw sessions per
case/backend/lane. Cold session totals use fresh child processes and include one primary-context
initialization. The same child also records first-corner latency and the later-corner steady-state
distribution after structure analysis. A separate persistent-process diagnostic for both the fixed
CPU worker pool and CUDA executor records one cold session followed by at least twenty timed steady
sessions in the same process. CPU steady sessions retain the worker pool and per-worker symbolic
analysis; CUDA steady sessions retain the primary context, library objects, allocations, structure,
and analysis. The diagnostic applies the same lane definitions and reports a CPU/GPU steady-session
ratio, but it is not substituted for the fresh-session total. Two complete independent invocations
are preserved.

For each case and lane, evidence reports CPU/GPU session median and P95 ratios, accepted-member
throughput, first-result latency, later-corner median/P95, the exact corner and member counts,
structure analyses/uploads, value/RHS refreshes, factor/solve calls, validation/certification solves,
and peak CPU/GPU memory. It also reports the smallest observed corner/member count whose replicated
median ratio is at least `1.25` and P95 ratio is at least `1.10`, if one exists. Cold latency is a
separate operational result rather than a conjunctive steady-state gate. The 2 GiB peak batch-memory
bound remains mandatory. A crossover is a property only of the exact compiler-derived case,
hardware, lane, and measured session envelope; it is not automatic-dispatch eligibility.

The v1 session corpus contains one permanently ineligible 64-point control and three large-sweep
candidate shapes: a 33-by-33 grid with 512 frequency points, a 65-by-65 grid with 256 points, and a
1,024-node four-source ring with 2,048 points. Every candidate contains four same-structure
component/source corners. The batches were selected before canonical evidence from the bounded
memory-safe upper envelope, not in response to an observed gate result. Counts and transforms are
checksum-bound in the manifest and tested before evidence. GPU-02S may report that no crossover
exists. Thresholds, identities, raw samples, or failing points may not be removed or redefined
after measurement.

### GPU-02S exclusions

GPU-02S changes no Rust/Cargo source, ordinary `SimulateAc`, CSV output, supported CPU behavior,
linear-AC semantics, or GPU-02 replay/evidence bytes. It adds no automatic or production-default
dispatch, universal solver/backend abstraction, nonlinear device semantics, Newton work, transient
GPU execution, mixed precision, MPI/NCCL/RAS, multi-node execution, domain decomposition, or
downstream MOSFET/CMOS GPU work. Native complex FP64 cuDSS uniform batching remains the only CUDA
algorithm.

### GPU-02 preservation and exclusions

GPU-02 changes none of Phase 1--3C or GPU-01 semantics and does not edit Rust/Cargo. It adds no
MOSFET or other nonlinear model, Newton work, transient GPU execution, mixed precision,
automatic or production-default CUDA dispatch, CUDA CSV path, universal solver/backend/device
abstraction, MPI/NCCL/RAS, domain decomposition, multi-node execution, multi-GPU execution, or
downstream GPU work. NL-04 and every later GPU phase remain unstarted.

## Follow-up epics

Post-Phase 3C work is divided into three independently reviewable epics. Their detailed planning
contract is recorded in [the C++/CUDA roadmap](../roadmap.md). Recording an epic does not authorize
its implementation; each epic requires an exact bounded ADR contract before code changes begin.

1. **NL-04: Deterministic FP64 CPU MOSFET DC authority.** Add a strict minimal NMOS/PMOS DC model
   on the existing Newton and KLU path, with an explicit terminal/bulk policy, analytic and ngspice
   evidence, and a representative CMOS/corner corpus. It does not add transient charge, AC/noise,
   temperature behavior, CUDA, or another solver.
2. **GPU-01: Prepared workload and evidence foundation.** Define a backend-neutral prepared-batch
   contract, CPU KLU reference and fallback, replay corpus, hostile-result boundary, parallel CPU
   performance baseline, complete timing model, and a falsifiable crossover hypothesis. It adds no
   CUDA circuit kernel or dispatch.
3. **GPU-02: Native FP64 CUDA batched-AC vertical slice.** Implement one opt-in CUDA path for the
   already-authoritative linear AC semantics using immutable uploaded structure, CPU validation,
   differential replay, and end-to-end benchmarks. CUDA is not dispatch-eligible unless it beats
   the crossover thresholds frozen before implementation without weakening correctness.

GPU-02 depends on GPU-01. NL-04 does not block the batched-linear-AC epics, but it and GPU-02 both
precede any future nonlinear MOSFET/CMOS CUDA proposal. Evaluate batched AC points, parameter
corners, Monte Carlo runs, and independent circuits before considering single-circuit domain
decomposition.

Performance evidence compares complete GPU time---preparation, upload, device work,
synchronization, readback, and CPU validation---with a fair parallel host baseline for independent
KLU solves. The deterministic single-thread KLU path remains the correctness authority, but it is
not the sole performance comparator. Kernel-only timing or comparison only against the
single-thread oracle cannot justify automatic CUDA dispatch.

## Consequences

The project temporarily carries two implementations, and Phase 1 supports less SPICE syntax than
the legacy prototype. That duplication is intentional: it keeps the new contracts reviewable and
prevents incomplete feature parity from being mistaken for correctness.

The decision narrows GPU portability to gain numerical control, direct CUDA tooling, and alignment
with the project's actual deployment hardware. Future portability is an explicit backend decision,
not a constraint on the core circuit model.
