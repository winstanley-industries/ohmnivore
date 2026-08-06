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

## Follow-up phases

1. Extend the CPU nonlinear authority only through separately bounded additional-device or
   charge-storage phases.
2. Add one CUDA vertical slice with immutable uploaded structure, native `double`, hostile result
   validation, replay, and end-to-end benchmarks.
3. Evaluate batched AC points, parameter corners, Monte Carlo runs, and independent circuits before
   considering single-circuit domain decomposition.

## Consequences

The project temporarily carries two implementations, and Phase 1 supports less SPICE syntax than
the legacy prototype. That duplication is intentional: it keeps the new contracts reviewable and
prevents incomplete feature parity from being mistaken for correctness.

The decision narrows GPU portability to gain numerical control, direct CUDA tooling, and alignment
with the project's actual deployment hardware. Future portability is an explicit backend decision,
not a constraint on the core circuit model.
