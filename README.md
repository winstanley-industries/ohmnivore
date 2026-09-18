# Ohmnivore

Ohmnivore is a GPU-accelerated circuit simulation project that parses a SPICE subset, compiles
Modified Nodal Analysis (MNA) systems, and solves DC, AC, and transient analyses.

## Migration status

Ohmnivore is migrating from its original Rust/wgpu prototype to a C++20 core with a CUDA-first
GPU backend. The accepted rationale, invariants, and phased plan are recorded in
[ADR-001](docs/adr/ADR-001-cpp-cuda-migration.md).

The C++ path preserves deterministic linear DC, AC, and transient analysis plus bounded
nonlinear diode DC operating-point and memoryless-diode transient analysis and strict BJT DC
operating-point analysis on the production FP64 CPU sparse-direct correctness path. It also adds
an explicit prepared linear-AC evidence boundary without changing ordinary simulation. It
provides:

- hermetic Bazel C++ and opt-in CUDA toolchains;
- typed domain errors at library boundaries;
- resistor, capacitor, inductor, and independent voltage/current-source parsing, with sources
  accepting strict DC, AC, PULSE, SIN, PWL, and EXP forms in legacy order; every source must
  specify DC, AC, and/or transient excitation, and passive values must be positive;
- `.DC`/`.OP`, validated `.AC DEC|OCT|LIN`, and validated
  `.TRAN tstep tstop [tstart] [UIC]` execution, with `.PRINT` accepted as a compatibility no-op
  because the CLI emits every solved variable;
- insertion-ordered Circuit IR, deterministic non-ground node order, and one interleaved
  insertion-ordered voltage-source/inductor branch sequence;
- deterministic independent CSR conductance (`G`) and dynamic (`C`) matrices plus real DC and
  complex AC right-hand sides;
- canonical current-source, capacitor, and inductor MNA stamps, including capacitor-open and
  inductor-short operating-point behavior;
- checksum-pinned SuiteSparse KLU 2.3.6 real and complex FP64 sparse-direct solves for DC,
  `A(omega) = G + j * omega * C`, transient companion, UIC, and discontinuity-projection systems;
- deterministic CSR-to-CSC conversion, fixed AMD/BTF/pivot/scaling/single-thread policy,
  symbolic-analysis reuse, pivot-safe numeric refactorization, and finite normwise plus rowwise
  componentwise backward-error checks;
- strict diode instances and case-insensitive `.MODEL ... D(IS=... N=...)` lookup with deterministic
  defaults, typed malformed/unsupported/model failures, and one fixed explicit-zero CSR union
  pattern for every linear and possible diode Jacobian stamp;
- FP64 Shockley current/conductance evaluation at fixed `VT=0.02585 V`, deterministic PN-junction
  limiting, Newton update plus nonlinear-residual convergence, fixed DC direct/source/GMIN strategy
  order, original-system final validation, and reused KLU symbolic analyses;
- strict five-token BJT instances and case-insensitive NPN/PNP models with only
  `IS/BF/BR/NF/NR`, legacy-compatible FP64 Ebers--Moll currents, full 3-by-3 Jacobian stamping,
  deterministic BE/BC limiting, and typed rejection of BJT AC or transient execution;
- inclusive, strictly increasing AC frequency grids: LIN emits exactly its total point count,
  while DEC/OCT use points per decade/octave and include the exact stop frequency once;
- backward Euler for initial/recovery/breakpoint steps, trapezoidal integration otherwise,
  deterministic scaled local-error control on both methods, left/right source-discontinuity
  projection, exact waveform/output-start/stop boundaries, nonlinear companion residual validation,
  deterministic half-step Newton retry, and bounded typed timestep failures;
- linear or nonlinear DC operating-point initialization without UIC and deterministic zero
  capacitor-voltage/zero inductor-current constraints with UIC;
- the legacy DC, AC, and transient CSV schemas;
- a versioned backend-neutral prepared linear-AC batch contract with immutable canonical sparse
  structure, ordered value/RHS members, content-bound circuit/corner/frequency identities, typed
  hostile-result rejection, mandatory CPU KLU certification, and explicit whole-batch CPU KLU
  fallback;
- a byte- and identity-pinned four-class replay corpus spanning path/tree/grid/ring sparsity,
  dimensions 65--1025, 192--4994 stored entries, batches 16--517, broad FP64 value/frequency
  ranges, one-to-four source branches, and preparation reuse counts 2--8;
- a manual CPU-only evidence target containing the deterministic serial KLU authority and an
  isolated, solve-counted parallel-host KLU comparator with complete cold/prepared timing, raw
  identities/metadata, per-sample child-process memory, and a terminal completeness record;
- one manual native-complex-FP64 CUDA prepared-AC executor using checksum-pinned static cuDSS and
  cuBLAS, with one full native uniform batch per factor/solve phase, immutable structure
  upload/reuse, CPU KLU certification, hostile-failure coverage, explicit full-batch fallback,
  checked owner-thread teardown, and no ordinary simulator routing;
- two complete GPU-02 replay-v1 evidence invocations comparing cold/prepared CUDA with both serial
  and parallel-host KLU; native batching improved prepared factor/solve device time by 2.5x--22.8x
  over the preserved per-member diagnostic, but fresh-process context initialization and mandatory
  CPU certification still leave every frozen timing class ineligible, so this is a correct negative
  experiment rather than a production speedup claim or dispatch authorization;
- a bounded GPU-02S persistent-session experiment using compiler-derived 64--2,048 point linear-AC
  sweeps, same-structure value/RHS refresh without structure re-upload or re-analysis, persistent
  CPU and CUDA owners, separate inline-certified and candidate-runtime lanes, and both honest
  fresh-session and long-lived-process evidence; and
- a checksum-pinned Bazel-built ngspice 46 acceptance harness for representative linear DC, AC,
  and transient circuits plus bounded forward-biased diode DC, memoryless-diode transient, and BJT
  DC fixtures; and
- a deterministic CUDA platform smoke test checked against a CPU oracle.

Native MOSFET primitives, BJT transient/charge/AC/noise/temperature behavior, and diode AC/charge/noise/temperature
behavior have not yet been ported. Ordinary CUDA solver dispatch, CUDA device-model kernels, mixed
precision, and distributed solving are also absent. GPU-02 does not alter the ordinary
single-thread CPU execution path or make a production performance claim. Bare `.DC` is an
operating-point request; DC source sweeps are not executed.
Unsupported input is rejected explicitly, malformed input is reported separately, and the Rust
implementation remains in `src/` and `tests/` as a behavioral reference until C++ parity is
accepted. The ordinary frontend acceptance fixtures cover representative linear circuits, one diode
DC fixture, one memoryless-diode transient fixture, one BJT DC fixture, and tolerances recorded in
`third_party/ngspice/PROVENANCE.md`. Sparse-solver selection,
storage, numerical, license, and reproducibility details are recorded in
`third_party/suitesparse/PROVENANCE.md`.

## Next direction: inverter output-filter design studies

The accepted next direction is evaluating many combined common-mode and differential-mode output
filter designs for switching motor-drive inverters, seeking lower mass with sufficient predicted
EMI margin. The unit of parallel work is a complete coupled circuit for one filter candidate and
operating/tolerance corner. The primary metric is validated candidate/corner evaluations per hour
against a fair parallel CPU baseline at fixed accuracy.

EMI-01 supplies the frozen external reference, measurement and mass contracts, and CPU runtime
breakdown. EMI-02 adds disjoint coupled-inductor pairs, bounded behavioral expressions and
Jacobians, simultaneous controlled displacement-current transient equations, and an opt-in bridge
for the exact reference SiC model. Its [qualification harness](reference/emi02/README.md) compares
all thirty frozen trajectories with ngspice and fails closed on incomplete or inaccurate jobs.
See the [EMI-02 implementation and qualification results](docs/emi02-results.md).
Full CPU qualification passes: all sixty CPU/reference jobs, eighty refinement checks and thirty
differential comparisons pass within the original resource limits. The earlier failed run is retained.
The ordinary parser does not accept general behavioral sources or vendor model libraries.
EMI-03 adds an [opt-in GPU transient ensemble experiment](reference/emi03/README.md)
with native-FP64 expression evaluation and real sparse solves, persistent CPU/GPU
workers and fresh CPU validation. Its [bounded contract](docs/adr/ADR-008-emi03-transient-ensembles.md)
retains the frozen accuracy and complete-study performance gates. The
[retained experiment results](docs/emi03-results.md) document its resource rejection;
EMI-03 remains incomplete until every qualification, resource and performance target
passes. No GPU speedup is established. Exact aviation requirements and laboratory correlation
remain to be established. See [ADR-002](docs/adr/ADR-002-inverter-emi-design-study.md) and
[the C++/CUDA roadmap](docs/roadmap.md) for scope, dependencies, and acceptance gates.

GPU-01, GPU-02, and GPU-02S remain completed prepared-linear-AC experiments. Neither GPU-02's native
cuDSS uniform batching nor GPU-02S's persistent sessions met the frozen end-to-end performance
gates. Their historical evidence and [ADR-001](docs/adr/ADR-001-cpp-cuda-migration.md) contracts are
preserved. This roadmap update does not refresh those measurements or enable ordinary CUDA
execution. Automatic dispatch remains unauthorized.

## Build and test the C++ path

Install Bazelisk as `bazel` on `PATH`; all compilers, headers, libraries, and lint tools are
downloaded from pinned, checksum-verified dependencies.

```sh
bazel lint
bazel build //...
bazel test //...
bazel test //cpp:phase3a_test
bazel test //cpp:phase3b_test
bazel test //cpp:phase3c_test
bazel test //cpp:gpu01_prepared_ac_test //cpp:gpu01_replay_test
bazel test //acceptance:ngspice_acceptance_test
bazel test //cpp:solver_hermeticity_test
bazel run -c opt //cpp:solver_selection_benchmark -- --warmups=3 --repetitions=15
bazel run -c opt //cpp:prepared_ac_replay_benchmark -- --warmups=2 --repetitions=9
bazel run //:ohmnivore -- examples/voltage_divider.spice
```

Expected CSV columns are compatible with the legacy implementation:

```csv
Variable,Value
V(in),10
V(mid),4.9999999975
I(V1),-0.005000000012499999
```

The small difference from the ideal 5 V / -5 mA values is the intentional `1e-12 S` GMIN
conductance applied to each non-ground node, matching the existing MNA convention.

Run the opt-in CUDA smoke test on the reference NVIDIA platform:

```sh
bazel test --config=cuda //:cuda_smoke_test
bazel test --config=cuda //:gpu02_cuda_test
```

See [Building Ohmnivore](docs/BUILDING.md) for sanitizer, lockfile, and hermeticity details.

## Legacy Rust prototype

The legacy implementation supports a broader SPICE subset and contains the current research
implementations of:

- resistors, capacitors, inductors, voltage/current sources, diodes, BJTs, and MOSFETs;
- DC, AC, and linear transient analyses;
- wgpu BiCGSTAB and double-single nonlinear kernels;
- CPU dense and sparse-LU fallbacks; and
- experimental MPI/RAS domain decomposition.

It is retained for semantic reference, fixtures, and differential migration tests. It is not the
target architecture for new solver development.

If a Rust toolchain is installed, its historical commands remain:

```sh
cargo build
cargo test
cargo test --features ngspice-compare
```

## Documentation

- [ADR-001: Migrate the Solver Core to C++20 and CUDA](docs/adr/ADR-001-cpp-cuda-migration.md)
- [ADR-002: Batched Inverter Output-Filter Design Studies](docs/adr/ADR-002-inverter-emi-design-study.md)
- [C++/CUDA Roadmap](docs/roadmap.md) — reference study, CPU qualification, and GPU ensemble stages
- [Building Ohmnivore](docs/BUILDING.md)
- [Netlist Format](docs/netlist-format.md) — active C++ subset and identified legacy-only forms
- [Analysis Types](docs/analyses.md) — active C++ behavior and identified legacy-only behavior
- [Solver Reference](docs/solver-reference.md) — legacy Rust/wgpu architecture
