# Ohmnivore

Ohmnivore is a GPU-accelerated circuit simulation project that parses a SPICE subset, compiles
Modified Nodal Analysis (MNA) systems, and solves DC, AC, and transient analyses.

## Migration status

Ohmnivore is migrating from its original Rust/wgpu prototype to a C++20 core with a CUDA-first
GPU backend. The accepted rationale, invariants, and phased plan are recorded in
[ADR-001](docs/adr/ADR-001-cpp-cuda-migration.md).

The C++ Phase 3C path provides deterministic linear DC, AC, and transient analysis plus bounded
nonlinear diode DC operating-point and memoryless-diode transient analysis and strict BJT DC
operating-point analysis on the production FP64 CPU sparse-direct correctness path. It provides:

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
- a checksum-pinned Bazel-built ngspice 46 acceptance harness for representative linear DC, AC,
  and transient circuits plus bounded forward-biased diode DC, memoryless-diode transient, and BJT
  DC fixtures; and
- a deterministic CUDA platform smoke test checked against a CPU oracle.

MOSFETs, BJT transient/charge/AC/noise/temperature behavior, and diode AC/charge/noise/temperature
behavior have not yet been ported. CUDA circuit kernels or solver dispatch, mixed precision, and
distributed solving are also absent. Bare `.DC` is an operating-point request; DC source sweeps are
not executed.
Unsupported input is rejected explicitly, malformed input is reported separately, and the Rust
implementation remains in `src/` and `tests/` as a behavioral reference until C++ parity is
accepted. The differential claim is limited to the representative linear fixtures, one diode DC
fixture, one memoryless-diode transient fixture, one BJT DC fixture, and tolerances recorded in
`third_party/ngspice/PROVENANCE.md`. Sparse-solver selection,
storage, numerical, license, and reproducibility details are recorded in
`third_party/suitesparse/PROVENANCE.md`.

## Post-Phase 3C roadmap

Planned work is split into three bounded epics: deterministic FP64 CPU MOSFET DC authority
(`NL-04`), a backend-neutral prepared-workload and evidence foundation (`GPU-01`), and one native
FP64 CUDA batched-AC vertical slice (`GPU-02`). GPU-02 depends on GPU-01; MOSFET DC is independent
of batched linear AC but is required before any later MOSFET/CMOS CUDA work. The performance thesis,
parallel CPU comparison, full timing boundary, dispatch gate, scope exclusions, and dependency
details are recorded in [the C++/CUDA roadmap](docs/roadmap.md). Planning these epics does not
authorize their implementation or make their behavior supported.

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
bazel test //acceptance:ngspice_acceptance_test
bazel test //cpp:solver_hermeticity_test
bazel run -c opt //cpp:solver_selection_benchmark -- --warmups=3 --repetitions=15
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
- [C++/CUDA Roadmap After Phase 3C](docs/roadmap.md) — planned epic boundaries and GPU evidence gate
- [Building Ohmnivore](docs/BUILDING.md)
- [Netlist Format](docs/netlist-format.md) — active C++ subset and identified legacy-only forms
- [Analysis Types](docs/analyses.md) — active C++ behavior and identified legacy-only behavior
- [Solver Reference](docs/solver-reference.md) — legacy Rust/wgpu architecture
