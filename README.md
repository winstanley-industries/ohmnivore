# Ohmnivore

Ohmnivore is a GPU-accelerated circuit simulation project that parses a SPICE subset, compiles
Modified Nodal Analysis (MNA) systems, and solves DC, AC, and transient analyses.

## Migration status

Ohmnivore is migrating from its original Rust/wgpu prototype to a C++20 core with a CUDA-first
GPU backend. The accepted rationale, invariants, and phased plan are recorded in
[ADR-001](docs/adr/ADR-001-cpp-cuda-migration.md).

The C++ Phase 2C path is intentionally limited to deterministic linear DC, AC, and transient
analysis on the FP64 CPU correctness path. It
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
- deterministic dense real and complex FP64 partial-pivoting CPU reference solves for
  `.DC`/`.OP` and `A(omega) = G + j * omega * C`;
- inclusive, strictly increasing AC frequency grids: LIN emits exactly its total point count,
  while DEC/OCT use points per decade/octave and include the exact stop frequency once;
- backward Euler for initial/recovery/breakpoint steps, trapezoidal integration otherwise,
  deterministic scaled local-error control on both methods, left/right source-discontinuity
  projection, exact waveform/output-start/stop boundaries, and bounded typed timestep failures;
- DC operating-point initialization without UIC and deterministic zero capacitor-voltage/zero
  inductor-current constraints with UIC;
- the legacy DC, AC, and transient CSV schemas;
- a checksum-pinned Bazel-built ngspice 46 acceptance harness for representative linear DC, AC,
  and transient circuits; and
- a deterministic CUDA platform smoke test checked against a CPU oracle.

Nonlinear devices and nonlinear transient analysis, production sparse-direct solver selection,
CUDA circuit kernels or solver dispatch, mixed precision, and distributed solving have not yet
been ported. Bare `.DC` is an operating-point request; DC source sweeps are not executed.
Unsupported input is rejected explicitly, malformed input is reported separately, and the Rust
implementation remains in `src/` and `tests/` as a behavioral reference until C++ parity is
accepted. The differential claim is limited to the representative linear fixtures and tolerances
recorded in `third_party/ngspice/PROVENANCE.md`.

## Build and test the C++ path

Install Bazelisk as `bazel` on `PATH`; all compilers, headers, libraries, and lint tools are
downloaded from pinned, checksum-verified dependencies.

```sh
bazel lint
bazel build //...
bazel test //...
bazel test //acceptance:ngspice_acceptance_test
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
- [Building Ohmnivore](docs/BUILDING.md)
- [Netlist Format](docs/netlist-format.md) — legacy prototype coverage
- [Analysis Types](docs/analyses.md) — legacy prototype coverage
- [Solver Reference](docs/solver-reference.md) — legacy Rust/wgpu architecture
