# Ohmnivore

Ohmnivore is a GPU-accelerated circuit simulation project that parses a SPICE subset, compiles
Modified Nodal Analysis (MNA) systems, and solves DC, AC, and transient analyses.

## Migration status

Ohmnivore is migrating from its original Rust/wgpu prototype to a C++20 core with a CUDA-first
GPU backend. The accepted rationale, invariants, and phased plan are recorded in
[ADR-001](docs/adr/ADR-001-cpp-cuda-migration.md).

The first C++ phase is intentionally narrow. It provides:

- hermetic Bazel C++ and opt-in CUDA toolchains;
- typed domain errors at library boundaries;
- resistor and independent DC voltage-source parsing, with `.PRINT` accepted as a compatibility
  no-op because the CLI emits every solved variable;
- insertion-ordered Circuit IR and CSR MNA compilation;
- a deterministic FP64 CPU reference solve for `.DC`/`.OP`;
- the legacy DC CSV schema; and
- a deterministic CUDA platform smoke test checked against a CPU oracle.

Capacitors, inductors, current sources, nonlinear devices, AC, transient analysis, CUDA solver
kernels, and distributed solving have not yet been ported. Unsupported input is rejected
explicitly. The Rust implementation remains in `src/` and `tests/` as a behavioral reference
until C++ parity is accepted.

## Build and test the C++ path

Install Bazelisk as `bazel` on `PATH`; all compilers, headers, libraries, and lint tools are
downloaded from pinned, checksum-verified dependencies.

```sh
bazel lint
bazel build //...
bazel test //...
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
