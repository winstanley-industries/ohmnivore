# Ohmnivore

GPU-accelerated circuit simulation solver. The active implementation target is a C++20 core with
a CUDA-first GPU backend; the original Rust/wgpu solver remains in-tree as a migration reference.

## Active C++/CUDA migration authority

ADR-001 (`docs/adr/ADR-001-cpp-cuda-migration.md`) is the authority for new work. Ohmnivore is
migrating to a C++20 core with a CUDA-first backend and hermetic Bazel toolchains. The Rust/wgpu
implementation is retained as a behavioral reference, not the target architecture for new solver
development.

Phase 3B is deliberately bounded to deterministic FP64 CPU nonlinear transient analysis for the
strict, memoryless Phase 3A `D`/`.MODEL ... D(IS=... N=...)` diode subset. SuiteSparse KLU 2.3.6
from the checksum-pinned SuiteSparse 7.12.3 archive remains the only production linear solve path
and reuses symbolic analysis for each fixed nonlinear union pattern. The former dense
partial-pivoting implementation remains a test-only exact-small oracle. Do not pull BJTs, MOSFETs,
diode AC/charge/noise/temperature work, CUDA circuit kernels, mixed precision, distributed solving,
or semantic changes to linear DC, AC, transient, waveforms, GMIN, ordering, signs, or CSV into
Phase 3B.

Before changing the C++ path:

1. Read ADR-001 and `docs/BUILDING.md`.
2. Preserve the CPU FP64 path as the correctness oracle and supported no-GPU implementation.
3. Treat CUDA results as untrusted until CPU differential checks accept them.
4. Use Bazel as the canonical build interface; never use system CUDA, nvcc, GCC, Clang, headers, or
   libraries.
5. Return typed errors for unsupported input rather than silently weakening semantics.
6. Keep the Rust source intact until a documented parity decision retires it.

Canonical C++ validation:

```sh
bazel lint
bazel build //...
bazel test //...
bazel test --config=asan //...
bazel test --config=ubsan //...
bazel test --lockfile_mode=error //...
bazel test --config=cuda //:cuda_smoke_test
bazel test //acceptance:ngspice_acceptance_test
git diff --check
```

Verify the declared CUDA/sanitizer incompatibility with explicit targets so it is an analysis
failure rather than a skipped test-suite member:

```sh
bazel build --config=cuda --config=asan //cuda:smoke_test
bazel build --config=cuda --config=ubsan //cuda:smoke_test
```

The architecture and commands below describe the legacy Rust prototype unless stated otherwise.

## Architecture

```
Netlist (.spice) -> Parser (nom) -> Circuit IR -> MNA Compiler -> Solver -> Analysis Results (CSV)
```

### Module Map

| Module | Purpose |
|---|---|
| `parser` | SPICE-subset netlist parser (R, L, C, V, I, D, Q, M, `.DC`, `.AC`) |
| `ir` | Circuit intermediate representation (components, analyses, models) |
| `compiler` | Builds MNA G/C matrices and b vectors in CSR sparse format |
| `sparse` | Generic CSR sparse matrix (f64 and Complex64) |
| `solver/` | Linear solvers (CPU direct, GPU BiCGSTAB), nonlinear Newton-Raphson, distributed RAS |
| `analysis/` | DC operating point and AC frequency sweep engines |
| `output` | CSV result formatting |
| `error` | `OhmnivoreError` enum via `thiserror`, project-wide `Result<T>` alias |

### Solver Traits

- `LinearSolver` — main interface (`solve_real`, `solve_complex`)
- `SolverBackend` — GPU-agnostic vector/matrix ops (SpMV, dot, AXPY)
- `NonlinearBackend` — GPU-resident diode/BJT/MOSFET evaluation for Newton-Raphson
- `CommunicationBackend` — inter-process communication (all-reduce, halo exchange). Implementations: `SingleProcessComm` (no-op), `MpiComm` (behind `distributed` feature)
- `Partitioner` — graph partitioning for domain decomposition. Implementation: `MetisPartitioner` (via `metis-rs`)
- `DistributedPreconditioner` — distributed preconditioner apply with halo exchange. Implementation: `RasIsaiPreconditioner` (local ISAI(1) per subdomain)

### GPU Details

- **Backend**: wgpu (WebGPU — Vulkan/Metal/DX12)
- **Precision**: f32 on GPU, f64 on CPU interface
- **Shaders**: WGSL compute shaders in `solver/gpu_shaders.rs`
- **Workgroup size**: 64

## Building & Testing

```sh
cargo build                                          # build
cargo test                                           # all tests (unit + integration)
cargo test --features ngspice-compare                # include ngspice regression tests (requires ngspice in PATH)
cargo test --test integration_test                   # linear circuit integration tests only
cargo test --test diode_integration_test             # diode tests only
cargo test --test transistor_integration_test        # BJT/MOSFET tests only
cargo test --test gpu_integration_test               # GPU-specific tests only
mpirun -n 2 cargo test --features distributed --test distributed_test  # MPI distributed tests (requires MPI)
```

### Feature Flags

| Feature | Purpose |
|---|---|
| `ngspice-compare` | Enable ngspice regression tests |
| `distributed` | Enable MPI communication backend (`MpiComm`) for multi-GPU/multi-node |

GPU tests require a GPU-capable environment. They will fail in headless CI without a compatible adapter.

### Regression Framework

`tests/regression/` contains circuits validated against ngspice. Config lives in `tests/regression/manifest.toml` — each entry specifies circuit file, analysis type, nodes to compare, and tolerances.

## Conventions

### Dependencies

- Verify latest stable before adding: `cargo search <crate> --limit 1`
- Use major version specifiers (e.g., `nom = "8"`) unless a specific minor/patch is needed

### Error Handling

- Use `OhmnivoreError` variants from `src/error.rs` — never `panic!` or `unwrap` in library code
- Propagate errors with `?`; tests may use `.expect("descriptive message")`

### Code Style

- Default `rustfmt` and `clippy` settings (no custom config files)
- Comments on non-obvious algorithms (MNA stamps, BiCGSTAB iteration, voltage limiting)
- Integration tests go in `tests/`, unit tests in `#[cfg(test)]` blocks within source files
- Test helpers (e.g., `dc_solve`, `ac_solve`) live at the top of integration test files

### MNA Conventions

- Ground nodes: `"0"` or `"GND"` — excluded from matrix
- Voltage sources and inductors add branch current variables (rows/cols after node variables)
- Matrix variable naming: `g` (conductance/G), `c` (capacitance/C), `b_dc`/`b_ac` (RHS vectors)

### Nonlinear Elements

- Each element type gets a `Gpu*Descriptor` struct (bytemuck Pod) with CSR value indices for stamp injection
- Newton-Raphson runs entirely on GPU — evaluation, assembly, and linear solve per iteration
- Voltage limiting applied per-iteration to improve convergence

### Distributed Solver

Ohmnivore supports multi-GPU and multi-node execution via domain decomposition:

- **Partitioning**: `solver/partition.rs` — METIS graph partitioning (`MetisPartitioner`) with `SubdomainMap` providing 1-layer overlap, global-to-local index mapping, and submatrix/subvector extraction
- **Communication**: `solver/comm.rs` — `CommunicationBackend` trait (all-reduce sum/max, halo exchange, barrier). `SingleProcessComm` (no-op) for single-GPU. `solver/comm_mpi.rs` — `MpiComm` behind `distributed` feature flag
- **Distributed BiCGSTAB**: `solver/distributed_bicgstab.rs` — BiCGSTAB with `all_reduce_sum` for dot products, halo exchange before SpMV and preconditioner apply
- **RAS Preconditioner**: `solver/distributed_preconditioner.rs` — Restricted Additive Schwarz with local ISAI(1) per subdomain. Halo exchange → local ISAI SpMVs → restrict to owned nodes
- **Newton-Raphson**: `solver/distributed_newton.rs` — `solve_dc` routing through distributed BiCGSTAB + RAS for linear circuits, existing `newton_solve` for nonlinear single-GPU

Single-GPU degenerates to one subdomain with no-op communication. RoCE (RDMA) backend planned for low-latency production clusters.

### Design Documents

Architecture and design docs live in `.docs/plans/` (gitignored, local only). Reference them for context on past decisions but don't assume they reflect current implementation — code is the source of truth.
