# Ohmnivore

GPU-accelerated circuit simulation solver. Parses a SPICE-subset netlist, compiles to Modified Nodal Analysis (MNA) matrices, and solves via GPU (wgpu/BiCGSTAB) or CPU (direct LU). Designed to scale to multi-GPU and multi-node clusters via domain decomposition.

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
| `solver/` | Linear solvers (CPU direct, GPU BiCGSTAB) and nonlinear Newton-Raphson |
| `analysis/` | DC operating point and AC frequency sweep engines |
| `output` | CSV result formatting |
| `error` | `OhmnivoreError` enum via `thiserror`, project-wide `Result<T>` alias |

### Solver Traits

- `LinearSolver` — main interface (`solve_real`, `solve_complex`)
- `SolverBackend` — GPU-agnostic vector/matrix ops (SpMV, dot, AXPY)
- `NonlinearBackend` — GPU-resident diode/BJT/MOSFET evaluation for Newton-Raphson

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
```

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

### Distributed Solver Vision

Ohmnivore targets multi-GPU and multi-node execution:

- **Domain decomposition**: Circuit graph partitioned across GPUs via METIS (one subdomain per GPU), coordinated by a global Krylov solver (BiCGSTAB/GMRES) with distributed reductions (all-reduce for dot products)
- **Preconditioning**: Restricted Additive Schwarz (RAS) with 1-layer overlap. Each GPU applies a local ISAI(1) preconditioner to its subdomain. `DistributedPreconditioner` trait abstracts over single-GPU vs. distributed apply
- **Communication**: `CommunicationBackend` trait abstracting over transport (halo exchange, all-reduce) — MPI first, RoCE (RDMA over Converged Ethernet) planned for low-latency production clusters. Single-process no-op backend for single-GPU

Design the solver traits, preconditioner interfaces, and data structures with this distribution model in mind, even when implementing single-GPU features.

### Design Documents

Architecture and design docs live in `.docs/plans/` (gitignored, local only). Reference them for context on past decisions but don't assume they reflect current implementation — code is the source of truth.
