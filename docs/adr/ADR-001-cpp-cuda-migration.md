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

## Follow-up phases

1. Port and differentially accept the complete linear CPU path: parser/IR/CSR, RLCVI stamps, DC,
   AC, transient, CSV, and ngspice harness.
2. Select the hermetic sparse-direct FP64 CPU oracle using circuit-representative correctness and
   performance evidence.
3. Port nonlinear device evaluation, Newton iteration, limiting, continuation, and nonlinear
   transient on CPU.
4. Add one CUDA vertical slice with immutable uploaded structure, native `double`, hostile result
   validation, replay, and end-to-end benchmarks.
5. Evaluate batched AC points, parameter corners, Monte Carlo runs, and independent circuits before
   considering single-circuit domain decomposition.

## Consequences

The project temporarily carries two implementations, and Phase 1 supports less SPICE syntax than
the legacy prototype. That duplication is intentional: it keeps the new contracts reviewable and
prevents incomplete feature parity from being mistaken for correctness.

The decision narrows GPU portability to gain numerical control, direct CUDA tooling, and alignment
with the project's actual deployment hardware. Future portability is an explicit backend decision,
not a constraint on the core circuit model.
