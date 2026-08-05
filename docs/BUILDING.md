# Building Ohmnivore

Bazel is the canonical build interface for the C++/CUDA migration. Bzlmod resolves dependencies,
and Bazelisk reads the pinned Bazel version from `.bazelversion`.

## Prerequisites

- Bazelisk installed as `bazel` on `PATH`.
- Network access on the first build for checksum-verified archives.
- An NVIDIA driver compatible with the pinned toolkit for CUDA tests.

No host C/C++ compiler, CUDA Toolkit, nvcc, GCC, standard-library headers, or Python interpreter is
required for the C++ phase. Bazel downloads the declared toolchains and lint tools. The NVIDIA
kernel driver remains part of the execution environment.

## Standard commands

```sh
bazel lint
bazel build //...
bazel test //...
bazel run //:ohmnivore -- examples/voltage_divider.spice
```

The current C++ execution surface is `.DC`/`.OP` operating-point analysis for resistors,
capacitors, inductors, and independent voltage/current sources with DC values. Capacitors are open
and inductors are ideal shorts in the operating-point solve. Their dynamic stamps are retained in
the compiled `C` matrix, but AC and transient execution remain unsupported.

Use `bazel lint --fix` to apply supported formatting fixes. Individual language checks are
available with `--only cpp`, `--only python`, `--only shell`, and `--only starlark`.

## Sanitizers

The CPU implementation is checked separately under the pinned LLVM sanitizer runtimes:

```sh
bazel test --config=asan //...
bazel test --config=ubsan //...
```

CUDA targets are explicitly incompatible with either sanitizer configuration because the CUDA
host/device/driver boundary cannot be instrumented end to end by those LLVM runtimes. Verify the
analysis-time rejection by requesting the CUDA target directly:

```sh
bazel build --config=cuda --config=asan //cuda:smoke_test
bazel build --config=cuda --config=ubsan //cuda:smoke_test
```

Both commands must fail during Bazel analysis as incompatible. A wildcard or test-suite request
can skip an incompatible test and is not evidence of rejection.

## CUDA smoke test

CUDA is opt-in and excluded from default wildcard builds:

```sh
bazel test --config=cuda //:cuda_smoke_test
```

The configuration selects checksum-pinned CUDA Toolkit 13.0.2 redistributables, nvcc, GCC 15.2.0
and its sysroot, and native plus PTX code for compute capability 12.0. The smoke test performs a
real allocation, kernel dispatch, synchronization, and device-to-host copy, compares the result
with a deterministic CPU oracle, reports device/runtime metadata, and rejects dynamically loaded
`libstdc++` or `libgcc_s`.

## Dependency lock

After changing `MODULE.bazel`, update and inspect `MODULE.bazel.lock` through a successful Bazel
command. Final validation must reject implicit lock changes:

```sh
bazel test --lockfile_mode=error //...
```

## Hermeticity boundary

Normal C++ builds use a downloaded zero-sysroot LLVM toolchain. Local C++ and Apple toolchain
discovery are disabled so an incompatible platform fails resolution instead of silently reading
host headers or libraries.

The CUDA configuration selects only checksum-pinned toolkit and GCC inputs. Pinned `libstdc++` and
`libgcc` are linked statically. Host glibc and the NVIDIA kernel driver are the declared Linux
execution ABI; the system CUDA Toolkit and system compiler stack are outside the build boundary.

## Legacy Rust validation

The Rust/wgpu implementation remains a migration reference, but Rust is not installed or managed by
the Bazel phase. When an explicitly pinned legacy environment is added, its regression suite will
be used to emit differential migration fixtures. Until then, do not present unexecuted Cargo tests
as current validation.
