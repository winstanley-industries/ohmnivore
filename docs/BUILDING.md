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

The current C++ execution surface is `.DC`/`.OP` operating-point analysis and deterministic linear
`.AC DEC|OCT|LIN` analysis for resistors, capacitors, inductors, and independent voltage/current
sources. Sources accept a bare DC value, `DC value`, `AC magnitude [phase_degrees]`, or a DC form
followed by an AC form. Capacitors are open and inductors are ideal shorts at DC; AC solves
`(G + j * 2*pi*f*C)x = b_ac` through the temporary dense complex FP64 CPU correctness path.

AC point counts and frequencies are validated before execution. LIN requires at least two total
points and includes the requested endpoints. DEC/OCT require a positive points-per-interval value,
emit their geometric grid from the exact start, and include the exact stop once. Every sweep must
have finite positive frequencies with `stop > start`, and generated sweeps are limited to one
million points. A requested grid that cannot be represented as strictly increasing FP64 values is
rejected instead of silently dropping points. Transient analysis and waveform sources remain typed
unsupported errors.

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
