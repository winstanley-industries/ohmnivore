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

The current C++ execution surface is deterministic linear `.DC`/`.OP`, `.AC DEC|OCT|LIN`, and
`.TRAN tstep tstop [tstart] [UIC]` analysis for resistors, capacitors, inductors, and independent
voltage/current sources. Sources accept strict DC, AC, PULSE, SIN, PWL, and EXP forms in legacy
order. Capacitors are open and inductors are ideal shorts at DC; AC solves
`(G + j * 2*pi*f*C)x = b_ac`; transient solves `G*x + C*dx/dt = b(t)`. Both use temporary dense
FP64 CPU correctness solvers.

AC point counts and frequencies are validated before execution. LIN requires at least two total
points and includes the requested endpoints. DEC/OCT require a positive points-per-interval value,
emit their geometric grid from the exact start, and include the exact stop once. Every sweep must
have finite positive frequencies with `stop > start`, and generated sweeps are limited to one
million points. A requested grid that cannot be represented as strictly increasing FP64 values is
rejected instead of silently dropping points.

Transient `tstep` is the hard maximum accepted step, integration always begins at zero, `tstart`
is an exact output boundary, and `tstop` is included exactly. Waveform breakpoints are hard
boundaries. The initial, post-rejection recovery, and waveform-breakpoint landing steps use
backward Euler; other steps use trapezoidal integration. Every dynamic BE step is error-controlled
by one full step versus two half steps, while trapezoidal steps use a deterministic BE comparison.
At a discontinuous source edge, integration reaches the edge with the left-limit forcing and then
projects algebraic variables to the right limit while preserving capacitor voltages and inductor
currents. The controller uses a scaled infinity error with `1e-9` absolute and `1e-3` relative tolerances,
clamps step changes to `[0.5, 2]` with a `0.9` safety factor, and uses `tstep/10000` as the adaptive
minimum. Mandatory hard-boundary clips may be smaller. Unrepresentable time progress,
minimum-step exhaustion, non-finite arithmetic, singular systems, and step/attempt limits are
typed failures. The production limits are 1,000,000 accepted steps and 2,000,000 total attempts.

Without UIC, the existing DC solve initializes the state. UIC instead enforces zero capacitor
voltage and zero inductor current while solving the remaining algebraic constraints; inconsistent
constraints fail explicitly, while redundant constraints consistent with voltage sources are
accepted. Transient-only sources have zero DC initialization, and a transient
waveform replaces rather than adds to a source's DC value during transient execution.

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

## Hermetic ngspice acceptance

The Phase 2C differential gate builds checksum-pinned ngspice 46 source through Bazel and invokes
that exact executable. It does not search `PATH` or use a system ngspice, compiler, header, or
library. Configure and Make receive only checksum-pinned BusyBox POSIX tools and GCC binutils on
their `PATH`; the required execution-platform `/bin/bash` is checksum-verified before configure.
The runner requires a little-endian x86-64 static executable, rejects any ELF program
header containing `PT_INTERP` or `PT_DYNAMIC`, and checks the exact version before comparing the
bounded linear fixtures:

```sh
bazel test //acceptance:ngspice_acceptance_test
```

Exact source/build provenance, configure inputs, fixtures, sampling rules, and tolerances are in
`third_party/ngspice/PROVENANCE.md`. The harness reconstructs and validates complete transient
reference grids and exact CSV schemas before comparing values. Failure to prove provenance or
obtain bracketing comparison samples fails the test closed.

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
