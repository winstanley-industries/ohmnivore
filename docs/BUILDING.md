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
voltage/current sources, plus nonlinear diode `.DC`/`.OP`, memoryless-diode `.TRAN`, and strict
NPN/PNP BJT `.DC`/`.OP`. Sources
accept strict DC, AC, PULSE, SIN, PWL, and EXP forms in legacy order. Capacitors are open and
inductors are ideal shorts at DC; AC solves
`(G + j * 2*pi*f*C)x = b_ac`; transient solves `G*x + C*dx/dt = b(t)`. DC, AC,
transient companion, UIC, and discontinuity-projection systems use the checksum-pinned KLU 2.3.6
real or complex FP64 sparse-direct path. The old dense partial-pivoting implementation is linked
only into the exact-small test oracle.

GPU-01 additionally provides an explicit prepared linear-AC batch and replay/evidence surface.
It is not used by ordinary `SimulateAc` or the CLI, and it has no CUDA implementation or automatic
dispatch. CPU KLU remains the only implementation and full-batch fallback.

GPU-02 adds one opt-in native-complex-FP64 CUDA executor for that prepared surface. It uses the
checksum-pinned static NVIDIA cuDSS 0.8.0.10 archive and the cuBLAS package from the existing
checksum-pinned CUDA 13.0.2 redistributable manifest. CUDA remains outside ordinary simulator and
CSV execution; CPU KLU remains the correctness authority, supported no-GPU path, and explicit
whole-batch fallback. Exact source, linkage, architecture, and license provenance is recorded in
`third_party/cudss/PROVENANCE.md`.

GPU-02S is an evidence-only persistent-session follow-up. It reuses the same CUDA algorithm and
backend-neutral contract, permits same-structure FP64 values/RHS refresh while retaining device
structure and analysis, and compares long-lived CUDA ownership with a persistent parallel KLU
worker pool. It remains unreachable from ordinary simulation and CSV output.

## Nonlinear diode and BJT analysis

Phase 3A accepts exactly `Dname anode cathode modelname` and diode models in one of these forms:

```text
.MODEL modelname D
.MODEL modelname D()
.MODEL modelname D(IS=value N=value)
```

`IS` (amperes, default `1e-14`) and `N` (dimensionless, default `1`) are the only parameters.
Fields inside parentheses are whitespace-separated `key=value` tokens, may appear in either order,
and may each appear at most once. Model names and references match `[A-Za-z0-9_]+`; lookup and
parameter keys are ASCII case-insensitive. `IS` and `N` must be finite in `(0,1e100]`, and the
descriptor value must satisfy `0 < N*0.02585 V <= 2.585e98 V`. Commas, detached/nested/unclosed
parentheses, trailing fields, invalid identifiers, duplicate parameters or model names,
unsupported parameters or model types, and missing references fail with typed errors.

For `v=va-vc`, the FP64 device path uses `i=IS*expm1(v/(N*VT))` and
`g=IS*exp(v/(N*VT))/(N*VT)` with `VT=0.02585 V` and the exponent clamped to `[-80,80]`. Every input,
intermediate, Jacobian/residual entry, and solution value must be finite and no larger than `1e100`
in magnitude. This includes Newton deltas/sums, normalization values, and limiting arithmetic. A
mathematically positive conductance that underflows to zero is rejected. The diode contributes
`+i` at its anode KCL row and `-i` at its cathode; its Jacobian stamp is
`+g,-g,-g,+g` at `aa,ac,ca,cc`.

Newton starts from zero, solves `J*delta=-F` with production KLU, and applies SPICE3 logarithmic PN
limiting in diode insertion order. Acceptance requires both scaled voltage/current update checks
and a freshly recomputed nonlinear residual: voltage uses `1e-9 V + 1e-6*scale`, current uses
`1e-12 A + 1e-6*scale`. The fixed bounded strategy order is direct Newton (50 iterations), source
scales `0.0,0.1,...,1.0` (30 iterations each), then extra-GMIN values
`1e-3,1e-4,...,1e-12,0 S` (30 iterations at nonzero values and 50 at zero). There are no adaptive
subdivisions. The final accepted point is independently checked with full sources and no extra
GMIN, so only the existing production `1e-12 S` node conductance remains.

Every accepted point, including an iteration-zero point with an already acceptable residual,
requires numeric factorization and a zero-right-hand-side solve of its current Jacobian through
KLU. This makes numerical rank deficiency a typed failure instead of allowing a zero residual to
bypass the production sparse solver.

Source stepping starts from zero; every later continuation point starts from the preceding
accepted point. If source stepping fails, GMIN stepping starts from its last accepted source point.

Compilation constructs one canonical CSR union pattern containing the linear matrix and all diode
Jacobian coordinates. Diode-required numerical zeros are retained; linear-only matrices keep their
existing zero-elision and exact fast path. One KLU symbolic analysis is reused across every Newton
and continuation step; changed values use numeric refactorization plus the Phase 2D pivot-safe
fresh-numeric retry and backward-error checks. No sparse or nonlinear failure can dispatch to the
dense test oracle. Nonlinear CPU work is deterministic and single-threaded. Diode AC remains
explicitly unsupported.

Repeated direct, source-stepping, and GMIN-stepping runs on one supported toolchain/platform are
required to produce bitwise-identical solutions, traces, and solver statistics. There is no
cross-libm or cross-platform equality guarantee; independent-oracle and ngspice checks use only
their individually stated tolerances.

Phase 3C accepts exactly `Qname collector base emitter modelname` and NPN/PNP models in bare,
empty-parentheses, or attached-parentheses forms. The only parameters are `IS`, `BF`, `BR`, `NF`,
and `NR`, with defaults `1e-16`, `100`, `1`, `1`, and `1`. Model identifiers, full-token parsing,
case-insensitive lookup, duplicate detection, finite `(0,1e100]` bounds, and fixed
`VT=0.02585 V` follow the strict Phase 3A policy. Collector/base aliases are supported for
diode-connected devices; an all-terminal self-connection is rejected.

For polarity-adjusted `vbe` and `vbc`, the bounded legacy-compatible Ebers--Moll path evaluates
`IF=IS*expm1(clamp(vbe/(NF*VT),-80,80))` and the corresponding `IR`, then
`Ic=p*(BF/(BF+1)*IF-IR/(BR+1))`, `Ib=p*(IF/(BF+1)+IR/(BR+1))`, and
`Ie=-(Ic+Ib)`. The complete 3-by-3 Jacobian is accumulated into the immutable nonlinear union.
BJT descriptors are evaluated and limited after diode descriptors, with BE before BC. The same
Newton tolerances, direct/source/GMIN schedule, original-system residual validation, accepted
Jacobian KLU solve, one symbolic analysis, and deterministic FP64 reductions apply. BJT AC and
transient requests are typed unsupported errors; no device stamp is silently omitted.

Phase 3B applies the same memoryless diode current, conductance, limiting, update, residual, and
accepted-Jacobian checks to the Phase 2C transient DAE. Backward Euler solves
`(G+C/h)x_n-(b_n+C*x_p/h)+d(x_n)=0`; trapezoidal integration includes the previous diode current as
`(G+2C/h)x_n-(b_n+b_p+(2C/h-G)x_p-d(x_p))+d(x_n)=0`. Every full, half, and LTE comparison solve is
independently converged. Only Newton nonconvergence retries by halving the timestep and forcing a BE
recovery step; singularity, non-finite arithmetic, factorization, and validation failures propagate
immediately.

Without UIC, the Phase 3A nonlinear DC operating point initializes diode transient state. With UIC,
zero capacitor voltage and zero inductor current remain exact constraints while the other algebraic
rows include diode current. Source discontinuity projection preserves those reactive quantities and
solves the right-limit nonlinear algebraic equations. The canonical transient matrix retains the
fixed G/C/diode coordinate union so KLU symbolic analysis is reused across BE, TRAP, LTE, rejection,
and accepted steps.

Focused nonlinear validation is available as:

```sh
bazel test //cpp:phase3a_test
bazel test //cpp:phase3b_test
bazel test //cpp:phase3c_test
```

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

## Production sparse solver

Phase 2D builds the 32-bit-index, serial KLU 2.3.6, AMD, BTF, COLAMD, and SuiteSparse_config C
sources directly from the checksum-pinned SuiteSparse 7.12.3 archive. It does not run upstream
CMake and declares no system sparse library, BLAS, LAPACK, Fortran, OpenMP, or path-discovery
dependency. Normal ASan and UBSan test configurations instrument these sources with the rest of
the CPU implementation.

The explicit hermeticity/boundary test parses the production ELF rather than relying only on
substring matching. It allowlists exact `DT_NEEDED` host-ABI entries, requires embedded
`klu_factor`/`klu_refactor` real and complex symbols, rejects the dense-oracle symbol, checks a
Bazel-generated production dependency manifest, and consumes an analysis-time `CcInfo` manifest
that rejects system include/library paths, dense-oracle inputs, excluded int64 SuiteSparse
sources, and external numerical libraries:

```sh
bazel test //cpp:solver_hermeticity_test
```

The two constituent evidence targets can also be built and inspected directly:

```sh
bazel build //cpp:production_solver_dependency_manifest //cpp:production_solver_cc_manifest
```

The pinned zero-sysroot compiler action and sanitizer instrumentation remain reproducible with:

```sh
bazel aquery 'mnemonic("CppCompile", @suitesparse_7_12_3//:klu)' --output=commands
bazel aquery --config=asan 'mnemonic("CppCompile", @suitesparse_7_12_3//:klu)' --output=commands
bazel aquery --config=ubsan 'mnemonic("CppCompile", @suitesparse_7_12_3//:klu)' --output=commands
```

The manual selection-evidence target has no timing pass/fail thresholds. It emits machine and
toolchain metadata, every raw sample, and deterministic min/P25/median/P75/max summaries for a
fixed circuit-representative matrix corpus:

```sh
bazel run -c opt //cpp:solver_selection_benchmark -- --warmups=3 --repetitions=15
```

Exact dependency provenance, rejected candidates, build inputs, storage and failure contracts,
license obligations, and the recorded benchmark are in `third_party/suitesparse/PROVENANCE.md`.

## GPU-01 prepared AC replay evidence

The version-1 replay manifest is `replays/prepared_ac/v1/corpus.csv`. Its exact bytes and generated
identities are pinned. It materializes deterministic MNA path, binary-tree, 32-by-32-grid, and
multi-source-ring workload classes covering dimensions 65--1025, 192--4994 G/C-union entries,
batch sizes 16--517, frequency ranges from `1e-3` Hz through `1e9` Hz, multiple
conductance/dynamic scales, one-to-four source branches, and prepared reuse counts 2--8. The exact
builder and fingerprint serialization are documented beside the manifest.

Focused correctness and hostile-result gates are:

```sh
bazel test //cpp:gpu01_prepared_ac_test //cpp:gpu01_replay_test
```

The manual CPU-only evidence target measures cold and prepared/reused serial KLU authority plus a
fair parallel-host KLU comparator. Every scheduled member is solved exactly once; fresh CPU KLU
certification is recorded separately. Each raw sample runs in a new child process and includes
preparation, scheduling-plus-KLU, association/residual/KLU/differential validation, complete
source/binary/host/target metadata, throughput, nearest-rank tail-latency inputs, child-process
peak memory, exact solve counts, and failure count. The stream emits every prepared identity and a
terminal record binding all preceding records and expected counts:

```sh
bazel run -c opt //cpp:prepared_ac_replay_benchmark -- --warmups=2 --repetitions=9
```

`--threads=0` (the default) uses the available hardware-thread count, bounded by batch size. A
positive `--threads=N` is an explicitly recorded evidence override. This CPU-only measurement has
no CUDA upload/device/synchronization/readback path and cannot establish a GPU speedup. The
crossover hypothesis and pre-CUDA thresholds are frozen in ADR-001 before GPU-02.

The preserved canonical CPU-only run uses two warmups and nine recorded samples per mode; its
complete metadata, identities, raw samples, summaries, solve counts, and terminal completeness
record are in `docs/evidence/gpu01-prepared-ac-cpu-baseline-2026-08-07.csv`. It is a host baseline
only, not evidence that a future CUDA implementation will win.

## GPU-02 native-FP64 CUDA prepared AC

GPU-02 is manual and opt-in. The focused analytic/hostile test, the complete frozen replay-v1
CPU/CUDA differential, and the static-link audit are explicit CUDA targets:

```sh
bazel test --config=cuda //cuda:gpu02_prepared_ac_test
bazel test --config=cuda //cuda:gpu02_replay_differential_test
bazel test --config=cuda //cuda:gpu02_linkage_test
```

The executor uploads canonical CSR structure once per preparation, uses native complex FP64 cuDSS,
and submits the complete same-pattern member set as one native uniform batch per factor/solve
phase. It exposes owner-thread teardown so every cleanup status and allocation imbalance fails closed.
The linkage audit requires embedded static cuDSS and cuBLAS symbols and permits only the declared
glibc host ABI in `DT_NEEDED`; cudart, libstdc++, and libgcc are static. No ordinary simulator or
CSV target depends on this executor.

The canonical evidence invocation enforces at least three warmups and twenty retained samples for
all four cases and all six serial-KLU, parallel-KLU, and CUDA cold/prepared modes. Run it twice as
two independent processes and preserve the complete streams:

```sh
bazel run -c opt --config=cuda //cuda:gpu02_evidence_benchmark -- \
  --warmups=3 --repetitions=20 \
  > docs/evidence/gpu02-native-fp64-cuda-uniform-batch-run-1-2026-08-08.csv
bazel run -c opt --config=cuda //cuda:gpu02_evidence_benchmark -- \
  --warmups=3 --repetitions=20 \
  > docs/evidence/gpu02-native-fp64-cuda-uniform-batch-run-2-2026-08-08.csv
```

Every raw CUDA row reconciles the complete measured `Execute` wall into classified upload,
submission, synchronization, and readback intervals plus an explicit remainder that contains all
CUDA/cuDSS setup, host packing, status/memory queries, and result association. Fresh KLU CPU
certification remains a separate validation interval. Release time and zero post-release device
allocation balance are recorded, while release/destructor time is excluded symmetrically from the
CPU and GPU latency comparators. The terminal record binds the expected identity, sample, summary,
verdict, and failure counts to all preceding records. These artifacts are experimental evidence,
not a production speedup claim or authorization for automatic dispatch.
The complete 2026-08-07 measurements are retained separately for comparison with the superseded
per-member `UBATCH_SIZE=1` diagnostic; only the uniform-batch streams above bind the final GPU-02
performance verdict. The exact pre-fix source and binary are not retained.

## GPU-02S persistent-session evidence

The focused compiler/corpus, CUDA refresh, and static-link tests are explicit targets:

```sh
bazel test //cpp:gpu02s_session_test
bazel test --config=cuda //cuda:gpu02s_session_test
bazel test --config=cuda //cuda:gpu02s_linkage_test
```

The benchmark emits fresh-child CPU/CUDA session samples for every corner prefix and a separate
same-process diagnostic containing one cold session plus twenty steady sessions for both persistent
backends and both validation lanes. Canonical evidence requires three warmups and twenty recorded
fresh sessions; the candidate lane still performs and records mandatory fresh KLU certification
outside its technical interval. Run two independent invocations:

```sh
bazel run -c opt --config=cuda //cuda:gpu02s_session_benchmark -- \
  --warmups=3 --repetitions=20 --persistent-sessions=20 \
  > docs/evidence/gpu02s-persistent-session-run-1-2026-08-16.csv
bazel run -c opt --config=cuda //cuda:gpu02s_session_benchmark -- \
  --warmups=3 --repetitions=20 --persistent-sessions=20 \
  > docs/evidence/gpu02s-persistent-session-run-2-2026-08-16.csv
```

The session manifest is synthetic and compiler-derived, not a customer or production workload.
Its control is permanently ineligible; a passing candidate would be only a technical crossover for
the exact hardware, validation lane, and session envelope. Automatic dispatch remains unauthorized.

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
bazel build --config=cuda --config=asan //cuda:prepared_ac_cuda
bazel build --config=cuda --config=ubsan //cuda:prepared_ac_cuda
bazel build --config=cuda --config=asan //cuda:gpu02s_session_benchmark
bazel build --config=cuda --config=ubsan //cuda:gpu02s_session_benchmark
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

The Phase 3C differential gate builds checksum-pinned ngspice 46 source through Bazel and invokes
that exact executable. It does not search `PATH` or use a system ngspice, compiler, header, or
library. Configure and Make receive only checksum-pinned BusyBox POSIX tools and GCC binutils on
their `PATH`; the required execution-platform `/bin/bash` is checksum-verified before configure.
The runner requires a little-endian x86-64 static executable, rejects any ELF program
header containing `PT_INTERP` or `PT_DYNAMIC`, and checks the exact version before comparing the
bounded linear fixtures plus one forward-biased diode DC fixture, one memoryless-diode transient
fixture, and one forward-active BJT DC fixture:

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
