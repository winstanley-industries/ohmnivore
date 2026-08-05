# Hermetic production sparse-direct solver

Ohmnivore Phase 2D selects SuiteSparse KLU as the production FP64 CPU solver for the completed
linear DC, AC, and transient correctness path. KLU is linked into production; the former dense
partial-pivoting implementation is an independent exact-small oracle in the Bazel `testonly`
target `//cpp:dense_oracle`.

## Selection requirements and candidates

The selection review was performed on 2026-08-05 against upstream release documentation and
source. A candidate had to support real and complex FP64, deterministic single-process sparse LU,
circuit-MNA-appropriate pivoting and scaling, CSC or deterministic conversion from CSR, separate
symbolic analysis and numeric refactorization, and a checksum-pinned static Bazel build with no
system sparse solver, BLAS/LAPACK, Fortran runtime, compiler, header, library, or `PATH` discovery.

| Candidate reviewed | Evidence | Decision |
|---|---|---|
| SuiteSparse KLU 2.3.6 | The upstream KLU guide identifies SPICE-like circuit matrices as its target, documents real and interleaved-complex CSC APIs, BTF plus AMD/COLAMD ordering, threshold partial pivoting, row scaling, structural/numerical rank, and `klu_refactor` for a fixed pattern. Its direct dependencies are BTF, AMD, COLAMD, and SuiteSparse_config; none requires BLAS/LAPACK or Fortran. | Selected. It meets every required numerical and hermetic-build contract with a small serial C source boundary. |
| Eigen SparseLU 5.0.1 | The public SparseLU interface supports general real/complex compressed column matrices and split `analyzePattern`/`factorize` calls under MPL-2.0. It provides sparse ordering and threshold pivoting, but does not expose a documented built-in row-equilibration policy matching the mixed-unit MNA requirement; Ohmnivore would have to own and validate another numerical transformation layer. | Rejected before timing. It is otherwise viable, but adds project-owned scaling behavior when KLU supplies an explicit circuit-oriented policy. |
| SuperLU 7.0.1 | The sequential library supports real/complex FP64, equilibration, partial pivoting, reused sparsity, and CSC. Its supported build consumes BLAS; the bundled reference CBLAS is still a BLAS implementation. | Rejected by the explicit no-BLAS build boundary. |
| SuiteSparse UMFPACK 6.3.8 | General unsymmetric real/complex sparse LU with symbolic/numeric separation and scaling. Upstream declares BLAS/LAPACK build dependencies and licenses UMFPACK as GPL-2.0+. | Rejected by the no-BLAS/LAPACK boundary; its stronger copyleft was also unnecessary when KLU met the requirements under LGPL-2.1+. |

Primary selection inputs:

- SuiteSparse 7.12.3 release and component versions:
  `https://github.com/DrTimothyAldenDavis/SuiteSparse/releases/tag/v7.12.3`
- KLU user guide in the selected archive: `KLU/Doc/KLU_UserGuide.pdf` and its exact source
  `KLU/Doc/KLU_UserGuide.tex`
- Eigen 5.0.1 release: `https://gitlab.com/libeigen/eigen/-/releases/5.0.1`
- Eigen SparseLU reference: `https://libeigen.gitlab.io/eigen/docs-nightly/classEigen_1_1SparseLU.html`
- SuperLU 7.0.1 release: `https://github.com/xiaoyeli/superlu/releases/tag/v7.0.1`
- SuperLU project/BLAS documentation: `https://portal.nersc.gov/project/sparse/superlu/`
  and `https://portal.nersc.gov/project/sparse/superlu/faq.html`
- UMFPACK license and SuiteSparse dependency documentation in the selected archive:
  `UMFPACK/Doc/License.txt` and the top-level `README.md`

Performance was not used to rescue a candidate that failed a hard requirement. The fixed KLU
corpus confirms that both cold analysis/factor/solve and fixed-pattern numeric-refactor/solve paths
are reproducibly exercisable on circuit-representative shapes. It is bounded selection evidence,
not a comparison against rejected dependencies or a general performance/scalability claim.

## Selected source identity and license

- Upstream project and tag: SuiteSparse `v7.12.3`, released 2026-07-31
- Selected component: KLU 2.3.6
- Required components: AMD 3.3.4, BTF 2.3.3, COLAMD 3.3.5, and SuiteSparse_config 7.12.3
- Official archive URL:
  `https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/refs/tags/v7.12.3.tar.gz`
- Archive size observed during selection: 95,359,625 bytes
- SHA-256: `158ee4ed2ce3fdcbf52c4e47e94b0d1a8ae13344b4a835991d78a3ad20f08086`
- KLU and BTF license: LGPL-2.1-or-later
- AMD, COLAMD, and SuiteSparse_config license: BSD-3-Clause

The exact upstream notices are exposed by `@suitesparse_7_12_3//:licenses`. Distribution must
preserve those notices and satisfy LGPL-2.1-or-later source/relinking requirements for the
statically linked KLU and BTF objects. The selected licenses permit this integration subject to
those obligations; this record is technical provenance, not legal advice.

The direct `http_archive` repository rule and checksum are declared in `MODULE.bazel`. It rejects
missing, redirected-to-different, or modified archive bytes. The rule is not a module-registry
dependency and therefore does not add a SuiteSparse entry to `MODULE.bazel.lock`; lockfile-error
validation still proves that resolving it does not mutate the module lock.

## Hermetic Bazel build

`third_party/suitesparse/BUILD.suitesparse.bazel` compiles the upstream C files directly with the
repository's checksum-pinned zero-sysroot LLVM toolchain. Upstream CMake, Makefiles, generated
package discovery, and optional CHOLMOD user ordering are not invoked. The build inputs are:

- `SuiteSparse_config/SuiteSparse_config.c` and `.h`;
- all 32-bit AMD `AMD/Source/*.c` plus `AMD/Include/*.h`, excluding `amd_l*.c`;
- all 32-bit BTF `BTF/Source/*.c` plus `BTF/Include/*.h`, excluding `btf_l*.c`;
- all 32-bit COLAMD `COLAMD/Source/*.c` plus `COLAMD/Include/*.h`, excluding `colamd_l.c`; and
- all 32-bit real and complex KLU compilation units plus `KLU/Include/*.h`, excluding `klu_l*.c`
  and `klu_zl*.c`. The generic `klu_*.c` implementations are also textual inputs because the
  selected `klu_z*.c` int32 complex wrappers include them after defining `COMPLEX`.

The AMD, BTF, and COLAMD int64 wrappers and the KLU int64 real/complex wrappers are neither
compiled nor declared as textual inputs. The hermetic provider manifest fails analysis if an
excluded long-index source enters the production compile boundary.

KLU, BTF, AMD, COLAMD, and SuiteSparse_config are ordinary Bazel `cc_library` static inputs. No
target declares BLAS, LAPACK, Fortran, OpenMP, a system sparse package, `rules_foreign_cc`, CMake,
pkg-config, or shell actions. The production boundary check is:

```sh
bazel test //cpp:solver_hermeticity_test
```

It compile-time checks the exact KLU 2.3.6 headers and parses the Bazel-built ELF. The ELF audit
allowlists exact host-ABI `DT_NEEDED` entries, requires embedded `klu_factor`, `klu_refactor`,
`klu_z_factor`, and `klu_z_refactor` symbols, and rejects the dense-oracle symbol. A `genquery`
manifest proves that `//cpp:ohmnivore` depends on
`//cpp:core -> @suitesparse_7_12_3//:klu`, has no path to the `testonly`
`//cpp:dense_oracle` target, and contains none of the excluded int64 sources. An analysis-time
`CcInfo` manifest rejects system include/library paths, dense-oracle inputs, excluded int64 source
inputs, and external BLAS/LAPACK/Fortran/sparse libraries while requiring the pinned headers and
static `libklu.a`. Normal `--config=asan` and `--config=ubsan` tests compile and exercise the
selected C sources; no incompatibility or skip tag is applied.

The dependency/provider manifests are reproducible targets:

```sh
bazel build //cpp:production_solver_dependency_manifest //cpp:production_solver_cc_manifest
```

The exact zero-sysroot KLU compile commands and ASan/UBSan instrumentation can be inspected without
executing a system compiler:

```sh
bazel aquery 'mnemonic("CppCompile", @suitesparse_7_12_3//:klu)' --output=commands
bazel aquery --config=asan 'mnemonic("CppCompile", @suitesparse_7_12_3//:klu)' --output=commands
bazel aquery --config=ubsan 'mnemonic("CppCompile", @suitesparse_7_12_3//:klu)' --output=commands
```

## Storage and numerical contract

Ohmnivore retains canonical CSR as its semantic matrix format. Before KLU sees a matrix, the
adapter requires:

- a square matrix with `row_offsets.size() == rows + 1`;
- equally sized `column_indices` and `values`;
- offsets starting at zero, ending at the value count, monotone, and within the value array;
- in-range columns strictly increasing within every row, so duplicates are rejected rather than
  summed;
- finite real values, or finite real and imaginary components; and
- dimensions and stored-entry count no greater than `INT32_MAX`.

Conversion is deterministic. It counts entries by column, takes one prefix sum, then scans CSR rows
and each row's columns in canonical order. The resulting CSC columns therefore contain increasing
row indexes. Every stored entry is retained, including explicit numerical zeros. A parallel
`csr_value_indices` array maps each CSC slot to its exact CSR slot; refactorization gathers values
through that immutable map. No sorting, duplicate coalescing, zero dropping, or fill prediction is
performed by the adapter. Complex values and right-hand sides use KLU's documented interleaved
`real, imaginary` double representation.

The fixed KLU policy is:

- 32-bit signed indexes and one right-hand side;
- BTF enabled (`btf = 1`);
- AMD fill-reducing ordering (`ordering = 0`); COLAMD is built only because it is a declared KLU
  dependency, not selected dynamically;
- maximum-magnitude row scaling (`scale = 2`);
- full threshold partial pivoting (`tol = 1.0`);
- halt on singularity (`halt_if_singular = 1`); and
- serial single-process execution with no OpenMP or threaded numerical dependency.

`klu_analyze` creates one symbolic object. The first solve uses `klu_factor`; a changed value array
with the same exact converted pattern first uses `klu_refactor`; an exactly identical value array
reuses the numeric object; and every solve uses `klu_solve` or `klu_z_solve`. KLU documents that
`klu_refactor` preserves the first numeric pivot order and performs no new numerical pivoting. If
refactorization itself fails, or its solve fails finite/componentwise-backward-error validation,
Ohmnivore deterministically discards only the numeric object, calls `klu_factor`/`klu_z_factor`
again under the existing symbolic analysis to select fresh full-threshold pivots, solves, and
revalidates. This is a retry within the selected sparse solver, never dense fallback.

AC holds one complex object for the frequency sweep. Transient uses a deterministic exact-pattern
cache for DC initialization, UIC/projection, BE, and trapezoidal matrices. G and C are merged into
a fixed union pattern and explicit zero cancellations are retained, so changing frequency,
timestep, or source values does not accidentally change symbolic structure.

## Validation, determinism, and failures

Every returned component must be finite. The adapter independently recomputes the residual in
`long double` and requires both of these bounds:

```text
max_i |D(Ax-b)_i| / (max_i sum_j |(DA)_ij| * ||x||inf + ||Db||inf) <= 1e-10
D_ii = 1 / max(|b_i|, max_j |a_ij|), or 1 for an all-zero row

max_i |(Ax-b)_i| / (|b_i| + sum_j |a_ij|*|x_j|) <= 1e-5
0/0 is defined as zero; a nonzero residual over a zero denominator is infinite
```

The first bound is the tight dimensionless acceptance criterion for row-equilibrated mixed-unit
MNA. The second is a sparsity-preserving row-local guard: its looser bound accommodates the
observed few-parts-per-million componentwise residual of valid GMIN-inclusive RLC equations while
still preventing an unrelated large-unit row or variable from hiding a grossly bad small-unit
equation. These are solution validation, not a promise of forward error for ill-conditioned
systems.

The adapter returns distinct typed failures for invalid CSR structure, unsupported index size,
structural or numerical singularity/rank deficiency, allocation or factorization/refactor/solve
failure, non-finite input/result, and backward-error rejection. The direct solver has no iterative
non-convergence state; inability to factor or solve is a factorization failure. Sparse failures
never dispatch to or fall back to the dense oracle.

For a fixed build, hardware, matrix, and right-hand side, repeated solves are required bitwise
identical and are tested as such. Cross-toolchain/platform correctness is expressed by exact
structural contracts plus analytic, dense-oracle, backward-error, and bounded ngspice tolerances;
bitwise identity is not claimed across different floating-point implementations.

## Reproduce the selection evidence

Build with optimization and run the fixed corpus explicitly:

```sh
bazel run -c opt //cpp:solver_selection_benchmark -- --warmups=3 --repetitions=15
```

The target reports Bazel/compiler/kernel/CPU metadata, matrix dimensions and stored-entry counts,
warmups, repetitions, every measured `steady_clock` nanosecond sample, and sorted
min/P25/median/P75/max summaries. It contains no pass/fail timing threshold. The recorded raw run is
`docs/evidence/phase2d-solver-selection-2026-08-05.csv`.

The corpus sparsity is:

| Matrix | Scalar | Size | Stored entries | Stored density |
|---|---:|---:|---:|---:|
| `dc_mna_ladder_128` | real | 129 | 384 | 2.307553633% |
| `transient_companion_ladder_512` | real | 513 | 1,536 | 0.583655370% |
| `transient_companion_ladder_2048` | real | 2,049 | 6,144 | 0.146341429% |
| `ac_mna_ladder_256` | complex | 257 | 768 | 1.162773093% |

Each has an asymmetric voltage-source branch and tridiagonal circuit body. Both cold
analyze/factor/solve and fixed-union-pattern numeric-refactor/solve are measured. Synthetic ladders
make the corpus source-controlled and reproducible; the results do not imply universal performance
or scalability.

## Remaining limitations

Phase 2D supports only the completed linear RLCVI DC/AC/transient subset. It does not add nonlinear
devices, Newton iteration, limiting, continuation, nonlinear transient analysis, new netlist or
initial-condition syntax, CUDA circuit kernels or GPU dispatch, mixed precision, distributed
solving, broader ngspice fixture claims, or performance guarantees. KLU's 32-bit API bounds both
matrix dimension and stored-entry count to `INT32_MAX`; larger matrices fail before conversion.
