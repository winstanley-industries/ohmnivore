# Time-window decomposition evidence

**EMI-03 acceptance remains incomplete.** These are manual fixed-matrix and
fixed-schedule nonlinear probes. They do not replace the frozen complete
transient, DPT, q0/q1/q2, external-reference, full-study or resource gates.

The [experiment report](../../../../emi03-time-window-decomposition.md) explains
the exact coupled recurrence, validation boundaries, measured results and
remaining integration work. [nonlinear-results.json](nonlinear-results.json)
accounts for all 64 final nonlinear windows: 56 pass and eight reject after
bounded backtracking. Passing windows contain 6,784 distinct original timestep
states. Repeated measurements are additional executions of those inputs.

Every archived input/source/log is losslessly gzip encoded. The
[artifact index](artifact-index.json) records original and stored SHA-256 hashes,
byte counts, source paths and binary identities. Executables are not committed.
The selected resident implementation remains unchanged from the base commit.
The archive also retains the independent CPU oracles, GPU differential results,
failed attempts and hardware-counter report.

## Experiment identities

- **131:** the initial offline matrix calculation mistakenly used the reference
  case's mass matrix for other cases. Its results are invalid for those cases,
  regardless of any permissive status label in the original output. **131b/c**
  use independently compiled, case-specific mass matrices; 131c also checks the
  original per-step componentwise denominator. All 36 corrected windows pass.
- **132/132b:** transfer-column construction rejects on the componentwise guard,
  including after cached native-FP64 dense fallback. **132c** treats internal
  transfer/particular vectors as preconditioner work while preserving both
  original guards on the recovered full timestep states.
- **133:** independent CPU-only sequential KLU oracle, 4,608 states. Recompiling
  the formatted source reproduces the retained oracle byte for byte.
- **134:** shared status readback removes repeated per-case host waits. Its
  retained Nsight Compute profile identifies one active block per SM, low
  occupancy and warp/barrier imbalance in `TimeIndependent`.
- **135/135b:** independent CPU nonlinear fixtures and offline nonlinear
  feasibility. The first delta formulation rejects zero/near-zero componentwise
  rows; the absolute-state affine formulation follows the solver's convention.
  The offline method's initial backtracking policy differs from the GPU method.
- **136:** eight independent RHS solves per block, zero dense retries in the
  manufactured screen. Complete memory/synchronization/initialization checks pass;
  race checking covers every packed-warp launch, supplementing 134's full race
  check. Nine-matrix 512-step median is 4.691 ms versus 35.356 ms sequentially.
- **137:** actual nonlinear behavioral evaluation, bounded window iteration,
  per-step GPU rank checks and independent CPU state/residual/rank certification.
  Early build/layout adjustments are retained. **137c** passes 30/36 windows.
- **138/139:** device-controlled conditional graphs. Ordinary numerical results
  pass, but Compute Sanitizer fails with an unknown CUDA error. The initial 139
  generation assertion did not change source; its cached build is not a new
  solver result. **139b** is the actual conditional-graph timing variant.
- **140:** seven additional independently generated reference/nominal windows
  throughout the cycle. These use their captured step sizes. Some schedules
  cross source breakpoints and therefore are discretized-equation probes, not
  accepted adaptive trajectories.
- **141:** minimal one-level and nested conditional-graph reproductions. Both
  pass ordinarily and fail memory checking with the same unknown CUDA error,
  independently of all solver code. This does not qualify conditional graphs.
- **142:** ordinary graph per nonlinear iteration. Memory, race and synchronization
  checks pass; initialization checking finds padding in the returned status.
- **143:** explicit initialized reserved field removes that padding. All four
  sanitizer tools pass for the reference/nominal 128-step switching window.
  The final 64-window corpus and matched 128-step timing screens use this source.
- **144:** build/test, independent-oracle identity, sanitizer configuration and
  artifact validation. An initially requested pair of nonexistent CUDA test
  labels is retained separately from the corrected successful test invocation.

## Reproduction

Bazel remains the only compiler/toolchain interface. The tools are manual and
`testonly`; none is linked into the selected simulator.

```sh
bazel build -c opt //cpp:emi03_time_cpu_probe //cpp:emi03_time_nonlinear_fixture
bazel build -c opt --config=cuda //cuda:emi03_time_matrix_probe //cuda:emi03_time_nonlinear_probe
```

Decompress the retained replay, mass matrix and CPU oracle to a working folder,
then run the matrix executable with those three absolute paths. Add
`--validate-only` to run one invocation of each length/mapping without the nine
measured repetitions. The nonlinear executable takes a compiled behavioral
deck, a retained `EMI03_NONLINEAR_TIME_1` fixture, a window length, and `--simple`
or `--simple-compare`. The latter performs one warmup and nine alternating
ordinary-graph/sequential measurements. It checks each result independently on
the CPU. Exact original commands and source generators are archived.

The nonlinear fixture contains the captured initial state and independent CPU
reference for validation. The GPU solver receives the initial state and circuit;
the CPU reference is not uploaded or used to generate the GPU trajectory. A full
transient implementation must obtain subsequent boundaries exclusively from its
own accepted GPU state.
