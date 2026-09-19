# ADR-008: EMI-03 native-FP64 transient ensemble experiment

- **Status:** Implemented experiment contract; qualification and results are reported separately
- **CPU baseline:** `734a9da`, qualified EMI-02 run-3

## Scope and authority

This experiment implements the bounded ensemble execution and evidence surface in
[ADR-002](ADR-002-inverter-emi-design-study.md) and preserves the frozen
[EMI-01 GPU experiment contract](../emi01-gpu-experiment-contract.md). The user
authorized implementation and a draft PR. Automatic dispatch and EMI-04 remain
outside this request.

CPU FP64 KLU remains the correctness authority and the supported no-GPU path.
The complete coupled inverter/filter/load/chassis circuit remains one independent
initial-value problem. Candidate and corner replicas have private initialization,
Newton, integration, accepted-state, rejected-state and output storage. No solver
state or trajectory is shared across replicas.

## CPU diagnosis before selecting the experiment

The diagnostic runner compiles the qualified solver with optional timing scopes
for nonlinear evaluation/assembly, real sparse factor/solve/refinement/validation,
and accepted-state output. Normal builds preprocess those scopes away. The
diagnostic runs a complete frozen candidate/corner job and a fresh ordinary job,
then requires identical raw bytes, measurement fields and numerical work counts.
The remaining runner wall time is reported explicitly. Instrumented phase times
are diagnosis, not benchmark evidence or an assertion about GPU performance.

The fresh `q0/reference/nominal` diagnostic passes exact parity and places 49.49%
of instrumented runner time in assembly/evaluation and 33.98% in the complete
linear pipeline. Even eliminating that entire linear pipeline would yield only
1.515x on this circuit, before transfer/certification costs. A follow-up nested
profile records 111,243,720 full/AD and 49,913,520 value-only expression calls.
Per-expression clocks materially perturb that run (37.998 versus 33.375 seconds),
so its apparent combined expression/linear share is not an uninstrumented bound.

The first candidate therefore combines batched device expression evaluation with
real-FP64 cuDSS linear solves. This is a falsifiable screening experiment; retained
host assembly, validation, millions of transfers and the GPU's FP64 throughput may
still prevent useful acceleration. Neither diagnostic predicts a passing gate.

## Frozen GPU candidate

The separate CUDA core build preserves the CPU host controller and assembly while
replacing behavioral expression batches and real factorizations. One CUDA thread
evaluates each immutable compiled expression with the original lazy branch/domain
rules, analytic reverse derivatives and ordered dependency output. Device kernels
use native `double`, ordinary math functions and `--fmad=false`, with no fast math.
Every uncached assembly state and fresh final value check executes independently;
exact host cache hits retain their established state/descriptor identity rules.
Runtime-owned compiled graphs remain temporary and are never published as model text.

The linear candidate uses the existing checksum-pinned cuDSS 0.8.0.10 and CUDA
13.0.2 toolchain: GENERAL/FULL CSR, signed 32-bit indices, real FP64, BTF_COLAMD,
default factorization, global-column pivot search, threshold 1.0, static epsilon
`std::numeric_limits<double>::min()` (`2.2250738585072014e-308`), matching disabled,
internal iterative refinement disabled, hybrid execution
and hybrid memory disabled, host thread count one and host registration disabled.
BTF does not support the deterministic mode. Each job retains its own structure,
factors, stream and library objects; jobs do not share timesteps or numeric state.
No uniform batch forces divergent jobs into one implicit step.

Uploads equilibrate each row by its maximum absolute coefficient, retaining the
same scale for its RHS and refinement corrections. A nonzero coefficient that
would underflow to zero is rejected. All residual and backward-error acceptance
uses the original unscaled matrix and RHS; equilibration changes no tolerances.

Every factor/refactor synchronizes and checks both library status and perturbed
pivots. Any pivot perturbation fails closed, including for a zero RHS. The original
host residual guards and mandatory first nonzero residual correction remain; at
most four corrections are solved on the GPU. A failed refactor may trigger the
same bounded fresh-factor retry policy, entirely on the GPU. This is a numerical
factorization retry, not a failed-job retry or CPU fallback.
The minimum-normal epsilon avoids imposing an arbitrary `1e-13` conditioning
cutoff on mixed-unit MNA equations; zero or under-normal pivots still fail closed.

The frozen GPU mode uses four persistent workers. Each shares a 256 MiB ledger
between explicit buffers and cuDSS allocations, with explicit cleanup checking.
The parent also samples aggregate resident memory, including CUDA context and
kernel local-memory residency, against a pre-worker device baseline. The ledger
exactly bounds controlled allocations; sampled residency does not establish an
exact peak for opaque driver allocations. Both observed bounds must pass. Primary
context residency may survive jobs, while per-job graphs/factors/storage are released.

## Unchanged numerical and output policy

The experimental worker reuses the qualified EMI-02 parser, model import,
nonlinear assembly, Newton line search, physical-difference convergence checks,
source breakpoints, initialization and transactional adaptive BE/TRAP controller.
It selects the same audited derivative-history estimator and the same refinement
policy as the CPU reference. Ordinary solver targets retain their KLU implementation.

Each accepted output contains every frozen observable at every accepted output
time. Files close before completion is reported. Partial output cannot count as
a completed job. Parent-side admission independently checks job/input identity,
model/source/binary hashes, raw dimensions, finite values, monotone time coverage,
spectra, stress/loss metrics and final feasibility against fresh CPU trajectories.
Every invocation regenerates its reference trajectories; historical waveforms are
never substituted for new execution.

## Qualification, persistent workers and performance gates

Qualification covers all three DPT refinements and all nine candidate/corner
inputs at all three refinement levels. The CPU references also pass the existing
external ngspice and integration/observation refinement checks. GPU results must
pass complete CPU differential checks and refinement before timing is eligible.
A failed qualification is retained as negative evidence, and no performance gate
is evaluated from its surviving subset.

Persistent worker processes execute a tab-separated request protocol with exactly
three absolute paths: input deck, raw output and statistics. Each response binds
the request input path and reports completion or failure. Workers retain process
and library residency; numerical state is recreated for every job on both backends.
Workers neither respawn per successful job nor reuse solved trajectories.

The frozen timed domain is nine unique jobs and four replica-major copies of
those nine jobs. CPU modes use 1, 4 and 16 persistent workers, with single-threaded
numerical libraries. One complete warmup and five measured studies per mode and
size are required in each of two independent invocations. All scheduled jobs,
including failures, remain in terminal accounting. Every GPU replica independently
solves the entire initial-value problem and receives fresh CPU validation.

Complete-study timing charges input preparation, scheduling, transfers, device
work, synchronization, CPU result validation, spectrum/metric processing and closed
required files. Cold timing additionally includes setup and fresh qualification;
qualification is charged once. Whole-invocation wall time is also retained. Build,
dependency fetch and physical fsync are excluded. Instrumentation and kernel timing
cannot replace the complete-study boundary.

Passing requires, for both sizes in both invocations, at least 2.0x median and
1.5x nearest-rank empirical P95 speedup against the fastest qualified CPU mode for
that size. Cold GPU study time must not exceed the best cold CPU study. Memory,
accuracy, complete accounting, zero failures and zero GPU fallbacks are conjunctive.
Five observations do not estimate a population tail. Failure leaves dispatch closed.

## Resource and failure contract

At most sixteen jobs may be active; aggregate host memory is bounded to 16 GiB and
device allocation to 4 GiB. Per-job limits remain 120 seconds wall, 110 seconds CPU,
512 MiB raw output and two million points. CPU oracle workers retain the 1 GiB
address-space limit. CUDA virtual address reservations are not resident memory;
GPU workers require explicit resident host/device accounting rather than a 1 GiB
virtual-address cap. The parent enforces job deadlines and terminates an exhausted
worker. No selective retry repairs an existing failed result.

Unsupported/malformed input, wrong association, missing/truncated output, nonfinite
values, invalid timing grids, numerical failure, allocation exhaustion and cleanup
failure all prevent publication as a validated result or inclusion in a feasible
ranking. A valid predicted-infeasible circuit remains a successful simulation.
This experiment uses fail-closed execution and has no automatic CPU fallback.

## Resident follow-up candidate (not qualified)

`//cuda:emi03_resident_worker` is a separate experimental executor. The initial
host-controller/cuDSS candidate and its two failed evidence invocations remain
available. Neither candidate has passed the frozen acceptance gates.

The resident candidate keeps the adaptive BE/TRAP controller, derivative-history
estimator, Newton iterations, expression evaluation, factorization, refinement,
and accepted-state accounting on the device in bounded chunks. Each block owns
one complete circuit and uses native FP64. Output is streamed through the same
bounded observer and atomically published only after successful completion.
Ordinary CPU execution remains unchanged. GPU cuDSS still performs initialization;
KLU supplies symbolic ordering only, never a CPU numeric solve for this executor.

Immutable expression trees are scheduled by dependency level. Guards preserve
lazy conditional evaluation and inactive-branch domains. Reverse derivatives
retain the original ordered accumulation for aliased state dependencies.
The original node magnitude and finite-result checks remain. Affine RHS and
original-equation residuals use compensated FP64 products and sums.

The sparse numeric plan combines structural ordering and device-discovered row
pivots. Native FP64 row equilibration uses finite reciprocals, with direct division
when a reciprocal is not representable. Zero, under-normal, or invalid fixed
pivots trigger genuine GPU partial pivoting without pivot perturbation. A failed
original-equation residual guard permits one fresh GPU factorization and bounded
refinement; it never permits a CPU solve. Accepted Jacobians are checked even when
the response is zero. A private chunk may reuse a factor only after exact bitwise
comparison of every Jacobian value; a new chunk or changed matrix invalidates
reuse. The first dynamic factorization in a chunk ends that chunk
after its current trial, so the host can rebuild the symbolic plan from the
encountered device Jacobian. At most 32 plans are prepared per job; further numeric
pivoting remains on the GPU. Old plans are released as they are replaced.
These are numerical retries inside one IVP, not failed-job retries or CPU fallbacks.

State vectors, expression scratch, and sparse factors use explicitly bounded
shared storage. Controlled global buffers remain in the existing allocation
ledger. Job state and expression caches belong to their host thread; library
allocation callbacks use their explicit owner context. Private nonblocking streams
allow independent jobs to share the primary context. Each allocation owner has
bounded pinned staging of at most 526,336 bytes. Completion queries sleep for
250 us between attempts because blocking waits consumed a CPU core on the measured
WSL driver. Transfer, polling and waiting costs remain inside measured wall time.
All writers finish an output row before its shared index advances. The current
study
harness still uses its declared four persistent worker processes; the sixteen-job
shared-context test does not establish a new qualified scheduling mode.

The resident tests cover the 21 independent analytic accuracy fixtures, lazy
branches and active invalid domains, changing exact pivots, singular accepted
Jacobians with zero response, observer and attempt-budget failure, and sixteen
concurrent private jobs with one isolated allocation fault, plus 192 distinct
voltage outputs across multiple chunks. Passing those tests is not the thirty-job EMI qualification or a performance pass. Full frozen
qualification, resource accounting, and both independent performance invocations
remain required before EMI-03 can be called complete.

## Exclusions

No mixed precision, new physical model, repaired vendor `vt` interpretation,
optimizer, novel-candidate performance claim, automatic dispatch, distributed
execution, or laboratory/compliance claim is introduced. Historical EMI and AC
evidence and Rust source remain unchanged. Results are research predictions for
the frozen finite input set.
