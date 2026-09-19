# Resident owner-pool and synchronization checkpoint

EMI-03 is incomplete. This checkpoint validates a private 16-owner process pool,
three fresh DPT trajectories and targeted CUDA memory/synchronization checks.
It does not pass the required full thirty-job qualification or either independent
9/36-job performance invocation. The frozen limits and timing targets are unchanged.

The worker uses one CUDA process/context with sixteen private owner threads,
channels, CPU affinities, numerical states and output paths. Device allocation
completes on each job's private allocation stream. One private pinned staging
buffer survives factor-plan replacement. The parent checks owner/thread/request
identity, counts shared-process host memory once, sums device allocation peaks
across owners and conservatively charges whole-process CPU time to each active
GPU request. Shared-process failure retains every affected failure without retry.

Compute Sanitizer found bugs that ordinary numerical tests had missed:

- v77 freed factor-plan buffers on an upload stream before the simulation stream
  finished consuming them. Allocation owners now synchronize their consumer
  before release, including exception cleanup.
- v78/v79 allowed one warp to invalidate expression-cache metadata while another
  still read it. All warps now finish the decision before invalidation.
- v80 exposed a related nonlinear recovery race: clearing an error could change
  a slower warp's recovery branch. Barriers now separate shared control decisions
  from mutation, and the shared maximum is consumed before its next reduction.

The v81 targeted memcheck, initcheck and synccheck runs cover all launches in the
wide-output, dynamic-pivot and 16-concurrent-job tests with zero errors. Racecheck
covers all launches in the nested lazy-branch/error-recovery and dynamic-pivot
cases. The independent 64-expression and wide-output race checks cover their
first resident kernel chunk; all ordinary numerical tests cover full trajectories.
Exact commands, scope, binary/DSO hashes and unsuppressed reports are retained.
A separate all-launch concurrent racecheck reached its 600-second diagnostic
ceiling without completing; `sanitizers/81-concurrent` records this timeout as
failed coverage, not a pass. Its instrumented processes exited before ordinary
performance diagnostics resumed.

Failed attempts remain under `sanitizers/77*` through `sanitizers/80`. v77's first
attempt could not load a copied binary's shared libraries and is not a CUDA
finding; its corrected-runtime memcheck reported 2,801 memory errors. The v78
race run found hazards, then hit an internal tool error during the concurrent
case and was interrupted; it is not a pass. v79's conditional invalidation still
raced. v80 passed its initial selected checks but failed the broader expression
recovery case. Source snapshots are retained for the pre-lifetime fix and v80;
intermediate failed runs also identify their exact test binaries, but no complete
source-reconstruction claim is made for every intermediate variant.

## Fresh DPT through the real pool

| Refinement | CPU request wall | GPU request wall | CPU/GPU points |
|---|---:|---:|---:|
| q0 | 0.596 s | 8.526 s | 17,971 / 17,766 |
| q1 | 0.807 s | 11.669 s | 24,817 / 24,868 |
| q2 | 1.405 s | 14.492 s | 39,171 / 39,171 |

All three full DPT waveform/switching comparisons and all eight integration/output
refinement checks pass against fresh CPU runs. The GPU requests run concurrently
through three of the sixteen available owners; CPU uses one owner. Complete DPT
study wall time is 14.587 s GPU and 2.885 s CPU. This is a correctness/resource
diagnostic, not a performance comparison against the required best CPU mode.

The GPU pool observes 300 MiB incremental device residency, 3,168,165 bytes summed
native owner peaks and 1,146,712,064 bytes aggregate host high-water memory, with
108 device samples and zero monitoring errors. All six CPU/GPU records pass the
DPT resource audit. This does not establish the resource envelope for all sixteen
active coupled circuits. GPU jobs report no fallback, allocation, cleanup or
outstanding-allocation failures.

`identity.json` binds all measured source files and both executables.
`sources.tar.gz` contains those exact repository source files; `source.patch.gz`
is the tracked-file diff from the recorded base and the archive also includes new
files. `dpt/` retains raw trajectories, metrics, resources and worker identities.
The private adapted vendor deck and worker logs are not published; their input and
log identities are recorded. `audit.json` independently reconstructs all six raw
trajectories and recomputes every DPT metric, comparison and refinement.

`validation/` retains the canonical CPU build/tests/lint, ASAN, UBSAN, lockfile,
ngspice/prepared-AC checks and optimized replay benchmark, six CUDA targets, the
actual worker-process integration test, and eight expected CUDA/sanitizer analysis
rejections. The protected-path scope audit remains clean. CPU checks precede the
final GPU-only synchronization edits; the final CUDA and sanitizer checks cover
the current kernel. `validation/extra-sources.tar.gz` supplies the additional
canonical-test and linkage sources, including the analytic accuracy fixtures,
with a separate hash manifest.

## Reproducing CUDA tooling

`profiler/` reconstructs the separate checksum-pinned Bazel profiler workspace.
It uses NVIDIA Compute Sanitizer 2026.3.0.0 from CUDA redistributable 13.4.2;
the solver remains built with CUDA 13.0.2. Build
`@compute_sanitizer_current//:files`, then use the inner
`compute-sanitizer/compute-sanitizer` executable. Copied test binaries need the
recorded sibling `_solib_local` libraries to preserve their Bazel RPATH.

Nsight Systems software traces are retained in the adjacent
[profiling diagnostic](../resident-profiling/README.md). Nsight Compute hardware
counters remain permission-blocked. Source line information is now included in
the EMI-03 CUDA library; it does not change the toolchain or enable GPU dispatch.
Profiler and sanitizer runs are excluded from acceptance timing.
