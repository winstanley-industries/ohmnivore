# EMI-01 independent accounting, provenance, and build review

Reviewed the reference harness at source commit `bf1165f`, separately from the
primary implementation work. The reviewer also implemented the bounded fixes
listed below. This document records source review and tests completed before
retained measurements. At writing, retained invocation 1 was running; this is
not a claim that either retained invocation completed or passed qualification.
No builds, benchmarks, or evidence audits ran while preparing this document.

| Finding | Resolution and discriminating coverage |
|---|---|
| Missing, duplicated, reordered, or unknown jobs could corrupt completion accounting or ranking. | Audit independently reconstructs the ordered 30-job qualification or 102-job full schedule. Ranking requires exact candidate/corner identities, all required corners, and successful global qualification. Missing or failed simulations cannot count as feasible. Tests cover every failure type and malformed identity/order cases. |
| A self-consistent rehashed result could assert false numerical success. | Audit regenerates deck/driver bytes, verifies model/source identities, restores raw output, validates process/log/telemetry consistency, and recomputes successful metrics, spectra, qualification, and ranking. Tests forge metrics, spectra, raw titles, decks, and qualification claims. Failed jobs remain explicit failures. Numerical recomputation shares the reference metric routines; it complements the separate numerical review rather than constituting an independent physics oracle. |
| Worker termination could omit a terminal record. | Failed futures produce durable typed records with expected identities and hashes of available partial artifacts. They remain in the denominator. An injected worker failure test verifies durable records and zero validated throughput. |
| Generated launcher names differed between study and report entry points. Runtime/build inputs were insufficiently bound. | An explicit 24-file identity set replaces directory scanning. Narrow runfiles groups provide identical inputs to both entry points. Tests ignore launcher/test-only files but reject changes to declared source bytes and metadata. |
| Corrupt compressed output could allocate unbounded memory or escape the blob directory. | Restore validates SHA256-only chunk names, index schema, declared lengths, chunk counts, and total size before reconstruction. Reads allow at most 4 MiB plus one sentinel byte per inflated chunk; raw reconstruction is bounded to 512 MiB, headers to 64 KiB, and parsed waveforms to 2,000,000 points. Tests cover missing/invalid gzip, a compressed expansion beyond the chunk bound, path injection, and inconsistent sizes. Original raw-byte SHA256 identities remain authoritative. |
| Fractional telemetry counters were silently truncated. | Integer counters now reject fractional values before conversion. Unavailable device-evaluation and assembly-only telemetry remains explicitly unavailable. |
| Heavy-candidate refinement required a different frozen integration grid. | Scheduling and independent audit reconstruction apply 0.625/0.3125/0.15625 ns to every heavy-candidate corner. Light/medium and DPT grids remain distinct. Tests cover every level/corner, measured finest-step selection, and rejection of a rehashed default-grid heavy job. |
| Time-to-lightest could precede completion of a lighter infeasible candidate. | The reported proof time conservatively waits for all study-job completions. A test makes the feasible heavy candidate finish before slower infeasible lighter candidates. |
| Preparation, derived throughput, and acceleration budgets could be misrepresented. | Batch timing starts before job materialization. Throughput is recomputed from validated counts and measured wall time. One-time setup is reported separately and included in invocation time. FIFO counterfactuals preserve non-simulation service and fitted nonnegative scheduling overhead; negative fit residuals are exposed. Factor-plus-solve Amdahl ceilings are separate, optimistic analysis-only models with nested timers. Both budget types are unavailable after failed global qualification. |
| The paired report could accept one invocation twice. | Resolved paths and terminal SHA256 identities must differ; each terminal identity is included in the report. Tests reject duplicate paths and copied terminal records. This is an accidental-reuse guard, not cryptographic proof of independently executed invocations. |

The 24 bound files comprise twelve files under `reference/emi01` (`BUILD.bazel`,
`CPU_GAPS.md`, `MODEL_PROVENANCE.md`, `README.md`, `adapter.py`, `circuits.py`,
`manifest.json`, `metrics.py`, `report.py`, `requirements.txt`, `signals.py`, and
`study.py`); four under `third_party/emi_python` (`runtime.py`,
`runtime_rules.bzl`, `BUILD.bazel`, and `PROVENANCE.md`); two under
`third_party/ngspice` (`BUILD.bazel` and `PROVENANCE.md`); ADR-002; and root
`MODULE.bazel`, `MODULE.bazel.lock`, `BUILD.bazel`, `.bazelrc`, and `.bazelversion`.
Results/review documents are intentionally outside that source-input set.
Model archive/member/adapter identities and actual simulator, interpreter, and
native runtime hashes are also recorded.

The build uses pinned Bazel Python 3.12.13 and NumPy 2.4.3, with pinned native
runtime dependencies and one numerical-library thread per worker. Source review
and runfiles comparison verified the shared 24-file set. The existing ngspice
build remains the external reference; these additions do not change production
CPU FP64/KLU authority, Rust/Cargo, C++/CUDA behavior, or historical AC evidence.
No EMI-02 implementation, GPU executor, kernels, dispatch, or optimizer was added.

Validation completed before retained measurements:

- `bazel test //reference/emi01:study_test //reference/emi01:report_test`: **43 study tests and 18 report tests passed**.
- `bazel build //reference/emi01:study //reference/emi01:report`: passed.
- Focused Ruff checks and `git diff --check`: passed.

Timing and memory remain observations of the running processes, not externally
attested measurements. Internal timers exclude interpreter/import startup and
Bazel build/fetch. Required output includes a complete summary write; the final
timing-field rewrite is administrative. Completion means closed files, without
an `fsync` durability claim. Process RSS maxima do not measure simultaneous
system peak memory. Acceleration figures are counterfactual CPU budgets, not GPU
measurements. This review does not establish laboratory correlation, compliance,
vendor-model accuracy outside the frozen scope, or successful retained-run
completion; those conclusions require the separate numerical evidence and
terminal audits after both invocations finish.
