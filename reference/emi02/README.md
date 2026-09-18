# EMI-02 bounded CPU model bridge

This opt-in bridge imports the exact externally fetched EMI-01 v2 Microchip model
into the CPU FP64/KLU behavioral path. It expands two selected definitions, up to
four packages, retaining package RLC and the sensing-capacitor current multiplier
networks. It does not extend the ordinary simulator's SPICE parser. Model text
and generated flattened decks exist only in temporary local directories and are
not included in evidence or distributions.

The governing contracts are ADR-003 through ADR-006. The external measurements,
source circuits, v2 candidate/corner identities and tolerances come unchanged from
`reference/emi01`. All thirty mandatory CPU jobs and thirty independent ngspice
jobs are run before complete qualification can pass. An incomplete or failed job
suppresses whole-study qualification and feasible ranking. Each invocation needs
a fresh output directory; individual retries cannot repair an existing record.

The model runner selects ADR-005's opt-in physical derivative-history error
estimator for TRAP. BE startup and recovery retain step doubling. Actual
full/two-half audits run before history is available, periodically, at guarded
transitions and during fallback; sixteen consecutive accepted audit agreements
are required to recover. This numerical policy is limited to the pinned workload,
with unchanged physical tolerances and external qualification gates. The public
default retains the existing estimator. Prepared invocation-owned expression,
matrix and validation caches preserve the checked path's results; final original
system and KLU backward-error checks remain mandatory. CPU statistics schema
`emi02-cpu-v2` records method `trapezoidal`, estimator
`derivative-history-audited-v1`, audit/fallback counts and KLU work counts.

ADR-007 adds a generic analytic transient corpus and exact companion-matrix
preparation. An admitted immutable companion updates only entries present in C;
immutable G-only entries retain their already checked values. This changes no numerical policy or tolerance. The endpoint-scaling
experiment was rejected after broader circuits exposed increased global error.
Run `bazel test //cpp:transient_accuracy_test` for the independent accuracy and
prepared/checked trajectory gates; per-case benchmarks also retain coarse
stress-case failures rather than treating local error tolerance as a global bound.

```sh
bazel test //reference/emi02:importer_test \
  //reference/emi02:importer_oracle_test //reference/emi02:device_dynamic_test \
  //reference/emi02:qualification_test
bazel run -c opt //reference/emi02:qualification -- --out=/absolute/new/emi02-run
bazel run -c opt //reference/emi02:qualification -- --audit=/absolute/new/emi02-run
```

For diagnostic development, `--probe` runs just the coarsest DPT pair. A probe
always has `pass=false`, `complete=false`, and exits nonzero; it provides no
qualification claim. Full runs use the frozen per-job resource limits, original
candidate-specific integration refinements and independent observation grids.
The sixty engine/job executions use four spawned worker processes, each with
private simulator/model working files and the unchanged 110 s CPU / 120 s wall
limits. Completion logs stream immediately; retained records keep the frozen job
order. Worker count, spawn method and concurrent execution mode are audited.
This is concurrent qualification, not a CPU performance baseline or speedup test.
The audit independently recomputes raw measurements, spectra, refinement, all-level
differential checks and ranking, and verifies all source/model/binary/job/file
identities. It rejects probes, missing or reordered jobs, non-finite/duplicate
JSON fields and changed artifacts. It reports an honestly failed complete study
as `audit_pass=true, qualification_pass=false`; it cannot repair or promote it.
The C++ runner streams accepted states in the existing validated binary waveform
schema and publishes its output only after successful full coverage. Provenance
records bind model/archive/adapter, source files, both simulator binaries, import
counts, source/flattened deck hashes and individual job identities. Raw waveform
chunks and hashes use the existing external format. Diagnostic logs are hashed,
not published, because parser failures can include proprietary equation text.

## Exact compatibility and model fidelity

The independent DC import check discovered that ngspice 46 `ps` compatibility
reserves `vt`. It rewrites that token to ambient thermal voltage even when the
vendor defines a local parameter with the same name. The frozen EMI-01 reference
therefore executes `(27 + 273.15) * 8.6173303e-5` V in those expressions at both
27 and 125 degree C fixed junction temperatures. The bridge explicitly reproduces
that binding and records it; the generic expression evaluator does not reserve
`vt`. This is numerical parity with the historical ngspice interpretation, not
qualification of the vendor-intended local threshold equation. Repairing that
collision would define a different reference and require new evidence. Historical
EMI-01 results and model bytes are left intact.

The model's documented omitted bipolar reverse recovery, self-heating, avalanche
and hardware correlation remain limitations. No GPU dispatch, GPU speedup,
hardware EMI compliance or intrinsic switching-loss claim follows from this CPU
port. Current qualification state must be read from the actual retained evidence;
the existence of this bridge is not evidence that all solver gates pass.
