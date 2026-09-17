# EMI-01 CPU reference study

This is an external research evaluator for the frozen [ADR-002 contract](../../docs/adr/ADR-002-inverter-emi-design-study.md).
It does not add SiC devices, model parsing, coupled inductors or GPU transient execution to Ohmnivore.
The four switches, both filtered conductors, harness, load and chassis stay coupled inside every
ngspice job. Nine independent candidate/corner jobs define one study.

The public Microchip model is fetched directly from its vendor with exact checksums. Original and
mechanically adapted model text remain in temporary local directories and are never included in
this repository or the evidence. Read [model provenance and limitations](MODEL_PROVENANCE.md),
[Python/runtime provenance](../../third_party/emi_python/PROVENANCE.md) and the
[proposed CPU capability slices](CPU_GAPS.md).

## Reproduction

From the repository root on Linux x86-64:

```sh
bazel test //reference/emi01:signals_test //reference/emi01:adapter_test \
  //reference/emi01:metrics_test //reference/emi01:study_test //reference/emi01:report_test \
  //third_party/emi_python:runtime_test
bazel run //reference/emi01:study -- --out=/absolute/new/directory/run-1
bazel run //reference/emi01:study -- --out=/absolute/new/directory/run-2
bazel run //reference/emi01:study -- --audit=/absolute/new/directory/run-1
bazel run //reference/emi01:study -- --audit=/absolute/new/directory/run-2
```

Output directories must be new. Do not run another build/benchmark during measured invocations.
The runner enforces one warmup and three measured full studies with one and four workers, plus
three DPT and 27 ensemble refinement solves: **102 terminal job records per invocation**.
No automatic retries, reused simulation results or silent failed-job exclusions are permitted.
Each serial/parallel job runs ngspice afresh; worker processes persist within one study. This
reference measures independent processes, not a persistent production KLU executor.

`--qualification-only` produces 30 qualification jobs, with an explicitly different terminal
schedule. `--exploratory-probe` produces only six diagnostic jobs and deliberately has no evidence
terminal; it cannot be audited or presented as a complete study.

## Output interpretation

`manifest.json` binds the ordered candidate/corner domain and settings. `metadata.json` identifies
source/model/binary/runtime bytes and hardware. `qualification.json` records each refinement
comparison. Each job directory retains its generated deck, simulator log, lossless raw links,
spectra and terminal result. `terminal.json` reconciles all identities, file hashes and counts.
`predicted_infeasible` is a complete numerical result, distinct from every typed execution or
validation failure. Candidate feasibility requires all three exact corners and passed invocation
qualification. No failing or missing simulation may rank as feasible.

Waveform storage preserves original ngspice bytes without repeatedly storing identical payloads.
`raw.header` contains the exact header through `Binary:\n`; `raw.json` orders 4 MiB payload chunks.
Each `blobs/<sha256>.gz` decompresses to the exact chunk whose SHA-256 is its filename. Concatenate
header and decompressed chunks to reconstruct the original `.raw`; `raw_sha256` verifies it.
Repetition only deduplicates required output storage; every job still reruns the simulation and
validation. `spectra.f64` is a little-endian float64 matrix described by `spectra.json`, with
frequency followed by conductor A/B, CM and DM RMS-equivalent amplitudes in amperes. dBuA is
`20*log10(max(amplitude,1e-15)/1e-6)`.

Per-job times separate preparation, simulation (including process startup, initialization and
ngspice raw-file write), raw validation, metrics/spectral output, and compressed waveform output.
The remaining interval includes cleanup and hashing. Study batch time additionally charges
worker creation, scheduling, IPC, shutdown and a complete summary write. One-time input verification/model adaptation precedes the batches and is reported as `setup_s`;
whole invocation time includes it. Import/interpreter startup before `main` and Bazel execution
are outside these internal timers. The final timing-field
rewrite is administrative self-report serialization; file closure, not physical `fsync`, defines
required I/O completion. Job latency is worker service time; queue-to-completion latency is also
recorded. CPU/RSS telemetry is measured separately for ngspice and Python; aggregate simultaneous
system peak is unavailable and must not be inferred by summing independent process maxima.

The mask is a research-only 90 dBuA flat single-bin RMS-equivalent tone mask from 150 kHz to
10 MHz, with 6 dB reserve, 10 kHz grid and 15 kHz Hann ENBW. It is not an aviation receiver,
quasi-peak detector, EMC certification, motor impedance validation or laboratory correlation.
The vendor model omits bipolar reverse-recovery charge and self-heating. Mass/rating and linear
magnetic assumptions are explicit hypothetical designs, not purchasable qualified components.

The load is an assumed passive `20 ohm + (100 uH || 100 ohm)` network with explicit terminal
capacitances. The winding-loss resistor damps the harness resonance and changes load power;
its measured simulation dissipation is reported separately from filter loss. It is not a
measured motor impedance. The DC-link capacitor sits before the explicit 20 nH commutation
inductance. Both decisions and candidate-specific integration refinements are frozen in ADR-002
(light/medium 2.5/1.25/0.625 ns; heavy 0.625/0.3125/0.15625 ns);
the original spectral accuracy limits remain unchanged.

See [retained results and reproduction evidence](../../docs/emi01-results.md) and the
[proposed future GPU experiment gates](../../docs/emi01-gpu-experiment-contract.md).
After both complete invocations, regenerate the CPU budget with:

```sh
bazel run //reference/emi01:report -- \
  --run=/absolute/new/directory/run-1 --run=/absolute/new/directory/run-2 \
  --out=/absolute/new/directory/cpu-budget.json
```

The report audits all raw outputs first. Its acceleration projections are explicitly idealized
counterfactuals; matrix factor/solve timers are nested within ngspice analysis and never added to
external wall-time phases. Failed or incomplete jobs cannot establish a fixed-accuracy speedup
budget. No measured GPU throughput is claimed.
