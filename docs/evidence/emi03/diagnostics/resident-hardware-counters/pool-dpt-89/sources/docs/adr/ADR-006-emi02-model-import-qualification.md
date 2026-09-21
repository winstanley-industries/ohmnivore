# ADR-006: EMI-02D bounded SiC import and coupled CPU qualification

- **Status:** Implementation contract authorized by the all-four-slices EMI-02 request
- **Authority:** ADR-001, ADR-002 and `reference/emi01/CPU_GAPS.md`

## Exact model and import boundary

The import bridge accepts only the checksum-pinned Microchip archive, member and
adapted hash in `reference/emi01/MODEL_PROVENANCE.md`, using the unchanged EMI-01
adapter. The complete archive and member are checked before parsing. Only the
reachable `MSC040SMA120B` and `MSCSICFET1200` definitions are expanded. At most four
package instances, sixteen total expansions, depth three, 512 MNA unknowns and the
ADR-004 expression budgets are admitted. Junction temperature is exactly 27 or
125 degrees C; GM, VTO and MDE keep the pinned defaults. Unknown package,
parameter, card or model identity fails closed. There is no include search,
command interpreter, model catalogue or model substitution.

The bridge emits a local temporary flattened deck. RLC and independent sources
retain native descriptors; E/G/B equations retain their original syntax, graph,
output orientation and auxiliary unknowns, except for deterministic namespace
binding and numerical parameter substitution. Local parameters resolve forward
references and reject cycles; overrides evaluate in caller scope, including
`MDE=MDE`. Parameter exponentiation follows ngspice's parameter dialect and is
separate from runtime behavioral exponentiation. Node and branch identifiers are
case-insensitive. Internal names cannot alias external terminals or other
instances. All expanded identities are checked before producing output.

Neither original/adapted model text nor flattened decks containing model equations
may be committed or included in retained evidence. Retained data include hashes,
source identities, job identities, named waveform measurements, diagnostics and
comparison results. Temporary files are removed on completion and failure.

## Frozen ngspice compatibility discovery

Independent DC comparison against the actual pinned ngspice expansion revealed a
load-bearing reserved-token collision absent from the EMI-01 audit: in `ps` mode,
`vt` becomes `(temper + 273.15) * 8.6173303e-5`, including occurrences inside
expanded local parameter expressions. The selected library's local `vt` definition
therefore does not control the executed reference threshold. The reference uses
ambient temperature 27 degrees C at both fixed junction-temperature corners.
The narrow bridge reproduces this exact frontend binding before resolving
parameter dependencies; it does not silently repair the historical reference.
This explicit compatibility rule is separately recorded in import provenance and
checked across forward/reverse DC bias at both temperatures. CPU DC comparisons
use 1e-7 SI absolute plus 1e-6 relative; original-versus-flattened ngspice checks
use 1e-9 SI absolute plus 1e-8 relative. A vendor-intended
threshold implementation would be a different model/reference requiring new
qualification. Existing EMI-01 evidence remains untouched and is not asserted to
validate the vendor-intended threshold equation or hardware switching behavior.

## Initialization, source and output contracts

Only frozen EMI-01 v2 DPT and candidate/corner circuit generation is supported by
the qualification entry point. Every PWL/PULSE source gains an explicit DC value
equal to its value at time zero. This gives initially -3 V DPT gates, -3 V study
high-side gates and +20 V study low-side gates. High-side references, exact finite
10 ns transitions, periods and durations are preserved. The import does not change
ordinary parser defaults. No UIC or injected hidden state is admitted.

The four-argument reference TRAN card maps to the existing maximum timestep,
complete integration from zero and exact stop. No reference `options`, `save` or
control-language syntax enters the production parser. Named accepted CPU state
samples are extracted into the exact unchanged external measurement schemas.
Missing, duplicate, truncated, non-finite, non-monotonic or unbracketed samples
are typed failures, never extrapolated. Output sampling is independent of the
adaptive accepted timestep sequence.

## Standalone selected-device dynamic gates

In addition to the bias grid, execute one package at TJ_C 27 and 125 degrees C
with (a) drain clamped to 400 V and gate ramped -3 to +20 to -3 V, and (b) gate
clamped to -3 V and drain ramped 0 to -3.5 to 0 V. Both fixtures integrate
0..80 ns, hold their initial level through 10 ns, ramp over 10..20 ns, hold
through 40 ns, return over 40..50 ns, then settle. This is a numerical
clamped-terminal test, not a device SOA or hardware operating recommendation.
Retain the entire package/model graph and no-UIC initialization. Compare CPU and
original external ngspice at maximum steps 0.5, 0.25 and 0.125 ns, using independent
0.1 and 0.05 ns observation grids. Current waveform RMS gates are 2 mA plus 2%
reference RMS for gate current and 20 mA plus 2% for drain current. The imposed
terminal voltage RMS gate is 1 microvolt plus 1e-6 reference RMS. Signed charge
in each 0..30 ns and 30..80 ns window must agree within 0.2 nC plus 2% reference
magnitude. Apply those same gates to adjacent integration levels and both
observation grids for each backend; retain signed current orientation rather
than absolute-value integration. No tolerance changes depend on measured output.

## Bounded raw emission

The runner keeps one preallocated record of time followed by the selected saved
state values. Each observation retains the original point/byte budget, time,
shape and finite-value checks, then writes that complete record in one binary
stream operation. FP64 representations, column order, point order and file
contents remain identical. The record is private scratch; failures still remove
unpublished output and metadata. This changes stream-call overhead only and does
not buffer complete trajectories or relax output/resource limits.

## Qualification and failure accounting

Each full qualification invocation runs DPT at all three frozen integration
refinements and all nine coupled bridge/filter/load/chassis jobs at each of their
three frozen candidate-specific refinements, with three independent observation
sample spacings. The CPU and ngspice references use the same frozen source,
manifest, model and job identities. CPU BE/TRAP remains as defined in ADR-005;
the model runner explicitly selects its audited derivative-history policy and
records method `trapezoidal` and estimator `derivative-history-audited-v1` in `emi02-cpu-v2` statistics. The audit
checks estimator, fallback, refinement and step accounting as well as raw output.
Refinement corrections are counted separately from successful top-level solves.
Their admitted upper bound is four times the sum of successful numeric
factorizations, refactorizations and numeric reuses, including work in failed
solve attempts; a successful-solve count alone is not a valid correction bound.
ngspice retains its original Gear-2 options. No tolerance or timestep gate is
relaxed to produce a passing result. Reusing prior oracle evidence requires
verifying the entire evidence source/manifest/model identities and waveform
hashes. No selective single-job repair is allowed.

Qualification schedules the sixty independent engine/job executions in a fixed
four-worker process pool using Python's `spawn` start method. Each worker uses a
separate simulator process, private model/working directory and engine/job output
directory. The original per-job limits remain 110 CPU seconds, 120 wall seconds,
1 GiB address space and 512 MiB per output file. CPU and reference records are
reassembled in the frozen thirty-job order, regardless of completion order; each
completion updates bounded progress evidence and emits a flushed status record.
Shared numeric raw chunks are published atomically. No missing, duplicate or
failed job may become a successful record, and an infrastructure exception aborts
without complete qualification or selective retry. The invocation and audit bind
exactly four workers, the spawn method and `concurrent-qualification-v1` execution
mode. A DPT probe submits only its two engine jobs to that same pool and remains
incomplete. This concurrency bounds qualification turnaround; it is not a fair
CPU baseline benchmark and provides no performance or speedup claim.

The retained execution record includes the launcher's allowed logical CPU mask,
logical-to-physical core topology, processor/virtualization identity and exact
command. A launcher affinity mask is inherited equally by all CPU and reference
jobs; it does not change the four-worker execution contract or any per-job limit.
An allowed set of four physical cores does not guarantee exclusive core ownership.
Resource qualification applies to the recorded environment and is not a portable
runtime guarantee.

The additional DPT differential waveform gate uses 10 V drain RMS, 0.5 V gate
RMS and 0.02 A plus 2% reference RMS for each saved current.
The unchanged EMI-01 measurement/refinement functions determine waveform, signed
terminal energy, switching time, peak Vds, ringing frequency/damping, CM/DM
spectrum, flux, stress and physical screens. DPT unresolved ringing remains an
accuracy failure. All above-floor bins satisfy 1 dB agreement and below-floor
bins satisfy the frozen 1 microamp absolute gate. Whole-candidate feasible
ranking requires all mandatory corners and every qualification gate. Missing or
failed jobs suppress a success claim; failure is reported as provenance,
unsupported input, parser, simulator/numerical, timeout/resource, output or
accuracy failure, with exact bounded execution diagnostics. Solver failures
never become predicted EMI infeasibility.

Acceptance requires all EMI-02A-C gates, independent importer/identity/source
initialization tests, all thirty CPU job trajectories and matching external
trajectories, independent integration/observation refinement, canonical C++
checks and review of model equations, Jacobians, dynamic state and identities.
A draft may report an unresolved qualification gate honestly; it must not label
EMI-02D qualified or authorize GPU work without that evidence. The imported model
retains its documented omissions, including bipolar reverse recovery,
self-heating and avalanche qualification. No hardware or GPU speedup claim follows.
