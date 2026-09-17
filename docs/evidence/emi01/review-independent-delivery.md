# EMI-01 independent delivery review

Review date: 2026-09-17. Frozen implementation:
`bf1165fe92fb5f527bbddfcb53374956d62929a5`; baseline:
`17519fc869a7088eccdd8fed53ca1a3758758972`.

**Disposition: no unresolved blocker to the bounded, unmerged draft delivery.**
The two retained invocations qualify numerically, reconcile all jobs, and support
the reported CPU measurements and negative finite-candidate result. The ngspice
build is **not byte reproducible across Bazel sandbox paths**. That limitation
is explicitly recorded and does not become a claim of binary reproducibility or
performance for the subsequently rebuilt executable.

This review is by `independent_final_review`, who did not author the harness,
adapter, fixtures, metrics, tests, numerical-review helper, runtime investigation,
or performance report. This reviewer authored only the earlier
[source review](review-independent-final.md) and this completion record. The
earlier review remains a historical account written during retained invocation 1.
No frozen source or retained run directory was edited by this reviewer.

## Accounting and numerical acceptance

Independent standard-library inspection reconstructed the exact ordered schedule
without calling harness scheduling, ranking, metric, or audit helpers. For each
invocation it checked terminal/result identities, process success and one-attempt
records, all required candidate/corner identities, final numerical status, the
40 qualification checks, complete rankings, and positive timing relationships.
It rehashed all 708 indexed small per-job artifacts per run and every completion
record. The canonical paired report separately restored and audited all raw
payloads; this reviewer did not repeat that full large-data operation.

| Check | Run 1 | Run 2 |
|---|---:|---:|
| Expected / terminal / validated jobs | 102 / 102 / 102 | 102 / 102 / 102 |
| Failed or missing jobs | 0 | 0 |
| Qualified DPT records | 3 | 3 |
| Numerically valid, predicted-infeasible ensemble records | 99 | 99 |
| Passed qualification checks | 40 / 40 | 40 / 40 |
| Independent spectral / waveform / settling checks | 144 / 216 / 54 | 144 / 216 / 54 |
| Independent failed checks or metric discrepancies | 0 | 0 |

The two terminal identities are distinct and match their audit/report bindings:

- Run 1: `68aafdb695edc2e3d50bc08f19a8dcdc0114f2aa0ac88370824051db86409b35`.
- Run 2: `6a53b691412525090f7a104f95f90d6a55cb65508477faa37463d05f21285930`.

All warmup/measured metrics equal their corresponding finest qualification
metrics, and all 102 job metrics agree across invocations. Every full-study
ranking contains the complete three-corner set for each candidate; none is
feasible. Lightest-feasible identity and its completion time remain null. No
solver failure, missing job, or failed corner is counted as a feasible design.

The separately authored numerical reports are bound to the correct helper,
manifest and terminal records. Their maximum spectral errors are 0.555320919 dB
above the accepted 10 uA floor and 0.273035457 uA below it. The original 1 dB /
1 uA gates therefore pass. Historical exploratory reports with alternate floor
fields are clearly distinguished by the sidecar and are not used to replace
retained qualification.

This reviewer additionally applied the frozen DPT observability and adjacent
refinement gate arithmetic to the independently recomputed DPT values in both
reports. Loaded current, signed energy, edge timing, gate samples/drop, Vds peak,
and measured decaying ringing pass. The 24.263 V full gate peak is retained and
reported. The [-3,23] V requirement concerns the specified 50% Vds crossings;
it is not an absolute gate-oxide rating test. No gate-rating, parasitic-turn-on,
complete SOA, hardware, thermal, or aviation qualification follows.

## Provenance and the executable rebuild limitation

Both retained invocations bind the same 24 frozen sources, exact model/archive/
adapted identities, interpreter and eight native runtime artifacts. Current
source hashes were independently compared with both metadata records. The
successful pre-rebuild canonical-artifact audit is historical evidence for the
recorded `b2574df8...` executable, not a statement about the later rebuilt file.

The subsequent strict artifact check failed when rebuilding produced
`afb6b348...`. The failure remains in the delivery. The
[investigation](review/ngspice-reproducibility.md) identifies absolute installation
prefixes containing Bazel sandbox numbers. A longer prefix changes linked data
placement as well as literal strings; normalized disassembly is expressly not
treated as a general semantic proof.

This reviewer independently hashed the recovered historical executable and
confirmed the complete retained SHA256:
`b2574df80f88a02b3d0597206277b189686e8708dcf8449fec42f250e41d74a6`.
Its exact six changed byte offsets relative to the preserved exploratory copy
match the recovery record. The recovery did not substitute a new identity for
either retained invocation.

The rebuilt executable then completed a separate 30-job qualification-only
diagnostic with zero failures and all 40 checks passing. The independent
comparison records exact equality of all numerical payload bytes, headers with
only the Date line removed, spectra, metrics, input/status identities and
qualification results. This reviewer checked all 30 comparison rows, their
terminal bindings, and the investigation artifact hashes. Diagnostic metadata
also matches all 24 sources, model/manifest, Python/NumPy, interpreter, all eight
native libraries and resource settings; only the ngspice executable identity
differs.

This closes numerical reproduction for the finite workload, with an explicit
remaining build limitation. The original timings remain measurements of
`b2574df8...`; diagnostic timing is neither a third retained invocation nor a
measurement of equivalent performance. New builds must record and qualify their
own executable identities. Neither these hashes nor local process observations
constitute independent execution attestation. Proprietary model bytes remain
outside the repository and evidence.

## Performance, scope and final interpretation

Independent arithmetic from the retained records reproduced all four mode
aggregations, empirical job P95s, queue/completion statistics, physical/mass and
stress tables, whole-invocation memory maxima, and all twelve FIFO counterfactual
samples. The reported serial/four-worker median study times are respectively
541.444/184.187 s and 538.076/185.386 s. The 27-job P95s are empirical observations,
not population-tail estimates; service latency excludes separately reported
queueing. Required output, startup/shutdown and summary work are charged to the
declared timing boundaries. RSS maxima do not establish simultaneous aggregate
system memory.

Factorization plus solve occupies approximately 11.44--11.54% of ngspice analysis.
The reported 1.129--1.130x optimistic analysis-only ceiling supports rejecting
factor/solve-only acceleration as a route to the proposed 2x complete-study gate.
The broader simulator-phase counterfactuals preserve measured outside-process
work but omit any new GPU transfer/setup/certification costs; they remain
optimistic opportunity bounds, not GPU results. The future GPU proposal now
precisely limits warm speedup to fixed-input replay after charged fresh CPU
reference generation. It does not claim faster discovery or novel-candidate
throughput once those inputs' feasibility is already known.

Protected-path Git comparison remains empty for production C++/CUDA, Rust/Cargo,
acceptance fixtures and replay inputs. The scope record preserves all eleven
historical AC evidence files and the 24 frozen source inputs. The evidence-only
`.gitattributes` rules preserve original bytes and exempt original progress-log
whitespace; they do not alter recorded simulator logs. Validation-index log
hashes were checked. No new solver/device semantics, GPU kernel, dispatch,
optimizer, mixed precision, or distributed execution is delivered.

The results and evidence index disclose the negative physical/research-mask
result, hypothetical load/magnetics/mass assumptions, missing device physics,
historical exploratory failures, timing exclusions, and executable rebuild
limitation. Those are material limits of the accepted reference, not assertions
that the designs are suitable hardware. This review authorizes no downstream
implementation or merge; publication/head checks remain separate delivery steps.
