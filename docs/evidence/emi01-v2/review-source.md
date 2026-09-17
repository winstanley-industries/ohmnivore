# Independent pre-evidence source review

Reviewer: `independent_final_review`, a separate reviewer who did not author the v2
implementation, tests, manifest, or numerical diagnostics. This reviewer proposed the
control-role correction and additional ranking coverage; the implementation author made them.
This review was completed on 2026-09-17 against the working tree based on
`9021ad4f5820b7e0fff757369e5212c7ddcf66e5`, comparing the follow-up with historical delivery
`c621e5cfcedf2de934c8046595dbb9f7aa59e607`.

**Disposition: no remaining source-review blocker found before canonical qualification and
the two retained v2 invocations.** This is not a claim that those invocations have completed,
that retained numerical results pass, or that delivery is ready to merge. A separate final
numerical, provenance, accounting, and scope review remains necessary.

## Scope and method

Read the governing ADR-002 additions, v2 manifest, study/report/circuit/build changes, related
tests, reference documentation, future GPU proposal, and exploratory diagnostic reports and
their helper. Inspected strict audit reconstruction and failure paths in the unchanged code
surrounding the edits. This reviewer ran no build, test, simulator, FFT, raw-waveform audit, or
benchmark during this review. Reported exploratory values below come from the separately
authored independent diagnostic reports, not new numerical computation by this reviewer.

The reviewed diff contains no changes to production C++/CUDA, Rust/Cargo, acceptance circuits,
replay inputs, or historical `docs/evidence/emi01` files. The original manifest, adapter,
spectral/physical metric implementations, and Python/ngspice dependency definitions are unchanged.
Circuit code changes only versioned titles and their arguments; the new electrical values enter
through the explicitly selected manifest. The old evidence remains auditable from its detached
historical source commit, without replacing its recorded source identities.

## Findings and closure

The initial v2 role gate checked the reference and boundary but accepted a passing `light`
control. That conflicted with the declared failing-control coverage. The corrected manifest
and ADR require `light` to remain valid, settled, and predicted infeasible at every corner.
The role checker now requires nonempty violations consistent with its recorded stress and
margin values. Missing, failed, unsettled, unexpectedly passing, and forged-violation cases
fail the role gate without rewriting numerical results. Tests specifically discriminate these
cases. The ordinary audit still recomputes physical metrics and classifications from raw data.

The earlier manifest-order ranking risk is resolved: ranking uses computed mass, then candidate
identity for exact ties, while execution retains manifest order. Reviewed tests now exercise
both equal-mass ties and nonmonotonic masses that conflict with name and execution order.
Incomplete candidates and candidates with failed/infeasible corners cannot rank as feasible;
global numerical qualification is also required.

## Contract conclusions

- Version selection is explicit, defaults to v1, and requires exact canonical manifest bytes.
  Both manifests are source-bound. Metadata, job inputs, generated deck titles, terminal records,
  and report pairing carry or verify the selected version. Mixed or incorrectly selected report
  pairs fail. Current source audits cannot silently certify historical source hashes.
- Numerical qualification, simulation failure accounting, and fixture-role acceptance remain
  separate. The 30-job/40-check qualification and 102-job complete schedule are reconstructed.
  Roles are checked at the finest qualification and every warmup/measured study. Performance
  budgets require numerical qualification, role acceptance, and zero failures; valid infeasible
  designs remain numerical results, not solver failures.
- The reference must pass every corner. The boundary must pass all physical/settling screens
  and have a worst margin in inclusive [5,7] dB. Exactly 6 dB still passes predicted feasibility.
  Reports locate the worst retained bin across all four observables and report corner, frequency,
  distance from 6 dB, all integration-level margins, and classification changes.
- The static ngspice executable is hashed, copied into an invocation-private directory, set to
  mode 0500, rehashed, and used by every worker. Preparation is charged to invocation time.
  This limits interference from subsequent build-output replacement; it is not execution
  attestation and does not repair the documented ngspice build-path byte-reproducibility limit.
  Final runtime identity checks and strict evidence audits remain separate requirements.
- Exploratory finest-integration reports show approximately 5.516619 dB for the 80 nF boundary
  and 10.102246 dB for the reference, with stable classifications and passing recorded refinement
  checks. These support selection only. Retained qualification must independently establish the
  final values and all original numerical gates without weakening the 1 dB/1 uA spectral test.
- The roughly 3.8 kg designs remain hypothetical. Geometry consistently enters the existing
  mass, copper-resistance, and flux equations; fixed CM parasitics, omitted DM self-capacitance,
  and missing fill/fringing/core-loss validation prevent hardware or minimum-mass claims.
  New CPU budgets apply only to v2. Future GPU comparisons retain complete-job timing and fresh
  CPU validation of the fixed inputs; no GPU implementation or dispatch is authorized here.

## Reviewed file identities

These SHA256 values identify the principal source and contract bytes read at closure. A later
source change requires review of its delta; final evidence independently binds the full source set.

| File | SHA256 |
|---|---|
| `reference/emi01/study.py` | `8d059af34b269e532c74d7240a850d6b1e6065daa085c44ab725bd0150042e35` |
| `reference/emi01/report.py` | `4d117fddc162ee8802973dcdd3de307274d2b91a17c3bec296e1438f1fabb571` |
| `reference/emi01/circuits.py` | `5d23ab959b45a9194f766b7503eebf260bd0a14bd57f6dd4b4797e838743772c` |
| `reference/emi01/BUILD.bazel` | `9fc53a55fa656def3e516110a59929107d51e12725db2c0e3511bb9374dabcb3` |
| `reference/emi01/manifest-v2.json` | `ec68ddb48866d8127fbdd0ecf411dfc326b6c7bc7929c10429e40edd9cd2c81f` |
| `reference/emi01/study_test.py` | `784a16be0dfd1c8136daf6ee7b005f3be0637f831604dc6a545b3320bfe0d9dc` |
| `reference/emi01/report_test.py` | `a12eca0f386464e7ca54642f9683c3037a85583bb1eb186bb5b7e377ce6f31a1` |
| `reference/emi01/metrics_test.py` | `acedeac58fde252426289dd400df776418f125d2c7552794a46604147b94213c` |
| `docs/adr/ADR-002-inverter-emi-design-study.md` | `b0b153eed33cf00db778941ebc5e58bfd3575f7ccb93bc891ed0f4f50e5e3ab4` |
