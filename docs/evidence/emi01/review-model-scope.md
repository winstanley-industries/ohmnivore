# EMI-01 model, provenance and scope review

Reviewed source: `bf1165fe92fb5f527bbddfcb53374956d62929a5` (`bf1165f`).
Baseline: `17519fc869a7088eccdd8fed53ca1a3758758972` (`17519fc`).

**Signed by role:** model/provenance and physical-scope reviewer, `model_audit`
subagent, separate from the main harness author and retained-study executor.
This role previously contributed the adapter, provenance document and proposed
CPU contracts; this record discloses that authorship and does not represent an
independent second-author certification of those files.

The review covered the selected public model and mechanical translation,
fixture connections and measurement directions, frozen numerical identities,
dependency provenance, and implementation boundaries. Retained measurements
were in progress when this document was written. This review does **not** claim
that either complete invocation, numerical qualification, or performance gate
has passed. No build, simulation, audit or benchmark was launched during those
measurements for this record.

## Findings addressed before retained measurements

| Finding | Resolution checked in the reviewed source |
|---|---|
| Unmodified vendor syntax is not directly supported by pinned ngspice 46. | The original is explicitly rejected; archive/member/adapted hashes bind the narrowly identified TEMP rename and two behavioral-current syntax translations. Branch directions, baseline capacitors and all model parameters are preserved. |
| A capacitor connected directly at bridge node `p` bypassed the declared 20 nH at switching frequencies. | Both fixtures now place Cbus at `p1`, before the 20 nH commutation inductance. The held-off DPT gate has its explicit 4.7-ohm resistor. The final DPT contract requires resolved, decaying ringing. |
| The ideal inductive load left a nearly lossless differential harness mode near 5.14 MHz. | The declared hypothetical load is `20 ohm + (100 uH || 100 ohm)`. Its separate resistor dissipation and load-inductor current are retained. This changes the physical workload, not the device equations or spectral acceptance limits. |
| Executed runtime setup was missing from source identities. | The source binding now covers 24 explicit files, including runtime setup, governing ADR, dependency lock and relevant build/provenance inputs. |
| Numerical prose and fixture wording lagged the final choices. | ADR, manifest and code agree on original solver tolerances, light/medium and heavy refinement grids, 512 MiB raw/file budget, CM-only winding capacitance, and failure for unresolved DPT ringing. |

The two coupled-inductor pairs have positive inverter-side dot conventions;
equal outward currents reinforce CM flux. Gate sources use the appropriate
local source reference. Port current sensors point from filter toward harness.
Each job retains the complete coupled bridge/filter/load/chassis network.

The manifest's heavy-candidate integration override is applied both by job
generation and by the separately reconstructed audit schedule. The common
output grids and original 1 dB above 20 dBuA / 1 microampere below-floor
spectral gates remain unchanged. The two-million-point, 1 GiB simulator
address-space, 110 s CPU, 120 s wall and zero-retry limits remain explicit.

## Provenance, limitations and scope

The model is proprietary public evaluation material, not an open-source model.
Original and adapted vendor bytes remain externally fetched/local; this record
contains neither. The exact terms, source links and hashes are documented in
[MODEL_PROVENANCE.md](../../../reference/emi01/MODEL_PROVENANCE.md).

The reference retains dynamic Miller/output capacitance and reverse conduction,
but omits bipolar body-diode recovery charge, self-heating and avalanche
qualification. The omission can materially affect energy, ringing and EMI.
The load impedance, magnetic geometry/mass and rating screens are hypothetical
bounded assumptions. They establish neither a measured motor model nor thermal,
manufacturing, aviation-compliance or laboratory-correlation claims. The
research reserve does not prove that missing physics is bounded.

Read-only `git diff --name-status 17519fc bf1165f` and the following zero-difference
check confirmed that production C++/CUDA, Rust/Cargo, acceptance fixtures and
historical evidence were unchanged:

```sh
git diff --exit-code 17519fc bf1165f -- \
  cpp cuda src tests Cargo.toml Cargo.lock acceptance docs/evidence
```

The ngspice build changes expose existing outputs/provenance and reorder
attributes; its simulator configure options are unchanged. New Python/runtime
integration serves external reference tooling. The CPU capability slices and
GPU experiment gates are explicitly proposals: no EMI-02 production semantics,
GPU kernels, transient GPU executor or dispatch are implemented or authorized.

**Disposition:** no unresolved source/provenance/scope blocker found within this
review. Completion remains contingent on the two full retained invocations,
their numerical and accounting checks, the measured CPU decision, and final
delivery validation.
