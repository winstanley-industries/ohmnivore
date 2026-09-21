# Rejected factor-dependency and triangular-row trials

These three counter-guided trials are rejected. They do not qualify EMI-03 or
change any frozen acceptance requirement. Each has ten passing resident tests,
a source/binary identity, exact source patch, build and execution logs, and fresh
CPU waveform comparisons on the 1.03 us reference/nominal prefix.

| Trial | Change | Ordinary median request wall time |
|---|---|---|
| 91 | Symbolic flags preserve factor entries independent of state/companion changes | 2.088 s vs 2.096 s baseline; effectively neutral |
| 92 | Also skip wholly unchanged factorization levels | 2.104 s vs 2.088 s; 0.8% slower |
| 93 | Warp-cooperative reductions for triangular rows with at least eight terms | 2.205 s vs 2.082 s; 5.9% slower |

Dependency flags conservatively include row equilibration, companion coefficients,
behavioral rows and transitive factor/pivot dependencies. They reduce arithmetic,
but every encountered factorization level still includes changing entries. The
extra setup and branching erase the benefit. Trial 93 changes long-row FP64
accumulation order and adds block synchronization per level. Its CPU waveform
checks pass, but its elapsed time increases. No candidate is selected.

Measurements use one warmup and three alternating ordinary observations, plus
separate phase-print runs. These are diagnostic medians, not acceptance results.
The baseline is rerun beside each candidate on driver 616.92. Different internal
point counts and raw floating-point trajectories are retained without assuming
bitwise equality. The generic freezer's scope labels are preserved as historical
metadata; the table above defines the actual changes.

The audit reconstructs all three candidate sources from the owner-pool archive,
all 33 raw trajectories and all 30 GPU/CPU waveform comparisons. `files.json`
inventories retained files. Compressed scripts retain execution and audit details.
There is no fresh full DPT, coupled-window or Compute Sanitizer qualification for
these rejected candidates. The full thirty-job and throughput gates remain unmet.
