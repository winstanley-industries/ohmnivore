# Prepared linear-AC persistent-session corpus v1

This corpus is the immutable GPU-02S compiler-derived synthetic session envelope. It does not
modify or reinterpret `../v1`, and it is not labeled as a customer or production workload.

Each case constructs a public linear `Circuit` IR containing one resistor and capacitor from every
non-ground node to ground, matching resistor/capacitor pairs on every declared topology edge, and
one or more ground-referenced AC voltage sources. The authoritative `CompileMna` and
`PrepareLinearAcBatch` paths create every corner batch. All corners retain identical component
insertion order, node/branch order, sparse structure, frequency count, and uniform-batch size.

For zero-based corner `k`, conductance/capacitance values are transformed deterministically:

```text
g_series(k) = g_series * (1 + 0.0125*k)
g_shunt(k)  = g_shunt  * (1 + 0.0025*(k mod 5))
c_series(k) = c_series * (1 + 0.02*((3*k) mod 7))
c_shunt(k)  = c_shunt  * (1 + 0.01*((5*k) mod 11))
```

Source branch `b` uses magnitude `(1 + 0.01*k)/(b+1)` and phase
`(7*k + 19*b) mod 360` degrees. These mutations require new complex-FP64 matrix values and RHS
uploads while preserving the canonical CSR structure and cuDSS analysis.

The four cases deliberately span a moderate 64-point control tree and three large-sweep candidates:
a 33-by-33 grid with 512 points, a 65-by-65 grid with 256 points, and a 1,024-node four-source ring
with 2,048 points. Pre-evidence linear scaling from the smaller diagnostic shapes estimates roughly
0.4--0.8 GiB of CUDA batch memory, below the unchanged 2 GiB bound; the measured peak remains an
acceptance condition. These are a frozen upper envelope, not a post-measurement eligibility
adjustment. Each candidate has four same-structure component/source corners. All counts and exact
ASCII bytes are fingerprint-bound by the loader. Input uses the same canonical unsigned integer
and lowercase-decimal rules as replay v1 and must be LF-terminated printable ASCII.

This corpus may establish a technical crossover only for its exact compiler-derived shapes,
corner counts, target hardware, and validation lane. It cannot authorize automatic dispatch.
