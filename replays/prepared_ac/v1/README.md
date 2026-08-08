# Prepared linear-AC replay corpus v1

`corpus.csv` is the immutable GPU-01 replay authority. The loader pins its exact byte fingerprint,
and replay tests pin every declared field plus generated structure, batch, aggregate-member, first-
member, and last-member fingerprints. Any semantic change requires a new versioned directory; v1
must not be reinterpreted in place.

The four deterministic MNA builders exercise a path ladder, binary tree, 32-by-32 grid, and ring
with four ideal voltage-source branches. Every non-ground node receives the declared shunt.
Topology edges receive symmetric negative off-diagonal series stamps and matching positive
diagonal stamps. Conductance source incidence is `+1` at both `(node, branch)` and `(branch, node)`;
source `b` connects to node `b*node_count/branch_count`. Dynamic matrices have empty branch rows.
The AC RHS for branch `b` is
`(1/(b+1)) + j*((b even ? 0.25 : -0.25)/(b+1))`. CSR rows follow node/branch order and columns are
strictly increasing.

LIN uses `sweep_points` as the inclusive batch size. DEC uses `sweep_points` per decade and has
`decades*sweep_points+1` members, including exact endpoints through the existing
`GenerateAcFrequencies` authority. The parser binds declared topology, node, branch, dimension,
union-nonzero, sweep, batch, frequency, scale, and reuse fields before preparation.

Identifiers use ASCII letters, digits, underscore, and hyphen. Unsigned integers have no leading
zeros. Decimal fields use lowercase `e`, no leading plus, no redundant leading integer zero, no
trailing fractional zero, and no leading-zero exponent. Input is LF-terminated printable ASCII;
the exact 18-column header, four known classes/topologies, unique IDs, bounded counts, and all
numeric relationships are mandatory.

The manifest identity is two FNV-1a 64-bit lanes over every file byte, including the final LF.
Both use prime `1099511628211`; lane offsets are `14695981039346656037` and
`9521211207457086692`. The rendered identity is `v1-` followed by the two 16-digit lowercase
hexadecimal lane values.

Prepared structure/member/batch and aggregate replay-member identities use the same lanes.
Integers and FP64 bit patterns are fed least-significant byte first; strings are prefixed by their
unsigned 64-bit byte length; complex values feed real then imaginary FP64 bits. Domain strings are
`prepared-ac-structure-v1`, `prepared-ac-member-v1`, `prepared-ac-batch-v1`, and
`prepared-ac-replay-members-v1`. These are stable content identities, not cryptographic
authentication.
