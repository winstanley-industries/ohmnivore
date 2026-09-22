# Breakdown of the nine-job GPU diagnostic

The original batch took **46.313031 s**, bounded by `light-hot_high_c` at
**46.298639 s**. Jobs ran concurrently; their request times must not be added.

| Job | GPU request seconds | CPU request seconds |
| --- | ---: | ---: |
| light-nominal | 40.626 | 6.414 |
| light-fast_low_lc | 39.875 | 6.463 |
| light-hot_high_c | 46.299 | 6.999 |
| boundary-nominal | 21.867 | 3.489 |
| boundary-fast_low_lc | 20.850 | 3.410 |
| boundary-hot_high_c | 21.016 | 3.283 |
| reference-nominal | 28.066 | 3.971 |
| reference-fast_low_lc | 25.161 | 4.704 |
| reference-hot_high_c | 23.517 | 3.425 |

The critical job records **45.906111 s** in GPU completion waits, **0.386333 s**
in other worker activity, and **0.006194 s** in the request wrapper/protocol remainder.
The batch exceeds the critical request duration by **0.014392 s**.
These are nested timer differences. Completion waits include resident GPU execution,
copy completion, queueing and event polling; they are not synchronization overhead alone.
Initialization, parsing and raw-file output are within request time; pool construction,
deck generation, waveform comparison and spectral processing are outside this diagnostic timer.

## Fresh phase replay

The exact original CPU/GPU binaries and all nine exact flattened input hashes match.
One GPU warmup and one measured batch were run with the existing device phase counters
printed at job completion. The new measured batch took **47.658892 s**.
All nine fresh CPU trajectories, eighteen GPU waveform comparisons and resource checks pass.
The slowest job remains `light-hot_high_c` and finishes **5.904 s** after the next job.
Its final contiguous six-line profile group is therefore isolated; its chord-iteration
count matches that job's native numerical-reuse telemetry.

| Resident GPU phase | Device-cycle share |
| --- | ---: |
| Triangular solves, including refinement | 28.14% |
| Factorization and factor reuse checks | 20.05% |
| Behavioral expressions and derivatives | 20.10% |
| Linear residual and backward-error checks | 11.25% |
| Remaining assembly, Newton and timestep work | 20.45% |

These are new device-cycle shares, not directly measured phase seconds from the original
46.3-second run. The remaining category is subtraction and includes nonlinear assembly,
Newton control/norms, companion/history/error-estimation work, output/state staging and
unclassified barriers. Value evaluation and AD are included together in expressions;
refinement solves are included in triangular solves, avoiding double counting.

For the original critical job, GPU made 102,789 timestep attempts versus
101,863 on CPU. It made 521,511
primary linear solves and 354,380
refinement solves. Comparable CPU counts are 515,551
and 349,348.
The dominant discrepancy is execution cost per attempt, rather than a large increase
in attempted steps.

The separately retained Nsight Compute report for selected variant 100 profiles one
reference/nominal resident chunk, not this entire batch. It reports one 256-thread
block on an 84-SM GPU, zero spills, 95.08% no-eligible issue opportunities and substantial
barrier waiting. These are supporting mechanism diagnostics, not wall-time percentages
for this nine-job run. Current execution has at most nine circuit blocks in this batch;
its triangular solve executes in one warp per circuit block.

Sources: original `/home/adam/code/ohmnivore/docs/evidence/emi03/diagnostics/resident-shared-addresses/prefix-corpus100`, replay `/home/adam/emi03-work/prefix-phases106`, and
`/home/adam/code/ohmnivore/docs/evidence/emi03/diagnostics/resident-shared-addresses/resident-full-counters100/metrics.txt`.
Exact derived values and caveats are in `/home/adam/emi03-work/prefix-breakdown106.json`.
