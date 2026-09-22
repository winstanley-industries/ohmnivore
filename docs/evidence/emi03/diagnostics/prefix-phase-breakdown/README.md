# Selected resident prefix phase breakdown

This archive retains the fresh phase replay underlying
[`prefix-breakdown106.md`](prefix-breakdown106.md). It reuses selected checkpoint
100's exact CPU/GPU binaries and all nine original flattened input hashes.
The original request-only batch took 46.313 s GPU versus 7.017 s CPU; this fresh
instrumented GPU batch takes 47.659 s. These are shortened 20 us diagnostics,
not complete-study or acceptance measurements.

`run/` contains all nine CPU trajectories, eighteen GPU trajectories (warmup and
measurement), input decks, comparisons, resource records and worker phase logs.
Raw files are compressed without changing their contents. Analysis and execution
scripts retain their original workspace paths; their Python sources are compressed
to keep historical evidence outside the live lint/build inputs. `manifest.json`
records hashes of the retained artifacts.

The measured critical job's exclusive device-cycle shares are 28.14% triangular
solves, 20.05% factorization/reuse checks, 20.10% expressions/derivatives, 11.25%
linear residual checks and 20.45% remaining work. These are not directly measured
phase seconds from the earlier 46.313 s run. Device completion waits include GPU
execution and must not be classified wholesale as avoidable host synchronization.
