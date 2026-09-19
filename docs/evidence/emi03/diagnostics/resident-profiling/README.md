# Resident profiling and concurrent-owner diagnostics

EMI-03 remains incomplete. These shortened 1.03 us reference/nominal diagnostics
identify bottlenecks; they do not establish the full thirty-job qualification,
resource admission or the 9/36-job performance gates.

Nsight Systems 2026.3.2.476, from the checksum-pinned CUDA 13.4.2 redistributable,
successfully collects software CUDA tracing on the recorded WSL host with GPU
metrics and CPU sampling disabled. This updates the external profiler only;
the solver still uses the pinned CUDA 13.0.2 build toolchain. The older profiler
bundled with CUDA 13.0.2 collected host APIs but no GPU kernel timestamps and
reported an unsupported driver version. Both reports remain visible.

The single-owner trace attributes 99.6% of kernel time to `Emi03Advance`: 1.860 s
across 19 launches, occupying 97.6% of the 1.905 s first-to-last resident interval.
Device copies total 0.610 ms. It does not identify the instruction/cache/stall
cause inside the kernel. Nsight Compute's hardware-counter collection is blocked
by `ERR_NVGPUCTRPERM`; its failed attempt is retained. No hardware-counter results
are claimed.

The trace also exposes interference between owners. Four ordinary processes take
8.449 s for four shortened jobs. Moving complete private jobs into threads in one
process takes 4.873 s. Software tracing then shows large waits in legacy device
allocation/free and in pinned-memory frees during factor-plan rebuilds.

| Diagnostic prototype | One owner | Four owners | Sixteen owners |
|---|---:|---:|---:|
| v74: one process, original allocation | 2.319 s | 4.873 s | 26.815 s |
| v75: private allocation stream | 2.318 s | 3.671 s | 14.742 s |
| v76: also reuse private job staging | 2.319 s | 2.318 s | 5.475 s |
| v76 independent repeat | — | 2.319 s | 5.576 s |

These are ordinary diagnostic wall times, not medians/P95 or acceptance ratios.
The traced v76 API totals fall to 0.125 s in copies across the four host threads;
individual resident kernels retain their single-job duration. Aggregate API
and kernel durations across threads may overlap and must not be summed into
whole-study time. Cross-process kernel intervals may also include preemption.

The v75 prototype completes allocation/free on a private nonblocking stream and
waits before publishing memory or decrementing its ledger. Library frees first
complete their supplied use stream. Its 20 existing CUDA and 10 resident test
cases pass. The v76 resident path retains one private pinned staging buffer per
job across all factor-plan owners; each transfer completes before the next uses
that staging buffer. Its ten resident tests pass, including 16 concurrent private
jobs and isolated allocation failure. These are development prototypes, not an
admitted ensemble scheduling mode. Pool residency and the full resource envelope
still require qualification.

All 88 retained ordinary/profiling-probe output identities pass the diagnostic
CPU waveform audit and have zero fallback, allocation or cleanup failures. That
audit covers every saved observable on a 0.5 ns grid using the original waveform
tolerance formulas. It does not cover the missing full-window spectra or physical
classification. Exact raw hashes are retained; local raw probe files remain under
`/home/adam/emi03-work/`. No shortened output substitutes for a required full run.

`owner-prototype-*/source.patch.gz` applies to `c6eb2ef`; the separately retained
`emi03_concurrent_probe.cc.gz` supplies its new translation unit. Binary identities
are recorded in each summary. The frozen resident numerical candidate is the
[Jacobian-reuse checkpoint](../resident-jacobian-reuse/README.md). The prototypes
add ownership/allocation experiments to that candidate and retain their own
sources. The private adapted vendor deck is not published.

The `.nsys-rep` files can be opened in a compatible Nsight Systems GUI. The CSV
summaries, raw resident intervals and diagnostic messages are retained beside
each report. `profiler-MODULE.bazel.gz`, its lockfile, pinned Bazel version and
NVIDIA manifests reconstruct a separate Bazel profiler workspace. Build the
`@nsight_systems_current//:files` or `@nsight_compute_current//:files` target there;
no system CUDA compiler or profiling package is used.

The software trace command is:

```sh
nsys profile --trace=cuda-sw,osrt --sample=none --cpuctxsw=none \
  --gpu-metrics-devices=none --resolve-symbols=false --output=report \
  worker input.cir output.raw stats.json
```

Once host counter access is enabled, use the pinned Nsight Compute with a
separately identified profiling binary, for example:

```sh
ncu --set full --kernel-name Emi03Advance --launch-skip 4 --launch-count 1 \
  --clock-control none --cache-control none --export resident \
  worker input.cir output.raw stats.json
```

Profiler replay and instrumented timings are excluded from acceptance timing.
Source-correlated profiling can use a separately recorded Bazel build with
`--@rules_cuda//cuda:copts=-lineinfo`.
