# Resident Jacobian-reuse diagnostic

EMI-03 remains incomplete. This candidate preserves the frozen acceptance gates.
A complete q0 reference/nominal trajectory passes the original CPU waveform,
all-bin spectral and physical-classification comparison, but takes 286.349 s on
GPU versus 32.947 s on CPU. The GPU result is `resource_limit`: the separate
600 s diagnostic ceiling does not change the frozen 120 s job limit.

Newton may reuse a validated Jacobian at an identical companion coefficient.
Reuse within a solve requires a falling residual and periodically refreshes the
analytic derivatives. Failed lagged linear solves invalidate the factor cache
and refresh at the same state within the bounded Newton iteration. Compensated
FP64 products form the lagged RHS; linear certification and mandatory residual
correction remain. Acceptance always checks the actual analytic Jacobian, a fresh
nonlinear value/residual evaluation, the physical update, and nonsingularity.
The timestep controller, rollback, source boundaries and error thresholds remain.

The original full run used the unformatted v68 sources. A separately rebuilt
formatted v73 source snapshot produces the identical worker SHA256. Both original
and formatted patches and identities are retained; historical timing is not a
new run. The full diagnostic emits 482,889 GPU points and 461,672 CPU points.

`full-reference/` retains the complete raw trajectories, spectra, metrics,
progress records and resource failure. `dpt-refinements/` contains fresh runs at
all three frozen DPT levels: all three CPU/GPU comparisons and all eight
integration/output-refinement checks pass. GPU q0/q1/q2 times are
8.583 / 11.639 / 14.544 s, versus 0.617 / 0.817 / 1.418 s CPU.
`validation/` records all eighteen passing canonical checks and the
protected-path audit. The source/raw audit reproduces the numerical comparisons;
that audit does not turn failed resource or absent throughput gates into passes.

Still required: fresh complete thirty-job CPU/ngspice/GPU qualification, aggregate
resource accounting, zero failed/missing jobs, and both independent invocations
of the 9/36-job median/P95/cold-time gates. No automatic dispatch is enabled.
