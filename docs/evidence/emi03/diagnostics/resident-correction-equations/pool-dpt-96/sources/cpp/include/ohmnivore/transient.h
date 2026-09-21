#ifndef OHMNIVORE_TRANSIENT_H_
#define OHMNIVORE_TRANSIENT_H_

#include <cstddef>
#include <functional>
#include <vector>

#include "ohmnivore/compiler.h"
#include "ohmnivore/ir.h"
#include "ohmnivore/nonlinear.h"
#include "ohmnivore/solver.h"
#include "ohmnivore/sparse.h"
#include "ohmnivore/status.h"

namespace ohmnivore {

inline constexpr double kTransientAbsoluteTolerance = 1e-9;
inline constexpr double kTransientRelativeTolerance = 1e-3;
inline constexpr std::size_t kMaximumTransientAcceptedSteps = 1'000'000;
inline constexpr std::size_t kMaximumTransientStepAttempts = 2'000'000;

enum class BehavioralErrorEstimator {
  kStepDoubling,
  kDerivativeHistory,
};

// Defaults are production contracts. Explicit limits keep every typed bound
// directly testable without million-step unit tests.
struct TransientExecutionLimits {
  std::size_t maximum_accepted_steps = kMaximumTransientAcceptedSteps;
  std::size_t maximum_step_attempts = kMaximumTransientStepAttempts;
  double minimum_step_divisor = 10'000.0;
  // Production uses the Phase 3A bound. Focused tests may only reduce it to
  // exercise deterministic nonlinear timestep retry and exhaustion.
  std::size_t nonlinear_maximum_iterations = kDirectNewtonMaximumIterations;
  // Experimental bounded-output callers can stream accepted states. A failing
  // observer aborts the entire run; ordinary retained results are unchanged.
  std::function<Result<bool>(double, const std::vector<double> &)>
      accepted_state_observer = {};
  bool retain_output_states = true;
  // Opt-in EMI-02 model runner policy; native circuits retain their estimator.
  BehavioralErrorEstimator behavioral_error_estimator =
      BehavioralErrorEstimator::kStepDoubling;
};

enum class TransientIntegrationMethod {
  kBackwardEuler,
  kTrapezoidal,
};

enum class TransientStepRejectionReason {
  kNone,
  kLocalError,
  kNonlinearConvergence,
};

struct TransientStepRecord {
  double start_time_seconds;
  double end_time_seconds;
  double step_size_seconds;
  TransientIntegrationMethod method;
  bool accepted;
  double normalized_local_error;
  bool landed_on_hard_point;
  // These occupy existing alignment space before the rejection enum.
  // Audited means a valid history estimate was checked against two half steps.
  bool derivative_history_audited = false;
  bool derivative_history_audit_agreed = false;
  // Policy state after this trial, including rejected-audit entry, accepted
  // 16-agreement recovery, or accepted BE reset. No physical history is
  // exposed.
  bool derivative_history_fallback_active = false;
  TransientStepRejectionReason rejection_reason =
      TransientStepRejectionReason::kNone;
};

// Full FP64 states are ordered exactly like MnaSystem: insertion-ordered node
// voltages followed by insertion-ordered voltage-source/inductor currents.
// Samples before .TRAN tstart are integrated but are not returned.
struct TransientResult {
  std::vector<double> times_seconds;
  std::vector<std::vector<double>> states;
  std::vector<TransientStepRecord> step_trace;
  SparseSolverStatistics solver_statistics;
  std::size_t emitted_points = 0;
  // Completed LTE evaluations include rejected trials, but not failed solves.
  std::size_t derivative_history_error_estimates = 0;
  // Opt-in TRAP audits and fallbacks that actually computed two half steps.
  std::size_t derivative_history_step_doubling_checks = 0;
  // All BE/behavioral-TRAP full/two-half comparisons, including default policy.
  std::size_t step_doubling_error_estimates = 0;
  std::size_t derivative_history_fallback_entries = 0;
  // Recovery requires accepted audit agreement; BE resets are not counted.
  std::size_t derivative_history_fallback_recoveries = 0;
};

// Builds b(t) by replacing each transient source's own DC contribution with
// its waveform value while retaining every non-transient contribution.
[[nodiscard]] Result<std::vector<double>>
BuildTransientRhs(const MnaSystem &system, double time_seconds);

// Forms the canonical independent-pattern CSR union
// A = G + alpha * C / step_size_seconds. alpha=1 is BE and alpha=2 is TRAP.
[[nodiscard]] Result<CsrMatrix>
FormTransientCompanionMatrix(const CsrMatrix &g, const CsrMatrix &c,
                             double step_size_seconds, double alpha);

// BE: b_n + C*x_(n-1)/h.
[[nodiscard]] Result<std::vector<double>> BuildBackwardEulerRhs(
    const CsrMatrix &c, const std::vector<double> &previous_state,
    const std::vector<double> &current_rhs, double step_size_seconds);

// TRAP: b_n + b_(n-1) + (2*C/h - G)*x_(n-1).
[[nodiscard]] Result<std::vector<double>>
BuildTrapezoidalRhs(const CsrMatrix &g, const CsrMatrix &c,
                    const std::vector<double> &previous_state,
                    const std::vector<double> &previous_rhs,
                    const std::vector<double> &current_rhs,
                    double step_size_seconds);

// Without UIC this is the existing DC operating-point solve. With UIC it
// enforces zero capacitor voltage and zero inductor current while preserving
// algebraic source constraints and accepting redundant capacitor constraints.
[[nodiscard]] Result<std::vector<double>>
BuildTransientInitialState(const MnaSystem &system,
                           bool use_initial_conditions);

[[nodiscard]] Result<TransientResult> RunTransientAnalysis(
    const MnaSystem &system, const TranAnalysis &analysis,
    const TransientExecutionLimits &limits = TransientExecutionLimits{});

} // namespace ohmnivore

#endif // OHMNIVORE_TRANSIENT_H_
