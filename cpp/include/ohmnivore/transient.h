#ifndef OHMNIVORE_TRANSIENT_H_
#define OHMNIVORE_TRANSIENT_H_

#include <cstddef>
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

// Defaults are production contracts. Explicit limits keep every typed bound
// directly testable without million-step unit tests.
struct TransientExecutionLimits {
  std::size_t maximum_accepted_steps = kMaximumTransientAcceptedSteps;
  std::size_t maximum_step_attempts = kMaximumTransientStepAttempts;
  double minimum_step_divisor = 10'000.0;
  // Production uses the Phase 3A bound. Focused tests may only reduce it to
  // exercise deterministic nonlinear timestep retry and exhaustion.
  std::size_t nonlinear_maximum_iterations = kDirectNewtonMaximumIterations;
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
