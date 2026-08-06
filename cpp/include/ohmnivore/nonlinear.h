#ifndef OHMNIVORE_NONLINEAR_H_
#define OHMNIVORE_NONLINEAR_H_

#include <cstddef>
#include <vector>

#include "ohmnivore/compiler.h"
#include "ohmnivore/solver.h"
#include "ohmnivore/status.h"

namespace ohmnivore {

inline constexpr double kDiodeMinimumExponent = -80.0;
inline constexpr double kDiodeMaximumExponent = 80.0;
inline constexpr double kNonlinearMaximumMagnitude = 1e100;
inline constexpr double kNewtonVoltageAbsoluteTolerance = 1e-9;
inline constexpr double kNewtonCurrentAbsoluteTolerance = 1e-12;
inline constexpr double kNewtonRelativeTolerance = 1e-6;
inline constexpr std::size_t kDirectNewtonMaximumIterations = 50;
inline constexpr std::size_t kContinuationNewtonMaximumIterations = 30;

struct DiodeEvaluation {
  double current_amperes;
  double conductance_siemens;
  double exponent;
};

[[nodiscard]] Result<DiodeEvaluation>
EvaluateDiode(double junction_voltage_volts, double saturation_current_amperes,
              double emission_voltage_volts);

[[nodiscard]] Result<double> LimitDiodeJunctionVoltage(
    double proposed_voltage_volts, double previous_voltage_volts,
    double saturation_current_amperes, double emission_voltage_volts);

enum class NonlinearStrategy {
  kDirect,
  kSourceStepping,
  kGminStepping,
};

struct NonlinearIterationRecord {
  NonlinearStrategy strategy;
  double continuation_value;
  std::size_t iteration;
  double maximum_normalized_update;
  double maximum_normalized_residual;
  bool accepted;
};

struct NonlinearAttemptRecord {
  NonlinearStrategy strategy;
  double continuation_value;
  std::size_t iterations;
  bool converged;
};

struct NonlinearDcOptions {
  // Production uses these defaults. Focused tests may only reduce the bounds
  // to exercise bounded failure and continuation paths.
  std::size_t direct_maximum_iterations = kDirectNewtonMaximumIterations;
  std::size_t source_step_maximum_iterations =
      kContinuationNewtonMaximumIterations;
  std::size_t gmin_step_maximum_iterations =
      kContinuationNewtonMaximumIterations;
  std::size_t final_gmin_maximum_iterations = kDirectNewtonMaximumIterations;
};

struct NonlinearDcResult {
  std::vector<double> solution;
  std::vector<NonlinearIterationRecord> iteration_trace;
  std::vector<NonlinearAttemptRecord> attempt_trace;
  SparseSolverStatistics solver_statistics;
};

struct NonlinearLinearization {
  CsrMatrix jacobian;
  std::vector<double> residual;
};

[[nodiscard]] Result<NonlinearLinearization> BuildNonlinearDcLinearization(
    const MnaSystem &system, const std::vector<double> &solution,
    double source_scale = 1.0, double extra_gmin_siemens = 0.0);

// Recomputes the nonlinear residual in stable row/device order. The source
// scale and extra GMIN arguments expose continuation systems to focused tests;
// final production acceptance always calls this with 1 and 0 respectively.
[[nodiscard]] Result<double> ValidateNonlinearResidual(
    const MnaSystem &system, const std::vector<double> &solution,
    double source_scale = 1.0, double extra_gmin_siemens = 0.0);

[[nodiscard]] Result<NonlinearDcResult>
RunNonlinearDc(const MnaSystem &system, const NonlinearDcOptions &options = {});

} // namespace ohmnivore

#endif // OHMNIVORE_NONLINEAR_H_
