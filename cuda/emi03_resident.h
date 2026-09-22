#ifndef OHMNIVORE_CUDA_EMI03_RESIDENT_H_
#define OHMNIVORE_CUDA_EMI03_RESIDENT_H_

#include "ohmnivore/transient.h"

namespace ohmnivore {
struct Emi03ResidentResult {
  std::size_t emitted_points = 0;
  std::size_t attempts = 0;
  std::size_t rejected = 0;
  std::size_t nonlinear_rejections = 0;
  std::size_t history_estimates = 0;
  std::size_t history_checks = 0;
  std::size_t doubling_estimates = 0;
  std::size_t history_fallback_entries = 0;
  std::size_t history_fallback_recoveries = 0;
  SparseSolverStatistics solver_statistics;
};

// Private opt-in device-resident controller. Ordinary transient execution and
// the CPU oracle continue to use RunTransientAnalysis.
[[nodiscard]] Result<Emi03ResidentResult>
RunEmi03ResidentTransient(const MnaSystem &system, const TranAnalysis &analysis,
                          const TransientExecutionLimits &limits);
} // namespace ohmnivore
#endif
