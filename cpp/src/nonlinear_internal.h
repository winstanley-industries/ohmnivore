#ifndef OHMNIVORE_CPP_SRC_NONLINEAR_INTERNAL_H_
#define OHMNIVORE_CPP_SRC_NONLINEAR_INTERNAL_H_

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "ohmnivore/compiler.h"
#include "ohmnivore/nonlinear.h"
#include "ohmnivore/solver.h"
#include "ohmnivore/status.h"
#include "ohmnivore/transient.h"

namespace ohmnivore::internal {

// Private invocation-owned immutable G/C plan. Creation performs the ordinary
// first companion construction before preparing any shortcuts. The returned
// matrix reference remains valid only until the next Form call.
class PreparedTransientCompanion {
public:
  class StateProducts {
  public:
    StateProducts(StateProducts &&) noexcept = default;
    StateProducts &operator=(StateProducts &&) noexcept = default;
    [[nodiscard]] const std::vector<double> &g_product() const { return g; }
    [[nodiscard]] const std::vector<double> &c_product() const { return c; }

  private:
    friend class PreparedTransientCompanion;
    StateProducts() = default;
    StateProducts(const PreparedTransientCompanion *source,
                  std::vector<double> g_product, std::vector<double> c_product)
        : owner(source), g(std::move(g_product)), c(std::move(c_product)) {}
    const PreparedTransientCompanion *owner = nullptr;
    std::vector<double> g;
    std::vector<double> c;
  };

  [[nodiscard]] static Result<std::unique_ptr<PreparedTransientCompanion>>
  Create(const CsrMatrix &g, const CsrMatrix &c, double step, double alpha);
  [[nodiscard]] const CsrMatrix &matrix() const { return matrix_; }
  [[nodiscard]] Result<const CsrMatrix *> Form(double step, double alpha);
  [[nodiscard]] Result<std::vector<double>>
  BackwardEulerRhs(const std::vector<double> &state,
                   const std::vector<double> &rhs, double step) const;
  // The optional products belong to one integration attempt and exactly this
  // immutable accepted state. Never pass them to a different state/second half.
  [[nodiscard]] Result<std::vector<double>> TrapezoidalRhs(
      const std::vector<double> &state, const std::vector<double> &previous_rhs,
      const std::vector<double> &current_rhs, double step,
      std::optional<StateProducts> *same_state_products = nullptr) const;

private:
  struct UnionEntry {
    std::size_t g_index;
    std::size_t c_index;
  };
  PreparedTransientCompanion(CsrMatrix g, CsrMatrix c, CsrMatrix matrix,
                             std::vector<UnionEntry> entries)
      : g_(std::move(g)), c_(std::move(c)), matrix_(std::move(matrix)),
        entries_(std::move(entries)) {}
  CsrMatrix g_;
  CsrMatrix c_;
  CsrMatrix matrix_;
  std::vector<UnionEntry> entries_;
};

// Private diagnostics count per-expression fresh evaluations and per-state
// cache hits. They do not affect solver acceptance or numeric statistics.
struct PreparedExpressionCacheStatistics {
  std::size_t initial_hits = 0;
  std::size_t initial_misses = 0;
  std::size_t history_hits = 0;
  std::size_t history_misses = 0;
  std::size_t fresh_initial_evaluations = 0;
  std::size_t fresh_trial_evaluations = 0;
  std::size_t fresh_final_evaluations = 0;
  std::size_t fresh_final_value_evaluations = 0;
  std::size_t reused_final_derivatives = 0;
  std::size_t fresh_history_evaluations = 0;
  std::size_t publications = 0;
  std::size_t retained_entries = 0;
};

class PreparedExpressionCache;
class PreparedNewtonWorkspace;

struct PreparedAssemblyWorkspaceStatistics {
  std::size_t acquisitions = 0;
  std::size_t reused_buffers = 0;
  std::size_t pattern_initializations = 0;
  std::size_t active_buffers = 0;
  std::size_t maximum_active_buffers = 0;
  std::size_t retained_buffers = 0;
  std::size_t admitted_linear_solves = 0;
};

// An invocation-owned behavioral companion workspace. The factory validates and
// copies its metadata; callers can update only checked numeric values and RHS.
class PreparedNonlinearPointSolver {
public:
  ~PreparedNonlinearPointSolver();
  [[nodiscard]] static Result<std::unique_ptr<PreparedNonlinearPointSolver>>
  Create(const MnaSystem &system);
  [[nodiscard]] Result<NonlinearPointResult>
  Solve(const CsrMatrix &matrix, const std::vector<double> &rhs,
        const std::vector<double> &initial_guess,
        std::size_t maximum_iterations);
  [[nodiscard]] Result<std::vector<double>>
  History(const std::vector<double> &state) const;
  [[nodiscard]] SparseSolverStatistics statistics() const;
  [[nodiscard]] PreparedExpressionCacheStatistics
  expression_cache_statistics() const;
  [[nodiscard]] PreparedAssemblyWorkspaceStatistics
  assembly_workspace_statistics() const;

private:
  PreparedNonlinearPointSolver(
      MnaSystem system, std::unique_ptr<SparseRealFactorization> factorization);
  MnaSystem system_;
  std::unique_ptr<SparseRealFactorization> factorization_;
  std::unique_ptr<PreparedExpressionCache> expression_cache_;
  std::unique_ptr<PreparedNewtonWorkspace> assembly_workspace_;
};

// Fully checked point-by-point execution for parity tests. It shares the
// scheduler and numerical policy, but does not use the prepared workspace.
[[nodiscard]] Result<TransientResult>
RunTransientAnalysisUnpreparedForTest(const MnaSystem &system,
                                      const TranAnalysis &analysis,
                                      const TransientExecutionLimits &limits);

// A false node row must already have been replaced by an independently
// selected reactive-state constraint. Canonical public nonlinear entry points
// always stamp both non-ground diode terminal equations.
[[nodiscard]] Result<NonlinearPointResult> RunNonlinearPointForProjection(
    const MnaSystem &system, const std::vector<double> &initial_guess,
    const std::vector<bool> &active_node_equations,
    SparseRealFactorization *factorization, std::size_t maximum_iterations);

} // namespace ohmnivore::internal

#endif // OHMNIVORE_CPP_SRC_NONLINEAR_INTERNAL_H_
