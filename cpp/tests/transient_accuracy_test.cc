#include "cpp/tests/google_test.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>

#include "cpp/src/nonlinear_internal.h"
#include "cpp/tests/transient_accuracy_cases.h"
#include "ohmnivore/behavioral.h"

namespace ohmnivore {
namespace {

TransientExecutionLimits AccuracyLimits(BehavioralErrorEstimator policy) {
  TransientExecutionLimits limits;
  limits.behavioral_error_estimator = policy;
  limits.minimum_step_divisor = 1e6;
  limits.nonlinear_maximum_iterations = 100;
  return limits;
}

void EqualBits(const std::vector<double> &a, const std::vector<double> &b) {
  ASSERT_EQ(a.size(), b.size());
  for (std::size_t i = 0; i < a.size(); ++i)
    ASSERT_EQ(std::bit_cast<std::uint64_t>(a[i]),
              std::bit_cast<std::uint64_t>(b[i]));
}

TEST(TransientAccuracy, FineResolutionMatchesIndependentPhysicalOracles) {
  const auto cases = test::TransientAccuracyCases();
  ASSERT_EQ(cases.size(), 21U);
  for (const auto &fixture : cases) {
    SCOPED_TRACE(fixture.name);
    const auto parsed = ParseBehavioralNetlist(fixture.deck);
    ASSERT_TRUE(parsed.ok()) << parsed.error().message;
    const auto compiled = CompileBehavioralMna(parsed.value());
    ASSERT_TRUE(compiled.ok()) << compiled.error().message;
    const auto &system = compiled.value();
    const auto node =
        std::find(system.node_names.begin(), system.node_names.end(), "out");
    ASSERT_NE(node, system.node_names.end());
    const auto out = static_cast<std::size_t>(node - system.node_names.begin());
    auto analysis = fixture.analysis;
    analysis.time_step_seconds /= 32;
    for (const auto policy : {BehavioralErrorEstimator::kStepDoubling,
                              BehavioralErrorEstimator::kDerivativeHistory}) {
      SCOPED_TRACE(static_cast<int>(policy));
      const auto solved =
          RunTransientAnalysis(system, analysis, AccuracyLimits(policy));
      ASSERT_TRUE(solved.ok()) << solved.error().message;
      const auto &result = solved.value();
      double voltage_error = 0, current_error = 0;
      for (std::size_t i = 0; i < result.states.size(); ++i) {
        const auto exact = fixture.exact(result.times_seconds[i]);
        voltage_error =
            std::max(voltage_error,
                     std::abs(result.states[i][out] - fixture.bias - exact[0]));
        if (fixture.has_inductor) {
          const auto branch =
              system.inductor_initial_constraints.at(0).branch_index;
          current_error = std::max(
              current_error, std::abs(result.states[i][branch] - exact[1]));
        }
      }
      EXPECT_LE(voltage_error, fixture.voltage_limit);
      EXPECT_LE(current_error, fixture.current_limit);
    }
  }
}

TEST(TransientAccuracy,
     PreparedCoarseTrajectoriesRetainEveryDecisionAndStateBit) {
  for (const auto &fixture : test::TransientAccuracyCases()) {
    SCOPED_TRACE(fixture.name);
    const auto parsed = ParseBehavioralNetlist(fixture.deck);
    ASSERT_TRUE(parsed.ok());
    const auto compiled = CompileBehavioralMna(parsed.value());
    ASSERT_TRUE(compiled.ok());
    for (const auto policy : {BehavioralErrorEstimator::kStepDoubling,
                              BehavioralErrorEstimator::kDerivativeHistory}) {
      const auto limits = AccuracyLimits(policy);
      const auto prepared =
          RunTransientAnalysis(compiled.value(), fixture.analysis, limits);
      const auto checked = internal::RunTransientAnalysisUnpreparedForTest(
          compiled.value(), fixture.analysis, limits);
      ASSERT_TRUE(prepared.ok()) << prepared.error().message;
      ASSERT_TRUE(checked.ok()) << checked.error().message;
      const auto &a = prepared.value(), &b = checked.value();
      EqualBits(a.times_seconds, b.times_seconds);
      ASSERT_EQ(a.states.size(), b.states.size());
      for (std::size_t i = 0; i < a.states.size(); ++i)
        EqualBits(a.states[i], b.states[i]);
      ASSERT_EQ(a.step_trace.size(), b.step_trace.size());
      for (std::size_t i = 0; i < a.step_trace.size(); ++i) {
        const auto &x = a.step_trace[i], &y = b.step_trace[i];
        EqualBits({x.start_time_seconds, x.end_time_seconds,
                   x.step_size_seconds, x.normalized_local_error},
                  {y.start_time_seconds, y.end_time_seconds,
                   y.step_size_seconds, y.normalized_local_error});
        EXPECT_EQ(x.method, y.method);
        EXPECT_EQ(x.accepted, y.accepted);
        EXPECT_EQ(x.landed_on_hard_point, y.landed_on_hard_point);
        EXPECT_EQ(x.rejection_reason, y.rejection_reason);
        EXPECT_EQ(x.derivative_history_audited, y.derivative_history_audited);
        EXPECT_EQ(x.derivative_history_audit_agreed,
                  y.derivative_history_audit_agreed);
        EXPECT_EQ(x.derivative_history_fallback_active,
                  y.derivative_history_fallback_active);
      }
      EXPECT_EQ(a.solver_statistics.symbolic_analyses,
                b.solver_statistics.symbolic_analyses);
      EXPECT_EQ(a.solver_statistics.numeric_factorizations,
                b.solver_statistics.numeric_factorizations);
      EXPECT_EQ(a.solver_statistics.numeric_refactorizations,
                b.solver_statistics.numeric_refactorizations);
      EXPECT_EQ(a.solver_statistics.numeric_refactorization_fallbacks,
                b.solver_statistics.numeric_refactorization_fallbacks);
      EXPECT_EQ(a.solver_statistics.numeric_reuses,
                b.solver_statistics.numeric_reuses);
      EXPECT_EQ(a.solver_statistics.solves, b.solver_statistics.solves);
      EXPECT_EQ(a.solver_statistics.iterative_refinement_solves,
                b.solver_statistics.iterative_refinement_solves);
    }
  }
}

} // namespace
} // namespace ohmnivore
