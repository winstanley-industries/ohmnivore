#include "cpp/tests/google_test.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "cpp/src/nonlinear_internal.h"
#include "ohmnivore/behavioral.h"
#include "ohmnivore/transient.h"

namespace ohmnivore {
namespace {

MnaSystem PreparedCompile(const std::string &deck) {
  auto parsed = ParseBehavioralNetlist(deck);
  if (!parsed.ok()) {
    ADD_FAILURE() << parsed.error().message;
    return {};
  }
  auto compiled = CompileBehavioralMna(parsed.value());
  if (!compiled.ok()) {
    ADD_FAILURE() << compiled.error().message;
    return {};
  }
  return compiled.TakeValue();
}

MnaSystem NonlinearCharge() {
  return PreparedCompile("Icharge 0 out DC 0 PWL(0 0 100u 10m 1m 10m)\n"
                         "Cbase out sense 1u\nVsense sense 0 0\n"
                         "Bextra out 0 I={i(Vsense)*(1+10*v(out)*v(out))}\n");
}

Result<TransientResult>
RunPreparedOrChecked(bool prepared, const MnaSystem &system,
                     const TranAnalysis &analysis,
                     const TransientExecutionLimits &limits = {}) {
  return prepared ? RunTransientAnalysis(system, analysis, limits)
                  : internal::RunTransientAnalysisUnpreparedForTest(
                        system, analysis, limits);
}

void EqualBits(double first, double second) {
  EXPECT_EQ(std::bit_cast<std::uint64_t>(first),
            std::bit_cast<std::uint64_t>(second));
}

void EqualState(const std::vector<double> &first,
                const std::vector<double> &second) {
  ASSERT_EQ(first.size(), second.size());
  for (std::size_t i = 0; i < first.size(); ++i)
    EqualBits(first[i], second[i]);
}

void EqualMatrix(const CsrMatrix &first, const CsrMatrix &second) {
  EXPECT_EQ(first.rows, second.rows);
  EXPECT_EQ(first.columns, second.columns);
  EXPECT_EQ(first.row_offsets, second.row_offsets);
  EXPECT_EQ(first.column_indices, second.column_indices);
  EqualState(first.values, second.values);
}

void EqualVectorResult(const Result<std::vector<double>> &first,
                       const Result<std::vector<double>> &second) {
  ASSERT_EQ(first.ok(), second.ok());
  if (first.ok())
    EqualState(first.value(), second.value());
  else {
    EXPECT_EQ(first.error().code, second.error().code);
    EXPECT_EQ(first.error().message, second.error().message);
  }
}

void EqualResult(const TransientResult &first, const TransientResult &second) {
  ASSERT_EQ(first.times_seconds.size(), second.times_seconds.size());
  ASSERT_EQ(first.states.size(), second.states.size());
  ASSERT_EQ(first.step_trace.size(), second.step_trace.size());
  EXPECT_EQ(first.emitted_points, second.emitted_points);
  EXPECT_EQ(first.derivative_history_error_estimates,
            second.derivative_history_error_estimates);
  EXPECT_EQ(first.derivative_history_step_doubling_checks,
            second.derivative_history_step_doubling_checks);
  EXPECT_EQ(first.step_doubling_error_estimates,
            second.step_doubling_error_estimates);
  EXPECT_EQ(first.derivative_history_fallback_entries,
            second.derivative_history_fallback_entries);
  EXPECT_EQ(first.derivative_history_fallback_recoveries,
            second.derivative_history_fallback_recoveries);
  EqualState(first.times_seconds, second.times_seconds);
  for (std::size_t i = 0; i < first.states.size(); ++i)
    EqualState(first.states[i], second.states[i]);
  for (std::size_t i = 0; i < first.step_trace.size(); ++i) {
    const auto &a = first.step_trace[i];
    const auto &b = second.step_trace[i];
    EqualBits(a.start_time_seconds, b.start_time_seconds);
    EqualBits(a.end_time_seconds, b.end_time_seconds);
    EqualBits(a.step_size_seconds, b.step_size_seconds);
    EqualBits(a.normalized_local_error, b.normalized_local_error);
    EXPECT_EQ(a.method, b.method);
    EXPECT_EQ(a.accepted, b.accepted);
    EXPECT_EQ(a.landed_on_hard_point, b.landed_on_hard_point);
    EXPECT_EQ(a.derivative_history_audited, b.derivative_history_audited);
    EXPECT_EQ(a.derivative_history_audit_agreed,
              b.derivative_history_audit_agreed);
    EXPECT_EQ(a.derivative_history_fallback_active,
              b.derivative_history_fallback_active);
    EXPECT_EQ(a.rejection_reason, b.rejection_reason);
  }
  const auto &a = first.solver_statistics;
  const auto &b = second.solver_statistics;
  EXPECT_EQ(a.symbolic_analyses, b.symbolic_analyses);
  EXPECT_EQ(a.numeric_factorizations, b.numeric_factorizations);
  EXPECT_EQ(a.numeric_refactorizations, b.numeric_refactorizations);
  EXPECT_EQ(a.numeric_refactorization_fallbacks,
            b.numeric_refactorization_fallbacks);
  EXPECT_EQ(a.numeric_reuses, b.numeric_reuses);
  EXPECT_EQ(a.iterative_refinement_solves, b.iterative_refinement_solves);
  EXPECT_EQ(a.solves, b.solves);
}

void EqualPoint(const NonlinearPointResult &first,
                const NonlinearPointResult &second) {
  EqualState(first.solution, second.solution);
  ASSERT_EQ(first.iteration_trace.size(), second.iteration_trace.size());
  for (std::size_t i = 0; i < first.iteration_trace.size(); ++i) {
    const auto &a = first.iteration_trace[i];
    const auto &b = second.iteration_trace[i];
    EXPECT_EQ(a.strategy, b.strategy);
    EqualBits(a.continuation_value, b.continuation_value);
    EXPECT_EQ(a.iteration, b.iteration);
    EqualBits(a.maximum_normalized_update, b.maximum_normalized_update);
    EqualBits(a.maximum_normalized_residual, b.maximum_normalized_residual);
    EXPECT_EQ(a.accepted, b.accepted);
  }
}

TEST(Emi02PreparedTransient,
     PreparedCompanionOwnsUnionKindsSignedZerosAndCancellation) {
  CsrMatrix g{.rows = 4,
              .columns = 4,
              .values = {2, -0.0, 1e-200, 0.0, 7, 1e100},
              .column_indices = {0, 2, 3, 1, 0, 2},
              .row_offsets = {0, 3, 4, 6, 6}};
  CsrMatrix c{.rows = 4,
              .columns = 4,
              .values = {-1, -0.0, -1e-200, -0.0, 3, 1e-100, 0.0},
              .column_indices = {0, 1, 3, 1, 1, 2, 3},
              .row_offsets = {0, 3, 4, 6, 7}};
  const auto original_g = g;
  const auto original_c = c;
  auto prepared = internal::PreparedTransientCompanion::Create(g, c, .5, 1);
  ASSERT_TRUE(prepared.ok()) << prepared.error().message;
  const CsrMatrix expected{
      .rows = 4,
      .columns = 4,
      .values = {0.0, -0.0, -0.0, -1e-200, 0.0, 7, 6, 1e100, 0.0},
      .column_indices = {0, 1, 2, 3, 1, 0, 1, 2, 3},
      .row_offsets = {0, 4, 5, 8, 9}};
  EqualMatrix(prepared.value()->matrix(), expected);
  EXPECT_EQ(prepared.value()->numeric_entry_count(), original_c.values.size());
  EXPECT_LT(prepared.value()->numeric_entry_count(), expected.values.size());
  // Different input pairs with the same checked scale have identical values.
  const auto same_scale = prepared.value()->Form(1, 2);
  ASSERT_TRUE(same_scale.ok());
  EqualMatrix(*same_scale.value(), expected);
  EXPECT_TRUE(std::signbit(prepared.value()->matrix().values[1]));
  EXPECT_TRUE(std::signbit(prepared.value()->matrix().values[2]));
  const auto *const storage = prepared.value()->matrix().values.data();
  const auto *const columns = prepared.value()->matrix().column_indices.data();
  g.values.clear();
  g.row_offsets.clear();
  c.values.assign(c.values.size(), std::numeric_limits<double>::infinity());
  for (const double step : {.5, 1.0, 2.0, 1e100, 1e-100}) {
    for (const double alpha : {1.0, 2.0}) {
      const auto checked =
          FormTransientCompanionMatrix(original_g, original_c, step, alpha);
      const auto actual = prepared.value()->Form(step, alpha);
      ASSERT_TRUE(checked.ok());
      ASSERT_TRUE(actual.ok());
      EqualMatrix(*actual.value(), checked.value());
      EXPECT_EQ(actual.value()->values.data(), storage);
      EXPECT_EQ(actual.value()->column_indices.data(), columns);
    }
  }
  // Positive alpha and h may produce a zero scale by underflow. G-only -0
  // must still be copied, rather than recomputed as -0 + (+0 * missing C).
  const auto zero_scale =
      prepared.value()->Form(std::numeric_limits<double>::max(),
                             std::numeric_limits<double>::denorm_min());
  ASSERT_TRUE(zero_scale.ok());
  const auto checked = FormTransientCompanionMatrix(
      original_g, original_c, std::numeric_limits<double>::max(),
      std::numeric_limits<double>::denorm_min());
  ASSERT_TRUE(checked.ok());
  EqualMatrix(*zero_scale.value(), checked.value());
}

TEST(Emi02PreparedTransient,
     PreparedCompanionRetainsFirstValidationAndNumericFailurePrecedence) {
  const CsrMatrix g{.rows = 2,
                    .columns = 2,
                    .values = {1, 1},
                    .column_indices = {0, 1},
                    .row_offsets = {0, 1, 2}};
  const CsrMatrix c{.rows = 2,
                    .columns = 2,
                    .values = {1, 1e308},
                    .column_indices = {0, 1},
                    .row_offsets = {0, 1, 2}};
  const auto factory_matches = [&](const CsrMatrix &bad_g,
                                   const CsrMatrix &bad_c, double h,
                                   double alpha) {
    const auto checked = FormTransientCompanionMatrix(bad_g, bad_c, h, alpha);
    const auto prepared =
        internal::PreparedTransientCompanion::Create(bad_g, bad_c, h, alpha);
    ASSERT_FALSE(checked.ok());
    ASSERT_FALSE(prepared.ok());
    EXPECT_EQ(prepared.error().code, checked.error().code);
    EXPECT_EQ(prepared.error().message, checked.error().message);
  };
  auto bad_g = g;
  auto bad_c = c;
  bad_g.row_offsets.back() = 3;
  bad_c.values[0] = std::numeric_limits<double>::quiet_NaN();
  factory_matches(bad_g, bad_c, 1, 1);
  factory_matches(bad_g, bad_c, -1, 1);
  bad_g.rows = 3;
  factory_matches(bad_g, bad_c, -1, 1);
  factory_matches(g, bad_c, 1, 1);

  auto prepared = internal::PreparedTransientCompanion::Create(g, c, 1, .25);
  ASSERT_TRUE(prepared.ok());
  const auto reject_then_recover = [&](double h, double alpha,
                                       const std::string &reason) {
    const auto checked = FormTransientCompanionMatrix(g, c, h, alpha);
    const auto actual = prepared.value()->Form(h, alpha);
    ASSERT_FALSE(checked.ok());
    ASSERT_FALSE(actual.ok());
    EXPECT_EQ(actual.error().code, checked.error().code);
    EXPECT_EQ(actual.error().message, checked.error().message);
    EXPECT_NE(actual.error().message.find(reason), std::string::npos);
    const auto recovered = prepared.value()->Form(1, .25);
    const auto expected = FormTransientCompanionMatrix(g, c, 1, .25);
    ASSERT_TRUE(recovered.ok());
    ASSERT_TRUE(expected.ok());
    EqualMatrix(*recovered.value(), expected.value());
  };
  reject_then_recover(0, 1, "finite and positive");
  // Even a previously used ratio cannot excuse invalid inputs.
  reject_then_recover(-4, -1, "finite and positive");
  reject_then_recover(1, std::numeric_limits<double>::infinity(),
                      "finite and positive");
  reject_then_recover(std::numeric_limits<double>::denorm_min(), 1, "scaling");
  // This fails on the second entry, after the first numeric slot was changed.
  reject_then_recover(.25, 1, "matrix produced");
}

TEST(Emi02PreparedTransient, ConstantCompanionStillChecksScalingInputs) {
  const CsrMatrix g{.rows = 1,
                    .columns = 1,
                    .values = {-0.0},
                    .column_indices = {0},
                    .row_offsets = {0, 1}};
  const CsrMatrix c{.rows = 1,
                    .columns = 1,
                    .values = {},
                    .column_indices = {},
                    .row_offsets = {0, 0}};
  auto prepared = internal::PreparedTransientCompanion::Create(g, c, 1, 1);
  ASSERT_TRUE(prepared.ok());
  EXPECT_EQ(prepared.value()->numeric_entry_count(), 0U);
  const auto reused = prepared.value()->Form(.5, .5);
  ASSERT_TRUE(reused.ok());
  EqualMatrix(*reused.value(), g);
  const auto overflow =
      prepared.value()->Form(std::numeric_limits<double>::denorm_min(), 1);
  const auto checked = FormTransientCompanionMatrix(
      g, c, std::numeric_limits<double>::denorm_min(), 1);
  ASSERT_FALSE(overflow.ok());
  ASSERT_FALSE(checked.ok());
  EXPECT_EQ(overflow.error().code, checked.error().code);
  EXPECT_EQ(overflow.error().message, checked.error().message);
  const auto recovered = prepared.value()->Form(1, 1);
  ASSERT_TRUE(recovered.ok());
  EqualMatrix(*recovered.value(), g);
}

TEST(Emi02PreparedTransient,
     PreparedRhsKeepsPartialSumGuardsAndReusesOnlyExplicitSameStateProducts) {
  const CsrMatrix g{.rows = 2,
                    .columns = 2,
                    .values = {2, 3, -1, 4},
                    .column_indices = {0, 1, 0, 1},
                    .row_offsets = {0, 2, 4}};
  const CsrMatrix c{.rows = 2,
                    .columns = 2,
                    .values = {.5, -.5, -.5, .5},
                    .column_indices = {0, 1, 0, 1},
                    .row_offsets = {0, 2, 4}};
  auto prepared = internal::PreparedTransientCompanion::Create(g, c, 1, 1);
  ASSERT_TRUE(prepared.ok());
  const std::vector<double> state{2, -.5};
  std::optional<internal::PreparedTransientCompanion::StateProducts> products;
  for (const double h : {1.0, .5, .01}) {
    EqualVectorResult(prepared.value()->BackwardEulerRhs(state, {1, 3}, h),
                      BuildBackwardEulerRhs(c, state, {1, 3}, h));
    EqualVectorResult(
        prepared.value()->TrapezoidalRhs(state, {1, 3}, {4, -2}, h, &products),
        BuildTrapezoidalRhs(g, c, state, {1, 3}, {4, -2}, h));
    ASSERT_TRUE(products.has_value());
    EqualState(products->g_product(), {2.5, -4});
    EqualState(products->c_product(), {1.25, -1.25});
  }
  auto other_owner = internal::PreparedTransientCompanion::Create(g, c, 1, 1);
  ASSERT_TRUE(other_owner.ok());
  const auto wrong_owner = other_owner.value()->TrapezoidalRhs(
      state, {1, 3}, {4, -2}, .5, &products);
  ASSERT_FALSE(wrong_owner.ok());
  EXPECT_EQ(wrong_owner.error().code, ErrorCode::kInvalidStructure);
  auto retained_products = std::move(*products);
  const auto moved_from =
      prepared.value()->TrapezoidalRhs(state, {1, 3}, {4, -2}, .5, &products);
  ASSERT_FALSE(moved_from.ok());
  EXPECT_EQ(moved_from.error().code, ErrorCode::kInvalidStructure);
  *products = std::move(retained_products);
  EqualVectorResult(
      prepared.value()->TrapezoidalRhs(state, {1, 3}, {4, -2}, .5, &products),
      BuildTrapezoidalRhs(g, c, state, {1, 3}, {4, -2}, .5));
  EqualVectorResult(
      prepared.value()->TrapezoidalRhs({3, 4}, {4, -2}, {7, 8}, .5),
      BuildTrapezoidalRhs(g, c, {3, 4}, {4, -2}, {7, 8}, .5));

  const auto nan = std::numeric_limits<double>::quiet_NaN();
  const auto tiny = std::numeric_limits<double>::denorm_min();
  for (const auto &trial :
       std::vector<std::vector<double>>{{}, {nan, 0}, {1, 2}}) {
    for (const double h : {tiny, 1.0}) {
      EqualVectorResult(prepared.value()->BackwardEulerRhs(trial, {nan, 0}, h),
                        BuildBackwardEulerRhs(c, trial, {nan, 0}, h));
      EqualVectorResult(
          prepared.value()->TrapezoidalRhs(trial, {nan, 0}, {0, 0}, h),
          BuildTrapezoidalRhs(g, c, trial, {nan, 0}, {0, 0}, h));
    }
  }
  // Partial sums must fail before a later exact cancellation or invalid source.
  // All matrix entries are finite, and the complete mathematical sum is finite.
  const CsrMatrix large{.rows = 3,
                        .columns = 3,
                        .values = {1e308, 1e308, -1e308},
                        .column_indices = {0, 1, 2},
                        .row_offsets = {0, 3, 3, 3}};
  const CsrMatrix zero{.rows = 3,
                       .columns = 3,
                       .values = {},
                       .column_indices = {},
                       .row_offsets = {0, 0, 0, 0}};
  for (const bool overflow_in_g : {false, true}) {
    const auto &overflow_g = overflow_in_g ? large : zero;
    const auto &overflow_c = overflow_in_g ? zero : large;
    auto overflow = internal::PreparedTransientCompanion::Create(
        overflow_g, overflow_c, 1, .25);
    ASSERT_TRUE(overflow.ok());
    std::optional<internal::PreparedTransientCompanion::StateProducts>
        failed_products;
    const auto actual = overflow.value()->TrapezoidalRhs(
        {1, 1, 1}, {nan, 0, 0}, {0, 0, 0}, tiny, &failed_products);
    const auto checked = BuildTrapezoidalRhs(overflow_g, overflow_c, {1, 1, 1},
                                             {nan, 0, 0}, {0, 0, 0}, tiny);
    EqualVectorResult(actual, checked);
    ASSERT_FALSE(actual.ok());
    EXPECT_NE(actual.error().message.find("multiplication produced"),
              std::string::npos);
    EXPECT_FALSE(failed_products.has_value());
    if (!overflow_in_g)
      EqualVectorResult(
          overflow.value()->BackwardEulerRhs({1, 1, 1}, {nan, 0, 0}, tiny),
          BuildBackwardEulerRhs(overflow_c, {1, 1, 1}, {nan, 0, 0}, tiny));
  }
}

TEST(Emi02PreparedTransient,
     PreparedPointOwnsMetadataAndMatchesChangingCheckedPoints) {
  const auto original =
      PreparedCompile("Iin 0 out 1m\nRload out 0 1k\nCstore out 0 1u\n"
                      "Bload out 0 I={.001*v(out)*v(out)}\n");
  auto first_matrix =
      FormTransientCompanionMatrix(original.g, original.c, 1e-6, 1);
  ASSERT_TRUE(first_matrix.ok());
  auto admitted = original;
  admitted.g = first_matrix.value();
  ASSERT_TRUE(RemapBehavioralDescriptors(&admitted).ok());
  auto prepared = internal::PreparedNonlinearPointSolver::Create(admitted);
  ASSERT_TRUE(prepared.ok()) << prepared.error().message;
  auto checked_factor = SparseRealFactorization::Analyze(admitted.g);
  ASSERT_TRUE(checked_factor.ok());
  auto checked_system = admitted;
  // Mutating the factory input after construction must not invalidate its
  // topology, descriptors, reactive coordinates, or expression ownership.
  admitted.g.column_indices.clear();
  admitted.capacitor_initial_constraints.clear();
  admitted.behavioral_descriptors.clear();
  admitted.node_names.clear();
  std::size_t iteration = 0;
  for (const double h : {1e-6, 5e-6, 20e-6, 1e-6}) {
    auto matrix = FormTransientCompanionMatrix(original.g, original.c, h,
                                               iteration % 2 == 0 ? 1.0 : 2.0);
    ASSERT_TRUE(matrix.ok());
    const std::vector<double> rhs{.001 +
                                  static_cast<double>(iteration) * .0001};
    const std::vector<double> guess{static_cast<double>(iteration) * .02};
    checked_system.g = matrix.value();
    checked_system.b_dc = rhs;
    auto checked = RunNonlinearPoint(checked_system, guess,
                                     checked_factor.value().get(), 100);
    ASSERT_TRUE(checked.ok()) << checked.error().message;
    auto actual = prepared.value()->Solve(matrix.value(), rhs, guess, 100);
    ASSERT_TRUE(actual.ok()) << actual.error().message;
    EqualPoint(actual.value(), checked.value());
    auto expected_history =
        BuildDiodeResidualContribution(checked_system, actual.value().solution);
    auto actual_history = prepared.value()->History(actual.value().solution);
    ASSERT_TRUE(expected_history.ok());
    ASSERT_TRUE(actual_history.ok());
    EqualState(actual_history.value(), expected_history.value());
    const auto &expected_stats = checked_factor.value()->statistics();
    const auto actual_stats = prepared.value()->statistics();
    EXPECT_EQ(actual_stats.symbolic_analyses, 1U);
    EXPECT_EQ(actual_stats.numeric_refactorizations,
              expected_stats.numeric_refactorizations);
    EXPECT_EQ(actual_stats.numeric_reuses, expected_stats.numeric_reuses);
    EXPECT_EQ(actual_stats.iterative_refinement_solves,
              expected_stats.iterative_refinement_solves);
    EXPECT_EQ(actual_stats.solves, expected_stats.solves);
    ++iteration;
  }
}

TEST(Emi02PreparedTransient,
     PreparedPointRejectsNumericAndStructuralChangesAndRecovers) {
  const auto system =
      PreparedCompile("Iin 0 out 1m\nRload out 0 1k\nCstore out 0 1u\n"
                      "Bload out 0 I={.001*v(out)*v(out)}\n");
  auto prepared = internal::PreparedNonlinearPointSolver::Create(system);
  ASSERT_TRUE(prepared.ok());
  auto checked_factor = SparseRealFactorization::Analyze(system.g);
  ASSERT_TRUE(checked_factor.ok());
  const auto baseline =
      RunNonlinearPoint(system, {0}, checked_factor.value().get(), 100);
  ASSERT_TRUE(baseline.ok()) << baseline.error().message;
  const auto reject = [&](const CsrMatrix &matrix,
                          const std::vector<double> &rhs,
                          const std::vector<double> &guess,
                          std::size_t iterations, ErrorCode expected) {
    const auto failed = prepared.value()->Solve(matrix, rhs, guess, iterations);
    ASSERT_FALSE(failed.ok());
    EXPECT_EQ(failed.error().code, expected);
    const auto recovered =
        prepared.value()->Solve(system.g, system.b_dc, {0}, 100);
    ASSERT_TRUE(recovered.ok()) << recovered.error().message;
    EqualPoint(recovered.value(), baseline.value());
  };
  auto bad = system.g;
  bad.column_indices.front() = bad.columns;
  reject(bad, system.b_dc, {0}, 100, ErrorCode::kInvalidStructure);
  bad = system.g;
  bad.row_offsets.back() = 0;
  reject(bad, system.b_dc, {0}, 100, ErrorCode::kInvalidStructure);
  bad = system.g;
  bad.values.clear();
  reject(bad, system.b_dc, {0}, 100, ErrorCode::kInvalidStructure);
  bad = system.g;
  bad.values.front() = std::numeric_limits<double>::quiet_NaN();
  reject(bad, system.b_dc, {0}, 100, ErrorCode::kNonFinite);
  bad.values.front() = 1e101;
  reject(bad, system.b_dc, {0}, 100, ErrorCode::kNonFinite);
  reject(system.g, {}, {0}, 100, ErrorCode::kInvalidStructure);
  reject(system.g, {std::numeric_limits<double>::infinity()}, {0}, 100,
         ErrorCode::kNonFinite);
  reject(system.g, system.b_dc, {}, 100, ErrorCode::kInvalidStructure);
  reject(system.g, system.b_dc, {std::numeric_limits<double>::quiet_NaN()}, 100,
         ErrorCode::kNonFinite);
  reject(system.g, system.b_dc, {0}, 301, ErrorCode::kInvalidStructure);
  reject(system.g, system.b_dc, {0}, 0, ErrorCode::kNonConvergence);
}

TEST(Emi02PreparedTransient,
     PreparedHistoryKeepsDerivativeGuardsAndHasNoTrialState) {
  const auto system = PreparedCompile(
      "Iin 0 out 6\nRload out 0 1\nBroot out 0 I={v(out)**.5}\n");
  auto prepared = internal::PreparedNonlinearPointSolver::Create(system);
  ASSERT_TRUE(prepared.ok());
  // The value at zero is finite, but its derivative is not. A value-only
  // shortcut for TRAP history would incorrectly accept this input.
  const auto zero = prepared.value()->History({0});
  ASSERT_FALSE(zero.ok());
  EXPECT_EQ(zero.error().code, ErrorCode::kNonFinite);
  const auto nonfinite =
      prepared.value()->History({std::numeric_limits<double>::infinity()});
  ASSERT_FALSE(nonfinite.ok());
  EXPECT_EQ(nonfinite.error().code, ErrorCode::kNonFinite);
  const auto dimension = prepared.value()->History({});
  ASSERT_FALSE(dimension.ok());
  EXPECT_EQ(dimension.error().code, ErrorCode::kInvalidStructure);
  for (const double state : {4.0, -4.0, 9.0, 4.0}) {
    const auto expected = BuildDiodeResidualContribution(system, {state});
    const auto actual = prepared.value()->History({state});
    ASSERT_TRUE(expected.ok());
    ASSERT_TRUE(actual.ok());
    EqualState(actual.value(), expected.value());
  }
  auto checked_factor = SparseRealFactorization::Analyze(system.g);
  ASSERT_TRUE(checked_factor.ok());
  const auto checked =
      RunNonlinearPoint(system, {4}, checked_factor.value().get(), 100);
  const auto actual = prepared.value()->Solve(system.g, system.b_dc, {4}, 100);
  ASSERT_TRUE(checked.ok()) << checked.error().message;
  ASSERT_TRUE(actual.ok()) << actual.error().message;
  EqualPoint(actual.value(), checked.value());
}

TEST(Emi02PreparedTransient,
     CompletePreparedTracesMatchCheckedPointPathBitForBit) {
  const std::vector<MnaSystem> systems{
      NonlinearCharge(),
      PreparedCompile(
          "Vcommand command 0 DC 0 PWL(0 0 10u 1 100u 1)\n"
          "Edrive drive 0 VALUE={v(command)}\nRdrive drive primary 5\n"
          "Lfirst primary 0 1m\nLsecond secondary 0 2m\n"
          "Kpair Lfirst Lsecond -.4\nCstore secondary 0 1u\n"
          "Bload secondary 0 I={.01*v(secondary)}\n"),
      PreparedCompile("Vinput input 0 DC 0 PWL(0 0 1m 1 2m 1)\n"
                      "Edrive drive 0 VALUE={v(input)*v(input)}\n"
                      "Rcharge drive out 1k\nCstate out 0 1u\n"
                      "Bload out 0 I={.001*v(out)}\n")};
  const std::vector<TranAnalysis> analyses{{5e-6, 1e-3, 0, false},
                                           {1e-6, 100e-6, 0, false},
                                           {50e-6, 2e-3, 0, false}};
  for (std::size_t i = 0; i < systems.size(); ++i) {
    SCOPED_TRACE(i);
    ASSERT_GT(systems[i].g.rows, 0U);
    const auto checked = RunPreparedOrChecked(false, systems[i], analyses[i]);
    ASSERT_TRUE(checked.ok()) << checked.error().message;
    const auto prepared = RunPreparedOrChecked(true, systems[i], analyses[i]);
    ASSERT_TRUE(prepared.ok()) << prepared.error().message;
    EqualResult(prepared.value(), checked.value());
    if (i == 0) {
      // The nonlinear charge fixture has changing behavioral history and a
      // nonzero step-doubling error. Bitwise LTE parity therefore discriminates
      // reuse at the private midpoint or across separate attempts, as well as
      // changes to the accepted full-step trajectory.
      const auto first_history = BuildDiodeResidualContribution(
          systems[i], checked.value().states.front());
      const auto last_history = BuildDiodeResidualContribution(
          systems[i], checked.value().states.back());
      ASSERT_TRUE(first_history.ok());
      ASSERT_TRUE(last_history.ok());
      EXPECT_NE(first_history.value(), last_history.value());
      EXPECT_TRUE(std::any_of(
          checked.value().step_trace.begin(), checked.value().step_trace.end(),
          [](const auto &step) {
            return step.accepted &&
                   step.method == TransientIntegrationMethod::kTrapezoidal &&
                   step.normalized_local_error > 0;
          }));
    }
    EXPECT_EQ(prepared.value().solver_statistics.symbolic_analyses, 1U);
    EXPECT_GT(prepared.value().solver_statistics.numeric_refactorizations, 0U);
    EXPECT_TRUE(std::any_of(
        prepared.value().step_trace.begin(), prepared.value().step_trace.end(),
        [](const auto &step) {
          return step.accepted &&
                 step.method == TransientIntegrationMethod::kTrapezoidal;
        }));
  }
}

TEST(Emi02PreparedTransient, ObserverCannotMutateTheAdmittedCircuitSnapshot) {
  const auto original = NonlinearCharge();
  const TranAnalysis analysis{5e-6, 1e-3, 0, false};
  const auto baseline = RunTransientAnalysis(original, analysis);
  ASSERT_TRUE(baseline.ok()) << baseline.error().message;
  for (const bool prepared : {false, true}) {
    for (const std::size_t mutate_at : {1U, 7U}) {
      SCOPED_TRACE(prepared);
      SCOPED_TRACE(mutate_at);
      auto external_alias = original;
      std::size_t calls = 0;
      TransientExecutionLimits limits;
      limits.accepted_state_observer = [&](double,
                                           const std::vector<double> &) {
        if (++calls == mutate_at) {
          // The first callback precedes lazy preparation. The later one occurs
          // after it. Neither may change sources, history or matrix metadata.
          external_alias.g.column_indices.clear();
          external_alias.c.values.front() =
              std::numeric_limits<double>::quiet_NaN();
          external_alias.transient_sources.clear();
          external_alias.b_dc.assign(external_alias.g.rows, 123.0);
          external_alias.behavioral_descriptors.front().rows.clear();
          external_alias.capacitor_initial_constraints.clear();
        }
        return Result<bool>::Ok(true);
      };
      const auto result =
          RunPreparedOrChecked(prepared, external_alias, analysis, limits);
      ASSERT_TRUE(result.ok()) << result.error().message;
      EXPECT_EQ(calls, baseline.value().emitted_points);
      EqualResult(result.value(), baseline.value());
      // Ownership is per invocation; a later call must validate the newly
      // malformed input instead of finding an old cached admitted circuit.
      const auto next =
          RunPreparedOrChecked(prepared, external_alias, analysis);
      ASSERT_FALSE(next.ok());
      EXPECT_EQ(next.error().code, ErrorCode::kNonFinite);
    }
  }
}

TEST(Emi02PreparedTransient,
     RejectedTrialsAndFailedObserversDoNotEscapeOrPersist) {
  const auto system = NonlinearCharge();
  const TranAnalysis analysis{5e-6, 1e-3, 0, false};
  TransientExecutionLimits limited_newton;
  limited_newton.nonlinear_maximum_iterations = 2;
  const auto checked =
      RunPreparedOrChecked(false, system, analysis, limited_newton);
  ASSERT_TRUE(checked.ok()) << checked.error().message;
  const auto baseline =
      RunPreparedOrChecked(true, system, analysis, limited_newton);
  ASSERT_TRUE(baseline.ok()) << baseline.error().message;
  EqualResult(baseline.value(), checked.value());
  for (const auto reason :
       {TransientStepRejectionReason::kLocalError,
        TransientStepRejectionReason::kNonlinearConvergence}) {
    const auto found = std::find_if(
        baseline.value().step_trace.begin(), baseline.value().step_trace.end(),
        [reason](const auto &step) {
          return !step.accepted && step.rejection_reason == reason;
        });
    ASSERT_NE(found, baseline.value().step_trace.end());
    const auto attempts =
        static_cast<std::size_t>(found - baseline.value().step_trace.begin()) +
        1;
    for (const bool prepared : {false, true}) {
      auto limits = limited_newton;
      limits.maximum_step_attempts = attempts;
      limits.maximum_accepted_steps = attempts;
      std::size_t observed = 0;
      limits.accepted_state_observer = [&](double time,
                                           const std::vector<double> &state) {
        EXPECT_LT(observed, baseline.value().states.size());
        if (observed < baseline.value().states.size()) {
          EqualBits(time, baseline.value().times_seconds[observed]);
          EqualState(state, baseline.value().states[observed]);
        }
        ++observed;
        return Result<bool>::Ok(true);
      };
      const auto stopped =
          RunPreparedOrChecked(prepared, system, analysis, limits);
      ASSERT_FALSE(stopped.ok());
      EXPECT_EQ(stopped.error().code,
                reason == TransientStepRejectionReason::kNonlinearConvergence
                    ? ErrorCode::kNonConvergence
                    : ErrorCode::kSolve);
      const auto accepted_before = static_cast<std::size_t>(
          std::count_if(baseline.value().step_trace.begin(), found,
                        [](const auto &step) { return step.accepted; }));
      EXPECT_EQ(observed, accepted_before + 1);
      const auto restarted =
          RunPreparedOrChecked(prepared, system, analysis, limited_newton);
      ASSERT_TRUE(restarted.ok()) << restarted.error().message;
      EqualResult(restarted.value(), baseline.value());
    }
  }
  for (const bool prepared : {false, true}) {
    auto limits = limited_newton;
    std::size_t observed = 0;
    limits.accepted_state_observer = [&](double time,
                                         const std::vector<double> &state) {
      EqualBits(time, baseline.value().times_seconds[observed]);
      EqualState(state, baseline.value().states[observed]);
      return ++observed == 7 ? Result<bool>::Fail(ErrorCode::kIo,
                                                  "injected observer failure")
                             : Result<bool>::Ok(true);
    };
    const auto failed =
        RunPreparedOrChecked(prepared, system, analysis, limits);
    ASSERT_FALSE(failed.ok());
    EXPECT_EQ(failed.error().code, ErrorCode::kIo);
    EXPECT_EQ(observed, 7U);
    const auto restart =
        RunPreparedOrChecked(prepared, system, analysis, limited_newton);
    ASSERT_TRUE(restart.ok()) << restart.error().message;
    EqualResult(restart.value(), baseline.value());
  }
}

TEST(Emi02PreparedTransient, InterleavedInvocationOwnsItsWorkspaceAndHistory) {
  const auto outer = NonlinearCharge();
  const auto inner = PreparedCompile(
      "Vinput in 0 DC 0 PWL(0 0 1u 1 10u 1)\n"
      "Edrive drive 0 VALUE={v(in)}\nRload drive out 1k\nCload out 0 1n\n");
  const TranAnalysis outer_analysis{5e-6, 1e-3, 0, false};
  const TranAnalysis inner_analysis{100e-9, 10e-6, 0, false};
  const auto expected_outer =
      RunPreparedOrChecked(false, outer, outer_analysis);
  const auto expected_inner =
      RunPreparedOrChecked(false, inner, inner_analysis);
  ASSERT_TRUE(expected_outer.ok()) << expected_outer.error().message;
  ASSERT_TRUE(expected_inner.ok()) << expected_inner.error().message;
  std::size_t calls = 0;
  bool nested_done = false;
  TransientExecutionLimits limits;
  limits.accepted_state_observer = [&](double, const std::vector<double> &) {
    if (++calls == 7) {
      const auto nested = RunTransientAnalysis(inner, inner_analysis);
      EXPECT_TRUE(nested.ok());
      if (nested.ok())
        EqualResult(nested.value(), expected_inner.value());
      nested_done = true;
    }
    return Result<bool>::Ok(true);
  };
  const auto actual = RunTransientAnalysis(outer, outer_analysis, limits);
  ASSERT_TRUE(actual.ok()) << actual.error().message;
  EXPECT_TRUE(nested_done);
  EqualResult(actual.value(), expected_outer.value());
}

TEST(Emi02PreparedTransient,
     RejectedDomainTrialsRetainCheckedRetryFailureAndAcceptedPrefix) {
  const auto system = PreparedCompile(
      "Vcontrol control 0 DC 0 PWL(0 0 1m 1)\n"
      "Edrive drive 0 VALUE={if(v(control)>.5,1e100*10,v(control))}\n"
      "Rload drive 0 1k\nCload drive 0 1u\n");
  ASSERT_GT(system.g.rows, 0U);
  const TranAnalysis analysis{100e-6, 1e-3, 0, false};
  std::vector<double> checked_times;
  std::vector<std::vector<double>> checked_states;
  TransientExecutionLimits checked_limits;
  checked_limits.accepted_state_observer =
      [&](double time, const std::vector<double> &state) {
        checked_times.push_back(time);
        checked_states.push_back(state);
        return Result<bool>::Ok(true);
      };
  const auto checked =
      RunPreparedOrChecked(false, system, analysis, checked_limits);
  ASSERT_FALSE(checked.ok());
  // The domain error is first encountered by a private Newton trial. The
  // unchanged policy backtracks to finite trials and retries shorter steps;
  // it ultimately reports nonlinear exhaustion, not an immediate domain error.
  EXPECT_EQ(checked.error().code, ErrorCode::kNonConvergence)
      << checked.error().message;
  ASSERT_GT(checked_states.size(), 2U);
  EXPECT_GE(checked_times.back(), 400e-6);
  const auto control = static_cast<std::size_t>(
      std::find(system.node_names.begin(), system.node_names.end(), "control") -
      system.node_names.begin());
  ASSERT_LT(control, system.node_names.size());
  for (const auto &state : checked_states)
    EXPECT_LE(state[control], .5);
  std::size_t observed = 0;
  TransientExecutionLimits prepared_limits;
  prepared_limits.accepted_state_observer =
      [&](double time, const std::vector<double> &state) {
        EXPECT_LT(observed, checked_states.size());
        if (observed < checked_states.size()) {
          EqualBits(time, checked_times[observed]);
          EqualState(state, checked_states[observed]);
        }
        ++observed;
        return Result<bool>::Ok(true);
      };
  const auto prepared =
      RunPreparedOrChecked(true, system, analysis, prepared_limits);
  ASSERT_FALSE(prepared.ok());
  EXPECT_EQ(prepared.error().code, checked.error().code);
  EXPECT_EQ(observed, checked_states.size());
}

TEST(Emi02PreparedTransient,
     LteRetriesRetainTheirMethodAndNewtonFailureRestartsWithBackwardEuler) {
  const auto ringing =
      PreparedCompile("Vdrive drive 0 DC 0 PWL(0 0 1u 1 50u 1)\n"
                      "Rloss drive package 10\nLpackage package out 100u\n"
                      "Cpackage out 0 10n\nBzero out 0 I={0*v(out)}\n");
  const auto result = RunTransientAnalysis(ringing, {1e-6, 50e-6, 0, false});
  ASSERT_TRUE(result.ok()) << result.error().message;
  std::size_t rejected_be = 0;
  std::size_t rejected_trap = 0;
  for (std::size_t i = 0; i + 1 < result.value().step_trace.size(); ++i) {
    const auto &step = result.value().step_trace[i];
    if (step.accepted ||
        step.rejection_reason != TransientStepRejectionReason::kLocalError)
      continue;
    const auto &retry = result.value().step_trace[i + 1];
    EqualBits(retry.start_time_seconds, step.start_time_seconds);
    EXPECT_LT(retry.step_size_seconds, step.step_size_seconds);
    EXPECT_EQ(retry.method, step.method);
    step.method == TransientIntegrationMethod::kBackwardEuler ? ++rejected_be
                                                              : ++rejected_trap;
  }
  EXPECT_GT(rejected_be, 0U);
  EXPECT_GT(rejected_trap, 0U);

  TransientExecutionLimits limits;
  limits.nonlinear_maximum_iterations = 2;
  const auto nonlinear =
      RunTransientAnalysis(NonlinearCharge(), {5e-6, 1e-3, 0, false}, limits);
  ASSERT_TRUE(nonlinear.ok()) << nonlinear.error().message;
  std::size_t rejected_newton = 0;
  for (std::size_t i = 0; i + 1 < nonlinear.value().step_trace.size(); ++i) {
    const auto &step = nonlinear.value().step_trace[i];
    if (step.accepted ||
        step.rejection_reason !=
            TransientStepRejectionReason::kNonlinearConvergence)
      continue;
    const auto &retry = nonlinear.value().step_trace[i + 1];
    EqualBits(retry.start_time_seconds, step.start_time_seconds);
    EXPECT_EQ(retry.method, TransientIntegrationMethod::kBackwardEuler);
    EqualBits(retry.step_size_seconds, step.step_size_seconds * .5);
    ++rejected_newton;
  }
  EXPECT_GT(rejected_newton, 0U);
}

TEST(Emi02PreparedTransient,
     DerivativeHistoryPreparedAndCheckedInvocationsRemainBitExact) {
  const auto system = NonlinearCharge();
  const TranAnalysis analysis{5e-6, 1e-3, 0, false};
  TransientExecutionLimits limits;
  limits.behavioral_error_estimator =
      BehavioralErrorEstimator::kDerivativeHistory;
  const auto checked = RunPreparedOrChecked(false, system, analysis, limits);
  ASSERT_TRUE(checked.ok()) << checked.error().message;
  const auto prepared = RunPreparedOrChecked(true, system, analysis, limits);
  ASSERT_TRUE(prepared.ok()) << prepared.error().message;
  EqualResult(prepared.value(), checked.value());
  EXPECT_GT(prepared.value().derivative_history_error_estimates, 0U);
  EXPECT_GT(prepared.value().derivative_history_step_doubling_checks, 0U);
  EXPECT_GT(prepared.value().step_doubling_error_estimates,
            prepared.value().derivative_history_step_doubling_checks);

  auto external_alias = system;
  std::size_t observed = 0;
  limits.accepted_state_observer = [&](double time,
                                       const std::vector<double> &state) {
    if (++observed == 1) {
      external_alias.capacitor_initial_constraints.clear();
      external_alias.transient_sources.clear();
      external_alias.g.values.assign(external_alias.g.values.size(), 123.0);
    }
    EXPECT_LT(observed - 1, checked.value().states.size());
    if (observed - 1 < checked.value().states.size()) {
      EqualBits(time, checked.value().times_seconds[observed - 1]);
      EqualState(state, checked.value().states[observed - 1]);
    }
    return observed == 17 ? Result<bool>::Fail(
                                ErrorCode::kIo,
                                "injected derivative-history observer failure")
                          : Result<bool>::Ok(true);
  };
  const auto failed = RunTransientAnalysis(external_alias, analysis, limits);
  ASSERT_FALSE(failed.ok());
  EXPECT_EQ(failed.error().code, ErrorCode::kIo);
  EXPECT_EQ(observed, 17U);
  limits.accepted_state_observer = {};
  const auto restarted = RunTransientAnalysis(system, analysis, limits);
  ASSERT_TRUE(restarted.ok()) << restarted.error().message;
  EqualResult(restarted.value(), prepared.value());
}

TEST(Emi02PreparedTransient,
     DerivativeHistoryUsesTimestampDurationAndRespectsRoundedMaximumStep) {
  const auto system = PreparedCompile(
      "Rhold out 0 1k\nCstore out 0 1u\nBzero out 0 I={0*v(out)}\n"
      "Ihold 0 out DC 0 PWL(0 0 .55 0 1 0)\n");
  const TranAnalysis analysis{.1, 1.0, 0, false};
  // Decimal tenths discriminate a proposed h from fl(t+h)-t. This default
  // reference retains its established arithmetic and demonstrates that the
  // fixture actually encounters a rounded duration, rather than exact dyadics.
  const auto original = RunTransientAnalysis(system, analysis);
  ASSERT_TRUE(original.ok()) << original.error().message;
  EXPECT_TRUE(
      std::any_of(original.value().step_trace.begin(),
                  original.value().step_trace.end(), [](const auto &step) {
                    return step.step_size_seconds !=
                           step.end_time_seconds - step.start_time_seconds;
                  }));

  TransientExecutionLimits limits;
  limits.behavioral_error_estimator =
      BehavioralErrorEstimator::kDerivativeHistory;
  const auto result = RunTransientAnalysis(system, analysis, limits);
  ASSERT_TRUE(result.ok()) << result.error().message;
  const auto checked = RunPreparedOrChecked(false, system, analysis, limits);
  ASSERT_TRUE(checked.ok()) << checked.error().message;
  EqualResult(result.value(), checked.value());
  std::size_t waveform_landings = 0;
  for (const auto &step : result.value().step_trace) {
    EqualBits(step.step_size_seconds,
              step.end_time_seconds - step.start_time_seconds);
    EXPECT_GT(step.step_size_seconds, 0.0);
    EXPECT_LE(step.step_size_seconds, analysis.time_step_seconds);
    if (step.accepted && step.end_time_seconds == .55) {
      EXPECT_TRUE(step.landed_on_hard_point);
      EXPECT_EQ(step.method, TransientIntegrationMethod::kBackwardEuler);
      ++waveform_landings;
    }
  }
  EXPECT_EQ(waveform_landings, 1U);
  EqualBits(result.value().times_seconds.back(), 1.0);
}

TEST(Emi02PreparedTransient,
     DerivativeHistoryRecoveryRequiresSixteenUninterruptedAcceptedAgreements) {
  const auto system =
      PreparedCompile("Vdrive drive 0 DC 0 PWL(0 0 1u 1 200u 1)\n"
                      "Rloss drive package 10\nLpackage package out 100u\n"
                      "Cpackage out 0 10n\nBzero out 0 I={0*v(out)}\n");
  TransientExecutionLimits limits;
  limits.behavioral_error_estimator =
      BehavioralErrorEstimator::kDerivativeHistory;
  const auto result =
      RunTransientAnalysis(system, {1e-6, 200e-6, 0, false}, limits);
  ASSERT_TRUE(result.ok()) << result.error().message;

  // Reconstruct the policy from inspectable trial outcomes, including rejected
  // attempts. A fallback interval may end only after sixteen uninterrupted,
  // accepted agreements or an accepted BE reset. In particular, the fifteenth
  // agreement must still expose active fallback.
  bool fallback = false;
  std::size_t agreements = 0;
  std::size_t entries = 0;
  std::size_t recoveries = 0;
  std::size_t fifteenth_agreements = 0;
  std::size_t partial_lte_resets = 0;
  std::size_t partial_disagreement_resets = 0;
  for (const auto &step : result.value().step_trace) {
    SCOPED_TRACE(step.start_time_seconds);
    EXPECT_FALSE(step.derivative_history_audit_agreed &&
                 !step.derivative_history_audited);
    const bool disagreement = step.derivative_history_audited &&
                              !step.derivative_history_audit_agreed;
    if (disagreement && !fallback) {
      fallback = true;
      ++entries;
    }
    if (!step.accepted || disagreement) {
      if (fallback && agreements != 0) {
        if (step.rejection_reason == TransientStepRejectionReason::kLocalError)
          ++partial_lte_resets;
        if (disagreement)
          ++partial_disagreement_resets;
      }
      agreements = 0;
    }
    if (step.accepted &&
        step.method == TransientIntegrationMethod::kBackwardEuler) {
      fallback = false;
      agreements = 0;
    } else if (step.accepted && fallback &&
               step.derivative_history_audit_agreed) {
      ++agreements;
      if (agreements == 15) {
        EXPECT_TRUE(step.derivative_history_fallback_active);
        ++fifteenth_agreements;
      } else if (agreements == 16) {
        EXPECT_FALSE(step.derivative_history_fallback_active);
        fallback = false;
        agreements = 0;
        ++recoveries;
      }
    }
    EXPECT_EQ(step.derivative_history_fallback_active, fallback);
  }
  EXPECT_EQ(entries, result.value().derivative_history_fallback_entries);
  EXPECT_EQ(recoveries, result.value().derivative_history_fallback_recoveries);
  EXPECT_GT(entries, 0U);
  EXPECT_GT(recoveries, 0U);
  EXPECT_GE(fifteenth_agreements, recoveries);
  EXPECT_GT(partial_lte_resets, 0U);
  EXPECT_GT(partial_disagreement_resets, 0U);
}

} // namespace
} // namespace ohmnivore
