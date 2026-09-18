#include "cpp/tests/google_test.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "cpp/src/nonlinear_internal.h"
#include "ohmnivore/behavioral.h"

namespace ohmnivore {
namespace {

MnaSystem CacheCompile(const std::string &deck) {
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

std::size_t CacheNode(const MnaSystem &system, const std::string &name) {
  const auto found =
      std::find(system.node_names.begin(), system.node_names.end(), name);
  EXPECT_NE(found, system.node_names.end());
  return static_cast<std::size_t>(found - system.node_names.begin());
}

void CacheEqualBits(double a, double b) {
  EXPECT_EQ(std::bit_cast<std::uint64_t>(a), std::bit_cast<std::uint64_t>(b));
}

void CacheEqualState(const std::vector<double> &a,
                     const std::vector<double> &b) {
  ASSERT_EQ(a.size(), b.size());
  for (std::size_t i = 0; i < a.size(); ++i)
    CacheEqualBits(a[i], b[i]);
}

void CacheEqualPoint(const NonlinearPointResult &a,
                     const NonlinearPointResult &b) {
  CacheEqualState(a.solution, b.solution);
  ASSERT_EQ(a.iteration_trace.size(), b.iteration_trace.size());
  for (std::size_t i = 0; i < a.iteration_trace.size(); ++i) {
    const auto &x = a.iteration_trace[i], &y = b.iteration_trace[i];
    EXPECT_EQ(x.strategy, y.strategy);
    CacheEqualBits(x.continuation_value, y.continuation_value);
    EXPECT_EQ(x.iteration, y.iteration);
    CacheEqualBits(x.maximum_normalized_update, y.maximum_normalized_update);
    CacheEqualBits(x.maximum_normalized_residual,
                   y.maximum_normalized_residual);
    EXPECT_EQ(x.accepted, y.accepted);
  }
}

void CacheEqualStatistics(const SparseSolverStatistics &a,
                          const SparseSolverStatistics &b) {
  EXPECT_EQ(a.symbolic_analyses, b.symbolic_analyses);
  EXPECT_EQ(a.numeric_factorizations, b.numeric_factorizations);
  EXPECT_EQ(a.numeric_refactorizations, b.numeric_refactorizations);
  EXPECT_EQ(a.numeric_refactorization_fallbacks,
            b.numeric_refactorization_fallbacks);
  EXPECT_EQ(a.numeric_reuses, b.numeric_reuses);
  EXPECT_EQ(a.solves, b.solves);
  EXPECT_EQ(a.iterative_refinement_solves, b.iterative_refinement_solves);
}

TEST(Emi02ExpressionCache,
     ReusesInitialHistoryAndImmediateAcceptedTrialDerivatives) {
  auto system = CacheCompile("Iin 0 out 1m\nRload out 0 1k\n"
                             "Bload out 0 I={.001*v(out)*v(out)}\n");
  auto prepared = internal::PreparedNonlinearPointSolver::Create(system);
  ASSERT_TRUE(prepared.ok()) << prepared.error().message;
  auto factor = SparseRealFactorization::Analyze(system.g);
  ASSERT_TRUE(factor.ok());
  std::vector<double> state(system.g.rows, 0);
  std::size_t trials = 0;
  for (std::size_t i = 0; i < 4; ++i) {
    // Same expressions and initial state, different companion and source.
    // Cached expression values cannot substitute for rebuilding these rows.
    system.g.values.front() = .001 + 1e-12 + i * .0001;
    system.b_dc.front() = .001 + i * .0003;
    auto expected = RunNonlinearPoint(system, state, factor.value().get(), 100);
    ASSERT_TRUE(expected.ok()) << expected.error().message;
    auto actual = prepared.value()->Solve(system.g, system.b_dc, state, 100);
    ASSERT_TRUE(actual.ok()) << actual.error().message;
    CacheEqualPoint(actual.value(), expected.value());
    trials += actual.value().iteration_trace.size();
    state = actual.value().solution;
    auto history = prepared.value()->History(state);
    auto checked = BuildDiodeResidualContribution(system, state);
    ASSERT_TRUE(history.ok());
    ASSERT_TRUE(checked.ok());
    CacheEqualState(history.value(), checked.value());
    const auto counts = prepared.value()->expression_cache_statistics();
    EXPECT_EQ(counts.initial_hits, i);
    EXPECT_EQ(counts.initial_misses, 1U);
    EXPECT_EQ(counts.history_hits, i + 1);
    EXPECT_EQ(counts.history_misses, 0U);
    EXPECT_EQ(counts.fresh_initial_evaluations, 1U);
    EXPECT_EQ(counts.fresh_trial_evaluations, trials);
    EXPECT_EQ(counts.fresh_final_value_evaluations, i + 1);
    EXPECT_EQ(counts.reused_final_derivatives, i + 1);
    EXPECT_EQ(counts.fresh_final_evaluations, 0U);
    EXPECT_EQ(counts.fresh_history_evaluations, 0U);
    EXPECT_EQ(counts.publications, i + 1);
  }
  CacheEqualStatistics(prepared.value()->statistics(),
                       factor.value()->statistics());
}

TEST(Emi02ExpressionCache,
     FourEntriesRetainHalfStepStatesAndFailedSolveDoesNotEvict) {
  const auto system = CacheCompile("Rload out 0 1k\n"
                                   "Bload out 0 I={.001*v(out)*v(out)}\n");
  auto prepared = internal::PreparedNonlinearPointSolver::Create(system);
  ASSERT_TRUE(prepared.ok()) << prepared.error().message;
  std::vector<std::vector<double>> states;
  for (const double rhs : {.001, .0011, .0012, .0013}) {
    auto point = prepared.value()->Solve(system.g, {rhs}, {0}, 100);
    ASSERT_TRUE(point.ok()) << point.error().message;
    states.push_back(point.value().solution);
  }
  auto counts = prepared.value()->expression_cache_statistics();
  EXPECT_EQ(counts.retained_entries, 4U);
  auto exhausted =
      prepared.value()->Solve(system.g, {.0014}, states.front(), 0);
  ASSERT_FALSE(exhausted.ok());
  EXPECT_EQ(exhausted.error().code, ErrorCode::kNonConvergence);
  EXPECT_EQ(prepared.value()->expression_cache_statistics().publications, 4U);
  // The fourth entry models the next full step; the original accepted state
  // remains available for its first half after two prior half-step
  // publications.
  for (const auto &state : states)
    ASSERT_TRUE(prepared.value()->History(state).ok());
  counts = prepared.value()->expression_cache_statistics();
  EXPECT_EQ(counts.history_hits, 4U);
  EXPECT_EQ(counts.history_misses, 0U);
  auto fifth = prepared.value()->Solve(system.g, {.0014}, states.back(), 100);
  ASSERT_TRUE(fifth.ok()) << fifth.error().message;
  ASSERT_TRUE(prepared.value()->History(states.front()).ok());
  counts = prepared.value()->expression_cache_statistics();
  EXPECT_EQ(counts.history_misses, 1U);
  EXPECT_EQ(counts.retained_entries, 4U);
  EXPECT_EQ(counts.publications, 5U);
}

TEST(Emi02ExpressionCache, BitKeysDistinguishSignedZeroAndOneUlpChanges) {
  const auto system = CacheCompile("Rload out 0 1k\nBzero out 0 I={0}\n");
  auto prepared = internal::PreparedNonlinearPointSolver::Create(system);
  ASSERT_TRUE(prepared.ok());
  auto zero = prepared.value()->Solve(system.g, {0}, {0}, 100);
  ASSERT_TRUE(zero.ok()) << zero.error().message;
  ASSERT_EQ(zero.value().solution[0], 0);
  const auto before = prepared.value()->expression_cache_statistics();
  auto repeated =
      prepared.value()->Solve(system.g, {0}, zero.value().solution, 100);
  ASSERT_TRUE(repeated.ok()) << repeated.error().message;
  CacheEqualState(repeated.value().solution, zero.value().solution);
  const auto after = prepared.value()->expression_cache_statistics();
  EXPECT_EQ(after.initial_hits, before.initial_hits + 1);
  // Even when the proposed and final state are already cached, these checks
  // must execute the evaluator again. An initial hit alone cannot accept it.
  EXPECT_EQ(after.fresh_trial_evaluations, before.fresh_trial_evaluations + 1);
  EXPECT_EQ(after.fresh_final_value_evaluations,
            before.fresh_final_value_evaluations + 1);
  EXPECT_EQ(after.reused_final_derivatives,
            before.reused_final_derivatives + 1);
  EXPECT_EQ(after.fresh_final_evaluations, 0U);
  ASSERT_TRUE(prepared.value()->History(zero.value().solution).ok());
  auto changed = zero.value().solution;
  changed[0] = std::copysign(0.0, std::signbit(changed[0]) ? 1.0 : -1.0);
  ASSERT_TRUE(prepared.value()->History(changed).ok());
  changed[0] = std::nextafter(0.0, 1.0);
  ASSERT_TRUE(prepared.value()->History(changed).ok());
  const auto counts = prepared.value()->expression_cache_statistics();
  EXPECT_EQ(counts.history_hits, 1U);
  EXPECT_EQ(counts.history_misses, 2U);
  EXPECT_EQ(counts.fresh_history_evaluations, 2U);
  EXPECT_EQ(counts.publications, 2U);
}

TEST(Emi02ExpressionCache, StateValidationAndLazyDomainFailuresCannotBeHidden) {
  const auto system =
      CacheCompile("Vgate gate 0 -1\nRload out 0 1k\nRspare spare 0 1k\n"
                   "Bsafe out 0 I={.001*v(out)}\n"
                   "Bguard out 0 I={if(v(gate)>0,v(spare)+0**-1,0)}\n");
  auto prepared = internal::PreparedNonlinearPointSolver::Create(system);
  ASSERT_TRUE(prepared.ok()) << prepared.error().message;
  auto point = prepared.value()->Solve(
      system.g, system.b_dc, std::vector<double>(system.g.rows, 0), 100);
  ASSERT_TRUE(point.ok()) << point.error().message;
  const auto valid = point.value().solution;
  auto malformed = valid;
  malformed.pop_back();
  auto bad_size = prepared.value()->History(malformed);
  ASSERT_FALSE(bad_size.ok());
  EXPECT_EQ(bad_size.error().code, ErrorCode::kInvalidStructure);
  malformed = valid;
  malformed[CacheNode(system, "spare")] =
      std::numeric_limits<double>::quiet_NaN();
  auto bad_state = prepared.value()->History(malformed);
  ASSERT_FALSE(bad_state.ok());
  EXPECT_EQ(bad_state.error().code, ErrorCode::kNonFinite);
  auto bad_initial =
      prepared.value()->Solve(system.g, system.b_dc, malformed, 100);
  ASSERT_FALSE(bad_initial.ok());
  EXPECT_EQ(bad_initial.error().code, ErrorCode::kNonFinite);
  auto counts = prepared.value()->expression_cache_statistics();
  // Size/bounds rejection precedes even lookup, including an inactive branch.
  EXPECT_EQ(counts.initial_misses, 1U);
  EXPECT_EQ(counts.history_hits + counts.history_misses, 0U);
  malformed = valid;
  malformed[CacheNode(system, "gate")] = 1;
  auto domain = prepared.value()->History(malformed);
  ASSERT_FALSE(domain.ok());
  EXPECT_EQ(domain.error().code, ErrorCode::kNonFinite);
  auto checked_domain = BuildDiodeResidualContribution(system, malformed);
  ASSERT_FALSE(checked_domain.ok());
  EXPECT_EQ(domain.error().message, checked_domain.error().message);
  auto failed_point =
      prepared.value()->Solve(system.g, system.b_dc, malformed, 100);
  ASSERT_FALSE(failed_point.ok());
  EXPECT_EQ(failed_point.error().code, ErrorCode::kNonFinite);
  counts = prepared.value()->expression_cache_statistics();
  EXPECT_EQ(counts.publications, 1U);
  EXPECT_EQ(counts.retained_entries, 1U);
  EXPECT_EQ(counts.fresh_history_evaluations, 2U);
  auto restored = prepared.value()->History(valid);
  auto checked = BuildDiodeResidualContribution(system, valid);
  ASSERT_TRUE(restored.ok());
  ASSERT_TRUE(checked.ok());
  CacheEqualState(restored.value(), checked.value());
  EXPECT_EQ(prepared.value()->expression_cache_statistics().history_hits, 1U);
}

TEST(Emi02ExpressionCache,
     ProgramsRemainOwnedAndCacheIsNotSharedBetweenOwners) {
  const auto original = CacheCompile("Iin 0 out 1m\nRload out 0 1k\n"
                                     "Bload out 0 I={.001*v(out)*v(out)}\n");
  auto aliased = original;
  auto first = internal::PreparedNonlinearPointSolver::Create(aliased);
  auto second = internal::PreparedNonlinearPointSolver::Create(original);
  ASSERT_TRUE(first.ok());
  ASSERT_TRUE(second.ok());
  auto point = first.value()->Solve(original.g, original.b_dc, {0}, 100);
  ASSERT_TRUE(point.ok());
  aliased.behavioral_descriptors.clear();
  aliased.g.values.clear();
  auto cached = first.value()->History(point.value().solution);
  auto uncached = second.value()->History(point.value().solution);
  ASSERT_TRUE(cached.ok());
  ASSERT_TRUE(uncached.ok());
  CacheEqualState(cached.value(), uncached.value());
  EXPECT_EQ(first.value()->expression_cache_statistics().history_hits, 1U);
  EXPECT_EQ(second.value()->expression_cache_statistics().history_misses, 1U);
  EXPECT_EQ(second.value()->expression_cache_statistics().publications, 0U);
}

TEST(Emi02ExpressionCache,
     PreparedFinalResidualMatchesFullVoltageAndTwoTerminalCurrentAssembly) {
  auto system = CacheCompile(
      "Vcontrol control 0 .3\nRload auxiliary 0 1k\n"
      "Eaux auxiliary 0 VALUE={if(v(control)>0,v(control)**2,-v(control))}\n"
      "Bload auxiliary 0 I={.001*v(auxiliary)*v(auxiliary)}\n"
      "Bbridge auxiliary control I={.0001*v(auxiliary,control)}\n");
  auto prepared = internal::PreparedNonlinearPointSolver::Create(system);
  auto ordinary = SparseRealFactorization::Analyze(system.g);
  ASSERT_TRUE(prepared.ok()) << prepared.error().message;
  ASSERT_TRUE(ordinary.ok());
  const auto forcing = std::find_if(system.b_dc.begin(), system.b_dc.end(),
                                    [](double value) { return value != 0.0; });
  ASSERT_NE(forcing, system.b_dc.end());
  const auto source_row =
      static_cast<std::size_t>(forcing - system.b_dc.begin());
  std::vector<double> state(system.g.rows, 0.0);
  std::size_t solved_points = 0;
  for (const double control : {-.4, .2, .6, -.1}) {
    system.b_dc[source_row] = control;
    auto expected =
        RunNonlinearPoint(system, state, ordinary.value().get(), 100);
    auto actual = prepared.value()->Solve(system.g, system.b_dc, state, 100);
    ASSERT_TRUE(expected.ok()) << expected.error().message;
    ASSERT_TRUE(actual.ok()) << actual.error().message;
    CacheEqualPoint(actual.value(), expected.value());
    CacheEqualStatistics(prepared.value()->statistics(),
                         ordinary.value()->statistics());
    state = actual.value().solution;
    EXPECT_NEAR(state[CacheNode(system, "auxiliary")],
                control > 0 ? control * control : -control, 1e-12);
    ASSERT_TRUE(ValidateNonlinearResidual(system, state).ok());
    auto history = prepared.value()->History(state);
    auto checked = BuildDiodeResidualContribution(system, state);
    ASSERT_TRUE(history.ok());
    ASSERT_TRUE(checked.ok());
    CacheEqualState(history.value(), checked.value());
    ++solved_points;
    const auto counts = prepared.value()->expression_cache_statistics();
    EXPECT_EQ(counts.fresh_final_value_evaluations, 3 * solved_points);
    EXPECT_EQ(counts.reused_final_derivatives, 3 * solved_points);
    EXPECT_EQ(counts.fresh_final_evaluations, 0U);
    EXPECT_EQ(counts.publications, solved_points);
    EXPECT_EQ(counts.history_hits, solved_points);
  }
}

TEST(Emi02ExpressionCache,
     InitialAndTrialBoundsRemainMandatoryAtZeroBehavioralResidual) {
  struct Fixture {
    const char *cards;
    double unsafe_root;
    double safe_initial;
    const char *message;
  };
  for (const auto fixture :
       {Fixture{"Bfirst out 0 I={if(v(out)>.5,v(out),6e99*v(out))}\n"
                "Bsecond out 0 I={if(v(out)>.5,0,6e99*v(out))}\n",
                0.0, 1.0, "nonlinear Jacobian contains"},
        Fixture{"Bload out 0 I={if(v(out)<4e99,v(out)-8e99,"
                "2*(v(out)-8e99))}\n",
                8e99, 0.0, "behavioral affine RHS overflow"}}) {
    SCOPED_TRACE(fixture.message);
    auto system = CacheCompile(std::string("Rout out 0 1\n") + fixture.cards);
    ASSERT_EQ(system.g.rows, 1U);
    // Remove the numerical shunt: each fixture has exactly zero residual at
    // its unsafe root, while a different required assembly bound fails.
    system.g.values[0] = 0.0;
    const std::vector<double> unsafe{fixture.unsafe_root};
    auto residual = BuildDiodeResidualContribution(system, unsafe);
    ASSERT_TRUE(residual.ok()) << residual.error().message;
    EXPECT_DOUBLE_EQ(residual.value()[0], 0.0);
    auto full = BuildNonlinearDcLinearization(system, unsafe);
    ASSERT_FALSE(full.ok());
    EXPECT_EQ(full.error().code, ErrorCode::kNonFinite);
    EXPECT_NE(full.error().message.find(fixture.message), std::string::npos);

    auto prepared = internal::PreparedNonlinearPointSolver::Create(system);
    auto ordinary = SparseRealFactorization::Analyze(system.g);
    ASSERT_TRUE(prepared.ok());
    ASSERT_TRUE(ordinary.ok());
    auto initial = prepared.value()->Solve(system.g, system.b_dc, unsafe, 100);
    ASSERT_FALSE(initial.ok());
    EXPECT_EQ(initial.error().code, full.error().code);
    EXPECT_EQ(initial.error().message, full.error().message);

    // From the safe initial state, the first Newton proposal is exactly that
    // unsafe root. Full trial assembly must reject it and backtrack; with one
    // iteration the remaining bounded trial cannot be published as converged.
    auto expected = RunNonlinearPoint(system, {fixture.safe_initial},
                                      ordinary.value().get(), 1);
    auto actual = prepared.value()->Solve(system.g, system.b_dc,
                                          {fixture.safe_initial}, 1);
    ASSERT_FALSE(expected.ok());
    ASSERT_FALSE(actual.ok());
    EXPECT_EQ(actual.error().code, ErrorCode::kNonConvergence);
    EXPECT_EQ(actual.error().code, expected.error().code);
    EXPECT_EQ(actual.error().message, expected.error().message);
    CacheEqualStatistics(prepared.value()->statistics(),
                         ordinary.value()->statistics());
    const auto counts = prepared.value()->expression_cache_statistics();
    EXPECT_GE(counts.fresh_trial_evaluations,
              3 * system.behavioral_descriptors.size());
    EXPECT_EQ(counts.fresh_final_evaluations, 0U);
    EXPECT_EQ(counts.fresh_final_value_evaluations, 0U);
    EXPECT_EQ(counts.reused_final_derivatives, 0U);
    EXPECT_EQ(counts.publications, 0U);
    EXPECT_EQ(counts.retained_entries, 0U);
  }
}

TEST(Emi02ExpressionCache, DirectLinearPointKeepsIndependentTwoIterationTrace) {
  auto system = CacheCompile("Iin 0 out 2m\nRout out 0 1k\n"
                             "Bload out 0 I={.001*v(out)}\n");
  ASSERT_EQ(system.g.values.size(), 1U);
  // State the exact independent equation .001*x + .001*x = .002 without
  // the compiler's extra numerical shunt. Its solution is exactly one volt.
  system.g.values[0] = .001;
  auto prepared = internal::PreparedNonlinearPointSolver::Create(system);
  auto ordinary = SparseRealFactorization::Analyze(system.g);
  ASSERT_TRUE(prepared.ok());
  ASSERT_TRUE(ordinary.ok());
  auto expected = RunNonlinearPoint(system, {0.0}, ordinary.value().get(), 100);
  auto actual = prepared.value()->Solve(system.g, system.b_dc, {0.0}, 100);
  ASSERT_TRUE(expected.ok()) << expected.error().message;
  ASSERT_TRUE(actual.ok()) << actual.error().message;
  CacheEqualPoint(actual.value(), expected.value());
  CacheEqualStatistics(prepared.value()->statistics(),
                       ordinary.value()->statistics());
  ASSERT_EQ(actual.value().iteration_trace.size(), 2U);
  CacheEqualState(actual.value().solution, {1.0});
  const auto &first = actual.value().iteration_trace[0];
  const auto &second = actual.value().iteration_trace[1];
  EXPECT_EQ(first.iteration, 1U);
  EXPECT_EQ(first.strategy, NonlinearStrategy::kDirect);
  EXPECT_DOUBLE_EQ(first.maximum_normalized_update,
                   1.0 /
                       (BehavioralNumericalPolicy::voltage_absolute_tolerance +
                        BehavioralNumericalPolicy::relative_tolerance));
  EXPECT_DOUBLE_EQ(first.maximum_normalized_residual, 0.0);
  EXPECT_FALSE(first.accepted);
  EXPECT_EQ(second.iteration, 2U);
  EXPECT_DOUBLE_EQ(second.maximum_normalized_update, 0.0);
  EXPECT_DOUBLE_EQ(second.maximum_normalized_residual, 0.0);
  EXPECT_TRUE(second.accepted);
  const auto counts = prepared.value()->expression_cache_statistics();
  EXPECT_EQ(counts.fresh_trial_evaluations, 2U);
  EXPECT_EQ(counts.fresh_final_value_evaluations, 1U);
  EXPECT_EQ(counts.reused_final_derivatives, 1U);
  EXPECT_EQ(counts.fresh_final_evaluations, 0U);
  EXPECT_EQ(counts.publications, 1U);
  EXPECT_EQ(prepared.value()->statistics().solves, 3U);
  EXPECT_EQ(prepared.value()->statistics().iterative_refinement_solves, 0U);
}

TEST(Emi02ExpressionCache,
     AssemblyStorageResetsAfterFailuresAndNumericChanges) {
  auto system =
      CacheCompile("Rload out 0 1\n"
                   "Bfirst out 0 I={if(v(out)>.5,6e99*v(out),v(out)*v(out))}\n"
                   "Bsecond out 0 I={if(v(out)>.5,6e99*v(out),v(out))}\n");
  auto prepared = internal::PreparedNonlinearPointSolver::Create(system);
  ASSERT_TRUE(prepared.ok());
  auto factor = SparseRealFactorization::Analyze(system.g);
  ASSERT_TRUE(factor.ok());
  const auto compare = [&](const std::vector<double> &guess,
                           std::size_t iterations) {
    const auto expected =
        RunNonlinearPoint(system, guess, factor.value().get(), iterations);
    const auto actual =
        prepared.value()->Solve(system.g, system.b_dc, guess, iterations);
    EXPECT_EQ(actual.ok(), expected.ok());
    if (actual.ok() && expected.ok()) {
      CacheEqualPoint(actual.value(), expected.value());
    } else if (!actual.ok() && !expected.ok()) {
      EXPECT_EQ(actual.error().code, expected.error().code);
      EXPECT_EQ(actual.error().message, expected.error().message);
    }
    CacheEqualStatistics(prepared.value()->statistics(),
                         factor.value()->statistics());
    const auto storage = prepared.value()->assembly_workspace_statistics();
    EXPECT_EQ(storage.active_buffers, 0U);
    EXPECT_LE(storage.maximum_active_buffers, 2U);
    EXPECT_LE(storage.retained_buffers, 2U);
    return actual.ok();
  };
  system.b_dc[0] = .25;
  ASSERT_TRUE(compare({0}, 100));
  const auto warm = prepared.value()->assembly_workspace_statistics();
  EXPECT_EQ(warm.pattern_initializations, 2U);
  EXPECT_EQ(warm.retained_buffers, 2U);
  const auto publications =
      prepared.value()->expression_cache_statistics().publications;
  // Leave oversized stamped Jacobian/residual contents in an acquired buffer;
  // the following successful point must reset every reused numerical entry.
  EXPECT_FALSE(compare({1}, 100));
  EXPECT_EQ(prepared.value()->expression_cache_statistics().publications,
            publications);
  EXPECT_FALSE(compare({0}, 0));
  const auto before_bad_input =
      prepared.value()->assembly_workspace_statistics();
  EXPECT_FALSE(compare({std::numeric_limits<double>::quiet_NaN()}, 100));
  EXPECT_EQ(prepared.value()->assembly_workspace_statistics().acquisitions,
            before_bad_input.acquisitions);
  for (std::size_t i = 0; i < 8; ++i) {
    system.g.values[0] = 1 + 1e-12 + .05 * i;
    system.b_dc[0] = i % 2 == 0 ? .25 : -.2;
    ASSERT_TRUE(compare({0}, 100));
  }
  const auto after = prepared.value()->assembly_workspace_statistics();
  EXPECT_EQ(after.pattern_initializations, 2U);
  EXPECT_EQ(after.maximum_active_buffers, 2U);
  EXPECT_EQ(after.reused_buffers + 2, after.acquisitions);
  EXPECT_GT(after.reused_buffers, 20U);
  EXPECT_EQ(after.retained_buffers, 2U);
}

TEST(Emi02ExpressionCache, AssemblyStorageIsOwnedPerImmutablePointSolver) {
  auto first = CacheCompile("Rload out 0 1\nBload out 0 I={v(out)*v(out)}\n");
  auto second = CacheCompile("Rload out 0 2\nBload out 0 I={3*v(out)}\n");
  first.b_dc[0] = .5;
  second.b_dc[0] = -.75;
  auto a = internal::PreparedNonlinearPointSolver::Create(first);
  auto b = internal::PreparedNonlinearPointSolver::Create(second);
  ASSERT_TRUE(a.ok());
  ASSERT_TRUE(b.ok());
  auto fa = SparseRealFactorization::Analyze(first.g);
  auto fb = SparseRealFactorization::Analyze(second.g);
  ASSERT_TRUE(fa.ok());
  ASSERT_TRUE(fb.ok());
  for (std::size_t i = 0; i < 4; ++i) {
    const auto expected_a =
        RunNonlinearPoint(first, {0}, fa.value().get(), 100);
    const auto actual_a = a.value()->Solve(first.g, first.b_dc, {0}, 100);
    const auto expected_b =
        RunNonlinearPoint(second, {0}, fb.value().get(), 100);
    const auto actual_b = b.value()->Solve(second.g, second.b_dc, {0}, 100);
    ASSERT_TRUE(expected_a.ok());
    ASSERT_TRUE(actual_a.ok());
    ASSERT_TRUE(expected_b.ok());
    ASSERT_TRUE(actual_b.ok());
    CacheEqualPoint(actual_a.value(), expected_a.value());
    CacheEqualPoint(actual_b.value(), expected_b.value());
    CacheEqualStatistics(a.value()->statistics(), fa.value()->statistics());
    CacheEqualStatistics(b.value()->statistics(), fb.value()->statistics());
  }
  EXPECT_EQ(a.value()->assembly_workspace_statistics().pattern_initializations,
            2U);
  EXPECT_EQ(b.value()->assembly_workspace_statistics().pattern_initializations,
            2U);
  EXPECT_EQ(a.value()->assembly_workspace_statistics().active_buffers, 0U);
  EXPECT_EQ(b.value()->assembly_workspace_statistics().active_buffers, 0U);
}

TEST(Emi02ExpressionCache,
     DerivativeOnlyFailureCannotBecomeAcceptedTrialProof) {
  auto system =
      CacheCompile("Rload out 0 1\nBfirst out 0 I={.25*v(out)}\n"
                   "Bguard out 0 I={if(v(out)<.5,v(out)-1,(v(out)-1)**.5)}\n");
  ASSERT_EQ(system.g.rows, 1U);
  system.g.values[0] = 0.0;
  system.b_dc[0] = .25;
  // At x=1 the guarded expression's exact value is sqrt(0)=0, but its
  // state derivative is singular. The complete physical residual is zero.
  // A value-only acceptance would incorrectly admit this point.
  EXPECT_DOUBLE_EQ(.25 * 1.0 + std::sqrt(1.0 - 1.0) - .25, 0.0);
  const auto full = BuildNonlinearDcLinearization(system, {1.0});
  ASSERT_FALSE(full.ok());
  EXPECT_EQ(full.error().code, ErrorCode::kNonFinite);
  auto prepared = internal::PreparedNonlinearPointSolver::Create(system);
  auto ordinary = SparseRealFactorization::Analyze(system.g);
  ASSERT_TRUE(prepared.ok());
  ASSERT_TRUE(ordinary.ok());
  auto initial = prepared.value()->Solve(system.g, system.b_dc, {1.0}, 100);
  ASSERT_FALSE(initial.ok());
  EXPECT_EQ(initial.error().code, full.error().code);
  EXPECT_EQ(initial.error().message, full.error().message);
  // From zero, the initial linear branch sends the first full Newton trial
  // exactly to the singular point. Bfirst has already yielded a valid result
  // before Bguard fails, so this also leaves an incomplete trial capture.
  auto expected = RunNonlinearPoint(system, {0.0}, ordinary.value().get(), 1);
  auto actual = prepared.value()->Solve(system.g, system.b_dc, {0.0}, 1);
  ASSERT_FALSE(expected.ok());
  ASSERT_FALSE(actual.ok());
  EXPECT_EQ(actual.error().code, ErrorCode::kNonConvergence);
  EXPECT_EQ(actual.error().code, expected.error().code);
  EXPECT_EQ(actual.error().message, expected.error().message);
  CacheEqualStatistics(prepared.value()->statistics(),
                       ordinary.value()->statistics());
  auto counts = prepared.value()->expression_cache_statistics();
  EXPECT_GE(counts.fresh_trial_evaluations, 4U);
  EXPECT_EQ(counts.fresh_final_evaluations, 0U);
  EXPECT_EQ(counts.fresh_final_value_evaluations, 0U);
  EXPECT_EQ(counts.reused_final_derivatives, 0U);
  EXPECT_EQ(counts.publications, 0U);
  EXPECT_EQ(prepared.value()->assembly_workspace_statistics().active_buffers,
            0U);
  // Change only the forcing: .25*x + x - 1 = -.6875 has the nonsingular
  // exact root x=.25. Recovery must rebuild the complete accepted payload,
  // publish only that state, and match the public full-AD result and history.
  system.b_dc[0] = -.6875;
  auto recovered_expected =
      RunNonlinearPoint(system, {0.0}, ordinary.value().get(), 100);
  auto recovered = prepared.value()->Solve(system.g, system.b_dc, {0.0}, 100);
  ASSERT_TRUE(recovered_expected.ok()) << recovered_expected.error().message;
  ASSERT_TRUE(recovered.ok()) << recovered.error().message;
  CacheEqualPoint(recovered.value(), recovered_expected.value());
  CacheEqualState(recovered.value().solution, {.25});
  CacheEqualStatistics(prepared.value()->statistics(),
                       ordinary.value()->statistics());
  auto history = prepared.value()->History(recovered.value().solution);
  auto full_history =
      BuildDiodeResidualContribution(system, recovered.value().solution);
  ASSERT_TRUE(history.ok());
  ASSERT_TRUE(full_history.ok());
  CacheEqualState(history.value(), full_history.value());
  counts = prepared.value()->expression_cache_statistics();
  EXPECT_EQ(counts.fresh_final_evaluations, 0U);
  EXPECT_EQ(counts.fresh_final_value_evaluations, 2U);
  EXPECT_EQ(counts.reused_final_derivatives, 2U);
  EXPECT_EQ(counts.publications, 1U);
  EXPECT_EQ(counts.history_hits, 1U);
}

TEST(Emi02ExpressionCache, CompletedAssemblyAloneAdmitsPreparedLinearInputs) {
  auto system = CacheCompile(
      "Rload out 0 1\nBfirst out 0 I={if(v(out)>.5,v(out),6e99*v(out))}\n"
      "Bsecond out 0 I={if(v(out)>.5,v(out)*v(out),6e99*v(out))}\n");
  auto prepared = internal::PreparedNonlinearPointSolver::Create(system);
  auto checked = SparseRealFactorization::Analyze(system.g);
  ASSERT_TRUE(prepared.ok());
  ASSERT_TRUE(checked.ok());
  const auto compare = [&](const std::vector<double> &guess) {
    auto expected =
        RunNonlinearPoint(system, guess, checked.value().get(), 100);
    auto actual = prepared.value()->Solve(system.g, system.b_dc, guess, 100);
    EXPECT_EQ(actual.ok(), expected.ok());
    if (actual.ok() && expected.ok()) {
      CacheEqualPoint(actual.value(), expected.value());
    } else if (!actual.ok() && !expected.ok()) {
      EXPECT_EQ(actual.error().code, expected.error().code);
      EXPECT_EQ(actual.error().message, expected.error().message);
    }
    CacheEqualStatistics(prepared.value()->statistics(),
                         checked.value()->statistics());
    return actual.ok();
  };
  system.b_dc[0] = 3.0;
  ASSERT_TRUE(compare({1.0}));
  const auto before = prepared.value()->statistics();
  const auto admissions =
      prepared.value()->assembly_workspace_statistics().admitted_linear_solves;
  ASSERT_GT(admissions, 0U);
  // At zero both values and affine terms are zero and the residual is merely
  // -3, but the individually bounded slopes overflow the combined Jacobian.
  // Its specific admission guard must fail before authorizing a solver call.
  auto overflow = BuildNonlinearDcLinearization(system, {0.0});
  ASSERT_FALSE(overflow.ok());
  EXPECT_EQ(overflow.error().code, ErrorCode::kNonFinite);
  EXPECT_NE(overflow.error().message.find("nonlinear Jacobian contains"),
            std::string::npos);
  EXPECT_FALSE(compare({0.0}));
  CacheEqualStatistics(prepared.value()->statistics(), before);
  EXPECT_EQ(
      prepared.value()->assembly_workspace_statistics().admitted_linear_solves,
      admissions);
  // New numeric inputs remain checked before even obtaining an assembly lease.
  auto invalid_matrix = system.g;
  invalid_matrix.values[0] = std::numeric_limits<double>::infinity();
  auto invalid =
      prepared.value()->Solve(invalid_matrix, system.b_dc, {0.0}, 100);
  ASSERT_FALSE(invalid.ok());
  EXPECT_EQ(invalid.error().code, ErrorCode::kNonFinite);
  invalid = prepared.value()->Solve(
      system.g, {std::numeric_limits<double>::quiet_NaN()}, {0.0}, 100);
  ASSERT_FALSE(invalid.ok());
  EXPECT_EQ(invalid.error().code, ErrorCode::kNonFinite);
  CacheEqualStatistics(prepared.value()->statistics(), before);
  EXPECT_EQ(
      prepared.value()->assembly_workspace_statistics().admitted_linear_solves,
      admissions);
  // A valid later point must complete a new assembly and preserve checked KLU
  // results/statistics rather than inheriting an admission flag from scratch.
  system.g.values[0] = 2.0;
  system.b_dc[0] = 4.5;
  ASSERT_TRUE(compare({1.0}));
  EXPECT_GT(
      prepared.value()->assembly_workspace_statistics().admitted_linear_solves,
      admissions);
  EXPECT_EQ(prepared.value()->assembly_workspace_statistics().active_buffers,
            0U);
}

} // namespace
} // namespace ohmnivore
