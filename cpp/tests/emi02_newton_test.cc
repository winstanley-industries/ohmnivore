#include "cpp/tests/google_test.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "ohmnivore/behavioral.h"
#include "ohmnivore/nonlinear.h"
#include "ohmnivore/transient.h"

namespace ohmnivore {
namespace {

TEST(Emi02Newton, SimultaneousDirectStepCrossesAuxiliaryRationalPole) {
  // Independently authored complete circuit. A monotone residual line search
  // from zero gets trapped approaching the pole at control=0.5, even though the
  // finite target is control=1, auxiliary=2. All unknowns remain in the solve.
  auto parsed = ParseBehavioralNetlist(
      "* rational auxiliary equation\nVcontrol control 0 1\n"
      "Eaux auxiliary 0 VALUE={1/(v(control)-0.5)}\n"
      "Rload auxiliary 0 1k\n.OP\n.END\n");
  ASSERT_TRUE(parsed.ok());
  auto compiled = CompileBehavioralMna(parsed.value());
  ASSERT_TRUE(compiled.ok());
  const auto &system = compiled.value();
  auto solved = RunNonlinearDc(system);
  ASSERT_TRUE(solved.ok()) << solved.error().message;
  ASSERT_EQ(solved.value().attempt_trace.size(), 1U);
  EXPECT_EQ(solved.value().attempt_trace[0].strategy,
            NonlinearStrategy::kDirect);
  EXPECT_LE(solved.value().attempt_trace[0].iterations, 4U);
  const auto position = std::find(system.node_names.begin(),
                                  system.node_names.end(), "auxiliary");
  ASSERT_NE(position, system.node_names.end());
  const std::size_t row =
      static_cast<std::size_t>(position - system.node_names.begin());
  EXPECT_NEAR(solved.value().solution[row], 2.0, 1e-12);
  EXPECT_TRUE(ValidateNonlinearResidual(system, solved.value().solution).ok());
  EXPECT_TRUE(solved.value().iteration_trace.back().accepted);
  EXPECT_LE(solved.value().iteration_trace.back().maximum_normalized_update,
            1.0);
  EXPECT_LE(solved.value().iteration_trace.back().maximum_normalized_residual,
            1.0);
}

TEST(Emi02Newton, ReducedBudgetCannotPublishIntermediateAcrossPole) {
  auto parsed = ParseBehavioralNetlist(
      "* bounded rational auxiliary equation\nVcontrol control 0 1\n"
      "Eaux auxiliary 0 VALUE={1/(v(control)-0.5)}\n"
      "Rload auxiliary 0 1k\n.OP\n.END\n");
  ASSERT_TRUE(parsed.ok());
  auto compiled = CompileBehavioralMna(parsed.value());
  ASSERT_TRUE(compiled.ok());
  auto factorization = SparseRealFactorization::Analyze(compiled.value().g);
  ASSERT_TRUE(factorization.ok());
  const std::vector<double> initial(compiled.value().g.rows, 0.0);
  auto failed = RunNonlinearPoint(compiled.value(), initial,
                                  factorization.value().get(), 1);
  ASSERT_FALSE(failed.ok());
  EXPECT_EQ(failed.error().code, ErrorCode::kNonConvergence);
  auto recovered = RunNonlinearPoint(compiled.value(), initial,
                                     factorization.value().get(), 4);
  ASSERT_TRUE(recovered.ok()) << recovered.error().message;
  EXPECT_TRUE(
      ValidateNonlinearResidual(compiled.value(), recovered.value().solution)
          .ok());
}

TEST(Emi02Newton, CapacitorDifferentialConvergenceIsIndependentOfCommonMode) {
  // Independently authored nonlinear RC. The physical capacitor voltage is
  // only a few microvolts. A 400 V common-mode offset must not turn a large
  // differential-state Newton error into an acceptable node-relative update.
  // Icomp cancels exactly the known GMIN current caused by shifting "out";
  // ideal voltage sources supply the other nodes' GMIN currents.
  constexpr double step = 0.1;
  constexpr double capacitance = 1e-6;
  constexpr double conductance = 1e-5;
  constexpr double quadratic = 1000.0;
  constexpr double drive = 1e-3;
  constexpr double linear = capacitance / step + conductance + 1e-12;
  constexpr double forcing = conductance * drive;
  // Stable positive root of a*u^2+b*u-c=0, the independent BE equation.
  const double expected =
      2.0 * forcing /
      (linear + std::sqrt(linear * linear + 4.0 * quadratic * forcing));
  std::vector<double> physical_states;
  for (const int bias : {0, 400}) {
    const std::string deck =
        "* common-mode nonlinear RC\nVcommon ref 0 " + std::to_string(bias) +
        "\nVdrive drive ref DC 0 PWL(0 0 100m 1m)\n"
        "Rdrive drive out 100k\nCstore out ref 1u\n"
        "Bload out ref I={1000*v(out,ref)*v(out,ref)}\n"
        "Icomp 0 out " +
        std::to_string(bias) + "p\n.TRAN 100m 100m\n.END\n";
    auto parsed = ParseBehavioralNetlist(deck);
    ASSERT_TRUE(parsed.ok()) << parsed.error().message;
    auto compiled = CompileBehavioralMna(parsed.value());
    ASSERT_TRUE(compiled.ok()) << compiled.error().message;
    MnaSystem system = compiled.TakeValue();
    auto initial = RunNonlinearDc(system);
    ASSERT_TRUE(initial.ok()) << initial.error().message;
    auto source = BuildTransientRhs(system, step);
    ASSERT_TRUE(source.ok());
    auto rhs = BuildBackwardEulerRhs(system.c, initial.value().solution,
                                     source.value(), step);
    ASSERT_TRUE(rhs.ok());
    auto companion = FormTransientCompanionMatrix(system.g, system.c, step, 1);
    ASSERT_TRUE(companion.ok());
    system.g = companion.TakeValue();
    system.b_dc = rhs.TakeValue();
    ASSERT_TRUE(RemapBehavioralDescriptors(&system).ok());
    auto factorization = SparseRealFactorization::Analyze(system.g);
    ASSERT_TRUE(factorization.ok());
    auto solved = RunNonlinearPoint(system, initial.value().solution,
                                    factorization.value().get(), 100);
    ASSERT_TRUE(solved.ok()) << solved.error().message;
    const auto node = [&](const std::string &name) {
      return static_cast<std::size_t>(
          std::find(system.node_names.begin(), system.node_names.end(), name) -
          system.node_names.begin());
    };
    const double physical = solved.value().solution[node("out")] -
                            solved.value().solution[node("ref")];
    EXPECT_NEAR(physical, expected, 1e-7) << "common-mode bias=" << bias;
    physical_states.push_back(physical);
  }
  ASSERT_EQ(physical_states.size(), 2U);
  EXPECT_NEAR(physical_states[0], physical_states[1], 1e-7);
}

TEST(Emi02Newton, LinearCompanionSolutionDoesNotDependOnNewtonStartingGuess) {
  // Two unit shunts and equal 128 A forcing have common-mode solution 128 V.
  // The capacitor contributes only differential conductance, so changing it
  // cannot change this exact linear solution. Its large C/h deliberately makes
  // J*x-F cancellation sensitive to the arbitrary Newton starting guess.
  auto parsed =
      ParseBehavioralNetlist("* independent stiff symmetric companion\n"
                             "Rleft a 0 1\nRright b 0 1\n"
                             "Ileft 0 a 128\nIright 0 b 128\n"
                             "Ccouple a b 1\nBzero a b I={0}\n.OP\n.END\n");
  ASSERT_TRUE(parsed.ok());
  auto compiled = CompileBehavioralMna(parsed.value());
  ASSERT_TRUE(compiled.ok());
  MnaSystem system = compiled.TakeValue();
  const double step = std::ldexp(1.0, -40);
  auto companion = FormTransientCompanionMatrix(system.g, system.c, step, 1);
  ASSERT_TRUE(companion.ok());
  system.g = companion.TakeValue();
  ASSERT_TRUE(RemapBehavioralDescriptors(&system).ok());
  ASSERT_EQ(system.g.rows, 2U);
  ASSERT_EQ(system.g.values.size(), 4U);
  // The retained FP64 companion has exact diagonal 2^40+1: node GMIN is
  // smaller than half an ulp here. Its exact two-row solution is [128,128].
  EXPECT_DOUBLE_EQ(system.g.values[0], std::ldexp(1.0, 40) + 1.0);
  EXPECT_DOUBLE_EQ(system.g.values[1], -std::ldexp(1.0, 40));
  EXPECT_DOUBLE_EQ(system.g.values[2], -std::ldexp(1.0, 40));
  EXPECT_DOUBLE_EQ(system.g.values[3], std::ldexp(1.0, 40) + 1.0);
  std::vector<double> baseline;
  for (const std::vector<double> &initial :
       std::vector<std::vector<double>>{{0.0, 0.0},
                                        {128.01, 128.01},
                                        {400.1, 399.93},
                                        {-1200.125, -1199.999},
                                        {10000.1, -9999.8},
                                        {0.2, -0.7}}) {
    auto factorization = SparseRealFactorization::Analyze(system.g);
    ASSERT_TRUE(factorization.ok());
    auto solved =
        RunNonlinearPoint(system, initial, factorization.value().get(), 100);
    ASSERT_TRUE(solved.ok()) << solved.error().message;
    for (const double value : solved.value().solution)
      EXPECT_NEAR(value, 128.0, 1e-7);
    if (baseline.empty())
      baseline = solved.value().solution;
    else
      EXPECT_EQ(solved.value().solution, baseline);
  }
}

TEST(Emi02Newton, BoundedHalfStepRecoversAnOverBoundFullProposal) {
  auto parsed = ParseBehavioralNetlist(
      "* independently scaled scalar polynomial\n"
      "Bquadratic out 0 I={(v(out)*1e-99-4)**2-4}\n.OP\n.END\n");
  ASSERT_TRUE(parsed.ok());
  auto compiled = CompileBehavioralMna(parsed.value());
  ASSERT_TRUE(compiled.ok());
  auto system = compiled.TakeValue();
  ASSERT_EQ(system.g.rows, 1U);
  ASSERT_EQ(system.g.values.size(), 1U);
  // Explicitly remove the compiler's numerical shunt to state the independent
  // scalar point equation F(x)=(x*1e-99-4)^2-4, with roots 2e99 and 6e99.
  system.g.values[0] = 0.0;
  const std::vector<double> initial{4.3e99};
  const double scaled_initial = initial[0] * 1e-99;
  const double delta = -((scaled_initial - 4) * (scaled_initial - 4) - 4) /
                       (2 * (scaled_initial - 4) * 1e-99);
  ASSERT_LT(std::abs(delta), kNonlinearMaximumMagnitude);
  ASSERT_GT(initial[0] + delta, kNonlinearMaximumMagnitude);
  ASSERT_LT(initial[0] + delta * .5, kNonlinearMaximumMagnitude);

  auto factorization = SparseRealFactorization::Analyze(system.g);
  ASSERT_TRUE(factorization.ok());
  auto stopped =
      RunNonlinearPoint(system, initial, factorization.value().get(), 1);
  ASSERT_FALSE(stopped.ok());
  // The rejected full proposal must reach a bounded half-step trial. The
  // native junction limiter previously returned kNonFinite before backtracking.
  EXPECT_EQ(stopped.error().code, ErrorCode::kNonConvergence);
  auto solved =
      RunNonlinearPoint(system, initial, factorization.value().get(), 100);
  ASSERT_TRUE(solved.ok()) << solved.error().message;
  ASSERT_EQ(solved.value().solution.size(), 1U);
  EXPECT_NEAR(solved.value().solution[0] / 1e99, 6.0, 1e-10);
  EXPECT_LE(std::abs(solved.value().solution[0]), kNonlinearMaximumMagnitude);
  EXPECT_TRUE(ValidateNonlinearResidual(system, solved.value().solution).ok());
  EXPECT_EQ(initial, (std::vector<double>{4.3e99}));
}

TEST(Emi02Newton, LinearRowMagnitudeBudgetSurvivesCancellationAndUnderflow) {
  // Exercise the public assembly path with independently specified row
  // equations. A small final residual must never conceal an excessive sum of
  // absolute terms; source magnitude and extra GMIN consume the same budget.
  constexpr double bound = kNonlinearMaximumMagnitude;
  const double tiny = std::numeric_limits<double>::denorm_min();
  struct RowCase {
    const char *name;
    std::array<double, 3> coefficients;
    std::array<double, 3> state;
    double source;
    double gmin;
    bool accepted;
    double residual;
  };
  const std::vector<RowCase> cases{
      {"signed zeros",
       {0.0, -0.0, 0.0},
       {-0.0, 0.0, -1.0},
       -0.0,
       0.0,
       true,
       0.0},
      {"inclusive bound",
       {bound, 0.0, 0.0},
       {1.0, 0.0, 0.0},
       0.0,
       0.0,
       true,
       bound},
      {"exact bounded cancellation",
       {bound / 2, -bound / 2, 0.0},
       {1.0, 1.0, 0.0},
       0.0,
       0.0,
       true,
       0.0},
      {"rounded underflow",
       {tiny, -tiny, 0.0},
       {0.5, 0.5, 0.0},
       0.0,
       0.0,
       true,
       0.0},
      {"cancelled excessive magnitudes",
       {0.6 * bound, -0.6 * bound, 0.0},
       {1.0, 1.0, 0.0},
       0.0,
       0.0,
       false,
       0.0},
      {"excessive intermediate product",
       {0.6 * bound, 0.6 * bound, -0.6 * bound},
       {1.0, 1.0, 1.0},
       0.0,
       0.0,
       false,
       0.0},
      {"excessive single term",
       {bound, 0.0, 0.0},
       {2.0, 0.0, 0.0},
       0.0,
       0.0,
       false,
       0.0},
      {"large finite multiplication",
       {bound, 0.0, 0.0},
       {bound, 0.0, 0.0},
       0.0,
       0.0,
       false,
       0.0},
      {"source participates in budget",
       {bound / 2, -bound / 2, 0.0},
       {1.0, 1.0, 0.0},
       0.1 * bound,
       0.0,
       false,
       0.0},
      {"bounded GMIN cancellation",
       {-0.4 * bound, 0.0, 0.0},
       {1.0, 0.0, 0.0},
       0.0,
       0.4 * bound,
       true,
       0.0},
      {"excessive GMIN cancellation",
       {-0.6 * bound, 0.0, 0.0},
       {1.0, 0.0, 0.0},
       0.0,
       0.6 * bound,
       false,
       0.0},
      {"excessive GMIN term",
       {0.0, 0.0, 0.0},
       {2.0, 0.0, 0.0},
       0.0,
       bound,
       false,
       0.0},
  };
  auto parsed = ParseBehavioralNetlist(
      "* independent row-budget fixture\nRa a 0 1\nRb b 0 1\nRc c 0 1\n"
      "Bzero a 0 I={0}\n.OP\n.END\n");
  ASSERT_TRUE(parsed.ok());
  auto compiled = CompileBehavioralMna(parsed.value());
  ASSERT_TRUE(compiled.ok());
  auto system = compiled.TakeValue();
  system.g = {.rows = 3,
              .columns = 3,
              .values = {0.0, 0.0, 0.0, 0.0, 0.0},
              .column_indices = {0, 1, 2, 1, 2},
              .row_offsets = {0, 3, 4, 5}};
  ASSERT_TRUE(RemapBehavioralDescriptors(&system).ok());
  for (const auto &row : cases) {
    SCOPED_TRACE(row.name);
    std::copy(row.coefficients.begin(), row.coefficients.end(),
              system.g.values.begin());
    system.b_dc = {row.source, 0.0, 0.0};
    const std::vector<double> state(row.state.begin(), row.state.end());
    auto result = BuildNonlinearDcLinearization(system, state, 1.0, row.gmin);
    ASSERT_EQ(result.ok(), row.accepted)
        << (result.ok() ? "unexpected success" : result.error().message);
    if (row.accepted) {
      EXPECT_DOUBLE_EQ(result.value().residual[0], row.residual);
    } else {
      EXPECT_EQ(result.error().code, ErrorCode::kNonFinite);
      EXPECT_NE(result.error().message.find(row.gmin == 0.0
                                                ? "linear-row evaluation"
                                                : "extra GMIN evaluation"),
                std::string::npos);
    }
  }
}

} // namespace
} // namespace ohmnivore
