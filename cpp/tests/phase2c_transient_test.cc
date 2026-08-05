#include "google_test.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ohmnivore/compiler.h"
#include "ohmnivore/ir.h"
#include "ohmnivore/transient.h"
#include "ohmnivore/waveform.h"

namespace ohmnivore {
namespace {

[[nodiscard]] MnaSystem Compile(std::vector<Component> components) {
  Result<MnaSystem> compiled = CompileMna(Circuit{
      .components = std::move(components),
      .analyses = {},
  });
  if (!compiled.ok()) {
    ADD_FAILURE() << compiled.error().message;
    return MnaSystem{};
  }
  return compiled.TakeValue();
}

[[nodiscard]] std::size_t NodeIndex(const MnaSystem &system,
                                    const std::string &name) {
  const auto found =
      std::find(system.node_names.begin(), system.node_names.end(), name);
  EXPECT_NE(found, system.node_names.end());
  return static_cast<std::size_t>(found - system.node_names.begin());
}

[[nodiscard]] std::size_t BranchIndex(const MnaSystem &system,
                                      const std::string &name) {
  const auto found =
      std::find(system.branch_names.begin(), system.branch_names.end(), name);
  EXPECT_NE(found, system.branch_names.end());
  return system.node_names.size() +
         static_cast<std::size_t>(found - system.branch_names.begin());
}

TEST(Phase2CCompilerTest, BuildsExactTransientRhsWithCanonicalSourceSigns) {
  MnaSystem system = Compile({
      VoltageSource{
          .name = "V1",
          .positive_node = "a",
          .negative_node = "0",
          .dc_volts = std::nullopt,
          .ac = std::nullopt,
          .transient =
              PwlWaveform{.time_value_pairs = {{0.0, 1.0}, {1.0, 3.0}}},
      },
      CurrentSource{
          .name = "I1",
          .positive_node = "a",
          .negative_node = "b",
          .dc_amperes = 7.0,
          .ac = AcSourceSpecification{.magnitude = 2.0, .phase_degrees = 90.0},
          .transient =
              PwlWaveform{.time_value_pairs = {{0.0, 2.0}, {1.0, 4.0}}},
      },
      Resistor{.name = "R1",
               .positive_node = "a",
               .negative_node = "0",
               .resistance_ohms = 1.0},
      Resistor{.name = "R2",
               .positive_node = "b",
               .negative_node = "0",
               .resistance_ohms = 1.0},
  });

  ASSERT_EQ(system.transient_sources.size(), 2U);
  Result<std::vector<double>> rhs = BuildTransientRhs(system, 0.5);
  ASSERT_TRUE(rhs.ok()) << rhs.error().message;
  ASSERT_EQ(rhs.value().size(), 3U);
  EXPECT_DOUBLE_EQ(rhs.value()[NodeIndex(system, "a")], -3.0);
  EXPECT_DOUBLE_EQ(rhs.value()[NodeIndex(system, "b")], 3.0);
  EXPECT_DOUBLE_EQ(rhs.value()[BranchIndex(system, "V1")], 2.0);
}

TEST(Phase2CCompilerTest, RejectsInvalidDirectIrWaveform) {
  Result<MnaSystem> compiled = CompileMna(Circuit{
      .components = {VoltageSource{
          .name = "V1",
          .positive_node = "1",
          .negative_node = "0",
          .dc_volts = std::nullopt,
          .ac = std::nullopt,
          .transient =
              PwlWaveform{.time_value_pairs = {{1.0, 0.0}, {0.5, 1.0}}},
      }},
      .analyses = {},
  });
  ASSERT_FALSE(compiled.ok());
  EXPECT_EQ(compiled.error().code, ErrorCode::kCompile);
}

TEST(Phase2CCompanionTest, MergesIndependentPatternsAndBuildsExactRhs) {
  const CsrMatrix g{
      .rows = 2,
      .columns = 2,
      .values = {2.0, 1.0, 5.0},
      .column_indices = {0, 0, 1},
      .row_offsets = {0, 1, 3},
  };
  const CsrMatrix c{
      .rows = 2,
      .columns = 2,
      .values = {3.0, 4.0},
      .column_indices = {1, 1},
      .row_offsets = {0, 1, 2},
  };

  Result<CsrMatrix> be = FormTransientCompanionMatrix(g, c, 2.0, 1.0);
  ASSERT_TRUE(be.ok()) << be.error().message;
  EXPECT_EQ(be.value().column_indices, (std::vector<std::size_t>{0, 1, 0, 1}));
  EXPECT_EQ(be.value().row_offsets, (std::vector<std::size_t>{0, 2, 4}));
  EXPECT_EQ(be.value().values, (std::vector<double>{2.0, 1.5, 1.0, 7.0}));

  Result<CsrMatrix> trap = FormTransientCompanionMatrix(g, c, 2.0, 2.0);
  ASSERT_TRUE(trap.ok()) << trap.error().message;
  EXPECT_EQ(trap.value().values, (std::vector<double>{2.0, 3.0, 1.0, 9.0}));

  const std::vector<double> previous_state = {2.0, 3.0};
  const std::vector<double> previous_rhs = {5.0, 6.0};
  const std::vector<double> current_rhs = {7.0, 11.0};
  Result<std::vector<double>> be_rhs =
      BuildBackwardEulerRhs(c, previous_state, current_rhs, 2.0);
  ASSERT_TRUE(be_rhs.ok()) << be_rhs.error().message;
  EXPECT_EQ(be_rhs.value(), (std::vector<double>{11.5, 17.0}));
  Result<std::vector<double>> trap_rhs =
      BuildTrapezoidalRhs(g, c, previous_state, previous_rhs, current_rhs, 2.0);
  ASSERT_TRUE(trap_rhs.ok()) << trap_rhs.error().message;
  EXPECT_EQ(trap_rhs.value(), (std::vector<double>{17.0, 12.0}));
}

TEST(Phase2CUicTest, SupportsRedundantCapsAndRejectsInconsistentConstraints) {
  MnaSystem redundant = Compile({
      VoltageSource{.name = "V1",
                    .positive_node = "in",
                    .negative_node = "0",
                    .dc_volts = 5.0,
                    .ac = std::nullopt,
                    .transient = std::nullopt},
      Resistor{.name = "R1",
               .positive_node = "in",
               .negative_node = "out",
               .resistance_ohms = 1000.0},
      Capacitor{.name = "C1",
                .positive_node = "out",
                .negative_node = "0",
                .capacitance_farads = 1e-6},
      Capacitor{.name = "C2",
                .positive_node = "out",
                .negative_node = "0",
                .capacitance_farads = 2e-6},
  });
  Result<std::vector<double>> uic = BuildTransientInitialState(redundant, true);
  ASSERT_TRUE(uic.ok()) << uic.error().message;
  EXPECT_DOUBLE_EQ(uic.value()[NodeIndex(redundant, "out")], 0.0);
  EXPECT_DOUBLE_EQ(uic.value()[NodeIndex(redundant, "in")], 5.0);

  MnaSystem inconsistent = Compile({
      VoltageSource{.name = "V1",
                    .positive_node = "forced",
                    .negative_node = "0",
                    .dc_volts = 5.0,
                    .ac = std::nullopt,
                    .transient = std::nullopt},
      Capacitor{.name = "C1",
                .positive_node = "forced",
                .negative_node = "0",
                .capacitance_farads = 1e-6},
  });
  Result<std::vector<double>> rejected =
      BuildTransientInitialState(inconsistent, true);
  ASSERT_FALSE(rejected.ok());
  EXPECT_EQ(rejected.error().code, ErrorCode::kSolve);
}

TEST(Phase2CUicTest, DistinguishesDcAndZeroReactiveStateInitialization) {
  MnaSystem rc = Compile({
      VoltageSource{.name = "V1",
                    .positive_node = "in",
                    .negative_node = "0",
                    .dc_volts = 5.0,
                    .ac = std::nullopt,
                    .transient = std::nullopt},
      Resistor{.name = "R1",
               .positive_node = "in",
               .negative_node = "out",
               .resistance_ohms = 1000.0},
      Capacitor{.name = "C1",
                .positive_node = "out",
                .negative_node = "0",
                .capacitance_farads = 1e-6},
  });
  Result<std::vector<double>> dc = BuildTransientInitialState(rc, false);
  Result<std::vector<double>> uic = BuildTransientInitialState(rc, true);
  ASSERT_TRUE(dc.ok()) << dc.error().message;
  ASSERT_TRUE(uic.ok()) << uic.error().message;
  EXPECT_NEAR(dc.value()[NodeIndex(rc, "out")], 5.0, 2e-8);
  EXPECT_DOUBLE_EQ(uic.value()[NodeIndex(rc, "out")], 0.0);

  MnaSystem rl = Compile({
      VoltageSource{.name = "V1",
                    .positive_node = "in",
                    .negative_node = "0",
                    .dc_volts = 5.0,
                    .ac = std::nullopt,
                    .transient = std::nullopt},
      Resistor{.name = "R1",
               .positive_node = "in",
               .negative_node = "out",
               .resistance_ohms = 100.0},
      Inductor{.name = "L1",
               .positive_node = "out",
               .negative_node = "0",
               .inductance_henries = 10e-3},
  });
  Result<std::vector<double>> rl_uic = BuildTransientInitialState(rl, true);
  ASSERT_TRUE(rl_uic.ok()) << rl_uic.error().message;
  EXPECT_DOUBLE_EQ(rl_uic.value()[BranchIndex(rl, "L1")], 0.0);
}

TEST(Phase2CUicTest, AcceptsCapacitorConstraintDuplicatedByZeroVoltSource) {
  MnaSystem system = Compile({
      VoltageSource{.name = "V1",
                    .positive_node = "n",
                    .negative_node = "0",
                    .dc_volts = 0.0,
                    .ac = std::nullopt,
                    .transient = std::nullopt},
      Capacitor{.name = "C1",
                .positive_node = "n",
                .negative_node = "0",
                .capacitance_farads = 1.0},
  });
  Result<std::vector<double>> state = BuildTransientInitialState(system, true);
  ASSERT_TRUE(state.ok()) << state.error().message;
  EXPECT_DOUBLE_EQ(state.value()[NodeIndex(system, "n")], 0.0);
  EXPECT_DOUBLE_EQ(state.value()[BranchIndex(system, "V1")], 0.0);
}

TEST(Phase2CTransientTest, HasDeterministicRejectionAndRecoveryTrace) {
  MnaSystem system = Compile({
      VoltageSource{.name = "V1",
                    .positive_node = "in",
                    .negative_node = "0",
                    .dc_volts = 5.0,
                    .ac = std::nullopt,
                    .transient = std::nullopt},
      Resistor{.name = "R1",
               .positive_node = "in",
               .negative_node = "out",
               .resistance_ohms = 1000.0},
      Capacitor{.name = "C1",
                .positive_node = "out",
                .negative_node = "0",
                .capacitance_farads = 1e-6},
  });
  Result<TransientResult> result = RunTransientAnalysis(
      system, TranAnalysis{.time_step_seconds = 5e-4,
                           .stop_time_seconds = 2e-3,
                           .start_time_seconds = 0.0,
                           .use_initial_conditions = true});
  ASSERT_TRUE(result.ok()) << result.error().message;
  ASSERT_GE(result.value().step_trace.size(), 3U);
  EXPECT_EQ(result.value().step_trace[0].method,
            TransientIntegrationMethod::kBackwardEuler);
  EXPECT_FALSE(result.value().step_trace[0].accepted);
  const auto accepted_be = std::find_if(
      result.value().step_trace.begin() + 1, result.value().step_trace.end(),
      [](const TransientStepRecord &record) {
        return record.method == TransientIntegrationMethod::kBackwardEuler &&
               record.accepted;
      });
  ASSERT_NE(accepted_be, result.value().step_trace.end());
  const auto trap = std::find_if(
      accepted_be + 1, result.value().step_trace.end(),
      [](const TransientStepRecord &record) {
        return record.method == TransientIntegrationMethod::kTrapezoidal;
      });
  ASSERT_NE(trap, result.value().step_trace.end());

  Result<TransientResult> repeated = RunTransientAnalysis(
      system, TranAnalysis{.time_step_seconds = 5e-4,
                           .stop_time_seconds = 2e-3,
                           .start_time_seconds = 0.0,
                           .use_initial_conditions = true});
  ASSERT_TRUE(repeated.ok()) << repeated.error().message;
  ASSERT_EQ(repeated.value().step_trace.size(),
            result.value().step_trace.size());
  for (std::size_t index = 0; index < result.value().step_trace.size();
       ++index) {
    const TransientStepRecord &first = result.value().step_trace[index];
    const TransientStepRecord &second = repeated.value().step_trace[index];
    EXPECT_DOUBLE_EQ(first.start_time_seconds, second.start_time_seconds);
    EXPECT_DOUBLE_EQ(first.end_time_seconds, second.end_time_seconds);
    EXPECT_DOUBLE_EQ(first.step_size_seconds, second.step_size_seconds);
    EXPECT_EQ(first.method, second.method);
    EXPECT_EQ(first.accepted, second.accepted);
    EXPECT_DOUBLE_EQ(first.normalized_local_error,
                     second.normalized_local_error);
    EXPECT_EQ(first.landed_on_hard_point, second.landed_on_hard_point);
  }
}

TEST(Phase2CTransientTest, LandsExactlyOnOutputAndWaveformHardPoints) {
  const PulseWaveform pulse{
      .initial_value = 0.0,
      .pulsed_value = 5.0,
      .delay_seconds = 0.35,
      .rise_time_seconds = 0.05,
      .fall_time_seconds = 0.05,
      .pulse_width_seconds = 0.20,
      .period_seconds = 1.0,
  };
  MnaSystem system = Compile({
      VoltageSource{.name = "V1",
                    .positive_node = "out",
                    .negative_node = "0",
                    .dc_volts = std::nullopt,
                    .ac = std::nullopt,
                    .transient = pulse},
      Resistor{.name = "R1",
               .positive_node = "out",
               .negative_node = "0",
               .resistance_ohms = 1000.0},
  });
  Result<TransientResult> result = RunTransientAnalysis(
      system, TranAnalysis{.time_step_seconds = 0.2,
                           .stop_time_seconds = 0.8,
                           .start_time_seconds = 0.13,
                           .use_initial_conditions = false});
  ASSERT_TRUE(result.ok()) << result.error().message;
  EXPECT_DOUBLE_EQ(result.value().times_seconds.front(), 0.13);
  EXPECT_DOUBLE_EQ(result.value().times_seconds.back(), 0.8);

  Result<std::vector<double>> breakpoints =
      CollectTransientWaveformBreakpoints(pulse, 0.8);
  ASSERT_TRUE(breakpoints.ok()) << breakpoints.error().message;
  for (double breakpoint : breakpoints.value()) {
    if (breakpoint == 0.0) {
      continue;
    }
    const auto found = std::find_if(
        result.value().step_trace.begin(), result.value().step_trace.end(),
        [breakpoint](const TransientStepRecord &record) {
          return record.accepted && record.end_time_seconds == breakpoint;
        });
    ASSERT_NE(found, result.value().step_trace.end());
    EXPECT_EQ(found->method, TransientIntegrationMethod::kBackwardEuler);
  }
  for (const TransientStepRecord &record : result.value().step_trace) {
    EXPECT_LE(record.step_size_seconds, 0.2);
  }
}

TEST(Phase2CTransientTest, AllowsMandatoryHardPointBelowMinimumStep) {
  MnaSystem system = Compile({
      VoltageSource{.name = "V1",
                    .positive_node = "out",
                    .negative_node = "0",
                    .dc_volts = 1.0,
                    .ac = std::nullopt,
                    .transient = std::nullopt},
      Resistor{.name = "R1",
               .positive_node = "out",
               .negative_node = "0",
               .resistance_ohms = 1.0},
  });
  Result<TransientResult> result = RunTransientAnalysis(
      system, TranAnalysis{.time_step_seconds = 1.0,
                           .stop_time_seconds = 2e-5,
                           .start_time_seconds = 1e-5,
                           .use_initial_conditions = false});
  ASSERT_TRUE(result.ok()) << result.error().message;
  EXPECT_EQ(result.value().times_seconds, (std::vector<double>{1e-5, 2e-5}));
  ASSERT_EQ(result.value().step_trace.size(), 2U);
  EXPECT_LT(result.value().step_trace[0].step_size_seconds, 1e-4);
}

TEST(Phase2CTransientTest, PreservesReactiveStateAtDiscontinuousPulseEdges) {
  const auto make_system = [](double delay, double width) {
    return Compile({
        VoltageSource{
            .name = "V1",
            .positive_node = "in",
            .negative_node = "0",
            .dc_volts = std::nullopt,
            .ac = std::nullopt,
            .transient = PulseWaveform{.initial_value = 0.0,
                                       .pulsed_value = 1.0,
                                       .delay_seconds = delay,
                                       .rise_time_seconds = 0.0,
                                       .fall_time_seconds = 0.0,
                                       .pulse_width_seconds = width,
                                       .period_seconds = 2.0},
        },
        Resistor{.name = "R1",
                 .positive_node = "in",
                 .negative_node = "out",
                 .resistance_ohms = 1.0},
        Capacitor{.name = "C1",
                  .positive_node = "out",
                  .negative_node = "0",
                  .capacitance_farads = 1.0},
    });
  };

  MnaSystem rising = make_system(0.5, 1.0);
  Result<TransientResult> rise = RunTransientAnalysis(
      rising, TranAnalysis{.time_step_seconds = 0.5,
                           .stop_time_seconds = 0.5,
                           .start_time_seconds = 0.0,
                           .use_initial_conditions = true});
  ASSERT_TRUE(rise.ok()) << rise.error().message;
  EXPECT_DOUBLE_EQ(rise.value().states.back()[NodeIndex(rising, "in")], 1.0);
  EXPECT_DOUBLE_EQ(rise.value().states.back()[NodeIndex(rising, "out")], 0.0);

  MnaSystem falling = make_system(0.0, 0.5);
  Result<TransientResult> fall = RunTransientAnalysis(
      falling, TranAnalysis{.time_step_seconds = 0.5,
                            .stop_time_seconds = 0.5,
                            .start_time_seconds = 0.0,
                            .use_initial_conditions = true});
  ASSERT_TRUE(fall.ok()) << fall.error().message;
  EXPECT_DOUBLE_EQ(fall.value().states.front()[NodeIndex(falling, "in")], 1.0);
  EXPECT_DOUBLE_EQ(fall.value().states.back()[NodeIndex(falling, "in")], 0.0);
  EXPECT_GT(fall.value().states.back()[NodeIndex(falling, "out")], 0.0);
}

TEST(Phase2CTransientTest, ErrorControlsInitialBackwardEulerStep) {
  MnaSystem system = Compile({
      VoltageSource{.name = "V1",
                    .positive_node = "in",
                    .negative_node = "0",
                    .dc_volts = 1.0,
                    .ac = std::nullopt,
                    .transient = std::nullopt},
      Resistor{.name = "R1",
               .positive_node = "in",
               .negative_node = "out",
               .resistance_ohms = 1.0},
      Capacitor{.name = "C1",
                .positive_node = "out",
                .negative_node = "0",
                .capacitance_farads = 1.0},
  });
  Result<TransientResult> result = RunTransientAnalysis(
      system, TranAnalysis{.time_step_seconds = 1.0,
                           .stop_time_seconds = 1.0,
                           .start_time_seconds = 0.0,
                           .use_initial_conditions = true});
  ASSERT_TRUE(result.ok()) << result.error().message;
  ASSERT_FALSE(result.value().step_trace.empty());
  EXPECT_EQ(result.value().step_trace.front().method,
            TransientIntegrationMethod::kBackwardEuler);
  EXPECT_FALSE(result.value().step_trace.front().accepted);
  EXPECT_NEAR(result.value().states.back()[NodeIndex(system, "out")],
              1.0 - std::exp(-1.0), 2e-3);
}

TEST(Phase2CTransientTest, EnforcesMinimumAttemptAndAcceptedStepLimits) {
  MnaSystem dynamic = Compile({
      VoltageSource{.name = "V1",
                    .positive_node = "in",
                    .negative_node = "0",
                    .dc_volts = 1.0,
                    .ac = std::nullopt,
                    .transient = std::nullopt},
      Resistor{.name = "R1",
               .positive_node = "in",
               .negative_node = "out",
               .resistance_ohms = 1.0},
      Capacitor{.name = "C1",
                .positive_node = "out",
                .negative_node = "0",
                .capacitance_farads = 1.0},
  });
  Result<TransientResult> below_minimum = RunTransientAnalysis(
      dynamic,
      TranAnalysis{.time_step_seconds = 1.0,
                   .stop_time_seconds = 1.0,
                   .start_time_seconds = 0.0,
                   .use_initial_conditions = true},
      TransientExecutionLimits{.maximum_accepted_steps = 2,
                               .maximum_step_attempts = 2,
                               .minimum_step_divisor = 1.0});
  ASSERT_FALSE(below_minimum.ok());
  EXPECT_NE(below_minimum.error().message.find("below h_min"),
            std::string::npos);

  MnaSystem static_system = Compile({
      VoltageSource{.name = "V1",
                    .positive_node = "out",
                    .negative_node = "0",
                    .dc_volts = 1.0,
                    .ac = std::nullopt,
                    .transient = std::nullopt},
      Resistor{.name = "R1",
               .positive_node = "out",
               .negative_node = "0",
               .resistance_ohms = 1.0},
  });
  const TranAnalysis two_steps{.time_step_seconds = 1.0,
                               .stop_time_seconds = 2.0,
                               .start_time_seconds = 0.0,
                               .use_initial_conditions = false};
  Result<TransientResult> accepted_limit = RunTransientAnalysis(
      static_system, two_steps,
      TransientExecutionLimits{.maximum_accepted_steps = 1,
                               .maximum_step_attempts = 2,
                               .minimum_step_divisor = 10'000.0});
  ASSERT_FALSE(accepted_limit.ok());
  EXPECT_NE(accepted_limit.error().message.find("accepted timesteps"),
            std::string::npos);

  Result<TransientResult> attempt_limit = RunTransientAnalysis(
      static_system, two_steps,
      TransientExecutionLimits{.maximum_accepted_steps = 1,
                               .maximum_step_attempts = 1,
                               .minimum_step_divisor = 10'000.0});
  ASSERT_FALSE(attempt_limit.ok());
  EXPECT_NE(attempt_limit.error().message.find("timestep attempts"),
            std::string::npos);
}

TEST(Phase2CTransientTest, RejectsUnrepresentableAndNonFiniteWork) {
  MnaSystem system = Compile({
      VoltageSource{.name = "V1",
                    .positive_node = "out",
                    .negative_node = "0",
                    .dc_volts = 1.0,
                    .ac = std::nullopt,
                    .transient = std::nullopt},
      Resistor{.name = "R1",
               .positive_node = "out",
               .negative_node = "0",
               .resistance_ohms = 1.0},
  });
  Result<TransientResult> too_small = RunTransientAnalysis(
      system, TranAnalysis{.time_step_seconds =
                               std::numeric_limits<double>::denorm_min(),
                           .stop_time_seconds = 1.0,
                           .start_time_seconds = 0.0,
                           .use_initial_conditions = false});
  ASSERT_FALSE(too_small.ok());
  EXPECT_EQ(too_small.error().code, ErrorCode::kSolve);

  Result<TransientResult> invalid_maximum = RunTransientAnalysis(
      system,
      TranAnalysis{.time_step_seconds = std::numeric_limits<double>::infinity(),
                   .stop_time_seconds = 1.0,
                   .start_time_seconds = 0.0,
                   .use_initial_conditions = false});
  ASSERT_FALSE(invalid_maximum.ok());
  EXPECT_EQ(invalid_maximum.error().code, ErrorCode::kSolve);

  CsrMatrix non_finite{
      .rows = 1,
      .columns = 1,
      .values = {std::numeric_limits<double>::infinity()},
      .column_indices = {0},
      .row_offsets = {0, 1},
  };
  CsrMatrix empty{
      .rows = 1,
      .columns = 1,
      .values = {},
      .column_indices = {},
      .row_offsets = {0, 0},
  };
  Result<CsrMatrix> rejected =
      FormTransientCompanionMatrix(non_finite, empty, 1.0, 1.0);
  ASSERT_FALSE(rejected.ok());
  EXPECT_EQ(rejected.error().code, ErrorCode::kSolve);
}

TEST(Phase2CTransientTest, MatchesRcAndRlStepResponsesAndBranchSigns) {
  MnaSystem rc = Compile({
      VoltageSource{.name = "V1",
                    .positive_node = "in",
                    .negative_node = "0",
                    .dc_volts = 5.0,
                    .ac = std::nullopt,
                    .transient = std::nullopt},
      Resistor{.name = "R1",
               .positive_node = "in",
               .negative_node = "out",
               .resistance_ohms = 1000.0},
      Capacitor{.name = "C1",
                .positive_node = "out",
                .negative_node = "0",
                .capacitance_farads = 1e-6},
  });
  Result<TransientResult> rc_result =
      RunTransientAnalysis(rc, TranAnalysis{.time_step_seconds = 1e-5,
                                            .stop_time_seconds = 1e-3,
                                            .start_time_seconds = 0.0,
                                            .use_initial_conditions = true});
  ASSERT_TRUE(rc_result.ok()) << rc_result.error().message;
  const double expected_rc = 5.0 * (1.0 - std::exp(-1.0));
  EXPECT_NEAR(rc_result.value().states.back()[NodeIndex(rc, "out")],
              expected_rc, 2e-3);
  EXPECT_LT(rc_result.value().states.front()[BranchIndex(rc, "V1")], 0.0);

  MnaSystem rl = Compile({
      VoltageSource{.name = "V1",
                    .positive_node = "in",
                    .negative_node = "0",
                    .dc_volts = 5.0,
                    .ac = std::nullopt,
                    .transient = std::nullopt},
      Resistor{.name = "R1",
               .positive_node = "in",
               .negative_node = "out",
               .resistance_ohms = 100.0},
      Inductor{.name = "L1",
               .positive_node = "out",
               .negative_node = "0",
               .inductance_henries = 10e-3},
  });
  Result<TransientResult> rl_result =
      RunTransientAnalysis(rl, TranAnalysis{.time_step_seconds = 1e-6,
                                            .stop_time_seconds = 1e-4,
                                            .start_time_seconds = 0.0,
                                            .use_initial_conditions = true});
  ASSERT_TRUE(rl_result.ok()) << rl_result.error().message;
  const double expected_rl = 0.05 * (1.0 - std::exp(-1.0));
  EXPECT_NEAR(rl_result.value().states.back()[BranchIndex(rl, "L1")],
              expected_rl, 2e-5);
  EXPECT_LT(rl_result.value().states.back()[BranchIndex(rl, "V1")], 0.0);
}

TEST(Phase2CTransientTest, MatchesRcDischargeFromDcOperatingPoint) {
  MnaSystem rc = Compile({
      VoltageSource{
          .name = "V1",
          .positive_node = "in",
          .negative_node = "0",
          .dc_volts = 5.0,
          .ac = std::nullopt,
          .transient = PwlWaveform{.time_value_pairs = {{0.0, 0.0}}},
      },
      Resistor{.name = "R1",
               .positive_node = "in",
               .negative_node = "out",
               .resistance_ohms = 1000.0},
      Capacitor{.name = "C1",
                .positive_node = "out",
                .negative_node = "0",
                .capacitance_farads = 1e-6},
  });
  Result<TransientResult> result =
      RunTransientAnalysis(rc, TranAnalysis{.time_step_seconds = 1e-5,
                                            .stop_time_seconds = 1e-3,
                                            .start_time_seconds = 0.0,
                                            .use_initial_conditions = false});
  ASSERT_TRUE(result.ok()) << result.error().message;
  const double initial = 5e-3 / (1e-3 + kGminSiemens);
  const double time_constant = 1e-6 / (1e-3 + kGminSiemens);
  EXPECT_NEAR(result.value().states.back()[NodeIndex(rc, "out")],
              initial * std::exp(-1e-3 / time_constant), 2e-3);
}

TEST(Phase2CTransientTest, MatchesRepresentativeUnderdampedRlcResponse) {
  constexpr double resistance = 10.0;
  constexpr double inductance = 1e-3;
  constexpr double capacitance = 1e-6;
  constexpr double stop = 50e-6;
  MnaSystem rlc = Compile({
      VoltageSource{.name = "V1",
                    .positive_node = "in",
                    .negative_node = "0",
                    .dc_volts = 1.0,
                    .ac = std::nullopt,
                    .transient = std::nullopt},
      Resistor{.name = "R1",
               .positive_node = "in",
               .negative_node = "mid",
               .resistance_ohms = resistance},
      Inductor{.name = "L1",
               .positive_node = "mid",
               .negative_node = "out",
               .inductance_henries = inductance},
      Capacitor{.name = "C1",
                .positive_node = "out",
                .negative_node = "0",
                .capacitance_farads = capacitance},
  });
  Result<TransientResult> result =
      RunTransientAnalysis(rlc, TranAnalysis{.time_step_seconds = 2.5e-7,
                                             .stop_time_seconds = stop,
                                             .start_time_seconds = 0.0,
                                             .use_initial_conditions = true});
  ASSERT_TRUE(result.ok()) << result.error().message;

  const double alpha = resistance / (2.0 * inductance);
  const double omega_zero = 1.0 / std::sqrt(inductance * capacitance);
  const double omega_damped =
      std::sqrt(omega_zero * omega_zero - alpha * alpha);
  const double expected =
      1.0 - std::exp(-alpha * stop) *
                (std::cos(omega_damped * stop) +
                 alpha / omega_damped * std::sin(omega_damped * stop));
  EXPECT_NEAR(result.value().states.back()[NodeIndex(rlc, "out")], expected,
              2e-3);
}

TEST(Phase2CTransientTest, MatchesNaturalRlcResponseFromDcOperatingPoint) {
  constexpr double resistance = 10.0;
  constexpr double inductance = 1e-3;
  constexpr double capacitance = 1e-6;
  constexpr double stop = 50e-6;
  MnaSystem rlc = Compile({
      VoltageSource{
          .name = "V1",
          .positive_node = "in",
          .negative_node = "0",
          .dc_volts = 1.0,
          .ac = std::nullopt,
          .transient = PwlWaveform{.time_value_pairs = {{0.0, 0.0}}},
      },
      Resistor{.name = "R1",
               .positive_node = "in",
               .negative_node = "mid",
               .resistance_ohms = resistance},
      Inductor{.name = "L1",
               .positive_node = "mid",
               .negative_node = "out",
               .inductance_henries = inductance},
      Capacitor{.name = "C1",
                .positive_node = "out",
                .negative_node = "0",
                .capacitance_farads = capacitance},
  });
  Result<TransientResult> result =
      RunTransientAnalysis(rlc, TranAnalysis{.time_step_seconds = 1e-8,
                                             .stop_time_seconds = stop,
                                             .start_time_seconds = 0.0,
                                             .use_initial_conditions = false});
  ASSERT_TRUE(result.ok()) << result.error().message;

  const double alpha = resistance / (2.0 * inductance);
  const double omega_zero = 1.0 / std::sqrt(inductance * capacitance);
  const double omega_damped =
      std::sqrt(omega_zero * omega_zero - alpha * alpha);
  const double expected =
      std::exp(-alpha * stop) *
      (std::cos(omega_damped * stop) +
       alpha / omega_damped * std::sin(omega_damped * stop));
  EXPECT_NEAR(result.value().states.back()[NodeIndex(rlc, "out")], expected,
              2e-3);
}

TEST(Phase2CTransientTest, PreservesGminInclusiveStaticReference) {
  MnaSystem system = Compile({
      VoltageSource{.name = "V1",
                    .positive_node = "out",
                    .negative_node = "0",
                    .dc_volts = 5.0,
                    .ac = std::nullopt,
                    .transient = std::nullopt},
      Resistor{.name = "R1",
               .positive_node = "out",
               .negative_node = "0",
               .resistance_ohms = 1000.0},
  });
  Result<TransientResult> result = RunTransientAnalysis(
      system, TranAnalysis{.time_step_seconds = 1e-3,
                           .stop_time_seconds = 1e-3,
                           .start_time_seconds = 0.0,
                           .use_initial_conditions = false});
  ASSERT_TRUE(result.ok()) << result.error().message;
  const double expected_current = -5.0 * (1.0 / 1000.0 + kGminSiemens);
  EXPECT_NEAR(result.value().states.back()[BranchIndex(system, "V1")],
              expected_current, 1e-15);
}

TEST(Phase2CTransientTest, PropagatesSingularSolveAsTypedFailure) {
  const MnaSystem singular{
      .g = CsrMatrix{.rows = 1,
                     .columns = 1,
                     .values = {},
                     .column_indices = {},
                     .row_offsets = {0, 0}},
      .c = CsrMatrix{.rows = 1,
                     .columns = 1,
                     .values = {},
                     .column_indices = {},
                     .row_offsets = {0, 0}},
      .b_dc = {0.0},
      .b_ac = {{0.0, 0.0}},
      .node_names = {"floating"},
      .branch_names = {},
  };
  Result<TransientResult> result = RunTransientAnalysis(
      singular, TranAnalysis{.time_step_seconds = 1e-3,
                             .stop_time_seconds = 1e-3,
                             .start_time_seconds = 0.0,
                             .use_initial_conditions = false});
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.error().code, ErrorCode::kSolve);
}

} // namespace
} // namespace ohmnivore
