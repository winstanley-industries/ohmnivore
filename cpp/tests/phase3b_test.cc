#include "cpp/tests/google_test.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

#include "SuiteSparse_config.h"
#include "ohmnivore/compiler.h"
#include "ohmnivore/ir.h"
#include "ohmnivore/nonlinear.h"
#include "ohmnivore/parser.h"
#include "ohmnivore/simulator.h"
#include "ohmnivore/status.h"
#include "ohmnivore/transient.h"

namespace ohmnivore {
namespace {

using SuiteSparseMalloc = void *(*)(std::size_t);
using SuiteSparseCalloc = void *(*)(std::size_t, std::size_t);
using SuiteSparseRealloc = void *(*)(void *, std::size_t);

SuiteSparseMalloc g_original_malloc = nullptr;
SuiteSparseCalloc g_original_calloc = nullptr;
SuiteSparseRealloc g_original_realloc = nullptr;
std::size_t g_allocations_before_failure = 0;

[[nodiscard]] bool RejectThisSuiteSparseAllocation() {
  if (g_allocations_before_failure == 0) {
    return true;
  }
  --g_allocations_before_failure;
  return false;
}

void *FailSuiteSparseMallocAfter(std::size_t size) {
  return RejectThisSuiteSparseAllocation() ? nullptr : g_original_malloc(size);
}

void *FailSuiteSparseCallocAfter(std::size_t count, std::size_t size) {
  return RejectThisSuiteSparseAllocation() ? nullptr
                                           : g_original_calloc(count, size);
}

void *FailSuiteSparseReallocAfter(void *pointer, std::size_t size) {
  return RejectThisSuiteSparseAllocation() ? nullptr
                                           : g_original_realloc(pointer, size);
}

[[nodiscard]] MnaSystem Compile(std::string_view netlist) {
  Result<Circuit> parsed = ParseNetlist(netlist);
  EXPECT_TRUE(parsed.ok()) << parsed.error().message;
  if (!parsed.ok()) {
    return {};
  }
  Result<MnaSystem> compiled = CompileMna(parsed.value());
  EXPECT_TRUE(compiled.ok()) << compiled.error().message;
  return compiled.ok() ? compiled.TakeValue() : MnaSystem{};
}

[[nodiscard]] std::size_t NodeIndex(const MnaSystem &system,
                                    std::string_view name) {
  const auto found =
      std::find(system.node_names.begin(), system.node_names.end(), name);
  EXPECT_NE(found, system.node_names.end());
  return static_cast<std::size_t>(found - system.node_names.begin());
}

[[nodiscard]] std::size_t BranchIndex(const MnaSystem &system,
                                      std::string_view name) {
  const auto found =
      std::find(system.branch_names.begin(), system.branch_names.end(), name);
  EXPECT_NE(found, system.branch_names.end());
  return system.node_names.size() +
         static_cast<std::size_t>(found - system.branch_names.begin());
}

[[nodiscard]] double
IndependentResistorDiodeRoot(double source_volts, double resistance_ohms,
                             double saturation_current_amperes,
                             double ideality_factor) {
  const double emission = ideality_factor * kDiodeThermalVoltageVolts;
  const auto residual = [&](double voltage) {
    return (voltage - source_volts) / resistance_ohms + kGminSiemens * voltage +
           saturation_current_amperes * std::expm1(voltage / emission);
  };
  double lower = 0.0;
  double upper = source_volts;
  for (std::size_t iteration = 0; iteration < 200; ++iteration) {
    const double middle = std::midpoint(lower, upper);
    if (residual(middle) > 0.0) {
      upper = middle;
    } else {
      lower = middle;
    }
  }
  return std::midpoint(lower, upper);
}

[[nodiscard]] double IndependentRcDiodeRk4(double stop_time_seconds) {
  constexpr double source_volts = 1.0;
  constexpr double resistance_ohms = 1000.0;
  constexpr double capacitance_farads = 1e-6;
  constexpr double saturation_current_amperes = 1e-12;
  constexpr double emission_voltage = kDiodeThermalVoltageVolts;
  constexpr std::size_t steps = 200'000;
  const double step = stop_time_seconds / static_cast<double>(steps);
  const auto derivative = [](double voltage) {
    const double diode =
        saturation_current_amperes * std::expm1(voltage / emission_voltage);
    return ((source_volts - voltage) / resistance_ohms -
            kGminSiemens * voltage - diode) /
           capacitance_farads;
  };
  double voltage = 0.0;
  for (std::size_t index = 0; index < steps; ++index) {
    const double k1 = derivative(voltage);
    const double k2 = derivative(voltage + 0.5 * step * k1);
    const double k3 = derivative(voltage + 0.5 * step * k2);
    const double k4 = derivative(voltage + step * k3);
    voltage += step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
  }
  return voltage;
}

[[nodiscard]] std::vector<double>
MultiplyIndependently(const CsrMatrix &matrix,
                      const std::vector<double> &state) {
  std::vector<double> product(matrix.rows, 0.0);
  for (std::size_t row = 0; row < matrix.rows; ++row) {
    for (std::size_t index = matrix.row_offsets[row];
         index < matrix.row_offsets[row + 1]; ++index) {
      product[row] +=
          matrix.values[index] * state[matrix.column_indices[index]];
    }
  }
  return product;
}

void ExpectTrapezoidalDaeResidual(const MnaSystem &system,
                                  const std::vector<double> &previous,
                                  const std::vector<double> &current,
                                  double previous_time, double current_time) {
  const double step = current_time - previous_time;
  ASSERT_GT(step, 0.0);
  Result<std::vector<double>> previous_rhs =
      BuildTransientRhs(system, previous_time);
  Result<std::vector<double>> current_rhs =
      BuildTransientRhs(system, current_time);
  Result<std::vector<double>> previous_diode =
      BuildDiodeResidualContribution(system, previous);
  Result<std::vector<double>> current_diode =
      BuildDiodeResidualContribution(system, current);
  ASSERT_TRUE(previous_rhs.ok()) << previous_rhs.error().message;
  ASSERT_TRUE(current_rhs.ok()) << current_rhs.error().message;
  ASSERT_TRUE(previous_diode.ok()) << previous_diode.error().message;
  ASSERT_TRUE(current_diode.ok()) << current_diode.error().message;
  const std::vector<double> previous_g =
      MultiplyIndependently(system.g, previous);
  const std::vector<double> current_g =
      MultiplyIndependently(system.g, current);
  const std::vector<double> previous_c =
      MultiplyIndependently(system.c, previous);
  const std::vector<double> current_c =
      MultiplyIndependently(system.c, current);
  for (std::size_t row = 0; row < system.g.rows; ++row) {
    const double dynamic = 2.0 * (current_c[row] - previous_c[row]) / step;
    const double residual =
        current_g[row] + previous_g[row] + dynamic - current_rhs.value()[row] -
        previous_rhs.value()[row] + current_diode.value()[row] +
        previous_diode.value()[row];
    const double scale = std::abs(current_g[row]) + std::abs(previous_g[row]) +
                         std::abs(dynamic) +
                         std::abs(current_rhs.value()[row]) +
                         std::abs(previous_rhs.value()[row]) +
                         std::abs(current_diode.value()[row]) +
                         std::abs(previous_diode.value()[row]);
    const double absolute = row < system.node_names.size()
                                ? kNewtonCurrentAbsoluteTolerance
                                : kNewtonVoltageAbsoluteTolerance;
    EXPECT_LE(std::abs(residual), absolute + kNewtonRelativeTolerance * scale)
        << "row=" << row << " previous_time=" << previous_time
        << " current_time=" << current_time;
  }
}

void ExpectBitwiseEqual(const TransientResult &first,
                        const TransientResult &second) {
  ASSERT_EQ(first.times_seconds.size(), second.times_seconds.size());
  ASSERT_EQ(first.states.size(), second.states.size());
  ASSERT_EQ(first.step_trace.size(), second.step_trace.size());
  for (std::size_t index = 0; index < first.times_seconds.size(); ++index) {
    EXPECT_EQ(std::bit_cast<std::uint64_t>(first.times_seconds[index]),
              std::bit_cast<std::uint64_t>(second.times_seconds[index]));
    ASSERT_EQ(first.states[index].size(), second.states[index].size());
    for (std::size_t variable = 0; variable < first.states[index].size();
         ++variable) {
      EXPECT_EQ(std::bit_cast<std::uint64_t>(first.states[index][variable]),
                std::bit_cast<std::uint64_t>(second.states[index][variable]));
    }
  }
  for (std::size_t index = 0; index < first.step_trace.size(); ++index) {
    const TransientStepRecord &left = first.step_trace[index];
    const TransientStepRecord &right = second.step_trace[index];
    EXPECT_EQ(std::bit_cast<std::uint64_t>(left.start_time_seconds),
              std::bit_cast<std::uint64_t>(right.start_time_seconds));
    EXPECT_EQ(std::bit_cast<std::uint64_t>(left.end_time_seconds),
              std::bit_cast<std::uint64_t>(right.end_time_seconds));
    EXPECT_EQ(std::bit_cast<std::uint64_t>(left.step_size_seconds),
              std::bit_cast<std::uint64_t>(right.step_size_seconds));
    EXPECT_EQ(left.method, right.method);
    EXPECT_EQ(left.accepted, right.accepted);
    EXPECT_EQ(std::bit_cast<std::uint64_t>(left.normalized_local_error),
              std::bit_cast<std::uint64_t>(right.normalized_local_error));
    EXPECT_EQ(left.landed_on_hard_point, right.landed_on_hard_point);
    EXPECT_EQ(left.rejection_reason, right.rejection_reason);
  }
  EXPECT_EQ(first.solver_statistics.symbolic_analyses,
            second.solver_statistics.symbolic_analyses);
  EXPECT_EQ(first.solver_statistics.numeric_factorizations,
            second.solver_statistics.numeric_factorizations);
  EXPECT_EQ(first.solver_statistics.numeric_refactorizations,
            second.solver_statistics.numeric_refactorizations);
  EXPECT_EQ(first.solver_statistics.numeric_refactorization_fallbacks,
            second.solver_statistics.numeric_refactorization_fallbacks);
  EXPECT_EQ(first.solver_statistics.numeric_reuses,
            second.solver_statistics.numeric_reuses);
  EXPECT_EQ(first.solver_statistics.solves, second.solver_statistics.solves);
}

TEST(Phase3BAnalyticTest,
     MemorylessDrivenDiodePreservesVoltageAndCurrentSigns) {
  constexpr char netlist[] = R"(V1 drive 0 PWL(0 0 1u 0.2 2u 0.4)
.MODEL DM D(IS=1e-12 N=1.5)
D1 drive 0 DM
.TRAN 0.5u 2u UIC
.END
)";
  MnaSystem system = Compile(netlist);
  Result<TransientResult> integrated = RunTransientAnalysis(
      system, TranAnalysis{.time_step_seconds = 0.5e-6,
                           .stop_time_seconds = 2e-6,
                           .start_time_seconds = 0.0,
                           .use_initial_conditions = true});
  ASSERT_TRUE(integrated.ok()) << integrated.error().message;
  const std::size_t drive = NodeIndex(system, "drive");
  const std::size_t branch = BranchIndex(system, "V1");
  ASSERT_EQ(integrated.value().times_seconds.size(),
            integrated.value().states.size());
  for (std::size_t index = 0; index < integrated.value().states.size();
       ++index) {
    const double time = integrated.value().times_seconds[index];
    const double expected_voltage = 0.4 * time / 2e-6;
    EXPECT_NEAR(integrated.value().states[index][drive], expected_voltage,
                1e-15);
    Result<DiodeEvaluation> diode =
        EvaluateDiode(expected_voltage, 1e-12, 1.5 * kDiodeThermalVoltageVolts);
    ASSERT_TRUE(diode.ok()) << diode.error().message;
    const double expected_current =
        -kGminSiemens * expected_voltage - diode.value().current_amperes;
    EXPECT_NEAR(integrated.value().states[index][branch], expected_current,
                1e-15);

    MnaSystem point = system;
    Result<std::vector<double>> rhs = BuildTransientRhs(system, time);
    ASSERT_TRUE(rhs.ok()) << rhs.error().message;
    point.b_dc = rhs.TakeValue();
    Result<double> residual =
        ValidateNonlinearResidual(point, integrated.value().states[index]);
    ASSERT_TRUE(residual.ok()) << residual.error().message;
  }
}

TEST(Phase3BAnalyticTest, MatchesIndependentHighAccuracyRcDiodeOracle) {
  constexpr char netlist[] = R"(V1 in 0 DC 1
R1 in out 1k
C1 out 0 1u
.MODEL DM D(IS=1e-12 N=1)
D1 out 0 DM
.TRAN 20u 200u UIC
.END
)";
  MnaSystem system = Compile(netlist);
  Result<TransientResult> integrated = RunTransientAnalysis(
      system, TranAnalysis{.time_step_seconds = 20e-6,
                           .stop_time_seconds = 200e-6,
                           .start_time_seconds = 0.0,
                           .use_initial_conditions = true});
  ASSERT_TRUE(integrated.ok()) << integrated.error().message;
  const double actual =
      integrated.value().states.back()[NodeIndex(system, "out")];
  const double independent = IndependentRcDiodeRk4(200e-6);
  EXPECT_NEAR(actual, independent, 2e-5);
  std::size_t state_index = 0;
  std::size_t trapezoidal_steps = 0;
  for (const TransientStepRecord &step : integrated.value().step_trace) {
    if (!step.accepted) {
      continue;
    }
    ++state_index;
    ASSERT_LT(state_index, integrated.value().states.size());
    if (step.method == TransientIntegrationMethod::kTrapezoidal) {
      ++trapezoidal_steps;
      ExpectTrapezoidalDaeResidual(
          system, integrated.value().states[state_index - 1],
          integrated.value().states[state_index], step.start_time_seconds,
          step.end_time_seconds);
    }
  }
  EXPECT_GT(trapezoidal_steps, 0U);
  EXPECT_EQ(state_index + 1, integrated.value().states.size());
  EXPECT_LE(integrated.value().solver_statistics.symbolic_analyses, 2U);
  EXPECT_GT(integrated.value().solver_statistics.numeric_refactorizations, 0U);
}

TEST(Phase3BInitializationTest, SupportsDcOperatingPointAndZeroStateUic) {
  constexpr char netlist[] = R"(V1 in 0 DC 1
R1 in out 1k
C1 out 0 1u
.MODEL DM D(IS=1e-12 N=1)
D1 out 0 DM
.TRAN 10u 20u
.END
)";
  MnaSystem system = Compile(netlist);
  Result<std::vector<double>> dc = BuildTransientInitialState(system, false);
  ASSERT_TRUE(dc.ok()) << dc.error().message;
  const double expected = IndependentResistorDiodeRoot(1.0, 1000.0, 1e-12, 1.0);
  EXPECT_NEAR(dc.value()[NodeIndex(system, "out")], expected, 1e-10);

  Result<std::vector<double>> uic = BuildTransientInitialState(system, true);
  ASSERT_TRUE(uic.ok()) << uic.error().message;
  EXPECT_DOUBLE_EQ(uic.value()[NodeIndex(system, "out")], 0.0);
  EXPECT_DOUBLE_EQ(uic.value()[NodeIndex(system, "in")], 1.0);
}

TEST(Phase3BProjectionTest, PreservesCapacitorVoltageAtPulseDiscontinuity) {
  constexpr char netlist[] = R"(V1 in 0 PULSE(0 1 0.5m 0 0 1m 2m)
R1 in out 1k
C1 out 0 1u
.MODEL DM D(IS=1e-12 N=1)
D1 out 0 DM
.TRAN 0.25m 0.5m UIC
.END
)";
  MnaSystem system = Compile(netlist);
  Result<TransientResult> integrated = RunTransientAnalysis(
      system, TranAnalysis{.time_step_seconds = 0.25e-3,
                           .stop_time_seconds = 0.5e-3,
                           .start_time_seconds = 0.0,
                           .use_initial_conditions = true});
  ASSERT_TRUE(integrated.ok()) << integrated.error().message;
  ASSERT_EQ(integrated.value().times_seconds.back(), 0.5e-3);
  const std::vector<double> &edge = integrated.value().states.back();
  EXPECT_DOUBLE_EQ(edge[NodeIndex(system, "in")], 1.0);
  EXPECT_NEAR(edge[NodeIndex(system, "out")], 0.0, 1e-15);
  EXPECT_TRUE(std::any_of(
      integrated.value().step_trace.begin(),
      integrated.value().step_trace.end(), [](const TransientStepRecord &step) {
        return step.accepted && step.end_time_seconds == 0.5e-3 &&
               step.method == TransientIntegrationMethod::kBackwardEuler &&
               step.landed_on_hard_point;
      }));
}

TEST(Phase3BProjectionTest,
     PreservesTwoTerminalCapacitorAndRetainedPhysicalDiodeRow) {
  constexpr char netlist[] = R"(V1 left 0 PULSE(0 1 0.5m 0 0 1m 2m)
R1 right 0 1k
C1 left right 1u
.MODEL DM D(IS=1e-12 N=1)
D1 left right DM
.TRAN 0.25m 0.5m UIC
.END
)";
  MnaSystem system = Compile(netlist);
  Result<TransientResult> integrated = RunTransientAnalysis(
      system, TranAnalysis{.time_step_seconds = 0.25e-3,
                           .stop_time_seconds = 0.5e-3,
                           .start_time_seconds = 0.0,
                           .use_initial_conditions = true});
  ASSERT_TRUE(integrated.ok()) << integrated.error().message;
  const std::vector<double> &edge = integrated.value().states.back();
  const std::size_t left = NodeIndex(system, "left");
  const std::size_t right = NodeIndex(system, "right");
  EXPECT_DOUBLE_EQ(edge[left], 1.0);
  EXPECT_NEAR(edge[left] - edge[right], 0.0, 1e-15);

  Result<DiodeEvaluation> diode =
      EvaluateDiode(edge[left] - edge[right], 1e-12, kDiodeThermalVoltageVolts);
  ASSERT_TRUE(diode.ok()) << diode.error().message;
  const double retained_left_kcl = edge[BranchIndex(system, "V1")] +
                                   kGminSiemens * edge[left] +
                                   diode.value().current_amperes;
  EXPECT_NEAR(retained_left_kcl, 0.0, 1e-15);
}

TEST(Phase3BProjectionTest,
     AcceptsConsistentRedundantCapacitorAndVoltageSourceConstraint) {
  constexpr char netlist[] = R"(V1 n 0 DC 0
C1 n 0 1u
.MODEL DM D(IS=1e-12 N=1)
D1 n 0 DM
.TRAN 1u 1u UIC
.END
)";
  MnaSystem system = Compile(netlist);
  Result<std::vector<double>> initial =
      BuildTransientInitialState(system, true);
  ASSERT_TRUE(initial.ok()) << initial.error().message;
  EXPECT_DOUBLE_EQ(initial.value()[NodeIndex(system, "n")], 0.0);

  Result<TransientResult> integrated = RunTransientAnalysis(
      system, TranAnalysis{.time_step_seconds = 1e-6,
                           .stop_time_seconds = 1e-6,
                           .start_time_seconds = 0.0,
                           .use_initial_conditions = true});
  ASSERT_TRUE(integrated.ok()) << integrated.error().message;
  EXPECT_DOUBLE_EQ(integrated.value().states.back()[NodeIndex(system, "n")],
                   0.0);
}

TEST(Phase3BProjectionTest,
     RejectsConflictingRedundantCapacitorAndVoltageSourceConstraint) {
  constexpr char netlist[] = R"(V1 n 0 DC 1
C1 n 0 1u
.MODEL DM D(IS=1e-12 N=1)
D1 n 0 DM
.TRAN 1u 1u UIC
.END
)";
  MnaSystem system = Compile(netlist);
  Result<std::vector<double>> rejected =
      BuildTransientInitialState(system, true);
  ASSERT_FALSE(rejected.ok());
  EXPECT_EQ(rejected.error().code, ErrorCode::kSolutionValidation);
}

TEST(Phase3BDeterminismTest, RepeatsStatesTracesAndSparseStatisticsBitwise) {
  constexpr char netlist[] = R"(V1 in 0 DC 1
R1 in out 1k
C1 out 0 1u
.MODEL DM D(IS=1e-12 N=1)
D1 out 0 DM
.TRAN 20u 100u UIC
.END
)";
  MnaSystem system = Compile(netlist);
  const TranAnalysis analysis{.time_step_seconds = 20e-6,
                              .stop_time_seconds = 100e-6,
                              .start_time_seconds = 0.0,
                              .use_initial_conditions = true};
  Result<TransientResult> first = RunTransientAnalysis(system, analysis);
  Result<TransientResult> second = RunTransientAnalysis(system, analysis);
  ASSERT_TRUE(first.ok()) << first.error().message;
  ASSERT_TRUE(second.ok()) << second.error().message;
  ExpectBitwiseEqual(first.value(), second.value());
}

TEST(Phase3BTimestepRetryTest, HalvesOnlyNonconvergedImplicitSteps) {
  constexpr char netlist[] = R"(V1 in 0 DC 5
R1 in out 1k
C1 out 0 1u
.MODEL DM D(IS=1e-14 N=1)
D1 out 0 DM
.TRAN 1m 1m UIC
.END
)";
  MnaSystem system = Compile(netlist);
  Result<TransientResult> integrated = RunTransientAnalysis(
      system,
      TranAnalysis{.time_step_seconds = 1e-3,
                   .stop_time_seconds = 1e-3,
                   .start_time_seconds = 0.0,
                   .use_initial_conditions = true},
      TransientExecutionLimits{.maximum_accepted_steps = 1000,
                               .maximum_step_attempts = 2000,
                               .minimum_step_divisor = 10'000.0,
                               .nonlinear_maximum_iterations = 2});
  ASSERT_TRUE(integrated.ok()) << integrated.error().message;
  const auto rejected = std::find_if(
      integrated.value().step_trace.begin(),
      integrated.value().step_trace.end(), [](const TransientStepRecord &step) {
        return !step.accepted &&
               step.rejection_reason ==
                   TransientStepRejectionReason::kNonlinearConvergence;
      });
  ASSERT_NE(rejected, integrated.value().step_trace.end());
  const auto recovered = std::find_if(
      rejected + 1, integrated.value().step_trace.end(),
      [](const TransientStepRecord &step) { return step.accepted; });
  ASSERT_NE(recovered, integrated.value().step_trace.end());
  EXPECT_EQ(recovered->method, TransientIntegrationMethod::kBackwardEuler);
  EXPECT_LE(recovered->step_size_seconds, rejected->step_size_seconds * 0.5);
}

TEST(Phase3BFailureTest, ExhaustsOnlyNonconvergenceRetry) {
  constexpr char netlist[] = R"(V1 in 0 DC 5
R1 in out 1k
C1 out 0 1u
.MODEL DM D(IS=1e-14 N=1)
D1 out 0 DM
.TRAN 1m 1m UIC
.END
)";
  MnaSystem system = Compile(netlist);
  Result<TransientResult> exhausted = RunTransientAnalysis(
      system,
      TranAnalysis{.time_step_seconds = 1e-3,
                   .stop_time_seconds = 1e-3,
                   .start_time_seconds = 0.0,
                   .use_initial_conditions = true},
      TransientExecutionLimits{.maximum_accepted_steps = 100,
                               .maximum_step_attempts = 100,
                               .minimum_step_divisor = 4.0,
                               .nonlinear_maximum_iterations = 0});
  ASSERT_FALSE(exhausted.ok());
  EXPECT_EQ(exhausted.error().code, ErrorCode::kNonConvergence);
}

TEST(Phase3BFailureTest,
     PropagatesImplicitSingularityAndNonFiniteCompanionWithoutRetry) {
  constexpr double saturation_current = 1e-14;
  Result<DiodeEvaluation> zero_bias =
      EvaluateDiode(0.0, saturation_current, kDiodeThermalVoltageVolts);
  ASSERT_TRUE(zero_bias.ok()) << zero_bias.error().message;
  const auto make_system = [&](double g_value, CsrMatrix c) {
    return MnaSystem{
        .g = CsrMatrix{.rows = 1,
                       .columns = 1,
                       .values = {g_value},
                       .column_indices = {0},
                       .row_offsets = {0, 1}},
        .c = std::move(c),
        .b_dc = {0.0},
        .b_ac = {{0.0, 0.0}},
        .node_names = {"n"},
        .branch_names = {},
        .capacitor_initial_constraints = {CapacitorInitialConstraint{
            .name = "Ctest",
            .positive_node_index = 0,
            .negative_node_index = std::nullopt,
        }},
        .diode_descriptors = {DiodeDescriptor{
            .name = "D1",
            .anode_node_index = 0,
            .cathode_node_index = std::nullopt,
            .saturation_current_amperes = saturation_current,
            .emission_voltage_volts = kDiodeThermalVoltageVolts,
            .anode_anode_value_index = 0,
            .anode_cathode_value_index = std::nullopt,
            .cathode_anode_value_index = std::nullopt,
            .cathode_cathode_value_index = std::nullopt,
        }},
    };
  };

  const MnaSystem singular = make_system(-zero_bias.value().conductance_siemens,
                                         CsrMatrix{.rows = 1,
                                                   .columns = 1,
                                                   .values = {},
                                                   .column_indices = {},
                                                   .row_offsets = {0, 0}});
  Result<TransientResult> singular_result = RunTransientAnalysis(
      singular, TranAnalysis{.time_step_seconds = 1e-3,
                             .stop_time_seconds = 1e-3,
                             .start_time_seconds = 0.0,
                             .use_initial_conditions = true});
  ASSERT_FALSE(singular_result.ok());
  EXPECT_EQ(singular_result.error().code, ErrorCode::kSingular);
  EXPECT_NE(singular_result.error().message.find(
                "backward-Euler transient solve failed"),
            std::string::npos);

  const MnaSystem non_finite =
      make_system(0.0, CsrMatrix{.rows = 1,
                                 .columns = 1,
                                 .values = {1e308},
                                 .column_indices = {0},
                                 .row_offsets = {0, 1}});
  Result<TransientResult> non_finite_result = RunTransientAnalysis(
      non_finite, TranAnalysis{.time_step_seconds = 1e-308,
                               .stop_time_seconds = 1e-308,
                               .start_time_seconds = 0.0,
                               .use_initial_conditions = true});
  ASSERT_FALSE(non_finite_result.ok());
  EXPECT_EQ(non_finite_result.error().code, ErrorCode::kSolve);
  EXPECT_NE(non_finite_result.error().message.find(
                "transient companion matrix produced a non-finite value"),
            std::string::npos);
}

TEST(Phase3BFailureTest,
     PropagatesImplicitNumericAllocationFailureWithoutRetry) {
  constexpr char netlist[] = R"(V1 in 0 DC 1
R1 in out 1k
C1 out 0 1u
.MODEL DM D(IS=1e-12 N=1)
D1 out 0 DM
.TRAN 20u 20u UIC
.END
)";
  const MnaSystem system = Compile(netlist);
  const TranAnalysis analysis{.time_step_seconds = 20e-6,
                              .stop_time_seconds = 20e-6,
                              .start_time_seconds = 0.0,
                              .use_initial_conditions = true};
  g_original_malloc = SuiteSparse_config_malloc_func_get();
  g_original_calloc = SuiteSparse_config_calloc_func_get();
  g_original_realloc = SuiteSparse_config_realloc_func_get();

  bool observed_implicit_failure = false;
  for (std::size_t allocation = 0; allocation < 256; ++allocation) {
    g_allocations_before_failure = allocation;
    SuiteSparse_config_malloc_func_set(FailSuiteSparseMallocAfter);
    SuiteSparse_config_calloc_func_set(FailSuiteSparseCallocAfter);
    SuiteSparse_config_realloc_func_set(FailSuiteSparseReallocAfter);
    Result<TransientResult> result = RunTransientAnalysis(system, analysis);
    SuiteSparse_config_malloc_func_set(g_original_malloc);
    SuiteSparse_config_calloc_func_set(g_original_calloc);
    SuiteSparse_config_realloc_func_set(g_original_realloc);

    if (!result.ok() && result.error().code == ErrorCode::kFactorization &&
        result.error().message.find("backward-Euler transient solve failed") !=
            std::string::npos) {
      observed_implicit_failure = true;
      break;
    }
  }
  EXPECT_TRUE(observed_implicit_failure);
}

TEST(Phase3BSimulationTest, EmitsLegacyCsvAndKeepsDiodeAcUnsupported) {
  constexpr char transient[] = R"(V1 in 0 PWL(0 0 1u 0.2)
.MODEL DM D(IS=1e-12 N=1)
D1 in 0 DM
.TRAN 0.5u 1u UIC
.END
)";
  Result<std::string> csv = SimulateTransientToCsv(transient);
  ASSERT_TRUE(csv.ok()) << csv.error().message;
  EXPECT_EQ(csv.value().substr(0, csv.value().find('\n')), "time,V(in),I(V1)");

  constexpr char ac[] = R"(V1 in 0 DC 1 AC 1
.MODEL DM D
D1 in 0 DM
.AC LIN 2 1 2
.END
)";
  Result<AcResult> unsupported = SimulateAc(ac);
  ASSERT_FALSE(unsupported.ok());
  EXPECT_EQ(unsupported.error().code, ErrorCode::kUnsupported);
}

} // namespace
} // namespace ohmnivore
