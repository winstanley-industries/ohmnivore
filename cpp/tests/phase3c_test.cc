#include "cpp/tests/google_test.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "ohmnivore/compiler.h"
#include "ohmnivore/ir.h"
#include "ohmnivore/nonlinear.h"
#include "ohmnivore/parser.h"
#include "ohmnivore/simulator.h"
#include "ohmnivore/status.h"
#include "ohmnivore/transient.h"

namespace ohmnivore {
namespace {

[[nodiscard]] MnaSystem Compile(const std::string &netlist) {
  auto parsed = ParseNetlist(netlist);
  if (!parsed.ok()) {
    ADD_FAILURE() << parsed.error().message;
    return {};
  }
  auto compiled = CompileMna(parsed.value());
  if (!compiled.ok()) {
    ADD_FAILURE() << compiled.error().message;
    return {};
  }
  return compiled.TakeValue();
}

[[nodiscard]] double NodeVoltage(const DcResult &result,
                                 const std::string &name) {
  const auto found =
      std::find_if(result.node_voltages.begin(), result.node_voltages.end(),
                   [&](const auto &entry) { return entry.first == name; });
  EXPECT_NE(found, result.node_voltages.end());
  return found == result.node_voltages.end() ? 0.0 : found->second;
}

void ExpectBitwiseEqual(double first, double second) {
  EXPECT_EQ(std::bit_cast<std::uint64_t>(first),
            std::bit_cast<std::uint64_t>(second));
}

void ExpectNonlinearResultsBitwiseEqual(const NonlinearDcResult &first,
                                        const NonlinearDcResult &second) {
  ASSERT_EQ(first.solution.size(), second.solution.size());
  for (std::size_t index = 0; index < first.solution.size(); ++index) {
    ExpectBitwiseEqual(first.solution[index], second.solution[index]);
  }
  ASSERT_EQ(first.iteration_trace.size(), second.iteration_trace.size());
  for (std::size_t index = 0; index < first.iteration_trace.size(); ++index) {
    const auto &left = first.iteration_trace[index];
    const auto &right = second.iteration_trace[index];
    EXPECT_EQ(left.strategy, right.strategy);
    ExpectBitwiseEqual(left.continuation_value, right.continuation_value);
    EXPECT_EQ(left.iteration, right.iteration);
    ExpectBitwiseEqual(left.maximum_normalized_update,
                       right.maximum_normalized_update);
    ExpectBitwiseEqual(left.maximum_normalized_residual,
                       right.maximum_normalized_residual);
    EXPECT_EQ(left.accepted, right.accepted);
  }
  ASSERT_EQ(first.attempt_trace.size(), second.attempt_trace.size());
  for (std::size_t index = 0; index < first.attempt_trace.size(); ++index) {
    const auto &left = first.attempt_trace[index];
    const auto &right = second.attempt_trace[index];
    EXPECT_EQ(left.strategy, right.strategy);
    ExpectBitwiseEqual(left.continuation_value, right.continuation_value);
    EXPECT_EQ(left.iterations, right.iterations);
    EXPECT_EQ(left.converged, right.converged);
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

template <typename Residual>
[[nodiscard]] double Bisect(double lower, double upper, Residual residual) {
  for (std::size_t iteration = 0; iteration < 250; ++iteration) {
    const double middle = std::midpoint(lower, upper);
    if (residual(middle) > 0.0) {
      upper = middle;
    } else {
      lower = middle;
    }
  }
  return std::midpoint(lower, upper);
}

TEST(Phase3CParserTest, ParsesStrictInstancesModelsDefaultsCaseAndOrder) {
  constexpr char netlist[] =
      R"(.model Forward npn(nr=1.2 is=2e-14 bf=150 nf=1.1 br=3)
Qfirst c b 0 fOrWaRd
.MODEL Reverse PNP()
qsecond x y z reverse
.MODEL Bare NPN
Qthird d e f BARE
.MODEL Partial PNP(BF=250)
Qfourth g h i partial
.OP
)";
  auto parsed = ParseNetlist(netlist);
  ASSERT_TRUE(parsed.ok()) << parsed.error().message;
  ASSERT_EQ(parsed.value().components.size(), 4U);
  ASSERT_EQ(parsed.value().bjt_models.size(), 4U);
  const BjtModel &forward = parsed.value().bjt_models[0];
  EXPECT_EQ(forward.name, "Forward");
  EXPECT_TRUE(forward.is_npn);
  EXPECT_DOUBLE_EQ(forward.saturation_current_amperes, 2e-14);
  EXPECT_DOUBLE_EQ(forward.forward_current_gain, 150.0);
  EXPECT_DOUBLE_EQ(forward.reverse_current_gain, 3.0);
  EXPECT_DOUBLE_EQ(forward.forward_ideality_factor, 1.1);
  EXPECT_DOUBLE_EQ(forward.reverse_ideality_factor, 1.2);
  const BjtModel &reverse = parsed.value().bjt_models[1];
  EXPECT_FALSE(reverse.is_npn);
  EXPECT_DOUBLE_EQ(reverse.saturation_current_amperes, 1e-16);
  EXPECT_DOUBLE_EQ(reverse.forward_current_gain, 100.0);
  EXPECT_DOUBLE_EQ(reverse.reverse_current_gain, 1.0);
  EXPECT_DOUBLE_EQ(reverse.forward_ideality_factor, 1.0);
  EXPECT_DOUBLE_EQ(reverse.reverse_ideality_factor, 1.0);
  EXPECT_DOUBLE_EQ(parsed.value().bjt_models[3].forward_current_gain, 250.0);
  const auto *first = std::get_if<Bjt>(&parsed.value().components[0]);
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(first->name, "Qfirst");
  EXPECT_EQ(first->collector_node, "c");
  EXPECT_EQ(first->base_node, "b");
  EXPECT_EQ(first->emitter_node, "0");
  EXPECT_EQ(first->model_name, "fOrWaRd");
}

TEST(Phase3CParserTest, RejectsMalformedUnsupportedAndDuplicateInput) {
  struct Case {
    std::string line;
    ErrorCode code;
  };
  const std::vector<Case> cases = {
      {"Q1 c b e", ErrorCode::kParse},
      {"Q1 c b e QM extra", ErrorCode::kParse},
      {"Q1 c b e bad-name", ErrorCode::kParse},
      {".MODEL QM NPN(", ErrorCode::kParse},
      {".MODEL QM NPN)", ErrorCode::kParse},
      {".MODEL QM PNP+", ErrorCode::kParse},
      {".MODEL QM NPN (IS=1e-14)", ErrorCode::kParse},
      {".MODEL QM NPN((IS=1e-14))", ErrorCode::kParse},
      {".MODEL QM NPN(IS=1e-14) trailing", ErrorCode::kParse},
      {".MODEL QM NPN(IS)", ErrorCode::kParse},
      {".MODEL QM NPN(IS=1e-14 is=2e-14)", ErrorCode::kParse},
      {".MODEL QM NPN(BF=1 BF=2)", ErrorCode::kParse},
      {".MODEL QM NPN(IS=0)", ErrorCode::kParse},
      {".MODEL QM NPN(BF=-1)", ErrorCode::kParse},
      {".MODEL QM NPN(NF=1e-323)", ErrorCode::kParse},
      {".MODEL QM NPN(NR=1e101)", ErrorCode::kParse},
      {".MODEL QM NPN(IS=1e-14,BF=100)", ErrorCode::kParse},
      {".MODEL QM NPN(VAF=10)", ErrorCode::kUnsupported},
      {".MODEL QM NMOS(KP=1e-3)", ErrorCode::kUnsupported},
      {"M1 d g s MM", ErrorCode::kUnsupported},
  };
  for (const Case &test_case : cases) {
    SCOPED_TRACE(test_case.line);
    auto parsed = ParseNetlist(test_case.line + "\n.OP\n");
    ASSERT_FALSE(parsed.ok());
    EXPECT_EQ(parsed.error().code, test_case.code);
    EXPECT_NE(parsed.error().message.find("line 1"), std::string::npos);
  }

  for (const std::string &duplicate :
       {".MODEL Qmod NPN\n.MODEL qMOD PNP()\n.OP\n",
        ".MODEL Shared D\n.MODEL shared NPN\n.OP\n"}) {
    auto parsed = ParseNetlist(duplicate);
    ASSERT_FALSE(parsed.ok());
    EXPECT_EQ(parsed.error().code, ErrorCode::kParse);
    EXPECT_NE(parsed.error().message.find("duplicate"), std::string::npos);
  }
}

TEST(Phase3CCompilerTest, BuildsExactFullUnionAndOrderedDescriptors) {
  MnaSystem system = Compile(R"(R1 c 0 1k
R2 b 0 2k
R3 e 0 3k
.MODEL QM NPN(IS=2e-14 BF=150 BR=3 NF=1.1 NR=1.2)
Q1 c b e qm
.OP
)");
  EXPECT_TRUE(system.diode_descriptors.empty());
  ASSERT_EQ(system.bjt_descriptors.size(), 1U);
  EXPECT_EQ(system.node_names, (std::vector<std::string>{"c", "b", "e"}));
  EXPECT_EQ(system.g.row_offsets, (std::vector<std::size_t>{0, 3, 6, 9}));
  EXPECT_EQ(system.g.column_indices,
            (std::vector<std::size_t>{0, 1, 2, 0, 1, 2, 0, 1, 2}));
  const BjtDescriptor &descriptor = system.bjt_descriptors[0];
  EXPECT_EQ(descriptor.name, "Q1");
  EXPECT_EQ(descriptor.collector_node_index, 0U);
  EXPECT_EQ(descriptor.base_node_index, 1U);
  EXPECT_EQ(descriptor.emitter_node_index, 2U);
  EXPECT_DOUBLE_EQ(descriptor.polarity, 1.0);
  EXPECT_DOUBLE_EQ(descriptor.saturation_current_amperes, 2e-14);
  EXPECT_DOUBLE_EQ(descriptor.forward_current_gain, 150.0);
  EXPECT_DOUBLE_EQ(descriptor.reverse_current_gain, 3.0);
  EXPECT_DOUBLE_EQ(descriptor.forward_emission_voltage_volts,
                   1.1 * kDiodeThermalVoltageVolts);
  EXPECT_DOUBLE_EQ(descriptor.reverse_emission_voltage_volts,
                   1.2 * kDiodeThermalVoltageVolts);
  for (std::size_t index = 0; index < 9; ++index) {
    EXPECT_EQ(descriptor.jacobian_value_indices[index], index);
  }
}

TEST(Phase3CCompilerTest, AccumulatesAliasedTerminalsAndRejectsSelfConnection) {
  MnaSystem system = Compile(R"(V1 in 0 1
R1 in out 100k
.MODEL QM NPN
Q1 out out 0 QM
.OP
)");
  ASSERT_EQ(system.bjt_descriptors.size(), 1U);
  const BjtDescriptor &descriptor = system.bjt_descriptors[0];
  ASSERT_EQ(descriptor.collector_node_index, descriptor.base_node_index);
  EXPECT_FALSE(descriptor.emitter_node_index.has_value());
  EXPECT_EQ(descriptor.jacobian_value_indices[0],
            descriptor.jacobian_value_indices[1]);
  EXPECT_EQ(descriptor.jacobian_value_indices[0],
            descriptor.jacobian_value_indices[3]);
  EXPECT_EQ(descriptor.jacobian_value_indices[0],
            descriptor.jacobian_value_indices[4]);
  for (std::size_t index : {2U, 5U, 6U, 7U, 8U}) {
    EXPECT_FALSE(descriptor.jacobian_value_indices[index].has_value());
  }

  for (const std::string &terminals : {"x x x", "0 GND 0"}) {
    auto parsed = ParseNetlist("R1 x 0 1k\n.MODEL QM NPN\nQ1 " + terminals +
                               " QM\n.OP\n");
    ASSERT_TRUE(parsed.ok()) << parsed.error().message;
    auto compiled = CompileMna(parsed.value());
    ASSERT_FALSE(compiled.ok());
    EXPECT_EQ(compiled.error().code, ErrorCode::kInvalidStructure);
  }
}

TEST(Phase3CCompilerTest, RejectsMissingWrongTypeAndMalformedDirectIr) {
  auto missing = ParseNetlist("R1 c 0 1k\nQ1 c b 0 Missing\n.OP\n");
  ASSERT_TRUE(missing.ok()) << missing.error().message;
  auto missing_compiled = CompileMna(missing.value());
  ASSERT_FALSE(missing_compiled.ok());
  EXPECT_EQ(missing_compiled.error().code, ErrorCode::kCompile);

  Circuit wrong_type{
      .components = {Resistor{.name = "R1",
                              .positive_node = "c",
                              .negative_node = "0",
                              .resistance_ohms = 1000.0},
                     Bjt{.name = "Q1",
                         .collector_node = "c",
                         .base_node = "b",
                         .emitter_node = "0",
                         .model_name = "DM"}},
      .analyses = {DcAnalysis{}},
      .diode_models = {DiodeModel{.name = "DM"}},
  };
  auto wrong = CompileMna(wrong_type);
  ASSERT_FALSE(wrong.ok());
  EXPECT_EQ(wrong.error().code, ErrorCode::kCompile);

  wrong_type.diode_models.clear();
  wrong_type.bjt_models = {BjtModel{.name = "QM"}};
  std::get<Bjt>(wrong_type.components[1]).model_name = "bad reference";
  auto malformed_reference = CompileMna(wrong_type);
  ASSERT_FALSE(malformed_reference.ok());
  EXPECT_EQ(malformed_reference.error().code, ErrorCode::kCompile);

  std::get<Bjt>(wrong_type.components[1]).model_name = "QM";
  wrong_type.bjt_models[0].forward_current_gain =
      std::numeric_limits<double>::quiet_NaN();
  auto invalid_model = CompileMna(wrong_type);
  ASSERT_FALSE(invalid_model.ok());
  EXPECT_EQ(invalid_model.error().code, ErrorCode::kCompile);
}

TEST(Phase3CDeviceTest, MatchesLegacyEquationsPolarityClampAndDerivatives) {
  constexpr double saturation = 1e-14;
  constexpr double forward_gain = 200.0;
  constexpr double reverse_gain = 2.0;
  constexpr double forward_emission = 1.1 * kDiodeThermalVoltageVolts;
  constexpr double reverse_emission = 1.2 * kDiodeThermalVoltageVolts;
  auto zero = EvaluateBjt(0.0, 0.0, 0.0, 1.0, saturation, forward_gain,
                          reverse_gain, forward_emission, reverse_emission);
  ASSERT_TRUE(zero.ok()) << zero.error().message;
  EXPECT_DOUBLE_EQ(zero.value().collector_current_amperes, 0.0);
  EXPECT_DOUBLE_EQ(zero.value().base_current_amperes, 0.0);
  EXPECT_NEAR(zero.value().collector_vbe_derivative_siemens,
              forward_gain / (forward_gain + 1.0) * saturation /
                  forward_emission,
              1e-27);
  EXPECT_NEAR(zero.value().collector_vbc_derivative_siemens,
              -saturation / ((reverse_gain + 1.0) * reverse_emission), 1e-27);

  auto npn = EvaluateBjt(2.0, 0.7, 0.0, 1.0, saturation, forward_gain,
                         reverse_gain, forward_emission, reverse_emission);
  auto pnp = EvaluateBjt(-2.0, -0.7, 0.0, -1.0, saturation, forward_gain,
                         reverse_gain, forward_emission, reverse_emission);
  ASSERT_TRUE(npn.ok()) << npn.error().message;
  ASSERT_TRUE(pnp.ok()) << pnp.error().message;
  EXPECT_DOUBLE_EQ(pnp.value().collector_current_amperes,
                   -npn.value().collector_current_amperes);
  EXPECT_DOUBLE_EQ(pnp.value().base_current_amperes,
                   -npn.value().base_current_amperes);
  EXPECT_DOUBLE_EQ(pnp.value().collector_vbe_derivative_siemens,
                   npn.value().collector_vbe_derivative_siemens);
  EXPECT_DOUBLE_EQ(pnp.value().collector_vbc_derivative_siemens,
                   npn.value().collector_vbc_derivative_siemens);
  EXPECT_DOUBLE_EQ(pnp.value().base_vbe_derivative_siemens,
                   npn.value().base_vbe_derivative_siemens);
  EXPECT_DOUBLE_EQ(pnp.value().base_vbc_derivative_siemens,
                   npn.value().base_vbc_derivative_siemens);

  auto clamped = EvaluateBjt(-1e6, 1e6, 0.0, 1.0, saturation, forward_gain,
                             reverse_gain, forward_emission, reverse_emission);
  ASSERT_TRUE(clamped.ok()) << clamped.error().message;
  EXPECT_DOUBLE_EQ(clamped.value().forward_exponent, 80.0);
  EXPECT_DOUBLE_EQ(clamped.value().reverse_exponent, 80.0);

  auto hostile =
      EvaluateBjt(0.0, 0.0, 0.0, 1.0, std::numeric_limits<double>::denorm_min(),
                  1e100, 1e100, 1e100, 1e100);
  ASSERT_FALSE(hostile.ok());
  EXPECT_EQ(hostile.error().code, ErrorCode::kNonFinite);
}

TEST(Phase3CDeviceTest, JacobianMatchesIndependentFiniteDifferences) {
  constexpr double epsilon = 1e-7;
  const auto evaluate = [](double collector, double base, double emitter) {
    return EvaluateBjt(collector, base, emitter, 1.0, 1e-14, 150.0, 3.0,
                       1.1 * kDiodeThermalVoltageVolts,
                       1.2 * kDiodeThermalVoltageVolts);
  };
  auto center = evaluate(2.0, 0.55, 0.0);
  auto forward_plus = evaluate(2.0 + epsilon, 0.55 + epsilon, 0.0);
  auto forward_minus = evaluate(2.0 - epsilon, 0.55 - epsilon, 0.0);
  auto reverse_plus = evaluate(2.0, 0.55 + epsilon, epsilon);
  auto reverse_minus = evaluate(2.0, 0.55 - epsilon, -epsilon);
  ASSERT_TRUE(center.ok());
  ASSERT_TRUE(forward_plus.ok());
  ASSERT_TRUE(forward_minus.ok());
  ASSERT_TRUE(reverse_plus.ok());
  ASSERT_TRUE(reverse_minus.ok());
  const double collector_vbe =
      (forward_plus.value().collector_current_amperes -
       forward_minus.value().collector_current_amperes) /
      (2.0 * epsilon);
  const double base_vbe = (forward_plus.value().base_current_amperes -
                           forward_minus.value().base_current_amperes) /
                          (2.0 * epsilon);
  const double collector_vbc =
      (reverse_plus.value().collector_current_amperes -
       reverse_minus.value().collector_current_amperes) /
      (2.0 * epsilon);
  const double base_vbc = (reverse_plus.value().base_current_amperes -
                           reverse_minus.value().base_current_amperes) /
                          (2.0 * epsilon);
  EXPECT_NEAR(collector_vbe, center.value().collector_vbe_derivative_siemens,
              std::abs(collector_vbe) * 1e-8 + 1e-18);
  EXPECT_NEAR(base_vbe, center.value().base_vbe_derivative_siemens,
              std::abs(base_vbe) * 1e-8 + 1e-18);
  EXPECT_NEAR(collector_vbc, center.value().collector_vbc_derivative_siemens,
              std::abs(collector_vbc) * 1e-8 + 1e-18);
  EXPECT_NEAR(base_vbc, center.value().base_vbc_derivative_siemens,
              std::abs(base_vbc) * 1e-8 + 1e-18);
}

TEST(Phase3CNewtonTest, MatchesIndependentDiodeConnectedScalarOracle) {
  constexpr char netlist[] = R"(V1 in 0 1
R1 in out 100k
.MODEL QM NPN(IS=1e-14 BF=200 BR=2 NF=1.5 NR=.7)
Q1 out out 0 QM
.OP
)";
  auto simulated = SimulateDc(netlist);
  ASSERT_TRUE(simulated.ok()) << simulated.error().message;
  const double emission = 1.5 * kDiodeThermalVoltageVolts;
  const auto residual = [&](double voltage) {
    return (voltage - 1.0) / 100000.0 + kGminSiemens * voltage +
           1e-14 * std::expm1(voltage / emission);
  };
  const double expected = Bisect(0.0, 1.0, residual);
  EXPECT_NEAR(NodeVoltage(simulated.value(), "out"), expected, 1e-12);
  EXPECT_NEAR(expected, 0.7498191042879521, 1e-14);
  ASSERT_EQ(simulated.value().branch_currents.size(), 1U);
  const double expected_source_current =
      -kGminSiemens - (1.0 - expected) / 100000.0;
  EXPECT_NEAR(simulated.value().branch_currents[0].second,
              expected_source_current, 1e-15);
}

TEST(Phase3CNewtonTest, MatchesIndependentBaseKclAndPnpSymmetry) {
  constexpr char npn_netlist[] = R"(V1 in 0 5
R1 in out 100k
.MODEL QM NPN(IS=1e-14 BF=200 BR=2 NF=1 NR=1)
Q1 in out 0 QM
.OP
)";
  auto npn = SimulateDc(npn_netlist);
  ASSERT_TRUE(npn.ok()) << npn.error().message;
  const auto base_residual = [](double voltage) {
    const double forward =
        1e-14 * std::expm1(voltage / kDiodeThermalVoltageVolts);
    const double reverse =
        1e-14 * std::expm1((voltage - 5.0) / kDiodeThermalVoltageVolts);
    const double base_current = forward / 201.0 + reverse / 3.0;
    return (voltage - 5.0) / 100000.0 + kGminSiemens * voltage + base_current;
  };
  const double expected = Bisect(0.0, 1.0, base_residual);
  EXPECT_NEAR(NodeVoltage(npn.value(), "out"), expected, 1e-12);
  EXPECT_NEAR(expected, 0.7104292785708113, 1e-14);

  constexpr char pnp_netlist[] = R"(V1 in 0 -5
R1 in out 100k
.MODEL QM PNP(IS=1e-14 BF=200 BR=2 NF=1 NR=1)
Q1 in out 0 QM
.OP
)";
  auto pnp = SimulateDc(pnp_netlist);
  ASSERT_TRUE(pnp.ok()) << pnp.error().message;
  EXPECT_DOUBLE_EQ(NodeVoltage(pnp.value(), "in"),
                   -NodeVoltage(npn.value(), "in"));
  EXPECT_DOUBLE_EQ(NodeVoltage(pnp.value(), "out"),
                   -NodeVoltage(npn.value(), "out"));
  ASSERT_EQ(npn.value().branch_currents.size(), 1U);
  ASSERT_EQ(pnp.value().branch_currents.size(), 1U);
  EXPECT_DOUBLE_EQ(pnp.value().branch_currents[0].second,
                   -npn.value().branch_currents[0].second);
}

TEST(Phase3CDeterminismTest, RepeatsTracesResidualAndKluReuseBitwise) {
  MnaSystem system = Compile(R"(V1 in 0 5
R1 in out 100k
.MODEL QM NPN(IS=1e-14 BF=200 BR=2 NF=1 NR=1)
Q1 in out 0 QM
.OP
)");
  auto first = RunNonlinearDc(system);
  auto second = RunNonlinearDc(system);
  ASSERT_TRUE(first.ok()) << first.error().message;
  ASSERT_TRUE(second.ok()) << second.error().message;
  ExpectNonlinearResultsBitwiseEqual(first.value(), second.value());
  EXPECT_EQ(first.value().solver_statistics.symbolic_analyses, 1U);
  EXPECT_GT(first.value().solver_statistics.numeric_refactorizations, 0U);
  auto residual = ValidateNonlinearResidual(system, first.value().solution);
  ASSERT_TRUE(residual.ok()) << residual.error().message;
  std::vector<double> perturbed = first.value().solution;
  perturbed[1] += 1e-3;
  auto rejected = ValidateNonlinearResidual(system, perturbed);
  ASSERT_FALSE(rejected.ok());
  EXPECT_EQ(rejected.error().code, ErrorCode::kSolutionValidation);
}

TEST(Phase3CContinuationTest, ReusesFixedSourceSteppingSchedule) {
  MnaSystem system = Compile(R"(V1 in 0 5
R1 in out 100k
.MODEL QM NPN(IS=1e-14 BF=200 BR=2 NF=1 NR=1)
Q1 in out 0 QM
.OP
)");
  NonlinearDcOptions options;
  options.direct_maximum_iterations = 0;
  auto solved = RunNonlinearDc(system, options);
  auto repeated = RunNonlinearDc(system, options);
  ASSERT_TRUE(solved.ok()) << solved.error().message;
  ASSERT_TRUE(repeated.ok()) << repeated.error().message;
  ExpectNonlinearResultsBitwiseEqual(solved.value(), repeated.value());
  std::vector<double> source_values;
  for (const NonlinearAttemptRecord &attempt : solved.value().attempt_trace) {
    if (attempt.strategy == NonlinearStrategy::kSourceStepping) {
      source_values.push_back(attempt.continuation_value);
    }
  }
  ASSERT_EQ(source_values.size(), 11U);
  for (std::size_t index = 0; index <= 10; ++index) {
    EXPECT_DOUBLE_EQ(source_values[index], static_cast<double>(index) / 10.0);
  }
}

TEST(Phase3CFailureTest, RejectsMalformedDescriptorAndZeroResidualSingularity) {
  MnaSystem malformed = Compile(R"(R1 c 0 1k
R2 b 0 1k
.MODEL QM NPN
Q1 c b 0 QM
.OP
)");
  malformed.bjt_descriptors[0].jacobian_value_indices[0] = std::nullopt;
  auto invalid = RunNonlinearDc(malformed);
  ASSERT_FALSE(invalid.ok());
  EXPECT_EQ(invalid.error().code, ErrorCode::kInvalidStructure);

  auto singular = SimulateDc(R"(V1 a b 0
V2 b c 0
V3 c a 0
.MODEL QM NPN
Q1 a b 0 QM
.OP
)");
  ASSERT_FALSE(singular.ok());
  EXPECT_EQ(singular.error().code, ErrorCode::kSingular);
}

TEST(Phase3CSimulationTest, RejectsAcAndTransientAtEveryExecutionBoundary) {
  constexpr char ac[] = R"(V1 in 0 DC 1 AC 1
.MODEL QM NPN
Q1 in base 0 QM
.AC LIN 2 1 2
)";
  auto ac_result = SimulateAc(ac);
  ASSERT_FALSE(ac_result.ok());
  EXPECT_EQ(ac_result.error().code, ErrorCode::kUnsupported);

  constexpr char transient[] = R"(V1 in 0 PULSE(0 1 0 0 0 1m 2m)
R1 in base 1k
.MODEL QM NPN
Q1 in base 0 QM
.TRAN 1u 2u
)";
  auto transient_result = SimulateTransient(transient);
  ASSERT_FALSE(transient_result.ok());
  EXPECT_EQ(transient_result.error().code, ErrorCode::kUnsupported);
  auto parsed = ParseNetlist(transient);
  ASSERT_TRUE(parsed.ok()) << parsed.error().message;
  auto compiled = CompileMna(parsed.value());
  ASSERT_TRUE(compiled.ok()) << compiled.error().message;
  auto direct_initial = BuildTransientInitialState(compiled.value(), false);
  ASSERT_FALSE(direct_initial.ok());
  EXPECT_EQ(direct_initial.error().code, ErrorCode::kUnsupported);
  auto direct_transient = RunTransientAnalysis(
      compiled.value(), std::get<TranAnalysis>(parsed.value().analyses[0]));
  ASSERT_FALSE(direct_transient.ok());
  EXPECT_EQ(direct_transient.error().code, ErrorCode::kUnsupported);
}

TEST(Phase3CPreservationTest, KeepsLinearAndDiodeResultsExact) {
  constexpr char linear[] = R"(V1 in 0 10
R1 in out 1k
R2 out 0 1k
.OP
)";
  auto linear_csv = SimulateDcToCsv(linear);
  ASSERT_TRUE(linear_csv.ok()) << linear_csv.error().message;
  EXPECT_EQ(linear_csv.value(),
            "Variable,Value\nV(in),10\nV(out),4.9999999975\n"
            "I(V1),-0.005000000012499999\n");

  constexpr char diode[] = R"(V1 supply 0 5
R1 supply out 1k
.MODEL DM D(IS=1e-14 N=1)
D1 out 0 DM
.OP
)";
  auto first = SimulateDcToCsv(diode);
  auto second = SimulateDcToCsv(diode);
  ASSERT_TRUE(first.ok()) << first.error().message;
  ASSERT_TRUE(second.ok()) << second.error().message;
  EXPECT_EQ(first.value(), second.value());

  MnaSystem mixed = Compile(R"(V1 supply 0 5
R1 supply diode 1k
.MODEL DM D
D1 diode 0 DM
R2 supply base 100k
.MODEL QM NPN(IS=1e-14 BF=200)
Q1 supply base 0 QM
.OP
)");
  ASSERT_EQ(mixed.diode_descriptors.size(), 1U);
  ASSERT_EQ(mixed.bjt_descriptors.size(), 1U);
  auto mixed_first = RunNonlinearDc(mixed);
  auto mixed_second = RunNonlinearDc(mixed);
  ASSERT_TRUE(mixed_first.ok()) << mixed_first.error().message;
  ASSERT_TRUE(mixed_second.ok()) << mixed_second.error().message;
  ExpectNonlinearResultsBitwiseEqual(mixed_first.value(), mixed_second.value());
}

} // namespace
} // namespace ohmnivore
