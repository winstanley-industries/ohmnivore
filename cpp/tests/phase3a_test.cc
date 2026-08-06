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

[[nodiscard]] double IndependentResistorDiodeRoot(double source_volts,
                                                  double resistance_ohms,
                                                  double saturation_current,
                                                  double ideality_factor) {
  const double emission = ideality_factor * kDiodeThermalVoltageVolts;
  const auto residual = [&](double voltage) {
    return (voltage - source_volts) / resistance_ohms + kGminSiemens * voltage +
           saturation_current * std::expm1(voltage / emission);
  };
  double lower = -1.0;
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

void *RejectAllocation(std::size_t) { return nullptr; }
void *RejectCalloc(std::size_t, std::size_t) { return nullptr; }
void *RejectReallocation(void *, std::size_t) { return nullptr; }

TEST(Phase3AParserTest, ParsesStrictDiodesModelsDefaultsCaseAndOrder) {
  constexpr char netlist[] = R"(.model First d(n=1.752 is=2.52e-9)
Dfirst a 0 fIrSt
.MODEL Defaulted D()
dsecond b a DEFAULTED
.MODEL Bare D
Dthird c b bare
.MODEL IsOnly D(IS=3e-12)
Dfourth d c isonly
.MODEL NOnly D(N=2)
Dfifth e d nonly
.MODEL m_1 D
Dsixth f e M_1
.OP
)";
  auto parsed = ParseNetlist(netlist);
  ASSERT_TRUE(parsed.ok()) << parsed.error().message;
  ASSERT_EQ(parsed.value().components.size(), 6U);
  ASSERT_EQ(parsed.value().diode_models.size(), 6U);
  EXPECT_EQ(parsed.value().diode_models[0].name, "First");
  EXPECT_DOUBLE_EQ(parsed.value().diode_models[0].saturation_current_amperes,
                   2.52e-9);
  EXPECT_DOUBLE_EQ(parsed.value().diode_models[0].ideality_factor, 1.752);
  EXPECT_DOUBLE_EQ(parsed.value().diode_models[1].saturation_current_amperes,
                   1e-14);
  EXPECT_DOUBLE_EQ(parsed.value().diode_models[1].ideality_factor, 1.0);
  EXPECT_DOUBLE_EQ(parsed.value().diode_models[2].saturation_current_amperes,
                   1e-14);
  EXPECT_DOUBLE_EQ(parsed.value().diode_models[3].saturation_current_amperes,
                   3e-12);
  EXPECT_DOUBLE_EQ(parsed.value().diode_models[3].ideality_factor, 1.0);
  EXPECT_DOUBLE_EQ(parsed.value().diode_models[4].saturation_current_amperes,
                   1e-14);
  EXPECT_DOUBLE_EQ(parsed.value().diode_models[4].ideality_factor, 2.0);
  const auto *first = std::get_if<Diode>(&parsed.value().components[0]);
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(first->name, "Dfirst");
  EXPECT_EQ(first->positive_node, "a");
  EXPECT_EQ(first->negative_node, "0");
  EXPECT_EQ(first->model_name, "fIrSt");
}

TEST(Phase3AParserTest, RejectsMalformedDuplicateUnsupportedAndInvalidModels) {
  struct Case {
    std::string line;
    ErrorCode code;
  };
  const std::vector<Case> cases = {
      {"D1 a 0", ErrorCode::kParse},
      {"D1 a 0 DM extra", ErrorCode::kParse},
      {".MODEL", ErrorCode::kParse},
      {".MODEL DM", ErrorCode::kParse},
      {".MODEL DM D(", ErrorCode::kParse},
      {".MODEL DM D(IS=1e-14", ErrorCode::kParse},
      {".MODEL DM D((IS=1e-14))", ErrorCode::kParse},
      {".MODEL DM D(IS=1e-14) trailing", ErrorCode::kParse},
      {".MODEL DM D(IS)", ErrorCode::kParse},
      {".MODEL DM D(IS=1e-14 is=2e-14)", ErrorCode::kParse},
      {".MODEL DM D(N=1 N=2)", ErrorCode::kParse},
      {".MODEL DM D(IS=0)", ErrorCode::kParse},
      {".MODEL DM D(IS=-1)", ErrorCode::kParse},
      {".MODEL DM D(N=0)", ErrorCode::kParse},
      {".MODEL DM D(N=-1)", ErrorCode::kParse},
      {".MODEL DM D(N=1e-323)", ErrorCode::kParse},
      {".MODEL DM D(IS=1e309)", ErrorCode::kParse},
      {".MODEL DM D(IS=1e101)", ErrorCode::kParse},
      {".MODEL DM D(N=1e101)", ErrorCode::kParse},
      {".MODEL bad-name D", ErrorCode::kParse},
      {".MODEL foo( D", ErrorCode::kParse},
      {"D1 a 0 bad-name", ErrorCode::kParse},
      {".MODEL DM D(RS=1)", ErrorCode::kUnsupported},
      {".MODEL QM NMOS(KP=1e-3)", ErrorCode::kUnsupported},
      {".MODEL DM DIODE", ErrorCode::kUnsupported},
      {".MODEL DM D(IS=1e-14,N=1)", ErrorCode::kParse},
      {".MODEL DM D)", ErrorCode::kParse},
  };
  for (const Case &test_case : cases) {
    SCOPED_TRACE(test_case.line);
    auto parsed = ParseNetlist(test_case.line + "\n.OP\n");
    ASSERT_FALSE(parsed.ok());
    EXPECT_EQ(parsed.error().code, test_case.code);
    EXPECT_NE(parsed.error().message.find("line 1"), std::string::npos);
  }

  auto duplicate = ParseNetlist(".MODEL Dmod D\n.MODEL dMOD D()\n.OP\n");
  ASSERT_FALSE(duplicate.ok());
  EXPECT_EQ(duplicate.error().code, ErrorCode::kParse);
  EXPECT_NE(duplicate.error().message.find("duplicate"), std::string::npos);
}

TEST(Phase3ACompilerTest, BuildsCanonicalUnionPatternAndOrderedDescriptors) {
  MnaSystem system = Compile(R"(R1 a 0 1k
R2 b 0 2k
.MODEL DM D(IS=1e-14 N=1)
Dfirst a b dm
Dsecond b 0 DM
.OP
)");
  EXPECT_EQ(system.node_names, (std::vector<std::string>{"a", "b"}));
  EXPECT_EQ(system.g.row_offsets, (std::vector<std::size_t>{0, 2, 4}));
  EXPECT_EQ(system.g.column_indices, (std::vector<std::size_t>{0, 1, 0, 1}));
  ASSERT_EQ(system.g.values.size(), 4U);
  EXPECT_DOUBLE_EQ(system.g.values[0], 1e-3 + kGminSiemens);
  EXPECT_DOUBLE_EQ(system.g.values[1], 0.0);
  EXPECT_DOUBLE_EQ(system.g.values[2], 0.0);
  EXPECT_DOUBLE_EQ(system.g.values[3], 5e-4 + kGminSiemens);
  ASSERT_EQ(system.diode_descriptors.size(), 2U);
  EXPECT_EQ(system.diode_descriptors[0].name, "Dfirst");
  EXPECT_EQ(system.diode_descriptors[1].name, "Dsecond");
  EXPECT_DOUBLE_EQ(system.diode_descriptors[0].saturation_current_amperes,
                   1e-14);
  EXPECT_DOUBLE_EQ(system.diode_descriptors[0].emission_voltage_volts,
                   kDiodeThermalVoltageVolts);
  EXPECT_EQ(system.diode_descriptors[0].anode_anode_value_index, 0U);
  EXPECT_EQ(system.diode_descriptors[0].anode_cathode_value_index, 1U);
  EXPECT_EQ(system.diode_descriptors[0].cathode_anode_value_index, 2U);
  EXPECT_EQ(system.diode_descriptors[0].cathode_cathode_value_index, 3U);
  EXPECT_EQ(system.diode_descriptors[1].anode_anode_value_index, 3U);
  EXPECT_FALSE(
      system.diode_descriptors[1].cathode_cathode_value_index.has_value());

  auto csc = ConvertCsrToSolverCsc(system.g);
  ASSERT_TRUE(csc.ok()) << csc.error().message;
  EXPECT_EQ(csc.value().column_offsets, (std::vector<std::int32_t>{0, 2, 4}));
  EXPECT_EQ(csc.value().row_indices, (std::vector<std::int32_t>{0, 1, 0, 1}));
}

TEST(Phase3ACompilerTest, BuildsExactGroundAndTwoNodeJacobianResidualStamps) {
  MnaSystem ground = Compile(R"(R1 a 0 1k
.MODEL DM D(IS=1e-12 N=1)
D1 a 0 DM
.OP
)");
  const double voltage = 0.1;
  auto diode = EvaluateDiode(voltage, 1e-12, kDiodeThermalVoltageVolts);
  ASSERT_TRUE(diode.ok()) << diode.error().message;
  auto ground_linearization = BuildNonlinearDcLinearization(ground, {voltage});
  ASSERT_TRUE(ground_linearization.ok())
      << ground_linearization.error().message;
  EXPECT_DOUBLE_EQ(ground_linearization.value().jacobian.values[0],
                   1e-3 + kGminSiemens + diode.value().conductance_siemens);
  EXPECT_DOUBLE_EQ(ground_linearization.value().residual[0],
                   (1e-3 + kGminSiemens) * voltage +
                       diode.value().current_amperes);

  MnaSystem floating = Compile(R"(R1 a 0 1k
R2 b 0 2k
.MODEL DM D(IS=1e-12 N=1)
D1 a b DM
.OP
)");
  const std::vector<double> state = {0.2, -0.1};
  auto two_node = BuildNonlinearDcLinearization(floating, state);
  ASSERT_TRUE(two_node.ok()) << two_node.error().message;
  auto evaluation =
      EvaluateDiode(state[0] - state[1], 1e-12, kDiodeThermalVoltageVolts);
  ASSERT_TRUE(evaluation.ok());
  EXPECT_DOUBLE_EQ(two_node.value().jacobian.values[0],
                   1e-3 + kGminSiemens +
                       evaluation.value().conductance_siemens);
  EXPECT_DOUBLE_EQ(two_node.value().jacobian.values[1],
                   -evaluation.value().conductance_siemens);
  EXPECT_DOUBLE_EQ(two_node.value().jacobian.values[2],
                   -evaluation.value().conductance_siemens);
  EXPECT_DOUBLE_EQ(two_node.value().jacobian.values[3],
                   5e-4 + kGminSiemens +
                       evaluation.value().conductance_siemens);
  EXPECT_DOUBLE_EQ(two_node.value().residual[0],
                   (1e-3 + kGminSiemens) * 0.2 +
                       evaluation.value().current_amperes);
  EXPECT_DOUBLE_EQ(two_node.value().residual[1],
                   (5e-4 + kGminSiemens) * -0.1 -
                       evaluation.value().current_amperes);

  MnaSystem anode_ground = Compile(R"(R1 b 0 2k
.MODEL DM D(IS=1e-12 N=1)
D1 0 b DM
.OP
)");
  auto anode_ground_evaluation =
      EvaluateDiode(-0.1, 1e-12, kDiodeThermalVoltageVolts);
  ASSERT_TRUE(anode_ground_evaluation.ok());
  auto anode_ground_linearization =
      BuildNonlinearDcLinearization(anode_ground, {0.1});
  ASSERT_TRUE(anode_ground_linearization.ok())
      << anode_ground_linearization.error().message;
  EXPECT_DOUBLE_EQ(anode_ground_linearization.value().jacobian.values[0],
                   5e-4 + kGminSiemens +
                       anode_ground_evaluation.value().conductance_siemens);
  EXPECT_DOUBLE_EQ(anode_ground_linearization.value().residual[0],
                   (5e-4 + kGminSiemens) * 0.1 -
                       anode_ground_evaluation.value().current_amperes);

  MnaSystem parallel = Compile(R"(R1 a 0 1k
R2 b 0 2k
.MODEL DM1 D(IS=1e-12 N=1)
.MODEL DM2 D(IS=2e-12 N=2)
D1 a b DM1
D2 a b DM2
.OP
)");
  const std::vector<double> parallel_state = {0.2, -0.1};
  const double parallel_junction = parallel_state[0] - parallel_state[1];
  auto first_parallel =
      EvaluateDiode(parallel_junction, 1e-12, kDiodeThermalVoltageVolts);
  auto second_parallel =
      EvaluateDiode(parallel_junction, 2e-12, 2.0 * kDiodeThermalVoltageVolts);
  ASSERT_TRUE(first_parallel.ok());
  ASSERT_TRUE(second_parallel.ok());
  auto parallel_linearization =
      BuildNonlinearDcLinearization(parallel, parallel_state);
  ASSERT_TRUE(parallel_linearization.ok())
      << parallel_linearization.error().message;
  double parallel_anode_diagonal = 1e-3 + kGminSiemens;
  parallel_anode_diagonal += first_parallel.value().conductance_siemens;
  parallel_anode_diagonal += second_parallel.value().conductance_siemens;
  double parallel_cathode_diagonal = 5e-4 + kGminSiemens;
  parallel_cathode_diagonal += first_parallel.value().conductance_siemens;
  parallel_cathode_diagonal += second_parallel.value().conductance_siemens;
  double parallel_off_diagonal = 0.0;
  parallel_off_diagonal -= first_parallel.value().conductance_siemens;
  parallel_off_diagonal -= second_parallel.value().conductance_siemens;
  double parallel_anode_residual = (1e-3 + kGminSiemens) * 0.2;
  parallel_anode_residual += first_parallel.value().current_amperes;
  parallel_anode_residual += second_parallel.value().current_amperes;
  double parallel_cathode_residual = (5e-4 + kGminSiemens) * -0.1;
  parallel_cathode_residual -= first_parallel.value().current_amperes;
  parallel_cathode_residual -= second_parallel.value().current_amperes;
  EXPECT_DOUBLE_EQ(parallel_linearization.value().jacobian.values[0],
                   parallel_anode_diagonal);
  EXPECT_DOUBLE_EQ(parallel_linearization.value().jacobian.values[1],
                   parallel_off_diagonal);
  EXPECT_DOUBLE_EQ(parallel_linearization.value().jacobian.values[2],
                   parallel_off_diagonal);
  EXPECT_DOUBLE_EQ(parallel_linearization.value().jacobian.values[3],
                   parallel_cathode_diagonal);
  EXPECT_DOUBLE_EQ(parallel_linearization.value().residual[0],
                   parallel_anode_residual);
  EXPECT_DOUBLE_EQ(parallel_linearization.value().residual[1],
                   parallel_cathode_residual);
}

TEST(Phase3ACompilerTest, RejectsMissingInvalidDuplicateAndMalformedDirectIr) {
  auto missing = ParseNetlist("D1 a 0 missing\n.OP\n");
  ASSERT_TRUE(missing.ok());
  auto missing_compiled = CompileMna(missing.value());
  ASSERT_FALSE(missing_compiled.ok());
  EXPECT_EQ(missing_compiled.error().code, ErrorCode::kCompile);

  Circuit duplicate{
      .components = {Diode{.name = "D1",
                           .positive_node = "a",
                           .negative_node = "0",
                           .model_name = "dm"}},
      .analyses = {DcAnalysis{}},
      .diode_models = {DiodeModel{.name = "DM"}, DiodeModel{.name = "dm"}},
  };
  auto duplicate_compiled = CompileMna(duplicate);
  ASSERT_FALSE(duplicate_compiled.ok());
  EXPECT_EQ(duplicate_compiled.error().code, ErrorCode::kCompile);

  Circuit invalid = duplicate;
  invalid.diode_models.resize(1);
  invalid.diode_models[0].saturation_current_amperes = -1.0;
  auto invalid_compiled = CompileMna(invalid);
  ASSERT_FALSE(invalid_compiled.ok());
  EXPECT_EQ(invalid_compiled.error().code, ErrorCode::kCompile);

  invalid.diode_models[0].saturation_current_amperes = 1e101;
  auto over_bound_model = CompileMna(invalid);
  ASSERT_FALSE(over_bound_model.ok());
  EXPECT_EQ(over_bound_model.error().code, ErrorCode::kCompile);

  invalid.diode_models[0].saturation_current_amperes = 1e-14;
  invalid.diode_models[0].ideality_factor =
      std::numeric_limits<double>::infinity();
  auto non_finite_ideality = CompileMna(invalid);
  ASSERT_FALSE(non_finite_ideality.ok());
  EXPECT_EQ(non_finite_ideality.error().code, ErrorCode::kCompile);

  invalid.diode_models[0].ideality_factor = 1e101;
  auto over_bound_ideality = CompileMna(invalid);
  ASSERT_FALSE(over_bound_ideality.ok());
  EXPECT_EQ(over_bound_ideality.error().code, ErrorCode::kCompile);

  invalid.diode_models[0].ideality_factor = 1.0;
  invalid.diode_models[0].name.clear();
  auto missing_model_name = CompileMna(invalid);
  ASSERT_FALSE(missing_model_name.ok());
  EXPECT_EQ(missing_model_name.error().code, ErrorCode::kCompile);

  invalid.diode_models[0].name = "bad name";
  auto malformed_model_identifier = CompileMna(invalid);
  ASSERT_FALSE(malformed_model_identifier.ok());
  EXPECT_EQ(malformed_model_identifier.error().code, ErrorCode::kCompile);

  invalid.diode_models[0].name = "DM";
  std::get<Diode>(invalid.components[0]).model_name = "bad reference";
  auto malformed_model_reference = CompileMna(invalid);
  ASSERT_FALSE(malformed_model_reference.ok());
  EXPECT_EQ(malformed_model_reference.error().code, ErrorCode::kCompile);

  invalid.components[0] = Diode{.name = "D1",
                                .positive_node = "a",
                                .negative_node = "a",
                                .model_name = "DM"};
  auto same_node = CompileMna(invalid);
  ASSERT_FALSE(same_node.ok());
  EXPECT_EQ(same_node.error().code, ErrorCode::kInvalidStructure);

  Circuit same_ground_circuit{
      .components = {Resistor{.name = "R1",
                              .positive_node = "a",
                              .negative_node = "0",
                              .resistance_ohms = 1000.0},
                     Diode{.name = "D1",
                           .positive_node = "0",
                           .negative_node = "GND",
                           .model_name = "DM"}},
      .analyses = {DcAnalysis{}},
      .diode_models = {DiodeModel{.name = "DM"}},
  };
  auto same_ground = CompileMna(same_ground_circuit);
  ASSERT_FALSE(same_ground.ok());
  EXPECT_EQ(same_ground.error().code, ErrorCode::kInvalidStructure);

  MnaSystem malformed = Compile(R"(R1 a 0 1k
.MODEL DM D
D1 a 0 DM
.OP
)");
  malformed.diode_descriptors[0].name.clear();
  auto missing_descriptor_name = RunNonlinearDc(malformed);
  ASSERT_FALSE(missing_descriptor_name.ok());
  EXPECT_EQ(missing_descriptor_name.error().code, ErrorCode::kInvalidStructure);

  malformed.diode_descriptors[0].name = "D1";
  malformed.diode_descriptors[0].anode_anode_value_index = std::nullopt;
  auto rejected = RunNonlinearDc(malformed);
  ASSERT_FALSE(rejected.ok());
  EXPECT_EQ(rejected.error().code, ErrorCode::kInvalidStructure);

  MnaSystem invalid_parameters = Compile(R"(R1 a 0 1k
.MODEL DM D
D1 a 0 DM
.OP
)");
  invalid_parameters.diode_descriptors[0].emission_voltage_volts = 1e99;
  auto over_bound_emission = RunNonlinearDc(invalid_parameters);
  ASSERT_FALSE(over_bound_emission.ok());
  EXPECT_EQ(over_bound_emission.error().code, ErrorCode::kCompile);

  invalid_parameters.diode_descriptors[0].emission_voltage_volts =
      kDiodeThermalVoltageVolts;
  invalid_parameters.diode_descriptors[0].saturation_current_amperes =
      std::numeric_limits<double>::quiet_NaN();
  auto non_finite_descriptor = RunNonlinearDc(invalid_parameters);
  ASSERT_FALSE(non_finite_descriptor.ok());
  EXPECT_EQ(non_finite_descriptor.error().code, ErrorCode::kCompile);
}

TEST(Phase3ADeviceTest, EvaluatesZeroForwardReverseAndExponentBoundaries) {
  constexpr double saturation = 1e-14;
  constexpr double emission = kDiodeThermalVoltageVolts;
  auto zero = EvaluateDiode(0.0, saturation, emission);
  ASSERT_TRUE(zero.ok());
  EXPECT_DOUBLE_EQ(zero.value().current_amperes, 0.0);
  EXPECT_DOUBLE_EQ(zero.value().conductance_siemens, saturation / emission);

  auto forward = EvaluateDiode(emission, saturation, emission);
  ASSERT_TRUE(forward.ok());
  EXPECT_NEAR(forward.value().current_amperes, saturation * std::expm1(1.0),
              1e-28);
  EXPECT_NEAR(forward.value().conductance_siemens,
              saturation * std::exp(1.0) / emission, 1e-26);

  auto reverse = EvaluateDiode(-10.0 * emission, saturation, emission);
  ASSERT_TRUE(reverse.ok());
  EXPECT_NEAR(reverse.value().current_amperes, saturation * std::expm1(-10.0),
              1e-28);

  auto underflow = EvaluateDiode(-1e6, saturation, emission);
  auto overflow = EvaluateDiode(1e6, saturation, emission);
  ASSERT_TRUE(underflow.ok());
  ASSERT_TRUE(overflow.ok());
  EXPECT_DOUBLE_EQ(underflow.value().exponent, -80.0);
  EXPECT_DOUBLE_EQ(overflow.value().exponent, 80.0);
  EXPECT_GT(underflow.value().conductance_siemens, 0.0);
  EXPECT_TRUE(std::isfinite(overflow.value().current_amperes));

  auto hostile = EvaluateDiode(1.0, 1e90, emission);
  ASSERT_FALSE(hostile.ok());
  EXPECT_EQ(hostile.error().code, ErrorCode::kNonFinite);
  auto invalid = EvaluateDiode(0.0, 0.0, emission);
  ASSERT_FALSE(invalid.ok());
  EXPECT_EQ(invalid.error().code, ErrorCode::kNonFinite);
  auto underflowed_model =
      EvaluateDiode(0.0, std::numeric_limits<double>::denorm_min(), 1e100);
  ASSERT_FALSE(underflowed_model.ok());
  EXPECT_EQ(underflowed_model.error().code, ErrorCode::kNonFinite);
  auto over_bound_model = EvaluateDiode(0.0, 1e101, emission);
  ASSERT_FALSE(over_bound_model.ok());
  EXPECT_EQ(over_bound_model.error().code, ErrorCode::kNonFinite);
  auto over_bound_exponent = EvaluateDiode(1e100, saturation, emission);
  ASSERT_FALSE(over_bound_exponent.ok());
  EXPECT_EQ(over_bound_exponent.error().code, ErrorCode::kNonFinite);
}

TEST(Phase3ALimitingTest, AppliesDeterministicPnjlimAndRejectsHostileValues) {
  constexpr double saturation = 1e-14;
  constexpr double emission = kDiodeThermalVoltageVolts;
  auto limited = LimitDiodeJunctionVoltage(5.0, 0.0, saturation, emission);
  ASSERT_TRUE(limited.ok()) << limited.error().message;
  EXPECT_LT(limited.value(), 1.0);
  EXPECT_GT(limited.value(), 0.0);
  auto unchanged = LimitDiodeJunctionVoltage(-1.0, 0.0, saturation, emission);
  ASSERT_TRUE(unchanged.ok());
  EXPECT_DOUBLE_EQ(unchanged.value(), -1.0);
  auto repeated = LimitDiodeJunctionVoltage(5.0, 0.0, saturation, emission);
  ASSERT_TRUE(repeated.ok());
  EXPECT_EQ(std::bit_cast<std::uint64_t>(limited.value()),
            std::bit_cast<std::uint64_t>(repeated.value()));

  auto nonfinite = LimitDiodeJunctionVoltage(
      std::numeric_limits<double>::infinity(), 0.0, saturation, emission);
  ASSERT_FALSE(nonfinite.ok());
  EXPECT_EQ(nonfinite.error().code, ErrorCode::kNonFinite);
  auto over_bound_difference =
      LimitDiodeJunctionVoltage(1e100, -1e100, saturation, emission);
  ASSERT_FALSE(over_bound_difference.ok());
  EXPECT_EQ(over_bound_difference.error().code, ErrorCode::kNonFinite);
}

TEST(Phase3ANewtonTest, MatchesIndependentResistorDiodeScalarOracle) {
  constexpr char netlist[] = R"(V1 supply 0 5
R1 supply out 1k
.MODEL DM D(IS=1e-14 N=1)
D1 out 0 DM
.OP
)";
  auto simulated = SimulateDc(netlist);
  ASSERT_TRUE(simulated.ok()) << simulated.error().message;
  const double expected = IndependentResistorDiodeRoot(5.0, 1000.0, 1e-14, 1.0);
  EXPECT_NEAR(NodeVoltage(simulated.value(), "out"), expected, 1e-12);
  ASSERT_EQ(simulated.value().branch_currents.size(), 1U);
  const double expected_source_current =
      -kGminSiemens * 5.0 - (5.0 - expected) / 1000.0;
  EXPECT_NEAR(simulated.value().branch_currents[0].second,
              expected_source_current, 1e-14);
}

TEST(Phase3ANewtonTest, MatchesIndependentCurrentDrivenDiodeOracle) {
  constexpr char netlist[] = R"(I1 0 out 1m
.MODEL DM D(IS=1e-14 N=1)
D1 out 0 DM
.OP
)";
  auto simulated = SimulateDc(netlist);
  ASSERT_TRUE(simulated.ok()) << simulated.error().message;
  const double emission = kDiodeThermalVoltageVolts;
  const auto residual = [&](double voltage) {
    return kGminSiemens * voltage + 1e-14 * std::expm1(voltage / emission) -
           1e-3;
  };
  double lower = 0.0;
  double upper = 1.0;
  for (std::size_t iteration = 0; iteration < 200; ++iteration) {
    const double middle = std::midpoint(lower, upper);
    if (residual(middle) > 0.0) {
      upper = middle;
    } else {
      lower = middle;
    }
  }
  EXPECT_NEAR(NodeVoltage(simulated.value(), "out"),
              std::midpoint(lower, upper), 1e-12);
}

TEST(Phase3ANewtonTest, CoversZeroReverseOrdinaryAndStrongForwardBias) {
  auto zero = SimulateDc(R"(R1 out 0 1k
.MODEL DM D
D1 out 0 DM
.OP
)");
  ASSERT_TRUE(zero.ok()) << zero.error().message;
  EXPECT_DOUBLE_EQ(NodeVoltage(zero.value(), "out"), 0.0);

  auto reverse = SimulateDc(R"(V1 supply 0 -2
R1 supply out 1k
.MODEL DM D
D1 out 0 DM
.OP
)");
  ASSERT_TRUE(reverse.ok()) << reverse.error().message;
  EXPECT_NEAR(NodeVoltage(reverse.value(), "out"), -2.0, 1e-8);

  for (double source : {1.0, 100.0}) {
    const std::string netlist =
        "V1 supply 0 " + std::to_string(source) +
        "\nR1 supply out 1k\n.MODEL DM D\nD1 out 0 DM\n.OP\n";
    auto forward = SimulateDc(netlist);
    ASSERT_TRUE(forward.ok()) << forward.error().message;
    const double expected =
        IndependentResistorDiodeRoot(source, 1000.0, 1e-14, 1.0);
    EXPECT_NEAR(NodeVoltage(forward.value(), "out"), expected, 1e-11);
  }
}

TEST(Phase3ANewtonTest, AcceptsBadlyScaledMixedUnitMnaSystem) {
  constexpr char netlist[] = R"(V1 supply 0 1
Rlarge supply out 1T
Itrim 0 out 1p
.MODEL DM D(IS=1e-18 N=2)
D1 out 0 DM
.OP
)";
  auto simulated = SimulateDc(netlist);
  ASSERT_TRUE(simulated.ok()) << simulated.error().message;
  EXPECT_TRUE(std::isfinite(NodeVoltage(simulated.value(), "out")));
  MnaSystem system = Compile(netlist);
  auto nonlinear = RunNonlinearDc(system);
  ASSERT_TRUE(nonlinear.ok()) << nonlinear.error().message;
  auto residual = ValidateNonlinearResidual(system, nonlinear.value().solution);
  ASSERT_TRUE(residual.ok()) << residual.error().message;
}

TEST(Phase3ADeterminismTest, RepeatsResultsTracesAndKluReuseBitwise) {
  MnaSystem system = Compile(R"(V1 supply 0 5
R1 supply out 1k
.MODEL DM D(IS=1e-14 N=1)
D1 out 0 DM
.OP
)");
  auto first = RunNonlinearDc(system);
  auto second = RunNonlinearDc(system);
  ASSERT_TRUE(first.ok()) << first.error().message;
  ASSERT_TRUE(second.ok()) << second.error().message;
  ExpectNonlinearResultsBitwiseEqual(first.value(), second.value());
  EXPECT_EQ(first.value().solver_statistics.symbolic_analyses, 1U);
  EXPECT_GT(first.value().solver_statistics.numeric_refactorizations, 0U);
}

TEST(Phase3AContinuationTest, RecordsFixedSourceSteppingSchedule) {
  MnaSystem system = Compile(R"(V1 supply 0 5
R1 supply out 1k
.MODEL DM D
D1 out 0 DM
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
  for (const auto &attempt : solved.value().attempt_trace) {
    if (attempt.strategy == NonlinearStrategy::kSourceStepping) {
      source_values.push_back(attempt.continuation_value);
    }
  }
  ASSERT_EQ(source_values.size(), 11U);
  for (std::size_t index = 0; index <= 10; ++index) {
    EXPECT_DOUBLE_EQ(source_values[index], static_cast<double>(index) / 10.0);
  }
  EXPECT_EQ(solved.value().attempt_trace.front().strategy,
            NonlinearStrategy::kDirect);
  EXPECT_TRUE(std::any_of(
      solved.value().iteration_trace.begin(),
      solved.value().iteration_trace.end(), [](const auto &record) {
        return record.strategy == NonlinearStrategy::kSourceStepping;
      }));
  auto residual = ValidateNonlinearResidual(system, solved.value().solution);
  ASSERT_TRUE(residual.ok()) << residual.error().message;
}

TEST(Phase3AContinuationTest, RecordsFixedGminScheduleAfterSourceFailure) {
  MnaSystem system = Compile(R"(V1 supply 0 5
R1 supply out 1k
.MODEL DM D
D1 out 0 DM
.OP
)");
  NonlinearDcOptions options;
  options.direct_maximum_iterations = 0;
  options.source_step_maximum_iterations = 0;
  auto solved = RunNonlinearDc(system, options);
  auto repeated = RunNonlinearDc(system, options);
  ASSERT_TRUE(solved.ok()) << solved.error().message;
  ASSERT_TRUE(repeated.ok()) << repeated.error().message;
  ExpectNonlinearResultsBitwiseEqual(solved.value(), repeated.value());
  const std::vector<double> expected = {1e-3, 1e-4,  1e-5,  1e-6,  1e-7, 1e-8,
                                        1e-9, 1e-10, 1e-11, 1e-12, 0.0};
  std::vector<double> values;
  for (const auto &attempt : solved.value().attempt_trace) {
    if (attempt.strategy == NonlinearStrategy::kGminStepping) {
      values.push_back(attempt.continuation_value);
    }
  }
  EXPECT_EQ(values, expected);
  EXPECT_TRUE(
      std::any_of(solved.value().iteration_trace.begin(),
                  solved.value().iteration_trace.end(), [](const auto &record) {
                    return record.strategy == NonlinearStrategy::kGminStepping;
                  }));
  auto residual = ValidateNonlinearResidual(system, solved.value().solution);
  ASSERT_TRUE(residual.ok()) << residual.error().message;
  EXPECT_EQ(solved.value().solver_statistics.symbolic_analyses, 1U);
}

TEST(Phase3AFailureTest, RejectsHostileResidualAndExhaustedContinuation) {
  MnaSystem system = Compile(R"(V1 supply 0 5
R1 supply out 1k
.MODEL DM D
D1 out 0 DM
.OP
)");
  auto hostile = ValidateNonlinearResidual(
      system, std::vector<double>(system.g.rows, 0.0));
  ASSERT_FALSE(hostile.ok());
  EXPECT_EQ(hostile.error().code, ErrorCode::kSolutionValidation);

  const MnaSystem cancellation{
      .g = CsrMatrix{.rows = 2,
                     .columns = 2,
                     .values = {6e99, -6e99, 1.0},
                     .column_indices = {0, 1, 1},
                     .row_offsets = {0, 2, 3}},
      .c = CsrMatrix{.rows = 2,
                     .columns = 2,
                     .values = {},
                     .column_indices = {},
                     .row_offsets = {0, 0, 0}},
      .b_dc = {0.0, 0.0},
      .b_ac = {{0.0, 0.0}, {0.0, 0.0}},
      .node_names = {"a", "b"},
      .branch_names = {},
      .diode_descriptors = {DiodeDescriptor{
          .name = "D1",
          .anode_node_index = 0,
          .cathode_node_index = std::nullopt,
          .saturation_current_amperes = 1e-14,
          .emission_voltage_volts = kDiodeThermalVoltageVolts,
          .anode_anode_value_index = 0,
          .anode_cathode_value_index = std::nullopt,
          .cathode_anode_value_index = std::nullopt,
          .cathode_cathode_value_index = std::nullopt,
      }},
  };
  auto over_bound_intermediate =
      BuildNonlinearDcLinearization(cancellation, {1.0, 1.0});
  ASSERT_FALSE(over_bound_intermediate.ok());
  EXPECT_EQ(over_bound_intermediate.error().code, ErrorCode::kNonFinite);

  MnaSystem over_bound_base = system;
  over_bound_base.g.values[0] = 1e101;
  auto rejected_base = BuildNonlinearDcLinearization(
      over_bound_base, std::vector<double>(system.g.rows, 0.0));
  ASSERT_FALSE(rejected_base.ok());
  EXPECT_EQ(rejected_base.error().code, ErrorCode::kNonFinite);

  MnaSystem over_bound_rhs = system;
  over_bound_rhs.b_dc[0] = 1e101;
  auto rejected_rhs = ValidateNonlinearResidual(
      over_bound_rhs, std::vector<double>(system.g.rows, 0.0), 0.0, 0.0);
  ASSERT_FALSE(rejected_rhs.ok());
  EXPECT_EQ(rejected_rhs.error().code, ErrorCode::kNonFinite);

  NonlinearDcOptions options;
  options.direct_maximum_iterations = 0;
  options.source_step_maximum_iterations = 0;
  options.gmin_step_maximum_iterations = 0;
  options.final_gmin_maximum_iterations = 0;
  auto exhausted = RunNonlinearDc(system, options);
  ASSERT_FALSE(exhausted.ok());
  EXPECT_EQ(exhausted.error().code, ErrorCode::kNonConvergence);

  options.direct_maximum_iterations = std::numeric_limits<std::size_t>::max();
  auto unbounded = RunNonlinearDc(system, options);
  ASSERT_FALSE(unbounded.ok());
  EXPECT_EQ(unbounded.error().code, ErrorCode::kInvalidStructure);
}

TEST(Phase3AFailureTest, PropagatesSingularAndSparseAllocationFailures) {
  auto zero_residual_singular = SimulateDc(R"(V1 a b 0
V2 b c 0
V3 c a 0
.MODEL DM D
D1 a 0 DM
.OP
)");
  ASSERT_FALSE(zero_residual_singular.ok());
  EXPECT_EQ(zero_residual_singular.error().code, ErrorCode::kSingular);

  const MnaSystem singular{
      .g = CsrMatrix{.rows = 2,
                     .columns = 2,
                     .values = {0.0},
                     .column_indices = {0},
                     .row_offsets = {0, 1, 1}},
      .c = CsrMatrix{.rows = 2,
                     .columns = 2,
                     .values = {},
                     .column_indices = {},
                     .row_offsets = {0, 0, 0}},
      .b_dc = {1.0, 0.0},
      .b_ac = {{0.0, 0.0}, {0.0, 0.0}},
      .node_names = {"n"},
      .branch_names = {"Vbad"},
      .diode_descriptors = {DiodeDescriptor{
          .name = "D1",
          .anode_node_index = 0,
          .cathode_node_index = std::nullopt,
          .saturation_current_amperes = 1e-14,
          .emission_voltage_volts = kDiodeThermalVoltageVolts,
          .anode_anode_value_index = 0,
          .anode_cathode_value_index = std::nullopt,
          .cathode_anode_value_index = std::nullopt,
          .cathode_cathode_value_index = std::nullopt,
      }},
  };
  auto singular_result = RunNonlinearDc(singular);
  ASSERT_FALSE(singular_result.ok());
  EXPECT_EQ(singular_result.error().code, ErrorCode::kSingular);

  MnaSystem valid = Compile(R"(V1 supply 0 5
R1 supply out 1k
.MODEL DM D
D1 out 0 DM
.OP
)");
  const auto original_malloc = SuiteSparse_config_malloc_func_get();
  const auto original_calloc = SuiteSparse_config_calloc_func_get();
  const auto original_realloc = SuiteSparse_config_realloc_func_get();
  SuiteSparse_config_malloc_func_set(RejectAllocation);
  SuiteSparse_config_calloc_func_set(RejectCalloc);
  SuiteSparse_config_realloc_func_set(RejectReallocation);
  auto allocation_failure = RunNonlinearDc(valid);
  SuiteSparse_config_malloc_func_set(original_malloc);
  SuiteSparse_config_calloc_func_set(original_calloc);
  SuiteSparse_config_realloc_func_set(original_realloc);
  ASSERT_FALSE(allocation_failure.ok());
  EXPECT_EQ(allocation_failure.error().code, ErrorCode::kFactorization);
}

TEST(Phase3ASimulationTest, KeepsNonlinearAcExplicitlyUnsupported) {
  constexpr char ac[] = R"(V1 in 0 DC 1 AC 1
.MODEL DM D
D1 in 0 DM
.AC LIN 2 1 2
)";
  auto ac_result = SimulateAc(ac);
  ASSERT_FALSE(ac_result.ok());
  EXPECT_EQ(ac_result.error().code, ErrorCode::kUnsupported);
}

TEST(Phase3ASimulationTest, PreservesOrderingSignsGminAndCsvEscaping) {
  constexpr char netlist[] = R"(V"1 in,node 0 5
R1 in,node out 1k
.MODEL DM D(IS=1e-14 N=1)
D1 out 0 DM
.OP
)";
  auto simulated = SimulateDc(netlist);
  ASSERT_TRUE(simulated.ok()) << simulated.error().message;
  EXPECT_EQ(simulated.value().node_voltages[0].first, "in,node");
  EXPECT_EQ(simulated.value().node_voltages[1].first, "out");
  EXPECT_EQ(simulated.value().branch_currents[0].first, "V\"1");
  EXPECT_LT(simulated.value().branch_currents[0].second, 0.0);
  auto csv = SimulateDcToCsv(netlist);
  ASSERT_TRUE(csv.ok()) << csv.error().message;
  EXPECT_NE(csv.value().find("\"V(in,node)\","), std::string::npos);
  EXPECT_NE(csv.value().find("\"I(V\"\"1)\","), std::string::npos);
}

TEST(Phase3ALinearPreservationTest, RetainsExactLinearCsrFastPathAndCsv) {
  constexpr char netlist[] = R"(V1 in 0 10
R1 in out 1k
R2 out 0 1k
.OP
)";
  MnaSystem system = Compile(netlist);
  EXPECT_TRUE(system.diode_descriptors.empty());
  EXPECT_EQ(system.g.row_offsets, (std::vector<std::size_t>{0, 3, 5, 6}));
  EXPECT_EQ(system.g.column_indices,
            (std::vector<std::size_t>{0, 1, 2, 0, 1, 0}));
  auto csv = SimulateDcToCsv(netlist);
  ASSERT_TRUE(csv.ok()) << csv.error().message;
  EXPECT_EQ(csv.value(), "Variable,Value\nV(in),10\nV(out),4.9999999975\n"
                         "I(V1),-0.005000000012499999\n");
}

} // namespace
} // namespace ohmnivore
