#include "cpp/tests/google_test.h"

#include <string>
#include <variant>
#include <vector>

#include "ohmnivore/compiler.h"
#include "ohmnivore/ir.h"
#include "ohmnivore/parser.h"
#include "ohmnivore/simulator.h"
#include "ohmnivore/solver.h"
#include "ohmnivore/sparse.h"
#include "ohmnivore/status.h"

namespace ohmnivore {
namespace {

TEST(Phase2AParserTest, ParsesLinearDevicesAndDcSourceForms) {
  constexpr char netlist[] = R"(R1 in mid 2k
C1 mid 0 3u
L1 mid out 4m
V1 in 0 5
I1 0 out 2m
.OP
.END
)";

  auto parsed = ParseNetlist(netlist);
  ASSERT_TRUE(parsed.ok()) << parsed.error().message;
  ASSERT_EQ(parsed.value().components.size(), 5U);

  const auto *resistor = std::get_if<Resistor>(&parsed.value().components[0]);
  ASSERT_NE(resistor, nullptr);
  EXPECT_DOUBLE_EQ(resistor->resistance_ohms, 2000.0);

  const auto *capacitor = std::get_if<Capacitor>(&parsed.value().components[1]);
  ASSERT_NE(capacitor, nullptr);
  EXPECT_DOUBLE_EQ(capacitor->capacitance_farads, 3e-6);

  const auto *inductor = std::get_if<Inductor>(&parsed.value().components[2]);
  ASSERT_NE(inductor, nullptr);
  EXPECT_DOUBLE_EQ(inductor->inductance_henries, 4e-3);

  const auto *voltage =
      std::get_if<VoltageSource>(&parsed.value().components[3]);
  ASSERT_NE(voltage, nullptr);
  ASSERT_TRUE(voltage->dc_volts.has_value());
  EXPECT_DOUBLE_EQ(*voltage->dc_volts, 5.0);

  const auto *current =
      std::get_if<CurrentSource>(&parsed.value().components[4]);
  ASSERT_NE(current, nullptr);
  EXPECT_EQ(current->positive_node, "0");
  EXPECT_EQ(current->negative_node, "out");
  ASSERT_TRUE(current->dc_amperes.has_value());
  EXPECT_DOUBLE_EQ(*current->dc_amperes, 2e-3);
}

TEST(Phase2AParserTest, ClassifiesMalformedAndUnsupportedSources) {
  struct Case {
    std::string netlist;
    ErrorCode expected_code;
  };
  const std::vector<Case> cases = {
      {.netlist = "V1 1 0\n.OP\n", .expected_code = ErrorCode::kParse},
      {.netlist = "I1 1 0 DC\n.OP\n", .expected_code = ErrorCode::kParse},
      {.netlist = "V1 1 0 DC 5 extra\n.OP\n",
       .expected_code = ErrorCode::kParse},
      {.netlist = "V1 1 0 SINUSOIDAL\n.OP\n",
       .expected_code = ErrorCode::kParse},
      {.netlist = "V1 1 0 PULSE(0)\n.OP\n", .expected_code = ErrorCode::kParse},
  };

  for (const Case &test_case : cases) {
    SCOPED_TRACE(test_case.netlist);
    auto parsed = ParseNetlist(test_case.netlist);
    ASSERT_FALSE(parsed.ok());
    EXPECT_EQ(parsed.error().code, test_case.expected_code);
    EXPECT_NE(parsed.error().message.find("line 1"), std::string::npos);
  }
}

TEST(Phase2ACompilerTest, BuildsExactDeterministicRlcviCsrSystem) {
  constexpr char netlist[] = R"(V1 in 0 DC 5
R1 in mid 1k
C1 mid out 2u
L1 mid out 3m
I1 0 out DC 4m
R2 out 0 2k
.OP
.END
)";
  auto parsed = ParseNetlist(netlist);
  ASSERT_TRUE(parsed.ok()) << parsed.error().message;
  auto compiled = CompileMna(parsed.value());
  ASSERT_TRUE(compiled.ok()) << compiled.error().message;

  const MnaSystem &system = compiled.value();
  EXPECT_EQ(system.node_names, (std::vector<std::string>{"in", "mid", "out"}));
  EXPECT_EQ(system.branch_names, (std::vector<std::string>{"V1", "L1"}));
  ASSERT_EQ(system.b_dc.size(), 5U);
  EXPECT_DOUBLE_EQ(system.b_dc[0], 0.0);
  EXPECT_DOUBLE_EQ(system.b_dc[1], 0.0);
  EXPECT_DOUBLE_EQ(system.b_dc[2], 4e-3);
  EXPECT_DOUBLE_EQ(system.b_dc[3], 5.0);
  EXPECT_DOUBLE_EQ(system.b_dc[4], 0.0);

  EXPECT_EQ(system.g.row_offsets,
            (std::vector<std::size_t>{0, 3, 6, 8, 9, 11}));
  EXPECT_EQ(system.g.column_indices,
            (std::vector<std::size_t>{0, 1, 3, 0, 1, 4, 2, 4, 0, 1, 2}));
  const std::vector<double> expected_g = {
      1e-3 + kGminSiemens,
      -1e-3,
      1.0,
      -1e-3,
      1e-3 + kGminSiemens,
      1.0,
      5e-4 + kGminSiemens,
      -1.0,
      1.0,
      1.0,
      -1.0,
  };
  ASSERT_EQ(system.g.values.size(), expected_g.size());
  for (std::size_t index = 0; index < expected_g.size(); ++index) {
    EXPECT_DOUBLE_EQ(system.g.values[index], expected_g[index]);
  }

  EXPECT_EQ(system.c.row_offsets, (std::vector<std::size_t>{0, 0, 2, 4, 4, 5}));
  EXPECT_EQ(system.c.column_indices, (std::vector<std::size_t>{1, 2, 1, 2, 4}));
  ASSERT_EQ(system.c.values.size(), 5U);
  EXPECT_DOUBLE_EQ(system.c.values[0], 2e-6);
  EXPECT_DOUBLE_EQ(system.c.values[1], -2e-6);
  EXPECT_DOUBLE_EQ(system.c.values[2], -2e-6);
  EXPECT_DOUBLE_EQ(system.c.values[3], 2e-6);
  EXPECT_DOUBLE_EQ(system.c.values[4], -3e-3);
}

TEST(Phase2ACompilerTest, PreservesNodeAndInterleavedBranchInsertionOrder) {
  constexpr char netlist[] = R"(Lfirst z a 1m
Vmiddle b z 2
Llast c 0 3m
Vlast d 0 4
.OP
.END
)";
  auto parsed = ParseNetlist(netlist);
  ASSERT_TRUE(parsed.ok()) << parsed.error().message;
  auto compiled = CompileMna(parsed.value());
  ASSERT_TRUE(compiled.ok()) << compiled.error().message;

  EXPECT_EQ(compiled.value().node_names,
            (std::vector<std::string>{"z", "a", "b", "c", "d"}));
  EXPECT_EQ(compiled.value().branch_names,
            (std::vector<std::string>{"Lfirst", "Vmiddle", "Llast", "Vlast"}));
}

TEST(Phase2ACompilerTest, AppliesCurrentSourceRhsSignBetweenNodes) {
  Circuit circuit{
      .components = {CurrentSource{.name = "I1",
                                   .positive_node = "plus",
                                   .negative_node = "minus",
                                   .dc_amperes = 3.0,
                                   .ac = {}}},
      .analyses = {DcAnalysis{}},
  };
  auto compiled = CompileMna(circuit);
  ASSERT_TRUE(compiled.ok()) << compiled.error().message;
  EXPECT_EQ(compiled.value().node_names,
            (std::vector<std::string>{"plus", "minus"}));
  EXPECT_EQ(compiled.value().b_dc, (std::vector<double>{-3.0, 3.0}));
}

TEST(Phase2ACompilerTest, TreatsSameNodePassiveAndCurrentDevicesAsNoOps) {
  Circuit circuit{
      .components =
          {
              CurrentSource{.name = "Ibase",
                            .positive_node = "0",
                            .negative_node = "n",
                            .dc_amperes = 1.0,
                            .ac = {}},
              CurrentSource{.name = "Iself",
                            .positive_node = "n",
                            .negative_node = "n",
                            .dc_amperes = 1e16,
                            .ac = {}},
              Resistor{.name = "Rbase",
                       .positive_node = "n",
                       .negative_node = "0",
                       .resistance_ohms = 1.0},
              Resistor{.name = "Rself",
                       .positive_node = "n",
                       .negative_node = "n",
                       .resistance_ohms = 1e-20},
              Capacitor{.name = "Cbase",
                        .positive_node = "n",
                        .negative_node = "0",
                        .capacitance_farads = 1.0},
              Capacitor{.name = "Cself",
                        .positive_node = "n",
                        .negative_node = "n",
                        .capacitance_farads = 1e20},
          },
      .analyses = {DcAnalysis{}},
  };
  auto compiled = CompileMna(circuit);
  ASSERT_TRUE(compiled.ok()) << compiled.error().message;
  EXPECT_EQ(compiled.value().g.row_offsets, (std::vector<std::size_t>{0, 1}));
  EXPECT_EQ(compiled.value().g.column_indices, (std::vector<std::size_t>{0}));
  EXPECT_EQ(compiled.value().g.values,
            (std::vector<double>{1.0 + kGminSiemens}));
  EXPECT_EQ(compiled.value().c.row_offsets, (std::vector<std::size_t>{0, 1}));
  EXPECT_EQ(compiled.value().c.column_indices, (std::vector<std::size_t>{0}));
  EXPECT_EQ(compiled.value().c.values, (std::vector<double>{1.0}));
  EXPECT_EQ(compiled.value().b_dc, (std::vector<double>{1.0}));
}

TEST(Phase2ACompilerTest, RejectsInvalidDirectIrWithTypedCompileError) {
  Circuit circuit{
      .components = {Inductor{.name = "L1",
                              .positive_node = "1",
                              .negative_node = "0",
                              .inductance_henries = -1.0}},
      .analyses = {DcAnalysis{}},
  };
  auto compiled = CompileMna(circuit);
  ASSERT_FALSE(compiled.ok());
  EXPECT_EQ(compiled.error().code, ErrorCode::kCompile);
}

TEST(Phase2ACompilerTest, RejectsNonfiniteAccumulatedMatricesAndRhs) {
  Circuit capacitor_overflow{
      .components =
          {
              VoltageSource{.name = "V1",
                            .positive_node = "n",
                            .negative_node = "0",
                            .dc_volts = 1.0,
                            .ac = {}},
              Capacitor{.name = "C1",
                        .positive_node = "n",
                        .negative_node = "0",
                        .capacitance_farads = 1e308},
              Capacitor{.name = "C2",
                        .positive_node = "n",
                        .negative_node = "0",
                        .capacitance_farads = 1e308},
          },
      .analyses = {DcAnalysis{}},
  };
  auto compiled_capacitors = CompileMna(capacitor_overflow);
  ASSERT_FALSE(compiled_capacitors.ok());
  EXPECT_EQ(compiled_capacitors.error().code, ErrorCode::kCompile);
  EXPECT_NE(compiled_capacitors.error().message.find("dynamic-matrix"),
            std::string::npos);

  Circuit rhs_overflow{
      .components =
          {
              CurrentSource{.name = "I1",
                            .positive_node = "0",
                            .negative_node = "n",
                            .dc_amperes = 1e308,
                            .ac = {}},
              CurrentSource{.name = "I2",
                            .positive_node = "0",
                            .negative_node = "n",
                            .dc_amperes = 1e308,
                            .ac = {}},
              Resistor{.name = "R1",
                       .positive_node = "n",
                       .negative_node = "0",
                       .resistance_ohms = 1.0},
          },
      .analyses = {DcAnalysis{}},
  };
  auto compiled_rhs = CompileMna(rhs_overflow);
  ASSERT_FALSE(compiled_rhs.ok());
  EXPECT_EQ(compiled_rhs.error().code, ErrorCode::kCompile);
  EXPECT_NE(compiled_rhs.error().message.find("right-hand-side"),
            std::string::npos);
}

TEST(Phase2ASolverTest, RejectsNonzeroInitialCsrRowOffset) {
  const CsrMatrix invalid = {
      .rows = 1,
      .columns = 1,
      .values = {1.0},
      .column_indices = {0},
      .row_offsets = {1, 1},
  };
  auto solved = SolveCpuReference(invalid, {1.0});
  ASSERT_FALSE(solved.ok());
  EXPECT_EQ(solved.error().code, ErrorCode::kSolve);
  EXPECT_NE(solved.error().message.find("start at zero"), std::string::npos);
}

TEST(Phase2ASolverTest, RejectsNoncanonicalCsrColumnOrder) {
  const CsrMatrix invalid = {
      .rows = 2,
      .columns = 2,
      .values = {1.0, 2.0},
      .column_indices = {1, 0},
      .row_offsets = {0, 2, 2},
  };
  auto solved = SolveCpuReference(invalid, {1.0, 0.0});
  ASSERT_FALSE(solved.ok());
  EXPECT_EQ(solved.error().code, ErrorCode::kSolve);
  EXPECT_NE(solved.error().message.find("strictly increasing"),
            std::string::npos);
}

TEST(Phase2ASimulationTest, SolvesCurrentSourceOperatingPoint) {
  auto simulated = SimulateDc("I1 0 out DC 1m\nR1 out 0 1k\n.DC\n.END\n");
  ASSERT_TRUE(simulated.ok()) << simulated.error().message;
  ASSERT_EQ(simulated.value().node_voltages.size(), 1U);
  const double expected = 1e-3 / (1e-3 + kGminSiemens);
  EXPECT_NEAR(simulated.value().node_voltages[0].second, expected, 1e-12);
  EXPECT_TRUE(simulated.value().branch_currents.empty());
}

TEST(Phase2ASimulationTest, TreatsCapacitorAsOpenAtDc) {
  auto simulated = SimulateDc("V1 in 0 5\nC1 in 0 1u\n.OP\n.END\n");
  ASSERT_TRUE(simulated.ok()) << simulated.error().message;
  ASSERT_EQ(simulated.value().node_voltages.size(), 1U);
  ASSERT_EQ(simulated.value().branch_currents.size(), 1U);
  EXPECT_DOUBLE_EQ(simulated.value().node_voltages[0].second, 5.0);
  EXPECT_NEAR(simulated.value().branch_currents[0].second, -5.0 * kGminSiemens,
              1e-24);
}

TEST(Phase2ASimulationTest, TreatsInductorAsShortAtDc) {
  constexpr char netlist[] = R"(V1 in 0 5
L1 in out 10m
R1 out 0 1k
.OP
.END
)";
  auto simulated = SimulateDc(netlist);
  ASSERT_TRUE(simulated.ok()) << simulated.error().message;
  ASSERT_EQ(simulated.value().node_voltages.size(), 2U);
  ASSERT_EQ(simulated.value().branch_currents.size(), 2U);
  EXPECT_DOUBLE_EQ(simulated.value().node_voltages[0].second, 5.0);
  EXPECT_DOUBLE_EQ(simulated.value().node_voltages[1].second, 5.0);
  EXPECT_EQ(simulated.value().branch_currents[0].first, "V1");
  EXPECT_EQ(simulated.value().branch_currents[1].first, "L1");
  EXPECT_NEAR(simulated.value().branch_currents[1].second,
              5.0 * (1e-3 + kGminSiemens), 1e-15);
}

TEST(Phase2ASimulationTest, SolvesCombinedRlcviOperatingPoint) {
  constexpr char netlist[] = R"(V1 supply 0 5
R1 supply mid 1k
C1 mid 0 1u
L1 mid out 1m
I1 0 out 1m
R2 out 0 1k
.OP
.END
)";
  auto simulated = SimulateDc(netlist);
  ASSERT_TRUE(simulated.ok()) << simulated.error().message;
  ASSERT_EQ(simulated.value().node_voltages.size(), 3U);
  const double expected = 6e-3 / (2e-3 + 2.0 * kGminSiemens);
  EXPECT_NEAR(simulated.value().node_voltages[1].second, expected, 1e-10);
  EXPECT_NEAR(simulated.value().node_voltages[2].second, expected, 1e-10);
}

} // namespace
} // namespace ohmnivore
