#include "cpp/tests/google_test.h"

#include <sstream>
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

constexpr char kVoltageDivider[] = R"(* Voltage Divider
V1 1 0 DC 10
R1 1 2 1k
R2 2 0 1k
.DC
.END
)";

TEST(Phase1ParserTest, PreservesComponentsAndEngineeringValues) {
  auto parsed = ParseNetlist(kVoltageDivider);
  ASSERT_TRUE(parsed.ok()) << parsed.error().message;
  ASSERT_EQ(parsed.value().components.size(), 3U);
  ASSERT_EQ(parsed.value().analyses.size(), 1U);

  const auto *source =
      std::get_if<VoltageSource>(&parsed.value().components[0]);
  ASSERT_NE(source, nullptr);
  EXPECT_EQ(source->name, "V1");
  EXPECT_DOUBLE_EQ(source->dc_volts, 10.0);

  const auto *resistor = std::get_if<Resistor>(&parsed.value().components[1]);
  ASSERT_NE(resistor, nullptr);
  EXPECT_EQ(resistor->name, "R1");
  EXPECT_DOUBLE_EQ(resistor->resistance_ohms, 1000.0);
}

TEST(Phase1ParserTest, RejectsUnsupportedElementsExplicitly) {
  auto parsed = ParseNetlist("C1 1 0 1u\n.DC\n.END\n");
  ASSERT_FALSE(parsed.ok());
  EXPECT_EQ(parsed.error().code, ErrorCode::kUnsupported);
  EXPECT_NE(parsed.error().message.find("line 1"), std::string::npos);
}

TEST(Phase1ParserTest, RecognizesPrintAsCompatibilityNoOp) {
  auto parsed =
      ParseNetlist("V1 1 0 5\nR1 1 0 1k\n.OP\n.PRINT OP V(1)\n.END\n");
  ASSERT_TRUE(parsed.ok()) << parsed.error().message;
  EXPECT_EQ(parsed.value().components.size(), 2U);
  EXPECT_EQ(parsed.value().analyses.size(), 1U);
}

TEST(Phase1CompilerTest, BuildsCanonicalVoltageDividerMna) {
  auto parsed = ParseNetlist(kVoltageDivider);
  ASSERT_TRUE(parsed.ok()) << parsed.error().message;
  auto compiled = CompileMna(parsed.value());
  ASSERT_TRUE(compiled.ok()) << compiled.error().message;

  EXPECT_EQ(compiled.value().node_names, (std::vector<std::string>{"1", "2"}));
  EXPECT_EQ(compiled.value().branch_names, (std::vector<std::string>{"V1"}));
  EXPECT_EQ(compiled.value().b_dc, (std::vector<double>{0.0, 0.0, 10.0}));

  const std::vector<double> dense = compiled.value().g.ToDense();
  ASSERT_EQ(dense.size(), 9U);
  EXPECT_NEAR(dense[0], 0.001 + kGminSiemens, 1e-15);
  EXPECT_NEAR(dense[1], -0.001, 1e-15);
  EXPECT_DOUBLE_EQ(dense[2], 1.0);
  EXPECT_NEAR(dense[3], -0.001, 1e-15);
  EXPECT_NEAR(dense[4], 0.002 + kGminSiemens, 1e-15);
  EXPECT_DOUBLE_EQ(dense[5], 0.0);
  EXPECT_DOUBLE_EQ(dense[6], 1.0);
  EXPECT_DOUBLE_EQ(dense[7], 0.0);
  EXPECT_DOUBLE_EQ(dense[8], 0.0);
}

TEST(Phase1CompilerTest, TreatsGndCaseInsensitivelyAsGround) {
  auto parsed = ParseNetlist("V1 1 Gnd 5\nR1 1 GND 1k\n.DC\n.END\n");
  ASSERT_TRUE(parsed.ok()) << parsed.error().message;
  auto compiled = CompileMna(parsed.value());
  ASSERT_TRUE(compiled.ok()) << compiled.error().message;
  EXPECT_EQ(compiled.value().node_names, (std::vector<std::string>{"1"}));
}

TEST(Phase1SimulationTest, MatchesVoltageDividerBehaviorAndCsvSchema) {
  auto simulated = SimulateDc(kVoltageDivider);
  ASSERT_TRUE(simulated.ok()) << simulated.error().message;
  ASSERT_EQ(simulated.value().node_voltages.size(), 2U);
  ASSERT_EQ(simulated.value().branch_currents.size(), 1U);
  EXPECT_EQ(simulated.value().node_voltages[0].first, "1");
  EXPECT_NEAR(simulated.value().node_voltages[0].second, 10.0, 1e-9);
  EXPECT_EQ(simulated.value().node_voltages[1].first, "2");
  EXPECT_NEAR(simulated.value().node_voltages[1].second, 5.0, 1e-8);
  EXPECT_EQ(simulated.value().branch_currents[0].first, "V1");
  EXPECT_NEAR(simulated.value().branch_currents[0].second, -0.005, 1e-9);

  auto csv = SimulateDcToCsv(kVoltageDivider);
  ASSERT_TRUE(csv.ok()) << csv.error().message;
  std::istringstream rows(csv.value());
  std::string row;
  ASSERT_TRUE(std::getline(rows, row));
  EXPECT_EQ(row, "Variable,Value");
  ASSERT_TRUE(std::getline(rows, row));
  EXPECT_EQ(row.substr(0, row.find(',')), "V(1)");
  ASSERT_TRUE(std::getline(rows, row));
  EXPECT_EQ(row.substr(0, row.find(',')), "V(2)");
  ASSERT_TRUE(std::getline(rows, row));
  EXPECT_EQ(row.substr(0, row.find(',')), "I(V1)");
  EXPECT_FALSE(std::getline(rows, row));
}

TEST(Phase1SimulationTest, RequiresDcAnalysis) {
  auto simulated = SimulateDc("V1 1 0 5\nR1 1 0 1k\n.END\n");
  ASSERT_FALSE(simulated.ok());
  EXPECT_EQ(simulated.error().code, ErrorCode::kUnsupported);
}

TEST(Phase1SolverTest, RejectsOutOfBoundsCsrColumnBeforeDenseConversion) {
  const CsrMatrix invalid = {
      .rows = 1,
      .columns = 1,
      .values = {1.0},
      .column_indices = {1},
      .row_offsets = {0, 1},
  };
  auto solved = SolveCpuReference(invalid, {1.0});
  ASSERT_FALSE(solved.ok());
  EXPECT_EQ(solved.error().code, ErrorCode::kSolve);
}

} // namespace
} // namespace ohmnivore
