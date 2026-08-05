#include "cpp/tests/google_test.h"

#include <sstream>
#include <string>

#include "ohmnivore/compiler.h"
#include "ohmnivore/simulator.h"
#include "ohmnivore/status.h"

namespace ohmnivore {
namespace {

TEST(Phase2CSimulationTest, UsesDcOperatingPointWithoutUic) {
  constexpr char netlist[] = R"(V1 in 0 DC 5
R1 in out 1k
C1 out 0 1u
.TRAN 100u 1m
.END
)";
  auto simulated = SimulateTransient(netlist);
  ASSERT_TRUE(simulated.ok()) << simulated.error().message;
  ASSERT_FALSE(simulated.value().times_seconds.empty());
  EXPECT_DOUBLE_EQ(simulated.value().times_seconds.front(), 0.0);
  EXPECT_DOUBLE_EQ(simulated.value().times_seconds.back(), 1e-3);
  ASSERT_EQ(simulated.value().node_voltages.size(), 2U);
  EXPECT_NEAR(simulated.value().node_voltages[1].second.front(),
              5.0 * 1e-3 / (1e-3 + kGminSiemens), 1e-10);
  EXPECT_NEAR(simulated.value().node_voltages[1].second.back(),
              simulated.value().node_voltages[1].second.front(), 1e-10);
}

TEST(Phase2CSimulationTest, EmitsTransientCsvInIrOrderWithEscaping) {
  constexpr char netlist[] = R"(V"1 in,node 0 DC 5
R1 in,node out 1k
C1 out 0 1u
L1 out sense 1m
R2 sense 0 10
.TRAN 10u 20u UIC
.END
)";
  auto csv = SimulateTransientToCsv(netlist);
  ASSERT_TRUE(csv.ok()) << csv.error().message;
  std::istringstream rows(csv.value());
  std::string row;
  ASSERT_TRUE(std::getline(rows, row));
  EXPECT_EQ(row, "time,\"V(in,node)\",V(out),V(sense),\"I(V\"\"1)\",I(L1)");
  ASSERT_TRUE(std::getline(rows, row));
  EXPECT_EQ(row.substr(0, row.find(',')), "0");
  EXPECT_NE(row.find(",5,"), std::string::npos);
  std::string last_row = row;
  while (std::getline(rows, row)) {
    last_row = row;
  }
  EXPECT_DOUBLE_EQ(std::stod(last_row.substr(0, last_row.find(','))), 20e-6);
}

TEST(Phase2CSimulationTest, PreservesAnalysisExecutionOrder) {
  constexpr char netlist[] = R"(V1 in 0 DC 1 AC 1 PULSE(0 1 0 0 0 1 2)
R1 in 0 1k
.OP
.AC LIN 2 1 2
.TRAN 0.5 1 UIC
.END
)";
  auto csv = SimulateToCsv(netlist);
  ASSERT_TRUE(csv.ok()) << csv.error().message;
  const std::size_t dc = csv.value().find("Variable,Value\n");
  const std::size_t ac = csv.value().find("Frequency,V(in)_mag");
  const std::size_t transient = csv.value().find("time,V(in),I(V1)\n");
  ASSERT_NE(dc, std::string::npos);
  ASSERT_NE(ac, std::string::npos);
  ASSERT_NE(transient, std::string::npos);
  EXPECT_LT(dc, ac);
  EXPECT_LT(ac, transient);
}

TEST(Phase2CSimulationTest, RequiresTransientAnalysisBeforeCompilation) {
  auto simulated = SimulateTransient("V1 n n PULSE(0 1)\n.OP\n.END\n");
  ASSERT_FALSE(simulated.ok());
  EXPECT_EQ(simulated.error().code, ErrorCode::kUnsupported);
}

TEST(Phase2CSimulationTest, PreservesCurrentAndBranchSignsWithGmin) {
  constexpr char netlist[] = R"(V1 drive 0 PULSE(0 2 0 0 0 1 2)
R1 drive load 1k
I1 load 0 PULSE(0 1m 0 0 0 1 2)
R2 load 0 1k
.TRAN 0.1 0.1 UIC
.END
)";
  auto simulated = SimulateTransient(netlist);
  ASSERT_TRUE(simulated.ok()) << simulated.error().message;
  ASSERT_EQ(simulated.value().times_seconds.size(), 2U);
  ASSERT_EQ(simulated.value().branch_currents.size(), 1U);
  const double load = simulated.value().node_voltages[1].second.back();
  const double expected_load = (2e-3 - 1e-3) / (2e-3 + kGminSiemens);
  EXPECT_NEAR(load, expected_load, 1e-12);
  const double expected_source_current =
      -kGminSiemens * 2.0 - (2.0 - expected_load) / 1000.0;
  EXPECT_NEAR(simulated.value().branch_currents[0].second.back(),
              expected_source_current, 1e-14);
}

} // namespace
} // namespace ohmnivore
