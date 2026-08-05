#include "cpp/tests/google_test.h"

#include <cmath>
#include <complex>
#include <limits>
#include <numbers>
#include <sstream>
#include <stdexcept>
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

void ExpectComplexNear(std::complex<double> actual,
                       std::complex<double> expected, double tolerance) {
  EXPECT_NEAR(actual.real(), expected.real(), tolerance);
  EXPECT_NEAR(actual.imag(), expected.imag(), tolerance);
}

[[nodiscard]] const std::vector<std::complex<double>> &
FindNodeValues(const AcResult &result, const std::string &name) {
  for (const auto &[candidate, values] : result.node_voltages) {
    if (candidate == name) {
      return values;
    }
  }
  throw std::runtime_error("missing AC node result: " + name);
}

[[nodiscard]] const std::vector<std::complex<double>> &
FindBranchValues(const AcResult &result, const std::string &name) {
  for (const auto &[candidate, values] : result.branch_currents) {
    if (candidate == name) {
      return values;
    }
  }
  throw std::runtime_error("missing AC branch result: " + name);
}

[[nodiscard]] std::vector<std::string> SplitCsvRow(const std::string &row) {
  std::istringstream stream(row);
  std::vector<std::string> columns;
  for (std::string column; std::getline(stream, column, ',');) {
    columns.push_back(std::move(column));
  }
  return columns;
}

TEST(Phase2BParserTest, ParsesDcOnlyAcOnlyAndCombinedSources) {
  constexpr char netlist[] = R"(Vbare n1 0 5
IDC n2 0 DC 2m
Vac n3 0 AC 3
Iac n4 0 AC 4m -90
Vcombined n5 0 DC -2 AC 6 45
Ibarecombined n6 0 1m AC 2m 180
.AC DEC 10 1 1k
.END
)";

  auto parsed = ParseNetlist(netlist);
  ASSERT_TRUE(parsed.ok()) << parsed.error().message;
  ASSERT_EQ(parsed.value().components.size(), 6U);

  const auto *bare = std::get_if<VoltageSource>(&parsed.value().components[0]);
  ASSERT_NE(bare, nullptr);
  ASSERT_TRUE(bare->dc_volts.has_value());
  EXPECT_DOUBLE_EQ(*bare->dc_volts, 5.0);
  EXPECT_FALSE(bare->ac.has_value());

  const auto *dc = std::get_if<CurrentSource>(&parsed.value().components[1]);
  ASSERT_NE(dc, nullptr);
  ASSERT_TRUE(dc->dc_amperes.has_value());
  EXPECT_DOUBLE_EQ(*dc->dc_amperes, 2e-3);
  EXPECT_FALSE(dc->ac.has_value());

  const auto *ac = std::get_if<VoltageSource>(&parsed.value().components[2]);
  ASSERT_NE(ac, nullptr);
  EXPECT_FALSE(ac->dc_volts.has_value());
  ASSERT_TRUE(ac->ac.has_value());
  EXPECT_DOUBLE_EQ(ac->ac->magnitude, 3.0);
  EXPECT_DOUBLE_EQ(ac->ac->phase_degrees, 0.0);

  const auto *combined =
      std::get_if<VoltageSource>(&parsed.value().components[4]);
  ASSERT_NE(combined, nullptr);
  ASSERT_TRUE(combined->dc_volts.has_value());
  ASSERT_TRUE(combined->ac.has_value());
  EXPECT_DOUBLE_EQ(*combined->dc_volts, -2.0);
  EXPECT_DOUBLE_EQ(combined->ac->magnitude, 6.0);
  EXPECT_DOUBLE_EQ(combined->ac->phase_degrees, 45.0);

  const auto *bare_combined =
      std::get_if<CurrentSource>(&parsed.value().components[5]);
  ASSERT_NE(bare_combined, nullptr);
  ASSERT_TRUE(bare_combined->dc_amperes.has_value());
  ASSERT_TRUE(bare_combined->ac.has_value());
  EXPECT_DOUBLE_EQ(*bare_combined->dc_amperes, 1e-3);
  EXPECT_DOUBLE_EQ(bare_combined->ac->magnitude, 2e-3);
  EXPECT_DOUBLE_EQ(bare_combined->ac->phase_degrees, 180.0);
}

TEST(Phase2BParserTest, RejectsMalformedSourcesWithFullTokenConsumption) {
  const std::vector<std::string> netlists = {
      "V1 1 0\n.OP\n",
      "I1 1 0 DC\n.OP\n",
      "V1 1 0 AC\n.AC LIN 2 1 2\n",
      "I1 1 0 AC nope\n.AC LIN 2 1 2\n",
      "V1 1 0 AC -1\n.AC LIN 2 1 2\n",
      "V1 1 0 DC 1 trailing\n.OP\n",
      "I1 1 0 AC 1 0 trailing\n.AC LIN 2 1 2\n",
      "V1 1 0 DC 1 DC 2\n.OP\n",
      "I1 1 0 AC 1 DC 2\n.AC LIN 2 1 2\n",
      "V1 1 0 +-1\n.OP\n",
      "I1 1 0 ++1\n.OP\n",
      "V1 1 0 AC 1 +-90\n.AC LIN 2 1 2\n",
      "I1 1 0 AC 1 ++90\n.AC LIN 2 1 2\n",
  };
  for (const std::string &netlist : netlists) {
    SCOPED_TRACE(netlist);
    auto parsed = ParseNetlist(netlist);
    ASSERT_FALSE(parsed.ok());
    EXPECT_EQ(parsed.error().code, ErrorCode::kParse);
    EXPECT_NE(parsed.error().message.find("line 1"), std::string::npos);
  }
}

TEST(Phase2BParserTest, AcceptsLegacyLeadingPlusEngineeringValues) {
  constexpr char netlist[] = R"(V1 in 0 DC +2 AC +3 +90
I1 0 in +4m AC +5m +180
.AC LIN 2 +1 +2
.END
)";
  auto parsed = ParseNetlist(netlist);
  ASSERT_TRUE(parsed.ok()) << parsed.error().message;
  const auto *voltage =
      std::get_if<VoltageSource>(&parsed.value().components[0]);
  ASSERT_NE(voltage, nullptr);
  EXPECT_EQ(voltage->dc_volts, 2.0);
  ASSERT_TRUE(voltage->ac.has_value());
  EXPECT_DOUBLE_EQ(voltage->ac->magnitude, 3.0);
  EXPECT_DOUBLE_EQ(voltage->ac->phase_degrees, 90.0);
  const auto *analysis =
      std::get_if<AcAnalysis>(&parsed.value().analyses.front());
  ASSERT_NE(analysis, nullptr);
  EXPECT_DOUBLE_EQ(analysis->start_frequency_hz, 1.0);
  EXPECT_DOUBLE_EQ(analysis->stop_frequency_hz, 2.0);

  auto bare_plus = ParseNetlist("V1 in 0 +\n.OP\n");
  ASSERT_FALSE(bare_plus.ok());
  EXPECT_EQ(bare_plus.error().code, ErrorCode::kParse);
}

TEST(Phase2BParserTest, RejectsMalformedRecognizedTransientWaveforms) {
  const std::vector<std::string> specifications = {
      "PULSE(0)",
      "SIN(0 1)",
      "PWL(0 0 0 1)",
      "EXP(0 1 0 0)",
      "DC 1 AC 2 0 PULSE(0 1) trailing",
  };
  for (const std::string &specification : specifications) {
    SCOPED_TRACE(specification);
    auto parsed = ParseNetlist("V1 1 0 " + specification + "\n.OP\n");
    ASSERT_FALSE(parsed.ok());
    EXPECT_EQ(parsed.error().code, ErrorCode::kParse);
    EXPECT_NE(parsed.error().message.find("line 1"), std::string::npos);
  }
}

TEST(Phase2BParserTest, ParsesTypedAcAnalysis) {
  auto parsed = ParseNetlist(".ac oct 7 10 10k\n.END\n");
  ASSERT_TRUE(parsed.ok()) << parsed.error().message;
  ASSERT_EQ(parsed.value().analyses.size(), 1U);
  const auto *analysis =
      std::get_if<AcAnalysis>(&parsed.value().analyses.front());
  ASSERT_NE(analysis, nullptr);
  EXPECT_EQ(analysis->sweep_type, AcSweepType::kOct);
  EXPECT_EQ(analysis->points, 7U);
  EXPECT_DOUBLE_EQ(analysis->start_frequency_hz, 10.0);
  EXPECT_DOUBLE_EQ(analysis->stop_frequency_hz, 10000.0);
}

TEST(Phase2BParserTest, ValidatesAcAnalysisStrictly) {
  const std::vector<std::string> directives = {
      ".AC",
      ".AC LOG 10 1 10",
      ".AC DEC 0 1 10",
      ".AC DEC 1.5 1 10",
      ".AC LIN 1 1 10",
      ".AC OCT 2 0 10",
      ".AC DEC 2 -1 10",
      ".AC DEC 2 10 10",
      ".AC DEC 2 20 10",
      ".AC DEC 2 1 10 extra",
  };
  for (const std::string &directive : directives) {
    SCOPED_TRACE(directive);
    auto parsed = ParseNetlist(directive + "\n.END\n");
    ASSERT_FALSE(parsed.ok());
    EXPECT_EQ(parsed.error().code, ErrorCode::kParse);
    EXPECT_NE(parsed.error().message.find("line 1"), std::string::npos);
  }
}

TEST(Phase2BCompilerTest, BuildsExactDcAndComplexAcRightHandSides) {
  constexpr char netlist[] = R"(V1 drive 0 DC 5 AC 2 90
I1 plus minus DC 4 AC 3 -90
R1 drive plus 1k
R2 minus 0 2k
.AC LIN 2 10 20
.END
)";
  auto parsed = ParseNetlist(netlist);
  ASSERT_TRUE(parsed.ok()) << parsed.error().message;
  auto compiled = CompileMna(parsed.value());
  ASSERT_TRUE(compiled.ok()) << compiled.error().message;

  const MnaSystem &system = compiled.value();
  EXPECT_EQ(system.node_names,
            (std::vector<std::string>{"drive", "plus", "minus"}));
  EXPECT_EQ(system.branch_names, (std::vector<std::string>{"V1"}));
  EXPECT_EQ(system.b_dc, (std::vector<double>{0.0, -4.0, 4.0, 5.0}));
  ASSERT_EQ(system.b_ac.size(), 4U);
  ExpectComplexNear(system.b_ac[0], {0.0, 0.0}, 0.0);
  ExpectComplexNear(system.b_ac[1], {0.0, 3.0}, 1e-15);
  ExpectComplexNear(system.b_ac[2], {0.0, -3.0}, 1e-15);
  ExpectComplexNear(system.b_ac[3], {0.0, 2.0}, 1e-15);
}

TEST(Phase2BCompilerTest, MergesIndependentGAndCCsrPatternsExactly) {
  const CsrMatrix g = {
      .rows = 3,
      .columns = 3,
      .values = {1.0, 2.0},
      .column_indices = {0, 2},
      .row_offsets = {0, 1, 2, 2},
  };
  const CsrMatrix c = {
      .rows = 3,
      .columns = 3,
      .values = {3.0, 4.0, 5.0},
      .column_indices = {1, 2, 0},
      .row_offsets = {0, 1, 2, 3},
  };

  auto formed = FormAcMatrix(g, c, 2.0);
  ASSERT_TRUE(formed.ok()) << formed.error().message;
  EXPECT_EQ(formed.value().row_offsets, (std::vector<std::size_t>{0, 2, 3, 4}));
  EXPECT_EQ(formed.value().column_indices,
            (std::vector<std::size_t>{0, 1, 2, 0}));
  EXPECT_EQ(formed.value().values,
            (std::vector<std::complex<double>>{
                {1.0, 0.0}, {0.0, 6.0}, {2.0, 8.0}, {0.0, 10.0}}));
}

TEST(Phase2BCompilerTest, RejectsInvalidDirectSourceIr) {
  Circuit circuit{
      .components = {VoltageSource{.name = "V1",
                                   .positive_node = "1",
                                   .negative_node = "0",
                                   .dc_volts = {},
                                   .ac = {}}},
      .analyses = {DcAnalysis{}},
  };
  auto compiled = CompileMna(circuit);
  ASSERT_FALSE(compiled.ok());
  EXPECT_EQ(compiled.error().code, ErrorCode::kCompile);
}

TEST(Phase2BCompilerTest, ReducesHugeFiniteAcPhasesBeforeConversion) {
  constexpr char netlist[] = R"(V1 drive 0 AC 1 1e308
I1 drive load AC 2 1e308
R1 load 0 1
.AC LIN 2 1 2
.END
)";
  auto parsed = ParseNetlist(netlist);
  ASSERT_TRUE(parsed.ok()) << parsed.error().message;
  auto compiled = CompileMna(parsed.value());
  ASSERT_TRUE(compiled.ok()) << compiled.error().message;
  for (std::complex<double> value : compiled.value().b_ac) {
    EXPECT_TRUE(std::isfinite(value.real()));
    EXPECT_TRUE(std::isfinite(value.imag()));
  }
  EXPECT_NEAR(std::abs(compiled.value().b_ac.back()), 1.0, 1e-15);
}

TEST(Phase2BFrequencyTest, GeneratesOrderedInclusiveDecOctAndLinEndpoints) {
  auto dec = GenerateAcFrequencies(AcAnalysis{
      .sweep_type = AcSweepType::kDec,
      .points = 2,
      .start_frequency_hz = 1.0,
      .stop_frequency_hz = 100.0,
  });
  ASSERT_TRUE(dec.ok()) << dec.error().message;
  ASSERT_EQ(dec.value().size(), 5U);
  EXPECT_DOUBLE_EQ(dec.value().front(), 1.0);
  EXPECT_NEAR(dec.value()[1], std::sqrt(10.0), 1e-14);
  EXPECT_DOUBLE_EQ(dec.value()[2], 10.0);
  EXPECT_NEAR(dec.value()[3], 10.0 * std::sqrt(10.0), 1e-13);
  EXPECT_DOUBLE_EQ(dec.value().back(), 100.0);

  auto oct = GenerateAcFrequencies(AcAnalysis{
      .sweep_type = AcSweepType::kOct,
      .points = 2,
      .start_frequency_hz = 1.0,
      .stop_frequency_hz = 4.0,
  });
  ASSERT_TRUE(oct.ok()) << oct.error().message;
  ASSERT_EQ(oct.value().size(), 5U);
  EXPECT_DOUBLE_EQ(oct.value().front(), 1.0);
  EXPECT_NEAR(oct.value()[1], std::sqrt(2.0), 1e-14);
  EXPECT_DOUBLE_EQ(oct.value()[2], 2.0);
  EXPECT_NEAR(oct.value()[3], 2.0 * std::sqrt(2.0), 1e-14);
  EXPECT_DOUBLE_EQ(oct.value().back(), 4.0);

  auto lin = GenerateAcFrequencies(AcAnalysis{
      .sweep_type = AcSweepType::kLin,
      .points = 3,
      .start_frequency_hz = 1.0,
      .stop_frequency_hz = 3.0,
  });
  ASSERT_TRUE(lin.ok()) << lin.error().message;
  EXPECT_EQ(lin.value(), (std::vector<double>{1.0, 2.0, 3.0}));

  auto partial_dec = GenerateAcFrequencies(AcAnalysis{
      .sweep_type = AcSweepType::kDec,
      .points = 10,
      .start_frequency_hz = 1.0,
      .stop_frequency_hz = 2.0,
  });
  ASSERT_TRUE(partial_dec.ok()) << partial_dec.error().message;
  EXPECT_EQ(partial_dec.value().size(), 5U);
  EXPECT_DOUBLE_EQ(partial_dec.value().front(), 1.0);
  EXPECT_DOUBLE_EQ(partial_dec.value().back(), 2.0);
  for (std::size_t index = 1; index < partial_dec.value().size(); ++index) {
    EXPECT_LT(partial_dec.value()[index - 1], partial_dec.value()[index]);
  }

  for (std::size_t density : {1U, 2U, 3U, 7U, 10U, 31U, 100U}) {
    SCOPED_TRACE(density);
    auto exact_decades = GenerateAcFrequencies(AcAnalysis{
        .sweep_type = AcSweepType::kDec,
        .points = density,
        .start_frequency_hz = 1e3,
        .stop_frequency_hz = 1e6,
    });
    ASSERT_TRUE(exact_decades.ok()) << exact_decades.error().message;
    ASSERT_EQ(exact_decades.value().size(), density * 3 + 1);
    EXPECT_DOUBLE_EQ(exact_decades.value().front(), 1e3);
    EXPECT_DOUBLE_EQ(exact_decades.value().back(), 1e6);
    for (std::size_t index = 1; index < exact_decades.value().size(); ++index) {
      EXPECT_LT(exact_decades.value()[index - 1], exact_decades.value()[index]);
    }
  }
}

TEST(Phase2BFrequencyTest, RejectsInvalidSweepTypeAndCollapsedLogPoints) {
  auto invalid = GenerateAcFrequencies(AcAnalysis{
      .sweep_type = static_cast<AcSweepType>(99),
      .points = 1,
      .start_frequency_hz = 1.0,
      .stop_frequency_hz = 2.0,
  });
  ASSERT_FALSE(invalid.ok());
  EXPECT_EQ(invalid.error().code, ErrorCode::kParse);

  constexpr std::size_t dense_points = 1'000'000'000'000'000'000ULL;
  const double adjacent = std::nextafter(1.0, 2.0);
  for (AcSweepType sweep : {AcSweepType::kDec, AcSweepType::kOct}) {
    auto sparse = GenerateAcFrequencies(AcAnalysis{
        .sweep_type = sweep,
        .points = 1,
        .start_frequency_hz = 1.0,
        .stop_frequency_hz = adjacent,
    });
    ASSERT_TRUE(sparse.ok()) << sparse.error().message;
    EXPECT_EQ(sparse.value(), (std::vector<double>{1.0, adjacent}));

    auto collapsed = GenerateAcFrequencies(AcAnalysis{
        .sweep_type = sweep,
        .points = dense_points,
        .start_frequency_hz = 1.0,
        .stop_frequency_hz = adjacent,
    });
    ASSERT_FALSE(collapsed.ok());
    EXPECT_EQ(collapsed.error().code, ErrorCode::kParse);
    EXPECT_NE(collapsed.error().message.find("strictly increasing"),
              std::string::npos);
  }

  const double just_above_wide_decades =
      std::nextafter(1e300, std::numeric_limits<double>::infinity());
  auto wide_decades = GenerateAcFrequencies(AcAnalysis{
      .sweep_type = AcSweepType::kDec,
      .points = 1,
      .start_frequency_hz = 1e-300,
      .stop_frequency_hz = just_above_wide_decades,
  });
  ASSERT_TRUE(wide_decades.ok()) << wide_decades.error().message;
  ASSERT_EQ(wide_decades.value().size(), 602U);
  EXPECT_DOUBLE_EQ(wide_decades.value().front(), 1e-300);
  EXPECT_LT(wide_decades.value()[600], just_above_wide_decades);
  EXPECT_DOUBLE_EQ(wide_decades.value().back(), just_above_wide_decades);
}

TEST(Phase2BSolverTest, SolvesComplexSystemWithDeterministicPivoting) {
  const ComplexCsrMatrix matrix = {
      .rows = 2,
      .columns = 2,
      .values = {{1.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}},
      .column_indices = {1, 0, 1},
      .row_offsets = {0, 1, 3},
  };
  auto solved = SolveCpuComplexReference(matrix, {{2.0, -1.0}, {2.0, 3.0}});
  ASSERT_TRUE(solved.ok()) << solved.error().message;
  ASSERT_EQ(solved.value().size(), 2U);
  ExpectComplexNear(solved.value()[0], {1.0, 1.0}, 1e-14);
  ExpectComplexNear(solved.value()[1], {2.0, -1.0}, 1e-14);
}

TEST(Phase2BSolverTest, SolvesNonsingularMixedScaleRealAndComplexSystems) {
  const CsrMatrix real_matrix = {
      .rows = 2,
      .columns = 2,
      .values = {1e20, 1.0},
      .column_indices = {0, 1},
      .row_offsets = {0, 1, 2},
  };
  auto real = SolveCpuReference(real_matrix, {1e20, 1.0});
  ASSERT_TRUE(real.ok()) << real.error().message;
  EXPECT_EQ(real.value(), (std::vector<double>{1.0, 1.0}));

  const ComplexCsrMatrix complex_matrix = {
      .rows = 2,
      .columns = 2,
      .values = {{1e20, 0.0}, {0.0, 1.0}},
      .column_indices = {0, 1},
      .row_offsets = {0, 1, 2},
  };
  auto complex =
      SolveCpuComplexReference(complex_matrix, {{1e20, 0.0}, {0.0, 1.0}});
  ASSERT_TRUE(complex.ok()) << complex.error().message;
  ExpectComplexNear(complex.value()[0], {1.0, 0.0}, 0.0);
  ExpectComplexNear(complex.value()[1], {1.0, 0.0}, 0.0);
}

TEST(Phase2BSolverTest, SolvesLargeGminDiagonalAndRejectsTrueSingularity) {
  constexpr std::size_t size = 282;
  ComplexCsrMatrix gmin_diagonal;
  gmin_diagonal.rows = size;
  gmin_diagonal.columns = size;
  gmin_diagonal.values.assign(size, {kGminSiemens, 0.0});
  gmin_diagonal.column_indices.reserve(size);
  gmin_diagonal.row_offsets.reserve(size + 1);
  for (std::size_t index = 0; index < size; ++index) {
    gmin_diagonal.column_indices.push_back(index);
    gmin_diagonal.row_offsets.push_back(index);
  }
  gmin_diagonal.row_offsets.push_back(size);
  auto solved = SolveCpuComplexReference(
      gmin_diagonal,
      std::vector<std::complex<double>>(size, {kGminSiemens, 0.0}));
  ASSERT_TRUE(solved.ok()) << solved.error().message;
  for (std::complex<double> value : solved.value()) {
    ExpectComplexNear(value, {1.0, 0.0}, 0.0);
  }

  const ComplexCsrMatrix singular = {
      .rows = 2,
      .columns = 2,
      .values = {{1.0, 0.0}, {2.0, 0.0}, {2.0, 0.0}, {4.0, 0.0}},
      .column_indices = {0, 1, 0, 1},
      .row_offsets = {0, 2, 4},
  };
  auto rejected = SolveCpuComplexReference(singular, {{3.0, 0.0}, {6.0, 0.0}});
  ASSERT_FALSE(rejected.ok());
  EXPECT_EQ(rejected.error().code, ErrorCode::kSolve);
}

TEST(Phase2BSolverTest, RejectsMalformedComplexInputs) {
  const ComplexCsrMatrix identity = {
      .rows = 1,
      .columns = 1,
      .values = {{1.0, 0.0}},
      .column_indices = {0},
      .row_offsets = {0, 1},
  };
  auto dimensions = SolveCpuComplexReference(identity, {});
  ASSERT_FALSE(dimensions.ok());
  EXPECT_EQ(dimensions.error().code, ErrorCode::kSolve);

  const ComplexCsrMatrix invalid_order = {
      .rows = 2,
      .columns = 2,
      .values = {{1.0, 0.0}, {1.0, 0.0}},
      .column_indices = {1, 0},
      .row_offsets = {0, 2, 2},
  };
  auto order =
      SolveCpuComplexReference(invalid_order, {{1.0, 0.0}, {0.0, 0.0}});
  ASSERT_FALSE(order.ok());
  EXPECT_EQ(order.error().code, ErrorCode::kSolve);

  const ComplexCsrMatrix nonfinite = {
      .rows = 1,
      .columns = 1,
      .values = {{std::numeric_limits<double>::infinity(), 0.0}},
      .column_indices = {0},
      .row_offsets = {0, 1},
  };
  auto finite = SolveCpuComplexReference(nonfinite, {{1.0, 0.0}});
  ASSERT_FALSE(finite.ok());
  EXPECT_EQ(finite.error().code, ErrorCode::kSolve);

  const ComplexCsrMatrix bad_offset = {
      .rows = 1,
      .columns = 1,
      .values = {{1.0, 0.0}},
      .column_indices = {0},
      .row_offsets = {1, 1},
  };
  auto offset = SolveCpuComplexReference(bad_offset, {{1.0, 0.0}});
  ASSERT_FALSE(offset.ok());
  EXPECT_EQ(offset.error().code, ErrorCode::kSolve);

  const ComplexCsrMatrix bad_column = {
      .rows = 1,
      .columns = 1,
      .values = {{1.0, 0.0}},
      .column_indices = {1},
      .row_offsets = {0, 1},
  };
  auto column = SolveCpuComplexReference(bad_column, {{1.0, 0.0}});
  ASSERT_FALSE(column.ok());
  EXPECT_EQ(column.error().code, ErrorCode::kSolve);
}

TEST(Phase2BSimulationTest, MatchesAnalyticRcLowPass) {
  constexpr double frequency = 159.15494309189535;
  constexpr char netlist[] = R"(V1 in 0 AC 1
R1 in out 1k
C1 out 0 1u
.AC LIN 2 159.15494309189535 318.3098861837907
.END
)";
  auto simulated = SimulateAc(netlist);
  ASSERT_TRUE(simulated.ok()) << simulated.error().message;
  const auto &output = FindNodeValues(simulated.value(), "out");
  ASSERT_EQ(output.size(), 2U);
  const std::complex<double> expected =
      1e-3 / std::complex<double>(1e-3 + kGminSiemens,
                                  2.0 * std::numbers::pi * frequency * 1e-6);
  ExpectComplexNear(output.front(), expected, 1e-12);
  const auto &source_current = FindBranchValues(simulated.value(), "V1");
  const std::complex<double> expected_source_current =
      -kGminSiemens - 1e-3 * (1.0 - expected);
  ExpectComplexNear(source_current.front(), expected_source_current, 1e-15);
}

TEST(Phase2BSimulationTest, MatchesAnalyticRlCircuit) {
  constexpr double frequency = 159.15494309189535;
  constexpr char netlist[] = R"(V1 in 0 AC 1
L1 in out 10m
R1 out 0 10
.AC LIN 2 159.15494309189535 318.3098861837907
.END
)";
  auto simulated = SimulateAc(netlist);
  ASSERT_TRUE(simulated.ok()) << simulated.error().message;
  const auto &output = FindNodeValues(simulated.value(), "out");
  const std::complex<double> expected =
      1.0 / std::complex<double>(1.0, 2.0 * std::numbers::pi * frequency *
                                          10e-3 * (0.1 + kGminSiemens));
  ExpectComplexNear(output.front(), expected, 1e-12);
  const std::complex<double> expected_inductor_current =
      (0.1 + kGminSiemens) * expected;
  const auto &inductor_current = FindBranchValues(simulated.value(), "L1");
  ExpectComplexNear(inductor_current.front(), expected_inductor_current, 1e-14);
  const auto &source_current = FindBranchValues(simulated.value(), "V1");
  ExpectComplexNear(source_current.front(),
                    -kGminSiemens - expected_inductor_current, 1e-14);
}

TEST(Phase2BSimulationTest, MatchesRepresentativeSeriesRlcCircuit) {
  constexpr double frequency = 1591.5494309189535;
  constexpr double resistance = 10.0;
  constexpr double inductance = 10e-3;
  constexpr double capacitance = 1e-6;
  constexpr char netlist[] = R"(V1 in 0 AC 1
R1 in mid 10
L1 mid out 10m
C1 out 0 1u
.AC LIN 2 1591.5494309189535 3183.098861837907
.END
)";
  auto simulated = SimulateAc(netlist);
  ASSERT_TRUE(simulated.ok()) << simulated.error().message;
  const auto &output = FindNodeValues(simulated.value(), "out");
  const double omega = 2.0 * std::numbers::pi * frequency;
  const std::complex<double> s(0.0, omega);
  const std::complex<double> output_admittance = kGminSiemens + s * capacitance;
  const std::complex<double> inductor_factor =
      1.0 + s * inductance * output_admittance;
  const double conductance = 1.0 / resistance;
  const std::complex<double> expected =
      conductance /
      ((conductance + kGminSiemens) * inductor_factor + output_admittance);
  ExpectComplexNear(output.front(), expected, 1e-11);
  const std::complex<double> expected_inductor_current =
      output_admittance * expected;
  const auto &inductor_current = FindBranchValues(simulated.value(), "L1");
  ExpectComplexNear(inductor_current.front(), expected_inductor_current, 1e-12);
  const std::complex<double> expected_mid = inductor_factor * expected;
  const std::complex<double> expected_source_current =
      -kGminSiemens - conductance * (1.0 - expected_mid);
  const auto &source_current = FindBranchValues(simulated.value(), "V1");
  ExpectComplexNear(source_current.front(), expected_source_current, 1e-12);
}

TEST(Phase2BSimulationTest, EmitsLegacyCompatibleAcCsvInIrOrder) {
  constexpr char netlist[] = R"(V1 in 0 DC 5 AC 1 0
R1 in out 1k
C1 out 0 1u
.AC LIN 2 10 20
.END
)";
  auto csv = SimulateAcToCsv(netlist);
  ASSERT_TRUE(csv.ok()) << csv.error().message;
  std::istringstream rows(csv.value());
  std::string row;
  ASSERT_TRUE(std::getline(rows, row));
  EXPECT_EQ(row, "Frequency,V(in)_mag,V(in)_phase_deg,V(out)_mag,"
                 "V(out)_phase_deg,I(V1)_mag,I(V1)_phase_deg");
  ASSERT_TRUE(std::getline(rows, row));
  EXPECT_EQ(row.substr(0, row.find(',')), "10");
  const std::vector<std::string> first_values = SplitCsvRow(row);
  ASSERT_EQ(first_values.size(), 7U);
  const double omega = 2.0 * std::numbers::pi * 10.0;
  const std::complex<double> expected_output =
      1e-3 / std::complex<double>(1e-3 + kGminSiemens, omega * 1e-6);
  const std::complex<double> expected_source_current =
      -kGminSiemens - 1e-3 * (1.0 - expected_output);
  EXPECT_DOUBLE_EQ(std::stod(first_values[1]), 1.0);
  EXPECT_DOUBLE_EQ(std::stod(first_values[2]), 0.0);
  EXPECT_NEAR(std::stod(first_values[3]), std::abs(expected_output), 1e-14);
  EXPECT_NEAR(std::stod(first_values[4]),
              std::arg(expected_output) * 180.0 / std::numbers::pi, 1e-12);
  EXPECT_NEAR(std::stod(first_values[5]), std::abs(expected_source_current),
              1e-14);
  EXPECT_NEAR(std::stod(first_values[6]),
              std::arg(expected_source_current) * 180.0 / std::numbers::pi,
              1e-12);
  ASSERT_TRUE(std::getline(rows, row));
  EXPECT_EQ(row.substr(0, row.find(',')), "20");
  EXPECT_FALSE(std::getline(rows, row));
}

TEST(Phase2BSimulationTest, EscapesCsvIdentifiersWithoutChangingColumns) {
  constexpr char netlist[] = R"(V"1 in,node 0 DC 1 AC 1
R1 in,node 0 1k
.OP
.AC LIN 2 1 2
.END
)";
  auto ac_csv = SimulateAcToCsv(netlist);
  ASSERT_TRUE(ac_csv.ok()) << ac_csv.error().message;
  std::istringstream ac_rows(ac_csv.value());
  std::string header;
  ASSERT_TRUE(std::getline(ac_rows, header));
  EXPECT_EQ(header, "Frequency,\"V(in,node)_mag\",\"V(in,node)_phase_deg\","
                    "\"I(V\"\"1)_mag\",\"I(V\"\"1)_phase_deg\"");

  auto dc_csv = SimulateDcToCsv(netlist);
  ASSERT_TRUE(dc_csv.ok()) << dc_csv.error().message;
  EXPECT_NE(dc_csv.value().find("\"V(in,node)\","), std::string::npos);
  EXPECT_NE(dc_csv.value().find("\"I(V\"\"1)\","), std::string::npos);
}

TEST(Phase2BSimulationTest, RejectsNonfiniteDerivedCsvMagnitude) {
  constexpr char netlist[] = R"(I1 0 n AC 2e296 45
.AC LIN 2 1 2
.END
)";
  auto simulated = SimulateAc(netlist);
  ASSERT_TRUE(simulated.ok()) << simulated.error().message;
  const std::complex<double> value =
      FindNodeValues(simulated.value(), "n").front();
  EXPECT_TRUE(std::isfinite(value.real()));
  EXPECT_TRUE(std::isfinite(value.imag()));

  auto csv = SimulateAcToCsv(netlist);
  ASSERT_FALSE(csv.ok());
  EXPECT_EQ(csv.error().code, ErrorCode::kIo);
  EXPECT_NE(csv.error().message.find("non-finite"), std::string::npos);
}

TEST(Phase2BSimulationTest, ChecksAnalysisBeforeCompilingCircuit) {
  auto dc = SimulateDc("V1 n n DC 1\n.END\n");
  ASSERT_FALSE(dc.ok());
  EXPECT_EQ(dc.error().code, ErrorCode::kUnsupported);

  auto ac = SimulateAc("V1 n n AC 1\n.OP\n.END\n");
  ASSERT_FALSE(ac.ok());
  EXPECT_EQ(ac.error().code, ErrorCode::kUnsupported);

  auto automatic = SimulateToCsv("V1 n n DC 1\n.END\n");
  ASSERT_FALSE(automatic.ok());
  EXPECT_EQ(automatic.error().code, ErrorCode::kUnsupported);
}

TEST(Phase2BSimulationTest, PreservesDcBehaviorForCombinedSource) {
  constexpr char netlist[] = R"(V1 in 0 DC 10 AC 2 90
R1 in out 1k
R2 out 0 1k
.OP
.END
)";
  auto simulated = SimulateDc(netlist);
  ASSERT_TRUE(simulated.ok()) << simulated.error().message;
  ASSERT_EQ(simulated.value().node_voltages.size(), 2U);
  EXPECT_DOUBLE_EQ(simulated.value().node_voltages[0].second, 10.0);
  EXPECT_NEAR(simulated.value().node_voltages[1].second, 5.0, 1e-8);
  ASSERT_EQ(simulated.value().branch_currents.size(), 1U);
  EXPECT_NEAR(simulated.value().branch_currents[0].second, -0.005, 1e-9);
}

} // namespace
} // namespace ohmnivore
