#include "cpp/tests/google_test.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "ohmnivore/behavioral.h"
#include "ohmnivore/nonlinear.h"
#include "ohmnivore/parser.h"
#include "ohmnivore/prepared_ac.h"
#include "ohmnivore/transient.h"

namespace ohmnivore {
namespace {
MnaSystem Compile(const std::string &text) {
  auto parsed = ParseBehavioralNetlist(text);
  if (!parsed.ok()) {
    ADD_FAILURE() << parsed.error().message;
    return {};
  }
  auto result = CompileBehavioralMna(parsed.value());
  if (!result.ok()) {
    ADD_FAILURE() << result.error().message;
    return {};
  }
  return result.TakeValue();
}
std::size_t Node(const MnaSystem &system, const std::string &name) {
  return static_cast<std::size_t>(
      std::find(system.node_names.begin(), system.node_names.end(), name) -
      system.node_names.begin());
}
std::size_t Branch(const MnaSystem &system, const std::string &name) {
  return system.node_names.size() +
         static_cast<std::size_t>(std::find(system.branch_names.begin(),
                                            system.branch_names.end(), name) -
                                  system.branch_names.begin());
}
double Entry(const CsrMatrix &matrix, std::size_t row, std::size_t column) {
  for (std::size_t i = matrix.row_offsets[row]; i < matrix.row_offsets[row + 1];
       ++i)
    if (matrix.column_indices[i] == column)
      return matrix.values[i];
  return 0.0;
}

TEST(Emi02C, BehavioralSourcesResidualAndCurrentCrossDerivatives) {
  const auto system =
      Compile("* independent mixed source graph\nVcontrol a 0 2\nVsense b 0 "
              "0\nRload b 0 1\nEout c 0 VALUE={v(a)*v(a)}\nRout c 0 1k\nBmult "
              "a b I={i(Vsense)*(2+3*v(a,b))}\n.OP\n.end\n");
  ASSERT_GT(system.g.rows, 0U);
  std::vector<double> x(system.g.rows, 0.0);
  x[Node(system, "a")] = 2;
  x[Node(system, "b")] = 0.5;
  x[Branch(system, "Vsense")] = 0.2;
  auto linearized = BuildNonlinearDcLinearization(system, x);
  ASSERT_TRUE(linearized.ok()) << linearized.error().message;
  const auto &j = linearized.value().jacobian;
  const auto a = Node(system, "a"), b = Node(system, "b"),
             sense = Branch(system, "Vsense"), e = Branch(system, "Eout");
  EXPECT_NEAR(Entry(j, a, a) - Entry(system.g, a, a), 0.6, 1e-12);
  EXPECT_NEAR(Entry(j, a, b) - Entry(system.g, a, b), -0.6, 1e-12);
  EXPECT_NEAR(Entry(j, a, sense) - Entry(system.g, a, sense), 6.5, 1e-12);
  EXPECT_NEAR(Entry(j, b, sense) - Entry(system.g, b, sense), -6.5, 1e-12);
  EXPECT_NEAR(Entry(j, e, a) - Entry(system.g, e, a), -4, 1e-12);
  auto contribution = BuildDiodeResidualContribution(system, x);
  ASSERT_TRUE(contribution.ok());
  EXPECT_NEAR(contribution.value()[a], 1.3, 1e-12);
  EXPECT_NEAR(contribution.value()[b], -1.3, 1e-12);
  EXPECT_NEAR(contribution.value()[e], -4, 1e-12);
  for (std::size_t col = 0; col < x.size(); ++col) {
    auto plus = x, minus = x;
    plus[col] += 1e-6;
    minus[col] -= 1e-6;
    auto fp = BuildNonlinearDcLinearization(system, plus),
         fm = BuildNonlinearDcLinearization(system, minus);
    ASSERT_TRUE(fp.ok());
    ASSERT_TRUE(fm.ok());
    for (std::size_t row = 0; row < x.size(); ++row)
      EXPECT_NEAR((fp.value().residual[row] - fm.value().residual[row]) / 2e-6,
                  Entry(j, row, col), 1e-8);
  }
  auto invalid = system;
  invalid.behavioral_descriptors.back()
      .rows.front()
      .jacobian_value_indices.front() = system.g.values.size();
  auto rejected = BuildNonlinearDcLinearization(invalid, x);
  ASSERT_FALSE(rejected.ok());
  EXPECT_EQ(rejected.error().code, ErrorCode::kInvalidStructure);
  for (int fault = 0; fault < 4; ++fault) {
    auto corrupted = system;
    auto &d = corrupted.behavioral_descriptors.back();
    if (fault == 0)
      d.rows.back().coefficient = 1.0;
    if (fault == 1)
      d.rows.pop_back();
    if (fault == 2)
      d.is_voltage = true;
    if (fault == 3)
      d.branch_index = sense;
    auto invalid_result = BuildNonlinearDcLinearization(corrupted, x);
    ASSERT_FALSE(invalid_result.ok()) << fault;
    EXPECT_EQ(invalid_result.error().code, ErrorCode::kInvalidStructure);
  }
}

TEST(Emi02C, SignedAffineCapacitorRampsAndCharge) {
  for (double polarity : {-1.0, 1.0}) {
    const auto system =
        Compile("* capacitor multiplier\nVdrive drive 0 DC 0 PWL(0 0 1m " +
                std::to_string(polarity) +
                ")\nCbase drive sense 1u\nVsense sense 0 0\nBextra drive 0 "
                "I={i(Vsense)*(2+0.5*v(drive))}\n.TRAN 2u 1m\n.end\n");
    ASSERT_GT(system.g.rows, 0U);
    auto solved = RunTransientAnalysis(system, {2e-6, 1e-3, 0, false});
    ASSERT_TRUE(solved.ok()) << solved.error().message;
    const auto drive = Branch(system, "Vdrive");
    double charge = 0;
    // At t=0 DC displacement current is zero; integrate from first transient
    // sample.
    for (std::size_t i = 2; i < solved.value().states.size(); ++i) {
      const double t = solved.value().times_seconds[i];
      const double v = polarity * t / 1e-3;
      const double expected = 1e-6 * (3 + 0.5 * v) * polarity / 1e-3;
      EXPECT_NEAR(-solved.value().states[i][drive], expected, 1e-8);
      const double h = t - solved.value().times_seconds[i - 1];
      charge -= 0.5 * h *
                (solved.value().states[i][drive] +
                 solved.value().states[i - 1][drive]);
    }
    const double v0 = polarity * solved.value().times_seconds[1] / 1e-3;
    const double expected_charge =
        1e-6 * (3 * (polarity - v0) + 0.25 * (1 - v0 * v0));
    EXPECT_NEAR(charge, expected_charge, 1e-9);
    // Mutating the current coupling derivative must be rejected before Newton.
    auto bad = system;
    bad.behavioral_descriptors[0].rows[0].jacobian_value_indices.pop_back();
    EXPECT_FALSE(RunTransientAnalysis(bad, {2e-6, 1e-3, 0, false}).ok());
  }
}

TEST(Emi02C, StreamingFailureAndReducedBoundsDoNotChangeAcceptedState) {
  auto system = Compile(
      "* independent RC driven by behavioral source\nVinput input 0 DC 0 PWL(0 "
      "0 10u 1 100u 1)\nEdrive drive 0 VALUE={v(input)*v(input)}\nR1 drive out "
      "1k\nC1 out 0 1u\n.TRAN 10u 100u\n.end\n");
  ASSERT_GT(system.g.rows, 0U);
  const TranAnalysis analysis{10e-6, 100e-6, 0, false};
  auto baseline = RunTransientAnalysis(system, analysis);
  ASSERT_TRUE(baseline.ok()) << baseline.error().message;
  std::vector<double> times;
  std::vector<std::vector<double>> states;
  TransientExecutionLimits stream;
  stream.retain_output_states = false;
  stream.accepted_state_observer = [&](double t, const std::vector<double> &x) {
    times.push_back(t);
    states.push_back(x);
    return Result<bool>::Ok(true);
  };
  auto streamed = RunTransientAnalysis(system, analysis, stream);
  ASSERT_TRUE(streamed.ok()) << streamed.error().message;
  EXPECT_EQ(times, baseline.value().times_seconds);
  EXPECT_EQ(states, baseline.value().states);
  EXPECT_TRUE(streamed.value().states.empty());
  EXPECT_EQ(streamed.value().emitted_points, times.size());
  std::size_t calls = 0;
  stream.accepted_state_observer = [&](double, const std::vector<double> &) {
    return ++calls == 7
               ? Result<bool>::Fail(ErrorCode::kIo, "injected write failure")
               : Result<bool>::Ok(true);
  };
  auto failed = RunTransientAnalysis(system, analysis, stream);
  ASSERT_FALSE(failed.ok());
  EXPECT_EQ(failed.error().code, ErrorCode::kIo);
  EXPECT_EQ(calls, 7U);
  TransientExecutionLimits exhausted;
  exhausted.nonlinear_maximum_iterations = 0;
  auto bounded = RunTransientAnalysis(system, analysis, exhausted);
  ASSERT_FALSE(bounded.ok());
  EXPECT_EQ(bounded.error().code, ErrorCode::kNonConvergence);
  auto repeat = RunTransientAnalysis(system, analysis);
  ASSERT_TRUE(repeat.ok());
  EXPECT_EQ(repeat.value().states, baseline.value().states);
  EXPECT_EQ(repeat.value().times_seconds, baseline.value().times_seconds);
  EXPECT_TRUE(std::any_of(baseline.value().step_trace.begin(),
                          baseline.value().step_trace.end(),
                          [](const auto &step) { return !step.accepted; }));
}

TEST(Emi02C, ExplicitBoundaryAndHostileInputs) {
  EXPECT_FALSE(ParseNetlist("E1 a 0 VALUE={1}\n.OP\n.end\n").ok());
  EXPECT_FALSE(ParseBehavioralNetlist("B1 a 0 V={1}\n.OP\n.end\n").ok());
  auto unknown =
      ParseBehavioralNetlist("B1 a 0 I={v(missing)}\nR1 a 0 1\n.OP\n.end\n");
  ASSERT_TRUE(unknown.ok());
  EXPECT_FALSE(CompileBehavioralMna(unknown.value()).ok());
  auto system = Compile(
      "V1 a 0 PWL(0 1 1m 2)\nB1 b 0 I={v(a)}\nR1 b 0 1\n.TRAN 1u 1m\n.end\n");
  auto inconsistent = RunTransientAnalysis(system, {1e-6, 1e-3, 0, false});
  ASSERT_FALSE(inconsistent.ok());
  EXPECT_EQ(inconsistent.error().code, ErrorCode::kUnsupported);
  system = Compile("V1 a 0 1\nB1 b 0 I={v(a)}\nR1 b 0 1\n.TRAN 1u 1m\n.end\n");
  auto uic = RunTransientAnalysis(system, {1e-6, 1e-3, 0, true});
  ASSERT_FALSE(uic.ok());
  EXPECT_EQ(uic.error().code, ErrorCode::kUnsupported);
  auto prepared = PrepareLinearAcBatch(system, {AcSweepType::kLin, 2, 1, 2},
                                       "emi02", "c", "n");
  EXPECT_FALSE(prepared.ok());
}

TEST(Emi02C, CaseInsensitiveTopologyAndBindingsKeepOnePhysicalNode) {
  const auto system =
      Compile("Vdrive Input gNd 2\nR1 INPUT Output 1k\n"
              "R2 output 0 1k\nEcopy COPY GND VALUE={v(oUtPuT)}\n"
              "R3 copy 0 1k\nB1 COPY 0 I={i(vDrIvE)}\n.OP\n.end\n");
  ASSERT_EQ(system.node_names.size(), 3U);
  auto solved = RunNonlinearDc(system);
  ASSERT_TRUE(solved.ok()) << solved.error().message;
  EXPECT_NEAR(solved.value().solution[Node(system, "Input")], 2.0, 1e-12);
  EXPECT_NEAR(solved.value().solution[Node(system, "Output")], 1.0, 1e-8);
  EXPECT_NEAR(solved.value().solution[Node(system, "copy")], 1.0, 1e-8);
}

TEST(Emi02C, DirectIrSourceCurrentRequiresAnIndependentVoltageSource) {
  BehavioralCircuit input;
  input.circuit.components.emplace_back(Inductor{"Vsense", "a", "0", 1e-3});
  input.sources.push_back({"B1", "a", "0", "{i(Vsense)}", false});
  auto inductor = CompileBehavioralMna(input);
  ASSERT_FALSE(inductor.ok());
  input.circuit.components.clear();
  input.sources.push_back({"Vsense", "a", "0", "{1}", true});
  auto behavioral_voltage = CompileBehavioralMna(input);
  ASSERT_FALSE(behavioral_voltage.ok());
  input.sources.pop_back();
  input.circuit.components.emplace_back(
      VoltageSource{"Vsense", "a", "0", 1.0, std::nullopt});
  EXPECT_TRUE(CompileBehavioralMna(input).ok());
}
} // namespace
} // namespace ohmnivore
