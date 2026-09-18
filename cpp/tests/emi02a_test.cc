#include "cpp/tests/google_test.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <numbers>
#include <string>
#include <variant>
#include <vector>

#include "ohmnivore/compiler.h"
#include "ohmnivore/ir.h"
#include "ohmnivore/parser.h"
#include "ohmnivore/simulator.h"
#include "ohmnivore/status.h"
#include "ohmnivore/transient.h"

namespace ohmnivore {
namespace {

constexpr std::array<double, 7> kCoefficients = {0.0,    0.2,   -0.2,  0.995,
                                                 -0.995, 0.999, -0.999};

Circuit Pair(double coefficient, double first = 0.002, double second = 0.008) {
  Circuit circuit;
  circuit.components = {
      VoltageSource{"V1", "p", "0", 1.0, AcSourceSpecification{1.0, 0.0}},
      Resistor{"R1", "p", "a", 2.0},
      Inductor{"L1", "a", "0", first},
      VoltageSource{"V2", "q", "0", -0.3, AcSourceSpecification{0.3, 180.0}},
      Resistor{"R2", "q", "b", 3.0},
      Inductor{"L2", "b", "0", second}};
  circuit.inductor_couplings = {{"Kpair", "l1", "L2", coefficient}};
  return circuit;
}

// Oracle algebra is derived from the physical two-winding equations, without
// using production identity, stamping, matrix formation, or solver helpers.
struct PhysicalPair {
  double l1;
  double l2;
  double mutual;

  static PhysicalPair Make(double l1, double l2, double k) {
    return {l1, l2, k * std::sqrt(l1 * l2)};
  }

  std::array<double, 2> Slopes(std::array<double, 2> voltage) const {
    const double determinant = l1 * l2 - mutual * mutual;
    return {(l2 * voltage[0] - mutual * voltage[1]) / determinant,
            (l1 * voltage[1] - mutual * voltage[0]) / determinant};
  }

  double Energy(double first, double second) const {
    return 0.5 * (l1 * first * first + 2.0 * mutual * first * second +
                  l2 * second * second);
  }
};

std::vector<std::vector<double>> Dense(const CsrMatrix &matrix) {
  std::vector<std::vector<double>> result(
      matrix.rows, std::vector<double>(matrix.columns, 0.0));
  for (std::size_t row = 0; row < matrix.rows; ++row) {
    for (std::size_t i = matrix.row_offsets[row];
         i < matrix.row_offsets[row + 1]; ++i) {
      result[row][matrix.column_indices[i]] = matrix.values[i];
    }
  }
  return result;
}

void ScalarNear(double actual, double expected) {
  EXPECT_NEAR(actual, expected, 1e-12 + 1e-10 * std::abs(expected));
}

TEST(Emi02AParser, ForwardReferencesDoNotAffectComponentOrBranchOrder) {
  const auto parsed =
      ParseNetlist("kPair lFIRST Lsecond -0.2\nLfirst a 0 2m\nVmid z 0 1\n"
                   "Lsecond b 0 8m\n.OP\n");
  ASSERT_TRUE(parsed.ok()) << parsed.error().message;
  ASSERT_EQ(parsed.value().components.size(), 3U);
  ASSERT_EQ(parsed.value().inductor_couplings.size(), 1U);
  EXPECT_EQ(parsed.value().inductor_couplings[0].name, "kPair");
  const auto compiled = CompileMna(parsed.value());
  ASSERT_TRUE(compiled.ok()) << compiled.error().message;
  EXPECT_EQ(compiled.value().node_names,
            (std::vector<std::string>{"a", "z", "b"}));
  EXPECT_EQ(compiled.value().branch_names,
            (std::vector<std::string>{"Lfirst", "Vmid", "Lsecond"}));
  const auto reordered = ParseNetlist(
      "Lfirst a 0 2m\nVmid z 0 1\nLsecond b 0 8m\nkPair Lsecond lFIRST -0.2\n");
  ASSERT_TRUE(reordered.ok());
  const auto second = CompileMna(reordered.value());
  ASSERT_TRUE(second.ok());
  EXPECT_EQ(compiled.value().c.values, second.value().c.values);
  EXPECT_EQ(compiled.value().c.column_indices, second.value().c.column_indices);
  EXPECT_EQ(compiled.value().c.row_offsets, second.value().c.row_offsets);
}

TEST(Emi02AParser, DistinguishesMalformedUnsupportedAndNonfiniteCards) {
  const std::vector<std::pair<std::string, ErrorCode>> cases = {
      {"K1 L1 L2", ErrorCode::kParse},
      {"K1 L1 L2 0.2 0.3", ErrorCode::kParse},
      {"K1 L1 L2 --0.2", ErrorCode::kParse},
      {"K1 L1 L2 0.2bad", ErrorCode::kParse},
      {"K1 L1 L2 1e999bad", ErrorCode::kParse},
      {"K1 L1 L2 nan 0", ErrorCode::kParse},
      {"K1 L1 L2 L3 0.2", ErrorCode::kUnsupported},
      {"K1 L1 L2 Core", ErrorCode::kUnsupported},
      {"K1 L1 L2 0.2 Core", ErrorCode::kUnsupported},
      {"K1 L1 L2 1", ErrorCode::kUnsupported},
      {"K1 L1 L2 -0.9991", ErrorCode::kUnsupported},
      {"K1 L1 L2 nan", ErrorCode::kNonFinite},
      {"K1 L1 L2 +inf", ErrorCode::kNonFinite},
      {"K1 L1 L2 -infinity", ErrorCode::kNonFinite},
      {"K1 L1 L2 1e999", ErrorCode::kNonFinite},
      {"K1 L1 L2 1e-320F", ErrorCode::kNonFinite},
      {"K1 L1 L2 1e308T", ErrorCode::kNonFinite}};
  for (const auto &[card, expected] : cases) {
    SCOPED_TRACE(card);
    const auto parsed = ParseNetlist(card + "\n");
    ASSERT_FALSE(parsed.ok());
    EXPECT_EQ(parsed.error().code, expected);
    EXPECT_NE(parsed.error().message.find("line 1"), std::string::npos);
  }
  for (double coefficient : kCoefficients) {
    EXPECT_TRUE(ParseNetlist("K1 L1 L2 " + std::to_string(coefficient)).ok());
  }
}

TEST(Emi02ACompiler, FullMatricesSignedEnergyAndConstitutiveEquations) {
  for (double k : kCoefficients) {
    SCOPED_TRACE(k);
    const auto physical = PhysicalPair::Make(0.002, 0.008, k);
    const auto compiled = CompileMna(Pair(k));
    ASSERT_TRUE(compiled.ok()) << compiled.error().message;
    const MnaSystem &system = compiled.value();
    const auto g = Dense(system.g);
    const auto c = Dense(system.c);
    std::vector<std::vector<double>> expected_g(8, std::vector<double>(8, 0.0));
    expected_g[0][0] = expected_g[1][1] = 0.5 + kGminSiemens;
    expected_g[2][2] = expected_g[3][3] = 1.0 / 3.0 + kGminSiemens;
    expected_g[0][1] = expected_g[1][0] = -0.5;
    expected_g[2][3] = expected_g[3][2] = -1.0 / 3.0;
    expected_g[0][4] = expected_g[4][0] = 1.0;
    expected_g[1][5] = expected_g[5][1] = 1.0;
    expected_g[2][6] = expected_g[6][2] = 1.0;
    expected_g[3][7] = expected_g[7][3] = 1.0;
    for (std::size_t row = 0; row < 8; ++row) {
      for (std::size_t col = 0; col < 8; ++col) {
        ScalarNear(g[row][col], expected_g[row][col]);
        double expected_c = 0.0;
        if (row == 5 && col == 5)
          expected_c = -physical.l1;
        if (row == 7 && col == 7)
          expected_c = -physical.l2;
        if ((row == 5 && col == 7) || (row == 7 && col == 5))
          expected_c = -physical.mutual;
        ScalarNear(c[row][col], expected_c);
      }
    }
    EXPECT_EQ(system.c.values.size(), 4U); // Retains k=0 coordinates.
    const double discriminant =
        std::hypot(physical.l1 - physical.l2, 2 * physical.mutual);
    const double minimum_eigenvalue =
        (physical.l1 + physical.l2 - discriminant) / 2;
    ASSERT_GT(minimum_eigenvalue, 0.0);
    for (double first : {-10.0, -1.0, 0.0, 1.0, 10.0}) {
      for (double second : {-10.0, -1.0, 0.0, 1.0, 10.0}) {
        const double stored = -0.5 * (c[5][5] * first * first +
                                      (c[5][7] + c[7][5]) * first * second +
                                      c[7][7] * second * second);
        ScalarNear(stored, physical.Energy(first, second));
        EXPECT_GE(stored + 1e-14,
                  minimum_eigenvalue * (first * first + second * second) / 2);
        // Keep the independent voltage grid in [-1,1] V; at the endpoint
        // condition number, a 10 V zero-row cancellation alone exceeds 1 pV.
        const auto slope = physical.Slopes({first / 10.0, second / 10.0});
        ScalarNear(-c[5][5] * slope[0] - c[5][7] * slope[1], first / 10.0);
        ScalarNear(-c[7][5] * slope[0] - c[7][7] * slope[1], second / 10.0);
      }
    }
  }
}

TEST(Emi02ACompiler, DirectIrReferenceIdentityAndPassPrecedence) {
  const auto expect_error = [](const Circuit &circuit, ErrorCode expected) {
    const auto compiled = CompileMna(circuit);
    ASSERT_FALSE(compiled.ok());
    EXPECT_EQ(compiled.error().code, expected) << compiled.error().message;
  };
  for (const std::string &name : {"", "K", "Other", "Kbad!"}) {
    Circuit circuit = Pair(0.2);
    circuit.inductor_couplings[0].name = name;
    expect_error(circuit, ErrorCode::kCompile);
  }
  for (const std::string &reference : {"missing", "R1", "L1"}) {
    Circuit circuit = Pair(0.2);
    circuit.inductor_couplings[0].second_inductor = reference;
    expect_error(circuit, ErrorCode::kCompile);
  }
  Circuit ambiguous = Pair(0.2);
  ambiguous.components.push_back(Inductor{"l1", "extra", "0", 1.0});
  expect_error(ambiguous, ErrorCode::kCompile);
  ambiguous.components.back() = Resistor{"l1", "extra", "0", 1.0};
  expect_error(ambiguous, ErrorCode::kCompile);
  Circuit empty_reference = Pair(0.2);
  std::get<Inductor>(empty_reference.components[2]).name.clear();
  empty_reference.inductor_couplings[0].first_inductor.clear();
  expect_error(empty_reference, ErrorCode::kCompile);
  Circuit duplicate = Pair(0.0);
  duplicate.inductor_couplings.push_back({"Kother", "L2", "L1", 0.0});
  expect_error(duplicate, ErrorCode::kCompile);
  duplicate.inductor_couplings.back().name = "kPAIR";
  expect_error(duplicate, ErrorCode::kCompile);
  duplicate.inductor_couplings[0].first_inductor = "missing";
  duplicate.inductor_couplings.back().coefficient = 1.0;
  expect_error(duplicate, ErrorCode::kUnsupported);
  duplicate.inductor_couplings[0].coefficient = 2.0;
  duplicate.inductor_couplings.back().coefficient =
      std::numeric_limits<double>::quiet_NaN();
  expect_error(duplicate, ErrorCode::kNonFinite);
  duplicate = Pair(0.2, std::numeric_limits<double>::infinity());
  duplicate.inductor_couplings.push_back({"Kother", "missing", "L2", 0.0});
  expect_error(duplicate, ErrorCode::kCompile); // Identity pass precedes L.
  duplicate.inductor_couplings.pop_back();
  expect_error(duplicate, ErrorCode::kNonFinite);
  expect_error(Pair(0.2, -1.0), ErrorCode::kCompile);
  expect_error(Pair(0.2, 0.0), ErrorCode::kCompile);
  expect_error(Pair(std::numeric_limits<double>::infinity()),
               ErrorCode::kNonFinite);
}

TEST(Emi02ACompiler, DisjointDeclarationOrderIsStableAndDuplicateNamesFail) {
  Circuit circuit = Pair(0.2);
  circuit.components.push_back(Inductor{"L3", "c", "0", 0.001});
  circuit.components.push_back(Inductor{"L4", "d", "0", 0.003});
  circuit.inductor_couplings.push_back({"Kother", "L3", "L4", -0.995});
  const auto first = CompileMna(circuit);
  ASSERT_TRUE(first.ok());
  std::reverse(circuit.inductor_couplings.begin(),
               circuit.inductor_couplings.end());
  const auto second = CompileMna(circuit);
  ASSERT_TRUE(second.ok());
  EXPECT_EQ(first.value().c.values, second.value().c.values);
  EXPECT_EQ(first.value().c.column_indices, second.value().c.column_indices);
  EXPECT_EQ(first.value().c.row_offsets, second.value().c.row_offsets);
  circuit.inductor_couplings[0].name = "kPAIR";
  const auto duplicate = CompileMna(circuit);
  ASSERT_FALSE(duplicate.ok());
  EXPECT_EQ(duplicate.error().code, ErrorCode::kCompile);
}

TEST(Emi02ACompiler, ScaledArithmeticRejectsOnlyUnrepresentablePhysicalPair) {
  EXPECT_TRUE(CompileMna(Pair(0.999, 1e308, 1e308)).ok());
  EXPECT_TRUE(CompileMna(Pair(0.995, 1e-300, 1e300)).ok());
  EXPECT_TRUE(CompileMna(Pair(1e-300, 1e-300, 1e300)).ok());
  for (double coefficient : {0.2, 0.999}) {
    const auto result =
        CompileMna(Pair(coefficient, std::numeric_limits<double>::denorm_min(),
                        std::numeric_limits<double>::denorm_min()));
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.error().code, ErrorCode::kNonFinite);
  }
}

TEST(Emi02AAc, IndependentUnequalInductanceAnalyticSolveAndWindingReversal) {
  for (double k : kCoefficients) {
    SCOPED_TRACE(k);
    const auto physical = PhysicalPair::Make(0.002, 0.008, k);
    const std::string circuit =
        "V1 p 0 AC 1\nR1 p a 2\nL1 a 0 2m\n"
        "V2 q 0 AC 0.3 180\nR2 q b 3\nL2 b 0 8m\nK1 L1 L2 " +
        std::to_string(k) + "\n.AC LIN 3 20 50000\n";
    const auto actual = SimulateAc(circuit);
    ASSERT_TRUE(actual.ok()) << actual.error().message;
    for (std::size_t index = 0; index < actual.value().frequencies_hz.size();
         ++index) {
      const std::complex<double> jw(
          0.0, 2 * std::numbers::pi * actual.value().frequencies_hz[index]);
      const auto z1 = 2.0 / (1 + 2 * kGminSiemens) + jw * physical.l1;
      const auto z2 = 3.0 / (1 + 3 * kGminSiemens) + jw * physical.l2;
      const auto zm = jw * physical.mutual;
      const double e1 = 1.0 / (1 + 2 * kGminSiemens);
      const double e2 = -0.3 / (1 + 3 * kGminSiemens);
      const std::array<std::complex<double>, 2> expected = {
          (e1 * z2 - e2 * zm) / (z1 * z2 - zm * zm),
          (e2 * z1 - e1 * zm) / (z1 * z2 - zm * zm)};
      for (std::size_t winding = 0; winding < 2; ++winding) {
        const auto observed =
            actual.value().branch_currents[2 * winding + 1].second[index];
        ScalarNear(observed.real(), expected[winding].real());
        ScalarNear(observed.imag(), expected[winding].imag());
      }
    }
    std::string reversed = circuit;
    reversed.replace(reversed.find("L2 b 0"), 6, "L2 0 b");
    const std::string old_k = "K1 L1 L2 " + std::to_string(k);
    reversed.replace(reversed.find(old_k), old_k.size(),
                     "K1 L1 L2 " + std::to_string(-k));
    const auto changed = SimulateAc(reversed);
    ASSERT_TRUE(changed.ok()) << changed.error().message;
    for (std::size_t index = 0; index < actual.value().frequencies_hz.size();
         ++index) {
      EXPECT_NEAR(std::abs(actual.value().branch_currents[1].second[index] -
                           changed.value().branch_currents[1].second[index]),
                  0.0, 1e-12);
      EXPECT_NEAR(std::abs(actual.value().branch_currents[3].second[index] +
                           changed.value().branch_currents[3].second[index]),
                  0.0, 1e-12);
    }
  }
}

std::array<double, 2> Derivative(const PhysicalPair &pair,
                                 const std::array<double, 2> &state) {
  return pair.Slopes({(1.0 - 2 * state[0]) / (1 + 2 * kGminSiemens),
                      (-0.3 - 3 * state[1]) / (1 + 3 * kGminSiemens)});
}

std::array<double, 2> IntegrateIndependent(const PhysicalPair &pair,
                                           std::array<double, 2> state,
                                           double interval) {
  const std::size_t count =
      static_cast<std::size_t>(std::ceil(interval / 1e-8));
  if (count == 0)
    return state;
  const double h = interval / static_cast<double>(count);
  for (std::size_t step = 0; step < count; ++step) {
    const auto a = Derivative(pair, state);
    const auto b =
        Derivative(pair, {state[0] + h * a[0] / 2, state[1] + h * a[1] / 2});
    const auto c =
        Derivative(pair, {state[0] + h * b[0] / 2, state[1] + h * b[1] / 2});
    const auto d = Derivative(pair, {state[0] + h * c[0], state[1] + h * c[1]});
    for (std::size_t winding = 0; winding < 2; ++winding)
      state[winding] +=
          h * (a[winding] + 2 * b[winding] + 2 * c[winding] + d[winding]) / 6;
  }
  return state;
}

TEST(Emi02ATransient, IndependentRk4TrajectoriesInitializationAndRefinement) {
  for (double k : kCoefficients) {
    SCOPED_TRACE(k);
    const auto physical = PhysicalPair::Make(0.002, 0.008, k);
    const auto compiled = CompileMna(Pair(k));
    ASSERT_TRUE(compiled.ok());
    const auto dc = BuildTransientInitialState(compiled.value(), false);
    const auto uic = BuildTransientInitialState(compiled.value(), true);
    ASSERT_TRUE(dc.ok());
    ASSERT_TRUE(uic.ok());
    ScalarNear(dc.value()[5], 0.5);
    ScalarNear(dc.value()[7], -0.1);
    ScalarNear(uic.value()[5], 0.0);
    ScalarNear(uic.value()[7], 0.0);
    const auto no_uic = RunTransientAnalysis(compiled.value(),
                                             {.time_step_seconds = 1e-5,
                                              .stop_time_seconds = 2e-4,
                                              .start_time_seconds = 0.0,
                                              .use_initial_conditions = false});
    ASSERT_TRUE(no_uic.ok()) << no_uic.error().message;
    for (const auto &state : no_uic.value().states) {
      ScalarNear(state[5], 0.5);
      ScalarNear(state[7], -0.1);
      ScalarNear(state[1], 0.0);
      ScalarNear(state[3], 0.0);
    }
    std::array<double, 2> final_errors = {};
    for (std::size_t refinement = 0; refinement < 2; ++refinement) {
      // Resolve the endpoint leakage mode and its startup independently of
      // the controller's LTE scale; halving a controller-limited maximum
      // step can otherwise leave the actual trajectory almost unchanged.
      const double maximum_step = std::abs(k) > 0.9 ? 1e-8 : 1e-7;
      const auto result = RunTransientAnalysis(
          compiled.value(),
          {.time_step_seconds = maximum_step / (refinement == 0 ? 1 : 2),
           .stop_time_seconds = 2e-4,
           .start_time_seconds = 0.0,
           .use_initial_conditions = true});
      ASSERT_TRUE(result.ok()) << result.error().message;
      std::array<double, 2> oracle = {0.0, 0.0};
      double previous = 0.0;
      for (std::size_t sample = 0; sample < result.value().states.size();
           ++sample) {
        oracle = IntegrateIndependent(
            physical, oracle, result.value().times_seconds[sample] - previous);
        previous = result.value().times_seconds[sample];
        for (std::size_t winding = 0; winding < 2; ++winding) {
          const double actual = result.value().states[sample][5 + 2 * winding];
          EXPECT_NEAR(actual, oracle[winding],
                      1e-6 + 1e-3 * std::abs(oracle[winding]));
          const double resistance = winding == 0 ? 2.0 : 3.0;
          const double source = winding == 0 ? 1.0 : -0.3;
          const double voltage = (source - resistance * oracle[winding]) /
                                 (1 + resistance * kGminSiemens);
          EXPECT_NEAR(result.value().states[sample][1 + 2 * winding], voltage,
                      1e-5 + 1e-3 * std::abs(voltage));
          if (sample + 1 == result.value().states.size())
            final_errors[refinement] = std::max(
                final_errors[refinement], std::abs(actual - oracle[winding]));
        }
      }
    }
    EXPECT_LE(final_errors[1], final_errors[0] * 1.05 + 1e-10);
  }
}

TEST(Emi02ATransient, MutualHistoryAndCommonDifferentialModes) {
  for (double k : kCoefficients) {
    const auto compiled = CompileMna(Pair(k, 0.002, 0.002));
    ASSERT_TRUE(compiled.ok());
    const auto physical = PhysicalPair::Make(0.002, 0.002, k);
    for (double sign : {-1.0, 1.0}) {
      std::vector<double> previous(8, 0.0);
      previous[5] = 2.0;
      previous[7] = 2.0 * sign;
      const auto be = BuildBackwardEulerRhs(compiled.value().c, previous,
                                            std::vector<double>(8, 0.0), 0.01);
      const auto trap = BuildTrapezoidalRhs(
          compiled.value().g, compiled.value().c, previous,
          std::vector<double>(8, 0.0), std::vector<double>(8, 0.0), 0.01);
      ASSERT_TRUE(be.ok());
      ASSERT_TRUE(trap.ok());
      const double modal = 0.002 + sign * physical.mutual;
      ScalarNear(be.value()[5], -2.0 * modal / 0.01);
      ScalarNear(be.value()[7], -2.0 * sign * modal / 0.01);
      ScalarNear(trap.value()[5], -4.0 * modal / 0.01);
      ScalarNear(trap.value()[7], -4.0 * sign * modal / 0.01);
      EXPECT_GT(modal, 0.0);
    }
  }
}

TEST(Emi02ATransient, RejectedTrialsHardPointsBudgetsAndRepeatability) {
  Circuit circuit = Pair(0.995);
  std::get<VoltageSource>(circuit.components[0]).transient =
      PulseWaveform{1.0, 2.0, 7e-5, 0.0, 0.0, 7e-5, 1.0};
  const auto compiled = CompileMna(circuit);
  ASSERT_TRUE(compiled.ok());
  const TranAnalysis analysis{1e-5, 2e-4, 0.0, true};
  const auto first = RunTransientAnalysis(compiled.value(), analysis);
  ASSERT_TRUE(first.ok()) << first.error().message;
  ASSERT_TRUE(std::any_of(first.value().step_trace.begin(),
                          first.value().step_trace.end(),
                          [](const auto &step) { return !step.accepted; }));
  for (double edge : {7e-5, 14e-5}) {
    EXPECT_NE(std::find(first.value().times_seconds.begin(),
                        first.value().times_seconds.end(), edge),
              first.value().times_seconds.end());
  }
  const auto original_c = compiled.value().c.values;
  const auto failed = RunTransientAnalysis(
      compiled.value(), analysis,
      {.maximum_accepted_steps = 1, .maximum_step_attempts = 1});
  ASSERT_FALSE(failed.ok());
  EXPECT_EQ(failed.error().code, ErrorCode::kSolve);
  EXPECT_EQ(compiled.value().c.values, original_c);
  const auto repeat = RunTransientAnalysis(compiled.value(), analysis);
  ASSERT_TRUE(repeat.ok());
  EXPECT_EQ(repeat.value().times_seconds, first.value().times_seconds);
  EXPECT_EQ(repeat.value().states, first.value().states);
  ASSERT_EQ(repeat.value().step_trace.size(), first.value().step_trace.size());
  double accepted_time = 0.0;
  for (std::size_t i = 0; i < first.value().step_trace.size(); ++i) {
    const auto &step = first.value().step_trace[i];
    EXPECT_EQ(step.start_time_seconds, accepted_time);
    EXPECT_EQ(step.accepted, repeat.value().step_trace[i].accepted);
    EXPECT_EQ(step.normalized_local_error,
              repeat.value().step_trace[i].normalized_local_error);
    if (step.accepted)
      accepted_time = step.end_time_seconds;
  }
  const auto initial = BuildTransientInitialState(compiled.value(), true);
  ASSERT_TRUE(initial.ok());
  ScalarNear(initial.value()[5], 0.0);
  ScalarNear(initial.value()[7], 0.0);
  const auto singular = SimulateDc("L1 a 0 1m\nL2 a 0 2m\nK1 L1 L2 .2\n.OP\n");
  ASSERT_FALSE(singular.ok());
  EXPECT_EQ(singular.error().code, ErrorCode::kSingular);
}

} // namespace
} // namespace ohmnivore
