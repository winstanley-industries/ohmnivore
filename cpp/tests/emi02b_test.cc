#include "cpp/tests/google_test.h"

#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "ohmnivore/expression.h"

#include "cpp/src/expression_internal.h"

namespace ohmnivore {
namespace {

ExpressionBindings Bindings() {
  return {.state_size = 4,
          .node_indices = {{"a", 0}, {"b", 1}, {"0", std::nullopt}},
          .current_indices = {{"vsense", 3}},
          .parameters = {{"gain", 2.5}}};
}

ExpressionEvaluation Eval(const std::string &text,
                          const std::array<double, 4> &state = {3.0, 2.0, 0.0,
                                                                0.4}) {
  auto expression = CompileExpression(text, Bindings());
  if (!expression.ok()) {
    ADD_FAILURE() << expression.error().message;
    return {};
  }
  auto result = EvaluateExpression(expression.value(), state);
  if (!result.ok()) {
    ADD_FAILURE() << result.error().message;
    return {};
  }
  return result.TakeValue();
}

double Derivative(const ExpressionEvaluation &result, std::size_t index) {
  for (const auto &[variable, value] : result.derivatives) {
    if (variable == index)
      return value;
  }
  return 0.0;
}

void Near(double actual, double expected) {
  EXPECT_NEAR(actual, expected, 1e-12 + 1e-11 * std::abs(expected));
}

TEST(Emi02B, IndependentAnalyticOperatorValuesAndJacobians) {
  struct Case {
    const char *text;
    double value;
    double da;
    double db;
  };
  const std::array<Case, 12> cases = {{
      {"+v(a)", 3, 1, 0},
      {"-v(a)", -3, -1, 0},
      {"v(a)+v(b)", 5, 1, 1},
      {"v(a)-v(b)", 1, 1, -1},
      {"v(a)*v(b)", 6, 2, 3},
      {"v(a)/v(b)", 1.5, .5, -.75},
      {"v(a)**2", 9, 6, 0},
      {"v(a)**v(b)", 9, 6, 9 * std::log(3.0)},
      {"2**v(b)", 4, 0, 4 * std::log(2.0)},
      {"exp(v(b))", std::exp(2.0), 0, std::exp(2.0)},
      {"v(a)<v(b)", 0, 0, 0},
      {"v(a)>v(b)", 1, 0, 0},
  }};
  for (const auto &test : cases) {
    SCOPED_TRACE(test.text);
    const auto result = Eval(test.text);
    Near(result.value, test.value);
    Near(Derivative(result, 0), test.da);
    Near(Derivative(result, 1), test.db);
  }
}

TEST(Emi02B, EngineeringLiteralsGroundPrecedenceAndCase) {
  Near(Eval("{GAIN*(V(a,GND)-v(b,0)) + 2k*3m - 4u/2n}").value, -1991.5);
  Near(Eval("1T+2G+3MEG+4K+5M+6U+7N+8P+9F").value,
       1e12 + 2e9 + 3e6 + 4e3 + 5e-3 + 6e-6 + 7e-9 + 8e-12 + 9e-15);
  Near(Eval("-2**2 + 2**3**2").value, 508.0);
  Near(Eval("2**-2").value, .25);
}

TEST(Emi02B, AbsoluteBaseDynamicAndConstantPowers) {
  const auto runtime = Eval("v(a)**3", {-2, 0, 0, 0});
  Near(runtime.value, 8.0);
  Near(Derivative(runtime, 0), -12.0);
  Near(Eval("(-2)**3").value, 8.0);
  auto parameter =
      CompileExpression("(-2)**3", {}, ExpressionDialect::kParameter);
  ASSERT_TRUE(parameter.ok());
  auto result = EvaluateExpression(parameter.value(), {});
  ASSERT_TRUE(result.ok());
  Near(result.value().value, 8.0);
  const auto fractional = Eval("v(a)**2.5", {-4, 0, 0, 0});
  Near(fractional.value, 32.0);
  Near(Derivative(fractional, 0), -20.0);
  const auto zero = Eval("v(a)**2", {0, 0, 0, 0});
  Near(zero.value, 0);
  Near(Derivative(zero, 0), 0);
  auto cusp = CompileExpression("v(a)**0.5", Bindings());
  ASSERT_TRUE(cusp.ok());
  const std::array<double, 4> state = {0, 0, 0, 0};
  auto invalid = EvaluateExpression(cusp.value(), state);
  ASSERT_FALSE(invalid.ok());
  EXPECT_EQ(invalid.error().code, ErrorCode::kNonFinite);
}

TEST(Emi02B, SignedDivisionOffsetAndExactLocalDerivative) {
  constexpr double delta = 1e-32;
  for (double denominator : {0.0, -0.0, delta, -delta, 1.0, -1.0}) {
    const auto result = Eval("v(a)/v(b)", {2e-32, denominator, 0, 0});
    const double adjusted = denominator + (denominator >= 0 ? delta : -delta);
    Near(result.value, 2e-32 / adjusted);
    Near(Derivative(result, 0), 1.0 / adjusted);
    Near(Derivative(result, 1), -(2e-32 / adjusted) / adjusted);
  }
  auto ordinary = CompileExpression("1/0", {}, ExpressionDialect::kParameter);
  ASSERT_TRUE(ordinary.ok());
  auto rejected = EvaluateExpression(ordinary.value(), {});
  ASSERT_FALSE(rejected.ok());
  EXPECT_EQ(rejected.error().code, ErrorCode::kNonFinite);
  Near(Eval("1/0").value, 1e32);
}

TEST(Emi02B, PsExponentialContinuationAndSeparateParameterExp) {
  constexpr double slope = 1202604.284;
  for (double input : {13.0, 14.0, 14.5, 20.0, 1000.0}) {
    const auto result = Eval("exp(v(a))", {input, 0, 0, 0});
    Near(result.value, input <= 14 ? std::exp(input) : slope * (input - 13));
    Near(Derivative(result, 0), input <= 14 ? std::exp(input) : slope);
  }
  auto parameter =
      CompileExpression("exp(20)", {}, ExpressionDialect::kParameter);
  ASSERT_TRUE(parameter.ok());
  auto result = EvaluateExpression(parameter.value(), {});
  ASSERT_TRUE(result.ok());
  Near(result.value().value, std::exp(20.0));
  auto too_large =
      CompileExpression("exp(1000)", {}, ExpressionDialect::kParameter);
  ASSERT_TRUE(too_large.ok());
  EXPECT_FALSE(EvaluateExpression(too_large.value(), {}).ok());
}

TEST(Emi02B, LazyConditionsBindBothArmsAndUseSameDerivativeBranch) {
  auto expression =
      CompileExpression("if(v(a)>0, v(a)*i(vsense), v(a)**-1)", Bindings());
  ASSERT_TRUE(expression.ok());
  const std::array<double, 4> positive = {3, 0, 0, .4};
  auto value = EvaluateExpression(expression.value(), positive);
  ASSERT_TRUE(value.ok());
  Near(value.value().value, 1.2);
  Near(Derivative(value.value(), 0), .4);
  Near(Derivative(value.value(), 3), 3.0);
  const std::array<double, 4> equal = {0, 0, 0, .4};
  auto failed = EvaluateExpression(expression.value(), equal);
  ASSERT_FALSE(failed.ok());
  EXPECT_EQ(failed.error().code, ErrorCode::kNonFinite);
  for (double input : {-1.0, 0.0, 1.0}) {
    const auto result = Eval("if(v(a)<0,-v(a),v(a)*2)", {input, 0, 0, 0});
    Near(Derivative(result, 0), input < 0 ? -1.0 : 2.0);
  }
  auto unknown_dead_arm = CompileExpression("if(1,2,v(missing))", Bindings());
  ASSERT_FALSE(unknown_dead_arm.ok());
  EXPECT_EQ(unknown_dead_arm.error().code, ErrorCode::kUnsupported);
  auto structural = CompileExpression("if(v(a)>0,i(vsense),v(b))", Bindings());
  ASSERT_TRUE(structural.ok());
  EXPECT_EQ(std::vector<std::size_t>(structural.value().dependencies().begin(),
                                     structural.value().dependencies().end()),
            (std::vector<std::size_t>{0, 1, 3}));
}

TEST(Emi02B, CurrentMultiplierHasSimultaneousCrossDerivatives) {
  const auto value = Eval("i(vsense)*(1+gain*v(a,b))");
  Near(value.value, 1.4);
  Near(Derivative(value, 0), 1.0);
  Near(Derivative(value, 1), -1.0);
  Near(Derivative(value, 3), 3.5);
  const auto repeated = Eval("v(a)-v(a)+i(vsense)-i(vsense)");
  EXPECT_TRUE(repeated.derivatives.empty());
  // Source-current state indices need not be smaller than AST size.
  const auto standalone = Eval("i(vsense)");
  Near(standalone.value, .4);
  Near(Derivative(standalone, 3), 1.0);
}

TEST(Emi02B, SparseStateSlotsPreserveExactOrderCancellationAndLazyValidation) {
  const ExpressionBindings bindings{
      .state_size = 512,
      .node_indices = {{"a", 511}, {"b", 4}, {"c", 127}},
      .current_indices = {{"vsense", 256}},
      .parameters = {}};
  auto expression =
      CompileExpression("if(v(a)>0,v(a)-v(a)+v(b)*i(vsense)+v(c)*2,"
                        "v(b)-v(b)+i(vsense)*0)",
                        bindings);
  ASSERT_TRUE(expression.ok());
  EXPECT_EQ(std::vector<std::size_t>(expression.value().dependencies().begin(),
                                     expression.value().dependencies().end()),
            (std::vector<std::size_t>{4, 127, 256, 511}));
  std::array<double, 512> state{};
  state[511] = 1;
  state[4] = 0.5;
  state[127] = -2;
  state[256] = 4;
  const std::vector<std::pair<std::size_t, double>> expected{
      {4, 4}, {127, 2}, {256, 0.5}};
  auto positive = EvaluateExpression(expression.value(), state);
  ASSERT_TRUE(positive.ok());
  EXPECT_EQ(positive.value().value, -2);
  EXPECT_EQ(positive.value().derivatives, expected);
  state[511] = -1;
  auto cancelled = EvaluateExpression(expression.value(), state);
  ASSERT_TRUE(cancelled.ok());
  EXPECT_EQ(cancelled.value().value, 0);
  EXPECT_TRUE(cancelled.value().derivatives.empty());
  state[127] = std::numeric_limits<double>::quiet_NaN();
  auto inactive_nonfinite = EvaluateExpression(expression.value(), state);
  ASSERT_FALSE(inactive_nonfinite.ok());
  EXPECT_EQ(inactive_nonfinite.error().code, ErrorCode::kNonFinite);
  state[127] = -2;
  state[511] = 1;
  auto repeated = EvaluateExpression(expression.value(), state);
  ASSERT_TRUE(repeated.ok());
  EXPECT_EQ(repeated.value().value, positive.value().value);
  EXPECT_EQ(repeated.value().derivatives, expected);
}

TEST(Emi02B, IndependentFiniteDifferencesAndComplexStepOnSmoothBranch) {
  const std::string expression = "exp(v(a)/5)*(v(a)-v(b))**2+i(vsense)*v(b)";
  std::array<double, 4> state = {3, 2, 0, .4};
  const auto value = Eval(expression, state);
  for (std::size_t variable : {0U, 1U, 3U}) {
    constexpr double h = 1e-5;
    auto positive = state;
    auto negative = state;
    positive[variable] += h;
    negative[variable] -= h;
    const double finite_difference =
        (Eval(expression, positive).value - Eval(expression, negative).value) /
        (2 * h);
    EXPECT_NEAR(Derivative(value, variable), finite_difference, 1e-8);
    std::array<std::complex<double>, 4> z;
    for (std::size_t i = 0; i < state.size(); ++i)
      z[i] = state[i];
    constexpr double complex_h = 1e-20;
    z[variable] += std::complex<double>(0, complex_h);
    // Independently written analytic formula, valid because a-b is positive.
    const auto independent =
        std::exp(z[0] / 5.0) * (z[0] - z[1]) * (z[0] - z[1]) + z[3] * z[1];
    Near(Derivative(value, variable), independent.imag() / complex_h);
  }
}

TEST(Emi02B, ForwardParametersCallerOverridesAndShadowing) {
  const std::vector<ParameterDefinition> definitions = {
      {"output", "late*2+MDE"},
      {"late", "exp(base)"},
      {"base", "3"},
      {"MDE", "99"}};
  const ParameterValues caller = {{"mde", 4.0}, {"base", 2.0}};
  auto result = ResolveParameters(definitions, caller,
                                  {{"mde", "MDE"}, {"base", "base+1"}});
  ASSERT_TRUE(result.ok()) << result.error().message;
  Near(result.value().at("MDE"), 4);
  Near(result.value().at("BASE"), 3);
  Near(result.value().at("LATE"), std::exp(3.0));
  Near(result.value().at("OUTPUT"), 2 * std::exp(3.0) + 4);
  auto local_shadow = ResolveParameters({{"a", "b"}, {"b", "7"}}, {{"b", 1}});
  ASSERT_TRUE(local_shadow.ok());
  Near(local_shadow.value().at("A"), 7);
}

TEST(Emi02B, ParameterCyclesDuplicatesMissingNamesAndCallerScopeFailClosed) {
  const std::vector<std::vector<ParameterDefinition>> invalid = {
      {{"a", "a"}}, {{"a", "b"}, {"b", "a"}}, {{"a", "unknown"}}};
  for (const auto &definitions : invalid) {
    auto result = ResolveParameters(definitions);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.error().code, ErrorCode::kUnsupported);
  }
  auto duplicate = ResolveParameters({{"a", "1"}, {"A", "2"}});
  ASSERT_FALSE(duplicate.ok());
  EXPECT_EQ(duplicate.error().code, ErrorCode::kCompile);
  auto unknown_override = ResolveParameters({{"a", "1"}}, {}, {{"b", "2"}});
  ASSERT_FALSE(unknown_override.ok());
  EXPECT_EQ(unknown_override.error().code, ErrorCode::kUnsupported);
  auto callee_only =
      ResolveParameters({{"a", "1"}, {"b", "2"}}, {}, {{"a", "b"}});
  ASSERT_FALSE(callee_only.ok());
  EXPECT_EQ(callee_only.error().code, ErrorCode::kUnsupported);
  auto duplicate_override =
      ResolveParameters({{"a", "1"}}, {}, {{"a", "2"}, {"A", "3"}});
  ASSERT_FALSE(duplicate_override.ok());
  EXPECT_EQ(duplicate_override.error().code, ErrorCode::kCompile);
  auto state =
      CompileExpression("v(a)", Bindings(), ExpressionDialect::kParameter);
  ASSERT_FALSE(state.ok());
  EXPECT_EQ(state.error().code, ErrorCode::kUnsupported);
}

TEST(Emi02B, MalformedUnknownNonfiniteAndResourceLimits) {
  for (const char *text :
       {"", "1+", "(1", "if(1,2)", "exp(1,2)", "i(vsense,a)"}) {
    SCOPED_TRACE(text);
    auto result = CompileExpression(text, Bindings());
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.error().code, ErrorCode::kParse);
  }
  for (const char *text :
       {"sqrt(4)", "v(a)^2", "v(a)<=2", "1xyz", "time", "i(R1)"}) {
    SCOPED_TRACE(text);
    EXPECT_FALSE(CompileExpression(text, Bindings()).ok());
  }
  auto overflow = CompileExpression("1e999", Bindings());
  ASSERT_FALSE(overflow.ok());
  EXPECT_EQ(overflow.error().code, ErrorCode::kNonFinite);
  auto nesting = CompileExpression(
      std::string(64, '(') + "1" + std::string(64, ')'), Bindings());
  ASSERT_FALSE(nesting.ok());
  EXPECT_EQ(nesting.error().code, ErrorCode::kUnsupportedSize);
  auto oversized = CompileExpression(std::string(65537, ' '), Bindings());
  ASSERT_FALSE(oversized.ok());
  EXPECT_EQ(oversized.error().code, ErrorCode::kUnsupportedSize);
  std::string balanced = "1";
  for (int i = 0; i < 9; ++i)
    balanced = "(" + balanced + "+" + balanced + ")";
  auto nodes = CompileExpression(balanced, Bindings());
  ASSERT_FALSE(nodes.ok());
  EXPECT_EQ(nodes.error().code, ErrorCode::kUnsupportedSize);
  std::vector<ParameterDefinition> chain;
  for (int i = 0; i < 65; ++i)
    chain.push_back(
        {"p" + std::to_string(i), i == 64 ? "1" : "p" + std::to_string(i + 1)});
  auto depth = ResolveParameters(chain);
  ASSERT_FALSE(depth.ok());
  EXPECT_EQ(depth.error().code, ErrorCode::kUnsupportedSize);
  std::vector<ParameterDefinition> total;
  for (int i = 0; i < 400; ++i)
    total.push_back(
        {"p" + std::to_string(i),
         "1+2+3+4+5+6+7+8+9+10+11+12+13+14+15+16+17+18+19+20+21+22"});
  auto budget = ResolveParameters(total);
  ASSERT_FALSE(budget.ok());
  EXPECT_EQ(budget.error().code, ErrorCode::kUnsupportedSize);
}

TEST(Emi02B, InvalidBindingsAndStateNeverProducePartialResult) {
  auto bindings = Bindings();
  bindings.node_indices["A"] = 1;
  auto duplicate = CompileExpression("1", bindings);
  ASSERT_FALSE(duplicate.ok());
  EXPECT_EQ(duplicate.error().code, ErrorCode::kCompile);
  bindings = Bindings();
  bindings.current_indices["vbad"] = 4;
  auto index = CompileExpression("1", bindings);
  ASSERT_FALSE(index.ok());
  EXPECT_EQ(index.error().code, ErrorCode::kCompile);
  bindings = Bindings();
  bindings.parameters["bad"] = std::numeric_limits<double>::infinity();
  auto nonfinite = CompileExpression("1", bindings);
  ASSERT_FALSE(nonfinite.ok());
  EXPECT_EQ(nonfinite.error().code, ErrorCode::kNonFinite);
  auto expression = CompileExpression("v(a)", Bindings());
  ASSERT_TRUE(expression.ok());
  auto size = EvaluateExpression(expression.value(), {});
  ASSERT_FALSE(size.ok());
  EXPECT_EQ(size.error().code, ErrorCode::kInvalidStructure);
  const std::array<double, 4> bad_state = {
      std::numeric_limits<double>::quiet_NaN(), 0, 0, 0};
  auto bad = EvaluateExpression(expression.value(), bad_state);
  ASSERT_FALSE(bad.ok());
  EXPECT_EQ(bad.error().code, ErrorCode::kNonFinite);
  EXPECT_FALSE(EvaluateExpression(CompiledExpression{}, {}).ok());
}

TEST(Emi02B, MagnitudeBoundCoversDiscardedIntermediatesAndDerivatives) {
  const auto boundary = Eval("v(a)", {1e100, 0, 0, 0});
  EXPECT_EQ(boundary.value, 1e100);
  auto literal = CompileExpression("1.01e100", {});
  ASSERT_FALSE(literal.ok());
  EXPECT_EQ(literal.error().code, ErrorCode::kNonFinite);
  auto intermediate = CompileExpression("(v(a)*1e100)*0", Bindings());
  ASSERT_TRUE(intermediate.ok());
  const std::array<double, 4> two = {2, 0, 0, 0};
  auto failed = EvaluateExpression(intermediate.value(), two);
  ASSERT_FALSE(failed.ok());
  EXPECT_EQ(failed.error().code, ErrorCode::kNonFinite);
  auto derivative = CompileExpression("1e100*(v(a)*2)", Bindings());
  ASSERT_TRUE(derivative.ok());
  const std::array<double, 4> zero = {0, 0, 0, 0};
  auto overflowed = EvaluateExpression(derivative.value(), zero);
  ASSERT_FALSE(overflowed.ok());
  EXPECT_EQ(overflowed.error().code, ErrorCode::kNonFinite);
  // A lazy arm is never an evaluated intermediate.
  Near(Eval("if(v(a)>0,2,v(a)*1e100)", two).value, 2);
  auto parameters = ResolveParameters({{"a", "exp(231)"}});
  ASSERT_FALSE(parameters.ok());
  EXPECT_EQ(parameters.error().code, ErrorCode::kNonFinite);
}

TEST(Emi02B, SimultaneousInvalidInputsHaveDeterministicPrecedence) {
  auto bindings = Bindings();
  bindings.state_size = 513;
  auto shape = CompileExpression("", bindings);
  ASSERT_FALSE(shape.ok());
  EXPECT_EQ(shape.error().code, ErrorCode::kUnsupportedSize);
  bindings = Bindings();
  bindings.node_indices["A"] = 1;
  auto duplicate_before_syntax = CompileExpression("(", bindings);
  ASSERT_FALSE(duplicate_before_syntax.ok());
  EXPECT_EQ(duplicate_before_syntax.error().code, ErrorCode::kCompile);
  auto binding_first = CompileExpression("unknown+", Bindings());
  ASSERT_FALSE(binding_first.ok());
  EXPECT_EQ(binding_first.error().code, ErrorCode::kUnsupported);
  auto syntax_first = CompileExpression("*unknown", Bindings());
  ASSERT_FALSE(syntax_first.ok());
  EXPECT_EQ(syntax_first.error().code, ErrorCode::kParse);
}

void SameEvaluationBits(const Result<ExpressionEvaluation> &actual,
                        const Result<ExpressionEvaluation> &expected) {
  ASSERT_EQ(actual.ok(), expected.ok());
  if (!actual.ok()) {
    EXPECT_EQ(actual.error().code, expected.error().code);
    EXPECT_EQ(actual.error().message, expected.error().message);
    return;
  }
  EXPECT_EQ(std::bit_cast<std::uint64_t>(actual.value().value),
            std::bit_cast<std::uint64_t>(expected.value().value));
  const auto &a = actual.value().derivatives;
  const auto &b = expected.value().derivatives;
  ASSERT_EQ(a.size(), b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a[i].first, b[i].first);
    EXPECT_EQ(std::bit_cast<std::uint64_t>(a[i].second),
              std::bit_cast<std::uint64_t>(b[i].second));
  }
}

void SameScalarEvaluation(const Result<double> &actual,
                          const Result<ExpressionEvaluation> &expected) {
  ASSERT_EQ(actual.ok(), expected.ok());
  if (!actual.ok()) {
    EXPECT_EQ(actual.error().code, expected.error().code);
    EXPECT_EQ(actual.error().message, expected.error().message);
    return;
  }
  EXPECT_EQ(std::bit_cast<std::uint64_t>(actual.value()),
            std::bit_cast<std::uint64_t>(expected.value().value));
}

TEST(Emi02B, SimpleExpressionsMatchOriginalGenericValueAndDerivativeBits) {
  auto bindings = Bindings();
  bindings.node_indices.emplace("same", 0);
  bindings.parameters.emplace("negative_zero", -0.0);
  const double minimum = std::numeric_limits<double>::denorm_min();
  const double infinity = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> values{0.0,
                             -0.0,
                             minimum,
                             -minimum,
                             std::numeric_limits<double>::min(),
                             -std::numeric_limits<double>::min(),
                             1.0,
                             -1.0,
                             2.5,
                             1e100,
                             -1e100,
                             std::nextafter(1e100, 0.0),
                             std::nextafter(1e100, infinity),
                             -std::nextafter(1e100, infinity),
                             std::numeric_limits<double>::max(),
                             infinity,
                             -infinity,
                             nan};
  std::uint64_t bits = 0x7b63f194de50a28cULL;
  for (std::size_t i = 0; i < 16; ++i) {
    bits = bits * 6364136223846793005ULL + 1442695040888963407ULL;
    values.push_back(std::bit_cast<double>(bits));
  }
  const std::vector<std::string> expressions{
      "v(a)",           "i(vsense)", "v(0)",      "0",          "gain",
      "negative_zero",  "v(a)-gain", "gain-v(a)", "1e100-v(a)", "v(a)-1e100",
      "v(a,b)",         "v(b,a)",    "v(a,a)",    "v(a,same)",  "v(a,0)",
      "v(0,a)",         "v(0,0)",    "0-v(a)",    "v(a)-0",     "v(a)-v(b)",
      "i(vsense)-v(a)", "0-0",       "+v(a)",     "-v(a)",      "v(a)-(-gain)"};
  for (const auto &text : expressions) {
    SCOPED_TRACE(text);
    auto simple = CompileExpression(text, bindings);
    // The root IF forces the original recursive evaluator. Its chosen arm
    // keeps exactly the simple expression's operations, including signed zero
    // and reverse traversal; the extra adjoint factor is exactly one.
    auto generic = CompileExpression("if(1," + text + ",0)", bindings);
    ASSERT_TRUE(simple.ok());
    ASSERT_TRUE(generic.ok());
    EXPECT_EQ(generic.value().node_count(), simple.value().node_count() + 3);
    EXPECT_EQ(std::vector<std::size_t>(simple.value().dependencies().begin(),
                                       simple.value().dependencies().end()),
              std::vector<std::size_t>(generic.value().dependencies().begin(),
                                       generic.value().dependencies().end()));
    for (const double a : values) {
      for (const double b : values) {
        // Unused state remains intentionally nonfinite. Public expression
        // evaluation admits only its structural dependencies, as before.
        const std::array<double, 4> state{a, b, nan, b};
        SameEvaluationBits(EvaluateExpression(simple.value(), state),
                           EvaluateExpression(generic.value(), state));
        SameScalarEvaluation(
            internal::EvaluateExpressionValue(simple.value(), state),
            internal::EvaluateExpressionOriginalAdForTesting(simple.value(),
                                                             state));
      }
    }
    const std::array<double, 3> wrong_size{nan, infinity, 0};
    SameEvaluationBits(EvaluateExpression(simple.value(), wrong_size),
                       EvaluateExpression(generic.value(), wrong_size));
    SameScalarEvaluation(
        internal::EvaluateExpressionValue(simple.value(), wrong_size),
        internal::EvaluateExpressionOriginalAdForTesting(simple.value(),
                                                         wrong_size));
  }
}

TEST(Emi02B, SimpleShapesPreserveSparseBindingsAndParameterDialect) {
  const ExpressionBindings bindings{
      .state_size = 512,
      .node_indices = {{"high", 511}, {"low", 4}, {"alias", 511}},
      .current_indices = {{"sense", 256}},
      .parameters = {{"negative_zero", -0.0}}};
  std::array<double, 512> state{};
  state[511] = -0.0;
  state[4] = 0.0;
  state[256] = std::numeric_limits<double>::denorm_min();
  for (const std::string text :
       {"v(high)", "v(high,low)", "v(low,high)", "v(high,alias)", "v(0,high)",
        "i(sense)", "negative_zero-v(high)"}) {
    auto simple = CompileExpression(text, bindings);
    auto generic = CompileExpression("if(1," + text + ",0)", bindings);
    ASSERT_TRUE(simple.ok());
    ASSERT_TRUE(generic.ok());
    SameEvaluationBits(EvaluateExpression(simple.value(), state),
                       EvaluateExpression(generic.value(), state));
    SameScalarEvaluation(
        internal::EvaluateExpressionValue(simple.value(), state),
        internal::EvaluateExpressionOriginalAdForTesting(simple.value(),
                                                         state));
    state[511] = std::numeric_limits<double>::quiet_NaN();
    SameEvaluationBits(EvaluateExpression(simple.value(), state),
                       EvaluateExpression(generic.value(), state));
    SameScalarEvaluation(
        internal::EvaluateExpressionValue(simple.value(), state),
        internal::EvaluateExpressionOriginalAdForTesting(simple.value(),
                                                         state));
    state[511] = -0.0;
  }
  ExpressionBindings parameters;
  parameters.parameters.emplace("negative_zero", -0.0);
  for (const std::string text :
       {"negative_zero", "negative_zero-0", "0-negative_zero", "1e100-1e100",
        "1e100-(-1e100)"}) {
    auto simple =
        CompileExpression(text, parameters, ExpressionDialect::kParameter);
    // Parameter syntax excludes IF. Two exact sign reflections force the
    // generic path and preserve every finite constant bit, including -0.
    auto generic = CompileExpression("-(-(" + text + "))", parameters,
                                     ExpressionDialect::kParameter);
    ASSERT_TRUE(simple.ok());
    ASSERT_TRUE(generic.ok());
    SameEvaluationBits(EvaluateExpression(simple.value(), {}),
                       EvaluateExpression(generic.value(), {}));
    SameScalarEvaluation(
        internal::EvaluateExpressionValue(simple.value(), {}),
        internal::EvaluateExpressionOriginalAdForTesting(simple.value(), {}));
  }
}

TEST(Emi02B, CompiledReverseOrderMatchesOriginalFullTraversal) {
  const std::vector<std::string> expressions{
      "v(a)",
      "v(a,b)",
      "v(a,a)",
      "i(vsense)",
      "v(0,a)",
      "1+2*3",
      "-gain",
      "v(a)-gain",
      "gain-v(a)",
      "v(a)<v(b)",
      "v(a)>v(b)",
      "v(a)<v(b)<i(vsense)",
      "v(a)*v(b)+i(vsense)*v(a)",
      "v(a)/(2+v(b))",
      "exp(v(a)/5)*(v(a)-v(b))**2+i(vsense)*v(b)",
      "v(a)**v(b)",
      "v(a)**(v(b)>0)",
      "2**v(a)",
      "v(a)**2.62",
      "exp(v(a))",
      "exp(1e100)",
      "(v(a)*1e100)*0",
      "1e100*(v(a)*2)",
      "1e100*(v(a)>0)*2",
      "(1e100*v(a))-(1e100*v(a))",
      "(v(a)-v(a)+v(b)-v(b))**v(a)",
      "if(v(a)>0,v(a)+3,v(b)*i(vsense))",
      "if(v(a)>0,3,if(v(b)<0,exp(v(b)),v(a)**.5))",
      "if(v(a)>0,v(a),v(b)**.5)",
      "if(v(a)>0,1e100*(v(b)>0)*2,v(b))",
      "if(v(a)>0,exp(v(b)),exp(i(vsense)))",
      "if(1,2,0**-1)",
      "if(0,2,0**-1)",
      "if(1,v(a),v(b)**-1)"};
  const double minimum = std::numeric_limits<double>::denorm_min();
  const double infinity = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const std::vector<double> values{
      0.0,      -0.0,   minimum, -minimum,
      1e-32,    -1e-32, 1.0,     -1.0,
      2.0,      -2.0,   14.0,    14.5,
      1000.0,   1e100,  -1e100,  std::nextafter(1e100, infinity),
      infinity, nan};
  for (const auto &text : expressions) {
    SCOPED_TRACE(text);
    auto compiled = CompileExpression(text, Bindings());
    ASSERT_TRUE(compiled.ok()) << compiled.error().message;
    for (std::size_t i = 0; i < values.size(); ++i) {
      for (std::size_t j = 0; j < values.size(); ++j) {
        const std::array<double, 4> state{values[i], values[j], nan,
                                          values[(i + j) % values.size()]};
        SameEvaluationBits(EvaluateExpression(compiled.value(), state),
                           internal::EvaluateExpressionOriginalAdForTesting(
                               compiled.value(), state));
      }
    }
    SameEvaluationBits(
        EvaluateExpression(compiled.value(), {}),
        internal::EvaluateExpressionOriginalAdForTesting(compiled.value(), {}));
  }
  SameEvaluationBits(EvaluateExpression(CompiledExpression{}, {}),
                     internal::EvaluateExpressionOriginalAdForTesting(
                         CompiledExpression{}, {}));
}

TEST(Emi02B, OmittedComparisonVisitsKeepIncomingAdjointAndDomainGuards) {
  auto comparison = CompileExpression("1e100*(v(a)>0)*2", Bindings());
  ASSERT_TRUE(comparison.ok());
  // The value is exactly zero and the mathematical derivative of comparison
  // is zero. Its incoming adjoint nevertheless exceeds the original budget;
  // omitting only the comparison's no-op visit cannot hide that failure.
  const std::array<double, 4> state{-1, 0, 0, 0};
  auto checked = EvaluateExpression(comparison.value(), state);
  ASSERT_FALSE(checked.ok());
  EXPECT_EQ(checked.error().code, ErrorCode::kNonFinite);
  SameEvaluationBits(checked, internal::EvaluateExpressionOriginalAdForTesting(
                                  comparison.value(), state));

  auto lazy = CompileExpression("if(v(a)>0,2,0**-1)", Bindings());
  ASSERT_TRUE(lazy.ok());
  for (const double condition : {1.0, 0.0, -1.0}) {
    const std::array<double, 4> input{condition, 0, 0, 0};
    auto result = EvaluateExpression(lazy.value(), input);
    EXPECT_EQ(result.ok(), condition > 0);
    SameEvaluationBits(result, internal::EvaluateExpressionOriginalAdForTesting(
                                   lazy.value(), input));
  }
  auto inactive = CompileExpression("if(v(a)>0,2,v(b))", Bindings());
  ASSERT_TRUE(inactive.ok());
  const std::array<double, 4> nonfinite{
      1, std::numeric_limits<double>::quiet_NaN(), 0, 0};
  auto rejected = EvaluateExpression(inactive.value(), nonfinite);
  ASSERT_FALSE(rejected.ok());
  EXPECT_EQ(rejected.error().code, ErrorCode::kNonFinite);
  SameEvaluationBits(rejected, internal::EvaluateExpressionOriginalAdForTesting(
                                   inactive.value(), nonfinite));
}

TEST(Emi02B, AlternatingLazyPathsMatchOriginalEvaluation) {
  const std::vector<std::string> texts{
      "if(v(a)>0,0**-1,-exp(v(b)))",
      "if(v(a)>0,0**-1,if(v(b)>0,-i(vsense),exp(i(vsense))))",
      "if(v(a)>0,v(b)*i(vsense),v(b)/i(vsense))",
      "if(v(a)>0,v(b)**i(vsense),-v(b)+i(vsense))",
      "if(v(a)>0,if(v(b)>0,v(b)**.5,exp(i(vsense))),v(b)-i(vsense))",
      "if(v(a)>0,1e100*(v(b)>0)*2,if(v(b)>0,v(b),-i(vsense)))"};
  std::vector<CompiledExpression> expressions;
  for (const auto &text : texts) {
    auto compiled = CompileExpression(text, Bindings());
    ASSERT_TRUE(compiled.ok()) << compiled.error().message;
    expressions.push_back(compiled.TakeValue());
  }
  const double minimum = std::numeric_limits<double>::denorm_min();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const std::array<std::array<double, 4>, 10> states{{
      {-1, 2, nan, 3},
      {1, -2, nan, 3},
      {-1, -2, nan, -3},
      {1, 0, nan, .5},
      {-0.0, -0.0, nan, minimum},
      {1, 1e100, nan, 2},
      {-1, minimum, nan, -minimum},
      {1, nan, 0, 0},
      {-1, 14.5, nan, 1},
      {1, 2, nan, 0},
  }};
  // Reuse stack storage through successes, value failures and AD failures with
  // different selected paths and AST sizes. Inactive arms remain unevaluated;
  // their domains must not affect a later successful traversal.
  for (std::size_t round = 0; round < 12; ++round) {
    for (std::size_t j = 0; j < states.size(); ++j) {
      for (std::size_t i = 0; i < expressions.size(); ++i) {
        const auto &expression = expressions[(i + round) % expressions.size()];
        const auto &state = states[(j + round) % states.size()];
        SameEvaluationBits(EvaluateExpression(expression, state),
                           internal::EvaluateExpressionOriginalAdForTesting(
                               expression, state));
      }
    }
  }

  const auto inactive_domain = EvaluateExpression(expressions[0], states[0]);
  ASSERT_TRUE(inactive_domain.ok());
  EXPECT_EQ(inactive_domain.value().value, -std::exp(2.0));
  EXPECT_EQ(Derivative(inactive_domain.value(), 1), -std::exp(2.0));
  EXPECT_EQ(Derivative(inactive_domain.value(), 0), 0.0);
  const auto active_domain = EvaluateExpression(expressions[0], states[1]);
  ASSERT_FALSE(active_domain.ok());
  EXPECT_EQ(active_domain.error().code, ErrorCode::kNonFinite);
}

TEST(Emi02B, PrivateScalarEvaluationMatchesFullReferenceValueBitsAndErrors) {
  const std::vector<std::string> texts{
      "v(a)",
      "-v(a)",
      "v(a,b)",
      "v(a,a)",
      "v(0,a)",
      "i(vsense)",
      "gain-v(a)",
      "v(a)+v(b)-i(vsense)",
      "v(a)*v(b)",
      "v(a)/(2+v(b))",
      "v(a)**2.5",
      "exp(v(a))",
      "v(a)<v(b)",
      "v(a)>v(b)",
      "if(v(a)>0,exp(v(b)),v(b)**2)",
      "if(v(a)>0,if(v(b)>0,v(a),-v(b)),i(vsense))",
      "if(0,0**-1,v(a))",
      "if(1,v(a),0**-1)"};
  const double minimum = std::numeric_limits<double>::denorm_min();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const std::array<double, 10> values{-14.0,   -1.0, -minimum, -0.0, 0.0,
                                      minimum, 1.0,  2.0,      14.0, 14.5};
  for (const auto &text : texts) {
    SCOPED_TRACE(text);
    auto expression = CompileExpression(text, Bindings());
    ASSERT_TRUE(expression.ok());
    for (double a : values) {
      for (double b : values) {
        const std::array<double, 4> state{a, b, nan, b};
        auto original = internal::EvaluateExpressionOriginalAdForTesting(
            expression.value(), state);
        ASSERT_TRUE(original.ok()) << original.error().message;
        SameScalarEvaluation(
            internal::EvaluateExpressionValue(expression.value(), state),
            original);
      }
    }
    SameScalarEvaluation(
        internal::EvaluateExpressionValue(expression.value(), {}),
        internal::EvaluateExpressionOriginalAdForTesting(expression.value(),
                                                         {}));
  }

  for (const std::string text :
       {"0**-1", "exp(1e100)", "1e100*(v(a)+2)", "if(1,2,v(b))"}) {
    auto expression = CompileExpression(text, Bindings());
    ASSERT_TRUE(expression.ok());
    const std::array<double, 4> state{1, nan, 0, 0};
    auto original = internal::EvaluateExpressionOriginalAdForTesting(
        expression.value(), state);
    ASSERT_FALSE(original.ok());
    SameScalarEvaluation(
        internal::EvaluateExpressionValue(expression.value(), state), original);
  }
  for (const std::string text :
       {"-0", "0-(-0)", "exp(2)", "exp(1000)", "1/0", "0**-1"}) {
    auto expression =
        CompileExpression(text, {}, ExpressionDialect::kParameter);
    ASSERT_TRUE(expression.ok());
    SameScalarEvaluation(
        internal::EvaluateExpressionValue(expression.value(), {}),
        internal::EvaluateExpressionOriginalAdForTesting(expression.value(),
                                                         {}));
  }
  SameScalarEvaluation(
      internal::EvaluateExpressionValue({}, {}),
      internal::EvaluateExpressionOriginalAdForTesting({}, {}));
}

TEST(Emi02B, PrivateScalarSuccessDoesNotProveDerivativeAdmissibility) {
  for (const auto &[text, input] :
       std::array<std::pair<const char *, double>, 2>{
           {{"v(a)**.5", 0}, {"1e100*(v(a)>0)*2", -1}}}) {
    auto expression = CompileExpression(text, Bindings());
    ASSERT_TRUE(expression.ok());
    const std::array<double, 4> state{input, 0, 0, 0};
    auto value = internal::EvaluateExpressionValue(expression.value(), state);
    ASSERT_TRUE(value.ok());
    EXPECT_EQ(std::bit_cast<std::uint64_t>(value.value()),
              std::bit_cast<std::uint64_t>(0.0));
    auto full = EvaluateExpression(expression.value(), state);
    ASSERT_FALSE(full.ok());
    EXPECT_EQ(full.error().code, ErrorCode::kNonFinite);
    SameEvaluationBits(full, internal::EvaluateExpressionOriginalAdForTesting(
                                 expression.value(), state));
  }
}

} // namespace
} // namespace ohmnivore
