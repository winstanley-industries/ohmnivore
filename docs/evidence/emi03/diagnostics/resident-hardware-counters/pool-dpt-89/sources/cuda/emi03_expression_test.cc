#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "cuda/emi03_expression.h"
#include "cuda/emi03_real_solver.h"

namespace ohmnivore {
namespace {

ExpressionBindings Bindings() {
  ExpressionBindings bindings;
  bindings.state_size = 2;
  bindings.node_indices = {{"A", 0}, {"B", 1}};
  return bindings;
}

CompiledExpression Compile(const std::string &source) {
  auto result = CompileExpression(source, Bindings());
  EXPECT_TRUE(result.ok()) << (result.ok() ? "" : result.error().message);
  return result.ok() ? result.TakeValue() : CompiledExpression{};
}

class Emi03Expression : public ::testing::Test {
protected:
  void SetUp() override {
    auto started = BeginEmi03CudaJob("expression-test");
    ASSERT_TRUE(started.ok()) << started.error().message;
  }
  void TearDown() override {
    auto ended = EndEmi03CudaJob();
    ASSERT_TRUE(ended.ok()) << ended.error().message;
    EXPECT_EQ(ended.value().outstanding_device_bytes, 0U);
    EXPECT_EQ(ended.value().cleanup_failures, 0U);
  }
};

TEST_F(Emi03Expression, AllOperationsAndDerivativesMatchCpuAuthority) {
  std::vector<CompiledExpression> programs;
  for (const auto *source :
       {"1", "V(A)", "-V(A)", "V(A)+V(B)", "V(A)-V(B)", "V(A)*V(B)",
        "V(A)/V(B)", "V(A)**V(B)", "EXP(V(A))", "V(A)<V(B)", "V(A)>V(B)",
        "IF(V(A)>0,V(A)*V(B),-V(B))",
        "IF(V(A)>0,IF(V(B)>2,EXP(V(A)),V(B)),V(A)**2)"})
    programs.push_back(Compile(source));
  for (const auto &state : std::vector<std::vector<double>>{
           {2, 3}, {-2, 3}, {0, 3}, {14, 0.5}, {14.00001, 2}}) {
    auto gpu = EvaluateEmi03CudaExpressions(programs, state, true);
    ASSERT_TRUE(gpu.ok()) << gpu.error().message;
    ASSERT_EQ(gpu.value().size(), programs.size());
    auto values = EvaluateEmi03CudaExpressions(programs, state, false);
    ASSERT_TRUE(values.ok()) << values.error().message;
    for (std::size_t i = 0; i < programs.size(); ++i) {
      auto cpu = EvaluateExpression(programs[i], state);
      ASSERT_TRUE(cpu.ok()) << cpu.error().message;
      const auto &actual = gpu.value()[i];
      EXPECT_NEAR(actual.value, cpu.value().value,
                  2e-12 * std::max(1.0, std::abs(cpu.value().value)));
      EXPECT_EQ(
          std::memcmp(&actual.value, &values.value()[i].value, sizeof(double)),
          0);
      ASSERT_EQ(actual.derivatives.size(), cpu.value().derivatives.size());
      for (std::size_t j = 0; j < actual.derivatives.size(); ++j) {
        EXPECT_EQ(actual.derivatives[j].first,
                  cpu.value().derivatives[j].first);
        EXPECT_NEAR(
            actual.derivatives[j].second, cpu.value().derivatives[j].second,
            2e-12 * std::max(1.0, std::abs(cpu.value().derivatives[j].second)));
      }
    }
  }
  EXPECT_EQ(SnapshotEmi03CudaJob().expression_program_uploads, 1U);
  EXPECT_EQ(SnapshotEmi03CudaJob().expression_batches, 10U);
}

TEST_F(Emi03Expression, LazyIfAndValueOnlyDerivativeBoundaryPreserveErrors) {
  const std::vector<CompiledExpression> lazy{Compile("IF(V(A)>0,V(B),0**-1)")};
  const std::vector<double> positive{1, 2}, negative{-1, 2}, zero{0, 2};
  auto accepted = EvaluateEmi03CudaExpressions(lazy, positive, true);
  ASSERT_TRUE(accepted.ok()) << accepted.error().message;
  EXPECT_EQ(accepted.value()[0].value, 2);
  auto domain = EvaluateEmi03CudaExpressions(lazy, negative, true);
  ASSERT_FALSE(domain.ok());
  EXPECT_EQ(domain.error().code, ErrorCode::kNonFinite);
  const std::vector<CompiledExpression> derivative{Compile("V(A)**0.5")};
  auto value = EvaluateEmi03CudaExpressions(derivative, zero, false);
  ASSERT_TRUE(value.ok()) << value.error().message;
  EXPECT_EQ(value.value()[0].value, 0);
  auto gradient = EvaluateEmi03CudaExpressions(derivative, zero, true);
  ASSERT_FALSE(gradient.ok());
  EXPECT_EQ(gradient.error().code, ErrorCode::kNonFinite);
}

TEST_F(Emi03Expression, ConstantOnlyZeroStateAndStrictDivisionAreSupported) {
  const ExpressionBindings empty;
  auto constant = CompileExpression("2+3*4", empty);
  ASSERT_TRUE(constant.ok());
  const std::vector<CompiledExpression> constants{constant.TakeValue()};
  auto value = EvaluateEmi03CudaExpressions(constants, {}, true);
  ASSERT_TRUE(value.ok()) << value.error().message;
  EXPECT_EQ(value.value()[0].value, 14);
  EXPECT_TRUE(value.value()[0].derivatives.empty());
  auto behavioral = CompileExpression("1/0", empty);
  auto strict = CompileExpression("1/0", empty, ExpressionDialect::kParameter);
  ASSERT_TRUE(behavioral.ok());
  ASSERT_TRUE(strict.ok());
  const std::vector<CompiledExpression> behavioral_program{
      behavioral.TakeValue()};
  const std::vector<CompiledExpression> strict_program{strict.TakeValue()};
  auto finite = EvaluateEmi03CudaExpressions(behavioral_program, {}, true);
  ASSERT_TRUE(finite.ok()) << finite.error().message;
  EXPECT_NEAR(finite.value()[0].value, 1e32, 1e18);
  auto invalid = EvaluateEmi03CudaExpressions(strict_program, {}, true);
  ASSERT_FALSE(invalid.ok());
  EXPECT_EQ(invalid.error().code, ErrorCode::kNonFinite);
}

TEST_F(Emi03Expression, DuplicateDependenciesAccumulateInOriginalAdOrder) {
  const std::vector<CompiledExpression> programs{
      Compile("V(A)*V(A)+V(A)*V(B)-V(A)")};
  const std::vector<double> state{3, 5};
  auto result = EvaluateEmi03CudaExpressions(programs, state, true);
  ASSERT_TRUE(result.ok()) << result.error().message;
  EXPECT_EQ(result.value()[0].value, 21);
  ASSERT_EQ(result.value()[0].derivatives.size(), 2U);
  EXPECT_EQ(result.value()[0].derivatives[0].second, 10);
  EXPECT_EQ(result.value()[0].derivatives[1].second, 3);
}

TEST_F(Emi03Expression, UnequalProgramWorkspacesStayPrivateAcrossBlocks) {
  std::vector<CompiledExpression> programs;
  for (std::size_t i = 0; i < 130; ++i) {
    std::string expression = "V(A)*" + std::to_string(i + 1);
    if (i % 2 == 0)
      expression += "+IF(V(B)>0,V(B)*V(B),-V(B))";
    programs.push_back(Compile(expression));
  }
  for (const auto &state :
       std::vector<std::vector<double>>{{2, 3}, {-1, -4}, {0, 0}, {2, 3}}) {
    for (bool derivatives : {false, true}) {
      auto gpu = EvaluateEmi03CudaExpressions(programs, state, derivatives);
      ASSERT_TRUE(gpu.ok()) << gpu.error().message;
      ASSERT_EQ(gpu.value().size(), programs.size());
      for (std::size_t i = 0; i < programs.size(); ++i) {
        auto cpu = EvaluateExpression(programs[i], state);
        ASSERT_TRUE(cpu.ok());
        EXPECT_EQ(gpu.value()[i].value, cpu.value().value);
        if (derivatives) {
          EXPECT_EQ(gpu.value()[i].derivatives, cpu.value().derivatives);
        }
      }
    }
  }
  EXPECT_EQ(SnapshotEmi03CudaJob().expression_program_uploads, 1U);
  EXPECT_LT(SnapshotEmi03CudaJob().peak_device_bytes, 1024U * 1024U);
}

TEST_F(Emi03Expression, DependencyChecksIncludeInactiveArms) {
  const std::vector<CompiledExpression> programs{Compile("IF(V(A)>0,1,V(B))")};
  const std::vector<double> state{1, std::numeric_limits<double>::quiet_NaN()};
  auto result = EvaluateEmi03CudaExpressions(programs, state, true);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.error().code, ErrorCode::kNonFinite);
}

TEST_F(Emi03Expression, ProgramOwnershipOrderAndChangingStateStayPrivate) {
  std::vector<CompiledExpression> programs{Compile("V(A)"), Compile("V(B)")};
  const std::vector<double> state{2, 3};
  auto first = EvaluateEmi03CudaExpressions(programs, state, true);
  ASSERT_TRUE(first.ok());
  std::swap(programs[0], programs[1]);
  auto reordered = EvaluateEmi03CudaExpressions(programs, state, true);
  ASSERT_TRUE(reordered.ok());
  EXPECT_EQ(reordered.value()[0].value, 3);
  EXPECT_EQ(reordered.value()[1].value, 2);
  programs = {Compile("V(A)+1"), Compile("V(B)+1")};
  auto replaced = EvaluateEmi03CudaExpressions(programs, state, true);
  ASSERT_TRUE(replaced.ok());
  EXPECT_EQ(replaced.value()[0].value, 3);
  EXPECT_EQ(replaced.value()[1].value, 4);
  EXPECT_EQ(SnapshotEmi03CudaJob().expression_program_uploads, 3U);
}

TEST_F(Emi03Expression, UncompiledAndWrongStateAreTypedFailures) {
  const std::vector<CompiledExpression> uncompiled(1);
  const std::vector<double> state{1, 2};
  auto invalid = EvaluateEmi03CudaExpressions(uncompiled, state, true);
  ASSERT_FALSE(invalid.ok());
  EXPECT_EQ(invalid.error().code, ErrorCode::kInvalidStructure);
  const std::vector<CompiledExpression> compiled{Compile("V(A)")};
  const std::vector<double> short_state{1};
  auto dimensions = EvaluateEmi03CudaExpressions(compiled, short_state, true);
  ASSERT_FALSE(dimensions.ok());
  EXPECT_EQ(dimensions.error().code, ErrorCode::kInvalidStructure);
}

TEST(Emi03ExpressionFailures, NonfiniteReadbackAndAllocationFailClosed) {
  const std::vector<CompiledExpression> programs{Compile("V(A)+V(B)")};
  const std::vector<double> state{2, 3};
  for (const auto fault : {Emi03CudaFault::kNonFiniteExpression,
                           Emi03CudaFault::kAllocationFailure}) {
    Emi03CudaOptions options;
    options.fault = fault;
    ASSERT_TRUE(BeginEmi03CudaJob("expression-fault", options).ok());
    auto result = EvaluateEmi03CudaExpressions(programs, state, true);
    EXPECT_FALSE(result.ok());
    auto ended = EndEmi03CudaJob();
    ASSERT_TRUE(ended.ok()) << ended.error().message;
    EXPECT_EQ(ended.value().outstanding_device_bytes, 0U);
  }
}

TEST(Emi03ExpressionFailures, FiniteCorruptionRequiresDifferentialValidation) {
  const std::vector<CompiledExpression> programs{Compile("V(A)+V(B)")};
  const std::vector<double> state{2, 3};
  Emi03CudaOptions options;
  options.fault = Emi03CudaFault::kWrongExpression;
  ASSERT_TRUE(BeginEmi03CudaJob("finite-expression-fault", options).ok());
  auto gpu = EvaluateEmi03CudaExpressions(programs, state, true);
  ASSERT_TRUE(gpu.ok());
  auto cpu = EvaluateExpression(programs[0], state);
  ASSERT_TRUE(cpu.ok());
  EXPECT_NE(gpu.value()[0].value, cpu.value().value);
  // Intrinsic finiteness is deliberately insufficient to certify a trajectory.
  auto ended = EndEmi03CudaJob();
  ASSERT_TRUE(ended.ok()) << ended.error().message;
}

} // namespace
} // namespace ohmnivore
