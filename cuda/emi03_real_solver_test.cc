#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "cuda/emi03_real_solver.h"
#include "ohmnivore/behavioral.h"
#include "ohmnivore/solver.h"
#include "ohmnivore/transient.h"

namespace ohmnivore {
namespace {

CsrMatrix Matrix() {
  return {.rows = 2,
          .columns = 2,
          .values = {0, 2, 3, 1},
          .column_indices = {0, 1, 0, 1},
          .row_offsets = {0, 2, 4}};
}

class Emi03RealSolver : public ::testing::Test {
protected:
  void SetUp() override {
    auto started = BeginEmi03CudaJob("real-solver-test");
    ASSERT_TRUE(started.ok()) << started.error().message;
  }
  void TearDown() override {
    auto ended = EndEmi03CudaJob();
    ASSERT_TRUE(ended.ok()) << ended.error().message;
    EXPECT_EQ(ended.value().outstanding_device_bytes, 0U);
    EXPECT_EQ(ended.value().cleanup_failures, 0U);
    EXPECT_LE(ended.value().peak_device_bytes,
              kEmi03WorkerDeviceAllocationLimit);
  }
};

TEST_F(Emi03RealSolver, PivotingRefactorReuseAndOriginalEquationValidation) {
  auto matrix = Matrix();
  auto analyzed = SparseRealFactorization::Analyze(matrix);
  ASSERT_TRUE(analyzed.ok()) << analyzed.error().message;
  auto first = analyzed.value()->FactorAndSolve(matrix, {4, 11});
  ASSERT_TRUE(first.ok()) << first.error().message;
  EXPECT_NEAR(first.value()[0], 3, 1e-13);
  EXPECT_NEAR(first.value()[1], 2, 1e-13);
  EXPECT_TRUE(ValidateSparseSolution(matrix, {4, 11}, first.value()).ok());
  auto second = analyzed.value()->FactorAndSolve(matrix, {8, 7});
  ASSERT_TRUE(second.ok()) << second.error().message;
  EXPECT_NEAR(second.value()[0], 1, 1e-13);
  EXPECT_NEAR(second.value()[1], 4, 1e-13);
  matrix.values[1] = 4;
  auto refreshed = analyzed.value()->FactorAndSolve(matrix, {8, 11});
  ASSERT_TRUE(refreshed.ok()) << refreshed.error().message;
  EXPECT_NEAR(refreshed.value()[0], 3, 1e-13);
  EXPECT_NEAR(refreshed.value()[1], 2, 1e-13);
  EXPECT_EQ(analyzed.value()->statistics().numeric_reuses, 1U);
  EXPECT_EQ(analyzed.value()->statistics().numeric_refactorizations, 1U);
  EXPECT_EQ(SnapshotEmi03CudaJob().successful_solves, 3U);
}

TEST_F(Emi03RealSolver, RowEquilibrationPreservesMixedPhysicalUnits) {
  const CsrMatrix matrix{.rows = 2,
                         .columns = 2,
                         .values = {1e-20, 1e20},
                         .column_indices = {0, 1},
                         .row_offsets = {0, 1, 2}};
  auto result = SolveSparseReal(matrix, {3e-20, -2e20});
  ASSERT_TRUE(result.ok()) << result.error().message;
  EXPECT_NEAR(result.value()[0], 3, 1e-14);
  EXPECT_NEAR(result.value()[1], -2, 1e-14);
}

TEST_F(Emi03RealSolver, MandatoryOriginalResidualRefinesWeakCommonMode) {
  const CsrMatrix matrix{.rows = 2,
                         .columns = 2,
                         .values = {1e8 + 1, -1e8, -1e8, 1e8 + 1},
                         .column_indices = {0, 1, 0, 1},
                         .row_offsets = {0, 2, 4}};
  auto analyzed = SparseRealFactorization::Analyze(matrix);
  ASSERT_TRUE(analyzed.ok()) << analyzed.error().message;
  auto refined = analyzed.value()->FactorAndSolveRefined(matrix, {128, 128});
  ASSERT_TRUE(refined.ok()) << refined.error().message;
  EXPECT_NEAR(refined.value()[0], 128, 5e-9);
  EXPECT_NEAR(refined.value()[1], 128, 5e-9);
  EXPECT_GT(analyzed.value()->statistics().iterative_refinement_solves, 0U);
  EXPECT_LE(analyzed.value()->statistics().iterative_refinement_solves, 4U);
}

TEST_F(Emi03RealSolver, ZeroRhsCannotHideSingularAcceptedJacobian) {
  auto matrix = Matrix();
  auto analyzed = SparseRealFactorization::Analyze(matrix);
  ASSERT_TRUE(analyzed.ok()) << analyzed.error().message;
  ASSERT_TRUE(analyzed.value()->FactorAndSolve(matrix, {0, 0}).ok());
  matrix.values = {1, 2, 2, 4};
  auto singular = analyzed.value()->FactorAndSolve(matrix, {0, 0});
  ASSERT_FALSE(singular.ok());
  EXPECT_TRUE(singular.error().code == ErrorCode::kSingular ||
              singular.error().code == ErrorCode::kFactorization);
  EXPECT_EQ(analyzed.value()->statistics().solves, 1U);
  EXPECT_EQ(SnapshotEmi03CudaJob().successful_solves, 1U);
}

TEST_F(Emi03RealSolver, MalformedChangedPatternAndNonfiniteInputsFail) {
  auto matrix = Matrix();
  auto analyzed = SparseRealFactorization::Analyze(matrix);
  ASSERT_TRUE(analyzed.ok()) << analyzed.error().message;
  auto short_rhs = analyzed.value()->FactorAndSolve(matrix, {1});
  ASSERT_FALSE(short_rhs.ok());
  EXPECT_EQ(short_rhs.error().code, ErrorCode::kInvalidStructure);
  auto nonfinite = analyzed.value()->FactorAndSolve(
      matrix, {std::numeric_limits<double>::infinity(), 0});
  ASSERT_FALSE(nonfinite.ok());
  EXPECT_EQ(nonfinite.error().code, ErrorCode::kNonFinite);
  auto excessive = analyzed.value()->FactorAndSolveRefined(matrix, {0, 0}, 5);
  ASSERT_FALSE(excessive.ok());
  EXPECT_EQ(excessive.error().code, ErrorCode::kUnsupportedSize);
  matrix.column_indices[0] = 1;
  EXPECT_FALSE(analyzed.value()->FactorAndSolve(matrix, {0, 0}).ok());
  matrix = Matrix();
  matrix.values[0] = std::numeric_limits<double>::quiet_NaN();
  auto nan = analyzed.value()->FactorAndSolve(matrix, {0, 0});
  ASSERT_FALSE(nan.ok());
  EXPECT_EQ(nan.error().code, ErrorCode::kNonFinite);
  EXPECT_EQ(SnapshotEmi03CudaJob().successful_solves, 0U);
}

TEST_F(Emi03RealSolver, EmptySystemUsesNoDeviceBuffers) {
  const CsrMatrix empty{.rows = 0,
                        .columns = 0,
                        .values = {},
                        .column_indices = {},
                        .row_offsets = {0}};
  auto result = SolveSparseReal(empty, {});
  ASSERT_TRUE(result.ok());
  EXPECT_TRUE(result.value().empty());
  EXPECT_EQ(SnapshotEmi03CudaJob().peak_device_bytes, 0U);
}

TEST_F(Emi03RealSolver, BehavioralTransientPreservesDisplacementCurrent) {
  auto parsed = ParseBehavioralNetlist(
      "* affine charge oracle\nVdrive drive 0 DC 0 PWL(0 0 50u 0.5 100u 1)\n"
      "Cbase drive sense 1u\nVsense sense 0 0\n"
      "Bextra drive 0 I={i(Vsense)*(2+0.5*v(drive))}\n"
      ".TRAN 2u 100u\n.end\n");
  ASSERT_TRUE(parsed.ok()) << parsed.error().message;
  auto compiled = CompileBehavioralMna(parsed.value());
  ASSERT_TRUE(compiled.ok()) << compiled.error().message;
  const auto &system = compiled.value();
  TransientExecutionLimits limits;
  limits.behavioral_error_estimator =
      BehavioralErrorEstimator::kDerivativeHistory;
  auto solved = RunTransientAnalysis(system, {2e-6, 1e-4, 0, false}, limits);
  ASSERT_TRUE(solved.ok()) << solved.error().message;
  ASSERT_GT(solved.value().states.size(), 3U);
  const auto drive = system.node_names.size();
  for (std::size_t i = 2; i < solved.value().states.size(); ++i) {
    const double voltage = solved.value().times_seconds[i] / 1e-4;
    EXPECT_NEAR(solved.value().states[i][0], voltage, 1e-12);
    // The ideal PWL ramp stops at the final sample: its derivative (hence
    // displacement current) has a discontinuity there. Test the ramp current
    // on its open interval and the voltage on every emitted sample.
    if (solved.value().times_seconds[i] >= 1e-4 - 1e-15)
      continue;
    const double current = 1e-6 * (3 + 0.5 * voltage) / 1e-4;
    EXPECT_NEAR(-solved.value().states[i][drive], current, 1e-8);
  }
  EXPECT_GT(SnapshotEmi03CudaJob().expression_batches, 0U);
  EXPECT_GT(SnapshotEmi03CudaJob().successful_solves, 0U);
  EXPECT_NE(
      std::find_if(solved.value().times_seconds.begin(),
                   solved.value().times_seconds.end(),
                   [](double time) { return std::abs(time - 50e-6) < 1e-15; }),
      solved.value().times_seconds.end());
  auto repeated = RunTransientAnalysis(system, {2e-6, 1e-4, 0, false}, limits);
  ASSERT_TRUE(repeated.ok()) << repeated.error().message;
  ASSERT_EQ(repeated.value().times_seconds, solved.value().times_seconds);
  ASSERT_EQ(repeated.value().states.size(), solved.value().states.size());
  for (std::size_t i = 0; i < repeated.value().states.size(); ++i)
    for (std::size_t j = 0; j < repeated.value().states[i].size(); ++j)
      EXPECT_NEAR(repeated.value().states[i][j], solved.value().states[i][j],
                  1e-12);
  limits.maximum_accepted_steps = 2;
  auto exhausted = RunTransientAnalysis(system, {2e-6, 1e-4, 0, false}, limits);
  ASSERT_FALSE(exhausted.ok());
  EXPECT_EQ(exhausted.error().code, ErrorCode::kSolve);
}

TEST(Emi03RealFailures, FaultsFailClosedAndReleaseAllDeviceBuffers) {
  for (auto fault :
       {Emi03CudaFault::kAllocationFailure,
        Emi03CudaFault::kCudssAllocationFailure, Emi03CudaFault::kCudssFailure,
        Emi03CudaFault::kCudaFailure, Emi03CudaFault::kDataInfoFailure,
        Emi03CudaFault::kNonFiniteSolve, Emi03CudaFault::kWrongSolve}) {
    SCOPED_TRACE(static_cast<int>(fault));
    Emi03CudaOptions options;
    options.fault = fault;
    ASSERT_TRUE(BeginEmi03CudaJob("fault-test", options).ok());
    auto result = SolveSparseReal(Matrix(), {4, 11});
    EXPECT_FALSE(result.ok());
    if (!result.ok() && (fault == Emi03CudaFault::kAllocationFailure ||
                         fault == Emi03CudaFault::kCudssAllocationFailure)) {
      EXPECT_EQ(result.error().code, ErrorCode::kUnsupportedSize);
    }
    EXPECT_EQ(SnapshotEmi03CudaJob().successful_solves, 0U);
    auto ended = EndEmi03CudaJob();
    ASSERT_TRUE(ended.ok()) << ended.error().message;
    EXPECT_EQ(ended.value().outstanding_device_bytes, 0U);
  }
  ASSERT_TRUE(BeginEmi03CudaJob("clean-after-fault").ok());
  EXPECT_TRUE(SolveSparseReal(Matrix(), {4, 11}).ok());
  EXPECT_TRUE(EndEmi03CudaJob().ok());
}

TEST(Emi03RealFailures, DeviceCapIsEnforcedBeforeAllocation) {
  Emi03CudaOptions options;
  options.maximum_device_bytes = 16;
  ASSERT_TRUE(BeginEmi03CudaJob("small-budget", options).ok());
  auto result = SolveSparseReal(Matrix(), {4, 11});
  EXPECT_FALSE(result.ok());
  auto ended = EndEmi03CudaJob();
  ASSERT_TRUE(ended.ok()) << ended.error().message;
  EXPECT_LE(ended.value().peak_device_bytes, 16U);
  EXPECT_GT(ended.value().allocation_failures, 0U);
}

TEST(Emi03RealFailures, ExplicitJobAndClosedOwnershipAreRequired) {
  auto outside = SparseRealFactorization::Analyze(Matrix());
  ASSERT_FALSE(outside.ok());
  EXPECT_EQ(outside.error().code, ErrorCode::kUnsupported);
  ASSERT_TRUE(BeginEmi03CudaJob("ownership").ok());
  {
    auto analyzed = SparseRealFactorization::Analyze(Matrix());
    ASSERT_TRUE(analyzed.ok()) << analyzed.error().message;
    EXPECT_FALSE(EndEmi03CudaJob().ok());
    EXPECT_FALSE(BeginEmi03CudaJob("overlap").ok());
  }
  auto ended = EndEmi03CudaJob();
  ASSERT_TRUE(ended.ok()) << ended.error().message;
  EXPECT_EQ(ended.value().job_id, "ownership");
}

} // namespace
} // namespace ohmnivore
