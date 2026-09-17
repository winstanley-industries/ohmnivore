#include <gtest/gtest.h>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "cuda/prepared_ac_cuda.h"
#include "ohmnivore/compiler.h"
#include "ohmnivore/ir.h"
#include "ohmnivore/prepared_ac.h"
#include "ohmnivore/status.h"

namespace ohmnivore {
namespace {

[[nodiscard]] PreparedAcBatch
PrepareAnalyticBatch(std::string circuit_id = "diagonal-two",
                     double conductance_a = 2.0, double conductance_b = 4.0,
                     std::complex<double> solution_a = {2.0, 3.0},
                     std::complex<double> solution_b = {2.0, -1.0}) {
  MnaSystem system{
      .g =
          CsrMatrix{
              .rows = 2,
              .columns = 2,
              .values = {conductance_a, conductance_b},
              .column_indices = {0, 1},
              .row_offsets = {0, 1, 2},
          },
      .c =
          CsrMatrix{
              .rows = 2,
              .columns = 2,
              .values = {},
              .column_indices = {},
              .row_offsets = {0, 0, 0},
          },
      .b_dc = {0.0, 0.0},
      .b_ac = {conductance_a * solution_a, conductance_b * solution_b},
      .node_names = {"a", "b"},
      .branch_names = {},
  };
  auto prepared = PrepareLinearAcBatch(
      system,
      AcAnalysis{.sweep_type = AcSweepType::kLin,
                 .points = 2,
                 .start_frequency_hz = 1.0,
                 .stop_frequency_hz = 2.0},
      "gpu02-analytic/v1", std::move(circuit_id), "nominal");
  if (!prepared.ok()) {
    ADD_FAILURE() << prepared.error().message;
    return {};
  }
  return prepared.TakeValue();
}

void ExpectRejected(const PreparedAcBatch &batch,
                    const PreparedAcBatchResult &result,
                    ErrorCode expected_code) {
  auto accepted = ValidatePreparedAcBatchResult(batch, result);
  ASSERT_FALSE(accepted.ok());
  EXPECT_EQ(accepted.error().code, expected_code) << accepted.error().message;
}

void ExpectResultsEqual(const PreparedAcBatchResult &left,
                        const PreparedAcBatchResult &right) {
  ASSERT_EQ(left.members.size(), right.members.size());
  for (std::size_t member = 0; member < left.members.size(); ++member) {
    EXPECT_EQ(left.members[member].identity, right.members[member].identity);
    EXPECT_EQ(left.members[member].solution, right.members[member].solution);
  }
}

[[nodiscard]] PreparedAcBatchResult
SolveAndCertify(const PreparedAcBatch &batch,
                CudaPreparedAcBatchBackend *backend) {
  auto solved = backend->Execute(batch);
  if (!solved.ok()) {
    ADD_FAILURE() << solved.error().message;
    return {};
  }
  auto accepted = ValidatePreparedAcBatchResult(batch, solved.value());
  if (!accepted.ok()) {
    ADD_FAILURE() << accepted.error().message;
    return {};
  }
  return solved.TakeValue();
}

TEST(Gpu02PreparedAcCudaTest,
     SolvesExactComplexFp64CaseAndReusesOneImmutablePreparation) {
  const PreparedAcBatch batch = PrepareAnalyticBatch();
  auto valid = ValidatePreparedAcBatch(batch);
  ASSERT_TRUE(valid.ok()) << valid.error().message;

  CudaPreparedAcBatchBackend backend;
  const PreparedAcBatchResult first = SolveAndCertify(batch, &backend);
  const PreparedAcBatchResult second = SolveAndCertify(batch, &backend);

  constexpr std::complex<double> kExpectedA{2.0, 3.0};
  constexpr std::complex<double> kExpectedB{2.0, -1.0};
  for (const PreparedAcBatchResult *result : {&first, &second}) {
    ASSERT_EQ(result->members.size(), batch.members.size());
    for (const PreparedAcResultMember &member : result->members) {
      ASSERT_EQ(member.solution.size(), 2U);
      EXPECT_NEAR(member.solution[0].real(), kExpectedA.real(), 1.0e-13);
      EXPECT_NEAR(member.solution[0].imag(), kExpectedA.imag(), 1.0e-13);
      EXPECT_NEAR(member.solution[1].real(), kExpectedB.real(), 1.0e-13);
      EXPECT_NEAR(member.solution[1].imag(), kExpectedB.imag(), 1.0e-13);
    }
  }

  const CudaPreparedAcStatistics &statistics = backend.statistics();
  EXPECT_EQ(statistics.preparations, 1U);
  EXPECT_EQ(statistics.structure_uploads, 1U);
  EXPECT_EQ(statistics.values_rhs_uploads, 1U);
  EXPECT_EQ(statistics.same_structure_refreshes, 0U);
  EXPECT_EQ(statistics.analyses, 1U);
  EXPECT_EQ(statistics.executions, 2U);
  EXPECT_EQ(statistics.factorizations, 2U * batch.members.size());
  EXPECT_EQ(statistics.solves, 2U * batch.members.size());
  EXPECT_EQ(statistics.factorization_calls, 2U);
  EXPECT_EQ(statistics.solve_calls, 2U);
  EXPECT_EQ(statistics.uniform_batch_size, batch.members.size());
  EXPECT_GT(statistics.controlled_device_bytes, 0U);
  EXPECT_GE(statistics.peak_batch_device_bytes,
            statistics.controlled_device_bytes);
  auto released = backend.Release();
  ASSERT_TRUE(released.ok()) << released.error().message;
  EXPECT_EQ(backend.statistics().releases, 1U);
  EXPECT_EQ(backend.statistics().controlled_device_bytes, 0U);
  EXPECT_EQ(backend.statistics().cudss_outstanding_device_bytes, 0U);
  EXPECT_EQ(backend.statistics().last_release_outstanding_device_bytes, 0U);
  EXPECT_EQ(backend.statistics().uniform_batch_size, 0U);
}

TEST(Gpu02PreparedAcCudaTest,
     RefreshesValuesRhsAndAssociationsWithoutReplacingStructureAnalysis) {
  const PreparedAcBatch batch_a = PrepareAnalyticBatch();
  const std::complex<double> expected_b_a{1.5, -2.0};
  const std::complex<double> expected_b_b{-0.5, 0.25};
  const PreparedAcBatch batch_b = PrepareAnalyticBatch(
      "diagonal-replacement", 3.0, 5.0, expected_b_a, expected_b_b);
  CudaPreparedAcBatchBackend backend;

  const PreparedAcBatchResult first_a = SolveAndCertify(batch_a, &backend);
  const PreparedAcBatchResult only_b = SolveAndCertify(batch_b, &backend);
  const PreparedAcBatchResult second_a = SolveAndCertify(batch_a, &backend);

  ASSERT_EQ(only_b.members.size(), batch_b.members.size());
  for (const PreparedAcResultMember &member : only_b.members) {
    ASSERT_EQ(member.solution.size(), 2U);
    EXPECT_NEAR(member.solution[0].real(), expected_b_a.real(), 1.0e-13);
    EXPECT_NEAR(member.solution[0].imag(), expected_b_a.imag(), 1.0e-13);
    EXPECT_NEAR(member.solution[1].real(), expected_b_b.real(), 1.0e-13);
    EXPECT_NEAR(member.solution[1].imag(), expected_b_b.imag(), 1.0e-13);
  }
  ExpectResultsEqual(first_a, second_a);
  EXPECT_EQ(backend.statistics().preparations, 1U);
  EXPECT_EQ(backend.statistics().structure_uploads, 1U);
  EXPECT_EQ(backend.statistics().values_rhs_uploads, 3U);
  EXPECT_EQ(backend.statistics().same_structure_refreshes, 2U);
  EXPECT_EQ(backend.statistics().analyses, 1U);
  EXPECT_EQ(backend.statistics().executions, 3U);
  EXPECT_EQ(backend.statistics().factorization_calls, 3U);
  EXPECT_EQ(backend.statistics().solve_calls, 3U);
  auto released = backend.Release();
  ASSERT_TRUE(released.ok()) << released.error().message;
  EXPECT_EQ(backend.statistics().last_release_outstanding_device_bytes, 0U);
}

TEST(Gpu02PreparedAcCudaTest,
     BackendNeutralFailClosedWrapperAcceptsValidCudaWithoutFallback) {
  const PreparedAcBatch batch = PrepareAnalyticBatch();
  CudaPreparedAcBatchBackend backend;
  auto executed = ExecutePreparedAcBatch(batch, backend,
                                         PreparedAcFallbackPolicy::kFailClosed);
  ASSERT_TRUE(executed.ok()) << executed.error().message;
  EXPECT_FALSE(executed.value().used_cpu_fallback);
  auto accepted =
      ValidatePreparedAcBatchResult(batch, executed.value().batch_result);
  EXPECT_TRUE(accepted.ok()) << accepted.error().message;
  auto released = backend.Release();
  EXPECT_TRUE(released.ok()) << released.error().message;
}

TEST(Gpu02PreparedAcCudaTest,
     CpuAcceptanceRejectsEveryHostileCudaOutputAssociationAndValueClass) {
  const PreparedAcBatch batch = PrepareAnalyticBatch();
  CudaPreparedAcBatchBackend backend;
  const PreparedAcBatchResult valid = SolveAndCertify(batch, &backend);

  PreparedAcBatchResult non_finite = valid;
  non_finite.members[0].solution[0] = {std::numeric_limits<double>::quiet_NaN(),
                                       0.0};
  ExpectRejected(batch, non_finite, ErrorCode::kPreparedInvalidResult);

  PreparedAcBatchResult excessive_residual = valid;
  excessive_residual.members[0].solution[0] += std::complex<double>{1.0, 1.0};
  ExpectRejected(batch, excessive_residual, ErrorCode::kPreparedInvalidResult);

  PreparedAcBatchResult wrong_dimension = valid;
  wrong_dimension.members[0].solution.pop_back();
  ExpectRejected(batch, wrong_dimension, ErrorCode::kPreparedInvalidResult);

  PreparedAcBatchResult missing = valid;
  missing.members.pop_back();
  ExpectRejected(batch, missing, ErrorCode::kPreparedResultMissing);

  PreparedAcBatchResult duplicate = valid;
  duplicate.members.back() = duplicate.members.front();
  ExpectRejected(batch, duplicate, ErrorCode::kPreparedResultDuplicate);

  PreparedAcBatchResult reordered = valid;
  std::swap(reordered.members[0], reordered.members[1]);
  ExpectRejected(batch, reordered, ErrorCode::kPreparedResultReordered);

  PreparedAcBatchResult stale_envelope = valid;
  stale_envelope.batch_fingerprint = "v1-stale";
  ExpectRejected(batch, stale_envelope, ErrorCode::kPreparedBatchStale);

  PreparedAcBatchResult stale_member = valid;
  stale_member.members[0].identity.content_fingerprint = "v1-stale";
  ExpectRejected(batch, stale_member, ErrorCode::kPreparedBatchStale);

  PreparedAcBatchResult wrong_association = valid;
  wrong_association.members[0].identity.circuit_id = "other-circuit";
  ExpectRejected(batch, wrong_association,
                 ErrorCode::kPreparedResultAssociation);

  auto released = backend.Release();
  ASSERT_TRUE(released.ok()) << released.error().message;
  EXPECT_EQ(backend.statistics().last_release_outstanding_device_bytes, 0U);
}

TEST(Gpu02PreparedAcCudaTest,
     LiveAllocationCudaCudssAndDataInfoFailuresInvalidateThenReprepare) {
  const PreparedAcBatch batch = PrepareAnalyticBatch();
  for (const auto [fault, name] :
       std::vector<std::pair<CudaPreparedAcFault, std::string_view>>{
           {CudaPreparedAcFault::kAllocationFailure, "allocation"},
           {CudaPreparedAcFault::kCudaFailure, "CUDA"},
           {CudaPreparedAcFault::kCudssFailure, "cuDSS"},
           {CudaPreparedAcFault::kDataInfoFailure, "cuDSS data-info"},
       }) {
    SCOPED_TRACE(name);
    CudaPreparedAcBatchBackend backend(CudaPreparedAcOptions{.fault = fault});
    auto solved = backend.Execute(batch);
    ASSERT_FALSE(solved.ok());
    EXPECT_EQ(solved.error().code, ErrorCode::kPreparedBackendFailure)
        << solved.error().message;
    EXPECT_GT(backend.statistics().last_upload_ns, 0U);

    auto recovered = backend.Execute(batch);
    ASSERT_TRUE(recovered.ok()) << recovered.error().message;
    auto accepted = ValidatePreparedAcBatchResult(batch, recovered.value());
    ASSERT_TRUE(accepted.ok()) << accepted.error().message;
    const std::uint64_t expected_completed_preparations =
        fault == CudaPreparedAcFault::kAllocationFailure ? 1U : 2U;
    EXPECT_EQ(backend.statistics().structure_uploads,
              expected_completed_preparations);
    EXPECT_EQ(backend.statistics().analyses, expected_completed_preparations);
    auto released = backend.Release();
    ASSERT_TRUE(released.ok()) << released.error().message;
    EXPECT_EQ(backend.statistics().last_release_outstanding_device_bytes, 0U);
  }
}

TEST(Gpu02PreparedAcCudaTest,
     PartialCudaSuccessIsRejectedAndExplicitFullBatchFallbackRecovers) {
  const PreparedAcBatch batch = PrepareAnalyticBatch();
  CudaPreparedAcBatchBackend backend(
      CudaPreparedAcOptions{.fault = CudaPreparedAcFault::kPartialResult});
  auto partial = backend.Execute(batch);
  ASSERT_TRUE(partial.ok()) << partial.error().message;
  ExpectRejected(batch, partial.value(), ErrorCode::kPreparedResultMissing);

  auto fail_closed = ExecutePreparedAcBatch(
      batch, backend, PreparedAcFallbackPolicy::kFailClosed);
  ASSERT_FALSE(fail_closed.ok());
  EXPECT_EQ(fail_closed.error().code, ErrorCode::kPreparedResultMissing)
      << fail_closed.error().message;

  auto recovered = ExecutePreparedAcBatch(
      batch, backend, PreparedAcFallbackPolicy::kCpuKluFullBatch);
  ASSERT_TRUE(recovered.ok()) << recovered.error().message;
  EXPECT_TRUE(recovered.value().used_cpu_fallback);
  auto accepted =
      ValidatePreparedAcBatchResult(batch, recovered.value().batch_result);
  EXPECT_TRUE(accepted.ok()) << accepted.error().message;
  ASSERT_FALSE(partial.value().members.empty());
  ASSERT_FALSE(recovered.value().batch_result.members.empty());
  EXPECT_NE(partial.value().members.front().solution.front(),
            recovered.value().batch_result.members.front().solution.front());

  CpuKluPreparedAcBatchBackend cpu;
  auto authoritative = cpu.Execute(batch);
  ASSERT_TRUE(authoritative.ok()) << authoritative.error().message;
  ExpectResultsEqual(recovered.value().batch_result, authoritative.value());
  auto released = backend.Release();
  EXPECT_TRUE(released.ok()) << released.error().message;
}

} // namespace
} // namespace ohmnivore
