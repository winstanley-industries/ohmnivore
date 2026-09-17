#include <gtest/gtest.h>

#include <cstddef>
#include <string>

#include "cpp/benchmarks/prepared_ac_session.h"
#include "cuda/prepared_ac_cuda.h"
#include "ohmnivore/prepared_ac.h"

namespace ohmnivore::benchmarks {
namespace {

[[nodiscard]] std::string CorpusPath() {
  EXPECT_EQ(testing::internal::GetArgvs().size(), 2U);
  return testing::internal::GetArgvs()[1];
}

TEST(Gpu02sCudaSessionTest,
     CertifiesChangedCornersWithOneStructureUploadAndAnalysis) {
  auto corpus = LoadPreparedAcSessionCorpus(CorpusPath());
  ASSERT_TRUE(corpus.ok()) << corpus.error().message;
  const PreparedAcSessionCase &item = corpus.value().cases.front();
  auto first = PrepareAcSessionCorner(item, 0);
  auto second = PrepareAcSessionCorner(item, 1);
  ASSERT_TRUE(first.ok()) << first.error().message;
  ASSERT_TRUE(second.ok()) << second.error().message;
  ASSERT_EQ(first.value().structure.fingerprint,
            second.value().structure.fingerprint);
  ASSERT_NE(first.value().batch_fingerprint, second.value().batch_fingerprint);

  CudaPreparedAcBatchBackend backend;
  auto first_result = backend.Execute(first.value());
  ASSERT_TRUE(first_result.ok()) << first_result.error().message;
  auto first_runtime = ValidatePreparedAcBatchResultForEvidence(
      first.value(), first_result.value());
  ASSERT_TRUE(first_runtime.ok()) << first_runtime.error().message;
  auto first_certified =
      ValidatePreparedAcBatchResult(first.value(), first_result.value());
  ASSERT_TRUE(first_certified.ok()) << first_certified.error().message;

  auto second_result = backend.Execute(second.value());
  ASSERT_TRUE(second_result.ok()) << second_result.error().message;
  auto second_runtime = ValidatePreparedAcBatchResultForEvidence(
      second.value(), second_result.value());
  ASSERT_TRUE(second_runtime.ok()) << second_runtime.error().message;
  auto second_certified =
      ValidatePreparedAcBatchResult(second.value(), second_result.value());
  ASSERT_TRUE(second_certified.ok()) << second_certified.error().message;

  const CudaPreparedAcStatistics &statistics = backend.statistics();
  EXPECT_EQ(statistics.preparations, 1U);
  EXPECT_EQ(statistics.structure_uploads, 1U);
  EXPECT_EQ(statistics.values_rhs_uploads, 2U);
  EXPECT_EQ(statistics.same_structure_refreshes, 1U);
  EXPECT_EQ(statistics.analyses, 1U);
  EXPECT_EQ(statistics.executions, 2U);
  EXPECT_EQ(statistics.factorization_calls, 2U);
  EXPECT_EQ(statistics.solve_calls, 2U);
  EXPECT_EQ(statistics.factorizations, 2U * item.batch_size);
  EXPECT_EQ(statistics.solves, 2U * item.batch_size);
  EXPECT_LE(statistics.peak_batch_device_bytes, 2ULL * 1024 * 1024 * 1024);

  auto released = backend.Release();
  ASSERT_TRUE(released.ok()) << released.error().message;
  EXPECT_EQ(backend.statistics().last_release_outstanding_device_bytes, 0U);
}

} // namespace
} // namespace ohmnivore::benchmarks
