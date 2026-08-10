#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <string>
#include <string_view>

#include "cpp/benchmarks/prepared_ac_replay.h"
#include "cuda/prepared_ac_cuda.h"
#include "ohmnivore/prepared_ac.h"

namespace ohmnivore {
namespace {

inline constexpr std::array<std::string_view, 4> kFrozenCaseOrder{
    "ladder_s_65",
    "tree_m_257",
    "grid_l_1025",
    "ring_multi_m_260",
};

[[nodiscard]] std::string CorpusPath() {
  EXPECT_EQ(testing::internal::GetArgvs().size(), 2U);
  return testing::internal::GetArgvs()[1];
}

TEST(Gpu02PreparedAcCudaReplayTest,
     FullFrozenReplayV1BatchDifferentiatesAgainstCpuKluAndCertifies) {
  auto corpus = benchmarks::LoadPreparedAcReplayCorpus(CorpusPath());
  ASSERT_TRUE(corpus.ok()) << corpus.error().message;
  EXPECT_EQ(corpus.value().schema_version, 1U);
  EXPECT_EQ(corpus.value().manifest_fingerprint,
            benchmarks::kPreparedAcReplayV1ManifestFingerprint);
  ASSERT_EQ(corpus.value().cases.size(), kFrozenCaseOrder.size());

  for (std::size_t case_index = 0; case_index < corpus.value().cases.size();
       ++case_index) {
    const benchmarks::PreparedAcReplayCase &replay_case =
        corpus.value().cases[case_index];
    SCOPED_TRACE(replay_case.case_id);
    EXPECT_EQ(replay_case.case_id, kFrozenCaseOrder[case_index]);

    auto prepared = benchmarks::PrepareAcReplayCase(replay_case);
    ASSERT_TRUE(prepared.ok()) << prepared.error().message;
    const PreparedAcBatch &batch = prepared.value();
    ASSERT_EQ(batch.members.size(), replay_case.batch_size);

    CpuKluPreparedAcBatchBackend cpu;
    auto authoritative = cpu.Execute(batch);
    ASSERT_TRUE(authoritative.ok()) << authoritative.error().message;
    auto cpu_accepted =
        ValidatePreparedAcBatchResult(batch, authoritative.value());
    ASSERT_TRUE(cpu_accepted.ok()) << cpu_accepted.error().message;

    CudaPreparedAcBatchBackend gpu;
    for (std::size_t execution = 0; execution < 2; ++execution) {
      SCOPED_TRACE(execution);
      auto candidate = gpu.Execute(batch);
      ASSERT_TRUE(candidate.ok()) << candidate.error().message;

      // This unchanged acceptance boundary independently reconstructs every
      // authoritative matrix/RHS, checks its residual, performs fresh KLU
      // certification, and applies the componentwise differential bound.
      auto gpu_accepted =
          ValidatePreparedAcBatchResult(batch, candidate.value());
      ASSERT_TRUE(gpu_accepted.ok()) << gpu_accepted.error().message;
      ASSERT_EQ(candidate.value().members.size(),
                authoritative.value().members.size());

      for (std::size_t member = 0; member < batch.members.size(); ++member) {
        EXPECT_EQ(candidate.value().members[member].identity,
                  authoritative.value().members[member].identity);
        ASSERT_EQ(candidate.value().members[member].solution.size(),
                  authoritative.value().members[member].solution.size());
        for (std::size_t component = 0;
             component < candidate.value().members[member].solution.size();
             ++component) {
          const std::complex<double> gpu_value =
              candidate.value().members[member].solution[component];
          const std::complex<double> cpu_value =
              authoritative.value().members[member].solution[component];
          const double scale =
              std::max(std::abs(gpu_value), std::abs(cpu_value));
          EXPECT_LE(std::abs(gpu_value - cpu_value), 1.0e-12 + 1.0e-9 * scale);
        }
      }
    }

    const CudaPreparedAcStatistics &statistics = gpu.statistics();
    EXPECT_EQ(statistics.preparations, 1U);
    EXPECT_EQ(statistics.structure_uploads, 1U);
    EXPECT_EQ(statistics.analyses, 1U);
    EXPECT_EQ(statistics.executions, 2U);
    EXPECT_EQ(statistics.factorizations, 2U * batch.members.size());
    EXPECT_EQ(statistics.solves, 2U * batch.members.size());
    EXPECT_EQ(statistics.factorization_calls, 2U);
    EXPECT_EQ(statistics.solve_calls, 2U);
    EXPECT_EQ(statistics.uniform_batch_size, batch.members.size());
    auto released = gpu.Release();
    ASSERT_TRUE(released.ok()) << released.error().message;
    EXPECT_EQ(gpu.statistics().last_release_outstanding_device_bytes, 0U);
  }
}

} // namespace
} // namespace ohmnivore
