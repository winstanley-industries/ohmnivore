#include "cpp/tests/google_test.h"

#include <array>
#include <complex>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

#include <unistd.h>

#include "cpp/benchmarks/prepared_ac_session.h"
#include "ohmnivore/prepared_ac.h"
#include "ohmnivore/status.h"

namespace ohmnivore::benchmarks {
namespace {

struct SessionGolden {
  std::string_view case_id;
  std::string_view workload_class;
  std::string_view topology;
  std::size_t nodes;
  std::size_t branches;
  std::size_t dimension;
  std::size_t nonzeros;
  std::size_t batch;
  std::size_t corners;
};

inline constexpr std::array<SessionGolden, 4> kGoldens{{
    {.case_id = "tree_session_control",
     .workload_class = "session_control",
     .topology = "binary_tree",
     .nodes = 256,
     .branches = 1,
     .dimension = 257,
     .nonzeros = 768,
     .batch = 64,
     .corners = 4},
    {.case_id = "grid_33_session",
     .workload_class = "session_grid_medium",
     .topology = "square_grid",
     .nodes = 1089,
     .branches = 1,
     .dimension = 1090,
     .nonzeros = 5315,
     .batch = 512,
     .corners = 4},
    {.case_id = "grid_65_session",
     .workload_class = "session_grid_large",
     .topology = "square_grid",
     .nodes = 4225,
     .branches = 1,
     .dimension = 4226,
     .nonzeros = 20867,
     .batch = 256,
     .corners = 4},
    {.case_id = "ring_1024_session",
     .workload_class = "session_wide",
     .topology = "ring_multi",
     .nodes = 1024,
     .branches = 4,
     .dimension = 1028,
     .nonzeros = 3080,
     .batch = 2048,
     .corners = 4},
}};

[[nodiscard]] std::string CorpusPath() {
  EXPECT_EQ(testing::internal::GetArgvs().size(), 2U);
  return testing::internal::GetArgvs()[1];
}

[[nodiscard]] std::string ReadCorpus() {
  std::ifstream input(CorpusPath(), std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

class TemporaryCorpus {
public:
  explicit TemporaryCorpus(std::string_view bytes) {
    static std::size_t sequence = 0;
    path_ = std::filesystem::temp_directory_path() /
            ("ohmnivore_gpu02s_session_" + std::to_string(getpid()) + "_" +
             std::to_string(sequence++) + ".csv");
    std::ofstream output(path_, std::ios::binary);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }

  ~TemporaryCorpus() {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }

  [[nodiscard]] const std::filesystem::path &path() const { return path_; }

private:
  std::filesystem::path path_;
};

void ReplaceOnce(std::string *text, std::string_view old_value,
                 std::string_view new_value) {
  const std::size_t position = text->find(old_value);
  ASSERT_NE(position, std::string::npos) << old_value;
  text->replace(position, old_value.size(), new_value);
}

void ExpectMalformed(std::string bytes) {
  TemporaryCorpus corpus(bytes);
  auto loaded = LoadPreparedAcSessionCorpus(corpus.path().string());
  ASSERT_FALSE(loaded.ok());
  EXPECT_EQ(loaded.error().code, ErrorCode::kPreparedBatchMalformed);
}

TEST(Gpu02sSessionTest, PinsCorpusAndCompilerDerivedSameStructureCorners) {
  auto corpus = LoadPreparedAcSessionCorpus(CorpusPath());
  ASSERT_TRUE(corpus.ok()) << corpus.error().message;
  EXPECT_EQ(corpus.value().manifest_fingerprint,
            kPreparedAcSessionV1ManifestFingerprint);
  ASSERT_EQ(corpus.value().cases.size(), kGoldens.size());

  for (std::size_t index = 0; index < kGoldens.size(); ++index) {
    const PreparedAcSessionCase &item = corpus.value().cases[index];
    const SessionGolden &golden = kGoldens[index];
    EXPECT_EQ(item.case_id, golden.case_id);
    EXPECT_EQ(item.workload_class, golden.workload_class);
    EXPECT_EQ(item.topology, golden.topology);
    EXPECT_EQ(item.node_count, golden.nodes);
    EXPECT_EQ(item.branch_count, golden.branches);
    EXPECT_EQ(item.dimension, golden.dimension);
    EXPECT_EQ(item.union_nonzeros, golden.nonzeros);
    EXPECT_EQ(item.batch_size, golden.batch);
    EXPECT_EQ(item.corner_count, golden.corners);

    auto first = PrepareAcSessionCorner(item, 0);
    ASSERT_TRUE(first.ok()) << item.case_id << ": " << first.error().message;
    auto last = PrepareAcSessionCorner(item, item.corner_count - 1);
    ASSERT_TRUE(last.ok()) << item.case_id << ": " << last.error().message;
    EXPECT_EQ(first.value().structure.fingerprint,
              last.value().structure.fingerprint);
    EXPECT_NE(first.value().batch_fingerprint, last.value().batch_fingerprint);
    EXPECT_EQ(first.value().members.size(), item.batch_size);
    EXPECT_EQ(last.value().members.size(), item.batch_size);
    EXPECT_EQ(first.value().members.front().identity.corner_id, "corner-0");
    EXPECT_EQ(last.value().members.front().identity.corner_id,
              "corner-" + std::to_string(item.corner_count - 1));
    auto first_valid = ValidatePreparedAcBatch(first.value());
    auto last_valid = ValidatePreparedAcBatch(last.value());
    EXPECT_TRUE(first_valid.ok()) << first_valid.error().message;
    EXPECT_TRUE(last_valid.ok()) << last_valid.error().message;
  }
}

TEST(Gpu02sSessionTest, RejectsMalformedOrAlteredSessionManifest) {
  const std::string original = ReadCorpus();
  ExpectMalformed("");
  ExpectMalformed(original.substr(0, original.size() - 1));

  std::string changed = original;
  ReplaceOnce(&changed, "\n1,tree_session_control",
              "\r\n1,tree_session_control");
  ExpectMalformed(changed);
  changed = original;
  ReplaceOnce(&changed, "schema_version,case_id",
              "schema_version,extra,case_id");
  ExpectMalformed(changed);
  changed = original;
  ReplaceOnce(&changed, "1,tree_session_control", "2,tree_session_control");
  ExpectMalformed(changed);
  changed = original;
  ReplaceOnce(&changed, "session_control,binary_tree",
              "session_control,unknown");
  ExpectMalformed(changed);
  changed = original;
  ReplaceOnce(&changed, ",256,1,257,768,", ",256,1,258,768,");
  ExpectMalformed(changed);
  changed = original;
  ReplaceOnce(&changed, ",257,768,64,", ",257,769,64,");
  ExpectMalformed(changed);
  changed = original;
  ReplaceOnce(&changed, ",1e-3,1e-9,", ",0,1e-9,");
  ExpectMalformed(changed);
  changed = original;
  ReplaceOnce(&changed, ",1e-12,4\n", ",1e-12,5\n");
  ExpectMalformed(changed);
}

TEST(Gpu02sSessionTest, PublicCpuAuthorityRetainsExactBatchCacheSemantics) {
  auto corpus = LoadPreparedAcSessionCorpus(CorpusPath());
  ASSERT_TRUE(corpus.ok()) << corpus.error().message;
  const PreparedAcSessionCase &item = corpus.value().cases.front();
  auto first = PrepareAcSessionCorner(item, 0);
  auto second = PrepareAcSessionCorner(item, 1);
  ASSERT_TRUE(first.ok()) << first.error().message;
  ASSERT_TRUE(second.ok()) << second.error().message;

  CpuKluPreparedAcBatchBackend backend;
  auto first_result = backend.Execute(first.value());
  ASSERT_TRUE(first_result.ok()) << first_result.error().message;
  auto repeated_result = backend.Execute(first.value());
  ASSERT_TRUE(repeated_result.ok()) << repeated_result.error().message;
  EXPECT_EQ(backend.statistics().symbolic_analyses, 1U);
  EXPECT_EQ(backend.statistics().solves, 2U * item.batch_size);

  auto second_result = backend.Execute(second.value());
  ASSERT_TRUE(second_result.ok()) << second_result.error().message;
  auto first_valid =
      ValidatePreparedAcBatchResult(first.value(), first_result.value());
  auto second_valid =
      ValidatePreparedAcBatchResult(second.value(), second_result.value());
  ASSERT_TRUE(first_valid.ok()) << first_valid.error().message;
  ASSERT_TRUE(second_valid.ok()) << second_valid.error().message;
  EXPECT_EQ(backend.statistics().symbolic_analyses, 1U);
  EXPECT_EQ(backend.statistics().solves, item.batch_size);
}

TEST(Gpu02sSessionTest, EvidenceValidatorIsTestOnlyAndFailsClosed) {
  auto corpus = LoadPreparedAcSessionCorpus(CorpusPath());
  ASSERT_TRUE(corpus.ok()) << corpus.error().message;
  auto batch = PrepareAcSessionCorner(corpus.value().cases.front(), 0);
  ASSERT_TRUE(batch.ok()) << batch.error().message;

  CpuKluPreparedAcBatchBackend backend;
  auto solved = backend.Execute(batch.value());
  ASSERT_TRUE(solved.ok()) << solved.error().message;
  auto runtime =
      ValidatePreparedAcBatchResultForEvidence(batch.value(), solved.value());
  ASSERT_TRUE(runtime.ok()) << runtime.error().message;

  const auto expect_rejected = [&](PreparedAcBatchResult hostile) {
    auto rejected =
        ValidatePreparedAcBatchResultForEvidence(batch.value(), hostile);
    ASSERT_FALSE(rejected.ok());
  };

  PreparedAcBatchResult hostile = solved.value();
  hostile.batch_fingerprint = "v1-stale";
  expect_rejected(std::move(hostile));

  hostile = solved.value();
  std::swap(hostile.members[0], hostile.members[1]);
  expect_rejected(std::move(hostile));

  hostile = solved.value();
  hostile.members[0].solution.pop_back();
  expect_rejected(std::move(hostile));

  hostile = solved.value();
  hostile.members[0].solution[0] = {std::numeric_limits<double>::quiet_NaN(),
                                    0.0};
  expect_rejected(std::move(hostile));

  hostile = solved.value();
  hostile.members[0].solution[0] += std::complex<double>{1.0, 0.0};
  expect_rejected(std::move(hostile));
}

TEST(Gpu02sSessionTest, RejectsOutOfRangeCornerWithoutCompilation) {
  auto corpus = LoadPreparedAcSessionCorpus(CorpusPath());
  ASSERT_TRUE(corpus.ok()) << corpus.error().message;
  auto circuit = BuildPreparedAcSessionCircuit(
      corpus.value().cases.front(), corpus.value().cases.front().corner_count);
  ASSERT_FALSE(circuit.ok());
  EXPECT_EQ(circuit.error().code, ErrorCode::kPreparedBatchMalformed);
}

} // namespace
} // namespace ohmnivore::benchmarks
