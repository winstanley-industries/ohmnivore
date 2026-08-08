#include "cpp/tests/google_test.h"

#include <array>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>

#include <unistd.h>

#include "cpp/benchmarks/prepared_ac_replay.h"
#include "ohmnivore/prepared_ac.h"
#include "ohmnivore/status.h"

namespace ohmnivore::benchmarks {
namespace {

struct ReplayGolden {
  std::string_view case_id;
  std::string_view workload_class;
  std::string_view topology;
  std::size_t nodes;
  std::size_t branches;
  std::size_t dimension;
  std::size_t nonzeros;
  AcSweepType sweep_type;
  std::size_t sweep_points;
  std::size_t batch;
  std::size_t reuses;
  std::string_view structure;
  std::string_view batch_fingerprint;
  std::string_view aggregate;
  std::string_view first_member;
  std::string_view last_member;
};

inline constexpr std::array<ReplayGolden, 4> kGoldens{{
    {.case_id = "ladder_s_65",
     .workload_class = "small_control",
     .topology = "ladder",
     .nodes = 64,
     .branches = 1,
     .dimension = 65,
     .nonzeros = 192,
     .sweep_type = AcSweepType::kLin,
     .sweep_points = 16,
     .batch = 16,
     .reuses = 8,
     .structure = "v1-80c3f428ebbdf94c126f9ac4410ecd7d",
     .batch_fingerprint = "v1-dd7db9180c25e40edcdce264e3951eb7",
     .aggregate = "v1-9284546700ab44ee262303e5ab1b717b",
     .first_member = "v1-29c01ada890a3521c9f9ca378b2727d6",
     .last_member = "v1-68c7ea3b486f220582cc79eca263e4a2"},
    {.case_id = "tree_m_257",
     .workload_class = "medium",
     .topology = "binary_tree",
     .nodes = 256,
     .branches = 1,
     .dimension = 257,
     .nonzeros = 768,
     .sweep_type = AcSweepType::kDec,
     .sweep_points = 10,
     .batch = 61,
     .reuses = 4,
     .structure = "v1-b7dbae64c3b47c80919d1f9a135cb449",
     .batch_fingerprint = "v1-de9b9c71249acdac028c77a1eb368537",
     .aggregate = "v1-7bdcbd3e84c9c977e1ce93b12af500be",
     .first_member = "v1-0606cef24f7444db020c3871ca7c0580",
     .last_member = "v1-6f40cb62b91944b36f16e3a5e5abb204"},
    {.case_id = "grid_l_1025",
     .workload_class = "large",
     .topology = "grid_32x32",
     .nodes = 1024,
     .branches = 1,
     .dimension = 1025,
     .nonzeros = 4994,
     .sweep_type = AcSweepType::kDec,
     .sweep_points = 20,
     .batch = 121,
     .reuses = 2,
     .structure = "v1-ff2842b6baeef40e02cf7832a34af3fb",
     .batch_fingerprint = "v1-705c5fd1588f63b1883c87c1413a929c",
     .aggregate = "v1-14f25da677bb84abb249792780edc16c",
     .first_member = "v1-2903515c876f0b8367b54aca8c1c9dd8",
     .last_member = "v1-21913e2903ad47a7acc38ec936227bf8"},
    {.case_id = "ring_multi_m_260",
     .workload_class = "medium_wide",
     .topology = "ring_multi",
     .nodes = 256,
     .branches = 4,
     .dimension = 260,
     .nonzeros = 776,
     .sweep_type = AcSweepType::kDec,
     .sweep_points = 43,
     .batch = 517,
     .reuses = 2,
     .structure = "v1-fc4da1872698073e575a302aeb1c9d7f",
     .batch_fingerprint = "v1-4c1d93c6a556a748ef7991b941bb2bbb",
     .aggregate = "v1-59200e21062e4d16d78995cec8ba6ec7",
     .first_member = "v1-add827d4465570d05c57c8c2344f6c6f",
     .last_member = "v1-8e8223354626c97c7f336623c10fe6ab"},
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
            ("ohmnivore_gpu01_replay_" + std::to_string(getpid()) + "_" +
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
  auto loaded = LoadPreparedAcReplayCorpus(corpus.path().string());
  ASSERT_FALSE(loaded.ok());
  EXPECT_EQ(loaded.error().code, ErrorCode::kPreparedBatchMalformed);
}

TEST(Gpu01ReplayTest, LoadsAndPinsTheExactVersionOneCorpusAndBuilder) {
  auto corpus = LoadPreparedAcReplayCorpus(CorpusPath());
  ASSERT_TRUE(corpus.ok()) << corpus.error().message;
  EXPECT_EQ(corpus.value().schema_version, 1U);
  EXPECT_EQ(corpus.value().manifest_fingerprint,
            kPreparedAcReplayV1ManifestFingerprint);
  ASSERT_EQ(corpus.value().cases.size(), kGoldens.size());

  for (std::size_t index = 0; index < kGoldens.size(); ++index) {
    const PreparedAcReplayCase &item = corpus.value().cases[index];
    const ReplayGolden &golden = kGoldens[index];
    EXPECT_EQ(item.case_id, golden.case_id);
    EXPECT_EQ(item.workload_class, golden.workload_class);
    EXPECT_EQ(item.topology, golden.topology);
    EXPECT_EQ(item.node_count, golden.nodes);
    EXPECT_EQ(item.branch_count, golden.branches);
    EXPECT_EQ(item.dimension, golden.dimension);
    EXPECT_EQ(item.union_nonzeros, golden.nonzeros);
    EXPECT_EQ(item.sweep_type, golden.sweep_type);
    EXPECT_EQ(item.sweep_points, golden.sweep_points);
    EXPECT_EQ(item.batch_size, golden.batch);
    EXPECT_EQ(item.reuse_count, golden.reuses);

    auto prepared = PrepareAcReplayCase(item);
    ASSERT_TRUE(prepared.ok())
        << item.case_id << ": " << prepared.error().message;
    EXPECT_EQ(prepared.value().structure.fingerprint, golden.structure);
    EXPECT_EQ(prepared.value().batch_fingerprint, golden.batch_fingerprint);
    EXPECT_EQ(FingerprintPreparedAcReplayMembers(prepared.value()),
              golden.aggregate);
    EXPECT_EQ(prepared.value().members.front().identity.content_fingerprint,
              golden.first_member);
    EXPECT_EQ(prepared.value().members.back().identity.content_fingerprint,
              golden.last_member);
    EXPECT_EQ(prepared.value().members.front().identity.contract_version, 1U);
    EXPECT_EQ(prepared.value().members.front().identity.replay_id,
              "prepared-ac-replay-v1/" + item.case_id);
    EXPECT_DOUBLE_EQ(prepared.value().members.front().identity.frequency_hz,
                     item.start_frequency_hz);
    EXPECT_DOUBLE_EQ(prepared.value().members.back().identity.frequency_hz,
                     item.stop_frequency_hz);
    auto valid = ValidatePreparedAcBatch(prepared.value());
    EXPECT_TRUE(valid.ok()) << valid.error().message;
  }
}

TEST(Gpu01ReplayTest, RejectsMalformedSerializationsAndAlteredV1Meaning) {
  const std::string original = ReadCorpus();
  ExpectMalformed("");
  ExpectMalformed(original.substr(0, original.size() - 1));

  std::string changed = original;
  ReplaceOnce(&changed, "\n1,ladder_s_65", "\r\n1,ladder_s_65");
  ExpectMalformed(changed);
  changed = original;
  changed.insert(changed.find('\n') + 1, "\n");
  ExpectMalformed(changed);
  changed = original;
  ReplaceOnce(&changed, "schema_version,case_id",
              "schema_version,extra,case_id");
  ExpectMalformed(changed);
  changed = original;
  ReplaceOnce(&changed, "1,ladder_s_65", "2,ladder_s_65");
  ExpectMalformed(changed);
  changed = original;
  ReplaceOnce(&changed, "small_control,ladder", "unknown,ladder");
  ExpectMalformed(changed);
  changed = original;
  ReplaceOnce(&changed, "small_control,ladder", "small_control,unknown");
  ExpectMalformed(changed);
  changed = original;
  ReplaceOnce(&changed, ",LIN,16,16,", ",OCT,16,16,");
  ExpectMalformed(changed);
  changed = original;
  ReplaceOnce(&changed, ",64,1,65,", ",064,1,65,");
  ExpectMalformed(changed);
  changed = original;
  ReplaceOnce(&changed, ",64,1,65,192,", ",64,1,66,192,");
  ExpectMalformed(changed);
  changed = original;
  ReplaceOnce(&changed, ",65,192,LIN", ",65,193,LIN");
  ExpectMalformed(changed);
  changed = original;
  ReplaceOnce(&changed, ",DEC,10,61,", ",DEC,10,60,");
  ExpectMalformed(changed);
  changed = original;
  ReplaceOnce(&changed, ",10,10000,", ",10000,10,");
  ExpectMalformed(changed);
  changed = original;
  ReplaceOnce(&changed, ",1e-3,1e-12,", ",0,1e-12,");
  ExpectMalformed(changed);
  changed = original;
  ReplaceOnce(&changed, ",1e-12,8\n", ",1e-12,9\n");
  ExpectMalformed(changed); // valid bounded data, but not the pinned v1 bytes.
  changed = original;
  const std::size_t first_row_end = changed.find('\n', changed.find('\n') + 1);
  const std::string first_row = changed.substr(
      changed.find('\n') + 1, first_row_end - changed.find('\n') - 1);
  changed.insert(first_row_end + 1, first_row + "\n");
  ExpectMalformed(changed);
  changed = original;
  changed.insert(changed.find('\n') + 1, "\x01");
  ExpectMalformed(changed);
}

} // namespace
} // namespace ohmnivore::benchmarks
