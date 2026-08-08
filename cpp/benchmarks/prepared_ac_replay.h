#ifndef OHMNIVORE_BENCHMARKS_PREPARED_AC_REPLAY_H_
#define OHMNIVORE_BENCHMARKS_PREPARED_AC_REPLAY_H_

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "ohmnivore/compiler.h"
#include "ohmnivore/prepared_ac.h"
#include "ohmnivore/status.h"

namespace ohmnivore::benchmarks {

inline constexpr std::string_view kPreparedAcReplayV1ManifestFingerprint =
    "v1-c87168bf25d75aa720fa8916a340ee5a";

struct PreparedAcReplayCase {
  std::string case_id;
  std::string workload_class;
  std::string topology;
  std::size_t node_count = 0;
  std::size_t branch_count = 0;
  std::size_t dimension = 0;
  std::size_t union_nonzeros = 0;
  AcSweepType sweep_type = AcSweepType::kLin;
  std::size_t sweep_points = 0;
  std::size_t batch_size = 0;
  double start_frequency_hz = 0.0;
  double stop_frequency_hz = 0.0;
  double g_series = 0.0;
  double g_shunt = 0.0;
  double c_series = 0.0;
  double c_shunt = 0.0;
  std::size_t reuse_count = 0;
};

struct PreparedAcReplayCorpus {
  std::size_t schema_version = 0;
  std::string manifest_fingerprint;
  std::vector<PreparedAcReplayCase> cases;
};

[[nodiscard]] Result<PreparedAcReplayCorpus>
LoadPreparedAcReplayCorpus(const std::string &path);

[[nodiscard]] Result<MnaSystem>
BuildPreparedAcReplaySystem(const PreparedAcReplayCase &replay_case);

[[nodiscard]] Result<PreparedAcBatch>
PrepareAcReplayCase(const PreparedAcReplayCase &replay_case);

[[nodiscard]] std::string
FingerprintPreparedAcReplayMembers(const PreparedAcBatch &batch);

[[nodiscard]] std::string_view
PreparedAcReplaySweepName(AcSweepType sweep_type);

} // namespace ohmnivore::benchmarks

#endif // OHMNIVORE_BENCHMARKS_PREPARED_AC_REPLAY_H_
