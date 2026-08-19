#ifndef OHMNIVORE_BENCHMARKS_PREPARED_AC_SESSION_H_
#define OHMNIVORE_BENCHMARKS_PREPARED_AC_SESSION_H_

#include <cstddef>
#include <string>
#include <vector>

#include "ohmnivore/ir.h"
#include "ohmnivore/prepared_ac.h"
#include "ohmnivore/status.h"

namespace ohmnivore::benchmarks {

inline constexpr const char *kPreparedAcSessionV1ManifestFingerprint =
    "v1-de52b10506655ec327cab7e010fbe106";

struct PreparedAcSessionCase {
  std::string case_id;
  std::string workload_class;
  std::string topology;
  std::size_t node_count = 0;
  std::size_t branch_count = 0;
  std::size_t dimension = 0;
  std::size_t union_nonzeros = 0;
  std::size_t sweep_points = 0;
  std::size_t batch_size = 0;
  double start_frequency_hz = 0.0;
  double stop_frequency_hz = 0.0;
  double g_series = 0.0;
  double g_shunt = 0.0;
  double c_series = 0.0;
  double c_shunt = 0.0;
  std::size_t corner_count = 0;
};

struct PreparedAcSessionCorpus {
  std::size_t schema_version = 0;
  std::string manifest_fingerprint;
  std::vector<PreparedAcSessionCase> cases;
};

[[nodiscard]] Result<PreparedAcSessionCorpus>
LoadPreparedAcSessionCorpus(const std::string &path);

// Constructs a deterministic linear RLC Circuit IR for one declared corner.
// The public compiler and prepared-AC paths remain the sole MNA authorities.
[[nodiscard]] Result<Circuit>
BuildPreparedAcSessionCircuit(const PreparedAcSessionCase &session_case,
                              std::size_t corner_ordinal);

[[nodiscard]] Result<PreparedAcBatch>
PrepareAcSessionCorner(const PreparedAcSessionCase &session_case,
                       std::size_t corner_ordinal);

// Evidence-only candidate-runtime boundary. It performs the prepared envelope,
// association, dimension, finiteness, residual, and backward-error checks but
// deliberately omits fresh KLU differential certification. This test-only
// helper is not an acceptance boundary; every result must still pass
// ValidatePreparedAcBatchResult before an evidence stream can complete.
[[nodiscard]] Result<bool>
ValidatePreparedAcBatchResultForEvidence(const PreparedAcBatch &batch,
                                         const PreparedAcBatchResult &result);

} // namespace ohmnivore::benchmarks

#endif // OHMNIVORE_BENCHMARKS_PREPARED_AC_SESSION_H_
