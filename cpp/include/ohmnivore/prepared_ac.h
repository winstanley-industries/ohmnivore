#ifndef OHMNIVORE_PREPARED_AC_H_
#define OHMNIVORE_PREPARED_AC_H_

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "ohmnivore/compiler.h"
#include "ohmnivore/ir.h"
#include "ohmnivore/solver.h"
#include "ohmnivore/sparse.h"
#include "ohmnivore/status.h"

namespace ohmnivore {

inline constexpr std::uint32_t kPreparedAcContractVersion = 1;

struct PreparedAcStructure {
  std::size_t dimension = 0;
  std::vector<std::size_t> row_offsets;
  std::vector<std::size_t> column_indices;
  std::string fingerprint;
};

struct PreparedAcMemberIdentity {
  std::uint32_t contract_version = kPreparedAcContractVersion;
  std::string replay_id;
  std::string circuit_id;
  std::string corner_id;
  std::size_t ordinal = 0;
  double frequency_hz = 0.0;
  std::string structure_fingerprint;
  std::string content_fingerprint;

  bool operator==(const PreparedAcMemberIdentity &) const = default;
};

struct PreparedAcMember {
  PreparedAcMemberIdentity identity;
  std::vector<std::complex<double>> matrix_values;
  std::vector<std::complex<double>> rhs;
};

struct PreparedAcBatch {
  std::uint32_t contract_version = kPreparedAcContractVersion;
  std::string replay_id;
  PreparedAcStructure structure;
  std::vector<PreparedAcMember> members;
  std::string batch_fingerprint;
};

struct PreparedAcResultMember {
  PreparedAcMemberIdentity identity;
  std::vector<std::complex<double>> solution;
};

struct PreparedAcBatchResult {
  std::uint32_t contract_version = kPreparedAcContractVersion;
  std::string replay_id;
  std::string structure_fingerprint;
  std::string batch_fingerprint;
  std::vector<PreparedAcResultMember> members;
};

class PreparedAcBatchBackend {
public:
  virtual ~PreparedAcBatchBackend() = default;

  [[nodiscard]] virtual Result<PreparedAcBatchResult>
  Execute(const PreparedAcBatch &batch) = 0;
};

// The only GPU-01 implementation. It is deterministic, serial, FP64, and
// retains a KLU symbolic analysis while the exact prepared batch is reused.
class CpuKluPreparedAcBatchBackend final : public PreparedAcBatchBackend {
public:
  CpuKluPreparedAcBatchBackend();
  ~CpuKluPreparedAcBatchBackend() override;
  CpuKluPreparedAcBatchBackend(CpuKluPreparedAcBatchBackend &&) noexcept;
  CpuKluPreparedAcBatchBackend &
  operator=(CpuKluPreparedAcBatchBackend &&) noexcept;
  CpuKluPreparedAcBatchBackend(const CpuKluPreparedAcBatchBackend &) = delete;
  CpuKluPreparedAcBatchBackend &
  operator=(const CpuKluPreparedAcBatchBackend &) = delete;

  [[nodiscard]] Result<PreparedAcBatchResult>
  Execute(const PreparedAcBatch &batch) override;
  [[nodiscard]] const SparseSolverStatistics &statistics() const;

private:
  std::string cached_structure_fingerprint_;
  std::unique_ptr<SparseComplexFactorization> factorization_;
  SparseSolverStatistics statistics_;
};

enum class PreparedAcFallbackPolicy {
  kFailClosed,
  kCpuKluFullBatch,
};

struct PreparedAcExecutionResult {
  PreparedAcBatchResult batch_result;
  bool used_cpu_fallback = false;
};

// Identifiers admit only ASCII letters, digits, underscore, hyphen, period,
// colon, and slash. The default corner identity is deliberately observable.
[[nodiscard]] Result<PreparedAcBatch>
PrepareLinearAcBatch(const MnaSystem &system, const AcAnalysis &analysis,
                     std::string replay_id, std::string circuit_id,
                     std::string corner_id = "nominal");

[[nodiscard]] Result<bool>
ValidatePreparedAcBatch(const PreparedAcBatch &batch);

// This is the only acceptance boundary for a backend-produced batch. It
// verifies identities before independently checking every numerical result.
[[nodiscard]] Result<bool>
ValidatePreparedAcBatchResult(const PreparedAcBatch &batch,
                              const PreparedAcBatchResult &result);

// Evidence-only candidate-runtime boundary. It performs the complete prepared
// envelope, association, dimension, finiteness, residual, and backward-error
// checks but deliberately omits fresh KLU differential certification. It is
// not an acceptance boundary and is never used by ExecutePreparedAcBatch,
// ordinary simulation, CSV output, or fallback. Evidence using it must still
// call ValidatePreparedAcBatchResult before the sample can be accepted.
[[nodiscard]] Result<bool>
ValidatePreparedAcBatchResultForEvidence(const PreparedAcBatch &batch,
                                         const PreparedAcBatchResult &result);

// Executes an explicitly supplied backend. Fallback, when requested, discards
// the complete preferred result and re-solves the complete batch with CPU KLU.
[[nodiscard]] Result<PreparedAcExecutionResult>
ExecutePreparedAcBatch(const PreparedAcBatch &batch,
                       PreparedAcBatchBackend &preferred_backend,
                       PreparedAcFallbackPolicy fallback_policy);

// Reconstructs a semantic complex CSR member from immutable structure and
// per-member values. It is exposed for replay evidence and hostile tests only.
[[nodiscard]] Result<ComplexCsrMatrix>
MaterializePreparedAcMatrix(const PreparedAcBatch &batch,
                            std::size_t member_ordinal);

} // namespace ohmnivore

#endif // OHMNIVORE_PREPARED_AC_H_
