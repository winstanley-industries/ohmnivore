#include "ohmnivore/prepared_ac.h"

#include "prepared_ac_internal.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <numbers>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ohmnivore/compiler.h"
#include "ohmnivore/simulator.h"
#include "ohmnivore/solver.h"
#include "ohmnivore/sparse.h"
#include "ohmnivore/status.h"

namespace ohmnivore {
namespace {

inline constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
inline constexpr std::uint64_t kFnvOffsetFirst = 14695981039346656037ULL;
inline constexpr std::uint64_t kFnvOffsetSecond = 9521211207457086692ULL;

class FingerprintBuilder {
public:
  explicit FingerprintBuilder(std::string_view domain) { AddString(domain); }

  void AddByte(std::uint8_t value) {
    first_ = (first_ ^ value) * kFnvPrime;
    second_ = (second_ ^ value) * kFnvPrime;
  }

  void AddUint64(std::uint64_t value) {
    for (std::size_t index = 0; index < 8; ++index) {
      AddByte(static_cast<std::uint8_t>(value & 0xffU));
      value >>= 8U;
    }
  }

  void AddString(std::string_view value) {
    AddUint64(value.size());
    for (const unsigned char character : value) {
      AddByte(character);
    }
  }

  void AddDouble(double value) {
    AddUint64(std::bit_cast<std::uint64_t>(value));
  }

  void AddComplex(std::complex<double> value) {
    AddDouble(value.real());
    AddDouble(value.imag());
  }

  [[nodiscard]] std::string Finish() const {
    constexpr char kHex[] = "0123456789abcdef";
    std::string result = "v1-";
    result.reserve(3 + 32);
    const auto append = [&](std::uint64_t value, std::string *output) {
      for (int shift = 60; shift >= 0; shift -= 4) {
        output->push_back(kHex[(value >> shift) & 0xfU]);
      }
    };
    append(first_, &result);
    append(second_, &result);
    return result;
  }

private:
  std::uint64_t first_ = kFnvOffsetFirst;
  std::uint64_t second_ = kFnvOffsetSecond;
};

[[nodiscard]] bool IsIdentifier(std::string_view value) {
  if (value.empty()) {
    return false;
  }
  return std::all_of(value.begin(), value.end(), [](char character) {
    return (character >= 'A' && character <= 'Z') ||
           (character >= 'a' && character <= 'z') ||
           (character >= '0' && character <= '9') || character == '_' ||
           character == '-' || character == '.' || character == ':' ||
           character == '/';
  });
}

[[nodiscard]] bool IsFinite(std::complex<double> value) {
  return std::isfinite(value.real()) && std::isfinite(value.imag());
}

[[nodiscard]] std::string
FingerprintStructure(std::size_t dimension,
                     const std::vector<std::size_t> &row_offsets,
                     const std::vector<std::size_t> &column_indices) {
  FingerprintBuilder fingerprint("prepared-ac-structure-v1");
  fingerprint.AddUint64(dimension);
  fingerprint.AddUint64(row_offsets.size());
  for (const std::size_t offset : row_offsets) {
    fingerprint.AddUint64(offset);
  }
  fingerprint.AddUint64(column_indices.size());
  for (const std::size_t column : column_indices) {
    fingerprint.AddUint64(column);
  }
  return fingerprint.Finish();
}

[[nodiscard]] std::string
FingerprintMember(const PreparedAcMemberIdentity &identity,
                  const std::vector<std::complex<double>> &matrix_values,
                  const std::vector<std::complex<double>> &rhs) {
  FingerprintBuilder fingerprint("prepared-ac-member-v1");
  fingerprint.AddUint64(identity.contract_version);
  fingerprint.AddString(identity.replay_id);
  fingerprint.AddString(identity.circuit_id);
  fingerprint.AddString(identity.corner_id);
  fingerprint.AddUint64(identity.ordinal);
  fingerprint.AddDouble(identity.frequency_hz);
  fingerprint.AddString(identity.structure_fingerprint);
  fingerprint.AddUint64(matrix_values.size());
  for (const std::complex<double> value : matrix_values) {
    fingerprint.AddComplex(value);
  }
  fingerprint.AddUint64(rhs.size());
  for (const std::complex<double> value : rhs) {
    fingerprint.AddComplex(value);
  }
  return fingerprint.Finish();
}

[[nodiscard]] std::string FingerprintBatch(const PreparedAcBatch &batch) {
  FingerprintBuilder fingerprint("prepared-ac-batch-v1");
  fingerprint.AddUint64(batch.contract_version);
  fingerprint.AddString(batch.replay_id);
  fingerprint.AddString(batch.structure.fingerprint);
  fingerprint.AddUint64(batch.members.size());
  for (const PreparedAcMember &member : batch.members) {
    fingerprint.AddUint64(member.identity.contract_version);
    fingerprint.AddString(member.identity.replay_id);
    fingerprint.AddString(member.identity.circuit_id);
    fingerprint.AddString(member.identity.corner_id);
    fingerprint.AddUint64(member.identity.ordinal);
    fingerprint.AddDouble(member.identity.frequency_hz);
    fingerprint.AddString(member.identity.content_fingerprint);
  }
  return fingerprint.Finish();
}

[[nodiscard]] bool SameBaseIdentity(const PreparedAcMemberIdentity &first,
                                    const PreparedAcMemberIdentity &second) {
  return first.contract_version == second.contract_version &&
         first.replay_id == second.replay_id &&
         first.circuit_id == second.circuit_id &&
         first.corner_id == second.corner_id &&
         first.ordinal == second.ordinal &&
         std::bit_cast<std::uint64_t>(first.frequency_hz) ==
             std::bit_cast<std::uint64_t>(second.frequency_hz) &&
         first.structure_fingerprint == second.structure_fingerprint;
}

[[nodiscard]] PreparedAcBatchResult
MakeResultEnvelope(const PreparedAcBatch &batch) {
  return PreparedAcBatchResult{
      .contract_version = batch.contract_version,
      .replay_id = batch.replay_id,
      .structure_fingerprint = batch.structure.fingerprint,
      .batch_fingerprint = batch.batch_fingerprint,
      .members = {},
  };
}

[[nodiscard]] Result<bool>
ValidatePreparedAcBatchImpl(const PreparedAcBatch &batch) {
  if (batch.contract_version != kPreparedAcContractVersion) {
    return Result<bool>::Fail(ErrorCode::kPreparedBatchMalformed,
                              "prepared AC contract version must be 1");
  }
  if (!IsIdentifier(batch.replay_id)) {
    return Result<bool>::Fail(
        ErrorCode::kPreparedBatchMalformed,
        "prepared AC replay identity contains invalid characters");
  }
  if (batch.structure.dimension == 0 || batch.members.empty()) {
    return Result<bool>::Fail(
        ErrorCode::kPreparedBatchMalformed,
        "prepared AC batch requires a nonempty square system and members");
  }
  const std::string structure_fingerprint = FingerprintStructure(
      batch.structure.dimension, batch.structure.row_offsets,
      batch.structure.column_indices);
  if (batch.structure.fingerprint != structure_fingerprint) {
    return Result<bool>::Fail(
        ErrorCode::kPreparedBatchMalformed,
        "prepared AC structure fingerprint does not match its content");
  }

  std::set<std::string> content_fingerprints;
  bool canonical_structure_validated = false;
  double prior_frequency = 0.0;
  std::string circuit_id;
  std::string corner_id;
  for (std::size_t index = 0; index < batch.members.size(); ++index) {
    const PreparedAcMember &member = batch.members[index];
    if (member.identity.contract_version != batch.contract_version ||
        member.identity.replay_id != batch.replay_id) {
      return Result<bool>::Fail(
          ErrorCode::kPreparedBatchMalformed,
          "prepared AC member version or replay identity differs from batch");
    }
    if (!IsIdentifier(member.identity.circuit_id) ||
        !IsIdentifier(member.identity.corner_id)) {
      return Result<bool>::Fail(
          ErrorCode::kPreparedBatchMalformed,
          "prepared AC member identity contains invalid characters");
    }
    if (index == 0) {
      circuit_id = member.identity.circuit_id;
      corner_id = member.identity.corner_id;
    } else if (member.identity.circuit_id != circuit_id ||
               member.identity.corner_id != corner_id) {
      return Result<bool>::Fail(ErrorCode::kPreparedBatchMalformed,
                                "prepared AC v1 batch members must share "
                                "circuit and corner identity");
    }
    if (member.identity.ordinal != index) {
      return Result<bool>::Fail(
          ErrorCode::kPreparedBatchMalformed,
          "prepared AC member ordinals must be contiguous and ordered");
    }
    if (!std::isfinite(member.identity.frequency_hz) ||
        member.identity.frequency_hz <= 0.0 ||
        (index != 0 && member.identity.frequency_hz <= prior_frequency)) {
      return Result<bool>::Fail(
          ErrorCode::kPreparedBatchMalformed,
          "prepared AC member frequencies must be finite, positive, and "
          "strictly increasing");
    }
    prior_frequency = member.identity.frequency_hz;
    if (member.identity.structure_fingerprint != structure_fingerprint) {
      return Result<bool>::Fail(
          ErrorCode::kPreparedBatchMalformed,
          "prepared AC member structure identity does not match the batch");
    }
    if (member.matrix_values.size() != batch.structure.column_indices.size() ||
        member.rhs.size() != batch.structure.dimension) {
      return Result<bool>::Fail(
          ErrorCode::kPreparedBatchMalformed,
          "prepared AC member values or right-hand side have wrong dimensions");
    }
    for (const std::complex<double> value : member.rhs) {
      if (!IsFinite(value)) {
        return Result<bool>::Fail(
            ErrorCode::kPreparedBatchMalformed,
            "prepared AC member right-hand side contains a non-finite value");
      }
    }
    if (!canonical_structure_validated) {
      ComplexCsrMatrix matrix{
          .rows = batch.structure.dimension,
          .columns = batch.structure.dimension,
          .values = member.matrix_values,
          .column_indices = batch.structure.column_indices,
          .row_offsets = batch.structure.row_offsets,
      };
      auto converted = ConvertCsrToSolverCsc(matrix);
      if (!converted.ok()) {
        return Result<bool>::Fail(
            ErrorCode::kPreparedBatchMalformed,
            "prepared AC member has invalid canonical structure or values: " +
                converted.error().message);
      }
      canonical_structure_validated = true;
    } else if (std::any_of(member.matrix_values.begin(),
                           member.matrix_values.end(),
                           [](std::complex<double> value) {
                             return !IsFinite(value);
                           })) {
      return Result<bool>::Fail(
          ErrorCode::kPreparedBatchMalformed,
          "prepared AC member has invalid canonical structure or values: "
          "matrix contains a non-finite value");
    }
    const std::string content_fingerprint =
        FingerprintMember(member.identity, member.matrix_values, member.rhs);
    if (member.identity.content_fingerprint != content_fingerprint) {
      return Result<bool>::Fail(
          ErrorCode::kPreparedBatchMalformed,
          "prepared AC member fingerprint does not match its content");
    }
    if (!content_fingerprints.insert(content_fingerprint).second) {
      return Result<bool>::Fail(ErrorCode::kPreparedBatchMalformed,
                                "prepared AC batch has a duplicate member");
    }
  }
  if (batch.batch_fingerprint != FingerprintBatch(batch)) {
    return Result<bool>::Fail(
        ErrorCode::kPreparedBatchMalformed,
        "prepared AC batch fingerprint does not match its ordered members");
  }
  return Result<bool>::Ok(true);
}

} // namespace

Result<PreparedAcBatch> PrepareLinearAcBatch(const MnaSystem &system,
                                             const AcAnalysis &analysis,
                                             std::string replay_id,
                                             std::string circuit_id,
                                             std::string corner_id) {
  try {
    if (!IsIdentifier(replay_id) || !IsIdentifier(circuit_id) ||
        !IsIdentifier(corner_id)) {
      return Result<PreparedAcBatch>::Fail(
          ErrorCode::kPreparedBatchMalformed,
          "prepared AC identities contain invalid characters");
    }
    if (!system.diode_descriptors.empty() || !system.bjt_descriptors.empty() ||
        !system.behavioral_descriptors.empty()) {
      return Result<PreparedAcBatch>::Fail(
          ErrorCode::kUnsupported,
          "prepared AC v1 supports only the authoritative linear AC subset");
    }
    auto frequencies = GenerateAcFrequencies(analysis);
    if (!frequencies.ok()) {
      return Result<PreparedAcBatch>::Fail(frequencies.error().code,
                                           frequencies.error().message);
    }

    PreparedAcBatch batch;
    batch.replay_id = std::move(replay_id);
    batch.members.reserve(frequencies.value().size());
    for (std::size_t ordinal = 0; ordinal < frequencies.value().size();
         ++ordinal) {
      const double frequency_hz = frequencies.value()[ordinal];
      const double angular_frequency = 2.0 * std::numbers::pi * frequency_hz;
      auto matrix = FormAcMatrix(system.g, system.c, angular_frequency);
      if (!matrix.ok()) {
        return Result<PreparedAcBatch>::Fail(matrix.error().code,
                                             matrix.error().message);
      }
      if (ordinal == 0) {
        batch.structure.dimension = matrix.value().rows;
        batch.structure.row_offsets = matrix.value().row_offsets;
        batch.structure.column_indices = matrix.value().column_indices;
        batch.structure.fingerprint = FingerprintStructure(
            batch.structure.dimension, batch.structure.row_offsets,
            batch.structure.column_indices);
      } else if (matrix.value().rows != batch.structure.dimension ||
                 matrix.value().row_offsets != batch.structure.row_offsets ||
                 matrix.value().column_indices !=
                     batch.structure.column_indices) {
        return Result<PreparedAcBatch>::Fail(
            ErrorCode::kPreparedBatchMalformed,
            "prepared AC matrices do not share one immutable structure");
      }
      PreparedAcMember member{
          .identity =
              PreparedAcMemberIdentity{
                  .contract_version = kPreparedAcContractVersion,
                  .replay_id = batch.replay_id,
                  .circuit_id = circuit_id,
                  .corner_id = corner_id,
                  .ordinal = ordinal,
                  .frequency_hz = frequency_hz,
                  .structure_fingerprint = batch.structure.fingerprint,
                  .content_fingerprint = {},
              },
          .matrix_values = matrix.TakeValue().values,
          .rhs = system.b_ac,
      };
      member.identity.content_fingerprint =
          FingerprintMember(member.identity, member.matrix_values, member.rhs);
      batch.members.push_back(std::move(member));
    }
    batch.batch_fingerprint = FingerprintBatch(batch);
    auto validated = ValidatePreparedAcBatchImpl(batch);
    if (!validated.ok()) {
      return Result<PreparedAcBatch>::Fail(validated.error().code,
                                           validated.error().message);
    }
    return Result<PreparedAcBatch>::Ok(std::move(batch));
  } catch (const std::bad_alloc &) {
    return Result<PreparedAcBatch>::Fail(ErrorCode::kFactorization,
                                         "prepared AC batch allocation failed");
  }
}

Result<bool> ValidatePreparedAcBatch(const PreparedAcBatch &batch) {
  try {
    return ValidatePreparedAcBatchImpl(batch);
  } catch (const std::bad_alloc &) {
    return Result<bool>::Fail(ErrorCode::kFactorization,
                              "prepared AC validation allocation failed");
  }
}

Result<ComplexCsrMatrix>
MaterializePreparedAcMatrix(const PreparedAcBatch &batch,
                            std::size_t member_ordinal) {
  try {
    if (member_ordinal >= batch.members.size()) {
      return Result<ComplexCsrMatrix>::Fail(
          ErrorCode::kPreparedBatchMalformed,
          "prepared AC member ordinal is outside the batch");
    }
    const PreparedAcMember &member = batch.members[member_ordinal];
    if (batch.structure.row_offsets.size() != batch.structure.dimension + 1 ||
        batch.structure.column_indices.size() != member.matrix_values.size()) {
      return Result<ComplexCsrMatrix>::Fail(
          ErrorCode::kPreparedBatchMalformed,
          "prepared AC structure and member values disagree");
    }
    return Result<ComplexCsrMatrix>::Ok(ComplexCsrMatrix{
        .rows = batch.structure.dimension,
        .columns = batch.structure.dimension,
        .values = member.matrix_values,
        .column_indices = batch.structure.column_indices,
        .row_offsets = batch.structure.row_offsets,
    });
  } catch (const std::bad_alloc &) {
    return Result<ComplexCsrMatrix>::Fail(
        ErrorCode::kFactorization,
        "prepared AC matrix materialization allocation failed");
  }
}

namespace {

Result<bool>
ValidatePreparedAcBatchResultImpl(const PreparedAcBatch &batch,
                                  const PreparedAcBatchResult &result,
                                  bool certify_with_fresh_klu) {
  try {
    auto batch_valid = ValidatePreparedAcBatchImpl(batch);
    if (!batch_valid.ok()) {
      return batch_valid;
    }
    if (result.contract_version != batch.contract_version ||
        result.replay_id != batch.replay_id ||
        result.structure_fingerprint != batch.structure.fingerprint ||
        result.batch_fingerprint != batch.batch_fingerprint) {
      return Result<bool>::Fail(
          ErrorCode::kPreparedBatchStale,
          "prepared AC result envelope belongs to another batch generation");
    }

    std::vector<bool> ordinal_seen(batch.members.size(), false);
    bool has_unknown_ordinal = false;
    for (const PreparedAcResultMember &member : result.members) {
      if (member.identity.ordinal >= batch.members.size()) {
        has_unknown_ordinal = true;
      } else if (ordinal_seen[member.identity.ordinal]) {
        return Result<bool>::Fail(
            ErrorCode::kPreparedResultDuplicate,
            "prepared AC result contains a duplicate identity ordinal");
      } else {
        ordinal_seen[member.identity.ordinal] = true;
      }
    }
    if (result.members.size() < batch.members.size()) {
      return Result<bool>::Fail(ErrorCode::kPreparedResultMissing,
                                "prepared AC result is missing members");
    }
    if (has_unknown_ordinal) {
      return Result<bool>::Fail(
          ErrorCode::kPreparedResultAssociation,
          "prepared AC result ordinal is outside the prepared batch");
    }

    for (const PreparedAcResultMember &returned : result.members) {
      const PreparedAcMemberIdentity &expected =
          batch.members[returned.identity.ordinal].identity;
      if (returned.identity.contract_version != batch.contract_version ||
          returned.identity.replay_id != batch.replay_id ||
          returned.identity.structure_fingerprint !=
              batch.structure.fingerprint) {
        return Result<bool>::Fail(
            ErrorCode::kPreparedBatchStale,
            "prepared AC result member belongs to another batch generation");
      }
      if (SameBaseIdentity(expected, returned.identity) &&
          expected.content_fingerprint !=
              returned.identity.content_fingerprint) {
        return Result<bool>::Fail(
            ErrorCode::kPreparedBatchStale,
            "prepared AC result member belongs to another content generation");
      }
      if (!SameBaseIdentity(expected, returned.identity) ||
          expected.content_fingerprint !=
              returned.identity.content_fingerprint) {
        return Result<bool>::Fail(
            ErrorCode::kPreparedResultAssociation,
            "prepared AC result contains an unknown or incorrectly associated "
            "member");
      }
    }
    if (result.members.size() != batch.members.size()) {
      return Result<bool>::Fail(ErrorCode::kPreparedResultAssociation,
                                "prepared AC result has extra members");
    }
    for (std::size_t index = 0; index < batch.members.size(); ++index) {
      if (result.members[index].identity.ordinal != index) {
        return Result<bool>::Fail(
            ErrorCode::kPreparedResultReordered,
            "prepared AC result members are not in prepared order");
      }
    }

    std::unique_ptr<SparseComplexFactorization> cpu_authority;
    for (std::size_t index = 0; index < batch.members.size(); ++index) {
      if (result.members[index].solution.size() != batch.structure.dimension) {
        return Result<bool>::Fail(
            ErrorCode::kPreparedInvalidResult,
            "prepared AC result solution has the wrong dimension");
      }
      auto matrix = MaterializePreparedAcMatrix(batch, index);
      if (!matrix.ok()) {
        return Result<bool>::Fail(matrix.error().code, matrix.error().message);
      }
      auto validation =
          ValidateSparseSolution(matrix.value(), batch.members[index].rhs,
                                 result.members[index].solution);
      if (!validation.ok()) {
        return Result<bool>::Fail(
            ErrorCode::kPreparedInvalidResult,
            "prepared AC result failed authoritative validation: " +
                validation.error().message);
      }
      if (!certify_with_fresh_klu) {
        continue;
      }
      if (cpu_authority == nullptr) {
        auto analyzed = SparseComplexFactorization::Analyze(matrix.value());
        if (!analyzed.ok()) {
          return Result<bool>::Fail(
              ErrorCode::kPreparedInvalidResult,
              "prepared AC result failed CPU KLU structural certification: " +
                  analyzed.error().message);
        }
        cpu_authority = analyzed.TakeValue();
      }
      auto certified = cpu_authority->FactorAndSolve(matrix.value(),
                                                     batch.members[index].rhs);
      if (!certified.ok()) {
        return Result<bool>::Fail(
            ErrorCode::kPreparedInvalidResult,
            "prepared AC result failed CPU KLU numeric certification: " +
                certified.error().message);
      }
      for (std::size_t component = 0;
           component < result.members[index].solution.size(); ++component) {
        const std::complex<double> returned =
            result.members[index].solution[component];
        const std::complex<double> authoritative = certified.value()[component];
        const double scale =
            std::max(std::abs(returned), std::abs(authoritative));
        if (std::abs(returned - authoritative) > 1.0e-12 + 1.0e-9 * scale) {
          return Result<bool>::Fail(
              ErrorCode::kPreparedInvalidResult,
              "prepared AC result disagrees with CPU KLU certification");
        }
      }
    }
    return Result<bool>::Ok(true);
  } catch (const std::bad_alloc &) {
    return Result<bool>::Fail(
        ErrorCode::kPreparedBackendFailure,
        "prepared AC result validation allocation failed");
  } catch (const std::exception &error) {
    return Result<bool>::Fail(
        ErrorCode::kPreparedBackendFailure,
        "prepared AC result validation threw an exception: " +
            std::string(error.what()));
  } catch (...) {
    return Result<bool>::Fail(ErrorCode::kPreparedBackendFailure,
                              "prepared AC result validation threw");
  }
}

} // namespace

Result<bool>
ValidatePreparedAcBatchResult(const PreparedAcBatch &batch,
                              const PreparedAcBatchResult &result) {
  return ValidatePreparedAcBatchResultImpl(batch, result, true);
}

Result<bool> internal::ValidatePreparedAcBatchResultForEvidence(
    const PreparedAcBatch &batch, const PreparedAcBatchResult &result) {
  return ValidatePreparedAcBatchResultImpl(batch, result, false);
}

CpuKluPreparedAcBatchBackend::CpuKluPreparedAcBatchBackend() = default;
CpuKluPreparedAcBatchBackend::~CpuKluPreparedAcBatchBackend() = default;
CpuKluPreparedAcBatchBackend::CpuKluPreparedAcBatchBackend(
    CpuKluPreparedAcBatchBackend &&) noexcept = default;
CpuKluPreparedAcBatchBackend &CpuKluPreparedAcBatchBackend::operator=(
    CpuKluPreparedAcBatchBackend &&) noexcept = default;

Result<PreparedAcBatchResult>
CpuKluPreparedAcBatchBackend::Execute(const PreparedAcBatch &batch) {
  try {
    auto valid = ValidatePreparedAcBatch(batch);
    if (!valid.ok()) {
      return Result<PreparedAcBatchResult>::Fail(valid.error().code,
                                                 valid.error().message);
    }
    if (cached_batch_fingerprint_ != batch.batch_fingerprint ||
        factorization_ == nullptr) {
      auto first_matrix = MaterializePreparedAcMatrix(batch, 0);
      if (!first_matrix.ok()) {
        return Result<PreparedAcBatchResult>::Fail(
            first_matrix.error().code, first_matrix.error().message);
      }
      auto analyzed = SparseComplexFactorization::Analyze(first_matrix.value());
      if (!analyzed.ok()) {
        return Result<PreparedAcBatchResult>::Fail(analyzed.error().code,
                                                   analyzed.error().message);
      }
      factorization_ = analyzed.TakeValue();
      cached_batch_fingerprint_ = batch.batch_fingerprint;
    }

    PreparedAcBatchResult result = MakeResultEnvelope(batch);
    result.members.reserve(batch.members.size());
    for (std::size_t index = 0; index < batch.members.size(); ++index) {
      auto matrix = MaterializePreparedAcMatrix(batch, index);
      if (!matrix.ok()) {
        return Result<PreparedAcBatchResult>::Fail(matrix.error().code,
                                                   matrix.error().message);
      }
      auto solved = factorization_->FactorAndSolve(matrix.value(),
                                                   batch.members[index].rhs);
      if (!solved.ok()) {
        return Result<PreparedAcBatchResult>::Fail(solved.error().code,
                                                   solved.error().message);
      }
      result.members.push_back(PreparedAcResultMember{
          .identity = batch.members[index].identity,
          .solution = solved.TakeValue(),
      });
    }
    statistics_ = factorization_->statistics();
    return Result<PreparedAcBatchResult>::Ok(std::move(result));
  } catch (const std::bad_alloc &) {
    return Result<PreparedAcBatchResult>::Fail(
        ErrorCode::kPreparedBackendFailure,
        "prepared AC CPU backend allocation failed");
  } catch (const std::exception &error) {
    return Result<PreparedAcBatchResult>::Fail(
        ErrorCode::kPreparedBackendFailure,
        "prepared AC CPU backend threw an exception: " +
            std::string(error.what()));
  } catch (...) {
    return Result<PreparedAcBatchResult>::Fail(
        ErrorCode::kPreparedBackendFailure,
        "prepared AC CPU backend threw an unknown exception");
  }
}

const SparseSolverStatistics &CpuKluPreparedAcBatchBackend::statistics() const {
  return statistics_;
}

Result<PreparedAcExecutionResult>
ExecutePreparedAcBatch(const PreparedAcBatch &batch,
                       PreparedAcBatchBackend &preferred_backend,
                       PreparedAcFallbackPolicy fallback_policy) {
  auto input = ValidatePreparedAcBatch(batch);
  if (!input.ok()) {
    return Result<PreparedAcExecutionResult>::Fail(input.error().code,
                                                   input.error().message);
  }

  const auto invoke_backend = [](PreparedAcBatchBackend &backend,
                                 const PreparedAcBatch &prepared) {
    try {
      return backend.Execute(prepared);
    } catch (const std::bad_alloc &) {
      return Result<PreparedAcBatchResult>::Fail(
          ErrorCode::kPreparedBackendFailure,
          "prepared AC backend allocation escaped its typed boundary");
    } catch (const std::exception &error) {
      return Result<PreparedAcBatchResult>::Fail(
          ErrorCode::kPreparedBackendFailure,
          "prepared AC backend exception escaped its typed boundary: " +
              std::string(error.what()));
    } catch (...) {
      return Result<PreparedAcBatchResult>::Fail(
          ErrorCode::kPreparedBackendFailure,
          "prepared AC backend exception escaped its typed boundary");
    }
  };

  auto preferred = invoke_backend(preferred_backend, batch);
  if (preferred.ok()) {
    auto accepted = ValidatePreparedAcBatchResult(batch, preferred.value());
    if (accepted.ok()) {
      return Result<PreparedAcExecutionResult>::Ok(PreparedAcExecutionResult{
          .batch_result = preferred.TakeValue(),
          .used_cpu_fallback = false,
      });
    }
    if (fallback_policy == PreparedAcFallbackPolicy::kFailClosed) {
      return Result<PreparedAcExecutionResult>::Fail(accepted.error().code,
                                                     accepted.error().message);
    }
  } else if (fallback_policy == PreparedAcFallbackPolicy::kFailClosed) {
    return Result<PreparedAcExecutionResult>::Fail(preferred.error().code,
                                                   preferred.error().message);
  }

  CpuKluPreparedAcBatchBackend cpu_fallback;
  auto fallback = invoke_backend(cpu_fallback, batch);
  if (!fallback.ok()) {
    return Result<PreparedAcExecutionResult>::Fail(fallback.error().code,
                                                   fallback.error().message);
  }
  auto accepted = ValidatePreparedAcBatchResult(batch, fallback.value());
  if (!accepted.ok()) {
    return Result<PreparedAcExecutionResult>::Fail(accepted.error().code,
                                                   accepted.error().message);
  }
  return Result<PreparedAcExecutionResult>::Ok(PreparedAcExecutionResult{
      .batch_result = fallback.TakeValue(),
      .used_cpu_fallback = true,
  });
}

} // namespace ohmnivore
