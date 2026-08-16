#include "cpp/tests/google_test.h"

#include <bit>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ohmnivore/compiler.h"
#include "ohmnivore/parser.h"
#include "ohmnivore/prepared_ac.h"
#include "ohmnivore/simulator.h"
#include "ohmnivore/status.h"

namespace ohmnivore {
namespace {

inline constexpr char kAcNetlist[] = R"(V1 in 0 AC 1
R1 in mid 10
L1 mid out 1m
C1 out 0 1u
.AC LIN 5 10 10000
.END
)";

[[nodiscard]] MnaSystem Compile(std::string_view netlist) {
  auto parsed = ParseNetlist(netlist);
  if (!parsed.ok()) {
    ADD_FAILURE() << parsed.error().message;
    return {};
  }
  auto compiled = CompileMna(parsed.value());
  if (!compiled.ok()) {
    ADD_FAILURE() << compiled.error().message;
    return {};
  }
  return compiled.TakeValue();
}

[[nodiscard]] PreparedAcBatch Prepare() {
  MnaSystem system = Compile(kAcNetlist);
  auto prepared =
      PrepareLinearAcBatch(system,
                           AcAnalysis{.sweep_type = AcSweepType::kLin,
                                      .points = 5,
                                      .start_frequency_hz = 10.0,
                                      .stop_frequency_hz = 10000.0},
                           "gpu01-test/v1", "rlc-series", "nominal");
  if (!prepared.ok()) {
    ADD_FAILURE() << prepared.error().message;
    return {};
  }
  return prepared.TakeValue();
}

[[nodiscard]] PreparedAcBatchResult Solve(const PreparedAcBatch &batch) {
  CpuKluPreparedAcBatchBackend backend;
  auto solved = backend.Execute(batch);
  if (!solved.ok()) {
    ADD_FAILURE() << solved.error().message;
    return {};
  }
  auto accepted = ValidatePreparedAcBatchResult(batch, solved.value());
  if (!accepted.ok()) {
    ADD_FAILURE() << accepted.error().message;
    return {};
  }
  auto evidence =
      ValidatePreparedAcBatchResultForEvidence(batch, solved.value());
  if (!evidence.ok()) {
    ADD_FAILURE() << evidence.error().message;
    return {};
  }
  return solved.TakeValue();
}

void ExpectRejected(const PreparedAcBatch &batch,
                    const PreparedAcBatchResult &result,
                    ErrorCode expected_code,
                    bool evidence_validator_must_reject = true) {
  auto accepted = ValidatePreparedAcBatchResult(batch, result);
  ASSERT_FALSE(accepted.ok());
  EXPECT_EQ(accepted.error().code, expected_code) << accepted.error().message;
  if (evidence_validator_must_reject) {
    auto evidence = ValidatePreparedAcBatchResultForEvidence(batch, result);
    ASSERT_FALSE(evidence.ok());
    EXPECT_EQ(evidence.error().code, expected_code) << evidence.error().message;
  }
}

class StaticBackend final : public PreparedAcBatchBackend {
public:
  explicit StaticBackend(PreparedAcBatchResult result)
      : result_(std::move(result)) {}

  Result<PreparedAcBatchResult>
  Execute(const PreparedAcBatch &unused) override {
    static_cast<void>(unused);
    return Result<PreparedAcBatchResult>::Ok(result_);
  }

private:
  PreparedAcBatchResult result_;
};

class ErrorBackend final : public PreparedAcBatchBackend {
public:
  Result<PreparedAcBatchResult>
  Execute(const PreparedAcBatch &unused) override {
    static_cast<void>(unused);
    return Result<PreparedAcBatchResult>::Fail(ErrorCode::kSolve,
                                               "hostile backend failure");
  }
};

class ThrowingBackend final : public PreparedAcBatchBackend {
public:
  Result<PreparedAcBatchResult>
  Execute(const PreparedAcBatch &unused) override {
    static_cast<void>(unused);
    throw std::runtime_error("hostile backend exception");
  }
};

class CountingBackend final : public PreparedAcBatchBackend {
public:
  Result<PreparedAcBatchResult>
  Execute(const PreparedAcBatch &unused) override {
    static_cast<void>(unused);
    ++calls;
    return Result<PreparedAcBatchResult>::Fail(ErrorCode::kSolve,
                                               "must not be called");
  }

  std::size_t calls = 0;
};

[[nodiscard]] PreparedAcBatch PrepareSingular(bool structurally_empty) {
  MnaSystem system;
  if (structurally_empty) {
    system.g = CsrMatrix{.rows = 1,
                         .columns = 1,
                         .values = {},
                         .column_indices = {},
                         .row_offsets = {0, 0}};
    system.c = system.g;
    system.b_dc = {0.0};
    system.b_ac = {{0.0, 0.0}};
    system.node_names = {"n"};
  } else {
    system.g = CsrMatrix{.rows = 2,
                         .columns = 2,
                         .values = {1.0, 1.0, 1.0, 1.0},
                         .column_indices = {0, 1, 0, 1},
                         .row_offsets = {0, 2, 4}};
    system.c = CsrMatrix{.rows = 2,
                         .columns = 2,
                         .values = {},
                         .column_indices = {},
                         .row_offsets = {0, 0, 0}};
    system.b_dc = {0.0, 0.0};
    system.b_ac = {{0.0, 0.0}, {0.0, 0.0}};
    system.node_names = {"a", "b"};
  }
  auto prepared = PrepareLinearAcBatch(
      system,
      AcAnalysis{.sweep_type = AcSweepType::kLin,
                 .points = 2,
                 .start_frequency_hz = 1.0,
                 .stop_frequency_hz = 2.0},
      structurally_empty ? "gpu01-singular/empty" : "gpu01-singular/numeric",
      structurally_empty ? "empty" : "numeric", "nominal");
  if (!prepared.ok()) {
    ADD_FAILURE() << prepared.error().message;
    return {};
  }
  return prepared.TakeValue();
}

[[nodiscard]] PreparedAcBatchResult
ZeroResidualHostileResult(const PreparedAcBatch &batch) {
  PreparedAcBatchResult result{
      .contract_version = batch.contract_version,
      .replay_id = batch.replay_id,
      .structure_fingerprint = batch.structure.fingerprint,
      .batch_fingerprint = batch.batch_fingerprint,
      .members = {},
  };
  result.members.reserve(batch.members.size());
  for (const PreparedAcMember &member : batch.members) {
    result.members.push_back(PreparedAcResultMember{
        .identity = member.identity,
        .solution = std::vector<std::complex<double>>(batch.structure.dimension,
                                                      {0.0, 0.0}),
    });
  }
  return result;
}

TEST(Gpu01PreparedAcTest,
     PreservesFrequencyStructureOrderingKluReuseAndOrdinaryResults) {
  const PreparedAcBatch batch = Prepare();
  auto valid = ValidatePreparedAcBatch(batch);
  ASSERT_TRUE(valid.ok()) << valid.error().message;
  ASSERT_EQ(batch.contract_version, kPreparedAcContractVersion);
  ASSERT_EQ(batch.members.size(), 5U);
  EXPECT_EQ(batch.members.front().identity.ordinal, 0U);
  EXPECT_EQ(batch.members.back().identity.ordinal, 4U);
  EXPECT_DOUBLE_EQ(batch.members.front().identity.frequency_hz, 10.0);
  EXPECT_DOUBLE_EQ(batch.members.back().identity.frequency_hz, 10000.0);
  EXPECT_FALSE(batch.structure.fingerprint.empty());
  EXPECT_FALSE(batch.batch_fingerprint.empty());
  for (const PreparedAcMember &member : batch.members) {
    EXPECT_EQ(member.identity.contract_version, batch.contract_version);
    EXPECT_EQ(member.identity.replay_id, batch.replay_id);
    EXPECT_EQ(member.identity.structure_fingerprint,
              batch.structure.fingerprint);
    EXPECT_EQ(member.matrix_values.size(),
              batch.structure.column_indices.size());
    EXPECT_EQ(member.rhs.size(), batch.structure.dimension);
  }

  CpuKluPreparedAcBatchBackend backend;
  auto solved = backend.Execute(batch);
  ASSERT_TRUE(solved.ok()) << solved.error().message;
  auto accepted = ValidatePreparedAcBatchResult(batch, solved.value());
  ASSERT_TRUE(accepted.ok()) << accepted.error().message;
  EXPECT_EQ(backend.statistics().symbolic_analyses, 1U);
  EXPECT_EQ(backend.statistics().solves, batch.members.size());

  auto ordinary = SimulateAc(kAcNetlist);
  ASSERT_TRUE(ordinary.ok()) << ordinary.error().message;
  ASSERT_EQ(ordinary.value().frequencies_hz.size(), batch.members.size());
  for (std::size_t frequency = 0; frequency < batch.members.size();
       ++frequency) {
    EXPECT_DOUBLE_EQ(ordinary.value().frequencies_hz[frequency],
                     batch.members[frequency].identity.frequency_hz);
    std::vector<std::complex<double>> ordinary_solution;
    for (const auto &[unused, values] : ordinary.value().node_voltages) {
      static_cast<void>(unused);
      ordinary_solution.push_back(values[frequency]);
    }
    for (const auto &[unused, values] : ordinary.value().branch_currents) {
      static_cast<void>(unused);
      ordinary_solution.push_back(values[frequency]);
    }
    EXPECT_EQ(solved.value().members[frequency].solution, ordinary_solution);
  }
}

TEST(Gpu01PreparedAcTest, ReusesPreparedCpuStateAndRepeatsBitwise) {
  const PreparedAcBatch batch = Prepare();
  CpuKluPreparedAcBatchBackend backend;
  auto first = backend.Execute(batch);
  auto second = backend.Execute(batch);
  ASSERT_TRUE(first.ok()) << first.error().message;
  ASSERT_TRUE(second.ok()) << second.error().message;
  EXPECT_EQ(first.value().contract_version, second.value().contract_version);
  EXPECT_EQ(first.value().replay_id, second.value().replay_id);
  EXPECT_EQ(first.value().batch_fingerprint, second.value().batch_fingerprint);
  ASSERT_EQ(first.value().members.size(), second.value().members.size());
  for (std::size_t member = 0; member < first.value().members.size();
       ++member) {
    EXPECT_EQ(first.value().members[member].identity,
              second.value().members[member].identity);
    ASSERT_EQ(first.value().members[member].solution.size(),
              second.value().members[member].solution.size());
    for (std::size_t value = 0;
         value < first.value().members[member].solution.size(); ++value) {
      EXPECT_EQ(std::bit_cast<std::uint64_t>(
                    first.value().members[member].solution[value].real()),
                std::bit_cast<std::uint64_t>(
                    second.value().members[member].solution[value].real()));
      EXPECT_EQ(std::bit_cast<std::uint64_t>(
                    first.value().members[member].solution[value].imag()),
                std::bit_cast<std::uint64_t>(
                    second.value().members[member].solution[value].imag()));
    }
  }
  EXPECT_EQ(backend.statistics().symbolic_analyses, 1U);
  EXPECT_EQ(backend.statistics().solves, 2U * batch.members.size());
}

TEST(Gpu01PreparedAcTest, RejectsMalformedAndNonlinearPreparation) {
  PreparedAcBatch malformed = Prepare();
  malformed.structure.row_offsets.back() -= 1;
  auto rejected = ValidatePreparedAcBatch(malformed);
  ASSERT_FALSE(rejected.ok());
  EXPECT_EQ(rejected.error().code, ErrorCode::kPreparedBatchMalformed);

  malformed = Prepare();
  malformed.members[0].identity.contract_version = 2;
  rejected = ValidatePreparedAcBatch(malformed);
  ASSERT_FALSE(rejected.ok());
  EXPECT_EQ(rejected.error().code, ErrorCode::kPreparedBatchMalformed);

  malformed = Prepare();
  malformed.members[0].identity.replay_id = "another-replay";
  rejected = ValidatePreparedAcBatch(malformed);
  ASSERT_FALSE(rejected.ok());
  EXPECT_EQ(rejected.error().code, ErrorCode::kPreparedBatchMalformed);

  malformed = Prepare();
  ASSERT_GT(malformed.members.size(), 1U);
  malformed.members[1].matrix_values[0] = {
      std::numeric_limits<double>::quiet_NaN(), 0.0};
  rejected = ValidatePreparedAcBatch(malformed);
  ASSERT_FALSE(rejected.ok());
  EXPECT_EQ(rejected.error().code, ErrorCode::kPreparedBatchMalformed);
  EXPECT_NE(rejected.error().message.find("non-finite"), std::string::npos);

  MnaSystem system = Compile(kAcNetlist);
  auto invalid_identity =
      PrepareLinearAcBatch(system,
                           AcAnalysis{.sweep_type = AcSweepType::kLin,
                                      .points = 2,
                                      .start_frequency_hz = 1.0,
                                      .stop_frequency_hz = 2.0},
                           "bad identity", "circuit", "nominal");
  ASSERT_FALSE(invalid_identity.ok());
  EXPECT_EQ(invalid_identity.error().code, ErrorCode::kPreparedBatchMalformed);

  MnaSystem nonlinear = Compile(R"(.MODEL dm D
D1 out 0 dm
I1 0 out AC 1
.AC LIN 2 1 2
.END
)");
  auto unsupported =
      PrepareLinearAcBatch(nonlinear,
                           AcAnalysis{.sweep_type = AcSweepType::kLin,
                                      .points = 2,
                                      .start_frequency_hz = 1.0,
                                      .stop_frequency_hz = 2.0},
                           "gpu01-test/v1", "diode", "nominal");
  ASSERT_FALSE(unsupported.ok());
  EXPECT_EQ(unsupported.error().code, ErrorCode::kUnsupported);
}

TEST(Gpu01PreparedAcTest, RejectsHostileValuesResidualsAndAssociations) {
  const PreparedAcBatch batch = Prepare();
  const PreparedAcBatchResult valid = Solve(batch);

  PreparedAcBatchResult non_finite = valid;
  non_finite.members[0].solution[0] = {std::numeric_limits<double>::quiet_NaN(),
                                       0.0};
  ExpectRejected(batch, non_finite, ErrorCode::kPreparedInvalidResult);

  PreparedAcBatchResult excessive_residual = valid;
  excessive_residual.members[0].solution[0] += std::complex<double>{1.0, 1.0};
  ExpectRejected(batch, excessive_residual, ErrorCode::kPreparedInvalidResult);

  PreparedAcBatchResult wrong_association = valid;
  wrong_association.members[1].identity.circuit_id = "another-circuit";
  ExpectRejected(batch, wrong_association,
                 ErrorCode::kPreparedResultAssociation);

  PreparedAcBatchResult swapped_solutions = valid;
  std::swap(swapped_solutions.members[0].solution,
            swapped_solutions.members[1].solution);
  ExpectRejected(batch, swapped_solutions, ErrorCode::kPreparedInvalidResult);

  PreparedAcBatchResult wrong_corner = valid;
  wrong_corner.members[1].identity.corner_id = "another-corner";
  ExpectRejected(batch, wrong_corner, ErrorCode::kPreparedResultAssociation);

  PreparedAcBatchResult wrong_frequency = valid;
  wrong_frequency.members[1].identity.frequency_hz = 123.0;
  ExpectRejected(batch, wrong_frequency, ErrorCode::kPreparedResultAssociation);

  PreparedAcBatchResult wrong_dimension = valid;
  wrong_dimension.members[0].solution.pop_back();
  ExpectRejected(batch, wrong_dimension, ErrorCode::kPreparedInvalidResult);

  PreparedAcBatchResult extra = valid;
  PreparedAcResultMember extra_member = extra.members.back();
  extra_member.identity.ordinal = batch.members.size();
  extra.members.push_back(std::move(extra_member));
  ExpectRejected(batch, extra, ErrorCode::kPreparedResultAssociation);
}

TEST(Gpu01PreparedAcTest, RejectsStaleMissingDuplicateAndReorderedResults) {
  const PreparedAcBatch batch = Prepare();
  const PreparedAcBatchResult valid = Solve(batch);

  PreparedAcBatchResult stale_batch = valid;
  stale_batch.contract_version = 2;
  ExpectRejected(batch, stale_batch, ErrorCode::kPreparedBatchStale);

  stale_batch = valid;
  stale_batch.replay_id = "another-replay";
  ExpectRejected(batch, stale_batch, ErrorCode::kPreparedBatchStale);

  stale_batch = valid;
  stale_batch.structure_fingerprint = "v1-stale";
  ExpectRejected(batch, stale_batch, ErrorCode::kPreparedBatchStale);

  stale_batch = valid;
  stale_batch.batch_fingerprint = "v1-stale";
  ExpectRejected(batch, stale_batch, ErrorCode::kPreparedBatchStale);

  PreparedAcBatchResult stale_member = valid;
  stale_member.members[0].identity.content_fingerprint = "v1-stale";
  ExpectRejected(batch, stale_member, ErrorCode::kPreparedBatchStale);

  stale_member = valid;
  stale_member.members[0].identity.content_fingerprint =
      stale_member.members[1].identity.content_fingerprint;
  ExpectRejected(batch, stale_member, ErrorCode::kPreparedBatchStale);

  stale_member = valid;
  stale_member.members[0].identity.contract_version = 2;
  ExpectRejected(batch, stale_member, ErrorCode::kPreparedBatchStale);

  stale_member = valid;
  stale_member.members[0].identity.replay_id = "another-replay";
  ExpectRejected(batch, stale_member, ErrorCode::kPreparedBatchStale);

  stale_member = valid;
  stale_member.members[0].identity.structure_fingerprint = "v1-stale";
  ExpectRejected(batch, stale_member, ErrorCode::kPreparedBatchStale);

  PreparedAcBatchResult missing = valid;
  missing.members.pop_back();
  ExpectRejected(batch, missing, ErrorCode::kPreparedResultMissing);

  PreparedAcBatchResult duplicate = valid;
  duplicate.members.back() = duplicate.members.front();
  ExpectRejected(batch, duplicate, ErrorCode::kPreparedResultDuplicate);

  PreparedAcBatchResult reordered = valid;
  std::swap(reordered.members[0], reordered.members[1]);
  ExpectRejected(batch, reordered, ErrorCode::kPreparedResultReordered);
}

TEST(Gpu01PreparedAcTest,
     RejectsZeroResidualResultsForStructurallyAndNumericallySingularSystems) {
  for (const bool structurally_empty : {true, false}) {
    const PreparedAcBatch batch = PrepareSingular(structurally_empty);
    auto input = ValidatePreparedAcBatch(batch);
    ASSERT_TRUE(input.ok()) << input.error().message;
    const PreparedAcBatchResult hostile = ZeroResidualHostileResult(batch);
    ExpectRejected(batch, hostile, ErrorCode::kPreparedInvalidResult, false);
    auto residual_only =
        ValidatePreparedAcBatchResultForEvidence(batch, hostile);
    ASSERT_TRUE(residual_only.ok()) << residual_only.error().message;

    CpuKluPreparedAcBatchBackend cpu;
    auto solved = cpu.Execute(batch);
    ASSERT_FALSE(solved.ok());
    EXPECT_TRUE(solved.error().code == ErrorCode::kSingular ||
                solved.error().code == ErrorCode::kInvalidStructure)
        << solved.error().message;
  }
}

TEST(Gpu01PreparedAcTest, ExplicitFallbackDiscardsTheCompleteHostileBatch) {
  const PreparedAcBatch batch = Prepare();
  PreparedAcBatchResult hostile = Solve(batch);
  hostile.members[0].solution[0] = {std::numeric_limits<double>::infinity(),
                                    0.0};

  StaticBackend fail_closed_backend(hostile);
  auto fail_closed = ExecutePreparedAcBatch(
      batch, fail_closed_backend, PreparedAcFallbackPolicy::kFailClosed);
  ASSERT_FALSE(fail_closed.ok());
  EXPECT_EQ(fail_closed.error().code, ErrorCode::kPreparedInvalidResult);

  StaticBackend fallback_backend(std::move(hostile));
  auto recovered = ExecutePreparedAcBatch(
      batch, fallback_backend, PreparedAcFallbackPolicy::kCpuKluFullBatch);
  ASSERT_TRUE(recovered.ok()) << recovered.error().message;
  EXPECT_TRUE(recovered.value().used_cpu_fallback);
  auto accepted =
      ValidatePreparedAcBatchResult(batch, recovered.value().batch_result);
  EXPECT_TRUE(accepted.ok()) << accepted.error().message;
}

TEST(Gpu01PreparedAcTest,
     TypedBackendFailuresExceptionsAndInvalidInputsObeyFallbackPolicy) {
  const PreparedAcBatch batch = Prepare();

  ErrorBackend error_backend;
  auto failed = ExecutePreparedAcBatch(batch, error_backend,
                                       PreparedAcFallbackPolicy::kFailClosed);
  ASSERT_FALSE(failed.ok());
  EXPECT_EQ(failed.error().code, ErrorCode::kSolve);

  auto recovered = ExecutePreparedAcBatch(
      batch, error_backend, PreparedAcFallbackPolicy::kCpuKluFullBatch);
  ASSERT_TRUE(recovered.ok()) << recovered.error().message;
  EXPECT_TRUE(recovered.value().used_cpu_fallback);

  ThrowingBackend throwing_backend;
  failed = ExecutePreparedAcBatch(batch, throwing_backend,
                                  PreparedAcFallbackPolicy::kFailClosed);
  ASSERT_FALSE(failed.ok());
  EXPECT_EQ(failed.error().code, ErrorCode::kPreparedBackendFailure);

  recovered = ExecutePreparedAcBatch(
      batch, throwing_backend, PreparedAcFallbackPolicy::kCpuKluFullBatch);
  ASSERT_TRUE(recovered.ok()) << recovered.error().message;
  EXPECT_TRUE(recovered.value().used_cpu_fallback);

  PreparedAcBatch malformed = batch;
  malformed.structure.row_offsets.back() -= 1;
  CountingBackend counting_backend;
  failed = ExecutePreparedAcBatch(malformed, counting_backend,
                                  PreparedAcFallbackPolicy::kCpuKluFullBatch);
  ASSERT_FALSE(failed.ok());
  EXPECT_EQ(failed.error().code, ErrorCode::kPreparedBatchMalformed);
  EXPECT_EQ(counting_backend.calls, 0U);
}

} // namespace
} // namespace ohmnivore
