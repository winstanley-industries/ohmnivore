#include "cpp/benchmarks/prepared_ac_replay.h"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "ohmnivore/compiler.h"
#include "ohmnivore/ir.h"
#include "ohmnivore/prepared_ac.h"
#include "ohmnivore/status.h"

namespace ohmnivore::benchmarks {
namespace {

inline constexpr std::string_view kHeader =
    "schema_version,case_id,workload_class,topology,node_count,branch_count,"
    "dimension,union_nnz,sweep_type,sweep_points,batch_size,"
    "start_frequency_hz,stop_frequency_hz,g_series,g_shunt,c_series,c_shunt,"
    "reuse_count";
inline constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
inline constexpr std::uint64_t kFnvOffsetFirst = 14695981039346656037ULL;
inline constexpr std::uint64_t kFnvOffsetSecond = 9521211207457086692ULL;

class ReplayFingerprintBuilder {
public:
  explicit ReplayFingerprintBuilder(std::string_view domain) {
    AddString(domain);
  }

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

  [[nodiscard]] std::string Finish() const {
    constexpr char kHex[] = "0123456789abcdef";
    std::string result = "v1-";
    const auto append = [&](std::uint64_t value) {
      for (int shift = 60; shift >= 0; shift -= 4) {
        result.push_back(kHex[(value >> shift) & 0xfU]);
      }
    };
    append(first_);
    append(second_);
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
           character == '-';
  });
}

[[nodiscard]] bool IsCanonicalDecimal(std::string_view value) {
  if (value.empty() || value.front() == '+' || value.front() == '-' ||
      value.front() == '.' || value.back() == '.') {
    return false;
  }
  const std::size_t exponent = value.find('e');
  if (value.find('E') != std::string_view::npos ||
      (exponent != std::string_view::npos &&
       value.find('e', exponent + 1) != std::string_view::npos)) {
    return false;
  }
  const std::string_view mantissa = value.substr(0, exponent);
  const std::size_t decimal = mantissa.find('.');
  if (mantissa.find('.', decimal == std::string_view::npos ? 0 : decimal + 1) !=
      std::string_view::npos) {
    return false;
  }
  const std::string_view integer = mantissa.substr(0, decimal);
  if (integer.empty() || (integer.size() > 1 && integer.front() == '0') ||
      !std::all_of(integer.begin(), integer.end(), [](char character) {
        return character >= '0' && character <= '9';
      })) {
    return false;
  }
  if (decimal != std::string_view::npos) {
    const std::string_view fraction = mantissa.substr(decimal + 1);
    if (fraction.empty() || fraction.back() == '0' ||
        !std::all_of(fraction.begin(), fraction.end(), [](char character) {
          return character >= '0' && character <= '9';
        })) {
      return false;
    }
  }
  if (exponent != std::string_view::npos) {
    std::string_view digits = value.substr(exponent + 1);
    if (!digits.empty() && digits.front() == '-') {
      digits.remove_prefix(1);
    }
    if (digits.empty() || (digits.size() > 1 && digits.front() == '0') ||
        !std::all_of(digits.begin(), digits.end(), [](char character) {
          return character >= '0' && character <= '9';
        })) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] Result<std::size_t> ParseSize(std::string_view text,
                                            std::string_view field) {
  if (text.empty() || (text.size() > 1 && text.front() == '0')) {
    return Result<std::size_t>::Fail(
        ErrorCode::kPreparedBatchMalformed,
        "replay field '" + std::string(field) +
            "' is not a canonical unsigned integer");
  }
  std::size_t value = 0;
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) {
    return Result<std::size_t>::Fail(ErrorCode::kPreparedBatchMalformed,
                                     "replay field '" + std::string(field) +
                                         "' is not an integer");
  }
  return Result<std::size_t>::Ok(value);
}

[[nodiscard]] Result<double> ParseDouble(std::string_view text,
                                         std::string_view field) {
  if (!IsCanonicalDecimal(text)) {
    return Result<double>::Fail(ErrorCode::kPreparedBatchMalformed,
                                "replay field '" + std::string(field) +
                                    "' is not a canonical decimal");
  }
  double value = 0.0;
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
      !std::isfinite(value)) {
    return Result<double>::Fail(ErrorCode::kPreparedBatchMalformed,
                                "replay field '" + std::string(field) +
                                    "' is not finite FP64");
  }
  return Result<double>::Ok(value);
}

[[nodiscard]] std::vector<std::string_view> Split(std::string_view line) {
  std::vector<std::string_view> fields;
  std::size_t begin = 0;
  while (begin <= line.size()) {
    const std::size_t comma = line.find(',', begin);
    if (comma == std::string_view::npos) {
      fields.push_back(line.substr(begin));
      break;
    }
    fields.push_back(line.substr(begin, comma - begin));
    begin = comma + 1;
  }
  return fields;
}

[[nodiscard]] std::string FingerprintManifest(std::string_view bytes) {
  std::uint64_t first = kFnvOffsetFirst;
  std::uint64_t second = kFnvOffsetSecond;
  for (const unsigned char byte : bytes) {
    first = (first ^ byte) * kFnvPrime;
    second = (second ^ byte) * kFnvPrime;
  }
  constexpr char kHex[] = "0123456789abcdef";
  std::string result = "v1-";
  const auto append = [&](std::uint64_t value) {
    for (int shift = 60; shift >= 0; shift -= 4) {
      result.push_back(kHex[(value >> shift) & 0xfU]);
    }
  };
  append(first);
  append(second);
  return result;
}

[[nodiscard]] std::size_t EdgeCount(const PreparedAcReplayCase &item) {
  if (item.topology == "ladder" || item.topology == "binary_tree") {
    return item.node_count - 1;
  }
  if (item.topology == "grid_32x32") {
    return 2 * 32 * 31;
  }
  return item.node_count;
}

void AddEdge(std::vector<std::map<std::size_t, double>> *rows,
             std::size_t first, std::size_t second, double value) {
  (*rows)[first][first] += value;
  (*rows)[second][second] += value;
  (*rows)[first][second] -= value;
  (*rows)[second][first] -= value;
}

[[nodiscard]] CsrMatrix BuildMatrix(const PreparedAcReplayCase &item,
                                    bool dynamic) {
  const double series = dynamic ? item.c_series : item.g_series;
  const double shunt = dynamic ? item.c_shunt : item.g_shunt;
  std::vector<std::map<std::size_t, double>> rows(item.dimension);
  for (std::size_t node = 0; node < item.node_count; ++node) {
    rows[node][node] += shunt;
  }
  if (item.topology == "ladder") {
    for (std::size_t node = 1; node < item.node_count; ++node) {
      AddEdge(&rows, node - 1, node, series);
    }
  } else if (item.topology == "binary_tree") {
    for (std::size_t node = 1; node < item.node_count; ++node) {
      AddEdge(&rows, (node - 1) / 2, node, series);
    }
  } else if (item.topology == "grid_32x32") {
    for (std::size_t row = 0; row < 32; ++row) {
      for (std::size_t column = 0; column < 32; ++column) {
        const std::size_t node = row * 32 + column;
        if (column + 1 < 32) {
          AddEdge(&rows, node, node + 1, series);
        }
        if (row + 1 < 32) {
          AddEdge(&rows, node, node + 32, series);
        }
      }
    }
  } else {
    for (std::size_t node = 0; node < item.node_count; ++node) {
      AddEdge(&rows, node, (node + 1) % item.node_count, series);
    }
  }
  if (!dynamic) {
    for (std::size_t branch = 0; branch < item.branch_count; ++branch) {
      const std::size_t node = branch * item.node_count / item.branch_count;
      const std::size_t variable = item.node_count + branch;
      rows[node][variable] += 1.0;
      rows[variable][node] += 1.0;
    }
  }

  CsrMatrix matrix{
      .rows = item.dimension,
      .columns = item.dimension,
      .values = {},
      .column_indices = {},
      .row_offsets = {},
  };
  matrix.row_offsets.reserve(item.dimension + 1);
  matrix.row_offsets.push_back(0);
  for (const auto &row : rows) {
    for (const auto &[column, value] : row) {
      matrix.column_indices.push_back(column);
      matrix.values.push_back(value);
    }
    matrix.row_offsets.push_back(matrix.values.size());
  }
  return matrix;
}

[[nodiscard]] Result<AcSweepType> ParseSweepType(std::string_view value) {
  if (value == "LIN") {
    return Result<AcSweepType>::Ok(AcSweepType::kLin);
  }
  if (value == "DEC") {
    return Result<AcSweepType>::Ok(AcSweepType::kDec);
  }
  return Result<AcSweepType>::Fail(ErrorCode::kPreparedBatchMalformed,
                                   "replay sweep type is not LIN or DEC");
}

} // namespace

std::string_view PreparedAcReplaySweepName(AcSweepType sweep_type) {
  switch (sweep_type) {
  case AcSweepType::kLin:
    return "LIN";
  case AcSweepType::kDec:
    return "DEC";
  case AcSweepType::kOct:
    return "OCT";
  }
  return "unknown";
}

Result<PreparedAcReplayCorpus>
LoadPreparedAcReplayCorpus(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return Result<PreparedAcReplayCorpus>::Fail(
        ErrorCode::kIo, "could not open prepared AC replay corpus: " + path);
  }
  const std::string bytes((std::istreambuf_iterator<char>(input)),
                          std::istreambuf_iterator<char>());
  if (bytes.empty() || bytes.back() != '\n' ||
      bytes.find('\r') != std::string::npos ||
      std::any_of(bytes.begin(), bytes.end(), [](unsigned char byte) {
        return byte != '\n' && (byte < 0x20 || byte > 0x7e);
      })) {
    return Result<PreparedAcReplayCorpus>::Fail(
        ErrorCode::kPreparedBatchMalformed,
        "prepared AC replay corpus must be nonempty LF-terminated ASCII");
  }

  PreparedAcReplayCorpus corpus{
      .schema_version = 1,
      .manifest_fingerprint = FingerprintManifest(bytes),
      .cases = {},
  };
  std::set<std::string> case_ids;
  std::size_t line_number = 0;
  std::size_t begin = 0;
  while (begin < bytes.size()) {
    const std::size_t end = bytes.find('\n', begin);
    const std::string_view line(bytes.data() + begin, end - begin);
    begin = end + 1;
    ++line_number;
    if (line_number == 1) {
      if (line != kHeader) {
        return Result<PreparedAcReplayCorpus>::Fail(
            ErrorCode::kPreparedBatchMalformed,
            "prepared AC replay corpus has an unknown header");
      }
      continue;
    }
    if (line.empty()) {
      return Result<PreparedAcReplayCorpus>::Fail(
          ErrorCode::kPreparedBatchMalformed,
          "prepared AC replay corpus contains a blank row");
    }
    const std::vector<std::string_view> fields = Split(line);
    if (fields.size() != 18 || fields[0] != "1") {
      return Result<PreparedAcReplayCorpus>::Fail(
          ErrorCode::kPreparedBatchMalformed,
          "prepared AC replay row has wrong field count or schema version");
    }
    if (!IsIdentifier(fields[1]) ||
        (fields[2] != "small_control" && fields[2] != "medium" &&
         fields[2] != "large" && fields[2] != "medium_wide") ||
        (fields[3] != "ladder" && fields[3] != "binary_tree" &&
         fields[3] != "grid_32x32" && fields[3] != "ring_multi")) {
      return Result<PreparedAcReplayCorpus>::Fail(
          ErrorCode::kPreparedBatchMalformed,
          "prepared AC replay row has invalid case, class, or topology");
    }
    if (!case_ids.insert(std::string(fields[1])).second) {
      return Result<PreparedAcReplayCorpus>::Fail(
          ErrorCode::kPreparedBatchMalformed,
          "prepared AC replay corpus contains a duplicate case ID");
    }

    auto node_count = ParseSize(fields[4], "node_count");
    auto branch_count = ParseSize(fields[5], "branch_count");
    auto dimension = ParseSize(fields[6], "dimension");
    auto union_nonzeros = ParseSize(fields[7], "union_nnz");
    auto sweep_type = ParseSweepType(fields[8]);
    auto sweep_points = ParseSize(fields[9], "sweep_points");
    auto batch_size = ParseSize(fields[10], "batch_size");
    auto start_frequency = ParseDouble(fields[11], "start_frequency_hz");
    auto stop_frequency = ParseDouble(fields[12], "stop_frequency_hz");
    auto g_series = ParseDouble(fields[13], "g_series");
    auto g_shunt = ParseDouble(fields[14], "g_shunt");
    auto c_series = ParseDouble(fields[15], "c_series");
    auto c_shunt = ParseDouble(fields[16], "c_shunt");
    auto reuse_count = ParseSize(fields[17], "reuse_count");
    const Error *parse_error = nullptr;
    const auto capture_error = [&](const auto &parsed) {
      if (parse_error == nullptr && !parsed.ok()) {
        parse_error = &parsed.error();
      }
    };
    capture_error(node_count);
    capture_error(branch_count);
    capture_error(dimension);
    capture_error(union_nonzeros);
    capture_error(sweep_type);
    capture_error(sweep_points);
    capture_error(batch_size);
    capture_error(start_frequency);
    capture_error(stop_frequency);
    capture_error(g_series);
    capture_error(g_shunt);
    capture_error(c_series);
    capture_error(c_shunt);
    capture_error(reuse_count);
    if (parse_error != nullptr) {
      return Result<PreparedAcReplayCorpus>::Fail(parse_error->code,
                                                  parse_error->message);
    }

    PreparedAcReplayCase item{
        .case_id = std::string(fields[1]),
        .workload_class = std::string(fields[2]),
        .topology = std::string(fields[3]),
        .node_count = node_count.TakeValue(),
        .branch_count = branch_count.TakeValue(),
        .dimension = dimension.TakeValue(),
        .union_nonzeros = union_nonzeros.TakeValue(),
        .sweep_type = sweep_type.TakeValue(),
        .sweep_points = sweep_points.TakeValue(),
        .batch_size = batch_size.TakeValue(),
        .start_frequency_hz = start_frequency.TakeValue(),
        .stop_frequency_hz = stop_frequency.TakeValue(),
        .g_series = g_series.TakeValue(),
        .g_shunt = g_shunt.TakeValue(),
        .c_series = c_series.TakeValue(),
        .c_shunt = c_shunt.TakeValue(),
        .reuse_count = reuse_count.TakeValue(),
    };
    const bool topology_valid =
        (item.topology == "grid_32x32" && item.node_count == 1024 &&
         item.branch_count == 1) ||
        (item.topology == "ring_multi" && item.node_count >= 3 &&
         item.branch_count >= 2) ||
        ((item.topology == "ladder" || item.topology == "binary_tree") &&
         item.node_count >= 2 && item.branch_count == 1);
    const bool scales_valid =
        item.g_series > 0.0 && item.g_shunt > 0.0 && item.c_series > 0.0 &&
        item.c_shunt > 0.0 &&
        std::isfinite(4.0 * item.g_series + item.g_shunt) &&
        std::isfinite(4.0 * item.c_series + item.c_shunt);
    bool sweep_valid = item.sweep_points >= 2 && item.sweep_points <= 4096 &&
                       item.batch_size >= 2 && item.batch_size <= 4096 &&
                       item.start_frequency_hz > 0.0 &&
                       item.stop_frequency_hz > item.start_frequency_hz;
    if (sweep_valid && item.sweep_type == AcSweepType::kLin) {
      sweep_valid = item.batch_size == item.sweep_points;
    } else if (sweep_valid) {
      const double decades =
          std::log10(item.stop_frequency_hz / item.start_frequency_hz);
      const double rounded_decades = std::round(decades);
      sweep_valid =
          std::abs(decades - rounded_decades) <= 1.0e-12 &&
          item.batch_size ==
              static_cast<std::size_t>(rounded_decades) * item.sweep_points + 1;
    }
    if (!topology_valid || item.node_count > 2048 || item.branch_count > 16 ||
        item.dimension != item.node_count + item.branch_count ||
        item.union_nonzeros !=
            item.node_count + 2 * EdgeCount(item) + 2 * item.branch_count ||
        !sweep_valid || !scales_valid || item.reuse_count == 0 ||
        item.reuse_count > 64) {
      return Result<PreparedAcReplayCorpus>::Fail(
          ErrorCode::kPreparedBatchMalformed,
          "prepared AC replay row violates the bounded v1 case contract");
    }
    corpus.cases.push_back(std::move(item));
  }
  if (corpus.cases.size() != 4) {
    return Result<PreparedAcReplayCorpus>::Fail(
        ErrorCode::kPreparedBatchMalformed,
        "prepared AC replay v1 requires exactly four declared cases");
  }
  if (corpus.manifest_fingerprint != kPreparedAcReplayV1ManifestFingerprint) {
    return Result<PreparedAcReplayCorpus>::Fail(
        ErrorCode::kPreparedBatchMalformed,
        "prepared AC replay v1 manifest fingerprint mismatch: observed " +
            corpus.manifest_fingerprint);
  }
  return Result<PreparedAcReplayCorpus>::Ok(std::move(corpus));
}

Result<MnaSystem>
BuildPreparedAcReplaySystem(const PreparedAcReplayCase &replay_case) {
  CsrMatrix g = BuildMatrix(replay_case, false);
  CsrMatrix c = BuildMatrix(replay_case, true);
  std::vector<std::string> node_names;
  node_names.reserve(replay_case.node_count);
  for (std::size_t index = 0; index < replay_case.node_count; ++index) {
    node_names.push_back("n" + std::to_string(index));
  }
  std::vector<std::string> branch_names;
  branch_names.reserve(replay_case.branch_count);
  std::vector<std::complex<double>> b_ac(replay_case.dimension, {0.0, 0.0});
  for (std::size_t branch = 0; branch < replay_case.branch_count; ++branch) {
    branch_names.push_back("V_replay_" + std::to_string(branch));
    const double scale = 1.0 / static_cast<double>(branch + 1);
    b_ac[replay_case.node_count + branch] = {
        scale, (branch % 2 == 0 ? 0.25 : -0.25) * scale};
  }
  return Result<MnaSystem>::Ok(MnaSystem{
      .g = std::move(g),
      .c = std::move(c),
      .b_dc = std::vector<double>(replay_case.dimension, 0.0),
      .b_ac = std::move(b_ac),
      .node_names = std::move(node_names),
      .branch_names = std::move(branch_names),
  });
}

Result<PreparedAcBatch>
PrepareAcReplayCase(const PreparedAcReplayCase &replay_case) {
  auto system = BuildPreparedAcReplaySystem(replay_case);
  if (!system.ok()) {
    return Result<PreparedAcBatch>::Fail(system.error().code,
                                         system.error().message);
  }
  auto batch = PrepareLinearAcBatch(
      system.value(),
      AcAnalysis{.sweep_type = replay_case.sweep_type,
                 .points = replay_case.sweep_points,
                 .start_frequency_hz = replay_case.start_frequency_hz,
                 .stop_frequency_hz = replay_case.stop_frequency_hz},
      "prepared-ac-replay-v1/" + replay_case.case_id, replay_case.case_id,
      "nominal");
  if (!batch.ok()) {
    return batch;
  }
  if (batch.value().structure.dimension != replay_case.dimension ||
      batch.value().structure.column_indices.size() !=
          replay_case.union_nonzeros ||
      batch.value().members.size() != replay_case.batch_size) {
    return Result<PreparedAcBatch>::Fail(
        ErrorCode::kPreparedBatchMalformed,
        "prepared AC replay case does not match its declared shape");
  }
  return batch;
}

std::string FingerprintPreparedAcReplayMembers(const PreparedAcBatch &batch) {
  ReplayFingerprintBuilder fingerprint("prepared-ac-replay-members-v1");
  fingerprint.AddUint64(batch.contract_version);
  fingerprint.AddString(batch.replay_id);
  fingerprint.AddString(batch.structure.fingerprint);
  fingerprint.AddString(batch.batch_fingerprint);
  fingerprint.AddUint64(batch.members.size());
  for (const PreparedAcMember &member : batch.members) {
    fingerprint.AddUint64(member.identity.contract_version);
    fingerprint.AddString(member.identity.replay_id);
    fingerprint.AddString(member.identity.circuit_id);
    fingerprint.AddString(member.identity.corner_id);
    fingerprint.AddUint64(member.identity.ordinal);
    fingerprint.AddDouble(member.identity.frequency_hz);
    fingerprint.AddString(member.identity.structure_fingerprint);
    fingerprint.AddString(member.identity.content_fingerprint);
  }
  return fingerprint.Finish();
}

} // namespace ohmnivore::benchmarks
