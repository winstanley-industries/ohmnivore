#include "cpp/benchmarks/prepared_ac_session.h"

#include "cpp/src/prepared_ac_internal.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
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
    "dimension,union_nnz,sweep_points,batch_size,start_frequency_hz,"
    "stop_frequency_hz,g_series,g_shunt,c_series,c_shunt,corner_count";
inline constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
inline constexpr std::uint64_t kFnvOffsetFirst = 14695981039346656037ULL;
inline constexpr std::uint64_t kFnvOffsetSecond = 9521211207457086692ULL;

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
        "session field '" + std::string(field) +
            "' is not a canonical unsigned integer");
  }
  std::size_t value = 0;
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) {
    return Result<std::size_t>::Fail(ErrorCode::kPreparedBatchMalformed,
                                     "session field '" + std::string(field) +
                                         "' is not an integer");
  }
  return Result<std::size_t>::Ok(value);
}

[[nodiscard]] Result<double> ParseDouble(std::string_view text,
                                         std::string_view field) {
  if (!IsCanonicalDecimal(text)) {
    return Result<double>::Fail(ErrorCode::kPreparedBatchMalformed,
                                "session field '" + std::string(field) +
                                    "' is not a canonical decimal");
  }
  double value = 0.0;
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
      !std::isfinite(value)) {
    return Result<double>::Fail(ErrorCode::kPreparedBatchMalformed,
                                "session field '" + std::string(field) +
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

[[nodiscard]] std::size_t EdgeCount(const PreparedAcSessionCase &item) {
  if (item.topology == "binary_tree") {
    return item.node_count - 1;
  }
  if (item.topology == "square_grid") {
    const std::size_t width = static_cast<std::size_t>(
        std::llround(std::sqrt(static_cast<double>(item.node_count))));
    return 2 * width * (width - 1);
  }
  return item.node_count;
}

[[nodiscard]] bool HasValidShape(const PreparedAcSessionCase &item) {
  bool topology_valid = false;
  if (item.topology == "binary_tree") {
    topology_valid = item.node_count >= 2 && item.branch_count == 1;
  } else if (item.topology == "square_grid") {
    const std::size_t width = static_cast<std::size_t>(
        std::llround(std::sqrt(static_cast<double>(item.node_count))));
    topology_valid = width >= 2 && width * width == item.node_count &&
                     item.branch_count == 1;
  } else if (item.topology == "ring_multi") {
    topology_valid = item.node_count >= 3 && item.branch_count >= 2;
  }
  return topology_valid && item.node_count <= 8192 && item.branch_count <= 16 &&
         item.dimension == item.node_count + item.branch_count &&
         item.union_nonzeros ==
             item.node_count + 2 * EdgeCount(item) + 2 * item.branch_count &&
         item.sweep_points == item.batch_size && item.batch_size >= 2 &&
         item.batch_size <= 2048 && item.start_frequency_hz > 0.0 &&
         item.stop_frequency_hz > item.start_frequency_hz &&
         item.g_series > 0.0 && item.g_shunt > 0.0 && item.c_series > 0.0 &&
         item.c_shunt > 0.0 && item.corner_count >= 2 &&
         item.corner_count <= 64;
}

[[nodiscard]] std::string NodeName(std::size_t node) {
  return "n" + std::to_string(node);
}

void AddEdgeComponents(Circuit *circuit, std::size_t edge, std::size_t first,
                       std::size_t second, double resistance_ohms,
                       double capacitance_farads) {
  circuit->components.push_back(Resistor{.name = "Rs" + std::to_string(edge),
                                         .positive_node = NodeName(first),
                                         .negative_node = NodeName(second),
                                         .resistance_ohms = resistance_ohms});
  circuit->components.push_back(
      Capacitor{.name = "Cs" + std::to_string(edge),
                .positive_node = NodeName(first),
                .negative_node = NodeName(second),
                .capacitance_farads = capacitance_farads});
}

} // namespace

Result<PreparedAcSessionCorpus>
LoadPreparedAcSessionCorpus(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return Result<PreparedAcSessionCorpus>::Fail(
        ErrorCode::kIo, "could not open prepared AC session corpus: " + path);
  }
  const std::string bytes((std::istreambuf_iterator<char>(input)),
                          std::istreambuf_iterator<char>());
  if (bytes.empty() || bytes.back() != '\n' ||
      bytes.find('\r') != std::string::npos ||
      std::any_of(bytes.begin(), bytes.end(), [](unsigned char byte) {
        return byte != '\n' && (byte < 0x20 || byte > 0x7e);
      })) {
    return Result<PreparedAcSessionCorpus>::Fail(
        ErrorCode::kPreparedBatchMalformed,
        "prepared AC session corpus must be nonempty LF-terminated ASCII");
  }

  PreparedAcSessionCorpus corpus{
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
        return Result<PreparedAcSessionCorpus>::Fail(
            ErrorCode::kPreparedBatchMalformed,
            "prepared AC session corpus has an unknown header");
      }
      continue;
    }
    const std::vector<std::string_view> fields = Split(line);
    if (line.empty() || fields.size() != 17 || fields[0] != "1") {
      return Result<PreparedAcSessionCorpus>::Fail(
          ErrorCode::kPreparedBatchMalformed,
          "prepared AC session row has wrong field count or schema version");
    }
    if (!IsIdentifier(fields[1]) || !IsIdentifier(fields[2]) ||
        (fields[3] != "binary_tree" && fields[3] != "square_grid" &&
         fields[3] != "ring_multi") ||
        !case_ids.insert(std::string(fields[1])).second) {
      return Result<PreparedAcSessionCorpus>::Fail(
          ErrorCode::kPreparedBatchMalformed,
          "prepared AC session row has invalid or duplicate identity");
    }

    auto node_count = ParseSize(fields[4], "node_count");
    auto branch_count = ParseSize(fields[5], "branch_count");
    auto dimension = ParseSize(fields[6], "dimension");
    auto union_nonzeros = ParseSize(fields[7], "union_nnz");
    auto sweep_points = ParseSize(fields[8], "sweep_points");
    auto batch_size = ParseSize(fields[9], "batch_size");
    auto start_frequency = ParseDouble(fields[10], "start_frequency_hz");
    auto stop_frequency = ParseDouble(fields[11], "stop_frequency_hz");
    auto g_series = ParseDouble(fields[12], "g_series");
    auto g_shunt = ParseDouble(fields[13], "g_shunt");
    auto c_series = ParseDouble(fields[14], "c_series");
    auto c_shunt = ParseDouble(fields[15], "c_shunt");
    auto corner_count = ParseSize(fields[16], "corner_count");
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
    capture_error(sweep_points);
    capture_error(batch_size);
    capture_error(start_frequency);
    capture_error(stop_frequency);
    capture_error(g_series);
    capture_error(g_shunt);
    capture_error(c_series);
    capture_error(c_shunt);
    capture_error(corner_count);
    if (parse_error != nullptr) {
      return Result<PreparedAcSessionCorpus>::Fail(parse_error->code,
                                                   parse_error->message);
    }

    PreparedAcSessionCase item{
        .case_id = std::string(fields[1]),
        .workload_class = std::string(fields[2]),
        .topology = std::string(fields[3]),
        .node_count = node_count.TakeValue(),
        .branch_count = branch_count.TakeValue(),
        .dimension = dimension.TakeValue(),
        .union_nonzeros = union_nonzeros.TakeValue(),
        .sweep_points = sweep_points.TakeValue(),
        .batch_size = batch_size.TakeValue(),
        .start_frequency_hz = start_frequency.TakeValue(),
        .stop_frequency_hz = stop_frequency.TakeValue(),
        .g_series = g_series.TakeValue(),
        .g_shunt = g_shunt.TakeValue(),
        .c_series = c_series.TakeValue(),
        .c_shunt = c_shunt.TakeValue(),
        .corner_count = corner_count.TakeValue(),
    };
    if (!HasValidShape(item)) {
      return Result<PreparedAcSessionCorpus>::Fail(
          ErrorCode::kPreparedBatchMalformed,
          "prepared AC session row violates the bounded v1 case contract");
    }
    corpus.cases.push_back(std::move(item));
  }
  if (corpus.cases.size() != 4) {
    return Result<PreparedAcSessionCorpus>::Fail(
        ErrorCode::kPreparedBatchMalformed,
        "prepared AC session v1 requires exactly four declared cases");
  }
  if (corpus.manifest_fingerprint != kPreparedAcSessionV1ManifestFingerprint) {
    return Result<PreparedAcSessionCorpus>::Fail(
        ErrorCode::kPreparedBatchMalformed,
        "prepared AC session v1 manifest fingerprint mismatch: observed " +
            corpus.manifest_fingerprint);
  }
  return Result<PreparedAcSessionCorpus>::Ok(std::move(corpus));
}

Result<Circuit>
BuildPreparedAcSessionCircuit(const PreparedAcSessionCase &session_case,
                              std::size_t corner_ordinal) {
  if (!HasValidShape(session_case) ||
      corner_ordinal >= session_case.corner_count) {
    return Result<Circuit>::Fail(
        ErrorCode::kPreparedBatchMalformed,
        "prepared AC session case or corner is outside the v1 contract");
  }
  const double ordinal = static_cast<double>(corner_ordinal);
  const double series_g = session_case.g_series * (1.0 + 0.0125 * ordinal);
  const double shunt_g =
      session_case.g_shunt * (1.0 + 0.0025 * (corner_ordinal % 5));
  const double series_c =
      session_case.c_series * (1.0 + 0.02 * ((corner_ordinal * 3) % 7));
  const double shunt_c =
      session_case.c_shunt * (1.0 + 0.01 * ((corner_ordinal * 5) % 11));
  if (!std::isfinite(series_g) || !std::isfinite(shunt_g) ||
      !std::isfinite(series_c) || !std::isfinite(shunt_c)) {
    return Result<Circuit>::Fail(ErrorCode::kPreparedBatchMalformed,
                                 "prepared AC corner scaling is non-finite");
  }

  Circuit circuit;
  const std::size_t component_count = 2 * session_case.node_count +
                                      2 * EdgeCount(session_case) +
                                      session_case.branch_count;
  circuit.components.reserve(component_count);
  for (std::size_t node = 0; node < session_case.node_count; ++node) {
    circuit.components.push_back(Resistor{.name = "Rg" + std::to_string(node),
                                          .positive_node = NodeName(node),
                                          .negative_node = "0",
                                          .resistance_ohms = 1.0 / shunt_g});
    circuit.components.push_back(Capacitor{.name = "Cg" + std::to_string(node),
                                           .positive_node = NodeName(node),
                                           .negative_node = "0",
                                           .capacitance_farads = shunt_c});
  }

  std::size_t edge = 0;
  if (session_case.topology == "binary_tree") {
    for (std::size_t node = 1; node < session_case.node_count; ++node) {
      AddEdgeComponents(&circuit, edge++, (node - 1) / 2, node, 1.0 / series_g,
                        series_c);
    }
  } else if (session_case.topology == "square_grid") {
    const std::size_t width = static_cast<std::size_t>(
        std::llround(std::sqrt(static_cast<double>(session_case.node_count))));
    for (std::size_t row = 0; row < width; ++row) {
      for (std::size_t column = 0; column < width; ++column) {
        const std::size_t node = row * width + column;
        if (column + 1 < width) {
          AddEdgeComponents(&circuit, edge++, node, node + 1, 1.0 / series_g,
                            series_c);
        }
        if (row + 1 < width) {
          AddEdgeComponents(&circuit, edge++, node, node + width,
                            1.0 / series_g, series_c);
        }
      }
    }
  } else {
    for (std::size_t node = 0; node < session_case.node_count; ++node) {
      AddEdgeComponents(&circuit, edge++, node,
                        (node + 1) % session_case.node_count, 1.0 / series_g,
                        series_c);
    }
  }
  if (edge != EdgeCount(session_case)) {
    return Result<Circuit>::Fail(ErrorCode::kPreparedBatchMalformed,
                                 "prepared AC session edge count disagrees");
  }

  for (std::size_t branch = 0; branch < session_case.branch_count; ++branch) {
    const std::size_t node =
        branch * session_case.node_count / session_case.branch_count;
    circuit.components.push_back(VoltageSource{
        .name = "V" + std::to_string(branch),
        .positive_node = NodeName(node),
        .negative_node = "0",
        .dc_volts = std::nullopt,
        .ac =
            AcSourceSpecification{
                .magnitude =
                    (1.0 + 0.01 * ordinal) / static_cast<double>(branch + 1),
                .phase_degrees = static_cast<double>(
                    (corner_ordinal * 7 + branch * 19) % 360),
            },
    });
  }
  circuit.analyses.push_back(AcAnalysis{
      .sweep_type = AcSweepType::kLin,
      .points = session_case.sweep_points,
      .start_frequency_hz = session_case.start_frequency_hz,
      .stop_frequency_hz = session_case.stop_frequency_hz,
  });
  return Result<Circuit>::Ok(std::move(circuit));
}

Result<PreparedAcBatch>
PrepareAcSessionCorner(const PreparedAcSessionCase &session_case,
                       std::size_t corner_ordinal) {
  auto circuit = BuildPreparedAcSessionCircuit(session_case, corner_ordinal);
  if (!circuit.ok()) {
    return Result<PreparedAcBatch>::Fail(circuit.error().code,
                                         circuit.error().message);
  }
  auto system = CompileMna(circuit.value());
  if (!system.ok()) {
    return Result<PreparedAcBatch>::Fail(system.error().code,
                                         system.error().message);
  }
  auto prepared = PrepareLinearAcBatch(
      system.value(),
      AcAnalysis{.sweep_type = AcSweepType::kLin,
                 .points = session_case.sweep_points,
                 .start_frequency_hz = session_case.start_frequency_hz,
                 .stop_frequency_hz = session_case.stop_frequency_hz},
      "prepared-ac-session-v1/" + session_case.case_id, session_case.case_id,
      "corner-" + std::to_string(corner_ordinal));
  if (!prepared.ok()) {
    return prepared;
  }
  if (prepared.value().structure.dimension != session_case.dimension ||
      prepared.value().structure.column_indices.size() !=
          session_case.union_nonzeros ||
      prepared.value().members.size() != session_case.batch_size) {
    return Result<PreparedAcBatch>::Fail(
        ErrorCode::kPreparedBatchMalformed,
        "prepared AC session corner does not match its declared shape");
  }
  return prepared;
}

Result<bool>
ValidatePreparedAcBatchResultForEvidence(const PreparedAcBatch &batch,
                                         const PreparedAcBatchResult &result) {
  return internal::ValidatePreparedAcBatchResultForEvidence(batch, result);
}

} // namespace ohmnivore::benchmarks
