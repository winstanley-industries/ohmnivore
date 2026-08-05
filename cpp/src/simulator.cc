#include "ohmnivore/simulator.h"

#include <charconv>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <numbers>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

#include "ohmnivore/compiler.h"
#include "ohmnivore/ir.h"
#include "ohmnivore/parser.h"
#include "ohmnivore/solver.h"

namespace ohmnivore {
namespace {

inline constexpr std::size_t kMaxAcFrequencyPoints = 1'000'000;

[[nodiscard]] Result<std::string> FormatDouble(double value) {
  if (!std::isfinite(value)) {
    return Result<std::string>::Fail(
        ErrorCode::kIo, "cannot format a non-finite floating-point value");
  }
  char buffer[64];
  const auto formatted = std::to_chars(std::begin(buffer), std::end(buffer),
                                       value, std::chars_format::general);
  if (formatted.ec != std::errc{}) {
    return Result<std::string>::Fail(ErrorCode::kIo,
                                     "failed to format floating-point output");
  }
  return Result<std::string>::Ok(std::string(buffer, formatted.ptr));
}

[[nodiscard]] std::string EscapeCsvField(std::string_view field) {
  if (field.find_first_of(",\"\r\n") == std::string_view::npos) {
    return std::string(field);
  }
  std::string escaped;
  escaped.reserve(field.size() + 2);
  escaped.push_back('"');
  for (char character : field) {
    if (character == '"') {
      escaped.push_back('"');
    }
    escaped.push_back(character);
  }
  escaped.push_back('"');
  return escaped;
}

[[nodiscard]] Result<DcResult> RunDc(const MnaSystem &system) {
  auto solved = SolveCpuReference(system.g, system.b_dc);
  if (!solved.ok()) {
    return Result<DcResult>::Fail(solved.error().code, solved.error().message);
  }
  const std::vector<double> &solution = solved.value();

  DcResult result;
  for (std::size_t index = 0; index < system.node_names.size(); ++index) {
    result.node_voltages.emplace_back(system.node_names[index],
                                      solution[index]);
  }
  const std::size_t branch_offset = system.node_names.size();
  for (std::size_t index = 0; index < system.branch_names.size(); ++index) {
    result.branch_currents.emplace_back(system.branch_names[index],
                                        solution[branch_offset + index]);
  }
  return Result<DcResult>::Ok(std::move(result));
}

[[nodiscard]] Result<AcResult> RunAc(const MnaSystem &system,
                                     const AcAnalysis &analysis) {
  auto generated = GenerateAcFrequencies(analysis);
  if (!generated.ok()) {
    return Result<AcResult>::Fail(generated.error().code,
                                  generated.error().message);
  }

  AcResult result;
  result.frequencies_hz = generated.TakeValue();
  for (const std::string &name : system.node_names) {
    result.node_voltages.emplace_back(name,
                                      std::vector<std::complex<double>>{});
    result.node_voltages.back().second.reserve(result.frequencies_hz.size());
  }
  for (const std::string &name : system.branch_names) {
    result.branch_currents.emplace_back(name,
                                        std::vector<std::complex<double>>{});
    result.branch_currents.back().second.reserve(result.frequencies_hz.size());
  }

  for (double frequency_hz : result.frequencies_hz) {
    const double angular_frequency = 2.0 * std::numbers::pi * frequency_hz;
    auto matrix = FormAcMatrix(system.g, system.c, angular_frequency);
    if (!matrix.ok()) {
      return Result<AcResult>::Fail(matrix.error().code,
                                    matrix.error().message);
    }
    auto solved = SolveCpuComplexReference(matrix.value(), system.b_ac);
    if (!solved.ok()) {
      return Result<AcResult>::Fail(solved.error().code,
                                    solved.error().message);
    }
    const std::vector<std::complex<double>> &solution = solved.value();
    for (std::size_t index = 0; index < result.node_voltages.size(); ++index) {
      result.node_voltages[index].second.push_back(solution[index]);
    }
    const std::size_t branch_offset = system.node_names.size();
    for (std::size_t index = 0; index < result.branch_currents.size();
         ++index) {
      result.branch_currents[index].second.push_back(
          solution[branch_offset + index]);
    }
  }
  return Result<AcResult>::Ok(std::move(result));
}

[[nodiscard]] Result<std::string> FormatDcCsv(const DcResult &result) {
  std::string csv = "Variable,Value\n";
  const auto append_row = [&](std::string variable,
                              double value) -> Result<std::string> {
    auto formatted = FormatDouble(value);
    if (!formatted.ok()) {
      return Result<std::string>::Fail(formatted.error().code,
                                       formatted.error().message);
    }
    return Result<std::string>::Ok(EscapeCsvField(variable) + "," +
                                   formatted.TakeValue() + "\n");
  };

  for (const auto &[name, voltage] : result.node_voltages) {
    auto row = append_row("V(" + name + ")", voltage);
    if (!row.ok()) {
      return row;
    }
    csv += row.TakeValue();
  }
  for (const auto &[name, current] : result.branch_currents) {
    auto row = append_row("I(" + name + ")", current);
    if (!row.ok()) {
      return row;
    }
    csv += row.TakeValue();
  }
  return Result<std::string>::Ok(std::move(csv));
}

[[nodiscard]] Result<std::string> FormatAcCsv(const AcResult &result) {
  std::string csv = "Frequency";
  for (const auto &[name, unused] : result.node_voltages) {
    static_cast<void>(unused);
    csv += "," + EscapeCsvField("V(" + name + ")_mag");
    csv += "," + EscapeCsvField("V(" + name + ")_phase_deg");
  }
  for (const auto &[name, unused] : result.branch_currents) {
    static_cast<void>(unused);
    csv += "," + EscapeCsvField("I(" + name + ")_mag");
    csv += "," + EscapeCsvField("I(" + name + ")_phase_deg");
  }
  csv += "\n";

  for (std::size_t frequency_index = 0;
       frequency_index < result.frequencies_hz.size(); ++frequency_index) {
    auto frequency = FormatDouble(result.frequencies_hz[frequency_index]);
    if (!frequency.ok()) {
      return frequency;
    }
    csv += frequency.TakeValue();

    const auto append_complex =
        [&](std::complex<double> value) -> Result<std::string> {
      const double magnitude = std::abs(value);
      const double phase_degrees =
          magnitude == 0.0 ? 0.0 : std::arg(value) * 180.0 / std::numbers::pi;
      auto formatted_magnitude = FormatDouble(magnitude);
      if (!formatted_magnitude.ok()) {
        return formatted_magnitude;
      }
      auto formatted_phase = FormatDouble(phase_degrees);
      if (!formatted_phase.ok()) {
        return formatted_phase;
      }
      return Result<std::string>::Ok("," + formatted_magnitude.TakeValue() +
                                     "," + formatted_phase.TakeValue());
    };

    for (const auto &[unused, values] : result.node_voltages) {
      static_cast<void>(unused);
      auto columns = append_complex(values[frequency_index]);
      if (!columns.ok()) {
        return columns;
      }
      csv += columns.TakeValue();
    }
    for (const auto &[unused, values] : result.branch_currents) {
      static_cast<void>(unused);
      auto columns = append_complex(values[frequency_index]);
      if (!columns.ok()) {
        return columns;
      }
      csv += columns.TakeValue();
    }
    csv += "\n";
  }
  return Result<std::string>::Ok(std::move(csv));
}

} // namespace

Result<std::vector<double>> GenerateAcFrequencies(const AcAnalysis &analysis) {
  switch (analysis.sweep_type) {
  case AcSweepType::kDec:
  case AcSweepType::kOct:
  case AcSweepType::kLin:
    break;
  default:
    return Result<std::vector<double>>::Fail(ErrorCode::kParse,
                                             ".AC sweep type is invalid");
  }
  if (analysis.points == 0) {
    return Result<std::vector<double>>::Fail(
        ErrorCode::kParse, ".AC points must be a positive integer");
  }
  if (analysis.sweep_type == AcSweepType::kLin && analysis.points < 2) {
    return Result<std::vector<double>>::Fail(
        ErrorCode::kParse,
        ".AC LIN requires at least two points to include both endpoints");
  }
  if (!std::isfinite(analysis.start_frequency_hz) ||
      !std::isfinite(analysis.stop_frequency_hz) ||
      analysis.start_frequency_hz <= 0.0 ||
      analysis.stop_frequency_hz <= analysis.start_frequency_hz) {
    return Result<std::vector<double>>::Fail(
        ErrorCode::kParse,
        ".AC requires finite positive frequencies with stop greater than "
        "start");
  }

  if (analysis.sweep_type == AcSweepType::kLin) {
    if (analysis.points > kMaxAcFrequencyPoints) {
      return Result<std::vector<double>>::Fail(
          ErrorCode::kParse, ".AC sweep exceeds the 1000000-point limit");
    }
    std::vector<double> frequencies;
    frequencies.reserve(analysis.points);
    frequencies.push_back(analysis.start_frequency_hz);
    const long double start = analysis.start_frequency_hz;
    const long double span =
        static_cast<long double>(analysis.stop_frequency_hz) - start;
    for (std::size_t index = 1; index + 1 < analysis.points; ++index) {
      const long double fraction =
          static_cast<long double>(index) /
          static_cast<long double>(analysis.points - 1);
      const double frequency = static_cast<double>(start + span * fraction);
      if (!std::isfinite(frequency) || frequency <= frequencies.back() ||
          frequency >= analysis.stop_frequency_hz) {
        return Result<std::vector<double>>::Fail(
            ErrorCode::kParse,
            ".AC LIN frequencies cannot be represented as a strictly "
            "increasing FP64 grid");
      }
      frequencies.push_back(frequency);
    }
    frequencies.push_back(analysis.stop_frequency_hz);
    return Result<std::vector<double>>::Ok(std::move(frequencies));
  }

  const long double ratio =
      static_cast<long double>(analysis.stop_frequency_hz) /
      static_cast<long double>(analysis.start_frequency_hz);
  const long double interval_count = analysis.sweep_type == AcSweepType::kDec
                                         ? std::log10(ratio)
                                         : std::log2(ratio);
  const long double raw_steps =
      interval_count * static_cast<long double>(analysis.points);
  if (!std::isfinite(raw_steps)) {
    return Result<std::vector<double>>::Fail(
        ErrorCode::kParse, ".AC sweep exceeds the 1000000-point limit");
  }
  // Exact decade/octave boundaries can land a few ULPs above an integer after
  // logarithm evaluation. Snap only that arithmetic noise before applying the
  // documented ceiling contract.
  const long double nearest_steps = std::round(raw_steps);
  const long double tolerance =
      16.0L * std::numeric_limits<long double>::epsilon() *
      (std::abs(raw_steps) < 1.0L ? 1.0L : std::abs(raw_steps));
  const long double base =
      analysis.sweep_type == AcSweepType::kDec ? 10.0L : 2.0L;
  const double reconstructed_stop = static_cast<double>(
      static_cast<long double>(analysis.start_frequency_hz) *
      std::pow(base,
               nearest_steps / static_cast<long double>(analysis.points)));
  const long double normalized_steps =
      nearest_steps >= 1.0L &&
              std::abs(raw_steps - nearest_steps) <= tolerance &&
              reconstructed_stop == analysis.stop_frequency_hz
          ? nearest_steps
          : raw_steps;
  if (normalized_steps > static_cast<long double>(kMaxAcFrequencyPoints - 1)) {
    return Result<std::vector<double>>::Fail(
        ErrorCode::kParse, ".AC sweep exceeds the 1000000-point limit");
  }
  const std::size_t steps =
      static_cast<std::size_t>(std::ceil(normalized_steps));
  std::vector<double> frequencies;
  frequencies.reserve(steps + 1);
  for (std::size_t index = 0; index < steps; ++index) {
    const long double exponent = static_cast<long double>(index) /
                                 static_cast<long double>(analysis.points);
    const double frequency = static_cast<double>(
        static_cast<long double>(analysis.start_frequency_hz) *
        std::pow(base, exponent));
    if (!std::isfinite(frequency)) {
      return Result<std::vector<double>>::Fail(
          ErrorCode::kParse, ".AC generated a non-finite frequency");
    }
    if (frequency >= analysis.stop_frequency_hz ||
        (!frequencies.empty() && frequency <= frequencies.back())) {
      return Result<std::vector<double>>::Fail(
          ErrorCode::kParse,
          ".AC logarithmic frequencies cannot be represented as a strictly "
          "increasing FP64 grid");
    }
    frequencies.push_back(frequency);
  }
  if (frequencies.size() != steps || frequencies.empty() ||
      frequencies.back() >= analysis.stop_frequency_hz) {
    return Result<std::vector<double>>::Fail(
        ErrorCode::kParse,
        ".AC logarithmic frequencies cannot be represented as a strictly "
        "increasing FP64 grid");
  }
  frequencies.push_back(analysis.stop_frequency_hz);
  return Result<std::vector<double>>::Ok(std::move(frequencies));
}

Result<DcResult> SimulateDc(std::string_view netlist) {
  auto parsed = ParseNetlist(netlist);
  if (!parsed.ok()) {
    return Result<DcResult>::Fail(parsed.error().code, parsed.error().message);
  }
  Circuit circuit = parsed.TakeValue();
  bool has_dc = false;
  for (const Analysis &analysis : circuit.analyses) {
    has_dc = has_dc || std::holds_alternative<DcAnalysis>(analysis);
  }
  if (!has_dc) {
    return Result<DcResult>::Fail(
        ErrorCode::kUnsupported,
        "DC simulation requires a .DC or .OP analysis");
  }

  auto compiled = CompileMna(circuit);
  if (!compiled.ok()) {
    return Result<DcResult>::Fail(compiled.error().code,
                                  compiled.error().message);
  }
  return RunDc(compiled.value());
}

Result<std::string> SimulateDcToCsv(std::string_view netlist) {
  auto simulated = SimulateDc(netlist);
  if (!simulated.ok()) {
    return Result<std::string>::Fail(simulated.error().code,
                                     simulated.error().message);
  }
  return FormatDcCsv(simulated.value());
}

Result<AcResult> SimulateAc(std::string_view netlist) {
  auto parsed = ParseNetlist(netlist);
  if (!parsed.ok()) {
    return Result<AcResult>::Fail(parsed.error().code, parsed.error().message);
  }
  Circuit circuit = parsed.TakeValue();
  const AcAnalysis *requested_analysis = nullptr;
  for (const Analysis &analysis : circuit.analyses) {
    if (const auto *ac = std::get_if<AcAnalysis>(&analysis)) {
      requested_analysis = ac;
      break;
    }
  }
  if (requested_analysis == nullptr) {
    return Result<AcResult>::Fail(ErrorCode::kUnsupported,
                                  "AC simulation requires a .AC analysis");
  }

  auto compiled = CompileMna(circuit);
  if (!compiled.ok()) {
    return Result<AcResult>::Fail(compiled.error().code,
                                  compiled.error().message);
  }
  return RunAc(compiled.value(), *requested_analysis);
}

Result<std::string> SimulateAcToCsv(std::string_view netlist) {
  auto simulated = SimulateAc(netlist);
  if (!simulated.ok()) {
    return Result<std::string>::Fail(simulated.error().code,
                                     simulated.error().message);
  }
  return FormatAcCsv(simulated.value());
}

Result<std::string> SimulateToCsv(std::string_view netlist) {
  auto parsed = ParseNetlist(netlist);
  if (!parsed.ok()) {
    return Result<std::string>::Fail(parsed.error().code,
                                     parsed.error().message);
  }
  Circuit circuit = parsed.TakeValue();
  if (circuit.analyses.empty()) {
    return Result<std::string>::Fail(
        ErrorCode::kUnsupported,
        "simulation requires a .DC, .OP, or .AC analysis");
  }

  auto compiled = CompileMna(circuit);
  if (!compiled.ok()) {
    return Result<std::string>::Fail(compiled.error().code,
                                     compiled.error().message);
  }

  std::string csv;
  for (const Analysis &analysis : circuit.analyses) {
    Result<std::string> table = [&]() {
      if (std::holds_alternative<DcAnalysis>(analysis)) {
        auto simulated = RunDc(compiled.value());
        if (!simulated.ok()) {
          return Result<std::string>::Fail(simulated.error().code,
                                           simulated.error().message);
        }
        return FormatDcCsv(simulated.value());
      }
      auto simulated = RunAc(compiled.value(), std::get<AcAnalysis>(analysis));
      if (!simulated.ok()) {
        return Result<std::string>::Fail(simulated.error().code,
                                         simulated.error().message);
      }
      return FormatAcCsv(simulated.value());
    }();
    if (!table.ok()) {
      return table;
    }
    csv += table.TakeValue();
  }
  return Result<std::string>::Ok(std::move(csv));
}

} // namespace ohmnivore
