#include "ohmnivore/parser.h"

#include <charconv>
#include <cmath>
#include <cstddef>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace ohmnivore {
namespace {

[[nodiscard]] std::string_view Trim(std::string_view value) {
  const std::size_t first = value.find_first_not_of(" \t\r\n");
  if (first == std::string_view::npos) {
    return {};
  }
  const std::size_t last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

[[nodiscard]] std::string Upper(std::string_view value) {
  std::string upper(value);
  for (char &character : upper) {
    if (character >= 'a' && character <= 'z') {
      character = static_cast<char>(character - 'a' + 'A');
    }
  }
  return upper;
}

[[nodiscard]] std::vector<std::string> SplitWhitespace(std::string_view line) {
  std::istringstream stream{std::string(line)};
  std::vector<std::string> tokens;
  for (std::string token; stream >> token;) {
    tokens.push_back(std::move(token));
  }
  return tokens;
}

[[nodiscard]] Result<double> ParseEngineeringValue(std::string_view token) {
  const std::string original(token);
  if (!token.empty() && token.front() == '+') {
    if (token.size() == 1 || token[1] == '+' || token[1] == '-') {
      return Result<double>::Fail(
          ErrorCode::kParse, "invalid engineering value '" + original + "'");
    }
    token.remove_prefix(1);
  }
  if (token.empty()) {
    return Result<double>::Fail(ErrorCode::kParse,
                                "invalid engineering value '" + original + "'");
  }

  double value = 0.0;
  const char *begin = token.data();
  const char *end = token.data() + token.size();
  const auto parsed =
      std::from_chars(begin, end, value, std::chars_format::general);
  if (parsed.ec != std::errc{} || parsed.ptr == begin) {
    return Result<double>::Fail(ErrorCode::kParse,
                                "invalid engineering value '" + original + "'");
  }

  const std::string suffix = Upper(std::string_view(parsed.ptr, end));
  double multiplier = 1.0;
  if (suffix == "T") {
    multiplier = 1e12;
  } else if (suffix == "G") {
    multiplier = 1e9;
  } else if (suffix == "MEG") {
    multiplier = 1e6;
  } else if (suffix == "K") {
    multiplier = 1e3;
  } else if (suffix == "M") {
    multiplier = 1e-3;
  } else if (suffix == "U") {
    multiplier = 1e-6;
  } else if (suffix == "N") {
    multiplier = 1e-9;
  } else if (suffix == "P") {
    multiplier = 1e-12;
  } else if (suffix == "F") {
    multiplier = 1e-15;
  } else if (!suffix.empty()) {
    return Result<double>::Fail(
        ErrorCode::kParse, "unknown engineering suffix in '" + original + "'");
  }

  value *= multiplier;
  if (!std::isfinite(value)) {
    return Result<double>::Fail(
        ErrorCode::kParse, "non-finite engineering value '" + original + "'");
  }
  return Result<double>::Ok(value);
}

[[nodiscard]] Result<Component>
ParseResistor(const std::vector<std::string> &tokens) {
  if (tokens.size() != 4) {
    return Result<Component>::Fail(
        ErrorCode::kParse, "resistor syntax is: Rname n+ n- resistance");
  }
  auto resistance = ParseEngineeringValue(tokens[3]);
  if (!resistance.ok()) {
    return Result<Component>::Fail(resistance.error().code,
                                   resistance.error().message);
  }
  if (resistance.value() <= 0.0) {
    return Result<Component>::Fail(ErrorCode::kParse,
                                   "resistance must be greater than zero");
  }
  return Result<Component>::Ok(Resistor{
      .name = tokens[0],
      .positive_node = tokens[1],
      .negative_node = tokens[2],
      .resistance_ohms = resistance.value(),
  });
}

[[nodiscard]] Result<Component>
ParseCapacitor(const std::vector<std::string> &tokens) {
  if (tokens.size() != 4) {
    return Result<Component>::Fail(
        ErrorCode::kParse, "capacitor syntax is: Cname n+ n- capacitance");
  }
  auto capacitance = ParseEngineeringValue(tokens[3]);
  if (!capacitance.ok()) {
    return Result<Component>::Fail(capacitance.error().code,
                                   capacitance.error().message);
  }
  if (capacitance.value() <= 0.0) {
    return Result<Component>::Fail(ErrorCode::kParse,
                                   "capacitance must be greater than zero");
  }
  return Result<Component>::Ok(Capacitor{
      .name = tokens[0],
      .positive_node = tokens[1],
      .negative_node = tokens[2],
      .capacitance_farads = capacitance.value(),
  });
}

[[nodiscard]] Result<Component>
ParseInductor(const std::vector<std::string> &tokens) {
  if (tokens.size() != 4) {
    return Result<Component>::Fail(
        ErrorCode::kParse, "inductor syntax is: Lname n+ n- inductance");
  }
  auto inductance = ParseEngineeringValue(tokens[3]);
  if (!inductance.ok()) {
    return Result<Component>::Fail(inductance.error().code,
                                   inductance.error().message);
  }
  if (inductance.value() <= 0.0) {
    return Result<Component>::Fail(ErrorCode::kParse,
                                   "inductance must be greater than zero");
  }
  return Result<Component>::Ok(Inductor{
      .name = tokens[0],
      .positive_node = tokens[1],
      .negative_node = tokens[2],
      .inductance_henries = inductance.value(),
  });
}

[[nodiscard]] bool IsUnsupportedSourceToken(std::string_view token) {
  const std::string upper = Upper(token);
  const auto is_waveform = [&](std::string_view keyword) {
    return upper == keyword || upper.starts_with(std::string(keyword) + "(");
  };
  return is_waveform("PULSE") || is_waveform("SIN") || is_waveform("PWL") ||
         is_waveform("EXP");
}

struct SourceSpecifications {
  std::optional<double> dc;
  std::optional<AcSourceSpecification> ac;
};

[[nodiscard]] Result<SourceSpecifications>
ParseSourceSpecifications(const std::vector<std::string> &tokens,
                          std::string_view source_kind) {
  for (std::size_t index = 3; index < tokens.size(); ++index) {
    if (IsUnsupportedSourceToken(tokens[index])) {
      return Result<SourceSpecifications>::Fail(
          ErrorCode::kUnsupported, "phase 2B does not support transient " +
                                       std::string(source_kind) +
                                       "-source waveforms");
    }
  }

  const std::string syntax =
      std::string(source_kind) +
      "-source syntax is: " + (source_kind == "voltage" ? "V" : "I") +
      "name n+ n- ([DC] value [AC magnitude [phase_degrees]] | AC "
      "magnitude [phase_degrees])";
  if (tokens.size() <= 3) {
    return Result<SourceSpecifications>::Fail(
        ErrorCode::kParse, std::string(source_kind) +
                               " source requires a DC and/or AC specification");
  }

  SourceSpecifications specifications;
  std::size_t index = 3;
  if (Upper(tokens[index]) == "DC") {
    ++index;
    if (index >= tokens.size()) {
      return Result<SourceSpecifications>::Fail(ErrorCode::kParse, syntax);
    }
    auto dc = ParseEngineeringValue(tokens[index]);
    if (!dc.ok()) {
      return Result<SourceSpecifications>::Fail(dc.error().code,
                                                dc.error().message);
    }
    specifications.dc = dc.value();
    ++index;
  } else if (Upper(tokens[index]) != "AC") {
    auto dc = ParseEngineeringValue(tokens[index]);
    if (!dc.ok()) {
      return Result<SourceSpecifications>::Fail(dc.error().code,
                                                dc.error().message);
    }
    specifications.dc = dc.value();
    ++index;
  }

  if (index < tokens.size() && Upper(tokens[index]) == "AC") {
    ++index;
    if (index >= tokens.size()) {
      return Result<SourceSpecifications>::Fail(ErrorCode::kParse, syntax);
    }
    auto magnitude = ParseEngineeringValue(tokens[index]);
    if (!magnitude.ok()) {
      return Result<SourceSpecifications>::Fail(magnitude.error().code,
                                                magnitude.error().message);
    }
    if (magnitude.value() < 0.0) {
      return Result<SourceSpecifications>::Fail(
          ErrorCode::kParse, "AC source magnitude must not be negative");
    }
    ++index;

    double phase_degrees = 0.0;
    if (index < tokens.size()) {
      auto phase = ParseEngineeringValue(tokens[index]);
      if (!phase.ok()) {
        return Result<SourceSpecifications>::Fail(phase.error().code,
                                                  phase.error().message);
      }
      phase_degrees = phase.value();
      ++index;
    }
    specifications.ac = AcSourceSpecification{.magnitude = magnitude.value(),
                                              .phase_degrees = phase_degrees};
  }

  if (index != tokens.size()) {
    return Result<SourceSpecifications>::Fail(ErrorCode::kParse, syntax);
  }
  if (!specifications.dc.has_value() && !specifications.ac.has_value()) {
    return Result<SourceSpecifications>::Fail(
        ErrorCode::kParse, std::string(source_kind) +
                               " source requires a DC and/or AC specification");
  }
  return Result<SourceSpecifications>::Ok(std::move(specifications));
}

[[nodiscard]] Result<Component>
ParseVoltageSource(const std::vector<std::string> &tokens) {
  auto specifications = ParseSourceSpecifications(tokens, "voltage");
  if (!specifications.ok()) {
    return Result<Component>::Fail(specifications.error().code,
                                   specifications.error().message);
  }
  return Result<Component>::Ok(VoltageSource{
      .name = tokens[0],
      .positive_node = tokens[1],
      .negative_node = tokens[2],
      .dc_volts = specifications.value().dc,
      .ac = specifications.value().ac,
  });
}

[[nodiscard]] Result<Component>
ParseCurrentSource(const std::vector<std::string> &tokens) {
  auto specifications = ParseSourceSpecifications(tokens, "current");
  if (!specifications.ok()) {
    return Result<Component>::Fail(specifications.error().code,
                                   specifications.error().message);
  }
  return Result<Component>::Ok(CurrentSource{
      .name = tokens[0],
      .positive_node = tokens[1],
      .negative_node = tokens[2],
      .dc_amperes = specifications.value().dc,
      .ac = specifications.value().ac,
  });
}

[[nodiscard]] Result<Analysis>
ParseAcAnalysis(const std::vector<std::string> &tokens) {
  constexpr std::string_view kSyntax =
      ".AC syntax is: .AC DEC|OCT|LIN points start_frequency "
      "stop_frequency";
  if (tokens.size() != 5) {
    return Result<Analysis>::Fail(ErrorCode::kParse, std::string(kSyntax));
  }

  AcSweepType sweep_type;
  const std::string sweep = Upper(tokens[1]);
  if (sweep == "DEC") {
    sweep_type = AcSweepType::kDec;
  } else if (sweep == "OCT") {
    sweep_type = AcSweepType::kOct;
  } else if (sweep == "LIN") {
    sweep_type = AcSweepType::kLin;
  } else {
    return Result<Analysis>::Fail(ErrorCode::kParse,
                                  ".AC sweep type must be DEC, OCT, or LIN");
  }

  std::size_t points = 0;
  const char *points_begin = tokens[2].data();
  const char *points_end = points_begin + tokens[2].size();
  const auto parsed_points = std::from_chars(points_begin, points_end, points);
  if (parsed_points.ec != std::errc{} || parsed_points.ptr != points_end ||
      points == 0) {
    return Result<Analysis>::Fail(ErrorCode::kParse,
                                  ".AC points must be a positive integer");
  }
  if (sweep_type == AcSweepType::kLin && points < 2) {
    return Result<Analysis>::Fail(
        ErrorCode::kParse,
        ".AC LIN requires at least two points to include both endpoints");
  }

  auto start = ParseEngineeringValue(tokens[3]);
  if (!start.ok()) {
    return Result<Analysis>::Fail(start.error().code, start.error().message);
  }
  auto stop = ParseEngineeringValue(tokens[4]);
  if (!stop.ok()) {
    return Result<Analysis>::Fail(stop.error().code, stop.error().message);
  }
  if (start.value() <= 0.0 || stop.value() <= 0.0) {
    return Result<Analysis>::Fail(ErrorCode::kParse,
                                  ".AC frequencies must be greater than zero");
  }
  if (stop.value() <= start.value()) {
    return Result<Analysis>::Fail(
        ErrorCode::kParse,
        ".AC stop frequency must be greater than start frequency");
  }

  return Result<Analysis>::Ok(AcAnalysis{
      .sweep_type = sweep_type,
      .points = points,
      .start_frequency_hz = start.value(),
      .stop_frequency_hz = stop.value(),
  });
}

[[nodiscard]] std::string WithLine(std::size_t line_number,
                                   std::string message) {
  return "line " + std::to_string(line_number) + ": " + std::move(message);
}

} // namespace

Result<Circuit> ParseNetlist(std::string_view input) {
  Circuit circuit;
  std::istringstream lines{std::string(input)};
  std::string raw_line;
  std::size_t line_number = 0;

  while (std::getline(lines, raw_line)) {
    ++line_number;
    const std::string_view line = Trim(raw_line);
    if (line.empty() || line.front() == '*') {
      continue;
    }

    const std::string upper = Upper(line);
    if (upper == ".END") {
      break;
    }
    if (upper == ".DC" || upper == ".OP") {
      circuit.analyses.push_back(DcAnalysis{});
      continue;
    }
    // The CLI emits every solved variable, so the legacy .PRINT selection is
    // recognized as a compatibility no-op rather than silently interpreted.
    if (upper.starts_with(".PRINT ")) {
      continue;
    }
    const std::vector<std::string> tokens = SplitWhitespace(line);
    if (!tokens.empty() && Upper(tokens.front()) == ".AC") {
      auto analysis = ParseAcAnalysis(tokens);
      if (!analysis.ok()) {
        return Result<Circuit>::Fail(
            analysis.error().code,
            WithLine(line_number, analysis.error().message));
      }
      circuit.analyses.push_back(analysis.TakeValue());
      continue;
    }
    if (line.front() == '.') {
      return Result<Circuit>::Fail(
          ErrorCode::kUnsupported,
          WithLine(line_number, "phase 2B does not support directive '" +
                                    std::string(line) + "'"));
    }

    if (tokens.empty()) {
      continue;
    }

    Result<Component> component = [&]() {
      const char kind = Upper(tokens.front()).front();
      if (kind == 'R') {
        return ParseResistor(tokens);
      }
      if (kind == 'C') {
        return ParseCapacitor(tokens);
      }
      if (kind == 'L') {
        return ParseInductor(tokens);
      }
      if (kind == 'V') {
        return ParseVoltageSource(tokens);
      }
      if (kind == 'I') {
        return ParseCurrentSource(tokens);
      }
      return Result<Component>::Fail(
          ErrorCode::kUnsupported,
          "phase 2B supports RLC elements and independent DC/AC "
          "voltage/current sources only");
    }();
    if (!component.ok()) {
      return Result<Circuit>::Fail(
          component.error().code,
          WithLine(line_number, component.error().message));
    }
    circuit.components.push_back(component.TakeValue());
  }

  return Result<Circuit>::Ok(std::move(circuit));
}

} // namespace ohmnivore
