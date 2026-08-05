#include "ohmnivore/parser.h"

#include <charconv>
#include <cmath>
#include <cstddef>
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
  double value = 0.0;
  const char *begin = token.data();
  const char *end = token.data() + token.size();
  const auto parsed =
      std::from_chars(begin, end, value, std::chars_format::general);
  if (parsed.ec != std::errc{} || parsed.ptr == begin) {
    return Result<double>::Fail(ErrorCode::kParse,
                                "invalid engineering value '" +
                                    std::string(token) + "'");
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
    return Result<double>::Fail(ErrorCode::kParse,
                                "unknown engineering suffix in '" +
                                    std::string(token) + "'");
  }

  value *= multiplier;
  if (!std::isfinite(value)) {
    return Result<double>::Fail(ErrorCode::kParse,
                                "non-finite engineering value '" +
                                    std::string(token) + "'");
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
  return upper == "AC" || is_waveform("PULSE") || is_waveform("SIN") ||
         is_waveform("PWL") || is_waveform("EXP");
}

[[nodiscard]] Result<double>
ParseDcSourceValue(const std::vector<std::string> &tokens,
                   std::string_view source_kind) {
  for (std::size_t index = 3; index < tokens.size(); ++index) {
    if (IsUnsupportedSourceToken(tokens[index])) {
      return Result<double>::Fail(ErrorCode::kUnsupported,
                                  "phase 2A " + std::string(source_kind) +
                                      " sources support DC values only");
    }
  }

  const std::string syntax = std::string(source_kind) + "-source syntax is: " +
                             (source_kind == "voltage" ? "V" : "I") +
                             "name n+ n- [DC] value";
  if (tokens.size() != 4 && tokens.size() != 5) {
    return Result<double>::Fail(ErrorCode::kParse, syntax);
  }

  std::size_t value_index = 3;
  if (tokens.size() == 5) {
    if (Upper(tokens[3]) != "DC") {
      return Result<double>::Fail(ErrorCode::kParse, syntax);
    }
    value_index = 4;
  } else if (Upper(tokens[3]) == "DC") {
    return Result<double>::Fail(ErrorCode::kParse, syntax);
  }

  auto parsed = ParseEngineeringValue(tokens[value_index]);
  if (!parsed.ok()) {
    return Result<double>::Fail(parsed.error().code, parsed.error().message);
  }
  return parsed;
}

[[nodiscard]] Result<Component>
ParseVoltageSource(const std::vector<std::string> &tokens) {
  auto dc_value = ParseDcSourceValue(tokens, "voltage");
  if (!dc_value.ok()) {
    return Result<Component>::Fail(dc_value.error().code,
                                   dc_value.error().message);
  }
  return Result<Component>::Ok(VoltageSource{
      .name = tokens[0],
      .positive_node = tokens[1],
      .negative_node = tokens[2],
      .dc_volts = dc_value.value(),
  });
}

[[nodiscard]] Result<Component>
ParseCurrentSource(const std::vector<std::string> &tokens) {
  auto dc_value = ParseDcSourceValue(tokens, "current");
  if (!dc_value.ok()) {
    return Result<Component>::Fail(dc_value.error().code,
                                   dc_value.error().message);
  }
  return Result<Component>::Ok(CurrentSource{
      .name = tokens[0],
      .positive_node = tokens[1],
      .negative_node = tokens[2],
      .dc_amperes = dc_value.value(),
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
      circuit.analyses.push_back(Analysis::kDc);
      continue;
    }
    // The CLI emits every solved variable, so the legacy .PRINT selection is
    // recognized as a compatibility no-op rather than silently interpreted.
    if (upper.starts_with(".PRINT ")) {
      continue;
    }
    if (line.front() == '.') {
      return Result<Circuit>::Fail(
          ErrorCode::kUnsupported,
          WithLine(line_number, "phase 2A does not support directive '" +
                                    std::string(line) + "'"));
    }

    const std::vector<std::string> tokens = SplitWhitespace(line);
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
          "phase 2A supports RLC elements and independent DC voltage/current "
          "sources only");
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
