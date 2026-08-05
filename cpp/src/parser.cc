#include "ohmnivore/parser.h"

#include <charconv>
#include <cmath>
#include <cstddef>
#include <limits>
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

struct SourceSpecifications {
  std::optional<double> dc;
  std::optional<AcSourceSpecification> ac;
  std::optional<TransientWaveform> transient;
};

[[nodiscard]] bool IsAsciiWhitespace(char character) {
  return character == ' ' || character == '\t' || character == '\r' ||
         character == '\n';
}

[[nodiscard]] bool StartsWithCaseInsensitive(std::string_view input,
                                             std::string_view prefix) {
  if (input.size() < prefix.size()) {
    return false;
  }
  return Upper(input.substr(0, prefix.size())) == prefix;
}

[[nodiscard]] bool StartsWithKeywordToken(std::string_view input,
                                          std::string_view keyword) {
  input = Trim(input);
  if (!StartsWithCaseInsensitive(input, keyword)) {
    return false;
  }
  return input.size() == keyword.size() ||
         IsAsciiWhitespace(input[keyword.size()]) ||
         input[keyword.size()] == '(';
}

[[nodiscard]] bool StartsWithWaveformKeyword(std::string_view input) {
  return StartsWithKeywordToken(input, "PULSE") ||
         StartsWithKeywordToken(input, "SIN") ||
         StartsWithKeywordToken(input, "PWL") ||
         StartsWithKeywordToken(input, "EXP");
}

[[nodiscard]] bool ConsumeKeyword(std::string_view keyword,
                                  std::string_view *input) {
  *input = Trim(*input);
  if (!StartsWithCaseInsensitive(*input, keyword)) {
    return false;
  }
  if (input->size() != keyword.size() &&
      !IsAsciiWhitespace((*input)[keyword.size()])) {
    return false;
  }
  input->remove_prefix(keyword.size());
  *input = Trim(*input);
  return true;
}

[[nodiscard]] std::optional<std::string_view>
TakeWhitespaceToken(std::string_view *input) {
  *input = Trim(*input);
  if (input->empty()) {
    return std::nullopt;
  }
  std::size_t length = 0;
  while (length < input->size() && !IsAsciiWhitespace((*input)[length])) {
    ++length;
  }
  const std::string_view token = input->substr(0, length);
  input->remove_prefix(length);
  *input = Trim(*input);
  return token;
}

[[nodiscard]] Result<double> TakeEngineeringValue(std::string_view *input,
                                                  std::string_view syntax) {
  const auto token = TakeWhitespaceToken(input);
  if (!token.has_value()) {
    return Result<double>::Fail(ErrorCode::kParse, std::string(syntax));
  }
  return ParseEngineeringValue(*token);
}

[[nodiscard]] Result<std::vector<double>>
ParseWaveformArguments(std::string_view input) {
  std::vector<double> values;
  input = Trim(input);
  while (!input.empty()) {
    if (input.front() == ',') {
      return Result<std::vector<double>>::Fail(
          ErrorCode::kParse, "waveform contains an empty parameter");
    }

    std::size_t length = 0;
    while (length < input.size() && input[length] != ',' &&
           !IsAsciiWhitespace(input[length])) {
      ++length;
    }
    auto value = ParseEngineeringValue(input.substr(0, length));
    if (!value.ok()) {
      return Result<std::vector<double>>::Fail(value.error().code,
                                               value.error().message);
    }
    values.push_back(value.value());
    input.remove_prefix(length);

    bool saw_comma = false;
    while (!input.empty() && IsAsciiWhitespace(input.front())) {
      input.remove_prefix(1);
    }
    if (!input.empty() && input.front() == ',') {
      saw_comma = true;
      input.remove_prefix(1);
      while (!input.empty() && IsAsciiWhitespace(input.front())) {
        input.remove_prefix(1);
      }
    }
    if (saw_comma && (input.empty() || input.front() == ',')) {
      return Result<std::vector<double>>::Fail(
          ErrorCode::kParse, "waveform contains an empty parameter");
    }
  }
  return Result<std::vector<double>>::Ok(std::move(values));
}

[[nodiscard]] Result<TransientWaveform>
ParseTransientWaveform(std::string_view input) {
  input = Trim(input);
  std::string_view keyword;
  if (StartsWithKeywordToken(input, "PULSE")) {
    keyword = "PULSE";
  } else if (StartsWithKeywordToken(input, "SIN")) {
    keyword = "SIN";
  } else if (StartsWithKeywordToken(input, "PWL")) {
    keyword = "PWL";
  } else if (StartsWithKeywordToken(input, "EXP")) {
    keyword = "EXP";
  } else {
    return Result<TransientWaveform>::Fail(
        ErrorCode::kParse,
        "expected one PULSE, SIN, PWL, or EXP waveform at the end of the "
        "source specification");
  }

  if (input.size() <= keyword.size() || input[keyword.size()] != '(') {
    return Result<TransientWaveform>::Fail(ErrorCode::kParse,
                                           "expected '(' immediately after " +
                                               std::string(keyword));
  }
  const std::size_t close = input.find(')', keyword.size() + 1);
  if (close == std::string_view::npos) {
    return Result<TransientWaveform>::Fail(
        ErrorCode::kParse, "expected ')' to close " + std::string(keyword));
  }
  const std::string_view inner =
      input.substr(keyword.size() + 1, close - keyword.size() - 1);
  if (inner.find_first_of("()") != std::string_view::npos) {
    return Result<TransientWaveform>::Fail(
        ErrorCode::kParse, "nested waveform parentheses are not allowed");
  }
  if (!Trim(input.substr(close + 1)).empty()) {
    return Result<TransientWaveform>::Fail(
        ErrorCode::kParse, "trailing text after transient waveform");
  }

  auto parsed_values = ParseWaveformArguments(inner);
  if (!parsed_values.ok()) {
    return Result<TransientWaveform>::Fail(parsed_values.error().code,
                                           parsed_values.error().message);
  }
  const std::vector<double> &values = parsed_values.value();
  const double maximum = std::numeric_limits<double>::max();
  if (keyword == "PULSE") {
    if (values.size() < 2 || values.size() > 7) {
      return Result<TransientWaveform>::Fail(
          ErrorCode::kParse, "PULSE requires between two and seven parameters");
    }
    const double delay = values.size() > 2 ? values[2] : 0.0;
    const double rise = values.size() > 3 ? values[3] : 0.0;
    const double fall = values.size() > 4 ? values[4] : 0.0;
    const double width = values.size() > 5 ? values[5] : maximum;
    const double period = values.size() > 6 ? values[6] : maximum;
    if (delay < 0.0 || rise < 0.0 || fall < 0.0 || width < 0.0) {
      return Result<TransientWaveform>::Fail(
          ErrorCode::kParse, "PULSE delay and durations must be nonnegative");
    }
    if (period <= 0.0) {
      return Result<TransientWaveform>::Fail(
          ErrorCode::kParse, "PULSE period must be greater than zero");
    }
    return Result<TransientWaveform>::Ok(PulseWaveform{
        .initial_value = values[0],
        .pulsed_value = values[1],
        .delay_seconds = delay,
        .rise_time_seconds = rise,
        .fall_time_seconds = fall,
        .pulse_width_seconds = width,
        .period_seconds = period,
    });
  }
  if (keyword == "SIN") {
    if (values.size() < 3 || values.size() > 5) {
      return Result<TransientWaveform>::Fail(
          ErrorCode::kParse, "SIN requires between three and five parameters");
    }
    const double delay = values.size() > 3 ? values[3] : 0.0;
    const double damping = values.size() > 4 ? values[4] : 0.0;
    if (values[2] < 0.0 || delay < 0.0 || damping < 0.0) {
      return Result<TransientWaveform>::Fail(
          ErrorCode::kParse,
          "SIN frequency, delay, and damping factor must be nonnegative");
    }
    return Result<TransientWaveform>::Ok(SinWaveform{
        .offset = values[0],
        .amplitude = values[1],
        .frequency_hz = values[2],
        .delay_seconds = delay,
        .damping_factor_per_second = damping,
    });
  }
  if (keyword == "PWL") {
    if (values.size() < 2 || values.size() % 2 != 0) {
      return Result<TransientWaveform>::Fail(
          ErrorCode::kParse,
          "PWL requires one or more complete time-value pairs");
    }
    PwlWaveform waveform;
    waveform.time_value_pairs.reserve(values.size() / 2);
    for (std::size_t index = 0; index < values.size(); index += 2) {
      if (values[index] < 0.0) {
        return Result<TransientWaveform>::Fail(ErrorCode::kParse,
                                               "PWL times must be nonnegative");
      }
      if (!waveform.time_value_pairs.empty() &&
          values[index] <= waveform.time_value_pairs.back().first) {
        return Result<TransientWaveform>::Fail(
            ErrorCode::kParse, "PWL times must be strictly increasing");
      }
      waveform.time_value_pairs.emplace_back(values[index], values[index + 1]);
    }
    return Result<TransientWaveform>::Ok(std::move(waveform));
  }

  if (values.size() < 2 || values.size() > 6) {
    return Result<TransientWaveform>::Fail(
        ErrorCode::kParse, "EXP requires between two and six parameters");
  }
  const double rise_delay = values.size() > 2 ? values[2] : 0.0;
  const double rise_time_constant = values.size() > 3 ? values[3] : maximum;
  const double fall_delay = values.size() > 4 ? values[4] : maximum;
  const double fall_time_constant = values.size() > 5 ? values[5] : maximum;
  if (rise_delay < 0.0 || fall_delay < 0.0) {
    return Result<TransientWaveform>::Fail(ErrorCode::kParse,
                                           "EXP delays must be nonnegative");
  }
  if (rise_time_constant <= 0.0 || fall_time_constant <= 0.0) {
    return Result<TransientWaveform>::Fail(
        ErrorCode::kParse, "EXP time constants must be greater than zero");
  }
  if (fall_delay < rise_delay) {
    return Result<TransientWaveform>::Fail(
        ErrorCode::kParse, "EXP fall delay must not precede its rise delay");
  }
  return Result<TransientWaveform>::Ok(ExpWaveform{
      .initial_value = values[0],
      .pulsed_value = values[1],
      .rise_delay_seconds = rise_delay,
      .rise_time_constant_seconds = rise_time_constant,
      .fall_delay_seconds = fall_delay,
      .fall_time_constant_seconds = fall_time_constant,
  });
}

[[nodiscard]] Result<SourceSpecifications>
ParseSourceSpecifications(std::string_view input,
                          std::string_view source_kind) {
  const std::string syntax =
      std::string(source_kind) +
      "-source syntax is: " + (source_kind == "voltage" ? "V" : "I") +
      "name n+ n- [[DC] value] [AC magnitude [phase_degrees]] "
      "[PULSE(...)|SIN(...)|PWL(...)|EXP(...)]";
  input = Trim(input);
  if (input.empty()) {
    return Result<SourceSpecifications>::Fail(
        ErrorCode::kParse, std::string(source_kind) +
                               " source requires a DC, AC, and/or transient "
                               "specification");
  }

  SourceSpecifications specifications;
  if (ConsumeKeyword("DC", &input)) {
    auto dc = TakeEngineeringValue(&input, syntax);
    if (!dc.ok()) {
      return Result<SourceSpecifications>::Fail(dc.error().code,
                                                dc.error().message);
    }
    specifications.dc = dc.value();
  } else if (!StartsWithKeywordToken(input, "AC") &&
             !StartsWithWaveformKeyword(input)) {
    auto dc = TakeEngineeringValue(&input, syntax);
    if (!dc.ok()) {
      return Result<SourceSpecifications>::Fail(dc.error().code,
                                                dc.error().message);
    }
    specifications.dc = dc.value();
  }

  if (ConsumeKeyword("AC", &input)) {
    auto magnitude = TakeEngineeringValue(&input, syntax);
    if (!magnitude.ok()) {
      return Result<SourceSpecifications>::Fail(magnitude.error().code,
                                                magnitude.error().message);
    }
    if (magnitude.value() < 0.0) {
      return Result<SourceSpecifications>::Fail(
          ErrorCode::kParse, "AC source magnitude must not be negative");
    }

    double phase_degrees = 0.0;
    if (!input.empty() && !StartsWithWaveformKeyword(input)) {
      auto phase = TakeEngineeringValue(&input, syntax);
      if (!phase.ok()) {
        return Result<SourceSpecifications>::Fail(phase.error().code,
                                                  phase.error().message);
      }
      phase_degrees = phase.value();
    }
    specifications.ac = AcSourceSpecification{.magnitude = magnitude.value(),
                                              .phase_degrees = phase_degrees};
  }

  if (!input.empty()) {
    auto waveform = ParseTransientWaveform(input);
    if (!waveform.ok()) {
      return Result<SourceSpecifications>::Fail(waveform.error().code,
                                                waveform.error().message);
    }
    specifications.transient = waveform.TakeValue();
  }
  if (!specifications.dc.has_value() && !specifications.ac.has_value() &&
      !specifications.transient.has_value()) {
    return Result<SourceSpecifications>::Fail(
        ErrorCode::kParse, std::string(source_kind) +
                               " source requires a DC, AC, and/or transient "
                               "specification");
  }
  return Result<SourceSpecifications>::Ok(std::move(specifications));
}

[[nodiscard]] Result<Component> ParseSource(std::string_view line,
                                            bool is_voltage) {
  std::string_view remaining = line;
  const auto name = TakeWhitespaceToken(&remaining);
  const auto positive_node = TakeWhitespaceToken(&remaining);
  const auto negative_node = TakeWhitespaceToken(&remaining);
  const std::string_view source_kind = is_voltage ? "voltage" : "current";
  if (!name.has_value() || !positive_node.has_value() ||
      !negative_node.has_value()) {
    return Result<Component>::Fail(
        ErrorCode::kParse, std::string(source_kind) +
                               "-source syntax requires a name and two nodes");
  }

  auto specifications = ParseSourceSpecifications(remaining, source_kind);
  if (!specifications.ok()) {
    return Result<Component>::Fail(specifications.error().code,
                                   specifications.error().message);
  }
  if (is_voltage) {
    return Result<Component>::Ok(VoltageSource{
        .name = std::string(*name),
        .positive_node = std::string(*positive_node),
        .negative_node = std::string(*negative_node),
        .dc_volts = specifications.value().dc,
        .ac = specifications.value().ac,
        .transient = specifications.value().transient,
    });
  }
  return Result<Component>::Ok(CurrentSource{
      .name = std::string(*name),
      .positive_node = std::string(*positive_node),
      .negative_node = std::string(*negative_node),
      .dc_amperes = specifications.value().dc,
      .ac = specifications.value().ac,
      .transient = specifications.value().transient,
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

[[nodiscard]] Result<Analysis>
ParseTranAnalysis(const std::vector<std::string> &tokens) {
  constexpr std::string_view kSyntax =
      ".TRAN syntax is: .TRAN tstep tstop [tstart] [UIC]";
  if (tokens.size() < 3 || tokens.size() > 5) {
    return Result<Analysis>::Fail(ErrorCode::kParse, std::string(kSyntax));
  }

  auto time_step = ParseEngineeringValue(tokens[1]);
  if (!time_step.ok()) {
    return Result<Analysis>::Fail(time_step.error().code,
                                  time_step.error().message);
  }
  auto stop_time = ParseEngineeringValue(tokens[2]);
  if (!stop_time.ok()) {
    return Result<Analysis>::Fail(stop_time.error().code,
                                  stop_time.error().message);
  }
  if (time_step.value() <= 0.0 || stop_time.value() <= 0.0) {
    return Result<Analysis>::Fail(
        ErrorCode::kParse, ".TRAN tstep and tstop must be greater than zero");
  }

  double start_time = 0.0;
  bool use_initial_conditions = false;
  if (tokens.size() >= 4) {
    if (Upper(tokens[3]) == "UIC") {
      if (tokens.size() != 4) {
        return Result<Analysis>::Fail(ErrorCode::kParse, std::string(kSyntax));
      }
      use_initial_conditions = true;
    } else {
      auto parsed_start = ParseEngineeringValue(tokens[3]);
      if (!parsed_start.ok()) {
        return Result<Analysis>::Fail(parsed_start.error().code,
                                      parsed_start.error().message);
      }
      start_time = parsed_start.value();
      if (tokens.size() == 5) {
        if (Upper(tokens[4]) != "UIC") {
          return Result<Analysis>::Fail(ErrorCode::kParse,
                                        std::string(kSyntax));
        }
        use_initial_conditions = true;
      }
    }
  }
  if (start_time < 0.0 || start_time > stop_time.value()) {
    return Result<Analysis>::Fail(
        ErrorCode::kParse,
        ".TRAN tstart must be nonnegative and no greater than tstop");
  }

  return Result<Analysis>::Ok(TranAnalysis{
      .time_step_seconds = time_step.value(),
      .stop_time_seconds = stop_time.value(),
      .start_time_seconds = start_time,
      .use_initial_conditions = use_initial_conditions,
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
    if (!tokens.empty() && Upper(tokens.front()) == ".TRAN") {
      auto analysis = ParseTranAnalysis(tokens);
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
          WithLine(line_number, "phase 2C does not support directive '" +
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
        return ParseSource(line, true);
      }
      if (kind == 'I') {
        return ParseSource(line, false);
      }
      return Result<Component>::Fail(
          ErrorCode::kUnsupported,
          "phase 2C supports RLC elements and independent DC/AC/transient "
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
