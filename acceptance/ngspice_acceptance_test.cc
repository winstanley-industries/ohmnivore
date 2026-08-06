#include <algorithm>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

struct ProcessResult {
  int exit_code = -1;
  std::string output;
};

struct Table {
  std::vector<std::string> header;
  std::vector<std::vector<double>> rows;
};

struct RawTable {
  bool complex = false;
  std::vector<std::string> names;
  std::vector<std::vector<std::complex<double>>> rows;
};

[[noreturn]] void Fail(const std::string &message) {
  throw std::runtime_error(message);
}

std::string ReadFile(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    Fail("cannot read " + path.string());
  }
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

void WriteFile(const std::filesystem::path &path, std::string_view contents) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output || !(output << contents)) {
    Fail("cannot write " + path.string());
  }
  output.close();
  if (!output) {
    Fail("cannot write " + path.string());
  }
}

std::string Trim(std::string value) {
  const std::size_t first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return {};
  }
  const std::size_t last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

double ParseDouble(std::string_view text, const std::string &context) {
  double value = 0.0;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(),
                                      value, std::chars_format::general);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
      !std::isfinite(value)) {
    Fail(context + ": invalid finite number '" + std::string(text) + "'");
  }
  return value;
}

std::size_t ParseSize(std::string_view text, const std::string &context) {
  std::size_t value = 0;
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) {
    Fail(context + ": invalid integer '" + std::string(text) + "'");
  }
  return value;
}

ProcessResult Run(const std::vector<std::string> &arguments,
                  const std::filesystem::path &temporary_directory) {
  if (arguments.empty()) {
    Fail("empty process arguments");
  }
  int descriptors[2];
  if (pipe(descriptors) != 0) {
    Fail("pipe failed");
  }
  const pid_t child = fork();
  if (child < 0) {
    close(descriptors[0]);
    close(descriptors[1]);
    Fail("fork failed");
  }
  if (child == 0) {
    if (dup2(descriptors[1], STDOUT_FILENO) < 0 ||
        dup2(descriptors[1], STDERR_FILENO) < 0) {
      _exit(126);
    }
    close(descriptors[0]);
    close(descriptors[1]);
    std::vector<char *> argv;
    argv.reserve(arguments.size() + 1);
    for (const std::string &argument : arguments) {
      argv.push_back(const_cast<char *>(argument.c_str()));
    }
    argv.push_back(nullptr);
    const std::string home = "HOME=" + (temporary_directory / "home").string();
    const std::string temp = "TMPDIR=" + temporary_directory.string();
    std::vector<std::string> environment = {"PATH=",  "LC_ALL=C", "LANG=C",
                                            "TZ=UTC", home,       temp};
    std::vector<char *> envp;
    for (std::string &entry : environment) {
      envp.push_back(entry.data());
    }
    envp.push_back(nullptr);
    execve(arguments.front().c_str(), argv.data(), envp.data());
    _exit(errno == ENOENT ? 127 : 126);
  }
  close(descriptors[1]);
  ProcessResult result;
  char buffer[8192];
  for (;;) {
    const ssize_t count = read(descriptors[0], buffer, sizeof(buffer));
    if (count > 0) {
      result.output.append(buffer, static_cast<std::size_t>(count));
    } else if (count == 0) {
      break;
    } else if (errno != EINTR) {
      close(descriptors[0]);
      Fail("read from child failed");
    }
  }
  close(descriptors[0]);
  int status = 0;
  if (waitpid(child, &status, 0) != child) {
    Fail("waitpid failed");
  }
  result.exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : 128;
  return result;
}

std::vector<std::string> ParseCsvLine(std::string_view line) {
  std::vector<std::string> fields;
  std::string field;
  bool quoted = false;
  for (std::size_t i = 0; i < line.size(); ++i) {
    const char c = line[i];
    if (quoted) {
      if (c == '"') {
        if (i + 1 < line.size() && line[i + 1] == '"') {
          field.push_back('"');
          ++i;
        } else {
          quoted = false;
        }
      } else {
        field.push_back(c);
      }
    } else if (c == ',') {
      fields.push_back(std::move(field));
      field.clear();
    } else if (c == '"') {
      if (!field.empty()) {
        Fail("malformed CSV quote");
      }
      quoted = true;
    } else {
      field.push_back(c);
    }
  }
  if (quoted) {
    Fail("unterminated CSV quote");
  }
  fields.push_back(std::move(field));
  return fields;
}

Table ParseCsv(std::string_view contents, const std::string &context) {
  std::istringstream input{std::string(contents)};
  std::string line;
  Table table;
  if (!std::getline(input, line)) {
    Fail(context + ": empty CSV");
  }
  if (!line.empty() && line.back() == '\r') {
    line.pop_back();
  }
  table.header = ParseCsvLine(line);
  while (std::getline(input, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (line.empty()) {
      continue;
    }
    const auto fields = ParseCsvLine(line);
    if (fields.size() != table.header.size()) {
      Fail(context + ": CSV row width mismatch");
    }
    std::vector<double> row;
    for (const std::string &field : fields) {
      row.push_back(ParseDouble(field, context));
    }
    table.rows.push_back(std::move(row));
  }
  if (table.header.empty() || table.rows.empty()) {
    Fail(context + ": CSV has no data");
  }
  return table;
}

void RequireExactHeader(const Table &table,
                        const std::vector<std::string> &expected,
                        const std::string &context) {
  if (table.header != expected) {
    Fail(context + ": CSV header order or schema mismatch");
  }
}

std::size_t Column(const std::vector<std::string> &header,
                   std::string_view name) {
  const auto found = std::find(header.begin(), header.end(), name);
  if (found == header.end()) {
    Fail("missing column " + std::string(name));
  }
  if (std::find(found + 1, header.end(), name) != header.end()) {
    Fail("duplicate column " + std::string(name));
  }
  return static_cast<std::size_t>(found - header.begin());
}

RawTable ParseRaw(const std::filesystem::path &path) {
  const std::string contents = ReadFile(path);
  std::istringstream lines(contents);
  std::string line;
  std::size_t variable_count = 0;
  std::size_t point_count = 0;
  bool variables = false;
  bool values = false;
  std::string value_text;
  RawTable table;
  while (std::getline(lines, line)) {
    const std::string trimmed = Trim(line);
    if (values) {
      value_text += trimmed;
      value_text.push_back(' ');
    } else if (trimmed.rfind("Flags:", 0) == 0) {
      table.complex = trimmed.find("complex") != std::string::npos;
    } else if (trimmed.rfind("No. Variables:", 0) == 0) {
      variable_count = ParseSize(Trim(trimmed.substr(14)), "raw variables");
    } else if (trimmed.rfind("No. Points:", 0) == 0) {
      point_count = ParseSize(Trim(trimmed.substr(11)), "raw points");
    } else if (trimmed == "Variables:") {
      variables = true;
    } else if (trimmed == "Values:") {
      variables = false;
      values = true;
    } else if (variables && !trimmed.empty()) {
      std::istringstream row(trimmed);
      std::size_t index = 0;
      std::string name;
      std::string type;
      if (!(row >> index >> name >> type) || index != table.names.size()) {
        Fail("malformed raw variable table");
      }
      table.names.push_back(std::move(name));
    }
  }
  if (!values || variable_count == 0 || point_count == 0 ||
      table.names.size() != variable_count) {
    Fail("incomplete ASCII raw header");
  }
  for (std::size_t index = 0; index < table.names.size(); ++index) {
    if (std::find(table.names.begin() + static_cast<std::ptrdiff_t>(index + 1),
                  table.names.end(), table.names[index]) != table.names.end()) {
      Fail("duplicate raw variable " + table.names[index]);
    }
  }
  std::replace(value_text.begin(), value_text.end(), ',', ' ');
  std::istringstream tokens(value_text);
  for (std::size_t point = 0; point < point_count; ++point) {
    std::string token;
    if (!(tokens >> token) || ParseSize(token, "raw point index") != point) {
      Fail("raw point indexes are not consecutive");
    }
    std::vector<std::complex<double>> row;
    for (std::size_t variable = 0; variable < variable_count; ++variable) {
      std::string real_text;
      std::string imaginary_text;
      if (!(tokens >> real_text)) {
        Fail("truncated raw values");
      }
      const double real = ParseDouble(real_text, "raw value");
      double imaginary = 0.0;
      if (table.complex) {
        if (!(tokens >> imaginary_text)) {
          Fail("truncated complex raw values");
        }
        imaginary = ParseDouble(imaginary_text, "raw value");
      }
      row.emplace_back(real, imaginary);
    }
    table.rows.push_back(std::move(row));
  }
  std::string trailing;
  if (tokens >> trailing) {
    Fail("trailing raw values");
  }
  return table;
}

std::filesystem::path MakeOracleNetlist(const std::filesystem::path &fixture,
                                        const std::filesystem::path &raw,
                                        bool transient) {
  std::string netlist = ReadFile(fixture);
  std::size_t end = netlist.find_last_not_of(" \t\r\n");
  if (end == std::string::npos) {
    Fail("empty fixture");
  }
  const std::size_t line = netlist.rfind('\n', end);
  const std::size_t start = line == std::string::npos ? 0 : line + 1;
  if (Trim(netlist.substr(start, end - start + 1)) != ".END") {
    Fail("fixture must end with .END");
  }
  netlist.erase(start);
  netlist += ".options numdgt=17 method=trap\n.control\n"
             "set filetype=ascii\nset numdgt=17\nrun\n";
  if (transient) {
    netlist += "linearize v(out)\n";
  }
  netlist += "write " + raw.string() + " v(out)\nquit\n.endc\n.END\n";
  const std::filesystem::path generated = raw.string() + ".spice";
  WriteFile(generated, netlist);
  return generated;
}

void RequireSuccess(const ProcessResult &result, const std::string &context) {
  if (result.exit_code != 0) {
    Fail(context + " failed with exit " + std::to_string(result.exit_code) +
         ":\n" + result.output);
  }
}

std::uint16_t ReadU16(const std::string &bytes, std::size_t offset) {
  if (offset + 2 > bytes.size()) {
    Fail("truncated ELF16 field");
  }
  return static_cast<std::uint16_t>(
      static_cast<unsigned char>(bytes[offset]) |
      (static_cast<unsigned char>(bytes[offset + 1]) << 8));
}

std::uint32_t ReadU32(const std::string &bytes, std::size_t offset) {
  if (offset + 4 > bytes.size()) {
    Fail("truncated ELF32 field");
  }
  std::uint32_t value = 0;
  for (std::size_t index = 0; index < 4; ++index) {
    value |= static_cast<std::uint32_t>(
                 static_cast<unsigned char>(bytes[offset + index]))
             << (8 * index);
  }
  return value;
}

std::uint64_t ReadU64(const std::string &bytes, std::size_t offset) {
  if (offset + 8 > bytes.size()) {
    Fail("truncated ELF64 field");
  }
  std::uint64_t value = 0;
  for (std::size_t index = 0; index < 8; ++index) {
    value |= static_cast<std::uint64_t>(
                 static_cast<unsigned char>(bytes[offset + index]))
             << (8 * index);
  }
  return value;
}

void VerifyStaticElf(const std::filesystem::path &ngspice,
                     const std::filesystem::path &temporary_directory) {
  const std::string elf = ReadFile(ngspice);
  if (elf.size() < 64 ||
      elf.compare(0, 4,
                  "\x7f"
                  "ELF") != 0 ||
      static_cast<unsigned char>(elf[4]) != 2 ||
      static_cast<unsigned char>(elf[5]) != 1 || ReadU16(elf, 16) != 2 ||
      ReadU16(elf, 18) != 62) {
    Fail("ngspice is not a little-endian x86-64 ET_EXEC ELF");
  }
  const std::uint64_t program_offset = ReadU64(elf, 32);
  const std::uint16_t entry_size = ReadU16(elf, 54);
  const std::uint16_t entry_count = ReadU16(elf, 56);
  if (entry_size < 56 || entry_count == 0 || program_offset > elf.size() ||
      static_cast<std::uint64_t>(entry_count) * entry_size >
          elf.size() - program_offset) {
    Fail("ngspice has a malformed ELF program-header table");
  }
  for (std::size_t index = 0; index < entry_count; ++index) {
    const std::uint32_t type = ReadU32(
        elf, static_cast<std::size_t>(program_offset) + index * entry_size);
    if (type == 2 || type == 3) {
      Fail("ngspice ELF contains forbidden PT_DYNAMIC or PT_INTERP");
    }
  }
  const ProcessResult version =
      Run({ngspice.string(), "--version"}, temporary_directory);
  RequireSuccess(version, "ngspice version check");
  std::istringstream version_lines(version.output);
  std::string version_line;
  bool exact_version = false;
  while (std::getline(version_lines, version_line)) {
    if (Trim(version_line) ==
        "** ngspice-46 : Circuit level simulation program") {
      exact_version = true;
    }
  }
  if (!exact_version) {
    Fail("built oracle did not report exact ngspice-46");
  }
}

RawTable RunNgspice(const std::filesystem::path &ngspice,
                    const std::filesystem::path &fixture,
                    const std::filesystem::path &temporary_directory,
                    std::string_view stem, bool transient) {
  const std::filesystem::path raw =
      temporary_directory / (std::string(stem) + ".raw");
  const auto netlist = MakeOracleNetlist(fixture, raw, transient);
  const ProcessResult result = Run(
      {ngspice.string(), "-n", "-b", netlist.string()}, temporary_directory);
  RequireSuccess(result, "ngspice " + std::string(stem));
  return ParseRaw(raw);
}

Table RunOhmnivore(const std::filesystem::path &program,
                   const std::filesystem::path &fixture,
                   const std::filesystem::path &temporary_directory) {
  const ProcessResult result =
      Run({program.string(), fixture.string()}, temporary_directory);
  RequireSuccess(result, "Ohmnivore " + fixture.filename().string());
  return ParseCsv(result.output, fixture.filename().string());
}

std::size_t RawColumn(const RawTable &table, std::string_view name) {
  const auto found = std::find(table.names.begin(), table.names.end(), name);
  if (found == table.names.end()) {
    Fail("ngspice raw missing " + std::string(name));
  }
  if (std::find(found + 1, table.names.end(), name) != table.names.end()) {
    Fail("ngspice raw has duplicate " + std::string(name));
  }
  return static_cast<std::size_t>(found - table.names.begin());
}

void Near(double actual, double reference, double absolute, double relative,
          const std::string &context) {
  const double tolerance = std::max(absolute, relative * std::abs(reference));
  if (!std::isfinite(actual) || std::abs(actual - reference) > tolerance) {
    Fail(context + ": actual=" + std::to_string(actual) + " reference=" +
         std::to_string(reference) + " tolerance=" + std::to_string(tolerance));
  }
}

double Interpolate(const Table &actual, std::size_t time_column,
                   std::size_t value_column, double time) {
  if (actual.rows.empty()) {
    Fail("empty transient result");
  }
  for (std::size_t index = 1; index < actual.rows.size(); ++index) {
    if (!(actual.rows[index][time_column] >
          actual.rows[index - 1][time_column])) {
      Fail("Ohmnivore transient times are not strictly increasing");
    }
  }
  const auto found = std::lower_bound(
      actual.rows.begin(), actual.rows.end(), time,
      [time_column](const std::vector<double> &row, double query) {
        return row[time_column] < query;
      });
  const double epsilon = 32.0 * std::numeric_limits<double>::epsilon() *
                         std::max(1.0, std::abs(time));
  if (found != actual.rows.end() &&
      std::abs((*found)[time_column] - time) <= epsilon) {
    return (*found)[value_column];
  }
  if (found != actual.rows.begin()) {
    const auto previous = found - 1;
    if (std::abs((*previous)[time_column] - time) <= epsilon) {
      return (*previous)[value_column];
    }
  }
  if (found == actual.rows.begin() || found == actual.rows.end()) {
    Fail("transient interpolation would extrapolate or clamp");
  }
  const auto &right = *found;
  const auto &left = *(found - 1);
  const double fraction =
      (time - left[time_column]) / (right[time_column] - left[time_column]);
  return left[value_column] +
         fraction * (right[value_column] - left[value_column]);
}

void CompareAc(const Table &actual, const RawTable &reference) {
  RequireExactHeader(actual,
                     {"Frequency", "V(in)_mag", "V(in)_phase_deg", "V(out)_mag",
                      "V(out)_phase_deg", "I(V1)_mag", "I(V1)_phase_deg"},
                     "AC");
  const std::size_t frequency = Column(actual.header, "Frequency");
  const std::size_t magnitude = Column(actual.header, "V(out)_mag");
  const std::size_t phase = Column(actual.header, "V(out)_phase_deg");
  const std::size_t raw_frequency = RawColumn(reference, "frequency");
  const std::size_t raw_out = RawColumn(reference, "v(out)");
  if (!reference.complex || actual.rows.size() != reference.rows.size()) {
    Fail("AC point count or complex flag mismatch");
  }
  constexpr double kPi = 3.141592653589793238462643383279502884;
  for (std::size_t i = 0; i < actual.rows.size(); ++i) {
    const double f = reference.rows[i][raw_frequency].real();
    Near(actual.rows[i][frequency], f, 1e-12, 1e-12, "AC frequency alignment");
    const std::complex<double> value = reference.rows[i][raw_out];
    Near(actual.rows[i][magnitude], std::abs(value), 1e-12, 1e-2,
         "AC magnitude");
    const double reference_phase = std::arg(value) * 180.0 / kPi;
    double difference =
        std::remainder(actual.rows[i][phase] - reference_phase, 360.0);
    if (!std::isfinite(difference) || std::abs(difference) > 1.0) {
      Fail("AC phase error exceeds one degree");
    }
  }
}

std::vector<double> RequestedTransientGrid(double step_time, double stop_time,
                                           double start_time,
                                           const std::string &context) {
  std::vector<double> grid;
  for (std::size_t index = 0; index < 1'000'000; ++index) {
    const long double candidate = static_cast<long double>(start_time) +
                                  static_cast<long double>(index) * step_time;
    if (candidate >= static_cast<long double>(stop_time)) {
      break;
    }
    const double value = static_cast<double>(candidate);
    if (!std::isfinite(value) || (!grid.empty() && !(value > grid.back()))) {
      Fail(context + ": requested transient grid is not representable");
    }
    grid.push_back(value);
  }
  if (grid.size() == 1'000'000) {
    Fail(context + ": requested transient grid exceeds point limit");
  }
  if (grid.empty() || grid.back() != stop_time) {
    grid.push_back(stop_time);
  }
  return grid;
}

void CompareTransient(const Table &actual, const RawTable &reference,
                      double step_time, double stop_time, double start_time,
                      const std::vector<std::string> &expected_header,
                      const std::string &context) {
  RequireExactHeader(actual, expected_header, context);
  const std::size_t time = Column(actual.header, "time");
  const std::size_t out = Column(actual.header, "V(out)");
  const std::size_t raw_time = RawColumn(reference, "time");
  const std::size_t raw_out = RawColumn(reference, "v(out)");
  if (reference.complex || reference.rows.empty()) {
    Fail(context + ": invalid transient reference");
  }
  if (actual.rows.back()[time] != stop_time) {
    Fail(context + ": Ohmnivore omitted exact tstop");
  }
  if (actual.rows.front()[time] != start_time) {
    Fail(context + ": Ohmnivore omitted exact tstart");
  }
  const std::vector<double> requested =
      RequestedTransientGrid(step_time, stop_time, start_time, context);
  if (reference.rows.size() != requested.size()) {
    Fail(context + ": ngspice reference point count does not match .TRAN grid");
  }
  for (std::size_t index = 0; index < requested.size(); ++index) {
    const double query = reference.rows[index][raw_time].real();
    if (index > 0 && !(query > reference.rows[index - 1][raw_time].real())) {
      Fail(context + ": ngspice reference times are not strictly increasing");
    }
    Near(query, requested[index], 1e-15, 1e-12, context + " reference grid");
    Near(Interpolate(actual, time, out, query),
         reference.rows[index][raw_out].real(), 1e-2, 2e-2, context);
  }
}

// DC is the only legacy CSV schema whose first field is textual.
std::pair<std::string, double> ParseDcOut(std::string_view contents) {
  std::istringstream input{std::string(contents)};
  std::string line;
  if (!std::getline(input, line) || line != "Variable,Value") {
    Fail("DC CSV schema mismatch");
  }
  const std::vector<std::string> expected = {"V(in)", "V(out)", "I(V1)"};
  std::vector<std::pair<std::string, double>> rows;
  while (std::getline(input, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    const auto fields = ParseCsvLine(line);
    if (line.empty()) {
      continue;
    }
    if (fields.size() != 2) {
      Fail("DC CSV row width mismatch");
    }
    rows.emplace_back(fields[0], ParseDouble(fields[1], "DC value"));
  }
  if (rows.size() != expected.size()) {
    Fail("DC CSV row count mismatch");
  }
  for (std::size_t index = 0; index < expected.size(); ++index) {
    if (rows[index].first != expected[index]) {
      Fail("DC CSV variable order mismatch");
    }
  }
  return rows[1];
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 10) {
      Fail("acceptance runner requires nine runfile arguments");
    }
    const std::filesystem::path temporary_root =
        std::getenv("TEST_TMPDIR") == nullptr
            ? std::filesystem::temp_directory_path()
            : std::filesystem::path(std::getenv("TEST_TMPDIR"));
    const std::filesystem::path temporary =
        temporary_root / ("ngspice_acceptance_" + std::to_string(getpid()));
    std::filesystem::create_directories(temporary / "home");

    const std::filesystem::path ohmnivore = std::filesystem::absolute(argv[1]);
    const std::filesystem::path ngspice = std::filesystem::absolute(argv[2]);
    VerifyStaticElf(ngspice, temporary);

    const std::filesystem::path dc_fixture = std::filesystem::absolute(argv[3]);
    const RawTable dc_reference =
        RunNgspice(ngspice, dc_fixture, temporary, "dc", false);
    const ProcessResult dc_actual =
        Run({ohmnivore.string(), dc_fixture.string()}, temporary);
    RequireSuccess(dc_actual, "Ohmnivore DC");
    const double dc_value = ParseDcOut(dc_actual.output).second;
    Near(dc_value,
         dc_reference.rows.front()[RawColumn(dc_reference, "v(out)")].real(),
         1e-9, 1e-3, "DC V(out)");

    const std::filesystem::path diode_fixture =
        std::filesystem::absolute(argv[4]);
    const RawTable diode_reference =
        RunNgspice(ngspice, diode_fixture, temporary, "dc_diode", false);
    const ProcessResult diode_actual =
        Run({ohmnivore.string(), diode_fixture.string()}, temporary);
    RequireSuccess(diode_actual, "Ohmnivore diode DC");
    const double diode_value = ParseDcOut(diode_actual.output).second;
    Near(diode_value,
         diode_reference.rows.front()[RawColumn(diode_reference, "v(out)")]
             .real(),
         1e-6, 2e-3, "diode DC V(out)");

    const std::filesystem::path ac_fixture = std::filesystem::absolute(argv[5]);
    CompareAc(RunOhmnivore(ohmnivore, ac_fixture, temporary),
              RunNgspice(ngspice, ac_fixture, temporary, "ac", false));

    struct TransientFixture {
      std::filesystem::path path;
      std::string stem;
      double step;
      double stop;
      double start;
      std::vector<std::string> header;
    };
    const std::vector<TransientFixture> transient_fixtures = {
        {std::filesystem::absolute(argv[6]),
         "tran_rc_uic",
         10e-6,
         5e-3,
         0.0,
         {"time", "V(in)", "V(out)", "I(V1)"}},
        {std::filesystem::absolute(argv[7]),
         "tran_pulse_rc",
         1e-6,
         2e-3,
         0.0,
         {"time", "V(in)", "V(out)", "I(V1)"}},
        {std::filesystem::absolute(argv[8]),
         "tran_rl_step",
         2e-6,
         2e-3,
         0.0,
         {"time", "V(in)", "V(out)", "I(V1)", "I(L1)"}},
        {std::filesystem::absolute(argv[9]),
         "tran_current_rlc_non_uic",
         0.1e-6,
         1e-3,
         0.1e-3,
         {"time", "V(in)", "V(mid)", "V(out)", "I(L1)"}},
    };
    for (const auto &fixture : transient_fixtures) {
      CompareTransient(
          RunOhmnivore(ohmnivore, fixture.path, temporary),
          RunNgspice(ngspice, fixture.path, temporary, fixture.stem, true),
          fixture.step, fixture.stop, fixture.start, fixture.header,
          fixture.stem);
    }
    std::cout
        << "ngspice-46 hermetic linear DC/AC/transient plus bounded diode DC "
           "acceptance passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "ngspice acceptance failure: " << error.what() << '\n';
    return 1;
  }
}
