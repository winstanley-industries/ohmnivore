#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include "ohmnivore/expression.h"

namespace {

void Require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

std::string Read(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  Require(input.good(), "cannot open " + path.string());
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

void Write(const std::filesystem::path &path, const std::string &contents) {
  std::ofstream output(path, std::ios::binary);
  Require(static_cast<bool>(output << contents), "cannot write test fixture");
  output.close();
  Require(output.good(), "cannot close test fixture");
}

std::string Run(const std::filesystem::path &binary,
                const std::filesystem::path &scratch,
                const std::vector<std::string> &arguments) {
  const std::string executable = binary.string();
  const pid_t process = fork();
  Require(process >= 0, "cannot fork hermetic ngspice");
  if (process == 0) {
    if (chdir(scratch.c_str()) != 0)
      _exit(124);
    const int log = open("ngspice.log", O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (log < 0 || dup2(log, STDOUT_FILENO) < 0 || dup2(log, STDERR_FILENO) < 0)
      _exit(125);
    close(log);
    std::vector<char *> argv = {const_cast<char *>(executable.c_str())};
    for (const auto &argument : arguments)
      argv.push_back(const_cast<char *>(argument.c_str()));
    argv.push_back(nullptr);
    std::vector<std::string> environment = {"PATH=",
                                            "LC_ALL=C",
                                            "LANG=C",
                                            "TZ=UTC",
                                            "HOME=" + scratch.string(),
                                            "TMPDIR=" + scratch.string()};
    std::vector<char *> envp;
    for (auto &entry : environment)
      envp.push_back(entry.data());
    envp.push_back(nullptr);
    execve(executable.c_str(), argv.data(), envp.data());
    _exit(errno == ENOENT ? 127 : 126);
  }
  int status = 0;
  while (waitpid(process, &status, 0) < 0)
    Require(errno == EINTR, "waitpid failed");
  const std::string log = Read(scratch / "ngspice.log");
  Require(WIFEXITED(status) && WEXITSTATUS(status) == 0,
          "ngspice failed: " + log);
  return log;
}

void CheckStaticOracle(const std::filesystem::path &binary) {
  const std::string bytes = Read(binary);
  Require(bytes.size() >= 64 &&
              bytes.substr(0, 4) == std::string("\177ELF", 4) &&
              bytes[4] == 2 && bytes[5] == 1,
          "oracle is not little-endian ELF64");
  auto integer = [&](std::size_t start, std::size_t length) {
    Require(start + length <= bytes.size(), "truncated ELF integer");
    std::uint64_t result = 0;
    for (std::size_t i = 0; i < length; ++i)
      result |= static_cast<std::uint64_t>(
                    static_cast<unsigned char>(bytes[start + i]))
                << (8 * i);
    return result;
  };
  Require(integer(18, 2) == 62, "oracle is not x86-64");
  const auto offset = integer(32, 8), width = integer(54, 2),
             count = integer(56, 2);
  Require(width >= 56 && offset <= bytes.size() &&
              count <= (bytes.size() - offset) / width,
          "invalid ELF program headers");
  for (std::uint64_t i = 0; i < count; ++i) {
    const auto kind = integer(static_cast<std::size_t>(offset + i * width), 4);
    Require(kind != 2 && kind != 3, "oracle dynamically loads host libraries");
  }
}

double Reference(const std::filesystem::path &binary,
                 const std::filesystem::path &scratch,
                 const std::string &expression,
                 const std::array<double, 3> &state,
                 const std::string &parameters = "") {
  std::ostringstream deck;
  deck << std::setprecision(17) << "Independent EMI02B expression fixture\n"
       << "Va a 0 " << state[0] << "\nVb b 0 " << state[1]
       << "\nVsense sensed 0 0\nIbias 0 sensed " << state[2]
       << "\n.include expressions.lib\n"
       << ".options gmin=1e-12 reltol=1e-10 abstol=1e-15 "
          "vntol=1e-12\n.op\n.end\n";
  // ps conversion applies to included libraries, as in the selected model.
  Write(scratch / "expressions.lib",
        parameters + "Bresult result 0 V={" + expression + "}\n");
  Write(scratch / "circuit.cir", deck.str());
  Write(scratch / "driver.cir",
        "Independent ps expression driver\n.control\nset ngbehavior=ps\n"
        "set numdgt=17\nset wr_singlescale\nsource circuit.cir\nrun\n"
        "wrdata result.txt v(result)\nquit\n.endc\n.end\n");
  std::filesystem::remove(scratch / "result.txt");
  const auto log = Run(binary, scratch, {"-n", "-b", "driver.cir"});
  Require(std::filesystem::exists(scratch / "result.txt"),
          "oracle omitted result: " + log);
  std::istringstream table(Read(scratch / "result.txt"));
  double scale = 0, value = 0;
  Require(static_cast<bool>(table >> scale >> value) && std::isfinite(scale) &&
              std::isfinite(value),
          "invalid oracle output: " + log);
  std::string extra;
  Require(!(table >> extra), "oracle returned multiple/unexpected rows");
  return value;
}

void Near(double actual, double expected, const std::string &description,
          double relative = 1e-9) {
  if (std::abs(actual - expected) > 1e-11 + relative * std::abs(expected)) {
    std::ostringstream message;
    message << std::setprecision(17) << description << ": " << actual
            << " != " << expected;
    throw std::runtime_error(message.str());
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    Require(argc == 2, "expected checksum-pinned ngspice path");
    const auto binary = std::filesystem::absolute(argv[1]);
    CheckStaticOracle(binary);
    const char *temporary = std::getenv("TEST_TMPDIR");
    Require(temporary != nullptr, "Bazel TEST_TMPDIR is required");
    const std::filesystem::path scratch =
        std::filesystem::path(temporary) / "expression-oracle";
    Require(std::filesystem::create_directory(scratch),
            "oracle directory already exists");
    Require(Run(binary, scratch, {"--version"}).find("ngspice-46") !=
                std::string::npos,
            "oracle version mismatch");
    struct Case {
      std::string expression;
      std::array<double, 3> state;
      bool smooth;
    };
    const std::vector<Case> cases = {
        {"-v(a)+v(b)*2", {3, 2, .4}, true},
        {"v(a)/v(b)", {3, 2, .4}, true},
        {"v(a)/v(b)", {3, -2, .4}, true},
        {"v(a)**3", {-2, 2, .4}, true},
        {"v(a)**2.5", {-4, 2, .4}, true},
        {"v(a)**v(b)", {-4, 2.5, .4}, true},
        {"exp(v(a))", {2, 2, .4}, true},
        {"exp(v(a))", {20, 2, .4}, true},
        {"if(v(a)>0, v(a)*i(vsense), exp(v(a)*1e100))", {3, 2, .4}, true},
        {"if(v(a)<2,v(b),-v(b))", {2, 3, .4}, false},
        {"i(vsense)*(1+v(a,b)*2.5)", {3, 2, .4}, true},
        {"(-2)**3", {3, 2, .4}, true},
        {"exp(20)", {3, 2, .4}, true},
        {"1/0", {3, 2, .4}, false},
        {"2e-32/v(b)", {3, 1e-32, .4}, false},
        {"2e-32/v(b)", {3, -1e-32, .4}, false},
        {"2e-32/v(b)", {3, 0, .4}, false},
    };
    ohmnivore::ExpressionBindings bindings{.state_size = 3,
                                           .node_indices = {{"a", 0}, {"b", 1}},
                                           .current_indices = {{"vsense", 2}},
                                           .parameters = {}};
    std::size_t comparisons = 0;
    for (const auto &test : cases) {
      auto compiled = ohmnivore::CompileExpression(test.expression, bindings);
      Require(compiled.ok(), "production expression compile failed");
      auto actual = ohmnivore::EvaluateExpression(compiled.value(), test.state);
      Require(actual.ok(), "production expression evaluation failed");
      Near(actual.value().value,
           Reference(binary, scratch, test.expression, test.state),
           test.expression);
      ++comparisons;
      if (!test.smooth)
        continue;
      for (std::size_t variable = 0; variable < 3; ++variable) {
        double derivative = 0;
        for (const auto &[index, value] : actual.value().derivatives)
          if (index == variable)
            derivative = value;
        constexpr double h = 1e-5;
        auto left = test.state, right = test.state;
        left[variable] -= h;
        right[variable] += h;
        const double numerical =
            (Reference(binary, scratch, test.expression, right) -
             Reference(binary, scratch, test.expression, left)) /
            (2 * h);
        Near(derivative, numerical, "oracle derivative " + test.expression,
             1e-6);
        ++comparisons;
      }
    }
    // The selected parameter language uses mathematical exp, while behavioral
    // exp uses the PSpice runtime continuation. This fixture keeps them
    // distinct.
    auto params = ohmnivore::ResolveParameters(
        {{"a1", "b1*2"}, {"b1", "3"}, {"c1", "exp(20)"}});
    Require(params.ok(), "production parameter graph rejected");
    Near(params.value().at("A1") + params.value().at("C1"),
         Reference(binary, scratch, "a1+c1", {0, 0, 0},
                   ".param a1={b1*2} b1=3 c1={exp(20)}\n"),
         "parameter dialect");
    // Upstream reserves vt during frontend expression conversion, even when
    // a local parameter spells the same name. The bounded importer must lower
    // this contextual collision explicitly; the generic evaluator has no magic
    // names.
    Near(Reference(binary, scratch, "vt+v(a)*0", {0, 0, 0}, ".param vt=1.91\n"),
         (27.0 + 273.15) * 8.6173303e-5, "ngspice reserved vt collision");
    std::cout << "EMI02B hermetic ngspice46 ps values/derivatives: "
              << comparisons << "; parameter dialect and reserved vt checked\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
