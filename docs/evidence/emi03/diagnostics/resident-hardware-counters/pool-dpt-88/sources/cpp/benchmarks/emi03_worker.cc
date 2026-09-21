#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifdef OHMNIVORE_EMI03_CUDA
#include "cuda/emi03_real_solver.h"
#endif

#ifdef OHMNIVORE_EMI03_RESIDENT
#include "cuda/emi03_worker_pool.h"
#endif

// The ordinary EMI-02 runner supplies the same bounded parser, numerical
// policy, streaming output and atomic publication in both worker builds.
int RunEmi02Job(int argc, char **argv);

namespace {

std::string JsonString(const std::string &value) {
  std::string result = "\"";
  constexpr char hex[] = "0123456789abcdef";
  for (unsigned char c : value) {
    if (c == '\"' || c == '\\') {
      result += '\\';
      result += static_cast<char>(c);
    } else if (c < 32) {
      result += "\\u00";
      result += hex[c >> 4];
      result += hex[c & 15];
    } else {
      result += static_cast<char>(c);
    }
  }
  return result + '\"';
}

int ExecuteChecked(std::vector<std::string> arguments) {
#ifdef OHMNIVORE_EMI03_CUDA
  using namespace ohmnivore;
  const std::filesystem::path telemetry = arguments[3] + ".gpu.json";
  const std::filesystem::path temporary = telemetry.string() + ".partial";
  struct Cleanup {
    const std::vector<std::string> &arguments;
    const std::filesystem::path &temporary;
    bool active = false;
    bool owns_output = false;
    bool owns_temporary = false;
    bool complete = false;
    ~Cleanup() {
      if (active) {
        try {
          static_cast<void>(EndEmi03CudaJob());
        } catch (...) {
          // The outer boundary already reports failure; never throw in cleanup.
        }
      }
      std::error_code ignored;
      if (owns_temporary)
        std::filesystem::remove(temporary, ignored);
      if (owns_output && !complete) {
        std::filesystem::remove(arguments[2], ignored);
        std::filesystem::remove(arguments[3], ignored);
      }
    }
  } cleanup{arguments, temporary};
  for (const auto &path : {telemetry, temporary}) {
    if (std::filesystem::exists(std::filesystem::symlink_status(path))) {
      std::cerr << "io: GPU telemetry path already exists\n";
      return 1;
    }
    for (std::size_t i = 1; i < arguments.size(); ++i) {
      if (std::filesystem::weakly_canonical(path) ==
          std::filesystem::weakly_canonical(arguments[i])) {
        std::cerr << "io: GPU telemetry aliases a job path\n";
        return 1;
      }
    }
  }
  auto begun = BeginEmi03CudaJob(arguments[1]);
  if (!begun.ok()) {
    std::cerr << ErrorCodeName(begun.error().code) << ": "
              << begun.error().message << '\n';
    return 1;
  }
  cleanup.active = true;
#endif
  std::vector<char *> pointers;
  for (auto &argument : arguments)
    pointers.push_back(argument.data());
  int code = RunEmi02Job(static_cast<int>(pointers.size()), pointers.data());
#ifdef OHMNIVORE_EMI03_CUDA
  cleanup.owns_output = code == 0;
  // RunEmi02Job has destroyed every factorization before this explicit release.
  // A cleanup/telemetry failure invalidates otherwise successful raw output.
  auto ended = EndEmi03CudaJob();
  cleanup.active = false;
  bool valid = ended.ok();
  if (!valid) {
    std::cerr << ErrorCodeName(ended.error().code) << ": "
              << ended.error().message << '\n';
  }
  std::ofstream output(temporary);
  cleanup.owns_temporary = static_cast<bool>(output);
  output << Emi03CudaStatisticsJson(ended.ok() ? ended.value()
                                               : SnapshotEmi03CudaJob())
         << '\n';
  output.close();
  if (!output) {
    valid = false;
    std::cerr << "io: GPU telemetry write failed\n";
  } else {
    std::error_code error;
    std::filesystem::rename(temporary, telemetry, error);
    if (error) {
      valid = false;
      std::cerr << "io: GPU telemetry publication failed\n";
    }
  }
  if (!valid) {
    code = 1;
  }
  cleanup.complete = valid && code == 0;
#endif
  return code;
}

int Execute(std::vector<std::string> arguments) {
  try {
    return ExecuteChecked(std::move(arguments));
  } catch (const std::exception &) {
    std::cerr << "io: worker execution or output failed\n";
    return 1;
  } catch (...) {
    std::cerr << "factorization: worker execution failed\n";
    return 1;
  }
}

} // namespace

int main(int argc, char **argv) {
#ifdef OHMNIVORE_EMI03_RESIDENT
  if (argc >= 2 && std::string(argv[1]) == "--worker-fds")
    return RunEmi03WorkerPool(argc, argv, Execute);
#endif
  if (argc == 4)
    return Execute({argv[0], argv[1], argv[2], argv[3]});
  if (argc != 2 || std::string(argv[1]) != "--worker") {
    std::cerr << "usage: emi03_worker input.spice output.raw metadata.json\n"
                 "       emi03_worker --worker\n";
    return 2;
  }
  // A worker owns one job at a time. Every call constructs fresh nonlinear and
  // integration state; only process/library residency survives requests.
  while (true) {
    std::string line;
    char byte;
    while (std::cin.get(byte) && byte != '\n') {
      if (line.size() == 16 * 1024) {
        std::cerr << "unsupported-size: worker request exceeds 16 KiB\n";
        return 2;
      }
      line.push_back(byte);
    }
    if (!std::cin && line.empty())
      break;
    if (!std::cin) {
      std::cerr << "invalid-structure: truncated worker request\n";
      return 2;
    }
    const auto first = line.find('\t');
    const auto second = first == std::string::npos ? std::string::npos
                                                   : line.find('\t', first + 1);
    std::string input = line.substr(0, first);
    int code = 2;
    if (first != std::string::npos && second != std::string::npos &&
        line.find('\t', second + 1) == std::string::npos &&
        line.find('\0') == std::string::npos) {
      std::vector<std::string> arguments{
          argv[0], input, line.substr(first + 1, second - first - 1),
          line.substr(second + 1)};
      bool absolute = true;
      for (std::size_t i = 1; i < arguments.size(); ++i)
        absolute &= std::filesystem::path(arguments[i]).is_absolute();
      if (absolute)
        code = Execute(std::move(arguments));
    }
    std::cout << "{\"status\":\"" << (code == 0 ? "complete" : "typed_failure")
              << "\",\"input\":" << JsonString(input)
              << ",\"exit_code\":" << code << "}\n"
              << std::flush;
    if (!std::cout)
      return 1;
  }
  return std::cin.eof() ? 0 : 1;
}
