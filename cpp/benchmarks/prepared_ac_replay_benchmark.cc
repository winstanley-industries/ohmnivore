#include <algorithm>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <sys/resource.h>
#include <sys/types.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <unistd.h>

#include "cpp/benchmarks/prepared_ac_replay.h"
#include "ohmnivore/ir.h"
#include "ohmnivore/prepared_ac.h"
#include "ohmnivore/solver.h"
#include "ohmnivore/status.h"

#ifdef OHMNIVORE_GPU02_EVIDENCE
#include <cuda_runtime_api.h>

#include "cuda/prepared_ac_cuda.h"
#endif

namespace {

using Clock = std::chrono::steady_clock;
using ohmnivore::PreparedAcBatch;
using ohmnivore::PreparedAcBatchResult;
using ohmnivore::PreparedAcResultMember;
using ohmnivore::Result;
using ohmnivore::benchmarks::PreparedAcReplayCase;
using ohmnivore::benchmarks::PreparedAcReplayCorpus;

inline constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
inline constexpr std::uint64_t kFnvOffsetFirst = 14695981039346656037ULL;
inline constexpr std::uint64_t kFnvOffsetSecond = 9521211207457086692ULL;
inline constexpr std::uint32_t kChildOutcomeMagic = 0x47505531U;

class EvidenceFingerprint {
public:
  explicit EvidenceFingerprint(std::string_view domain) { AddString(domain); }

  void AddByte(std::uint8_t byte) {
    first_ = (first_ ^ byte) * kFnvPrime;
    second_ = (second_ ^ byte) * kFnvPrime;
  }

  void AddBytes(std::string_view bytes) {
    for (const unsigned char byte : bytes) {
      AddByte(byte);
    }
  }

  void AddUint64(std::uint64_t value) {
    for (std::size_t index = 0; index < 8; ++index) {
      AddByte(static_cast<std::uint8_t>(value & 0xffU));
      value >>= 8U;
    }
  }

  void AddString(std::string_view value) {
    AddUint64(value.size());
    AddBytes(value);
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

class EvidenceWriter {
public:
  void Write(std::string line) {
    fingerprint_.AddBytes(line);
    fingerprint_.AddByte('\n');
    std::cout << line << '\n';
  }

  void WriteCompletion(std::string line) { std::cout << line << '\n'; }

  [[nodiscard]] std::string Fingerprint() const {
    return fingerprint_.Finish();
  }

private:
#ifdef OHMNIVORE_GPU02_EVIDENCE
  EvidenceFingerprint fingerprint_{"gpu02-evidence-records-v1"};
#else
  EvidenceFingerprint fingerprint_{"gpu01-evidence-records-v1"};
#endif
};

struct Options {
  std::string corpus_path;
  std::vector<std::string> source_paths;
#ifdef OHMNIVORE_GPU02_EVIDENCE
  std::size_t warmups = 3;
  std::size_t repetitions = 20;
#else
  std::size_t warmups = 2;
  std::size_t repetitions = 9;
#endif
  std::size_t requested_threads = 0;
  std::string command;
};

struct Environment {
  std::string build_mode;
  std::string compiler;
  std::string kernel;
  std::string cpu;
  std::string target_cpu = "AMD Ryzen 9 9950X3D 16-Core Processor";
  std::string target_gpu = "NVIDIA GeForce RTX 5080";
  std::string target_driver_api = "13030";
  std::string target_cuda_runtime = "13.3";
  std::string observed_gpu;
  std::string observed_driver;
#ifdef OHMNIVORE_GPU02_EVIDENCE
  std::string observed_gpu_uuid;
  std::string observed_compute_capability;
  int observed_driver_api = 0;
  int observed_cuda_runtime = 0;
#endif
  std::size_t hardware_threads = 1;
};

struct TimingSample {
  std::uint64_t preparation_ns = 0;
  std::uint64_t schedule_klu_ns = 0;
  std::uint64_t validation_ns = 0;
  std::uint64_t total_ns = 0;
  double throughput_members_per_second = 0.0;
  std::uint64_t cpu_peak_rss_bytes = 0;
  std::size_t participating_threads = 1;
  std::size_t measured_reuses = 1;
  std::size_t scheduled_klu_solves = 0;
  std::size_t validation_klu_solves = 0;
  std::size_t actual_klu_solves = 0;
#ifdef OHMNIVORE_GPU02_EVIDENCE
  std::uint64_t executor_create_ns = 0;
  std::uint64_t executor_execute_ns = 0;
  std::uint64_t executor_overhead_ns = 0;
  std::uint64_t context_setup_ns = 0;
  std::uint64_t library_setup_ns = 0;
  std::uint64_t host_pack_ns = 0;
  std::uint64_t upload_ns = 0;
  std::uint64_t matrix_setup_ns = 0;
  std::uint64_t analysis_submit_ns = 0;
  std::uint64_t analysis_sync_ns = 0;
  std::uint64_t factor_solve_submit_ns = 0;
  std::uint64_t factor_solve_sync_ns = 0;
  std::uint64_t readback_ns = 0;
  std::uint64_t phase_status_ns = 0;
  std::uint64_t memory_accounting_ns = 0;
  std::uint64_t result_assembly_ns = 0;
  std::uint64_t analysis_device_ns = 0;
  std::uint64_t factor_solve_device_ns = 0;
  std::size_t cuda_factorizations = 0;
  std::size_t cuda_solves = 0;
  std::size_t cuda_factorization_calls = 0;
  std::size_t cuda_solve_calls = 0;
  std::size_t cuda_uniform_batch_size = 0;
  std::uint64_t gpu_peak_batch_memory_bytes = 0;
  std::uint64_t gpu_context_memory_bytes = 0;
  std::uint64_t cudss_estimated_peak_device_bytes = 0;
  std::uint64_t cudss_observed_peak_device_bytes = 0;
  std::uint64_t release_ns = 0;
  std::size_t release_count = 0;
  std::uint64_t outstanding_device_bytes_after_release = 0;
#endif
};

struct ChildOutcome {
  std::uint32_t magic = kChildOutcomeMagic;
  std::uint32_t success = 0;
  TimingSample sample;
  char error[512]{};
};

[[nodiscard]] std::size_t ParseCount(std::string_view text,
                                     std::string_view name, bool allow_zero) {
  std::size_t value = 0;
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
      (!allow_zero && value == 0) || value > 10'000) {
    throw std::runtime_error(std::string(name) +
                             " must be an integer in the supported range");
  }
  return value;
}

[[nodiscard]] std::string ReconstructCommand(int argc, char **argv) {
  std::string command;
  for (int index = 0; index < argc; ++index) {
    if (index != 0) {
      command.push_back(' ');
    }
    command.push_back('\'');
    for (const char character : std::string_view(argv[index])) {
      if (character == '\'') {
        command += "'\\''";
      } else {
        command.push_back(character);
      }
    }
    command.push_back('\'');
  }
  return command;
}

[[nodiscard]] Options ParseOptions(int argc, char **argv) {
  Options options;
  options.command = ReconstructCommand(argc, argv);
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    constexpr std::string_view kCorpus = "--corpus=";
    constexpr std::string_view kSource = "--source=";
    constexpr std::string_view kWarmups = "--warmups=";
    constexpr std::string_view kRepetitions = "--repetitions=";
    constexpr std::string_view kThreads = "--threads=";
    if (argument.starts_with(kCorpus)) {
      options.corpus_path = argument.substr(kCorpus.size());
    } else if (argument.starts_with(kSource)) {
      options.source_paths.emplace_back(argument.substr(kSource.size()));
    } else if (argument.starts_with(kWarmups)) {
      options.warmups =
          ParseCount(argument.substr(kWarmups.size()), "warmups", false);
    } else if (argument.starts_with(kRepetitions)) {
      options.repetitions = ParseCount(argument.substr(kRepetitions.size()),
                                       "repetitions", false);
    } else if (argument.starts_with(kThreads)) {
      options.requested_threads =
          ParseCount(argument.substr(kThreads.size()), "threads", true);
    } else {
      throw std::runtime_error("unknown benchmark argument: " +
                               std::string(argument));
    }
  }
  if (options.corpus_path.empty() || options.source_paths.empty()) {
    throw std::runtime_error("--corpus and at least one --source are required");
  }
#ifdef OHMNIVORE_GPU02_EVIDENCE
  if (options.warmups < 3 || options.repetitions < 20) {
    throw std::runtime_error(
        "GPU-02 evidence requires at least 3 warmups and 20 repetitions");
  }
#endif
  return options;
}

[[nodiscard]] std::string ReadFirstLine(const std::filesystem::path &path) {
  std::ifstream input(path);
  std::string line;
  return std::getline(input, line) ? line : "unavailable";
}

[[nodiscard]] std::string ReadBinary(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("could not read evidence identity input: " +
                             path.string());
  }
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

[[nodiscard]] std::string
FingerprintSources(const std::vector<std::string> &paths) {
#ifdef OHMNIVORE_GPU02_EVIDENCE
  EvidenceFingerprint fingerprint("gpu02-source-set-v1");
#else
  EvidenceFingerprint fingerprint("gpu01-source-set-v1");
#endif
  for (const std::string &path : paths) {
    const std::string bytes = ReadBinary(path);
    fingerprint.AddString(path);
    fingerprint.AddUint64(bytes.size());
    fingerprint.AddBytes(bytes);
  }
  return fingerprint.Finish();
}

[[nodiscard]] std::string FingerprintBinary() {
  const std::string bytes = ReadBinary("/proc/self/exe");
#ifdef OHMNIVORE_GPU02_EVIDENCE
  EvidenceFingerprint fingerprint("gpu02-binary-v1");
#else
  EvidenceFingerprint fingerprint("gpu01-binary-v1");
#endif
  fingerprint.AddUint64(bytes.size());
  fingerprint.AddBytes(bytes);
  return fingerprint.Finish();
}

[[nodiscard]] std::string CpuModel() {
  std::ifstream input("/proc/cpuinfo");
  for (std::string line; std::getline(input, line);) {
    constexpr std::string_view kModel = "model name";
    if (line.starts_with(kModel)) {
      const std::size_t colon = line.find(':');
      if (colon != std::string::npos) {
        const std::size_t value = line.find_first_not_of(" \t", colon + 1);
        return value == std::string::npos ? "unavailable" : line.substr(value);
      }
    }
  }
  return "unavailable";
}

[[nodiscard]] std::string KernelIdentity() {
  utsname identity{};
  if (uname(&identity) != 0) {
    return "unavailable";
  }
  return std::string(identity.sysname) + " " + identity.release + " " +
         identity.machine;
}

[[nodiscard]] std::string ObservedGpu() {
  const std::filesystem::path directory("/proc/driver/nvidia/gpus");
  std::error_code error;
  if (!std::filesystem::is_directory(directory, error)) {
    return "unavailable";
  }
  for (const auto &entry :
       std::filesystem::directory_iterator(directory, error)) {
    std::ifstream input(entry.path() / "information");
    for (std::string line; std::getline(input, line);) {
      constexpr std::string_view kModel = "Model:";
      if (line.starts_with(kModel)) {
        const std::size_t value = line.find_first_not_of(" \t", kModel.size());
        return value == std::string::npos ? "unavailable" : line.substr(value);
      }
    }
  }
  return "unavailable";
}

[[nodiscard]] Environment InspectEnvironment() {
  const std::size_t hardware = std::thread::hardware_concurrency();
  return Environment{
#ifdef __OPTIMIZE__
      .build_mode = "optimized",
#else
      .build_mode = "nonoptimized",
#endif
      .compiler = __VERSION__,
      .kernel = KernelIdentity(),
      .cpu = CpuModel(),
      .observed_gpu = ObservedGpu(),
      .observed_driver = ReadFirstLine("/proc/driver/nvidia/version"),
#ifdef OHMNIVORE_GPU02_EVIDENCE
      .observed_gpu_uuid = {},
      .observed_compute_capability = {},
      .observed_driver_api = 0,
      .observed_cuda_runtime = 0,
#endif
      .hardware_threads = hardware == 0 ? 1 : hardware,
  };
}

#ifdef OHMNIVORE_GPU02_EVIDENCE
struct GpuEnvironmentOutcome {
  std::uint32_t magic = 0x47505532U;
  std::uint32_t success = 0;
  int driver_api = 0;
  int runtime_api = 0;
  int compute_major = 0;
  int compute_minor = 0;
  char name[256]{};
  char uuid[64]{};
  char error[256]{};
};

[[nodiscard]] GpuEnvironmentOutcome InspectGpuEnvironmentInChild() {
  int descriptors[2]{};
  if (pipe(descriptors) != 0) {
    throw std::runtime_error("could not create GPU identity pipe");
  }
  const pid_t child = fork();
  if (child < 0) {
    close(descriptors[0]);
    close(descriptors[1]);
    throw std::runtime_error("could not fork GPU identity child");
  }
  if (child == 0) {
    close(descriptors[0]);
    GpuEnvironmentOutcome outcome;
    cudaDeviceProp properties{};
    const cudaError_t device_status = cudaGetDeviceProperties(&properties, 0);
    const cudaError_t driver_status = cudaDriverGetVersion(&outcome.driver_api);
    const cudaError_t runtime_status =
        cudaRuntimeGetVersion(&outcome.runtime_api);
    if (device_status == cudaSuccess && driver_status == cudaSuccess &&
        runtime_status == cudaSuccess) {
      std::snprintf(outcome.name, sizeof(outcome.name), "%s", properties.name);
      std::ostringstream uuid;
      uuid << std::hex << std::setfill('0');
      for (const char byte : properties.uuid.bytes) {
        uuid << std::setw(2)
             << static_cast<unsigned int>(static_cast<unsigned char>(byte));
      }
      std::snprintf(outcome.uuid, sizeof(outcome.uuid), "%s",
                    uuid.str().c_str());
      outcome.compute_major = properties.major;
      outcome.compute_minor = properties.minor;
      outcome.success = 1;
    } else {
      std::snprintf(outcome.error, sizeof(outcome.error),
                    "device=%s driver=%s runtime=%s",
                    cudaGetErrorString(device_status),
                    cudaGetErrorString(driver_status),
                    cudaGetErrorString(runtime_status));
    }
    const bool written = write(descriptors[1], &outcome, sizeof(outcome)) ==
                         static_cast<ssize_t>(sizeof(outcome));
    close(descriptors[1]);
    _exit(written && outcome.success != 0 ? 0 : 1);
  }
  close(descriptors[1]);
  GpuEnvironmentOutcome outcome;
  const ssize_t received = read(descriptors[0], &outcome, sizeof(outcome));
  close(descriptors[0]);
  int status = 0;
  while (waitpid(child, &status, 0) < 0 && errno == EINTR) {
  }
  if (received != static_cast<ssize_t>(sizeof(outcome)) ||
      outcome.magic != 0x47505532U || !WIFEXITED(status) ||
      WEXITSTATUS(status) != 0 || outcome.success == 0) {
    throw std::runtime_error(
        std::string("GPU identity inspection failed: ") +
        (outcome.error[0] == '\0' ? "missing child result" : outcome.error));
  }
  return outcome;
}

void AddGpuEnvironment(Environment *environment) {
  const GpuEnvironmentOutcome gpu = InspectGpuEnvironmentInChild();
  environment->observed_gpu = gpu.name;
  environment->observed_gpu_uuid = gpu.uuid;
  environment->observed_compute_capability = std::to_string(gpu.compute_major) +
                                             "." +
                                             std::to_string(gpu.compute_minor);
  environment->observed_driver_api = gpu.driver_api;
  environment->observed_cuda_runtime = gpu.runtime_api;
  if (environment->observed_driver == "unavailable") {
    environment->observed_driver =
        "cudaDriverGetVersion=" + std::to_string(gpu.driver_api);
  }
}
#endif

[[nodiscard]] std::string UtcNow() {
  const std::time_t now = std::time(nullptr);
  std::tm utc{};
  if (gmtime_r(&now, &utc) == nullptr) {
    return "unavailable";
  }
  char buffer[32]{};
  if (std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc) == 0) {
    return "unavailable";
  }
  return buffer;
}

[[nodiscard]] std::uint64_t ElapsedNanoseconds(Clock::time_point start) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start)
          .count());
}

[[nodiscard]] std::uint64_t PeakResidentBytes() {
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0 || usage.ru_maxrss < 0) {
    return 0;
  }
  return static_cast<std::uint64_t>(usage.ru_maxrss) * 1024U;
}

[[nodiscard]] std::string Csv(std::string_view value) {
  if (value.find_first_of(",\"\r\n") == std::string_view::npos) {
    return std::string(value);
  }
  std::string escaped = "\"";
  for (const char character : value) {
    if (character == '\"') {
      escaped.push_back('\"');
    }
    escaped.push_back(character);
  }
  escaped.push_back('\"');
  return escaped;
}

[[nodiscard]] std::string FrequencyBits(double frequency) {
  std::ostringstream output;
  output << "0x" << std::hex << std::setw(16) << std::setfill('0')
         << std::bit_cast<std::uint64_t>(frequency);
  return output.str();
}

[[nodiscard]] PreparedAcBatch PrepareCase(const PreparedAcReplayCase &item) {
  auto prepared = ohmnivore::benchmarks::PrepareAcReplayCase(item);
  if (!prepared.ok()) {
    throw std::runtime_error(
        item.case_id + " preparation failed: " + prepared.error().message);
  }
  return prepared.TakeValue();
}

void RequireAuthorityEqual(const PreparedAcBatchResult &actual,
                           const PreparedAcBatchResult &authority,
                           bool bitwise) {
  if (actual.contract_version != authority.contract_version ||
      actual.replay_id != authority.replay_id ||
      actual.structure_fingerprint != authority.structure_fingerprint ||
      actual.batch_fingerprint != authority.batch_fingerprint ||
      actual.members.size() != authority.members.size()) {
    throw std::runtime_error("prepared result envelope differs from authority");
  }
  for (std::size_t member = 0; member < actual.members.size(); ++member) {
    if (!(actual.members[member].identity ==
          authority.members[member].identity) ||
        actual.members[member].solution.size() !=
            authority.members[member].solution.size()) {
      throw std::runtime_error(
          "prepared result association differs from authority");
    }
    for (std::size_t value = 0; value < actual.members[member].solution.size();
         ++value) {
      const std::complex<double> left = actual.members[member].solution[value];
      const std::complex<double> right =
          authority.members[member].solution[value];
      if (bitwise) {
        if (std::bit_cast<std::uint64_t>(left.real()) !=
                std::bit_cast<std::uint64_t>(right.real()) ||
            std::bit_cast<std::uint64_t>(left.imag()) !=
                std::bit_cast<std::uint64_t>(right.imag())) {
          throw std::runtime_error(
              "serial prepared authority is not bitwise repeatable");
        }
      } else {
        const double scale = std::max(std::abs(left), std::abs(right));
        if (std::abs(left - right) > 1.0e-12 + 1.0e-9 * scale) {
          throw std::runtime_error(
              "parallel prepared result differs from CPU authority");
        }
      }
    }
  }
}

class ParallelCpuExecutor {
public:
  explicit ParallelCpuExecutor(std::size_t worker_count)
      : factorization_(worker_count) {}

  [[nodiscard]] Result<PreparedAcBatchResult>
  Execute(const PreparedAcBatch &batch) {
    auto valid = ohmnivore::ValidatePreparedAcBatch(batch);
    if (!valid.ok()) {
      return Result<PreparedAcBatchResult>::Fail(valid.error().code,
                                                 valid.error().message);
    }
    if (cached_batch_fingerprint_ != batch.batch_fingerprint) {
      for (auto &factorization : factorization_) {
        factorization.reset();
      }
      cached_batch_fingerprint_ = batch.batch_fingerprint;
    }

    PreparedAcBatchResult result{
        .contract_version = batch.contract_version,
        .replay_id = batch.replay_id,
        .structure_fingerprint = batch.structure.fingerprint,
        .batch_fingerprint = batch.batch_fingerprint,
        .members = std::vector<PreparedAcResultMember>(batch.members.size()),
    };
    std::vector<std::optional<ohmnivore::Error>> errors(factorization_.size());
    std::atomic<std::size_t> next_member = 0;
    std::atomic<std::size_t> solve_count = 0;
    std::vector<std::thread> workers;
    workers.reserve(factorization_.size());
    for (std::size_t worker = 0; worker < factorization_.size(); ++worker) {
      workers.emplace_back([&, worker]() {
        while (true) {
          const std::size_t member = next_member.fetch_add(1);
          if (member >= batch.members.size()) {
            break;
          }
          auto matrix = ohmnivore::MaterializePreparedAcMatrix(batch, member);
          if (!matrix.ok()) {
            errors[worker] = matrix.error();
            return;
          }
          if (factorization_[worker] == nullptr) {
            auto analyzed =
                ohmnivore::SparseComplexFactorization::Analyze(matrix.value());
            if (!analyzed.ok()) {
              errors[worker] = analyzed.error();
              return;
            }
            factorization_[worker] = analyzed.TakeValue();
          }
          auto solved = factorization_[worker]->FactorAndSolve(
              matrix.value(), batch.members[member].rhs);
          if (!solved.ok()) {
            errors[worker] = solved.error();
            return;
          }
          solve_count.fetch_add(1);
          result.members[member] = PreparedAcResultMember{
              .identity = batch.members[member].identity,
              .solution = solved.TakeValue(),
          };
        }
      });
    }
    for (std::thread &worker : workers) {
      worker.join();
    }
    for (const auto &error : errors) {
      if (error.has_value()) {
        return Result<PreparedAcBatchResult>::Fail(error->code, error->message);
      }
    }
    last_solve_count_ = solve_count.load();
    if (last_solve_count_ != batch.members.size()) {
      return Result<PreparedAcBatchResult>::Fail(
          ohmnivore::ErrorCode::kPreparedBackendFailure,
          "parallel CPU comparator solve count differs from batch size");
    }
    return Result<PreparedAcBatchResult>::Ok(std::move(result));
  }

  [[nodiscard]] std::size_t last_solve_count() const {
    return last_solve_count_;
  }

private:
  std::string cached_batch_fingerprint_;
  std::vector<std::unique_ptr<ohmnivore::SparseComplexFactorization>>
      factorization_;
  std::size_t last_solve_count_ = 0;
};

[[nodiscard]] PreparedAcBatchResult
BuildAuthority(const PreparedAcReplayCase &item) {
  PreparedAcBatch batch = PrepareCase(item);
  const auto solve_once = [&]() {
    ohmnivore::CpuKluPreparedAcBatchBackend backend;
    auto solved = backend.Execute(batch);
    if (!solved.ok()) {
      throw std::runtime_error(
          item.case_id + " authority solve failed: " + solved.error().message);
    }
    auto accepted =
        ohmnivore::ValidatePreparedAcBatchResult(batch, solved.value());
    if (!accepted.ok()) {
      throw std::runtime_error(
          item.case_id + " authority rejected: " + accepted.error().message);
    }
    return solved.TakeValue();
  };
  PreparedAcBatchResult first = solve_once();
  PreparedAcBatchResult second = solve_once();
  RequireAuthorityEqual(second, first, true);
  return first;
}

[[nodiscard]] TimingSample Measure(const PreparedAcReplayCase &item,
                                   const PreparedAcBatchResult &authority,
                                   bool parallel, bool prepared_reuse,
                                   std::size_t participating_threads) {
  TimingSample sample;
  sample.participating_threads = parallel ? participating_threads : 1;
  sample.measured_reuses = prepared_reuse ? item.reuse_count : 1;

  const Clock::time_point preparation_start = Clock::now();
  PreparedAcBatch batch = PrepareCase(item);
  sample.preparation_ns = ElapsedNanoseconds(preparation_start);

  std::unique_ptr<ohmnivore::CpuKluPreparedAcBatchBackend> serial;
  std::unique_ptr<ParallelCpuExecutor> parallel_executor;
  const Clock::time_point executor_start = Clock::now();
  if (parallel) {
    parallel_executor =
        std::make_unique<ParallelCpuExecutor>(participating_threads);
  } else {
    serial = std::make_unique<ohmnivore::CpuKluPreparedAcBatchBackend>();
  }
  sample.schedule_klu_ns += ElapsedNanoseconds(executor_start);

  for (std::size_t reuse = 0; reuse < sample.measured_reuses; ++reuse) {
    const std::size_t serial_before =
        parallel ? 0 : serial->statistics().solves;
    const Clock::time_point solve_start = Clock::now();
    Result<PreparedAcBatchResult> solved =
        parallel ? parallel_executor->Execute(batch) : serial->Execute(batch);
    sample.schedule_klu_ns += ElapsedNanoseconds(solve_start);
    if (!solved.ok()) {
      throw std::runtime_error(
          item.case_id + " measured solve failed: " + solved.error().message);
    }
    const std::size_t execution_solves =
        parallel ? parallel_executor->last_solve_count()
                 : serial->statistics().solves - serial_before;
    if (execution_solves != item.batch_size) {
      throw std::runtime_error(item.case_id +
                               " measured scheduled solve count is unfair");
    }
    sample.scheduled_klu_solves += execution_solves;

    const Clock::time_point validation_start = Clock::now();
    auto accepted =
        ohmnivore::ValidatePreparedAcBatchResult(batch, solved.value());
    if (!accepted.ok()) {
      throw std::runtime_error(item.case_id + " measured result rejected: " +
                               accepted.error().message);
    }
    RequireAuthorityEqual(solved.value(), authority, !parallel);
    sample.validation_ns += ElapsedNanoseconds(validation_start);
    sample.validation_klu_solves += item.batch_size;
  }
  const std::size_t expected = item.batch_size * sample.measured_reuses;
  if (sample.scheduled_klu_solves != expected ||
      sample.validation_klu_solves != expected) {
    throw std::runtime_error(item.case_id +
                             " sample KLU solve accounting is incomplete");
  }
  sample.actual_klu_solves =
      sample.scheduled_klu_solves + sample.validation_klu_solves;
  sample.total_ns =
      sample.preparation_ns + sample.schedule_klu_ns + sample.validation_ns;
  sample.throughput_members_per_second = static_cast<double>(expected) * 1e9 /
                                         static_cast<double>(sample.total_ns);
  sample.cpu_peak_rss_bytes = PeakResidentBytes();
  return sample;
}

#ifdef OHMNIVORE_GPU02_EVIDENCE
[[nodiscard]] TimingSample MeasureGpu(const PreparedAcReplayCase &item,
                                      const PreparedAcBatchResult &authority,
                                      bool prepared_reuse) {
  TimingSample sample;
  sample.participating_threads = 0;
  sample.measured_reuses = prepared_reuse ? item.reuse_count : 1;

  const Clock::time_point preparation_start = Clock::now();
  PreparedAcBatch batch = PrepareCase(item);
  sample.preparation_ns = ElapsedNanoseconds(preparation_start);

  const Clock::time_point executor_start = Clock::now();
  ohmnivore::CudaPreparedAcBatchBackend executor;
  sample.executor_create_ns = ElapsedNanoseconds(executor_start);

  for (std::size_t reuse = 0; reuse < sample.measured_reuses; ++reuse) {
    const Clock::time_point execute_start = Clock::now();
    auto solved = executor.Execute(batch);
    const std::uint64_t execute_ns = ElapsedNanoseconds(execute_start);
    const ohmnivore::CudaPreparedAcStatistics &statistics =
        executor.statistics();
    const std::uint64_t classified_execute_ns =
        statistics.last_context_setup_ns + statistics.last_library_setup_ns +
        statistics.last_host_pack_ns + statistics.last_upload_ns +
        statistics.last_matrix_setup_ns + statistics.last_analysis_submit_ns +
        statistics.last_analysis_sync_ns +
        statistics.last_factor_solve_submit_ns +
        statistics.last_factor_solve_sync_ns + statistics.last_readback_ns +
        statistics.last_phase_status_ns + statistics.last_memory_accounting_ns +
        statistics.last_result_assembly_ns;
    if (execute_ns < classified_execute_ns) {
      throw std::runtime_error(item.case_id +
                               " CUDA phase timing exceeds Execute wall");
    }
    sample.executor_execute_ns += execute_ns;
    sample.executor_overhead_ns += execute_ns - classified_execute_ns;
    sample.context_setup_ns += statistics.last_context_setup_ns;
    sample.library_setup_ns += statistics.last_library_setup_ns;
    sample.host_pack_ns += statistics.last_host_pack_ns;
    sample.upload_ns += statistics.last_upload_ns;
    sample.matrix_setup_ns += statistics.last_matrix_setup_ns;
    sample.analysis_submit_ns += statistics.last_analysis_submit_ns;
    sample.analysis_sync_ns += statistics.last_analysis_sync_ns;
    sample.factor_solve_submit_ns += statistics.last_factor_solve_submit_ns;
    sample.factor_solve_sync_ns += statistics.last_factor_solve_sync_ns;
    sample.readback_ns += statistics.last_readback_ns;
    sample.phase_status_ns += statistics.last_phase_status_ns;
    sample.memory_accounting_ns += statistics.last_memory_accounting_ns;
    sample.result_assembly_ns += statistics.last_result_assembly_ns;
    sample.analysis_device_ns += statistics.last_analysis_device_ns;
    sample.factor_solve_device_ns += statistics.last_factor_solve_device_ns;
    sample.gpu_peak_batch_memory_bytes = std::max<std::uint64_t>(
        sample.gpu_peak_batch_memory_bytes, statistics.peak_batch_device_bytes);
    sample.cudss_estimated_peak_device_bytes =
        std::max<std::uint64_t>(sample.cudss_estimated_peak_device_bytes,
                                statistics.cudss_estimated_peak_device_bytes);
    sample.cudss_observed_peak_device_bytes =
        std::max<std::uint64_t>(sample.cudss_observed_peak_device_bytes,
                                statistics.cudss_observed_peak_device_bytes);
    sample.gpu_context_memory_bytes = std::max<std::uint64_t>(
        sample.gpu_context_memory_bytes, statistics.context_device_bytes);
    if (!solved.ok()) {
      throw std::runtime_error(item.case_id + " CUDA measured solve failed: " +
                               solved.error().message);
    }

    const Clock::time_point validation_start = Clock::now();
    auto accepted =
        ohmnivore::ValidatePreparedAcBatchResult(batch, solved.value());
    if (!accepted.ok()) {
      throw std::runtime_error(
          item.case_id +
          " CUDA measured result rejected: " + accepted.error().message);
    }
    RequireAuthorityEqual(solved.value(), authority, false);
    sample.validation_ns += ElapsedNanoseconds(validation_start);
    sample.validation_klu_solves += item.batch_size;
  }

  const ohmnivore::CudaPreparedAcStatistics &statistics = executor.statistics();
  const std::size_t expected = item.batch_size * sample.measured_reuses;
  sample.cuda_factorizations = statistics.factorizations;
  sample.cuda_solves = statistics.solves;
  sample.cuda_factorization_calls = statistics.factorization_calls;
  sample.cuda_solve_calls = statistics.solve_calls;
  sample.cuda_uniform_batch_size = statistics.uniform_batch_size;
  if (statistics.preparations != 1 || statistics.structure_uploads != 1 ||
      statistics.analyses != 1 ||
      statistics.executions != sample.measured_reuses ||
      sample.cuda_factorizations != expected ||
      sample.cuda_solves != expected ||
      sample.cuda_factorization_calls != sample.measured_reuses ||
      sample.cuda_solve_calls != sample.measured_reuses ||
      sample.cuda_uniform_batch_size != item.batch_size ||
      sample.validation_klu_solves != expected) {
    throw std::runtime_error(
        item.case_id + " CUDA solve or preparation accounting is incomplete");
  }
  const Clock::time_point release_start = Clock::now();
  auto released = executor.Release();
  sample.release_ns = ElapsedNanoseconds(release_start);
  if (!released.ok()) {
    throw std::runtime_error(
        item.case_id + " CUDA release failed: " + released.error().message);
  }
  sample.release_count = executor.statistics().releases;
  sample.outstanding_device_bytes_after_release =
      executor.statistics().last_release_outstanding_device_bytes;
  if (sample.release_count != 1 ||
      sample.outstanding_device_bytes_after_release != 0 ||
      executor.statistics().cudss_outstanding_device_bytes != 0 ||
      executor.statistics().controlled_device_bytes != 0) {
    throw std::runtime_error(
        item.case_id + " CUDA teardown or allocation balance is incomplete");
  }
  sample.actual_klu_solves = sample.validation_klu_solves;
  sample.total_ns = sample.preparation_ns + sample.executor_create_ns +
                    sample.executor_overhead_ns + sample.context_setup_ns +
                    sample.library_setup_ns + sample.host_pack_ns +
                    sample.upload_ns + sample.matrix_setup_ns +
                    sample.analysis_submit_ns + sample.analysis_sync_ns +
                    sample.factor_solve_submit_ns +
                    sample.factor_solve_sync_ns + sample.readback_ns +
                    sample.phase_status_ns + sample.memory_accounting_ns +
                    sample.result_assembly_ns + sample.validation_ns;
  if (sample.total_ns != sample.preparation_ns + sample.executor_create_ns +
                             sample.executor_execute_ns +
                             sample.validation_ns) {
    throw std::runtime_error(item.case_id +
                             " CUDA end-to-end timing reconciliation failed");
  }
  sample.throughput_members_per_second = static_cast<double>(expected) * 1e9 /
                                         static_cast<double>(sample.total_ns);
  sample.cpu_peak_rss_bytes = PeakResidentBytes();
  return sample;
}
#endif

[[nodiscard]] bool WriteAll(int descriptor, const void *data,
                            std::size_t size) {
  const auto *bytes = static_cast<const unsigned char *>(data);
  while (size != 0) {
    const ssize_t written = write(descriptor, bytes, size);
    if (written < 0 && errno == EINTR) {
      continue;
    }
    if (written <= 0) {
      return false;
    }
    bytes += written;
    size -= static_cast<std::size_t>(written);
  }
  return true;
}

[[nodiscard]] bool ReadAll(int descriptor, void *data, std::size_t size) {
  auto *bytes = static_cast<unsigned char *>(data);
  while (size != 0) {
    const ssize_t received = read(descriptor, bytes, size);
    if (received < 0 && errno == EINTR) {
      continue;
    }
    if (received <= 0) {
      return false;
    }
    bytes += received;
    size -= static_cast<std::size_t>(received);
  }
  return true;
}

[[nodiscard]] TimingSample
MeasureInChild(const PreparedAcReplayCase &item,
               const PreparedAcBatchResult &authority, bool parallel,
               bool prepared_reuse, std::size_t participating_threads) {
  int descriptors[2]{};
  if (pipe(descriptors) != 0) {
    throw std::runtime_error("could not create measurement pipe");
  }
  const pid_t child = fork();
  if (child < 0) {
    close(descriptors[0]);
    close(descriptors[1]);
    throw std::runtime_error("could not fork isolated measurement child");
  }
  if (child == 0) {
    close(descriptors[0]);
    ChildOutcome outcome;
    try {
      outcome.sample = Measure(item, authority, parallel, prepared_reuse,
                               participating_threads);
      outcome.success = 1;
    } catch (const std::exception &error) {
      std::snprintf(outcome.error, sizeof(outcome.error), "%s", error.what());
    } catch (...) {
      std::snprintf(outcome.error, sizeof(outcome.error), "%s",
                    "unknown measurement exception");
    }
    const bool written = WriteAll(descriptors[1], &outcome, sizeof(outcome));
    close(descriptors[1]);
    _exit(written && outcome.success != 0 ? 0 : 1);
  }

  close(descriptors[1]);
  ChildOutcome outcome;
  const bool received = ReadAll(descriptors[0], &outcome, sizeof(outcome));
  close(descriptors[0]);
  int status = 0;
  while (waitpid(child, &status, 0) < 0) {
    if (errno != EINTR) {
      throw std::runtime_error("could not wait for measurement child");
    }
  }
  if (!received || outcome.magic != kChildOutcomeMagic || !WIFEXITED(status) ||
      WEXITSTATUS(status) != 0 || outcome.success == 0) {
    throw std::runtime_error(
        std::string("isolated measurement failed: ") +
        (outcome.error[0] == '\0' ? "missing child result" : outcome.error));
  }
  return outcome.sample;
}

#ifdef OHMNIVORE_GPU02_EVIDENCE
[[nodiscard]] TimingSample
MeasureGpuInChild(const PreparedAcReplayCase &item,
                  const PreparedAcBatchResult &authority, bool prepared_reuse) {
  int descriptors[2]{};
  if (pipe(descriptors) != 0) {
    throw std::runtime_error("could not create CUDA measurement pipe");
  }
  const pid_t child = fork();
  if (child < 0) {
    close(descriptors[0]);
    close(descriptors[1]);
    throw std::runtime_error("could not fork isolated CUDA measurement child");
  }
  if (child == 0) {
    close(descriptors[0]);
    ChildOutcome outcome;
    try {
      outcome.sample = MeasureGpu(item, authority, prepared_reuse);
      outcome.success = 1;
    } catch (const std::exception &error) {
      std::snprintf(outcome.error, sizeof(outcome.error), "%s", error.what());
    } catch (...) {
      std::snprintf(outcome.error, sizeof(outcome.error), "%s",
                    "unknown CUDA measurement exception");
    }
    const bool written = WriteAll(descriptors[1], &outcome, sizeof(outcome));
    close(descriptors[1]);
    _exit(written && outcome.success != 0 ? 0 : 1);
  }

  close(descriptors[1]);
  ChildOutcome outcome;
  const bool received = ReadAll(descriptors[0], &outcome, sizeof(outcome));
  close(descriptors[0]);
  int status = 0;
  while (waitpid(child, &status, 0) < 0) {
    if (errno != EINTR) {
      throw std::runtime_error("could not wait for CUDA measurement child");
    }
  }
  if (!received || outcome.magic != kChildOutcomeMagic || !WIFEXITED(status) ||
      WEXITSTATUS(status) != 0 || outcome.success == 0) {
    throw std::runtime_error(
        std::string("isolated CUDA measurement failed: ") +
        (outcome.error[0] == '\0' ? "missing child result" : outcome.error));
  }
  return outcome.sample;
}
#endif

[[nodiscard]] std::string
SampleLine(const PreparedAcReplayCorpus &corpus,
           const PreparedAcReplayCase &item, const PreparedAcBatch &prepared,
           const Environment &environment, const Options &options,
           std::string_view mode, std::size_t repetition,
           const TimingSample &sample) {
  std::ostringstream output;
  output << std::setprecision(17) << "sample," << corpus.schema_version << ','
         << corpus.manifest_fingerprint << ',' << item.case_id << ','
         << item.workload_class << ',' << item.topology << ',' << mode << ','
         << environment.build_mode << ',' << Csv(environment.compiler) << ','
         << OHMNIVORE_REPOSITORY_BAZEL_PIN << ',' << Csv(environment.kernel)
         << ',' << Csv(environment.cpu) << ',' << Csv(environment.target_cpu)
         << ',' << Csv(environment.target_gpu) << ','
         << Csv(environment.target_driver_api) << ','
         << Csv(environment.target_cuda_runtime) << ','
         << Csv(environment.observed_gpu) << ','
         << Csv(environment.observed_driver) << ','
         << environment.hardware_threads << ',' << options.requested_threads
         << ',' << sample.participating_threads << ',' << options.warmups << ','
         << options.repetitions << ',' << repetition << ',' << item.node_count
         << ',' << item.branch_count << ',' << item.dimension << ','
         << item.union_nonzeros << ','
         << ohmnivore::benchmarks::PreparedAcReplaySweepName(item.sweep_type)
         << ',' << item.sweep_points << ',' << item.batch_size << ','
         << item.start_frequency_hz << ',' << item.stop_frequency_hz << ','
         << item.g_series << ',' << item.g_shunt << ',' << item.c_series << ','
         << item.c_shunt << ',' << sample.measured_reuses << ','
         << prepared.structure.fingerprint << ',' << prepared.batch_fingerprint
         << ','
         << ohmnivore::benchmarks::FingerprintPreparedAcReplayMembers(prepared)
         << ',' << sample.preparation_ns << ',' << sample.schedule_klu_ns << ','
         << sample.validation_ns << ',' << sample.total_ns << ','
         << sample.scheduled_klu_solves << ',' << sample.validation_klu_solves
         << ',' << sample.actual_klu_solves << ','
         << sample.throughput_members_per_second << ','
         << sample.cpu_peak_rss_bytes;
#ifdef OHMNIVORE_GPU02_EVIDENCE
  output << ',' << sample.executor_create_ns << ','
         << sample.executor_execute_ns << ',' << sample.executor_overhead_ns
         << ',' << sample.context_setup_ns << ',' << sample.library_setup_ns
         << ',' << sample.host_pack_ns << ',' << sample.upload_ns << ','
         << sample.matrix_setup_ns << ',' << sample.analysis_submit_ns << ','
         << sample.analysis_sync_ns << ',' << sample.factor_solve_submit_ns
         << ',' << sample.factor_solve_sync_ns << ',' << sample.readback_ns
         << ',' << sample.phase_status_ns << ',' << sample.memory_accounting_ns
         << ',' << sample.result_assembly_ns << ',' << sample.analysis_device_ns
         << ',' << sample.factor_solve_device_ns << ','
         << sample.cuda_factorizations << ',' << sample.cuda_solves << ','
         << sample.cuda_factorization_calls << ',' << sample.cuda_solve_calls
         << ',' << sample.cuda_uniform_batch_size << ','
         << sample.gpu_peak_batch_memory_bytes << ','
         << sample.gpu_context_memory_bytes << ','
         << sample.cudss_estimated_peak_device_bytes << ','
         << sample.cudss_observed_peak_device_bytes << ',' << sample.release_ns
         << ',' << sample.release_count << ','
         << sample.outstanding_device_bytes_after_release << ",0";
#else
  output << ",0,0";
#endif
  return output.str();
}

[[nodiscard]] std::string
SummaryLine(const PreparedAcReplayCase &item, std::string_view mode,
            const std::vector<TimingSample> &samples) {
  std::vector<std::uint64_t> totals;
  std::vector<double> throughputs;
  totals.reserve(samples.size());
  throughputs.reserve(samples.size());
  for (const TimingSample &sample : samples) {
    totals.push_back(sample.total_ns);
    throughputs.push_back(sample.throughput_members_per_second);
  }
  std::sort(totals.begin(), totals.end());
  std::sort(throughputs.begin(), throughputs.end());
  const auto nearest_rank = [&](std::size_t numerator,
                                std::size_t denominator) {
    const std::size_t rank =
        (numerator * totals.size() + denominator - 1) / denominator;
    return totals[rank - 1];
  };
  std::ostringstream output;
  output << std::setprecision(17) << "summary," << item.case_id << ','
         << item.workload_class << ',' << item.topology << ',' << mode << ','
         << samples.size() << ',' << totals.front() << ',' << nearest_rank(1, 4)
         << ',' << nearest_rank(1, 2) << ',' << nearest_rank(3, 4) << ','
         << nearest_rank(95, 100) << ',' << totals.back() << ','
         << throughputs[(throughputs.size() - 1) / 2] << ",nearest_rank";
#ifdef OHMNIVORE_GPU02_EVIDENCE
  std::uint64_t peak_gpu_memory = 0;
  for (const TimingSample &sample : samples) {
    peak_gpu_memory =
        std::max(peak_gpu_memory, sample.gpu_peak_batch_memory_bytes);
  }
  output << ',' << peak_gpu_memory;
#endif
  output << ",0";
  return output.str();
}

#ifdef OHMNIVORE_GPU02_EVIDENCE
struct SummaryStatistics {
  std::uint64_t median_ns = 0;
  std::uint64_t p95_ns = 0;
  std::uint64_t peak_gpu_memory_bytes = 0;
};

[[nodiscard]] SummaryStatistics
Summarize(const std::vector<TimingSample> &samples) {
  std::vector<std::uint64_t> totals;
  totals.reserve(samples.size());
  SummaryStatistics result;
  for (const TimingSample &sample : samples) {
    totals.push_back(sample.total_ns);
    result.peak_gpu_memory_bytes = std::max(result.peak_gpu_memory_bytes,
                                            sample.gpu_peak_batch_memory_bytes);
  }
  std::sort(totals.begin(), totals.end());
  const auto nearest_rank = [&](std::size_t numerator,
                                std::size_t denominator) {
    const std::size_t rank =
        (numerator * totals.size() + denominator - 1) / denominator;
    return totals[rank - 1];
  };
  result.median_ns = nearest_rank(1, 2);
  result.p95_ns = nearest_rank(95, 100);
  return result;
}

[[nodiscard]] std::string VerdictLine(
    const PreparedAcReplayCase &item, const SummaryStatistics &parallel_cold,
    const SummaryStatistics &parallel_prepared,
    const SummaryStatistics &gpu_cold, const SummaryStatistics &gpu_prepared) {
  const double median_ratio = static_cast<double>(parallel_prepared.median_ns) /
                              static_cast<double>(gpu_prepared.median_ns);
  const double p95_ratio = static_cast<double>(parallel_prepared.p95_ns) /
                           static_cast<double>(gpu_prepared.p95_ns);
  const double cold_ratio = static_cast<double>(gpu_cold.median_ns) /
                            static_cast<double>(parallel_cold.median_ns);
  const bool median_pass = median_ratio >= 1.25;
  const bool p95_pass = p95_ratio >= 1.10;
  const bool cold_pass = cold_ratio <= 1.10;
  const std::uint64_t gpu_peak_batch_memory_bytes = std::max(
      gpu_cold.peak_gpu_memory_bytes, gpu_prepared.peak_gpu_memory_bytes);
  const bool memory_pass = gpu_peak_batch_memory_bytes <= 2'147'483'648ULL;
  const bool control = item.workload_class == "small_control";
  const bool run_pass =
      !control && median_pass && p95_pass && cold_pass && memory_pass;
  std::ostringstream output;
  output << std::setprecision(17) << "verdict," << item.case_id << ','
         << item.workload_class << ',' << parallel_prepared.median_ns << ','
         << gpu_prepared.median_ns << ',' << median_ratio << ','
         << (median_pass ? "pass" : "fail") << ',' << parallel_prepared.p95_ns
         << ',' << gpu_prepared.p95_ns << ',' << p95_ratio << ','
         << (p95_pass ? "pass" : "fail") << ',' << parallel_cold.median_ns
         << ',' << gpu_cold.median_ns << ',' << cold_ratio << ','
         << (cold_pass ? "pass" : "fail") << ',' << gpu_peak_batch_memory_bytes
         << ',' << (memory_pass ? "pass" : "fail") << ','
         << (control ? "permanently_ineligible"
                     : (run_pass ? "run_bounds_pass" : "run_bounds_fail"))
         << ",automatic_dispatch_unauthorized";
  return output.str();
}
#endif

void WriteIdentityRecords(EvidenceWriter *writer,
                          const PreparedAcReplayCorpus &corpus,
                          const PreparedAcReplayCase &item,
                          const PreparedAcBatch &batch,
                          std::size_t *identity_count) {
  const std::string aggregate =
      ohmnivore::benchmarks::FingerprintPreparedAcReplayMembers(batch);
  for (const auto &member : batch.members) {
    std::ostringstream output;
    output << std::setprecision(17) << "identity," << corpus.schema_version
           << ',' << corpus.manifest_fingerprint << ',' << item.case_id << ','
           << batch.structure.fingerprint << ',' << batch.batch_fingerprint
           << ',' << aggregate << ',' << member.identity.contract_version << ','
           << member.identity.replay_id << ',' << member.identity.circuit_id
           << ',' << member.identity.corner_id << ',' << member.identity.ordinal
           << ',' << member.identity.frequency_hz << ','
           << FrequencyBits(member.identity.frequency_hz) << ','
           << member.identity.content_fingerprint;
    writer->Write(output.str());
    ++*identity_count;
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = ParseOptions(argc, argv);
    Environment environment = InspectEnvironment();
#ifdef OHMNIVORE_GPU02_EVIDENCE
    AddGpuEnvironment(&environment);
#endif
    const std::string start_utc = UtcNow();
    const std::string source_fingerprint =
        FingerprintSources(options.source_paths);
    const std::string binary_fingerprint = FingerprintBinary();
    auto loaded =
        ohmnivore::benchmarks::LoadPreparedAcReplayCorpus(options.corpus_path);
    if (!loaded.ok()) {
      throw std::runtime_error(loaded.error().message);
    }
    const PreparedAcReplayCorpus corpus = loaded.TakeValue();

    EvidenceWriter writer;
    writer.Write("metadata,key,value");
#ifdef OHMNIVORE_GPU02_EVIDENCE
    writer.Write("metadata,evidence_contract,gpu02-native-fp64-cuda-v1");
#else
    writer.Write("metadata,evidence_contract,gpu01-cpu-only-v1");
#endif
    writer.Write("metadata,speedup_claim,false");
#ifdef OHMNIVORE_GPU02_EVIDENCE
    writer.Write("metadata,automatic_dispatch_authorized,false");
#endif
    writer.Write("metadata,status,incomplete_until_completion_record");
    writer.Write("metadata,command," + Csv(options.command));
    writer.Write("metadata,start_utc," + start_utc);
#ifdef OHMNIVORE_GPU02_EVIDENCE
    writer.Write(
        "metadata,base_revision,03d755997c261c5618a952ab2c4c032f38a70d6c");
#else
    writer.Write(
        "metadata,base_revision,371e401a14942321848fc308f7ddabe9c4be68b4");
#endif
    writer.Write("metadata,source_fingerprint," + source_fingerprint);
    writer.Write("metadata,binary_fingerprint," + binary_fingerprint);
    writer.Write("metadata,source_input_count," +
                 std::to_string(options.source_paths.size()));
    writer.Write("metadata,corpus_fingerprint," + corpus.manifest_fingerprint);
    writer.Write("metadata,suitesparse_version,7.12.3");
    writer.Write(
        "metadata,suitesparse_archive_sha256,"
        "158ee4ed2ce3fdcbf52c4e47e94b0d1a8ae13344b4a835991d78a3ad20f08086");
    writer.Write("metadata,klu_version,2.3.6");
#ifdef OHMNIVORE_GPU02_EVIDENCE
    writer.Write("metadata,cudss_version,0.8.0.10");
    writer.Write("metadata,cudss_execution_shape,uniform_batch_all_members");
    writer.Write("metadata,cudss_matching,disabled");
    writer.Write("metadata,cudss_deterministic_mode,disabled");
    writer.Write("metadata,cudss_fp64_iterative_refinement_steps,1");
    writer.Write("metadata,cudss_reordering,default");
    writer.Write("metadata,cudss_factorization,default");
    writer.Write(
        "metadata,cudss_archive_sha256,"
        "ba18f5fd80dcbbe905d158caac5b3061d848442bb5abd477b5f296b4257a4937");
    writer.Write("metadata,cuda_toolkit_redist_version,13.0.2");
    writer.Write("metadata,nvcc_redist_version,13.0.88");
    writer.Write("metadata,cublas_redist_version,13.1.0.3");
    writer.Write(
        "metadata,cublas_archive_sha256,"
        "88bc951efd906032a371153ca61975e0d9c4761e4012169169a6b3a47931606e");
    writer.Write("metadata,cuda_architectures,compute_120+sm_120");
    writer.Write("metadata,cuda_runtime_linkage,static");
    writer.Write("metadata,bazel_configuration,opt+cuda");
    writer.Write(
        "metadata,canonical_bazel_invocation,bazel run -c opt --config=cuda "
        "//cuda:gpu02_evidence_benchmark -- --warmups=3 --repetitions=20");
    writer.Write("metadata,host_toolchain,nvcc-13.0.88+gcc-15.2.0");
    writer.Write(
        "metadata,source_identity_scope,benchmark+replay+prepared_contract+"
        "complete_cpp_core+cuda_executor+relevant_bazel_module_lock_and_"
        "vendor_build_inputs");
#endif
    writer.Write("metadata,bazel_version," +
                 std::string(OHMNIVORE_REPOSITORY_BAZEL_PIN));
    writer.Write("metadata,compiler," + Csv(environment.compiler));
    writer.Write("metadata,build_mode," + environment.build_mode);
    writer.Write("metadata,clock,std::chrono::steady_clock");
#ifdef OHMNIVORE_GPU02_EVIDENCE
    writer.Write(
        "metadata,timing_boundary,cpu=prepare+schedule_and_klu+validation;cuda="
        "prepare+executor_create+executor_overhead+context_setup+library_setup+"
        "host_pack+upload+matrix_setup+analysis_submit+analysis_sync+"
        "factor_solve_submit+factor_solve_sync+readback+phase_status+"
        "memory_accounting+result_assembly+validation");
#else
    writer.Write(
        "metadata,timing_boundary,prepare+schedule_and_klu+validation");
#endif
    writer.Write(
        "metadata,validation_boundary,residual+fresh_cpu_klu+differential");
#ifdef OHMNIVORE_GPU02_EVIDENCE
    writer.Write("metadata,memory_boundary," +
                 Csv("isolated_child_ru_maxrss+max(controlled_plus_cudss_peak,"
                     "cuda_mem_info_batch_delta)"));
    writer.Write("metadata,cuda_event_boundary,device_telemetry_not_added_to_"
                 "end_to_end_total");
    writer.Write(
        "metadata,executor_overhead_boundary,complete_execute_wall_minus_"
        "all_classified_setup_pack_upload_device_sync_status_memory_readback_"
        "and_result_intervals");
    writer.Write(
        "metadata,release_boundary,owner_thread_release_timed_and_allocation_"
        "balanced_but_excluded_symmetrically_with_cpu_destructors");
    writer.Write(
        "metadata,primary_context_memory_boundary,not_observable_before_cuda_"
        "runtime_initialization_and_excluded_from_batch_gate");
#else
    writer.Write("metadata,memory_boundary,isolated_child_process_ru_maxrss");
#endif
    writer.Write("metadata,fork_overhead_in_timing,false");
    writer.Write("metadata,quantile_rule,nearest_rank_sorted_ceil_pn_minus_1");
    writer.Write("metadata,target_cpu," + Csv(environment.target_cpu));
    writer.Write("metadata,target_hardware_threads,32");
    writer.Write("metadata,target_gpu," + Csv(environment.target_gpu));
    writer.Write("metadata,target_driver_api," + environment.target_driver_api);
    writer.Write("metadata,target_cuda_runtime," +
                 environment.target_cuda_runtime);
    writer.Write("metadata,observed_kernel," + Csv(environment.kernel));
    writer.Write("metadata,observed_cpu," + Csv(environment.cpu));
    writer.Write("metadata,observed_hardware_threads," +
                 std::to_string(environment.hardware_threads));
    writer.Write("metadata,observed_gpu," + Csv(environment.observed_gpu));
    writer.Write("metadata,observed_driver," +
                 Csv(environment.observed_driver));
#ifdef OHMNIVORE_GPU02_EVIDENCE
    writer.Write("metadata,observed_gpu_uuid," +
                 Csv(environment.observed_gpu_uuid));
    writer.Write("metadata,observed_compute_capability," +
                 environment.observed_compute_capability);
    writer.Write("metadata,observed_driver_api," +
                 std::to_string(environment.observed_driver_api));
    writer.Write("metadata,observed_cuda_runtime_api," +
                 std::to_string(environment.observed_cuda_runtime));
#endif
    writer.Write("metadata,warmups," + std::to_string(options.warmups));
    writer.Write("metadata,repetitions," + std::to_string(options.repetitions));
#ifdef OHMNIVORE_GPU02_EVIDENCE
    writer.Write("metadata,gpu_memory_gate_bytes,2147483648");
    writer.Write("metadata,warmup_minimum,3");
    writer.Write("metadata,repetition_minimum,20");
#else
    writer.Write("metadata,gpu_memory_bytes,0");
#endif
    writer.Write(
        "identity,schema_version,manifest_fingerprint,case_id,"
        "structure_fingerprint,batch_fingerprint,aggregate_member_fingerprint,"
        "contract_version,replay_id,circuit_id,corner_id,ordinal,frequency_hz,"
        "frequency_hz_bits,content_fingerprint");
    std::string sample_header =
        "sample,schema_version,manifest_fingerprint,case_id,workload_class,"
        "topology,mode,build_mode,compiler,bazel_pin,kernel,cpu,target_cpu,"
        "target_gpu,target_driver_api,target_cuda_runtime,observed_gpu,"
        "observed_driver,hardware_threads,requested_threads,"
        "participating_threads,warmups,repetitions,repetition,node_count,"
        "branch_count,dimension,nnz,sweep_type,sweep_points,batch_size,"
        "start_frequency_hz,stop_frequency_hz,g_series,g_shunt,c_series,"
        "c_shunt,reuse_count,structure_fingerprint,batch_fingerprint,"
        "aggregate_member_fingerprint,preparation_ns,schedule_and_klu_ns,"
        "validation_ns,total_ns,scheduled_klu_solves,validation_klu_solves,"
        "actual_klu_solves,throughput_members_per_second,cpu_peak_rss_bytes";
#ifdef OHMNIVORE_GPU02_EVIDENCE
    sample_header +=
        ",executor_create_ns,executor_execute_ns,executor_overhead_ns,"
        "context_setup_ns,library_setup_ns,host_pack_ns,upload_ns,"
        "matrix_setup_ns,analysis_submit_ns,analysis_sync_ns,"
        "factor_solve_submit_ns,factor_solve_sync_ns,readback_ns,"
        "phase_status_ns,memory_accounting_ns,result_assembly_ns,"
        "analysis_device_ns,factor_solve_device_ns,cuda_factorizations,"
        "cuda_solves,cuda_factorization_calls,cuda_solve_calls,"
        "cuda_uniform_batch_size,gpu_peak_batch_memory_bytes,"
        "gpu_post_context_setup_"
        "memory_bytes,"
        "cudss_estimated_peak_device_bytes,cudss_observed_peak_device_bytes,"
        "release_ns,release_count,outstanding_device_bytes_after_release,"
        "failure_count";
#else
    sample_header += ",gpu_memory_bytes,failure_count";
#endif
    writer.Write(sample_header);
    writer.Write(
        "summary,case_id,workload_class,topology,mode,sample_count,min_total_"
        "ns,"
        "p25_total_ns,median_total_ns,p75_total_ns,p95_total_ns,max_total_ns,"
        "median_throughput_members_per_second,quantile_rule"
#ifdef OHMNIVORE_GPU02_EVIDENCE
        ",peak_gpu_memory_bytes"
#endif
        ",failure_count");
#ifdef OHMNIVORE_GPU02_EVIDENCE
    writer.Write(
        "verdict,case_id,workload_class,parallel_prepared_median_ns,"
        "gpu_prepared_median_ns,median_ratio,median_bound,"
        "parallel_prepared_p95_ns,gpu_prepared_p95_ns,p95_ratio,p95_bound,"
        "parallel_cold_median_ns,gpu_cold_median_ns,cold_ratio,cold_bound,"
        "gpu_peak_batch_memory_bytes,memory_bound,run_verdict,dispatch_status");
#endif

    std::size_t identity_count = 0;
    std::size_t sample_count = 0;
    std::size_t summary_count = 0;
#ifdef OHMNIVORE_GPU02_EVIDENCE
    std::size_t verdict_count = 0;
#endif
    for (const PreparedAcReplayCase &item : corpus.cases) {
      const PreparedAcBatch prepared = PrepareCase(item);
      WriteIdentityRecords(&writer, corpus, item, prepared, &identity_count);
      const PreparedAcBatchResult authority = BuildAuthority(item);
      const std::size_t requested = options.requested_threads == 0
                                        ? environment.hardware_threads
                                        : options.requested_threads;
      const std::size_t participating =
          std::max<std::size_t>(1, std::min(requested, item.batch_size));

#ifdef OHMNIVORE_GPU02_EVIDENCE
      struct Mode {
        std::string name;
        bool parallel = false;
        bool prepared_reuse = false;
        bool cuda = false;
      };
      std::optional<SummaryStatistics> parallel_cold;
      std::optional<SummaryStatistics> parallel_prepared;
      std::optional<SummaryStatistics> gpu_cold;
      std::optional<SummaryStatistics> gpu_prepared;
      const std::vector<Mode> modes = {
          {"single_cold", false, false, false},
          {"single_prepared", false, true, false},
          {"parallel_cold", true, false, false},
          {"parallel_prepared", true, true, false},
          {"gpu_cold", false, false, true},
          {"gpu_prepared", false, true, true},
      };
      for (const Mode &mode : modes) {
#else
      for (const auto &[mode, parallel, prepared_reuse] :
           std::vector<std::tuple<std::string, bool, bool>>{
               {"single_cold", false, false},
               {"single_prepared", false, true},
               {"parallel_cold", true, false},
               {"parallel_prepared", true, true}}) {
#endif
        for (std::size_t warmup = 0; warmup < options.warmups; ++warmup) {
#ifdef OHMNIVORE_GPU02_EVIDENCE
          static_cast<void>(
              mode.cuda
                  ? MeasureGpuInChild(item, authority, mode.prepared_reuse)
                  : MeasureInChild(item, authority, mode.parallel,
                                   mode.prepared_reuse, participating));
#else
          static_cast<void>(MeasureInChild(item, authority, parallel,
                                           prepared_reuse, participating));
#endif
        }
        std::vector<TimingSample> samples;
        samples.reserve(options.repetitions);
        for (std::size_t repetition = 0; repetition < options.repetitions;
             ++repetition) {
#ifdef OHMNIVORE_GPU02_EVIDENCE
          TimingSample sample =
              mode.cuda
                  ? MeasureGpuInChild(item, authority, mode.prepared_reuse)
                  : MeasureInChild(item, authority, mode.parallel,
                                   mode.prepared_reuse, participating);
          writer.Write(SampleLine(corpus, item, prepared, environment, options,
                                  mode.name, repetition, sample));
#else
          TimingSample sample = MeasureInChild(item, authority, parallel,
                                               prepared_reuse, participating);
          writer.Write(SampleLine(corpus, item, prepared, environment, options,
                                  mode, repetition, sample));
#endif
          ++sample_count;
          samples.push_back(sample);
        }
#ifdef OHMNIVORE_GPU02_EVIDENCE
        writer.Write(SummaryLine(item, mode.name, samples));
        const SummaryStatistics summary = Summarize(samples);
        if (mode.name == "parallel_cold") {
          parallel_cold = summary;
        } else if (mode.name == "parallel_prepared") {
          parallel_prepared = summary;
        } else if (mode.name == "gpu_cold") {
          gpu_cold = summary;
        } else if (mode.name == "gpu_prepared") {
          gpu_prepared = summary;
        }
#else
        writer.Write(SummaryLine(item, mode, samples));
#endif
        ++summary_count;
      }
#ifdef OHMNIVORE_GPU02_EVIDENCE
      if (!parallel_cold.has_value() || !parallel_prepared.has_value() ||
          !gpu_cold.has_value() || !gpu_prepared.has_value()) {
        throw std::runtime_error("GPU-02 verdict inputs are incomplete");
      }
      writer.Write(VerdictLine(item, *parallel_cold, *parallel_prepared,
                               *gpu_cold, *gpu_prepared));
      ++verdict_count;
#endif
    }

    const std::size_t expected_identities = 16 + 61 + 121 + 517;
    const std::size_t expected_samples =
#ifdef OHMNIVORE_GPU02_EVIDENCE
        corpus.cases.size() * 6 * options.repetitions;
    const std::size_t expected_summaries = corpus.cases.size() * 6;
    const std::size_t expected_verdicts = corpus.cases.size();
#else
        corpus.cases.size() * 4 * options.repetitions;
    const std::size_t expected_summaries = corpus.cases.size() * 4;
#endif
    if (identity_count != expected_identities ||
        sample_count != expected_samples || summary_count != expected_summaries
#ifdef OHMNIVORE_GPU02_EVIDENCE
        || verdict_count != expected_verdicts
#endif
    ) {
      throw std::runtime_error("evidence record counts are incomplete");
    }
    const std::string records_fingerprint = writer.Fingerprint();
    std::ostringstream completion;
    completion << "completion,complete," << start_utc << ',' << UtcNow() << ','
               << expected_identities << ',' << identity_count << ','
               << expected_samples << ',' << sample_count << ','
               << expected_summaries << ',' << summary_count
#ifdef OHMNIVORE_GPU02_EVIDENCE
               << ',' << expected_verdicts << ',' << verdict_count
#endif
               << ",0," << records_fingerprint;
    writer.WriteCompletion(completion.str());
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "prepared AC replay benchmark failed: " << error.what()
              << '\n';
    return 1;
  }
}
