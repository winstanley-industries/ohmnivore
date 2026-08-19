#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <complex>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
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

#include <cuda_runtime.h>

#include "cpp/benchmarks/prepared_ac_session.h"
#include "cuda/prepared_ac_cuda.h"
#include "ohmnivore/prepared_ac.h"
#include "ohmnivore/solver.h"
#include "ohmnivore/status.h"

#ifndef OHMNIVORE_REPOSITORY_BAZEL_PIN
#define OHMNIVORE_REPOSITORY_BAZEL_PIN "unavailable"
#endif

namespace {

using Clock = std::chrono::steady_clock;
using ohmnivore::CudaPreparedAcBatchBackend;
using ohmnivore::CudaPreparedAcStatistics;
using ohmnivore::Error;
using ohmnivore::ErrorCode;
using ohmnivore::PreparedAcBatch;
using ohmnivore::PreparedAcBatchResult;
using ohmnivore::PreparedAcResultMember;
using ohmnivore::Result;
using ohmnivore::SparseComplexFactorization;
using ohmnivore::benchmarks::PreparedAcSessionCase;

inline constexpr std::size_t kMaximumCorners = 64;
inline constexpr std::size_t kMaximumPersistentSessions = 64;
inline constexpr std::uint32_t kChildMagic = 0x47505332U;
inline constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
inline constexpr std::uint64_t kFnvOffsetFirst = 14695981039346656037ULL;
inline constexpr std::uint64_t kFnvOffsetSecond = 9521211207457086692ULL;
inline constexpr std::uint64_t kGpuMemoryGateBytes =
    2ULL * 1024ULL * 1024ULL * 1024ULL;

[[nodiscard]] std::uint64_t ElapsedNanoseconds(Clock::time_point start) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start)
          .count());
}

struct Options {
  std::string corpus_path;
  std::vector<std::string> source_paths;
  std::size_t warmups = 3;
  std::size_t repetitions = 20;
  std::size_t persistent_sessions = 20;
  std::string case_filter;
  bool diagnostic = false;
};

[[nodiscard]] std::size_t ParseSize(std::string_view text,
                                    std::string_view option) {
  std::size_t value = 0;
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (text.empty() || parsed.ec != std::errc{} ||
      parsed.ptr != text.data() + text.size()) {
    throw std::runtime_error(std::string(option) + " requires an integer");
  }
  return value;
}

[[nodiscard]] Options ParseOptions(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    constexpr std::string_view kCorpus = "--corpus=";
    constexpr std::string_view kSource = "--source=";
    constexpr std::string_view kWarmups = "--warmups=";
    constexpr std::string_view kRepetitions = "--repetitions=";
    constexpr std::string_view kPersistentSessions = "--persistent-sessions=";
    constexpr std::string_view kCase = "--case=";
    if (argument.starts_with(kCorpus)) {
      options.corpus_path = std::string(argument.substr(kCorpus.size()));
    } else if (argument.starts_with(kSource)) {
      options.source_paths.emplace_back(argument.substr(kSource.size()));
    } else if (argument.starts_with(kWarmups)) {
      options.warmups =
          ParseSize(argument.substr(kWarmups.size()), "--warmups");
    } else if (argument.starts_with(kRepetitions)) {
      options.repetitions =
          ParseSize(argument.substr(kRepetitions.size()), "--repetitions");
    } else if (argument.starts_with(kPersistentSessions)) {
      options.persistent_sessions = ParseSize(
          argument.substr(kPersistentSessions.size()), "--persistent-sessions");
    } else if (argument.starts_with(kCase)) {
      options.case_filter = std::string(argument.substr(kCase.size()));
    } else if (argument == "--diagnostic") {
      options.diagnostic = true;
    } else {
      throw std::runtime_error("unknown option: " + std::string(argument));
    }
  }
  if (options.corpus_path.empty()) {
    throw std::runtime_error("--corpus is required");
  }
  if (options.repetitions == 0 || options.persistent_sessions == 0 ||
      options.persistent_sessions > kMaximumPersistentSessions ||
      (!options.diagnostic &&
       (options.warmups < 3 || options.repetitions < 20 ||
        options.persistent_sessions < 20))) {
    throw std::runtime_error(
        "GPU-02S evidence requires at least 3 warmups, 20 repetitions, and "
        "20 persistent steady sessions");
  }
  return options;
}

class FingerprintBuilder {
public:
  explicit FingerprintBuilder(std::string_view domain) { AddString(domain); }

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

[[nodiscard]] std::string
FingerprintFiles(const std::vector<std::string> &paths,
                 std::string_view domain) {
  FingerprintBuilder fingerprint(domain);
  fingerprint.AddUint64(paths.size());
  for (const std::string &path : paths) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
      throw std::runtime_error("could not read fingerprint input: " + path);
    }
    const std::string bytes((std::istreambuf_iterator<char>(input)),
                            std::istreambuf_iterator<char>());
    fingerprint.AddString(path);
    fingerprint.AddString(bytes);
  }
  return fingerprint.Finish();
}

[[nodiscard]] std::string Csv(std::string_view value) {
  if (value.find_first_of(",\"\n\r") == std::string_view::npos) {
    return std::string(value);
  }
  std::string escaped = "\"";
  for (const char character : value) {
    if (character == '\"') {
      escaped += "\"\"";
    } else {
      escaped.push_back(character);
    }
  }
  escaped.push_back('\"');
  return escaped;
}

class EvidenceWriter {
public:
  void Write(std::string line) {
    fingerprint_.AddString(line);
    ++records_;
    std::cout << line << '\n';
  }

  [[nodiscard]] std::string fingerprint() const {
    return fingerprint_.Finish();
  }

  [[nodiscard]] std::size_t records() const { return records_; }

private:
  FingerprintBuilder fingerprint_{"gpu02s-evidence-records-v1"};
  std::size_t records_ = 0;
};

enum class BackendKind : std::uint32_t { kParallelCpu, kCuda };
enum class ValidationLane : std::uint32_t {
  kInlineCertified,
  kCandidateRuntime
};

[[nodiscard]] std::string_view ModeName(BackendKind backend,
                                        ValidationLane lane) {
  if (backend == BackendKind::kParallelCpu) {
    return lane == ValidationLane::kInlineCertified
               ? "parallel_inline_certified"
               : "parallel_candidate_runtime";
  }
  return lane == ValidationLane::kInlineCertified ? "gpu_inline_certified"
                                                  : "gpu_candidate_runtime";
}

struct PrefixSample {
  std::uint64_t preparation_ns = 0;
  std::uint64_t executor_create_ns = 0;
  std::uint64_t execute_ns = 0;
  std::uint64_t runtime_validation_ns = 0;
  std::uint64_t excluded_certification_ns = 0;
  std::uint64_t lane_total_ns = 0;
  std::uint64_t certified_total_ns = 0;
  std::uint64_t first_corner_ns = 0;
  std::uint64_t later_corner_median_ns = 0;
  std::uint64_t later_corner_p95_ns = 0;
  std::uint64_t accepted_members = 0;
  std::uint64_t context_setup_ns = 0;
  std::uint64_t library_setup_ns = 0;
  std::uint64_t host_pack_ns = 0;
  std::uint64_t upload_ns = 0;
  std::uint64_t matrix_setup_ns = 0;
  std::uint64_t analysis_ns = 0;
  std::uint64_t factor_solve_ns = 0;
  std::uint64_t analysis_device_ns = 0;
  std::uint64_t factor_solve_device_ns = 0;
  std::uint64_t readback_ns = 0;
  std::uint64_t phase_status_ns = 0;
  std::uint64_t memory_accounting_ns = 0;
  std::uint64_t result_assembly_ns = 0;
  std::uint64_t cuda_execute_phase_remainder_ns = 0;
};

struct SessionSample {
  std::uint32_t magic = kChildMagic;
  std::uint32_t success = 0;
  std::uint32_t prefix_count = 0;
  std::uint32_t participating_threads = 0;
  PrefixSample prefixes[kMaximumCorners]{};
  std::uint64_t cpu_symbolic_analyses = 0;
  std::uint64_t backend_solves = 0;
  std::uint64_t certification_solves = 0;
  std::uint64_t structure_preparations = 0;
  std::uint64_t structure_uploads = 0;
  std::uint64_t values_rhs_uploads = 0;
  std::uint64_t same_structure_refreshes = 0;
  std::uint64_t cuda_analyses = 0;
  std::uint64_t factorization_calls = 0;
  std::uint64_t solve_calls = 0;
  std::uint64_t context_setup_ns = 0;
  std::uint64_t library_setup_ns = 0;
  std::uint64_t host_pack_ns = 0;
  std::uint64_t upload_ns = 0;
  std::uint64_t matrix_setup_ns = 0;
  std::uint64_t analysis_ns = 0;
  std::uint64_t factor_solve_ns = 0;
  std::uint64_t analysis_device_ns = 0;
  std::uint64_t factor_solve_device_ns = 0;
  std::uint64_t readback_ns = 0;
  std::uint64_t phase_status_ns = 0;
  std::uint64_t memory_accounting_ns = 0;
  std::uint64_t result_assembly_ns = 0;
  std::uint64_t execute_phase_remainder_ns = 0;
  std::uint64_t release_ns = 0;
  std::uint64_t cpu_peak_rss_bytes = 0;
  std::uint64_t gpu_peak_batch_bytes = 0;
  std::uint64_t outstanding_device_bytes = 0;
  char error[512]{};
};

struct PersistentDiagnostic {
  std::uint32_t magic = kChildMagic;
  std::uint32_t success = 0;
  std::uint32_t sample_count = 0;
  SessionSample cold{};
  SessionSample steady[kMaximumPersistentSessions]{};
  std::uint64_t release_ns = 0;
  std::uint64_t outstanding_device_bytes = 0;
  char error[512]{};
};

[[nodiscard]] std::uint64_t PeakResidentBytes() {
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0) {
    throw std::runtime_error("getrusage failed");
  }
  return static_cast<std::uint64_t>(usage.ru_maxrss) * 1024ULL;
}

[[nodiscard]] std::uint64_t Quantile(std::vector<std::uint64_t> values,
                                     std::size_t numerator,
                                     std::size_t denominator) {
  if (values.empty()) {
    return 0;
  }
  std::sort(values.begin(), values.end());
  const std::size_t rank =
      (numerator * values.size() + denominator - 1) / denominator;
  return values[rank == 0 ? 0 : rank - 1];
}

struct GpuEnvironment {
  std::uint32_t magic = kChildMagic;
  std::uint32_t success = 0;
  int driver_api = 0;
  int runtime_api = 0;
  int compute_major = 0;
  int compute_minor = 0;
  std::uint64_t global_memory_bytes = 0;
  char name[256]{};
  char uuid[64]{};
  char error[256]{};
};

struct Summary {
  std::uint64_t minimum_ns = 0;
  std::uint64_t median_ns = 0;
  std::uint64_t p95_ns = 0;
  std::uint64_t maximum_ns = 0;
  std::uint64_t certified_median_ns = 0;
  std::uint64_t first_median_ns = 0;
  std::uint64_t steady_median_ns = 0;
  std::uint64_t steady_p95_ns = 0;
  std::uint64_t peak_cpu_bytes = 0;
  std::uint64_t peak_gpu_bytes = 0;
  double median_throughput = 0.0;
};

struct SummaryKey {
  std::string case_id;
  BackendKind backend;
  ValidationLane lane;
  std::size_t prefix;

  bool operator<(const SummaryKey &other) const {
    return std::tie(case_id, backend, lane, prefix) <
           std::tie(other.case_id, other.backend, other.lane, other.prefix);
  }
};

[[nodiscard]] SessionSample MeasureInChild(const PreparedAcSessionCase &item,
                                           BackendKind backend,
                                           ValidationLane lane);
[[nodiscard]] SessionSample MeasureCpuSession(const PreparedAcSessionCase &item,
                                              ValidationLane lane);
[[nodiscard]] SessionSample MeasureGpuSession(const PreparedAcSessionCase &item,
                                              ValidationLane lane);
[[nodiscard]] PersistentDiagnostic
MeasurePersistentInChild(const PreparedAcSessionCase &item, BackendKind backend,
                         ValidationLane lane, std::size_t steady_sessions);
[[nodiscard]] GpuEnvironment InspectGpuInChild();
[[nodiscard]] std::string UtcNow();
[[nodiscard]] std::string KernelIdentity();
[[nodiscard]] std::string CpuIdentity();
[[nodiscard]] Summary Summarize(const std::vector<SessionSample> &samples,
                                std::size_t prefix);

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = ParseOptions(argc, argv);
    auto corpus =
        ohmnivore::benchmarks::LoadPreparedAcSessionCorpus(options.corpus_path);
    if (!corpus.ok()) {
      throw std::runtime_error(corpus.error().message);
    }
    std::vector<PreparedAcSessionCase> cases;
    for (const PreparedAcSessionCase &item : corpus.value().cases) {
      if (options.case_filter.empty() || options.case_filter == item.case_id) {
        cases.push_back(item);
      }
    }
    if (cases.empty()) {
      throw std::runtime_error("--case did not select a session workload");
    }
    std::size_t corners_per_mode = 0;
    for (const PreparedAcSessionCase &item : cases) {
      corners_per_mode += item.corner_count;
    }
    constexpr std::size_t kModeCount = 4;
    const std::size_t expected_raw_sessions =
        cases.size() * kModeCount * options.repetitions;
    const std::size_t expected_raw_prefixes =
        corners_per_mode * kModeCount * options.repetitions;
    const std::size_t expected_persistent_samples =
        cases.size() * kModeCount * (options.persistent_sessions + 1);

    const GpuEnvironment gpu = InspectGpuInChild();
    std::ostringstream command;
    for (int index = 0; index < argc; ++index) {
      if (index != 0) {
        command << ' ';
      }
      command << argv[index];
    }
    const std::string source_fingerprint =
        FingerprintFiles(options.source_paths, "gpu02s-source-inputs-v1");
    const std::string binary_fingerprint =
        FingerprintFiles({"/proc/self/exe"}, "gpu02s-binary-v1");

    EvidenceWriter writer;
    writer.Write("metadata,key,value");
    writer.Write("metadata,evidence_contract,gpu02s-persistent-session-v1");
    writer.Write("metadata,base_revision,"
                 "1d0625589a6bb5b7744fe7319183c55a3032ff10");
    writer.Write("metadata,status,incomplete_until_completion_record");
    writer.Write(std::string("metadata,evidence_kind,") +
                 (options.diagnostic ? "diagnostic" : "canonical"));
    writer.Write("metadata,automatic_dispatch_authorized,false");
    writer.Write("metadata,production_acceptance_authorized,false");
    writer.Write("metadata,speedup_claim,false");
    writer.Write("metadata,workload_representation,compiler-derived synthetic "
                 "session envelope");
    writer.Write("metadata,command," + Csv(command.str()));
    writer.Write("metadata,start_utc," + UtcNow());
    writer.Write("metadata,manifest_fingerprint," +
                 corpus.value().manifest_fingerprint);
    writer.Write("metadata,source_fingerprint," + source_fingerprint);
    writer.Write("metadata,binary_fingerprint," + binary_fingerprint);
    writer.Write("metadata,source_input_count," +
                 std::to_string(options.source_paths.size()));
    writer.Write("metadata,source_identity_scope,governing_docs+benchmark+"
                 "session_corpus+prepared_contract+compiler+solver+cuda_"
                 "executor+bazel_module_lock+vendor_build_inputs");
    writer.Write("metadata,bazel_version," +
                 std::string(OHMNIVORE_REPOSITORY_BAZEL_PIN));
    writer.Write("metadata,compiler," + Csv(__VERSION__));
    writer.Write("metadata,build_mode,optimized_cuda");
    writer.Write("metadata,bazel_configuration,opt+cuda");
    writer.Write("metadata,canonical_bazel_invocation,bazel run -c opt "
                 "--config=cuda //cuda:gpu02s_session_benchmark -- "
                 "--warmups=3 --repetitions=20 --persistent-sessions=20");
    writer.Write("metadata,host_toolchain,nvcc-13.0.88+gcc-15.2.0");
    writer.Write("metadata,suitesparse_version,7.12.3");
    writer.Write(
        "metadata,suitesparse_archive_sha256,"
        "158ee4ed2ce3fdcbf52c4e47e94b0d1a8ae13344b4a835991d78a3ad20f08086");
    writer.Write("metadata,klu_version,2.3.6");
    writer.Write("metadata,cudss_version,0.8.0.10");
    writer.Write(
        "metadata,cudss_archive_sha256,"
        "ba18f5fd80dcbbe905d158caac5b3061d848442bb5abd477b5f296b4257a4937");
    writer.Write("metadata,cudss_execution_shape,uniform_batch_all_members");
    writer.Write("metadata,cudss_matching,disabled");
    writer.Write("metadata,cudss_deterministic_mode,disabled");
    writer.Write("metadata,cudss_fp64_iterative_refinement_steps,1");
    writer.Write("metadata,cudss_reordering,default");
    writer.Write("metadata,cudss_factorization,default");
    writer.Write("metadata,cuda_toolkit_redist_version,13.0.2");
    writer.Write("metadata,nvcc_redist_version,13.0.88");
    writer.Write("metadata,cublas_redist_version,13.1.0.3");
    writer.Write(
        "metadata,cublas_archive_sha256,"
        "88bc951efd906032a371153ca61975e0d9c4761e4012169169a6b3a47931606e");
    writer.Write("metadata,cuda_architectures,compute_120+sm_120");
    writer.Write("metadata,cuda_runtime_linkage,static");
    writer.Write("metadata,kernel," + Csv(KernelIdentity()));
    writer.Write("metadata,cpu," + Csv(CpuIdentity()));
    writer.Write("metadata,hardware_threads," +
                 std::to_string(std::thread::hardware_concurrency()));
    writer.Write("metadata,gpu," + Csv(gpu.name));
    writer.Write("metadata,gpu_uuid," + std::string(gpu.uuid));
    writer.Write("metadata,compute_capability," +
                 std::to_string(gpu.compute_major) + "." +
                 std::to_string(gpu.compute_minor));
    writer.Write("metadata,driver_api," + std::to_string(gpu.driver_api));
    writer.Write("metadata,runtime_api," + std::to_string(gpu.runtime_api));
    writer.Write("metadata,gpu_global_memory_bytes," +
                 std::to_string(gpu.global_memory_bytes));
    writer.Write("metadata,warmups," + std::to_string(options.warmups));
    writer.Write("metadata,repetitions," + std::to_string(options.repetitions));
    writer.Write("metadata,persistent_steady_sessions," +
                 std::to_string(options.persistent_sessions));
    writer.Write("metadata,expected_raw_sessions," +
                 std::to_string(expected_raw_sessions));
    writer.Write("metadata,expected_raw_prefixes," +
                 std::to_string(expected_raw_prefixes));
    writer.Write("metadata,expected_persistent_samples," +
                 std::to_string(expected_persistent_samples));
    writer.Write("metadata,quantile_rule,nearest_rank_sorted_ceil_pn_minus_1");
    writer.Write("metadata,clock,std::chrono::steady_clock");
    writer.Write("metadata,fork_overhead_in_timing,false");
    writer.Write("metadata,memory_boundary," +
                 Csv("isolated_child_ru_maxrss+max("
                     "controlled_plus_cudss_peak,cuda_mem_info_batch_delta)"));
    writer.Write("metadata,validation_boundary,all_results_residual_checked+"
                 "fresh_cpu_klu_componentwise_certified");
    writer.Write("metadata,cuda_phase_boundary,host_intervals_nonoverlapping;"
                 "remainder=complete_execute_wall_minus_classified_intervals");
    writer.Write("metadata,cuda_device_telemetry_boundary,event_elapsed_"
                 "overlaps_host_submit_and_sync_intervals");
    writer.Write("metadata,cold_boundary,fresh_child_one_context_per_session");
    writer.Write("metadata,steady_boundary,later_same-structure corners");
    writer.Write(
        "metadata,candidate_boundary,execute+residual_validation;fresh "
        "KLU certification measured separately and mandatory");
    writer.Write("metadata,inline_boundary,execute+complete fresh KLU "
                 "certification");
    writer.Write("metadata,technical_crossover_thresholds,median=1.25;p95=1.10;"
                 "gpu_peak_bytes=2147483648");
    writer.Write("sample,case_id,workload_class,mode,repetition,prefix_corners,"
                 "dimension,nnz,batch,accepted_members,participating_threads,"
                 "preparation_ns,executor_create_ns,execute_ns,"
                 "runtime_validation_ns,excluded_certification_ns,"
                 "lane_total_ns,certified_total_ns,first_corner_ns,"
                 "later_corner_median_ns,later_corner_p95_ns,"
                 "cpu_symbolic_analyses,backend_solves,certification_solves,"
                 "structure_preparations,structure_uploads,values_rhs_uploads,"
                 "same_structure_refreshes,cuda_analyses,"
                 "factorization_calls,solve_calls,context_setup_ns,"
                 "library_setup_ns,host_pack_ns,upload_ns,matrix_setup_ns,"
                 "analysis_ns,factor_solve_ns,analysis_device_ns,"
                 "factor_solve_device_ns,readback_ns,phase_status_ns,"
                 "memory_accounting_ns,result_assembly_ns,release_ns,"
                 "cuda_execute_phase_remainder_ns,cpu_peak_rss_bytes,"
                 "gpu_peak_batch_bytes,outstanding_device_bytes,failures");
    writer.Write(
        "persistent_sample,case_id,workload_class,mode,phase,ordinal,"
        "corner_count,accepted_members,participating_threads,"
        "cpu_symbolic_analyses,backend_solves,certification_solves,"
        "preparation_ns,executor_create_ns,"
        "execute_ns,runtime_validation_ns,excluded_certification_ns,"
        "lane_total_ns,certified_total_ns,"
        "first_corner_ns,later_corner_median_ns,later_corner_p95_ns,"
        "context_setup_ns,library_setup_ns,structure_preparations,"
        "structure_uploads,values_rhs_uploads,same_structure_refreshes,"
        "cuda_analyses,factorization_calls,solve_calls,"
        "host_pack_ns,upload_ns,matrix_setup_ns,analysis_ns,factor_solve_ns,"
        "analysis_device_ns,factor_solve_device_ns,readback_ns,"
        "phase_status_ns,memory_accounting_ns,result_assembly_ns,"
        "cuda_execute_phase_remainder_ns,cpu_peak_rss_bytes,gpu_peak_batch_"
        "bytes,"
        "failures");
    writer.Write("identity,case_id,corner_ordinal,structure_fingerprint,"
                 "batch_fingerprint,aggregate_member_fingerprint,member_count");
    writer.Write("case,case_id,workload_class,topology,node_count,branch_count,"
                 "dimension,nnz,batch,corner_count");
    writer.Write("summary,case_id,workload_class,mode,prefix_corners,"
                 "sample_count,min_total_ns,median_total_ns,p95_total_ns,"
                 "max_total_ns,certified_median_ns,first_corner_median_ns,"
                 "later_corner_median_ns,later_corner_p95_ns,"
                 "median_throughput_members_per_second,peak_cpu_bytes,"
                 "peak_gpu_bytes");
    writer.Write("crossover,case_id,lane,status,first_prefix_corners,"
                 "first_accepted_members");
    writer.Write("persistent_summary,case_id,workload_class,mode,"
                 "cold_total_ns,steady_sample_count,steady_min_ns,"
                 "steady_median_ns,steady_p95_ns,steady_max_ns,"
                 "steady_certified_median_ns,"
                 "steady_median_throughput_members_per_second,"
                 "steady_peak_cpu_bytes,steady_peak_gpu_bytes,"
                 "cold_to_steady_ratio,release_ns,outstanding_device_bytes");

    std::map<SummaryKey, Summary> summaries;
    const std::array<BackendKind, 2> backends = {BackendKind::kParallelCpu,
                                                 BackendKind::kCuda};
    const std::array<ValidationLane, 2> lanes = {
        ValidationLane::kInlineCertified, ValidationLane::kCandidateRuntime};
    std::size_t raw_session_count = 0;
    std::size_t raw_prefix_count = 0;
    for (const PreparedAcSessionCase &item : cases) {
      writer.Write("case," + item.case_id + "," + item.workload_class + "," +
                   item.topology + "," + std::to_string(item.node_count) + "," +
                   std::to_string(item.branch_count) + "," +
                   std::to_string(item.dimension) + "," +
                   std::to_string(item.union_nonzeros) + "," +
                   std::to_string(item.batch_size) + "," +
                   std::to_string(item.corner_count));
      for (std::size_t corner = 0; corner < item.corner_count; ++corner) {
        auto prepared =
            ohmnivore::benchmarks::PrepareAcSessionCorner(item, corner);
        if (!prepared.ok()) {
          throw std::runtime_error(
              item.case_id +
              " identity preparation failed: " + prepared.error().message);
        }
        auto valid = ohmnivore::ValidatePreparedAcBatch(prepared.value());
        if (!valid.ok()) {
          throw std::runtime_error(
              item.case_id +
              " identity validation failed: " + valid.error().message);
        }
        FingerprintBuilder members("gpu02s-member-identities-v1");
        for (const ohmnivore::PreparedAcMember &member :
             prepared.value().members) {
          members.AddString(member.identity.content_fingerprint);
        }
        writer.Write("identity," + item.case_id + "," + std::to_string(corner) +
                     "," + prepared.value().structure.fingerprint + "," +
                     prepared.value().batch_fingerprint + "," +
                     members.Finish() + "," +
                     std::to_string(prepared.value().members.size()));
      }
      for (const BackendKind backend : backends) {
        for (const ValidationLane lane : lanes) {
          for (std::size_t warmup = 0; warmup < options.warmups; ++warmup) {
            static_cast<void>(MeasureInChild(item, backend, lane));
          }
          std::vector<SessionSample> samples;
          samples.reserve(options.repetitions);
          for (std::size_t repetition = 0; repetition < options.repetitions;
               ++repetition) {
            SessionSample sample = MeasureInChild(item, backend, lane);
            if (sample.prefix_count != item.corner_count) {
              throw std::runtime_error(item.case_id +
                                       " session prefix count is incomplete");
            }
            ++raw_session_count;
            for (std::size_t prefix = 1; prefix <= item.corner_count;
                 ++prefix) {
              const PrefixSample &value = sample.prefixes[prefix - 1];
              const bool cuda = backend == BackendKind::kCuda;
              const std::uint64_t prefix_solves = prefix * item.batch_size;
              std::ostringstream line;
              line << "sample," << item.case_id << ',' << item.workload_class
                   << ',' << ModeName(backend, lane) << ',' << repetition << ','
                   << prefix << ',' << item.dimension << ','
                   << item.union_nonzeros << ',' << item.batch_size << ','
                   << value.accepted_members << ','
                   << sample.participating_threads << ','
                   << value.preparation_ns << ',' << value.executor_create_ns
                   << ',' << value.execute_ns << ','
                   << value.runtime_validation_ns << ','
                   << value.excluded_certification_ns << ','
                   << value.lane_total_ns << ',' << value.certified_total_ns
                   << ',' << value.first_corner_ns << ','
                   << value.later_corner_median_ns << ','
                   << value.later_corner_p95_ns << ','
                   << (cuda ? 0 : sample.cpu_symbolic_analyses) << ','
                   << prefix_solves << ',' << prefix_solves << ','
                   << (cuda ? 1 : 0) << ',' << (cuda ? 1 : 0) << ','
                   << (cuda ? prefix : 0) << ',' << (cuda ? prefix - 1 : 0)
                   << ',' << (cuda ? 1 : 0) << ',' << (cuda ? prefix : 0) << ','
                   << (cuda ? prefix : 0) << ',' << value.context_setup_ns
                   << ',' << value.library_setup_ns << ',' << value.host_pack_ns
                   << ',' << value.upload_ns << ',' << value.matrix_setup_ns
                   << ',' << value.analysis_ns << ',' << value.factor_solve_ns
                   << ',' << value.analysis_device_ns << ','
                   << value.factor_solve_device_ns << ',' << value.readback_ns
                   << ',' << value.phase_status_ns << ','
                   << value.memory_accounting_ns << ','
                   << value.result_assembly_ns << ','
                   << (prefix == item.corner_count ? sample.release_ns : 0)
                   << ',' << value.cuda_execute_phase_remainder_ns << ','
                   << sample.cpu_peak_rss_bytes << ','
                   << sample.gpu_peak_batch_bytes << ','
                   << (prefix == item.corner_count
                           ? sample.outstanding_device_bytes
                           : 0)
                   << ",0";
              writer.Write(line.str());
              ++raw_prefix_count;
            }
            samples.push_back(sample);
          }
          for (std::size_t prefix = 1; prefix <= item.corner_count; ++prefix) {
            const Summary summary = Summarize(samples, prefix);
            summaries.emplace(SummaryKey{.case_id = item.case_id,
                                         .backend = backend,
                                         .lane = lane,
                                         .prefix = prefix},
                              summary);
            std::ostringstream line;
            line << std::setprecision(17) << "summary," << item.case_id << ','
                 << item.workload_class << ',' << ModeName(backend, lane) << ','
                 << prefix << ',' << options.repetitions << ','
                 << summary.minimum_ns << ',' << summary.median_ns << ','
                 << summary.p95_ns << ',' << summary.maximum_ns << ','
                 << summary.certified_median_ns << ','
                 << summary.first_median_ns << ',' << summary.steady_median_ns
                 << ',' << summary.steady_p95_ns << ','
                 << summary.median_throughput << ',' << summary.peak_cpu_bytes
                 << ',' << summary.peak_gpu_bytes;
            writer.Write(line.str());
          }
        }
      }
    }

    writer.Write("verdict,case_id,workload_class,lane,prefix_corners,"
                 "cpu_median_ns,gpu_median_ns,median_ratio,median_pass,"
                 "cpu_p95_ns,gpu_p95_ns,p95_ratio,p95_pass,gpu_peak_bytes,"
                 "memory_pass,technical_crossover_this_run,"
                 "automatic_dispatch");
    for (const PreparedAcSessionCase &item : cases) {
      for (const ValidationLane lane : lanes) {
        std::optional<std::size_t> first_crossover;
        for (std::size_t prefix = 1; prefix <= item.corner_count; ++prefix) {
          const Summary &cpu = summaries.at(SummaryKey{
              item.case_id, BackendKind::kParallelCpu, lane, prefix});
          const Summary &gpu_summary = summaries.at(
              SummaryKey{item.case_id, BackendKind::kCuda, lane, prefix});
          const double median_ratio =
              static_cast<double>(cpu.median_ns) /
              static_cast<double>(gpu_summary.median_ns);
          const double p95_ratio = static_cast<double>(cpu.p95_ns) /
                                   static_cast<double>(gpu_summary.p95_ns);
          const bool median_pass = median_ratio >= 1.25;
          const bool p95_pass = p95_ratio >= 1.10;
          const bool memory_pass =
              gpu_summary.peak_gpu_bytes <= kGpuMemoryGateBytes;
          const bool control = item.workload_class == "session_control";
          const bool crossover =
              !control && median_pass && p95_pass && memory_pass;
          if (crossover && !first_crossover.has_value()) {
            first_crossover = prefix;
          }
          std::ostringstream line;
          line << std::setprecision(17) << "verdict," << item.case_id << ','
               << item.workload_class << ','
               << (lane == ValidationLane::kInlineCertified
                       ? "inline_certified"
                       : "candidate_runtime")
               << ',' << prefix << ',' << cpu.median_ns << ','
               << gpu_summary.median_ns << ',' << median_ratio << ','
               << (median_pass ? "pass" : "fail") << ',' << cpu.p95_ns << ','
               << gpu_summary.p95_ns << ',' << p95_ratio << ','
               << (p95_pass ? "pass" : "fail") << ','
               << gpu_summary.peak_gpu_bytes << ','
               << (memory_pass ? "pass" : "fail") << ','
               << (crossover ? "pass"
                   : control ? "ineligible"
                             : "fail")
               << ",unauthorized";
          writer.Write(line.str());
        }
        writer.Write(
            "crossover," + item.case_id + "," +
            (lane == ValidationLane::kInlineCertified ? "inline_certified"
                                                      : "candidate_runtime") +
            "," +
            (item.workload_class == "session_control" ? "ineligible"
             : first_crossover.has_value() ? "observed_this_run"
                                           : "not_observed_this_run") +
            "," +
            (first_crossover.has_value()
                 ? std::to_string(first_crossover.value())
                 : "none") +
            "," +
            (first_crossover.has_value()
                 ? std::to_string(first_crossover.value() * item.batch_size)
                 : "none"));
      }
    }

    std::size_t persistent_sample_count = 0;
    std::map<SummaryKey, Summary> persistent_summaries;
    for (const PreparedAcSessionCase &item : cases) {
      for (const BackendKind backend : backends) {
        for (const ValidationLane lane : lanes) {
          const PersistentDiagnostic diagnostic = MeasurePersistentInChild(
              item, backend, lane, options.persistent_sessions);
          if (diagnostic.sample_count != options.persistent_sessions) {
            throw std::runtime_error(item.case_id +
                                     " persistent diagnostic is incomplete");
          }
          const auto write_persistent = [&](const SessionSample &sample,
                                            std::string_view phase,
                                            std::size_t ordinal) {
            const PrefixSample &prefix = sample.prefixes[item.corner_count - 1];
            std::ostringstream line;
            line << "persistent_sample," << item.case_id << ','
                 << item.workload_class << ',' << ModeName(backend, lane) << ','
                 << phase << ',' << ordinal << ',' << item.corner_count << ','
                 << prefix.accepted_members << ','
                 << sample.participating_threads << ','
                 << sample.cpu_symbolic_analyses << ',' << sample.backend_solves
                 << ',' << sample.certification_solves << ','
                 << prefix.preparation_ns << ',' << prefix.executor_create_ns
                 << ',' << prefix.execute_ns << ','
                 << prefix.runtime_validation_ns << ','
                 << prefix.excluded_certification_ns << ','
                 << prefix.lane_total_ns << ',' << prefix.certified_total_ns
                 << ',' << prefix.first_corner_ns << ','
                 << prefix.later_corner_median_ns << ','
                 << prefix.later_corner_p95_ns << ',' << sample.context_setup_ns
                 << ',' << sample.library_setup_ns << ','
                 << sample.structure_preparations << ','
                 << sample.structure_uploads << ',' << sample.values_rhs_uploads
                 << ',' << sample.same_structure_refreshes << ','
                 << sample.cuda_analyses << ',' << sample.factorization_calls
                 << ',' << sample.solve_calls << ',' << sample.host_pack_ns
                 << ',' << sample.upload_ns << ',' << sample.matrix_setup_ns
                 << ',' << sample.analysis_ns << ',' << sample.factor_solve_ns
                 << ',' << sample.analysis_device_ns << ','
                 << sample.factor_solve_device_ns << ',' << sample.readback_ns
                 << ',' << sample.phase_status_ns << ','
                 << sample.memory_accounting_ns << ','
                 << sample.result_assembly_ns << ','
                 << sample.execute_phase_remainder_ns << ','
                 << sample.cpu_peak_rss_bytes << ','
                 << sample.gpu_peak_batch_bytes << ",0";
            writer.Write(line.str());
            ++persistent_sample_count;
          };
          write_persistent(diagnostic.cold, "cold", 0);
          std::vector<SessionSample> steady_samples;
          steady_samples.reserve(options.persistent_sessions);
          for (std::size_t session = 0; session < options.persistent_sessions;
               ++session) {
            write_persistent(diagnostic.steady[session], "steady", session + 1);
            steady_samples.push_back(diagnostic.steady[session]);
          }
          const Summary steady = Summarize(steady_samples, item.corner_count);
          persistent_summaries.emplace(SummaryKey{.case_id = item.case_id,
                                                  .backend = backend,
                                                  .lane = lane,
                                                  .prefix = item.corner_count},
                                       steady);
          const PrefixSample &cold =
              diagnostic.cold.prefixes[item.corner_count - 1];
          std::ostringstream line;
          line << std::setprecision(17) << "persistent_summary," << item.case_id
               << ',' << item.workload_class << ',' << ModeName(backend, lane)
               << ',' << cold.lane_total_ns << ','
               << options.persistent_sessions << ',' << steady.minimum_ns << ','
               << steady.median_ns << ',' << steady.p95_ns << ','
               << steady.maximum_ns << ',' << steady.certified_median_ns << ','
               << steady.median_throughput << ',' << steady.peak_cpu_bytes
               << ',' << steady.peak_gpu_bytes << ','
               << static_cast<double>(cold.lane_total_ns) /
                      static_cast<double>(steady.median_ns)
               << ',' << diagnostic.release_ns << ','
               << diagnostic.outstanding_device_bytes;
          writer.Write(line.str());
        }
      }
    }
    writer.Write("persistent_verdict,case_id,workload_class,lane,"
                 "cpu_steady_median_ns,gpu_steady_median_ns,median_ratio,"
                 "median_pass,cpu_steady_p95_ns,gpu_steady_p95_ns,p95_ratio,"
                 "p95_pass,gpu_peak_bytes,memory_pass,"
                 "technical_crossover_this_run,automatic_dispatch");
    for (const PreparedAcSessionCase &item : cases) {
      for (const ValidationLane lane : lanes) {
        const Summary &cpu = persistent_summaries.at(SummaryKey{
            item.case_id, BackendKind::kParallelCpu, lane, item.corner_count});
        const Summary &gpu_summary = persistent_summaries.at(SummaryKey{
            item.case_id, BackendKind::kCuda, lane, item.corner_count});
        const double median_ratio = static_cast<double>(cpu.median_ns) /
                                    static_cast<double>(gpu_summary.median_ns);
        const double p95_ratio = static_cast<double>(cpu.p95_ns) /
                                 static_cast<double>(gpu_summary.p95_ns);
        const bool median_pass = median_ratio >= 1.25;
        const bool p95_pass = p95_ratio >= 1.10;
        const bool memory_pass =
            gpu_summary.peak_gpu_bytes <= kGpuMemoryGateBytes;
        const bool control = item.workload_class == "session_control";
        const bool crossover =
            !control && median_pass && p95_pass && memory_pass;
        std::ostringstream line;
        line << std::setprecision(17) << "persistent_verdict," << item.case_id
             << ',' << item.workload_class << ','
             << (lane == ValidationLane::kInlineCertified ? "inline_certified"
                                                          : "candidate_runtime")
             << ',' << cpu.median_ns << ',' << gpu_summary.median_ns << ','
             << median_ratio << ',' << (median_pass ? "pass" : "fail") << ','
             << cpu.p95_ns << ',' << gpu_summary.p95_ns << ',' << p95_ratio
             << ',' << (p95_pass ? "pass" : "fail") << ','
             << gpu_summary.peak_gpu_bytes << ','
             << (memory_pass ? "pass" : "fail") << ','
             << (crossover ? "pass"
                 : control ? "ineligible"
                           : "fail")
             << ",unauthorized";
        writer.Write(line.str());
      }
    }
    if (raw_session_count != expected_raw_sessions ||
        raw_prefix_count != expected_raw_prefixes ||
        persistent_sample_count != expected_persistent_samples) {
      throw std::runtime_error("GPU-02S evidence record count is incomplete");
    }

    const std::size_t records_before_completion = writer.records();
    const std::string completion_fingerprint = writer.fingerprint();
    std::ostringstream completion;
    completion << "completion,"
               << (options.diagnostic ? "diagnostic" : "complete")
               << ",cases=" << cases.size()
               << ",raw_sessions=" << raw_session_count
               << ",raw_prefixes=" << raw_prefix_count
               << ",persistent_samples=" << persistent_sample_count
               << ",failures=0,records_before_completion="
               << records_before_completion << ",completed_utc=" << UtcNow()
               << ",fingerprint=" << completion_fingerprint;
    std::cout << completion.str() << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "GPU-02S session benchmark failed: " << error.what() << '\n';
    return 1;
  }
}

namespace {

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

[[nodiscard]] SessionSample MeasureInChild(const PreparedAcSessionCase &item,
                                           BackendKind backend,
                                           ValidationLane lane) {
  int descriptors[2]{};
  if (pipe(descriptors) != 0) {
    throw std::runtime_error("could not create session measurement pipe");
  }
  const pid_t child = fork();
  if (child < 0) {
    close(descriptors[0]);
    close(descriptors[1]);
    throw std::runtime_error("could not fork session measurement child");
  }
  if (child == 0) {
    close(descriptors[0]);
    SessionSample sample;
    try {
      sample = backend == BackendKind::kCuda ? MeasureGpuSession(item, lane)
                                             : MeasureCpuSession(item, lane);
    } catch (const std::exception &error) {
      std::snprintf(sample.error, sizeof(sample.error), "%s", error.what());
    } catch (...) {
      std::snprintf(sample.error, sizeof(sample.error), "%s",
                    "unknown session measurement failure");
    }
    const bool written = WriteAll(descriptors[1], &sample, sizeof(sample));
    close(descriptors[1]);
    _exit(written && sample.success != 0 ? 0 : 1);
  }

  close(descriptors[1]);
  SessionSample sample;
  const bool received = ReadAll(descriptors[0], &sample, sizeof(sample));
  close(descriptors[0]);
  int status = 0;
  while (waitpid(child, &status, 0) < 0) {
    if (errno != EINTR) {
      throw std::runtime_error("could not wait for session measurement child");
    }
  }
  if (!received || sample.magic != kChildMagic || !WIFEXITED(status) ||
      WEXITSTATUS(status) != 0 || sample.success == 0) {
    throw std::runtime_error(
        std::string("isolated session measurement failed: ") +
        (sample.error[0] == '\0' ? "missing child result" : sample.error));
  }
  return sample;
}

[[nodiscard]] GpuEnvironment InspectGpuInChild() {
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
    GpuEnvironment environment;
    cudaDeviceProp properties{};
    const cudaError_t device_status = cudaGetDeviceProperties(&properties, 0);
    const cudaError_t driver_status =
        cudaDriverGetVersion(&environment.driver_api);
    const cudaError_t runtime_status =
        cudaRuntimeGetVersion(&environment.runtime_api);
    if (device_status == cudaSuccess && driver_status == cudaSuccess &&
        runtime_status == cudaSuccess) {
      std::snprintf(environment.name, sizeof(environment.name), "%s",
                    properties.name);
      std::ostringstream uuid;
      uuid << std::hex << std::setfill('0');
      for (const char byte : properties.uuid.bytes) {
        uuid << std::setw(2)
             << static_cast<unsigned int>(static_cast<unsigned char>(byte));
      }
      std::snprintf(environment.uuid, sizeof(environment.uuid), "%s",
                    uuid.str().c_str());
      environment.compute_major = properties.major;
      environment.compute_minor = properties.minor;
      environment.global_memory_bytes = properties.totalGlobalMem;
      environment.success = 1;
    } else {
      std::snprintf(environment.error, sizeof(environment.error),
                    "device=%s driver=%s runtime=%s",
                    cudaGetErrorString(device_status),
                    cudaGetErrorString(driver_status),
                    cudaGetErrorString(runtime_status));
    }
    const bool written =
        WriteAll(descriptors[1], &environment, sizeof(environment));
    close(descriptors[1]);
    _exit(written && environment.success != 0 ? 0 : 1);
  }
  close(descriptors[1]);
  GpuEnvironment environment;
  const bool received =
      ReadAll(descriptors[0], &environment, sizeof(environment));
  close(descriptors[0]);
  int status = 0;
  while (waitpid(child, &status, 0) < 0) {
    if (errno != EINTR) {
      throw std::runtime_error("could not wait for GPU identity child");
    }
  }
  if (!received || environment.magic != kChildMagic || !WIFEXITED(status) ||
      WEXITSTATUS(status) != 0 || environment.success == 0) {
    throw std::runtime_error(std::string("GPU identity inspection failed: ") +
                             (environment.error[0] == '\0'
                                  ? "missing child result"
                                  : environment.error));
  }
  return environment;
}

[[nodiscard]] std::string UtcNow() {
  const std::time_t now = std::time(nullptr);
  std::tm utc{};
  gmtime_r(&now, &utc);
  char buffer[32]{};
  std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc);
  return buffer;
}

[[nodiscard]] std::string KernelIdentity() {
  utsname identity{};
  if (uname(&identity) != 0) {
    return "unavailable";
  }
  return std::string(identity.sysname) + " " + identity.release + " " +
         identity.machine;
}

[[nodiscard]] std::string CpuIdentity() {
  std::ifstream input("/proc/cpuinfo");
  std::string line;
  while (std::getline(input, line)) {
    constexpr std::string_view kModel = "model name";
    if (line.starts_with(kModel)) {
      const std::size_t colon = line.find(':');
      if (colon != std::string::npos) {
        std::size_t begin = colon + 1;
        while (begin < line.size() && line[begin] == ' ') {
          ++begin;
        }
        return line.substr(begin);
      }
    }
  }
  return "unavailable";
}

[[nodiscard]] Summary Summarize(const std::vector<SessionSample> &samples,
                                std::size_t prefix) {
  std::vector<std::uint64_t> totals;
  std::vector<std::uint64_t> certified;
  std::vector<std::uint64_t> first;
  std::vector<std::uint64_t> steady_median;
  std::vector<std::uint64_t> steady_p95;
  std::vector<double> throughput;
  Summary summary;
  for (const SessionSample &sample : samples) {
    const PrefixSample &value = sample.prefixes[prefix - 1];
    totals.push_back(value.lane_total_ns);
    certified.push_back(value.certified_total_ns);
    first.push_back(value.first_corner_ns);
    steady_median.push_back(value.later_corner_median_ns);
    steady_p95.push_back(value.later_corner_p95_ns);
    throughput.push_back(static_cast<double>(value.accepted_members) * 1e9 /
                         static_cast<double>(value.lane_total_ns));
    summary.peak_cpu_bytes =
        std::max(summary.peak_cpu_bytes, sample.cpu_peak_rss_bytes);
    summary.peak_gpu_bytes =
        std::max(summary.peak_gpu_bytes, sample.gpu_peak_batch_bytes);
  }
  std::sort(totals.begin(), totals.end());
  std::sort(throughput.begin(), throughput.end());
  summary.minimum_ns = totals.front();
  summary.median_ns = Quantile(totals, 1, 2);
  summary.p95_ns = Quantile(totals, 95, 100);
  summary.maximum_ns = totals.back();
  summary.certified_median_ns = Quantile(certified, 1, 2);
  summary.first_median_ns = Quantile(first, 1, 2);
  summary.steady_median_ns = Quantile(steady_median, 1, 2);
  summary.steady_p95_ns = Quantile(steady_p95, 95, 100);
  summary.median_throughput = throughput[(throughput.size() - 1) / 2];
  return summary;
}

} // namespace

namespace {

class PersistentParallelCpuExecutor {
public:
  explicit PersistentParallelCpuExecutor(std::size_t worker_count)
      : states_(worker_count), errors_(worker_count) {
    if (worker_count == 0) {
      throw std::runtime_error("persistent CPU executor requires workers");
    }
    threads_.reserve(worker_count);
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
      threads_.emplace_back([this, worker]() { WorkerLoop(worker); });
    }
    std::unique_lock lock(mutex_);
    ready_condition_.wait(lock, [&]() { return ready_ == states_.size(); });
  }

  ~PersistentParallelCpuExecutor() {
    {
      std::lock_guard lock(mutex_);
      stop_ = true;
      ++epoch_;
    }
    job_condition_.notify_all();
    for (std::thread &thread : threads_) {
      thread.join();
    }
  }

  PersistentParallelCpuExecutor(const PersistentParallelCpuExecutor &) = delete;
  PersistentParallelCpuExecutor &
  operator=(const PersistentParallelCpuExecutor &) = delete;

  [[nodiscard]] Result<PreparedAcBatchResult>
  Execute(const PreparedAcBatch &batch) {
    auto valid = ohmnivore::ValidatePreparedAcBatch(batch);
    if (!valid.ok()) {
      return Result<PreparedAcBatchResult>::Fail(valid.error().code,
                                                 valid.error().message);
    }
    PreparedAcBatchResult result{
        .contract_version = batch.contract_version,
        .replay_id = batch.replay_id,
        .structure_fingerprint = batch.structure.fingerprint,
        .batch_fingerprint = batch.batch_fingerprint,
        .members = std::vector<PreparedAcResultMember>(batch.members.size()),
    };
    {
      std::lock_guard lock(mutex_);
      batch_ = &batch;
      result_ = &result;
      completed_ = 0;
      last_solve_count_ = 0;
      std::fill(errors_.begin(), errors_.end(), std::nullopt);
      ++epoch_;
    }
    job_condition_.notify_all();
    {
      std::unique_lock lock(mutex_);
      done_condition_.wait(lock,
                           [&]() { return completed_ == states_.size(); });
      batch_ = nullptr;
      result_ = nullptr;
    }
    for (const std::optional<Error> &error : errors_) {
      if (error.has_value()) {
        return Result<PreparedAcBatchResult>::Fail(error->code, error->message);
      }
    }
    if (last_solve_count_ != batch.members.size()) {
      return Result<PreparedAcBatchResult>::Fail(
          ErrorCode::kPreparedBackendFailure,
          "persistent CPU comparator solve count differs from batch size");
    }
    return Result<PreparedAcBatchResult>::Ok(std::move(result));
  }

  [[nodiscard]] std::size_t symbolic_analyses() const {
    return symbolic_analyses_.load();
  }

  [[nodiscard]] std::size_t solves() const { return solves_.load(); }
  [[nodiscard]] std::size_t worker_count() const { return states_.size(); }

private:
  struct WorkerState {
    std::string structure_fingerprint;
    std::unique_ptr<SparseComplexFactorization> factorization;
  };

  void WorkerLoop(std::size_t worker) {
    std::size_t observed_epoch = 0;
    {
      std::lock_guard lock(mutex_);
      ++ready_;
      ready_condition_.notify_one();
    }
    while (true) {
      const PreparedAcBatch *batch = nullptr;
      PreparedAcBatchResult *result = nullptr;
      {
        std::unique_lock lock(mutex_);
        job_condition_.wait(
            lock, [&]() { return stop_ || epoch_ != observed_epoch; });
        if (stop_) {
          return;
        }
        observed_epoch = epoch_;
        batch = batch_;
        result = result_;
      }

      std::optional<Error> error;
      std::size_t local_solves = 0;
      WorkerState &state = states_[worker];
      if (state.structure_fingerprint != batch->structure.fingerprint) {
        state.factorization.reset();
        state.structure_fingerprint = batch->structure.fingerprint;
      }
      for (std::size_t member = worker; member < batch->members.size();
           member += states_.size()) {
        auto matrix = ohmnivore::MaterializePreparedAcMatrix(*batch, member);
        if (!matrix.ok()) {
          error = matrix.error();
          break;
        }
        if (state.factorization == nullptr) {
          auto analyzed = SparseComplexFactorization::Analyze(matrix.value());
          if (!analyzed.ok()) {
            error = analyzed.error();
            break;
          }
          state.factorization = analyzed.TakeValue();
          symbolic_analyses_.fetch_add(1);
        }
        auto solved = state.factorization->FactorAndSolve(
            matrix.value(), batch->members[member].rhs);
        if (!solved.ok()) {
          error = solved.error();
          break;
        }
        result->members[member] = PreparedAcResultMember{
            .identity = batch->members[member].identity,
            .solution = solved.TakeValue(),
        };
        ++local_solves;
      }
      solves_.fetch_add(local_solves);
      {
        std::lock_guard lock(mutex_);
        errors_[worker] = std::move(error);
        last_solve_count_ += local_solves;
        ++completed_;
        if (completed_ == states_.size()) {
          done_condition_.notify_one();
        }
      }
    }
  }

  std::vector<WorkerState> states_;
  std::vector<std::thread> threads_;
  std::vector<std::optional<Error>> errors_;
  mutable std::mutex mutex_;
  std::condition_variable ready_condition_;
  std::condition_variable job_condition_;
  std::condition_variable done_condition_;
  const PreparedAcBatch *batch_ = nullptr;
  PreparedAcBatchResult *result_ = nullptr;
  std::size_t ready_ = 0;
  std::size_t completed_ = 0;
  std::size_t last_solve_count_ = 0;
  std::size_t epoch_ = 0;
  bool stop_ = false;
  std::atomic<std::size_t> symbolic_analyses_ = 0;
  std::atomic<std::size_t> solves_ = 0;
};

void RequireOk(const Result<bool> &result, const std::string &context) {
  if (!result.ok()) {
    throw std::runtime_error(context + ": " + result.error().message);
  }
}

void ValidateResult(const PreparedAcBatch &batch,
                    const PreparedAcBatchResult &result, ValidationLane lane,
                    PrefixSample *prefix, std::uint64_t *corner_lane_ns) {
  if (lane == ValidationLane::kCandidateRuntime) {
    const Clock::time_point runtime_start = Clock::now();
    RequireOk(ohmnivore::benchmarks::ValidatePreparedAcBatchResultForEvidence(
                  batch, result),
              "candidate-runtime validation failed");
    const std::uint64_t runtime_ns = ElapsedNanoseconds(runtime_start);
    prefix->runtime_validation_ns += runtime_ns;
    *corner_lane_ns += runtime_ns;
    const Clock::time_point certification_start = Clock::now();
    RequireOk(ohmnivore::ValidatePreparedAcBatchResult(batch, result),
              "deferred KLU certification failed");
    prefix->excluded_certification_ns +=
        ElapsedNanoseconds(certification_start);
  } else {
    const Clock::time_point validation_start = Clock::now();
    RequireOk(ohmnivore::ValidatePreparedAcBatchResult(batch, result),
              "inline KLU certification failed");
    const std::uint64_t validation_ns = ElapsedNanoseconds(validation_start);
    prefix->runtime_validation_ns += validation_ns;
    *corner_lane_ns += validation_ns;
  }
}

void FinishPrefix(SessionSample *sample, std::size_t corner,
                  const std::vector<std::uint64_t> &later_corner_ns) {
  PrefixSample &prefix = sample->prefixes[corner];
  prefix.lane_total_ns = prefix.preparation_ns + prefix.executor_create_ns +
                         prefix.execute_ns + prefix.runtime_validation_ns;
  prefix.certified_total_ns =
      prefix.lane_total_ns + prefix.excluded_certification_ns;
  if (corner == 0) {
    prefix.first_corner_ns = prefix.lane_total_ns;
  } else {
    prefix.first_corner_ns = sample->prefixes[corner - 1].first_corner_ns;
  }
  prefix.later_corner_median_ns = Quantile(later_corner_ns, 1, 2);
  prefix.later_corner_p95_ns = Quantile(later_corner_ns, 95, 100);
}

[[nodiscard]] PreparedAcBatch PrepareCorner(const PreparedAcSessionCase &item,
                                            std::size_t corner) {
  auto prepared = ohmnivore::benchmarks::PrepareAcSessionCorner(item, corner);
  if (!prepared.ok()) {
    throw std::runtime_error(item.case_id + " corner preparation failed: " +
                             prepared.error().message);
  }
  return prepared.TakeValue();
}

[[nodiscard]] SessionSample MeasureCpuSessionWithExecutor(
    const PreparedAcSessionCase &item, ValidationLane lane,
    PersistentParallelCpuExecutor *executor, std::uint64_t executor_create_ns) {
  SessionSample sample;
  sample.prefix_count = static_cast<std::uint32_t>(item.corner_count);
  const std::size_t workers = executor->worker_count();
  sample.participating_threads = static_cast<std::uint32_t>(workers);
  const std::size_t before_analyses = executor->symbolic_analyses();
  const std::size_t before_solves = executor->solves();
  PrefixSample cumulative;
  cumulative.executor_create_ns = executor_create_ns;
  std::vector<std::uint64_t> later_corner_ns;
  std::string structure_fingerprint;
  for (std::size_t corner = 0; corner < item.corner_count; ++corner) {
    std::uint64_t corner_lane_ns = 0;
    const Clock::time_point preparation_start = Clock::now();
    PreparedAcBatch batch = PrepareCorner(item, corner);
    const std::uint64_t preparation_ns = ElapsedNanoseconds(preparation_start);
    cumulative.preparation_ns += preparation_ns;
    corner_lane_ns += preparation_ns;
    if (corner == 0) {
      structure_fingerprint = batch.structure.fingerprint;
    } else if (batch.structure.fingerprint != structure_fingerprint) {
      throw std::runtime_error(item.case_id +
                               " session corner changed sparse structure");
    }

    const Clock::time_point execute_start = Clock::now();
    auto solved = executor->Execute(batch);
    const std::uint64_t execute_ns = ElapsedNanoseconds(execute_start);
    cumulative.execute_ns += execute_ns;
    corner_lane_ns += execute_ns;
    if (!solved.ok()) {
      throw std::runtime_error(item.case_id + " CPU session solve failed: " +
                               solved.error().message);
    }
    ValidateResult(batch, solved.value(), lane, &cumulative, &corner_lane_ns);
    cumulative.accepted_members += item.batch_size;
    if (corner > 0) {
      later_corner_ns.push_back(corner_lane_ns);
    }
    sample.prefixes[corner] = cumulative;
    FinishPrefix(&sample, corner, later_corner_ns);
  }

  sample.cpu_symbolic_analyses =
      executor->symbolic_analyses() - before_analyses;
  sample.backend_solves = executor->solves() - before_solves;
  sample.certification_solves = item.corner_count * item.batch_size;
  const std::uint64_t expected_analyses = before_solves == 0 ? workers : 0;
  if (sample.cpu_symbolic_analyses != expected_analyses ||
      sample.backend_solves != item.corner_count * item.batch_size) {
    throw std::runtime_error(item.case_id +
                             " persistent CPU accounting is incomplete");
  }
  sample.cpu_peak_rss_bytes = PeakResidentBytes();
  sample.success = 1;
  return sample;
}

[[nodiscard]] SessionSample MeasureCpuSession(const PreparedAcSessionCase &item,
                                              ValidationLane lane) {
  const std::size_t hardware =
      std::max<std::size_t>(1, std::thread::hardware_concurrency());
  const std::size_t workers = std::min(hardware, item.batch_size);
  const Clock::time_point executor_start = Clock::now();
  auto executor = std::make_unique<PersistentParallelCpuExecutor>(workers);
  const std::uint64_t executor_create_ns = ElapsedNanoseconds(executor_start);
  SessionSample sample = MeasureCpuSessionWithExecutor(
      item, lane, executor.get(), executor_create_ns);
  const Clock::time_point release_start = Clock::now();
  executor.reset();
  sample.release_ns = ElapsedNanoseconds(release_start);
  return sample;
}

[[nodiscard]] SessionSample MeasureGpuSessionWithExecutor(
    const PreparedAcSessionCase &item, ValidationLane lane,
    CudaPreparedAcBatchBackend *executor, std::uint64_t executor_create_ns,
    bool release_after_session) {
  SessionSample sample;
  sample.prefix_count = static_cast<std::uint32_t>(item.corner_count);
  const CudaPreparedAcStatistics before = executor->statistics();
  PrefixSample cumulative;
  cumulative.executor_create_ns = executor_create_ns;
  std::vector<std::uint64_t> later_corner_ns;
  std::string structure_fingerprint;
  for (std::size_t corner = 0; corner < item.corner_count; ++corner) {
    std::uint64_t corner_lane_ns = 0;
    const Clock::time_point preparation_start = Clock::now();
    PreparedAcBatch batch = PrepareCorner(item, corner);
    const std::uint64_t preparation_ns = ElapsedNanoseconds(preparation_start);
    cumulative.preparation_ns += preparation_ns;
    corner_lane_ns += preparation_ns;
    if (corner == 0) {
      structure_fingerprint = batch.structure.fingerprint;
    } else if (batch.structure.fingerprint != structure_fingerprint) {
      throw std::runtime_error(item.case_id +
                               " session corner changed sparse structure");
    }

    const Clock::time_point execute_start = Clock::now();
    auto solved = executor->Execute(batch);
    const std::uint64_t execute_ns = ElapsedNanoseconds(execute_start);
    cumulative.execute_ns += execute_ns;
    corner_lane_ns += execute_ns;
    const CudaPreparedAcStatistics &statistics = executor->statistics();
    cumulative.context_setup_ns += statistics.last_context_setup_ns;
    cumulative.library_setup_ns += statistics.last_library_setup_ns;
    cumulative.host_pack_ns += statistics.last_host_pack_ns;
    cumulative.upload_ns += statistics.last_upload_ns;
    cumulative.matrix_setup_ns += statistics.last_matrix_setup_ns;
    cumulative.analysis_ns +=
        statistics.last_analysis_submit_ns + statistics.last_analysis_sync_ns;
    cumulative.factor_solve_ns += statistics.last_factor_solve_submit_ns +
                                  statistics.last_factor_solve_sync_ns;
    cumulative.analysis_device_ns += statistics.last_analysis_device_ns;
    cumulative.factor_solve_device_ns += statistics.last_factor_solve_device_ns;
    cumulative.readback_ns += statistics.last_readback_ns;
    cumulative.phase_status_ns += statistics.last_phase_status_ns;
    cumulative.memory_accounting_ns += statistics.last_memory_accounting_ns;
    cumulative.result_assembly_ns += statistics.last_result_assembly_ns;
    sample.gpu_peak_batch_bytes = std::max<std::uint64_t>(
        sample.gpu_peak_batch_bytes, statistics.peak_batch_device_bytes);
    if (!solved.ok()) {
      throw std::runtime_error(item.case_id + " CUDA session solve failed: " +
                               solved.error().message);
    }
    ValidateResult(batch, solved.value(), lane, &cumulative, &corner_lane_ns);
    cumulative.accepted_members += item.batch_size;
    const std::uint64_t classified_execute_ns =
        cumulative.context_setup_ns + cumulative.library_setup_ns +
        cumulative.host_pack_ns + cumulative.upload_ns +
        cumulative.matrix_setup_ns + cumulative.analysis_ns +
        cumulative.factor_solve_ns + cumulative.readback_ns +
        cumulative.phase_status_ns + cumulative.memory_accounting_ns +
        cumulative.result_assembly_ns;
    if (classified_execute_ns > cumulative.execute_ns) {
      throw std::runtime_error(item.case_id +
                               " CUDA phase accounting exceeds Execute wall");
    }
    cumulative.cuda_execute_phase_remainder_ns =
        cumulative.execute_ns - classified_execute_ns;
    if (corner > 0) {
      later_corner_ns.push_back(corner_lane_ns);
    }
    sample.prefixes[corner] = cumulative;
    FinishPrefix(&sample, corner, later_corner_ns);
  }

  const CudaPreparedAcStatistics &statistics = executor->statistics();
  sample.context_setup_ns = cumulative.context_setup_ns;
  sample.library_setup_ns = cumulative.library_setup_ns;
  sample.host_pack_ns = cumulative.host_pack_ns;
  sample.upload_ns = cumulative.upload_ns;
  sample.matrix_setup_ns = cumulative.matrix_setup_ns;
  sample.analysis_ns = cumulative.analysis_ns;
  sample.factor_solve_ns = cumulative.factor_solve_ns;
  sample.analysis_device_ns = cumulative.analysis_device_ns;
  sample.factor_solve_device_ns = cumulative.factor_solve_device_ns;
  sample.readback_ns = cumulative.readback_ns;
  sample.phase_status_ns = cumulative.phase_status_ns;
  sample.memory_accounting_ns = cumulative.memory_accounting_ns;
  sample.result_assembly_ns = cumulative.result_assembly_ns;
  sample.execute_phase_remainder_ns =
      cumulative.cuda_execute_phase_remainder_ns;
  sample.structure_preparations = statistics.preparations - before.preparations;
  sample.structure_uploads =
      statistics.structure_uploads - before.structure_uploads;
  sample.values_rhs_uploads =
      statistics.values_rhs_uploads - before.values_rhs_uploads;
  sample.same_structure_refreshes =
      statistics.same_structure_refreshes - before.same_structure_refreshes;
  sample.cuda_analyses = statistics.analyses - before.analyses;
  sample.backend_solves = statistics.solves - before.solves;
  sample.certification_solves = item.corner_count * item.batch_size;
  sample.factorization_calls =
      statistics.factorization_calls - before.factorization_calls;
  sample.solve_calls = statistics.solve_calls - before.solve_calls;
  const bool initial_generation = before.preparations == 0;
  const std::uint64_t expected_generation_count = initial_generation ? 1 : 0;
  const std::uint64_t expected_refreshes =
      item.corner_count - (initial_generation ? 1 : 0);
  if (sample.structure_preparations != expected_generation_count ||
      sample.structure_uploads != expected_generation_count ||
      sample.values_rhs_uploads != item.corner_count ||
      sample.same_structure_refreshes != expected_refreshes ||
      sample.cuda_analyses != expected_generation_count ||
      sample.backend_solves != item.corner_count * item.batch_size ||
      sample.factorization_calls != item.corner_count ||
      sample.solve_calls != item.corner_count) {
    throw std::runtime_error(item.case_id +
                             " persistent CUDA accounting is incomplete");
  }
  if (release_after_session) {
    const Clock::time_point release_start = Clock::now();
    auto released = executor->Release();
    sample.release_ns = ElapsedNanoseconds(release_start);
    if (!released.ok()) {
      throw std::runtime_error(
          item.case_id + " CUDA release failed: " + released.error().message);
    }
    sample.outstanding_device_bytes =
        executor->statistics().last_release_outstanding_device_bytes;
    if (sample.outstanding_device_bytes != 0) {
      throw std::runtime_error(item.case_id +
                               " CUDA release retained device allocations");
    }
  }
  sample.cpu_peak_rss_bytes = PeakResidentBytes();
  sample.success = 1;
  return sample;
}

[[nodiscard]] SessionSample MeasureGpuSession(const PreparedAcSessionCase &item,
                                              ValidationLane lane) {
  const Clock::time_point executor_start = Clock::now();
  CudaPreparedAcBatchBackend executor;
  const std::uint64_t executor_create_ns = ElapsedNanoseconds(executor_start);
  return MeasureGpuSessionWithExecutor(item, lane, &executor,
                                       executor_create_ns, true);
}

[[nodiscard]] PersistentDiagnostic
MeasurePersistentInChild(const PreparedAcSessionCase &item, BackendKind backend,
                         ValidationLane lane, std::size_t steady_sessions) {
  int descriptors[2]{};
  if (pipe(descriptors) != 0) {
    throw std::runtime_error("could not create persistent diagnostic pipe");
  }
  const pid_t child = fork();
  if (child < 0) {
    close(descriptors[0]);
    close(descriptors[1]);
    throw std::runtime_error("could not fork persistent diagnostic child");
  }
  if (child == 0) {
    close(descriptors[0]);
    PersistentDiagnostic diagnostic;
    try {
      if (backend == BackendKind::kCuda) {
        const Clock::time_point executor_start = Clock::now();
        CudaPreparedAcBatchBackend executor;
        const std::uint64_t executor_create_ns =
            ElapsedNanoseconds(executor_start);
        diagnostic.cold = MeasureGpuSessionWithExecutor(
            item, lane, &executor, executor_create_ns, false);
        for (std::size_t session = 0; session < steady_sessions; ++session) {
          diagnostic.steady[session] =
              MeasureGpuSessionWithExecutor(item, lane, &executor, 0, false);
        }
        const Clock::time_point release_start = Clock::now();
        auto released = executor.Release();
        diagnostic.release_ns = ElapsedNanoseconds(release_start);
        if (!released.ok()) {
          throw std::runtime_error(item.case_id + " CUDA release failed: " +
                                   released.error().message);
        }
        diagnostic.outstanding_device_bytes =
            executor.statistics().last_release_outstanding_device_bytes;
        if (diagnostic.outstanding_device_bytes != 0) {
          throw std::runtime_error(item.case_id +
                                   " CUDA release retained device allocations");
        }
      } else {
        const std::size_t hardware =
            std::max<std::size_t>(1, std::thread::hardware_concurrency());
        const std::size_t workers = std::min(hardware, item.batch_size);
        const Clock::time_point executor_start = Clock::now();
        auto executor =
            std::make_unique<PersistentParallelCpuExecutor>(workers);
        const std::uint64_t executor_create_ns =
            ElapsedNanoseconds(executor_start);
        diagnostic.cold = MeasureCpuSessionWithExecutor(
            item, lane, executor.get(), executor_create_ns);
        for (std::size_t session = 0; session < steady_sessions; ++session) {
          diagnostic.steady[session] =
              MeasureCpuSessionWithExecutor(item, lane, executor.get(), 0);
        }
        const Clock::time_point release_start = Clock::now();
        executor.reset();
        diagnostic.release_ns = ElapsedNanoseconds(release_start);
      }
      diagnostic.sample_count = static_cast<std::uint32_t>(steady_sessions);
      diagnostic.success = 1;
    } catch (const std::exception &error) {
      std::snprintf(diagnostic.error, sizeof(diagnostic.error), "%s",
                    error.what());
    } catch (...) {
      std::snprintf(diagnostic.error, sizeof(diagnostic.error), "%s",
                    "unknown persistent diagnostic failure");
    }
    const bool written =
        WriteAll(descriptors[1], &diagnostic, sizeof(diagnostic));
    close(descriptors[1]);
    _exit(written && diagnostic.success != 0 ? 0 : 1);
  }

  close(descriptors[1]);
  PersistentDiagnostic diagnostic;
  const bool received =
      ReadAll(descriptors[0], &diagnostic, sizeof(diagnostic));
  close(descriptors[0]);
  int status = 0;
  while (waitpid(child, &status, 0) < 0) {
    if (errno != EINTR) {
      throw std::runtime_error(
          "could not wait for persistent diagnostic child");
    }
  }
  if (!received || diagnostic.magic != kChildMagic || !WIFEXITED(status) ||
      WEXITSTATUS(status) != 0 || diagnostic.success == 0) {
    throw std::runtime_error(std::string("persistent diagnostic failed: ") +
                             (diagnostic.error[0] == '\0'
                                  ? "missing child result"
                                  : diagnostic.error));
  }
  return diagnostic;
}

} // namespace
