#ifndef OHMNIVORE_BENCHMARKS_EMI03_PROFILE_H_
#define OHMNIVORE_BENCHMARKS_EMI03_PROFILE_H_

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>

// Diagnostic-only scopes. Production translation units do not include this
// header unless the separate EMI-03 profiling build explicitly enables them.
namespace ohmnivore::emi03_profile {

enum class Phase : std::size_t {
  kAssembly,
  kLinearSolve,
  kOutput,
  kExpressionFull,
  kExpressionValue,
  kCount
};

struct Counter {
  std::uint64_t calls = 0;
  std::chrono::steady_clock::duration elapsed{};
};

inline thread_local std::array<Counter, static_cast<std::size_t>(Phase::kCount)>
    counters;

class Scope {
public:
  explicit Scope(Phase phase)
      : counter_(counters[static_cast<std::size_t>(phase)]),
        start_(std::chrono::steady_clock::now()) {}
  ~Scope() {
    counter_.elapsed += std::chrono::steady_clock::now() - start_;
    ++counter_.calls;
  }
  Scope(const Scope &) = delete;
  Scope &operator=(const Scope &) = delete;

private:
  Counter &counter_;
  std::chrono::steady_clock::time_point start_;
};

inline double Seconds(Phase phase) {
  return std::chrono::duration<double>(
             counters[static_cast<std::size_t>(phase)].elapsed)
      .count();
}

inline std::uint64_t Calls(Phase phase) {
  return counters[static_cast<std::size_t>(phase)].calls;
}

} // namespace ohmnivore::emi03_profile

#endif // OHMNIVORE_BENCHMARKS_EMI03_PROFILE_H_
