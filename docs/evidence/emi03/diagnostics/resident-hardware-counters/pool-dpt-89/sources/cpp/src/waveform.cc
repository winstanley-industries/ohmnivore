#include "ohmnivore/waveform.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace ohmnivore {
namespace {

inline constexpr std::size_t kMaxExpandedBreakpoints = 1'000'000;

[[nodiscard]] bool IsFinite(double value) { return std::isfinite(value); }

[[nodiscard]] std::optional<std::string>
ValidateWaveform(const PulseWaveform &waveform) {
  if (!IsFinite(waveform.initial_value) || !IsFinite(waveform.pulsed_value) ||
      !IsFinite(waveform.delay_seconds) ||
      !IsFinite(waveform.rise_time_seconds) ||
      !IsFinite(waveform.fall_time_seconds) ||
      !IsFinite(waveform.pulse_width_seconds) ||
      !IsFinite(waveform.period_seconds)) {
    return "PULSE parameters must be finite";
  }
  if (waveform.delay_seconds < 0.0 || waveform.rise_time_seconds < 0.0 ||
      waveform.fall_time_seconds < 0.0 || waveform.pulse_width_seconds < 0.0) {
    return "PULSE delay and durations must be nonnegative";
  }
  if (waveform.period_seconds <= 0.0) {
    return "PULSE period must be greater than zero";
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<std::string>
ValidateWaveform(const SinWaveform &waveform) {
  if (!IsFinite(waveform.offset) || !IsFinite(waveform.amplitude) ||
      !IsFinite(waveform.frequency_hz) || !IsFinite(waveform.delay_seconds) ||
      !IsFinite(waveform.damping_factor_per_second)) {
    return "SIN parameters must be finite";
  }
  if (waveform.frequency_hz < 0.0 || waveform.delay_seconds < 0.0 ||
      waveform.damping_factor_per_second < 0.0) {
    return "SIN frequency, delay, and damping factor must be nonnegative";
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<std::string>
ValidateWaveform(const PwlWaveform &waveform) {
  if (waveform.time_value_pairs.empty()) {
    return "PWL requires at least one time-value pair";
  }
  for (std::size_t index = 0; index < waveform.time_value_pairs.size();
       ++index) {
    const auto [time, value] = waveform.time_value_pairs[index];
    if (!IsFinite(time) || !IsFinite(value)) {
      return "PWL times and values must be finite";
    }
    if (time < 0.0) {
      return "PWL times must be nonnegative";
    }
    if (index > 0 && time <= waveform.time_value_pairs[index - 1].first) {
      return "PWL times must be strictly increasing";
    }
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<std::string>
ValidateWaveform(const ExpWaveform &waveform) {
  if (!IsFinite(waveform.initial_value) || !IsFinite(waveform.pulsed_value) ||
      !IsFinite(waveform.rise_delay_seconds) ||
      !IsFinite(waveform.rise_time_constant_seconds) ||
      !IsFinite(waveform.fall_delay_seconds) ||
      !IsFinite(waveform.fall_time_constant_seconds)) {
    return "EXP parameters must be finite";
  }
  if (waveform.rise_delay_seconds < 0.0 || waveform.fall_delay_seconds < 0.0) {
    return "EXP delays must be nonnegative";
  }
  if (waveform.rise_time_constant_seconds <= 0.0 ||
      waveform.fall_time_constant_seconds <= 0.0) {
    return "EXP time constants must be greater than zero";
  }
  if (waveform.fall_delay_seconds < waveform.rise_delay_seconds) {
    return "EXP fall delay must not precede its rise delay";
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<std::string>
ValidateWaveform(const TransientWaveform &waveform) {
  return std::visit([](const auto &typed) { return ValidateWaveform(typed); },
                    waveform);
}

[[nodiscard]] double EvaluatePulse(const PulseWaveform &waveform,
                                   double time_seconds) {
  if (time_seconds < waveform.delay_seconds) {
    return waveform.initial_value;
  }
  const double elapsed = time_seconds - waveform.delay_seconds;
  double relative = elapsed;
  if (waveform.period_seconds < std::numeric_limits<double>::max()) {
    relative = std::fmod(elapsed, waveform.period_seconds);
    if (relative < 0.0) {
      relative += waveform.period_seconds;
    }

    // Breakpoints are constructed as delay + cycle * period and then rounded
    // to FP64.  Subtracting delay again can put that exact scheduled time one
    // ulp on either side of the cycle boundary.  Canonicalize every PULSE edge
    // within the rounding error of the absolute time so breakpoint generation
    // and evaluation cannot disagree about which side of an edge is active.
    const double edge_tolerance =
        16.0 * std::numeric_limits<double>::epsilon() *
        std::max({std::abs(time_seconds), std::abs(waveform.delay_seconds),
                  std::abs(elapsed), waveform.period_seconds});
    if (relative <= edge_tolerance ||
        waveform.period_seconds - relative <= edge_tolerance) {
      relative = 0.0;
    }
    for (double edge :
         {waveform.rise_time_seconds,
          waveform.rise_time_seconds + waveform.pulse_width_seconds,
          waveform.rise_time_seconds + waveform.pulse_width_seconds +
              waveform.fall_time_seconds}) {
      if (edge < waveform.period_seconds &&
          std::abs(relative - edge) <= edge_tolerance) {
        relative = edge;
      }
    }
  }
  if (relative < waveform.rise_time_seconds) {
    if (waveform.rise_time_seconds > 0.0) {
      return waveform.initial_value +
             (waveform.pulsed_value - waveform.initial_value) * relative /
                 waveform.rise_time_seconds;
    }
    return waveform.pulsed_value;
  }
  if (relative < waveform.rise_time_seconds + waveform.pulse_width_seconds) {
    return waveform.pulsed_value;
  }
  if (relative < waveform.rise_time_seconds + waveform.pulse_width_seconds +
                     waveform.fall_time_seconds) {
    if (waveform.fall_time_seconds > 0.0) {
      return waveform.pulsed_value +
             (waveform.initial_value - waveform.pulsed_value) *
                 (relative - waveform.rise_time_seconds -
                  waveform.pulse_width_seconds) /
                 waveform.fall_time_seconds;
    }
    return waveform.initial_value;
  }
  return waveform.initial_value;
}

[[nodiscard]] double EvaluateSin(const SinWaveform &waveform,
                                 double time_seconds) {
  if (time_seconds < waveform.delay_seconds) {
    return waveform.offset;
  }
  const double elapsed = time_seconds - waveform.delay_seconds;
  const double envelope =
      waveform.damping_factor_per_second == 0.0
          ? 1.0
          : std::exp(-elapsed * waveform.damping_factor_per_second);
  return waveform.offset + waveform.amplitude *
                               std::sin(2.0 * std::numbers::pi *
                                        waveform.frequency_hz * elapsed) *
                               envelope;
}

[[nodiscard]] double EvaluatePwl(const PwlWaveform &waveform,
                                 double time_seconds) {
  if (time_seconds <= waveform.time_value_pairs.front().first) {
    return waveform.time_value_pairs.front().second;
  }
  if (time_seconds >= waveform.time_value_pairs.back().first) {
    return waveform.time_value_pairs.back().second;
  }
  for (std::size_t index = 1; index < waveform.time_value_pairs.size();
       ++index) {
    if (time_seconds <= waveform.time_value_pairs[index].first) {
      const auto [start_time, start_value] =
          waveform.time_value_pairs[index - 1];
      const auto [stop_time, stop_value] = waveform.time_value_pairs[index];
      const double fraction =
          (time_seconds - start_time) / (stop_time - start_time);
      return start_value + (stop_value - start_value) * fraction;
    }
  }
  return waveform.time_value_pairs.back().second;
}

[[nodiscard]] double EvaluateExp(const ExpWaveform &waveform,
                                 double time_seconds) {
  if (time_seconds < waveform.rise_delay_seconds) {
    return waveform.initial_value;
  }
  const double rise =
      (waveform.pulsed_value - waveform.initial_value) *
      (1.0 - std::exp(-(time_seconds - waveform.rise_delay_seconds) /
                      waveform.rise_time_constant_seconds));
  if (time_seconds < waveform.fall_delay_seconds) {
    return waveform.initial_value + rise;
  }
  const double fall =
      (waveform.initial_value - waveform.pulsed_value) *
      (1.0 - std::exp(-(time_seconds - waveform.fall_delay_seconds) /
                      waveform.fall_time_constant_seconds));
  return waveform.initial_value + rise + fall;
}

[[nodiscard]] bool AddBreakpoint(long double candidate,
                                 double stop_time_seconds,
                                 std::vector<double> *breakpoints) {
  if (candidate < 0.0L ||
      candidate > static_cast<long double>(stop_time_seconds) ||
      candidate >
          static_cast<long double>(std::numeric_limits<double>::max())) {
    return true;
  }
  const double value = static_cast<double>(candidate);
  if (!std::isfinite(value)) {
    return true;
  }
  breakpoints->push_back(value);
  return breakpoints->size() <= kMaxExpandedBreakpoints;
}

[[nodiscard]] bool AddPulseCycleBreakpoints(const PulseWaveform &waveform,
                                            long double cycle_start,
                                            double stop_time_seconds,
                                            std::vector<double> *breakpoints) {
  const long double period = waveform.period_seconds;
  const long double rise = waveform.rise_time_seconds;
  const long double high_end = rise + waveform.pulse_width_seconds;
  const long double fall_end = high_end + waveform.fall_time_seconds;
  for (long double offset : {0.0L, rise, high_end, fall_end}) {
    if (waveform.period_seconds < std::numeric_limits<double>::max() &&
        offset >= period) {
      continue;
    }
    if (!AddBreakpoint(cycle_start + offset, stop_time_seconds, breakpoints)) {
      return false;
    }
  }
  return true;
}

} // namespace

Result<double> EvaluateTransientWaveform(const TransientWaveform &waveform,
                                         double time_seconds) {
  if (const auto error = ValidateWaveform(waveform); error.has_value()) {
    return Result<double>::Fail(ErrorCode::kCompile,
                                "invalid transient waveform: " + *error);
  }
  if (!std::isfinite(time_seconds)) {
    return Result<double>::Fail(
        ErrorCode::kSolve, "transient waveform evaluation time must be finite");
  }

  const double value = std::visit(
      [&](const auto &typed) -> double {
        using Waveform = std::decay_t<decltype(typed)>;
        if constexpr (std::is_same_v<Waveform, PulseWaveform>) {
          return EvaluatePulse(typed, time_seconds);
        } else if constexpr (std::is_same_v<Waveform, SinWaveform>) {
          return EvaluateSin(typed, time_seconds);
        } else if constexpr (std::is_same_v<Waveform, PwlWaveform>) {
          return EvaluatePwl(typed, time_seconds);
        } else {
          return EvaluateExp(typed, time_seconds);
        }
      },
      waveform);
  if (!std::isfinite(value)) {
    return Result<double>::Fail(
        ErrorCode::kSolve,
        "transient waveform evaluation produced a non-finite value");
  }
  return Result<double>::Ok(value);
}

Result<std::vector<double>>
CollectTransientWaveformBreakpoints(const TransientWaveform &waveform,
                                    double stop_time_seconds) {
  if (const auto error = ValidateWaveform(waveform); error.has_value()) {
    return Result<std::vector<double>>::Fail(
        ErrorCode::kCompile, "invalid transient waveform: " + *error);
  }
  if (!std::isfinite(stop_time_seconds) || stop_time_seconds < 0.0) {
    return Result<std::vector<double>>::Fail(
        ErrorCode::kSolve,
        "waveform breakpoint stop time must be finite and nonnegative");
  }

  std::vector<double> breakpoints;
  const bool complete = std::visit(
      [&](const auto &typed) -> bool {
        using Waveform = std::decay_t<decltype(typed)>;
        if constexpr (std::is_same_v<Waveform, PulseWaveform>) {
          const long double delay = typed.delay_seconds;
          if (delay > static_cast<long double>(stop_time_seconds)) {
            return true;
          }
          if (typed.period_seconds == std::numeric_limits<double>::max()) {
            return AddPulseCycleBreakpoints(typed, delay, stop_time_seconds,
                                            &breakpoints);
          }
          for (std::size_t cycle = 0;; ++cycle) {
            const long double cycle_start =
                delay + static_cast<long double>(cycle) * typed.period_seconds;
            if (cycle_start > static_cast<long double>(stop_time_seconds)) {
              return true;
            }
            if (!AddPulseCycleBreakpoints(typed, cycle_start, stop_time_seconds,
                                          &breakpoints)) {
              return false;
            }
            if (cycle == std::numeric_limits<std::size_t>::max()) {
              return false;
            }
          }
        } else if constexpr (std::is_same_v<Waveform, SinWaveform>) {
          return AddBreakpoint(typed.delay_seconds, stop_time_seconds,
                               &breakpoints);
        } else if constexpr (std::is_same_v<Waveform, PwlWaveform>) {
          for (const auto &[time, unused] : typed.time_value_pairs) {
            static_cast<void>(unused);
            if (!AddBreakpoint(time, stop_time_seconds, &breakpoints)) {
              return false;
            }
          }
          return true;
        } else {
          return AddBreakpoint(typed.rise_delay_seconds, stop_time_seconds,
                               &breakpoints) &&
                 AddBreakpoint(typed.fall_delay_seconds, stop_time_seconds,
                               &breakpoints);
        }
      },
      waveform);
  if (!complete) {
    return Result<std::vector<double>>::Fail(
        ErrorCode::kSolve,
        "waveform expands beyond the 1000000-breakpoint limit");
  }

  std::sort(breakpoints.begin(), breakpoints.end());
  breakpoints.erase(std::unique(breakpoints.begin(), breakpoints.end()),
                    breakpoints.end());
  return Result<std::vector<double>>::Ok(std::move(breakpoints));
}

} // namespace ohmnivore
