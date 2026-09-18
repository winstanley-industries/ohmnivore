#include "cpp/tests/google_test.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "ohmnivore/behavioral.h"
#include "ohmnivore/transient.h"

namespace ohmnivore {
namespace {

MnaSystem HistoryCompile(const std::string &deck) {
  auto parsed = ParseBehavioralNetlist(deck);
  if (!parsed.ok()) {
    ADD_FAILURE() << parsed.error().message;
    return {};
  }
  auto compiled = CompileBehavioralMna(parsed.value());
  if (!compiled.ok()) {
    ADD_FAILURE() << compiled.error().message;
    return {};
  }
  return compiled.TakeValue();
}

std::size_t HistoryNode(const MnaSystem &system, const std::string &name) {
  const auto found =
      std::find(system.node_names.begin(), system.node_names.end(), name);
  EXPECT_NE(found, system.node_names.end());
  return static_cast<std::size_t>(found - system.node_names.begin());
}

TransientExecutionLimits HistoryLimits() {
  TransientExecutionLimits limits;
  limits.behavioral_error_estimator =
      BehavioralErrorEstimator::kDerivativeHistory;
  return limits;
}

MnaSystem ForcedPolynomial(bool quartic, bool corners = false) {
  const std::string clock =
      corners ? "Vclock clock 0 DC 0 PWL(0 0 40m .04 80m .04 120m 0)\n"
              : "Vclock clock 0 DC 0 PWL(0 0 1 1)\n";
  // DC is nonsingular. At positive clock values the independently specified
  // current cancels the 1-ohm shunt and known GMIN, giving exactly y'=3*t^2
  // or y'=4*t^3. No production matrix, derivative or LTE helper defines truth.
  return HistoryCompile(
      clock +
      "Rload out 0 1\nCstate out 0 1u\n"
      "Bforce out 0 I={if(v(clock)>0,-1.000000000001*v(out)-" +
      (quartic ? "4e-6*v(clock)*v(clock)*v(clock)" : "3e-6*v(clock)*v(clock)") +
      ",0)}\n");
}

struct ForcingChecks {
  std::size_t history_trials = 0;
  std::size_t unequal_history_trials = 0;
  std::size_t rejected = 0;
  std::size_t distinctive_controller = 0;
  std::size_t fallback_after_be = 0;
};

ForcingChecks CheckForcedTrace(
    const MnaSystem &system, const TransientResult &result, double maximum_step,
    const std::function<long double(long double)> &forcing,
    bool exact_cubic_defect = false, std::vector<double> hard_points = {}) {
  ForcingChecks checks;
  hard_points.push_back(result.times_seconds.back());
  const auto out = HistoryNode(system, "out");
  long double accepted_value = 0;
  std::size_t accepted_index = 0;
  bool have_derivative = false;
  std::optional<long double> older_time;
  for (std::size_t i = 0; i < result.step_trace.size(); ++i) {
    const auto &record = result.step_trace[i];
    SCOPED_TRACE(record.start_time_seconds);
    EXPECT_EQ(record.start_time_seconds, result.times_seconds[accepted_index]);
    EXPECT_NE(record.rejection_reason,
              TransientStepRejectionReason::kNonlinearConvergence);
    const long double start = record.start_time_seconds;
    const long double end = record.end_time_seconds;
    const long double h = end - start;
    const double middle_double =
        record.start_time_seconds + record.step_size_seconds / 2;
    const long double middle = middle_double;
    const bool trap = record.method == TransientIntegrationMethod::kTrapezoidal;
    const long double full =
        accepted_value +
        (trap ? h * (forcing(start) + forcing(end)) / 2 : h * forcing(end));
    const long double half =
        accepted_value +
        (trap ? (middle - start) * (forcing(start) + forcing(middle)) / 2 +
                    (end - middle) * (forcing(middle) + forcing(end)) / 2
              : (middle - start) * forcing(middle) +
                    (end - middle) * forcing(end));
    const bool history = trap && older_time.has_value();
    long double error;
    long double scale;
    if (history) {
      // Analytic forcing gives exact derivatives at accepted timestamps; this
      // deliberately does not reconstruct derivatives from numerical states.
      const long double k = start - *older_time;
      const long double curvature =
          ((forcing(end) - forcing(start)) / h -
           (forcing(start) - forcing(*older_time)) / k) /
          (h + k);
      const long double signed_error = h * h * h * curvature / 6;
      if (exact_cubic_defect)
        EXPECT_NEAR(static_cast<double>(signed_error),
                    static_cast<double>(h * h * h / 2), 1e-15);
      error = std::abs(signed_error);
      scale = 1e-7L + 1e-4L * std::max(std::abs(full),
                                       std::abs(full - .75L * signed_error));
      ++checks.history_trials;
      if (std::abs(h - k) > h * 1e-6L)
        ++checks.unequal_history_trials;
    } else {
      error = (trap ? 4.0L / 3 : 1.0L) * std::abs(full - half);
      scale = 1e-7L + 1e-4L * std::max(std::abs(full), std::abs(half));
      if (trap)
        ++checks.fallback_after_be;
    }
    double expected_error = static_cast<double>(error / scale);
    if (history) {
      // Governed audits may replace the history estimate with a real pair of
      // half steps. Independently derive both permitted estimates; the exact
      // quadrature oracle and accepted trajectory remain the same for either.
      const double audited_error = static_cast<double>(
          (4.0L / 3) * std::abs(full - half) /
          (1e-7L + 1e-4L * std::max(std::abs(full), std::abs(half))));
      if (std::abs(record.normalized_local_error - audited_error) <
          std::abs(record.normalized_local_error - expected_error))
        expected_error = audited_error;
    }
    EXPECT_NEAR(record.normalized_local_error, expected_error,
                2e-5 + 5e-4 * expected_error);
    if (std::abs(expected_error - 1) > 1e-3)
      EXPECT_EQ(record.accepted, expected_error <= 1);
    if (record.accepted) {
      accepted_value = trap ? full : half;
      ++accepted_index;
      EXPECT_NEAR(result.states[accepted_index][out],
                  static_cast<double>(accepted_value), 1e-10);
      if (trap && have_derivative)
        older_time = start;
      else
        older_time.reset();
      have_derivative = true;
    } else {
      ++checks.rejected;
    }
    if (expected_error > 0 && i + 1 < result.step_trace.size()) {
      const auto &next = result.step_trace[i + 1];
      const double factor =
          std::clamp(.9 * (trap ? std::cbrt(1 / expected_error)
                                : std::sqrt(1 / expected_error)),
                     .5, 2.0);
      const double proposed =
          std::min(maximum_step, record.step_size_seconds * factor);
      if (!next.landed_on_hard_point) {
        double scheduled = proposed;
        // A representable timestamp can leave a positive sub-minimum tail
        // before a known corner. The scheduler divides that remaining interval
        // in half, rather than creating a subsequent illegal tiny step.
        for (const double hard_point : hard_points) {
          const double tail = hard_point - next.start_time_seconds - proposed;
          if (hard_point > next.start_time_seconds && tail > 0 &&
              tail < maximum_step / 10000) {
            scheduled = (hard_point - next.start_time_seconds) / 2;
            break;
          }
        }
        EXPECT_NEAR(next.step_size_seconds, scheduled,
                    5e-4 * scheduled + 1e-12);
        if (history && proposed < maximum_step &&
            std::abs(factor - std::clamp(.9 * std::sqrt(1 / expected_error), .5,
                                         2.0)) > .02)
          ++checks.distinctive_controller;
      }
    }
  }
  EXPECT_EQ(accepted_index + 1, result.states.size());
  return checks;
}

TEST(Emi02HistoryLte, UnequalGridCubicMatchesIndependentQuadratureDefect) {
  const auto system = ForcedPolynomial(false);
  const auto result =
      RunTransientAnalysis(system, {.05, .2, 0, false}, HistoryLimits());
  ASSERT_TRUE(result.ok()) << result.error().message;
  const auto checks = CheckForcedTrace(
      system, result.value(), .05, [](long double t) { return 3 * t * t; },
      true);
  EXPECT_GT(checks.history_trials, 5U);
  EXPECT_GT(checks.unequal_history_trials, 3U);
  EXPECT_GT(checks.rejected, 0U);
  EXPECT_GT(checks.distinctive_controller, 0U);
  const auto out = HistoryNode(system, "out");
  for (std::size_t i = 0; i < result.value().states.size(); ++i) {
    const double t = result.value().times_seconds[i];
    EXPECT_NEAR(result.value().states[i][out], t * t * t, 2e-5);
  }
}

TEST(Emi02HistoryLte, HigherOrderForcingConvergesToItsIndependentPrimitive) {
  const auto system = ForcedPolynomial(true);
  const auto out = HistoryNode(system, "out");
  std::vector<double> errors;
  for (const double h : {.002, .001}) {
    const auto result =
        RunTransientAnalysis(system, {h, .2, 0, false}, HistoryLimits());
    ASSERT_TRUE(result.ok()) << result.error().message;
    const auto checks = CheckForcedTrace(
        system, result.value(), h, [](long double t) { return 4 * t * t * t; });
    EXPECT_GT(checks.history_trials, 20U);
    double maximum = 0;
    for (std::size_t i = 0; i < result.value().states.size(); ++i) {
      const double t = result.value().times_seconds[i];
      maximum = std::max(
          maximum, std::abs(result.value().states[i][out] - t * t * t * t));
    }
    EXPECT_LT(maximum, 2e-7);
    errors.push_back(maximum);
  }
  EXPECT_LT(errors[1], .35 * errors[0]);
}

TEST(Emi02HistoryLte, HardPointAndBackwardEulerDiscardOlderHistory) {
  const auto system = ForcedPolynomial(true, true);
  const auto result =
      RunTransientAnalysis(system, {.007, .11, 0, false}, HistoryLimits());
  ASSERT_TRUE(result.ok()) << result.error().message;
  const auto checks = CheckForcedTrace(
      system, result.value(), .007,
      [](long double t) {
        const long double clock = t < .04L ? t : t < .08L ? .04L : .12L - t;
        return 4 * clock * clock * clock;
      },
      false, {.04, .08});
  EXPECT_GE(checks.fallback_after_be, 3U);
  for (const double corner : {.04, .08}) {
    const auto found = std::find_if(
        result.value().step_trace.begin(), result.value().step_trace.end(),
        [corner](const auto &step) {
          return step.accepted && step.end_time_seconds == corner;
        });
    ASSERT_NE(found, result.value().step_trace.end());
    EXPECT_EQ(found->method, TransientIntegrationMethod::kBackwardEuler);
    EXPECT_TRUE(found->landed_on_hard_point);
    const auto next =
        std::find_if(found + 1, result.value().step_trace.end(),
                     [](const auto &step) { return step.accepted; });
    ASSERT_NE(next, result.value().step_trace.end());
    EXPECT_EQ(next->method, TransientIntegrationMethod::kBackwardEuler);
  }
}

TEST(Emi02HistoryLte, RcMatchesAnalyticRampAndSavesImplicitSolves) {
  const auto system = HistoryCompile(
      "Vdrive drive 0 DC 0 PWL(0 0 100u 1 1m 1)\n"
      "Rcharge drive out 1k\nCstore out 0 1u\nBzero out 0 I={0}\n");
  const auto out = HistoryNode(system, "out");
  const TranAnalysis analysis{5e-6, 1e-3, 0, false};
  const auto history = RunTransientAnalysis(system, analysis, HistoryLimits());
  ASSERT_TRUE(history.ok()) << history.error().message;
  const auto doubled = RunTransientAnalysis(system, analysis);
  ASSERT_TRUE(doubled.ok()) << doubled.error().message;
  const double gain = .001 / (.001 + 1e-12), tau = 1e-6 / (.001 + 1e-12);
  const auto ramp = [&](double t) {
    return t <= 0 ? 0.0 : gain * (t + tau * std::expm1(-t / tau)) / 1e-4;
  };
  for (const auto *result : {&history.value(), &doubled.value()}) {
    for (std::size_t i = 0; i < result->states.size(); ++i) {
      const double t = result->times_seconds[i];
      EXPECT_NEAR(result->states[i][out], ramp(t) - ramp(t - 1e-4), 2e-5);
    }
  }
  EXPECT_LT(history.value().solver_statistics.solves,
            .6 * doubled.value().solver_statistics.solves);
}

TEST(Emi02HistoryLte, CommonModeDoesNotHideSmallPhysicalRcChanges) {
  for (const int bias : {0, 400}) {
    const auto system = HistoryCompile(
        "Vreference ref 0 " + std::to_string(bias) +
        "\n"
        "Vdrive input ref DC 0 PWL(0 0 100u 1m 500u 1m)\n"
        "Rcharge input out 1k\nCstore out ref 1u\nBzero out ref I={0}\n"
        "Icomp 0 out " +
        std::to_string(bias) + "p\n");
    const auto out = HistoryNode(system, "out"),
               reference = HistoryNode(system, "ref");
    const auto result =
        RunTransientAnalysis(system, {1e-6, 500e-6, 0, false}, HistoryLimits());
    ASSERT_TRUE(result.ok()) << result.error().message;
    const double gain = .001 * .001 / (.001 + 1e-12),
                 tau = 1e-6 / (.001 + 1e-12);
    const auto ramp = [&](double t) {
      return t <= 0 ? 0.0 : gain * (t + tau * std::expm1(-t / tau)) / 1e-4;
    };
    for (std::size_t i = 0; i < result.value().states.size(); ++i) {
      const double t = result.value().times_seconds[i];
      EXPECT_NEAR(result.value().states[i][out] -
                      result.value().states[i][reference],
                  ramp(t) - ramp(t - 1e-4), 2e-8);
    }
  }
}

TEST(Emi02HistoryLte, StiffRlcMatchesTwoIndependentPhysicalTimeConstants) {
  const auto system = HistoryCompile(
      "Vdrive drive 0 DC 0 PWL(0 0 20u 1 80u 1)\n"
      "Rloss drive package 1k\nLpackage package out 100u\nCstore out 0 10n\n"
      "Bzero out 0 I={0}\n");
  const auto out = HistoryNode(system, "out");
  const auto branch = system.node_names.size() +
                      static_cast<std::size_t>(
                          std::find(system.branch_names.begin(),
                                    system.branch_names.end(), "Lpackage") -
                          system.branch_names.begin());
  ASSERT_LT(branch, system.g.rows);
  constexpr double sum = 1e7, product = 1e12;
  const double discriminant = std::sqrt(sum * sum - 4 * product);
  const double slow = -2 * product / (sum + discriminant),
               fast = -(sum + discriminant) / 2;
  EXPECT_GT(fast / slow, 90);
  const auto step = [&](double t) {
    return t <= 0
               ? 0.0
               : 1 + (fast * std::exp(slow * t) - slow * std::exp(fast * t)) /
                         (slow - fast);
  };
  const auto ramp = [&](double t) {
    return t <= 0 ? 0.0
                  : t + (fast * std::expm1(slow * t) / slow -
                         slow * std::expm1(fast * t) / fast) /
                            (slow - fast);
  };
  std::vector<double> errors;
  // Resolve the roughly 100 ns fast time constant at the source corner, where
  // the governed backward-Euler restart has first-order global accuracy.
  for (const double h : {12.5e-9, 6.25e-9}) {
    SCOPED_TRACE(h);
    const auto result =
        RunTransientAnalysis(system, {h, 80e-6, 0, false}, HistoryLimits());
    ASSERT_TRUE(result.ok()) << result.error().message;
    // Independently solve the two physical equations v'=i/C and
    // i'=(u-v-R*i)/L with scalar 2x2 formulas. This checks each accepted BE
    // or TRAP state separately from the continuous closed-form oracle below.
    const auto advance = [](std::array<long double, 2> prior, long double start,
                            long double end, bool trap) {
      const auto drive = [](long double t) {
        return std::min(t / 20e-6L, 1.0L);
      };
      const long double q = (end - start) / (trap ? 2 : 1);
      const long double rhs_v = prior[0] + (trap ? q * prior[1] / 1e-8L : 0);
      const long double rhs_i =
          prior[1] +
          q / 1e-4L *
              (drive(end) +
               (trap ? drive(start) - prior[0] - 1000 * prior[1] : 0));
      const long double b = -q / 1e-8L, c = q / 1e-4L, d = 1 + q * 1000 / 1e-4L;
      const long double determinant = d - b * c;
      return std::array<long double, 2>{(d * rhs_v - b * rhs_i) / determinant,
                                        (rhs_i - c * rhs_v) / determinant};
    };
    std::array<long double, 2> discrete{0, 0};
    std::size_t accepted = 0;
    for (const auto &record : result.value().step_trace) {
      if (!record.accepted)
        continue;
      if (record.method == TransientIntegrationMethod::kBackwardEuler) {
        const double middle =
            record.start_time_seconds + record.step_size_seconds / 2;
        discrete = advance(discrete, record.start_time_seconds, middle, false);
        discrete = advance(discrete, middle, record.end_time_seconds, false);
      } else {
        discrete = advance(discrete, record.start_time_seconds,
                           record.end_time_seconds, true);
      }
      ++accepted;
      ASSERT_NEAR(result.value().states[accepted][out],
                  static_cast<double>(discrete[0]), 1e-8);
      ASSERT_NEAR(result.value().states[accepted][branch],
                  static_cast<double>(discrete[1]), 1e-11);
    }
    double maximum = 0;
    for (std::size_t i = 0; i < result.value().states.size(); ++i) {
      const double t = result.value().times_seconds[i];
      SCOPED_TRACE(t);
      const double voltage = (ramp(t) - ramp(t - 20e-6)) / 20e-6;
      const double current = 1e-8 * (step(t) - step(t - 20e-6)) / 20e-6;
      maximum =
          std::max(maximum, std::abs(result.value().states[i][out] - voltage));
      EXPECT_NEAR(result.value().states[i][branch], current, 2e-8);
    }
    EXPECT_LT(maximum, 2e-5);
    errors.push_back(maximum);
  }
  EXPECT_LT(errors[1], .65 * errors[0]);
}

TEST(Emi02HistoryLte, NonlinearBranchTransitionMatchesPiecewiseAnalyticRc) {
  const auto system =
      HistoryCompile("Icharge 0 out DC 0 PWL(0 0 1u 1m 1m 1m)\n"
                     "Rload out 0 1k\nCstore out 0 1u\n"
                     "Bload out 0 I={if(v(out)>.2,.002*(v(out)-.2),0)}\n");
  const auto out = HistoryNode(system, "out");
  const double g = .001 + 1e-12, tau = 1e-6 / g, equilibrium = .001 / g;
  const auto ramp = [&](double t) {
    return equilibrium * (t + tau * std::expm1(-t / tau)) / 1e-6;
  };
  const double first = ramp(1e-6);
  const double crossing =
      1e-6 - tau * std::log((equilibrium - .2) / (equilibrium - first));
  const auto exact = [&](double t) {
    if (t <= 1e-6)
      return ramp(t);
    if (t <= crossing)
      return equilibrium + (first - equilibrium) * std::exp(-(t - 1e-6) / tau);
    const double final = .0014 / (g + .002);
    return final + (.2 - final) * std::exp(-(g + .002) * (t - crossing) / 1e-6);
  };
  const auto result =
      RunTransientAnalysis(system, {5e-6, 1e-3, 0, false}, HistoryLimits());
  ASSERT_TRUE(result.ok()) << result.error().message;
  bool before = false, after = false;
  for (std::size_t i = 0; i < result.value().states.size(); ++i) {
    const double t = result.value().times_seconds[i];
    EXPECT_NEAR(result.value().states[i][out], exact(t), 2e-5);
    before |= t > crossing - 20e-6 && t < crossing;
    after |= t > crossing && t < crossing + 20e-6;
  }
  EXPECT_TRUE(before);
  EXPECT_TRUE(after);
}

TEST(Emi02HistoryLte, PositiveFeedbackAuditsAStiffGrowingRcTrial) {
  constexpr double h = .00199, release = .01195;
  const auto system = HistoryCompile(
      "Vclock clock 0 DC 0 PWL(0 0 20m .02)\n"
      "Rload out 0 1k\nCstore out 0 1u\n"
      "Bfeedback out 0 I={-.002*v(out)-if(v(clock)>.01195,1e-12,0)}\n");
  const auto out = HistoryNode(system, "out");
  const auto result =
      RunTransientAnalysis(system, {h, 8 * h, 0, false}, HistoryLimits());
  ASSERT_TRUE(result.ok()) << result.error().message;
  constexpr long double lambda = (.001L - 1e-12L) / 1e-6L;
  const auto source = [](long double t) { return t <= release ? 0.0L : 1e-6L; };
  const auto trap = [&](long double y, long double start, long double end) {
    const long double dt = end - start;
    return ((1 + lambda * dt / 2) * y +
            dt * (source(start) + source(end)) / 2) /
           (1 - lambda * dt / 2);
  };
  std::size_t index = 0, guarded = 0, underestimated = 0;
  bool saw_coarse_release = false;
  for (const auto &record : result.value().step_trace) {
    const long double previous = result.value().states[index][out];
    if (record.method == TransientIntegrationMethod::kTrapezoidal) {
      const long double start = record.start_time_seconds,
                        end = record.end_time_seconds;
      const long double dt = end - start, middle = start + dt / 2;
      const long double full = trap(previous, start, end);
      const long double half = trap(trap(previous, start, middle), middle, end);
      const long double before = lambda * previous + source(start);
      const long double after = lambda * full + source(end);
      const long double delta = full - previous;
      const long double tolerance =
          1e-7L + 1e-4L * std::max(std::abs(previous), std::abs(full));
      if (std::abs(delta) > .01L * tolerance &&
          dt * (after - before) / delta >= .5L) {
        ++guarded;
        const double expected = static_cast<double>(
            (4.0L / 3) * std::abs(full - half) /
            (1e-7L + 1e-4L * std::max(std::abs(full), std::abs(half))));
        EXPECT_NEAR(record.normalized_local_error, expected,
                    2e-4 + 1e-3 * expected);
        if (expected > 1.001)
          EXPECT_FALSE(record.accepted);
        if (std::abs(record.start_time_seconds - 6 * h) < 1e-12 &&
            std::abs(record.end_time_seconds - 7 * h) < 1e-12) {
          saw_coarse_release = true;
          // The preceding accepted derivatives are zero at equilibrium. The
          // raw defect would accept this ~200-fold amplification, but the
          // independent two-half solve detects its error and must reject it.
          const long double raw = dt * (after - before) / 12;
          const long double raw_normalized =
              std::abs(raw) /
              (1e-7L +
               1e-4L * std::max(std::abs(full), std::abs(full - .75L * raw)));
          EXPECT_LT(raw_normalized, 1);
          EXPECT_GT(expected, 1);
          ++underestimated;
        }
      }
    }
    if (record.accepted)
      ++index;
  }
  EXPECT_GT(guarded, 0U);
  EXPECT_GT(underestimated, 0U);
  EXPECT_TRUE(saw_coarse_release);
  EXPECT_EQ(index + 1, result.value().states.size());
}

TEST(Emi02HistoryLte, ZeroEndpointCubicPulseCannotEvadeTheMidpointAudit) {
  for (const bool pulse : {false, true}) {
    const auto system = HistoryCompile(
        "Vclock clock 0 DC 0 PWL(0 0 6m .006)\n"
        "Rload out 0 1\nCstore out 0 1u\n"
        "Bforce out 0 I={if(v(clock)>0,-1.000000000001*v(out)-" +
        std::string(pulse ? "if(v(clock)>.003,if(v(clock)<.004,100*(v(clock)-."
                            "002)*(v(clock)-.003)*(.004-v(clock)),0),0)"
                          : "0") +
        ",0)}\n");
    const auto out = HistoryNode(system, "out");
    const auto result =
        RunTransientAnalysis(system, {.001, .006, 0, false}, HistoryLimits());
    ASSERT_TRUE(result.ok()) << result.error().message;
    const auto found = std::find_if(
        result.value().step_trace.begin(), result.value().step_trace.end(),
        [](const auto &step) {
          return std::abs(step.start_time_seconds - .003) < 1e-12 &&
                 std::abs(step.end_time_seconds - .004) < 1e-12;
        });
    ASSERT_NE(found, result.value().step_trace.end());
    EXPECT_EQ(found->method, TransientIntegrationMethod::kTrapezoidal);
    if (pulse) {
      EXPECT_FALSE(found->accepted);
      EXPECT_EQ(found->rejection_reason,
                TransientStepRejectionReason::kLocalError);
      // Analytic integral of 1e8*(h+u)*u*(h-u), u in [0,h], is 1e8*h^4/4.
      // All three derivative stencil samples vanish; the midpoint is nonzero.
      EXPECT_GT(found->normalized_local_error, 100);
      EXPECT_NEAR(result.value().states.back()[out], 2.5e-5, 5e-7);
    } else {
      EXPECT_TRUE(found->accepted);
      EXPECT_EQ(found->normalized_local_error, 0);
      EXPECT_EQ(result.value().states.back()[out], 0);
    }
  }
}

TEST(Emi02HistoryLte, RejectedTrialsCannotAdvancePublishedOrRestartedHistory) {
  const auto system =
      HistoryCompile("Icharge 0 out DC 0 PWL(0 0 100u 10m 1m 10m)\n"
                     "Cbase out sense 1u\nVsense sense 0 0\n"
                     "Bextra out 0 I={i(Vsense)*(1+10*v(out)*v(out))}\n");
  auto limits = HistoryLimits();
  limits.nonlinear_maximum_iterations = 2;
  const TranAnalysis analysis{5e-6, 3e-4, 0, false};
  const auto baseline = RunTransientAnalysis(system, analysis, limits);
  ASSERT_TRUE(baseline.ok()) << baseline.error().message;
  for (const auto reason :
       {TransientStepRejectionReason::kLocalError,
        TransientStepRejectionReason::kNonlinearConvergence}) {
    const auto found = std::find_if(
        baseline.value().step_trace.begin(), baseline.value().step_trace.end(),
        [reason](const auto &step) {
          return !step.accepted && step.rejection_reason == reason;
        });
    ASSERT_NE(found, baseline.value().step_trace.end());
    auto bounded = limits;
    bounded.maximum_step_attempts =
        static_cast<std::size_t>(found - baseline.value().step_trace.begin()) +
        1;
    bounded.maximum_accepted_steps = bounded.maximum_step_attempts;
    std::vector<double> times;
    std::vector<std::vector<double>> states;
    bounded.accepted_state_observer = [&](double t, const auto &state) {
      times.push_back(t);
      states.push_back(state);
      return Result<bool>::Ok(true);
    };
    const auto failed = RunTransientAnalysis(system, analysis, bounded);
    ASSERT_FALSE(failed.ok());
    const auto accepted = static_cast<std::size_t>(
        std::count_if(baseline.value().step_trace.begin(), found,
                      [](const auto &step) { return step.accepted; }));
    ASSERT_EQ(states.size(), accepted + 1);
    EXPECT_TRUE(std::equal(times.begin(), times.end(),
                           baseline.value().times_seconds.begin()));
    EXPECT_TRUE(std::equal(states.begin(), states.end(),
                           baseline.value().states.begin()));
  }
  const auto restarted = RunTransientAnalysis(system, analysis, limits);
  ASSERT_TRUE(restarted.ok()) << restarted.error().message;
  EXPECT_EQ(restarted.value().times_seconds, baseline.value().times_seconds);
  EXPECT_EQ(restarted.value().states, baseline.value().states);
}

TEST(Emi02HistoryLte,
     InvalidPolicyNativeSelectionAndDerivativeBoundsFailClosed) {
  const auto behavioral = ForcedPolynomial(false);
  auto invalid = HistoryLimits();
  invalid.behavioral_error_estimator =
      static_cast<BehavioralErrorEstimator>(99);
  const auto unknown =
      RunTransientAnalysis(behavioral, {.01, .1, 0, false}, invalid);
  ASSERT_FALSE(unknown.ok());
  EXPECT_EQ(unknown.error().code, ErrorCode::kInvalidStructure);
  const auto native =
      HistoryCompile("Vdrive in 0 1\nRload in out 1k\nCstore out 0 1u\n");
  const auto rejected =
      RunTransientAnalysis(native, {1e-6, 1e-4, 0, false}, HistoryLimits());
  ASSERT_FALSE(rejected.ok());
  EXPECT_EQ(rejected.error().code, ErrorCode::kUnsupported);
  EXPECT_TRUE(RunTransientAnalysis(native, {1e-6, 1e-4, 0, false}).ok());
  const auto steep =
      HistoryCompile("Vdrive out 0 DC 0 PWL(0 0 1p 1e99)\n"
                     "Rload out 0 1\nCstore out 0 1p\nBzero out 0 I={0}\n");
  std::size_t observed = 0;
  auto bounded = HistoryLimits();
  bounded.accepted_state_observer = [&](double, const auto &) {
    ++observed;
    return Result<bool>::Ok(true);
  };
  const auto overflow =
      RunTransientAnalysis(steep, {1e-12, 1e-12, 0, false}, bounded);
  ASSERT_FALSE(overflow.ok());
  EXPECT_EQ(overflow.error().code, ErrorCode::kNonFinite);
  EXPECT_EQ(observed, 1U);
}

} // namespace
} // namespace ohmnivore
