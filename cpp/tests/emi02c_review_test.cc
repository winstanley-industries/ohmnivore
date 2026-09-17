#include "cpp/tests/google_test.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <string>
#include <vector>

#include "ohmnivore/behavioral.h"
#include "ohmnivore/transient.h"

namespace ohmnivore {
namespace {

MnaSystem ReviewCompile(const std::string &text) {
  const auto parsed = ParseBehavioralNetlist(text);
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

std::size_t ReviewNode(const MnaSystem &system, const std::string &name) {
  const auto found =
      std::find(system.node_names.begin(), system.node_names.end(), name);
  EXPECT_NE(found, system.node_names.end());
  return static_cast<std::size_t>(found - system.node_names.begin());
}

std::size_t ReviewBranch(const MnaSystem &system, const std::string &name) {
  const auto found =
      std::find(system.branch_names.begin(), system.branch_names.end(), name);
  EXPECT_NE(found, system.branch_names.end());
  return system.node_names.size() +
         static_cast<std::size_t>(found - system.branch_names.begin());
}

// Independent RK4 integration of explicitly written physical equations. This
// never reads G/C, expression trees, stamps, residuals, or their derivatives.
template <std::size_t N, typename Derivative>
std::array<double, N> ReviewIntegrate(std::array<double, N> state, double start,
                                      double stop, Derivative derivative) {
  const auto steps = static_cast<std::size_t>(std::ceil((stop - start) / 2e-8));
  if (steps == 0)
    return state;
  const double h = (stop - start) / static_cast<double>(steps);
  for (std::size_t step = 0; step < steps; ++step) {
    const double time = start + static_cast<double>(step) * h;
    const auto a = derivative(time, state);
    auto shifted = state;
    for (std::size_t i = 0; i < N; ++i)
      shifted[i] += h * a[i] / 2;
    const auto b = derivative(time + h / 2, shifted);
    shifted = state;
    for (std::size_t i = 0; i < N; ++i)
      shifted[i] += h * b[i] / 2;
    const auto c = derivative(time + h / 2, shifted);
    shifted = state;
    for (std::size_t i = 0; i < N; ++i)
      shifted[i] += h * c[i];
    const auto d = derivative(time + h, shifted);
    for (std::size_t i = 0; i < N; ++i)
      state[i] += h * (a[i] + 2 * b[i] + 2 * c[i] + d[i]) / 6;
  }
  return state;
}

TEST(Emi02CReview,
     PeriodicAffineCapacitanceReturnsChargeAndKeepsCorrectPrimitive) {
  const auto system = ReviewCompile(
      "Vdrive drive 0 DC 0 PWL(0 0 10u 0 260u .5 510u 0 760u -.5 1010u 0 1020u "
      "0)\n"
      "Cbase drive sense 1u\nVsense sense 0 0\n"
      "Bextra drive 0 I={i(Vsense)*(2+0.5*v(drive))}\n.TRAN 250n 1020u\n");
  ASSERT_GT(system.g.rows, 0U);
  double last_accepted = 0.0;
  TransientExecutionLimits periodic_limits;
  periodic_limits.accepted_state_observer = [&](double time,
                                                const std::vector<double> &) {
    last_accepted = time;
    return Result<bool>::Ok(true);
  };
  const auto result = RunTransientAnalysis(
      system, {250e-9, 1020e-6, 0.0, false}, periodic_limits);
  ASSERT_TRUE(result.ok()) << result.error().message
                           << " after t=" << last_accepted;
  const auto voltage = ReviewNode(system, "drive");
  const auto drive = ReviewBranch(system, "Vdrive");
  const auto sensed = ReviewBranch(system, "Vsense");
  double charge = 0.0;
  for (std::size_t sample = 1; sample < result.value().states.size();
       ++sample) {
    const auto &previous = result.value().states[sample - 1];
    const auto &state = result.value().states[sample];
    const double h = result.value().times_seconds[sample] -
                     result.value().times_seconds[sample - 1];
    charge -= h * (state[drive] + previous[drive]) / 2;
    const double v = state[voltage];
    // C(v)=C0*(3+.5*v): its primitive is C0*(3*v+.25*v^2),
    // whereas the incorrect Q=C(v)*v doubles the quadratic term.
    const double expected_charge = 1e-6 * (3 * v + 0.25 * v * v);
    EXPECT_NEAR(charge, expected_charge, 3e-9);
    EXPECT_NEAR(-state[drive], state[sensed] * (3 + 0.5 * v), 1e-10);
    const double time = result.value().times_seconds[sample];
    const std::array<double, 5> corners = {10e-6, 260e-6, 510e-6, 760e-6,
                                           1010e-6};
    const bool near_corner =
        std::any_of(corners.begin(), corners.end(), [time](double corner) {
          return std::abs(time - corner) < 1e-6;
        });
    if (!near_corner) {
      const double derivative =
          time < 10e-6 || time > 1010e-6
              ? 0.0
              : (time < 260e-6 || time > 760e-6 ? 2000.0 : -2000.0);
      const double reference = 1e-6 * (3 + 0.5 * v) * derivative;
      EXPECT_NEAR(-state[drive], reference, 1e-6 + 1e-3 * std::abs(reference));
    }
  }
  EXPECT_NEAR(charge, 0.0, 3e-9);
  EXPECT_NEAR(result.value().states.back()[voltage], 0.0, 1e-12);
}

TEST(Emi02CReview, FinitePwellRetainsAnIndependentCapacitorVoltage) {
  const auto system = ReviewCompile(
      "Vexternal drain 0 DC 0 PWL(0 0 100u 1 300u 1)\n"
      "Rpwell drain well 10\nCbase well sense 1u\nVsense sense 0 0\n"
      "Bextra well 0 I={i(Vsense)*(2+0.5*v(drain))}\n.TRAN 500n 300u\n");
  ASSERT_GT(system.g.rows, 0U);
  double last_accepted = 0.0;
  TransientExecutionLimits pwell_limits;
  pwell_limits.accepted_state_observer = [&](double time,
                                             const std::vector<double> &) {
    last_accepted = time;
    return Result<bool>::Ok(true);
  };
  const auto result =
      RunTransientAnalysis(system, {50e-9, 300e-6, 0.0, false}, pwell_limits);
  ASSERT_TRUE(result.ok()) << result.error().message
                           << " after t=" << last_accepted;
  EXPECT_EQ(result.value().times_seconds.back(), 300e-6);
  for (const auto &record : result.value().step_trace) {
    EXPECT_LE(record.step_size_seconds, 50e-9);
  }
  const auto drain = ReviewNode(system, "drain");
  const auto well = ReviewNode(system, "well");
  const auto sense = ReviewBranch(system, "Vsense");
  const auto supply = ReviewBranch(system, "Vexternal");
  const auto external = [](double time) {
    return std::min(time / 100e-6, 1.0);
  };
  const auto derivative = [&](double time, std::array<double, 1> x) {
    const double v = external(time);
    return std::array<double, 1>{((v - x[0]) / 10 - 1e-12 * x[0]) /
                                 (1e-6 * (3 + 0.5 * v))};
  };
  std::array<double, 1> oracle = {0.0};
  double previous = 0.0;
  double maximum_lag = 0.0;
  for (std::size_t sample = 0; sample < result.value().states.size();
       ++sample) {
    const double time = result.value().times_seconds[sample];
    SCOPED_TRACE(time);
    oracle = ReviewIntegrate(oracle, previous, time, derivative);
    previous = time;
    const auto &state = result.value().states[sample];
    maximum_lag = std::max(maximum_lag, state[drain] - state[well]);
    ASSERT_NEAR(state[well], oracle[0], 1e-5 + 1e-3 * std::abs(oracle[0]));
    const double resistor_current = (external(time) - oracle[0]) / 10;
    ASSERT_NEAR(-state[supply], resistor_current,
                1e-6 + 1e-3 * std::abs(resistor_current));
    ASSERT_NEAR(state[sense], 1e-6 * derivative(time, oracle)[0], 1e-6);
    // Both sensed-current and voltage-control terms are simultaneous. The
    // finite resistor makes the two voltages measurably different.
    EXPECT_NEAR((state[drain] - state[well]) / 10,
                state[sense] * (3 + 0.5 * state[drain]) + 1e-12 * state[well],
                1e-9);
  }
  EXPECT_GT(maximum_lag, 0.2);
  EXPECT_LT(std::abs(result.value().states.back()[drain] -
                     result.value().states.back()[well]),
            0.002);
}

TEST(Emi02CReview, PackageRlcResonanceMatchesIndependentPhysicalState) {
  const auto system = ReviewCompile(
      "Vcommand command 0 DC 0 PWL(0 0 1u 1 1.2m 1)\n"
      "Edrive drive 0 VALUE={v(command)}\nRpackage drive pin 2\n"
      "Lpackage pin out 1m\nCbase out sense 1u\nVsense sense 0 0\n"
      "Bextra out 0 I={2*i(Vsense)}\n.TRAN 500n 1.2m\n");
  ASSERT_GT(system.g.rows, 0U);
  const auto result = RunTransientAnalysis(system, {2e-9, 900e-6, 0.0, false});
  ASSERT_TRUE(result.ok()) << result.error().message;
  const auto out = ReviewNode(system, "out");
  const auto current = ReviewBranch(system, "Lpackage");
  const auto sensed = ReviewBranch(system, "Vsense");
  const auto discrete_step = [](std::array<double, 2> x, double start,
                                double end, bool trap) {
    const double h = end - start;
    const double alpha = trap ? 2.0 : 1.0;
    const double resistance = 2.0 / (1.0 + 2e-12);
    const auto input = [](double time) {
      return std::min(time / 1e-6, 1.0) / (1.0 + 2e-12);
    };
    const double aa = alpha * 1e-3 / h + resistance;
    const double dd = alpha * 3e-6 / h + 1e-12;
    const double first = alpha * 1e-3 * x[0] / h + input(end) +
                         (trap ? input(start) - resistance * x[0] - x[1] : 0.0);
    const double second =
        alpha * 3e-6 * x[1] / h + (trap ? x[0] - 1e-12 * x[1] : 0.0);
    return std::array<double, 2>{(dd * first - second) / (aa * dd + 1),
                                 (first + aa * second) / (aa * dd + 1)};
  };
  std::array<double, 2> discrete = {0.0, 0.0};
  std::size_t discrete_index = 0;
  for (const auto &record : result.value().step_trace) {
    if (!record.accepted)
      continue;
    if (record.method == TransientIntegrationMethod::kBackwardEuler) {
      const double midpoint =
          record.start_time_seconds + record.step_size_seconds / 2;
      discrete =
          discrete_step(discrete, record.start_time_seconds, midpoint, false);
      discrete =
          discrete_step(discrete, midpoint, record.end_time_seconds, false);
    } else {
      discrete = discrete_step(discrete, record.start_time_seconds,
                               record.end_time_seconds, true);
    }
    ++discrete_index;
    ASSERT_NEAR(result.value().states[discrete_index][current], discrete[0],
                1e-9 + 1e-6 * std::abs(discrete[0]))
        << "discrete t=" << record.end_time_seconds;
    ASSERT_NEAR(result.value().states[discrete_index][out], discrete[1],
                1e-8 + 1e-6 * std::abs(discrete[1]))
        << "discrete t=" << record.end_time_seconds;
  }
  const auto derivative = [](double time, std::array<double, 2> x) {
    const double source = std::min(time / 1e-6, 1.0);
    const double package_pin = (source - 2 * x[0]) / (1 + 2e-12);
    return std::array<double, 2>{(package_pin - x[1]) / 1e-3,
                                 (x[0] - 1e-12 * x[1]) / 3e-6};
  };
  std::array<double, 2> oracle = {0.0, 0.0};
  double previous = 0.0;
  std::vector<std::array<double, 2>> peaks;
  const auto backward_steps = std::count_if(
      result.value().step_trace.begin(), result.value().step_trace.end(),
      [](const auto &step) {
        return step.accepted &&
               step.method == TransientIntegrationMethod::kBackwardEuler;
      });
  for (std::size_t sample = 0; sample < result.value().states.size();
       ++sample) {
    const double time = result.value().times_seconds[sample];
    SCOPED_TRACE(time);
    oracle = ReviewIntegrate(oracle, previous, time, derivative);
    previous = time;
    const auto &state = result.value().states[sample];
    ASSERT_NEAR(state[current], oracle[0], 1e-6 + 1e-3 * std::abs(oracle[0]))
        << "BE steps=" << backward_steps
        << " samples=" << result.value().states.size();
    ASSERT_NEAR(state[out], oracle[1], 1e-5 + 1e-3 * std::abs(oracle[1]));
    ASSERT_NEAR(state[current] - 3 * state[sensed] - 1e-12 * state[out], 0.0,
                1e-9);
    if (sample > 1 && sample + 1 < result.value().states.size() &&
        state[out] > result.value().states[sample - 1][out] &&
        state[out] >= result.value().states[sample + 1][out] &&
        state[out] > 1.1) {
      peaks.push_back({time, state[out] - 1});
    }
  }
  ASSERT_GE(peaks.size(), 3U);
  const double alpha = 2.0 / (2 * 1e-3);
  const double omega = std::sqrt(1.0 / (1e-3 * 3e-6) - alpha * alpha);
  const double period = 2 * std::numbers::pi / omega;
  for (std::size_t peak = 1; peak < peaks.size(); ++peak) {
    EXPECT_NEAR(peaks[peak][0] - peaks[peak - 1][0], period, 0.01 * period);
    EXPECT_NEAR(std::log(peaks[peak - 1][1] / peaks[peak][1]), alpha * period,
                0.01);
  }
}

TEST(Emi02CReview, NewtonAndLteRejectionsNeverEscapeAcceptedObserverHistory) {
  const auto system = ReviewCompile(
      "Icharge 0 out DC 0 PWL(0 0 100u 10m 1m 10m)\n"
      "Cbase out sense 1u\nVsense sense 0 0\n"
      "Bextra out 0 I={i(Vsense)*(1+10*v(out)*v(out))}\n.TRAN 100u 1m\n");
  ASSERT_GT(system.g.rows, 0U);
  const TranAnalysis analysis{5e-6, 1e-3, 0.0, false};
  TransientExecutionLimits limits;
  limits.nonlinear_maximum_iterations = 2;
  const auto baseline = RunTransientAnalysis(system, analysis, limits);
  ASSERT_TRUE(baseline.ok()) << baseline.error().message;
  const auto out = ReviewNode(system, "out");
  const auto sensed = ReviewBranch(system, "Vsense");
  // The intentionally reduced Newton budget forces repeated BE recovery. Its
  // accepted history is checked against an independently solved discrete
  // scalar constitutive equation, so rejection must never advance the oracle.
  const auto step = [](std::array<double, 2> previous, double start, double end,
                       bool trapezoidal) {
    const double h = end - start;
    const double input = 0.01 * std::min(end / 100e-6, 1.0);
    const double coefficient = (trapezoidal ? 2 : 1) * 1e-6 / h;
    const double history = trapezoidal ? previous[1] : 0.0;
    double low = previous[0], high = 10.0;
    for (int iteration = 0; iteration < 70; ++iteration) {
      const double v = (low + high) / 2;
      const double i = coefficient * (v - previous[0]) - history;
      if (i * (2 + 10 * v * v) + 1e-12 * v < input)
        low = v;
      else
        high = v;
    }
    const double v = (low + high) / 2;
    return std::array<double, 2>{v, coefficient * (v - previous[0]) - history};
  };
  std::array<double, 2> accepted_oracle = {0.0, 0.0};
  std::size_t accepted_index = 0;
  for (const auto &record : baseline.value().step_trace) {
    EXPECT_EQ(record.start_time_seconds,
              baseline.value().times_seconds[accepted_index]);
    if (!record.accepted)
      continue;
    if (record.method == TransientIntegrationMethod::kBackwardEuler) {
      const double midpoint =
          record.start_time_seconds + record.step_size_seconds / 2;
      const auto half =
          step(accepted_oracle, record.start_time_seconds, midpoint, false);
      accepted_oracle = step(half, midpoint, record.end_time_seconds, false);
    } else {
      accepted_oracle = step(accepted_oracle, record.start_time_seconds,
                             record.end_time_seconds, true);
    }
    ++accepted_index;
    ASSERT_NEAR(baseline.value().states[accepted_index][out],
                accepted_oracle[0], 1e-7 + 1e-5 * std::abs(accepted_oracle[0]));
    ASSERT_NEAR(baseline.value().states[accepted_index][sensed],
                accepted_oracle[1], 1e-9 + 1e-5 * std::abs(accepted_oracle[1]));
  }
  // Independently qualify a refined ordinary-policy trajectory against the
  // continuous charge primitive; fault-injection BE accuracy is a separate
  // question from whether rejected discrete histories remain transactional.
  const auto refined = RunTransientAnalysis(system, {100e-9, 1e-3, 0.0, false});
  ASSERT_TRUE(refined.ok()) << refined.error().message;
  for (std::size_t sample = 0; sample < refined.value().states.size();
       ++sample) {
    const double time = refined.value().times_seconds[sample];
    const double charge = time < 100e-6 ? 0.5 * 0.01 * time * time / 100e-6
                                        : 0.01 * (time - 50e-6);
    // Solve the independent monotone primitive Q/C0=2*v+(10/3)*v^3.
    double low = 0.0, high = 10.0;
    for (int iteration = 0; iteration < 60; ++iteration) {
      const double middle = (low + high) / 2;
      if (2 * middle + (10.0 / 3.0) * middle * middle * middle < charge / 1e-6)
        low = middle;
      else
        high = middle;
    }
    ASSERT_NEAR(refined.value().states[sample][out], (low + high) / 2,
                1e-5 + 1e-3 * high);
  }
  for (const auto reason : {TransientStepRejectionReason::kNonlinearConvergence,
                            TransientStepRejectionReason::kLocalError}) {
    const auto rejected = std::find_if(
        baseline.value().step_trace.begin(), baseline.value().step_trace.end(),
        [reason](const auto &step) {
          return !step.accepted && step.rejection_reason == reason;
        });
    ASSERT_NE(rejected, baseline.value().step_trace.end());
    const auto attempts = static_cast<std::size_t>(
                              rejected - baseline.value().step_trace.begin()) +
                          1;
    auto bounded = limits;
    bounded.maximum_step_attempts = attempts;
    bounded.maximum_accepted_steps = attempts;
    std::vector<double> times;
    std::vector<std::vector<double>> states;
    bounded.accepted_state_observer = [&](double time,
                                          const std::vector<double> &state) {
      times.push_back(time);
      states.push_back(state);
      return Result<bool>::Ok(true);
    };
    const auto failure = RunTransientAnalysis(system, analysis, bounded);
    ASSERT_FALSE(failure.ok());
    EXPECT_EQ(failure.error().code,
              reason == TransientStepRejectionReason::kNonlinearConvergence
                  ? ErrorCode::kNonConvergence
                  : ErrorCode::kSolve);
    const auto accepted = static_cast<std::size_t>(
        std::count_if(baseline.value().step_trace.begin(), rejected,
                      [](const auto &step) { return step.accepted; }));
    ASSERT_EQ(times.size(), accepted + 1);
    EXPECT_TRUE(std::equal(times.begin(), times.end(),
                           baseline.value().times_seconds.begin()));
    EXPECT_TRUE(std::equal(states.begin(), states.end(),
                           baseline.value().states.begin()));
    const auto restarted = RunTransientAnalysis(system, analysis, limits);
    ASSERT_TRUE(restarted.ok());
    EXPECT_EQ(restarted.value().times_seconds, baseline.value().times_seconds);
    EXPECT_EQ(restarted.value().states, baseline.value().states);
  }
}

TEST(Emi02CReview,
     TrapezoidalStepDoublingUsesPrivateMidpointForcingAndHistory) {
  const auto system = ReviewCompile("Vinput input 0 DC 0 PWL(0 0 1m 1 2m 1)\n"
                                    "Edrive drive 0 VALUE={v(input)*v(input)}\n"
                                    "Rcharge drive out 1k\nCstate out 0 1u\n"
                                    "Bload out 0 I={.001*v(out)}\n");
  ASSERT_GT(system.g.rows, 0U);
  constexpr double capacitance = 1e-6;
  constexpr double conductance = .002 + 1e-12;
  constexpr double ramp = 1e-3;
  constexpr double maximum_step = 50e-6;
  constexpr double stop = 2e-3;
  const auto forcing = [](double time) {
    const double input = std::min(time / ramp, 1.0);
    return .001 * input * input;
  };
  // Closed scalar constitutive solves for C*x' + (.001+.001+Gmin)*x
  // = .001*min(t/ramp,1)^2, independent of production G/C and residuals.
  const auto step = [&](double previous, double start, double end, bool trap) {
    const double coefficient = (trap ? 2.0 : 1.0) * capacitance / (end - start);
    return ((coefficient - (trap ? conductance : 0.0)) * previous +
            forcing(end) + (trap ? forcing(start) : 0.0)) /
           (coefficient + conductance);
  };
  const auto exact = [](double time) {
    constexpr double decay = conductance / capacitance;
    const auto ramp_value = [](double t) {
      const double u = decay * t;
      return (.001 / capacitance) / (ramp * ramp * decay * decay * decay) *
             (u * u - 2 * u - 2 * std::expm1(-u));
    };
    const double equilibrium = .001 / conductance;
    return time <= ramp ? ramp_value(time)
                        : equilibrium + (ramp_value(ramp) - equilibrium) *
                                            std::exp(-decay * (time - ramp));
  };
  const auto result =
      RunTransientAnalysis(system, {maximum_step, stop, 0, false});
  ASSERT_TRUE(result.ok()) << result.error().message;
  const auto out = ReviewNode(system, "out");
  double previous = 0;
  std::size_t accepted = 0, trapezoidal = 0, distinctive_controllers = 0;
  for (std::size_t index = 0; index < result.value().step_trace.size();
       ++index) {
    const auto &record = result.value().step_trace[index];
    SCOPED_TRACE(record.start_time_seconds);
    ASSERT_NE(record.rejection_reason,
              TransientStepRejectionReason::kNonlinearConvergence);
    const bool trap = record.method == TransientIntegrationMethod::kTrapezoidal;
    const double midpoint =
        record.start_time_seconds + record.step_size_seconds / 2;
    const double full = step(previous, record.start_time_seconds,
                             record.end_time_seconds, trap);
    const double first =
        step(previous, record.start_time_seconds, midpoint, trap);
    const double half = step(first, midpoint, record.end_time_seconds, trap);
    const double expected_error =
        (trap ? 4.0 / 3.0 : 1.0) * std::abs(full - half) /
        (1e-7 + 1e-4 * std::max(std::abs(full), std::abs(half)));
    EXPECT_NEAR(record.normalized_local_error, expected_error,
                2e-7 + 2e-5 * expected_error);
    EXPECT_EQ(record.accepted, expected_error <= 1.0);
    if (trap)
      ++trapezoidal;
    if (record.accepted) {
      previous = trap ? full : half;
      ++accepted;
      ASSERT_NEAR(result.value().states[accepted][out], previous,
                  2e-10 + 1e-8 * std::abs(previous));
      ASSERT_NEAR(result.value().states[accepted][out],
                  exact(record.end_time_seconds),
                  1e-5 + 1e-3 * std::abs(exact(record.end_time_seconds)));
    }
    if (trap && expected_error > 0 &&
        index + 1 < result.value().step_trace.size()) {
      const auto &next = result.value().step_trace[index + 1];
      const double factor =
          std::clamp(.9 * std::cbrt(1 / expected_error), .5, 2.0);
      const double square_root =
          std::clamp(.9 * std::sqrt(1 / expected_error), .5, 2.0);
      const double proposed =
          std::min(record.step_size_seconds * factor, maximum_step);
      const double hard_point = next.start_time_seconds < ramp ? ramp : stop;
      if (!next.landed_on_hard_point && next.start_time_seconds + proposed <
                                            hard_point - maximum_step / 10000) {
        EXPECT_NEAR(next.step_size_seconds, proposed, 1e-10 * proposed + 1e-13);
        if (std::abs(factor - square_root) > .05 && proposed < maximum_step)
          ++distinctive_controllers;
      }
    }
  }
  EXPECT_GT(trapezoidal, 10U);
  EXPECT_GT(distinctive_controllers, 0U);
  EXPECT_EQ(accepted + 1, result.value().states.size());
  EXPECT_EQ(result.value().times_seconds.back(), stop);

  // For a constant-source interval, the exact exponential independently checks
  // cubic local-error scaling and the 4/3 accepted-full-step error coefficient.
  const double start = 1.2e-3;
  const double initial = exact(start);
  const auto estimate = [&](double h) {
    const double full = step(initial, start, start + h, true);
    const double first = step(initial, start, start + h / 2, true);
    const double half = step(first, start + h / 2, start + h, true);
    const double error = 4.0 / 3.0 * std::abs(full - half);
    EXPECT_NEAR(error / std::abs(full - exact(start + h)), 1.0, .02);
    return error;
  };
  EXPECT_NEAR(estimate(10e-6) / estimate(5e-6), 8.0, .2);
}

TEST(Emi02CReview, ReactiveMetadataMustCoverTheEntirePhysicalDynamicMatrix) {
  const auto original = ReviewCompile(
      "Vdrive input 0 0\nRdrive input left 10\n"
      "Cleft left 0 1u\nCbridge left right 2u\n"
      "Cparallel left right 1p\nCscale left right 1f\n"
      "Lleft left 0 1m\nLright right 0 2m\nKpair Lleft Lright .5\n"
      "Bload right 0 I={v(right)*.1}\n");
  ASSERT_GT(original.g.rows, 0U);
  ASSERT_TRUE(ValidateBehavioralTransient(original).ok());
  const auto rejected = [&](const MnaSystem &system,
                            ErrorCode expected = ErrorCode::kInvalidStructure) {
    const auto validation = ValidateBehavioralTransient(system);
    ASSERT_FALSE(validation.ok());
    EXPECT_EQ(validation.error().code, expected);
    std::size_t observed = 0;
    TransientExecutionLimits limits;
    limits.accepted_state_observer = [&](double, const std::vector<double> &) {
      ++observed;
      return Result<bool>::Ok(true);
    };
    const auto result =
        RunTransientAnalysis(system, {1e-6, 2e-6, 0.0, false}, limits);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.error().code, expected);
    EXPECT_EQ(observed, 0U);
  };
  const auto set_c = [](MnaSystem *system, std::size_t row, std::size_t column,
                        double value) {
    auto dense = system->c.ToDense();
    dense[row * system->c.columns + column] = value;
    system->c.values.clear();
    system->c.column_indices.clear();
    system->c.row_offsets.assign(system->c.rows + 1, 0);
    for (std::size_t r = 0; r < system->c.rows; ++r) {
      for (std::size_t c = 0; c < system->c.columns; ++c) {
        const double entry = dense[r * system->c.columns + c];
        if (entry != 0.0) {
          system->c.values.push_back(entry);
          system->c.column_indices.push_back(c);
        }
      }
      system->c.row_offsets[r + 1] = system->c.values.size();
    }
  };
  auto changed = original;
  changed.capacitor_initial_constraints.clear();
  rejected(changed);
  changed = original;
  changed.capacitor_initial_constraints.erase(
      changed.capacitor_initial_constraints.begin() + 1);
  rejected(changed);
  changed = original;
  changed.capacitor_initial_constraints.push_back(
      changed.capacitor_initial_constraints.front());
  rejected(changed);
  changed = original;
  changed.capacitor_initial_constraints[1].positive_node_index =
      ReviewNode(original, "input");
  rejected(changed);
  for (const double capacitance : {0.0, -1e-6, 2e-6}) {
    changed = original;
    changed.capacitor_initial_constraints.front().capacitance_farads =
        capacitance;
    rejected(changed);
  }
  changed = original;
  changed.capacitor_initial_constraints.front().capacitance_farads =
      std::numeric_limits<double>::infinity();
  rejected(changed, ErrorCode::kNonFinite);
  changed = original;
  changed.inductor_initial_constraints.clear();
  rejected(changed);
  changed = original;
  changed.inductor_initial_constraints.pop_back();
  rejected(changed);
  changed = original;
  changed.inductor_initial_constraints.push_back(
      changed.inductor_initial_constraints.front());
  rejected(changed);
  changed = original;
  changed.inductor_initial_constraints.front().name = "Lwrong";
  rejected(changed);
  changed = original;
  changed.inductor_initial_constraints.front().branch_index =
      ReviewBranch(original, "Vdrive");
  rejected(changed);
  const auto inductor = ReviewBranch(original, "Lleft");
  for (const double diagonal : {0.0, 1e-3}) {
    changed = original;
    set_c(&changed, inductor, inductor, diagonal);
    rejected(changed);
  }
  changed = original;
  set_c(&changed, ReviewNode(original, "left"), inductor, 1e-6);
  rejected(changed);
  changed = original;
  set_c(&changed, inductor, ReviewBranch(original, "Vdrive"), 1e-6);
  rejected(changed);
  changed = original;
  set_c(&changed, ReviewNode(original, "input"), ReviewNode(original, "input"),
        1e-6);
  rejected(changed);
  changed = original;
  set_c(&changed, ReviewNode(original, "left"), ReviewNode(original, "left"),
        0.0);
  rejected(changed);
  changed = original;
  set_c(&changed, inductor, inductor, std::numeric_limits<double>::quiet_NaN());
  rejected(changed, ErrorCode::kNonFinite);
}

} // namespace
} // namespace ohmnivore
