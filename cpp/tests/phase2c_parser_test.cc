#include "cpp/tests/google_test.h"

#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "ohmnivore/ir.h"
#include "ohmnivore/parser.h"
#include "ohmnivore/status.h"
#include "ohmnivore/waveform.h"

namespace ohmnivore {
namespace {

TEST(Phase2CParserTest, ParsesWaveformDefaultsAndCombinedSourceForms) {
  constexpr char netlist[] = R"(Vpulse pulse 0 PULSE(0, 5)
Isin sin 0 DC 1 AC 2 30 SIN(1 2 3)
Vpwl pwl 0 AC 4 PWL(0,0 1u,5)
Iexp exp 0 7 EXP(0 5)
.TRAN 1u 10u
.END
)";

  auto parsed = ParseNetlist(netlist);
  ASSERT_TRUE(parsed.ok()) << parsed.error().message;
  ASSERT_EQ(parsed.value().components.size(), 4U);

  const auto *pulse = std::get_if<VoltageSource>(&parsed.value().components[0]);
  ASSERT_NE(pulse, nullptr);
  EXPECT_FALSE(pulse->dc_volts.has_value());
  EXPECT_FALSE(pulse->ac.has_value());
  ASSERT_TRUE(pulse->transient.has_value());
  const auto *pulse_waveform = std::get_if<PulseWaveform>(&*pulse->transient);
  ASSERT_NE(pulse_waveform, nullptr);
  EXPECT_DOUBLE_EQ(pulse_waveform->initial_value, 0.0);
  EXPECT_DOUBLE_EQ(pulse_waveform->pulsed_value, 5.0);
  EXPECT_DOUBLE_EQ(pulse_waveform->delay_seconds, 0.0);
  EXPECT_DOUBLE_EQ(pulse_waveform->rise_time_seconds, 0.0);
  EXPECT_DOUBLE_EQ(pulse_waveform->fall_time_seconds, 0.0);
  EXPECT_DOUBLE_EQ(pulse_waveform->pulse_width_seconds,
                   std::numeric_limits<double>::max());
  EXPECT_DOUBLE_EQ(pulse_waveform->period_seconds,
                   std::numeric_limits<double>::max());
  auto default_pulse_value = EvaluateTransientWaveform(*pulse->transient, 1.0);
  ASSERT_TRUE(default_pulse_value.ok()) << default_pulse_value.error().message;
  EXPECT_DOUBLE_EQ(default_pulse_value.value(), 5.0);

  const auto *sin = std::get_if<CurrentSource>(&parsed.value().components[1]);
  ASSERT_NE(sin, nullptr);
  ASSERT_TRUE(sin->dc_amperes.has_value());
  EXPECT_DOUBLE_EQ(*sin->dc_amperes, 1.0);
  ASSERT_TRUE(sin->ac.has_value());
  EXPECT_DOUBLE_EQ(sin->ac->magnitude, 2.0);
  EXPECT_DOUBLE_EQ(sin->ac->phase_degrees, 30.0);
  ASSERT_TRUE(sin->transient.has_value());
  const auto *sin_waveform = std::get_if<SinWaveform>(&*sin->transient);
  ASSERT_NE(sin_waveform, nullptr);
  EXPECT_DOUBLE_EQ(sin_waveform->offset, 1.0);
  EXPECT_DOUBLE_EQ(sin_waveform->amplitude, 2.0);
  EXPECT_DOUBLE_EQ(sin_waveform->frequency_hz, 3.0);
  EXPECT_DOUBLE_EQ(sin_waveform->delay_seconds, 0.0);
  EXPECT_DOUBLE_EQ(sin_waveform->damping_factor_per_second, 0.0);

  const auto *pwl = std::get_if<VoltageSource>(&parsed.value().components[2]);
  ASSERT_NE(pwl, nullptr);
  EXPECT_FALSE(pwl->dc_volts.has_value());
  ASSERT_TRUE(pwl->ac.has_value());
  ASSERT_TRUE(pwl->transient.has_value());
  const auto *pwl_waveform = std::get_if<PwlWaveform>(&*pwl->transient);
  ASSERT_NE(pwl_waveform, nullptr);
  EXPECT_EQ(pwl_waveform->time_value_pairs,
            (std::vector<std::pair<double, double>>{{0.0, 0.0}, {1e-6, 5.0}}));

  const auto *exp = std::get_if<CurrentSource>(&parsed.value().components[3]);
  ASSERT_NE(exp, nullptr);
  ASSERT_TRUE(exp->dc_amperes.has_value());
  EXPECT_DOUBLE_EQ(*exp->dc_amperes, 7.0);
  ASSERT_TRUE(exp->transient.has_value());
  const auto *exp_waveform = std::get_if<ExpWaveform>(&*exp->transient);
  ASSERT_NE(exp_waveform, nullptr);
  EXPECT_DOUBLE_EQ(exp_waveform->rise_delay_seconds, 0.0);
  EXPECT_DOUBLE_EQ(exp_waveform->rise_time_constant_seconds,
                   std::numeric_limits<double>::max());
  EXPECT_DOUBLE_EQ(exp_waveform->fall_delay_seconds,
                   std::numeric_limits<double>::max());
  EXPECT_DOUBLE_EQ(exp_waveform->fall_time_constant_seconds,
                   std::numeric_limits<double>::max());
  auto default_exp_value = EvaluateTransientWaveform(*exp->transient, 1.0);
  ASSERT_TRUE(default_exp_value.ok()) << default_exp_value.error().message;
  EXPECT_DOUBLE_EQ(default_exp_value.value(), 0.0);
}

TEST(Phase2CParserTest, ParsesAllExplicitWaveformParameters) {
  constexpr char netlist[] = R"(V1 a 0 PULSE(-1 2 3n 4n 5n 6n 7n)
V2 b 0 SIN(-2 3 4k 5u 6)
V3 c 0 PWL(0 -1, 2u 3, 4u -5)
V4 d 0 EXP(-1 2 3n 4n 5n 6n)
.TRAN 1n 10n UIC
)";
  auto parsed = ParseNetlist(netlist);
  ASSERT_TRUE(parsed.ok()) << parsed.error().message;

  const auto &pulse_source =
      std::get<VoltageSource>(parsed.value().components[0]);
  const auto &pulse = std::get<PulseWaveform>(*pulse_source.transient);
  EXPECT_DOUBLE_EQ(pulse.initial_value, -1.0);
  EXPECT_DOUBLE_EQ(pulse.pulsed_value, 2.0);
  EXPECT_DOUBLE_EQ(pulse.delay_seconds, 3e-9);
  EXPECT_DOUBLE_EQ(pulse.rise_time_seconds, 4e-9);
  EXPECT_DOUBLE_EQ(pulse.fall_time_seconds, 5e-9);
  EXPECT_DOUBLE_EQ(pulse.pulse_width_seconds, 6e-9);
  EXPECT_DOUBLE_EQ(pulse.period_seconds, 7e-9);

  const auto &sin_source =
      std::get<VoltageSource>(parsed.value().components[1]);
  const auto &sin = std::get<SinWaveform>(*sin_source.transient);
  EXPECT_DOUBLE_EQ(sin.frequency_hz, 4000.0);
  EXPECT_DOUBLE_EQ(sin.delay_seconds, 5e-6);
  EXPECT_DOUBLE_EQ(sin.damping_factor_per_second, 6.0);

  const auto &exp_source =
      std::get<VoltageSource>(parsed.value().components[3]);
  const auto &exp = std::get<ExpWaveform>(*exp_source.transient);
  EXPECT_DOUBLE_EQ(exp.rise_delay_seconds, 3e-9);
  EXPECT_DOUBLE_EQ(exp.rise_time_constant_seconds, 4e-9);
  EXPECT_DOUBLE_EQ(exp.fall_delay_seconds, 5e-9);
  EXPECT_DOUBLE_EQ(exp.fall_time_constant_seconds, 6e-9);
}

TEST(Phase2CParserTest, RejectsMalformedWaveformsAndTrailingSourceText) {
  const std::vector<std::string> specifications = {
      "",
      "PULSE(0)",
      "PULSE(0 1 0 0 0 1 0)",
      "PULSE(0 1 -1)",
      "PULSE (0 1)",
      "PULSE(0,,1)",
      "PULSE(0 1,)",
      "PULSE(0 1 0 0 0 1 2 3)",
      "PULSEX(0 1)",
      "SIN(0 1)",
      "SIN(0 1 -1)",
      "SIN(0 1 1 0 0 6)",
      "PWL()",
      "PWL(0)",
      "PWL(-1 0)",
      "PWL(0 0 0 1)",
      "PWL(1 0 0 1)",
      "EXP(0)",
      "EXP(0 1 0 0)",
      "EXP(0 1 2 1 1 1)",
      "EXP(0 1 0 1 2 1 3)",
      "PULSE(0 1) trailing",
      "PULSE(0 1) SIN(0 1 1)",
      "PULSE(0 1) AC 1",
      "DC 1 PULSE(0 1) AC 1",
      "DC 1e309",
      "PULSE(0 1e309)",
      "SIN(0 1 1e309)",
      "PWL(0 1e309)",
      "EXP(0 1 0 1e309)",
  };
  for (const std::string &specification : specifications) {
    SCOPED_TRACE(specification);
    auto parsed = ParseNetlist("V1 n 0 " + specification + "\n.OP\n");
    ASSERT_FALSE(parsed.ok());
    EXPECT_EQ(parsed.error().code, ErrorCode::kParse);
    EXPECT_NE(parsed.error().message.find("line 1"), std::string::npos);
  }
}

TEST(Phase2CParserTest, ParsesAndValidatesTranAnalysisStrictly) {
  auto basic = ParseNetlist(".TRAN 20 1\n");
  ASSERT_TRUE(basic.ok()) << basic.error().message;
  const auto *basic_analysis =
      std::get_if<TranAnalysis>(&basic.value().analyses.front());
  ASSERT_NE(basic_analysis, nullptr);
  EXPECT_DOUBLE_EQ(basic_analysis->time_step_seconds, 20.0);
  EXPECT_DOUBLE_EQ(basic_analysis->stop_time_seconds, 1.0);
  EXPECT_DOUBLE_EQ(basic_analysis->start_time_seconds, 0.0);
  EXPECT_FALSE(basic_analysis->use_initial_conditions);

  auto uic = ParseNetlist(".tran 1n 10n UIC\n");
  ASSERT_TRUE(uic.ok()) << uic.error().message;
  EXPECT_TRUE(std::get<TranAnalysis>(uic.value().analyses.front())
                  .use_initial_conditions);

  auto started = ParseNetlist(".TRAN 1u 10u 2u uic\n");
  ASSERT_TRUE(started.ok()) << started.error().message;
  const auto &started_analysis =
      std::get<TranAnalysis>(started.value().analyses.front());
  EXPECT_DOUBLE_EQ(started_analysis.start_time_seconds, 2e-6);
  EXPECT_TRUE(started_analysis.use_initial_conditions);

  const std::vector<std::string> invalid = {
      ".TRAN",
      ".TRAN 1",
      ".TRAN 0 1",
      ".TRAN -1 1",
      ".TRAN 1 0",
      ".TRAN 1 2 -1",
      ".TRAN 1 2 3",
      ".TRAN 1 2 nope",
      ".TRAN 1 2 0 nope",
      ".TRAN 1 2 UIC extra",
      ".TRAN 1e309 2",
      ".TRAN 1 2 0 UIC extra",
  };
  for (const std::string &directive : invalid) {
    SCOPED_TRACE(directive);
    auto parsed = ParseNetlist(directive + "\n");
    ASSERT_FALSE(parsed.ok());
    EXPECT_EQ(parsed.error().code, ErrorCode::kParse);
    EXPECT_NE(parsed.error().message.find("line 1"), std::string::npos);
  }
}

TEST(Phase2CWaveformTest, EvaluatesLegacyPulseSinPwlAndExpSemantics) {
  const TransientWaveform pulse = PulseWaveform{
      .initial_value = 0.0,
      .pulsed_value = 5.0,
      .delay_seconds = 1.0,
      .rise_time_seconds = 1.0,
      .fall_time_seconds = 1.0,
      .pulse_width_seconds = 2.0,
      .period_seconds = 6.0,
  };
  auto before = EvaluateTransientWaveform(pulse, 0.5);
  auto negative = EvaluateTransientWaveform(pulse, -1.0);
  auto rising = EvaluateTransientWaveform(pulse, 1.5);
  auto high = EvaluateTransientWaveform(pulse, 3.0);
  auto falling = EvaluateTransientWaveform(pulse, 4.5);
  auto repeated = EvaluateTransientWaveform(pulse, 7.5);
  ASSERT_TRUE(before.ok());
  ASSERT_TRUE(negative.ok());
  ASSERT_TRUE(rising.ok());
  ASSERT_TRUE(high.ok());
  ASSERT_TRUE(falling.ok());
  ASSERT_TRUE(repeated.ok());
  EXPECT_DOUBLE_EQ(before.value(), 0.0);
  EXPECT_DOUBLE_EQ(negative.value(), 0.0);
  EXPECT_DOUBLE_EQ(rising.value(), 2.5);
  EXPECT_DOUBLE_EQ(high.value(), 5.0);
  EXPECT_DOUBLE_EQ(falling.value(), 2.5);
  EXPECT_DOUBLE_EQ(repeated.value(), 2.5);

  const TransientWaveform sin = SinWaveform{
      .offset = 2.0,
      .amplitude = 1.0,
      .frequency_hz = 1.0,
      .delay_seconds = 1.0,
      .damping_factor_per_second = 1.0,
  };
  auto sin_before = EvaluateTransientWaveform(sin, 0.5);
  auto sin_quarter = EvaluateTransientWaveform(sin, 1.25);
  ASSERT_TRUE(sin_before.ok());
  ASSERT_TRUE(sin_quarter.ok());
  EXPECT_DOUBLE_EQ(sin_before.value(), 2.0);
  EXPECT_NEAR(sin_quarter.value(), 2.0 + std::exp(-0.25), 1e-14);

  const TransientWaveform pwl = PwlWaveform{
      .time_value_pairs = {{1.0, 0.0}, {2.0, 10.0}, {3.0, 5.0}},
  };
  auto pwl_before = EvaluateTransientWaveform(pwl, 0.0);
  auto pwl_middle = EvaluateTransientWaveform(pwl, 1.5);
  auto pwl_after = EvaluateTransientWaveform(pwl, 4.0);
  ASSERT_TRUE(pwl_before.ok());
  ASSERT_TRUE(pwl_middle.ok());
  ASSERT_TRUE(pwl_after.ok());
  EXPECT_DOUBLE_EQ(pwl_before.value(), 0.0);
  EXPECT_DOUBLE_EQ(pwl_middle.value(), 5.0);
  EXPECT_DOUBLE_EQ(pwl_after.value(), 5.0);

  const TransientWaveform exp = ExpWaveform{
      .initial_value = 0.0,
      .pulsed_value = 5.0,
      .rise_delay_seconds = 1.0,
      .rise_time_constant_seconds = 1.0,
      .fall_delay_seconds = 3.0,
      .fall_time_constant_seconds = 1.0,
  };
  auto exp_before = EvaluateTransientWaveform(exp, 0.5);
  auto exp_rise = EvaluateTransientWaveform(exp, 2.0);
  auto exp_fall = EvaluateTransientWaveform(exp, 4.0);
  ASSERT_TRUE(exp_before.ok());
  ASSERT_TRUE(exp_rise.ok());
  ASSERT_TRUE(exp_fall.ok());
  EXPECT_DOUBLE_EQ(exp_before.value(), 0.0);
  EXPECT_NEAR(exp_rise.value(), 5.0 * (1.0 - std::exp(-1.0)), 1e-14);
  EXPECT_NEAR(exp_fall.value(),
              5.0 * (1.0 - std::exp(-3.0)) - 5.0 * (1.0 - std::exp(-1.0)),
              1e-14);
}

TEST(Phase2CWaveformTest, RejectsInvalidDirectIrAndRuntimeValues) {
  const TransientWaveform invalid_pulse = PulseWaveform{
      .initial_value = 0.0,
      .pulsed_value = 1.0,
      .delay_seconds = -1.0,
  };
  auto compile_error = EvaluateTransientWaveform(invalid_pulse, 0.0);
  ASSERT_FALSE(compile_error.ok());
  EXPECT_EQ(compile_error.error().code, ErrorCode::kCompile);

  const TransientWaveform unordered_pwl = PwlWaveform{
      .time_value_pairs = {{1.0, 0.0}, {1.0, 1.0}},
  };
  auto pwl_error = EvaluateTransientWaveform(unordered_pwl, 0.0);
  ASSERT_FALSE(pwl_error.ok());
  EXPECT_EQ(pwl_error.error().code, ErrorCode::kCompile);

  const TransientWaveform valid = SinWaveform{
      .offset = 0.0,
      .amplitude = 1.0,
      .frequency_hz = 1.0,
  };
  auto nonfinite_time =
      EvaluateTransientWaveform(valid, std::numeric_limits<double>::infinity());
  ASSERT_FALSE(nonfinite_time.ok());
  EXPECT_EQ(nonfinite_time.error().code, ErrorCode::kSolve);

  const TransientWaveform overflow = SinWaveform{
      .offset = 0.0,
      .amplitude = 1.0,
      .frequency_hz = 1e308,
  };
  auto nonfinite_result = EvaluateTransientWaveform(overflow, 1e308);
  ASSERT_FALSE(nonfinite_result.ok());
  EXPECT_EQ(nonfinite_result.error().code, ErrorCode::kSolve);
}

TEST(Phase2CWaveformTest, CollectsSortedUniqueBreakpointsThroughStopTime) {
  const TransientWaveform pulse = PulseWaveform{
      .initial_value = 0.0,
      .pulsed_value = 1.0,
      .delay_seconds = 1.0,
      .rise_time_seconds = 0.1,
      .fall_time_seconds = 0.1,
      .pulse_width_seconds = 0.2,
      .period_seconds = 1.0,
  };
  auto pulse_breakpoints = CollectTransientWaveformBreakpoints(pulse, 2.2);
  ASSERT_TRUE(pulse_breakpoints.ok()) << pulse_breakpoints.error().message;
  const std::vector<double> expected = {1.0, 1.1, 1.3, 1.4, 2.0, 2.1};
  ASSERT_EQ(pulse_breakpoints.value().size(), expected.size());
  for (std::size_t index = 0; index < expected.size(); ++index) {
    EXPECT_NEAR(pulse_breakpoints.value()[index], expected[index], 1e-15);
  }

  const TransientWaveform pwl = PwlWaveform{
      .time_value_pairs = {{0.0, 0.0}, {1.0, 2.0}, {3.0, 4.0}},
  };
  auto pwl_breakpoints = CollectTransientWaveformBreakpoints(pwl, 1.0);
  ASSERT_TRUE(pwl_breakpoints.ok()) << pwl_breakpoints.error().message;
  EXPECT_EQ(pwl_breakpoints.value(), (std::vector<double>{0.0, 1.0}));
}

TEST(Phase2CWaveformTest, EvaluatesDecimalPeriodicPulseEdgesConsistently) {
  const TransientWaveform pulse = PulseWaveform{
      .initial_value = 0.0,
      .pulsed_value = 1.0,
      .delay_seconds = 0.1,
      .rise_time_seconds = 0.0,
      .fall_time_seconds = 0.0,
      .pulse_width_seconds = 0.005,
      .period_seconds = 0.01,
  };
  Result<std::vector<double>> breakpoints =
      CollectTransientWaveformBreakpoints(pulse, 0.12);
  ASSERT_TRUE(breakpoints.ok()) << breakpoints.error().message;
  const auto at = [&](double time) {
    Result<double> value = EvaluateTransientWaveform(pulse, time);
    EXPECT_TRUE(value.ok()) << value.error().message;
    return value.ok() ? value.value() : -1.0;
  };
  EXPECT_DOUBLE_EQ(at(0.1), 1.0);
  EXPECT_DOUBLE_EQ(at(0.105), 0.0);
  EXPECT_DOUBLE_EQ(at(0.11), 1.0);
  EXPECT_DOUBLE_EQ(at(0.115), 0.0);
  EXPECT_DOUBLE_EQ(at(0.12), 1.0);
  for (double breakpoint : breakpoints.value()) {
    if (breakpoint >= 0.1) {
      const double phase = std::fmod(breakpoint - 0.1, 0.01);
      const bool cycle_start =
          std::abs(phase) < 1e-12 || std::abs(phase - 0.01) < 1e-12;
      EXPECT_DOUBLE_EQ(at(breakpoint), cycle_start ? 1.0 : 0.0);
    }
  }
}

} // namespace
} // namespace ohmnivore
