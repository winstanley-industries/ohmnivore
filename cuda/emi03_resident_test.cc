#include <gtest/gtest.h>

#include <algorithm>
#include <barrier>
#include <cmath>
#include <future>
#include <string>
#include <vector>

#include "cpp/tests/transient_accuracy_cases.h"
#include "cuda/emi03_real_solver.h"
#include "cuda/emi03_resident.h"
#include "ohmnivore/behavioral.h"

namespace ohmnivore {
namespace {
class Emi03Resident : public ::testing::Test {
protected:
  void SetUp() override {
    auto begun = BeginEmi03CudaJob("resident-test");
    ASSERT_TRUE(begun.ok()) << begun.error().message;
  }
  void TearDown() override {
    auto ended = EndEmi03CudaJob();
    ASSERT_TRUE(ended.ok()) << ended.error().message;
    EXPECT_EQ(ended.value().outstanding_device_bytes, 0U);
    EXPECT_EQ(ended.value().cleanup_failures, 0U);
  }
};

TEST_F(Emi03Resident, FineResolutionMatchesIndependentPhysicalOracles) {
  const auto cases = test::TransientAccuracyCases();
  ASSERT_EQ(cases.size(), 21U);
  for (const auto &fixture : cases) {
    SCOPED_TRACE(fixture.name);
    auto closed = EndEmi03CudaJob();
    ASSERT_TRUE(closed.ok()) << closed.error().message;
    EXPECT_EQ(closed.value().outstanding_device_bytes, 0U);
    auto opened = BeginEmi03CudaJob(fixture.name);
    ASSERT_TRUE(opened.ok()) << opened.error().message;
    const auto parsed = ParseBehavioralNetlist(fixture.deck);
    ASSERT_TRUE(parsed.ok()) << parsed.error().message;
    const auto compiled = CompileBehavioralMna(parsed.value());
    ASSERT_TRUE(compiled.ok()) << compiled.error().message;
    const auto &system = compiled.value();
    const auto node =
        std::find(system.node_names.begin(), system.node_names.end(), "out");
    ASSERT_NE(node, system.node_names.end());
    const auto out = static_cast<std::size_t>(node - system.node_names.begin());
    auto analysis = fixture.analysis;
    analysis.time_step_seconds /= 32;
    TransientExecutionLimits limits;
    limits.retain_output_states = false;
    limits.behavioral_error_estimator =
        BehavioralErrorEstimator::kDerivativeHistory;
    limits.minimum_step_divisor = 1e6;
    limits.nonlinear_maximum_iterations = 100;
    double voltage_error = 0, current_error = 0, previous = -1;
    std::size_t points = 0;
    limits.accepted_state_observer = [&](double time,
                                         const std::vector<double> &state) {
      EXPECT_GT(time, previous);
      if (previous >= 0) {
        EXPECT_LE(time - previous, analysis.time_step_seconds * 1.000000000001);
      }
      previous = time;
      ++points;
      const auto exact = fixture.exact(time);
      voltage_error = std::max(voltage_error,
                               std::abs(state[out] - fixture.bias - exact[0]));
      if (fixture.has_inductor)
        current_error = std::max(
            current_error,
            std::abs(
                state[system.inductor_initial_constraints[0].branch_index] -
                exact[1]));
      return Result<bool>::Ok(true);
    };
    const auto solved = RunEmi03ResidentTransient(system, analysis, limits);
    ASSERT_TRUE(solved.ok()) << solved.error().message;
    EXPECT_EQ(solved.value().emitted_points, points);
    EXPECT_EQ(solved.value().attempts - solved.value().rejected, points - 1);
    EXPECT_EQ(previous, analysis.stop_time_seconds);
    EXPECT_LE(voltage_error, fixture.voltage_limit);
    if (fixture.has_inductor) {
      EXPECT_LE(current_error, fixture.current_limit);
    }
  }
}

TEST_F(Emi03Resident, WideOutputRowsCrossChunkBoundariesWithoutCorruption) {
  // Distinct DC values and reactive coordinates extend across every writer
  // warp. More than two chunks exercise publication of the last row in each
  // buffer.
  std::string deck = "Vdrive drive 0 1\nBzero drive 0 I={0}\n";
  for (int i = 0; i < 192; ++i) {
    const auto suffix = std::to_string(i);
    deck += "Rtop" + suffix + " drive out" + suffix + " " +
            std::to_string(1000 + 10 * i) + "\nRbottom" + suffix + " out" +
            suffix + " 0 1000\nCstore" + suffix + " out" + suffix + " 0 1n\n";
  }
  const auto parsed = ParseBehavioralNetlist(deck);
  ASSERT_TRUE(parsed.ok()) << parsed.error().message;
  const auto compiled = CompileBehavioralMna(parsed.value());
  ASSERT_TRUE(compiled.ok()) << compiled.error().message;
  const auto &system = compiled.value();
  std::vector<std::pair<std::size_t, double>> expected;
  for (int i = 0; i < 192; ++i) {
    const auto node =
        std::find(system.node_names.begin(), system.node_names.end(),
                  "out" + std::to_string(i));
    ASSERT_NE(node, system.node_names.end());
    const double resistance = 1000 + 10 * i;
    expected.emplace_back(node - system.node_names.begin(),
                          1 / (1 + resistance * (.001 + 1e-12)));
  }
  TransientExecutionLimits limits;
  limits.retain_output_states = false;
  limits.behavioral_error_estimator =
      BehavioralErrorEstimator::kDerivativeHistory;
  std::size_t count = 0;
  double last = -1, error = 0;
  limits.accepted_state_observer = [&](double time,
                                       const std::vector<double> &state) {
    EXPECT_GT(time, last);
    last = time;
    ++count;
    for (const auto &[index, value] : expected)
      error = std::max(error, std::abs(state[index] - value));
    return Result<bool>::Ok(true);
  };
  const auto result =
      RunEmi03ResidentTransient(system, {1e-9, 7e-7, 0, false}, limits);
  ASSERT_TRUE(result.ok()) << result.error().message;
  EXPECT_GT(count, 700U);
  EXPECT_EQ(result.value().emitted_points, count);
  EXPECT_EQ(last, 7e-7);
  EXPECT_LT(error, 1e-8);
}

TEST_F(Emi03Resident, ObserverFailureAndAttemptBudgetCannotPublishSuccess) {
  const auto parsed =
      ParseBehavioralNetlist("Vdrive in 0 PWL(0 0 1u 1)\nRseries in out "
                             "1\nCstore out 0 1u\nBzero out 0 I={0}\n");
  ASSERT_TRUE(parsed.ok());
  const auto compiled = CompileBehavioralMna(parsed.value());
  ASSERT_TRUE(compiled.ok());
  TransientExecutionLimits limits;
  limits.retain_output_states = false;
  limits.behavioral_error_estimator =
      BehavioralErrorEstimator::kDerivativeHistory;
  limits.accepted_state_observer = [](double, const std::vector<double> &) {
    return Result<bool>::Ok(false);
  };
  auto rejected = RunEmi03ResidentTransient(compiled.value(),
                                            {1e-8, 1e-6, 0, false}, limits);
  ASSERT_FALSE(rejected.ok());
  EXPECT_EQ(rejected.error().code, ErrorCode::kIo);
  limits.accepted_state_observer = [](double, const std::vector<double> &) {
    return Result<bool>::Ok(true);
  };
  limits.maximum_accepted_steps = 1;
  limits.maximum_step_attempts = 1;
  auto exhausted = RunEmi03ResidentTransient(compiled.value(),
                                             {1e-8, 1e-6, 0, false}, limits);
  ASSERT_FALSE(exhausted.ok());
  EXPECT_EQ(exhausted.error().code, ErrorCode::kUnsupportedSize);
}
TEST_F(Emi03Resident, DynamicGpuPivotsDoNotHideBehindZeroResponse) {
  // Cancel the compiler's declared 1e-12 node shunts so the changing
  // matrix has exact zero pivots, while its determinant remains nonzero.
  const auto parsed = ParseBehavioralNetlist(
      "Vdrive drive 0 DC 1 PWL(0 1 1u 0 2u 0)\n"
      "Bx x 0 I={if(v(drive)>0.5,v(x),v(y))-1e-12*v(x)}\n"
      "By y 0 I={if(v(drive)>0.5,-v(y),v(x))-1e-12*v(y)}\n");
  ASSERT_TRUE(parsed.ok()) << parsed.error().message;
  const auto compiled = CompileBehavioralMna(parsed.value());
  ASSERT_TRUE(compiled.ok()) << compiled.error().message;
  const auto &system = compiled.value();
  const auto index = [&](const std::string &name) {
    return static_cast<std::size_t>(
        std::find(system.node_names.begin(), system.node_names.end(), name) -
        system.node_names.begin());
  };
  const auto x = index("x"), y = index("y"), drive = index("drive");
  ASSERT_LT(x, system.g.rows);
  ASSERT_LT(y, system.g.rows);
  TransientExecutionLimits limits;
  limits.retain_output_states = false;
  limits.behavioral_error_estimator =
      BehavioralErrorEstimator::kDerivativeHistory;
  limits.nonlinear_maximum_iterations = 100;
  double last = -1, maximum = 0;
  limits.accepted_state_observer = [&](double time,
                                       const std::vector<double> &state) {
    EXPECT_GT(time, last);
    last = time;
    EXPECT_NEAR(state[drive], std::max(0.0, 1 - time / 1e-6), 1e-10);
    maximum = std::max({maximum, std::abs(state[x]), std::abs(state[y])});
    return Result<bool>::Ok(true);
  };
  const auto result =
      RunEmi03ResidentTransient(system, {1.25e-7, 2e-6, 0, false}, limits);
  ASSERT_TRUE(result.ok()) << result.error().message;
  EXPECT_EQ(last, 2e-6);
  EXPECT_LT(maximum, 1e-7);
  EXPECT_GT(result.value().solver_statistics.numeric_refactorization_fallbacks,
            0U);
  EXPECT_GT(result.value().solver_statistics.numeric_reuses, 0U);
}

TEST_F(Emi03Resident, ZeroResponseCannotHideASingularAcceptedJacobian) {
  const auto parsed = ParseBehavioralNetlist(
      "Vdrive drive 0 DC 1 PWL(0 1 1u 0 2u 0)\n"
      "Bx x 0 I={v(x)-1e-12*v(x)}\n"
      "By y 0 I={if(v(drive)>0.5,-v(y),v(x))-1e-12*v(y)}\n");
  ASSERT_TRUE(parsed.ok()) << parsed.error().message;
  const auto compiled = CompileBehavioralMna(parsed.value());
  ASSERT_TRUE(compiled.ok()) << compiled.error().message;
  TransientExecutionLimits limits;
  limits.retain_output_states = false;
  limits.behavioral_error_estimator =
      BehavioralErrorEstimator::kDerivativeHistory;
  limits.nonlinear_maximum_iterations = 100;
  double last = -1;
  limits.accepted_state_observer = [&](double time,
                                       const std::vector<double> &) {
    last = time;
    return Result<bool>::Ok(true);
  };
  const auto result = RunEmi03ResidentTransient(
      compiled.value(), {1.25e-7, 2e-6, 0, false}, limits);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.error().code, ErrorCode::kSingular);
  EXPECT_LT(last, 2e-6);
}

TEST_F(Emi03Resident, NestedLazyBranchesPreserveDomains) {
  for (bool invalid : {false, true}) {
    const std::string expression =
        invalid ? "if(v(in)>0,0**-1,0)"
                : "-if(v(in)>2,0**-1,if(v(in)>0,2*v(in),3*v(in)))";
    const auto parsed = ParseBehavioralNetlist(
        "Vdrive in 0 DC -1 PWL(0 -1 1u 1)\nRload out 0 1\nBload out 0 I={" +
        expression + "}\n");
    ASSERT_TRUE(parsed.ok()) << parsed.error().message;
    const auto compiled = CompileBehavioralMna(parsed.value());
    ASSERT_TRUE(compiled.ok()) << compiled.error().message;
    const auto &system = compiled.value();
    const auto out = static_cast<std::size_t>(
        std::find(system.node_names.begin(), system.node_names.end(), "out") -
        system.node_names.begin());
    ASSERT_LT(out, system.g.rows);
    TransientExecutionLimits limits;
    limits.retain_output_states = false;
    limits.behavioral_error_estimator =
        BehavioralErrorEstimator::kDerivativeHistory;
    limits.nonlinear_maximum_iterations = 100;
    double last = -1, maximum = 0;
    limits.accepted_state_observer = [&](double time,
                                         const std::vector<double> &state) {
      EXPECT_GT(time, last);
      last = time;
      const double input = -1 + 2 * time / 1e-6,
                   exact = (input > 0 ? 2 : 3) * input;
      maximum = std::max(maximum, std::abs(state[out] - exact));
      return Result<bool>::Ok(true);
    };
    const auto result =
        RunEmi03ResidentTransient(system, {1e-8, 1e-6, 0, false}, limits);
    if (invalid) {
      EXPECT_FALSE(result.ok());
      EXPECT_LT(last, 1e-6);
    } else {
      ASSERT_TRUE(result.ok()) << result.error().message;
      EXPECT_EQ(last, 1e-6);
      EXPECT_LT(maximum, 1e-7);
    }
  }
}

TEST_F(Emi03Resident,
       CompactExpressionIndicesCrossHighBitsAndPreserveLazyBranches) {
  // Ten large balanced trees cross the 4096-node boundary without exceeding
  // the per-program node/depth limits. Repeated state leaves also exercise
  // ordered derivative accumulation after decoding compact metadata.
  std::vector<std::string> terms(250, "v(drive)");
  while (terms.size() > 1) {
    std::vector<std::string> next;
    for (std::size_t i = 0; i < terms.size(); i += 2)
      next.push_back(i + 1 < terms.size()
                         ? "(" + terms[i] + "+" + terms[i + 1] + ")"
                         : terms[i]);
    terms = std::move(next);
  }
  std::string deck = "Vdrive drive 0 DC 1 PWL(0 1 100n 2)\n";
  for (int i = 0; i < 10; ++i) {
    const auto name = std::to_string(i);
    deck += "Rload" + name + " out" + name + " 0 1\nBload" + name + " out" +
            name + " 0 I={-if(v(drive)>-1," + terms[0] + ",0**-1)}\n";
  }
  const auto parsed = ParseBehavioralNetlist(deck);
  ASSERT_TRUE(parsed.ok()) << parsed.error().message;
  const auto compiled = CompileBehavioralMna(parsed.value());
  ASSERT_TRUE(compiled.ok()) << compiled.error().message;
  const auto &system = compiled.value();
  std::vector<std::size_t> outputs;
  for (int i = 0; i < 10; ++i) {
    const auto node =
        std::find(system.node_names.begin(), system.node_names.end(),
                  "out" + std::to_string(i));
    ASSERT_NE(node, system.node_names.end());
    outputs.push_back(node - system.node_names.begin());
  }
  TransientExecutionLimits limits;
  limits.retain_output_states = false;
  limits.behavioral_error_estimator =
      BehavioralErrorEstimator::kDerivativeHistory;
  double last = -1;
  std::size_t points = 0;
  limits.accepted_state_observer = [&](double time,
                                       const std::vector<double> &state) {
    EXPECT_GT(time, last);
    last = time;
    ++points;
    const double expected = 250 * (1 + time / 1e-7) / (1 + 1e-12);
    for (auto index : outputs)
      EXPECT_NEAR(state[index], expected, 1e-7);
    return Result<bool>::Ok(true);
  };
  const auto result =
      RunEmi03ResidentTransient(system, {1e-8, 1e-7, 0, false}, limits);
  ASSERT_TRUE(result.ok()) << result.error().message;
  EXPECT_GT(points, 10U);
  EXPECT_EQ(last, 1e-7);
}

TEST_F(Emi03Resident, ConcurrentJobsKeepStateAndFailureOwnershipPrivate) {
  const auto cases = test::TransientAccuracyCases();
  constexpr int count = 16;
  std::barrier ready(count);
  std::vector<std::future<void>> jobs;
  for (int index = 0; index < count; ++index) {
    jobs.push_back(std::async(std::launch::async, [&, index] {
      const auto &fixture = cases[index];
      const auto id = "concurrent-" + std::to_string(index);
      Emi03CudaOptions options;
      if (index == 0)
        options.fault = Emi03CudaFault::kCudssAllocationFailure;
      const auto opened = BeginEmi03CudaJob(id, options);
      EXPECT_TRUE(opened.ok());
      ready.arrive_and_wait();
      if (!opened.ok())
        return;
      const auto parsed = ParseBehavioralNetlist(fixture.deck);
      EXPECT_TRUE(parsed.ok());
      if (!parsed.ok()) {
        static_cast<void>(EndEmi03CudaJob());
        return;
      }
      const auto compiled = CompileBehavioralMna(parsed.value());
      EXPECT_TRUE(compiled.ok());
      if (!compiled.ok()) {
        static_cast<void>(EndEmi03CudaJob());
        return;
      }
      const auto &system = compiled.value();
      const auto node =
          std::find(system.node_names.begin(), system.node_names.end(), "out");
      EXPECT_NE(node, system.node_names.end());
      const auto out =
          static_cast<std::size_t>(node - system.node_names.begin());
      auto analysis = fixture.analysis;
      analysis.time_step_seconds /= 32;
      TransientExecutionLimits limits;
      limits.retain_output_states = false;
      limits.behavioral_error_estimator =
          BehavioralErrorEstimator::kDerivativeHistory;
      limits.minimum_step_divisor = 1e6;
      limits.nonlinear_maximum_iterations = 100;
      double last = -1, maximum_error = 0;
      limits.accepted_state_observer = [&](double time,
                                           const std::vector<double> &state) {
        EXPECT_GT(time, last);
        last = time;
        maximum_error =
            std::max(maximum_error, std::abs(state[out] - fixture.bias -
                                             fixture.exact(time)[0]));
        return Result<bool>::Ok(true);
      };
      const auto result = RunEmi03ResidentTransient(system, analysis, limits);
      if (index == 0) {
        EXPECT_FALSE(result.ok());
      } else {
        EXPECT_TRUE(result.ok()) << result.error().message;
        EXPECT_EQ(last, analysis.stop_time_seconds);
        EXPECT_LE(maximum_error, fixture.voltage_limit);
      }
      const auto closed = EndEmi03CudaJob();
      EXPECT_TRUE(closed.ok());
      if (closed.ok()) {
        EXPECT_EQ(closed.value().job_id, id);
        EXPECT_EQ(closed.value().outstanding_device_bytes, 0U);
        EXPECT_EQ(closed.value().cleanup_failures, 0U);
        EXPECT_EQ(closed.value().allocation_failures, index == 0 ? 1U : 0U);
        if (index != 0) {
          EXPECT_GT(closed.value().successful_solves, 0U);
        }
      }
    }));
  }
  for (auto &job : jobs)
    job.get();
  EXPECT_EQ(SnapshotEmi03CudaJob().job_id, "resident-test");
  EXPECT_EQ(SnapshotEmi03CudaJob().peak_device_bytes, 0U);
}

} // namespace
} // namespace ohmnivore
