#include "cuda/emi03_cuda_internal.h"
#include "cuda/emi03_expression_device.cuh"
#include "cuda/emi03_reduction.cuh"
#include "cuda/emi03_resident.h"
#include "ohmnivore/behavioral.h"
#include "ohmnivore/nonlinear.h"
#include "ohmnivore/waveform.h"

#include <klu.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

namespace ohmnivore {
namespace {
using namespace emi03_device;
using emi03_cuda_internal::BackendError;
using emi03_cuda_internal::CheckCuda;
#include "cuda/emi03_resident_device.cuh"

class Allocations {
public:
  Allocations() {
    CheckCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
              "resident stream creation");
  }
  cudaStream_t stream() const { return stream_; }
  void Copy(void *destination, const void *source, std::size_t bytes,
            cudaMemcpyKind kind) {
    if (kind != cudaMemcpyHostToDevice && kind != cudaMemcpyDeviceToHost)
      throw BackendError(ErrorCode::kUnsupported, "resident copy direction");
    // Pageable asynchronous copies can spin in the driver before returning.
    // Keep bounded pinned staging private to each allocation owner. Completion
    // queries sleep between attempts: blocking event waits still consumed a CPU
    // core on the measured WSL driver. All waiting remains in charged wall
    // time.
    if (!staging_)
      CheckCuda(cudaHostAlloc(&staging_, kStagingBytes, cudaHostAllocDefault),
                "resident pinned transfer staging");
    if (!completed_)
      CheckCuda(
          cudaEventCreateWithFlags(&completed_, cudaEventBlockingSync |
                                                    cudaEventDisableTiming),
          "resident blocking completion event");
    for (std::size_t offset = 0; offset < bytes; offset += kStagingBytes) {
      const auto count = std::min(kStagingBytes, bytes - offset);
      auto *target = static_cast<unsigned char *>(destination) + offset;
      const auto *input = static_cast<const unsigned char *>(source) + offset;
      if (kind == cudaMemcpyHostToDevice)
        std::memcpy(staging_, input, count);
      CheckCuda(
          cudaMemcpyAsync(kind == cudaMemcpyHostToDevice ? target : staging_,
                          kind == cudaMemcpyHostToDevice ? staging_ : input,
                          count, kind, stream_),
          "resident stream copy");
      CheckCuda(cudaEventRecord(completed_, stream_),
                "resident completion event record");
      const auto start = std::chrono::steady_clock::now();
      auto status = cudaEventQuery(completed_);
      while (status == cudaErrorNotReady) {
        std::this_thread::sleep_for(std::chrono::microseconds(250));
        status = cudaEventQuery(completed_);
      }
      emi03_cuda_internal::Statistics().synchronization_ns +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - start)
              .count();
      CheckCuda(status, "resident completion query");
      if (kind == cudaMemcpyDeviceToHost)
        std::memcpy(target, staging_, count);
    }
  }
  ~Allocations() {
    if (cudaStreamSynchronize(stream_) != cudaSuccess)
      ++emi03_cuda_internal::Statistics().cleanup_failures;
    for (auto it = owned_.rbegin(); it != owned_.rend(); ++it)
      emi03_cuda_internal::FreeDevice(it->first, it->second);
    if (completed_ && cudaEventDestroy(completed_) != cudaSuccess)
      ++emi03_cuda_internal::Statistics().cleanup_failures;
    if (staging_ && cudaFreeHost(staging_) != cudaSuccess)
      ++emi03_cuda_internal::Statistics().cleanup_failures;
    if (cudaStreamDestroy(stream_) != cudaSuccess)
      ++emi03_cuda_internal::Statistics().cleanup_failures;
  }
  template <class T> T *Allocate(std::size_t count) {
    const auto bytes = std::max<std::size_t>(1, count) * sizeof(T);
    void *p = emi03_cuda_internal::AllocateDevice(bytes);
    try {
      owned_.emplace_back(p, bytes);
    } catch (...) {
      emi03_cuda_internal::FreeDevice(p, bytes);
      throw;
    }
    return static_cast<T *>(p);
  }
  template <class T> T *Upload(const std::vector<T> &values) {
    auto *result = Allocate<T>(values.size());
    if (!values.empty()) {
      Copy(result, values.data(), values.size() * sizeof(T),
           cudaMemcpyHostToDevice);
      emi03_cuda_internal::Statistics().upload_bytes +=
          values.size() * sizeof(T);
    }
    return result;
  }

private:
  static constexpr std::size_t kStagingBytes = Chunk * (N + 1) * sizeof(double);
  cudaStream_t stream_ = nullptr;
  cudaEvent_t completed_ = nullptr;
  void *staging_ = nullptr;
  std::vector<std::pair<void *, std::size_t>> owned_;
};
void PrepareFactorPlan(const MnaSystem &system, const TranAnalysis &analysis,
                       const std::vector<double> &initial, Model &model,
                       Allocations &allocation,
                       const std::vector<double> *refresh = nullptr) {
  auto companion = FormTransientCompanionMatrix(system.g, system.c,
                                                analysis.time_step_seconds, 1);
  if (!companion.ok())
    throw BackendError(companion.error().code, companion.error().message);
  auto working = system;
  working.g = companion.TakeValue();
  auto remap = RemapBehavioralDescriptors(&working);
  if (!remap.ok())
    throw BackendError(remap.error().code, remap.error().message);
  auto assembled = BuildNonlinearDcLinearization(working, initial);
  if (!assembled.ok())
    throw BackendError(assembled.error().code, assembled.error().message);
  auto matrix = assembled.value().jacobian;
  if (refresh) {
    if (refresh->size() != matrix.values.size())
      throw BackendError(ErrorCode::kInvalidStructure,
                         "resident refresh structure changed");
    matrix.values = *refresh;
  }
  auto converted = ConvertCsrToSolverCsc(matrix);
  if (!converted.ok())
    throw BackendError(converted.error().code, converted.error().message);
  auto csc = converted.TakeValue();
  klu_common common;
  klu_defaults(&common);
  common.ordering =
      1; // COLAMD: structural ordering only, never a CPU numeric solve.
  const auto release_symbolic = [&common](klu_symbolic *value) {
    if (value)
      klu_free_symbolic(&value, &common);
  };
  std::unique_ptr<klu_symbolic, decltype(release_symbolic)> symbolic(
      klu_analyze(model.n, csc.column_offsets.data(), csc.row_indices.data(),
                  &common),
      release_symbolic);
  if (!symbolic)
    throw BackendError(ErrorCode::kSingular,
                       "resident symbolic ordering failed");
  std::vector<int> q(symbolic->Q, symbolic->Q + model.n);
  std::vector<int> structural_rows(symbolic->P, symbolic->P + model.n);
  symbolic.reset();
  std::vector<int> qi(model.n), p(model.n), pi(model.n);
  for (int j = 0; j < model.n; ++j)
    qi[q[j]] = j;
  {
    Allocations temporary;
    auto w = std::make_unique<Workspace>();
    std::vector<int> dense_offsets, dense_columns;
    std::vector<double> ordered_values;
    dense_offsets.push_back(0);
    for (int row : structural_rows) {
      for (auto k = matrix.row_offsets[row]; k < matrix.row_offsets[row + 1];
           ++k) {
        dense_columns.push_back(qi[matrix.column_indices[k]]);
        ordered_values.push_back(matrix.values[k]);
      }
      dense_offsets.push_back(static_cast<int>(dense_columns.size()));
    }
    w->jacobian = temporary.Upload(ordered_values);
    w->lu = temporary.Allocate<double>(model.n * N);
    w->equil = temporary.Allocate<double>(model.n);
    w->permutation = temporary.Allocate<int>(model.n);
    auto *device = temporary.Allocate<Workspace>(1);
    auto *error = temporary.Allocate<int>(1);
    auto ordering = model;
    ordering.row_offsets = temporary.Upload(dense_offsets);
    ordering.columns = temporary.Upload(dense_columns);
    temporary.Copy(device, w.get(), sizeof(Workspace), cudaMemcpyHostToDevice);
    DiscoverPivots<<<1, Threads, 0, temporary.stream()>>>(
        ordering, device, error, refresh == nullptr);
    CheckCuda(cudaGetLastError(), "resident pivot discovery launch");
    int status = 0;
    temporary.Copy(&status, error, sizeof(int), cudaMemcpyDeviceToHost);
    if (status)
      throw BackendError(static_cast<ErrorCode>(status - 1),
                         "resident initial GPU pivot discovery failed");
    temporary.Copy(p.data(), w->permutation, model.n * sizeof(int),
                   cudaMemcpyDeviceToHost);
    for (auto &row : p)
      row = structural_rows[row];
    auto &stats = emi03_cuda_internal::Statistics();
    stats.upload_bytes += sizeof(Workspace);
    stats.readback_bytes += (model.n + 1) * sizeof(int);
    ++stats.factorizations;
  }
  for (int j = 0; j < model.n; ++j)
    pi[p[j]] = j;
  std::vector<unsigned char> pattern(model.n * model.n, 0);
  for (int row = 0; row < model.n; ++row)
    for (auto k = matrix.row_offsets[row]; k < matrix.row_offsets[row + 1]; ++k)
      pattern[pi[row] * model.n + qi[matrix.column_indices[k]]] = 1;
  for (int k = 0; k < model.n; ++k)
    for (int row = k + 1; row < model.n; ++row)
      if (pattern[row * model.n + k])
        for (int col = k + 1; col < model.n; ++col)
          pattern[row * model.n + col] |= pattern[k * model.n + col];
  std::vector<int> offsets{0}, columns, diagonal, slots(model.n * model.n, -1);
  for (int row = 0; row < model.n; ++row) {
    for (int col = 0; col < model.n; ++col)
      if (pattern[row * model.n + col]) {
        slots[row * model.n + col] = static_cast<int>(columns.size());
        columns.push_back(col);
      }
    if (slots[row * model.n + row] < 0)
      throw BackendError(ErrorCode::kSingular,
                         "resident missing symbolic pivot");
    diagonal.push_back(slots[row * model.n + row]);
    offsets.push_back(static_cast<int>(columns.size()));
  }
  std::vector<int> inputs;
  for (int row = 0; row < model.n; ++row)
    for (auto k = matrix.row_offsets[row]; k < matrix.row_offsets[row + 1]; ++k)
      inputs.push_back(slots[pi[row] * model.n + qi[matrix.column_indices[k]]]);
  std::vector<FactorEntry> entries(columns.size());
  std::vector<FactorTerm> terms;
  std::vector<int> entry_levels(columns.size(), 0);
  for (int row = 0; row < model.n; ++row)
    for (int slot = offsets[row]; slot < offsets[row + 1]; ++slot) {
      const int col = columns[slot];
      auto &entry = entries[slot];
      if (terms.size() >= (1U << 24))
        throw BackendError(ErrorCode::kUnsupportedSize,
                           "resident factor term encoding bound");
      entry.begin = terms.size();
      entry.diagonal = row > col ? col : -1;
      entry.pivot = row == col ? row : -1;
      for (int k = 0; k < std::min(row, col); ++k) {
        const int lower = slots[row * model.n + k],
                  upper = slots[k * model.n + col];
        if (lower >= 0 && upper >= 0) {
          terms.push_back(FactorTerm{static_cast<std::uint16_t>(lower),
                                     static_cast<std::uint16_t>(upper)});
          entry_levels[slot] =
              std::max(entry_levels[slot],
                       std::max(entry_levels[lower], entry_levels[upper]) + 1);
        }
      }
      if (row > col)
        entry_levels[slot] =
            std::max(entry_levels[slot], entry_levels[diagonal[col]] + 1);
      if (terms.size() - entry.begin > 255)
        throw BackendError(ErrorCode::kUnsupportedSize,
                           "resident factor count encoding bound");
      entry.count = terms.size() - entry.begin;
    }
  model.factor_levels =
      *std::max_element(entry_levels.begin(), entry_levels.end()) + 1;
  std::vector<int> factor_order, factor_starts{0};
  for (int level = 0; level < model.factor_levels; ++level) {
    for (int slot = 0; slot < static_cast<int>(entries.size()); ++slot)
      if (entry_levels[slot] == level)
        factor_order.push_back(slot);
    factor_starts.push_back(static_cast<int>(factor_order.size()));
  }
  model.factor_entries = allocation.Upload(entries);
  model.factor_term = allocation.Upload(terms);
  model.factor_order = allocation.Upload(factor_order);
  model.factor_level_offsets = allocation.Upload(factor_starts);
  model.factor_terms = static_cast<int>(terms.size());
  std::vector<int> forward(model.n, 0), backward(model.n, 0);
  for (int row = 0; row < model.n; ++row)
    for (int j = offsets[row]; j < diagonal[row]; ++j)
      forward[row] = std::max(forward[row], forward[columns[j]] + 1);
  for (int row = model.n - 1; row >= 0; --row)
    for (int j = diagonal[row] + 1; j < offsets[row + 1]; ++j)
      backward[row] = std::max(backward[row], backward[columns[j]] + 1);
  const auto levels = [&](const std::vector<int> &level, const int *&rows,
                          const int *&bounds) {
    const int count = *std::max_element(level.begin(), level.end()) + 1;
    std::vector<int> ordered, starts{0};
    for (int i = 0; i < count; ++i) {
      for (int row = 0; row < model.n; ++row)
        if (level[row] == i)
          ordered.push_back(row);
      starts.push_back(static_cast<int>(ordered.size()));
    }
    rows = allocation.Upload(ordered);
    bounds = allocation.Upload(starts);
    return count;
  };
  model.forward_levels =
      levels(forward, model.forward_rows, model.forward_offsets);
  model.backward_levels =
      levels(backward, model.backward_rows, model.backward_offsets);
  model.factor_nonzeros = static_cast<int>(columns.size());
  model.factor_rows = allocation.Upload(offsets);
  model.factor_columns = allocation.Upload(columns);
  model.factor_diagonal = allocation.Upload(diagonal);
  model.factor_input_slots = allocation.Upload(inputs);
  model.row_permutation = allocation.Upload(p);
  model.column_permutation = allocation.Upload(q);
}
Model Prepare(const MnaSystem &system, const TranAnalysis &analysis,
              const TransientExecutionLimits &limits, Allocations &allocation,
              Workspace &workspace) {
  Model m{};
  if (system.g.rows == 0 || system.g.rows > N ||
      system.node_names.size() > system.g.rows ||
      system.behavioral_descriptors.empty() ||
      system.behavioral_descriptors.size() > 128 ||
      system.capacitor_initial_constraints.size() +
              system.inductor_initial_constraints.size() >
          N ||
      !system.diode_descriptors.empty() || !system.bjt_descriptors.empty() ||
      analysis.use_initial_conditions || limits.retain_output_states ||
      !limits.accepted_state_observer ||
      limits.behavioral_error_estimator !=
          BehavioralErrorEstimator::kDerivativeHistory)
    throw BackendError(
        ErrorCode::kUnsupported,
        "resident transient requires a bounded streamed behavioral job");
  if (!(std::isfinite(analysis.time_step_seconds) &&
        analysis.time_step_seconds > 0 &&
        std::isfinite(analysis.stop_time_seconds) &&
        analysis.stop_time_seconds > 0 &&
        std::isfinite(analysis.start_time_seconds) &&
        analysis.start_time_seconds >= 0 &&
        analysis.start_time_seconds <= analysis.stop_time_seconds &&
        std::isfinite(limits.minimum_step_divisor) &&
        limits.minimum_step_divisor >= 1 && limits.maximum_accepted_steps > 0 &&
        limits.maximum_step_attempts >= limits.maximum_accepted_steps &&
        limits.nonlinear_maximum_iterations > 0 &&
        limits.nonlinear_maximum_iterations <= 100))
    throw BackendError(ErrorCode::kInvalidStructure,
                       "invalid resident execution limits");
  auto valid = ValidateBehavioralTransient(system);
  if (!valid.ok())
    throw BackendError(valid.error().code, valid.error().message);
  auto pattern = FormTransientCompanionMatrix(system.g, system.c,
                                              analysis.time_step_seconds, 1);
  if (!pattern.ok())
    throw BackendError(pattern.error().code, pattern.error().message);
  m.n = static_cast<int>(system.g.rows);
  m.nodes = static_cast<int>(system.node_names.size());
  m.maximum_step = analysis.time_step_seconds;
  m.minimum_step = m.maximum_step / limits.minimum_step_divisor;
  m.stop = analysis.stop_time_seconds;
  m.start = analysis.start_time_seconds;
  m.maximum_attempts = limits.maximum_step_attempts;
  m.maximum_accepted = limits.maximum_accepted_steps;
  m.maximum_newton = static_cast<int>(limits.nonlinear_maximum_iterations);
  if (!(std::isfinite(m.minimum_step) && m.minimum_step > 0))
    throw BackendError(ErrorCode::kInvalidStructure,
                       "unrepresentable resident minimum step");
  std::vector<int> offsets, columns;
  std::vector<double> g, c;
  offsets.push_back(0);
  for (int row = 0; row < m.n; ++row) {
    auto ig = system.g.row_offsets[row], ic = system.c.row_offsets[row];
    for (auto k = pattern.value().row_offsets[row];
         k < pattern.value().row_offsets[row + 1]; ++k) {
      const auto column = pattern.value().column_indices[k];
      columns.push_back(static_cast<int>(column));
      const bool has_g = ig < system.g.row_offsets[row + 1] &&
                         system.g.column_indices[ig] == column;
      const bool has_c = ic < system.c.row_offsets[row + 1] &&
                         system.c.column_indices[ic] == column;
      g.push_back(has_g ? system.g.values[ig++] : 0);
      c.push_back(has_c ? system.c.values[ic++] : 0);
    }
    offsets.push_back(static_cast<int>(columns.size()));
  }
  m.nnz = static_cast<int>(columns.size());
  m.row_offsets = allocation.Upload(offsets);
  m.columns = allocation.Upload(columns);
  m.g = allocation.Upload(g);
  m.c = allocation.Upload(c);
  m.b = allocation.Upload(system.b_dc);
  std::vector<DeviceProgram> programs;
  std::vector<internal::ExportedExpressionNode> nodes;
  std::vector<std::uint32_t> ad, dependencies;
  for (const auto &descriptor : system.behavioral_descriptors) {
    auto exported = internal::ExportExpressionProgram(descriptor.expression);
    if (!exported.ok())
      throw BackendError(exported.error().code, exported.error().message);
    const auto &p = exported.value();
    if (p.nodes.empty() || p.nodes.size() > kMaximumNodes ||
        p.root >= p.nodes.size() ||
        p.state_size != static_cast<unsigned>(m.n) ||
        nodes.size() + p.nodes.size() > kMaximumTotalNodes)
      throw BackendError(ErrorCode::kUnsupportedSize,
                         "resident expression workspace bound");
    programs.push_back(DeviceProgram{
        static_cast<unsigned>(nodes.size()), static_cast<unsigned>(ad.size()),
        static_cast<unsigned>(dependencies.size()),
        static_cast<unsigned>(p.nodes.size()),
        static_cast<unsigned>(p.reverse_ad_indices.size()),
        static_cast<unsigned>(p.dependencies.size()), p.root, p.state_size,
        p.dialect});
    nodes.insert(nodes.end(), p.nodes.begin(), p.nodes.end());
    ad.insert(ad.end(), p.reverse_ad_indices.begin(),
              p.reverse_ad_indices.end());
    dependencies.insert(dependencies.end(), p.dependencies.begin(),
                        p.dependencies.end());
  }
  std::vector<internal::ExportedExpressionNode> parallel = nodes;
  std::vector<Guard> guards(nodes.size());
  std::vector<int> node_levels(nodes.size(), -1);
  for (const auto &program : programs) {
    std::function<int(unsigned, int, bool, int)> schedule;
    schedule = [&](unsigned local, int guard, bool positive, int floor) -> int {
      const auto index = program.node_offset + local;
      if (local >= program.nodes || node_levels[index] != -1)
        throw BackendError(ErrorCode::kInvalidStructure,
                           "resident expression must be a bounded tree");
      node_levels[index] = -2;
      guards[index] = Guard{
          guard, (program.dialect == ExpressionDialect::kBehavioral ? 1U : 0U) |
                     (positive ? 2U : 0U)};
      auto &node = parallel[index];
      int level = floor;
      if (node.op != ExportedExpressionOp::kConstant &&
          node.op != ExportedExpressionOp::kState) {
        const int first_level = schedule(node.first, guard, positive, floor);
        node.first += program.node_offset;
        level = std::max(level, first_level + 1);
        if (node.op == ExportedExpressionOp::kIf) {
          level = std::max(level,
                           schedule(node.second, static_cast<int>(node.first),
                                    true, first_level + 1) +
                               1);
          level =
              std::max(level, schedule(node.third, static_cast<int>(node.first),
                                       false, first_level + 1) +
                                  1);
          node.second += program.node_offset;
          node.third += program.node_offset;
        } else if (node.op != ExportedExpressionOp::kNegate &&
                   node.op != ExportedExpressionOp::kExp) {
          level = std::max(level,
                           schedule(node.second, guard, positive, floor) + 1);
          node.second += program.node_offset;
        }
      }
      node_levels[index] = level;
      return level;
    };
    schedule(program.root, -1, false, 0);
  }
  m.expression_levels =
      *std::max_element(node_levels.begin(), node_levels.end()) + 1;
  std::vector<int> expression_order, expression_starts{0};
  for (int level = 0; level < m.expression_levels; ++level) {
    std::vector<int> members;
    for (int i = 0; i < static_cast<int>(nodes.size()); ++i)
      if (node_levels[i] == level)
        members.push_back(i);
    std::stable_sort(members.begin(), members.end(), [&](int a, int b) {
      return parallel[a].op < parallel[b].op;
    });
    expression_order.insert(expression_order.end(), members.begin(),
                            members.end());
    expression_starts.push_back(static_cast<int>(expression_order.size()));
  }
  std::vector<PackedExpressionNode> compact;
  std::vector<double> literals;
  for (std::size_t index = 0; index < parallel.size(); ++index) {
    const auto &node = parallel[index];
    const auto &guard = guards[index];
    if (node.first >= 16384 || node.second >= 16384 || node.third >= 16384 ||
        guard.condition < -1 || guard.condition >= 16384 ||
        static_cast<unsigned>(node.op) > 11 || node.constant > 1)
      throw BackendError(ErrorCode::kInvalidStructure,
                         "resident expression encoding bounds");
    PackedExpressionNode packed{};
    packed.first = node.first;
    packed.second = node.second;
    packed.third = node.third;
    packed.guard = guard.condition + 1;
    packed.op = node.op;
    packed.constant = node.constant;
    packed.behavioral = bool(guard.flags & 1);
    packed.positive = bool(guard.flags & 2);
    compact.push_back(packed);
    literals.push_back(node.value);
  }
  m.parallel_nodes = allocation.Upload(compact);
  m.literal_values = allocation.Upload(literals);
  m.expression_order = allocation.Upload(expression_order);
  m.expression_level_offsets = allocation.Upload(expression_starts);
  m.dependency_count = static_cast<int>(dependencies.size());
  m.programs = static_cast<int>(programs.size());
  m.program = allocation.Upload(programs);
  m.expression_node_count = static_cast<int>(nodes.size());
  std::vector<int> leaf_offsets{0}, leaves;
  for (const auto &program : programs) {
    for (unsigned dependency = 0; dependency < program.dependency_count;
         ++dependency) {
      for (unsigned j = 0; j < program.ad_count; ++j) {
        const auto index = program.node_offset + ad[program.ad_offset + j];
        const auto &node = parallel[index];
        if (node.op == ExportedExpressionOp::kState &&
            node.second == dependency)
          leaves.push_back(static_cast<int>(index));
      }
      leaf_offsets.push_back(static_cast<int>(leaves.size()));
    }
  }
  m.gradient_leaf_offsets = allocation.Upload(leaf_offsets);
  m.gradient_leaves = allocation.Upload(leaves);
  m.dependencies = allocation.Upload(dependencies);
  std::vector<int> expression_offsets{0}, jacobian_slots;
  std::vector<ExpressionStamp> expression_stamps;
  for (int row = 0; row < m.n; ++row) {
    for (std::size_t i = 0; i < system.behavioral_descriptors.size(); ++i) {
      for (const auto &stamp : system.behavioral_descriptors[i].rows) {
        if (stamp.row != static_cast<std::size_t>(row))
          continue;
        expression_stamps.push_back(
            ExpressionStamp{static_cast<int>(i), stamp.coefficient,
                            static_cast<int>(jacobian_slots.size())});
        for (unsigned j = 0; j < programs[i].dependency_count; ++j) {
          const auto column = dependencies[programs[i].dependency_offset + j];
          const auto found = std::lower_bound(
              columns.begin() + offsets[row],
              columns.begin() + offsets[row + 1], static_cast<int>(column));
          if (found == columns.begin() + offsets[row + 1] ||
              *found != static_cast<int>(column))
            throw BackendError(
                ErrorCode::kInvalidStructure,
                "resident Jacobian dependency absent from union");
          jacobian_slots.push_back(static_cast<int>(found - columns.begin()));
        }
      }
    }
    expression_offsets.push_back(static_cast<int>(expression_stamps.size()));
  }
  m.expression_offsets = allocation.Upload(expression_offsets);
  m.expression_stamps = allocation.Upload(expression_stamps);
  m.expression_jacobian_slots = allocation.Upload(jacobian_slots);
  double *persistent = allocation.Allocate<double>(4 * m.n);
  CheckCuda(cudaMemsetAsync(persistent, 0, 4 * m.n * sizeof(double),
                            allocation.stream()),
            "resident state initialization");
  emi03_cuda_internal::Synchronize(allocation.stream());
  workspace.state = persistent;
  workspace.history_current = persistent + m.n;
  workspace.history_older = persistent + 2 * m.n;
  workspace.history_trial = persistent + 3 * m.n;
  workspace.factored_jacobian = allocation.Allocate<double>(m.nnz);
  workspace.lu = allocation.Allocate<double>(m.n * N);
  workspace.permutation = allocation.Allocate<int>(m.n);
  workspace.last_dynamic_jacobian = allocation.Allocate<double>(m.nnz);
  workspace.output = allocation.Allocate<double>(Chunk * (m.n + 1));
  std::vector<Reactive> coordinates;
  for (const auto &cap : system.capacitor_initial_constraints)
    coordinates.push_back(Reactive{
        cap.positive_node_index ? static_cast<int>(*cap.positive_node_index)
                                : -1,
        cap.negative_node_index ? static_cast<int>(*cap.negative_node_index)
                                : -1,
        1e-7});
  for (const auto &ind : system.inductor_initial_constraints)
    coordinates.push_back(
        Reactive{static_cast<int>(ind.branch_index), -1, 1e-9});
  m.reactive = static_cast<int>(coordinates.size());
  m.coordinates = allocation.Upload(coordinates);
  if (m.reactive > m.n)
    throw BackendError(ErrorCode::kUnsupportedSize,
                       "resident reactive coordinate workspace bound");
  std::vector<Source> sources;
  std::vector<Stamp> stamps;
  std::vector<Pair> pairs;
  std::vector<double> waves;
  for (const auto &input : system.transient_sources) {
    Source source{};
    source.dc = input.dc_value;
    source.stamp_offset = static_cast<int>(stamps.size());
    source.stamps = static_cast<int>(input.rhs_stamps.size());
    for (const auto &stamp : input.rhs_stamps)
      stamps.push_back(Stamp{static_cast<int>(stamp.index), stamp.coefficient});
    if (const auto *p = std::get_if<PwlWaveform>(&input.waveform)) {
      source.type = 0;
      source.offset = static_cast<int>(pairs.size());
      source.count = static_cast<int>(p->time_value_pairs.size());
      for (const auto &[time, value] : p->time_value_pairs)
        pairs.push_back(Pair{time, value});
    } else if (const auto *p = std::get_if<PulseWaveform>(&input.waveform)) {
      source.type = 1;
      source.pulse = *p;
    } else
      throw BackendError(ErrorCode::kUnsupported, "resident waveform type");
    sources.push_back(source);
    auto points = CollectTransientWaveformBreakpoints(input.waveform, m.stop);
    if (!points.ok())
      throw BackendError(points.error().code, points.error().message);
    waves.insert(waves.end(), points.value().begin(), points.value().end());
  }
  std::sort(waves.begin(), waves.end());
  waves.erase(std::unique(waves.begin(), waves.end()), waves.end());
  auto hard = waves;
  hard.push_back(m.stop);
  hard.push_back(m.start);
  std::sort(hard.begin(), hard.end());
  hard.erase(std::unique(hard.begin(), hard.end()), hard.end());
  hard.erase(
      std::remove_if(hard.begin(), hard.end(), [](double t) { return t <= 0; }),
      hard.end());
  std::vector<unsigned char> hard_wave;
  for (double time : hard)
    hard_wave.push_back(std::binary_search(waves.begin(), waves.end(), time));
  m.hard_count = static_cast<int>(hard.size());
  m.hard_points = allocation.Upload(hard);
  m.hard_wave = allocation.Upload(hard_wave);
  m.sources = static_cast<int>(sources.size());
  m.source = allocation.Upload(sources);
  m.source_stamps = allocation.Upload(stamps);
  m.pwl = allocation.Upload(pairs);
  workspace.progress.first_audit = true;
  workspace.progress.proposed_step = m.maximum_step;
  return m;
}
} // namespace

Result<Emi03ResidentResult>
RunEmi03ResidentTransient(const MnaSystem &system, const TranAnalysis &analysis,
                          const TransientExecutionLimits &limits) {
  using Outcome = Result<Emi03ResidentResult>;
  try {
    emi03_cuda_internal::RequireJob();
    emi03_cuda_internal::Statistics().transient_algorithm =
        "resident-be-trap-fp64-v1";
    Allocations allocation;
    auto workspace = std::make_unique<Workspace>();
    auto model = Prepare(system, analysis, limits, allocation, *workspace);
    auto initial = BuildTransientInitialState(system, false);
    if (!initial.ok())
      return Outcome::Fail(initial.error().code, initial.error().message);
    auto factor_allocation = std::make_unique<Allocations>();
    PrepareFactorPlan(system, analysis, initial.value(), model,
                      *factor_allocation);
    std::size_t plan_count = 1;
    const auto SharedBytes = [&]() -> std::size_t {
      return ((model.factor_nonzeros + 23 * model.n + 2 * model.nnz +
               model.dependency_count + 2 * model.expression_node_count) *
                  sizeof(double) +
              model.programs * sizeof(DeviceResult) +
              model.expression_node_count + 7) /
                 8 * 8 +
             (2 * model.factor_nonzeros + model.factor_levels + 1) *
                 sizeof(int);
    };
    std::size_t shared_bytes = (SharedBytes() + 7) / 8 * 8;
    int device_index = 0, shared_limit = 0;
    CheckCuda(cudaGetDevice(&device_index), "resident device query");
    CheckCuda(cudaDeviceGetAttribute(&shared_limit,
                                     cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                     device_index),
              "resident shared memory limit");
    cudaFuncAttributes attributes{};
    CheckCuda(cudaFuncGetAttributes(&attributes, Emi03Advance),
              "resident kernel attributes");
    const int dynamic_limit =
        (shared_limit - static_cast<int>(attributes.sharedSizeBytes)) / 256 *
        256;
    if (shared_bytes > static_cast<std::size_t>(dynamic_limit))
      throw BackendError(ErrorCode::kUnsupportedSize,
                         "resident sparse factors exceed shared memory");
    const auto CacheFactorMetadata = [&] {
      const auto additional = model.factor_nonzeros * sizeof(FactorEntry) +
                              model.factor_terms * sizeof(FactorTerm);
      model.shared_factor_metadata =
          shared_bytes + additional <= static_cast<std::size_t>(dynamic_limit);
      if (model.shared_factor_metadata)
        shared_bytes = (shared_bytes + additional + 7) / 8 * 8;
      const auto expression_bytes =
          model.expression_node_count * sizeof(PackedExpressionNode);
      model.shared_expression_metadata =
          shared_bytes + expression_bytes <=
          static_cast<std::size_t>(dynamic_limit);
      if (model.shared_expression_metadata)
        shared_bytes += expression_bytes;
    };
    CacheFactorMetadata();
    CheckCuda(cudaFuncSetAttribute(Emi03Advance,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   dynamic_limit),
              "resident shared memory reservation");
    allocation.Copy(workspace->state, initial.value().data(),
                    model.n * sizeof(double), cudaMemcpyHostToDevice);
    emi03_cuda_internal::Statistics().upload_bytes += model.n * sizeof(double);
    auto *device = allocation.Allocate<Workspace>(1);
    allocation.Copy(device, workspace.get(), sizeof(Workspace),
                    cudaMemcpyHostToDevice);
    emi03_cuda_internal::Statistics().upload_bytes += sizeof(Workspace);
    Emi03ResidentResult result;
    if (analysis.start_time_seconds == 0) {
      auto emitted = limits.accepted_state_observer(0, initial.value());
      if (!emitted.ok())
        return Outcome::Fail(emitted.error().code, emitted.error().message);
      if (!emitted.value())
        return Outcome::Fail(ErrorCode::kIo,
                             "resident initial observer rejected output");
      ++result.emitted_points;
    }
    std::vector<double> output(Chunk * (model.n + 1)), state(model.n);
    Progress progress{};
    std::uint64_t planned_dense = 0;
    while (progress.time < model.stop && !progress.error) {
      model.refresh_enabled = plan_count < 32;
      Emi03Advance<<<1, Threads, shared_bytes, allocation.stream()>>>(model,
                                                                      device);
      CheckCuda(cudaGetLastError(), "resident transient launch");
      allocation.Copy(&progress, device, sizeof(Progress),
                      cudaMemcpyDeviceToHost);
      emi03_cuda_internal::Statistics().readback_bytes += sizeof(Progress);
      if (progress.output_count < 0 || progress.output_count > Chunk)
        return Outcome::Fail(ErrorCode::kPreparedInvalidResult,
                             "resident output count mismatch");
      const auto count =
          static_cast<std::size_t>(progress.output_count) * (model.n + 1);
      if (count) {
        allocation.Copy(output.data(), workspace->output,
                        count * sizeof(double), cudaMemcpyDeviceToHost);
        emi03_cuda_internal::Statistics().readback_bytes +=
            count * sizeof(double);
        for (int row = 0; row < progress.output_count; ++row) {
          const auto *point = output.data() + row * (model.n + 1);
          if (point[0] < analysis.start_time_seconds)
            continue;
          std::copy(point + 1, point + model.n + 1, state.begin());
          auto emitted = limits.accepted_state_observer(point[0], state);
          if (!emitted.ok())
            return Outcome::Fail(emitted.error().code, emitted.error().message);
          if (!emitted.value())
            return Outcome::Fail(ErrorCode::kIo,
                                 "resident observer rejected output");
          ++result.emitted_points;
        }
      }
      if (!progress.error && progress.time < model.stop &&
          progress.dense_factors > planned_dense && plan_count < 32) {
        std::vector<double> values(model.nnz);
        allocation.Copy(values.data(), workspace->last_dynamic_jacobian,
                        values.size() * sizeof(double), cudaMemcpyDeviceToHost);
        emi03_cuda_internal::Statistics().readback_bytes +=
            values.size() * sizeof(double);
        auto next = std::make_unique<Allocations>();
        PrepareFactorPlan(system, analysis, initial.value(), model, *next,
                          &values);
        shared_bytes = (SharedBytes() + 7) / 8 * 8;
        CacheFactorMetadata();
        if (shared_bytes > static_cast<std::size_t>(dynamic_limit))
          throw BackendError(ErrorCode::kUnsupportedSize,
                             "resident refreshed factors exceed shared memory");
        factor_allocation = std::move(next);
        planned_dense = progress.dense_factors;
        ++plan_count;
      }
    }
    if (std::getenv("EMI03_RESIDENT_PROFILE")) {
      cudaFuncAttributes attributes;
      CheckCuda(cudaFuncGetAttributes(&attributes, Emi03Advance),
                "resident kernel attributes");
      std::fprintf(stderr,
                   "resident resources registers=%d local=%zu shared=%zu "
                   "dynamic=%zu nodes=%d programs=%d levels=%d/%d/%d\n",
                   attributes.numRegs, attributes.localSizeBytes,
                   attributes.sharedSizeBytes, shared_bytes,
                   model.expression_node_count, model.programs,
                   model.forward_levels, model.backward_levels,
                   model.factor_levels);
      std::fprintf(stderr,
                   "resident plans=%zu sparse retries=%llu dense=%llu terms=%d "
                   "metadata=%d\n",
                   plan_count,
                   static_cast<unsigned long long>(progress.linear_retries),
                   static_cast<unsigned long long>(progress.dense_factors),
                   model.factor_terms, model.shared_factor_metadata);
      std::fprintf(stderr,
                   "resident breakdown value=%llu ad=%llu factor_prepare=%llu "
                   "forward=%llu residual=%llu\n",
                   progress.expression_value_cycles,
                   progress.expression_ad_cycles,
                   progress.factor_prepare_cycles, progress.forward_cycles,
                   progress.linear_residual_cycles);
      std::fprintf(
          stderr, "resident expression cache hits=%llu\n",
          static_cast<unsigned long long>(progress.expression_cache_hits));
      std::fprintf(stderr,
                   "resident n=%d fill=%d expression=%llu factor=%llu "
                   "triangular=%llu total=%llu\n",
                   model.n, model.factor_nonzeros, progress.expression_cycles,
                   progress.factor_cycles, progress.triangular_cycles,
                   progress.total_cycles);
    }
    auto &stats = emi03_cuda_internal::Statistics();
    stats.analyses += plan_count;
    stats.factorizations += progress.factors;
    stats.reuses += progress.reuses;
    stats.solves += progress.solves + progress.refinements;
    stats.successful_solves += progress.solves;
    stats.refinements += progress.refinements;
    stats.expression_batches += progress.expression_batches;
    stats.expression_full_ad += progress.full_expressions;
    stats.expression_value_only += progress.value_expressions;
    ++stats.expression_program_uploads;
    if (progress.error)
      return Outcome::Fail(
          static_cast<ErrorCode>(progress.error - 1),
          "resident transient rejected at t=" + std::to_string(progress.time) +
              " attempts=" + std::to_string(progress.attempts));
    result.attempts = progress.attempts;
    result.rejected = progress.rejected;
    result.nonlinear_rejections = progress.nonlinear;
    result.history_estimates = progress.history_estimates;
    result.history_checks = progress.history_checks;
    result.doubling_estimates = progress.doubling;
    result.history_fallback_entries = progress.fallback_entries;
    result.history_fallback_recoveries = progress.fallback_recoveries;
    result.solver_statistics.symbolic_analyses = plan_count;
    result.solver_statistics.numeric_factorizations = progress.factors;
    result.solver_statistics.numeric_reuses = progress.reuses;
    result.solver_statistics.numeric_refactorization_fallbacks =
        progress.dense_factors;
    result.solver_statistics.solves = progress.solves;
    result.solver_statistics.iterative_refinement_solves = progress.refinements;
    return Outcome::Ok(std::move(result));
  } catch (const BackendError &e) {
    return Outcome::Fail(e.code, e.what());
  } catch (const std::bad_alloc &) {
    return Outcome::Fail(ErrorCode::kUnsupportedSize,
                         "resident host allocation failed");
  }
}
} // namespace ohmnivore
