// Private research adapter shared by the manual time-window probes.
#include "cuda/emi03_cuda_internal.h"
#include "cuda/emi03_expression_device.cuh"
#include "cuda/emi03_real_solver.h"
#include "cuda/emi03_reduction.cuh"
#include "cuda/emi03_resident.h"
#include "ohmnivore/behavioral.h"
#include "ohmnivore/nonlinear.h"
#include "ohmnivore/solver.h"
#include "ohmnivore/waveform.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

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
#define Emi03Advance UnusedAdvanceInMatrixProbe
#include "cuda/emi03_resident_device.cuh"
#undef Emi03Advance

class HostStaging {
public:
  static constexpr std::size_t kBytes = Chunk * (N + 1) * sizeof(double);
  HostStaging() {
    CheckCuda(cudaHostAlloc(&data_, kBytes, cudaHostAllocDefault),
              "resident private pinned transfer staging");
  }
  HostStaging(const HostStaging &) = delete;
  HostStaging &operator=(const HostStaging &) = delete;
  ~HostStaging() {
    if (data_ && cudaFreeHost(data_) != cudaSuccess)
      ++emi03_cuda_internal::Statistics().cleanup_failures;
  }
  void *data() const { return data_; }

private:
  void *data_ = nullptr;
};
class Allocations {
public:
  explicit Allocations(HostStaging &staging, cudaStream_t consumer = nullptr)
      : staging_(staging), consumer_(consumer) {
    CheckCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
              "resident stream creation");
  }
  Allocations(const Allocations &) = delete;
  Allocations &operator=(const Allocations &) = delete;
  cudaStream_t stream() const { return stream_; }
  HostStaging &staging() const { return staging_; }
  void Copy(void *destination, const void *source, std::size_t bytes,
            cudaMemcpyKind kind) {
    if (kind != cudaMemcpyHostToDevice && kind != cudaMemcpyDeviceToHost)
      throw BackendError(ErrorCode::kUnsupported, "resident copy direction");
    // Pageable asynchronous copies can spin in the driver before returning.
    // Keep bounded pinned staging private to each job. Every copy completes
    // before another stream within this job uses that storage. Completion
    // queries sleep between attempts: blocking event waits still consumed a CPU
    // core on the measured WSL driver. All waiting remains in charged wall
    // time.
    if (!completed_)
      CheckCuda(
          cudaEventCreateWithFlags(&completed_, cudaEventBlockingSync |
                                                    cudaEventDisableTiming),
          "resident blocking completion event");
    for (std::size_t offset = 0; offset < bytes;
         offset += HostStaging::kBytes) {
      const auto count = std::min(HostStaging::kBytes, bytes - offset);
      auto *target = static_cast<unsigned char *>(destination) + offset;
      const auto *input = static_cast<const unsigned char *>(source) + offset;
      if (kind == cudaMemcpyHostToDevice)
        std::memcpy(staging_.data(), input, count);
      CheckCuda(cudaMemcpyAsync(
                    kind == cudaMemcpyHostToDevice ? target : staging_.data(),
                    kind == cudaMemcpyHostToDevice ? staging_.data() : input,
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
        std::memcpy(target, staging_.data(), count);
    }
  }
  ~Allocations() {
    // Factor metadata is uploaded on this owner's stream, but read by the
    // resident simulation stream. Both must finish before asynchronous release,
    // including exceptional exits before the normal output-copy completion.
    if (consumer_ && consumer_ != stream_ &&
        cudaStreamSynchronize(consumer_) != cudaSuccess)
      ++emi03_cuda_internal::Statistics().cleanup_failures;
    if (cudaStreamSynchronize(stream_) != cudaSuccess)
      ++emi03_cuda_internal::Statistics().cleanup_failures;
    for (auto it = owned_.rbegin(); it != owned_.rend(); ++it)
      emi03_cuda_internal::FreeDevice(it->first, it->second);
    if (completed_ && cudaEventDestroy(completed_) != cudaSuccess)
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
  cudaStream_t stream_ = nullptr;
  cudaEvent_t completed_ = nullptr;
  HostStaging &staging_;
  cudaStream_t consumer_ = nullptr;
  std::vector<std::pair<void *, std::size_t>> owned_;
};
void PrepareFactorPlan(const MnaSystem &system, const TranAnalysis &analysis,
                       const std::vector<double> &initial, Model &model,
                       Allocations &allocation,
                       const std::vector<double> *refresh = nullptr) {
  // The replay already contains the complete actual Jacobian. Preserve the
  // selected structural ordering and GPU pivot discovery without reassembling
  // it.
  static_cast<void>(analysis);
  static_cast<void>(initial);
  auto matrix = system.g;
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
    Allocations temporary(allocation.staging());
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
struct TimeData {
  int length, rank;
  const int *active, *mass_rows, *mass_columns;
  const double *mass_values;
  double step;
  double *factor, *inverse, *equil, *inverse_equil, *row_norm;
  double *transfer, *powers, *rhs, *initial, *solution, *residual;
  double *transfer_certificate;
  unsigned long long *counters;
  double *z, *scan_a, *scan_b;
  int *error, *invalid, *nonzero;
};
__device__ void InitializeTime(const Model &m, const Workspace *workspace,
                               Shared &s, double *scratch) {
  auto &w = s.runtime;
  if (threadIdx.x == 0) {
    w = *workspace;
    s.error = 0;
    s.factor_valid = false;
    s.active_scale = 0;
    s.factored_scale = -1;
    s.expression_cache_valid = false;
    s.expression_cache_derivatives = false;
    w.progress.output_count = 0;
    s.factor = scratch;
    s.triangular = scratch + m.factor_nonzeros;
    s.inverse = s.triangular + m.n;
    double *vectors = s.inverse + m.n;
    w.state = vectors + 0 * m.n;
    w.full = vectors + 1 * m.n;
    w.half = vectors + 2 * m.n;
    w.current = vectors + 3 * m.n;
    w.proposed = vectors + 4 * m.n;
    w.delta = vectors + 5 * m.n;
    w.companion_rhs = vectors + 6 * m.n;
    w.affine_rhs = vectors + 7 * m.n;
    w.residual = vectors + 8 * m.n;
    w.row_scale = vectors + 9 * m.n;
    w.solution = vectors + 10 * m.n;
    w.equil = vectors + 11 * m.n;
    w.inverse_equil = vectors + 12 * m.n;
    w.linear_row_norm = vectors + 13 * m.n;
    s.expression_state = vectors + 14 * m.n;
    // Linear correction/forward work finish before the next nonlinear assembly
    // replaces residual/scale. The second half-step result is the final Newton
    // proposal and survives until its error/history checks have consumed it.
    w.correction = w.residual;
    w.work = w.row_scale;
    w.second = w.proposed;
    w.history_current = vectors + 15 * m.n;
    w.history_older = w.history_current + m.reactive;
    w.history_trial = w.history_older + m.reactive;
    w.base = w.history_trial + m.reactive;
    w.jacobian = w.base + m.nnz;
    w.gradients = w.jacobian + m.nnz;
    w.expressions =
        reinterpret_cast<DeviceResult *>(w.gradients + m.dependency_count);
    s.expression_values =
        reinterpret_cast<double *>(w.expressions + m.programs);
    s.expression_adjoints = s.expression_values + m.expression_node_count;
    s.expression_active = reinterpret_cast<unsigned char *>(
        s.expression_adjoints + m.expression_node_count);
    s.factor_columns = reinterpret_cast<int *>(
        (reinterpret_cast<std::uintptr_t>(s.expression_active +
                                          m.expression_node_count) +
         7) &
        ~std::uintptr_t{7});
    s.factor_order = s.factor_columns + m.factor_nonzeros;
    s.factor_level_offsets = s.factor_order + m.factor_nonzeros;
    s.factor_entries = m.factor_entries;
    s.factor_term = m.factor_term;
    auto end = (reinterpret_cast<std::uintptr_t>(s.factor_level_offsets +
                                                 m.factor_levels + 1) +
                7) &
               ~std::uintptr_t{7};
    if (m.shared_factor_metadata) {
      s.factor_entries = reinterpret_cast<const FactorEntry *>(end);
      s.factor_term = reinterpret_cast<const FactorTerm *>(s.factor_entries +
                                                           m.factor_nonzeros);
      end = (reinterpret_cast<std::uintptr_t>(s.factor_term + m.factor_terms) +
             7) &
            ~std::uintptr_t{7};
    }
    s.expression_nodes =
        m.shared_expression_metadata
            ? reinterpret_cast<const PackedExpressionNode *>(end)
            : m.parallel_nodes;
    if (m.shared_expression_metadata)
      end += m.expression_node_count * sizeof(PackedExpressionNode);
    s.matrix_rows =
        m.shared_structure ? reinterpret_cast<const int *>(end) : m.row_offsets;
    s.matrix_columns = m.shared_structure ? s.matrix_rows + m.n + 1 : m.columns;
  }
  __syncthreads();
  if (m.shared_structure) {
    for (int i = threadIdx.x; i <= m.n; i += Threads)
      const_cast<int *>(s.matrix_rows)[i] = m.row_offsets[i];
    for (int i = threadIdx.x; i < m.nnz; i += Threads)
      const_cast<int *>(s.matrix_columns)[i] = m.columns[i];
  }
  for (int i = threadIdx.x; i < m.n; i += Threads) {
    w.state[i] = workspace->state[i];
    if (i < m.reactive) {
      w.history_current[i] = workspace->history_current[i];
      w.history_older[i] = workspace->history_older[i];
      w.history_trial[i] = workspace->history_trial[i];
    }
    s.factor_diagonal[i] = m.factor_diagonal[i];
    s.row_permutation[i] = m.row_permutation[i];
    s.column_permutation[i] = m.column_permutation[i];
    s.forward_rows[i] = m.forward_rows[i];
    s.backward_rows[i] = m.backward_rows[i];
  }
  for (int i = threadIdx.x; i <= m.n; i += Threads) {
    s.factor_rows[i] = m.factor_rows[i];
  }
  for (int i = threadIdx.x; i <= m.forward_levels; i += Threads)
    s.forward_offsets[i] = m.forward_offsets[i];
  for (int i = threadIdx.x; i <= m.backward_levels; i += Threads)
    s.backward_offsets[i] = m.backward_offsets[i];
  for (int i = threadIdx.x; i < m.factor_nonzeros; i += Threads)
    s.factor_columns[i] = m.factor_columns[i];
  for (int i = threadIdx.x; i < m.factor_nonzeros; i += Threads)
    s.factor_order[i] = m.factor_order[i];
  for (int i = threadIdx.x; i <= m.factor_levels; i += Threads)
    s.factor_level_offsets[i] = m.factor_level_offsets[i];
  __syncthreads();
  if (m.shared_factor_metadata) {
    for (int i = threadIdx.x; i < m.factor_nonzeros; i += Threads)
      const_cast<FactorEntry *>(s.factor_entries)[i] = m.factor_entries[i];
    for (int i = threadIdx.x; i < m.factor_terms; i += Threads)
      const_cast<FactorTerm *>(s.factor_term)[i] = m.factor_term[i];
  }
  if (m.shared_expression_metadata)
    for (int index = threadIdx.x; index < m.expression_node_count;
         index += Threads)
      const_cast<PackedExpressionNode *>(s.expression_nodes)[index] =
          m.parallel_nodes[index];
  __syncthreads();
}

__device__ void CachedTimeLinear(const Model &m, Workspace &w, Shared &s) {
  if (s.error)
    return;
  for (int retry = 0; retry < 2 && !s.error; ++retry) {
    __syncthreads();
    Triangular(m, w, s, w.affine_rhs, w.solution);
    bool corrected = false, accepted = false;
    for (int iteration = 0; iteration <= 4 && !s.error; ++iteration) {
      __syncthreads();
      const auto residual_start = clock64();
      double component = 0, normalized = 0, matrix_norm = 0, rhs_norm = 0,
             solution_norm = 0, nonzero = 0;
      for (int row = threadIdx.x; row < m.n; row += Threads) {
        Sum residual;
        residual.Add(w.affine_rhs[row]);
        double scale = fabs(w.affine_rhs[row]),
               row_norm = s.sparse_factor ? w.linear_row_norm[row] : 0;
        for (int k = s.matrix_rows[row]; k < s.matrix_rows[row + 1]; ++k) {
          const int col = s.matrix_columns[k];
          const double a = w.jacobian[k];
          residual.Product(-a, w.solution[col]);
          scale += fabs(a * w.solution[col]);
          if (!s.sparse_factor)
            row_norm += fabs(a);
        }
        if (!s.sparse_factor)
          row_norm /= w.equil[row];
        const double r = residual.Value();
        w.correction[row] = r;
        if (!Bounded(r))
          Reject(s, Nonfinite);
        if (r != 0)
          nonzero = 1;
        component = PositiveMaximum(component, scale == 0 ? (r == 0 ? 0 : 1e100)
                                                          : fabs(r) / scale);
        normalized =
            PositiveMaximum(normalized, Equilibrated(fabs(r), row, w, s));
        matrix_norm = PositiveMaximum(matrix_norm, row_norm);
        rhs_norm = PositiveMaximum(
            rhs_norm, Equilibrated(fabs(w.affine_rhs[row]), row, w, s));
        solution_norm = PositiveMaximum(solution_norm, fabs(w.solution[row]));
      }
      double maxima[]{component, normalized,    matrix_norm,
                      rhs_norm,  solution_norm, nonzero};
      ValidationMaxima(maxima, s);
      component = maxima[0];
      normalized = maxima[1];
      matrix_norm = maxima[2];
      rhs_norm = maxima[3];
      solution_norm = maxima[4];
      nonzero = maxima[5];
      if (threadIdx.x == 0)
        w.progress.linear_residual_cycles += clock64() - residual_start;
      const double denom = matrix_norm * solution_norm + rhs_norm;
      const bool valid =
          component <= 1e-5 &&
          (denom == 0 ? normalized == 0 : normalized / denom <= 1e-10);
      if ((corrected || nonzero == 0) && valid) {
        accepted = true;
        break;
      }
      if (iteration == 4)
        break;
      Triangular(m, w, s, w.correction, w.delta);
      for (int j = threadIdx.x; j < m.n; j += Threads) {
        w.solution[j] += w.delta[j];
        if (!Bounded(w.solution[j]))
          Reject(s, Nonfinite);
      }
      if (threadIdx.x == 0)
        ++w.progress.refinements;
      corrected = true;
      __syncthreads();
    }
    if (accepted || s.error)
      break;
    if (retry == 0 && s.sparse_factor) {
      if (threadIdx.x == 0) {
        s.sparse_factor = false;
        ++w.progress.linear_retries;
      }
      __syncthreads();
    } else {
      if (threadIdx.x == 0)
        Reject(s, Invalid);
      __syncthreads();
    }
  }
  if (threadIdx.x == 0 && !s.error)
    ++w.progress.solves;
  __syncthreads();
}

// Internal transfer/particular solutions are preconditioner work, not accepted
// timestep states. Full original per-step componentwise certification occurs in
// TimeResidual and on the independent CPU after coupled recovery.
__device__ void IntermediateTimeLinear(const Model &m, Workspace &w, Shared &s,
                                       double *certificate) {
  if (s.error)
    return;
  for (int retry = 0; retry < 2 && !s.error; ++retry) {
    __syncthreads();
    Triangular(m, w, s, w.affine_rhs, w.solution);
    bool corrected = false, accepted = false;
    for (int iteration = 0; iteration <= 4 && !s.error; ++iteration) {
      __syncthreads();
      const auto residual_start = clock64();
      double component = 0, normalized = 0, matrix_norm = 0, rhs_norm = 0,
             solution_norm = 0, nonzero = 0;
      for (int row = threadIdx.x; row < m.n; row += Threads) {
        Sum residual;
        residual.Add(w.affine_rhs[row]);
        double scale = fabs(w.affine_rhs[row]),
               row_norm = s.sparse_factor ? w.linear_row_norm[row] : 0;
        for (int k = s.matrix_rows[row]; k < s.matrix_rows[row + 1]; ++k) {
          const int col = s.matrix_columns[k];
          const double a = w.jacobian[k];
          residual.Product(-a, w.solution[col]);
          scale += fabs(a * w.solution[col]);
          if (!s.sparse_factor)
            row_norm += fabs(a);
        }
        if (!s.sparse_factor)
          row_norm /= w.equil[row];
        const double r = residual.Value();
        w.correction[row] = r;
        if (!Bounded(r))
          Reject(s, Nonfinite);
        if (r != 0)
          nonzero = 1;
        component = PositiveMaximum(component, scale == 0 ? (r == 0 ? 0 : 1e100)
                                                          : fabs(r) / scale);
        normalized =
            PositiveMaximum(normalized, Equilibrated(fabs(r), row, w, s));
        matrix_norm = PositiveMaximum(matrix_norm, row_norm);
        rhs_norm = PositiveMaximum(
            rhs_norm, Equilibrated(fabs(w.affine_rhs[row]), row, w, s));
        solution_norm = PositiveMaximum(solution_norm, fabs(w.solution[row]));
      }
      double maxima[]{component, normalized,    matrix_norm,
                      rhs_norm,  solution_norm, nonzero};
      ValidationMaxima(maxima, s);
      component = maxima[0];
      normalized = maxima[1];
      matrix_norm = maxima[2];
      rhs_norm = maxima[3];
      solution_norm = maxima[4];
      nonzero = maxima[5];
      if (threadIdx.x == 0)
        w.progress.linear_residual_cycles += clock64() - residual_start;
      const double denom = matrix_norm * solution_norm + rhs_norm;
      const bool valid =

          (denom == 0 ? normalized == 0 : normalized / denom <= 1e-10);
      if ((corrected || nonzero == 0) && valid) {
        if (threadIdx.x == 0 && certificate) {
          certificate[0] = denom == 0 ? 0 : normalized / denom;
          certificate[1] = component;
        }
        accepted = true;
        break;
      }
      if (iteration == 4)
        break;
      Triangular(m, w, s, w.correction, w.delta);
      for (int j = threadIdx.x; j < m.n; j += Threads) {
        w.solution[j] += w.delta[j];
        if (!Bounded(w.solution[j]))
          Reject(s, Nonfinite);
      }
      if (threadIdx.x == 0)
        ++w.progress.refinements;
      corrected = true;
      __syncthreads();
    }
    if (accepted || s.error)
      break;
    if (retry == 0 && s.sparse_factor) {
      if (threadIdx.x == 0) {
        s.sparse_factor = false;
        ++w.progress.linear_retries;
      }
      __syncthreads();
    } else {
      if (threadIdx.x == 0)
        Reject(s, Invalid);
      __syncthreads();
    }
  }
  if (threadIdx.x == 0 && !s.error)
    ++w.progress.solves;
  __syncthreads();
}

__device__ void LoadTimeFactor(const Model &m, Workspace &w, Shared &s,
                               const TimeData &t) {
  for (int i = threadIdx.x; i < m.factor_nonzeros; i += Threads)
    s.factor[i] = t.factor[i];
  for (int i = threadIdx.x; i < m.n; i += Threads) {
    s.inverse[i] = t.inverse[i];
    w.equil[i] = t.equil[i];
    w.inverse_equil[i] = t.inverse_equil[i];
    w.linear_row_norm[i] = t.row_norm[i];
  }
  if (threadIdx.x == 0) {
    s.sparse_factor = true;
    s.factor_valid = true;
  }
  __syncthreads();
}
__global__ void CacheTimeFactors(const Model *models, const Workspace *all,
                                 TimeData *data) {
  const int job = blockIdx.x;
  const auto m = models[job];
  const auto t = data[job];
  __shared__ Shared s;
  extern __shared__ double time_storage[];
  InitializeTime(m, all + job, s, time_storage);
  auto &w = s.runtime;
  for (int i = threadIdx.x; i < m.nnz; i += Threads)
    w.jacobian[i] = all[job].jacobian[i];
  __syncthreads();
  Factor(m, w, s);
  if (threadIdx.x == 0 && (!s.sparse_factor || s.error))
    atomicCAS(t.error, 0, s.error ? s.error : Invalid);
  if (s.error || !s.sparse_factor)
    return;
  for (int i = threadIdx.x; i < m.factor_nonzeros; i += Threads)
    t.factor[i] = s.factor[i];
  for (int i = threadIdx.x; i < m.n; i += Threads) {
    t.inverse[i] = s.inverse[i];
    t.equil[i] = w.equil[i];
    t.inverse_equil[i] = w.inverse_equil[i];
    t.row_norm[i] = w.linear_row_norm[i];
  }
  __syncthreads();
  // Cache a complete native-FP64 pivoted fallback for this immutable matrix.
  // Later independent RHS solves only read these factors.
  DenseFactor(m, w, s);
  if (threadIdx.x == 0 && s.error)
    atomicCAS(t.error, 0, s.error);
}
__global__ void MakeTimeTransfer(const Model *models, const Workspace *all,
                                 TimeData *data, int rank) {
  const int job = blockIdx.x / rank, col = blockIdx.x % rank;
  const auto m = models[job];
  const auto t = data[job];
  __shared__ Shared s;
  extern __shared__ double time_storage[];
  InitializeTime(m, all + job, s, time_storage);
  auto &w = s.runtime;
  LoadTimeFactor(m, w, s, t);
  for (int i = threadIdx.x; i < m.nnz; i += Threads)
    w.jacobian[i] = all[job].jacobian[i];
  for (int row = threadIdx.x; row < m.n; row += Threads) {
    double value = 0;
    for (int j = t.mass_rows[row]; j < t.mass_rows[row + 1]; ++j)
      if (t.mass_columns[j] == t.active[col])
        value = t.mass_values[j] / t.step;
    w.affine_rhs[row] = value;
  }
  __syncthreads();
  IntermediateTimeLinear(m, w, s, t.transfer_certificate + 2 * col);
  if (threadIdx.x == 0 && s.error)
    atomicCAS(t.error, 0, s.error);
  for (int row = threadIdx.x; row < m.n && !s.error; row += Threads)
    t.transfer[row * rank + col] = w.solution[row];
}
__global__ void TimePower(TimeData *data, int rank, int level) {
  const int job = blockIdx.y, index = blockIdx.x * blockDim.x + threadIdx.x;
  auto t = data[job];
  if (index >= rank * rank)
    return;
  if (level == 0) {
    t.powers[index] = t.transfer[t.active[index / rank] * rank + index % rank];
    return;
  }
  const double *before = t.powers + (level - 1) * rank * rank;
  double value = 0;
  for (int k = 0; k < rank; ++k)
    value = fma(before[(index / rank) * rank + k],
                before[k * rank + index % rank], value);
  if (!Bounded(value))
    atomicCAS(t.error, 0, Nonfinite);
  t.powers[level * rank * rank + index] = value;
}
__global__ void TimeIndependent(const Model *models, const Workspace *all,
                                TimeData *data, int length, bool correction) {
  const int job = blockIdx.x / length, point = blockIdx.x % length;
  const auto m = models[job];
  const auto t = data[job];
  __shared__ Shared s;
  extern __shared__ double time_storage[];
  InitializeTime(m, all + job, s, time_storage);
  auto &w = s.runtime;
  LoadTimeFactor(m, w, s, t);
  for (int i = threadIdx.x; i < m.nnz; i += Threads)
    w.jacobian[i] = all[job].jacobian[i];
  const double *rhs = (correction ? t.residual : t.rhs) + point * m.n;
  for (int row = threadIdx.x; row < m.n; row += Threads)
    w.affine_rhs[row] = rhs[row];
  __syncthreads();
  IntermediateTimeLinear(m, w, s, nullptr);
  if (threadIdx.x == 0 && s.error)
    atomicCAS(t.error, 0, s.error);
  if (!s.error) {
    for (int row = threadIdx.x; row < m.n; row += Threads)
      t.z[point * m.n + row] = w.solution[row];
    for (int row = threadIdx.x; row < t.rank; row += Threads) {
      double value = w.solution[t.active[row]];
      if (point == 0 && !correction)
        for (int col = 0; col < t.rank; ++col)
          value = fma(t.powers[row * t.rank + col], t.initial[t.active[col]],
                      value);
      t.scan_a[point * t.rank + row] = value;
    }
  }
}
// Each warp owns one RHS and three disjoint vectors. Cached factors are
// read-only.
__device__ double TimeWarpMax(double value) {
  for (int off = 16; off; off /= 2)
    value = fmax(value, __shfl_down_sync(0xffffffff, value, off));
  return __shfl_sync(0xffffffff, value, 0);
}
__device__ bool TimeWarpTriangular(const Model &m, const Workspace &w,
                                   const TimeData &t, const double *rhs,
                                   double *result, double *values, bool dense) {
  const int lane = threadIdx.x % 32;
  bool bad = false;
  if (!dense) {
    for (int level = 0; level < m.forward_levels; ++level) {
      for (int at = m.forward_offsets[level] + lane;
           at < m.forward_offsets[level + 1]; at += 32) {
        int row = m.forward_rows[at], original = m.row_permutation[row];
        double value = isfinite(t.inverse_equil[original])
                           ? rhs[original] * t.inverse_equil[original]
                           : rhs[original] / t.equil[original];
        for (int j = m.factor_rows[row]; j < m.factor_diagonal[row]; ++j)
          value = fma(-t.factor[j], values[m.factor_columns[j]], value);
        values[row] = value;
      }
      __syncwarp();
    }
    for (int level = 0; level < m.backward_levels; ++level) {
      for (int at = m.backward_offsets[level] + lane;
           at < m.backward_offsets[level + 1]; at += 32) {
        int row = m.backward_rows[at];
        double value = values[row];
        for (int j = m.factor_diagonal[row] + 1; j < m.factor_rows[row + 1];
             ++j)
          value = fma(-t.factor[j], values[m.factor_columns[j]], value);
        values[row] = value * t.inverse[row];
        bad |= !Bounded(values[row]);
      }
      __syncwarp();
    }
    for (int row = lane; row < m.n; row += 32)
      result[m.column_permutation[row]] = values[row];
  } else {
    for (int row = lane; row < m.n; row += 32)
      values[row] = rhs[w.permutation[row]] / t.equil[w.permutation[row]];
    __syncwarp();
    for (int row = 0; row < m.n; ++row) {
      double sum = 0;
      for (int col = lane; col < row; col += 32)
        sum = fma(w.lu[row * N + col], values[col], sum);
      for (int off = 16; off; off /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, off);
      if (lane == 0)
        values[row] -= sum;
      __syncwarp();
    }
    for (int row = m.n - 1; row >= 0; --row) {
      double sum = 0;
      for (int col = row + 1 + lane; col < m.n; col += 32)
        sum = fma(w.lu[row * N + col], result[col], sum);
      for (int off = 16; off; off /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, off);
      if (lane == 0) {
        result[row] = (values[row] - sum) / w.lu[row * N + row];
        bad |= !Bounded(result[row]);
      }
      __syncwarp();
    }
  }
  __syncwarp();
  return !__any_sync(0xffffffff, bad);
}
__global__ void TimeWarpIndependent(const Model *models, const Workspace *all,
                                    TimeData *data, int length,
                                    bool correction) {
  const int jobs_per_case = (length + 7) / 8, job = blockIdx.x / jobs_per_case,
            point = (blockIdx.x % jobs_per_case) * 8 + threadIdx.x / 32,
            lane = threadIdx.x % 32;
  const auto m = models[job];
  const auto t = data[job];
  const auto w = all[job];
  if (point >= length)
    return;
  extern __shared__ double vectors[];
  double *values = vectors + (threadIdx.x / 32) * 3 * m.n, *x = values + m.n,
         *residual = x + m.n;
  const double *rhs = (correction ? t.residual : t.rhs) + point * m.n;
  bool accepted = false;
  int error = 0;
  unsigned long long refinements = 0, retries = 0;
  for (int attempt = 0; attempt < 2 && !accepted && !error; ++attempt) {
    const bool dense = attempt != 0;
    if (dense)
      ++retries;
    if (!TimeWarpTriangular(m, w, t, rhs, x, values, dense)) {
      error = Nonfinite;
      break;
    }
    for (int iteration = 0; iteration <= 4; ++iteration) {
      double normalized = 0, matrix_norm = 0, rhs_norm = 0, x_norm = 0;
      bool nonzero = false, bad = false;
      for (int row = lane; row < m.n; row += 32) {
        Sum sum;
        sum.Add(rhs[row]);
        for (int j = m.row_offsets[row]; j < m.row_offsets[row + 1]; ++j)
          sum.Product(-w.jacobian[j], x[m.columns[j]]);
        double value = sum.Value();
        residual[row] = value;
        bad |= !Bounded(value) || !Bounded(x[row]);
        nonzero |= value != 0;
        normalized = fmax(normalized, fabs(value) / t.equil[row]);
        matrix_norm = fmax(matrix_norm, t.row_norm[row]);
        rhs_norm = fmax(rhs_norm, fabs(rhs[row]) / t.equil[row]);
        x_norm = fmax(x_norm, fabs(x[row]));
      }
      __syncwarp();
      if (__any_sync(0xffffffff, bad)) {
        error = Nonfinite;
        break;
      }
      normalized = TimeWarpMax(normalized);
      matrix_norm = TimeWarpMax(matrix_norm);
      rhs_norm = TimeWarpMax(rhs_norm);
      x_norm = TimeWarpMax(x_norm);
      nonzero = __any_sync(0xffffffff, nonzero);
      const double denom = matrix_norm * x_norm + rhs_norm;
      if ((iteration > 0 || !nonzero) &&
          (denom == 0 ? normalized == 0 : normalized / denom <= 1e-10)) {
        accepted = true;
        break;
      }
      if (iteration == 4)
        break;
      if (!TimeWarpTriangular(m, w, t, residual, residual, values, dense)) {
        error = Nonfinite;
        break;
      }
      for (int row = lane; row < m.n; row += 32)
        x[row] += residual[row];
      ++refinements;
      __syncwarp();
    }
  }
  if (lane == 0) {
    atomicAdd(t.counters, 1ULL);
    atomicAdd(t.counters + 1, refinements);
    atomicAdd(t.counters + 2, retries);
    if (error || !accepted)
      atomicCAS(t.error, 0, error ? error : Invalid);
  }
  if (!error && accepted) {
    for (int row = lane; row < m.n; row += 32)
      t.z[point * m.n + row] = x[row];
    for (int row = lane; row < t.rank; row += 32) {
      double value = x[t.active[row]];
      if (point == 0 && !correction)
        for (int col = 0; col < t.rank; ++col)
          value = fma(t.powers[row * t.rank + col], t.initial[t.active[col]],
                      value);
      t.scan_a[point * t.rank + row] = value;
    }
  }
}

__global__ void TimeScan(TimeData *data, int length, int level, bool input_a) {
  const int job = blockIdx.x / length, point = blockIdx.x % length;
  const auto t = data[job];
  const int stride = 1 << level;
  const double *input = input_a ? t.scan_a : t.scan_b;
  double *output = input_a ? t.scan_b : t.scan_a;
  __shared__ double before[64];
  if (threadIdx.x < t.rank)
    before[threadIdx.x] =
        point >= stride ? input[(point - stride) * t.rank + threadIdx.x] : 0;
  __syncthreads();
  for (int row = threadIdx.x; row < t.rank; row += blockDim.x) {
    double value = input[point * t.rank + row];
    if (point >= stride)
      for (int col = 0; col < t.rank; ++col)
        value = fma(t.powers[(level * t.rank + row) * t.rank + col],
                    before[col], value);
    output[point * t.rank + row] = value;
    if (!Bounded(value))
      atomicCAS(t.error, 0, Nonfinite);
  }
}
__global__ void TimeRecover(const Model *models, TimeData *data, int length,
                            bool input_a, bool correction) {
  const int job = blockIdx.x / length, point = blockIdx.x % length;
  const auto t = data[job];
  const int n = models[job].n;
  const double *boundary = input_a ? t.scan_a : t.scan_b;
  __shared__ double before[64];
  if (threadIdx.x < t.rank)
    before[threadIdx.x] =
        point > 0 ? boundary[(point - 1) * t.rank + threadIdx.x]
                  : (correction ? 0 : t.initial[t.active[threadIdx.x]]);
  __syncthreads();
  for (int row = threadIdx.x; row < n; row += blockDim.x) {
    double value = t.z[point * n + row];
    for (int col = 0; col < t.rank; ++col)
      value = fma(t.transfer[row * t.rank + col], before[col], value);
    if (correction)
      value += t.solution[point * n + row];
    t.solution[point * n + row] = value;
    if (!Bounded(value))
      atomicCAS(t.error, 0, Nonfinite);
  }
}
__global__ void TimeResidual(const Model *models, const Workspace *all,
                             TimeData *data, int length) {
  const int job = blockIdx.x / length, point = blockIdx.x % length;
  const auto m = models[job];
  const auto t = data[job];
  __shared__ Shared s;
  if (threadIdx.x == 0)
    s.error = 0;
  __syncthreads();
  const double *previous = point ? t.solution + (point - 1) * m.n : t.initial,
               *x = t.solution + point * m.n, *a = all[job].jacobian;
  double component = 0, normalized = 0, matrix_norm = 0, rhs_norm = 0,
         solution_norm = 0, nonzero = 0;
  for (int row = threadIdx.x; row < m.n; row += Threads) {
    Sum rhs;
    rhs.Add(t.rhs[point * m.n + row]);
    for (int j = t.mass_rows[row]; j < t.mass_rows[row + 1]; ++j)
      rhs.Product(t.mass_values[j] / t.step, previous[t.mass_columns[j]]);
    const double effective = rhs.Value();
    Sum residual;
    residual.Add(effective);
    double denominator = fabs(effective);
    for (int j = m.row_offsets[row]; j < m.row_offsets[row + 1]; ++j) {
      residual.Product(-a[j], x[m.columns[j]]);
      denominator += fabs(a[j] * x[m.columns[j]]);
    }
    const double value = residual.Value();
    t.residual[point * m.n + row] = value;
    if (!Bounded(value) || !Bounded(effective) || !Bounded(x[row]))
      Reject(s, Nonfinite);
    nonzero = PositiveMaximum(nonzero, value != 0 ? 1. : 0.);
    component = PositiveMaximum(component, denominator == 0
                                               ? (value == 0 ? 0 : 1e100)
                                               : fabs(value) / denominator);
    normalized = PositiveMaximum(normalized, fabs(value) / t.equil[row]);
    matrix_norm = PositiveMaximum(matrix_norm, t.row_norm[row]);
    rhs_norm = PositiveMaximum(rhs_norm, fabs(effective) / t.equil[row]);
    solution_norm = PositiveMaximum(solution_norm, fabs(x[row]));
  }
  double maxima[]{component, normalized,    matrix_norm,
                  rhs_norm,  solution_norm, nonzero};
  ValidationMaxima(maxima, s);
  if (threadIdx.x == 0) {
    const double denominator = maxima[2] * maxima[4] + maxima[3];
    if (s.error)
      atomicCAS(t.error, 0, s.error);
    if (maxima[0] > 1e-5 ||
        (denominator == 0 ? maxima[1] != 0 : maxima[1] / denominator > 1e-10))
      atomicExch(t.invalid, 1);
    if (maxima[5] != 0)
      atomicExch(t.nonzero, 1);
  }
}
__global__ void SequentialTime(const Model *models, const Workspace *all,
                               TimeData *data, int length) {
  const int job = blockIdx.x;
  const auto m = models[job];
  const auto t = data[job];
  __shared__ Shared s;
  extern __shared__ double time_storage[];
  InitializeTime(m, all + job, s, time_storage);
  auto &w = s.runtime;
  LoadTimeFactor(m, w, s, t);
  for (int i = threadIdx.x; i < m.nnz; i += Threads)
    w.jacobian[i] = all[job].jacobian[i];
  for (int row = threadIdx.x; row < m.n; row += Threads)
    w.current[row] = t.initial[row];
  __syncthreads();
  for (int point = 0; point < length && !s.error; ++point) {
    for (int row = threadIdx.x; row < m.n; row += Threads) {
      Sum rhs;
      rhs.Add(t.rhs[point * m.n + row]);
      for (int j = t.mass_rows[row]; j < t.mass_rows[row + 1]; ++j)
        rhs.Product(t.mass_values[j] / t.step, w.current[t.mass_columns[j]]);
      w.affine_rhs[row] = rhs.Value();
    }
    __syncthreads();
    CachedTimeLinear(m, w, s);
    for (int row = threadIdx.x; row < m.n && !s.error; row += Threads) {
      t.solution[point * m.n + row] = w.solution[row];
      w.current[row] = w.solution[row];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0 && s.error)
    atomicCAS(t.error, 0, s.error);
}
template <class T> T ProbeChecked(Result<T> result) {
  if (!result.ok())
    throw std::runtime_error(result.error().message);
  return result.TakeValue();
}
struct ProbeSample {
  std::string name;
  CsrMatrix a;
  std::vector<double> b, oracle;
};
std::vector<ProbeSample> ReadProbe(const char *path) {
  std::ifstream input(path);
  std::string magic;
  int n = 0, ports = 0, groups = 0, count = 0;
  input >> magic >> n >> ports >> groups >> count;
  if (!input || magic != "EMI03_SCHUR_REPLAY_1" || n != 185 || ports != 14 ||
      groups != 15 || count != 144)
    throw std::runtime_error("invalid frozen replay");
  int ignored;
  for (int i = 0; i < ports; ++i)
    input >> ignored;
  for (int g = 0; g < groups; ++g) {
    int size = 0;
    input >> size;
    if (size < 1 || size > 64)
      throw std::runtime_error("bad group");
    for (int i = 0; i < size; ++i)
      input >> ignored;
  }
  std::vector<ProbeSample> samples(count);
  for (auto &s : samples) {
    std::size_t nnz = 0;
    input >> s.name >> nnz;
    if (!input || nnz > static_cast<std::size_t>(n * n))
      throw std::runtime_error("bad matrix");
    s.a.rows = s.a.columns = n;
    s.a.row_offsets.resize(n + 1);
    s.a.column_indices.resize(nnz);
    s.a.values.resize(nnz);
    s.b.resize(n);
    s.oracle.resize(n);
    for (auto &x : s.a.row_offsets)
      input >> x;
    for (auto &x : s.a.column_indices)
      input >> x;
    for (auto &x : s.a.values)
      input >> x;
    for (auto &x : s.b)
      input >> x;
    for (auto &x : s.oracle)
      input >> x;
    if (!input)
      throw std::runtime_error("truncated matrix");
    ProbeChecked(ValidateSparseSolution(s.a, s.b, s.oracle));
  }
  return samples;
}
std::size_t ProbePrepare(const ProbeSample &sample, Allocations &allocation,
                         Model &m, Workspace &w) {
  const int n = sample.a.rows;
  MnaSystem system;
  system.g = sample.a;
  system.c.rows = system.c.columns = n;
  system.c.row_offsets.resize(n + 1);
  system.b_dc.resize(n);
  system.b_ac.resize(n);
  for (int i = 0; i < n; ++i)
    system.node_names.push_back("v" + std::to_string(i));
  TranAnalysis analysis{1, 1, 0, false};
  m.n = m.nodes = n;
  m.nnz = sample.a.values.size();
  PrepareFactorPlan(system, analysis, std::vector<double>(n), m, allocation);
  m.row_offsets = allocation.Upload(std::vector<int>(
      sample.a.row_offsets.begin(), sample.a.row_offsets.end()));
  m.columns = allocation.Upload(std::vector<int>(
      sample.a.column_indices.begin(), sample.a.column_indices.end()));
  w.state = allocation.Upload(std::vector<double>(n));
  w.affine_rhs = allocation.Upload(sample.b);
  w.jacobian = allocation.Upload(sample.a.values);
  w.solution = allocation.Allocate<double>(n);
  w.factored_jacobian = allocation.Allocate<double>(m.nnz);
  w.lu = allocation.Allocate<double>(n * N);
  w.permutation = allocation.Allocate<int>(n);
  w.last_dynamic_jacobian = nullptr;
  std::size_t bytes =
      ((m.factor_nonzeros + 17 * n + 2 * m.nnz) * sizeof(double) + 7) / 8 * 8 +
      (2 * m.factor_nonzeros + m.factor_levels + 1) * sizeof(int);
  bytes = (bytes + 7) / 8 * 8;
  m.shared_factor_metadata = true;
  m.shared_structure = true;
  bytes = (bytes + m.factor_nonzeros * sizeof(FactorEntry) +
           m.factor_terms * sizeof(FactorTerm) + 7) /
          8 * 8;
  bytes += (n + 1 + m.nnz) * sizeof(int);
  return bytes;
}
struct HostTime {
  ProbeSample sample;
  CsrMatrix mass;
  double step;
  std::vector<double> rhs, truth, oracle;
};
std::vector<HostTime> ReadTime(const char *replay, const char *mass_path,
                               std::vector<int> &active) {
  auto samples = ReadProbe(replay);
  std::ifstream in(mass_path);
  std::string magic;
  int count = 0, n = 0, rank = 0;
  in >> magic >> count >> n >> rank;
  if (!in || magic != "EMI03_TIME_MASS_1" || count != 9 || n != 185 ||
      rank < 1 || rank > 64)
    throw std::runtime_error("invalid time replay dimensions");
  active.resize(rank);
  for (int &x : active)
    in >> x;
  if (!std::is_sorted(active.begin(), active.end()) ||
      std::adjacent_find(active.begin(), active.end()) != active.end() ||
      active.front() < 0 || active.back() >= n)
    throw std::runtime_error("invalid time state selectors");
  std::vector<HostTime> cases(count);
  for (int job = 0; job < count; ++job) {
    auto &h = cases[job];
    h.sample = samples[job * 16 + 4];
    std::string name;
    int nnz = 0;
    in >> name >> h.step >> nnz;
    if (!in || name != h.sample.name || !std::isfinite(h.step) || h.step <= 0 ||
        nnz < 0 || nnz > n * n)
      throw std::runtime_error("invalid time mass record");
    auto &c = h.mass;
    c.rows = c.columns = n;
    c.row_offsets.resize(n + 1);
    c.column_indices.resize(nnz);
    c.values.resize(nnz);
    for (auto &x : c.row_offsets)
      in >> x;
    for (auto &x : c.column_indices)
      in >> x;
    for (auto &x : c.values)
      in >> x;
    if (!in)
      throw std::runtime_error("truncated time mass matrix");
    ProbeChecked(ValidateSparseSolution(c, std::vector<double>(n),
                                        std::vector<double>(n)));
    for (std::size_t j = 0; j < c.values.size(); ++j)
      if (c.values[j] != 0 &&
          !std::binary_search(active.begin(), active.end(),
                              static_cast<int>(c.column_indices[j])))
        throw std::runtime_error("mass coupling omitted by selectors");
    h.truth.resize(513 * n);
    h.rhs.resize(512 * n);
    for (int point = 0; point <= 512; ++point)
      for (int row = 0; row < n; ++row)
        h.truth[point * n + row] = .2 * std::sin(.031 * point + .047 * row);
    for (int point = 0; point < 512; ++point)
      for (int row = 0; row < n; ++row) {
        long double value = 0;
        for (auto j = h.sample.a.row_offsets[row];
             j < h.sample.a.row_offsets[row + 1]; ++j)
          value += static_cast<long double>(h.sample.a.values[j]) *
                   h.truth[(point + 1) * n + h.sample.a.column_indices[j]];
        for (auto j = c.row_offsets[row]; j < c.row_offsets[row + 1]; ++j)
          value -= static_cast<long double>(c.values[j] / h.step) *
                   h.truth[point * n + c.column_indices[j]];
        h.rhs[point * n + row] = static_cast<double>(value);
      }
  }
  std::string extra;
  if (in >> extra)
    throw std::runtime_error("time mass trailing data");
  return cases;
}
void ReadTimeOracle(const char *path, std::vector<HostTime> &cases) {
  std::ifstream input(path);
  std::string magic;
  int jobs = 0, n = 0, length = 0;
  input >> magic >> jobs >> n >> length;
  if (!input || magic != "EMI03_TIME_ORACLE_1" ||
      jobs != static_cast<int>(cases.size()) || n != 185 || length != 512)
    throw std::runtime_error("invalid independent CPU time oracle");
  for (auto &h : cases) {
    std::string name;
    double step = 0;
    input >> name >> step;
    if (!input || name != h.sample.name || step != h.step)
      throw std::runtime_error("CPU time oracle identity mismatch");
    h.oracle.resize(n * length);
    for (double &v : h.oracle) {
      input >> v;
      if (!input || !std::isfinite(v))
        throw std::runtime_error("invalid CPU time oracle state");
    }
  }
  std::string extra;
  if (input >> extra)
    throw std::runtime_error("CPU time oracle trailing data");
}
TimeData PrepareTimeData(const HostTime &h, const std::vector<int> &active,
                         const Model &m, Allocations &allocation, int *flags) {
  TimeData t{};
  t.length = 512;
  t.rank = active.size();
  t.active = allocation.Upload(active);
  t.mass_rows = allocation.Upload(
      std::vector<int>(h.mass.row_offsets.begin(), h.mass.row_offsets.end()));
  t.mass_columns = allocation.Upload(std::vector<int>(
      h.mass.column_indices.begin(), h.mass.column_indices.end()));
  t.mass_values = allocation.Upload(h.mass.values);
  t.step = h.step;
  t.factor = allocation.Allocate<double>(m.factor_nonzeros);
  t.inverse = allocation.Allocate<double>(m.n);
  t.equil = allocation.Allocate<double>(m.n);
  t.inverse_equil = allocation.Allocate<double>(m.n);
  t.row_norm = allocation.Allocate<double>(m.n);
  t.transfer = allocation.Allocate<double>(m.n * t.rank);
  t.transfer_certificate = allocation.Allocate<double>(2 * t.rank);
  t.counters = allocation.Upload(std::vector<unsigned long long>(3));
  t.powers = allocation.Allocate<double>(9 * t.rank * t.rank);
  t.rhs = allocation.Upload(h.rhs);
  t.initial = allocation.Upload(
      std::vector<double>(h.truth.begin(), h.truth.begin() + m.n));
  t.solution = allocation.Allocate<double>(512 * m.n);
  t.residual = allocation.Allocate<double>(512 * m.n);
  t.z = allocation.Allocate<double>(512 * m.n);
  t.scan_a = allocation.Allocate<double>(512 * t.rank);
  t.scan_b = allocation.Allocate<double>(512 * t.rank);
  t.error = flags;
  t.invalid = t.error + 1;
  t.nonzero = t.error + 2;
  return t;
}
void CheckTimeErrors(const char *stage, const std::vector<TimeData> &data,
                     Allocations &allocation) {
  int error = 0;
  allocation.Copy(&error, data.front().error, sizeof(int),
                  cudaMemcpyDeviceToHost);
  if (error)
    throw std::runtime_error(std::string(stage) + " time GPU failure " +
                             std::to_string(error));
}
void ClearTimeFlags(const std::vector<TimeData> &data, Allocations &allocation,
                    bool errors) {
  const auto &t = data.front();
  CheckCuda(cudaMemsetAsync(errors ? t.error : t.invalid, 0,
                            (errors ? 3 : 2) * sizeof(int),
                            allocation.stream()),
            "time flag clear");
}
void TimePass(const Model *models, const Workspace *workspace, TimeData *device,
              int jobs, int rank, int length, std::size_t bytes,
              Allocations &allocation, bool correction) {
  TimeWarpIndependent<<<jobs *((length + 7) / 8), Threads,
                        8 * 3 * 185 * sizeof(double), allocation.stream()>>>(
      models, workspace, device, length, correction);
  CheckCuda(cudaGetLastError(), "time independent launch");
  bool input_a = true;
  for (int level = 0; (1 << level) < length; ++level) {
    TimeScan<<<jobs * length, Threads, 0, allocation.stream()>>>(
        device, length, level, input_a);
    CheckCuda(cudaGetLastError(), "time prefix launch");
    input_a = !input_a;
  }
  TimeRecover<<<jobs * length, Threads, 0, allocation.stream()>>>(
      models, device, length, input_a, correction);
  CheckCuda(cudaGetLastError(), "time recovery launch");
  static_cast<void>(rank);
  static_cast<void>(bytes);
}
int SolveWindow(const Model *models, const Workspace *workspace,
                TimeData *device, const std::vector<TimeData> &data, int length,
                std::size_t bytes, Allocations &allocation) {
  ClearTimeFlags(data, allocation, true);
  TimePass(models, workspace, device, data.size(), data[0].rank, length, bytes,
           allocation, false);
  for (int refinement = 0; refinement <= 4; ++refinement) {
    ClearTimeFlags(data, allocation, false);
    TimeResidual<<<data.size() * length, Threads, 0, allocation.stream()>>>(
        models, workspace, device, length);
    CheckCuda(cudaGetLastError(), "time residual launch");
    bool invalid = false, nonzero = false;
    {
      const auto &t = data.front();
      int flags[3]{};
      allocation.Copy(flags, t.error, sizeof(flags), cudaMemcpyDeviceToHost);
      if (flags[0])
        throw std::runtime_error("time GPU residual failure " +
                                 std::to_string(flags[0]));
      invalid |= flags[1] != 0;
      nonzero |= flags[2] != 0;
    }
    if (!invalid && (refinement > 0 || !nonzero))
      return refinement;
    if (refinement < 4)
      TimePass(models, workspace, device, data.size(), data[0].rank, length,
               bytes, allocation, true);
  }
  throw std::runtime_error("time window refinement limit");
}
void CheckTimeResult(const std::vector<HostTime> &cases,
                     const std::vector<TimeData> &data, Allocations &allocation,
                     int length, const char *method) {
  double maximum = 0, difference = 0, oracle_difference = 0;
  std::size_t checked = 0;
  for (std::size_t job = 0; job < cases.size(); ++job) {
    const auto &h = cases[job];
    const int n = h.sample.a.rows;
    std::vector<double> x(length * n);
    allocation.Copy(x.data(), data[job].solution, x.size() * sizeof(double),
                    cudaMemcpyDeviceToHost);
    for (int point = 0; point < length; ++point) {
      const double *previous =
          point ? x.data() + (point - 1) * n : h.truth.data();
      std::vector<double> rhs(n),
          solution(x.begin() + point * n, x.begin() + (point + 1) * n);
      for (int row = 0; row < n; ++row) {
        long double value = h.rhs[point * n + row];
        for (auto j = h.mass.row_offsets[row]; j < h.mass.row_offsets[row + 1];
             ++j)
          value += static_cast<long double>(h.mass.values[j] / h.step) *
                   previous[h.mass.column_indices[j]];
        rhs[row] = static_cast<double>(value);
        difference =
            std::max(difference,
                     std::abs(solution[row] - h.truth[(point + 1) * n + row]));
        oracle_difference =
            std::max(oracle_difference,
                     std::abs(solution[row] - h.oracle[point * n + row]));
      }
      maximum = std::max(maximum, ProbeChecked(ValidateSparseSolution(
                                      h.sample.a, rhs, solution)));
      ++checked;
    }
  }
  if (difference > 2e-10 || oracle_difference > 2e-10)
    throw std::runtime_error("time manufactured-state differential mismatch " +
                             std::to_string(difference));
  std::cout << "{\"kind\":\"time_validation\",\"method\":\"" << method
            << "\",\"length\":" << length << ",\"original_systems\":" << checked
            << ",\"componentwise\":" << maximum
            << ",\"manufactured_difference\":" << difference
            << ",\"klu_difference\":" << oracle_difference << "}\n";
}
int TimeProbe(const char *replay, const char *mass, const char *oracle,
              bool validation_only) {
  std::vector<int> active;
  auto cases = ReadTime(replay, mass, active);
  ReadTimeOracle(oracle, cases);
  ProbeChecked(BeginEmi03CudaJob("time-window-probe"));
  {
    HostStaging staging;
    Allocations allocation(staging);
    std::vector<Model> models(cases.size());
    std::vector<Workspace> workspace(cases.size());
    std::vector<TimeData> data;
    std::size_t bytes = 0;
    int *flags = allocation.Upload(std::vector<int>(3));
    for (std::size_t i = 0; i < cases.size(); ++i) {
      bytes = std::max(bytes, ProbePrepare(cases[i].sample, allocation,
                                           models[i], workspace[i]));
      data.push_back(
          PrepareTimeData(cases[i], active, models[i], allocation, flags));
    }
    auto *dm = allocation.Upload(models);
    auto *dw = allocation.Upload(workspace);
    auto *dt = allocation.Upload(data);
    CheckCuda(cudaFuncSetAttribute(CacheTimeFactors,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   bytes),
              "time cache shared");
    CheckCuda(cudaFuncSetAttribute(MakeTimeTransfer,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   bytes),
              "time transfer shared");
    CheckCuda(cudaFuncSetAttribute(TimeIndependent,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   bytes),
              "time independent shared");
    CheckCuda(cudaFuncSetAttribute(SequentialTime,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   bytes),
              "time sequential shared");
    CacheTimeFactors<<<cases.size(), Threads, bytes, allocation.stream()>>>(
        dm, dw, dt);
    CheckCuda(cudaGetLastError(), "time factors launch");
    CheckTimeErrors("factor cache", data, allocation);
    MakeTimeTransfer<<<cases.size() * active.size(), Threads, bytes,
                       allocation.stream()>>>(dm, dw, dt, active.size());
    CheckCuda(cudaGetLastError(), "time transfer launch");
    CheckTimeErrors("transfer", data, allocation);
    for (std::size_t job = 0; job < data.size(); ++job) {
      std::vector<double> certificate(2 * active.size());
      allocation.Copy(certificate.data(), data[job].transfer_certificate,
                      certificate.size() * sizeof(double),
                      cudaMemcpyDeviceToHost);
      double norm = 0, component = 0;
      for (std::size_t i = 0; i < active.size(); ++i) {
        norm = std::max(norm, certificate[2 * i]);
        component = std::max(component, certificate[2 * i + 1]);
      }
      std::cout << "{\"kind\":\"internal_transfer\",\"case\":" << job
                << ",\"normwise\":" << norm
                << ",\"componentwise\":" << component << "}\n";
    }

    for (int level = 0; level < 9; ++level) {
      TimePower<<<dim3((active.size() * active.size() + Threads - 1) / Threads,
                       cases.size()),
                  Threads, 0, allocation.stream()>>>(dt, active.size(), level);
      CheckCuda(cudaGetLastError(), "time powers launch");
    }
    CheckTimeErrors("completed", data, allocation);
    cudaEvent_t begin, end;
    CheckCuda(cudaEventCreate(&begin), "time event");
    CheckCuda(cudaEventCreate(&end), "time event");
    for (int length : {8, 32, 128, 512})
      for (int repetition = -1; repetition < (validation_only ? 0 : 9);
           ++repetition)
        for (int order = 0; order < 2; ++order) {
          const bool parallel = ((repetition + 1 + order) % 2) != 0;
          const char *method = parallel ? "prefix" : "sequential";
          ClearTimeFlags(data, allocation, true);
          for (const auto &t : data)
            CheckCuda(cudaMemsetAsync(t.counters, 0,
                                      3 * sizeof(unsigned long long),
                                      allocation.stream()),
                      "time counter clear");
          CheckCuda(cudaEventRecord(begin, allocation.stream()), "time begin");
          int refinements = 0;
          if (parallel)
            refinements =
                SolveWindow(dm, dw, dt, data, length, bytes, allocation);
          else {
            SequentialTime<<<cases.size(), Threads, bytes,
                             allocation.stream()>>>(dm, dw, dt, length);
            CheckCuda(cudaGetLastError(), "time sequential launch");
          }
          CheckCuda(cudaEventRecord(end, allocation.stream()), "time end");
          CheckCuda(cudaEventSynchronize(end), "time completion");
          float ms = 0;
          CheckCuda(cudaEventElapsedTime(&ms, begin, end), "time elapsed");
          CheckTimeErrors("completed", data, allocation);
          CheckTimeResult(cases, data, allocation, length, method);
          unsigned long long totals[3]{};
          for (const auto &t : data) {
            unsigned long long counts[3]{};
            allocation.Copy(counts, t.counters, sizeof(counts),
                            cudaMemcpyDeviceToHost);
            for (int j = 0; j < 3; ++j)
              totals[j] += counts[j];
          }
          std::cout << "{\"kind\":\"warp_counts\",\"method\":\"" << method
                    << "\",\"length\":" << length << ",\"solves\":" << totals[0]
                    << ",\"refinements\":" << totals[1]
                    << ",\"dense_retries\":" << totals[2] << "}\n";

          std::cout << "{\"kind\":\"time_timing\",\"method\":\"" << method
                    << "\",\"length\":" << length
                    << ",\"count\":" << cases.size()
                    << ",\"repetition\":" << repetition
                    << ",\"device_ms\":" << ms
                    << ",\"window_refinements\":" << refinements << "}\n";
        }
    CheckCuda(cudaEventDestroy(begin), "time event destroy");
    CheckCuda(cudaEventDestroy(end), "time event destroy");
  }
  const auto ended = ProbeChecked(EndEmi03CudaJob());
  if (ended.outstanding_device_bytes || ended.cleanup_failures)
    throw std::runtime_error("time cleanup failure");
  return 0;
}
} // namespace
} // namespace ohmnivore
