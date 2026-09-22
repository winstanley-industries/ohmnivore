// Isolated algebraic decomposition experiment. Never linked into a simulator.
#include "ohmnivore/solver.h"
#include <algorithm>
#include <cmath>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int N = 256, G = 16, L = 64, P = 32;
struct Plan {
  int n = 0, ports = 0, groups = 0, storage = 0;
  int port[P]{}, size[G]{}, index[G][L]{}, offset[G]{}, rhs_offset[G]{};
  int coupling_count[G]{}, coupling_port[G][P]{};
  const int *row_offsets = nullptr, *columns = nullptr;
  const int *interface_offsets = nullptr, *interface_columns = nullptr;
};
struct Work {
  double a[N * N], b[N], x[N], scale[N], q[N], response[N * P];
  double interface_solution[P];
  int error, failed_group, failed_pivot;
  double failed_best;
  unsigned long long cycles[3];
};
struct Sample {
  std::string name;
  ohmnivore::CsrMatrix a;
  std::vector<double> b, oracle;
};
void Check(cudaError_t code) {
  if (code != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(code));
}
template <class T> T Checked(ohmnivore::Result<T> result) {
  if (!result.ok())
    throw std::runtime_error(result.error().message);
  return result.TakeValue();
}
// One warp owns one interior. Numeric row pivots never cross the interface.
// A failed local pivot is a partition failure, not proof that full A is
// singular.
__device__ bool Factor(double *lu, int n, int *permutation, Work &w, int group,
                       unsigned long long *masks) {
  const int lane = threadIdx.x & 31;
  for (int i = lane; i < n; i += 32)
    permutation[i] = i;
  __syncwarp();
  for (int k = 0; k < n; ++k) {
    double best = 0;
    int selected = n;
    for (int i = k + lane; i < n; i += 32) {
      const double value = fabs(lu[i * n + k]);
      if (value > best) {
        best = value;
        selected = i;
      }
    }
    for (int off = 16; off; off /= 2) {
      const double other = __shfl_down_sync(0xffffffff, best, off);
      const int row = __shfl_down_sync(0xffffffff, selected, off);
      if (lane + off < 32 &&
          (other > best || (other == best && row < selected))) {
        best = other;
        selected = row;
      }
    }
    best = __shfl_sync(0xffffffff, best, 0);
    selected = __shfl_sync(0xffffffff, selected, 0);
    if (!(best >= 2.2250738585072014e-308) || !isfinite(best)) {
      if (lane == 0 && atomicCAS(&w.error, 0, 1) == 0) {
        w.failed_group = group;
        w.failed_pivot = k;
        w.failed_best = best;
      }
      return false;
    }
    if (selected != k) {
      for (int j = lane; j < n; j += 32) {
        const double temp = lu[k * n + j];
        lu[k * n + j] = lu[selected * n + j];
        lu[selected * n + j] = temp;
      }
      if (lane == 0) {
        const int temp = permutation[k];
        permutation[k] = permutation[selected];
        permutation[selected] = temp;
      }
    }
    __syncwarp();
    int *rows = permutation + n, *columns = permutation + 2 * n;
    int row_count = 0, column_count = 0;
    for (int base = 0; base < n; base += 32) {
      const int index = base + lane;
      const bool lower = index > k && index < n && lu[index * n + k] != 0;
      const bool upper = index > k && index < n && lu[k * n + index] != 0;
      const auto row_mask = __ballot_sync(0xffffffff, lower);
      const auto column_mask = __ballot_sync(0xffffffff, upper);
      const unsigned below = (1U << lane) - 1;
      if (lower)
        rows[row_count + __popc(row_mask & below)] = index;
      if (upper)
        columns[column_count + __popc(column_mask & below)] = index;
      row_count += __popc(row_mask);
      column_count += __popc(column_mask);
    }
    __syncwarp();
    for (int at = lane; at < row_count; at += 32)
      lu[rows[at] * n + k] /= lu[k * n + k];
    __syncwarp();
    for (int at = lane; at < row_count * column_count; at += 32) {
      const int row = rows[at / column_count], col = columns[at % column_count];
      lu[row * n + col] =
          fma(-lu[row * n + k], lu[k * n + col], lu[row * n + col]);
    }
    __syncwarp();
  }
  for (int row = lane; row < n; row += 32) {
    unsigned long long lower = 0, upper = 0;
    for (int col = 0; col < row; ++col)
      if (lu[row * n + col] != 0)
        lower |= 1ULL << col;
    for (int col = row + 1; col < n; ++col)
      if (lu[row * n + col] != 0)
        upper |= 1ULL << col;
    masks[row] = lower;
    masks[n + row] = upper;
  }
  __syncwarp();
  return true;
}
// Reuse only the row ordering. Rebuild the symbolic fill from this actual
// matrix; a failed pivot returns to the bounded local partial-pivot factor.
__device__ bool OrderedFactor(double *lu, int n, unsigned long long *masks) {
  const int lane = threadIdx.x & 31;
  for (int row = lane; row < n; row += 32) {
    unsigned long long pattern = 0;
    for (int col = 0; col < n; ++col)
      if (lu[row * n + col] != 0)
        pattern |= 1ULL << col;
    masks[row] = pattern;
  }
  __syncwarp();
  for (int k = 0; k < n; ++k) {
    const auto upper = k == 63 ? 0ULL : masks[k] & (~0ULL << (k + 1));
    for (int row = k + 1 + lane; row < n; row += 32)
      if (masks[row] & (1ULL << k))
        masks[row] |= upper;
    __syncwarp();
  }
  for (int row = lane; row < n; row += 32) {
    masks[n + row] = row == 63 ? 0ULL : masks[row] & (~0ULL << (row + 1));
    masks[row] &= (1ULL << row) - 1;
  }
  __syncwarp();
  for (int k = 0; k < n; ++k) {
    bool bad = false;
    for (int col = k + lane; col < n; col += 32) {
      double value = lu[k * n + col];
      auto active = masks[k];
      while (active) {
        const int j = __ffsll(active) - 1;
        value = fma(-lu[k * n + j], lu[j * n + col], value);
        active &= active - 1;
      }
      lu[k * n + col] = value;
      bad |= !isfinite(value);
    }
    __syncwarp();
    if (__any_sync(0xffffffff, bad) ||
        !(fabs(lu[k * n + k]) >= 2.2250738585072014e-308))
      return false;
    for (int row = k + 1 + lane; row < n; row += 32) {
      if (!(masks[row] & (1ULL << k)))
        continue;
      double value = lu[row * n + k];
      auto active = masks[row] & ((1ULL << k) - 1);
      while (active) {
        const int j = __ffsll(active) - 1;
        value = fma(-lu[row * n + j], lu[j * n + k], value);
        active &= active - 1;
      }
      lu[row * n + k] = value / lu[k * n + k];
      bad |= !isfinite(lu[row * n + k]);
    }
    __syncwarp();
    if (__any_sync(0xffffffff, bad))
      return false;
  }
  return true;
}
// All lanes cooperate across rows and RHS columns. Avoid serial per-column
// dot products with only one to four active lanes on the FP64 pipeline.
__device__ void Solve(const double *lu, int n, double *values, int count) {
  const int lane = threadIdx.x & 31;
  for (int k = 0; k < n; ++k) {
    for (int at = lane; at < (n - k - 1) * count; at += 32) {
      const int row = k + 1 + at / count, column = at % count;
      const double coefficient = lu[row * n + k];
      if (coefficient != 0)
        values[column * n + row] =
            fma(-coefficient, values[column * n + k], values[column * n + row]);
    }
    __syncwarp();
  }
  for (int k = n - 1; k >= 0; --k) {
    for (int column = lane; column < count; column += 32)
      values[column * n + k] /= lu[k * n + k];
    __syncwarp();
    for (int at = lane; at < k * count; at += 32) {
      const int row = at / count, column = at % count;
      const double coefficient = lu[row * n + k];
      if (coefficient != 0)
        values[column * n + row] =
            fma(-coefficient, values[column * n + k], values[column * n + row]);
    }
    __syncwarp();
  }
}
__device__ double Rhs(const Plan &p, const Work &w, int row, bool correction) {
  if (!correction)
    return w.b[row];
  // Compensated native-FP64 original-matrix residual, including product error.
  double hi = w.b[row], lo = 0;
  for (int at = p.row_offsets[row]; at < p.row_offsets[row + 1]; ++at) {
    const int col = p.columns[at];
    const double a = -w.a[row * N + col], x = w.x[col];
    const double product = a * x, total = hi + product;
    const double displacement = total - hi;
    lo += (hi - (total - displacement)) + (product - displacement) +
          fma(a, x, -product);
    hi = total;
  }
  return hi + lo;
}
__device__ void Interior(const Plan &p, Work &w, int group, double *storage,
                         int *permutations, unsigned long long *all_masks,
                         bool correction, bool ordered) {
  const int lane = threadIdx.x & 31, n = p.size[group];
  double *lu = storage + p.offset[group];
  int *permutation = permutations + group * L * 3;
  auto *masks = all_masks + group * L * 2;
  if (!correction) {
    for (int row = lane; row < n; row += 32) {
      const int original = p.index[group][row];
      double scale = 0;
      for (int at = p.row_offsets[original]; at < p.row_offsets[original + 1];
           ++at)
        scale = fmax(scale, fabs(w.a[original * N + p.columns[at]]));
      if (!(scale > 0) || !isfinite(scale))
        atomicExch(&w.error, 2);
      w.scale[original] = scale;
    }
    __syncwarp();
    for (int attempt = 0; attempt < 2; ++attempt) {
      const bool reuse = ordered && attempt == 0;
      for (int at = lane; at < n * n; at += 32) {
        const int row = p.index[group][reuse ? permutation[at / n] : at / n];
        const double value = w.a[row * N + p.index[group][at % n]];
        lu[at] = value / w.scale[row];
        if (!isfinite(lu[at]) || (value != 0 && lu[at] == 0))
          atomicExch(&w.error, 2);
      }
      __syncwarp();
      if (reuse) {
        if (OrderedFactor(lu, n, masks))
          break;
      } else {
        if (!Factor(lu, n, permutation, w, group, masks))
          return;
        break;
      }
    }
  }
  const int coupled = p.coupling_count[group];
  const int count = correction ? 1 : coupled + 1;
  double *values = storage + p.rhs_offset[group];
  for (int at = lane; at < count * n; at += 32) {
    const int column = at / n, row = at % n;
    const int original = p.index[group][permutation[row]];
    values[at] =
        (column == count - 1
             ? Rhs(p, w, original, correction)
             : w.a[original * N + p.port[p.coupling_port[group][column]]]) /
        w.scale[original];
  }
  __syncwarp();
  Solve(lu, n, values, count);
  for (int at = lane; at < count * n; at += 32) {
    const int column = at / n, row = at % n;
    const int original = p.index[group][row];
    if (column == count - 1)
      w.q[original] = values[at];
    else
      w.response[original * P + p.coupling_port[group][column]] = values[at];
    if (!isfinite(values[at]))
      atomicExch(&w.error, 2);
  }
}
__device__ void Interface(const Plan &p, Work &w, double *lu, int *permutation,
                          unsigned long long *masks, bool correction) {
  const int lane = threadIdx.x & 31;
  if (!correction) {
    for (int row = lane; row < p.ports; row += 32) {
      const int original = p.port[row];
      double scale = 0;
      for (int at = p.row_offsets[original]; at < p.row_offsets[original + 1];
           ++at)
        scale = fmax(scale, fabs(w.a[original * N + p.columns[at]]));
      if (!(scale > 0) || !isfinite(scale))
        atomicExch(&w.error, 2);
      w.scale[original] = scale;
    }
    __syncwarp();
    for (int at = lane; at < p.ports * p.ports; at += 32) {
      const int original = p.port[at / p.ports], col = at % p.ports;
      const double scale = w.scale[original];
      double value = w.a[original * N + p.port[col]] / scale;
      for (int j = p.interface_offsets[at / p.ports];
           j < p.interface_offsets[at / p.ports + 1]; ++j) {
        const int k = p.interface_columns[j];
        value =
            fma(-w.a[original * N + k] / scale, w.response[k * P + col], value);
      }
      lu[at] = value;
    }
    __syncwarp();
    if (!Factor(lu, p.ports, permutation, w, -1, masks))
      return;
  }
  double *values = lu + p.ports * p.ports;
  for (int row = lane; row < p.ports; row += 32) {
    const int original = p.port[permutation[row]];
    const double scale = w.scale[original];
    double value = Rhs(p, w, original, correction) / scale;
    for (int j = p.interface_offsets[permutation[row]];
         j < p.interface_offsets[permutation[row] + 1]; ++j) {
      const int k = p.interface_columns[j];
      value = fma(-w.a[original * N + k] / scale, w.q[k], value);
    }
    values[row] = value;
  }
  __syncwarp();
  Solve(lu, p.ports, values, 1);
  for (int row = lane; row < p.ports; row += 32)
    w.interface_solution[row] = values[row];
}
template <bool Cluster>
__global__ void Replay(Plan p, Work *all, int iterations) {
  const auto cluster = cooperative_groups::this_cluster();
  const int rank = Cluster ? cluster.block_rank() : 0;
  const int blocks = Cluster ? 4 : 1;
  const int job = blockIdx.x / blocks;
  Work &w = all[job];
  extern __shared__ double storage[];
  __shared__ int permutations[G * L * 3 + P * 3];
  __shared__ unsigned long long masks[G * L * 2 + P * 2];
  const int warp = threadIdx.x / 32, warps = blockDim.x / 32;
  const auto barrier = [&] {
    if constexpr (Cluster)
      cluster.sync();
    else
      __syncthreads();
  };
  if (rank == 0 && threadIdx.x == 0)
    for (int stage = 0; stage < 3; ++stage)
      w.cycles[stage] = 0;
  for (int iteration = 0; iteration < iterations; ++iteration) {
    if (rank == 0 && threadIdx.x == 0)
      w.error = 0;
    barrier();
    // Both paths execute a full factorization even when b is exactly zero.
    // One mandatory original-matrix correction is included in this probe.
    for (int pass = 0; pass < 2; ++pass) {
      auto start = clock64();
      for (int group = rank + blocks * warp; group < p.groups;
           group += blocks * warps)
        Interior(p, w, group, storage, permutations, masks, pass != 0,
                 iteration > 0);
      barrier();
      if (rank == 0 && threadIdx.x == 0)
        w.cycles[0] += clock64() - start;
      if (w.error)
        return;
      start = clock64();
      if (rank == 0 && warp == 0)
        Interface(p, w, storage + p.storage, permutations + G * L * 3,
                  masks + G * L * 2, pass != 0);
      barrier();
      if (rank == 0 && threadIdx.x == 0)
        w.cycles[1] += clock64() - start;
      if (w.error)
        return;
      start = clock64();
      for (int group = rank + blocks * warp; group < p.groups;
           group += blocks * warps) {
        for (int at = threadIdx.x & 31; at < p.size[group]; at += 32) {
          const int row = p.index[group][at];
          double value = w.q[row];
          for (int at = 0; at < p.coupling_count[group]; ++at) {
            const int col = p.coupling_port[group][at];
            value = fma(-w.response[row * P + col], w.interface_solution[col],
                        value);
          }
          w.x[row] = pass ? w.x[row] + value : value;
          if (!isfinite(w.x[row]))
            atomicExch(&w.error, 2);
        }
      }
      if (rank == 0)
        for (int at = threadIdx.x; at < p.ports; at += blockDim.x) {
          const int row = p.port[at];
          const double value = w.interface_solution[at];
          w.x[row] = pass ? w.x[row] + value : value;
          if (!isfinite(w.x[row]))
            atomicExch(&w.error, 2);
        }
      barrier();
      if (rank == 0 && threadIdx.x == 0)
        w.cycles[2] += clock64() - start;
    }
  }
}
void Launch(const Plan &p, Work *device, int count, int iterations,
            bool cluster, cudaStream_t stream) {
  const auto bytes = (p.storage + p.ports * p.ports + p.ports) * sizeof(double);
  if (cluster) {
    cudaLaunchConfig_t launch{};
    launch.gridDim = count * 4;
    launch.blockDim = 128;
    launch.dynamicSmemBytes = bytes;
    launch.stream = stream;
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim = {4, 1, 1};
    launch.attrs = &attribute;
    launch.numAttrs = 1;
    Check(cudaLaunchKernelEx(&launch, Replay<true>, p, device, iterations));
  } else {
    Replay<false><<<count, 512, bytes, stream>>>(p, device, iterations);
    Check(cudaGetLastError());
  }
}
struct Structure {
  std::vector<int *> device;
  const int *Upload(const std::vector<int> &values) {
    int *pointer = nullptr;
    Check(cudaMalloc(&pointer,
                     std::max<std::size_t>(1, values.size()) * sizeof(int)));
    Check(cudaMemcpy(pointer, values.data(), values.size() * sizeof(int),
                     cudaMemcpyHostToDevice));
    // The immutable preparation path has no outstanding numerical kernels.
    Check(cudaDeviceSynchronize());
    device.push_back(pointer);
    return pointer;
  }
  void Prepare(Plan &p, const std::vector<Sample> &samples) {
    std::vector<std::set<int>> pattern(p.n);
    for (const auto &sample : samples)
      for (int row = 0; row < p.n; ++row)
        for (auto at = sample.a.row_offsets[row];
             at < sample.a.row_offsets[row + 1]; ++at)
          pattern[row].insert(sample.a.column_indices[at]);
    std::vector<int> offsets{0}, columns;
    for (const auto &row : pattern) {
      columns.insert(columns.end(), row.begin(), row.end());
      offsets.push_back(columns.size());
    }
    p.row_offsets = Upload(offsets);
    p.columns = Upload(columns);
    std::set<int> ports(p.port, p.port + p.ports);
    offsets = {0};
    columns.clear();
    for (int row = 0; row < p.ports; ++row) {
      for (int col : pattern[p.port[row]])
        if (!ports.contains(col))
          columns.push_back(col);
      offsets.push_back(columns.size());
    }
    p.interface_offsets = Upload(offsets);
    p.interface_columns = Upload(columns);
    p.storage = 0;
    for (int group = 0; group < p.groups; ++group) {
      std::set<int> coupled;
      for (int i = 0; i < p.size[group]; ++i)
        for (int port = 0; port < p.ports; ++port)
          if (pattern[p.index[group][i]].contains(p.port[port]))
            coupled.insert(port);
      p.coupling_count[group] = coupled.size();
      std::copy(coupled.begin(), coupled.end(), p.coupling_port[group]);
      p.offset[group] = p.storage;
      p.storage += p.size[group] * p.size[group];
      p.rhs_offset[group] = p.storage;
      p.storage += p.size[group] * (coupled.size() + 1);
    }
  }
  ~Structure() {
    for (int *p : device)
      cudaFree(p);
  }
};
void Configure(const Plan &p) {
  const auto bytes = (p.storage + p.ports * p.ports + p.ports) * sizeof(double);
  Check(cudaFuncSetAttribute(
      Replay<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  Check(cudaFuncSetAttribute(
      Replay<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
}
std::vector<Work> Pack(const Plan &p, const std::vector<Sample> &samples) {
  std::vector<int> owner(p.n, -2);
  for (int i = 0; i < p.ports; ++i) {
    if (p.port[i] < 0 || p.port[i] >= p.n || owner[p.port[i]] != -2)
      throw std::runtime_error("invalid interface coverage");
    owner[p.port[i]] = -1;
  }
  for (int g = 0; g < p.groups; ++g)
    for (int j = 0; j < p.size[g]; ++j) {
      const int i = p.index[g][j];
      if (i < 0 || i >= p.n || owner[i] != -2)
        throw std::runtime_error("invalid interior coverage");
      owner[i] = g;
    }
  if (std::find(owner.begin(), owner.end(), -2) != owner.end())
    throw std::runtime_error("incomplete partition");
  std::vector<Work> packed(samples.size());
  for (std::size_t s = 0; s < samples.size(); ++s) {
    const auto &sample = samples[s];
    for (int i = 0; i < p.n; ++i) {
      packed[s].b[i] = sample.b[i];
      for (auto k = sample.a.row_offsets[i]; k < sample.a.row_offsets[i + 1];
           ++k) {
        const auto j = sample.a.column_indices[k];
        const double value = sample.a.values[k];
        if (!std::isfinite(value) || (value != 0 && owner[i] >= 0 &&
                                      owner[j] >= 0 && owner[i] != owner[j]))
          throw std::runtime_error(
              "nonfinite matrix or cross-interior coupling");
        packed[s].a[i * N + j] = value;
      }
    }
  }
  return packed;
}
void Validate(const Plan &p, const std::vector<Sample> &samples, bool cluster,
              cudaStream_t stream, bool zero) {
  auto packed = Pack(p, samples);
  if (zero)
    for (auto &w : packed)
      std::fill(std::begin(w.b), std::end(w.b), 0);
  Work *device = nullptr;
  // Upload at most sixteen independent matrices in any launch.
  Check(cudaMalloc(&device, 16 * sizeof(Work)));
  for (std::size_t start = 0; start < samples.size(); start += 16) {
    const int count = std::min<std::size_t>(16, samples.size() - start);
    Check(cudaMemcpyAsync(device, packed.data() + start, count * sizeof(Work),
                          cudaMemcpyHostToDevice, stream));
    Launch(p, device, count, 2, cluster, stream);
    Check(cudaStreamSynchronize(stream));
    Check(cudaMemcpy(packed.data() + start, device, count * sizeof(Work),
                     cudaMemcpyDeviceToHost));
    for (int i = 0; i < count; ++i) {
      const auto &sample = samples[start + i];
      const auto &work = packed[start + i];
      if (work.error)
        throw std::runtime_error(sample.name + " GPU partition failure " +
                                 std::to_string(work.error) +
                                 " group=" + std::to_string(work.failed_group) +
                                 " pivot=" + std::to_string(work.failed_pivot) +
                                 " best=" + std::to_string(work.failed_best));
      const std::vector<double> result(work.x, work.x + p.n);
      const std::vector<double> rhs(work.b, work.b + p.n);
      const double residual =
          Checked(ohmnivore::ValidateSparseSolution(sample.a, rhs, result));
      auto factor =
          Checked(ohmnivore::SparseRealFactorization::Analyze(sample.a));
      const auto oracle = Checked(factor->FactorAndSolveRefined(sample.a, rhs));
      double difference = 0, norm = 1;
      for (int row = 0; row < p.n; ++row) {
        difference = std::max(difference, std::abs(oracle[row] - result[row]));
        norm = std::max(norm, std::abs(oracle[row]));
      }
      if (difference / norm > 2e-10)
        throw std::runtime_error(sample.name + " KLU differential failure");
      std::cout << "{\"kind\":\"validation\",\"cluster\":" << cluster
                << ",\"zero_rhs\":" << zero << ",\"case\":\"" << sample.name
                << "\",\"componentwise\":" << residual
                << ",\"relative_klu_difference\":" << difference / norm
                << "}\n";
    }
  }
  Check(cudaFree(device));
}
void Hostile(bool cluster, cudaStream_t stream) {
  Plan p;
  p.n = 2;
  p.ports = p.groups = 1;
  p.port[0] = 1;
  p.size[0] = 1;
  p.index[0][0] = 0;
  p.storage = 1;
  Sample hostile;
  hostile.a.rows = hostile.a.columns = 2;
  hostile.a.row_offsets = {0, 2, 4};
  hostile.a.column_indices = {0, 1, 0, 1};
  hostile.a.values = {1, 1, 1, 1};
  Structure structure;
  structure.Prepare(p, {hostile});
  Configure(p);
  Work *device = nullptr;
  Check(cudaMalloc(&device, sizeof(Work)));
  for (int fixture = 0; fixture < 2; ++fixture) {
    Work work{};
    // A=[[0,1],[1,0]] is nonsingular but D is singular. Reject the partition.
    // A=[[1,1],[1,1]] has nonsingular D but singular S. Zero RHS cannot hide
    // it.
    work.a[1] = work.a[N] = 1;
    work.a[0] = work.a[N + 1] = fixture;
    Check(cudaMemcpyAsync(device, &work, sizeof(work), cudaMemcpyHostToDevice,
                          stream));
    Launch(p, device, 1, 1, cluster, stream);
    Check(cudaStreamSynchronize(stream));
    Check(cudaMemcpy(&work, device, sizeof(work), cudaMemcpyDeviceToHost));
    if (work.error != 1)
      throw std::runtime_error("hostile singularity was not rejected");
    std::cout << "{\"kind\":\"hostile\",\"cluster\":" << cluster
              << ",\"fixture\":" << fixture
              << ",\"partition_rejected\":true}\n";
  }
  Check(cudaFree(device));
}
void Benchmark(const Plan &p, const std::vector<Sample> &samples,
               cudaStream_t stream) {
  auto all = Pack(p, samples);
  std::vector<Work> batch;
  for (int i = 0; i < 16; ++i)
    batch.push_back(all[(i % 9) * 16 + 4]);
  Work *device = nullptr;
  Check(cudaMalloc(&device, batch.size() * sizeof(Work)));
  Check(cudaMemcpyAsync(device, batch.data(), batch.size() * sizeof(Work),
                        cudaMemcpyHostToDevice, stream));
  cudaEvent_t begin, end;
  Check(cudaEventCreate(&begin));
  Check(cudaEventCreate(&end));
  constexpr int iterations = 100;
  for (int count : {1, 9, 16})
    for (int repetition = -1; repetition < 9; ++repetition)
      for (int order = 0; order < 2; ++order) {
        const bool cluster = ((repetition + 1 + order) % 2) != 0;
        Check(cudaEventRecord(begin, stream));
        Launch(p, device, count, iterations, cluster, stream);
        Check(cudaEventRecord(end, stream));
        Check(cudaEventSynchronize(end));
        float milliseconds = 0;
        Check(cudaEventElapsedTime(&milliseconds, begin, end));
        std::vector<Work> output(count);
        Check(cudaMemcpy(output.data(), device, count * sizeof(Work),
                         cudaMemcpyDeviceToHost));
        for (int i = 0; i < count; ++i) {
          if (output[i].error)
            throw std::runtime_error("timed matrix solve failed");
          const auto &sample = samples[(i % 9) * 16 + 4];
          Checked(ohmnivore::ValidateSparseSolution(
              sample.a, sample.b,
              std::vector<double>(output[i].x, output[i].x + p.n)));
        }
        std::cout << "{\"kind\":\"timing\",\"cluster\":" << cluster
                  << ",\"count\":" << count << ",\"repetition\":" << repetition
                  << ",\"iterations\":" << iterations
                  << ",\"device_ms\":" << milliseconds << ",\"cycles\":["
                  << output[0].cycles[0] << "," << output[0].cycles[1] << ","
                  << output[0].cycles[2] << "]}\n";
      }
  Check(cudaEventDestroy(begin));
  Check(cudaEventDestroy(end));
  Check(cudaFree(device));
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 2 || argc > 3)
      throw std::runtime_error(
          "usage: emi03_schur_probe REPLAY [--validate-only]");
    std::ifstream input(argv[1]);
    std::string magic;
    input >> magic;
    if (magic != "EMI03_SCHUR_REPLAY_1")
      throw std::runtime_error("invalid replay header");
    Plan p;
    int count = 0;
    if (!(input >> p.n >> p.ports >> p.groups >> count))
      throw std::runtime_error("truncated replay dimensions");
    if (p.n < 2 || p.n > N || p.ports < 1 || p.ports > P || p.groups < 1 ||
        p.groups > G || count != 144)
      throw std::runtime_error("invalid bounded replay dimensions");
    for (int i = 0; i < p.ports; ++i)
      input >> p.port[i];
    for (int g = 0; g < p.groups; ++g) {
      input >> p.size[g];
      if (p.size[g] < 1 || p.size[g] > L)
        throw std::runtime_error("invalid local dimension");
      p.offset[g] = p.storage;
      p.storage += p.size[g] * p.size[g];
      for (int j = 0; j < p.size[g]; ++j)
        input >> p.index[g][j];
    }
    std::vector<Sample> samples(count);
    for (auto &sample : samples) {
      std::size_t nnz = 0;
      if (!(input >> sample.name >> nnz))
        throw std::runtime_error("truncated matrix header");
      if (nnz > static_cast<std::size_t>(p.n * p.n))
        throw std::runtime_error("invalid nonzero count");
      sample.a.rows = sample.a.columns = p.n;
      sample.a.row_offsets.resize(p.n + 1);
      sample.a.column_indices.resize(nnz);
      sample.a.values.resize(nnz);
      sample.b.resize(p.n);
      sample.oracle.resize(p.n);
      for (auto &x : sample.a.row_offsets)
        input >> x;
      for (auto &x : sample.a.column_indices)
        input >> x;
      for (auto &x : sample.a.values)
        input >> x;
      for (auto &x : sample.b)
        input >> x;
      for (auto &x : sample.oracle)
        input >> x;
      if (!input)
        throw std::runtime_error("truncated replay");
      // Validate CSR structure and oracle before any indexing or device launch.
      Checked(
          ohmnivore::ValidateSparseSolution(sample.a, sample.b, sample.oracle));
    }
    std::string trailing;
    if (input >> trailing)
      throw std::runtime_error("unexpected replay trailer");
    Structure structure;
    structure.Prepare(p, samples);
    std::cout << std::setprecision(17);
    cudaStream_t stream;
    Check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (bool cluster : {false, true}) {
      Configure(p);
      for (bool zero : {false, true})
        Validate(p, samples, cluster, stream, zero);
      Hostile(cluster, stream);
    }
    Configure(p);
    if (argc == 2)
      Benchmark(p, samples, stream);
    else if (std::string(argv[2]) != "--validate-only")
      throw std::runtime_error("unknown option");
    Check(cudaStreamDestroy(stream));
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
