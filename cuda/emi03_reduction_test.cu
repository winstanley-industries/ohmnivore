#include "cuda/emi03_reduction.cuh"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {
__global__ void Reduce(const double *input, double *output) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  output[index] = ohmnivore::emi03_device::WarpMagnitudeMaximum(input[index]);
}
bool Check(cudaError_t error) {
  if (error == cudaSuccess)
    return true;
  std::cerr << cudaGetErrorString(error) << '\n';
  return false;
}
struct Buffer {
  double *data = nullptr;
  ~Buffer() {
    if (data)
      cudaFree(data);
  }
};
} // namespace

int main() {
  constexpr int cases = 1024, lanes = 32;
  std::vector<double> input(cases * lanes), output(input.size());
  std::uint64_t random = 0x935d7025a8924b13ULL;
  for (auto &value : input) {
    random ^= random << 13;
    random ^= random >> 7;
    random ^= random << 17;
    value = std::bit_cast<double>(random);
  }
  for (int lane = 0; lane < lanes; ++lane) {
    // All NaNs, subnormals, ties in the high word, and a high-word winner
    // whose low word is zero. Each checks a different lexicographic boundary.
    input[lane] = std::bit_cast<double>(0x7ff8000000000000ULL + lane);
    input[lanes + lane] = std::bit_cast<double>(std::uint64_t(lane));
    input[2 * lanes + lane] =
        std::bit_cast<double>(0x3ff0000000000000ULL + lane);
    input[3 * lanes + lane] = std::bit_cast<double>(
        lane == 0 ? 0x3ff0000100000000ULL : 0x3ff00000ffffffffULL);
    input[4 * lanes + lane] = std::bit_cast<double>(
        lane == 31 ? 0xfff0000000000000ULL : 0x7ff8000000000000ULL);
    input[5 * lanes + lane] =
        std::bit_cast<double>(lane & 1 ? 0x8000000000000000ULL : 0ULL);
  }
  Buffer device_input, device_output;
  const auto bytes = input.size() * sizeof(double);
  if (!Check(cudaMalloc(&device_input.data, bytes)) ||
      !Check(cudaMalloc(&device_output.data, bytes)) ||
      !Check(cudaMemcpy(device_input.data, input.data(), bytes,
                        cudaMemcpyHostToDevice)))
    return 1;
  Reduce<<<cases / 8, 256>>>(device_input.data, device_output.data);
  if (!Check(cudaGetLastError()) ||
      !Check(cudaMemcpy(output.data(), device_output.data, bytes,
                        cudaMemcpyDeviceToHost)))
    return 1;
  for (int group = 0; group < cases; ++group) {
    double expected = 0;
    for (int lane = 0; lane < lanes; ++lane) {
      const double value = input[group * lanes + lane];
      // Explicitly ignore signaling as well as quiet NaNs. Host fmax may
      // propagate a signaling NaN and thereby discard an earlier maximum.
      if (!std::isnan(value))
        expected = std::max(expected, std::abs(value));
    }
    for (int lane = 0; lane < lanes; ++lane)
      if (std::bit_cast<std::uint64_t>(output[group * lanes + lane]) !=
          std::bit_cast<std::uint64_t>(expected)) {
        std::cerr << "FP64 magnitude maximum mismatch at " << group << '/'
                  << lane << '\n';
        return 1;
      }
  }
  std::cout
      << "1024 magnitude groups match the host FP64 oracle in all lanes\n";
}
