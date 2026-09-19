#ifndef OHMNIVORE_CUDA_EMI03_REDUCTION_CUH_
#define OHMNIVORE_CUDA_EMI03_REDUCTION_CUH_

#include <cuda_runtime.h>

namespace ohmnivore::emi03_device {
// Exact maximum of IEEE-754 magnitudes, with NaNs ignored and a zero floor.
// Every lane must participate. Integer reductions compare the high word first,
// then the low word only among lanes sharing that high word. No FP32 conversion
// or truncation of the FP64 mantissa occurs.
__device__ inline double WarpMagnitudeMaximum(double value) {
  auto bits = static_cast<unsigned long long>(__double_as_longlong(value)) &
              0x7fffffffffffffffULL;
  if (bits > 0x7ff0000000000000ULL)
    bits = 0;
  const auto high = static_cast<unsigned>(bits >> 32);
  const auto maximum_high = __reduce_max_sync(0xffffffff, high);
  const auto low = high == maximum_high ? static_cast<unsigned>(bits) : 0U;
  const auto maximum_low = __reduce_max_sync(0xffffffff, low);
  return __longlong_as_double(
      (static_cast<unsigned long long>(maximum_high) << 32) | maximum_low);
}
} // namespace ohmnivore::emi03_device

#endif
