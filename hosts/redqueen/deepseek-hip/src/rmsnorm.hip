#include "rmsnorm.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace {

constexpr int kThreads = 256;

enum Dtype : int {
  kF32 = REDQUEEN_DS4_RMSNORM_DTYPE_F32,
  kBf16 = REDQUEEN_DS4_RMSNORM_DTYPE_BF16,
};

__device__ __forceinline__ float load_f32(const void* pointer, std::size_t index) {
  return static_cast<const float*>(pointer)[index];
}

__device__ __forceinline__ float load_bf16(const void* pointer, std::size_t index) {
  const auto bits = static_cast<const std::uint16_t*>(pointer)[index];
  return __uint_as_float(static_cast<std::uint32_t>(bits) << 16U);
}

__device__ __forceinline__ std::uint16_t float_to_bf16_rne(float value) {
  std::uint32_t bits = __float_as_uint(value);
  const std::uint32_t exponent = bits & 0x7f800000U;
  const std::uint32_t mantissa = bits & 0x007fffffU;
  if (exponent == 0x7f800000U && mantissa != 0U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

template <int Dtype>
__device__ __forceinline__ float load_as_f32(const void* pointer, std::size_t index);

template <>
__device__ __forceinline__ float load_as_f32<kF32>(const void* pointer, std::size_t index) {
  return load_f32(pointer, index);
}

template <>
__device__ __forceinline__ float load_as_f32<kBf16>(const void* pointer, std::size_t index) {
  return load_bf16(pointer, index);
}

template <int InputDtype, int WeightDtype, typename Output>
__global__ void rmsnorm_kernel(
    const void* input,
    const void* weight,
    Output* output,
    int dim,
    float eps) {
  const int row = static_cast<int>(blockIdx.x);
  const std::size_t row_start = static_cast<std::size_t>(row) * static_cast<std::size_t>(dim);
  float thread_sum = 0.0f;

  for (std::size_t column = threadIdx.x; column < static_cast<std::size_t>(dim); column += blockDim.x) {
    const float value = load_as_f32<InputDtype>(input, row_start + column);
    thread_sum = thread_sum + value * value;
  }

  __shared__ float partials[kThreads];
  partials[threadIdx.x] = thread_sum;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partials[threadIdx.x] = partials[threadIdx.x] + partials[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float inverse_rms = rsqrtf(partials[0] / static_cast<float>(dim) + eps);
  for (std::size_t column = threadIdx.x; column < static_cast<std::size_t>(dim); column += blockDim.x) {
    const std::size_t index = row_start + column;
    const float value = load_as_f32<InputDtype>(input, index);
    const float scale = load_as_f32<WeightDtype>(weight, column);
    const float result = (value * inverse_rms) * scale;
    if constexpr (std::is_same<Output, float>::value) {
      output[index] = result;
    } else {
      output[index] = float_to_bf16_rne(result);
    }
  }
}

template <int InputDtype, int WeightDtype, typename Output>
int launch_rmsnorm(
    const void* input,
    const void* weight,
    void* output,
    int rows,
    int dim,
    float eps,
    hipStream_t stream) {
  hipLaunchKernelGGL(
      (rmsnorm_kernel<InputDtype, WeightDtype, Output>),
      dim3(static_cast<unsigned int>(rows)),
      dim3(kThreads),
      0,
      stream,
      input,
      weight,
      static_cast<Output*>(output),
      dim,
      eps);
  return static_cast<int>(hipGetLastError());
}

template <int InputDtype, int WeightDtype>
int launch_output(
    const void* input,
    const void* weight,
    void* output,
    int rows,
    int dim,
    float eps,
    hipStream_t stream) {
  if constexpr (InputDtype == kF32) {
    return launch_rmsnorm<InputDtype, WeightDtype, float>(input, weight, output, rows, dim, eps, stream);
  } else {
    return launch_rmsnorm<InputDtype, WeightDtype, std::uint16_t>(input, weight, output, rows, dim, eps, stream);
  }
}

int launch_weight_dtype(
    const void* input,
    const void* weight,
    void* output,
    int rows,
    int dim,
    int input_dtype,
    int weight_dtype,
    float eps,
    hipStream_t stream) {
  if (input_dtype == kF32) {
    return weight_dtype == kF32
               ? launch_output<kF32, kF32>(input, weight, output, rows, dim, eps, stream)
               : launch_output<kF32, kBf16>(input, weight, output, rows, dim, eps, stream);
  }
  return weight_dtype == kF32
             ? launch_output<kBf16, kF32>(input, weight, output, rows, dim, eps, stream)
             : launch_output<kBf16, kBf16>(input, weight, output, rows, dim, eps, stream);
}

}  // namespace

extern "C" int redqueen_ds4_rmsnorm(
    const void* input,
    const void* weight,
    void* output,
    int rows,
    int dim,
    int input_dtype,
    int weight_dtype,
    float eps,
    void* stream_handle) {
  if (input == nullptr || weight == nullptr || output == nullptr || rows <= 0 || dim <= 0 ||
      (input_dtype != kF32 && input_dtype != kBf16) ||
      (weight_dtype != kF32 && weight_dtype != kBf16) || !std::isfinite(eps) || eps <= 0.0f) {
    return static_cast<int>(hipErrorInvalidValue);
  }

  const std::int64_t row_count = rows;
  const std::int64_t width = dim;
  const std::int64_t element_count = row_count * width;
  if (element_count <= 0 ||
      element_count > std::numeric_limits<std::ptrdiff_t>::max() / sizeof(float)) {
    return static_cast<int>(hipErrorInvalidValue);
  }

  return launch_weight_dtype(
      input,
      weight,
      output,
      rows,
      dim,
      input_dtype,
      weight_dtype,
      eps,
      reinterpret_cast<hipStream_t>(stream_handle));
}
