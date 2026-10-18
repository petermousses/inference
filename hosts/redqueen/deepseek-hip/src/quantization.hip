#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <limits>
#include <cstddef>

namespace {

constexpr int kMaxGroupSize = 128;
constexpr float kFp8Max = 448.0f;
constexpr float kFp8MaxInverse = 1.0f / kFp8Max;

__device__ __forceinline__ float bf16_bits_to_float(std::uint16_t bits) {
  return __uint_as_float(static_cast<std::uint32_t>(bits) << 16);
}

__device__ __forceinline__ std::uint8_t e8m0_from_pow2_exponent(int exponent) {
  return static_cast<std::uint8_t>(exponent + 127);
}

__global__ void quantize_activation_kernel(
    const std::uint16_t* input_bf16,
    std::uint8_t* output_e4m3,
    std::uint8_t* output_e8m0,
    int cols,
    int group_size) {
  __shared__ float values[kMaxGroupSize];
  __shared__ float abs_values[kMaxGroupSize];
  __shared__ float scale;
  const int lane = threadIdx.x;
  const unsigned int group = blockIdx.x;
  const unsigned int group_count_per_row = static_cast<unsigned int>(cols / group_size);
  const unsigned int row = group / group_count_per_row;
  const unsigned int group_in_row = group % group_count_per_row;
  const std::size_t offset = static_cast<std::size_t>(row) * static_cast<std::size_t>(cols) +
                             static_cast<std::size_t>(group_in_row) * static_cast<std::size_t>(group_size) +
                             static_cast<std::size_t>(lane);

  const float value = bf16_bits_to_float(input_bf16[offset]);
  values[lane] = value;
  abs_values[lane] = fabsf(value);
  __syncthreads();

  for (int stride = group_size / 2; stride > 0; stride >>= 1) {
    if (lane < stride) {
      abs_values[lane] = fmaxf(abs_values[lane], abs_values[lane + stride]);
    }
    __syncthreads();
  }

  if (lane == 0) {
    const float amax = fmaxf(abs_values[0], 1.0e-4f);
    const float normalized_max = amax * kFp8MaxInverse;
    const std::uint32_t bits = __float_as_uint(normalized_max);
    int exponent = static_cast<int>((bits >> 23) & 0xffU) - 127;
    if ((bits & 0x7fffffU) != 0) {
      ++exponent;
    }
    scale = __uint_as_float(static_cast<std::uint32_t>(exponent + 127) << 23);
    output_e8m0[group] = e8m0_from_pow2_exponent(exponent);
  }
  __syncthreads();

  const float normalized = fminf(fmaxf(values[lane] / scale, -kFp8Max), kFp8Max);
  output_e4m3[offset] = __hip_cvt_float_to_fp8(normalized, __HIP_SATFINITE, __HIP_E4M3);
}

__device__ __forceinline__ float e8m0_to_float(std::uint8_t value) {
  if (value == 0xffU) {
    // Match ROCm's E8M0 conversion helper's canonical float NaN.
    return __uint_as_float(0x7fff0000U);
  }
  if (value == 0U) {
    // E8M0's minimum finite value is 2^-127, represented as a float subnormal.
    return __uint_as_float(0x00400000U);
  }
  return __uint_as_float(static_cast<std::uint32_t>(value) << 23);
}

__global__ void decode_e8m0_kernel(const std::uint8_t* input, float* output, int count) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = e8m0_to_float(input[index]);
  }
}

}  // namespace

extern "C" int redqueen_ds4_quantize_activation(
    const void* input_bf16,
    void* output_e4m3,
    void* output_e8m0,
    int rows,
    int cols,
    int group_size,
    void* stream_handle) {
  if (input_bf16 == nullptr || output_e4m3 == nullptr || output_e8m0 == nullptr || rows <= 0 || cols <= 0 ||
      (group_size != 32 && group_size != 128) || cols % group_size != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const std::int64_t groups = static_cast<std::int64_t>(rows) * (cols / group_size);
  // Keep the grid dimension within HIP's portable signed x-dimension limit.
  if (groups > std::numeric_limits<int>::max()) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const auto stream = reinterpret_cast<hipStream_t>(stream_handle);
  hipLaunchKernelGGL(
      quantize_activation_kernel,
      dim3(static_cast<unsigned int>(groups)),
      dim3(static_cast<unsigned int>(group_size)),
      0,
      stream,
      static_cast<const std::uint16_t*>(input_bf16),
      static_cast<std::uint8_t*>(output_e4m3),
      static_cast<std::uint8_t*>(output_e8m0),
      cols,
      group_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int redqueen_ds4_decode_e8m0(
    const void* input_e8m0,
    void* output_float32,
    int count,
    void* stream_handle) {
  if (input_e8m0 == nullptr || output_float32 == nullptr || count <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  const unsigned int blocks = static_cast<unsigned int>((count - 1) / threads + 1);
  const auto stream = reinterpret_cast<hipStream_t>(stream_handle);
  hipLaunchKernelGGL(
      decode_e8m0_kernel,
      dim3(blocks),
      dim3(threads),
      0,
      stream,
      static_cast<const std::uint8_t*>(input_e8m0),
      static_cast<float*>(output_float32),
      count);
  return static_cast<int>(hipGetLastError());
}
