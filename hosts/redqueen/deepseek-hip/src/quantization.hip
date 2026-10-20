#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <limits>
#include <cstddef>

namespace {

constexpr int kMaxGroupSize = 128;
constexpr unsigned int kGridXLimit = 65535U;
constexpr unsigned int kGridYLimit = 65535U;
constexpr float kFp8Max = 448.0f;
constexpr float kFp8MaxInverse = 1.0f / kFp8Max;

__device__ __forceinline__ float bf16_bits_to_float(std::uint16_t bits) {
  return __uint_as_float(static_cast<std::uint32_t>(bits) << 16);
}

__device__ __forceinline__ std::uint8_t e8m0_from_pow2_exponent(int exponent) {
  return static_cast<std::uint8_t>(exponent + 127);
}

__device__ __forceinline__ float e4m3_to_float(std::uint8_t value) {
  const float sign = (value & 0x80U) != 0U ? -1.0f : 1.0f;
  const int exponent = (value >> 3U) & 0x0fU;
  const int mantissa = value & 0x07U;
  if (exponent == 0) {
    return sign * static_cast<float>(mantissa) * 0x1.0p-9f;
  }
  if (exponent == 0x0f && mantissa == 0x07U) {
    return __uint_as_float(0x7fc00000U);
  }
  return sign * ldexpf(1.0f + static_cast<float>(mantissa) * 0.125f, exponent - 7);
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

template <bool Inplace>
__global__ void quantize_activation_kernel(
    std::uint16_t* input_bf16,
    std::uint8_t* output_e4m3,
    std::uint8_t* output_e8m0,
    int cols,
    int group_size,
    std::size_t group_count) {
  __shared__ float values[kMaxGroupSize];
  __shared__ float abs_values[kMaxGroupSize];
  __shared__ float scale;
  const int lane = threadIdx.x;
  const std::size_t group = static_cast<std::size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
  if (group >= group_count) {
    return;
  }
  const std::size_t group_count_per_row = static_cast<std::size_t>(cols / group_size);
  const std::size_t row = group / group_count_per_row;
  const std::size_t group_in_row = group % group_count_per_row;
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
    if constexpr (!Inplace) {
      output_e8m0[group] = e8m0_from_pow2_exponent(exponent);
    }
  }
  __syncthreads();

  const float normalized = fminf(fmaxf(values[lane] / scale, -kFp8Max), kFp8Max);
  const std::uint8_t quantized = __hip_cvt_float_to_fp8(normalized, __HIP_SATFINITE, __HIP_E4M3);
  if constexpr (Inplace) {
    input_bf16[offset] = float_to_bf16_rne(e4m3_to_float(quantized) * scale);
  } else {
    output_e4m3[offset] = quantized;
  }
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

template <bool Inplace>
int launch_quantize_activation(
    const void* input_bf16,
    void* output_e4m3,
    void* output_e8m0,
    int rows,
    int cols,
    int group_size,
    void* stream_handle) {
  const bool invalid_output = Inplace
                                  ? output_e4m3 != nullptr || output_e8m0 != nullptr
                                  : output_e4m3 == nullptr || output_e8m0 == nullptr;
  if (input_bf16 == nullptr || invalid_output || rows <= 0 || cols <= 0 ||
      (group_size != 32 && group_size != 128) || cols % group_size != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const std::int64_t groups = static_cast<std::int64_t>(rows) * (cols / group_size);
  // Keep the grid dimension within HIP's portable signed x-dimension limit.
  if (groups > std::numeric_limits<int>::max()) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const unsigned int grid_x = static_cast<unsigned int>(
      groups < kGridXLimit ? groups : kGridXLimit);
  const std::int64_t grid_y = (groups - 1) / grid_x + 1;
  if (grid_y > kGridYLimit) {
    return static_cast<int>(hipErrorInvalidConfiguration);
  }
  const auto stream = reinterpret_cast<hipStream_t>(stream_handle);
  hipLaunchKernelGGL(
      (quantize_activation_kernel<Inplace>),
      dim3(grid_x, static_cast<unsigned int>(grid_y)),
      dim3(static_cast<unsigned int>(group_size)),
      0,
      stream,
      const_cast<std::uint16_t*>(static_cast<const std::uint16_t*>(input_bf16)),
      static_cast<std::uint8_t*>(output_e4m3),
      static_cast<std::uint8_t*>(output_e8m0),
      cols,
      group_size,
      static_cast<std::size_t>(groups));
  return static_cast<int>(hipGetLastError());
}

extern "C" int redqueen_ds4_quantize_activation(
    const void* input_bf16,
    void* output_e4m3,
    void* output_e8m0,
    int rows,
    int cols,
    int group_size,
    void* stream_handle) {
  if (output_e4m3 == nullptr || output_e8m0 == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  return launch_quantize_activation<false>(
      input_bf16, output_e4m3, output_e8m0, rows, cols, group_size, stream_handle);
}

extern "C" int redqueen_ds4_quantize_dequantize_activation_bf16(
    void* input_bf16,
    int rows,
    int cols,
    int group_size,
    void* stream_handle) {
  return launch_quantize_activation<true>(
      input_bf16, nullptr, nullptr, rows, cols, group_size, stream_handle);
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
