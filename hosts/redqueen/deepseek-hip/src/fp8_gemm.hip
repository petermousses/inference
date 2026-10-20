#include "fp8_gemm.h"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>

namespace {

constexpr int kThreads = 256;
constexpr int kTileM = 16;
constexpr int kTileN = 16;
constexpr unsigned int kGridXLimit = 65535;
constexpr unsigned int kGridYLimit = 65535;

__device__ __forceinline__ float e4m3_to_float(std::uint8_t value) {
  const float sign = (value & 0x80U) != 0U ? -1.0f : 1.0f;
  const int exponent = (value >> 3U) & 0x0fU;
  const int mantissa = value & 0x07U;
  if (exponent == 0) {
    return sign * static_cast<float>(mantissa) * 0x1.0p-9f;
  }
  if (exponent == 0x0f && mantissa == 0x07) {
    return __uint_as_float(0x7fc00000U);
  }
  const float significand = 1.0f + static_cast<float>(mantissa) * 0.125f;
  return sign * ldexpf(significand, exponent - 7);
}

__device__ __forceinline__ float e8m0_to_float(std::uint8_t value) {
  if (value == 0xffU) {
    return __uint_as_float(0x7fff0000U);
  }
  if (value == 0U) {
    return __uint_as_float(0x00400000U);
  }
  return __uint_as_float(static_cast<std::uint32_t>(value) << 23U);
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

template <typename Output>
__device__ __forceinline__ void store_output(Output* output, std::size_t index, float value);

template <>
__device__ __forceinline__ void store_output<float>(float* output, std::size_t index, float value) {
  output[index] = value;
}

template <>
__device__ __forceinline__ void store_output<std::uint16_t>(
    std::uint16_t* output,
    std::size_t index,
    float value) {
  output[index] = float_to_bf16_rne(value);
}

template <int GroupSize, typename Output>
__global__ void grouped_fp8_gemm_kernel(
    const std::uint8_t* a,
    const std::uint8_t* scale_a,
    const std::uint8_t* b,
    const std::uint8_t* scale_b,
    Output* output,
    int m,
    int n,
    int k) {
  const std::size_t block_index =
      static_cast<std::size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
  const std::size_t output_index = block_index * blockDim.x + threadIdx.x;
  const std::size_t output_count = static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
  if (output_index >= output_count) {
    return;
  }

  const int row = static_cast<int>(output_index / static_cast<std::size_t>(n));
  const int column = static_cast<int>(output_index % static_cast<std::size_t>(n));
  const int k_groups = k / GroupSize;
  float accumulated = 0.0f;

  // Keep the pinned per-group accumulation order explicit in this correctness baseline.
  for (int group = 0; group < k_groups; ++group) {
    float partial = 0.0f;
    const std::size_t a_group_offset = static_cast<std::size_t>(row) * k +
                                       static_cast<std::size_t>(group) * GroupSize;
    const std::size_t b_group_offset = static_cast<std::size_t>(column) * k +
                                       static_cast<std::size_t>(group) * GroupSize;
    #pragma unroll
    for (int element = 0; element < GroupSize; ++element) {
      const float a_value = e4m3_to_float(a[a_group_offset + element]);
      const float b_value = e4m3_to_float(b[b_group_offset + element]);
      partial = partial + a_value * b_value;
    }

    const std::size_t scale_a_index = static_cast<std::size_t>(row) * k_groups + group;
    const std::size_t scale_b_index = static_cast<std::size_t>(column / GroupSize) * k_groups + group;
    const float scaled = (partial * e8m0_to_float(scale_a[scale_a_index])) *
                         e8m0_to_float(scale_b[scale_b_index]);
    accumulated = accumulated + scaled;
  }

  store_output(output, output_index, accumulated);
}

template <int GroupSize, typename Output>
__global__ void tiled_fp8_gemm_kernel(
    const std::uint8_t* a,
    const std::uint8_t* scale_a,
    const std::uint8_t* b,
    const std::uint8_t* scale_b,
    Output* output,
    int m,
    int n,
    int k,
    std::size_t tile_count_n,
    std::size_t tile_count_m) {
  __shared__ float a_tile[kTileM][GroupSize];
  __shared__ float b_tile[kTileN][GroupSize];

  const std::size_t tile_index =
      static_cast<std::size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
  const std::size_t tile_m = tile_index / tile_count_n;
  const std::size_t tile_n = tile_index % tile_count_n;
  if (tile_m >= tile_count_m) {
    return;
  }
  const std::size_t row_base = tile_m * kTileM;
  const std::size_t column_base = tile_n * kTileN;
  const int k_groups = k / GroupSize;
  const int local_row = static_cast<int>(threadIdx.x) / kTileN;
  const int local_column = static_cast<int>(threadIdx.x) % kTileN;
  const std::size_t global_row = row_base + static_cast<std::size_t>(local_row);
  const std::size_t global_column = column_base + static_cast<std::size_t>(local_column);
  const bool valid_output = global_row < static_cast<std::size_t>(m) &&
                            global_column < static_cast<std::size_t>(n);
  float accumulated = 0.0f;

  for (int group = 0; group < k_groups; ++group) {
    for (int index = static_cast<int>(threadIdx.x); index < kTileM * GroupSize; index += kThreads) {
      const int row = index / GroupSize;
      const int element = index % GroupSize;
      const std::size_t global_row = row_base + static_cast<std::size_t>(row);
      a_tile[row][element] = global_row < static_cast<std::size_t>(m)
          ? e4m3_to_float(a[global_row * static_cast<std::size_t>(k) +
                            static_cast<std::size_t>(group) * GroupSize + element])
          : 0.0f;
    }
    for (int index = static_cast<int>(threadIdx.x); index < kTileN * GroupSize; index += kThreads) {
      const int column = index / GroupSize;
      const int element = index % GroupSize;
      const std::size_t global_column = column_base + static_cast<std::size_t>(column);
      b_tile[column][element] = global_column < static_cast<std::size_t>(n)
          ? e4m3_to_float(b[global_column * static_cast<std::size_t>(k) +
                            static_cast<std::size_t>(group) * GroupSize + element])
          : 0.0f;
    }
    __syncthreads();

    if (valid_output) {
      float partial = 0.0f;
      for (int element = 0; element < GroupSize; ++element) {
        partial = partial + a_tile[local_row][element] * b_tile[local_column][element];
      }
      const std::size_t scale_a_index = global_row * k_groups + group;
      const std::size_t scale_b_index =
          (global_column / GroupSize) * k_groups + static_cast<std::size_t>(group);
      const float scaled = (partial * e8m0_to_float(scale_a[scale_a_index])) *
                           e8m0_to_float(scale_b[scale_b_index]);
      accumulated = accumulated + scaled;
    }
    __syncthreads();
  }
  if (valid_output) {
    const std::size_t output_index = global_row * static_cast<std::size_t>(n) + global_column;
    store_output(output, output_index, accumulated);
  }
}

template <int GroupSize, typename Output>
int launch_grouped_fp8_gemm(
    const std::uint8_t* a,
    const std::uint8_t* scale_a,
    const std::uint8_t* b,
    const std::uint8_t* scale_b,
    void* output,
    int m,
    int n,
    int k,
    hipStream_t stream,
    std::int64_t output_count) {
  const std::int64_t block_count = (output_count - 1) / kThreads + 1;
  const unsigned int blocks_x = static_cast<unsigned int>(
      block_count < kGridXLimit ? block_count : kGridXLimit);
  const std::int64_t blocks_y = (block_count - 1) / blocks_x + 1;
  if (blocks_y > kGridYLimit) {
    return static_cast<int>(hipErrorInvalidConfiguration);
  }
  hipLaunchKernelGGL(
      (grouped_fp8_gemm_kernel<GroupSize, Output>),
      dim3(blocks_x, static_cast<unsigned int>(blocks_y)),
      dim3(kThreads),
      0,
      stream,
      a,
      scale_a,
      b,
      scale_b,
      static_cast<Output*>(output),
      m,
      n,
      k);
  return static_cast<int>(hipGetLastError());
}

template <int GroupSize, typename Output>
int launch_tiled_fp8_gemm(
    const std::uint8_t* a,
    const std::uint8_t* scale_a,
    const std::uint8_t* b,
    const std::uint8_t* scale_b,
    void* output,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  const std::size_t tile_count_m =
      (static_cast<std::size_t>(m) + kTileM - 1U) / kTileM;
  const std::size_t tile_count_n =
      (static_cast<std::size_t>(n) + kTileN - 1U) / kTileN;
  const std::size_t tile_count = tile_count_m * tile_count_n;
  const unsigned int grid_x = static_cast<unsigned int>(
      tile_count < kGridXLimit ? tile_count : kGridXLimit);
  const std::size_t grid_y_size = (tile_count - 1U) / grid_x + 1U;
  if (grid_y_size > kGridYLimit) {
    return static_cast<int>(hipErrorInvalidConfiguration);
  }
  hipLaunchKernelGGL(
      (tiled_fp8_gemm_kernel<GroupSize, Output>),
      dim3(grid_x, static_cast<unsigned int>(grid_y_size)),
      dim3(kThreads),
      0,
      stream,
      a,
      scale_a,
      b,
      scale_b,
      static_cast<Output*>(output),
      m,
      n,
      k,
      tile_count_n,
      tile_count_m);
  return static_cast<int>(hipGetLastError());
}

}  // namespace

extern "C" int redqueen_ds4_fp8_gemm(
    const void* a_e4m3,
    const void* scale_a_e8m0,
    const void* b_e4m3,
    const void* scale_b_e8m0,
    void* output,
    int m,
    int n,
    int k,
    int group_size,
    int output_type,
    void* stream_handle) {
  if (a_e4m3 == nullptr || scale_a_e8m0 == nullptr || b_e4m3 == nullptr || scale_b_e8m0 == nullptr ||
      output == nullptr || m <= 0 || n <= 0 || k <= 0 || (group_size != 32 && group_size != 128) ||
      k % group_size != 0 || (output_type != REDQUEEN_DS4_FP8_GEMM_OUTPUT_F32 &&
                               output_type != REDQUEEN_DS4_FP8_GEMM_OUTPUT_BF16)) {
    return static_cast<int>(hipErrorInvalidValue);
  }

  const std::int64_t output_count = static_cast<std::int64_t>(m) * static_cast<std::int64_t>(n);
  const std::int64_t block_count = (output_count - 1) / kThreads + 1;
  if (output_count <= 0 || block_count > std::numeric_limits<int>::max()) {
    return static_cast<int>(hipErrorInvalidValue);
  }

  const auto* a = static_cast<const std::uint8_t*>(a_e4m3);
  const auto* scale_a = static_cast<const std::uint8_t*>(scale_a_e8m0);
  const auto* b = static_cast<const std::uint8_t*>(b_e4m3);
  const auto* scale_b = static_cast<const std::uint8_t*>(scale_b_e8m0);
  const auto stream = reinterpret_cast<hipStream_t>(stream_handle);
  if (output_type == REDQUEEN_DS4_FP8_GEMM_OUTPUT_F32) {
    if (group_size == 32) {
      return launch_grouped_fp8_gemm<32, float>(
          a, scale_a, b, scale_b, output, m, n, k, stream, output_count);
    }
    return launch_grouped_fp8_gemm<128, float>(
        a, scale_a, b, scale_b, output, m, n, k, stream, output_count);
  }
  if (group_size == 32) {
    return launch_grouped_fp8_gemm<32, std::uint16_t>(
        a, scale_a, b, scale_b, output, m, n, k, stream, output_count);
  }
  return launch_grouped_fp8_gemm<128, std::uint16_t>(
      a, scale_a, b, scale_b, output, m, n, k, stream, output_count);
}

extern "C" int redqueen_ds4_fp8_gemm_tiled(
    const void* a_e4m3,
    const void* scale_a_e8m0,
    const void* b_e4m3,
    const void* scale_b_e8m0,
    void* output,
    int m,
    int n,
    int k,
    int group_size,
    int output_type,
    void* stream_handle) {
  if (a_e4m3 == nullptr || scale_a_e8m0 == nullptr || b_e4m3 == nullptr || scale_b_e8m0 == nullptr ||
      output == nullptr || m <= 0 || n <= 0 || k <= 0 || (group_size != 32 && group_size != 128) ||
      k % group_size != 0 || (output_type != REDQUEEN_DS4_FP8_GEMM_OUTPUT_F32 &&
                               output_type != REDQUEEN_DS4_FP8_GEMM_OUTPUT_BF16)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const auto pending_error = hipGetLastError();
  if (pending_error != hipSuccess && pending_error != hipErrorNotReady) {
    return static_cast<int>(pending_error);
  }
  const auto* a = static_cast<const std::uint8_t*>(a_e4m3);
  const auto* scale_a = static_cast<const std::uint8_t*>(scale_a_e8m0);
  const auto* b = static_cast<const std::uint8_t*>(b_e4m3);
  const auto* scale_b = static_cast<const std::uint8_t*>(scale_b_e8m0);
  const auto stream = reinterpret_cast<hipStream_t>(stream_handle);
  if (output_type == REDQUEEN_DS4_FP8_GEMM_OUTPUT_F32) {
    if (group_size == 32) {
      return launch_tiled_fp8_gemm<32, float>(a, scale_a, b, scale_b, output, m, n, k, stream);
    }
    return launch_tiled_fp8_gemm<128, float>(a, scale_a, b, scale_b, output, m, n, k, stream);
  }
  if (group_size == 32) {
    return launch_tiled_fp8_gemm<32, std::uint16_t>(a, scale_a, b, scale_b, output, m, n, k, stream);
  }
  return launch_tiled_fp8_gemm<128, std::uint16_t>(a, scale_a, b, scale_b, output, m, n, k, stream);
}
