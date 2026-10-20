#include "fp4_gemm.h"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

constexpr int kThreads = 256;
constexpr int kTileM = 16;
constexpr int kTileN = 16;
constexpr int kWeightGroupSize = 32;
constexpr unsigned int kGridXLimit = 65535U;
constexpr unsigned int kGridYLimit = 65535U;

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

__device__ __forceinline__ float e2m1_to_float(std::uint8_t value) {
  const float sign = (value & 0x08U) != 0U ? -1.0f : 1.0f;
  const int exponent = (value >> 1U) & 0x03U;
  const int mantissa = value & 0x01U;
  if (exponent == 0 && mantissa == 0) {
    return 0.0f;
  }
  const float magnitude = exponent == 0
                              ? static_cast<float>(mantissa) * 0.5f
                              : ldexpf(1.0f + static_cast<float>(mantissa) * 0.5f, exponent - 1);
  return sign * magnitude;
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

template <typename Scale>
__device__ __forceinline__ float load_scale(const Scale* scales, std::size_t index);

template <>
__device__ __forceinline__ float load_scale<std::uint8_t>(const std::uint8_t* scales, std::size_t index) {
  return e8m0_to_float(scales[index]);
}

template <>
__device__ __forceinline__ float load_scale<float>(const float* scales, std::size_t index) {
  return scales[index];
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

template <typename Scale, typename Output>
__global__ void fp4_gemm_kernel(
    const std::uint8_t* a,
    const Scale* scale_a,
    const std::uint8_t* b,
    const Scale* scale_b,
    Output* output,
    int m,
    int n,
    int k,
    int activation_group_size,
    std::size_t output_count) {
  const std::size_t block_index = static_cast<std::size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
  const std::size_t output_index = block_index * blockDim.x + threadIdx.x;
  if (output_index >= output_count) {
    return;
  }

  const int row = static_cast<int>(output_index / static_cast<std::size_t>(n));
  const int column = static_cast<int>(output_index % static_cast<std::size_t>(n));
  const int weight_group_count = k / kWeightGroupSize;
  const int activation_groups_per_weight_group = activation_group_size / kWeightGroupSize;
  float accumulated = 0.0f;

  // Match the pinned kernel: one dot per 32 K values, scaled before group accumulation.
  for (int group = 0; group < weight_group_count; ++group) {
    float partial = 0.0f;
    const std::size_t a_group_offset = static_cast<std::size_t>(row) * static_cast<std::size_t>(k) +
                                       static_cast<std::size_t>(group) * kWeightGroupSize;
    const std::size_t b_group_offset = static_cast<std::size_t>(column) * static_cast<std::size_t>(k / 2) +
                                       static_cast<std::size_t>(group) * (kWeightGroupSize / 2);
    for (int element = 0; element < kWeightGroupSize; ++element) {
      const float a_value = e4m3_to_float(a[a_group_offset + static_cast<std::size_t>(element)]);
      const std::uint8_t packed = b[b_group_offset + static_cast<std::size_t>(element / 2)];
      const std::uint8_t code = (element & 1) == 0 ? packed & 0x0fU : packed >> 4U;
      const float b_value = e2m1_to_float(code);
      partial = partial + a_value * b_value;
    }

    const std::size_t activation_group = static_cast<std::size_t>(group / activation_groups_per_weight_group);
    const std::size_t scale_a_index = static_cast<std::size_t>(row) *
                                          static_cast<std::size_t>(k / activation_group_size) +
                                      activation_group;
    const std::size_t scale_b_index = static_cast<std::size_t>(column) *
                                          static_cast<std::size_t>(weight_group_count) +
                                      static_cast<std::size_t>(group);
    const float scaled = (partial * load_scale(scale_a, scale_a_index)) * load_scale(scale_b, scale_b_index);
    accumulated = accumulated + scaled;
  }

  store_output(output, output_index, accumulated);
}

template <typename Scale, typename Output>
__global__ void tiled_fp4_gemm_kernel(
    const std::uint8_t* a,
    const Scale* scale_a,
    const std::uint8_t* b,
    const Scale* scale_b,
    Output* output,
    int m,
    int n,
    int k,
    int activation_group_size,
    std::size_t tile_count_n,
    std::size_t tile_count_m) {
  __shared__ float a_tile[kTileM][kWeightGroupSize];
  __shared__ float b_tile[kTileN][kWeightGroupSize];

  const std::size_t tile_index =
      static_cast<std::size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
  const std::size_t tile_m = tile_index / tile_count_n;
  const std::size_t tile_n = tile_index % tile_count_n;
  if (tile_m >= tile_count_m) {
    return;
  }
  const std::size_t row_base = tile_m * kTileM;
  const std::size_t column_base = tile_n * kTileN;
  const int weight_group_count = k / kWeightGroupSize;
  const int activation_groups_per_weight_group = activation_group_size / kWeightGroupSize;
  const int local_row = static_cast<int>(threadIdx.x) / kTileN;
  const int local_column = static_cast<int>(threadIdx.x) % kTileN;
  const std::size_t global_row = row_base + static_cast<std::size_t>(local_row);
  const std::size_t global_column = column_base + static_cast<std::size_t>(local_column);
  const bool valid_output = global_row < static_cast<std::size_t>(m) &&
                            global_column < static_cast<std::size_t>(n);
  float accumulated = 0.0f;

  for (int group = 0; group < weight_group_count; ++group) {
    for (int index = static_cast<int>(threadIdx.x); index < kTileM * kWeightGroupSize; index += kThreads) {
      const int row = index / kWeightGroupSize;
      const int element = index % kWeightGroupSize;
      const std::size_t global_a_row = row_base + static_cast<std::size_t>(row);
      a_tile[row][element] = global_a_row < static_cast<std::size_t>(m)
          ? e4m3_to_float(a[global_a_row * static_cast<std::size_t>(k) +
                            static_cast<std::size_t>(group) * kWeightGroupSize + element])
          : 0.0f;
    }
    for (int index = static_cast<int>(threadIdx.x); index < kTileN * kWeightGroupSize; index += kThreads) {
      const int column = index / kWeightGroupSize;
      const int element = index % kWeightGroupSize;
      const std::size_t global_b_column = column_base + static_cast<std::size_t>(column);
      if (global_b_column < static_cast<std::size_t>(n)) {
        const std::size_t packed_index =
            global_b_column * static_cast<std::size_t>(k / 2) +
            static_cast<std::size_t>(group) * (kWeightGroupSize / 2) +
            static_cast<std::size_t>(element / 2);
        const std::uint8_t packed = b[packed_index];
        const std::uint8_t code = (element & 1) == 0 ? packed & 0x0fU : packed >> 4U;
        b_tile[column][element] = e2m1_to_float(code);
      } else {
        b_tile[column][element] = 0.0f;
      }
    }
    __syncthreads();

    if (valid_output) {
      float partial = 0.0f;
      for (int element = 0; element < kWeightGroupSize; ++element) {
        partial = partial + a_tile[local_row][element] * b_tile[local_column][element];
      }
      const std::size_t activation_group =
          static_cast<std::size_t>(group / activation_groups_per_weight_group);
      const std::size_t scale_a_index =
          global_row * static_cast<std::size_t>(k / activation_group_size) + activation_group;
      const std::size_t scale_b_index =
          global_column * static_cast<std::size_t>(weight_group_count) +
          static_cast<std::size_t>(group);
      const float scaled = (partial * load_scale(scale_a, scale_a_index)) *
                           load_scale(scale_b, scale_b_index);
      accumulated = accumulated + scaled;
    }
    __syncthreads();
  }
  if (valid_output) {
    const std::size_t output_index = global_row * static_cast<std::size_t>(n) + global_column;
    store_output(output, output_index, accumulated);
  }
}

template <typename Scale, typename Output>
int launch_fp4_gemm(
    const std::uint8_t* a,
    const Scale* scale_a,
    const std::uint8_t* b,
    const Scale* scale_b,
    void* output,
    int m,
    int n,
    int k,
    int activation_group_size,
    hipStream_t stream,
    std::size_t output_count,
    unsigned int block_count) {
  const unsigned int blocks_x = block_count < kGridXLimit ? block_count : kGridXLimit;
  const std::uint64_t blocks_y =
      (static_cast<std::uint64_t>(block_count) - 1U) / blocks_x + 1U;
  if (blocks_y > kGridYLimit) {
    return static_cast<int>(hipErrorInvalidConfiguration);
  }
  hipLaunchKernelGGL(
      (fp4_gemm_kernel<Scale, Output>),
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
      k,
      activation_group_size,
      output_count);
  return static_cast<int>(hipGetLastError());
}

template <typename Scale, typename Output>
int launch_tiled_fp4_gemm(
    const std::uint8_t* a,
    const Scale* scale_a,
    const std::uint8_t* b,
    const Scale* scale_b,
    void* output,
    int m,
    int n,
    int k,
    int activation_group_size,
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
      (tiled_fp4_gemm_kernel<Scale, Output>),
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
      activation_group_size,
      tile_count_n,
      tile_count_m);
  return static_cast<int>(hipGetLastError());
}

bool checked_multiply(std::size_t left, std::size_t right, std::size_t* product) {
  if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
    return false;
  }
  *product = left * right;
  return true;
}

bool fits_pointer_extent(std::size_t elements, std::size_t element_size) {
  return elements <= static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / element_size;
}

}  // namespace

extern "C" int redqueen_ds4_fp4_gemm(
    const void* a_e4m3,
    const void* scale_a,
    const void* b_e2m1fn_x2,
    const void* scale_b,
    void* output,
    int m,
    int n,
    int k,
    int activation_group_size,
    int scale_type,
    int output_type,
    void* stream_handle) {
  if (a_e4m3 == nullptr || scale_a == nullptr || b_e2m1fn_x2 == nullptr || scale_b == nullptr || output == nullptr ||
      m <= 0 || n <= 0 || k <= 0 || (activation_group_size != 32 && activation_group_size != 128) ||
      k % kWeightGroupSize != 0 || k % activation_group_size != 0 ||
      (scale_type != REDQUEEN_DS4_FP4_GEMM_SCALE_E8M0 && scale_type != REDQUEEN_DS4_FP4_GEMM_SCALE_F32) ||
      (output_type != REDQUEEN_DS4_FP4_GEMM_OUTPUT_F32 && output_type != REDQUEEN_DS4_FP4_GEMM_OUTPUT_BF16)) {
    return static_cast<int>(hipErrorInvalidValue);
  }

  std::size_t output_count = 0;
  std::size_t a_elements = 0;
  std::size_t b_elements = 0;
  std::size_t scale_a_elements = 0;
  std::size_t scale_b_elements = 0;
  if (!checked_multiply(static_cast<std::size_t>(m), static_cast<std::size_t>(n), &output_count) ||
      !checked_multiply(static_cast<std::size_t>(m), static_cast<std::size_t>(k), &a_elements) ||
      !checked_multiply(static_cast<std::size_t>(n), static_cast<std::size_t>(k / 2), &b_elements) ||
      !checked_multiply(
          static_cast<std::size_t>(m), static_cast<std::size_t>(k / activation_group_size), &scale_a_elements) ||
      !checked_multiply(static_cast<std::size_t>(n), static_cast<std::size_t>(k / kWeightGroupSize), &scale_b_elements) ||
      !fits_pointer_extent(a_elements, sizeof(std::uint8_t)) ||
      !fits_pointer_extent(b_elements, sizeof(std::uint8_t)) ||
      !fits_pointer_extent(output_count, output_type == REDQUEEN_DS4_FP4_GEMM_OUTPUT_BF16 ? sizeof(std::uint16_t)
                                                                                         : sizeof(float))) {
    return static_cast<int>(hipErrorInvalidValue);
  }

  const std::size_t scale_size = scale_type == REDQUEEN_DS4_FP4_GEMM_SCALE_E8M0 ? sizeof(std::uint8_t) : sizeof(float);
  if (!fits_pointer_extent(scale_a_elements, scale_size) || !fits_pointer_extent(scale_b_elements, scale_size)) {
    return static_cast<int>(hipErrorInvalidValue);
  }

  const std::size_t block_count_size = (output_count - 1) / kThreads + 1;
  if (block_count_size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return static_cast<int>(hipErrorInvalidValue);
  }

  const hipError_t pending_error = hipGetLastError();
  if (pending_error != hipSuccess && pending_error != hipErrorNotReady) {
    return static_cast<int>(pending_error);
  }

  const auto* a = static_cast<const std::uint8_t*>(a_e4m3);
  const auto* b = static_cast<const std::uint8_t*>(b_e2m1fn_x2);
  const auto stream = reinterpret_cast<hipStream_t>(stream_handle);
  const unsigned int block_count = static_cast<unsigned int>(block_count_size);
  if (scale_type == REDQUEEN_DS4_FP4_GEMM_SCALE_E8M0) {
    const auto* scale_a_e8m0 = static_cast<const std::uint8_t*>(scale_a);
    const auto* scale_b_e8m0 = static_cast<const std::uint8_t*>(scale_b);
    if (output_type == REDQUEEN_DS4_FP4_GEMM_OUTPUT_F32) {
      return launch_fp4_gemm<std::uint8_t, float>(
          a, scale_a_e8m0, b, scale_b_e8m0, output, m, n, k, activation_group_size, stream, output_count, block_count);
    }
    return launch_fp4_gemm<std::uint8_t, std::uint16_t>(
        a, scale_a_e8m0, b, scale_b_e8m0, output, m, n, k, activation_group_size, stream, output_count, block_count);
  }

  const auto* scale_a_f32 = static_cast<const float*>(scale_a);
  const auto* scale_b_f32 = static_cast<const float*>(scale_b);
  if (output_type == REDQUEEN_DS4_FP4_GEMM_OUTPUT_F32) {
    return launch_fp4_gemm<float, float>(
        a, scale_a_f32, b, scale_b_f32, output, m, n, k, activation_group_size, stream, output_count, block_count);
  }
  return launch_fp4_gemm<float, std::uint16_t>(
      a, scale_a_f32, b, scale_b_f32, output, m, n, k, activation_group_size, stream, output_count, block_count);
}

extern "C" int redqueen_ds4_fp4_gemm_tiled(
    const void* a_e4m3,
    const void* scale_a,
    const void* b_e2m1fn_x2,
    const void* scale_b,
    void* output,
    int m,
    int n,
    int k,
    int activation_group_size,
    int scale_type,
    int output_type,
    void* stream_handle) {
  if (a_e4m3 == nullptr || scale_a == nullptr || b_e2m1fn_x2 == nullptr || scale_b == nullptr || output == nullptr ||
      m <= 0 || n <= 0 || k <= 0 || (activation_group_size != 32 && activation_group_size != 128) ||
      k % kWeightGroupSize != 0 || k % activation_group_size != 0 ||
      (scale_type != REDQUEEN_DS4_FP4_GEMM_SCALE_E8M0 && scale_type != REDQUEEN_DS4_FP4_GEMM_SCALE_F32) ||
      (output_type != REDQUEEN_DS4_FP4_GEMM_OUTPUT_F32 && output_type != REDQUEEN_DS4_FP4_GEMM_OUTPUT_BF16)) {
    return static_cast<int>(hipErrorInvalidValue);
  }

  std::size_t output_count = 0;
  std::size_t a_elements = 0;
  std::size_t b_elements = 0;
  std::size_t scale_a_elements = 0;
  std::size_t scale_b_elements = 0;
  if (!checked_multiply(static_cast<std::size_t>(m), static_cast<std::size_t>(n), &output_count) ||
      !checked_multiply(static_cast<std::size_t>(m), static_cast<std::size_t>(k), &a_elements) ||
      !checked_multiply(static_cast<std::size_t>(n), static_cast<std::size_t>(k / 2), &b_elements) ||
      !checked_multiply(
          static_cast<std::size_t>(m), static_cast<std::size_t>(k / activation_group_size), &scale_a_elements) ||
      !checked_multiply(static_cast<std::size_t>(n), static_cast<std::size_t>(k / kWeightGroupSize), &scale_b_elements) ||
      !fits_pointer_extent(a_elements, sizeof(std::uint8_t)) ||
      !fits_pointer_extent(b_elements, sizeof(std::uint8_t)) ||
      !fits_pointer_extent(output_count, output_type == REDQUEEN_DS4_FP4_GEMM_OUTPUT_BF16 ? sizeof(std::uint16_t)
                                                                                         : sizeof(float))) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const std::size_t scale_size = scale_type == REDQUEEN_DS4_FP4_GEMM_SCALE_E8M0 ? sizeof(std::uint8_t) : sizeof(float);
  if (!fits_pointer_extent(scale_a_elements, scale_size) || !fits_pointer_extent(scale_b_elements, scale_size)) {
    return static_cast<int>(hipErrorInvalidValue);
  }

  const hipError_t pending_error = hipGetLastError();
  if (pending_error != hipSuccess && pending_error != hipErrorNotReady) {
    return static_cast<int>(pending_error);
  }

  const auto* a = static_cast<const std::uint8_t*>(a_e4m3);
  const auto* b = static_cast<const std::uint8_t*>(b_e2m1fn_x2);
  const auto stream = reinterpret_cast<hipStream_t>(stream_handle);
  if (scale_type == REDQUEEN_DS4_FP4_GEMM_SCALE_E8M0) {
    const auto* scale_a_e8m0 = static_cast<const std::uint8_t*>(scale_a);
    const auto* scale_b_e8m0 = static_cast<const std::uint8_t*>(scale_b);
    if (output_type == REDQUEEN_DS4_FP4_GEMM_OUTPUT_F32) {
      return launch_tiled_fp4_gemm<std::uint8_t, float>(
          a, scale_a_e8m0, b, scale_b_e8m0, output, m, n, k, activation_group_size, stream);
    }
    return launch_tiled_fp4_gemm<std::uint8_t, std::uint16_t>(
        a, scale_a_e8m0, b, scale_b_e8m0, output, m, n, k, activation_group_size, stream);
  }

  const auto* scale_a_f32 = static_cast<const float*>(scale_a);
  const auto* scale_b_f32 = static_cast<const float*>(scale_b);
  if (output_type == REDQUEEN_DS4_FP4_GEMM_OUTPUT_F32) {
    return launch_tiled_fp4_gemm<float, float>(
        a, scale_a_f32, b, scale_b_f32, output, m, n, k, activation_group_size, stream);
  }
  return launch_tiled_fp4_gemm<float, std::uint16_t>(
      a, scale_a_f32, b, scale_b_f32, output, m, n, k, activation_group_size, stream);
}
