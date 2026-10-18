#include "fp4_quant.h"

#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

constexpr int kMaxGroupSize = 32;
constexpr float kFp4Max = 6.0f;
constexpr float kFp4MaxInverse = 1.0f / kFp4Max;
constexpr float kE8M0MinimumAmax = 0x1.8p-124f;  // 6 * 2^-126
constexpr float kE4M3MinimumAmax = 0x1.8p-7f;    // 6 * 2^-9

enum ScaleType : int {
  kScaleE8M0 = REDQUEEN_DS4_FP4_QUANT_SCALE_E8M0,
  kScaleE4M3 = REDQUEEN_DS4_FP4_QUANT_SCALE_E4M3,
};

__device__ __forceinline__ float bf16_bits_to_float(std::uint16_t bits) {
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
  return sign * ldexpf(1.0f + static_cast<float>(mantissa) * 0.125f, exponent - 7);
}

__device__ __forceinline__ std::uint8_t fp4_encode_rne(float value) {
  const std::uint8_t sign = (__float_as_uint(value) & 0x80000000U) != 0U ? 0x08U : 0U;
  const float magnitude = fabsf(value);
  std::uint8_t code;
  if (magnitude <= 0.25f) {
    code = 0U;
  } else if (magnitude < 0.75f) {
    code = 1U;
  } else if (magnitude <= 1.25f) {
    code = 2U;
  } else if (magnitude < 1.75f) {
    code = 3U;
  } else if (magnitude <= 2.5f) {
    code = 4U;
  } else if (magnitude < 3.5f) {
    code = 5U;
  } else if (magnitude <= 5.0f) {
    code = 6U;
  } else {
    code = 7U;
  }
  return static_cast<std::uint8_t>(sign | code);
}

__device__ __forceinline__ float fp4_decode(std::uint8_t code) {
  float magnitude;
  switch (code & 0x07U) {
    case 0U: magnitude = 0.0f; break;
    case 1U: magnitude = 0.5f; break;
    case 2U: magnitude = 1.0f; break;
    case 3U: magnitude = 1.5f; break;
    case 4U: magnitude = 2.0f; break;
    case 5U: magnitude = 3.0f; break;
    case 6U: magnitude = 4.0f; break;
    default: magnitude = 6.0f; break;
  }
  return (code & 0x08U) != 0U ? -magnitude : magnitude;
}

__device__ __forceinline__ float e8m0_scale_from_amax(float amax, std::uint8_t* encoded) {
  const float normalized = amax * kFp4MaxInverse;
  const std::uint32_t bits = __float_as_uint(normalized);
  int exponent = static_cast<int>((bits >> 23U) & 0xffU) - 127;
  if ((bits & 0x007fffffU) != 0U) {
    ++exponent;
  }
  const int biased_exponent = exponent + 127;
  *encoded = static_cast<std::uint8_t>(biased_exponent);
  return __uint_as_float(static_cast<std::uint32_t>(biased_exponent) << 23U);
}

__device__ __forceinline__ float e4m3_scale_from_amax(float amax, std::uint8_t* encoded) {
  *encoded = __hip_cvt_float_to_fp8(amax / kFp4Max, __HIP_SATFINITE, __HIP_E4M3);
  return e4m3_to_float(*encoded);
}

template <ScaleType Scale>
__global__ void fp4_quantize_kernel(
    const std::uint16_t* input,
    void* output,
    std::uint8_t* output_scales,
    int cols,
    int block_size,
    bool inplace) {
  __shared__ float values[kMaxGroupSize];
  __shared__ float max_values[kMaxGroupSize];
  __shared__ float scale;

  const unsigned int lane = threadIdx.x;
  const unsigned int groups_per_row = static_cast<unsigned int>(cols / block_size);
  const unsigned int group = blockIdx.x;
  const unsigned int row = group / groups_per_row;
  const unsigned int group_in_row = group % groups_per_row;
  const std::size_t group_start = static_cast<std::size_t>(row) * static_cast<std::size_t>(cols) +
                                  static_cast<std::size_t>(group_in_row) * static_cast<std::size_t>(block_size);
  const float value = bf16_bits_to_float(input[group_start + lane]);
  values[lane] = value;
  max_values[lane] = fabsf(value);
  __syncthreads();

  for (int stride = block_size / 2; stride > 0; stride >>= 1) {
    if (lane < static_cast<unsigned int>(stride)) {
      max_values[lane] = fmaxf(max_values[lane], max_values[lane + stride]);
    }
    __syncthreads();
  }

  if (lane == 0U) {
    std::uint8_t encoded_scale;
    if constexpr (Scale == kScaleE8M0) {
      const float amax = fmaxf(max_values[0], kE8M0MinimumAmax);
      scale = e8m0_scale_from_amax(amax, &encoded_scale);
    } else {
      const float amax = fmaxf(max_values[0], kE4M3MinimumAmax);
      scale = e4m3_scale_from_amax(amax, &encoded_scale);
    }
    output_scales[group] = encoded_scale;
  }
  __syncthreads();

  if (inplace) {
    const float normalized = fminf(fmaxf(values[lane] / scale, -kFp4Max), kFp4Max);
    const std::uint8_t code = fp4_encode_rne(normalized);
    const float dequantized = fp4_decode(code) * scale;
    const_cast<std::uint16_t*>(input)[group_start + lane] = float_to_bf16_rne(dequantized);
  } else if ((lane & 1U) == 0U) {
    const float normalized_low = fminf(fmaxf(values[lane] / scale, -kFp4Max), kFp4Max);
    const float normalized_high = fminf(fmaxf(values[lane + 1U] / scale, -kFp4Max), kFp4Max);
    const std::uint8_t low = fp4_encode_rne(normalized_low);
    const std::uint8_t high = fp4_encode_rne(normalized_high);
    const std::size_t packed_offset = group_start / 2U + lane / 2U;
    static_cast<std::uint8_t*>(output)[packed_offset] = static_cast<std::uint8_t>(low | (high << 4U));
  }
}

template <ScaleType Scale>
int launch_fp4_quantize(
    const void* input_bf16,
    void* output,
    void* output_scales,
    int cols,
    int block_size,
    bool inplace,
    hipStream_t stream,
    std::uint64_t groups) {
  hipLaunchKernelGGL(
      (fp4_quantize_kernel<Scale>),
      dim3(static_cast<unsigned int>(groups)),
      dim3(static_cast<unsigned int>(block_size)),
      0,
      stream,
      static_cast<const std::uint16_t*>(input_bf16),
      output,
      static_cast<std::uint8_t*>(output_scales),
      cols,
      block_size,
      inplace);
  return static_cast<int>(hipGetLastError());
}

}  // namespace

extern "C" int redqueen_ds4_fp4_quantize_activation(
    const void* input_bf16,
    void* output,
    void* output_scales,
    int rows,
    int cols,
    int block_size,
    int scale_type,
    int inplace,
    void* stream_handle) {
  if (input_bf16 == nullptr || output == nullptr || output_scales == nullptr || rows <= 0 || cols <= 0 ||
      (block_size != 16 && block_size != 32) || cols % block_size != 0 || (inplace != 0 && inplace != 1) ||
      (scale_type != kScaleE8M0 && scale_type != kScaleE4M3) ||
      (inplace != 0 ? output != input_bf16 : output == input_bf16) || output_scales == input_bf16 ||
      output_scales == output) {
    return static_cast<int>(hipErrorInvalidValue);
  }

  const std::uint64_t groups = static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(cols / block_size);
  if (groups > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    return static_cast<int>(hipErrorInvalidValue);
  }

  const hipError_t pending_error = hipGetLastError();
  if (pending_error != hipSuccess && pending_error != hipErrorNotReady) {
    return static_cast<int>(pending_error);
  }

  const auto stream = reinterpret_cast<hipStream_t>(stream_handle);
  if (scale_type == kScaleE8M0) {
    return launch_fp4_quantize<kScaleE8M0>(
        input_bf16, output, output_scales, cols, block_size, inplace != 0, stream, groups);
  }
  return launch_fp4_quantize<kScaleE4M3>(
      input_bf16, output, output_scales, cols, block_size, inplace != 0, stream, groups);
}
