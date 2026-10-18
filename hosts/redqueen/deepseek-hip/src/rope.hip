#include "rope.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

constexpr int kThreads = 256;
constexpr int kF32 = REDQUEEN_DS4_ROPE_DTYPE_F32;
constexpr int kBf16 = REDQUEEN_DS4_ROPE_DTYPE_BF16;

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

__device__ __forceinline__ float load_as_float(const void* pointer, std::size_t index, int dtype) {
  return dtype == kF32 ? load_f32(pointer, index) : load_bf16(pointer, index);
}

__device__ __forceinline__ void store_from_float(void* pointer, std::size_t index, float value, int dtype) {
  if (dtype == kF32) {
    static_cast<float*>(pointer)[index] = value;
  } else {
    static_cast<std::uint16_t*>(pointer)[index] = float_to_bf16_rne(value);
  }
}

__global__ void rope_kernel(
    void* x,
    const float* freqs_cis,
    std::size_t sequence_heads,
    std::size_t batch_stride,
    std::size_t sequence_stride,
    std::size_t head_stride,
    int sequence,
    int heads,
    int rotary_dim,
    int dtype,
    int inverse) {
  const std::size_t row = static_cast<std::size_t>(blockIdx.x);
  const std::size_t batch_index = row / sequence_heads;
  const int sequence_index = static_cast<int>((row / static_cast<std::size_t>(heads)) % sequence);
  const int head_index = static_cast<int>(row % static_cast<std::size_t>(heads));
  const std::size_t x_base = batch_index * batch_stride +
                             static_cast<std::size_t>(sequence_index) * sequence_stride +
                             static_cast<std::size_t>(head_index) * head_stride;
  const std::size_t frequency_base =
      static_cast<std::size_t>(sequence_index) * static_cast<std::size_t>(rotary_dim);

  for (int pair = static_cast<int>(threadIdx.x); pair < rotary_dim / 2; pair += blockDim.x) {
    const std::size_t pair_offset = static_cast<std::size_t>(pair) * 2U;
    const std::size_t frequency_offset = frequency_base + pair_offset;
    const float cosine = freqs_cis[frequency_offset];
    const float signed_sine = inverse ? -freqs_cis[frequency_offset + 1U] : freqs_cis[frequency_offset + 1U];
    const float even = load_as_float(x, x_base + pair_offset, dtype);
    const float odd = load_as_float(x, x_base + pair_offset + 1U, dtype);
    const float rotated_even = even * cosine - odd * signed_sine;
    const float rotated_odd = even * signed_sine + odd * cosine;
    store_from_float(x, x_base + pair_offset, rotated_even, dtype);
    store_from_float(x, x_base + pair_offset + 1U, rotated_odd, dtype);
  }
}

bool checked_multiply(std::size_t left, std::size_t right, std::size_t* product) {
  if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
    return false;
  }
  *product = left * right;
  return true;
}

bool checked_add(std::size_t left, std::size_t right, std::size_t* sum) {
  if (left > std::numeric_limits<std::size_t>::max() - right) {
    return false;
  }
  *sum = left + right;
  return true;
}

bool checked_span(int count, std::size_t stride, std::size_t width, std::size_t* span) {
  std::size_t prefix = 0;
  if (!checked_multiply(static_cast<std::size_t>(count - 1), stride, &prefix)) {
    return false;
  }
  return checked_add(prefix, width, span);
}

bool fits_pointer_extent(std::size_t elements, std::size_t element_size) {
  return elements <= static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / element_size;
}

}  // namespace

extern "C" int redqueen_ds4_rope_inplace(
    void* x,
    const float* freqs_cis,
    int batch,
    int sequence,
    int heads,
    int rotary_dim,
    int dtype,
    int inverse,
    std::int64_t batch_stride_arg,
    std::int64_t sequence_stride_arg,
    std::int64_t head_stride_arg,
    void* stream_handle) {
  if (x == nullptr || freqs_cis == nullptr || batch <= 0 || sequence <= 0 || heads <= 0 || rotary_dim <= 0 ||
      (rotary_dim & 1) != 0 || (dtype != kF32 && dtype != kBf16) || (inverse != 0 && inverse != 1) ||
      batch_stride_arg < 0 || sequence_stride_arg < 0 || head_stride_arg < 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }

  const std::size_t batch_stride = static_cast<std::size_t>(batch_stride_arg);
  const std::size_t sequence_stride = static_cast<std::size_t>(sequence_stride_arg);
  const std::size_t head_stride = static_cast<std::size_t>(head_stride_arg);
  const std::size_t width = static_cast<std::size_t>(rotary_dim);
  std::size_t head_span = 0;
  std::size_t sequence_span = 0;
  std::size_t batch_span = 0;
  std::size_t sequence_heads = 0;
  std::size_t row_count = 0;
  std::size_t frequency_elements = 0;
  if ((heads > 1 && head_stride < width) || !checked_span(heads, head_stride, width, &head_span) ||
      (sequence > 1 && sequence_stride < head_span) ||
      !checked_span(sequence, sequence_stride, head_span, &sequence_span) ||
      (batch > 1 && batch_stride < sequence_span) ||
      !checked_span(batch, batch_stride, sequence_span, &batch_span) ||
      !checked_multiply(static_cast<std::size_t>(sequence), static_cast<std::size_t>(heads), &sequence_heads) ||
      !checked_multiply(static_cast<std::size_t>(batch), sequence_heads, &row_count) ||
      !checked_multiply(static_cast<std::size_t>(sequence), width, &frequency_elements) ||
      row_count > std::numeric_limits<unsigned int>::max() ||
      !fits_pointer_extent(batch_span, dtype == kF32 ? sizeof(float) : sizeof(std::uint16_t)) ||
      !fits_pointer_extent(frequency_elements, sizeof(float))) {
    return static_cast<int>(hipErrorInvalidValue);
  }

  const hipError_t pending_error = hipGetLastError();
  if (pending_error != hipSuccess && pending_error != hipErrorNotReady) {
    return static_cast<int>(pending_error);
  }

  hipLaunchKernelGGL(
      rope_kernel,
      dim3(static_cast<unsigned int>(row_count)),
      dim3(kThreads),
      0,
      reinterpret_cast<hipStream_t>(stream_handle),
      x,
      freqs_cis,
      sequence_heads,
      batch_stride,
      sequence_stride,
      head_stride,
      sequence,
      heads,
      rotary_dim,
      dtype,
      inverse);
  return static_cast<int>(hipGetLastError());
}
