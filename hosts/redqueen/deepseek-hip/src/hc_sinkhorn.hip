#include "hc_sinkhorn.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <limits>

namespace {

constexpr int kThreads = 64;
constexpr int kMaxHc = 16;
constexpr unsigned int kMaxGridX = 65535U;

__global__ void hc_split_sinkhorn_kernel(
    const float* mixes_f32,
    const float* hc_scale_f32,
    const float* hc_base_f32,
    float* pre_f32,
    float* post_f32,
    float* comb_f32,
    std::size_t rows,
    int hc_mult,
    std::size_t mix_width,
    int sinkhorn_iters,
    float eps) {
  __shared__ float mixes_shared[(kMaxHc + 2) * kMaxHc];
  __shared__ float comb_shared[kMaxHc * kMaxHc];
  __shared__ float row_max[kMaxHc];
  __shared__ float row_sum[kMaxHc];
  __shared__ float col_sum[kMaxHc];

  for (std::size_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const int thread = static_cast<int>(threadIdx.x);
    for (int index = thread; index < static_cast<int>(mix_width); index += kThreads) {
      mixes_shared[index] = mixes_f32[row * mix_width + static_cast<std::size_t>(index)];
    }
    __syncthreads();

    for (int index = thread; index < hc_mult; index += kThreads) {
      const float pre_logit = mixes_shared[index] * hc_scale_f32[0] + hc_base_f32[index];
      const float post_logit = mixes_shared[index + hc_mult] * hc_scale_f32[1] +
                               hc_base_f32[index + hc_mult];
      pre_f32[row * static_cast<std::size_t>(hc_mult) + static_cast<std::size_t>(index)] =
          1.0f / (1.0f + expf(-pre_logit)) + eps;
      post_f32[row * static_cast<std::size_t>(hc_mult) + static_cast<std::size_t>(index)] =
          2.0f / (1.0f + expf(-post_logit));
    }

    const int matrix_elements = hc_mult * hc_mult;
    for (int index = thread; index < matrix_elements; index += kThreads) {
      const int base = 2 * hc_mult + index;
      comb_shared[index] = mixes_shared[base] * hc_scale_f32[2] + hc_base_f32[base];
    }
    __syncthreads();

    for (int row_index = thread; row_index < hc_mult; row_index += kThreads) {
      float maximum = -INFINITY;
      for (int column = 0; column < hc_mult; ++column) {
        const float value = comb_shared[row_index * hc_mult + column];
        if (value != value) {
          maximum = value;
          break;
        }
        if (value > maximum) {
          maximum = value;
        }
      }
      row_max[row_index] = maximum;
    }
    __syncthreads();

    for (int index = thread; index < matrix_elements; index += kThreads) {
      const int row_index = index / hc_mult;
      comb_shared[index] = expf(comb_shared[index] - row_max[row_index]);
    }
    __syncthreads();

    for (int row_index = thread; row_index < hc_mult; row_index += kThreads) {
      float sum = 0.0f;
      for (int column = 0; column < hc_mult; ++column) {
        sum = sum + comb_shared[row_index * hc_mult + column];
      }
      row_sum[row_index] = sum;
    }
    __syncthreads();

    for (int index = thread; index < matrix_elements; index += kThreads) {
      const int row_index = index / hc_mult;
      comb_shared[index] = comb_shared[index] / row_sum[row_index] + eps;
    }
    __syncthreads();

    for (int column = thread; column < hc_mult; column += kThreads) {
      float sum = 0.0f;
      for (int row_index = 0; row_index < hc_mult; ++row_index) {
        sum = sum + comb_shared[row_index * hc_mult + column];
      }
      col_sum[column] = sum;
    }
    __syncthreads();

    for (int index = thread; index < matrix_elements; index += kThreads) {
      const int column = index % hc_mult;
      comb_shared[index] = comb_shared[index] / (col_sum[column] + eps);
    }
    __syncthreads();

    for (int iteration = 0; iteration < sinkhorn_iters - 1; ++iteration) {
      for (int row_index = thread; row_index < hc_mult; row_index += kThreads) {
        float sum = 0.0f;
        for (int column = 0; column < hc_mult; ++column) {
          sum = sum + comb_shared[row_index * hc_mult + column];
        }
        row_sum[row_index] = sum;
      }
      __syncthreads();

      for (int index = thread; index < matrix_elements; index += kThreads) {
        const int row_index = index / hc_mult;
        comb_shared[index] = comb_shared[index] / (row_sum[row_index] + eps);
      }
      __syncthreads();

      for (int column = thread; column < hc_mult; column += kThreads) {
        float sum = 0.0f;
        for (int row_index = 0; row_index < hc_mult; ++row_index) {
          sum = sum + comb_shared[row_index * hc_mult + column];
        }
        col_sum[column] = sum;
      }
      __syncthreads();

      for (int index = thread; index < matrix_elements; index += kThreads) {
        const int column = index % hc_mult;
        comb_shared[index] = comb_shared[index] / (col_sum[column] + eps);
      }
      __syncthreads();
    }

    for (int index = thread; index < matrix_elements; index += kThreads) {
      comb_f32[row * static_cast<std::size_t>(matrix_elements) + static_cast<std::size_t>(index)] =
          comb_shared[index];
    }
    __syncthreads();
  }
}

bool fits_pointer_extent(std::size_t elements) {
  return elements <= static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(float);
}

}  // namespace

extern "C" int redqueen_ds4_hc_split_sinkhorn(
    const float* mixes_f32,
    const float* hc_scale_f32,
    const float* hc_base_f32,
    float* pre_f32,
    float* post_f32,
    float* comb_f32,
    int rows,
    int hc_mult,
    int sinkhorn_iters,
    float eps,
    void* stream_handle) {
  if (mixes_f32 == nullptr || hc_scale_f32 == nullptr || hc_base_f32 == nullptr || pre_f32 == nullptr ||
      post_f32 == nullptr || comb_f32 == nullptr || rows <= 0 || hc_mult <= 0 || hc_mult > kMaxHc ||
      sinkhorn_iters <= 0 || !std::isfinite(eps) || eps < 0.0f || pre_f32 == post_f32 || pre_f32 == comb_f32 ||
      post_f32 == comb_f32 || mixes_f32 == pre_f32 || mixes_f32 == post_f32 || mixes_f32 == comb_f32 ||
      hc_scale_f32 == pre_f32 || hc_scale_f32 == post_f32 || hc_scale_f32 == comb_f32 ||
      hc_base_f32 == pre_f32 || hc_base_f32 == post_f32 || hc_base_f32 == comb_f32) {
    return static_cast<int>(hipErrorInvalidValue);
  }

  const std::size_t row_count = static_cast<std::size_t>(rows);
  const std::size_t hc = static_cast<std::size_t>(hc_mult);
  const std::size_t mix_width = (hc + 2U) * hc;
  const std::size_t output_width = hc * hc;
  if (row_count > std::numeric_limits<std::size_t>::max() / mix_width ||
      row_count > std::numeric_limits<std::size_t>::max() / output_width ||
      !fits_pointer_extent(row_count * mix_width) || !fits_pointer_extent(row_count * hc) ||
      !fits_pointer_extent(row_count * output_width) || !fits_pointer_extent(mix_width)) {
    return static_cast<int>(hipErrorInvalidValue);
  }

  const hipError_t pending_error = hipGetLastError();
  if (pending_error != hipSuccess && pending_error != hipErrorNotReady) {
    return static_cast<int>(pending_error);
  }

  unsigned int grid_x = static_cast<unsigned int>(rows);
  if (grid_x > kMaxGridX) {
    grid_x = kMaxGridX;
  }
  hipLaunchKernelGGL(
      hc_split_sinkhorn_kernel,
      dim3(grid_x),
      dim3(kThreads),
      0,
      reinterpret_cast<hipStream_t>(stream_handle),
      mixes_f32,
      hc_scale_f32,
      hc_base_f32,
      pre_f32,
      post_f32,
      comb_f32,
      row_count,
      hc_mult,
      mix_width,
      sinkhorn_iters,
      eps);
  return static_cast<int>(hipGetLastError());
}
