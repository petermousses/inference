#include "sparse_attn.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

constexpr int kThreads = 256;
constexpr int kKeyTile = 64;
constexpr int kTiledThreads = 512;
constexpr int kTiledKeyTile = 16;
constexpr int kTiledFeatureLanes = 32;
constexpr unsigned int kGridXLimit = 65535U;
constexpr unsigned int kGridYLimit = 65535U;

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

__device__ __forceinline__ float bf16_to_float(std::uint16_t value) {
  return __uint_as_float(static_cast<std::uint32_t>(value) << 16U);
}

// One CTA handles one query-head and one 256-wide output slice. Recomputing
// scores for additional output slices keeps this baseline allocation-free and
// supports arbitrary positive head dimensions without dynamic local storage.
__global__ void sparse_attn_kernel(
    const void* q_bf16,
    const void* kv_bf16,
    const float* sink_f32,
    const std::int32_t* indices_i32,
    std::uint16_t* output_bf16,
    std::size_t heads_per_batch,
    int heads,
    int dim,
    int keys,
    int topk,
    float scale) {
  __shared__ float tile_scores[kKeyTile];
  __shared__ std::uint16_t tile_probabilities_bf16[kKeyTile];
  __shared__ int selected_ids[kKeyTile];
  __shared__ float running_max;
  __shared__ float denominator;
  __shared__ float previous_scale;
  __shared__ float final_denominator;
  __shared__ int any_valid;

  const std::size_t row = static_cast<std::size_t>(blockIdx.x);
  const std::size_t batch_index = row / heads_per_batch;
  const std::size_t query_row = row / static_cast<std::size_t>(heads);
  const int head_index = static_cast<int>(row % static_cast<std::size_t>(heads));
  const std::size_t q_base = row * static_cast<std::size_t>(dim);
  const std::size_t indices_base = query_row * static_cast<std::size_t>(topk);
  const std::size_t output_slice = static_cast<std::size_t>(blockIdx.y) * kThreads;
  const std::size_t feature = output_slice + threadIdx.x;

  float output_accumulator = 0.0f;
  if (threadIdx.x == 0) {
    running_max = -1.0e30f;
    denominator = 0.0f;
    any_valid = 0;
  }
  __syncthreads();

  for (std::int64_t base = 0; base < topk; base += kKeyTile) {
    if (threadIdx.x < kKeyTile) {
      const std::int64_t position = base + threadIdx.x;
      int selected = -1;
      float score = -INFINITY;
      if (position < topk) {
        const int candidate = indices_i32[indices_base + static_cast<std::size_t>(position)];
        if (candidate >= 0 && candidate < keys) {
          selected = candidate;
          const std::size_t kv_base =
              (batch_index * static_cast<std::size_t>(keys) + static_cast<std::size_t>(candidate)) *
              static_cast<std::size_t>(dim);
          float dot = 0.0f;
          for (int column = 0; column < dim; ++column) {
            const float q_value = load_bf16(q_bf16, q_base + static_cast<std::size_t>(column));
            const float kv_value = load_bf16(kv_bf16, kv_base + static_cast<std::size_t>(column));
            dot = dot + q_value * kv_value;
          }
          score = dot * scale;
        }
      }
      selected_ids[threadIdx.x] = selected;
      tile_scores[threadIdx.x] = score;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      float tile_max = -INFINITY;
      for (int key = 0; key < kKeyTile; ++key) {
        tile_max = fmaxf(tile_max, tile_scores[key]);
      }
      const float new_max = fmaxf(running_max, tile_max);
      previous_scale = expf(running_max - new_max);
      denominator = denominator * previous_scale;
      float tile_sum = 0.0f;
      for (int key = 0; key < kKeyTile; ++key) {
        const float probability = selected_ids[key] >= 0 ? expf(tile_scores[key] - new_max) : 0.0f;
        tile_sum = tile_sum + probability;
        tile_probabilities_bf16[key] = float_to_bf16_rne(probability);
        any_valid |= selected_ids[key] >= 0;
      }
      denominator = denominator + tile_sum;
      running_max = new_max;
    }
    __syncthreads();

    if (feature < static_cast<std::size_t>(dim)) {
      output_accumulator = output_accumulator * previous_scale;
      for (int key = 0; key < kKeyTile; ++key) {
        const int selected = selected_ids[key];
        if (selected >= 0) {
          const std::size_t kv_index =
              (batch_index * static_cast<std::size_t>(keys) + static_cast<std::size_t>(selected)) *
                  static_cast<std::size_t>(dim) +
              feature;
          const float probability = bf16_to_float(tile_probabilities_bf16[key]);
          output_accumulator = output_accumulator + probability * load_bf16(kv_bf16, kv_index);
        }
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    final_denominator = any_valid != 0 ? denominator + expf(sink_f32[head_index] - running_max) : 0.0f;
  }
  __syncthreads();

  if (feature < static_cast<std::size_t>(dim)) {
    const float result = final_denominator > 0.0f ? output_accumulator / final_denominator : 0.0f;
    output_bf16[q_base + feature] = float_to_bf16_rne(result);
  }
}

// The baseline assigns one lane to each key and reduces all `dim` features
// serially. That makes a full-context prefill spend most of its time in a few
// long-running lanes. This variant assigns 32 lanes to each key, computes
// score tiles in parallel, and handles all output features in the same CTA.
// The production checkpoint uses dim=512, so this also halves its CTA count
// relative to the two output-slice baseline launch.
__global__ void sparse_attn_tiled_kernel(
    const void* q_bf16,
    const void* kv_bf16,
    const float* sink_f32,
    const std::int32_t* indices_i32,
    std::uint16_t* output_bf16,
    std::size_t query_head_rows,
    std::size_t heads_per_batch,
    int heads,
    int dim,
    int keys,
    int topk,
    float scale) {
  __shared__ float score_partials[kTiledKeyTile][kTiledFeatureLanes];
  __shared__ float tile_scores[kTiledKeyTile];
  __shared__ std::uint16_t tile_probabilities_bf16[kTiledKeyTile];
  __shared__ int selected_ids[kTiledKeyTile];
  __shared__ float running_max;
  __shared__ float denominator;
  __shared__ float previous_scale;
  __shared__ float final_denominator;
  __shared__ int any_valid;

  const std::size_t row = static_cast<std::size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
  if (row >= query_head_rows) {
    return;
  }
  const std::size_t batch_index = row / heads_per_batch;
  const std::size_t query_row = row / static_cast<std::size_t>(heads);
  const int head_index = static_cast<int>(row % static_cast<std::size_t>(heads));
  const std::size_t q_base = row * static_cast<std::size_t>(dim);
  const std::size_t indices_base = query_row * static_cast<std::size_t>(topk);
  const int key_group = static_cast<int>(threadIdx.x) / kTiledFeatureLanes;
  const int feature_lane = static_cast<int>(threadIdx.x) % kTiledFeatureLanes;
  const std::size_t output_base = q_base;

  float output_accumulator = 0.0f;
  if (threadIdx.x == 0) {
    running_max = -1.0e30f;
    denominator = 0.0f;
    any_valid = 0;
  }
  __syncthreads();

  for (std::int64_t base = 0; base < topk; base += kTiledKeyTile) {
    if (key_group < kTiledKeyTile) {
      const std::int64_t position = base + key_group;
      int selected = -1;
      if (position < topk) {
        const int candidate = indices_i32[indices_base + static_cast<std::size_t>(position)];
        if (candidate >= 0 && candidate < keys) {
          selected = candidate;
        }
      }
      if (feature_lane == 0) {
        selected_ids[key_group] = selected;
      }

      float dot = 0.0f;
      if (selected >= 0) {
        const std::size_t kv_base =
            (batch_index * static_cast<std::size_t>(keys) + static_cast<std::size_t>(selected)) *
            static_cast<std::size_t>(dim);
        for (int feature = feature_lane; feature < dim; feature += kTiledFeatureLanes) {
          dot = dot + load_bf16(q_bf16, q_base + static_cast<std::size_t>(feature)) *
                          load_bf16(kv_bf16, kv_base + static_cast<std::size_t>(feature));
        }
      }
      score_partials[key_group][feature_lane] = dot;
    }
    __syncthreads();

    if (threadIdx.x < kTiledKeyTile) {
      const int key = static_cast<int>(threadIdx.x);
      float dot = 0.0f;
      for (int lane = 0; lane < kTiledFeatureLanes; ++lane) {
        dot = dot + score_partials[key][lane];
      }
      tile_scores[key] = selected_ids[key] >= 0 ? dot * scale : -INFINITY;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      float tile_max = -INFINITY;
      for (int key = 0; key < kTiledKeyTile; ++key) {
        tile_max = fmaxf(tile_max, tile_scores[key]);
      }
      const float new_max = fmaxf(running_max, tile_max);
      previous_scale = expf(running_max - new_max);
      denominator = denominator * previous_scale;
      float tile_sum = 0.0f;
      for (int key = 0; key < kTiledKeyTile; ++key) {
        const float probability = selected_ids[key] >= 0 ? expf(tile_scores[key] - new_max) : 0.0f;
        tile_sum = tile_sum + probability;
        tile_probabilities_bf16[key] = float_to_bf16_rne(probability);
        any_valid |= selected_ids[key] >= 0;
      }
      denominator = denominator + tile_sum;
      running_max = new_max;
    }
    __syncthreads();

    const int feature = static_cast<int>(threadIdx.x);
    if (feature < dim) {
      output_accumulator = output_accumulator * previous_scale;
      for (int key = 0; key < kTiledKeyTile; ++key) {
        const int selected = selected_ids[key];
        if (selected >= 0) {
          const std::size_t kv_index =
              (batch_index * static_cast<std::size_t>(keys) + static_cast<std::size_t>(selected)) *
                  static_cast<std::size_t>(dim) +
              static_cast<std::size_t>(feature);
          const float probability = bf16_to_float(tile_probabilities_bf16[key]);
          output_accumulator = output_accumulator + probability * load_bf16(kv_bf16, kv_index);
        }
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    final_denominator = any_valid != 0 ? denominator + expf(sink_f32[head_index] - running_max) : 0.0f;
  }
  __syncthreads();

  const int feature = static_cast<int>(threadIdx.x);
  if (feature < dim) {
    const float result = final_denominator > 0.0f ? output_accumulator / final_denominator : 0.0f;
    output_bf16[output_base + static_cast<std::size_t>(feature)] = float_to_bf16_rne(result);
  }
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

int sparse_attn_impl(
    const void* q_bf16,
    const void* kv_bf16,
    const float* sink_f32,
    const std::int32_t* indices_i32,
    void* output_bf16,
    int batch,
    int queries,
    int heads,
    int dim,
    int keys,
    int topk,
    float scale,
    void* stream_handle,
    bool use_experimental_tiled) {
  if (q_bf16 == nullptr || kv_bf16 == nullptr || sink_f32 == nullptr || indices_i32 == nullptr ||
      output_bf16 == nullptr || batch <= 0 || queries <= 0 || heads <= 0 || dim <= 0 || keys <= 0 ||
      topk <= 0 || !std::isfinite(scale) || scale <= 0.0f) {
    return static_cast<int>(hipErrorInvalidValue);
  }

  std::size_t query_rows = 0;
  std::size_t query_head_rows = 0;
  std::size_t output_elements = 0;
  std::size_t kv_elements = 0;
  std::size_t index_elements = 0;
  std::size_t head_stride = 0;
  if (!checked_multiply(static_cast<std::size_t>(batch), static_cast<std::size_t>(queries), &query_rows) ||
      !checked_multiply(query_rows, static_cast<std::size_t>(heads), &query_head_rows) ||
      !checked_multiply(query_head_rows, static_cast<std::size_t>(dim), &output_elements) ||
      !checked_multiply(static_cast<std::size_t>(batch), static_cast<std::size_t>(keys), &kv_elements) ||
      !checked_multiply(kv_elements, static_cast<std::size_t>(dim), &kv_elements) ||
      !checked_multiply(query_rows, static_cast<std::size_t>(topk), &index_elements) ||
      !checked_multiply(static_cast<std::size_t>(queries), static_cast<std::size_t>(heads), &head_stride) ||
      query_head_rows > std::numeric_limits<unsigned int>::max() ||
      !fits_pointer_extent(output_elements, sizeof(std::uint16_t)) ||
      !fits_pointer_extent(kv_elements, sizeof(std::uint16_t)) ||
      !fits_pointer_extent(index_elements, sizeof(std::int32_t)) ||
      !fits_pointer_extent(static_cast<std::size_t>(heads), sizeof(float))) {
    return static_cast<int>(hipErrorInvalidValue);
  }

  const std::size_t output_slice_count =
      (static_cast<std::size_t>(dim) + static_cast<std::size_t>(kThreads - 1)) / kThreads;
  if (output_slice_count > 65535U) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const unsigned int output_slices = static_cast<unsigned int>(output_slice_count);
  const hipError_t pending_error = hipGetLastError();
  if (pending_error != hipSuccess && pending_error != hipErrorNotReady) {
    return static_cast<int>(pending_error);
  }

  if (use_experimental_tiled && dim >= 128 && dim <= kTiledThreads) {
    const unsigned int grid_x = static_cast<unsigned int>(
        query_head_rows < kGridXLimit ? query_head_rows : kGridXLimit);
    const std::size_t grid_y_size = (query_head_rows - 1) / grid_x + 1;
    if (grid_y_size > kGridYLimit) {
      return static_cast<int>(hipErrorInvalidConfiguration);
    }
    hipLaunchKernelGGL(
        sparse_attn_tiled_kernel,
        dim3(grid_x, static_cast<unsigned int>(grid_y_size)),
        dim3(kTiledThreads),
        0,
        reinterpret_cast<hipStream_t>(stream_handle),
        q_bf16,
        kv_bf16,
        sink_f32,
        indices_i32,
        static_cast<std::uint16_t*>(output_bf16),
        query_head_rows,
        head_stride,
        heads,
        dim,
        keys,
        topk,
        scale);
    return static_cast<int>(hipGetLastError());
  }

  hipLaunchKernelGGL(
      sparse_attn_kernel,
      dim3(static_cast<unsigned int>(query_head_rows), output_slices),
      dim3(kThreads),
      0,
      reinterpret_cast<hipStream_t>(stream_handle),
      q_bf16,
      kv_bf16,
      sink_f32,
      indices_i32,
      static_cast<std::uint16_t*>(output_bf16),
      head_stride,
      heads,
      dim,
      keys,
      topk,
      scale);
  return static_cast<int>(hipGetLastError());
}

}  // namespace

extern "C" int redqueen_ds4_sparse_attn(
    const void* q_bf16,
    const void* kv_bf16,
    const float* sink_f32,
    const std::int32_t* indices_i32,
    void* output_bf16,
    int batch,
    int queries,
    int heads,
    int dim,
    int keys,
    int topk,
    float scale,
    void* stream_handle) {
  return sparse_attn_impl(
      q_bf16,
      kv_bf16,
      sink_f32,
      indices_i32,
      output_bf16,
      batch,
      queries,
      heads,
      dim,
      keys,
      topk,
      scale,
      stream_handle,
      false);
}

extern "C" int redqueen_ds4_sparse_attn_experimental_tiled(
    const void* q_bf16,
    const void* kv_bf16,
    const float* sink_f32,
    const std::int32_t* indices_i32,
    void* output_bf16,
    int batch,
    int queries,
    int heads,
    int dim,
    int keys,
    int topk,
    float scale,
    void* stream_handle) {
  return sparse_attn_impl(
      q_bf16,
      kv_bf16,
      sink_f32,
      indices_i32,
      output_bf16,
      batch,
      queries,
      heads,
      dim,
      keys,
      topk,
      scale,
      stream_handle,
      true);
}
