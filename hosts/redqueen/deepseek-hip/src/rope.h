#ifndef REDQUEEN_DS4_ROPE_H
#define REDQUEEN_DS4_ROPE_H

#include <stdint.h>

enum redqueen_ds4_rope_dtype {
  REDQUEEN_DS4_ROPE_DTYPE_F32 = 0,
  REDQUEEN_DS4_ROPE_DTYPE_BF16 = 1,
};

#ifdef __cplusplus
extern "C" {
#endif

// In-place rotary embedding over the rotary tail of a tensor. x points at the
// first rotary element and may be strided between batches, sequences, and
// heads; strides are measured in elements. Supported logical shapes are
// [batch, sequence, rotary_dim] (heads=1) and
// [batch, sequence, heads, rotary_dim]. freqs_cis is contiguous interleaved
// complex FP32 [sequence, rotary_dim / 2, 2], stored as cosine then sine.
// Inverse rotation conjugates each frequency. x uses dtype and is BF16 or F32.
// x and freqs_cis storage must not overlap.
int redqueen_ds4_rope_inplace(
    void* x,
    const float* freqs_cis,
    int batch,
    int sequence,
    int heads,
    int rotary_dim,
    int dtype,
    int inverse,
    int64_t batch_stride,
    int64_t sequence_stride,
    int64_t head_stride,
    void* stream_handle);

#ifdef __cplusplus
}
#endif

#endif  // REDQUEEN_DS4_ROPE_H
