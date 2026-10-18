#ifndef REDQUEEN_DS4_SPARSE_ATTN_H
#define REDQUEEN_DS4_SPARSE_ATTN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Correctness baseline for contiguous tensors: q [batch, queries, heads, dim],
// kv [batch, keys, dim], sink [heads], indices [batch, queries, topk], and
// output [batch, queries, heads, dim]. q, kv, and output use BF16; sink uses
// FP32; indices use int32. Invalid indices (negative or >= keys) are masked as
// missing entries. The stream is a HIP stream handle.
int redqueen_ds4_sparse_attn(
    const void* q_bf16,
    const void* kv_bf16,
    const float* sink_f32,
    const int32_t* indices_i32,
    void* output_bf16,
    int batch,
    int queries,
    int heads,
    int dim,
    int keys,
    int topk,
    float scale,
    void* stream_handle);

#ifdef __cplusplus
}
#endif

#endif  // REDQUEEN_DS4_SPARSE_ATTN_H
