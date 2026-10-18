#ifndef REDQUEEN_DS4_HC_SINKHORN_H
#define REDQUEEN_DS4_HC_SINKHORN_H

#ifdef __cplusplus
extern "C" {
#endif

// Correctness baseline for DeepSeek-V4.1 Hyper-Connections mixing. `mixes` is
// contiguous FP32 [rows, (2 + hc_mult) * hc_mult], `hc_scale` is FP32 [3],
// `hc_base` is FP32 [(2 + hc_mult) * hc_mult], and outputs are FP32 `pre`
// [rows, hc_mult], `post` [rows, hc_mult], and `comb`
// [rows, hc_mult, hc_mult]. The `comb` normalization follows the pinned 1017
// Sinkhorn operation order. Buffers must not overlap. hc_mult is in [1, 16],
// sinkhorn_iters must be at least one, and eps must be finite and nonnegative.
// The stream is a HIP stream handle.
int redqueen_ds4_hc_split_sinkhorn(
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
    void* stream_handle);

#ifdef __cplusplus
}
#endif

#endif  // REDQUEEN_DS4_HC_SINKHORN_H
