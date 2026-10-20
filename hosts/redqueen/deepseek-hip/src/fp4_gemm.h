#ifndef REDQUEEN_DS4_FP4_GEMM_H
#define REDQUEEN_DS4_FP4_GEMM_H

enum redqueen_ds4_fp4_gemm_scale_type {
  REDQUEEN_DS4_FP4_GEMM_SCALE_E8M0 = 0,
  REDQUEEN_DS4_FP4_GEMM_SCALE_F32 = 1,
};

enum redqueen_ds4_fp4_gemm_output_type {
  REDQUEEN_DS4_FP4_GEMM_OUTPUT_F32 = 0,
  REDQUEEN_DS4_FP4_GEMM_OUTPUT_BF16 = 1,
};

#ifdef __cplusplus
extern "C" {
#endif

// Computes C[M,N] = A_e4m3[M,K] @ B_e2m1fn_x2[N,K]^T. Inputs and output are
// contiguous row-major tensors. B stores two FP4 values per byte along K:
// even K indices occupy the low nibble, odd K indices the high nibble.
// scale_a is [M,K/activation_group_size]; scale_b is [N,K/32]. Scales share
// scale_type (E8M0 bytes or FP32). activation_group_size must be 32 or 128.
// K must be divisible by both 32 and activation_group_size.
// Output is FP32 or BF16 with FP32 accumulation. Input, scale, and output
// buffers must not overlap; allocation sizes remain the caller's responsibility.
int redqueen_ds4_fp4_gemm(
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
    void* stream_handle);

// Tiled implementation with the same exact per-group accumulation contract.
int redqueen_ds4_fp4_gemm_tiled(
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
    void* stream_handle);

#ifdef __cplusplus
}
#endif

#endif  // REDQUEEN_DS4_FP4_GEMM_H
