#ifndef REDQUEEN_DS4_FP8_GEMM_H
#define REDQUEEN_DS4_FP8_GEMM_H

enum redqueen_ds4_fp8_gemm_output_type {
  REDQUEEN_DS4_FP8_GEMM_OUTPUT_F32 = 0,
  REDQUEEN_DS4_FP8_GEMM_OUTPUT_BF16 = 1,
};

#ifdef __cplusplus
extern "C" {
#endif

// Device pointers use row-major A[M,K], B[N,K], scale_a[M,K/G],
// scale_b[ceil(N/G),K/G], and output[M,N]. The stream is a HIP stream handle.
int redqueen_ds4_fp8_gemm(
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
    void* stream_handle);

#ifdef __cplusplus
}
#endif

#endif  // REDQUEEN_DS4_FP8_GEMM_H
