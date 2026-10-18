#ifndef REDQUEEN_DS4_RMSNORM_H
#define REDQUEEN_DS4_RMSNORM_H

enum redqueen_ds4_rmsnorm_dtype {
  REDQUEEN_DS4_RMSNORM_DTYPE_F32 = 0,
  REDQUEEN_DS4_RMSNORM_DTYPE_BF16 = 1,
};

#ifdef __cplusplus
extern "C" {
#endif

// Operates on contiguous row-major input [rows, dim] and weight [dim]. The
// output dtype matches input_dtype. Device pointers and stream are HIP handles.
int redqueen_ds4_rmsnorm(
    const void* input,
    const void* weight,
    void* output,
    int rows,
    int dim,
    int input_dtype,
    int weight_dtype,
    float eps,
    void* stream_handle);

#ifdef __cplusplus
}
#endif

#endif  // REDQUEEN_DS4_RMSNORM_H
