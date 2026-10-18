#ifndef REDQUEEN_DS4_FP4_QUANT_H
#define REDQUEEN_DS4_FP4_QUANT_H

enum redqueen_ds4_fp4_quant_scale_type {
  REDQUEEN_DS4_FP4_QUANT_SCALE_E8M0 = 0,
  REDQUEEN_DS4_FP4_QUANT_SCALE_E4M3 = 1,
};

#ifdef __cplusplus
extern "C" {
#endif

// Quantizes contiguous row-major BF16 input in groups of 16 or 32. E8M0
// scales use one uint8 per group; E4M3 scales use one float8_e4m3fn byte.
// Input values must be finite.
// With inplace == 0, output contains packed E2M1 values, with even columns in
// the low nibble and odd columns in the high nibble. With inplace != 0,
// output must equal input and receives FP4 quantize/dequantize results in BF16.
// Scale storage must be disjoint from input and output. In non-inplace mode,
// input and output must not overlap. Allocation sizes are the caller's job.
int redqueen_ds4_fp4_quantize_activation(
    const void* input_bf16,
    void* output,
    void* output_scales,
    int rows,
    int cols,
    int block_size,
    int scale_type,
    int inplace,
    void* stream_handle);

#ifdef __cplusplus
}
#endif

#endif  // REDQUEEN_DS4_FP4_QUANT_H
