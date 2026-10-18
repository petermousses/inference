"""Parity check for the native HIP port of the pinned 1017 reference quantizer."""

import ctypes
import math
import os
import unittest

import torch

REFERENCE_COMMIT = "1bae0e1088993b9b82de08f59de5be5a34e42170"
FP8_E4M3_MAX = 448.0


def reference_quantize(input_bf16: torch.Tensor, group_size: int):
    """Mirror inference/kernel.py act_quant with its UE8M0 power-of-two rounding."""
    rows, cols = input_bf16.shape
    assert cols % group_size == 0
    grouped = input_bf16.float().reshape(rows, cols // group_size, group_size)
    amax = grouped.abs().amax(dim=-1).clamp(min=1.0e-4)
    normalized_max = amax * (1.0 / FP8_E4M3_MAX)
    bits = normalized_max.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    exponent = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(torch.int64)
    scale_bits = ((exponent + 127) << 23).to(torch.int32)
    scale = scale_bits.view(torch.float32)
    quantized = torch.clamp(grouped / scale.unsqueeze(-1), -FP8_E4M3_MAX, FP8_E4M3_MAX)
    quantized_bytes = quantized.to(torch.float8_e4m3fn).contiguous().view(torch.uint8).reshape(rows, cols)
    e8m0_bytes = (exponent + 127).to(torch.uint8)
    return quantized_bytes, e8m0_bytes


class NativeQuantizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.version.hip is None:
            raise RuntimeError("requires a ROCm GPU; refusing to report a skipped self-test as success")
        library_path = os.environ.get("DS4_HIP_LIBRARY")
        if not library_path:
            raise RuntimeError("DS4_HIP_LIBRARY must name the compiled HIP library")
        cls.library = ctypes.CDLL(library_path)
        cls.library.redqueen_ds4_quantize_activation.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        cls.library.redqueen_ds4_quantize_activation.restype = ctypes.c_int
        cls.library.redqueen_ds4_decode_e8m0.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        cls.library.redqueen_ds4_decode_e8m0.restype = ctypes.c_int

    def quantize(self, values: torch.Tensor, group_size: int):
        rows, cols = values.shape
        source = values.contiguous().to(device="cuda")
        quantized = torch.empty((rows, cols), dtype=torch.uint8, device="cuda")
        scales = torch.empty((rows, cols // group_size), dtype=torch.uint8, device="cuda")
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        result = self.library.redqueen_ds4_quantize_activation(
            ctypes.c_void_p(source.data_ptr()),
            ctypes.c_void_p(quantized.data_ptr()),
            ctypes.c_void_p(scales.data_ptr()),
            rows,
            cols,
            group_size,
            stream,
        )
        self.assertEqual(result, 0)
        torch.cuda.synchronize()
        return quantized.cpu(), scales.cpu()

    def test_grouped_activation_quantization_matches_reference(self):
        generator = torch.Generator(device="cpu").manual_seed(1017)
        random_values = ((torch.rand((7, 256), generator=generator) * 2000.0) - 1000.0).to(torch.bfloat16)
        edge_values = torch.zeros((6, 256), dtype=torch.bfloat16)
        edge_values[0, 0] = -0.0
        edge_values[1].fill_(2.5)
        edge_values[2] = torch.tensor(
            [-900, -448, -447, -2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2, 447, 448, 449, 900],
            dtype=torch.bfloat16,
        ).repeat(16)
        edge_values[3] = torch.linspace(-8192, 8192, 256).to(torch.bfloat16)
        rounding_ties = torch.tensor(
            [448.0, 1.0625, 1.1875, 1.3125, -1.0625, -1.1875, -1.3125, -448.0],
            dtype=torch.bfloat16,
        )
        edge_values[4] = rounding_ties.repeat(32)
        edge_values[5] = rounding_ties.repeat(32)
        edge_values[5, 0] = 449.0
        values = torch.cat((edge_values, random_values), dim=0).contiguous()

        for group_size in (32, 128):
            with self.subTest(group_size=group_size):
                expected_quantized, expected_scales = reference_quantize(values, group_size)
                actual_quantized, actual_scales = self.quantize(values, group_size)
                self.assertTrue(torch.equal(actual_scales, expected_scales))
                if not torch.equal(actual_quantized, expected_quantized):
                    mismatch = torch.nonzero(actual_quantized != expected_quantized, as_tuple=False)
                    first = tuple(mismatch[0].tolist())
                    actual_values = actual_quantized.view(torch.float8_e4m3fn).float()
                    expected_values = expected_quantized.view(torch.float8_e4m3fn).float()
                    self.fail(
                        f"{mismatch.shape[0]} byte mismatches; first={first}, "
                        f"actual={actual_quantized[first].item()} ({actual_values[first].item()}), "
                        f"expected={expected_quantized[first].item()} ({expected_values[first].item()})"
                    )

    def test_rejects_unsupported_group_size(self):
        source = torch.zeros((1, 128), dtype=torch.bfloat16, device="cuda")
        quantized = torch.empty((1, 128), dtype=torch.uint8, device="cuda")
        scales = torch.empty((1, 2), dtype=torch.uint8, device="cuda")
        result = self.library.redqueen_ds4_quantize_activation(
            ctypes.c_void_p(source.data_ptr()),
            ctypes.c_void_p(quantized.data_ptr()),
            ctypes.c_void_p(scales.data_ptr()),
            1,
            128,
            64,
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
        )
        self.assertNotEqual(result, 0)

    def test_rejects_invalid_arguments_without_launching(self):
        source = torch.zeros((1, 128), dtype=torch.bfloat16, device="cuda")
        quantized = torch.empty((1, 128), dtype=torch.uint8, device="cuda")
        scales = torch.empty((1, 4), dtype=torch.uint8, device="cuda")
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

        invalid_calls = (
            (None, quantized, scales, 1, 128, 32),
            (source, quantized, scales, 1, 129, 32),
            (source, quantized, scales, 0, 128, 32),
        )
        for input_tensor, output_tensor, scale_tensor, rows, cols, group_size in invalid_calls:
            pointers = (
                ctypes.c_void_p(input_tensor.data_ptr()) if input_tensor is not None else None,
                ctypes.c_void_p(output_tensor.data_ptr()),
                ctypes.c_void_p(scale_tensor.data_ptr()),
            )
            result = self.library.redqueen_ds4_quantize_activation(
                *pointers,
                rows,
                cols,
                group_size,
                stream,
            )
            self.assertNotEqual(result, 0)

    def test_e8m0_decode_matches_all_finite_values_and_nan(self):
        source = torch.arange(256, dtype=torch.int16).to(torch.uint8).cuda()
        output = torch.empty(256, dtype=torch.float32, device="cuda")
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        result = self.library.redqueen_ds4_decode_e8m0(
            ctypes.c_void_p(source.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            256,
            stream,
        )
        self.assertEqual(result, 0)
        actual = output.cpu()
        expected = torch.tensor(
            [math.ldexp(1.0, -127) if value == 0 else math.ldexp(1.0, value - 127) for value in range(255)]
            + [float("nan")],
            dtype=torch.float32,
        )
        self.assertTrue(torch.equal(actual[:255], expected[:255]))
        self.assertTrue(torch.isnan(actual[255]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
