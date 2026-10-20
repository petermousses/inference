"""Correctness baseline for the pinned 1017 FP4 activation quantizer."""

import ctypes
import math
import os
import unittest

import torch

REFERENCE_COMMIT = "1bae0e1088993b9b82de08f59de5be5a34e42170"
SCALE_E8M0 = 0
SCALE_E4M3 = 1
FP4_MAGNITUDES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def fp4_rne_codes(values: torch.Tensor) -> torch.Tensor:
    magnitude = values.float().abs()
    code = torch.where(
        magnitude <= 0.25,
        0,
        torch.where(
            magnitude < 0.75,
            1,
            torch.where(
                magnitude <= 1.25,
                2,
                torch.where(
                    magnitude < 1.75,
                    3,
                    torch.where(
                        magnitude <= 2.5,
                        4,
                        torch.where(magnitude < 3.5, 5, torch.where(magnitude <= 5.0, 6, 7)),
                    ),
                ),
            ),
        ),
    ).to(torch.uint8)
    return code | (torch.signbit(values).to(torch.uint8) << 3)


def fp4_decode_codes(codes: torch.Tensor) -> torch.Tensor:
    magnitude = FP4_MAGNITUDES[(codes & 0x07).long()]
    return torch.where((codes & 0x08) != 0, -magnitude, magnitude)


def reference_fp4_quantize(values: torch.Tensor, block_size: int, scale_type: int):
    """Mirror inference/kernel.py scale selection, E2M1 cast, and BF16 writeback."""
    values_bf16 = values.detach().cpu().contiguous().to(torch.bfloat16)
    rows, cols = values_bf16.shape
    grouped = values_bf16.float().reshape(rows, cols // block_size, block_size)
    amax = grouped.abs().amax(dim=-1)

    if scale_type == SCALE_E8M0:
        amax = amax.clamp(min=6.0 * (2.0**-126))
        normalized_amax = amax * (1.0 / 6.0)
        bits = normalized_amax.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        exponent = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(torch.int64)
        biased_exponent = exponent + 127
        scale_bytes = biased_exponent.to(torch.uint8)
        scale_bits = (biased_exponent << 23).to(torch.int32)
        scale_values = scale_bits.view(torch.float32)
    else:
        amax = amax.clamp(min=6.0 * (2.0**-9))
        # TileLang's finite E4M3 cast saturates overflow to 448; torch.to
        # instead emits NaN for values above the format's finite range.
        scale_fp8 = (amax / 6.0).clamp(max=448.0).to(torch.float8_e4m3fn)
        scale_bytes = scale_fp8.contiguous().view(torch.uint8)
        scale_values = scale_fp8.float()

    normalized = torch.clamp(grouped / scale_values.unsqueeze(-1), -6.0, 6.0)
    codes = fp4_rne_codes(normalized)
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).reshape(rows, cols // 2)
    dequantized = (fp4_decode_codes(codes) * scale_values.unsqueeze(-1)).reshape(rows, cols).to(torch.bfloat16)
    return packed, scale_bytes.reshape(rows, cols // block_size), dequantized


class NativeFp4QuantizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.version.hip is None:
            raise RuntimeError("requires a ROCm GPU; refusing to report a skipped self-test as success")
        library_path = os.environ.get("DS4_HIP_LIBRARY")
        if not library_path:
            raise RuntimeError("DS4_HIP_LIBRARY must name the compiled HIP library")
        cls.library = ctypes.CDLL(library_path)
        cls.library.redqueen_ds4_fp4_quantize_activation.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        cls.library.redqueen_ds4_fp4_quantize_activation.restype = ctypes.c_int
        cls.library.redqueen_ds4_test_leave_pending_launch_error.argtypes = []
        cls.library.redqueen_ds4_test_leave_pending_launch_error.restype = ctypes.c_int

    def quantize(self, values, block_size, scale_type, inplace=False, guarded=False):
        values_bf16 = values.detach().cpu().contiguous().to(torch.bfloat16)
        rows, cols = values_bf16.shape
        source = values_bf16.cuda()
        scales = torch.empty((rows, cols // block_size), dtype=torch.uint8, device="cuda")

        if inplace:
            output = source
            storage = None
        elif guarded:
            guard_size = 32
            storage = torch.full((rows * cols // 2 + 2 * guard_size,), 0xA5, dtype=torch.uint8, device="cuda")
            output = storage.narrow(0, guard_size, rows * cols // 2)
        else:
            storage = None
            output = torch.empty((rows * cols // 2,), dtype=torch.uint8, device="cuda")

        result = self.library.redqueen_ds4_fp4_quantize_activation(
            ctypes.c_void_p(source.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            ctypes.c_void_p(scales.data_ptr()),
            rows,
            cols,
            block_size,
            scale_type,
            int(inplace),
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
        )
        self.assertEqual(result, 0)
        torch.cuda.synchronize()

        if guarded:
            guard = torch.cat((storage[:guard_size], storage[-guard_size:])).cpu()
            self.assertTrue(torch.all(guard == 0xA5), "packing kernel wrote outside its output tensor")

        result_output = source.cpu() if inplace else output.reshape(rows, cols // 2).cpu()
        return result_output, scales.cpu()

    def test_packed_quantization_matches_reference_for_supported_modes(self):
        generator = torch.Generator(device="cpu").manual_seed(1017041)
        random_values = ((torch.rand((7, 256), generator=generator) * 8000.0) - 4000.0).to(torch.bfloat16)
        edge_values = torch.zeros((3, 256), dtype=torch.bfloat16)
        edge_values[1] = torch.linspace(-2688.0, 2688.0, 256).to(torch.bfloat16)
        edge_values[2, 0] = 6.0
        edge_values[2, 16] = 6.015625
        edge_values[2, 32] = 0.01171875
        values = torch.cat((edge_values, random_values), dim=0).contiguous()

        for block_size in (16, 32):
            for scale_type in (SCALE_E8M0, SCALE_E4M3):
                with self.subTest(block_size=block_size, scale_type=scale_type):
                    expected_packed, expected_scales, _ = reference_fp4_quantize(values, block_size, scale_type)
                    actual_packed, actual_scales = self.quantize(values, block_size, scale_type, guarded=True)
                    self.assertTrue(torch.equal(actual_scales, expected_scales), "scale bytes differ from reference")
                    mismatch = torch.nonzero(actual_packed != expected_packed, as_tuple=False)
                    self.assertEqual(
                        mismatch.shape[0],
                        0,
                        f"packed bytes differ at {tuple(mismatch[0].tolist()) if mismatch.numel() else None}",
                    )

    def test_large_group_count_uses_two_dimensional_grid(self):
        values = torch.zeros((65536, 16), dtype=torch.bfloat16)
        values[-1] = torch.linspace(-12.0, 12.0, 16).to(torch.bfloat16)
        expected_packed, expected_scales, _ = reference_fp4_quantize(values, 16, SCALE_E8M0)
        actual_packed, actual_scales = self.quantize(values, 16, SCALE_E8M0)
        self.assertTrue(torch.equal(actual_scales, expected_scales))
        self.assertTrue(torch.equal(actual_packed, expected_packed))

    def test_inplace_dequantization_matches_reference_for_model_paths(self):
        generator = torch.Generator(device="cpu").manual_seed(41017)
        values = ((torch.rand((9, 128), generator=generator) * 24.0) - 12.0).to(torch.bfloat16)
        values[0].zero_()
        values[1, 0] = -0.0
        values[2, :8] = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0], dtype=torch.bfloat16)

        for block_size, scale_type in ((32, SCALE_E8M0), (16, SCALE_E4M3)):
            with self.subTest(block_size=block_size, scale_type=scale_type):
                _, expected_scales, expected_values = reference_fp4_quantize(values, block_size, scale_type)
                actual_values, actual_scales = self.quantize(values, block_size, scale_type, inplace=True)
                self.assertTrue(torch.equal(actual_scales, expected_scales))
                self.assertTrue(
                    torch.equal(
                        actual_values.contiguous().view(torch.uint16),
                        expected_values.contiguous().view(torch.uint16),
                    ),
                    "in-place BF16 values differ at the bit level",
                )

    def test_all_fp4_rounding_boundaries_and_signed_codes(self):
        boundaries = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
        exact_codes = (0, 2, 2, 4, 4, 6, 6)
        rows = []
        expected = []
        for boundary_index, (boundary, tie_code) in enumerate(zip(boundaries, exact_codes)):
            ulp = math.ldexp(1.0, math.floor(math.log2(boundary)) - 7)
            for magnitude, code in (
                (boundary - ulp, boundary_index),
                (boundary, tie_code),
                (boundary + ulp, boundary_index + 1),
            ):
                row = torch.zeros(16, dtype=torch.bfloat16)
                row[0] = magnitude
                row[1] = -magnitude
                row[-1] = 6.0  # fixes the E8M0 scale at one
                rows.append(row)
                expected.append((code, code | 0x08))

        values = torch.stack(rows)
        packed, scales = self.quantize(values, 16, SCALE_E8M0)
        self.assertTrue(torch.equal(scales, torch.full_like(scales, 127)))
        actual_low_high = packed[:, 0]
        expected_bytes = torch.tensor([low | (high << 4) for low, high in expected], dtype=torch.uint8)
        self.assertTrue(torch.equal(actual_low_high, expected_bytes))

        signed_zero_row = torch.zeros((1, 16), dtype=torch.bfloat16)
        signed_zero_row[0, 0] = 0.0
        signed_zero_row[0, 1] = -0.0
        signed_zero_row[0, 15] = 6.0  # fixes the E8M0 scale at one
        signed_zero_packed, signed_zero_scales = self.quantize(signed_zero_row, 16, SCALE_E8M0)
        self.assertEqual(signed_zero_scales[0, 0].item(), 127)
        self.assertEqual(signed_zero_packed[0, 0].item(), 0x80)

    def test_zero_groups_use_nonzero_minimum_scales(self):
        values = torch.zeros((2, 64), dtype=torch.bfloat16)
        for block_size in (16, 32):
            for scale_type in (SCALE_E8M0, SCALE_E4M3):
                with self.subTest(block_size=block_size, scale_type=scale_type):
                    packed, scales = self.quantize(values, block_size, scale_type)
                    self.assertTrue(torch.equal(packed, torch.zeros_like(packed)))
                    self.assertTrue(torch.equal(scales, torch.ones_like(scales)))

    def test_e4m3_scale_rounding_ties_and_finite_saturation(self):
        maxima = (
            3.171875,
            3.1875,
            3.203125,
            3.546875,
            3.5625,
            3.578125,
            2688.0,
            3000.0,
        )
        expected_scale_bytes = torch.tensor([0x30, 0x30, 0x31, 0x31, 0x32, 0x32, 0x7E, 0x7E], dtype=torch.uint8)
        values = torch.zeros((len(maxima), 16), dtype=torch.bfloat16)
        values[:, 0] = torch.tensor(maxima, dtype=torch.bfloat16)

        _, scales = self.quantize(values, 16, SCALE_E4M3)
        self.assertTrue(torch.equal(scales[:, 0], expected_scale_bytes))
        _, expected_scales, _ = reference_fp4_quantize(values, 16, SCALE_E4M3)
        self.assertTrue(torch.equal(scales, expected_scales))

    def test_rejects_invalid_arguments_without_launching(self):
        rows, cols = 2, 64
        source = torch.full((rows, cols), 3.0, dtype=torch.bfloat16, device="cuda")
        output = torch.full((rows * cols // 2,), 0xA5, dtype=torch.uint8, device="cuda")
        scales = torch.full((rows, cols // 16), 0x5A, dtype=torch.uint8, device="cuda")
        source_ptr = ctypes.c_void_p(source.data_ptr())
        output_ptr = ctypes.c_void_p(output.data_ptr())
        scales_ptr = ctypes.c_void_p(scales.data_ptr())
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        base = (source_ptr, output_ptr, scales_ptr)
        invalid_calls = (
            (None, output_ptr, scales_ptr, rows, cols, 16, SCALE_E8M0, 0, stream),
            (source_ptr, None, scales_ptr, rows, cols, 16, SCALE_E8M0, 0, stream),
            (source_ptr, output_ptr, None, rows, cols, 16, SCALE_E8M0, 0, stream),
            (*base, 0, cols, 16, SCALE_E8M0, 0, stream),
            (*base, rows, cols - 1, 16, SCALE_E8M0, 0, stream),
            (*base, rows, cols, 64, SCALE_E8M0, 0, stream),
            (*base, rows, cols, 16, 2, 0, stream),
            (*base, rows, cols, 16, SCALE_E8M0, 2, stream),
            (source_ptr, source_ptr, scales_ptr, rows, cols, 16, SCALE_E8M0, 0, stream),
            (source_ptr, output_ptr, scales_ptr, rows, cols, 16, SCALE_E8M0, 1, stream),
            (source_ptr, output_ptr, source_ptr, rows, cols, 16, SCALE_E8M0, 0, stream),
            (source_ptr, output_ptr, output_ptr, rows, cols, 16, SCALE_E8M0, 0, stream),
            (*base, 2**31 - 1, 64, 16, SCALE_E8M0, 0, stream),
        )
        for arguments in invalid_calls:
            with self.subTest(arguments=arguments[3:8]):
                self.assertNotEqual(self.library.redqueen_ds4_fp4_quantize_activation(*arguments), 0)

        torch.cuda.synchronize()
        self.assertTrue(torch.all(output == 0xA5), "invalid call modified packed output")
        self.assertTrue(torch.all(scales == 0x5A), "invalid call modified scale output")
        self.assertTrue(torch.all(source == 3.0), "invalid call modified input")

    def test_pending_launch_error_is_returned_before_enqueuing(self):
        values = torch.full((1, 32), 1.0, dtype=torch.bfloat16, device="cuda")
        scales = torch.full((1, 1), 0x5A, dtype=torch.uint8, device="cuda")
        pending_error = self.library.redqueen_ds4_test_leave_pending_launch_error()
        self.assertNotEqual(pending_error, 0, "test helper did not leave a pending HIP launch error")
        result = self.library.redqueen_ds4_fp4_quantize_activation(
            ctypes.c_void_p(values.data_ptr()),
            ctypes.c_void_p(values.data_ptr()),
            ctypes.c_void_p(scales.data_ptr()),
            1,
            32,
            32,
            SCALE_E8M0,
            1,
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
        )
        self.assertEqual(result, pending_error)
        torch.cuda.synchronize()
        self.assertTrue(torch.all(values == 1.0), "kernel was enqueued despite returning a prior HIP error")
        self.assertTrue(torch.all(scales == 0x5A), "kernel wrote scales despite returning a prior HIP error")


if __name__ == "__main__":
    unittest.main(verbosity=2)
