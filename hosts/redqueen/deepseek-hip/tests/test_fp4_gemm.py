"""Correctness baseline for the pinned 1017 FP8 x FP4 GEMM contract."""

import ctypes
import math
import os
import unittest

import torch

import kernel

REFERENCE_COMMIT = "1bae0e1088993b9b82de08f59de5be5a34e42170"
SCALE_E8M0 = 0
SCALE_F32 = 1
OUTPUT_F32 = 0
OUTPUT_BF16 = 1

FP4_E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def e8m0_values(encoded: torch.Tensor) -> torch.Tensor:
    return torch.tensor(
        [math.ldexp(1.0, int(value) - 127) if int(value) != 255 else float("nan") for value in encoded.flatten()],
        dtype=torch.float32,
    ).reshape(encoded.shape)


def unpack_fp4_x2(packed: torch.Tensor) -> torch.Tensor:
    raw = packed.to(torch.int16)
    low = raw & 0x0F
    high = raw >> 4
    codes = torch.stack((low, high), dim=-1).reshape(packed.shape[0], -1)
    return FP4_E2M1_VALUES[codes.long()]


def scale_values(encoded: torch.Tensor) -> torch.Tensor:
    return e8m0_values(encoded) if encoded.dtype == torch.uint8 else encoded.to(torch.float32)


def reference_fp4_gemm(
    a: torch.Tensor,
    scale_a: torch.Tensor,
    b_packed: torch.Tensor,
    scale_b: torch.Tensor,
    activation_group_size: int,
) -> torch.Tensor:
    """Mirror 32-wide FP4 weight groups and pinned scale/accumulation order."""
    m, k = a.shape
    n = b_packed.shape[0]
    a_values = a.contiguous().view(torch.float8_e4m3fn).float()
    b_values = unpack_fp4_x2(b_packed)
    a_scales = scale_values(scale_a)
    b_scales = scale_values(scale_b)
    result = torch.zeros((m, n), dtype=torch.float32)
    activation_groups_per_weight_group = activation_group_size // 32

    for group in range(k // 32):
        start = group * 32
        partial = a_values[:, start : start + 32] @ b_values[:, start : start + 32].T
        activation_group = group // activation_groups_per_weight_group
        scaled = (partial * a_scales[:, activation_group].unsqueeze(1)) * b_scales[:, group].unsqueeze(0)
        result = result + scaled
    return result


class NativeFp4GemmTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.version.hip is None:
            raise RuntimeError("requires a ROCm GPU; refusing to report a skipped self-test as success")
        library_path = os.environ.get("DS4_HIP_LIBRARY")
        if not library_path:
            raise RuntimeError("DS4_HIP_LIBRARY must name the compiled HIP library")
        cls.library = ctypes.CDLL(library_path)
        cls.library.redqueen_ds4_fp4_gemm.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        cls.library.redqueen_ds4_fp4_gemm.restype = ctypes.c_int
        cls.library.redqueen_ds4_test_leave_pending_launch_error.argtypes = []
        cls.library.redqueen_ds4_test_leave_pending_launch_error.restype = ctypes.c_int

    def call_gemm(
        self,
        a,
        scale_a,
        b_packed,
        scale_b,
        activation_group_size,
        output_type=torch.float32,
        guarded=False,
        output_fill=None,
    ):
        m, k = a.shape
        n = b_packed.shape[0]
        self.assertEqual(b_packed.shape[1] * 2, k)
        a_gpu = a.contiguous().cuda()
        scale_a_gpu = scale_a.contiguous().cuda()
        b_gpu = b_packed.contiguous().cuda()
        scale_b_gpu = scale_b.contiguous().cuda()
        scale_type = SCALE_E8M0 if scale_a.dtype == torch.uint8 else SCALE_F32
        self.assertEqual(scale_type, SCALE_E8M0 if scale_b.dtype == torch.uint8 else SCALE_F32)

        element_size = torch.empty((), dtype=output_type).element_size()
        if guarded:
            guard_bytes = 32
            storage = torch.full(
                (m * n * element_size + 2 * guard_bytes,), 0xA5, dtype=torch.uint8, device="cuda"
            )
            output = storage.narrow(0, guard_bytes, m * n * element_size).view(output_type).view(m, n)
        else:
            storage = None
            output = torch.empty((m, n), dtype=output_type, device="cuda")
            if output_fill is not None:
                output.fill_(output_fill)

        result = self.library.redqueen_ds4_fp4_gemm(
            ctypes.c_void_p(a_gpu.data_ptr()),
            ctypes.c_void_p(scale_a_gpu.data_ptr()),
            ctypes.c_void_p(b_gpu.data_ptr()),
            ctypes.c_void_p(scale_b_gpu.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            m,
            n,
            k,
            activation_group_size,
            scale_type,
            OUTPUT_F32 if output_type == torch.float32 else OUTPUT_BF16,
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
        )
        self.assertEqual(result, 0)
        torch.cuda.synchronize()
        if guarded:
            guard = torch.cat((storage[:guard_bytes], storage[-guard_bytes:])).cpu()
            self.assertTrue(torch.all(guard == 0xA5), "output kernel wrote outside its declared tensor")
        return output.cpu()

    def test_all_fp4_codes_in_both_packed_nibbles(self):
        k = 32
        m = k
        a = torch.zeros((m, k), dtype=torch.uint8)
        a[torch.arange(m), torch.arange(m)] = 0x38  # E4M3 1.0 isolates each FP4 code.
        codes = torch.arange(16, dtype=torch.uint8).repeat_interleave(2)
        b_packed = (codes[0::2] | (codes[1::2] << 4)).reshape(1, -1)
        scale_a = torch.full((m, 1), 127, dtype=torch.uint8)
        scale_b = torch.full((1, 1), 127, dtype=torch.uint8)

        actual = self.call_gemm(a, scale_a, b_packed, scale_b, 32).reshape(-1)
        expected = FP4_E2M1_VALUES[codes.long()]
        self.assertTrue(torch.equal(actual, expected))

    def test_e4m3_activation_decoder_matches_every_encoding(self):
        m, n, k = 256, 1, 32
        a = torch.zeros((m, k), dtype=torch.uint8)
        a[:, 0] = torch.arange(256, dtype=torch.int16).to(torch.uint8)
        b_codes = torch.zeros((n, k), dtype=torch.uint8)
        b_codes[:, 0] = 0x02  # E2M1 1.0.
        b_packed = (b_codes[:, 0::2] | (b_codes[:, 1::2] << 4)).contiguous()
        scale_a = torch.full((m, 1), 127, dtype=torch.uint8)
        scale_b = torch.full((n, 1), 127, dtype=torch.uint8)

        actual = self.call_gemm(a, scale_a, b_packed, scale_b, 32).reshape(m)
        expected = a[:, 0].contiguous().view(torch.float8_e4m3fn).float()
        finite = torch.isfinite(expected)
        self.assertTrue(torch.equal(actual[finite], expected[finite]))
        self.assertTrue(torch.equal(torch.isnan(actual), torch.isnan(expected)))

    def test_group_scales_cover_activation_and_weight_groups(self):
        generator = torch.Generator(device="cpu").manual_seed(41019)
        for activation_group_size, scale_type in ((32, SCALE_E8M0), (128, SCALE_E8M0), (32, SCALE_F32), (128, SCALE_F32)):
            with self.subTest(activation_group_size=activation_group_size, scale_type=scale_type):
                m, n, k = 3, 129, activation_group_size * 2
                a_values = ((torch.rand((m, k), generator=generator) * 4.0) - 2.0).to(torch.float8_e4m3fn)
                a = a_values.view(torch.uint8)
                fp4_codes = torch.randint(0, 16, (n, k), generator=generator, dtype=torch.uint8)
                b_packed = (fp4_codes[:, 0::2] | (fp4_codes[:, 1::2] << 4)).contiguous()

                if scale_type == SCALE_E8M0:
                    scale_a = torch.randint(124, 131, (m, k // activation_group_size), generator=generator, dtype=torch.uint8)
                    scale_b = torch.randint(124, 131, (n, k // 32), generator=generator, dtype=torch.uint8)
                else:
                    scale_a = torch.rand((m, k // activation_group_size), generator=generator) + 0.5
                    scale_b = torch.rand((n, k // 32), generator=generator) + 0.5

                actual = self.call_gemm(a, scale_a, b_packed, scale_b, activation_group_size, guarded=True)
                expected = reference_fp4_gemm(a, scale_a, b_packed, scale_b, activation_group_size)
                torch.testing.assert_close(actual, expected, rtol=3.0e-5, atol=3.0e-5)

    def test_tiled_exact_matches_scalar_reference_bitwise(self):
        generator = torch.Generator(device="cpu").manual_seed(4101921)
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float32)
            for activation_group_size in (32, 128):
                for scale_type in (SCALE_E8M0, SCALE_F32):
                    with self.subTest(activation_group_size=activation_group_size, scale_type=scale_type):
                        m, n, k = 17, 33, 256
                        a = torch.randn((m, k), generator=generator).to(torch.float8_e4m3fn)
                        codes = torch.randint(0, 16, (n, k), generator=generator, dtype=torch.uint8)
                        b_packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
                        if scale_type == SCALE_E8M0:
                            scale_a_raw = torch.randint(
                                124,
                                131,
                                (m, k // activation_group_size),
                                generator=generator,
                                dtype=torch.uint8,
                            )
                            scale_b_raw = torch.randint(
                                124,
                                131,
                                (n, k // 32),
                                generator=generator,
                                dtype=torch.uint8,
                            )
                            scale_a = scale_a_raw.cuda().view(torch.float8_e8m0fnu)
                            scale_b = scale_b_raw.cuda().view(torch.float8_e8m0fnu)
                        else:
                            scale_a_raw = torch.rand(
                                (m, k // activation_group_size), generator=generator
                            ) + 0.5
                            scale_b_raw = torch.rand((n, k // 32), generator=generator) + 0.5
                            scale_a = scale_a_raw.cuda()
                            scale_b = scale_b_raw.cuda()

                        scalar = self.call_gemm(
                            a.view(torch.uint8),
                            scale_a_raw,
                            b_packed,
                            scale_b_raw,
                            activation_group_size,
                        )
                        tiled = kernel.fp4_gemm(
                            a.cuda(),
                            scale_a,
                            b_packed.cuda().view(torch.float4_e2m1fn_x2),
                            scale_b,
                            scale_dtype=(torch.float8_e8m0fnu if scale_type == SCALE_E8M0 else torch.float32),
                            act_block_size=activation_group_size,
                        )
                        torch.cuda.synchronize()
                        self.assertTrue(torch.equal(tiled.cpu(), scalar))
        finally:
            torch.set_default_dtype(previous_dtype)

    def test_large_output_uses_two_dimensional_grid_and_writes_tail(self):
        m, n, k = 65536, 256, 32
        a = torch.full((m, k), 0x38, dtype=torch.uint8)
        scale_a = torch.full((m, 1), 127, dtype=torch.uint8)
        b_packed = torch.full((n, k // 2), 0x22, dtype=torch.uint8)
        scale_b = torch.full((n, 1), 127, dtype=torch.uint8)
        actual = self.call_gemm(
            a,
            scale_a,
            b_packed,
            scale_b,
            32,
            output_fill=-17.0,
        )
        self.assertTrue(torch.equal(actual, torch.full_like(actual, 32.0)))

    def test_fp32_result_and_bf16_round_to_nearest_even(self):
        m, n, k = 2, 1, 32
        a = torch.zeros((m, k), dtype=torch.uint8)
        a[:, 0] = 0x38  # E4M3 1.0.
        b_codes = torch.zeros((n, k), dtype=torch.uint8)
        b_codes[:, 0] = 0x02  # E2M1 1.0 in the low nibble.
        b_packed = (b_codes[:, 0::2] | (b_codes[:, 1::2] << 4)).contiguous()
        scale_a = torch.tensor([[1.00390625], [1.01171875]], dtype=torch.float32)
        scale_b = torch.ones((n, k // 32), dtype=torch.float32)

        fp32 = self.call_gemm(a, scale_a, b_packed, scale_b, 32, torch.float32, guarded=True)
        bf16 = self.call_gemm(a, scale_a, b_packed, scale_b, 32, torch.bfloat16, guarded=True)
        self.assertTrue(torch.equal(fp32[:, 0], torch.tensor([1.00390625, 1.01171875])))
        self.assertTrue(torch.equal(bf16[:, 0], torch.tensor([1.0, 1.015625], dtype=torch.bfloat16)))

    def test_rejects_invalid_arguments_without_launching(self):
        m, n, k = 1, 1, 128
        a = torch.full((m, k), 0x38, dtype=torch.uint8, device="cuda")
        scale_a = torch.full((m, k // 32), 127, dtype=torch.uint8, device="cuda")
        b = torch.full((n, k // 2), 0x22, dtype=torch.uint8, device="cuda")
        scale_b = torch.full((n, k // 32), 127, dtype=torch.uint8, device="cuda")
        output = torch.full((m, n), 17.0, dtype=torch.float32, device="cuda")
        pointers = [
            ctypes.c_void_p(a.data_ptr()),
            ctypes.c_void_p(scale_a.data_ptr()),
            ctypes.c_void_p(b.data_ptr()),
            ctypes.c_void_p(scale_b.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
        ]
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        invalid_calls = (
            (None, *pointers[1:], m, n, k, 32, SCALE_E8M0, OUTPUT_F32, stream),
            (pointers[0], None, *pointers[2:], m, n, k, 32, SCALE_E8M0, OUTPUT_F32, stream),
            (*pointers[:2], None, *pointers[3:], m, n, k, 32, SCALE_E8M0, OUTPUT_F32, stream),
            (*pointers[:3], None, *pointers[4:], m, n, k, 32, SCALE_E8M0, OUTPUT_F32, stream),
            (*pointers[:4], None, m, n, k, 32, SCALE_E8M0, OUTPUT_F32, stream),
            (*pointers, m, n, k, 64, SCALE_E8M0, OUTPUT_F32, stream),
            (*pointers, m, n, k - 1, 32, SCALE_E8M0, OUTPUT_F32, stream),
            (*pointers, m, n, k // 2, 128, SCALE_E8M0, OUTPUT_F32, stream),
            (*pointers, m, n, k, 32, 2, OUTPUT_F32, stream),
            (*pointers, m, n, k, 32, SCALE_E8M0, 2, stream),
            (*pointers, 0, n, k, 32, SCALE_E8M0, OUTPUT_F32, stream),
            (*pointers, 2**31 - 1, 2**31 - 1, k, 32, SCALE_E8M0, OUTPUT_F32, stream),
            (*pointers, 2_000_000, 2_000_000, k, 32, SCALE_E8M0, OUTPUT_BF16, stream),
        )
        for arguments in invalid_calls:
            with self.subTest(arguments=arguments[5:11]):
                self.assertNotEqual(self.library.redqueen_ds4_fp4_gemm(*arguments), 0)
        torch.cuda.synchronize()
        self.assertTrue(torch.all(output == 17.0), "invalid call enqueued a kernel")

    def test_pending_launch_error_is_returned_before_enqueuing(self):
        m, n, k = 1, 1, 32
        a = torch.full((m, k), 0x38, dtype=torch.uint8, device="cuda")
        scale_a = torch.full((m, 1), 127, dtype=torch.uint8, device="cuda")
        b = torch.full((n, k // 2), 0x22, dtype=torch.uint8, device="cuda")
        scale_b = torch.full((n, 1), 127, dtype=torch.uint8, device="cuda")
        output = torch.full((m, n), 17.0, dtype=torch.float32, device="cuda")
        pending_error = self.library.redqueen_ds4_test_leave_pending_launch_error()
        self.assertNotEqual(pending_error, 0, "test helper did not leave a pending HIP launch error")

        result = self.library.redqueen_ds4_fp4_gemm(
            ctypes.c_void_p(a.data_ptr()),
            ctypes.c_void_p(scale_a.data_ptr()),
            ctypes.c_void_p(b.data_ptr()),
            ctypes.c_void_p(scale_b.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            m,
            n,
            k,
            32,
            SCALE_E8M0,
            OUTPUT_F32,
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
        )
        self.assertEqual(result, pending_error)
        torch.cuda.synchronize()
        self.assertTrue(torch.all(output == 17.0), "kernel was enqueued despite returning a prior HIP error")


if __name__ == "__main__":
    unittest.main(verbosity=2)
