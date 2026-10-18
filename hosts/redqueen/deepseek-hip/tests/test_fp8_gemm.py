"""Correctness baseline for grouped FP8 GEMM on the pinned 1017 contract."""

import ctypes
import math
import os
import unittest

import torch

REFERENCE_COMMIT = "1bae0e1088993b9b82de08f59de5be5a34e42170"


def e8m0_values(encoded: torch.Tensor) -> torch.Tensor:
    return torch.tensor(
        [math.ldexp(1.0, int(value) - 127) if int(value) != 255 else float("nan") for value in encoded.flatten()],
        dtype=torch.float32,
    ).reshape(encoded.shape)


def reference_fp8_gemm(a: torch.Tensor, scale_a: torch.Tensor, b: torch.Tensor, scale_b: torch.Tensor, group_size: int):
    """Mirror the pinned kernel's per-K-group FP32 dot and scaling order."""
    m, k = a.shape
    n = b.shape[0]
    a_values = a.view(torch.float8_e4m3fn).float()
    b_values = b.view(torch.float8_e4m3fn).float()
    a_scales = e8m0_values(scale_a)
    b_scales = e8m0_values(scale_b)
    result = torch.zeros((m, n), dtype=torch.float32)
    k_groups = k // group_size
    for group in range(k_groups):
        start = group * group_size
        partial = a_values[:, start : start + group_size] @ b_values[:, start : start + group_size].T
        scale_a_group = a_scales[:, group].unsqueeze(1)
        scale_b_group = b_scales[:, group].index_select(
            0, torch.arange(n, dtype=torch.long) // group_size
        ).unsqueeze(0)
        result = result + ((partial * scale_a_group) * scale_b_group)
    return result


class NativeFp8GemmTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.version.hip is None:
            raise RuntimeError("requires a ROCm GPU; refusing to report a skipped self-test as success")
        library_path = os.environ.get("DS4_HIP_LIBRARY")
        if not library_path:
            raise RuntimeError("DS4_HIP_LIBRARY must name the compiled HIP library")
        cls.library = ctypes.CDLL(library_path)
        cls.library.redqueen_ds4_fp8_gemm.argtypes = [
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
            ctypes.c_void_p,
        ]
        cls.library.redqueen_ds4_fp8_gemm.restype = ctypes.c_int

    def call_gemm(self, a, scale_a, b, scale_b, group_size, output_type=torch.float32, guarded=False):
        m, k = a.shape
        n = b.shape[0]
        a_gpu = a.contiguous().cuda()
        scale_a_gpu = scale_a.contiguous().cuda()
        b_gpu = b.contiguous().cuda()
        scale_b_gpu = scale_b.contiguous().cuda()
        element_size = torch.empty((), dtype=output_type).element_size()
        if guarded:
            guard_bytes = 32
            storage = torch.full((m * n * element_size + 2 * guard_bytes,), 0xA5, dtype=torch.uint8, device="cuda")
            output = storage.narrow(0, guard_bytes, m * n * element_size).view(output_type).view(m, n)
        else:
            storage = None
            output = torch.empty((m, n), dtype=output_type, device="cuda")
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        result = self.library.redqueen_ds4_fp8_gemm(
            ctypes.c_void_p(a_gpu.data_ptr()),
            ctypes.c_void_p(scale_a_gpu.data_ptr()),
            ctypes.c_void_p(b_gpu.data_ptr()),
            ctypes.c_void_p(scale_b_gpu.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            m,
            n,
            k,
            group_size,
            0 if output_type == torch.float32 else 1,
            stream,
        )
        self.assertEqual(result, 0)
        torch.cuda.synchronize()
        if guarded:
            guard = torch.cat((storage[:guard_bytes], storage[-guard_bytes:])).cpu()
            self.assertTrue(torch.all(guard == 0xA5), "output kernel wrote outside its declared tensor")
        return output.cpu()

    def test_grouped_scales_cover_k_and_output_group_boundaries(self):
        for group_size, n in ((32, 128), (128, 256)):
            with self.subTest(group_size=group_size):
                m = 32
                k = group_size * 2
                a = torch.ones((m, k), dtype=torch.float8_e4m3fn).view(torch.uint8)
                b = torch.ones((n, k), dtype=torch.float8_e4m3fn).view(torch.uint8)
                scale_a = torch.tensor(
                    [[126 + ((row + group) % 5) for group in range(2)] for row in range(m)], dtype=torch.uint8
                )
                n_groups = (n + group_size - 1) // group_size
                scale_b = torch.tensor(
                    [[125 + ((row * 2 + group) % 7) for group in range(2)] for row in range(n_groups)],
                    dtype=torch.uint8,
                )

                actual = self.call_gemm(a, scale_a, b, scale_b, group_size)
                expected = reference_fp8_gemm(a, scale_a, b, scale_b, group_size)
                torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

                boundaries = {
                    boundary
                    for group in range(1, n_groups)
                    for boundary in (group * group_size - 1, group * group_size)
                }
                for column in boundaries:
                    self.assertTrue(torch.equal(actual[:, column], expected[:, column]))

    def test_e4m3_decoder_matches_every_encoding(self):
        group_size = 32
        m, n, k = 256, 1, group_size
        a = torch.zeros((m, k), dtype=torch.uint8)
        a[:, 0] = torch.arange(256, dtype=torch.int16).to(torch.uint8)
        b = torch.zeros((n, k), dtype=torch.uint8)
        b[0, 0] = torch.tensor(1.0, dtype=torch.float8_e4m3fn).view(torch.uint8)
        scale_a = torch.full((m, 1), 127, dtype=torch.uint8)
        scale_b = torch.full((1, 1), 127, dtype=torch.uint8)

        actual = self.call_gemm(a, scale_a, b, scale_b, group_size)
        expected = a[:, 0].contiguous().view(torch.float8_e4m3fn).float().reshape(m, 1)
        finite = torch.isfinite(expected[:, 0])
        self.assertTrue(torch.equal(actual[finite], expected[finite]))
        self.assertTrue(torch.equal(torch.isnan(actual[:, 0]), torch.isnan(expected[:, 0])))

    def test_e8m0_scale_decoder_matches_every_encoding(self):
        group_size = 32
        m, n, k = 256, 1, group_size
        a = torch.ones((m, k), dtype=torch.float8_e4m3fn).view(torch.uint8)
        b = torch.ones((n, k), dtype=torch.float8_e4m3fn).view(torch.uint8)
        scale_a = torch.arange(256, dtype=torch.int16).to(torch.uint8).reshape(m, 1)
        scale_b = torch.full((1, 1), 127, dtype=torch.uint8)

        actual = self.call_gemm(a, scale_a, b, scale_b, group_size)
        expected = (32.0 * e8m0_values(scale_a)).reshape(m, 1)
        finite = torch.isfinite(expected[:, 0])
        self.assertTrue(torch.equal(actual[finite], expected[finite]))
        self.assertTrue(torch.equal(torch.isnan(actual[:, 0]), torch.isnan(expected[:, 0])))

    def test_fp32_result_and_bf16_round_to_nearest_even(self):
        group_size = 32
        m, n, k = 2, 3, group_size
        a = torch.zeros((m, k), dtype=torch.float8_e4m3fn)
        b = torch.zeros((n, k), dtype=torch.float8_e4m3fn)
        a[:, 0] = 1.0
        a[:, 1] = 0.0625
        b[:, 0] = 1.0
        b[:, 1] = 0.0625
        a_bytes = a.view(torch.uint8)
        b_bytes = b.view(torch.uint8)
        scales_a = torch.full((m, 1), 127, dtype=torch.uint8)
        scales_b = torch.full((1, 1), 127, dtype=torch.uint8)

        fp32 = self.call_gemm(a_bytes, scales_a, b_bytes, scales_b, group_size, torch.float32, guarded=True)
        bf16 = self.call_gemm(a_bytes, scales_a, b_bytes, scales_b, group_size, torch.bfloat16, guarded=True)
        midpoint = 1.0 + 2.0**-8
        self.assertTrue(torch.equal(fp32, torch.full((m, n), midpoint, dtype=torch.float32)))
        self.assertTrue(torch.equal(bf16, torch.ones((m, n), dtype=torch.bfloat16)))

    def test_non_tile_aligned_shapes_are_bounds_safe(self):
        generator = torch.Generator(device="cpu").manual_seed(41017)
        group_size = 128
        m, n, k = 3, 129, 256
        a = ((torch.rand((m, k), generator=generator) * 8.0) - 4.0).to(torch.float8_e4m3fn).view(torch.uint8)
        b = ((torch.rand((n, k), generator=generator) * 8.0) - 4.0).to(torch.float8_e4m3fn).view(torch.uint8)
        scale_a = torch.randint(124, 131, (m, k // group_size), generator=generator, dtype=torch.uint8)
        scale_b = torch.randint(
            124, 131, ((n + group_size - 1) // group_size, k // group_size), generator=generator, dtype=torch.uint8
        )
        actual = self.call_gemm(a, scale_a, b, scale_b, group_size, guarded=True)
        expected = reference_fp8_gemm(a, scale_a, b, scale_b, group_size)
        torch.testing.assert_close(actual, expected, rtol=3.0e-5, atol=3.0e-5)

    def test_rejects_invalid_arguments_without_launching(self):
        m, n, k = 1, 1, 128
        a = torch.ones((m, k), dtype=torch.float8_e4m3fn).view(torch.uint8).cuda()
        b = torch.ones((n, k), dtype=torch.float8_e4m3fn).view(torch.uint8).cuda()
        scale_a = torch.full((m, 1), 127, dtype=torch.uint8, device="cuda")
        scale_b = torch.full((1, 1), 127, dtype=torch.uint8, device="cuda")
        output = torch.empty((m, n), dtype=torch.float32, device="cuda")
        pointers = [
            ctypes.c_void_p(a.data_ptr()),
            ctypes.c_void_p(scale_a.data_ptr()),
            ctypes.c_void_p(b.data_ptr()),
            ctypes.c_void_p(scale_b.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
        ]
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        common = (m, n, k, 128, 0, stream)
        invalid_calls = (
            (None, *pointers[1:], *common),
            (*pointers, m, n, k, 64, 0, stream),
            (*pointers, m, n, k - 1, 128, 0, stream),
            (*pointers, m, n, k, 128, 2, stream),
            (*pointers, 0, n, k, 128, 0, stream),
        )
        for arguments in invalid_calls:
            with self.subTest(arguments=arguments[5:10]):
                self.assertNotEqual(self.library.redqueen_ds4_fp8_gemm(*arguments), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
