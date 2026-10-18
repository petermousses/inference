"""Correctness baseline for the pinned DeepSeek-V4.1-Flash-1017 RMSNorm contract."""

import ctypes
import math
import os
import unittest

import torch

REFERENCE_COMMIT = "1bae0e1088993b9b82de08f59de5be5a34e42170"
F32 = 0
BF16 = 1


def reference_rmsnorm(input_tensor: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = input_tensor.dtype
    values = input_tensor.float()
    mean_square = values.square().mean(dim=-1, keepdim=True)
    result = (values * torch.rsqrt(mean_square + eps)) * weight.float()
    return result.to(input_dtype)


def bf16_ordered_bits(values: torch.Tensor) -> torch.Tensor:
    bits = values.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    return torch.where((bits & 0x8000) != 0, 0xFFFF - bits, bits + 0x8000)


class NativeRmsnormTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.version.hip is None:
            raise RuntimeError("requires a ROCm GPU; refusing to report a skipped self-test as success")
        library_path = os.environ.get("DS4_HIP_LIBRARY")
        if not library_path:
            raise RuntimeError("DS4_HIP_LIBRARY must name the compiled HIP library")
        cls.library = ctypes.CDLL(library_path)
        cls.library.redqueen_ds4_rmsnorm.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_void_p,
        ]
        cls.library.redqueen_ds4_rmsnorm.restype = ctypes.c_int

    def call_rmsnorm(self, input_tensor, weight, eps, guarded=True):
        input_shape = tuple(input_tensor.shape)
        dim = input_shape[-1]
        rows = math.prod(input_shape[:-1])
        input_gpu = input_tensor.contiguous().reshape(rows, dim).cuda()
        weight_gpu = weight.contiguous().cuda()
        output_dtype = input_tensor.dtype
        output_bytes = rows * dim * torch.empty((), dtype=output_dtype).element_size()
        if guarded:
            guard_bytes = 32
            storage = torch.full((output_bytes + 2 * guard_bytes,), 0xA5, dtype=torch.uint8, device="cuda")
            output = storage.narrow(0, guard_bytes, output_bytes).view(output_dtype).view(rows, dim)
        else:
            guard_bytes = 0
            storage = None
            output = torch.empty((rows, dim), dtype=output_dtype, device="cuda")

        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        result = self.library.redqueen_ds4_rmsnorm(
            ctypes.c_void_p(input_gpu.data_ptr()),
            ctypes.c_void_p(weight_gpu.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            rows,
            dim,
            F32 if input_tensor.dtype == torch.float32 else BF16,
            F32 if weight.dtype == torch.float32 else BF16,
            eps,
            stream,
        )
        self.assertEqual(result, 0)
        torch.cuda.synchronize()
        if guarded:
            guard = torch.cat((storage[:guard_bytes], storage[-guard_bytes:])).cpu()
            self.assertTrue(torch.all(guard == 0xA5), "RMSNorm wrote outside its declared output")
        return output.cpu().reshape(input_shape)

    def assert_matches_reference(self, actual, expected):
        if actual.dtype == torch.bfloat16:
            distance = (bf16_ordered_bits(actual) - bf16_ordered_bits(expected)).abs()
            self.assertLessEqual(int(distance.max()), 1, "BF16 result differs from reference by more than one ULP")
        else:
            torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-6)

    def test_reference_formula_for_supported_dtype_pairs(self):
        generator = torch.Generator(device="cpu").manual_seed(41017)
        raw = torch.randn((2, 3, 33), generator=generator, dtype=torch.float32)
        weights = torch.randn((33,), generator=generator, dtype=torch.float32)
        cases = (
            (torch.bfloat16, torch.bfloat16, 1.0e-20),
            (torch.bfloat16, torch.float32, 1.0e-6),
            (torch.float32, torch.bfloat16, 1.0e-20),
            (torch.float32, torch.float32, 1.0e-6),
        )
        for input_dtype, weight_dtype, eps in cases:
            with self.subTest(input_dtype=input_dtype, weight_dtype=weight_dtype, eps=eps):
                input_tensor = raw.to(input_dtype)
                weight = weights.to(weight_dtype)
                actual = self.call_rmsnorm(input_tensor, weight, eps)
                expected = reference_rmsnorm(input_tensor.cuda(), weight.cuda(), eps).cpu()
                self.assert_matches_reference(actual, expected)

    def test_irregular_tails_and_model_hidden_width(self):
        generator = torch.Generator(device="cpu").manual_seed(51017)
        for dim in (1, 3, 31, 33, 127, 257, 5120):
            with self.subTest(dim=dim):
                if dim == 33:
                    # This catches division by a padded block width such as 64.
                    input_tensor = torch.ones((3, dim), dtype=torch.float32)
                    weight = torch.ones((dim,), dtype=torch.float32)
                else:
                    input_tensor = torch.randn((3, dim), generator=generator, dtype=torch.float32)
                    weight = torch.randn((dim,), generator=generator, dtype=torch.float32)
                actual = self.call_rmsnorm(input_tensor, weight, 1.0e-6)
                expected = reference_rmsnorm(input_tensor.cuda(), weight.cuda(), 1.0e-6).cpu()
                self.assert_matches_reference(actual, expected)

    def test_bf16_final_rounding_and_no_intermediate_rounding(self):
        tie_cases = (
            (1.0 + 2.0**-8, 1.0),
            (1.0 + 3.0 * 2.0**-8, 1.015625),
        )
        for weight_value, expected_value in tie_cases:
            with self.subTest(weight_value=weight_value):
                tie_input = torch.ones((1, 1), dtype=torch.bfloat16)
                tie_weight = torch.tensor([weight_value], dtype=torch.float32)
                tie_output = self.call_rmsnorm(tie_input, tie_weight, 1.0e-20)
                expected = torch.tensor([[expected_value]], dtype=torch.bfloat16)
                self.assertTrue(torch.equal(tie_output, expected))

        input_tensor = torch.tensor([[1.0, 0.0]], dtype=torch.bfloat16)
        weight = torch.tensor([0.7099, 1.0], dtype=torch.float32)
        actual = self.call_rmsnorm(input_tensor, weight, 1.0e-20)
        expected = torch.tensor([[1.0078125, 0.0]], dtype=torch.bfloat16)
        self.assertTrue(torch.equal(actual, expected))

    def test_nan_and_infinity_propagate_by_row(self):
        input_tensor = torch.tensor([[float("nan"), 1.0], [float("inf"), 2.0]], dtype=torch.float32)
        weight = torch.ones((2,), dtype=torch.float32)
        actual = self.call_rmsnorm(input_tensor, weight, 1.0e-6)
        self.assertTrue(torch.isnan(actual[0]).all())
        self.assertTrue(torch.isnan(actual[1, 0]))
        self.assertEqual(float(actual[1, 1]), 0.0)

    def test_rejects_invalid_arguments_without_launching(self):
        input_gpu = torch.ones((2, 3), dtype=torch.float32, device="cuda")
        weight_gpu = torch.ones((3,), dtype=torch.float32, device="cuda")
        output_gpu = torch.empty_like(input_gpu)
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        input_pointer = ctypes.c_void_p(input_gpu.data_ptr())
        weight_pointer = ctypes.c_void_p(weight_gpu.data_ptr())
        output_pointer = ctypes.c_void_p(output_gpu.data_ptr())
        valid = (input_pointer, weight_pointer, output_pointer, 2, 3, F32, F32, 1.0e-6, stream)

        invalid_calls = (
            (None, *valid[1:]),
            (valid[0], None, *valid[2:]),
            (*valid[:2], None, *valid[3:]),
            (*valid[:3], 0, *valid[4:]),
            (*valid[:3], -1, *valid[4:]),
            (*valid[:4], 0, *valid[5:]),
            (*valid[:4], -1, *valid[5:]),
            (*valid[:5], 7, *valid[6:]),
            (*valid[:6], 7, *valid[7:]),
            (*valid[:7], 0.0, valid[8]),
            (*valid[:7], -1.0e-6, valid[8]),
            (*valid[:7], float("inf"), valid[8]),
            (*valid[:7], float("nan"), valid[8]),
        )
        for arguments in invalid_calls:
            with self.subTest(arguments=arguments[3:8]):
                self.assertNotEqual(self.library.redqueen_ds4_rmsnorm(*arguments), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
