"""Correctness tests for the pinned DeepSeek-V4.1-Flash-1017 RoPE contract."""

import ctypes
import os
import unittest

import torch

REFERENCE_COMMIT = "1bae0e1088993b9b82de08f59de5be5a34e42170"
F32 = 0
BF16 = 1


def make_frequencies(sequence, rotary_dim, seed, offset=0):
    generator = torch.Generator().manual_seed(seed)
    angles = torch.randn((sequence + offset, rotary_dim // 2), generator=generator) * 3.0
    rows = torch.stack((angles.cos(), angles.sin()), dim=-1)
    return rows.contiguous()


def reference_rope(x, frequencies, inverse=False):
    sequence = x.shape[1]
    rotary_dim = x.shape[-1]
    pairs = torch.view_as_complex(x.float().contiguous().view(*x.shape[:-1], rotary_dim // 2, 2))
    freqs_cis = torch.view_as_complex(frequencies.contiguous())
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, sequence, rotary_dim // 2)
    else:
        freqs_cis = freqs_cis.view(1, sequence, 1, rotary_dim // 2)
    return torch.view_as_real(pairs * freqs_cis).flatten(-2).to(x.dtype)


def bf16_ordered_bits(values):
    bits = values.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    return torch.where((bits & 0x8000) != 0, 0xFFFF - bits, bits + 0x8000)


class NativeRopeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.version.hip is None:
            raise RuntimeError("requires a ROCm GPU; refusing to report a skipped self-test as success")
        library_path = os.environ.get("DS4_HIP_LIBRARY")
        if not library_path:
            raise RuntimeError("DS4_HIP_LIBRARY must name the compiled HIP library")
        cls.library = ctypes.CDLL(library_path)
        cls.library.redqueen_ds4_rope_inplace.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_void_p,
        ]
        cls.library.redqueen_ds4_rope_inplace.restype = ctypes.c_int
        cls.library.redqueen_ds4_test_leave_pending_launch_error.argtypes = []
        cls.library.redqueen_ds4_test_leave_pending_launch_error.restype = ctypes.c_int

    def call_rope(self, x, frequencies, inverse=False):
        if x.ndim == 3:
            batch, sequence, rotary_dim = x.shape
            heads = 1
            batch_stride, sequence_stride, head_stride = x.stride(0), x.stride(1), 0
        elif x.ndim == 4:
            batch, sequence, heads, rotary_dim = x.shape
            batch_stride, sequence_stride, head_stride = x.stride(0), x.stride(1), x.stride(2)
        else:
            raise AssertionError(f"unsupported test rank {x.ndim}")
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        result = self.library.redqueen_ds4_rope_inplace(
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(frequencies.data_ptr()),
            batch,
            sequence,
            heads,
            rotary_dim,
            F32 if x.dtype == torch.float32 else BF16,
            int(inverse),
            batch_stride,
            sequence_stride,
            head_stride,
            stream,
        )
        self.assertEqual(result, 0)
        torch.cuda.synchronize()
        return x

    def assert_matches_reference(self, actual, expected):
        if actual.dtype == torch.bfloat16:
            distance = (bf16_ordered_bits(actual) - bf16_ordered_bits(expected)).abs()
            self.assertLessEqual(int(distance.max()), 1, "BF16 RoPE differs from reference by more than one ULP")
        else:
            torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-7)

    def test_bf16_and_fp32_match_complex_reference(self):
        generator = torch.Generator().manual_seed(41017)
        raw = torch.randn((2, 3, 4, 32), generator=generator)
        frequencies = make_frequencies(3, 32, seed=51017)
        frequencies_gpu = frequencies.cuda()

        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                x = raw.to(dtype).cuda()
                expected = reference_rope(x.cpu(), frequencies)
                actual = self.call_rope(x, frequencies_gpu).cpu()
                self.assert_matches_reference(actual, expected)

    def test_strided_tail_frequency_slice_and_inverse_rotation(self):
        generator = torch.Generator().manual_seed(61017)
        backing = torch.randn((2, 5, 3, 48), generator=generator).to(torch.bfloat16).cuda()
        original_backing = backing.cpu().clone()
        rotary_dim = 32
        prefix = 8
        tail = backing[..., prefix : prefix + rotary_dim]
        original_tail = tail.cpu().clone()
        frequencies_storage = make_frequencies(5, rotary_dim, seed=71017, offset=3)
        frequencies = frequencies_storage[2:7].contiguous()
        frequencies_gpu = frequencies_storage[2:7].cuda()

        expected_forward = reference_rope(original_tail, frequencies)
        self.call_rope(tail, frequencies_gpu)
        self.assert_matches_reference(tail.cpu(), expected_forward)

        after_forward = tail.cpu().clone()
        expected_inverse = reference_rope(after_forward, frequencies, inverse=True)
        self.call_rope(tail, frequencies_gpu, inverse=True)
        self.assert_matches_reference(tail.cpu(), expected_inverse)

        actual_backing = backing.cpu()
        self.assertTrue(torch.equal(actual_backing[..., :prefix], original_backing[..., :prefix]))
        self.assertTrue(torch.equal(actual_backing[..., prefix + rotary_dim :], original_backing[..., prefix + rotary_dim :]))

    def test_rank_three_single_head_with_more_than_256_pairs(self):
        generator = torch.Generator().manual_seed(81017)
        x = torch.randn((2, 3, 600), generator=generator).to(torch.bfloat16).cuda()
        frequencies = make_frequencies(3, 600, seed=91017)
        expected = reference_rope(x.cpu(), frequencies)
        self.call_rope(x, frequencies.cuda())
        self.assert_matches_reference(x.cpu(), expected)

    def test_rejects_invalid_arguments_without_mutating_input(self):
        x = torch.full((1, 2, 1, 4), 3.0, dtype=torch.float32, device="cuda")
        frequencies = torch.zeros((2, 2, 2), dtype=torch.float32, device="cuda")
        x_pointer = ctypes.c_void_p(x.data_ptr())
        frequency_pointer = ctypes.c_void_p(frequencies.data_ptr())
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        valid = (x_pointer, frequency_pointer, 1, 2, 1, 4, F32, 0, 8, 4, 4, stream)
        invalid_calls = (
            (None, *valid[1:]),
            (valid[0], None, *valid[2:]),
            (*valid[:2], 0, *valid[3:]),
            (*valid[:3], 0, *valid[4:]),
            (*valid[:4], 0, *valid[5:]),
            (*valid[:5], 0, *valid[6:]),
            (*valid[:5], 3, *valid[6:]),
            (*valid[:6], 9, *valid[7:]),
            (*valid[:7], 2, *valid[8:]),
            (*valid[:8], -1, *valid[9:]),
            (*valid[:8], 0, 1, 0, *valid[11:]),
            (valid[0], valid[1], 65536, 65536, 1, 2, F32, 0, 131072, 2, 0, valid[11]),
            (valid[0], valid[1], 2147483647, 1, 1, 2, F32, 0, 9223372036854775807, 2, 0, valid[11]),
        )
        for arguments in invalid_calls:
            with self.subTest(arguments=arguments[2:11]):
                self.assertNotEqual(self.library.redqueen_ds4_rope_inplace(*arguments), 0)
        torch.cuda.synchronize()
        self.assertTrue(torch.all(x == 3.0), "invalid call mutated its input")

    def test_pending_launch_error_is_returned_before_mutating_input(self):
        x = torch.ones((1, 1, 1, 4), dtype=torch.bfloat16, device="cuda")
        frequencies = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], dtype=torch.float32, device="cuda")
        original = x.clone()
        pending_error = self.library.redqueen_ds4_test_leave_pending_launch_error()
        self.assertNotEqual(pending_error, 0, "test helper did not leave a pending HIP launch error")

        result = self.library.redqueen_ds4_rope_inplace(
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(frequencies.data_ptr()),
            1,
            1,
            1,
            4,
            BF16,
            0,
            4,
            4,
            4,
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
        )
        self.assertEqual(result, pending_error)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(x, original), "kernel was enqueued despite returning a prior HIP error")


if __name__ == "__main__":
    unittest.main(verbosity=2)
