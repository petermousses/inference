"""Correctness tests for the pinned DeepSeek-V4.1-Flash-1017 HC mixing kernel."""

import ctypes
import os
import unittest

import torch

REFERENCE_COMMIT = "1bae0e1088993b9b82de08f59de5be5a34e42170"


def reference_hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps):
    mix_width = (2 + hc_mult) * hc_mult
    mixes = mixes.reshape(-1, mix_width)
    pre = torch.sigmoid(mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2.0 * torch.sigmoid(
        mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    )

    comb_base = 2 * hc_mult
    comb = (
        mixes[:, comb_base:] * hc_scale[2] + hc_base[comb_base:]
    ).reshape(-1, hc_mult, hc_mult)
    row_max = comb.amax(dim=-1, keepdim=True)
    comb = torch.exp(comb - row_max)
    row_sum = comb.sum(dim=-1, keepdim=True)
    comb = comb / row_sum + eps
    col_sum = comb.sum(dim=-2, keepdim=True)
    comb = comb / (col_sum + eps)

    for _ in range(sinkhorn_iters - 1):
        row_sum = comb.sum(dim=-1, keepdim=True)
        comb = comb / (row_sum + eps)
        col_sum = comb.sum(dim=-2, keepdim=True)
        comb = comb / (col_sum + eps)
    return pre, post, comb


class NativeHcSinkhornTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.version.hip is None:
            raise RuntimeError("requires a ROCm GPU; refusing to report a skipped self-test as success")
        library_path = os.environ.get("DS4_HIP_LIBRARY")
        if not library_path:
            raise RuntimeError("DS4_HIP_LIBRARY must name the compiled HIP library")
        cls.library = ctypes.CDLL(library_path)
        cls.library.redqueen_ds4_hc_split_sinkhorn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_void_p,
        ]
        cls.library.redqueen_ds4_hc_split_sinkhorn.restype = ctypes.c_int
        cls.library.redqueen_ds4_test_leave_pending_launch_error.argtypes = []
        cls.library.redqueen_ds4_test_leave_pending_launch_error.restype = ctypes.c_int

    def call_kernel(self, mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters=20, eps=1.0e-6):
        mixes_gpu = mixes.contiguous().cuda()
        scale_gpu = hc_scale.contiguous().cuda()
        base_gpu = hc_base.contiguous().cuda()
        rows = mixes_gpu.shape[0]
        pre_gpu = torch.empty((rows, hc_mult), dtype=torch.float32, device="cuda")
        post_gpu = torch.empty_like(pre_gpu)
        comb_gpu = torch.empty((rows, hc_mult, hc_mult), dtype=torch.float32, device="cuda")
        result = self.library.redqueen_ds4_hc_split_sinkhorn(
            ctypes.c_void_p(mixes_gpu.data_ptr()),
            ctypes.c_void_p(scale_gpu.data_ptr()),
            ctypes.c_void_p(base_gpu.data_ptr()),
            ctypes.c_void_p(pre_gpu.data_ptr()),
            ctypes.c_void_p(post_gpu.data_ptr()),
            ctypes.c_void_p(comb_gpu.data_ptr()),
            rows,
            hc_mult,
            sinkhorn_iters,
            eps,
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
        )
        self.assertEqual(result, 0)
        torch.cuda.synchronize()
        return pre_gpu.cpu(), post_gpu.cpu(), comb_gpu.cpu()

    def assert_matches_reference(self, mixes, scale, base, hc_mult, sinkhorn_iters, eps):
        expected = reference_hc_split_sinkhorn(mixes, scale, base, hc_mult, sinkhorn_iters, eps)
        actual = self.call_kernel(mixes, scale, base, hc_mult, sinkhorn_iters, eps)
        for actual_value, expected_value in zip(actual, expected):
            torch.testing.assert_close(actual_value, expected_value, rtol=2.0e-5, atol=2.0e-6)

    def test_matches_reference_for_default_1017_shape_and_iterations(self):
        generator = torch.Generator().manual_seed(1017042)
        hc_mult = 4
        mix_width = (2 + hc_mult) * hc_mult
        mixes = torch.randn((2, 7, mix_width), generator=generator).reshape(-1, mix_width)
        scale = torch.tensor([0.75, 1.25, 0.5], dtype=torch.float32)
        base = torch.randn((mix_width,), generator=generator)
        self.assert_matches_reference(mixes, scale, base, hc_mult, 20, 1.0e-6)

    def test_supports_hc_bounds_and_one_sinkhorn_iteration(self):
        generator = torch.Generator().manual_seed(2017042)
        for hc_mult in (1, 2, 16):
            with self.subTest(hc_mult=hc_mult):
                mix_width = (2 + hc_mult) * hc_mult
                mixes = torch.randn((3, mix_width), generator=generator)
                scale = torch.tensor([1.0, 0.5, 1.5], dtype=torch.float32)
                base = torch.randn((mix_width,), generator=generator)
                self.assert_matches_reference(mixes, scale, base, hc_mult, 1, 0.0)

    def test_stable_softmax_handles_large_finite_logits(self):
        hc_mult = 4
        mix_width = (2 + hc_mult) * hc_mult
        mixes = torch.linspace(-80.0, 80.0, 5 * mix_width).reshape(5, mix_width)
        scale = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        base = torch.linspace(80.0, -80.0, mix_width)
        self.assert_matches_reference(mixes, scale, base, hc_mult, 20, 1.0e-6)

    def test_rejects_invalid_arguments_without_mutating_outputs(self):
        hc_mult = 4
        mix_width = (2 + hc_mult) * hc_mult
        mixes = torch.zeros((2, mix_width), dtype=torch.float32, device="cuda")
        scale = torch.ones((3,), dtype=torch.float32, device="cuda")
        base = torch.ones((mix_width,), dtype=torch.float32, device="cuda")
        pre = torch.full((2, hc_mult), 17.0, dtype=torch.float32, device="cuda")
        post = torch.full_like(pre, 19.0)
        comb = torch.full((2, hc_mult, hc_mult), 23.0, dtype=torch.float32, device="cuda")
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        valid = [
            ctypes.c_void_p(mixes.data_ptr()),
            ctypes.c_void_p(scale.data_ptr()),
            ctypes.c_void_p(base.data_ptr()),
            ctypes.c_void_p(pre.data_ptr()),
            ctypes.c_void_p(post.data_ptr()),
            ctypes.c_void_p(comb.data_ptr()),
            2,
            hc_mult,
            20,
            1.0e-6,
            stream,
        ]

        invalid_calls = []
        for pointer_index in range(6):
            arguments = valid.copy()
            arguments[pointer_index] = None
            invalid_calls.append(arguments)
        for rows, multiplier, iterations, eps in (
            (0, 4, 20, 1.0e-6),
            (2, 0, 20, 1.0e-6),
            (2, 17, 20, 1.0e-6),
            (2, 4, 0, 1.0e-6),
            (2, 4, 20, -1.0e-6),
            (2, 4, 20, float("nan")),
        ):
            arguments = valid.copy()
            arguments[6:10] = [rows, multiplier, iterations, eps]
            invalid_calls.append(arguments)
        duplicate_output = valid.copy()
        duplicate_output[4] = duplicate_output[3]
        invalid_calls.append(duplicate_output)

        for arguments in invalid_calls:
            self.assertNotEqual(self.library.redqueen_ds4_hc_split_sinkhorn(*arguments), 0)
        torch.cuda.synchronize()
        self.assertTrue(torch.all(pre == 17.0))
        self.assertTrue(torch.all(post == 19.0))
        self.assertTrue(torch.all(comb == 23.0))

    def test_pending_launch_error_is_returned_before_writing_outputs(self):
        hc_mult = 4
        mix_width = (2 + hc_mult) * hc_mult
        mixes = torch.zeros((1, mix_width), dtype=torch.float32, device="cuda")
        scale = torch.ones((3,), dtype=torch.float32, device="cuda")
        base = torch.zeros((mix_width,), dtype=torch.float32, device="cuda")
        pre = torch.full((1, hc_mult), 17.0, dtype=torch.float32, device="cuda")
        post = torch.full_like(pre, 19.0)
        comb = torch.full((1, hc_mult, hc_mult), 23.0, dtype=torch.float32, device="cuda")
        pending_error = self.library.redqueen_ds4_test_leave_pending_launch_error()
        self.assertNotEqual(pending_error, 0, "test helper did not leave a pending HIP launch error")

        result = self.library.redqueen_ds4_hc_split_sinkhorn(
            ctypes.c_void_p(mixes.data_ptr()),
            ctypes.c_void_p(scale.data_ptr()),
            ctypes.c_void_p(base.data_ptr()),
            ctypes.c_void_p(pre.data_ptr()),
            ctypes.c_void_p(post.data_ptr()),
            ctypes.c_void_p(comb.data_ptr()),
            1,
            hc_mult,
            20,
            1.0e-6,
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
        )
        self.assertEqual(result, pending_error)
        torch.cuda.synchronize()
        self.assertTrue(torch.all(pre == 17.0))
        self.assertTrue(torch.all(post == 19.0))
        self.assertTrue(torch.all(comb == 23.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
