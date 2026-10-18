"""Correctness baseline for the pinned DeepSeek-V4.1-Flash-1017 sparse attention contract."""

import ctypes
import math
import os
import unittest

import torch

REFERENCE_COMMIT = "1bae0e1088993b9b82de08f59de5be5a34e42170"


def reference_sparse_attn(q, kv, sink, indices, scale):
    batch, queries, heads, dim = q.shape
    keys = kv.shape[1]
    topk = indices.shape[-1]
    result = torch.zeros((batch, queries, heads, dim), dtype=torch.bfloat16)

    for batch_index in range(batch):
        for query_index in range(queries):
            selected = indices[batch_index, query_index]
            for head_index in range(heads):
                running_max = -1.0e30
                denominator = 0.0
                accumulator = torch.zeros((dim,), dtype=torch.float32)
                any_valid = False
                query = q[batch_index, query_index, head_index].float()

                for base in range(0, topk, 64):
                    tile_ids = selected[base : base + 64]
                    tile_size = tile_ids.numel()
                    valid = (tile_ids >= 0) & (tile_ids < keys)
                    safe_ids = tile_ids.clamp(0, keys - 1)
                    tile_values = kv[batch_index, safe_ids].float()
                    tile_values = torch.where(valid[:, None], tile_values, torch.zeros_like(tile_values))
                    scores = torch.sum(tile_values * query[None, :], dim=-1) * scale
                    scores = torch.where(valid, scores, torch.full_like(scores, float("-inf")))
                    tile_max = float(scores.max()) if tile_size else float("-inf")
                    new_max = max(running_max, tile_max)
                    previous_scale = math.exp(running_max - new_max)
                    probabilities = torch.where(valid, torch.exp(scores - new_max), torch.zeros_like(scores))
                    denominator = denominator * previous_scale + float(probabilities.sum())
                    rounded_probabilities = probabilities.to(torch.bfloat16).float()
                    accumulator = accumulator * previous_scale + rounded_probabilities @ tile_values
                    running_max = new_max
                    any_valid = any_valid or bool(valid.any())

                if any_valid:
                    denominator += math.exp(float(sink[head_index]) - running_max)
                    result[batch_index, query_index, head_index] = (accumulator / denominator).to(torch.bfloat16)

    return result


def bf16_ordered_bits(values):
    bits = values.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    return torch.where((bits & 0x8000) != 0, 0xFFFF - bits, bits + 0x8000)


class NativeSparseAttnTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.version.hip is None:
            raise RuntimeError("requires a ROCm GPU; refusing to report a skipped self-test as success")
        library_path = os.environ.get("DS4_HIP_LIBRARY")
        if not library_path:
            raise RuntimeError("DS4_HIP_LIBRARY must name the compiled HIP library")
        cls.library = ctypes.CDLL(library_path)
        cls.library.redqueen_ds4_sparse_attn.argtypes = [
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
            ctypes.c_float,
            ctypes.c_void_p,
        ]
        cls.library.redqueen_ds4_sparse_attn.restype = ctypes.c_int
        cls.library.redqueen_ds4_test_leave_pending_launch_error.argtypes = []
        cls.library.redqueen_ds4_test_leave_pending_launch_error.restype = ctypes.c_int

    def call_sparse_attn(self, q, kv, sink, indices, scale, guarded=True):
        q_gpu = q.contiguous().cuda()
        kv_gpu = kv.contiguous().cuda()
        sink_gpu = sink.contiguous().cuda()
        indices_gpu = indices.to(torch.int32).contiguous().cuda()
        batch, queries, heads, dim = q.shape
        keys = kv.shape[1]
        topk = indices.shape[-1]

        if guarded:
            guard_bytes = 64
            output_bytes = batch * queries * heads * dim * 2
            storage = torch.full((output_bytes + 2 * guard_bytes,), 0xA5, dtype=torch.uint8, device="cuda")
            output = storage.narrow(0, guard_bytes, output_bytes).view(torch.bfloat16).view(
                batch, queries, heads, dim
            )
        else:
            guard_bytes = 0
            storage = None
            output = torch.empty((batch, queries, heads, dim), dtype=torch.bfloat16, device="cuda")

        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        result = self.library.redqueen_ds4_sparse_attn(
            ctypes.c_void_p(q_gpu.data_ptr()),
            ctypes.c_void_p(kv_gpu.data_ptr()),
            ctypes.c_void_p(sink_gpu.data_ptr()),
            ctypes.c_void_p(indices_gpu.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            batch,
            queries,
            heads,
            dim,
            keys,
            topk,
            scale,
            stream,
        )
        self.assertEqual(result, 0)
        torch.cuda.synchronize()
        if guarded:
            guard = torch.cat((storage[:guard_bytes], storage[-guard_bytes:])).cpu()
            self.assertTrue(torch.all(guard == 0xA5), "sparse attention wrote outside its declared output")
        return output.cpu()

    def assert_matches_reference(self, actual, expected, max_ulp=2):
        distance = (bf16_ordered_bits(actual) - bf16_ordered_bits(expected)).abs()
        self.assertLessEqual(int(distance.max()), max_ulp, "BF16 result differs from reference beyond tolerance")

    def test_sink_is_denominator_only_and_repeated_indices_count_twice(self):
        q = torch.zeros((1, 1, 2, 3), dtype=torch.bfloat16)
        kv = torch.tensor([[[3.0, 6.0, 9.0], [3.0, 6.0, 9.0]]], dtype=torch.bfloat16)
        sink = torch.tensor([0.0, 0.0], dtype=torch.float32)

        one_key = torch.tensor([[[0]]], dtype=torch.int32)
        actual = self.call_sparse_attn(q, kv, sink, one_key, 1.0)
        expected = torch.tensor([[[[1.5, 3.0, 4.5], [1.5, 3.0, 4.5]]]], dtype=torch.bfloat16)
        self.assertTrue(torch.equal(actual, expected))

        repeated = torch.tensor([[[0, 0]]], dtype=torch.int32)
        actual = self.call_sparse_attn(q, kv, sink, repeated, 1.0)
        expected = torch.tensor([[[[2.0, 4.0, 6.0], [2.0, 4.0, 6.0]]]], dtype=torch.bfloat16)
        self.assertTrue(torch.equal(actual, expected))

    def test_all_missing_and_out_of_range_indices_return_zero(self):
        q = torch.ones((1, 2, 3, 5), dtype=torch.bfloat16)
        kv = torch.randn((1, 4, 5), dtype=torch.bfloat16, generator=torch.Generator().manual_seed(1001))
        sink = torch.zeros((3,), dtype=torch.float32)
        indices = torch.tensor(
            [[[-1, -2, 4], [100, -1, -900]]],
            dtype=torch.int32,
        )
        actual = self.call_sparse_attn(q, kv, sink, indices, 0.25)
        self.assertTrue(torch.equal(actual, torch.zeros_like(actual)))

    def test_batch_head_isolation_and_irregular_dimension(self):
        generator = torch.Generator().manual_seed(41017)
        q = torch.randn((2, 2, 3, 5), generator=generator).to(torch.bfloat16)
        kv = torch.randn((2, 7, 5), generator=generator).to(torch.bfloat16)
        sink = torch.tensor([-0.3, 0.2, 0.7], dtype=torch.float32)
        indices = torch.tensor(
            [
                [[0, 1, 1, -1, 5], [6, 3, -1, 0, 2]],
                [[4, 2, 5, 6, 0], [-1, 3, 1, 1, 9]],
            ],
            dtype=torch.int32,
        )
        actual = self.call_sparse_attn(q, kv, sink, indices, 0.37)
        expected = reference_sparse_attn(q, kv, sink, indices, 0.37)
        self.assert_matches_reference(actual, expected)

    def test_64_key_tiles_rescale_and_mask_tail_entries(self):
        q = torch.zeros((1, 2, 2, 3), dtype=torch.bfloat16)
        q[0, 0, :, 0] = 1.0
        q[0, 1, :, 0] = 1.0
        kv = torch.zeros((1, 66, 3), dtype=torch.bfloat16)
        kv[0, :, 0] = torch.linspace(0.0, 1.0, 66).to(torch.bfloat16)
        kv[0, :, 1] = torch.linspace(-2.0, 2.0, 66).to(torch.bfloat16)
        kv[0, :, 2] = torch.arange(66).to(torch.bfloat16)
        kv[0, 65, 0] = 12.0
        sink = torch.tensor([0.2, -0.1], dtype=torch.float32)
        first = list(range(65)) + [65] + [999] * 63
        second = [-1] * 64 + [65] + [-2] * 64
        indices = torch.tensor([[first, second]], dtype=torch.int32)
        actual = self.call_sparse_attn(q, kv, sink, indices, 0.8)
        expected = reference_sparse_attn(q, kv, sink, indices, 0.8)
        self.assert_matches_reference(actual, expected, max_ulp=2)

    def test_multiple_tiles_preserve_online_softmax_and_bf16_probability_rounding(self):
        generator = torch.Generator().manual_seed(51017)
        q = torch.randn((1, 1, 2, 33), generator=generator).to(torch.bfloat16)
        kv = torch.randn((1, 130, 33), generator=generator).to(torch.bfloat16)
        sink = torch.tensor([0.15, -0.45], dtype=torch.float32)
        indices = torch.arange(130, dtype=torch.int32).view(1, 1, 130)
        actual = self.call_sparse_attn(q, kv, sink, indices, 0.11)
        expected = reference_sparse_attn(q, kv, sink, indices, 0.11)
        self.assert_matches_reference(actual, expected, max_ulp=2)

    def test_rejects_invalid_arguments_without_launching(self):
        q = torch.ones((1, 1, 2, 3), dtype=torch.bfloat16, device="cuda")
        kv = torch.ones((1, 2, 3), dtype=torch.bfloat16, device="cuda")
        sink = torch.zeros((2,), dtype=torch.float32, device="cuda")
        indices = torch.zeros((1, 1, 1), dtype=torch.int32, device="cuda")
        output = torch.full((1, 1, 2, 3), 17.0, dtype=torch.bfloat16, device="cuda")
        valid = (
            ctypes.c_void_p(q.data_ptr()),
            ctypes.c_void_p(kv.data_ptr()),
            ctypes.c_void_p(sink.data_ptr()),
            ctypes.c_void_p(indices.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            1,
            1,
            2,
            3,
            2,
            1,
            0.5,
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
        )
        invalid_calls = (
            (None, *valid[1:]),
            (valid[0], None, *valid[2:]),
            (*valid[:2], None, *valid[3:]),
            (*valid[:3], None, *valid[4:]),
            (*valid[:4], None, *valid[5:]),
            (*valid[:5], 0, *valid[6:]),
            (*valid[:5], 1, 0, *valid[7:]),
            (*valid[:5], 1, 1, 0, *valid[8:]),
            (*valid[:5], 1, 1, 2, 0, *valid[9:]),
            (*valid[:5], 1, 1, 2, 3, 0, *valid[10:]),
            (*valid[:11], 0.0, valid[12]),
            (*valid[:11], float("inf"), valid[12]),
        )
        for arguments in invalid_calls:
            with self.subTest(arguments=arguments[5:12]):
                self.assertNotEqual(self.library.redqueen_ds4_sparse_attn(*arguments), 0)
        torch.cuda.synchronize()
        self.assertTrue(torch.all(output == 17.0), "invalid call enqueued a kernel")

    def test_pending_launch_error_is_returned_before_enqueuing(self):
        q = torch.ones((1, 1, 2, 3), dtype=torch.bfloat16, device="cuda")
        kv = torch.ones((1, 2, 3), dtype=torch.bfloat16, device="cuda")
        sink = torch.zeros((2,), dtype=torch.float32, device="cuda")
        indices = torch.zeros((1, 1, 1), dtype=torch.int32, device="cuda")
        output = torch.full((1, 1, 2, 3), 17.0, dtype=torch.bfloat16, device="cuda")
        pending_error = self.library.redqueen_ds4_test_leave_pending_launch_error()
        self.assertNotEqual(pending_error, 0, "test helper did not leave a pending HIP launch error")

        result = self.library.redqueen_ds4_sparse_attn(
            ctypes.c_void_p(q.data_ptr()),
            ctypes.c_void_p(kv.data_ptr()),
            ctypes.c_void_p(sink.data_ptr()),
            ctypes.c_void_p(indices.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            1,
            1,
            2,
            3,
            2,
            1,
            0.5,
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
        )
        self.assertEqual(result, pending_error)
        torch.cuda.synchronize()
        self.assertTrue(torch.all(output == 17.0), "kernel was enqueued despite returning a prior HIP error")


if __name__ == "__main__":
    unittest.main(verbosity=2)
