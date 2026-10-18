"""Exercise the Python API used by the pinned 1017 model graph."""

import os
import unittest

import torch

import kernel


class NativeKernelModuleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.version.hip is None:
            raise RuntimeError("requires a ROCm GPU; refusing to report a skipped self-test as success")
        if not os.environ.get("DS4_HIP_LIBRARY"):
            raise RuntimeError("DS4_HIP_LIBRARY must name the compiled HIP library")
        kernel._library()
        cls.e8m0 = torch.float8_e8m0fnu

    def test_act_quant_returns_reference_shapes_and_inplace_dequantization(self):
        values = torch.linspace(-24, 24, 3 * 64, dtype=torch.float32).reshape(1, 3, 64).to(torch.bfloat16).cuda()
        quantized, scales = kernel.act_quant(values, 32, "ue8m0", self.e8m0)
        self.assertEqual(quantized.shape, values.shape)
        self.assertEqual(quantized.dtype, torch.float8_e4m3fn)
        self.assertEqual(scales.shape, (1, 3, 2))
        self.assertEqual(scales.dtype, self.e8m0)

        expected = (
            quantized.float().reshape(3, 64)
            * scales.float().reshape(3, 2).repeat_interleave(32, dim=-1)
        ).to(torch.bfloat16).reshape_as(values)
        inplace = values.clone()
        result = kernel.act_quant(inplace, 32, "ue8m0", self.e8m0, inplace=True)
        self.assertIs(result, inplace)
        self.assertTrue(torch.equal(inplace.cpu(), expected.cpu()))

    def test_act_quant_rejects_non_ue8m0_modes(self):
        values = torch.zeros((2, 64), dtype=torch.bfloat16, device="cuda")
        with self.assertRaises(NotImplementedError):
            kernel.act_quant(values)

    def test_act_quant_handles_noncontiguous_inputs_and_inplace_copyback(self):
        values = torch.linspace(-24, 24, 64 * 4, dtype=torch.float32, device="cuda")
        values = values.to(torch.bfloat16).reshape(64, 4).transpose(0, 1)
        self.assertFalse(values.is_contiguous())
        expected, expected_scales = kernel.act_quant(
            values.contiguous(), 32, "ue8m0", self.e8m0
        )
        quantized, scales = kernel.act_quant(values, 32, "ue8m0", self.e8m0)
        torch.testing.assert_close(quantized, expected, rtol=0, atol=0)
        torch.testing.assert_close(scales.float(), expected_scales.float(), rtol=0, atol=0)

        inplace = torch.linspace(-24, 24, 64 * 4, dtype=torch.float32, device="cuda")
        inplace = inplace.to(torch.bfloat16).reshape(64, 4).transpose(0, 1)
        self.assertFalse(inplace.is_contiguous())
        expected_inplace = inplace.contiguous()
        kernel.act_quant(expected_inplace, 32, "ue8m0", self.e8m0, inplace=True)
        result = kernel.act_quant(inplace, 32, "ue8m0", self.e8m0, inplace=True)
        self.assertIs(result, inplace)
        torch.testing.assert_close(inplace, expected_inplace, rtol=0, atol=0)

    def test_fp4_activation_quantization_supports_packed_and_inplace_paths(self):
        values = torch.linspace(-12, 12, 4 * 64, dtype=torch.float32).reshape(4, 64).to(torch.bfloat16).cuda()
        packed, scales = kernel.fp4_act_quant(values, 16, scale_dtype=torch.float8_e4m3fn)
        self.assertEqual(packed.shape, (4, 32))
        self.assertEqual(packed.dtype, torch.float4_e2m1fn_x2)
        self.assertEqual(scales.shape, (4, 4))
        self.assertEqual(scales.dtype, torch.float8_e4m3fn)

        inplace = values.clone()
        result = kernel.fp4_act_quant(inplace, 32, inplace=True)
        self.assertIs(result, inplace)
        self.assertTrue(torch.isfinite(inplace.float()).all().item())

    def test_fp4_activation_quantization_handles_noncontiguous_inputs(self):
        values = torch.linspace(-12, 12, 64 * 4, dtype=torch.float32, device="cuda")
        values = values.to(torch.bfloat16).reshape(64, 4).transpose(0, 1)
        self.assertFalse(values.is_contiguous())
        expected, expected_scales = kernel.fp4_act_quant(
            values.contiguous(), 16, scale_dtype=torch.float8_e4m3fn
        )
        packed, scales = kernel.fp4_act_quant(
            values, 16, scale_dtype=torch.float8_e4m3fn
        )
        torch.testing.assert_close(packed, expected, rtol=0, atol=0)
        torch.testing.assert_close(scales.float(), expected_scales.float(), rtol=0, atol=0)

        inplace = torch.linspace(-12, 12, 64 * 4, dtype=torch.float32, device="cuda")
        inplace = inplace.to(torch.bfloat16).reshape(64, 4).transpose(0, 1)
        self.assertFalse(inplace.is_contiguous())
        expected_inplace = inplace.contiguous()
        kernel.fp4_act_quant(expected_inplace, 32, inplace=True)
        result = kernel.fp4_act_quant(inplace, 32, inplace=True)
        self.assertIs(result, inplace)
        torch.testing.assert_close(inplace, expected_inplace, rtol=0, atol=0)

    def test_noncontiguous_activation_copy_keeps_source_alive_on_consumer_stream(self):
        def exercise(operation):
            producer = torch.cuda.Stream()
            consumer = torch.cuda.Stream()
            ready = torch.cuda.Event()
            with torch.cuda.stream(producer):
                backing = torch.full((64, 4), 3.0, dtype=torch.bfloat16, device="cuda")
                values = backing.transpose(0, 1)
                ready.record(producer)

            consumer.wait_event(ready)
            with torch.cuda.stream(consumer):
                # Hold the consumer behind a long-running device operation so
                # producer-side allocations can reuse freed source storage
                # before contiguous() reads it if record_stream is missing.
                torch.cuda._sleep(100_000_000)
                result = operation(values)
                del values, backing

            with torch.cuda.stream(producer):
                churn = [
                    torch.full((64, 4), -17.0, dtype=torch.bfloat16, device="cuda")
                    for _ in range(16)
                ]
            consumer.synchronize()
            return result, churn

        (quantized, scales), churn = exercise(
            lambda x: kernel.act_quant(x, 32, "ue8m0", self.e8m0)
        )
        expected_x = torch.full((4, 64), 3.0, dtype=torch.bfloat16, device="cuda")
        expected, expected_scales = kernel.act_quant(
            expected_x, 32, "ue8m0", self.e8m0
        )
        torch.testing.assert_close(quantized, expected, rtol=0, atol=0)
        torch.testing.assert_close(scales.float(), expected_scales.float(), rtol=0, atol=0)

        (packed, fp4_scales), churn = exercise(lambda x: kernel.fp4_act_quant(x, 32))
        expected_packed, expected_fp4_scales = kernel.fp4_act_quant(expected_x, 32)
        torch.testing.assert_close(packed, expected_packed, rtol=0, atol=0)
        torch.testing.assert_close(fp4_scales.float(), expected_fp4_scales.float(), rtol=0, atol=0)

    def test_fp8_gemm_dispatches_on_the_current_stream(self):
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                a = torch.linspace(-1, 1, 5 * 64, device="cuda").reshape(5, 64).to(torch.float8_e4m3fn)
                b = torch.linspace(1, -1, 7 * 64, device="cuda").reshape(7, 64).to(torch.float8_e4m3fn)
                a_s = torch.ones((5, 2), device="cuda").to(self.e8m0)
                b_s = torch.ones((1, 2), device="cuda").to(self.e8m0)
                output = kernel.fp8_gemm(a, a_s, b, b_s, self.e8m0, 32)
                expected = a.float() @ b.float().T
                event = torch.cuda.Event()
                event.record(stream)
            event.synchronize()
            self.assertEqual(output.shape, (5, 7))
            self.assertEqual(output.dtype, torch.float32)
            torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)
        finally:
            torch.set_default_dtype(previous_dtype)

    def test_fp4_gemm_dispatches_with_1017_packed_weights(self):
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            values = torch.linspace(-1, 1, 4 * 64, device="cuda").reshape(4, 64).to(torch.bfloat16)
            a, a_s = kernel.act_quant(values, 32, "ue8m0", self.e8m0)
            weights = torch.linspace(-2, 2, 6 * 64, device="cuda").reshape(6, 64).to(torch.bfloat16)
            b, b_s = kernel.fp4_act_quant(weights, 32)
            output = kernel.fp4_gemm(a, a_s, b, b_s, self.e8m0, 32)
            self.assertEqual(output.shape, (4, 6))
            self.assertEqual(output.dtype, torch.bfloat16)
            self.assertTrue(torch.isfinite(output.float()).all().item())
        finally:
            torch.set_default_dtype(previous_dtype)

    def test_sparse_attention_dispatches_reference_shapes(self):
        q = torch.randn((1, 2, 3, 8), dtype=torch.bfloat16, device="cuda")
        kv = torch.randn((1, 5, 8), dtype=torch.bfloat16, device="cuda")
        sink = torch.zeros((3,), dtype=torch.float32, device="cuda")
        indices = torch.tensor([[[0, 2, 4], [1, 3, -1]]], dtype=torch.int32, device="cuda")
        output = kernel.sparse_attn(q, kv, sink, indices, 8**-0.5)
        self.assertEqual(output.shape, q.shape)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(output.float()).all().item())

    def test_hc_split_sinkhorn_dispatches_reference_shapes(self):
        hc_mult = 4
        mixes = torch.randn((2, 3, (2 + hc_mult) * hc_mult), dtype=torch.float32, device="cuda")
        scale = torch.ones((3,), dtype=torch.float32, device="cuda")
        base = torch.zeros(((2 + hc_mult) * hc_mult,), dtype=torch.float32, device="cuda")
        pre, post, comb = kernel.hc_split_sinkhorn(mixes, scale, base, hc_mult, 4, 1e-6)
        self.assertEqual(pre.shape, (2, 3, hc_mult))
        self.assertEqual(post.shape, (2, 3, hc_mult))
        self.assertEqual(comb.shape, (2, 3, hc_mult, hc_mult))
        self.assertTrue(torch.isfinite(pre).all().item())
        self.assertTrue(torch.isfinite(post).all().item())
        self.assertTrue(torch.isfinite(comb).all().item())


if __name__ == "__main__":
    unittest.main(verbosity=2)
