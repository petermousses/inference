"""Reference-formula and lease-lifetime tests for cached FP4 experts."""

import unittest

try:
    import torch
    import torch.nn.functional as functional
except ImportError:
    torch = None
    functional = None

from expert_cache import ExpertCacheError, FP4ExpertCache, FP4ExpertKey
from expert_execution import execute_fp4_expert
from expert_weights import LoadedFP4Expert


@unittest.skipIf(torch is None, "requires PyTorch")
class FP4ExpertExecutionTest(unittest.TestCase):
    def setUp(self):
        self.key = FP4ExpertKey(0, 0)
        self.weights = {}
        for name, values in (
            ("w1", [[0.2, -0.4, 0.8, 0.1], [0.5, 0.3, -0.2, 0.7], [-0.1, 0.6, 0.4, -0.5]]),
            ("w2", [[0.2, -0.1, 0.4], [0.5, 0.3, -0.2], [-0.7, 0.6, 0.1], [0.3, 0.8, -0.4]]),
            ("w3", [[-0.2, 0.4, 0.1, 0.5], [0.7, -0.3, 0.6, 0.2], [0.4, 0.1, -0.5, 0.8]]),
        ):
            tensor = torch.tensor(values, dtype=torch.float32)
            tensor.scale = torch.empty(0, dtype=torch.float32)
            self.weights[name] = tensor
        self.expert = LoadedFP4Expert(
            layer_id=0,
            expert_id=0,
            mtp_layer=False,
            w1=self.weights["w1"],
            w2=self.weights["w2"],
            w3=self.weights["w3"],
        )
        self.cache = FP4ExpertCache(
            self.expert.resident_bytes,
            estimate_resident_bytes=lambda _: self.expert.resident_bytes,
            load=lambda _: self.expert,
        )

    def reference_linear(self, activations, weight):
        return functional.linear(activations.float(), weight).to(activations.dtype)

    def reference_forward(self, activations, limit, routing_weights):
        dtype = activations.dtype
        gate = self.reference_linear(activations, self.expert.w1).float()
        up = self.reference_linear(activations, self.expert.w3).float()
        if limit > 0:
            up = torch.clamp(up, min=-limit, max=limit)
            gate = torch.clamp(gate, max=limit)
        hidden = functional.silu(gate) * up
        if routing_weights is not None:
            hidden = routing_weights * hidden
        return self.reference_linear(hidden.to(dtype), self.expert.w2)

    def test_matches_reference_swiglu_and_holds_lease_through_fence(self):
        activations = torch.tensor(
            [[0.2, -0.7, 1.1, 0.3], [-0.4, 0.6, 0.1, 0.9]], dtype=torch.bfloat16
        )
        routing_weights = torch.tensor([[0.25], [0.8]], dtype=torch.float32)
        limit = 0.6
        expected = self.reference_forward(activations, limit, routing_weights)
        events = []

        def linear(values, weight):
            self.assertGreater(self.cache.stats.leased_bytes, 0)
            events.append("linear")
            return self.reference_linear(values, weight)

        def fence():
            self.assertGreater(self.cache.stats.leased_bytes, 0)
            events.append("fence")

        actual = execute_fp4_expert(
            self.cache,
            self.key,
            activations,
            linear,
            swiglu_limit=limit,
            routing_weights=routing_weights,
            fence=fence,
        )

        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(events, ["linear", "linear", "linear", "fence"])
        self.assertEqual(self.cache.stats.leased_bytes, 0)

    def test_fences_partial_work_when_projection_raises(self):
        activations = torch.ones((1, 4), dtype=torch.bfloat16)
        events = []
        calls = 0

        def linear(values, weight):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("synthetic projection failure")
            return self.reference_linear(values, weight)

        def fence():
            self.assertGreater(self.cache.stats.leased_bytes, 0)
            events.append("fence")

        with self.assertRaisesRegex(RuntimeError, "synthetic projection failure"):
            execute_fp4_expert(
                self.cache, self.key, activations, linear, swiglu_limit=10.0, fence=fence
            )

        self.assertEqual(events, ["fence"])
        self.assertEqual(self.cache.stats.leased_bytes, 0)

    def test_fence_failure_quarantines_cache_and_keeps_lease(self):
        activations = torch.ones((1, 4), dtype=torch.bfloat16)

        def failed_fence():
            raise RuntimeError("synthetic HIP fence failure")

        with self.assertRaisesRegex(RuntimeError, "synthetic HIP fence failure"):
            execute_fp4_expert(
                self.cache,
                self.key,
                activations,
                self.reference_linear,
                swiglu_limit=10.0,
                fence=failed_fence,
            )

        self.assertTrue(self.cache.stats.quarantined)
        self.assertEqual(self.cache.stats.leased_bytes, self.expert.resident_bytes)
        with self.assertRaisesRegex(ExpertCacheError, "quarantined; worker restart is required"):
            with self.cache.acquire(self.key):
                self.fail("quarantined cache unexpectedly admitted another expert")
        with self.assertRaisesRegex(ExpertCacheError, "quarantined; worker restart is required"):
            self.cache.resize(0)


if __name__ == "__main__":
    unittest.main()
