"""Routing, weighted accumulation, and shared-expert tests for cached MoE."""

import unittest

try:
    import torch
    import torch.nn.functional as functional
except ImportError:
    torch = None
    functional = None

from expert_cache import FP4ExpertCache, FP4ExpertKey
from expert_moe import execute_cached_moe
from expert_weights import LoadedFP4Expert


def make_expert(key, scale):
    weights = []
    for values in (
        [[0.2, -0.4], [0.5, 0.3]],
        [[0.2, -0.1], [0.5, 0.3]],
        [[-0.2, 0.4], [0.7, -0.3]],
    ):
        weight = torch.tensor(values, dtype=torch.float32) * scale
        weight.scale = torch.empty(0, dtype=torch.float32)
        weights.append(weight)
    return LoadedFP4Expert(
        layer_id=key.layer_id,
        expert_id=key.expert_id,
        mtp_layer=key.mtp_layer,
        w1=weights[0],
        w2=weights[1],
        w3=weights[2],
    )


@unittest.skipIf(torch is None, "requires PyTorch")
class CachedMoETest(unittest.TestCase):
    def setUp(self):
        self.layer_id = 3
        self.n_experts = 3
        self.keys = [FP4ExpertKey(self.layer_id, expert_id) for expert_id in range(3)]
        self.experts = {key: make_expert(key, key.expert_id + 1) for key in self.keys}
        self.loads = []
        resident_bytes = next(iter(self.experts.values())).resident_bytes
        self.cache = FP4ExpertCache(
            resident_bytes,
            estimate_resident_bytes=lambda key: self.experts[key].resident_bytes,
            load=self.load,
        )
        self.activations = torch.tensor(
            [[[0.2, -0.7], [-0.4, 0.6], [0.1, 0.9]]], dtype=torch.bfloat16
        )
        self.image_mask = torch.tensor([[False, True, False]])
        self.route_weights = torch.tensor(
            [[0.25, 0.75], [0.6, 0.4], [0.8, 0.2]], dtype=torch.float32
        )
        self.route_indices = torch.tensor([[0, 1], [1, 2], [2, 0]])
        self.seen_masks = []
        self.events = []

    def load(self, key):
        self.loads.append(key)
        return self.experts[key]

    def gate(self, activations, image_mask):
        self.seen_masks.append(None if image_mask is None else image_mask.clone())
        return self.route_weights, self.route_indices

    def linear(self, activations, weight):
        self.events.append("linear")
        return functional.linear(activations.float(), weight).to(activations.dtype)

    def expert_forward(self, expert, activations, weights):
        dtype = activations.dtype
        gate = self.linear(activations, expert.w1).float()
        up = self.linear(activations, expert.w3).float()
        if 0.6 > 0:
            up = torch.clamp(up, min=-0.6, max=0.6)
            gate = torch.clamp(gate, max=0.6)
        hidden = functional.silu(gate) * up
        hidden = weights * hidden
        return self.linear(hidden.to(dtype), expert.w2)

    def shared_expert(self, activations):
        self.events.append("shared")
        return activations * torch.tensor(0.5, dtype=activations.dtype)

    def reference_moe(self):
        shape = self.activations.size()
        flat = self.activations.view(-1, shape[-1])
        weights, indices = self.gate(flat, self.image_mask.flatten())
        output = torch.zeros_like(flat, dtype=torch.float32)
        counts = torch.bincount(indices.flatten(), minlength=self.n_experts).tolist()
        for expert_id, count in enumerate(counts):
            if count == 0:
                continue
            token_indices, route_positions = torch.where(indices == expert_id)
            output[token_indices] += self.expert_forward(
                self.experts[self.keys[expert_id]],
                flat[token_indices],
                weights[token_indices, route_positions, None],
            )
        output += self.shared_expert(flat)
        return output.type_as(flat).view(shape)

    def test_matches_reference_routes_image_mask_and_shared_expert(self):
        expected = self.reference_moe()
        self.events.clear()

        def fence():
            self.events.append("fence")

        actual = execute_cached_moe(
            self.cache,
            self.activations,
            self.gate,
            self.shared_expert,
            self.linear,
            layer_id=self.layer_id,
            n_routed_experts=self.n_experts,
            swiglu_limit=0.6,
            image_mask=self.image_mask,
            fence=lambda: self.events.append("fence"),
        )

        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(self.seen_masks[-1], self.image_mask.flatten()))
        self.assertEqual(self.loads, self.keys)
        self.assertEqual(self.events[-2], "shared")
        self.assertEqual(self.events.count("fence"), 4)
        self.assertEqual(self.cache.stats.leased_bytes, 0)
        self.assertEqual(self.cache.stats.entries, 1)

    def test_rejects_router_indices_outside_configured_experts(self):
        self.route_indices = torch.tensor([[0, 1], [1, 2], [2, 3]])
        with self.assertRaisesRegex(ValueError, "outside the configured range"):
            execute_cached_moe(
                self.cache,
                self.activations,
                self.gate,
                self.shared_expert,
                self.linear,
                layer_id=self.layer_id,
                n_routed_experts=self.n_experts,
                swiglu_limit=0.6,
                image_mask=self.image_mask,
            )
        self.assertEqual(self.cache.stats.resident_bytes, 0)

    def test_uses_mtp_expert_namespace(self):
        experts = {
            FP4ExpertKey(self.layer_id, expert_id, mtp_layer=True): make_expert(
                FP4ExpertKey(self.layer_id, expert_id, mtp_layer=True), expert_id + 1
            )
            for expert_id in range(self.n_experts)
        }
        loads = []
        resident_bytes = next(iter(experts.values())).resident_bytes
        cache = FP4ExpertCache(
            resident_bytes,
            estimate_resident_bytes=lambda key: experts[key].resident_bytes,
            load=lambda key: loads.append(key) or experts[key],
        )
        expected = self.reference_moe()
        actual = execute_cached_moe(
            cache,
            self.activations,
            self.gate,
            self.shared_expert,
            self.linear,
            layer_id=self.layer_id,
            n_routed_experts=self.n_experts,
            swiglu_limit=0.6,
            image_mask=self.image_mask,
            mtp_layer=True,
        )

        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(loads, list(experts))

    def test_final_fence_failure_quarantines_cache_after_shared_expert(self):
        calls = 0

        def fence():
            nonlocal calls
            calls += 1
            if calls == 4:
                raise RuntimeError("synthetic final MoE fence failure")

        with self.assertRaisesRegex(RuntimeError, "synthetic final MoE fence failure"):
            execute_cached_moe(
                self.cache,
                self.activations,
                self.gate,
                self.shared_expert,
                self.linear,
                layer_id=self.layer_id,
                n_routed_experts=self.n_experts,
                swiglu_limit=0.6,
                image_mask=self.image_mask,
                fence=fence,
            )

        self.assertTrue(self.cache.stats.quarantined)
        self.assertEqual(self.cache.stats.leased_bytes, 0)
        with self.assertRaisesRegex(RuntimeError, "quarantined; worker restart is required"):
            with self.cache.acquire(self.keys[0]):
                self.fail("quarantined cache unexpectedly admitted another expert")


if __name__ == "__main__":
    unittest.main()
