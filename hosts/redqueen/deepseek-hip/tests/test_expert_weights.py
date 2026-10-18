"""Bounded loading and memory estimates for one pinned 1017 FP4 expert."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

from checkpoint_reader import CheckpointError, SafeTensorCheckpoint
from expert_weights import estimate_fp4_expert_memory, load_fp4_expert
from test_checkpoint_reader import chunk_digest_bytes, checkpoint

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
try:
    import torch
except ImportError:
    torch = None


class FP4ExpertFixture:
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        packed_weight = bytes(range(256)) * 2
        scale = b"\x7f" * 32
        tensors = {}
        for prefix in ("layers.0", "mtp.0"):
            for projection in ("w1", "w2", "w3"):
                base = f"{prefix}.ffn.experts.2.{projection}"
                tensors[f"{base}.weight"] = ("I8", [32, 16], packed_weight)
                tensors[f"{base}.scale"] = ("F8_E8M0", [32, 1], scale)
        self.inventory = checkpoint(
            self.root,
            {"weights.safetensors": tensors},
        )
        self.reader = SafeTensorCheckpoint(
            self.root,
            expected_revision="abc123",
            expected_file_inventory=self.inventory,
            expected_chunk_digest_bytes=chunk_digest_bytes(self.root, self.inventory),
        )

    def tearDown(self):
        self.reader.close()
        self.temp.cleanup()


class FP4ExpertEstimateTest(FP4ExpertFixture, unittest.TestCase):
    def test_reports_resident_and_conservative_load_peak_bytes(self):
        estimate = estimate_fp4_expert_memory(self.reader, 0, 2)
        self.assertEqual(estimate.resident_bytes, 3 * (512 + 32))
        self.assertEqual(estimate.largest_tensor_bytes, 512)
        self.assertEqual(estimate.conservative_load_peak_bytes, 3 * (512 + 32) + 1024)

    def test_supports_mtp_keys_and_rejects_invalid_ids(self):
        mtp_estimate = estimate_fp4_expert_memory(self.reader, 0, 2, mtp_layer=True)
        self.assertEqual(mtp_estimate.resident_bytes, 3 * (512 + 32))
        with self.assertRaises(TypeError):
            estimate_fp4_expert_memory(self.reader, True, 2)
        with self.assertRaises(ValueError):
            estimate_fp4_expert_memory(self.reader, 0, -1)
        with self.assertRaises(TypeError):
            estimate_fp4_expert_memory(self.reader, 0, 2, mtp_layer=1)
        with self.assertRaises(CheckpointError):
            estimate_fp4_expert_memory(self.reader, 0, 3)


@unittest.skipUnless(TORCH_AVAILABLE, "requires PyTorch")
class FP4ExpertLoadTest(FP4ExpertFixture, unittest.TestCase):
    def test_reinterprets_packed_storage_and_attaches_scales(self):
        expert = load_fp4_expert(self.reader, 0, 2, device="cpu")
        self.assertEqual(expert.layer_id, 0)
        self.assertEqual(expert.expert_id, 2)
        self.assertFalse(expert.mtp_layer)
        for weight in (expert.w1, expert.w2, expert.w3):
            self.assertEqual(weight.dtype, torch.float4_e2m1fn_x2)
            self.assertEqual(tuple(weight.shape), (32, 16))
            self.assertEqual(weight.scale.dtype, torch.float8_e8m0fnu)
            self.assertEqual(tuple(weight.scale.shape), (32, 1))
        self.assertEqual(expert.resident_bytes, 3 * (512 + 32))

    def test_loads_mtp_expert(self):
        expert = load_fp4_expert(self.reader, 0, 2, device="cpu", mtp_layer=True)
        self.assertTrue(expert.mtp_layer)
        self.assertEqual(expert.resident_bytes, 3 * (512 + 32))


if __name__ == "__main__":
    unittest.main()
