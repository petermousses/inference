"""Admission, mapping, and lifecycle tests for dense checkpoint loading."""

from __future__ import annotations

import importlib.util
import gc
import tempfile
import threading
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import dense_checkpoint
from checkpoint_mapping_audit import MappingAuditError
from checkpoint_reader import CheckpointError, SafeTensorCheckpoint
from dense_checkpoint import load_dense_checkpoint, plan_dense_checkpoint_load
from test_checkpoint_reader import chunk_digest_bytes, checkpoint

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
try:
    import torch
except ImportError:
    torch = None

HAS_REQUIRED_DTYPES = TORCH_AVAILABLE and all(
    hasattr(torch, name)
    for name in ("float8_e4m3fn", "float8_e8m0fnu", "bfloat16")
)


@unittest.skipUnless(HAS_REQUIRED_DTYPES, "requires PyTorch BF16, FP8 E4M3, and E8M0")
class DenseCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model, self.values = self._model_and_values()
        self.inventory = checkpoint(self.root, {"weights.safetensors": self._checkpoint_tensors()})
        self.reader = SafeTensorCheckpoint(
            self.root,
            expected_revision="abc123",
            expected_file_inventory=self.inventory,
            expected_chunk_digest_bytes=chunk_digest_bytes(self.root, self.inventory),
        )

    def tearDown(self):
        self.reader.close()
        self.temp.cleanup()

    @staticmethod
    def _raw(tensor):
        return tensor.contiguous().view(torch.uint8).numpy().tobytes()

    @classmethod
    def _meta_parameter(cls, shape, dtype):
        return torch.nn.Parameter(torch.empty(shape, dtype=dtype, device="meta"))

    def _model_and_values(self):
        class DenseModule(torch.nn.Module):
            def __init__(self, shape, dtype):
                super().__init__()
                self.weight = DenseCheckpointTest._meta_parameter(shape, dtype)

        class Attention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.wo_a = DenseModule((64, 64), torch.bfloat16)
                self.linear = DenseModule((4, 4), torch.bfloat16)

        class Engram(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Module()
                self.embed.weight = DenseCheckpointTest._meta_parameter(
                    (5, 64), torch.float8_e4m3fn
                )
                self.embed.scale = DenseCheckpointTest._meta_parameter(
                    (5, 2), torch.float8_e8m0fnu
                )
                self.wkv = DenseModule((2, 2), torch.bfloat16)

        class Expert(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w1 = torch.nn.Module()
                self.w1.register_buffer(
                    "weight", torch.empty((2, 2), dtype=torch.uint8, device="meta")
                )

        class FeedForward(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.experts = torch.nn.ModuleList([Expert()])

        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = Attention()
                self.engram = Engram()
                self.ffn = FeedForward()

        class MTP(torch.nn.Module):
            def __init__(self, embed_weight, head_weight):
                super().__init__()
                self.embed = torch.nn.Module()
                self.embed.weight = embed_weight
                self.head = torch.nn.Module()
                self.head.weight = head_weight

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = DenseModule((2, 3), torch.bfloat16)
                self.head = DenseModule((2, 3), torch.bfloat16)
                self.vision = torch.nn.Module()
                self.vision.norm = DenseModule((2,), torch.float32)
                self.layers = torch.nn.ModuleList([Layer()])
                self.mtp = torch.nn.ModuleList(
                    [MTP(self.embed.weight, self.head.weight)]
                )

        model = Model()
        values = {
            "embed": torch.arange(6, dtype=torch.float32).reshape(2, 3).to(torch.bfloat16),
            "head": torch.arange(6, dtype=torch.float32).reshape(2, 3).add(1).to(torch.bfloat16),
            "wo_a": torch.arange(64 * 64, dtype=torch.float32)
            .remainder(11)
            .sub(5)
            .div(4)
            .reshape(64, 64)
            .to(torch.float8_e4m3fn),
            "wo_a_scale": torch.ones((2, 2), dtype=torch.float8_e8m0fnu),
            "linear": torch.arange(16, dtype=torch.float32).reshape(4, 4).to(torch.bfloat16),
            "vision_norm": torch.tensor([1.5, -2.25], dtype=torch.bfloat16),
            "expert": torch.tensor([[1, 2], [3, 4]], dtype=torch.uint8),
            "engram_weight": torch.arange(5 * 64, dtype=torch.float32)
            .remainder(13)
            .sub(6)
            .div(4)
            .reshape(5, 64)
            .to(torch.float8_e4m3fn),
            "engram_scale": torch.ones((5, 2), dtype=torch.float8_e8m0fnu),
            "engram_wkv": torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
        }
        return model, values

    def _checkpoint_tensors(self):
        values = self.values
        return {
            "model.embed.weight": ("BF16", [2, 3], self._raw(values["embed"])),
            "model.head.weight": ("BF16", [2, 3], self._raw(values["head"])),
            "mtp.0.embed.weight": ("BF16", [2, 3], self._raw(values["embed"])),
            "mtp.0.head.weight": ("BF16", [2, 3], self._raw(values["head"])),
            "model.layers.0.attn.wo_a.weight": (
                "F8_E4M3",
                [64, 64],
                self._raw(values["wo_a"]),
            ),
            "model.layers.0.attn.wo_a.scale": (
                "F8_E8M0",
                [2, 2],
                self._raw(values["wo_a_scale"]),
            ),
            "model.layers.0.attn.linear.weight": (
                "BF16",
                [4, 4],
                self._raw(values["linear"]),
            ),
            "model.vision.norm.weight": (
                "BF16",
                [2],
                self._raw(values["vision_norm"]),
            ),
            "model.layers.0.ffn.experts.0.w1.weight": (
                "U8",
                [2, 2],
                self._raw(values["expert"]),
            ),
            "model.layers.0.engram.embed.weight": (
                "F8_E4M3",
                [5, 64],
                self._raw(values["engram_weight"]),
            ),
            "model.layers.0.engram.embed.scale": (
                "F8_E8M0",
                [5, 2],
                self._raw(values["engram_scale"]),
            ),
            "model.layers.0.engram.wkv.weight": (
                "BF16",
                [2, 2],
                self._raw(values["engram_wkv"]),
            ),
        }

    def _admission(self, estimate, events, on_enter=None):
        events.append(("admission", estimate))

        class Lease:
            def __enter__(self):
                events.append(("enter", None))
                if on_enter is not None:
                    on_enter()

            def __exit__(self, exc_type, exc, traceback):
                events.append(("exit", exc_type))

        return Lease()

    def test_plan_admits_dense_state_but_excludes_experts_and_engram_tables(self):
        plan = plan_dense_checkpoint_load(self.reader, self.model, max_chunk_bytes=256)
        names = {item.target_name for item in plan.tensors}
        self.assertIn("layers.0.engram.wkv.weight", names)
        self.assertIn("layers.0.attn.wo_a.weight", names)
        self.assertNotIn("layers.0.ffn.experts.0.w1.weight", names)
        self.assertNotIn("layers.0.engram.embed.weight", names)
        self.assertEqual(plan.estimate.routed_expert_tensor_count, 1)
        self.assertEqual(plan.estimate.engram_table_tensor_count, 2)
        self.assertEqual(plan.estimate.wo_a_tensor_count, 1)
        self.assertGreater(plan.estimate.physical_peak_bytes, plan.estimate.device_peak_bytes)

    def test_load_matches_checkpoint_and_official_wo_a_conversion_then_restores_meta(self):
        events = []
        admitted = False
        original_empty = torch.empty

        def checked_empty(*args, **kwargs):
            self.assertTrue(admitted, "device allocation happened before memory admission")
            return original_empty(*args, **kwargs)

        def admission(estimate):
            nonlocal admitted

            def mark_admitted():
                nonlocal admitted
                admitted = True

            return self._admission(estimate, events, mark_admitted)

        state = None
        with patch.object(torch, "empty", side_effect=checked_empty):
            state = load_dense_checkpoint(
                self.reader,
                self.model,
                device="cpu",
                admission=admission,
                fence=lambda: events.append(("fence", None)),
                release_allocator=lambda: events.append(("release", None)),
                max_chunk_bytes=256,
                max_wo_a_output_bytes=4096,
            )

        loaded = self.model.state_dict(keep_vars=True)
        self.assertFalse(loaded["embed.weight"].is_meta)
        self.assertFalse(loaded["head.weight"].is_meta)
        self.assertFalse(loaded["layers.0.engram.wkv.weight"].is_meta)
        self.assertTrue(loaded["layers.0.ffn.experts.0.w1.weight"].is_meta)
        self.assertTrue(loaded["layers.0.engram.embed.weight"].is_meta)
        self.assertTrue(loaded["layers.0.engram.embed.scale"].is_meta)
        self.assertIs(loaded["mtp.0.embed.weight"], loaded["embed.weight"])
        self.assertIs(loaded["mtp.0.head.weight"], loaded["head.weight"])
        self.assertTrue(torch.equal(loaded["embed.weight"], self.values["embed"]))
        self.assertTrue(torch.equal(loaded["vision.norm.weight"], self.values["vision_norm"].float()))
        self.assertTrue(torch.equal(loaded["layers.0.engram.wkv.weight"], self.values["engram_wkv"]))

        expected_wo_a = (
            self.values["wo_a"]
            .unflatten(0, (-1, 32))
            .unflatten(-1, (-1, 32))
            .float()
            * self.values["wo_a_scale"][:, None, :, None].float()
        ).flatten(2, 3).flatten(0, 1).to(torch.bfloat16)
        torch.testing.assert_close(
            loaded["layers.0.attn.wo_a.weight"], expected_wo_a, rtol=0, atol=0
        )
        self.assertTrue(any(kind == "enter" for kind, _ in events))
        self.assertFalse(any(kind == "exit" for kind, _ in events))

        state.close()
        restored = self.model.state_dict(keep_vars=True)
        self.assertTrue(all(tensor.is_meta for tensor in restored.values()))
        self.assertIs(restored["mtp.0.embed.weight"], restored["embed.weight"])
        self.assertIn(("release", None), events)
        self.assertTrue(any(kind == "exit" for kind, _ in events))
        state.close()

    def test_concurrent_close_releases_allocator_and_admission_once(self):
        events = []
        fence_started = threading.Event()
        continue_fence = threading.Event()
        second_lock_attempt = threading.Event()
        count_lock = threading.Lock()
        close_fence_calls = 0
        block_close_fence = False

        def fence():
            nonlocal close_fence_calls
            if not block_close_fence:
                return
            with count_lock:
                close_fence_calls += 1
                call = close_fence_calls
            if call == 1:
                fence_started.set()
                if not continue_fence.wait(timeout=5):
                    raise TimeoutError("test did not release the first close fence")

        state = load_dense_checkpoint(
            self.reader,
            self.model,
            device="cpu",
            admission=lambda estimate: self._admission(estimate, events),
            fence=fence,
            release_allocator=lambda: events.append(("release", None)),
            max_chunk_bytes=256,
            max_wo_a_output_bytes=4096,
        )

        class SignalingLock:
            def __init__(self, lock):
                self.lock = lock
                self.attempts = 0

            def __enter__(self):
                with count_lock:
                    self.attempts += 1
                    if self.attempts == 2:
                        second_lock_attempt.set()
                self.lock.acquire()
                return self

            def __exit__(self, exc_type, exc, traceback):
                self.lock.release()
                return False

        state._close_lock = SignalingLock(state._close_lock)
        block_close_fence = True
        errors = []

        def close_state():
            try:
                state.close()
            except BaseException as exc:
                errors.append(exc)

        first = threading.Thread(target=close_state)
        first.start()
        self.assertTrue(fence_started.wait(timeout=5))

        second = threading.Thread(target=close_state)
        second.start()
        self.assertTrue(second_lock_attempt.wait(timeout=5))
        continue_fence.set()
        first.join(timeout=5)
        second.join(timeout=5)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(close_fence_calls, 2)
        self.assertEqual([kind for kind, _ in events if kind == "release"], ["release"])
        self.assertEqual([kind for kind, _ in events if kind == "exit"], ["exit"])
        self.assertTrue(all(tensor.is_meta for tensor in self.model.state_dict().values()))

    def test_failed_fence_quarantines_inflight_conversion_references(self):
        weak_temporary = []
        weak_destination = []
        events = []
        quarantine_start = len(dense_checkpoint._QUARANTINED_LOADS)

        def partial_copy(checkpoint, source_name, destination, *, max_chunk_bytes, torch):
            temporary = torch.empty((16,), dtype=torch.float32, device=destination.device)
            weak_temporary.append(weakref.ref(temporary))
            weak_destination.append(weakref.ref(destination))
            raise RuntimeError("injected partial copy failure")

        def failed_fence():
            raise RuntimeError("injected fence failure")

        caught_error = None
        with patch.object(dense_checkpoint, "_copy_tensor", side_effect=partial_copy):
            try:
                load_dense_checkpoint(
                    self.reader,
                    self.model,
                    device="cpu",
                    admission=lambda estimate: self._admission(estimate, events),
                    fence=failed_fence,
                    release_allocator=lambda: events.append(("release", None)),
                    max_chunk_bytes=256,
                    max_wo_a_output_bytes=4096,
                )
            except RuntimeError as exc:
                caught_error = exc

        self.assertIsNotNone(caught_error)
        quarantined = dense_checkpoint._QUARANTINED_LOADS[quarantine_start:]
        self.assertEqual(len(quarantined), 1)
        retained = quarantined[0][1]
        self.assertIs(caught_error, retained["inflight"][0])
        self.assertIs(weak_destination[0](), retained["inflight"][2])
        self.assertIsNotNone(weak_destination[0]())
        self.assertIsNotNone(weak_temporary[0]())
        self.assertEqual([kind for kind, _ in events if kind == "exit"], [])
        self.assertEqual([kind for kind, _ in events if kind == "release"], [])
        self.assertTrue(all(tensor.is_meta for tensor in self.model.state_dict().values()))

        # The test owns a CPU-only synthetic lease; remove its deliberate quarantine after proving
        # the worker-lifetime retention behavior.
        with dense_checkpoint._LIVE_STATES_LOCK:
            del dense_checkpoint._QUARANTINED_LOADS[quarantine_start:]

    def test_missing_admission_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "admission must be callable"):
            load_dense_checkpoint(
                self.reader,
                self.model,
                device="cpu",
                admission=None,
                fence=lambda: None,
                release_allocator=lambda: None,
            )

    def test_async_admission_context_is_rejected_before_allocation(self):
        class AsyncLease:
            async def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

        with self.assertRaisesRegex(TypeError, "must complete synchronously"):
            load_dense_checkpoint(
                self.reader,
                self.model,
                device="cpu",
                admission=lambda estimate: AsyncLease(),
                fence=lambda: None,
                release_allocator=lambda: None,
            )
        self.assertTrue(all(tensor.is_meta for tensor in self.model.state_dict().values()))

    def test_wo_a_chunk_limit_rejects_a_single_oversized_output_block(self):
        with self.assertRaisesRegex(CheckpointError, "one wo_a output block"):
            plan_dense_checkpoint_load(
                self.reader,
                self.model,
                max_chunk_bytes=256,
                max_wo_a_output_bytes=1024,
            )

    def test_audit_failure_occurs_before_admission(self):
        self.model.layers[0].attn.linear.weight = self._meta_parameter(
            (3, 3), torch.bfloat16
        )
        admission_calls = []
        with self.assertRaises(MappingAuditError):
            load_dense_checkpoint(
                self.reader,
                self.model,
                device="cpu",
                admission=lambda estimate: admission_calls.append(estimate),
                fence=lambda: None,
                release_allocator=lambda: None,
            )

        self.assertEqual(admission_calls, [])
        self.assertTrue(all(tensor.is_meta for tensor in self.model.state_dict().values()))

    def test_load_failure_cleans_allocations_and_releases_reservation(self):
        events = []
        with patch.object(
            dense_checkpoint,
            "_copy_tensor",
            side_effect=RuntimeError("injected copy failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected copy failure"):
                load_dense_checkpoint(
                    self.reader,
                    self.model,
                    device="cpu",
                    admission=lambda estimate: self._admission(estimate, events),
                    fence=lambda: events.append(("fence", None)),
                    release_allocator=lambda: events.append(("release", None)),
                    max_chunk_bytes=256,
                    max_wo_a_output_bytes=4096,
                )

        self.assertTrue(all(tensor.is_meta for tensor in self.model.state_dict().values()))
        self.assertEqual([kind for kind, _ in events if kind == "exit"], ["exit"])
        self.assertIn(("release", None), events)

    def test_assignment_interruption_rolls_back_mutated_slot(self):
        class MutateThenRaiseOnce(dict):
            def __init__(self, values):
                super().__init__(values)
                self.fail_once = True

            def __setitem__(self, key, value):
                super().__setitem__(key, value)
                if self.fail_once:
                    self.fail_once = False
                    raise KeyboardInterrupt("injected after parameter mutation")

        original_parameters = self.model.embed._parameters
        self.model.embed._parameters = MutateThenRaiseOnce(original_parameters)
        events = []
        try:
            with self.assertRaisesRegex(KeyboardInterrupt, "after parameter mutation"):
                load_dense_checkpoint(
                    self.reader,
                    self.model,
                    device="cpu",
                    admission=lambda estimate: self._admission(estimate, events),
                    fence=lambda: events.append(("fence", None)),
                    release_allocator=lambda: events.append(("release", None)),
                    max_chunk_bytes=256,
                    max_wo_a_output_bytes=4096,
                )
            self.assertTrue(all(tensor.is_meta for tensor in self.model.state_dict().values()))
            self.assertEqual([kind for kind, _ in events if kind == "exit"], ["exit"])
            self.assertIn(("release", None), events)
        finally:
            self.model.embed._parameters = original_parameters


if __name__ == "__main__":
    unittest.main()
