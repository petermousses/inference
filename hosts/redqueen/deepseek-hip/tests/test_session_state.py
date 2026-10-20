"""Planning, admission, aliasing, and teardown tests for DeepSeek session buffers."""

from __future__ import annotations

import threading
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch

import torch

import run_dense_checkpoint_smoke
from session_state import load_session_state, plan_session_state


def _args():
    return SimpleNamespace(
        max_batch_size=1,
        max_seq_len=16,
        vocab_size=129280,
        window_size=4,
        head_dim=8,
        index_head_dim=4,
        rope_head_dim=4,
        original_seq_len=8,
        rope_theta=10000.0,
        compress_rope_theta=40000.0,
        rope_factor=40.0,
        beta_fast=32,
        beta_slow=1,
    )


class Compressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.compress_ratio = 2
        self.register_buffer("kv_state", torch.zeros((1, 2, 8), dtype=torch.float32, device="meta"))
        self.register_buffer(
            "score_state",
            torch.full((1, 2, 8), -torch.inf, dtype=torch.float32, device="meta"),
        )


class Indexer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.compress_ratio = 2
        self.owns_k = True
        self.register_buffer("k_cache", torch.zeros((1, 8, 4), dtype=torch.bfloat16, device="meta"))


class NgramHashState(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layout = SimpleNamespace(max_ngram_size=4, layer_ids=(0, 1), n_heads=8)
        self.register_buffer("primes", torch.empty((2, 3, 8), dtype=torch.int64, device="meta"))
        self.register_buffer("offsets", torch.empty((2, 24), dtype=torch.int64, device="meta"))
        self.register_buffer("multipliers", torch.empty((2, 4), dtype=torch.int64, device="meta"))
        self.register_buffer("token_map", torch.empty((129280,), dtype=torch.int64, device="meta"))
        self.register_buffer("cache", torch.empty((1, 16), dtype=torch.int64, device="meta"))


class Attention(torch.nn.Module):
    def __init__(self, layer_id, ratio, frequency, is_source=False):
        super().__init__()
        self.layer_id = layer_id
        self.compress_ratio = ratio
        self.window_size = 4
        self.is_kv_source = is_source
        self.register_buffer(
            "window_kv_cache",
            torch.zeros((1, 4, 8), dtype=torch.bfloat16, device="meta"),
        )
        if is_source:
            self.register_buffer(
                "compress_kv_cache",
                torch.zeros((1, 8, 8), dtype=torch.bfloat16, device="meta"),
            )
            self.compressor = Compressor()
            self.indexer = Indexer()
        self.register_buffer("freqs_cis", frequency, persistent=False)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        args = _args()
        plain_frequency = torch.empty((16, 2), dtype=torch.complex64, device="meta")
        compressed_frequency = torch.empty((16, 2), dtype=torch.complex64, device="meta")
        self.layers = torch.nn.ModuleList(
            [
                Attention(0, 0, plain_frequency),
                Attention(1, 2, compressed_frequency, is_source=True),
                Attention(2, 2, compressed_frequency),
            ]
        )
        self.engram_hash = NgramHashState()
        self.shared = SimpleNamespace(compress_kv=None, index_k=None, topk_idxs=None, candidates=None)


def _factory(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    del original_seq_len, factor, beta_fast, beta_slow
    frequencies = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    positions = torch.arange(seqlen)
    return torch.polar(torch.ones_like(torch.outer(positions, frequencies)), torch.outer(positions, frequencies))


class Reservation:
    def __init__(self, events):
        self.events = events

    def __enter__(self):
        self.events.append("enter")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.events.append(("exit", exc_type))
        return False


class SessionStateTest(unittest.TestCase):
    def setUp(self):
        self.model = TinyModel()
        self.args = _args()
        self.events = []
        self.engram_factory_calls = 0

    def _engram_hash_factory(self, module, args, device):
        self.engram_factory_calls += 1
        self.assertIs(module, self.model.engram_hash)
        self.assertIs(args, self.args)
        return SimpleNamespace(
            _buffers={
                name: torch.full(
                    tuple(tensor.shape),
                    index + 1,
                    dtype=tensor.dtype,
                    device=device,
                )
                for index, (name, tensor) in enumerate(module._buffers.items())
            }
        )

    def _reset(self):
        self.events.append("reset")
        self.model.shared.compress_kv = None
        self.model.shared.index_k = None
        self.model.shared.topk_idxs = None
        self.model.shared.candidates = None
        for layer in self.model.layers:
            if hasattr(layer, "indexer"):
                layer.indexer.freqs_cis = None

    def _state(self, *, factory=_factory, fence=None):
        def admission(estimate):
            self.events.append(("admission", estimate))
            return Reservation(self.events)

        return load_session_state(
            self.model,
            self.args,
            device="cpu",
            frequency_factory=factory,
            engram_hash_factory=self._engram_hash_factory,
            admission=admission,
            fence=fence or (lambda: self.events.append("fence")),
            reset_runtime_references=self._reset,
            release_allocator=lambda: self.events.append("release"),
        )

    def test_plan_counts_shared_frequency_once_and_includes_all_220k_buffer_kinds(self):
        self.args.max_seq_len = 220000
        for _module_name, module in self.model.named_modules(remove_duplicate=False):
            for name, value in tuple(module._buffers.items()):
                if value is None:
                    continue
                shape = list(value.shape)
                if name in {"compress_kv_cache", "k_cache"}:
                    shape[1] = self.args.max_seq_len // 2
                elif name == "freqs_cis":
                    shape[0] = self.args.max_seq_len
                elif name == "cache":
                    shape[1] = self.args.max_seq_len
                module._buffers[name] = torch.empty(shape, dtype=value.dtype, device="meta")
        plan = plan_session_state(self.model, self.args)
        self.assertEqual(plan.max_seq_len, 220000)
        self.assertEqual(plan.estimate.window_kv_bytes, 3 * 4 * 8 * 2)
        self.assertEqual(plan.estimate.compressed_kv_bytes, 1 * 110000 * 8 * 2)
        self.assertEqual(plan.estimate.index_cache_bytes, 1 * 110000 * 4 * 2)
        self.assertEqual(plan.estimate.compressor_tail_bytes, 2 * 1 * 2 * 8 * 4)
        self.assertEqual(plan.estimate.rotary_bytes, 2 * 220000 * 2 * 8)
        self.assertEqual(
            plan.estimate.engram_hash_bytes,
            (2 * 3 * 8 + 2 * 24 + 2 * 4 + 129280 + 220000) * 8,
        )
        self.assertEqual(plan.estimate.engram_hash_bytes, 2_795_072)
        self.assertEqual(plan.estimate.physical_peak_bytes, plan.estimate.device_peak_bytes)

    def test_materializes_official_buffers_and_close_restores_meta(self):
        state = self._state()
        self.assertEqual(self.events[0], ("admission", state.plan.estimate))
        self.assertIn("enter", self.events)
        self.assertFalse(self.model.layers[0].window_kv_cache.is_meta)
        self.assertTrue(torch.equal(self.model.layers[0].window_kv_cache, torch.zeros_like(self.model.layers[0].window_kv_cache)))
        self.assertTrue(torch.isneginf(self.model.layers[1].compressor.score_state).all())
        self.assertEqual(self.engram_factory_calls, 1)
        for index, (name, tensor) in enumerate(self.model.engram_hash._buffers.items()):
            self.assertFalse(tensor.is_meta)
            self.assertTrue(torch.equal(tensor, torch.full_like(tensor, index + 1)))
        self.assertEqual(
            self.model.layers[1].freqs_cis.data_ptr(),
            self.model.layers[2].freqs_cis.data_ptr(),
        )
        self.assertIsNone(self.model.layers[1].indexer.freqs_cis)
        state.close()
        state.close()
        self.assertTrue(
            all(
                tensor is None or tensor.is_meta
                for _name, tensor in self.model.named_buffers(remove_duplicate=False)
            )
        )
        self.assertEqual(self.events[-1], ("exit", None))
        self.assertLess(self.events.index("reset"), self.events.index("release"))

    def test_reset_for_new_request_clears_mutable_buffers_and_preserves_static_state(self):
        state = self._state()
        for layer in self.model.layers:
            layer.window_kv_cache.fill_(7)
            if hasattr(layer, "compress_kv_cache"):
                layer.compress_kv_cache.fill_(7)
                layer.compressor.kv_state.fill_(7)
                layer.compressor.score_state.fill_(7)
                layer.indexer.k_cache.fill_(7)
        self.model.engram_hash.cache.fill_(7)
        static_token_map = self.model.engram_hash.token_map.clone()

        fence_count = self.events.count("fence")
        state.reset_for_new_request()

        self.assertEqual(self.events.count("fence") - fence_count, 2)
        for layer in self.model.layers:
            self.assertTrue(torch.count_nonzero(layer.window_kv_cache).item() == 0)
            if hasattr(layer, "compress_kv_cache"):
                self.assertTrue(torch.count_nonzero(layer.compress_kv_cache).item() == 0)
                self.assertTrue(torch.count_nonzero(layer.compressor.kv_state).item() == 0)
                self.assertTrue(torch.isneginf(layer.compressor.score_state).all().item())
                self.assertTrue(torch.count_nonzero(layer.indexer.k_cache).item() == 0)
        self.assertTrue(torch.count_nonzero(self.model.engram_hash.cache).item() == 0)
        self.assertTrue(torch.equal(self.model.engram_hash.token_map, static_token_map))
        self.assertNotIn(("exit", None), self.events)
        state.close()

    def test_close_drops_owned_tensors_before_releasing_backing_and_reservation(self):
        state = self._state()
        references = [weakref.ref(value) for value in state._loaded.values()]
        observed = []

        def release():
            observed.append(all(reference() is None for reference in references))

        state._release_allocator = release
        state.close()

        self.assertEqual(observed, [True])
        self.assertEqual(self.events[-1], ("exit", None))

    def test_reset_fence_failure_quarantines_the_session(self):
        fence_calls = 0

        def broken_fence():
            nonlocal fence_calls
            fence_calls += 1
            if fence_calls == 2:
                raise RuntimeError("request reset fence failed")

        state = self._state(fence=broken_fence)
        with self.assertRaisesRegex(RuntimeError, "request reset fence failed"):
            state.reset_for_new_request()
        self.assertTrue(state._quarantined)
        with self.assertRaisesRegex(RuntimeError, "already has a session load or active session"):
            self._state()

    def test_factory_failure_rolls_back_partial_allocations_and_reservation(self):
        calls = 0

        def broken_factory(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("second rotary table failed")
            return _factory(*args)

        with self.assertRaisesRegex(RuntimeError, "second rotary table failed"):
            self._state(factory=broken_factory)
        self.assertTrue(
            all(
                tensor is None or tensor.is_meta
                for _name, tensor in self.model.named_buffers(remove_duplicate=False)
            )
        )
        self.assertIn(("exit", RuntimeError), self.events)
        recovered = self._state()
        recovered.close()

    def test_missing_engram_hash_factory_fails_before_admission(self):
        with self.assertRaisesRegex(ValueError, "pinned Engram hash factory"):
            load_session_state(
                self.model,
                self.args,
                device="cpu",
                frequency_factory=_factory,
                admission=lambda _estimate: self.fail("admission ran before Engram factory validation"),
                fence=lambda: None,
                reset_runtime_references=self._reset,
                release_allocator=lambda: None,
            )
        self.assertFalse(any(isinstance(event, tuple) and event[0] == "admission" for event in self.events))

    def test_concurrent_load_is_rejected_until_active_state_closes(self):
        factory_entered = threading.Event()
        continue_factory = threading.Event()
        result = {}

        def blocking_factory(*args):
            factory_entered.set()
            if not continue_factory.wait(timeout=5):
                raise TimeoutError("test did not release the rotary factory")
            return _factory(*args)

        def load_in_thread():
            try:
                result["state"] = self._state(factory=blocking_factory)
            except BaseException as exc:
                result["error"] = exc

        thread = threading.Thread(target=load_in_thread)
        thread.start()
        try:
            self.assertTrue(factory_entered.wait(timeout=5))
            admissions_before = sum(
                isinstance(event, tuple) and event[0] == "admission" for event in self.events
            )
            with self.assertRaisesRegex(RuntimeError, "already has a session load or active session"):
                self._state()
            admissions_after = sum(
                isinstance(event, tuple) and event[0] == "admission" for event in self.events
            )
            self.assertEqual(admissions_after, admissions_before)
        finally:
            continue_factory.set()
            thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertNotIn("error", result)
        result["state"].close()
        reloaded = self._state()
        reloaded.close()

    def test_quarantined_load_keeps_model_lifecycle_locked(self):
        def broken_factory(*_args):
            raise RuntimeError("rotary creation failed")

        def broken_fence():
            raise RuntimeError("failed-load fence failed")

        with self.assertRaisesRegex(RuntimeError, "rotary creation failed"):
            self._state(factory=broken_factory, fence=broken_fence)
        with self.assertRaisesRegex(RuntimeError, "already has a session load or active session"):
            self._state()

    def test_close_fence_failure_quarantines_and_keeps_model_lifecycle_locked(self):
        fence_calls = 0

        def broken_fence():
            nonlocal fence_calls
            fence_calls += 1
            if fence_calls > 1:
                raise RuntimeError("GPU fence failed")

        state = self._state(fence=broken_fence)
        with self.assertRaisesRegex(RuntimeError, "GPU fence failed"):
            state.close()
        self.assertTrue(state._quarantined)
        self.assertFalse(any(isinstance(event, tuple) and event[0] == "exit" for event in self.events))
        with self.assertRaisesRegex(RuntimeError, "already has a session load or active session"):
            self._state()

    def test_concurrent_close_releases_reservation_once(self):
        state = self._state()
        barrier = threading.Barrier(3)
        failures = []

        def close():
            try:
                barrier.wait()
                state.close()
            except BaseException as exc:
                failures.append(exc)

        threads = [threading.Thread(target=close) for _ in range(2)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join()
        self.assertEqual(failures, [])
        self.assertEqual(sum(isinstance(event, tuple) and event[0] == "exit" for event in self.events), 1)


class ComfyUISmokeGuardTest(unittest.TestCase):
    def test_inactive_comfyui_needs_no_queue_probe_or_service_change(self):
        with (
            patch.object(run_dense_checkpoint_smoke, "_comfyui_state", return_value="inactive"),
            patch.object(run_dense_checkpoint_smoke, "_comfyui_queue_idle") as queue,
            patch.object(run_dense_checkpoint_smoke, "_comfyui_systemctl") as control,
        ):
            with run_dense_checkpoint_smoke._comfyui_smoke_window():
                pass
        queue.assert_not_called()
        control.assert_not_called()

    def test_active_comfyui_jobs_fail_closed_without_stopping_service(self):
        with (
            patch.object(run_dense_checkpoint_smoke, "_comfyui_state", return_value="active"),
            patch.object(run_dense_checkpoint_smoke, "_comfyui_guard_ready", return_value=True),
            patch.object(run_dense_checkpoint_smoke, "_comfyui_queue_idle", return_value=False),
            patch.object(run_dense_checkpoint_smoke, "_comfyui_systemctl") as control,
        ):
            with self.assertRaisesRegex(RuntimeError, "active or pending work"):
                with run_dense_checkpoint_smoke._comfyui_smoke_window():
                    self.fail("checkpoint work started over an active ComfyUI job")
        control.assert_not_called()

    def test_active_comfyui_without_submission_barrier_fails_closed(self):
        with (
            patch.object(run_dense_checkpoint_smoke, "_comfyui_state", return_value="active"),
            patch.object(run_dense_checkpoint_smoke, "_comfyui_guard_ready", return_value=False),
            patch.object(run_dense_checkpoint_smoke, "_comfyui_queue_idle") as queue,
            patch.object(run_dense_checkpoint_smoke, "_comfyui_systemctl") as control,
        ):
            with self.assertRaisesRegex(RuntimeError, "guard is not ready"):
                with run_dense_checkpoint_smoke._comfyui_smoke_window():
                    self.fail("checkpoint work ran without the direct submission barrier")
        queue.assert_not_called()
        control.assert_not_called()

    def test_idle_comfyui_is_stopped_for_the_window_and_restored(self):
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "comfyui-restore.json"
            with (
                patch.object(
                    run_dense_checkpoint_smoke,
                    "_comfyui_state",
                    side_effect=["active", "inactive", "active"],
                ),
                patch.object(run_dense_checkpoint_smoke, "_comfyui_guard_ready", return_value=True),
                patch.object(run_dense_checkpoint_smoke, "_comfyui_queue_idle", return_value=True),
                patch.object(run_dense_checkpoint_smoke, "_comfyui_systemctl") as control,
            ):
                with run_dense_checkpoint_smoke._comfyui_smoke_window(marker):
                    self.assertTrue(marker.exists())
                    self.assertEqual(control.call_args_list, [call("stop")])
            self.assertEqual(control.call_args_list, [call("stop"), call("start")])
            self.assertFalse(marker.exists())

    def test_comfyui_is_restored_when_checkpoint_run_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "comfyui-restore.json"
            with (
                patch.object(
                    run_dense_checkpoint_smoke,
                    "_comfyui_state",
                    side_effect=["active", "inactive", "active"],
                ),
                patch.object(run_dense_checkpoint_smoke, "_comfyui_guard_ready", return_value=True),
                patch.object(run_dense_checkpoint_smoke, "_comfyui_queue_idle", return_value=True),
                patch.object(run_dense_checkpoint_smoke, "_comfyui_systemctl") as control,
            ):
                with self.assertRaisesRegex(RuntimeError, "checkpoint failure"):
                    with run_dense_checkpoint_smoke._comfyui_smoke_window(marker):
                        raise RuntimeError("checkpoint failure")
            self.assertEqual(control.call_args_list, [call("stop"), call("start")])
            self.assertFalse(marker.exists())

    def test_failed_service_restore_leaves_a_marker_for_the_supervisor(self):
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "comfyui-restore.json"
            with (
                patch.object(run_dense_checkpoint_smoke, "_comfyui_state", return_value="active"),
                patch.object(run_dense_checkpoint_smoke, "_comfyui_guard_ready", return_value=True),
                patch.object(run_dense_checkpoint_smoke, "_comfyui_queue_idle", return_value=True),
                patch.object(run_dense_checkpoint_smoke, "_wait_for_comfyui_state"),
                patch.object(
                    run_dense_checkpoint_smoke,
                    "_comfyui_systemctl",
                    side_effect=[None, RuntimeError("start failed")],
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "start failed"):
                    with run_dense_checkpoint_smoke._comfyui_smoke_window(marker):
                        pass
            self.assertTrue(marker.is_file())


class AdmissionTest(unittest.TestCase):
    def test_checkpoint_smoke_requires_the_host_recovery_lease_directory(self):
        expected = run_dense_checkpoint_smoke._HOST_LEASE_DIRECTORY
        self.assertEqual(
            run_dense_checkpoint_smoke._validate_lease_directory(expected),
            expected.resolve(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "custom paths bypass host recovery coordination"):
                run_dense_checkpoint_smoke._validate_lease_directory(Path(temporary))

    def test_host_admission_keeps_eight_gib_minimum_plus_five_gib_headroom(self):
        estimate = SimpleNamespace(
            device_peak_bytes=1 * run_dense_checkpoint_smoke._GIB,
            physical_peak_bytes=2 * run_dense_checkpoint_smoke._GIB,
        )
        required_available = estimate.physical_peak_bytes + 13 * run_dense_checkpoint_smoke._GIB
        gtt_snapshot = 100 * run_dense_checkpoint_smoke._GIB
        with patch.object(
            run_dense_checkpoint_smoke,
            "_memory_snapshot",
            return_value=(gtt_snapshot, required_available - 1, gtt_snapshot),
        ):
            with self.assertRaisesRegex(RuntimeError, "8 GiB host minimum plus 5 GiB prefill headroom"):
                run_dense_checkpoint_smoke._check_admission(estimate)
        with patch.object(
            run_dense_checkpoint_smoke,
            "_memory_snapshot",
            return_value=(gtt_snapshot, required_available, gtt_snapshot),
        ):
            self.assertEqual(
                run_dense_checkpoint_smoke._check_admission(estimate),
                (gtt_snapshot, required_available, gtt_snapshot),
            )


if __name__ == "__main__":
    unittest.main()
