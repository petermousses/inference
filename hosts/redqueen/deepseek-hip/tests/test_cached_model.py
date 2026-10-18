from __future__ import annotations

import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from cached_model import install_cached_moe, run_reserved_sync


class _Cache:
    def __init__(self):
        self.quarantined = False

    def acquire(self, key):
        raise AssertionError("the patched-forward tests replace execute_cached_moe")

    def quarantine(self):
        self.quarantined = True


class _Reservation:
    def __init__(self, events):
        self.events = events
        self.active = False

    async def __aenter__(self):
        self.active = True
        self.events.append("enter")

    async def __aexit__(self, exc_type, exc, traceback):
        self.events.append("exit")
        self.active = False


def _moe(layer_id, expert_count=4, swiglu_limit=10.0):
    def original(activations, image_mask=None):
        return "reference", activations, image_mask

    def shared_expert(activations):
        return activations

    shared_expert.swiglu_limit = swiglu_limit

    return SimpleNamespace(
        layer_id=layer_id,
        n_routed_experts=expert_count,
        gate=lambda activations, image_mask=None: (activations, image_mask),
        shared_experts=shared_expert,
        forward=original,
    )


class CachedMoEInstallationTest(unittest.TestCase):
    def test_patches_backbone_and_mtp_with_local_checkpoint_keys(self):
        backbone = [_moe(0, 4), _moe(1, 4)]
        mtp = [_moe(2, 3), _moe(3, 3)]
        originals = [moe.forward for moe in (*backbone, *mtp)]
        transformer = SimpleNamespace(
            layers=[SimpleNamespace(ffn=moe) for moe in backbone],
            mtp=[SimpleNamespace(ffn=moe) for moe in mtp],
        )
        cache = _Cache()
        linear = lambda values, weight: values
        calls = []

        def execute(*args, **kwargs):
            calls.append((args, kwargs))
            return kwargs["layer_id"], kwargs["mtp_layer"]

        with patch("cached_model.execute_cached_moe", execute):
            with install_cached_moe(transformer, cache, linear) as installation:
                self.assertEqual(backbone[0].forward("x", "mask"), (0, False))
                self.assertEqual(backbone[1].forward("x", None), (1, False))
                self.assertEqual(mtp[0].forward("x", "mask"), (0, True))
                self.assertEqual(mtp[1].forward("x", None), (1, True))
                self.assertEqual(len(calls), 4)
                expected = [
                    (0, False, 4),
                    (1, False, 4),
                    (0, True, 3),
                    (1, True, 3),
                ]
                self.assertEqual(
                    [
                        (
                            call[1]["layer_id"],
                            call[1]["mtp_layer"],
                            call[1]["n_routed_experts"],
                        )
                        for call in calls
                    ],
                    expected,
                )
                self.assertIs(calls[0][0][0], cache)
                self.assertIs(calls[0][0][4], linear)
                self.assertEqual(calls[0][1]["image_mask"], "mask")
                self.assertIsNone(calls[1][1]["image_mask"])
                self.assertFalse(installation._closed)

        self.assertEqual(
            [moe.forward for moe in (*backbone, *mtp)], originals
        )

    def test_validates_entire_model_before_patching_any_moe(self):
        first = _moe(0)
        invalid_mtp = _moe(8)
        original = first.forward
        transformer = SimpleNamespace(
            layers=[SimpleNamespace(ffn=first)],
            mtp=[SimpleNamespace(ffn=invalid_mtp)],
        )
        with self.assertRaisesRegex(ValueError, r"mtp\[0\]"):
            install_cached_moe(transformer, _Cache(), lambda values, weight: values)
        self.assertIs(first.forward, original)

    def test_rejects_changes_to_a_patched_forward_before_restoring(self):
        moe = _moe(0)
        transformer = SimpleNamespace(layers=[SimpleNamespace(ffn=moe)], mtp=[])
        installation = install_cached_moe(
            transformer, _Cache(), lambda values, weight: values
        )
        moe.forward = lambda values, image_mask=None: "other"
        with self.assertRaisesRegex(RuntimeError, "changed"):
            installation.close()
        self.assertFalse(installation._closed)


class ReservedSyncOperationTest(unittest.IsolatedAsyncioTestCase):
    async def test_reservation_stays_held_through_final_gpu_fence(self):
        events = []
        reservation = _Reservation(events)
        cache = _Cache()

        def operation():
            events.append("operation")
            return "result"

        def fence():
            self.assertTrue(reservation.active)
            events.append("fence")

        result = await run_reserved_sync(
            operation,
            reservation,
            cache=cache,
            fence=fence,
        )
        self.assertEqual(result, "result")
        self.assertEqual(events, ["enter", "operation", "fence", "exit"])
        self.assertFalse(reservation.active)

    async def test_operation_error_is_raised_after_fence_and_release(self):
        events = []
        reservation = _Reservation(events)
        cache = _Cache()

        def operation():
            events.append("operation")
            raise LookupError("model failure")

        def fence():
            self.assertTrue(reservation.active)
            events.append("fence")

        with self.assertRaisesRegex(LookupError, "model failure"):
            await run_reserved_sync(
                operation, reservation, cache=cache, fence=fence
            )
        self.assertEqual(events, ["enter", "operation", "fence", "exit"])

    async def test_fence_failure_quarantines_before_reservation_release(self):
        events = []
        reservation = _Reservation(events)
        cache = _Cache()

        def fence():
            self.assertTrue(reservation.active)
            events.append("fence")
            raise RuntimeError("GPU fence failed")

        with self.assertRaisesRegex(RuntimeError, "GPU fence failed"):
            await run_reserved_sync(
                lambda: events.append("operation"),
                reservation,
                cache=cache,
                fence=fence,
            )
        self.assertTrue(cache.quarantined)
        self.assertEqual(events, ["enter", "operation", "fence", "exit"])

    async def test_cancelled_waiter_keeps_reservation_until_worker_finishes(self):
        events = []
        reservation = _Reservation(events)
        cache = _Cache()
        started = threading.Event()
        release = threading.Event()

        def operation():
            started.set()
            if not release.wait(5):
                raise TimeoutError("test worker was not released")
            events.append("operation-finished")

        def fence():
            self.assertTrue(reservation.active)
            events.append("fence")

        task = asyncio.create_task(
            run_reserved_sync(
                operation, reservation, cache=cache, fence=fence
            )
        )
        try:
            started_in_time = await asyncio.to_thread(started.wait, 2)
            self.assertTrue(started_in_time)
            task.cancel()
            with self.assertRaises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(task), timeout=0.05)
            self.assertTrue(reservation.active)
            self.assertNotIn("exit", events)
        finally:
            release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(reservation.active)
        self.assertEqual(events, ["enter", "operation-finished", "fence", "exit"])


if __name__ == "__main__":
    unittest.main()
