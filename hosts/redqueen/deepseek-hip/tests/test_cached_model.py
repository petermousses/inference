from __future__ import annotations

import asyncio
import threading
import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import cached_model
from cached_model import install_cached_moe
from expert_cache import ExpertCacheResize


class _Cache:
    def __init__(self, events=None):
        self.quarantined = False
        self.events = events if events is not None else []
        self.resize_calls = []
        self._stats = SimpleNamespace(
            capacity_bytes=100,
            resident_bytes=0,
            loading_bytes=0,
            releasing_bytes=0,
            leased_bytes=0,
            idle_bytes=0,
            entries=0,
            pending_loads=0,
            pending_backing_release=False,
            quarantined=False,
            within_capacity=True,
        )

    def acquire(self, key):
        raise AssertionError("the patched-forward tests replace execute_cached_moe")

    def quarantine(self):
        self.quarantined = True
        self._stats = SimpleNamespace(**{**vars(self._stats), "quarantined": True})

    @property
    def stats(self):
        return self._stats

    def resize_when_idle(self, capacity_bytes, *, after_drop=None):
        self.events.append("resize")
        self.resize_calls.append(capacity_bytes)
        pending_release = (
            self._stats.pending_backing_release
            or capacity_bytes < self._stats.capacity_bytes
        )
        self._stats = SimpleNamespace(
            **{
                **vars(self._stats),
                "capacity_bytes": capacity_bytes,
                "pending_backing_release": pending_release,
            }
        )
        if after_drop is not None and pending_release:
            after_drop()
            self._stats = SimpleNamespace(
                **{**vars(self._stats), "pending_backing_release": False}
            )
        return ExpertCacheResize(self._stats, evicted_logical_bytes=64)


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


def _installation(cache, fence):
    return install_cached_moe(
        SimpleNamespace(layers=[], mtp=[]),
        cache,
        lambda values, weight: values,
        fence=fence,
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

    def test_rejects_duplicate_installation_and_allows_reinstall_after_close(self):
        moe = _moe(0)
        original = moe.forward
        transformer = SimpleNamespace(layers=[SimpleNamespace(ffn=moe)], mtp=[])
        first_cache = _Cache()
        first = install_cached_moe(
            transformer, first_cache, lambda values, weight: values
        )
        first_forward = moe.forward

        with self.assertRaisesRegex(RuntimeError, "already belongs"):
            install_cached_moe(
                transformer, _Cache(), lambda values, weight: values
            )
        self.assertIs(moe.forward, first_forward)
        first.close()
        self.assertIs(moe.forward, original)

        second = install_cached_moe(
            transformer, _Cache(), lambda values, weight: values
        )
        self.assertIsNot(moe.forward, first_forward)
        second.close()
        self.assertIs(moe.forward, original)

    def test_concurrent_installation_has_one_owner(self):
        marker_check_barrier = threading.Barrier(2)

        class BarrierMoe(SimpleNamespace):
            def __getattribute__(self, name):
                if name == cached_model._INSTALLATION_MARKER:
                    try:
                        return super().__getattribute__(name)
                    except AttributeError:
                        try:
                            marker_check_barrier.wait(timeout=0.1)
                        except threading.BrokenBarrierError:
                            pass
                        raise
                return super().__getattribute__(name)

        moe = BarrierMoe(**vars(_moe(0)))
        transformer = SimpleNamespace(layers=[SimpleNamespace(ffn=moe)], mtp=[])

        def install():
            try:
                return "installed", install_cached_moe(
                    transformer, _Cache(), lambda values, weight: values
                )
            except RuntimeError as error:
                return "rejected", error

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(lambda _: install(), range(2)))
        installations = [value for state, value in outcomes if state == "installed"]
        rejected = [value for state, value in outcomes if state == "rejected"]
        self.assertEqual(len(installations), 1)
        self.assertEqual(len(rejected), 1)
        self.assertRegex(str(rejected[0]), "already belongs")
        installations[0].close()


class ReservedSyncOperationTest(unittest.IsolatedAsyncioTestCase):
    async def test_cache_resize_rejects_async_allocator_callback_before_mutation(self):
        events = []
        cache = _Cache(events)

        async def release_allocator():
            events.append("async-allocator-release")

        with _installation(cache, lambda: events.append("fence")) as installation:
            with self.assertRaisesRegex(TypeError, "release_allocator must be synchronous"):
                await installation.resize_cache(0, release_allocator=release_allocator)

        self.assertEqual(cache.resize_calls, [])
        self.assertEqual(cache.stats.capacity_bytes, 100)
        self.assertFalse(cache.stats.pending_backing_release)
        self.assertEqual(events, [])

    async def test_cache_resize_rejects_async_fence_before_mutation(self):
        events = []
        cache = _Cache(events)

        async def fence():
            events.append("async-fence")

        with _installation(cache, fence) as installation:
            with self.assertRaisesRegex(TypeError, "GPU fence must be synchronous"):
                await installation.resize_cache(
                    0, release_allocator=lambda: events.append("allocator-release")
                )

        self.assertEqual(cache.resize_calls, [])
        self.assertEqual(cache.stats.capacity_bytes, 100)
        self.assertFalse(cache.stats.pending_backing_release)
        self.assertEqual(events, [])

    async def test_cache_resize_keeps_release_pending_for_awaitable_result(self):
        events = []
        cache = _Cache(events)
        release_body_calls = []

        async def async_release_body():
            release_body_calls.append("released")

        def awaitable_release():
            return async_release_body()

        with _installation(cache, lambda: events.append("fence")) as installation:
            with self.assertRaisesRegex(TypeError, "returned an awaitable"):
                await installation.resize_cache(0, release_allocator=awaitable_release)

            self.assertTrue(cache.stats.pending_backing_release)
            self.assertEqual(cache.resize_calls, [0])
            self.assertEqual(release_body_calls, [])

            result = await installation.resize_cache(
                0, release_allocator=lambda: events.append("allocator-release")
            )

        self.assertFalse(result.stats.pending_backing_release)
        self.assertEqual(cache.resize_calls, [0, 0])
        self.assertEqual(release_body_calls, [])
        self.assertIn("allocator-release", events)

    async def test_cache_resize_waits_for_concurrent_future_before_unlocking(self):
        events = []
        cache = _Cache(events)
        release_started = threading.Event()
        finish_release = threading.Event()
        second_started = threading.Event()
        future = Future()
        self.assertTrue(future.set_running_or_notify_cancel())

        def finish_future():
            release_started.set()
            if not finish_release.wait(3):
                future.set_exception(TimeoutError("test did not finish allocator future"))
            else:
                future.set_result(None)

        completer = threading.Thread(target=finish_future, daemon=True)
        completer.start()
        with _installation(cache, lambda: events.append("fence")) as installation:
            resizing = asyncio.create_task(
                installation.resize_cache(0, release_allocator=lambda: future)
            )
            self.assertTrue(await asyncio.to_thread(release_started.wait, 2))
            second = asyncio.create_task(
                installation.run_reserved_sync(
                    lambda: second_started.set(), _Reservation(events)
                )
            )
            await asyncio.sleep(0.05)
            self.assertFalse(resizing.done())
            self.assertFalse(second_started.is_set())
            finish_release.set()
            with self.assertRaisesRegex(TypeError, "returned a future"):
                await resizing
            await second

        completer.join(timeout=2)
        self.assertTrue(future.done())
        self.assertTrue(cache.stats.pending_backing_release)
        self.assertEqual(cache.resize_calls, [0])
        self.assertTrue(second_started.is_set())

    async def test_cache_resize_cancels_and_drains_task_cleanup_before_unlocking(self):
        events = []
        cache = _Cache(events)
        cleanup_started = threading.Event()
        finish_cleanup = threading.Event()

        async def background_release():
            try:
                await asyncio.Event().wait()
            finally:
                cleanup_started.set()
                if not await asyncio.to_thread(finish_cleanup.wait, 3):
                    raise TimeoutError("test did not finish async cleanup")

        task = asyncio.create_task(background_release())
        await asyncio.sleep(0)
        with _installation(cache, lambda: events.append("fence")) as installation:
            resizing = asyncio.create_task(
                installation.resize_cache(0, release_allocator=lambda: task)
            )
            try:
                self.assertTrue(await asyncio.to_thread(cleanup_started.wait, 2))
                await asyncio.sleep(0.05)
                self.assertFalse(resizing.done())
            finally:
                finish_cleanup.set()
            with self.assertRaisesRegex(TypeError, "returned a future"):
                await resizing

        self.assertTrue(task.done())
        self.assertTrue(cache.stats.pending_backing_release)
        self.assertEqual(cache.resize_calls, [0])

    async def test_cache_resize_quarantines_when_opaque_awaitable_cannot_be_drained(self):
        events = []
        cache = _Cache(events)

        class OpaqueAwaitable:
            def __await__(self):
                yield

        with _installation(cache, lambda: events.append("fence")) as installation:
            with self.assertRaisesRegex(TypeError, "worker restart is required"):
                await installation.resize_cache(
                    0, release_allocator=OpaqueAwaitable
                )

            self.assertTrue(cache.quarantined)
            with self.assertRaisesRegex(RuntimeError, "worker restart is required"):
                await installation.run_reserved_sync(lambda: None, _Reservation(events))

        self.assertTrue(cache.stats.pending_backing_release)
        self.assertEqual(cache.resize_calls, [0])

    async def test_cache_resize_fences_and_releases_allocator_at_worker_boundary(self):
        events = []
        cache = _Cache(events)

        class WorkerContext:
            def __enter__(self):
                events.append("worker-enter")

            def __exit__(self, exc_type, exc, traceback):
                events.append("worker-exit")

        with _installation(cache, lambda: events.append("fence")) as installation:
            result = await installation.resize_cache(
                32,
                release_allocator=lambda: events.append("allocator-release"),
                worker_context=WorkerContext,
            )

        self.assertEqual(
            events,
            ["worker-enter", "fence", "resize", "allocator-release", "fence", "worker-exit"],
        )
        self.assertEqual(cache.resize_calls, [32])
        self.assertEqual(result.evicted_logical_bytes, 64)
        self.assertEqual(result.stats.capacity_bytes, 32)

    async def test_cache_resize_fence_failure_quarantines_without_releasing(self):
        events = []
        cache = _Cache(events)
        fence_calls = 0

        def fence():
            nonlocal fence_calls
            fence_calls += 1
            events.append("fence")
            if fence_calls == 1:
                raise RuntimeError("GPU fence failed")

        with _installation(cache, fence) as installation:
            with self.assertRaisesRegex(RuntimeError, "GPU fence failed"):
                await installation.resize_cache(
                    0, release_allocator=lambda: events.append("allocator-release")
                )
            self.assertTrue(installation._quarantined)
            with self.assertRaisesRegex(RuntimeError, "quarantined"):
                await installation.run_reserved_sync(lambda: None, _Reservation(events))
            with self.assertRaisesRegex(RuntimeError, "undrained callback"):
                installation.close()

        self.assertTrue(cache.quarantined)
        self.assertEqual(events, ["fence"])
        self.assertEqual(cache.resize_calls, [])

    async def test_cache_resize_post_release_fence_failure_quarantines_installation(self):
        events = []
        cache = _Cache(events)
        fence_calls = 0

        def fence():
            nonlocal fence_calls
            fence_calls += 1
            events.append("fence")
            if fence_calls == 2:
                raise RuntimeError("post-release GPU fence failed")

        with _installation(cache, fence) as installation:
            with self.assertRaisesRegex(RuntimeError, "post-release GPU fence failed"):
                await installation.resize_cache(
                    0, release_allocator=lambda: events.append("allocator-release")
                )
            self.assertTrue(installation._quarantined)
            with self.assertRaisesRegex(RuntimeError, "quarantined"):
                await installation.run_reserved_sync(lambda: None, _Reservation(events))

        self.assertTrue(cache.quarantined)
        self.assertEqual(cache.resize_calls, [0])
        self.assertIn("allocator-release", events)

    async def test_cancelled_resize_marks_installation_quarantined_after_fence_failure(self):
        events = []
        cache = _Cache(events)
        fence_started = threading.Event()
        finish_fence = threading.Event()
        fence_calls = 0

        def fence():
            nonlocal fence_calls
            fence_calls += 1
            if fence_calls == 2:
                fence_started.set()
                if not finish_fence.wait(3):
                    raise TimeoutError("test did not release final fence")
                raise RuntimeError("final fence failed")

        with _installation(cache, fence) as installation:
            resizing = asyncio.create_task(
                installation.resize_cache(
                    0, release_allocator=lambda: events.append("allocator-release")
                )
            )
            self.assertTrue(await asyncio.to_thread(fence_started.wait, 2))
            resizing.cancel()
            await asyncio.sleep(0.05)
            self.assertFalse(resizing.done())
            finish_fence.set()
            with self.assertRaises(asyncio.CancelledError):
                await resizing

            self.assertTrue(installation._quarantined)
            with self.assertRaisesRegex(RuntimeError, "quarantined"):
                await installation.run_reserved_sync(lambda: None, _Reservation(events))

        self.assertTrue(cache.quarantined)

    async def test_cache_resize_waits_until_model_execution_is_fenced(self):
        events = []
        cache = _Cache(events)
        reservation = _Reservation(events)
        operation_started = threading.Event()
        release_operation = threading.Event()
        resize_started = threading.Event()

        def operation():
            operation_started.set()
            if not release_operation.wait(3):
                raise TimeoutError("test did not release model operation")
            events.append("operation-finished")

        real_resize = cache.resize_when_idle

        def resize(capacity_bytes, *, after_drop=None):
            resize_started.set()
            return real_resize(capacity_bytes, after_drop=after_drop)

        cache.resize_when_idle = resize
        with _installation(cache, lambda: events.append("fence")) as installation:
            execution = asyncio.create_task(
                installation.run_reserved_sync(operation, reservation)
            )
            self.assertTrue(await asyncio.to_thread(operation_started.wait, 2))
            resizing = asyncio.create_task(
                installation.resize_cache(
                    32, release_allocator=lambda: events.append("allocator-release")
                )
            )
            await asyncio.sleep(0.05)
            self.assertFalse(resize_started.is_set())
            self.assertFalse(resizing.done())
            release_operation.set()
            await execution
            await resizing

        self.assertTrue(resize_started.is_set())
        self.assertLess(events.index("fence"), events.index("resize"))
        self.assertLess(events.index("allocator-release"), len(events))

    async def test_cancelled_resize_keeps_execution_lock_until_allocator_returns(self):
        events = []
        cache = _Cache(events)
        reservation = _Reservation(events)
        allocator_started = threading.Event()
        release_allocator = threading.Event()
        second_started = threading.Event()

        def slow_allocator_release():
            events.append("allocator-start")
            allocator_started.set()
            if not release_allocator.wait(3):
                raise TimeoutError("test did not release allocator")
            events.append("allocator-finished")

        with _installation(cache, lambda: events.append("fence")) as installation:
            resize = asyncio.create_task(
                installation.resize_cache(
                    32, release_allocator=slow_allocator_release
                )
            )
            self.assertTrue(await asyncio.to_thread(allocator_started.wait, 2))
            resize.cancel()
            with self.assertRaises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(resize), timeout=0.05)

            second = asyncio.create_task(
                installation.run_reserved_sync(
                    lambda: (second_started.set(), events.append("second-finished")),
                    reservation,
                )
            )
            await asyncio.sleep(0.05)
            self.assertFalse(second_started.is_set())
            self.assertFalse(reservation.active)
            release_allocator.set()
            with self.assertRaises(asyncio.CancelledError):
                await resize
            await second

        self.assertTrue(second_started.is_set())
        self.assertEqual(
            events,
            [
                "fence",
                "resize",
                "allocator-start",
                "allocator-finished",
                "fence",
                "enter",
                "second-finished",
                "fence",
                "exit",
            ],
        )

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

        with _installation(cache, fence) as installation:
            result = await installation.run_reserved_sync(operation, reservation)
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

        with _installation(cache, fence) as installation:
            with self.assertRaisesRegex(LookupError, "model failure"):
                await installation.run_reserved_sync(operation, reservation)
        self.assertEqual(events, ["enter", "operation", "fence", "exit"])

    async def test_worker_context_wraps_operation_and_restores_before_fence(self):
        events = []
        reservation = _Reservation(events)
        cache = _Cache()

        class WorkerContext:
            def __enter__(self):
                events.append("worker-enter")

            def __exit__(self, exc_type, exc, traceback):
                events.append("worker-exit")

        with _installation(cache, lambda: events.append("fence")) as installation:
            await installation.run_reserved_sync(
                lambda: events.append("operation"),
                reservation,
                worker_context=WorkerContext,
            )
        self.assertEqual(
            events,
            ["enter", "worker-enter", "operation", "worker-exit", "fence", "exit"],
        )

    async def test_fence_failure_quarantines_before_reservation_release(self):
        events = []
        reservation = _Reservation(events)
        cache = _Cache()

        def fence():
            self.assertTrue(reservation.active)
            events.append("fence")
            raise RuntimeError("GPU fence failed")

        with _installation(cache, fence) as installation:
            with self.assertRaisesRegex(RuntimeError, "GPU fence failed"):
                await installation.run_reserved_sync(
                    lambda: events.append("operation"), reservation
                )
            self.assertTrue(installation._quarantined)
            with self.assertRaisesRegex(RuntimeError, "quarantined"):
                await installation.run_reserved_sync(lambda: None, _Reservation(events))
            with self.assertRaisesRegex(RuntimeError, "undrained callback"):
                installation.close()
        self.assertTrue(cache.quarantined)
        self.assertEqual(events, ["enter", "operation", "fence", "exit"])

    async def test_cancelled_waiter_keeps_reservation_until_worker_finishes(self):
        events = []
        reservation = _Reservation(events)
        second_reservation = _Reservation(events)
        cache = _Cache()
        started = threading.Event()
        second_started = threading.Event()
        release = threading.Event()

        def operation():
            started.set()
            if not release.wait(5):
                raise TimeoutError("test worker was not released")
            events.append("operation-finished")

        def fence():
            self.assertTrue(reservation.active or second_reservation.active)
            events.append("fence")

        with _installation(cache, fence) as installation:
            task = asyncio.create_task(
                installation.run_reserved_sync(operation, reservation)
            )
            second_task = None
            try:
                started_in_time = await asyncio.to_thread(started.wait, 2)
                self.assertTrue(started_in_time)
                task.cancel()
                with self.assertRaises(asyncio.TimeoutError):
                    await asyncio.wait_for(asyncio.shield(task), timeout=0.05)
                self.assertTrue(reservation.active)
                self.assertNotIn("exit", events)
                second_task = asyncio.create_task(
                    installation.run_reserved_sync(
                        lambda: (second_started.set(), events.append("second-finished")),
                        second_reservation,
                    )
                )
                await asyncio.sleep(0.05)
                self.assertFalse(second_started.is_set())
                self.assertFalse(second_reservation.active)
            finally:
                release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            if second_task is not None:
                await second_task
        self.assertFalse(reservation.active)
        self.assertFalse(second_reservation.active)
        self.assertEqual(
            events,
            [
                "enter",
                "operation-finished",
                "fence",
                "exit",
                "enter",
                "second-finished",
                "fence",
                "exit",
            ],
        )

    async def test_model_execution_is_serialized_before_memory_admission(self):
        cache = _Cache()
        events = []
        first_reservation = _Reservation(events)
        second_reservation = _Reservation(events)
        first_started = threading.Event()
        second_started = threading.Event()
        release_first = threading.Event()
        active_guard = threading.Lock()
        active = 0
        peak_active = 0

        def operation(started, release=None):
            def run():
                nonlocal active, peak_active
                with active_guard:
                    active += 1
                    peak_active = max(peak_active, active)
                started.set()
                if release is not None and not release.wait(2):
                    raise TimeoutError("first operation was not released")
                with active_guard:
                    active -= 1
            return run

        def fence():
            self.assertTrue(first_reservation.active or second_reservation.active)

        with _installation(cache, fence) as installation:
            first = asyncio.create_task(
                installation.run_reserved_sync(
                    operation(first_started, release_first), first_reservation
                )
            )
            self.assertTrue(await asyncio.to_thread(first_started.wait, 2))
            second = asyncio.create_task(
                installation.run_reserved_sync(
                    operation(second_started), second_reservation
                )
            )
            await asyncio.sleep(0.05)
            self.assertFalse(second_started.is_set())
            self.assertFalse(second_reservation.active)
            with self.assertRaisesRegex(RuntimeError, "during model work"):
                installation.close()

            release_first.set()
            await asyncio.gather(first, second)
            self.assertTrue(second_started.is_set())
            self.assertEqual(peak_active, 1)

        with self.assertRaisesRegex(RuntimeError, "already closed"):
            await installation.run_reserved_sync(lambda: None, _Reservation([]))


if __name__ == "__main__":
    unittest.main()
