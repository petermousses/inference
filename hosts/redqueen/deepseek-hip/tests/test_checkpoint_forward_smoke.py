from __future__ import annotations

import unittest
import weakref
from types import SimpleNamespace

from checkpoint_forward_smoke import (
    _drain_request_state,
    _SystemMemorySampler,
)
from run_dense_checkpoint_smoke import (
    _boundary_peak_budget,
    _check_boundary_admission,
    _check_boundary_memory,
)


_GIB = 1024**3


class RequestDrainTest(unittest.TestCase):
    def test_shared_scratch_is_dropped_before_allocator_release(self):
        class Scratch:
            pass

        shared = SimpleNamespace(topk_idxs=Scratch())
        reference = weakref.ref(shared.topk_idxs)
        events = []

        class Cache:
            def resize_when_idle(self, capacity, *, after_drop):
                events.append(("resize", capacity))
                after_drop()

        def reset():
            events.append("reset")
            shared.topk_idxs = None

        def release():
            events.append("release")
            self.assertIsNone(reference(), "request scratch survived until session close")

        _drain_request_state(
            cache=Cache(),
            fence=lambda: events.append("fence"),
            reset_runtime_references=reset,
            release_allocator=release,
        )
        self.assertEqual(events, ["fence", "reset", ("resize", 0), "release", "fence"])

    def test_scratch_backing_is_released_even_when_expert_cache_is_empty(self):
        events = []
        cache = SimpleNamespace(resize_when_idle=lambda *_args, **_kwargs: None)
        _drain_request_state(
            cache=cache,
            fence=lambda: events.append("fence"),
            reset_runtime_references=lambda: events.append("reset"),
            release_allocator=lambda: events.append("release"),
        )
        self.assertEqual(events, ["fence", "reset", "release", "fence"])

    def test_failed_initial_fence_keeps_request_references(self):
        reset = []

        def failed_fence():
            raise RuntimeError("unfinished GPU work")

        with self.assertRaisesRegex(RuntimeError, "unfinished GPU work"):
            _drain_request_state(
                cache=None,
                fence=failed_fence,
                reset_runtime_references=lambda: reset.append(True),
                release_allocator=lambda: self.fail("released before GPU completion"),
            )
        self.assertEqual(reset, [])


class BoundaryPeakBudgetTest(unittest.TestCase):
    def test_peak_budget_uses_remaining_managed_gtt(self):
        budget = _boundary_peak_budget(lambda: (80 * _GIB, 90 * _GIB, 100 * _GIB))

        self.assertEqual(budget, 68 * _GIB)

    def test_peak_budget_uses_host_floor_and_headroom(self):
        budget = _boundary_peak_budget(lambda: (90 * _GIB, 50 * _GIB, 100 * _GIB))

        self.assertEqual(budget, 37 * _GIB)

    def test_peak_budget_subtracts_resident_managed_physical_memory(self):
        snapshot = lambda: (24 * _GIB, 30 * _GIB, 94 * _GIB)
        budget = _boundary_peak_budget(
            snapshot,
            managed_physical_resident_bytes=95 * _GIB,
        )

        self.assertEqual(budget, 9 * _GIB)
        _check_boundary_admission(
            SimpleNamespace(device_peak_bytes=budget, physical_peak_bytes=budget),
            snapshot,
            managed_physical_resident_bytes=95 * _GIB,
        )
        with self.assertRaisesRegex(RuntimeError, "104 GiB managed physical-memory ceiling"):
            _check_boundary_admission(
                SimpleNamespace(
                    device_peak_bytes=10 * _GIB,
                    physical_peak_bytes=10 * _GIB,
                ),
                snapshot,
                managed_physical_resident_bytes=95 * _GIB,
            )

    def test_peak_budget_rejects_no_remaining_capacity(self):
        with self.assertRaisesRegex(RuntimeError, "no 220k prefill capacity"):
            _boundary_peak_budget(lambda: (5 * _GIB, 13 * _GIB, 100 * _GIB))

    def test_admission_rechecks_live_gtt_ceiling_after_sizing(self):
        budget = _boundary_peak_budget(lambda: (90 * _GIB, 120 * _GIB, 100 * _GIB))
        estimate = SimpleNamespace(
            device_peak_bytes=budget,
            physical_peak_bytes=budget,
        )

        with self.assertRaisesRegex(RuntimeError, "88 GiB managed GTT ceiling"):
            _check_boundary_admission(
                estimate,
                lambda: (83 * _GIB, 120 * _GIB, 100 * _GIB),
            )

    def test_admission_enforces_physical_ceiling(self):
        estimate = SimpleNamespace(
            device_peak_bytes=1 * _GIB,
            physical_peak_bytes=105 * _GIB,
        )

        with self.assertRaisesRegex(RuntimeError, "104 GiB managed physical-memory ceiling"):
            _check_boundary_admission(
                estimate,
                lambda: (90 * _GIB, 120 * _GIB, 100 * _GIB),
            )

    def test_sampler_reports_minima_and_maximum_managed_gtt_use(self):
        snapshots = iter(
            (
                (80 * _GIB, 50 * _GIB, 100 * _GIB),
                (11 * _GIB, 40 * _GIB, 100 * _GIB),
            )
        )
        sampler = _SystemMemorySampler(lambda: next(snapshots), interval=60)

        sampler.start()
        result = sampler.stop()

        self.assertEqual(result["min_free_gtt"], 11 * _GIB)
        self.assertEqual(result["min_host_available"], 40 * _GIB)
        self.assertEqual(result["max_sampled_used_gtt"], 89 * _GIB)
        self.assertEqual(result["gtt_total"], 100 * _GIB)

    def test_memory_check_enforces_gtt_and_headroom_limits(self):
        passing = {
            "min_free_gtt": 5 * _GIB,
            "min_host_available": 13 * _GIB,
            "max_sampled_used_gtt": 88 * _GIB,
            "gtt_total": 100 * _GIB,
        }
        _check_boundary_memory(passing)

        with self.assertRaisesRegex(RuntimeError, "88 GiB managed GTT ceiling"):
            _check_boundary_memory(
                {**passing, "max_sampled_used_gtt": 88 * _GIB + 1}
            )

        with self.assertRaisesRegex(RuntimeError, "5 GiB GTT headroom"):
            _check_boundary_memory({**passing, "min_free_gtt": 5 * _GIB - 1})

        with self.assertRaisesRegex(RuntimeError, "8 GiB host minimum plus 5 GiB headroom"):
            _check_boundary_memory(
                {**passing, "min_host_available": 13 * _GIB - 1}
            )


if __name__ == "__main__":
    unittest.main()
