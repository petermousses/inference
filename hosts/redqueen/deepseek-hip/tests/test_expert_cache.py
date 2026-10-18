"""Lease and eviction behavior for the DeepSeek FP4 expert cache."""

import concurrent.futures
import gc
import threading
import unittest
import weakref

from expert_cache import (
    ExpertCacheCapacityError,
    ExpertCacheError,
    FP4ExpertCache,
    FP4ExpertKey,
)
from expert_weights import LoadedFP4Expert


class FakeScale:
    def __init__(self, nbytes):
        self.nbytes = nbytes


class FakeWeight:
    def __init__(self, nbytes):
        self.nbytes = nbytes
        self.scale = FakeScale(0)


def make_expert(key, resident_bytes):
    quotient, remainder = divmod(resident_bytes, 3)
    weights = [FakeWeight(quotient + (index < remainder)) for index in range(3)]
    return LoadedFP4Expert(
        layer_id=key.layer_id,
        expert_id=key.expert_id,
        mtp_layer=key.mtp_layer,
        w1=weights[0],
        w2=weights[1],
        w3=weights[2],
    )


class ExpertCacheTest(unittest.TestCase):
    def setUp(self):
        self.sizes = {}
        self.loads = []

    def key(self, expert_id):
        key = FP4ExpertKey(0, expert_id)
        self.sizes[key] = 100
        return key

    def cache(self, capacity=100, loader=None, estimate=None):
        return FP4ExpertCache(
            capacity,
            estimate_resident_bytes=estimate or (lambda key: self.sizes[key]),
            load=loader or self.load,
        )

    def load(self, key):
        self.loads.append(key)
        return make_expert(key, self.sizes[key])

    def test_lru_eviction_preserves_recently_used_expert(self):
        first, second, third = (self.key(value) for value in (1, 2, 3))
        cache = self.cache(capacity=200)

        for key in (first, second):
            with cache.acquire(key):
                pass
        with cache.acquire(first):
            pass
        with cache.acquire(third):
            pass

        self.assertEqual(self.loads, [first, second, third])
        self.assertEqual(cache.stats.resident_bytes, 200)
        with cache.acquire(first):
            pass
        self.assertEqual(self.loads, [first, second, third])
        with cache.acquire(second):
            pass
        self.assertEqual(self.loads[-1], second)

    def test_active_lease_cannot_be_evicted_or_overcommitted(self):
        first, second = self.key(1), self.key(2)
        cache = self.cache()

        with cache.acquire(first):
            with self.assertRaises(ExpertCacheCapacityError):
                with cache.acquire(second):
                    self.fail("capacity conflict unexpectedly loaded an expert")
            self.assertEqual(cache.stats.leased_bytes, 100)
            self.assertEqual(self.loads, [first])

        with cache.acquire(second):
            self.assertEqual(cache.stats.resident_bytes, 100)
        self.assertEqual(self.loads, [first, second])

    def test_impossible_admission_preserves_idle_entries(self):
        first, second, too_large = self.key(1), self.key(2), self.key(3)
        self.sizes[too_large] = 101
        cache = self.cache(capacity=100)
        with cache.acquire(first):
            pass

        with self.assertRaises(ExpertCacheCapacityError):
            with cache.acquire(too_large):
                self.fail("an oversized expert was unexpectedly admitted")
        self.assertEqual(cache.stats.resident_bytes, 100)
        self.assertEqual(cache.stats.entries, 1)
        with cache.acquire(first):
            pass
        self.assertEqual(self.loads, [first])

        cache = self.cache(capacity=200)
        for key in (first, second):
            with cache.acquire(key):
                pass
        with cache.acquire(first):
            too_large = self.key(4)
            self.sizes[too_large] = 101
            with self.assertRaises(ExpertCacheCapacityError):
                with cache.acquire(too_large):
                    self.fail("an infeasible expert was unexpectedly admitted")
            self.assertEqual(cache.stats.resident_bytes, 200)
            self.assertEqual(cache.stats.entries, 2)

    def test_resize_defers_eviction_until_leases_finish(self):
        first, second = self.key(1), self.key(2)
        cache = self.cache(capacity=200)
        for key in (first, second):
            with cache.acquire(key):
                pass

        with cache.acquire(first):
            resized = cache.resize(0)
            self.assertEqual(resized.evicted_logical_bytes, 100)
            self.assertFalse(resized.stats.within_capacity)
            self.assertEqual(resized.stats.leased_bytes, 100)
            with self.assertRaises(ExpertCacheCapacityError):
                with cache.acquire(second):
                    self.fail("new lease extended a cache that is shrinking")

        self.assertTrue(cache.stats.within_capacity)
        self.assertEqual(cache.stats.resident_bytes, 0)
        self.assertEqual(cache.stats.entries, 0)

    def test_concurrent_acquisition_loads_one_copy(self):
        key = self.key(1)
        loader_started = threading.Event()
        allow_load = threading.Event()
        both_leased = threading.Barrier(2)

        def slow_load(requested_key):
            self.loads.append(requested_key)
            loader_started.set()
            if not allow_load.wait(timeout=3):
                raise TimeoutError("test did not release expert loader")
            return make_expert(requested_key, self.sizes[requested_key])

        cache = self.cache(loader=slow_load)

        def use_expert():
            with cache.acquire(key) as expert:
                both_leased.wait(timeout=3)
                return id(expert.w1)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(use_expert)
            self.assertTrue(loader_started.wait(timeout=2))
            second = pool.submit(use_expert)
            allow_load.set()
            identities = (first.result(timeout=3), second.result(timeout=3))

        self.assertEqual(identities[0], identities[1])
        self.assertEqual(self.loads, [key])
        self.assertEqual(cache.stats.entries, 1)
        self.assertEqual(cache.stats.leased_bytes, 0)

    def test_concurrent_admission_waits_for_evicted_references_to_drop(self):
        first, second, third, fourth, fifth = (
            self.key(value) for value in (1, 2, 3, 4, 5)
        )
        cache = self.cache(capacity=300)
        for key in (first, second, third):
            with cache.acquire(key):
                pass

        release_started = threading.Event()
        allow_release = threading.Event()
        request_started = threading.Event()
        load_started = {fourth: threading.Event(), fifth: threading.Event()}
        original_drop = cache._drop_entries
        original_estimate = cache._estimate
        original_load = cache._load
        first_drop = True

        def delayed_drop(entries):
            nonlocal first_drop
            if first_drop:
                first_drop = False
                release_started.set()
                if not allow_release.wait(timeout=3):
                    raise TimeoutError("test did not release evicted references")
            return original_drop(entries)

        def track_estimate(key):
            if key == fifth:
                request_started.set()
            return original_estimate(key)

        def track_load(key):
            if key in load_started:
                load_started[key].set()
            return original_load(key)

        cache._drop_entries = delayed_drop
        cache._estimate = track_estimate
        cache._load = track_load

        def acquire_once(key):
            with cache.acquire(key):
                return key

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            first_admission = pool.submit(acquire_once, fourth)
            self.assertTrue(release_started.wait(timeout=2))
            second_admission = pool.submit(acquire_once, fifth)
            self.assertTrue(request_started.wait(timeout=2))
            self.assertFalse(load_started[fourth].wait(timeout=0.1))
            self.assertFalse(load_started[fifth].is_set())
            self.assertEqual(cache.stats.releasing_bytes, 100)

            allow_release.set()
            self.assertEqual(first_admission.result(timeout=3), fourth)
            self.assertEqual(second_admission.result(timeout=3), fifth)

        self.assertTrue(load_started[fourth].is_set())
        self.assertTrue(load_started[fifth].is_set())
        self.assertTrue(cache.stats.within_capacity)
        self.assertEqual(cache.stats.releasing_bytes, 0)

    def test_yielded_lease_invalidates_after_exit(self):
        key = self.key(1)
        cache = self.cache()
        with cache.acquire(key) as lease:
            self.assertEqual(lease.resident_bytes, 100)
        with self.assertRaisesRegex(ExpertCacheError, "no longer active"):
            _ = lease.w1
        cache.resize(0)
        self.assertEqual(cache.stats.resident_bytes, 0)

    def test_lease_wrapper_does_not_keep_evicted_expert_alive(self):
        key = self.key(1)
        references = []

        def record_expert(requested_key):
            expert = make_expert(requested_key, self.sizes[requested_key])
            references.append(weakref.ref(expert))
            return expert

        cache = self.cache(loader=record_expert)
        with cache.acquire(key) as lease:
            self.assertIsNotNone(lease.w1)
        cache.resize(0)
        gc.collect()
        self.assertIsNone(references[0]())

    def test_failed_load_releases_reservation_for_retry(self):
        key = self.key(1)
        calls = 0

        def fail_once(requested_key):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("synthetic read failure")
            return make_expert(requested_key, self.sizes[requested_key])

        cache = self.cache(loader=fail_once)
        with self.assertRaisesRegex(OSError, "synthetic read failure"):
            with cache.acquire(key):
                pass
        self.assertEqual(cache.stats.loading_bytes, 0)
        self.assertEqual(cache.stats.pending_loads, 0)

        with cache.acquire(key):
            pass
        self.assertEqual(calls, 2)
        self.assertEqual(cache.stats.resident_bytes, 100)

    def test_loader_must_match_estimate_and_key(self):
        key = self.key(1)
        cache = self.cache(loader=lambda requested: make_expert(requested, 99))
        with self.assertRaisesRegex(ExpertCacheError, "estimated 100"):
            with cache.acquire(key):
                pass
        self.assertEqual(cache.stats.loading_bytes, 0)
        self.assertEqual(cache.stats.resident_bytes, 0)

        wrong_key = self.key(2)
        cache = self.cache(loader=lambda requested: make_expert(key, 100))
        with self.assertRaisesRegex(ExpertCacheError, "different key"):
            with cache.acquire(wrong_key):
                pass
        self.assertEqual(cache.stats.loading_bytes, 0)
        self.assertEqual(cache.stats.resident_bytes, 0)


class ExpertCacheKeyTest(unittest.TestCase):
    def test_validates_ids_and_mtp_flag(self):
        with self.assertRaises(TypeError):
            FP4ExpertKey(True, 0)
        with self.assertRaises(ValueError):
            FP4ExpertKey(0, -1)
        with self.assertRaises(TypeError):
            FP4ExpertKey(0, 0, mtp_layer=1)


if __name__ == "__main__":
    unittest.main()
