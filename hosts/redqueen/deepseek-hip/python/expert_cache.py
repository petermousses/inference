"""Lease-aware, byte-bounded LRU cache for pinned DeepSeek FP4 experts."""

from __future__ import annotations

import threading
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator

from expert_weights import LoadedFP4Expert


@dataclass(frozen=True, order=True)
class FP4ExpertKey:
    layer_id: int
    expert_id: int
    mtp_layer: bool = False

    def __post_init__(self) -> None:
        for value, name in ((self.layer_id, "layer_id"), (self.expert_id, "expert_id")):
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if type(self.mtp_layer) is not bool:
            raise TypeError("mtp_layer must be a bool")


class ExpertCacheCapacityError(RuntimeError):
    """The requested expert cannot fit without evicting a leased expert."""


class ExpertCacheError(RuntimeError):
    """The loader returned an expert inconsistent with its key or estimate."""


@dataclass
class _Entry:
    expert: LoadedFP4Expert | None
    resident_bytes: int
    leases: int = 0


@dataclass(frozen=True)
class ExpertCacheStats:
    capacity_bytes: int
    resident_bytes: int
    loading_bytes: int
    releasing_bytes: int
    leased_bytes: int
    idle_bytes: int
    entries: int
    pending_loads: int
    quarantined: bool

    @property
    def within_capacity(self) -> bool:
        return (
            self.resident_bytes + self.loading_bytes + self.releasing_bytes
            <= self.capacity_bytes
        )


@dataclass(frozen=True)
class ExpertCacheResize:
    stats: ExpertCacheStats
    evicted_logical_bytes: int


class ExpertCacheLease:
    """Temporary view of a loaded expert; invalidated when its lease ends."""

    def __init__(self, cache: FP4ExpertCache, key: FP4ExpertKey, entry: _Entry) -> None:
        self._cache = cache
        self._key = key
        self._entry = entry
        self._expert = entry.expert
        self._quarantined = False

    def _get_expert(self) -> LoadedFP4Expert:
        if self._expert is None:
            raise ExpertCacheError("expert lease is no longer active")
        return self._expert

    @property
    def layer_id(self) -> int:
        return self._get_expert().layer_id

    @property
    def expert_id(self) -> int:
        return self._get_expert().expert_id

    @property
    def mtp_layer(self) -> bool:
        return self._get_expert().mtp_layer

    @property
    def w1(self):
        return self._get_expert().w1

    @property
    def w2(self):
        return self._get_expert().w2

    @property
    def w3(self):
        return self._get_expert().w3

    @property
    def resident_bytes(self) -> int:
        return self._get_expert().resident_bytes

    def _release(self) -> None:
        cache, key, entry = self._cache, self._key, self._entry
        if cache is None:
            return
        self._cache = None
        self._key = None
        self._entry = None
        self._expert = None
        if self._quarantined:
            # Keep the cache entry permanently leased. Its GPU completion state
            # is unknown, so only worker teardown may release these references.
            return
        cache._release_entry(key, entry)

    def quarantine(self) -> None:
        """Pin this entry and prevent cache reuse after an unconfirmed GPU fence."""
        if self._cache is None:
            raise ExpertCacheError("expert lease is no longer active")
        self._quarantined = True
        self._cache.quarantine()


class FP4ExpertCache:
    """Cache experts while protecting every entry held by an active lease.

    Capacity measures logical weight and scale payload bytes. Eviction drops the
    cache's tensor references, and new acquisitions wait until those references
    are dropped. This does not claim that a device allocator returned its backing
    to the operating system. A host memory broker must separately confirm
    physical memory returned before admitting another worker.

    Callers must hold a lease until all GPU work using the expert has completed
    and its device fence has passed. Do not retain the expert beyond the ``with``
    block; the yielded lease invalidates on exit. Do not retain individual tensor
    references beyond the lease. If acquisition runs in a worker thread, keep the
    root memory reservation until that thread exits; cancelling an async waiter
    does not stop the thread.
    """

    def __init__(
        self,
        capacity_bytes: int,
        *,
        estimate_resident_bytes: Callable[[FP4ExpertKey], int],
        load: Callable[[FP4ExpertKey], LoadedFP4Expert],
    ) -> None:
        self._validate_capacity(capacity_bytes)
        if not callable(estimate_resident_bytes) or not callable(load):
            raise TypeError("estimate_resident_bytes and load must be callable")
        self._capacity_bytes = capacity_bytes
        self._estimate = estimate_resident_bytes
        self._load = load
        self._condition = threading.Condition()
        self._entries: OrderedDict[FP4ExpertKey, _Entry] = OrderedDict()
        self._loading: dict[FP4ExpertKey, int] = {}
        self._resident_bytes = 0
        self._loading_bytes = 0
        self._releasing_bytes = 0
        self._quarantined = False

    @staticmethod
    def _quarantined_error() -> ExpertCacheError:
        return ExpertCacheError("expert cache is quarantined; worker restart is required")

    def quarantine(self) -> None:
        """Reject new work after a device fence fails; restart to clear it."""
        with self._condition:
            self._quarantined = True
            self._condition.notify_all()

    @staticmethod
    def _validate_capacity(capacity_bytes: int) -> None:
        if type(capacity_bytes) is not int or capacity_bytes < 0:
            raise ValueError("cache capacity must be a nonnegative integer")

    @staticmethod
    def _validate_size(size: int, name: str) -> None:
        if type(size) is not int or size <= 0:
            raise ValueError(f"{name} must be a positive integer")

    def _stats_locked(self) -> ExpertCacheStats:
        leased_bytes = sum(
            entry.resident_bytes for entry in self._entries.values() if entry.leases
        )
        return ExpertCacheStats(
            capacity_bytes=self._capacity_bytes,
            resident_bytes=self._resident_bytes,
            loading_bytes=self._loading_bytes,
            releasing_bytes=self._releasing_bytes,
            leased_bytes=leased_bytes,
            idle_bytes=self._resident_bytes - leased_bytes,
            entries=len(self._entries),
            pending_loads=len(self._loading),
            quarantined=self._quarantined,
        )

    @property
    def stats(self) -> ExpertCacheStats:
        with self._condition:
            return self._stats_locked()

    def _evict_idle_locked(self, required_bytes: int = 0) -> list[_Entry]:
        evicted = []
        while (
            self._resident_bytes + self._loading_bytes + required_bytes
            > self._capacity_bytes
        ):
            key = next(
                (key for key, entry in self._entries.items() if entry.leases == 0),
                None,
            )
            if key is None:
                break
            entry = self._entries.pop(key)
            self._resident_bytes -= entry.resident_bytes
            evicted.append(entry)
        self._releasing_bytes += sum(entry.resident_bytes for entry in evicted)
        return evicted

    @staticmethod
    def _drop_entries(entries: list[_Entry]) -> int:
        released = sum(entry.resident_bytes for entry in entries)
        for entry in entries:
            entry.expert = None
        entries.clear()
        return released

    def _finish_eviction(self, entries: list[_Entry]) -> None:
        released = sum(entry.resident_bytes for entry in entries)
        try:
            self._drop_entries(entries)
        finally:
            with self._condition:
                self._releasing_bytes -= released
                self._condition.notify_all()

    def _capacity_error_locked(self, request_bytes: int) -> ExpertCacheCapacityError:
        stats = self._stats_locked()
        return ExpertCacheCapacityError(
            "expert cache cannot admit "
            f"{request_bytes} bytes: capacity={stats.capacity_bytes}, "
            f"resident={stats.resident_bytes}, loading={stats.loading_bytes}, "
            f"releasing={stats.releasing_bytes}, leased={stats.leased_bytes}"
        )

    @contextmanager
    def acquire(self, key: FP4ExpertKey) -> Iterator[ExpertCacheLease]:
        """Lease an expert, loading it once and evicting idle LRU entries as needed."""
        if not isinstance(key, FP4ExpertKey):
            raise TypeError("key must be an FP4ExpertKey")
        estimate = self._estimate(key)
        self._validate_size(estimate, "estimated resident bytes")
        entry = None
        expert = None
        loading = False
        while True:
            dropped = []
            capacity_error = None
            with self._condition:
                while key in self._loading or self._releasing_bytes:
                    if self._quarantined:
                        raise self._quarantined_error()
                    self._condition.wait()
                if self._quarantined:
                    raise self._quarantined_error()
                if self._resident_bytes + self._loading_bytes > self._capacity_bytes:
                    capacity_error = self._capacity_error_locked(estimate)
                else:
                    entry = self._entries.get(key)
                    if entry is not None:
                        if entry.expert is None:
                            raise ExpertCacheError("resident expert entry lost its tensors")
                        entry.leases += 1
                        self._entries.move_to_end(key)
                        expert = entry.expert
                    else:
                        stats = self._stats_locked()
                        needed_bytes = (
                            stats.resident_bytes
                            + stats.loading_bytes
                            + estimate
                            - stats.capacity_bytes
                        )
                        if needed_bytes > stats.idle_bytes:
                            capacity_error = self._capacity_error_locked(estimate)
                        elif needed_bytes > 0:
                            dropped = self._evict_idle_locked(required_bytes=estimate)
                        else:
                            self._loading[key] = estimate
                            self._loading_bytes += estimate
                            loading = True
            if capacity_error is not None:
                raise capacity_error
            if entry is not None or loading:
                break
            # Do not reserve the incoming allocation until the old tensor
            # references have been dropped. Other acquisitions wait on the
            # releasing counter, so they cannot race this handoff either.
            if dropped:
                self._finish_eviction(dropped)

        if loading:
            try:
                expert = self._load(key)
                if not isinstance(expert, LoadedFP4Expert):
                    raise ExpertCacheError("loader did not return a LoadedFP4Expert")
                if (expert.layer_id, expert.expert_id, expert.mtp_layer) != (
                    key.layer_id,
                    key.expert_id,
                    key.mtp_layer,
                ):
                    raise ExpertCacheError("loader returned an expert for a different key")
                actual_bytes = expert.resident_bytes
                if type(actual_bytes) is not int or actual_bytes != estimate:
                    raise ExpertCacheError(
                        f"loaded expert uses {actual_bytes} bytes; estimated {estimate}"
                    )
            except BaseException:
                expert = None
                with self._condition:
                    reserved = self._loading.pop(key, None)
                    if reserved is not None:
                        self._loading_bytes -= reserved
                    dropped = self._evict_idle_locked()
                    self._condition.notify_all()
                self._finish_eviction(dropped)
                raise

            with self._condition:
                self._loading_bytes -= self._loading.pop(key)
                entry = _Entry(expert=expert, resident_bytes=estimate, leases=1)
                self._entries[key] = entry
                self._entries.move_to_end(key)
                self._resident_bytes += estimate
                self._condition.notify_all()
                if self._quarantined:
                    raise self._quarantined_error()

        lease = ExpertCacheLease(self, key, entry)
        try:
            yield lease
        finally:
            expert = None
            lease._release()
            entry = None

    def _release_entry(self, key: FP4ExpertKey, entry: _Entry) -> None:
        with self._condition:
            if self._entries.get(key) is not entry or entry.leases < 1:
                raise ExpertCacheError("expert lease state was corrupted")
            entry.leases -= 1
            dropped = self._evict_idle_locked()
            self._condition.notify_all()
        self._finish_eviction(dropped)

    def resize(self, capacity_bytes: int) -> ExpertCacheResize:
        """Set the logical payload cap and evict idle LRU entries immediately.

        If active leases or loads prevent reaching the cap, the smaller target is
        retained. New acquisitions fail until releases bring the cache within it.
        """
        self._validate_capacity(capacity_bytes)
        with self._condition:
            if self._quarantined:
                raise self._quarantined_error()
            self._capacity_bytes = capacity_bytes
            dropped = self._evict_idle_locked()
            evicted_bytes = sum(entry.resident_bytes for entry in dropped)
        self._finish_eviction(dropped)
        with self._condition:
            stats = self._stats_locked()
        return ExpertCacheResize(stats=stats, evicted_logical_bytes=evicted_bytes)
