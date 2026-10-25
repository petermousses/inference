from __future__ import annotations

import asyncio
import json
import math
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from redqueen_adapters.common import AdmissionError, UpstreamError
from redqueen_adapters import deploy_scheduler
from redqueen_adapters.image_api import ImageSettings, create_image_app
from redqueen_adapters.jev_api import JevSettings, create_jev_app
from redqueen_adapters.memory_profile import MIB, build_profiles, file_signature, fingerprint_artifacts, memory_contract, memory_rows, verify_profiles, weight_paths
from redqueen_adapters.ownership import OwnedRequest, records
from redqueen_adapters import scheduler_api
from redqueen_adapters.scheduler_api import ExpertCacheResizeError, ExpertCacheStateError, Memory, Runtime, Scheduler, create_apps, footprint
from redqueen_adapters.scheduler_client import (
    ResourceLock,
    SchedulerClient,
    SerializedExecution,
    run_thread_shielded,
)


class SSEFrameMetricsTests(unittest.TestCase):
    def test_chat_completion_terminal_markers_are_recognized(self):
        done = b"data: [DONE]\n\n"
        finish = b'data: {"choices":[{"finish_reason":"stop"}]}\n\n'

        self.assertTrue(scheduler_api._sse_frame_is_terminal(done, "/v1/chat/completions"))
        self.assertTrue(scheduler_api._sse_frame_is_terminal(finish, "/v1/chat/completions"))
        self.assertFalse(
            scheduler_api._sse_frame_is_terminal(
                b'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}\n\n',
                "/v1/chat/completions",
            )
        )

    def test_responses_terminal_event_is_recognized(self):
        frame = (
            b"event: response.completed\n"
            b'data: {"type":"response.completed","response":{"status":"completed"}}\n\n'
        )

        self.assertTrue(scheduler_api._sse_frame_is_terminal(frame, "/v1/responses"))

    def test_stream_shape_counters_distinguish_tool_calls_without_logging_arguments(self):
        payload = {
            "choices": [{
                "delta": {
                    "content": "",
                    "tool_calls": [{
                        "index": 0,
                        "function": {
                            "name": "lookup",
                            "arguments": "private tool arguments",
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"completion_tokens": 7},
        }
        frame = b"data: " + json.dumps(payload).encode() + b"\n\n"

        metrics = scheduler_api._sse_frame_metrics(frame)

        self.assertEqual(metrics["text_chars"], 0)
        self.assertEqual(metrics["tool_call_deltas"], 1)
        self.assertEqual(metrics["function_call_deltas"], 0)
        self.assertEqual(metrics["delta_fields"], {"content", "tool_calls"})
        self.assertEqual(metrics["finish_reasons"], {"tool_calls"})
        self.assertEqual(metrics["completion_tokens"], 7)
        self.assertNotIn("private tool arguments", repr(metrics))


class CacheReservationRoundingTests(unittest.TestCase):
    def test_decimal_multiplier_does_not_overreserve_on_integer_product(self):
        multiplier = 1.08
        available_bytes = 56_969_981_176
        deepseek_delta = 25_883_320_320
        glm_delta = 26_866_662_250

        deepseek_reservation = scheduler_api._reservation_for_cache_delta(
            deepseek_delta, multiplier
        )
        glm_reservation = scheduler_api._reservation_for_cache_delta(
            glm_delta, multiplier
        )

        self.assertEqual(deepseek_reservation, 27_953_985_946)
        self.assertEqual(glm_reservation, 29_015_995_230)
        self.assertEqual(math.ceil(glm_delta * multiplier), glm_reservation + 1)
        self.assertEqual(deepseek_reservation + glm_reservation, available_bytes)
        self.assertEqual(
            scheduler_api._cache_delta_for_reservation(glm_reservation, multiplier),
            glm_delta,
        )


class RequestShapeTests(unittest.TestCase):
    def test_chat_history_with_null_assistant_content_is_classified(self):
        body = {
            "messages": [
                {"role": "assistant", "content": None, "tool_calls": [{"type": "function"}]},
                {"role": "tool", "content": "done"},
                {"role": "user", "content": "continue"},
            ]
        }

        self.assertEqual(
            scheduler_api.request_shape("/v1/chat/completions", body),
            "openai.chat",
        )

    def test_nested_audio_content_is_still_detected(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_audio", "input_audio": {"data": "..."}}
                    ],
                }
            ]
        }

        self.assertEqual(
            scheduler_api.request_shape("/v1/chat/completions", body),
            "openai.multimodal_audio",
        )


def resource(size: int, kind: str = "text") -> dict:
    return {"gpu_bytes": size, "host_bytes": size, "kind": kind, "concurrency": 1, "context_tokens": 32768}


def adaptive_resource() -> dict:
    minimum, maximum = 10, 130
    memory = {
        "mandatory": {"gpu_bytes": minimum, "host_bytes": minimum},
        "reclaimable": {"gpu_bytes": maximum - minimum, "host_bytes": maximum - minimum},
        "transient_peak": {"gpu_bytes": 20, "host_bytes": 20},
    }
    return {
        "kind": "service",
        "gateway_url": "http://127.0.0.1:8085",
        "gpu_bytes": maximum + 20,
        "host_bytes": maximum + 20,
        "memory": memory,
        "adaptive_cache": {
            "minimum_capacity_bytes": minimum,
            "maximum_capacity_bytes": maximum,
            "reservation_multiplier": 1,
        },
        "expert_cache_url": "http://127.0.0.1:8085/expert-cache",
        "concurrency_limit": 1,
    }


class FakeRuntime:
    def __init__(self, resources: dict, *, capacity: int = 100):
        self.resources = resources
        self.limit = Memory(capacity, capacity)
        self.loaded: dict[str, str] = {}
        self.occupied: set[str] = set()
        self.stops: list[str] = []
        self.starts: list[str] = []
        self.external = Memory(0, 0)
        self.starting = asyncio.Event()
        self.load_gate: asyncio.Event | None = None
        self.stop_failure = False
        self.load_failure = False
        self.available = True

    async def healthy(self):
        return self.available

    async def snapshot(self):
        return dict(self.loaded)

    async def ready(self, name):
        return self.loaded.get(name) == "ready"

    async def busy(self, name):
        return name in self.occupied

    async def start(self, name):
        self.starts.append(name)
        self.loaded[name] = "starting"
        self.starting.set()
        if self.load_gate is not None:
            await self.load_gate.wait()
        if self.load_failure:
            raise RuntimeError("load failure")
        self.loaded[name] = "ready"

    async def stop(self, name):
        self.stops.append(name)
        if self.stop_failure:
            raise RuntimeError("stop failure")
        self.loaded.pop(name, None)

    def capacity(self):
        used = sum((Memory(self.resources[name]["gpu_bytes"], self.resources[name]["host_bytes"]) for name in self.loaded), self.external)
        return self.limit, used


class FakeAdaptiveRuntime(FakeRuntime):
    def __init__(self, resources: dict, *, capacity: int = 100):
        super().__init__(resources, capacity=capacity)
        self.cache_capacities = {
            name: spec["adaptive_cache"]["minimum_capacity_bytes"]
            for name, spec in resources.items()
            if spec.get("adaptive_cache")
        }
        self.resize_calls: list[tuple[str, int]] = []
        self.scheduler = None
        self.sticky_cache_backing = False
        self.physical_cache_capacities = dict(self.cache_capacities)
        self.cache_capacities_at_start: dict[str, dict[str, int]] = {}
        self.reject_cache_above: dict[str, int] = {}
        self.cache_rounding_quantum = 1
        self.pending_cache_release: set[str] = set()
        self.cache_resize_overshoots: dict[tuple[str, int], int] = {}
        self.capacity_calls = 0
        self.capacity_hook = None

    async def start(self, name):
        self.cache_capacities_at_start[name] = dict(self.cache_capacities)
        await super().start(name)

    async def expert_cache_info(self, name, *, allow_pending_backing_release=False):
        limits = self.resources[name]["adaptive_cache"]
        pending = name in self.pending_cache_release
        if pending and not allow_pending_backing_release:
            raise ExpertCacheStateError("pending_backing_release")
        return {
            **limits,
            "capacity_bytes": self.cache_capacities[name],
            "pending_backing_release": pending,
            "quarantined": False,
        }

    async def resize_expert_cache(self, name, capacity_bytes, *, operation_id=None):
        self.resize_calls.append((name, capacity_bytes))
        reject_above = self.reject_cache_above.get(name)
        if reject_above is not None and capacity_bytes > reject_above:
            raise ExpertCacheResizeError(503)
        minimum = self.resources[name]["adaptive_cache"]["minimum_capacity_bytes"]
        capacity_bytes = self.cache_resize_overshoots.get(
            (name, capacity_bytes),
            max(
                minimum,
                capacity_bytes // self.cache_rounding_quantum * self.cache_rounding_quantum,
            ),
        )
        self.cache_capacities[name] = capacity_bytes
        if not self.sticky_cache_backing or capacity_bytes > self.physical_cache_capacities[name]:
            self.physical_cache_capacities[name] = capacity_bytes
        self.pending_cache_release.discard(name)
        return await self.expert_cache_info(name)

    def capacity(self):
        self.capacity_calls += 1
        if self.capacity_hook is not None:
            self.capacity_hook(self.capacity_calls)
        used = self.external
        for name in self.loaded:
            spec = self.resources[name]
            profile = spec["memory"]
            mandatory = profile["mandatory"]
            peak = profile["transient_peak"]
            limits = spec["adaptive_cache"]
            extra = max(0, self.physical_cache_capacities[name] - limits["minimum_capacity_bytes"])
            active = self.scheduler is not None and self.scheduler.active.get(name, 0) > 0
            used += Memory(
                mandatory["gpu_bytes"] + extra + (peak["gpu_bytes"] if active else 0),
                mandatory["host_bytes"] + extra + (peak["host_bytes"] if active else 0),
            )
        return self.limit, used


class SchedulerTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.resources = {"a": resource(40), "b": resource(40), "large": resource(75)}
        self.runtime = FakeRuntime(self.resources)
        self.policy = {"load_timeout_s": 0.1, "stop_timeout_s": 0.1, "idle_timeout_s": 300}
        self.scheduler = Scheduler(self.resources, self.runtime, Path(self.directory.name) / "leases", self.policy, clock=lambda: 1000)

    def protect(self, name):
        lock = ResourceLock(self.scheduler.lock_dir, name)
        self.assertTrue(lock.acquire())
        self.addCleanup(lock.close)
        return lock

    async def test_refresh_accounts_pending_cache_backing_as_unknown_memory(self):
        resources = {name: adaptive_resource() for name in ("deepseek", "qwen")}
        runtime = FakeAdaptiveRuntime(resources, capacity=150)
        scheduler = Scheduler(
            resources, runtime, Path(self.directory.name) / "pending-cache-leases", self.policy
        )
        runtime.scheduler = scheduler
        runtime.loaded = {"deepseek": "ready"}
        runtime.cache_capacities["deepseek"] = 80
        runtime.physical_cache_capacities["deepseek"] = 120
        runtime.pending_cache_release.add("deepseek")
        scheduler.active["deepseek"] = 1

        await scheduler.refresh()

        self.assertEqual(scheduler.cache_allocations["deepseek"], 80)
        self.assertNotIn("deepseek", scheduler.cache_unready)
        capacity, usage = runtime.capacity()
        reserved = scheduler.reserved()
        unknown = Memory(max(0, usage.gpu - reserved.gpu), max(0, usage.host - reserved.host))
        self.assertTrue((reserved + unknown).fits(capacity))
        self.assertEqual(unknown, Memory(40, 40))

    async def test_lru_evicts_only_as_many_idle_models_as_required(self):
        self.runtime.loaded = {"a": "ready", "b": "ready"}
        self.scheduler.last_used = {"a": 10, "b": 20}
        await self.scheduler.admit("large")
        self.assertEqual(self.runtime.stops, ["a", "b"])
        self.assertEqual(self.runtime.loaded, {"large": "ready"})

    async def test_co_resident_models_fit_without_eviction(self):
        await self.scheduler.admit("a")
        await self.scheduler.admit("b")
        self.assertEqual(self.runtime.loaded, {"a": "ready", "b": "ready"})
        self.assertEqual(self.runtime.stops, [])

    async def test_active_moes_share_cache_and_idle_model_returns_its_share(self):
        resources = {name: adaptive_resource() for name in ("deepseek", "qwen")}
        runtime = FakeAdaptiveRuntime(resources, capacity=150)
        scheduler = Scheduler(
            resources, runtime, Path(self.directory.name) / "adaptive-leases", self.policy
        )
        runtime.scheduler = scheduler
        entered = {name: asyncio.Event() for name in resources}
        release = {name: asyncio.Event() for name in resources}

        async def serve(name):
            async with scheduler.request(name):
                runtime.occupied.add(name)
                entered[name].set()
                try:
                    await release[name].wait()
                finally:
                    runtime.occupied.discard(name)

        deepseek_task = asyncio.create_task(serve("deepseek"))
        qwen_task = None
        try:
            await asyncio.wait_for(entered["deepseek"].wait(), 1)
            self.assertEqual(runtime.cache_capacities["deepseek"], 130)

            qwen_task = asyncio.create_task(serve("qwen"))
            await asyncio.wait_for(entered["qwen"].wait(), 1)
            self.assertFalse(deepseek_task.done())
            self.assertEqual(runtime.loaded, {"deepseek": "ready", "qwen": "ready"})
            self.assertEqual(
                runtime.cache_capacities_at_start["qwen"],
                {"deepseek": 55, "qwen": 10},
            )
            self.assertEqual(runtime.cache_capacities["deepseek"], 55)
            self.assertEqual(runtime.cache_capacities["qwen"], 55)
            self.assertIn(("deepseek", 55), runtime.resize_calls)

            release["qwen"].set()
            await asyncio.wait_for(qwen_task, 1)
            deadline = asyncio.get_running_loop().time() + 1
            while runtime.cache_capacities["deepseek"] != 120:
                if asyncio.get_running_loop().time() >= deadline:
                    self.fail("active DeepSeek did not reclaim the idle Qwen cache share")
                await asyncio.sleep(0.01)
            self.assertEqual(runtime.cache_capacities["qwen"], 10)
            self.assertEqual(scheduler.active.get("deepseek"), 1)
        finally:
            release["deepseek"].set()
            release["qwen"].set()
            await asyncio.gather(deepseek_task, *([qwen_task] if qwen_task else []), return_exceptions=True)
            if scheduler.cache_rebalance_tasks:
                await asyncio.gather(*list(scheduler.cache_rebalance_tasks), return_exceptions=True)

    async def test_failed_incoming_start_restores_active_moe_cache(self):
        resources = {name: adaptive_resource() for name in ("deepseek", "qwen")}
        runtime = FakeAdaptiveRuntime(resources, capacity=150)
        scheduler = Scheduler(
            resources, runtime, Path(self.directory.name) / "failed-start-cache-leases", self.policy
        )
        runtime.scheduler = scheduler
        entered = asyncio.Event()
        release = asyncio.Event()

        async def serve_deepseek():
            async with scheduler.request("deepseek"):
                entered.set()
                await release.wait()

        deepseek_task = asyncio.create_task(serve_deepseek())
        try:
            await asyncio.wait_for(entered.wait(), 1)
            self.assertEqual(runtime.cache_capacities["deepseek"], 130)
            runtime.load_failure = True

            with self.assertRaises(AdmissionError):
                async with scheduler.request("qwen"):
                    self.fail("a worker whose startup failed must not serve")

            self.assertEqual(
                runtime.cache_capacities_at_start["qwen"],
                {"deepseek": 55, "qwen": 10},
            )
            deadline = asyncio.get_running_loop().time() + 1
            while runtime.cache_capacities["deepseek"] != 130:
                if asyncio.get_running_loop().time() >= deadline:
                    self.fail("active DeepSeek did not reclaim cache after Qwen failed to start")
                await asyncio.sleep(0.01)
            self.assertNotIn("qwen", runtime.loaded)
            self.assertEqual(scheduler.active.get("deepseek"), 1)
        finally:
            release.set()
            await asyncio.gather(deepseek_task, return_exceptions=True)
            if scheduler.cache_rebalance_tasks:
                await asyncio.gather(*list(scheduler.cache_rebalance_tasks), return_exceptions=True)

    async def test_pending_moe_requests_share_cache_before_fifo_admission(self):
        resources = {name: adaptive_resource() for name in ("qwen", "glm", "deepseek")}
        runtime = FakeAdaptiveRuntime(resources, capacity=150)
        scheduler = Scheduler(
            resources, runtime, Path(self.directory.name) / "pending-moe-leases", self.policy
        )
        runtime.scheduler = scheduler
        runtime.loaded = {"qwen": "ready"}
        scheduler.states = dict(runtime.loaded)
        scheduler.active = {"qwen": 1}
        scheduler.pending_requests = {"glm": 1, "deepseek": 1}
        runtime.cache_capacities["qwen"] = 130
        runtime.physical_cache_capacities["qwen"] = 130
        scheduler.cache_allocations["qwen"] = 130

        self.assertTrue(
            await scheduler._rebalance_caches("glm", runtime.limit, Memory(0, 0))
        )

        self.assertEqual(runtime.resize_calls, [("qwen", 30)])
        self.assertEqual(runtime.cache_capacities["qwen"], 30)

    async def test_idle_rebalance_keeps_unloaded_active_moe_in_demand(self):
        resources = {name: adaptive_resource() for name in ("qwen", "deepseek")}
        runtime = FakeAdaptiveRuntime(resources, capacity=150)
        scheduler = Scheduler(
            resources,
            runtime,
            Path(self.directory.name) / "active-unloaded-cache-leases",
            self.policy,
        )
        runtime.scheduler = scheduler
        runtime.loaded = {"qwen": "ready"}
        scheduler.states = dict(runtime.loaded)
        scheduler.active = {"deepseek": 1}
        scheduler.last_used = {"qwen": 1000}
        runtime.cache_capacities["qwen"] = 130
        runtime.physical_cache_capacities["qwen"] = 130
        scheduler.cache_allocations["qwen"] = 130

        self.assertTrue(
            await scheduler._rebalance_caches(
                None, runtime.limit, Memory(0, 0), reason="post_request_idle"
            )
        )

        self.assertEqual(runtime.resize_calls, [("qwen", 10)])
        self.assertEqual(scheduler.cache_allocations["qwen"], 10)
        self.assertEqual(scheduler.active["deepseek"], 1)

    async def test_cancelled_fifo_waiter_releases_pending_model_demand(self):
        resources = {"deepseek": adaptive_resource()}
        runtime = FakeAdaptiveRuntime(resources, capacity=150)
        scheduler = Scheduler(
            resources, runtime, Path(self.directory.name) / "pending-cancel-leases", self.policy
        )
        gate = ResourceLock(scheduler.lock_dir, "global-execution", exclusive=True)
        self.assertTrue(gate.acquire())

        async def wait_for_request_slot():
            async with scheduler.request("deepseek", "pending-cancel"):
                self.fail("request must remain behind the held execution gate")

        task = asyncio.create_task(wait_for_request_slot())
        try:
            deadline = asyncio.get_running_loop().time() + 1
            while not scheduler.pending_requests and not task.done():
                if asyncio.get_running_loop().time() >= deadline:
                    self.fail("request did not register as pending")
                await asyncio.sleep(0.005)
            self.assertEqual(scheduler.pending_requests, {"deepseek": 1})
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(scheduler.pending_requests, {})
        finally:
            gate.close()

    async def test_new_moe_load_reserves_equal_cache_share_during_startup(self):
        resources = {name: adaptive_resource() for name in ("deepseek", "glm")}
        runtime = FakeAdaptiveRuntime(resources, capacity=150)
        scheduler = Scheduler(
            resources, runtime, Path(self.directory.name) / "startup-cache-leases", self.policy
        )
        runtime.scheduler = scheduler
        runtime.loaded = {"deepseek": "ready"}
        scheduler.states = dict(runtime.loaded)
        scheduler.active = {"deepseek": 1, "glm": 1}
        runtime.cache_capacities["deepseek"] = 130
        runtime.physical_cache_capacities["deepseek"] = 130
        scheduler.cache_allocations["deepseek"] = 130

        self.assertTrue(await scheduler._rebalance_caches("glm", runtime.limit, Memory(0, 0)))

        self.assertEqual(runtime.cache_capacities["deepseek"], 55)
        self.assertEqual(runtime.cache_capacities["glm"], 10)
        self.assertEqual(runtime.resize_calls, [("deepseek", 55)])

        runtime.loaded["glm"] = "ready"
        scheduler.states["glm"] = "ready"
        self.assertTrue(await scheduler._rebalance_caches("glm", runtime.limit, Memory(0, 0)))

        self.assertEqual(runtime.cache_capacities["deepseek"], 55)
        self.assertEqual(runtime.cache_capacities["glm"], 55)

    async def test_rejected_cache_share_falls_back_to_largest_verified_allocation(self):
        resources = {name: adaptive_resource() for name in ("deepseek", "qwen")}
        runtime = FakeAdaptiveRuntime(resources, capacity=150)
        scheduler = Scheduler(
            resources, runtime, Path(self.directory.name) / "adaptive-fallback-leases", self.policy
        )
        runtime.scheduler = scheduler
        runtime.loaded = {"deepseek": "ready"}
        scheduler.states = dict(runtime.loaded)
        scheduler.active = {"deepseek": 1}
        runtime.cache_capacities["deepseek"] = 130
        runtime.physical_cache_capacities["deepseek"] = 130
        scheduler.cache_allocations["deepseek"] = 130
        runtime.reject_cache_above["deepseek"] = 52

        self.assertTrue(await scheduler._rebalance_caches("qwen", runtime.limit, Memory(0, 0)))

        self.assertEqual(runtime.cache_capacities["deepseek"], 52)
        self.assertEqual(scheduler.cache_fallback_caps["deepseek"], 52)
        self.assertEqual(
            runtime.resize_calls,
            [
                ("deepseek", 55),
                ("deepseek", 10),
                ("deepseek", 32),
                ("deepseek", 43),
                ("deepseek", 49),
                ("deepseek", 52),
            ],
        )

        runtime.loaded["qwen"] = "ready"
        runtime.cache_capacities["qwen"] = 10
        runtime.physical_cache_capacities["qwen"] = 10
        scheduler.states["qwen"] = "ready"
        scheduler.cache_allocations["qwen"] = 10
        self.assertTrue(await scheduler._rebalance_caches("qwen", runtime.limit, Memory(0, 0)))
        self.assertEqual(runtime.cache_capacities["qwen"], 58)
        self.assertTrue(runtime.capacity()[1].fits(runtime.limit))

        scheduler.active.pop("deepseek")
        self.assertTrue(await scheduler._rebalance_caches(None, runtime.limit, Memory(0, 0)))
        self.assertNotIn("deepseek", scheduler.cache_fallback_caps)

    async def test_failed_cache_growth_searches_for_largest_verified_capacity(self):
        resources = {"deepseek": adaptive_resource()}
        runtime = FakeAdaptiveRuntime(resources, capacity=100)
        scheduler = Scheduler(
            resources, runtime, Path(self.directory.name) / "growth-search-leases", self.policy
        )
        runtime.scheduler = scheduler
        runtime.loaded = {"deepseek": "ready"}
        scheduler.states = dict(runtime.loaded)
        runtime.reject_cache_above["deepseek"] = 52

        fits = await scheduler._rebalance_caches(None, runtime.limit, Memory(0, 0))

        self.assertTrue(fits)
        self.assertEqual(runtime.cache_capacities["deepseek"], 52)
        self.assertEqual(scheduler.cache_allocations["deepseek"], 52)
        self.assertEqual(scheduler.cache_fallback_caps["deepseek"], 52)
        self.assertEqual(
            runtime.resize_calls,
            [
                ("deepseek", 100),
                ("deepseek", 55),
                ("deepseek", 32),
                ("deepseek", 43),
                ("deepseek", 49),
                ("deepseek", 52),
                ("deepseek", 53),
            ],
        )
        self.assertTrue(runtime.capacity()[1].fits(runtime.limit))

        resize_calls = list(runtime.resize_calls)
        self.assertTrue(
            await scheduler._rebalance_caches(None, runtime.limit, Memory(0, 0))
        )
        self.assertEqual(runtime.resize_calls, resize_calls)

        runtime.limit = Memory(150, 150)
        self.assertTrue(
            await scheduler._rebalance_caches(None, runtime.limit, Memory(0, 0))
        )
        self.assertEqual(runtime.resize_calls[len(resize_calls)], ("deepseek", 130))

        resources["qwen"] = adaptive_resource()
        runtime.resources = resources
        runtime.cache_capacities["qwen"] = 10
        runtime.physical_cache_capacities["qwen"] = 10
        runtime.loaded["qwen"] = "ready"
        scheduler.resources = resources
        scheduler.states["qwen"] = "ready"
        scheduler.cache_allocations["qwen"] = 10
        scheduler.active.update(deepseek=1, qwen=1)
        runtime.resize_calls.clear()

        self.assertTrue(
            await scheduler._rebalance_caches("qwen", runtime.limit, Memory(0, 0))
        )
        self.assertIn(("deepseek", 55), runtime.resize_calls)
        self.assertEqual(
            scheduler.cache_fallback_demands["deepseek"], ("deepseek", "qwen")
        )

    async def test_repeated_runtime_allocation_failure_stops_growth_search(self):
        class RuntimeWithRepeatedAllocationFailure(FakeAdaptiveRuntime):
            async def resize_expert_cache(self, name, capacity_bytes, *, operation_id=None):
                self.resize_calls.append((name, capacity_bytes))
                if capacity_bytes > self.cache_capacities[name]:
                    raise ExpertCacheResizeError(
                        503,
                        reason_code="runtime_resize_failed",
                        detail="unable to allocate ROCm0 expert cache tensor blk.37.weight.stream_cache",
                    )
                return await super().resize_expert_cache(
                    name, capacity_bytes, operation_id=operation_id
                )

        resources = {"deepseek": adaptive_resource()}
        runtime = RuntimeWithRepeatedAllocationFailure(resources, capacity=100)
        scheduler = Scheduler(
            resources, runtime, Path(self.directory.name) / "repeated-allocation-failure-leases", self.policy
        )
        runtime.scheduler = scheduler
        runtime.loaded = {"deepseek": "ready"}
        scheduler.states = dict(runtime.loaded)

        self.assertTrue(await scheduler._rebalance_caches(None, runtime.limit, Memory(0, 0)))

        self.assertEqual(runtime.resize_calls, [("deepseek", 100), ("deepseek", 55)])
        self.assertEqual(runtime.cache_capacities["deepseek"], 10)
        self.assertEqual(scheduler.cache_fallback_caps["deepseek"], 10)

        resize_calls = list(runtime.resize_calls)
        self.assertTrue(await scheduler._rebalance_caches(None, runtime.limit, Memory(0, 0)))
        self.assertEqual(runtime.resize_calls, resize_calls)

    async def test_failed_cache_growth_stops_when_physical_usage_no_longer_fits(self):
        class RuntimeWithExternalPressure(FakeAdaptiveRuntime):
            async def resize_expert_cache(self, name, capacity_bytes, *, operation_id=None):
                try:
                    return await super().resize_expert_cache(
                        name, capacity_bytes, operation_id=operation_id
                    )
                except ExpertCacheResizeError:
                    self.external = Memory(141, 141)
                    raise

        resources = {"deepseek": adaptive_resource()}
        runtime = RuntimeWithExternalPressure(resources, capacity=150)
        scheduler = Scheduler(
            resources, runtime, Path(self.directory.name) / "growth-pressure-leases", self.policy
        )
        runtime.scheduler = scheduler
        runtime.loaded = {"deepseek": "ready"}
        scheduler.states = dict(runtime.loaded)
        runtime.reject_cache_above["deepseek"] = 52

        fits = await scheduler._rebalance_caches(None, runtime.limit, Memory(0, 0))

        self.assertFalse(fits)
        self.assertEqual(runtime.resize_calls, [("deepseek", 130)])
        self.assertEqual(runtime.cache_capacities["deepseek"], 10)
        self.assertEqual(scheduler.cache_allocations["deepseek"], 10)

    async def test_cache_growth_clamps_to_fresh_budget_before_resizing(self):
        resources = {"deepseek": adaptive_resource()}
        runtime = FakeAdaptiveRuntime(resources, capacity=150)
        scheduler = Scheduler(
            resources, runtime, Path(self.directory.name) / "fresh-growth-budget-leases", self.policy
        )
        runtime.scheduler = scheduler
        runtime.loaded = {"deepseek": "ready"}
        scheduler.states = dict(runtime.loaded)
        scheduler.active = {"deepseek": 1}

        def add_pressure_before_growth_check(read_number):
            if read_number == 4:
                runtime.external = Memory(60, 60)

        runtime.capacity_hook = add_pressure_before_growth_check

        fits = await scheduler._rebalance_caches(None, runtime.limit, Memory(0, 0))

        self.assertTrue(fits)
        self.assertEqual(runtime.resize_calls, [("deepseek", 70)])
        self.assertTrue((scheduler.reserved() + Memory(60, 60)).fits(runtime.limit))

    async def test_growth_that_loses_budget_mid_resize_is_rolled_back_then_replanned(self):
        class RuntimeWithLateExternalPressure(FakeAdaptiveRuntime):
            def __init__(self, resources, *, capacity):
                super().__init__(resources, capacity=capacity)
                self.pressure_injected = False

            async def resize_expert_cache(self, name, capacity_bytes, *, operation_id=None):
                growing = capacity_bytes > self.cache_capacities[name]
                info = await super().resize_expert_cache(
                    name, capacity_bytes, operation_id=operation_id
                )
                if growing and not self.pressure_injected:
                    self.pressure_injected = True
                    self.external = Memory(60, 60)
                return info

        resources = {"deepseek": adaptive_resource()}
        runtime = RuntimeWithLateExternalPressure(resources, capacity=150)
        scheduler = Scheduler(
            resources, runtime, Path(self.directory.name) / "growth-rollback-leases", self.policy
        )
        runtime.scheduler = scheduler
        runtime.loaded = {"deepseek": "ready"}
        scheduler.states = dict(runtime.loaded)
        scheduler.active = {"deepseek": 1}

        fits = await scheduler._rebalance_caches(None, runtime.limit, Memory(0, 0))

        self.assertTrue(fits)
        self.assertEqual(
            runtime.resize_calls,
            [("deepseek", 130), ("deepseek", 10), ("deepseek", 70)],
        )
        self.assertEqual(runtime.cache_capacities["deepseek"], 70)
        self.assertTrue(runtime.capacity()[1].fits(runtime.limit))

    async def test_growth_shrinks_to_minimum_if_post_resize_budget_cannot_be_measured(self):
        resources = {"deepseek": adaptive_resource()}
        runtime = FakeAdaptiveRuntime(resources, capacity=150)
        scheduler = Scheduler(
            resources,
            runtime,
            Path(self.directory.name) / "unverified-growth-budget-leases",
            self.policy,
        )
        runtime.scheduler = scheduler
        runtime.loaded = {"deepseek": "ready"}
        scheduler.states = dict(runtime.loaded)
        scheduler.active = {"deepseek": 1}

        def fail_post_resize_snapshot(read_number):
            if read_number == 5:
                raise RuntimeError("injected capacity snapshot failure")

        runtime.capacity_hook = fail_post_resize_snapshot

        with self.assertRaises(AdmissionError) as caught:
            await scheduler._rebalance_caches(None, runtime.limit, Memory(0, 0))

        self.assertEqual(caught.exception.code, "capacity_unavailable")
        self.assertEqual(runtime.resize_calls, [("deepseek", 130), ("deepseek", 10)])
        self.assertEqual(runtime.cache_capacities["deepseek"], 10)
        self.assertEqual(scheduler.cache_allocations["deepseek"], 10)
        self.assertIn("deepseek", scheduler.cache_unready)

    async def test_cancelled_growth_stays_unready_until_budget_reconciliation(self):
        class RuntimeWithPausedGrowth(FakeAdaptiveRuntime):
            def __init__(self, resources, *, capacity):
                super().__init__(resources, capacity=capacity)
                self.growth_started = asyncio.Event()
                self.growth_release = asyncio.Event()

            async def resize_expert_cache(self, name, capacity_bytes, *, operation_id=None):
                if capacity_bytes > self.cache_capacities[name]:
                    self.growth_started.set()
                    await self.growth_release.wait()
                return await super().resize_expert_cache(
                    name, capacity_bytes, operation_id=operation_id
                )

        resources = {"deepseek": adaptive_resource()}
        runtime = RuntimeWithPausedGrowth(resources, capacity=150)
        scheduler = Scheduler(
            resources,
            runtime,
            Path(self.directory.name) / "cancelled-growth-budget-leases",
            self.policy,
        )
        runtime.scheduler = scheduler
        runtime.loaded = {"deepseek": "ready"}
        scheduler.states = dict(runtime.loaded)
        scheduler.active = {"deepseek": 1}

        task = asyncio.create_task(
            scheduler._rebalance_caches(None, runtime.limit, Memory(0, 0))
        )
        await runtime.growth_started.wait()
        task.cancel()
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        runtime.growth_release.set()

        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertEqual(runtime.cache_capacities["deepseek"], 130)
        self.assertEqual(scheduler.cache_allocations["deepseek"], 130)
        self.assertIn("deepseek", scheduler.cache_unready)

    async def test_new_pending_model_reservation_is_kept_during_cache_growth(self):
        class RuntimeWithNewPendingModel(FakeAdaptiveRuntime):
            def __init__(self, resources, *, capacity):
                super().__init__(resources, capacity=capacity)
                self.scheduler = None
                self.pending_added = False

            async def resize_expert_cache(self, name, capacity_bytes, *, operation_id=None):
                growing = capacity_bytes > self.cache_capacities[name]
                info = await super().resize_expert_cache(
                    name, capacity_bytes, operation_id=operation_id
                )
                if growing and not self.pending_added:
                    self.pending_added = True
                    self.scheduler.pending_requests["qwen"] = 1
                return info

        resources = {name: adaptive_resource() for name in ("deepseek", "qwen")}
        runtime = RuntimeWithNewPendingModel(resources, capacity=150)
        scheduler = Scheduler(
            resources, runtime, Path(self.directory.name) / "new-pending-budget-leases", self.policy
        )
        runtime.scheduler = scheduler
        runtime.loaded = {"deepseek": "ready"}
        scheduler.states = dict(runtime.loaded)
        scheduler.active = {"deepseek": 1}

        fits = await scheduler._rebalance_caches(None, runtime.limit, Memory(0, 0))

        self.assertTrue(fits)
        self.assertEqual(
            runtime.resize_calls,
            [("deepseek", 130), ("deepseek", 10), ("deepseek", 100)],
        )
        self.assertEqual(runtime.cache_capacities["deepseek"], 100)
        self.assertTrue(scheduler.committed_reservation().fits(runtime.limit))

    async def test_idle_wait_does_not_reconcile_cache_state_outside_transition(self):
        resources = {"deepseek": adaptive_resource()}
        runtime = FakeAdaptiveRuntime(resources)
        scheduler = Scheduler(
            resources, runtime, Path(self.directory.name) / "idle-wait-transition-leases", self.policy
        )
        runtime.scheduler = scheduler
        runtime.loaded = {"deepseek": "ready"}
        scheduler.states = dict(runtime.loaded)
        runtime.cache_capacities["deepseek"] = 80
        scheduler.cache_allocations["deepseek"] = 10

        await scheduler.transition.acquire()
        try:
            await scheduler._wait_for_idle_worker("deepseek")
        finally:
            scheduler.transition.release()

        self.assertEqual(scheduler.cache_allocations["deepseek"], 10)

    async def test_fixed_model_load_is_stopped_if_actual_use_exceeds_budget(self):
        resources = {"dense": resource(75)}
        runtime = FakeRuntime(resources, capacity=80)
        scheduler = Scheduler(
            resources, runtime, Path(self.directory.name) / "post-load-budget-leases", self.policy
        )
        original_start = runtime.start

        async def start_with_unreserved_use(name):
            await original_start(name)
            runtime.external = Memory(10, 10)

        runtime.start = start_with_unreserved_use

        with self.assertRaises(AdmissionError) as caught:
            await scheduler.admit("dense")

        self.assertEqual(caught.exception.code, "capacity_unavailable")
        self.assertEqual(runtime.stops, ["dense"])
        self.assertNotIn("dense", runtime.loaded)

    async def test_cache_rounding_caps_unrepresentable_capacity_for_current_demand(self):
        resources = {name: adaptive_resource() for name in ("deepseek", "qwen")}
        runtime = FakeAdaptiveRuntime(resources, capacity=150)
        scheduler = Scheduler(
            resources, runtime, Path(self.directory.name) / "adaptive-rounding-leases", self.policy
        )
        runtime.scheduler = scheduler
        runtime.loaded = {"deepseek": "ready"}
        scheduler.states = dict(runtime.loaded)
        scheduler.active = {"deepseek": 1}
        runtime.cache_capacities["deepseek"] = 130
        runtime.physical_cache_capacities["deepseek"] = 130
        scheduler.cache_allocations["deepseek"] = 130
        runtime.cache_rounding_quantum = 7

        self.assertTrue(await scheduler._rebalance_caches("qwen", runtime.limit, Memory(0, 0)))

        self.assertEqual(runtime.cache_capacities["deepseek"], 49)
        self.assertEqual(scheduler.cache_fallback_caps["deepseek"], 49)
        self.assertEqual(runtime.resize_calls, [("deepseek", 55)])

    async def test_cache_transfers_shrink_before_growing_another_model(self):
        resources = {name: adaptive_resource() for name in ("a", "m", "z")}
        runtime = FakeAdaptiveRuntime(resources, capacity=180)
        scheduler = Scheduler(
            resources, runtime, Path(self.directory.name) / "transfer-leases", self.policy
        )
        runtime.scheduler = scheduler
        runtime.loaded = {"a": "ready", "z": "ready"}
        scheduler.states = dict(runtime.loaded)
        scheduler.active = {"a": 1, "z": 1}
        runtime.cache_capacities.update(a=10, z=130)
        runtime.physical_cache_capacities.update(a=10, z=130)
        scheduler.cache_allocations.update(a=10, z=130)

        self.assertTrue(await scheduler._rebalance_caches("m", runtime.limit, Memory(0, 0)))
        self.assertEqual(runtime.resize_calls, [("z", 40), ("a", 40)])
        self.assertEqual(runtime.capacity()[1], Memory(120, 120))

    async def test_cache_growth_overshoot_falls_back_to_a_fitting_capacity(self):
        resources = {name: adaptive_resource() for name in ("a", "m", "z")}
        runtime = FakeAdaptiveRuntime(resources, capacity=180)
        scheduler = Scheduler(
            resources, runtime, Path(self.directory.name) / "overshoot-cache-leases", self.policy
        )
        runtime.scheduler = scheduler
        runtime.loaded = {"a": "ready", "z": "ready"}
        scheduler.states = dict(runtime.loaded)
        scheduler.active = {"a": 1, "z": 1}
        runtime.cache_capacities.update(a=10, z=130)
        runtime.physical_cache_capacities.update(a=10, z=130)
        scheduler.cache_allocations.update(a=10, z=130)
        runtime.cache_resize_overshoots[("a", 40)] = 50
        runtime.cache_resize_overshoots[("a", 38)] = 50

        fits = await scheduler._rebalance_caches("m", runtime.limit, Memory(0, 0))

        self.assertTrue(fits)
        self.assertLessEqual(runtime.cache_capacities["a"], 40)
        self.assertEqual(
            scheduler.cache_fallback_caps["a"], runtime.cache_capacities["a"]
        )
        self.assertTrue(runtime.capacity()[1].fits(runtime.limit))

    async def test_rebalance_finishes_all_shrinks_when_first_release_is_not_enough(self):
        resources = {name: adaptive_resource() for name in ("a", "m", "z")}
        runtime = FakeAdaptiveRuntime(resources, capacity=180)
        scheduler = Scheduler(
            resources, runtime, Path(self.directory.name) / "multi-shrink-leases", self.policy
        )
        runtime.scheduler = scheduler
        runtime.loaded = {"a": "ready", "z": "ready"}
        scheduler.states = dict(runtime.loaded)
        scheduler.active = {"a": 1, "z": 1}
        runtime.cache_capacities.update(a=130, z=130)
        runtime.physical_cache_capacities.update(a=130, z=130)
        scheduler.cache_allocations.update(a=130, z=130)

        fits = await scheduler._rebalance_caches("m", runtime.limit, Memory(0, 0))

        self.assertTrue(fits)
        self.assertEqual(runtime.cache_capacities["a"], 40)
        self.assertEqual(runtime.cache_capacities["z"], 40)
        self.assertEqual(runtime.resize_calls, [("a", 40), ("z", 40)])
        self.assertTrue(runtime.capacity()[1].fits(runtime.limit))

    async def test_unreturned_cache_backing_is_not_spent_on_another_worker(self):
        resources = {name: adaptive_resource() for name in ("a", "m", "z")}
        runtime = FakeAdaptiveRuntime(resources, capacity=180)
        scheduler = Scheduler(
            resources, runtime, Path(self.directory.name) / "sticky-cache-leases", self.policy
        )
        runtime.scheduler = scheduler
        runtime.loaded = {"a": "ready", "z": "ready"}
        scheduler.states = dict(runtime.loaded)
        scheduler.active = {"a": 1, "z": 1}
        runtime.cache_capacities.update(a=10, z=130)
        runtime.physical_cache_capacities.update(a=10, z=130)
        scheduler.cache_allocations.update(a=10, z=130)
        runtime.sticky_cache_backing = True

        fits = await scheduler._rebalance_caches("m", runtime.limit, Memory(0, 0))

        self.assertFalse(fits)
        self.assertEqual(runtime.cache_capacities["z"], 10)
        self.assertEqual(runtime.cache_capacities["m"], 10)
        self.assertEqual(runtime.physical_cache_capacities["z"], 130)
        self.assertEqual(runtime.resize_calls, [("z", 40), ("z", 10)])

    async def test_lru_stops_after_reclaiming_enough_memory(self):
        self.resources["large"] = resource(50)
        self.runtime.loaded = {"a": "ready", "b": "ready"}
        self.scheduler.last_used = {"a": 10, "b": 20}
        await self.scheduler.admit("large")
        self.assertEqual(self.runtime.stops, ["a"])
        self.assertEqual(self.runtime.loaded, {"b": "ready", "large": "ready"})

    async def test_host_ram_can_reject_a_request_that_fits_in_gpu_memory(self):
        self.runtime.limit = Memory(100, 70)
        with self.assertRaises(AdmissionError) as caught:
            await self.scheduler.admit("large")
        self.assertEqual(caught.exception.code, "model_exceeds_budget")
        self.assertEqual(self.runtime.starts, [])

    async def test_adapter_only_worker_keeps_its_full_transient_peak_reserved(self):
        service = {
            "kind": "service",
            "gpu_bytes": 50,
            "host_bytes": 50,
            "memory": {
                "mandatory": {"gpu_bytes": 10, "host_bytes": 10},
                "reclaimable": {"gpu_bytes": 0, "host_bytes": 0},
                "transient_peak": {"gpu_bytes": 40, "host_bytes": 40},
            },
        }
        resources = {"comfyui": service}
        runtime = FakeRuntime(resources)
        scheduler = Scheduler(
            resources, runtime, Path(self.directory.name) / "adapter-leases", self.policy
        )
        scheduler.states["comfyui"] = "ready"
        self.assertEqual(scheduler.reserved(), Memory(50, 50))

    async def test_active_model_cannot_be_evicted_and_idle_models_are_not_pointlessly_stopped(self):
        self.runtime.loaded = {"a": "ready", "b": "ready"}
        self.protect("b")
        with self.assertRaises(AdmissionError) as caught:
            await asyncio.wait_for(self.scheduler.admit("large"), 0.05)
        self.assertEqual(caught.exception.status, 503)
        self.assertEqual(caught.exception.code, "capacity_unavailable")
        self.assertEqual(self.runtime.stops, [])
        self.assertEqual(self.runtime.starts, [])

    async def test_backend_work_outliving_client_is_still_protected(self):
        self.runtime.loaded = {"a": "ready"}
        self.runtime.occupied.add("a")
        with self.assertRaises(AdmissionError):
            await self.scheduler.admit("large")
        self.assertEqual(self.runtime.stops, [])

    async def test_unhealthy_resident_is_not_restarted_over_an_existing_request(self):
        self.runtime.loaded = {"a": "ready"}
        self.runtime.load_gate = asyncio.Event()
        self.protect("a")
        with patch.object(self.runtime, "ready", new=AsyncMock(return_value=False)):
            with self.assertRaises(AdmissionError):
                await self.scheduler.admit("a")
        self.assertEqual(self.runtime.starts, [])
        self.assertEqual(self.runtime.stops, [])

    async def test_file_lease_survives_scheduler_recreation(self):
        self.runtime.loaded = {"a": "ready"}
        self.protect("a")
        replacement = Scheduler(self.resources, self.runtime, self.scheduler.lock_dir, self.policy)
        with self.assertRaises(AdmissionError):
            await replacement.admit("large")
        self.assertEqual(self.runtime.stops, [])

    async def test_cold_requests_fail_fast_during_another_load_but_warm_work_proceeds(self):
        self.runtime.loaded = {"a": "ready"}
        self.runtime.load_gate = asyncio.Event()
        task = asyncio.create_task(self.scheduler.admit("b"))
        await self.runtime.starting.wait()
        with self.assertRaises(AdmissionError) as caught:
            await asyncio.wait_for(self.scheduler.admit("large"), 0.05)
        self.assertEqual(caught.exception.code, "model_transition")
        await self.scheduler.admit("a")
        self.runtime.load_gate.set()
        await task
        self.assertEqual(self.runtime.stops, [])

    async def test_cancelled_load_keeps_reservation_until_cleanup_completes(self):
        self.runtime.load_gate = asyncio.Event()
        cleanup_started, cleanup_release = asyncio.Event(), asyncio.Event()
        original = self.runtime.stop
        async def stop(name):
            cleanup_started.set()
            await cleanup_release.wait()
            await original(name)
        self.runtime.stop = stop
        async def request():
            async with self.scheduler.request("large"):
                self.fail("cancelled load was admitted")
        task = asyncio.create_task(request())
        await self.runtime.starting.wait()
        task.cancel()
        await cleanup_started.wait()
        task.cancel()
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        self.assertEqual(self.scheduler.states["large"], "starting")
        cleanup_release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.scheduler.active["large"], 0)
        self.assertEqual(self.runtime.loaded, {})

    async def test_cancelled_request_protects_its_worker_while_other_workers_run(self):
        started = asyncio.Event()

        async def first():
            async with self.scheduler.request("a"):
                started.set()
                await asyncio.Event().wait()

        async def second(name):
            async with self.scheduler.request(name):
                return "started"

        request = asyncio.create_task(first())
        await started.wait()
        self.runtime.occupied.add("a")
        request.cancel()
        await asyncio.sleep(0.05)

        self.assertEqual(await asyncio.wait_for(second("b"), 1), "started")
        queued = asyncio.create_task(second("a"))
        await asyncio.sleep(0.05)
        self.assertFalse(queued.done())
        self.runtime.occupied.clear()
        with self.assertRaises(asyncio.CancelledError):
            await request
        self.assertEqual(await asyncio.wait_for(queued, 1), "started")

    async def test_failed_load_cleans_up_and_failed_cleanup_does_not_release_memory(self):
        self.runtime.load_failure = self.runtime.stop_failure = True
        with self.assertRaises(AdmissionError):
            await self.scheduler.admit("large")
        self.assertEqual(self.scheduler.states["large"], "quarantined")
        with self.assertRaises(AdmissionError):
            await self.scheduler.admit("a")
        self.assertNotIn("a", self.runtime.starts)

    async def test_invalid_control_response_during_load_is_cleaned_up(self):
        async def start(name):
            self.runtime.starts.append(name)
            self.runtime.loaded[name] = "starting"
            raise UpstreamError(502, "invalid control response", "upstream_invalid_response")

        self.runtime.start = start
        with self.assertRaises(AdmissionError) as caught:
            await self.scheduler.admit("large")

        self.assertEqual((caught.exception.status, caught.exception.code), (503, "model_load_failed"))
        self.assertEqual(self.runtime.stops, ["large"])
        self.assertNotIn("large", self.runtime.loaded)
        self.assertNotIn("large", self.scheduler.states)

    async def test_rejected_admission_does_not_leave_an_abandoned_owner_record(self):
        self.resources["large"] = resource(101)
        self.runtime.loaded = {"a": "ready"}
        self.protect("a")
        with self.assertRaises(AdmissionError):
            async with self.scheduler.request("large"):
                self.fail("capacity rejection reached application work")
        self.assertEqual(records(Path(self.directory.name) / "requests", "large"), [])

    async def test_idle_comfy_backend_resets_only_for_abandoned_ambiguous_jobs(self):
        self.resources["comfyui"] = resource(40, "service")
        self.runtime.loaded = {"comfyui": "ready"}
        self.scheduler.last_used["comfyui"] = self.scheduler.clock()
        requests = Path(self.directory.name) / "requests"
        upload = OwnedRequest(requests, "comfyui", job={"kind": "comfyui_job", "job_id": "upload", "upload_pending": True}).begin()
        upload.detach()
        lost_ack = OwnedRequest(requests, "comfyui", job={"kind": "comfyui_job", "job_id": "ack", "submitted": True, "acknowledged": False}).begin()
        lost_ack.detach()
        await self.scheduler.reap()
        self.assertEqual(self.runtime.stops, ["comfyui"])
        self.assertEqual(self.runtime.loaded, {})
        self.assertEqual({item["job_id"] for item in records(requests, "comfyui")}, {"upload", "ack"})

    async def test_comfy_reaper_preserves_foreign_or_active_work(self):
        self.resources["comfyui"] = resource(40, "service")
        self.runtime.loaded = {"comfyui": "ready"}
        self.scheduler.last_used["comfyui"] = self.scheduler.clock() - 10000
        active = OwnedRequest(Path(self.directory.name) / "requests", "comfyui", job={"kind": "comfyui_job", "job_id": "active", "upload_pending": True}).begin()
        lease = self.protect("comfyui")
        await self.scheduler.reap()
        self.assertEqual(self.runtime.stops, [])
        self.assertTrue(active.path.exists())

    async def test_healthy_load_survives_former_deadline(self):
        self.policy["load_timeout_s"] = 0.01
        self.runtime.load_gate = asyncio.Event()
        task = asyncio.create_task(self.scheduler.admit("large"))
        await self.runtime.starting.wait()
        await asyncio.sleep(0.03)
        self.assertFalse(task.done())
        self.assertEqual(self.runtime.stops, [])
        self.runtime.load_gate.set()
        await asyncio.wait_for(task, 1)
        self.assertEqual(self.runtime.loaded, {"large": "ready"})

    async def test_external_gpu_memory_is_reserved(self):
        self.runtime.external = Memory(30, 0)
        with self.assertRaises(AdmissionError):
            await self.scheduler.admit("large")
        self.assertEqual(self.runtime.starts, [])

    async def test_partial_failed_backend_recovers_without_waiting_for_idle_age(self):
        self.runtime.loaded = {"a": "starting"}
        self.runtime.ready = AsyncMock(return_value=False)
        self.runtime.busy = AsyncMock(return_value=None)
        self.runtime.failed = AsyncMock(return_value=True)
        for _ in range(3):
            await self.scheduler.reap()
        self.assertEqual(self.runtime.stops, ["a"])
        self.assertEqual(self.runtime.loaded, {})
        self.assertEqual(self.scheduler.recovery["a"]["state"], "cooling-down")

    async def test_uncertain_live_backend_is_not_killed_based_on_age(self):
        self.runtime.loaded = {"a": "starting"}
        self.runtime.ready = AsyncMock(return_value=False)
        self.runtime.busy = AsyncMock(return_value=None)
        self.runtime.failed = AsyncMock(return_value=False)
        self.scheduler.last_used["a"] = -1000000000
        for _ in range(5):
            await self.scheduler.reap()
        self.assertEqual(self.runtime.stops, [])
        self.assertEqual(self.scheduler.recovery["a"]["state"], "quarantined")

    async def test_deployment_lock_rejects_admission_without_stopping_work(self):
        lock = ResourceLock(self.scheduler.lock_dir.parent, "operations.lock", exclusive=True)
        self.assertTrue(lock.acquire())
        try:
            with self.assertRaises(AdmissionError) as error:
                await self.scheduler.admit("a")
            self.assertEqual(error.exception.code, "maintenance")
            await self.scheduler.reap()
            self.assertEqual(self.runtime.stops, [])
        finally:
            lock.close()

    async def test_pending_systemd_work_blocks_admission_and_eviction(self):
        directory = Path(self.directory.name)
        self.scheduler = Scheduler(self.resources, self.runtime, directory / "leases", self.policy, clock=lambda: 1000)
        (directory / "maintenance.json").write_text('{"reason":"pending-service-job"}')
        self.runtime.loaded = {"a": "ready"}
        self.scheduler.last_used["a"] = 1
        with self.assertRaises(AdmissionError) as error:
            await self.scheduler.admit("b")
        self.assertEqual(error.exception.code, "maintenance")
        await self.scheduler.reap()
        self.assertEqual(self.runtime.stops, [])

    async def test_pending_comfyui_restore_blocks_model_load_after_operation_lock_release(self):
        directory = Path(self.directory.name)
        self.scheduler = Scheduler(self.resources, self.runtime, directory / "leases", self.policy, clock=lambda: 1000)
        (directory / "comfyui-restore.json").write_text(
            '{"version":1,"unit":"comfyui.service","was_active":true,"pid":123}'
        )

        with self.assertRaises(AdmissionError) as error:
            await self.scheduler.admit("a")

        self.assertEqual(error.exception.code, "maintenance")
        self.assertEqual(self.runtime.starts, [])
        self.assertEqual(self.runtime.stops, [])

    async def test_warm_admission_preserves_busy_and_unknown_activity_during_other_load(self):
        self.runtime.loaded = {"a": "ready"}
        await self.scheduler.refresh()
        async with self.scheduler.transition:
            for busy, status in ((True, 429), (None, 503)):
                self.runtime.busy = AsyncMock(return_value=busy)
                with self.assertRaises(AdmissionError) as error:
                    await self.scheduler.admit("a")
                self.assertEqual(error.exception.status, status)
        self.assertEqual(self.runtime.starts, [])

    async def test_reaper_preserves_active_work_and_stops_idle_models(self):
        self.runtime.loaded = {"a": "ready", "b": "ready"}
        self.scheduler.last_used = {"a": 10, "b": 20}
        self.protect("a")
        await self.scheduler.reap()
        self.assertEqual(self.runtime.stops, ["b"])


class ExecutionGateTest(unittest.IsolatedAsyncioTestCase):
    async def test_worker_slot_excludes_another_process(self):
        with tempfile.TemporaryDirectory() as directory:
            gate = SerializedExecution(Path(directory))
            async with gate.hold_worker("a") as admission:
                admission.ready()
                script = """
import fcntl, os, sys
fd = os.open(sys.argv[1], os.O_RDWR)
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    sys.exit(0)
sys.exit(1)
"""
                result = await asyncio.to_thread(
                    subprocess.run,
                    [sys.executable, "-c", script, str(Path(directory) / "worker-slots/a")],
                    capture_output=True,
                    timeout=5,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    async def test_shared_workers_drain_before_queued_maintenance_and_later_work(self):
        with tempfile.TemporaryDirectory() as directory:
            gate = SerializedExecution(Path(directory) / "leases")
            entered = {name: asyncio.Event() for name in ("a", "b", "maintenance", "later")}
            releases = {name: asyncio.Event() for name in entered}

            async def worker(name):
                async with gate.hold_worker(name) as admission:
                    admission.ready()
                    entered[name].set()
                    await releases[name].wait()

            async def maintenance():
                async with gate.hold_maintenance():
                    entered["maintenance"].set()
                    await releases["maintenance"].wait()

            async def queued(count):
                async with asyncio.timeout(2):
                    while len(list(gate.queue.glob("*.ticket"))) < count:
                        await asyncio.sleep(0)

            tasks = [asyncio.create_task(worker("a")), asyncio.create_task(worker("b"))]
            try:
                await asyncio.wait_for(entered["a"].wait(), 1)
                await asyncio.wait_for(entered["b"].wait(), 1)
                tasks.append(asyncio.create_task(maintenance()))
                await queued(1)
                tasks.append(asyncio.create_task(worker("later")))
                await queued(2)
                releases["a"].set()
                await tasks[0]
                self.assertFalse(entered["maintenance"].is_set())
                releases["b"].set()
                await asyncio.wait_for(entered["maintenance"].wait(), 1)
                self.assertFalse(entered["later"].is_set())
                releases["maintenance"].set()
                await asyncio.wait_for(entered["later"].wait(), 1)
            finally:
                for release in releases.values():
                    release.set()
                await asyncio.gather(*tasks)

    async def test_gate_serializes_distinct_workers_and_releases_after_cancellation(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = SerializedExecution(Path(directory)), SerializedExecution(Path(directory))
            started, release, second_started = asyncio.Event(), asyncio.Event(), asyncio.Event()

            async def operation():
                started.set()
                await release.wait()

            async def run_first():
                async with first.hold():
                    await operation()

            async def run_second():
                async with second.hold():
                    second_started.set()

            current = asyncio.create_task(run_first())
            await started.wait()
            queued = asyncio.create_task(run_second())
            await asyncio.sleep(0.05)
            self.assertFalse(second_started.is_set())
            queued.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await queued
            release.set()
            await current

            async with second.hold():
                self.assertFalse(second_started.is_set())

    async def test_gate_admits_waiters_in_ticket_order(self):
        with tempfile.TemporaryDirectory() as directory:
            gates = [SerializedExecution(Path(directory)) for _ in range(3)]
            queue = Path(directory) / "execution-queue"
            first_started, release_first = asyncio.Event(), asyncio.Event()
            second_started, release_second = asyncio.Event(), asyncio.Event()
            third_started = asyncio.Event()

            async def hold_first():
                async with gates[0].hold():
                    first_started.set()
                    await release_first.wait()

            async def hold_second():
                async with gates[1].hold():
                    second_started.set()
                    await release_second.wait()

            async def hold_third():
                async with gates[2].hold():
                    third_started.set()

            async def wait_for_tickets(count):
                for _ in range(100):
                    if len(list(queue.glob("*.ticket"))) >= count:
                        return
                    await asyncio.sleep(0.01)
                self.fail("execution ticket was not queued")

            first = asyncio.create_task(hold_first())
            await first_started.wait()
            second = asyncio.create_task(hold_second())
            await wait_for_tickets(1)
            third = asyncio.create_task(hold_third())
            await wait_for_tickets(2)

            release_first.set()
            await second_started.wait()
            self.assertFalse(third_started.is_set())
            release_second.set()
            await asyncio.gather(first, second, third)
            self.assertTrue(third_started.is_set())

    async def test_maintenance_window_keeps_fifo_and_host_operation_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lease_directory = root / "leases"
            normal = SerializedExecution(lease_directory)
            maintenance = SerializedExecution(lease_directory)
            later = SerializedExecution(lease_directory)
            queue = lease_directory / "execution-queue"
            first_started, release_first = asyncio.Event(), asyncio.Event()
            maintenance_started, release_maintenance = asyncio.Event(), asyncio.Event()
            later_started = asyncio.Event()

            async def hold_first():
                async with normal.hold():
                    first_started.set()
                    await release_first.wait()

            async def hold_maintenance():
                async with maintenance.hold_maintenance():
                    execution_probe = ResourceLock(
                        lease_directory, "global-execution", exclusive=True
                    )
                    self.assertFalse(execution_probe.acquire())
                    execution_probe.close()
                    operations_probe = ResourceLock(root, "operations.lock", exclusive=True)
                    self.assertFalse(operations_probe.acquire())
                    operations_probe.close()
                    maintenance_started.set()
                    await release_maintenance.wait()

            async def hold_later():
                async with later.hold():
                    later_started.set()

            async def wait_for_tickets(count):
                for _ in range(100):
                    if len(list(queue.glob("*.ticket"))) >= count:
                        return
                    await asyncio.sleep(0.01)
                self.fail("execution ticket was not queued")

            first = asyncio.create_task(hold_first())
            await first_started.wait()
            middle = asyncio.create_task(hold_maintenance())
            await wait_for_tickets(1)
            queued = asyncio.create_task(hold_later())
            await wait_for_tickets(2)

            release_first.set()
            await maintenance_started.wait()
            self.assertFalse(later_started.is_set())
            release_maintenance.set()
            await asyncio.gather(first, middle, queued)
            self.assertTrue(later_started.is_set())

    async def test_maintenance_window_fails_closed_when_host_operations_are_busy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lease_directory = root / "leases"
            operations = ResourceLock(root, "operations.lock", exclusive=True)
            self.assertTrue(operations.acquire())
            self.addCleanup(operations.close)

            with self.assertRaisesRegex(RuntimeError, "host operations are busy"):
                async with SerializedExecution(lease_directory).hold_maintenance():
                    self.fail("maintenance entered despite an occupied host operations lock")

            execution = ResourceLock(lease_directory, "global-execution", exclusive=True)
            self.assertTrue(execution.acquire())
            execution.close()

    async def test_maintenance_window_stays_held_until_cancelled_thread_finishes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lease_directory = root / "leases"
            worker_started, release_worker = threading.Event(), threading.Event()

            def worker():
                worker_started.set()
                if not release_worker.wait(2):
                    raise TimeoutError("test worker was not released")

            async def run_worker():
                async with SerializedExecution(lease_directory).hold_maintenance():
                    await run_thread_shielded(worker)

            task = asyncio.create_task(run_worker())
            self.assertTrue(await asyncio.to_thread(worker_started.wait, 2))
            task.cancel()
            await asyncio.sleep(0.05)
            self.assertFalse(task.done())

            execution_probe = ResourceLock(
                lease_directory, "global-execution", exclusive=True
            )
            self.assertFalse(execution_probe.acquire())
            execution_probe.close()
            operations_probe = ResourceLock(root, "operations.lock", exclusive=True)
            self.assertFalse(operations_probe.acquire())
            operations_probe.close()

            release_worker.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            execution = ResourceLock(lease_directory, "global-execution", exclusive=True)
            self.assertTrue(execution.acquire())
            execution.close()
            operations = ResourceLock(root, "operations.lock", exclusive=True)
            self.assertTrue(operations.acquire())
            operations.close()


class ProxyTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.release = asyncio.Event()
        self.started = asyncio.Event()
        self.api_ready = False
        self.expert_cache_state = {}
        self.expert_cache_post_capacity = None
        self.expert_cache_resize_failure = None
        self.progress_slots = []
        self.progress_metrics = ""
        self.custom_nodes = {}
        self.full_catalog_requests = 0
        self.resources = {"a": resource(60), "b": resource(60), "comfyui": resource(60, "service")}
        self.runtime = FakeRuntime(self.resources)
        self.forwarded = []

        async def handle(request):
            payload = await request.json()
            self.forwarded.append((await request.text(), request.headers.get("Content-Type")))
            if payload.get("hang"):
                await self.release.wait()
            if payload.get("stream"):
                response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
                await response.prepare(request)
                if payload.get("truncated_stream"):
                    await response.write(
                        b'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}\n\n'
                    )
                    return response
                if payload.get("fragmented"):
                    await response.write(b'data: {"content":"hel')
                    self.started.set()
                    await self.release.wait()
                    await response.write(b'lo"}\r\n\r')
                    await asyncio.sleep(0)
                    await response.write(b'\n')
                    await response.write(b'data: [DONE]\n\n')
                    return response
                await response.write(b'data: {"content":"hello"}\n\n')
                self.started.set()
                await self.release.wait()
                await response.write(b'data: [DONE]\n\n')
                return response
            return web.json_response({"model": payload["model"], "ok": True})
        upstream = web.Application()
        upstream.router.add_post("/v1/chat/completions", handle)
        upstream.router.add_post("/v1/responses", handle)
        upstream.router.add_post("/upstream/{model}/v1/chat/completions", handle)
        async def health(_):
            return web.json_response({"ok": True})
        async def ready(_):
            return web.json_response({"ok": self.api_ready}, status=200 if self.api_ready else 503)
        async def nodes(_):
            self.full_catalog_requests += 1
            return web.json_response(self.custom_nodes)
        async def node(request):
            name = request.match_info["node_class"]
            return web.json_response({name: self.custom_nodes[name]} if name in self.custom_nodes else {})
        async def expert_cache(request):
            operation_id = request.query.get("operation_id")
            if operation_id:
                return web.json_response({
                    "operation_id": operation_id,
                    "status": "completed",
                    "status_code": 200,
                    "result": {
                        "reason": "resize_done",
                        "phase": "restoring_residents",
                        "current_slots": 4,
                        "target_slots": 8,
                        "queued_loads": 2,
                        "loading_slots": 1,
                        "claimed_slots": 1,
                        "resident_slots": 6,
                    },
                })
            if request.method == "POST":
                payload = await request.json()
                if self.expert_cache_resize_failure is not None:
                    event_payload = {
                        "operation_id": payload.get("operation_id"),
                        "reason": "runtime_resize_failed",
                        "detail": self.expert_cache_resize_failure,
                        "status_code": 503,
                    }
                    response = web.StreamResponse(
                        headers={"Content-Type": "text/event-stream"}
                    )
                    await response.prepare(request)
                    await response.write(
                        f"event: resize.failed\ndata: {json.dumps(event_payload)}\n\n".encode()
                    )
                    return response
                self.expert_cache_state["capacity_bytes"] = (
                    self.expert_cache_post_capacity
                    if self.expert_cache_post_capacity is not None
                    else payload["capacity_bytes"]
                )
            return web.json_response(self.expert_cache_state)
        async def slots(_):
            return web.json_response(self.progress_slots)
        async def metrics(_):
            return web.Response(text=self.progress_metrics, content_type="text/plain")
        upstream.router.add_get("/health", health)
        upstream.router.add_get("/ready", ready)
        upstream.router.add_get("/slots", slots)
        upstream.router.add_get("/metrics", metrics)
        upstream.router.add_get("/object_info", nodes)
        upstream.router.add_get("/object_info/{node_class}", node)
        upstream.router.add_route("*", "/expert-cache", expert_cache)
        self.upstream = TestServer(upstream)
        await self.upstream.start_server()
        self.addAsyncCleanup(self.upstream.close)
        self.policy = {"socket_path": str(Path(self.directory.name) / "scheduler.sock"), "upstream_url": str(self.upstream.make_url("")).rstrip("/"), "load_timeout_s": 1, "stop_timeout_s": 1, "response_timeout_s": 1, "request_timeout_s": 5, "idle_timeout_s": 300}
        self.scheduler = Scheduler(
            self.resources,
            self.runtime,
            Path(self.directory.name) / "leases",
            self.policy,
        )
        public, local = create_apps(self.policy, self.resources, "secret", runtime_factory=lambda *args: self.runtime)
        self.public = public
        self.client = TestClient(TestServer(public, handler_cancellation=True))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)
        self.local_runner = web.AppRunner(local, handler_cancellation=True)
        await self.local_runner.setup()
        await web.UnixSite(self.local_runner, self.policy["socket_path"]).start()
        self.addAsyncCleanup(self.local_runner.cleanup)

    async def post(self, model, **parameters):
        return await self.client.post("/v1/chat/completions", headers={"Authorization": "Bearer secret"}, json={"model": model, **parameters})

    async def test_worker_progress_probe_returns_only_slot_counters_and_known_metrics(self):
        self.progress_slots = [{
            "id": 0,
            "id_task": 42,
            "is_processing": True,
            "n_prompt_tokens": 64,
            "n_prompt_tokens_processed": 32,
            "n_tokens": 0,
            "next_token": [{"n_decoded": 0, "has_next_token": True}],
            "n_ctx": 220160,
            "prompt": "private request text",
            "generated": "private generated text",
            "oversized": 2**100,
        }]
        self.progress_metrics = "\n".join((
            "# HELP llamacpp:tokens_predicted_total test",
            "llamacpp:tokens_predicted_total 0",
            "llamacpp:requests_processing 1",
            "llamacpp:n_decode_total 1e99",
            "llamacpp:unrelated_metric 999",
        ))
        self.resources["a"]["gateway_url"] = str(self.upstream.make_url(""))
        runtime = Runtime(self.policy, self.resources, self.client.session, "secret")

        progress = await runtime.inference_progress("a")

        self.assertEqual(progress["slots"], [{
            "id": 0,
            "id_task": 42,
            "is_processing": True,
            "n_prompt_tokens": 64,
            "n_prompt_tokens_processed": 32,
            "n_tokens": 0,
            "n_decoded": 0,
            "n_ctx": 220160,
        }])
        self.assertEqual(progress["metrics"], {
            "llamacpp:tokens_predicted_total": 0,
            "llamacpp:requests_processing": 1,
        })
        self.assertIsNone(progress["slots_error"])
        self.assertIsNone(progress["metrics_error"])
        self.assertNotIn("private request text", repr(progress))
        self.assertNotIn("private generated text", repr(progress))

    async def test_resize_progress_logs_live_memory_snapshot(self):
        self.resources["a"]["expert_cache_url"] = str(
            self.upstream.make_url("/expert-cache")
        )
        runtime = Runtime(self.policy, self.resources, self.client.session, "secret")
        with patch.object(
            runtime,
            "capacity",
            return_value=(Memory(100, 200), Memory(50, 80)),
        ), patch("redqueen_adapters.scheduler_api.asyncio.sleep", new=AsyncMock()), self.assertLogs(
            scheduler_api.LOG, level="INFO"
        ) as captured:
            await runtime._log_expert_cache_resize_progress(
                str(self.upstream.make_url("/expert-cache")),
                "a" * 32,
                "a",
                64,
            )

        self.assertTrue(any(
            "status=completed" in line
            and "phase=restoring_residents" in line
            and "current_slots=4" in line
            and "target_slots=8" in line
            and "queued_loads=2" in line
            and "loading_slots=1" in line
            and "claimed_slots=1" in line
            and "resident_slots=6" in line
            and "capacity_gpu_bytes=100" in line
            and "usage_gpu_bytes=50" in line
            and "usage_host_bytes=80" in line
            for line in captured.output
        ))

    async def test_stalled_stream_logs_correlated_worker_progress(self):
        self.policy["heartbeat_s"] = 0.01
        samples = [{
            "slots": [{
                "id": 0,
                "is_processing": True,
                "n_prompt_tokens": 64,
                "n_prompt_tokens_processed": 32,
                "n_tokens": 0,
                "n_decoded": 0,
                "n_ctx": 220160,
            }],
            "metrics": {"llamacpp:tokens_predicted_total": 0},
            "slots_error": None,
            "metrics_error": None,
        }, {
            "slots": [{
                "id": 0,
                "is_processing": True,
                "n_prompt_tokens": 64,
                "n_prompt_tokens_processed": 64,
                "n_tokens": 64,
                "n_decoded": 0,
                "n_ctx": 220160,
            }],
            "metrics": {"llamacpp:tokens_predicted_total": 0},
            "slots_error": None,
            "metrics_error": None,
        }, {
            "slots": [{
                "id": 0,
                "is_processing": True,
                "n_prompt_tokens": 64,
                "n_prompt_tokens_processed": 64,
                "n_tokens": 66,
                "n_decoded": 2,
                "n_ctx": 220160,
            }],
            "metrics": {"llamacpp:tokens_predicted_total": 2},
            "slots_error": None,
            "metrics_error": None,
        }]
        sample_calls = 0

        def sample_progress(_model):
            nonlocal sample_calls
            progress = samples[min(sample_calls, len(samples) - 1)]
            sample_calls += 1
            return progress

        probe = AsyncMock(side_effect=sample_progress)

        with patch.object(self.runtime, "inference_progress", new=probe, create=True):
            with self.assertLogs("redqueen.scheduler", level="INFO") as captured:
                response = await self.post("a", stream=True)
                await self.started.wait()
                await asyncio.sleep(0.04)
                self.release.set()
                await asyncio.wait_for(response.read(), 1)

        output = "\n".join(captured.output)
        self.assertIn("upstream_worker_progress", output)
        self.assertIn("prompt=32/64", output)
        self.assertIn("worker_phase=prefill", output)
        self.assertIn("worker_phase=decode", output)
        self.assertIn("prefill_tps=", output)
        self.assertIn("decode_tps=", output)
        self.assertIn("no_progress_s=", output)
        probe.assert_awaited()

    async def test_known_routes_return_405_with_allow(self):
        for method, path, allowed in (("GET", "/v1/chat/completions", "POST"), ("POST", "/v1/models", "GET"), ("DELETE", "/upstream/a/health", "GET")):
            response = await self.client.request(method, path, headers={"Authorization": "Bearer secret"})
            self.assertEqual(response.status, 405)
            self.assertIn(allowed, response.headers["Allow"])
            self.assertEqual((await response.json())["error"]["code"], "method_not_allowed")

    async def test_heartbeat_never_splits_fragmented_upstream_sse_event(self):
        self.policy["heartbeat_s"] = 0.01
        stream = await self.post("a", stream=True, fragmented=True)
        await self.started.wait()
        await asyncio.sleep(0.04)
        self.release.set()
        raw = await stream.read()
        self.assertIn(b": ping\n\n", raw)
        frames = [
            line[6:]
            for line in raw.splitlines()
            if line.startswith(b"data: ") and line[6:] != b"[DONE]"
        ]
        self.assertEqual([json.loads(frame) for frame in frames], [{"content": "hello"}])

    async def test_incomplete_chat_stream_is_reported_as_an_error_not_complete(self):
        with self.assertLogs("redqueen.scheduler", level="WARNING") as captured:
            stream = await self.post("a", stream=True, truncated_stream=True)
            raw = await stream.read()

        self.assertEqual(stream.status, 200)
        self.assertIn(b"upstream_invalid_response", raw)
        self.assertNotIn(b"[DONE]", raw)
        output = "\n".join(captured.output)
        self.assertIn("upstream_stream_incomplete", output)
        self.assertIn("text_chars=0", output)
        self.assertIn("reasoning_chars=8", output)
        self.assertNotIn("outcome=complete", output)

    async def test_new_model_waits_for_active_stream_then_evicts_it(self):
        stream = await self.post("a", stream=True)
        self.assertIn(b"hello", await stream.content.readany())
        queued = asyncio.create_task(self.post("b"))
        await asyncio.sleep(0.05)
        self.assertFalse(queued.done())
        self.assertEqual(self.runtime.stops, [])
        self.release.set()
        self.assertIn(b"[DONE]", await stream.read())
        admitted = await asyncio.wait_for(queued, 1)
        self.assertEqual(admitted.status, 200)
        self.assertEqual(self.runtime.stops, ["a"])

    async def test_distinct_workers_overlap_when_their_full_peaks_fit(self):
        self.runtime.limit = Memory(200, 200)
        stream = await self.post("a", stream=True)
        await self.started.wait()
        try:
            response = await asyncio.wait_for(self.post("b"), 1)
            self.assertEqual(response.status, 200)
            self.assertEqual(self.runtime.stops, [])
            self.assertEqual(self.runtime.loaded, {"a": "ready", "b": "ready"})
        finally:
            self.release.set()
            await stream.read()

    async def test_exclusive_worker_drains_other_workers_and_blocks_new_execution(self):
        self.runtime.limit = Memory(200, 200)
        self.resources["deepseek"] = resource(40, "service")
        self.resources["deepseek"].update(
            gateway_url="http://127.0.0.1:8085",
            exclusive_execution=True,
        )
        entered = {name: asyncio.Event() for name in ("a", "deepseek", "b")}
        release = {name: asyncio.Event() for name in entered}

        async def request(name):
            async with self.scheduler.request(name):
                entered[name].set()
                await release[name].wait()

        tasks = []
        try:
            tasks.append(asyncio.create_task(request("a")))
            await asyncio.wait_for(entered["a"].wait(), 1)
            tasks.append(asyncio.create_task(request("deepseek")))
            await asyncio.sleep(0.05)
            self.assertFalse(entered["deepseek"].is_set())

            release["a"].set()
            await tasks[0]
            await asyncio.wait_for(entered["deepseek"].wait(), 1)
            tasks.append(asyncio.create_task(request("b")))
            await asyncio.sleep(0.05)
            self.assertFalse(entered["b"].is_set())

            release["deepseek"].set()
            await tasks[1]
            await asyncio.wait_for(entered["b"].wait(), 1)
            release["b"].set()
            await tasks[2]
        finally:
            for event in release.values():
                event.set()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def test_adapter_and_gateway_overlap_when_their_full_peaks_fit(self):
        self.runtime.limit = Memory(200, 200)
        self.resources["jevk5"] = resource(40, "service")
        stream = await self.post("a", stream=True)
        await self.started.wait()
        adapter = SchedulerClient(self.policy["socket_path"])
        backend_started = asyncio.Event()

        async def score():
            backend_started.set()

        task = asyncio.create_task(adapter.run("jevk5", score))
        try:
            await asyncio.wait_for(backend_started.wait(), 1)
            await task
            self.assertEqual(self.runtime.stops, [])
        finally:
            self.release.set()
            await stream.read()
            await task

    async def test_gateway_alias_and_adapter_share_one_physical_worker_slot(self):
        self.resources["a"].update(
            kind="service",
            aliases=["a-alias"],
            gateway_url=str(self.upstream.make_url("")).rstrip("/"),
            supported_request_shapes=["openai.chat"],
        )
        stream = await self.post("a-alias", stream=True)
        await self.started.wait()
        backend_started = asyncio.Event()

        async def score():
            backend_started.set()

        task = asyncio.create_task(SchedulerClient(self.policy["socket_path"]).run("a", score))
        try:
            queue = Path(self.policy["socket_path"]).parent / "leases/execution-queue"
            async with asyncio.timeout(1):
                while not list(queue.glob("*.ticket")):
                    await asyncio.sleep(0)
            self.assertFalse(backend_started.is_set())
        finally:
            self.release.set()
            await stream.read()
            await asyncio.wait_for(task, 1)
        self.assertTrue(backend_started.is_set())

    async def test_adapter_admission_rejects_an_eviction_lease_without_a_worker_slot(self):
        directory = Path(self.policy["socket_path"]).parent / "leases"
        directory.mkdir(mode=0o700, exist_ok=True)
        lease = ResourceLock(directory, "comfyui")
        self.assertTrue(lease.acquire())
        try:
            async with aiohttp.ClientSession(connector=aiohttp.UnixConnector(path=self.policy["socket_path"])) as session:
                async with session.post("http://scheduler/admit", json={"resource": "comfyui"}) as response:
                    self.assertEqual(response.status, 400)
                    self.assertEqual((await response.json())["error"]["code"], "lease_required")
            self.assertEqual(self.runtime.starts, [])
        finally:
            lease.close()

    async def test_same_model_waits_for_active_stream_before_next_request(self):
        stream = await self.post("a", stream=True)
        self.assertIn(b"hello", await stream.content.readany())
        queued = asyncio.create_task(self.post("a"))
        await asyncio.sleep(0.05)
        self.assertFalse(queued.done())
        self.release.set()
        await stream.read()
        response = await asyncio.wait_for(queued, 1)
        self.assertEqual(response.status, 200)

    async def test_public_gateway_bounds_serialized_work_to_four_requests(self):
        stream = await self.post("a", stream=True)
        self.assertIn(b"hello", await stream.content.readany())
        queued = [asyncio.create_task(self.post("a")) for _ in range(3)]
        await asyncio.sleep(0.05)

        rejected = await self.post("b")
        self.assertEqual(rejected.status, 429)
        self.assertEqual((await rejected.json())["error"]["code"], "frontend_busy")
        self.release.set()
        await stream.read()
        responses = await asyncio.wait_for(asyncio.gather(*queued), 1)
        self.assertEqual([response.status for response in responses], [200, 200, 200])

    async def test_adapter_execution_waits_for_the_shared_gateway_lease(self):
        self.resources["jevk5"] = resource(60, "service")
        stream = await self.post("a", stream=True)
        self.assertIn(b"hello", await stream.content.readany())
        adapter = SchedulerClient(self.policy["socket_path"])
        backend_started = asyncio.Event()

        async def score():
            backend_started.set()

        task = asyncio.create_task(adapter.run("jevk5", score))
        await asyncio.sleep(0.05)
        self.assertFalse(backend_started.is_set())
        self.release.set()
        await stream.read()
        await asyncio.wait_for(task, 1)
        self.assertTrue(backend_started.is_set())

    async def test_service_admission_returns_its_profile_concurrency_limit(self):
        self.resources["jevk5"] = resource(60, "service")
        self.resources["jevk5"]["concurrency_limit"] = 3
        adapter = SchedulerClient(self.policy["socket_path"])
        async with adapter.lease("jevk5") as admission:
            self.assertEqual(admission["concurrency_limit"], 3)

    async def test_existing_comfy_operator_job_blocks_other_models_until_idle(self):
        self.runtime.loaded = {"comfyui": "ready"}
        self.runtime.occupied.add("comfyui")
        task = asyncio.create_task(self.post("a"))
        await asyncio.sleep(0.05)
        self.assertFalse(task.done())
        self.assertEqual(self.runtime.starts, [])
        self.runtime.occupied.clear()

        response = await asyncio.wait_for(task, 1)
        self.assertEqual(response.status, 200)
        self.assertEqual(self.runtime.stops, ["comfyui"])

    async def test_recovery_lease_can_reconcile_a_busy_comfy_queue(self):
        self.runtime.loaded = {"comfyui": "ready"}
        self.runtime.occupied.add("comfyui")
        adapter = SchedulerClient(self.policy["socket_path"])

        async with asyncio.timeout(1):
            async with adapter.lease(
                "comfyui", exclusive=True, wait_for_idle=False
            ):
                self.assertTrue(await self.runtime.busy("comfyui"))

    async def test_auth_and_management_routes_cannot_bypass_admission(self):
        response = await self.client.post("/v1/chat/completions", json={"model": "a"})
        self.assertEqual(response.status, 401)
        response = await self.client.post("/api/models/unload/a", headers={"Authorization": "Bearer secret"})
        self.assertEqual(response.status, 404)
        self.assertEqual(self.runtime.starts, [])

    async def test_readiness_fails_when_model_management_is_unavailable(self):
        self.runtime.available = False
        self.assertEqual((await self.client.get("/healthz")).status, 200)
        self.assertEqual((await self.client.get("/readyz")).status, 503)
        self.assertFalse(await SchedulerClient(self.policy["socket_path"]).ready())
        self.runtime.available = True
        self.assertEqual((await self.client.get("/readyz")).status, 200)
        self.assertTrue(await SchedulerClient(self.policy["socket_path"]).ready())

    async def test_runtime_readiness_probes_service_bus_and_memory_counters(self):
        self.resources["comfyui"]["units"] = ["comfyui.service"]
        runtime = Runtime(self.policy, self.resources, self.client.session, "secret")
        with patch.object(runtime, "capacity", return_value=(Memory(100, 100), Memory(0, 0))), patch.object(runtime, "command", new=AsyncMock(side_effect=["loaded\n\nloaded\n", "active\n"])):
            self.assertTrue(await runtime.healthy())
        with patch.object(runtime, "capacity", return_value=(Memory(100, 100), Memory(0, 0))), patch.object(runtime, "command", new=AsyncMock(return_value="not-found\nloaded\n")):
            self.assertFalse(await runtime.healthy())
        with patch.object(runtime, "capacity", return_value=(Memory(100, 100), Memory(0, 0))), patch.object(runtime, "command", new=AsyncMock(side_effect=RuntimeError("bus hidden"))):
            self.assertFalse(await runtime.healthy())
        with patch.object(runtime, "capacity", side_effect=RuntimeError("GPU counters hidden")):
            self.assertFalse(await runtime.healthy())

    async def test_duplicate_model_keys_cannot_route_a_different_unreserved_model(self):
        response = await self.client.post("/v1/chat/completions", headers={"Authorization": "Bearer secret", "Content-Type": "application/json"}, data='{"model":"b","model":"a"}')
        self.assertEqual(response.status, 200)
        pairs = json.loads(self.forwarded[-1][0], object_pairs_hook=list)
        self.assertEqual([value for key, value in pairs if key == "model"], ["a"])
        self.assertEqual(self.runtime.starts, ["a"])

    async def test_upstream_image_guard_and_backend_share_duplicate_key_interpretation(self):
        response = await self.client.post(
            "/upstream/a/v1/chat/completions",
            headers={"Authorization": "Bearer secret", "Content-Type": "application/json"},
            data=(
                '{"model":"a","messages":[{"role":"user","content":'
                '[{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]}],'
                '"messages":[{"role":"user","content":"text only"}]}'
            ),
        )

        self.assertEqual(response.status, 200)
        self.assertEqual(self.forwarded[-1][1], "application/json")
        self.assertEqual(self.forwarded[-1][0].count('"messages"'), 1)
        self.assertEqual(self.runtime.starts, ["a"])

    async def test_non_json_content_type_cannot_route_an_unreserved_query_model(self):
        response = await self.client.post("/v1/chat/completions?model=b", headers={"Authorization": "Bearer secret", "Content-Type": "text/plain"}, data='{"model":"a"}')
        self.assertEqual(response.status, 200)
        self.assertEqual(self.forwarded[-1][1], "application/json")
        self.assertEqual(self.runtime.starts, ["a"])

    async def test_direct_openai_service_is_routed_after_shared_admission(self):
        self.resources["native"] = resource(30, "service")
        direct_requests = []

        async def handle_direct(request):
            body = await request.json()
            direct_requests.append((request.path, request.headers.get("Authorization"), body))
            return web.json_response({"model": body["model"], "direct": True})

        direct_app = web.Application()
        direct_app.router.add_post("/v1/chat/completions", handle_direct)
        direct_server = TestServer(direct_app)
        await direct_server.start_server()
        self.addAsyncCleanup(direct_server.close)
        self.resources["native"].update(
            aliases=["native-alias"],
            gateway_url=str(direct_server.make_url("")).rstrip("/"),
            slots_url=str(direct_server.make_url("/slots")),
            health_url=str(direct_server.make_url("/health")),
            supported_request_shapes=["openai.chat", "openai.multimodal_chat"],
        )

        response = await self.post("native-alias", max_tokens=4)

        self.assertEqual(response.status, 200)
        self.assertEqual(await response.json(), {"model": "native", "direct": True})
        self.assertEqual(direct_requests, [("/v1/chat/completions", "Bearer secret", {"model": "native", "max_tokens": 4})])
        self.assertEqual(self.forwarded, [])
        self.assertEqual(self.runtime.starts, ["native"])

        image_response = await self.post(
            "native",
            messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]}],
        )
        self.assertEqual(image_response.status, 200)
        self.assertEqual(len(direct_requests), 2)

    async def test_direct_openai_service_rejects_unadvertised_request_shape(self):
        self.resources["native"] = resource(30, "service")
        self.resources["native"].update(
            gateway_url="http://127.0.0.1:8801",
            slots_url="http://127.0.0.1:8801/slots",
            health_url="http://127.0.0.1:8801/health",
            supported_request_shapes=["openai.chat", "openai.responses"],
        )

        response = await self.client.post(
            "/v1/embeddings",
            headers={"Authorization": "Bearer secret"},
            json={"model": "native", "input": "hello"},
        )

        self.assertEqual(response.status, 400)
        self.assertEqual((await response.json())["error"]["code"], "unsupported_request_shape")

        image_response = await self.client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer secret"},
            json={
                "model": "native",
                "input": [{"role": "user", "content": [{"type": "input_image", "image_url": "data:image/png;base64,AA=="}]}],
            },
        )
        self.assertEqual(image_response.status, 400)
        self.assertEqual((await image_response.json())["error"]["code"], "unsupported_request_shape")
        self.assertEqual(self.runtime.starts, [])

    async def test_text_worker_rejects_unaccounted_upstream_image_before_admission(self):
        self.resources["a"]["supported_request_shapes"] = [
            "openai.chat", "openai.completion", "openai.responses"
        ]

        response = await self.client.post(
            "/upstream/a/v1/chat/completions",
            headers={"Authorization": "Bearer secret"},
            json={
                "messages": [{
                    "role": "user",
                    "content": [{
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AA=="},
                    }],
                }],
            },
        )

        self.assertEqual(response.status, 400)
        self.assertEqual((await response.json())["error"]["code"], "unsupported_request_shape")
        self.assertEqual(self.runtime.starts, [])
        self.assertEqual(self.forwarded, [])

    async def test_text_worker_preserves_responses_images_when_profile_allows_them(self):
        self.resources["a"]["supported_request_shapes"] = [
            "openai.chat", "openai.completion", "openai.responses",
            "openai.multimodal_chat", "openai.multimodal_responses",
        ]

        response = await self.client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer secret"},
            json={
                "model": "a",
                "input": [{
                    "role": "user",
                    "content": [{
                        "type": "input_image",
                        "image_url": "data:image/png;base64,AA==",
                    }],
                }],
            },
        )

        self.assertEqual(response.status, 200)
        self.assertEqual(self.runtime.starts, ["a"])

    async def test_multimodal_image_count_limit_rejects_before_admission(self):
        self.resources["a"].update(
            supported_request_shapes=["openai.chat", "openai.multimodal_chat"],
            max_image_inputs=1,
        )

        response = await self.client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer secret"},
            json={
                "model": "a",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                        {"type": "input_image", "image_url": "data:image/png;base64,AA=="},
                    ],
                }],
            },
        )

        self.assertEqual(response.status, 400)
        self.assertEqual((await response.json())["error"]["code"], "image_input_limit_exceeded")
        self.assertEqual(self.runtime.starts, [])
        self.assertEqual(self.forwarded, [])

    async def test_legacy_completion_multimodal_image_limit_rejects_before_admission(self):
        self.resources["a"].update(
            supported_request_shapes=["openai.completion", "openai.multimodal_completion"],
            max_image_inputs=1,
        )
        for path in ("/completion", "/v1/completions"):
            with self.subTest(path=path):
                response = await self.client.post(
                    path,
                    headers={"Authorization": "Bearer secret"},
                    json={
                        "model": "a",
                        "prompt": {
                            "prompt_string": "<__media__><__media__>",
                            "multimodal_data": ["YQ==", "Yg=="],
                        },
                    },
                )

                self.assertEqual(response.status, 400)
                self.assertEqual(
                    (await response.json())["error"]["code"],
                    "image_input_limit_exceeded",
                )
        self.assertEqual(self.runtime.starts, [])
        self.assertEqual(self.forwarded, [])

    async def test_legacy_completion_rejects_multimodal_input_without_profile(self):
        for path, body in (
            (
                "/completion",
                {
                    "model": "a",
                    "prompt": {
                        "prompt_string": "<__media__>",
                        "multimodal_data": ["YQ=="],
                    },
                },
            ),
            (
                "/upstream/a/completions",
                {
                    "prompt": {
                        "prompt_string": "<__media__>",
                        "multimodal_data": ["YQ=="],
                    },
                },
            ),
        ):
            with self.subTest(path=path):
                response = await self.client.post(
                    path,
                    headers={"Authorization": "Bearer secret"},
                    json=body,
                )

                self.assertEqual(response.status, 400)
                self.assertEqual(
                    (await response.json())["error"]["code"],
                    "unsupported_request_shape",
                )
        self.assertEqual(self.runtime.starts, [])
        self.assertEqual(self.forwarded, [])

    async def test_audio_and_video_inputs_require_explicit_request_profiles(self):
        self.resources["a"]["supported_request_shapes"] = [
            "openai.chat", "openai.responses",
            "openai.multimodal_chat", "openai.multimodal_responses",
        ]
        cases = (
            (
                "/v1/chat/completions",
                {
                    "model": "a",
                    "messages": [{"role": "user", "content": [{
                        "type": "video_url",
                        "video_url": {"url": "data:video/mp4;base64,AA=="},
                    }]}],
                },
            ),
            (
                "/v1/responses",
                {
                    "model": "a",
                    "input": [{"role": "user", "content": [{
                        "type": "input_audio",
                        "input_audio": {"data": "AA==", "format": "wav"},
                    }]}],
                },
            ),
            (
                "/v1/responses",
                {
                    "model": "a",
                    "input": [{"role": "user", "content": [{
                        "type": "input_file",
                        "file_data": "data:application/pdf;base64,AA==",
                    }]}],
                },
            ),
            (
                "/upstream/a/chat/completions",
                {
                    "messages": [{"role": "user", "content": [{
                        "type": "input_video",
                        "video_url": "data:video/mp4;base64,AA==",
                    }]}],
                },
            ),
            (
                "/upstream/a/responses",
                {
                    "input": [{"role": "user", "content": [{
                        "type": "audio_url",
                        "audio_url": {"url": "data:audio/wav;base64,AA=="},
                    }]}],
                },
            ),
        )
        for path, body in cases:
            with self.subTest(path=path):
                response = await self.client.post(
                    path,
                    headers={"Authorization": "Bearer secret"},
                    json=body,
                )

                self.assertEqual(response.status, 400)
                self.assertEqual(
                    (await response.json())["error"]["code"],
                    "unsupported_request_shape",
                )
        self.assertEqual(self.runtime.starts, [])
        self.assertEqual(self.forwarded, [])

    async def test_upstream_chat_alias_enforces_image_limit(self):
        self.resources["a"].update(
            supported_request_shapes=["openai.chat", "openai.multimodal_chat"],
            max_image_inputs=1,
        )
        response = await self.client.post(
            "/upstream/a/chat/completions",
            headers={"Authorization": "Bearer secret"},
            json={"messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
            ]}]},
        )

        self.assertEqual(response.status, 400)
        self.assertEqual((await response.json())["error"]["code"], "image_input_limit_exceeded")
        self.assertEqual(self.runtime.starts, [])
        self.assertEqual(self.forwarded, [])

    async def test_upstream_media_token_count_routes_fail_closed(self):
        cases = (
            (
                "/upstream/a/chat/completions/input_tokens",
                {"messages": [{"role": "user", "content": [{
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AA=="},
                }]}]},
            ),
            (
                "/upstream/a/responses/input_tokens",
                {"input": [{"role": "user", "content": [{
                    "type": "input_image",
                    "image_url": "data:image/png;base64,AA==",
                }]}]},
            ),
            (
                "/upstream/a/v1/messages/count_tokens",
                {"messages": [{"role": "user", "content": [{
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": "AA=="},
                }]}]},
            ),
            (
                "/upstream/a/v1/messages",
                {"messages": [{"role": "user", "content": [{
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": "AA=="},
                }]}]},
            ),
        )
        for path, body in cases:
            with self.subTest(path=path):
                response = await self.client.post(
                    path,
                    headers={"Authorization": "Bearer secret"},
                    json=body,
                )

                self.assertEqual(response.status, 400)
                self.assertEqual(
                    (await response.json())["error"]["code"],
                    "unsupported_request_shape",
                )
        self.assertEqual(self.runtime.starts, [])
        self.assertEqual(self.forwarded, [])

    async def test_upstream_audio_transcription_routes_fail_closed(self):
        for path in (
            "/upstream/a/audio/transcriptions",
            "/upstream/a/v1/audio/transcriptions",
        ):
            with self.subTest(path=path):
                response = await self.client.post(
                    path,
                    headers={
                        "Authorization": "Bearer secret",
                        "Content-Type": "multipart/form-data; boundary=unused",
                    },
                    data=b"audio payload",
                )

                self.assertEqual(response.status, 400)
                self.assertEqual(
                    (await response.json())["error"]["code"],
                    "unsupported_request_shape",
                )
        self.assertEqual(self.runtime.starts, [])
        self.assertEqual(self.forwarded, [])

    async def test_text_worker_rejects_unaccounted_image_request_before_admission(self):
        self.resources["a"]["supported_request_shapes"] = [
            "openai.chat", "openai.completion", "openai.responses"
        ]

        response = await self.client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer secret"},
            json={
                "model": "a",
                "messages": [{
                    "role": "user",
                    "content": [{
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AA=="},
                    }],
                }],
            },
        )

        self.assertEqual(response.status, 400)
        self.assertEqual((await response.json())["error"]["code"], "unsupported_request_shape")
        self.assertEqual(self.runtime.starts, [])
        self.assertEqual(self.forwarded, [])

    async def test_direct_openai_service_does_not_accept_llama_swap_control_routes(self):
        self.resources["native"] = resource(30, "service")
        self.resources["native"].update(gateway_url="http://127.0.0.1:8801")

        response = await self.client.get(
            "/upstream/native/health",
            headers={"Authorization": "Bearer secret"},
        )

        self.assertEqual(response.status, 404)
        self.assertEqual((await response.json())["error"]["code"], "not_found")
        self.assertEqual(self.runtime.starts, [])
        self.assertEqual(self.forwarded, [])

    async def test_direct_openai_stream_holds_its_lease_until_the_worker_finishes(self):
        self.resources["native"] = resource(60, "service")
        self.resources["native"].update(
            gateway_url="http://127.0.0.1:8801",
            slots_url="http://127.0.0.1:8801/slots",
            health_url="http://127.0.0.1:8801/health",
            supported_request_shapes=["openai.chat"],
        )

        async def handle_direct(request):
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await response.write(b'data: {"content":"started"}\n\n')
            self.started.set()
            await self.release.wait()
            await response.write(b"data: [DONE]\n\n")
            return response

        direct_app = web.Application()
        direct_app.router.add_post("/v1/chat/completions", handle_direct)
        direct_server = TestServer(direct_app)
        await direct_server.start_server()
        self.addAsyncCleanup(direct_server.close)
        self.resources["native"]["gateway_url"] = str(direct_server.make_url(""))

        stream = await self.post("native", stream=True)
        await self.started.wait()
        queued = asyncio.create_task(self.post("b"))
        await asyncio.sleep(0.05)
        self.assertFalse(queued.done())
        self.assertEqual(self.runtime.stops, [])

        self.release.set()
        self.assertIn(b"[DONE]", await stream.read())
        admitted = await asyncio.wait_for(queued, 1)

        self.assertEqual(admitted.status, 200)
        self.assertEqual(self.runtime.stops, ["native"])

    async def test_adapter_lease_protects_model_across_http_calls_and_releases_afterwards(self):
        adapter = SchedulerClient(self.policy["socket_path"])
        async with adapter.lease("comfyui"):
            queued = asyncio.create_task(self.post("b"))
            await asyncio.sleep(0.05)
            self.assertFalse(queued.done())
            self.assertEqual(self.runtime.stops, [])
        response = await asyncio.wait_for(queued, 1)
        self.assertEqual(response.status, 200)
        self.assertEqual(self.runtime.stops, ["comfyui"])

    async def test_lost_scheduler_fails_closed(self):
        client = SchedulerClient(str(Path(self.directory.name) / "missing.sock"))
        with self.assertRaises(AdmissionError) as error:
            async with client.lease("a"):
                self.fail("backend ran without admission")
        self.assertEqual(error.exception.status, 503)

    async def test_backend_connection_failure_is_not_mislabeled_as_scheduler_failure(self):
        async def fail():
            raise aiohttp.ClientConnectionError("backend failure")
        with self.assertRaises(aiohttp.ClientConnectionError):
            await SchedulerClient(self.policy["socket_path"]).run("comfyui", fail)
        self.assertEqual((await self.post("b")).status, 200)

    async def test_image_and_both_decision_adapters_preserve_permanent_capacity_errors(self):
        self.resources.update(comfyui=resource(101, "service"), jevk5=resource(101, "service"), clm=resource(101, "service"))
        self.runtime.loaded = {"a": "ready"}
        active_model = ResourceLock(Path(self.policy["socket_path"]).parent / "leases", "a")
        self.assertTrue(active_model.acquire())
        backend = Mock()
        backend.close = AsyncMock()
        backend.generate = AsyncMock(side_effect=AssertionError("image backend ran without admission"))
        backend.decide = AsyncMock(side_effect=AssertionError("Jev backend ran without admission"))
        backend.evaluate = AsyncMock(side_effect=AssertionError("CLM backend ran without admission"))
        clients = []
        try:
            image = TestClient(TestServer(create_image_app(ImageSettings(api_key="secret", scheduler_socket=self.policy["socket_path"]), backend)))
            decision = TestClient(TestServer(create_jev_app(JevSettings(api_key="secret", clm_api_key="secret", scheduler_socket=self.policy["socket_path"]), backend, clm_backend=backend)))
            for client in (image, decision):
                await client.start_server()
                clients.append(client)
            for client, path, payload in (
                (image, "/v1/images/generations", {"prompt": "a cube"}),
                (decision, "/v1/systemone", {"model": "jevk5", "state": "hello", "questions": {"q": {"type": "noul", "instructions": "is this a greeting?"}}}),
                (decision, "/v1/systemone", {"model": "clm-latest", "state": "hello", "questions": {"q": {"type": "noul", "instructions": "is this a greeting?"}}}),
            ):
                response = await client.post(path, json=payload, headers={"Authorization": "Bearer secret"})
                self.assertEqual(response.status, 503)
                self.assertNotIn("Retry-After", response.headers)
                self.assertEqual((await response.json())["error"]["code"], "model_exceeds_budget")
            backend.generate.assert_not_awaited()
            backend.decide.assert_not_awaited()
            backend.evaluate.assert_not_awaited()
        finally:
            active_model.close()
            for client in reversed(clients):
                await client.close()

    async def test_adapter_cancel_keeps_lease_until_owned_cleanup_ends(self):
        client = SchedulerClient(self.policy["socket_path"])
        cleanup_started = asyncio.Event()
        async def operation():
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleanup_started.set()
                await self.release.wait()
        task = asyncio.create_task(client.run("comfyui", operation))
        await self.started.wait()
        task.cancel()
        await cleanup_started.wait()
        task.cancel()
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        queued = asyncio.create_task(self.post("b"))
        await asyncio.sleep(0.05)
        self.assertFalse(queued.done())
        self.release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        response = await asyncio.wait_for(queued, 1)
        self.assertEqual(response.status, 200)

    async def test_quiet_prefill_survives_former_response_deadline(self):
        self.policy.update(response_timeout_s=0.01, request_timeout_s=0.01)
        task = asyncio.create_task(self.post("a", hang=True))
        try:
            await asyncio.sleep(0.05)
            self.assertFalse(task.done())
            self.release.set()
            response = await asyncio.wait_for(task, 1)
            self.assertEqual(response.status, 200)
            self.assertTrue((await response.json())["ok"])
        finally:
            self.release.set()

    async def test_quiet_stream_survives_former_deadlines_and_emits_heartbeats(self):
        self.policy.update(response_timeout_s=0.01, request_timeout_s=0.01, heartbeat_s=0.01)
        response = await self.post("a", stream=True)
        first = await response.content.readany()
        self.assertIn(b"hello", first)
        queued = asyncio.create_task(self.post("b"))
        try:
            heartbeat = await asyncio.wait_for(response.content.readany(), 1)
            self.assertIn(b": ping", heartbeat)
        finally:
            self.release.set()
        rest = await asyncio.wait_for(response.read(), 1)
        self.assertIn(b"[DONE]", rest)
        self.assertNotIn(b"upstream_timeout", first + heartbeat + rest)
        self.assertEqual((await asyncio.wait_for(queued, 1)).status, 200)

    async def test_backpressure_does_not_expire_a_live_request(self):
        self.policy.update(request_timeout_s=0.01, response_timeout_s=0.01)
        original = web.StreamResponse.write
        write_started, write_release = asyncio.Event(), asyncio.Event()
        async def blocked(response, data):
            if response._req is not None and response._req.app is self.public:
                write_started.set()
                await write_release.wait()
            return await original(response, data)
        try:
            with patch.object(web.StreamResponse, "write", new=blocked):
                response = await self.post("a", stream=True)
                await asyncio.wait_for(write_started.wait(), 1)
                await asyncio.sleep(0.03)
                queued = asyncio.create_task(self.post("b"))
                await asyncio.sleep(0.03)
                self.assertFalse(queued.done())
                write_release.set()
                self.release.set()
                self.assertIn(b"[DONE]", await asyncio.wait_for(response.read(), 1))
                self.assertEqual((await asyncio.wait_for(queued, 1)).status, 200)
        finally:
            write_release.set()
            self.release.set()

    async def test_service_requires_all_health_endpoints_before_admission(self):
        self.resources["comfyui"].update(health_url=str(self.upstream.make_url("/health")), ready_url=str(self.upstream.make_url("/ready")))
        runtime = Runtime(self.policy, self.resources, self.client.session, "secret")
        self.assertFalse(await runtime.ready("comfyui"))
        self.api_ready = True
        self.assertTrue(await runtime.ready("comfyui"))

    async def test_native_expert_cache_url_works_before_running_proxy_is_discovered(self):
        native = adaptive_resource()
        native.pop("expert_cache_url")
        native["kind"] = "text"
        native.pop("gateway_url")
        self.resources["native-moe"] = native
        runtime = Runtime(self.policy, self.resources, self.client.session, "secret")

        self.assertEqual(
            runtime.expert_cache_url("native-moe"),
            f"{self.policy['upstream_url']}/upstream/native-moe/expert-cache",
        )

    async def test_expert_cache_accepts_runtime_maximum_above_profiled_limit(self):
        limits = adaptive_resource()["adaptive_cache"]
        runtime_maximum = limits["maximum_capacity_bytes"] + 10
        self.resources["moe"] = adaptive_resource()
        self.resources["moe"]["expert_cache_url"] = str(self.upstream.make_url("/expert-cache"))
        self.expert_cache_state = {
            **limits,
            "maximum_capacity_bytes": runtime_maximum,
            "capacity_bytes": limits["minimum_capacity_bytes"],
            "pending_backing_release": False,
            "quarantined": False,
        }
        runtime = Runtime(self.policy, self.resources, self.client.session, "secret")

        info = await runtime.expert_cache_info("moe")

        self.assertEqual(info["maximum_capacity_bytes"], runtime_maximum)

    async def test_expert_cache_rejects_runtime_maximum_below_profiled_limit(self):
        limits = adaptive_resource()["adaptive_cache"]
        self.resources["moe"] = adaptive_resource()
        self.resources["moe"]["expert_cache_url"] = str(self.upstream.make_url("/expert-cache"))
        self.expert_cache_state = {
            **limits,
            "maximum_capacity_bytes": limits["maximum_capacity_bytes"] - 1,
            "capacity_bytes": limits["minimum_capacity_bytes"],
            "pending_backing_release": False,
            "quarantined": False,
        }
        runtime = Runtime(self.policy, self.resources, self.client.session, "secret")

        with self.assertRaises(ExpertCacheStateError) as caught:
            await runtime.expert_cache_info("moe")

        self.assertEqual(caught.exception.reason_code, "profile_bounds_mismatch")

    async def test_expert_cache_still_rejects_capacity_above_profiled_limit(self):
        limits = adaptive_resource()["adaptive_cache"]
        self.resources["moe"] = adaptive_resource()
        self.resources["moe"]["expert_cache_url"] = str(self.upstream.make_url("/expert-cache"))
        self.expert_cache_state = {
            **limits,
            "maximum_capacity_bytes": limits["maximum_capacity_bytes"] + 10,
            "capacity_bytes": limits["maximum_capacity_bytes"] + 1,
            "pending_backing_release": False,
            "quarantined": False,
        }
        runtime = Runtime(self.policy, self.resources, self.client.session, "secret")

        with self.assertRaises(ExpertCacheStateError) as caught:
            await runtime.expert_cache_info("moe")

        self.assertEqual(caught.exception.reason_code, "capacity_out_of_bounds")

    async def test_quarantined_expert_cache_state_has_safe_reason_code(self):
        limits = adaptive_resource()["adaptive_cache"]
        self.resources["moe"] = adaptive_resource()
        self.resources["moe"]["expert_cache_url"] = str(self.upstream.make_url("/expert-cache"))
        self.expert_cache_state = {
            **limits,
            "capacity_bytes": limits["maximum_capacity_bytes"],
            "pending_backing_release": False,
            "quarantined": True,
        }
        runtime = Runtime(self.policy, self.resources, self.client.session, "secret")

        with self.assertRaises(ExpertCacheStateError) as caught:
            await runtime.expert_cache_info("moe")

        self.assertEqual(caught.exception.reason_code, "quarantined")

    async def test_pending_backing_is_ready_for_accounting_but_not_transfer(self):
        limits = adaptive_resource()["adaptive_cache"]
        self.resources["moe"] = adaptive_resource()
        self.resources["moe"].update(
            expert_cache_url=str(self.upstream.make_url("/expert-cache")),
            health_url=str(self.upstream.make_url("/health")),
        )
        self.expert_cache_state = {
            **limits,
            "capacity_bytes": limits["maximum_capacity_bytes"],
            "pending_backing_release": True,
            "quarantined": False,
        }
        runtime = Runtime(self.policy, self.resources, self.client.session, "secret")

        self.assertTrue(await runtime.ready("moe"))
        info = await runtime.expert_cache_info(
            "moe", allow_pending_backing_release=True
        )
        self.assertTrue(info["pending_backing_release"])
        with self.assertRaises(ExpertCacheStateError) as caught:
            await runtime.expert_cache_info("moe")
        self.assertEqual(caught.exception.reason_code, "pending_backing_release")

    async def test_resize_accepts_reported_allocator_capacity_for_scheduler_accounting(self):
        limits = adaptive_resource()["adaptive_cache"]
        self.resources["moe"] = adaptive_resource()
        self.resources["moe"].update(
            expert_cache_url=str(self.upstream.make_url("/expert-cache")),
            health_url=str(self.upstream.make_url("/health")),
        )
        self.expert_cache_state = {
            **limits,
            "capacity_bytes": 50,
            "pending_backing_release": False,
            "quarantined": False,
        }
        self.expert_cache_post_capacity = 35
        runtime = Runtime(self.policy, self.resources, self.client.session, "secret")

        info = await runtime.resize_expert_cache("moe", 30)

        self.assertEqual(info["capacity_bytes"], 35)

    async def test_resize_failure_logs_backend_detail_without_newlines(self):
        limits = adaptive_resource()["adaptive_cache"]
        self.resources["moe"] = adaptive_resource()
        self.resources["moe"].update(
            expert_cache_url=str(self.upstream.make_url("/expert-cache")),
            expert_cache_resize_events=True,
            health_url=str(self.upstream.make_url("/health")),
        )
        self.expert_cache_state = {
            **limits,
            "capacity_bytes": limits["minimum_capacity_bytes"],
            "pending_backing_release": False,
            "quarantined": False,
        }
        self.expert_cache_resize_failure = "std::bad_alloc:\n \x1bunable to commit backing"
        runtime = Runtime(self.policy, self.resources, self.client.session, "secret")

        with self.assertLogs(scheduler_api.LOG, level="INFO") as captured:
            with self.assertRaises(ExpertCacheResizeError) as raised:
                await runtime.resize_expert_cache(
                    "moe", limits["minimum_capacity_bytes"] + 1,
                    operation_id="a" * 32,
                )

        self.assertEqual(raised.exception.reason_code, "runtime_resize_failed")
        self.assertEqual(
            raised.exception.detail,
            "std::bad_alloc: unable to commit backing",
        )
        self.assertTrue(any(
            "event=resize.failed" in line
            and "detail=std::bad_alloc: unable to commit backing" in line
            for line in captured.output
        ))

    async def test_comfy_start_command_waits_past_systemd_readiness_deadline(self):
        self.resources["comfyui"].update(
            units=["comfyui.service"], start_command_timeout_s=240
        )
        runtime = Runtime(self.policy, self.resources, self.client.session, "secret")
        with patch.object(runtime, "command", new=AsyncMock(return_value="")) as command, patch.object(
            runtime, "ready", new=AsyncMock(return_value=True)
        ):
            await runtime.start("comfyui")
        self.assertEqual(command.await_args.args, ("start", "comfyui.service"))
        self.assertEqual(command.await_args.kwargs["timeout"], 240)

    async def test_direct_worker_readiness_is_bounded_by_startup_deadline(self):
        self.resources["deepseek"] = resource(40, "service")
        self.resources["deepseek"].update(
            units=["deepseek-hip.service"],
            start_command_timeout_s=1,
            startup_timeout_s=0.03,
        )
        runtime = Runtime(self.policy, self.resources, self.client.session, "secret")
        with patch.object(runtime, "command", new=AsyncMock(return_value="")) as command, patch.object(
            runtime, "ready", new=AsyncMock(return_value=False)
        ), patch.object(runtime, "failed", new=AsyncMock(return_value=False)):
            with self.assertRaisesRegex(TimeoutError, "startup deadline"):
                await runtime.start("deepseek")
        self.assertEqual(command.await_args.args, ("start", "deepseek-hip.service"))
        self.assertEqual(command.await_args.kwargs["timeout"], 0.03)

    async def test_image_readiness_requires_installed_workflow_nodes(self):
        self.resources["comfyui"].update(health_url=str(self.upstream.make_url("/health")), nodes_url=str(self.upstream.make_url("/object_info")), required_nodes=["UnetLoaderGGUF"])
        runtime = Runtime(self.policy, self.resources, self.client.session, "secret")
        self.assertFalse(await runtime.ready("comfyui"))
        self.custom_nodes["UnetLoaderGGUF"] = {}
        self.assertTrue(await runtime.ready("comfyui"))
        self.assertEqual(self.full_catalog_requests, 0)

    async def test_image_readiness_probes_only_required_node_metadata(self):
        required = ["UnetLoaderGGUF", "TextEncodeQwenImage21", "QwenImage21Cache"]
        self.resources["comfyui"].update(health_url=str(self.upstream.make_url("/health")), nodes_url=str(self.upstream.make_url("/object_info")), required_nodes=required)
        self.custom_nodes.update({name: {} for name in required})
        runtime = Runtime(self.policy, self.resources, self.client.session, "secret")

        self.assertTrue(await runtime.ready("comfyui"))
        self.assertEqual(self.full_catalog_requests, 0)

        del self.custom_nodes["QwenImage21Cache"]
        self.assertFalse(await runtime.ready("comfyui"))
        self.assertEqual(self.full_catalog_requests, 0)

    async def test_deployment_waits_for_existing_leases_before_stopping_anything(self):
        self.runtime.loaded = {"a": "ready"}
        lock = ResourceLock(Path(self.policy["socket_path"]).parent / "leases", "a")
        self.assertTrue(lock.acquire())
        try:
            with patch.object(deploy_scheduler, "Runtime", return_value=self.runtime), patch.object(deploy_scheduler, "time", Mock(monotonic=Mock(side_effect=[0, 1]))), patch.object(deploy_scheduler.asyncio, "sleep", new_callable=AsyncMock):
                with self.assertRaisesRegex(RuntimeError, "did not drain"):
                    await deploy_scheduler.drain(self.policy, {"resources": self.resources}, "secret", 0.5)
            self.assertEqual(self.runtime.stops, [])
        finally:
            lock.close()


class ProfileTest(unittest.TestCase):
    def test_capacity_limits_apply_redqueen_budgets_and_prefill_headroom(self):
        gib = 1024**3
        policy = {
            "gpu_budget_mib": 88 * 1024,
            "host_budget_mib": 104 * 1024,
            "minimum_host_available_mib": 8 * 1024,
            "prefill_headroom_mib": 5 * 1024,
        }
        limits = getattr(scheduler_api, "capacity_limits", None)
        self.assertTrue(callable(limits), "scheduler must expose its checked capacity calculation")

        normal = limits(policy, gpu_total=96 * gib, host_total=125 * gib)
        self.assertEqual(normal, Memory(88 * gib, 104 * gib))

        constrained = limits(policy, gpu_total=96 * gib, host_total=110 * gib)
        self.assertEqual(constrained, Memory(88 * gib, 97 * gib))

    def test_capacity_limits_reject_invalid_budgets(self):
        policy = {
            "gpu_budget_mib": 0,
            "host_budget_mib": 104 * 1024,
            "minimum_host_available_mib": 8 * 1024,
            "prefill_headroom_mib": 5 * 1024,
        }
        self.assertTrue(callable(getattr(scheduler_api, "capacity_limits", None)))
        with self.assertRaises(ValueError):
            scheduler_api.capacity_limits(
                policy, gpu_total=96 * 1024**3, host_total=125 * 1024**3
            )

    def test_worker_footprint_includes_each_memory_component(self):
        spec = {
            "gpu_bytes": 16, "host_bytes": 11,
            "memory": memory_contract((10, 8), reclaimable=(5, 2), transient_peak=(1, 1)),
        }
        self.assertEqual(footprint(spec), Memory(16, 11))
        spec["gpu_bytes"] += 1
        with self.assertRaises(ValueError):
            footprint(spec)

    def test_device_context_and_compute_are_included(self):
        rows = memory_rows("ROCm0 73486 1914 901\nHost 12620 429 47\n")
        self.assertGreater(sum(rows["ROCm0"]), (73486 + 1914 + 901) * MIB)
        with self.assertRaises(ValueError):
            memory_rows("not a memory estimate")

    def test_all_gguf_shards_are_accounted_for(self):
        self.assertEqual(len(weight_paths(Path("/models/model-00001-of-00003.gguf"))), 3)

    def test_profiles_fail_closed_after_configuration_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config"
            path.write_text("context=65536")
            identity = [{"model": "a", "repositories": {"repository": "test/a"}, "revisions": {"revision": "1"}, "manifest_sha256": "0" * 64}]
            profile = resource(1)
            profile.update({
                "memory": memory_contract((1, 1)),
                "artifact_identity": identity,
                "artifact_fingerprint": fingerprint_artifacts(identity),
                "runtime_identity": {"path": "/bin/llama-server", "size": 1, "mtime_ns": 1},
                "supported_request_shapes": ["openai.chat"],
                "concurrency_limit": 1,
            })
            document = {"version": 1, "worker_profile_version": 1, "resources": {"a": profile}, "sources": [file_signature(path, content=True)]}
            verify_profiles(document)
            profile.update({
                "kind": "service",
                "gateway_url": "http://127.0.0.1:8085",
                "slots_url": "http://127.0.0.1:8085/slots",
                "health_url": "http://127.0.0.1:8085/health",
                "units": ["native.service"],
                "runtime_identity": {"service_units": ["native.service"]},
            })
            verify_profiles(document)
            profile["gateway_url"] = "http://example.com:8085"
            with self.assertRaisesRegex(ValueError, "direct OpenAI service contract"):
                verify_profiles(document)
            for field, suffix in (
                ("gateway_url", ""),
                ("slots_url", "/slots"),
                ("health_url", "/health"),
            ):
                original = profile[field]
                profile[field] = f"http://127.0.0.1:0{suffix}"
                with self.assertRaisesRegex(ValueError, "direct OpenAI service contract"):
                    verify_profiles(document)
                profile[field] = original
            path.write_text("context=220000")
            with self.assertRaises(ValueError):
                verify_profiles(document)

    def test_profile_builder_passes_context_and_accounts_for_projectors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "llama-server").write_text("runtime")
            model_dir = root / "models/a"
            model_dir.mkdir(parents=True)
            (model_dir / "weights.gguf").write_bytes(b"weights")
            (model_dir / "projector.gguf").write_bytes(b"projector")
            metadata = root / "metadata/a"
            metadata.mkdir(parents=True)
            (metadata / "SOURCE_REVISION").write_text("gguf_repository=test/a\ngguf_revision=1234\n")
            (metadata / "SHA256SUMS").write_text("0" * 64 + "  weights.gguf\n" + "1" * 64 + "  projector.gguf\n")
            estimator = root / "llama-fit-params"
            estimator.write_text(f"#!{sys.executable}\nimport sys\nassert '--pooling' not in sys.argv\nctx = int(sys.argv[sys.argv.index('--ctx-size') + 1])\nprint(f'ROCm0 100 {{ctx}} 10\\nHost 20 0 1')\n")
            estimator.chmod(0o700)
            router_path, policy_path = root / "router.json", root / "policy.json"
            policy_path.write_text(json.dumps({"estimator": str(estimator), "model_root": str(root / "models"), "artifact_metadata_root": str(root / "metadata"), "estimate_margin": 1.08, "model_overhead_mib": 512, "services": {}}))
            def profile(context):
                command = f"{root}/llama-server --model {model_dir}/weights.gguf --mmproj {model_dir}/projector.gguf --alias a --ctx-size {context} --pooling mean --parallel 1 --gpu-layers 999"
                router_path.write_text(json.dumps({"models": {"a": {"cmd": command}}}))
                return build_profiles(router_path, policy_path)
            smaller, larger = profile(2048), profile(8192)
            self.assertGreater(larger["resources"]["a"]["gpu_bytes"], smaller["resources"]["a"]["gpu_bytes"])
            self.assertEqual(larger["resources"]["a"]["context_tokens"], 8192)
            self.assertEqual(larger["resources"]["a"]["weights_bytes"], 16)
            self.assertEqual(larger["version"], 1)
            self.assertEqual(larger["worker_profile_version"], 1)
            self.assertEqual(
                larger["resources"]["a"]["supported_request_shapes"],
                [
                    "openai.chat", "openai.completion", "openai.responses",
                    "openai.multimodal_chat", "openai.multimodal_responses",
                ],
            )
            self.assertEqual(larger["resources"]["a"]["memory"]["mandatory"]["gpu_bytes"], larger["resources"]["a"]["gpu_bytes"])
            self.assertEqual(larger["resources"]["a"]["concurrency_limit"], 1)
            self.assertEqual(larger["resources"]["a"]["artifact_identity"][0]["revisions"]["gguf_revision"], "1234")
            verify_profiles(larger)

    def test_streamed_moe_profile_uses_resident_fit_not_full_artifact_size(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "llama-server").write_text("runtime")
            model_dir = root / "models/a"
            model_dir.mkdir(parents=True)
            model_path = model_dir / "weights.gguf"
            with model_path.open("wb") as model:
                model.truncate(4 * 1024**3)
            projector = model_dir / "projector.gguf"
            projector.write_bytes(b"projector")
            metadata = root / "metadata/a"
            metadata.mkdir(parents=True)
            (metadata / "SOURCE_REVISION").write_text("gguf_repository=test/a\ngguf_revision=1234\n")
            (metadata / "SHA256SUMS").write_text("0" * 64 + "  weights.gguf\n" + "1" * 64 + "  projector.gguf\n")
            estimator = root / "llama-fit-params"
            estimator.write_text(
                f"#!{sys.executable}\nimport sys\n"
                "if '--moe-stream' in sys.argv:\n"
                " assert '--moe-stream-direct' in sys.argv\n"
                " assert sys.argv[sys.argv.index('--moe-stream-cache') + 1] == '64s'\n"
                " assert sys.argv[sys.argv.index('--moe-stream-io-threads') + 1] == '1'\n"
                "print('ROCm0 100 20 10\\nHost 20 0 1')\n"
            )
            estimator.chmod(0o700)
            router_path, policy_path = root / "router.json", root / "policy.json"
            command = (
                f"{root}/llama-server --model {model_path} --mmproj {projector} "
                "--ctx-size 2048 --parallel 1 --batch-size 32 --ubatch-size 1 "
                "--gpu-layers 999 --moe-stream --moe-stream-direct "
                "--moe-stream-cache 64s --moe-stream-io-threads 1"
            )
            router_path.write_text(json.dumps({"models": {"a": {"cmd": command}}}))
            policy_path.write_text(json.dumps({
                "estimator": str(estimator), "model_root": str(root / "models"),
                "artifact_metadata_root": str(root / "metadata"), "estimate_margin": 1.08,
                "model_overhead_mib": 512, "moe_stream_overhead_mib": 1536, "services": {},
            }))

            document = build_profiles(router_path, policy_path)
            profile = document["resources"]["a"]

            self.assertGreater(profile["weights_bytes"], 4 * 1024**3)
            self.assertGreater(profile["host_bytes"], 2 * 1024**3)
            self.assertLess(profile["host_bytes"], model_path.stat().st_size)
            self.assertEqual(profile["memory"]["mandatory"]["gpu_bytes"], profile["gpu_bytes"])
            self.assertNotIn("openai.multimodal_chat", profile["supported_request_shapes"])
            self.assertNotIn("openai.multimodal_responses", profile["supported_request_shapes"])
            self.assertNotIn("openai.multimodal_completion", profile["supported_request_shapes"])
            verify_profiles(document)

            policy = json.loads(policy_path.read_text())
            for invalid_overhead in (-1, True, "1536"):
                policy["moe_stream_overhead_mib"] = invalid_overhead
                policy_path.write_text(json.dumps(policy))
                with self.subTest(overhead=invalid_overhead), self.assertRaisesRegex(
                    ValueError, "moe_stream_overhead_mib"
                ):
                    build_profiles(router_path, policy_path)
            policy["moe_stream_overhead_mib"] = 1536
            policy_path.write_text(json.dumps(policy))

            invalid_commands = (
                command.replace("--moe-stream-direct ", ""),
                command.replace("--moe-stream-cache 64s", "--moe-stream-cache 0s"),
                command.replace("--moe-stream-io-threads 1", "--moe-stream-io-threads 0"),
                command.replace("--parallel 1", "--parallel 2"),
                command.replace("--ubatch-size 1", "--ubatch-size 2"),
                command + " --split-mode row",
                command + " --override-tensor 'blk.*=CPU'",
                command.replace("--moe-stream ", ""),
            )
            for invalid in invalid_commands:
                router_path.write_text(json.dumps({"models": {"a": {"cmd": invalid}}}))
                with self.subTest(command=invalid), self.assertRaisesRegex(ValueError, "MoE streaming"):
                    build_profiles(router_path, policy_path)

            vision_command = command + " --image-max-tokens 4096"
            router_path.write_text(json.dumps({"models": {"a": {"cmd": vision_command}}}))
            policy["multimodal_profiles"] = {"a": {"workspace_mib": 2, "max_images": 1}}
            policy_path.write_text(json.dumps(policy))
            vision_document = build_profiles(router_path, policy_path)
            vision_profile = vision_document["resources"]["a"]
            self.assertIn("openai.multimodal_chat", vision_profile["supported_request_shapes"])
            self.assertIn("openai.multimodal_responses", vision_profile["supported_request_shapes"])
            self.assertNotIn("openai.multimodal_completion", vision_profile["supported_request_shapes"])
            self.assertEqual(vision_profile["max_image_inputs"], 1)
            self.assertEqual(vision_profile["multimodal_workspace_mib"], 2)
            self.assertEqual(
                vision_profile["memory"]["transient_peak"]["gpu_bytes"],
                math.ceil(2 * MIB * 1.08),
            )
            self.assertEqual(
                vision_profile["memory"]["transient_peak"]["host_bytes"],
                math.ceil(2 * MIB * 1.08),
            )
            verify_profiles(vision_document)

            router_path.write_text(json.dumps({"models": {"a": {"cmd": command}}}))
            with self.assertRaisesRegex(ValueError, "needs --image-max-tokens"):
                build_profiles(router_path, policy_path)

    def test_adaptive_stream_profile_reserves_minimum_and_reclaimable_growth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "llama-server").write_text("runtime")
            model_dir = root / "models/a"
            model_dir.mkdir(parents=True)
            model_path = model_dir / "weights.gguf"
            model_path.write_bytes(b"weights")
            metadata = root / "metadata/a"
            metadata.mkdir(parents=True)
            (metadata / "SOURCE_REVISION").write_text("gguf_repository=test/a\ngguf_revision=1234\n")
            (metadata / "SHA256SUMS").write_text("0" * 64 + "  weights.gguf\n")
            estimator = root / "llama-fit-params"
            estimator.write_text(
                f"#!{sys.executable}\nimport sys\n"
                "cache = sys.argv[sys.argv.index('--moe-stream-cache') + 1]\n"
                "if cache == '4g':\n"
                " print('ROCm0 100 20 10\\nHost 20 0 1\\nMoEStreamCache 4294967296 4294967296 68719476736')\n"
                "else:\n"
                " print('ROCm0 200 20 10\\nHost 40 0 1\\nMoEStreamCache 66571993088 66571993088 68719476736')\n"
            )
            estimator.chmod(0o700)
            router_path, policy_path = root / "router.json", root / "policy.json"
            command = (
                f"{root}/llama-server --model {model_path} --alias a --ctx-size 2048 "
                "--parallel 1 --batch-size 32 --ubatch-size 1 --gpu-layers 999 "
                "--moe-stream --moe-stream-direct --moe-stream-cache 4g "
                "--moe-stream-cache-max 64g --moe-stream-io-threads 1"
            )
            router_path.write_text(json.dumps({"models": {"a": {"cmd": command}}}))
            policy_path.write_text(json.dumps({
                "estimator": str(estimator), "model_root": str(root / "models"),
                "artifact_metadata_root": str(root / "metadata"), "estimate_margin": 1.1,
                "model_overhead_mib": 1, "moe_stream_overhead_mib": 0, "services": {},
            }))

            profile = build_profiles(router_path, policy_path)["resources"]["a"]
            self.assertEqual(profile["adaptive_cache"], {
                "minimum_capacity_bytes": 4 * 1024**3,
                "maximum_capacity_bytes": 62 * 1024**3,
                "reservation_multiplier": 1.1,
            })
            self.assertEqual(
                profile["memory"]["reclaimable"]["gpu_bytes"], math.ceil(100 * MIB * 1.1)
            )
            self.assertEqual(
                profile["memory"]["reclaimable"]["host_bytes"], math.ceil(120 * MIB * 1.1)
            )

    def test_comfy_profile_separates_resident_weights_from_peak_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = root / "models/qwen-image"
            (model_dir / "diffusion_models").mkdir(parents=True)
            (model_dir / "text_encoders").mkdir()
            files = [model_dir / "diffusion_models/model.gguf", model_dir / "text_encoders/encoder.safetensors"]
            for path in files:
                path.write_bytes(b"weights")
            metadata = root / "metadata/qwen-image"
            metadata.mkdir(parents=True)
            (metadata / "SOURCE_REVISION").write_text("weights_repository=test/qwen-image\nweights_revision=abcd\n")
            (metadata / "SHA256SUMS").write_text("0" * 64 + "  diffusion_models/model.gguf\n" + "1" * 64 + "  text_encoders/encoder.safetensors\n")
            router_path, policy_path = root / "router.json", root / "policy.json"
            router_path.write_text(json.dumps({"models": {}}))
            policy_path.write_text(json.dumps({
                "model_root": str(root / "models"),
                "artifact_metadata_root": str(root / "metadata"),
                "estimate_margin": 1.1,
                "services": {"comfyui": {
                    "weights": [str(path) for path in files], "workspace_mib": 2,
                    "units": ["comfyui.service"],
                    "supported_request_shapes": ["openai.image_generation", "openai.image_edit"],
                    "concurrency_limit": 1,
                }},
            }))
            profile = build_profiles(router_path, policy_path)
            comfy = profile["resources"]["comfyui"]
            self.assertEqual(comfy["memory"]["mandatory"]["gpu_bytes"], math.ceil(14 * 1.1))
            self.assertEqual(comfy["memory"]["transient_peak"]["gpu_bytes"], 2 * MIB)
            self.assertEqual(comfy["memory"]["reclaimable"], {"gpu_bytes": 0, "host_bytes": 0})
            self.assertEqual(comfy["gpu_bytes"], comfy["memory"]["mandatory"]["gpu_bytes"] + 2 * MIB)
            verify_profiles(profile)

    def test_deepseek_profile_uses_allocation_free_resident_estimate_and_request_envelope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_root = root / "models"
            artifact_root = model_root / "deepseek-v4.1-flash-1017"
            inference_dir = artifact_root / "inference"
            inference_dir.mkdir(parents=True)
            artifact = inference_dir / "weights.bin"
            artifact.write_bytes(b"weights")
            metadata = root / "metadata/deepseek-v4.1-flash-1017"
            metadata.mkdir(parents=True)
            (metadata / "SOURCE_REVISION").write_text(
                "repository=deepseek-ai/test\nrevision=abcd\n"
            )
            digest = __import__("hashlib").sha256(artifact.read_bytes()).hexdigest()
            (metadata / "SHA256SUMS").write_text(f"{digest}  inference/weights.bin\n")
            inventory = root / "deepseek.files.tsv"
            inventory.write_text(f"inference/weights.bin\t{artifact.stat().st_size}\t{digest}\t{'a' * 40}\n")
            profiler = root / "profile.py"
            profiler.write_text(
                "import json; print(json.dumps({'resident_device_bytes': 100, 'resident_host_bytes': 80}))\n"
            )
            router_path, policy_path = root / "router.json", root / "policy.json"
            router_path.write_text(json.dumps({"models": {}}))
            service = {
                "units": ["deepseek-hip.service"],
                "start_command_timeout_s": 1200,
                "health_url": "http://127.0.0.1:8085/health",
                "slots_url": "http://127.0.0.1:8085/slots",
                "gateway_url": "http://127.0.0.1:8085",
                "bind": "127.0.0.1",
                "port": 8085,
                "profile_python": sys.executable,
                "profile_script": str(profiler),
                "artifact_root": str(artifact_root),
                "inference_dir": str(inference_dir),
                "artifact_inventory": str(inventory),
                "request_peak_mib": 36864,
                "expert_cache_mib": 1024,
                "expert_cache_max_mib": 65536,
                "expert_cache_url": "http://127.0.0.1:8085/expert-cache",
                "context_tokens": 220000,
                "supported_request_shapes": ["openai.chat", "openai.multimodal_chat"],
                "concurrency_limit": 1,
            }
            policy_path.write_text(json.dumps({
                "estimator": str(root / "estimator"),
                "model_root": str(model_root),
                "artifact_metadata_root": str(root / "metadata"),
                "estimate_margin": 1.1,
                "model_overhead_mib": 1,
                "services": {"deepseek": service},
            }))

            profile = build_profiles(router_path, policy_path)
            deepseek = profile["resources"]["deepseek"]
            self.assertEqual(deepseek["weights_bytes"], artifact.stat().st_size)
            self.assertNotIn("exclusive_execution", deepseek)
            self.assertEqual(deepseek["context_tokens"], 220000)
            cache_minimum = math.ceil(1024 * MIB * 1.1)
            self.assertEqual(deepseek["memory"]["mandatory"]["gpu_bytes"], math.ceil(100 * 1.1) + MIB + cache_minimum)
            self.assertEqual(deepseek["memory"]["mandatory"]["host_bytes"], math.ceil(80 * 1.1) + MIB + cache_minimum)
            self.assertEqual(deepseek["memory"]["reclaimable"]["gpu_bytes"], math.ceil((65536 - 1024) * MIB * 1.1))
            self.assertEqual(deepseek["adaptive_cache"]["maximum_capacity_bytes"], 65536 * MIB)
            self.assertEqual(deepseek["memory"]["transient_peak"]["gpu_bytes"], math.ceil(36864 * MIB * 1.1))
            self.assertEqual(deepseek["memory"]["transient_peak"]["host_bytes"], math.ceil(36864 * MIB * 1.1))
            verify_profiles(profile)


class DeploymentTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.repo = Path(self.directory.name)
        source = self.repo / "hosts/redqueen"
        (source / "model-router").mkdir(parents=True)
        (source / "systemd").mkdir()
        for unit in deploy_scheduler.FRONTS + deploy_scheduler.MANAGED:
            (source / "systemd" / unit).write_text("unit")
        self.profile_path = self.repo / "run/profiles.json"
        self.profile_path.parent.mkdir()
        self.profile_path.write_text("old profiles")
        self.policy = {
            "profile_path": str(self.profile_path),
            "socket_path": str(self.profile_path.parent / "scheduler.sock"),
        }
        (source / "model-router/scheduler.json").write_text(json.dumps(self.policy))
        self.lease = Mock()
        self.commands = []
        self.active_units = {"llama-swap.service"}
        self.user_units = self.repo / "user-units"
        self.user_units.mkdir()
        self.link_units()

    def link_units(self):
        for source in (self.repo / "hosts/redqueen/systemd").iterdir():
            target = self.user_units / source.name
            if not target.is_symlink():
                target.symlink_to(source)

    def run_command(self, *arguments, **kwargs):
        if arguments[0] == "bash" and arguments[1].endswith("link-runtime.sh"):
            self.link_units()
        return subprocess.CompletedProcess(arguments, 0, "", "")

    def command(self, *arguments, **kwargs):
        self.commands.append(arguments)
        if arguments[0] in {"is-enabled", "is-active"}:
            return subprocess.CompletedProcess(arguments, 0 if arguments[1] in self.active_units else 1, "", "")
        if arguments[0] == "disable":
            # systemctl also removes manually linked unit definitions outside
            # the unit search path, not just default.target.wants symlinks.
            for unit in arguments[1:]:
                (self.user_units / unit).unlink(missing_ok=True)
        return subprocess.CompletedProcess(arguments, 0, "", "")

    def test_failed_activation_restores_profiles_services_and_releases_drain_lock(self):
        with patch.object(deploy_scheduler.os, "geteuid", return_value=1002), patch.dict("os.environ", {"QWEN_API_KEY": "test"}), patch.object(deploy_scheduler, "run", side_effect=self.run_command) as commands, patch.object(deploy_scheduler, "systemctl", side_effect=self.command), patch.object(deploy_scheduler, "build_profiles", return_value={"resources": {"a": resource(40)}}), patch.object(deploy_scheduler, "drain", new=AsyncMock(return_value=[self.lease])), patch.object(deploy_scheduler, "healthy", new=AsyncMock(side_effect=RuntimeError("unhealthy"))):
            with self.assertRaisesRegex(RuntimeError, "unhealthy"):
                deploy_scheduler.deploy(self.repo, "previous")
        self.assertEqual(self.profile_path.read_text(), "old profiles")
        self.assertIn(("start", "llama-swap.service"), self.commands)
        self.assertIn(("enable", "llama-swap.service"), self.commands)
        self.assertIn(("disable", "comfyui.service"), self.commands)
        commands.assert_any_call("git", "-C", str(self.repo), "switch", "--detach", "previous")
        self.lease.close.assert_called_once()
        self.assertTrue(all((self.user_units / unit).exists() for unit in deploy_scheduler.FRONTS + deploy_scheduler.MANAGED))

    def test_disabling_boot_starts_keeps_backend_definitions_available(self):
        self.active_units.add("comfyui.service")
        async def healthy(policy):
            if not all((self.user_units / unit).exists() for unit in deploy_scheduler.MANAGED):
                raise RuntimeError("backend unit definitions are missing")
        with patch.object(deploy_scheduler.os, "geteuid", return_value=1002), patch.dict("os.environ", {"QWEN_API_KEY": "test"}), patch.object(deploy_scheduler, "run", side_effect=self.run_command), patch.object(deploy_scheduler, "systemctl", side_effect=self.command), patch.object(deploy_scheduler, "build_profiles", return_value={"resources": {"a": resource(40)}}), patch.object(deploy_scheduler, "drain", new=AsyncMock(return_value=[self.lease])), patch.object(deploy_scheduler, "healthy", new=healthy):
            deploy_scheduler.deploy(self.repo, "previous")
        self.assertTrue(all((self.user_units / unit).exists() for unit in deploy_scheduler.MANAGED))
        self.assertIn(("stop", "comfyui.service"), self.commands)
        self.lease.close.assert_called_once()

    def test_preflight_failure_does_not_touch_running_services(self):
        with patch.object(deploy_scheduler.os, "geteuid", return_value=1002), patch.object(deploy_scheduler, "run", return_value=subprocess.CompletedProcess([], 0, "", "")), patch.object(deploy_scheduler, "systemctl", side_effect=self.command), patch.object(deploy_scheduler, "build_profiles", side_effect=ValueError("bad profile")):
            with self.assertRaisesRegex(ValueError, "bad profile"):
                deploy_scheduler.deploy(self.repo, "previous")
        self.assertEqual(self.commands, [])
        self.assertEqual(self.profile_path.read_text(), "old profiles")


if __name__ == "__main__":
    unittest.main()
