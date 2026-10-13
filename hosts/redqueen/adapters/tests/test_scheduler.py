from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from redqueen_adapters.common import AdmissionError
from redqueen_adapters import deploy_scheduler
from redqueen_adapters.image_api import ImageSettings, create_image_app
from redqueen_adapters.jev_api import JevSettings, create_jev_app
from redqueen_adapters.memory_profile import MIB, build_profiles, file_signature, memory_rows, verify_profiles, weight_paths
from redqueen_adapters.scheduler_api import Memory, Runtime, Scheduler, create_apps
from redqueen_adapters.scheduler_client import ResourceLock, SchedulerClient


def resource(size: int, kind: str = "text") -> dict:
    return {"gpu_bytes": size, "host_bytes": size, "kind": kind, "concurrency": 1, "context_tokens": 32768}


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


class SchedulerTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.resources = {"a": resource(40), "b": resource(40), "large": resource(75)}
        self.runtime = FakeRuntime(self.resources)
        self.policy = {"load_timeout_s": 0.1, "stop_timeout_s": 0.1, "idle_timeout_s": 300}
        self.scheduler = Scheduler(self.resources, self.runtime, Path(self.directory.name), self.policy, clock=lambda: 1000)

    def protect(self, name):
        lock = ResourceLock(self.scheduler.lock_dir, name)
        self.assertTrue(lock.acquire())
        self.addCleanup(lock.close)
        return lock

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
        entered = asyncio.Event()
        async def request():
            async with self.scheduler.request("large"):
                entered.set()
        task = asyncio.create_task(request())
        await self.runtime.starting.wait()
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        self.assertEqual(self.scheduler.states["large"], "starting")
        self.runtime.load_gate.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(entered.is_set())
        self.assertEqual(self.scheduler.active["large"], 0)
        self.assertEqual(self.runtime.loaded["large"], "ready")

    async def test_failed_load_cleans_up_and_failed_cleanup_does_not_release_memory(self):
        self.runtime.load_failure = self.runtime.stop_failure = True
        with self.assertRaises(AdmissionError):
            await self.scheduler.admit("large")
        self.assertEqual(self.scheduler.states["large"], "quarantined")
        with self.assertRaises(AdmissionError):
            await self.scheduler.admit("a")
        self.assertNotIn("a", self.runtime.starts)

    async def test_load_deadline_stops_partial_model(self):
        self.runtime.load_gate = asyncio.Event()
        with self.assertRaises(AdmissionError):
            await self.scheduler.admit("large")
        self.assertEqual(self.runtime.stops, ["large"])
        self.assertEqual(self.runtime.loaded, {})

    async def test_external_gpu_memory_is_reserved(self):
        self.runtime.external = Memory(30, 0)
        with self.assertRaises(AdmissionError):
            await self.scheduler.admit("large")
        self.assertEqual(self.runtime.starts, [])

    async def test_reaper_preserves_active_work_and_stops_idle_models(self):
        self.runtime.loaded = {"a": "ready", "b": "ready"}
        self.scheduler.last_used = {"a": 10, "b": 20}
        self.protect("a")
        await self.scheduler.reap()
        self.assertEqual(self.runtime.stops, ["b"])


class ProxyTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.release = asyncio.Event()
        self.started = asyncio.Event()
        self.api_ready = False
        self.custom_nodes = {}
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
                await response.write(b'data: {"content":"hello"}\n\n')
                self.started.set()
                await self.release.wait()
                await response.write(b'data: [DONE]\n\n')
                return response
            return web.json_response({"model": payload["model"], "ok": True})
        upstream = web.Application()
        upstream.router.add_post("/v1/chat/completions", handle)
        async def health(_):
            return web.json_response({"ok": True})
        async def ready(_):
            return web.json_response({"ok": self.api_ready}, status=200 if self.api_ready else 503)
        async def nodes(_):
            return web.json_response(self.custom_nodes)
        upstream.router.add_get("/health", health)
        upstream.router.add_get("/ready", ready)
        upstream.router.add_get("/object_info", nodes)
        self.upstream = TestServer(upstream)
        await self.upstream.start_server()
        self.addAsyncCleanup(self.upstream.close)
        self.policy = {"socket_path": str(Path(self.directory.name) / "scheduler.sock"), "upstream_url": str(self.upstream.make_url("")).rstrip("/"), "load_timeout_s": 1, "stop_timeout_s": 1, "response_timeout_s": 1, "request_timeout_s": 5, "idle_timeout_s": 300}
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

    async def test_active_stream_survives_new_model_rejection_then_idle_model_is_evicted(self):
        stream = await self.post("a", stream=True)
        self.assertIn(b"hello", await stream.content.readany())
        rejected = await asyncio.wait_for(self.post("b"), 0.2)
        self.assertEqual(rejected.status, 503)
        self.assertEqual(rejected.headers["Retry-After"], "1")
        self.assertEqual((await rejected.json())["error"]["code"], "capacity_unavailable")
        self.assertEqual(self.runtime.stops, [])
        self.release.set()
        self.assertIn(b"[DONE]", await stream.read())
        admitted = await self.post("b")
        self.assertEqual(admitted.status, 200)
        self.assertEqual(self.runtime.stops, ["a"])

    async def test_same_model_returns_429_while_stream_is_active(self):
        stream = await self.post("a", stream=True)
        await stream.content.readany()
        rejected = await asyncio.wait_for(self.post("a"), 0.2)
        self.assertEqual(rejected.status, 429)
        self.release.set()
        await stream.read()

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

    async def test_non_json_content_type_cannot_route_an_unreserved_query_model(self):
        response = await self.client.post("/v1/chat/completions?model=b", headers={"Authorization": "Bearer secret", "Content-Type": "text/plain"}, data='{"model":"a"}')
        self.assertEqual(response.status, 200)
        self.assertEqual(self.forwarded[-1][1], "application/json")
        self.assertEqual(self.runtime.starts, ["a"])

    async def test_adapter_lease_protects_model_across_http_calls_and_releases_afterwards(self):
        adapter = SchedulerClient(self.policy["socket_path"])
        async with adapter.lease("comfyui"):
            response = await self.post("b")
            self.assertEqual(response.status, 503)
            self.assertEqual(self.runtime.stops, [])
        response = await self.post("b")
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

    async def test_image_and_both_decision_adapters_preserve_capacity_errors(self):
        self.resources.update(jevk5=resource(60, "service"), clm=resource(60, "service"))
        backend = Mock()
        backend.close = AsyncMock()
        backend.generate = AsyncMock(side_effect=AssertionError("image backend ran without admission"))
        backend.decide = AsyncMock(side_effect=AssertionError("Jev backend ran without admission"))
        backend.evaluate = AsyncMock(side_effect=AssertionError("CLM backend ran without admission"))
        clients = []
        stream = await self.post("a", stream=True)
        await stream.content.readany()
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
                self.assertEqual(response.headers["Retry-After"], "1")
                self.assertEqual((await response.json())["error"]["code"], "capacity_unavailable")
            backend.generate.assert_not_awaited()
            backend.decide.assert_not_awaited()
            backend.evaluate.assert_not_awaited()
        finally:
            self.release.set()
            await stream.read()
            for client in reversed(clients):
                await client.close()

    async def test_adapter_cancel_keeps_lease_until_background_work_ends(self):
        client = SchedulerClient(self.policy["socket_path"])
        async def operation():
            self.started.set()
            await self.release.wait()
        task = asyncio.create_task(client.run("comfyui", operation))
        await self.started.wait()
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        response = await self.post("b")
        self.assertEqual(response.status, 503)
        self.release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        response = await self.post("b")
        self.assertEqual(response.status, 200)

    async def test_unresponsive_upstream_returns_bounded_504(self):
        self.policy["response_timeout_s"] = 0.03
        try:
            response = await asyncio.wait_for(self.post("a", hang=True), 0.5)
            self.assertEqual(response.status, 504)
            self.assertEqual((await response.json())["error"]["code"], "upstream_timeout")
        finally:
            self.release.set()

    async def test_stalled_stream_returns_an_error_and_releases_the_lease(self):
        self.policy["response_timeout_s"] = 0.03
        try:
            response = await self.post("a", stream=True)
            result = await asyncio.wait_for(response.read(), 0.5)
            self.assertIn(b"upstream_timeout", result)
            self.assertNotIn(b"[DONE]", result)
            admitted = await self.post("b")
            self.assertEqual(admitted.status, 200)
        finally:
            self.release.set()

    async def test_client_not_reading_cannot_keep_a_lease_past_total_deadline(self):
        self.policy.update(request_timeout_s=0.03, response_timeout_s=0.03)
        original = web.StreamResponse.write
        write_started = asyncio.Event()
        async def blocked(response, data):
            if response._req is not None and response._req.app is self.public:
                write_started.set()
                await asyncio.Event().wait()
            return await original(response, data)
        try:
            with patch.object(web.StreamResponse, "write", new=blocked):
                response = await self.post("a", stream=True)
                await asyncio.wait_for(write_started.wait(), 0.5)
                with self.assertRaises(aiohttp.ClientPayloadError):
                    await asyncio.wait_for(response.read(), 0.5)
            admitted = await self.post("b")
            self.assertEqual(admitted.status, 200)
        finally:
            self.release.set()

    async def test_service_requires_all_health_endpoints_before_admission(self):
        self.resources["comfyui"].update(health_url=str(self.upstream.make_url("/health")), ready_url=str(self.upstream.make_url("/ready")))
        runtime = Runtime(self.policy, self.resources, self.client.session, "secret")
        self.assertFalse(await runtime.ready("comfyui"))
        self.api_ready = True
        self.assertTrue(await runtime.ready("comfyui"))

    async def test_image_readiness_requires_installed_workflow_nodes(self):
        self.resources["comfyui"].update(health_url=str(self.upstream.make_url("/health")), nodes_url=str(self.upstream.make_url("/object_info")), required_nodes=["UnetLoaderGGUF"])
        runtime = Runtime(self.policy, self.resources, self.client.session, "secret")
        self.assertFalse(await runtime.ready("comfyui"))
        self.custom_nodes["UnetLoaderGGUF"] = {}
        self.assertTrue(await runtime.ready("comfyui"))

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
            document = {"version": 1, "resources": {"a": resource(1)}, "sources": [file_signature(path, content=True)]}
            verify_profiles(document)
            path.write_text("context=220000")
            with self.assertRaises(ValueError):
                verify_profiles(document)

    def test_profile_builder_passes_context_and_accounts_for_projectors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "llama-server").write_text("runtime")
            (root / "weights.gguf").write_bytes(b"weights")
            (root / "projector.gguf").write_bytes(b"projector")
            estimator = root / "llama-fit-params"
            estimator.write_text(f"#!{sys.executable}\nimport sys\nassert '--pooling' not in sys.argv\nctx = int(sys.argv[sys.argv.index('--ctx-size') + 1])\nprint(f'ROCm0 100 {{ctx}} 10\\nHost 20 0 1')\n")
            estimator.chmod(0o700)
            router_path, policy_path = root / "router.json", root / "policy.json"
            policy_path.write_text(json.dumps({"estimator": str(estimator), "estimate_margin": 1.08, "model_overhead_mib": 512, "services": {}}))
            def profile(context):
                command = f"{root}/llama-server --model {root}/weights.gguf --mmproj {root}/projector.gguf --ctx-size {context} --pooling mean --parallel 1 --gpu-layers 999"
                router_path.write_text(json.dumps({"models": {"a": {"cmd": command}}}))
                return build_profiles(router_path, policy_path)
            smaller, larger = profile(2048), profile(8192)
            self.assertGreater(larger["resources"]["a"]["gpu_bytes"], smaller["resources"]["a"]["gpu_bytes"])
            self.assertEqual(larger["resources"]["a"]["context_tokens"], 8192)
            self.assertEqual(larger["resources"]["a"]["weights_bytes"], 16)
            verify_profiles(larger)


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
        self.policy = {"profile_path": str(self.profile_path)}
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
