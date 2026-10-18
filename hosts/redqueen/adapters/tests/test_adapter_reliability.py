from __future__ import annotations

import asyncio
import json
import math
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from PIL import PngImagePlugin

from redqueen_adapters import image_api
from redqueen_adapters.common import AdmissionError, UpstreamError
from redqueen_adapters.image_api import ComfyBackend, ImageSettings, _normalize_output, create_image_app, normalize_request
from redqueen_adapters.jev_api import JevSettings, LocalCLMBackend, LocalJevBackend, RouterClefBackend, create_jev_app
from redqueen_adapters.jevk5_gguf import spread
from redqueen_adapters.ownership import OwnedRequest, records
from redqueen_adapters.scheduler_client import ResourceLock, SchedulerClient
from test_adapters import FakeClefBackend, FakeImageBackend, FakeJevBackend, png


AUTH = {"Authorization": "Bearer test"}
QUESTION = {"type": "noul", "instructions": "is this a greeting?"}
WORKFLOWS = Path(__file__).resolve().parents[2] / "comfyui"


class ComfyFixture:
    """Small HTTP fixture; no model code, weights, GPU work or global interrupt."""

    def __init__(self, root: Path):
        self.root = root
        for area in ("input", "output"):
            (root / "state" / area).mkdir(parents=True)
        self.jobs: dict[str, dict] = {}
        self.interrupted: list[str] = []
        self.deleted_history: list[str] = []
        self.submitted = asyncio.Event()
        self.complete = asyncio.Event()
        self.allow_interrupt = asyncio.Event()
        self.allow_interrupt.set()
        self.allow_upload_response = asyncio.Event()
        self.allow_upload_response.set()
        self.interrupt_started = asyncio.Event()
        self.malformed_history = False
        self.malformed_prompt = False
        self.malformed_upload = False
        self.partial_output = False
        self.prompt_headers: dict[str, str] = {}
        self.image = png((512, 512), alpha=255)
        self.pre_recorded = False
        app = web.Application()
        app.router.add_post("/prompt", self.prompt)
        app.router.add_get("/history/{job}", self.history)
        app.router.add_get("/queue", self.queue)
        app.router.add_post("/queue", self.delete_queue)
        app.router.add_post("/interrupt", self.interrupt)
        app.router.add_post("/history", self.delete_history)
        app.router.add_get("/view", self.view)
        app.router.add_post("/upload/image", self.upload)
        self.server = TestServer(app)

    async def prompt(self, request):
        body = await request.json()
        self.prompt_headers = dict(request.headers)
        job_id = body["prompt_id"]
        uuid.UUID(job_id)
        manifests = records(self.root / "requests", "comfyui")
        owner = next(item for item in manifests if item.get("job_id") == job_id)
        self.pre_recorded = owner["submitted"] and owner["client_id"] == body["client_id"]
        self.jobs[job_id] = {"client_id": body["client_id"], "request_id": owner["id"], "running": True}
        if self.partial_output:
            output = self.root / "state/output/api" / owner["id"]
            output.mkdir(parents=True)
            (output / "partial.png").write_bytes(b"partial")
        self.submitted.set()
        if self.malformed_prompt:
            return web.json_response([])
        return web.json_response({"prompt_id": job_id})

    async def history(self, request):
        job_id = request.match_info["job"]
        if self.malformed_history:
            return web.Response(text="{broken", content_type="application/json")
        job = self.jobs.get(job_id)
        if job is None or not self.complete.is_set():
            return web.json_response({})
        job["running"] = False
        subfolder = f"api/{job['request_id']}"
        output = self.root / "state/output" / subfolder
        output.mkdir(parents=True, exist_ok=True)
        (output / "result.png").write_bytes(self.image)
        return web.json_response({job_id: {"status": {"status_str": "success"}, "outputs": {"9": {"images": [{"type": "output", "filename": "result.png", "subfolder": subfolder}]}}}})

    async def queue(self, request):
        return web.json_response({"queue_pending": [], "queue_running": [[0, job_id, {}, {"client_id": job["client_id"]}] for job_id, job in self.jobs.items() if job["running"]]})

    async def delete_queue(self, request):
        for job_id in (await request.json()).get("delete", []):
            self.jobs.pop(job_id, None)
        return web.Response()

    async def interrupt(self, request):
        body = await request.json()
        if not isinstance(body.get("prompt_id"), str):
            raise AssertionError("global interrupt is forbidden")
        self.interrupt_started.set()
        await self.allow_interrupt.wait()
        self.interrupted.append(body["prompt_id"])
        if body["prompt_id"] in self.jobs:
            self.jobs[body["prompt_id"]]["running"] = False
        return web.Response()

    async def delete_history(self, request):
        body = await request.json()
        if "clear" in body:
            raise AssertionError("global history deletion is forbidden")
        self.deleted_history.extend(body.get("delete", []))
        return web.Response()

    async def view(self, request):
        return web.Response(body=self.image, content_type="image/png")

    async def upload(self, request):
        body = await request.post()
        filename, subfolder = body["image"].filename, body["subfolder"]
        relative = str(Path(subfolder) / filename)
        manifests = records(self.root / "requests", "comfyui")
        if not any(relative in item.get("input_paths", []) for item in manifests):
            raise AssertionError("upload path was not recorded before sending bytes")
        directory = self.root / "state/input" / subfolder
        directory.mkdir(parents=True)
        with body["image"].file as uploaded:
            (directory / filename).write_bytes(uploaded.read())
        await self.allow_upload_response.wait()
        if self.malformed_upload:
            return web.json_response([])
        return web.json_response({"name": filename, "subfolder": subfolder})


class AdapterReliabilityTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    async def client(self, app):
        client = TestClient(TestServer(app, handler_cancellation=True))
        await client.start_server()
        self.addAsyncCleanup(client.close)
        return client

    async def comfy(self, **settings):
        fixture = ComfyFixture(self.root)
        await fixture.server.start_server()
        self.addAsyncCleanup(fixture.server.close)
        config = ImageSettings(api_key="test", comfy_url=str(fixture.server.make_url("")).rstrip("/"), workflow_dir=WORKFLOWS, comfy_state_dir=self.root / "state", ownership_directory=self.root / "requests", minimum_free_bytes=0, poll_interval_s=.001, cleanup_timeout_s=.4, **settings)
        backend = ComfyBackend(config)
        backend.generation = "generation-one"
        self.addAsyncCleanup(backend.close)
        return fixture, backend

    async def test_malformed_enums_and_credentials_are_client_errors(self):
        image = await self.client(create_image_app(ImageSettings(api_key="test"), FakeImageBackend()))
        for name in ("size", "quality", "background"):
            for value in ([], {}):
                response = await image.post("/v1/images/generations", headers=AUTH, json={"prompt": "x", name: value})
                self.assertEqual(response.status, 400)
                self.assertEqual(response.content_type, "application/json")
        response = await image.post("/v1/images/generations", headers={"Authorization": "Bearer é"}, json={"prompt": "x"})
        self.assertEqual(response.status, 401)
        decision = await self.client(create_jev_app(JevSettings(api_key="test"), FakeJevBackend()))
        response = await decision.post("/v1/systemone", headers=AUTH, json={"state": "x", "questions": {"q": {**QUESTION, "type": []}}})
        self.assertEqual(response.status, 400)

    async def test_frontend_rejects_before_multipart_buffering(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def parse(request):
            entered.set()
            await release.wait()
            return normalize_request({"prompt": "x", "size": "512x512"}, edit=True), []
        client = await self.client(create_image_app(ImageSettings(api_key="test", ingress_limit=1), FakeImageBackend()))
        with patch.object(image_api, "parse_edit", new=AsyncMock(side_effect=parse)) as parser:
            first = asyncio.create_task(client.post("/v1/images/edits", headers=AUTH))
            await entered.wait()
            try:
                rejected = await client.post("/v1/images/edits", headers=AUTH)
                self.assertEqual(rejected.status, 429)
                self.assertEqual((await rejected.json())["error"]["code"], "frontend_busy")
                self.assertEqual(parser.await_count, 1)
            finally:
                release.set()
            self.assertEqual((await first).status, 200)

    async def test_nested_multipart_image_is_a_client_error(self):
        client = await self.client(create_image_app(ImageSettings(api_key="test"), FakeImageBackend()))
        form = aiohttp.MultipartWriter("form-data")
        nested = aiohttp.MultipartWriter("mixed")
        nested.append(png((16, 16)), {"Content-Type": "image/png"})
        part = form.append(nested)
        part.set_content_disposition("form-data", name="image")
        response = await client.post("/v1/images/edits", headers=AUTH, data=form)
        self.assertEqual(response.status, 400)
        self.assertEqual((await response.json())["error"]["code"], "invalid_request")

    async def test_decision_json_reads_to_eof_without_an_inference_timer(self):
        release, first_chunk = asyncio.Event(), asyncio.Event()
        payload = await FakeClefBackend().evaluate("clef-flash-q4", {"questions": {"q": QUESTION}})
        encoded = json.dumps(payload).encode()
        async def fragmented(request):
            response = web.StreamResponse(headers={"Content-Type": "application/json"})
            await response.prepare(request)
            await response.write(encoded[:4])
            first_chunk.set()
            await release.wait()
            try:
                await response.write(encoded[4:])
                await response.write_eof()
            except ConnectionResetError:
                pass
            return response
        app = web.Application()
        app.router.add_post("/v1/systemone", fragmented)
        app.router.add_post("/upstream/clef-flash-q4/v1/systemone", fragmented)
        server = TestServer(app)
        await server.start_server()
        self.addAsyncCleanup(server.close)
        url = str(server.make_url("")).rstrip("/")
        settings = JevSettings(api_key="test", clm_api_key="test", clm_url=url, router_api_key="test", router_url=url)
        for backend, call in ((LocalCLMBackend(settings), lambda backend: backend.evaluate({})), (RouterClefBackend(settings), lambda backend: backend.evaluate("clef-flash-q4", {"questions": {"q": QUESTION}}))):
            release.clear(); first_chunk.clear()
            self.assertIsNone(backend.timeout.total)
            self.assertIsNone(backend.timeout.sock_read)
            task = asyncio.create_task(call(backend))
            await first_chunk.wait()
            try:
                done, _ = await asyncio.wait([task], timeout=.05)
                self.assertFalse(done, "a partial body must not be parsed")
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)
            self.assertEqual((await task)["answers"]["q"]["noul"], .9)

    async def test_clm_overload_keeps_status_retry_and_redacts_body(self):
        async def busy(request):
            return web.json_response({"error": {"message": "secret detail", "code": "model_busy"}}, status=503, headers={"Retry-After": "7"})
        app = web.Application(); app.router.add_post("/v1/systemone", busy)
        server = TestServer(app); await server.start_server(); self.addAsyncCleanup(server.close)
        settings = JevSettings(api_key="test", clm_api_key="test", clm_url=str(server.make_url("")).rstrip("/"))
        client = await self.client(create_jev_app(settings, FakeJevBackend(), clm_backend=LocalCLMBackend(settings)))
        response = await client.post("/v1/systemone", headers=AUTH, json={"model": "clm-latest", "state": "x", "questions": {"q": QUESTION}})
        self.assertEqual(response.status, 503)
        self.assertEqual(response.headers["Retry-After"], "7")
        self.assertNotIn("secret", await response.text())

    async def test_decision_connection_loss_is_retryable_without_leaking_details(self):
        async def disconnect(request):
            request.transport.close()
            return web.Response()
        app = web.Application()
        app.router.add_post("/v1/systemone", disconnect)
        app.router.add_post("/upstream/clef-flash-q4/v1/systemone", disconnect)
        server = TestServer(app); await server.start_server(); self.addAsyncCleanup(server.close)
        url = str(server.make_url("")).rstrip("/")
        settings = JevSettings(api_key="test", clm_api_key="test", clm_url=url, router_api_key="test", router_url=url)
        client = await self.client(create_jev_app(settings))
        for model in ("clm-latest", "clef-flash-q4"):
            response = await client.post("/v1/systemone", headers=AUTH, json={"model": model, "state": "x", "questions": {"q": QUESTION}})
            self.assertEqual(response.status, 503)
            self.assertEqual(response.headers["Retry-After"], "1")
            self.assertEqual((await response.json())["error"]["code"], "upstream_connection_failed")

    async def test_actual_backend_transport_timeouts_are_504(self):
        _, backend = await self.comfy()
        settings = JevSettings(api_key="test", clm_api_key="test", router_api_key="test")
        client = Mock()
        client.post.side_effect = TimeoutError("private transport detail")
        client.request.side_effect = TimeoutError("private transport detail")
        context = AsyncMock()
        context.__aenter__.return_value = client
        calls = [
            lambda: LocalJevBackend(settings).decide("x", QUESTION),
            lambda: LocalCLMBackend(settings).evaluate({}),
            lambda: RouterClefBackend(settings).evaluate("clef-flash-q4", {"questions": {"q": QUESTION}}),
            lambda: backend._json("GET", "/queue"),
            lambda: backend.edit(normalize_request({"prompt": "x", "size": "512x512"}, edit=True), [("reference.png", png((16, 16)), "image/png")]),
        ]
        with patch.object(aiohttp, "ClientSession", return_value=context), patch.object(backend, "_client", return_value=client):
            for call in calls:
                with self.assertRaises(UpstreamError) as error:
                    await call()
                self.assertEqual(error.exception.status, 504)
                self.assertEqual(error.exception.code, "transport_timeout")
                self.assertNotIn("private", str(error.exception))

    async def test_slow_response_writer_keeps_its_frontend_slot(self):
        app = create_image_app(ImageSettings(api_key="test", ingress_limit=1), FakeImageBackend())
        client = await self.client(app)
        started, release = asyncio.Event(), asyncio.Event()
        original = web.Response.write_eof
        blocked = None
        async def write_eof(response, data=b""):
            nonlocal blocked
            if blocked is None and response.status == 200 and response._req is not None and response._req.app is app:
                blocked = response
                started.set()
                await release.wait()
            return await original(response, data)
        with patch.object(web.Response, "write_eof", new=write_eof):
            first = asyncio.create_task(client.post("/v1/images/generations", headers=AUTH, json={"prompt": "x"}))
            try:
                await asyncio.wait_for(started.wait(), 2)
                second = await client.post("/v1/images/generations", headers=AUTH, json={"prompt": "x"})
                self.assertEqual(second.status, 429)
                await second.read()
            finally:
                release.set()
                response = await first
                await response.read()

    async def test_jev_http_cancellation_closes_connection_without_worker_threads(self):
        started, release = asyncio.Event(), asyncio.Event()
        transports = []
        async def tokenize(request):
            return web.json_response({"tokens": [1, 2]})
        async def completion(request):
            transports.append(request.transport)
            started.set()
            await release.wait()
            return web.json_response({})
        app = web.Application(); app.router.add_post("/tokenize", tokenize); app.router.add_post("/completion", completion)
        server = TestServer(app); await server.start_server(); self.addAsyncCleanup(server.close)
        backend = LocalJevBackend(JevSettings(api_key="test", llama_url=str(server.make_url("")).rstrip("/")))
        with patch.object(asyncio, "to_thread", side_effect=AssertionError("Jev inference must be cancellable async HTTP")):
            task = asyncio.create_task(backend.decide("hello", QUESTION))
            await started.wait()
            try:
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                for _ in range(50):
                    if transports[0].is_closing():
                        break
                    await asyncio.sleep(.005)
                self.assertTrue(transports[0].is_closing())
            finally:
                release.set()

    async def test_native_jev_probability_and_usage_contract(self):
        async def tokenize(request):
            return web.json_response({"tokens": [1, 2]})
        async def completion(request):
            payload = await request.json()
            self.assertEqual(payload["n_predict"], 1)
            return web.json_response({"tokens_evaluated": 5, "completion_probabilities": [{"top_logprobs": [{"token": "A", "logprob": 0}, {"token": "B", "logprob": -2}]}]})
        app = web.Application(); app.router.add_post("/tokenize", tokenize); app.router.add_post("/completion", completion)
        server = TestServer(app); await server.start_server(); self.addAsyncCleanup(server.close)
        settings = JevSettings(api_key="test", llama_url=str(server.make_url("")).rstrip("/"))
        client = await self.client(create_jev_app(settings))
        response = await client.post("/v1/systemone", headers=AUTH, json={"state": "hello", "questions": {"q": QUESTION}})
        self.assertEqual(response.status, 200)
        body = await response.json()
        self.assertEqual(body["usage"]["input_tokens"], 5)
        self.assertAlmostEqual(body["answers"]["q"]["noul"], 1 / (1 + math.exp(-2 / 1.22)))

    async def test_async_choice_math_preserves_normalized_probabilities(self):
        async def uniform(texts):
            return [1 / len(texts)] * len(texts)
        for count in (2, 16, 33, 256):
            values = await spread(uniform, [str(index) for index in range(count)], .93)
            self.assertEqual(len(values), count)
            self.assertAlmostEqual(sum(values), 1)
            if count == 33:
                # Frozen 6b7d424 math: the hierarchical selection is not uniform
                # for uneven groups; the async conversion must preserve it.
                a, b, c = .03055301847710799, .03438925007470106, .028267121166063905
                expected = [a] * 6 + [b] * 5 + [a] * 5 + [c] * 6 + [a] * 5 + [c] * 6
            else:
                expected = [1 / count] * count
            for value, original in zip(values, expected, strict=True):
                self.assertAlmostEqual(value, original)

    async def test_wrong_image_dimensions_are_rejected_before_decoding(self):
        encoded = png((16, 16))
        with patch.object(PngImagePlugin.PngImageFile, "load", side_effect=AssertionError("wrong image must not be decoded")) as decoder:
            with self.assertRaisesRegex(UpstreamError, "incorrect dimensions"):
                _normalize_output(encoded, {"size": "512x512", "background": "auto"})
            decoder.assert_not_called()

    async def test_image_job_can_outlive_old_timeout_and_cleans_history_files(self):
        fixture, backend = await self.comfy(request_timeout_s=.001)
        request = normalize_request({"prompt": "x", "size": "512x512"}, edit=False)
        task = asyncio.create_task(backend.generate(request))
        await fixture.submitted.wait()
        try:
            done, _ = await asyncio.wait([task], timeout=.05)
            self.assertFalse(done)
            self.assertTrue(fixture.pre_recorded)
            owners = records(self.root / "requests", "comfyui")
            self.assertEqual(len(owners), 1)
            self.assertEqual(
                fixture.prompt_headers["X-Redqueen-Request-ID"], owners[0]["id"]
            )
            self.assertEqual(
                fixture.prompt_headers["X-Redqueen-Job-ID"], owners[0]["job_id"]
            )
        finally:
            fixture.complete.set()
        self.assertTrue((await task).startswith(b"\x89PNG"))
        self.assertEqual(records(self.root / "requests", "comfyui"), [])
        self.assertEqual(len(fixture.deleted_history), 1)
        self.assertEqual(list((self.root / "state/output/api").glob("*")), [])

    async def test_malformed_history_cancels_only_owned_job_and_cleans_partial_files(self):
        fixture, backend = await self.comfy()
        fixture.malformed_history = True
        fixture.partial_output = True
        fixture.jobs["foreign"] = {"client_id": "foreign-client", "request_id": "foreign", "running": True}
        foreign = self.root / "state/output/foreign.png"; foreign.write_bytes(b"keep")
        with self.assertRaises(UpstreamError) as error:
            await backend.generate(normalize_request({"prompt": "x", "size": "512x512"}, edit=False))
        self.assertEqual(error.exception.status, 502)
        self.assertEqual(len(fixture.interrupted), 1)
        self.assertNotEqual(fixture.interrupted[0], "foreign")
        self.assertTrue(fixture.jobs["foreign"]["running"])
        self.assertTrue(foreign.exists())
        self.assertEqual(records(self.root / "requests", "comfyui"), [])
        self.assertEqual(list((self.root / "state/output/api").glob("*")), [])

    async def test_malformed_prompt_acknowledgment_still_cancels_the_recorded_job(self):
        fixture, backend = await self.comfy()
        fixture.malformed_prompt = True
        fixture.partial_output = True
        with self.assertRaises(UpstreamError) as error:
            await backend.generate(normalize_request({"prompt": "x", "size": "512x512"}, edit=False))
        self.assertEqual(error.exception.status, 502)
        self.assertTrue(fixture.pre_recorded)
        self.assertEqual(len(fixture.interrupted), 1)
        self.assertEqual(records(self.root / "requests", "comfyui"), [])
        self.assertEqual(list((self.root / "state/output/api").glob("*")), [])

    async def test_malformed_upload_cleans_prerecorded_file_before_submission(self):
        fixture, backend = await self.comfy()
        fixture.malformed_upload = True
        with self.assertRaises(UpstreamError) as error:
            await backend.edit(normalize_request({"prompt": "x", "size": "512x512"}, edit=True), [("reference.png", png((16, 16)), "image/png")])
        self.assertEqual(error.exception.status, 502)
        self.assertFalse(fixture.submitted.is_set())
        self.assertEqual(records(self.root / "requests", "comfyui"), [])
        self.assertEqual(list((self.root / "state/input/api").glob("*")), [])

    async def test_upload_acknowledgment_loss_preserves_manifest_until_backend_replacement(self):
        fixture, backend = await self.comfy(upload_timeout_s=.03)
        fixture.allow_upload_response.clear()
        try:
            with self.assertRaises(UpstreamError):
                await backend.edit(normalize_request({"prompt": "x", "size": "512x512"}, edit=True), [("reference.png", png((16, 16)), "image/png")])
            manifests = records(self.root / "requests", "comfyui")
            self.assertEqual(len(manifests), 1)
            self.assertTrue(manifests[0]["upload_pending"])
            self.assertEqual(manifests[0]["phase"], "abandoned")
            with self.assertRaises(AdmissionError):
                await backend.reconcile("generation-one")
            self.assertTrue(list((self.root / "state/input/api").glob("*/*")))
        finally:
            fixture.allow_upload_response.set()
        await backend.reconcile("generation-two")
        self.assertEqual(records(self.root / "requests", "comfyui"), [])
        self.assertEqual(list((self.root / "state/input/api").glob("*")), [])

    async def test_cleanup_timeout_retains_owned_work_for_retry(self):
        fixture, backend = await self.comfy()
        backend.settings = ImageSettings(**{**backend.settings.__dict__, "cleanup_timeout_s": .03})
        fixture.allow_interrupt.clear()
        fixture.malformed_history = True
        fixture.partial_output = True
        try:
            with self.assertRaises(UpstreamError):
                await backend.generate(normalize_request({"prompt": "x", "size": "512x512"}, edit=False))
            manifests = records(self.root / "requests", "comfyui")
            self.assertEqual(len(manifests), 1)
            self.assertEqual(manifests[0]["phase"], "abandoned")
            self.assertTrue(backend.has_abandoned())
            self.assertTrue(list((self.root / "state/output/api").glob("*/*")))
        finally:
            fixture.allow_interrupt.set()
        fixture.malformed_history = False
        await backend.reconcile("generation-one")
        self.assertEqual(records(self.root / "requests", "comfyui"), [])

    async def test_image_response_byte_limit_is_enforced(self):
        fixture, backend = await self.comfy()
        fixture.complete.set()
        with patch.object(image_api, "MAX_IMAGE_BYTES", len(fixture.image) - 1):
            with self.assertRaises(UpstreamError) as error:
                await backend.generate(normalize_request({"prompt": "x", "size": "512x512"}, edit=False))
        self.assertEqual(error.exception.status, 502)
        self.assertEqual(records(self.root / "requests", "comfyui"), [])

    def owner(self, **fields):
        request_id = uuid.uuid4().hex
        return OwnedRequest(self.root / "requests", "comfyui", request_id, {"kind": "comfyui_job", "job_id": str(uuid.uuid4()), "client_id": str(uuid.uuid4()), "backend_generation": "generation-one", "submitted": True, "acknowledged": False, **fields}).begin()

    async def test_abandoned_work_reconciles_but_live_owner_and_unowned_job_survive(self):
        fixture, backend = await self.comfy()
        dead = self.owner(acknowledged=True); dead.detach()
        live = self.owner(acknowledged=True)
        for owner in (dead, live):
            fixture.jobs[owner.record["job_id"]] = {"client_id": owner.record["client_id"], "request_id": owner.id, "running": True}
        fixture.jobs["foreign"] = {"client_id": "foreign-client", "request_id": "foreign", "running": True}
        leases = self.root / "leases"; leases.mkdir()
        lock = ResourceLock(leases, "comfyui", exclusive=True); self.assertTrue(lock.acquire())
        try:
            await backend.reconcile("generation-one")
            probe = ResourceLock(leases, "comfyui")
            self.assertFalse(probe.acquire())
            probe.close()
        finally:
            lock.close()
        self.assertEqual(fixture.interrupted, [dead.record["job_id"]])
        self.assertTrue(fixture.jobs[live.record["job_id"]]["running"])
        self.assertTrue(fixture.jobs["foreign"]["running"])
        self.assertEqual([item["id"] for item in records(self.root / "requests", "comfyui")], [live.id])
        live.close()

    async def test_ambiguous_submission_needs_terminal_evidence_or_new_generation(self):
        fixture, backend = await self.comfy()
        owner = self.owner(); owner.detach()
        owned_file = self.root / "state/output/api" / owner.id / "partial.png"
        owned_file.parent.mkdir(parents=True); owned_file.write_bytes(b"partial")
        with self.assertRaises(AdmissionError) as error:
            await backend.reconcile("generation-one")
        self.assertEqual(error.exception.status, 503)
        self.assertTrue(owner.path.exists())
        self.assertTrue(owned_file.exists())
        await backend.reconcile("generation-two")
        self.assertFalse(owner.path.exists())
        self.assertFalse(owned_file.exists())
        self.assertEqual(fixture.deleted_history, [], "new-generation jobs must not be touched")

    async def test_cleanup_limits_each_filesystem_pass_and_resumes_later(self):
        _, backend = await self.comfy()
        owner = self.owner(submitted=False); owner.detach()
        directory = self.root / "state/output/api" / owner.id
        directory.mkdir(parents=True)
        for index in range(4):
            (directory / f"partial-{index}.png").write_bytes(b"partial")
        with patch.object(image_api, "MAX_CLEANUP_ENTRIES", 2):
            with self.assertRaises(AdmissionError) as error:
                await backend.reconcile("generation-one")
        self.assertEqual(error.exception.code, "cleanup_pending")
        self.assertTrue(owner.path.exists())
        self.assertEqual(len(list(directory.iterdir())), 2)
        await backend.reconcile("generation-one")
        self.assertFalse(owner.path.exists())
        self.assertFalse(directory.exists())

    async def test_cancelled_job_keeps_resource_lease_until_cleanup_finishes(self):
        fixture, backend = await self.comfy()
        fixture.allow_interrupt.clear()
        local = web.Application()
        async def admit(request):
            return web.json_response({"ok": True, "generation": "generation-one"})
        local.router.add_post("/admit", admit)
        runner = web.AppRunner(local); await runner.setup(); self.addAsyncCleanup(runner.cleanup)
        socket = self.root / "scheduler.sock"
        await web.UnixSite(runner, str(socket)).start()
        scheduler = SchedulerClient(str(socket))
        task = asyncio.create_task(scheduler.run("comfyui", lambda: backend.generate(normalize_request({"prompt": "x", "size": "512x512"}, edit=False))))
        await fixture.submitted.wait()
        task.cancel()
        await fixture.interrupt_started.wait()
        probe = ResourceLock(self.root / "leases", "comfyui", exclusive=True)
        try:
            self.assertFalse(probe.acquire(), "cancellation cleanup still owns backend work")
            self.assertFalse(task.done())
        finally:
            fixture.allow_interrupt.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(probe.acquire())
        probe.close()
        self.assertFalse(any(item.get("kind") == "comfyui_job" for item in records(self.root / "requests", "comfyui")))

    async def test_disconnect_does_not_cancel_jev_sibling_cleanup_twice(self):
        started = [asyncio.Event(), asyncio.Event()]
        cleaning = [asyncio.Event(), asyncio.Event()]
        release = [asyncio.Event(), asyncio.Event()]
        interrupted = asyncio.Event()
        class CancellableBackend(FakeJevBackend):
            async def decide(self, state, question):
                index = int(question["instructions"])
                started[index].set()
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    cleaning[index].set()
                    try:
                        await release[index].wait()
                    except asyncio.CancelledError:
                        interrupted.set()
                        raise
                    raise
        local = web.Application()
        async def admit(request):
            return web.json_response({"ok": True, "generation": "generation-one"})
        local.router.add_post("/admit", admit)
        runner = web.AppRunner(local); await runner.setup(); self.addAsyncCleanup(runner.cleanup)
        socket = self.root / "scheduler.sock"
        await web.UnixSite(runner, str(socket)).start()
        client = await self.client(create_jev_app(JevSettings(api_key="test", scheduler_socket=str(socket)), CancellableBackend()))
        task = asyncio.create_task(client.post("/v1/systemone", headers=AUTH, json={"state": "x", "questions": {str(index): {**QUESTION, "instructions": str(index)} for index in range(2)}}))
        waiter = asyncio.create_task(interrupted.wait())
        probe = ResourceLock(self.root / "leases", "jevk5", exclusive=True)
        try:
            await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started)), 2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            await asyncio.wait_for(asyncio.gather(*(event.wait() for event in cleaning)), 2)
            release[0].set()
            done, _ = await asyncio.wait([waiter], timeout=.05)
            self.assertFalse(done, "cleanup must survive a sibling finishing cancellation first")
            self.assertFalse(probe.acquire(), "the remaining question still owns its cleanup lease")
        finally:
            for event in release:
                event.set()
            waiter.cancel()
            probe.close()

    async def test_disk_pressure_blocks_submission_without_creating_work(self):
        fixture, backend = await self.comfy()
        backend.settings = ImageSettings(**{**backend.settings.__dict__, "minimum_free_bytes": 1024})
        with patch.object(image_api.shutil, "disk_usage", return_value=type("Usage", (), {"free": 0})()):
            with self.assertRaises(AdmissionError) as error:
                await backend.generate(normalize_request({"prompt": "x", "size": "512x512"}, edit=False))
        self.assertEqual(error.exception.status, 503)
        self.assertEqual(error.exception.code, "storage_pressure")
        self.assertFalse(fixture.submitted.is_set())
        self.assertEqual(records(self.root / "requests", "comfyui"), [])


if __name__ == "__main__":
    unittest.main()
