import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from redqueen_adapters.common import IngressGate, UpstreamError, auth_middleware, read_json, read_limited, safe_errors
from redqueen_adapters.ownership import OwnedRequest, abandoned, records


class CommonReliabilityTest(unittest.IsolatedAsyncioTestCase):
    async def client(self, app):
        client = TestClient(TestServer(app))
        await client.start_server()
        self.addAsyncCleanup(client.close)
        return client

    async def test_non_ascii_bearer_is_json_401_and_unknown_exception_is_redacted(self):
        app = web.Application(middlewares=[safe_errors, auth_middleware("test")])
        async def broken(request):
            raise TypeError("secret-body")
        app.router.add_get("/broken", broken)
        client = await self.client(app)
        response = await client.get("/broken", headers={"Authorization": "Bearer é"})
        self.assertEqual(response.status, 401)
        response = await client.get("/broken", headers={"Authorization": "Bearer test"})
        self.assertEqual(response.status, 500)
        body = await response.json()
        self.assertEqual(body["error"]["code"], "internal_error")
        self.assertNotIn("secret-body", json.dumps(body))
        self.assertIn("X-Request-ID", response.headers)

    async def test_admission_precedes_body_handler(self):
        entered, release = asyncio.Event(), asyncio.Event()
        count = 0
        app = web.Application(middlewares=[safe_errors, IngressGate(1).middleware])
        async def handler(request):
            nonlocal count
            count += 1
            entered.set()
            await release.wait()
            return web.json_response({"ok": True})
        app.router.add_post("/work", handler)
        client = await self.client(app)
        first = asyncio.create_task(client.post("/work"))
        await entered.wait()
        response = await client.post("/work")
        self.assertEqual(response.status, 429)
        self.assertEqual(count, 1)
        release.set()
        self.assertEqual((await first).status, 200)

    async def test_bounded_read_waits_for_fragmented_eof(self):
        class Content:
            async def iter_chunked(self, size):
                yield b'{"ok":'
                await asyncio.sleep(0)
                yield b'true}'
        class Response:
            content, status, headers = Content(), 200, {}
        self.assertEqual(json.loads(await read_limited(Response())), {"ok": True})
        with self.assertRaises(UpstreamError) as error:
            await read_limited(Response(), 5)
        self.assertEqual(error.exception.status, 502)

    async def test_upstream_retry_after_is_preserved_except_permanent_configuration(self):
        class Content:
            async def iter_chunked(self, size):
                yield json.dumps({"error": {"code": response_code}}).encode()
        class Response:
            content, status, headers = Content(), 503, {"Retry-After": "Fri, 16 Oct 2026 22:00:00 GMT"}
        response_code = "capacity_unavailable"
        with self.assertRaises(UpstreamError) as error:
            await read_limited(Response())
        self.assertEqual(error.exception.retry_after, Response.headers["Retry-After"])
        response_code = "configuration_unready"
        with self.assertRaises(UpstreamError) as error:
            await read_limited(Response())
        self.assertIsNone(error.exception.retry_after)

    async def test_http_405_preserves_allow(self):
        app = web.Application(middlewares=[safe_errors])
        app.router.add_get("/route", lambda request: web.json_response({}))
        client = await self.client(app)
        response = await client.post("/route")
        self.assertEqual(response.status, 405)
        self.assertIn("GET", response.headers["Allow"])
        self.assertEqual((await response.json())["error"]["code"], "method_not_allowed")

    async def test_json_shape_is_bounded_before_decode(self):
        app = web.Application(client_max_size=32 * 1024 * 1024, middlewares=[safe_errors])
        async def handler(request):
            await read_json(request)
            return web.json_response({"ok": True})
        app.router.add_post("/json", handler)
        client = await self.client(app)
        malformed_encodings = [
            '{"model":"x"}'.encode("utf-16-le"),
            '{"model":"x"}'.encode("utf-16-be"),
            '{"model":"x"}'.encode("utf-32-le"),
            '{"model":"x"}'.encode("utf-32-be"),
        ]
        for raw in (b"[" * 65 + b"0" + b"]" * 65, b"[" + b'"",' * 125_001 + b'""]', *malformed_encodings):
            with patch("redqueen_adapters.common.json.loads", side_effect=AssertionError("must reject before decode")):
                response = await client.post("/json", data=raw)
            self.assertEqual(response.status, 400)
            self.assertEqual((await response.json())["error"]["code"], "invalid_request")
        for raw in (b"[" * 64 + b"0" + b"]" * 64, json.dumps({"escaped": '\\\"[]{}:,' * 100}).encode()):
            response = await client.post("/json", data=raw)
            self.assertEqual(response.status, 200)

    async def test_json_requires_strict_utf8_and_rejects_nonfinite_or_surrogate_values(self):
        app = web.Application(client_max_size=32 * 1024 * 1024, middlewares=[safe_errors])
        async def handler(request):
            await read_json(request)
            return web.json_response({"ok": True})
        app.router.add_post("/json", handler)
        client = await self.client(app)
        for raw in (b'{"x":"\\ud800"}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":"a\x00b"}'):
            response = await client.post("/json", data=raw)
            self.assertEqual(response.status, 400, raw)

    async def test_request_id_is_bounded_and_preserved_before_transmission(self):
        app = web.Application(middlewares=[safe_errors, IngressGate(1).middleware])
        async def handler(request):
            return web.json_response({"ok": True})
        app.router.add_post("/work", handler)
        client = await self.client(app)
        response = await client.post("/work", headers={"X-Request-ID": "client-123"})
        self.assertEqual(response.headers["X-Request-ID"], "client-123")
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        response = await client.post("/work", headers={"X-Request-ID": "x" * 81})
        self.assertEqual(len(response.headers["X-Request-ID"]), 32)


class OwnershipTest(unittest.TestCase):
    def test_live_ownership_never_expires_and_pid_reuse_is_abandoned(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with patch("redqueen_adapters.ownership.boot_id", return_value="boot"), patch("redqueen_adapters.ownership.process_start", return_value="start"):
                request = OwnedRequest(directory, "clm").begin()
                self.assertEqual(abandoned(directory, "clm"), [])
                self.assertNotIn("deadline", records(directory)[0])
                with patch("redqueen_adapters.ownership.process_start", return_value="new-process"):
                    self.assertEqual(abandoned(directory, "clm")[0]["id"], request.id)
                request.detach()
                self.assertEqual(abandoned(directory, "clm")[0]["phase"], "abandoned")
                request.close()
                self.assertEqual(records(directory), [])
