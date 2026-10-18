"""Tests for the direct ComfyUI submission maintenance barrier."""

from __future__ import annotations

import asyncio
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import asynccontextmanager

from aiohttp import web


GUARD_PATH = (
    Path(__file__).resolve().parents[1]
    / "custom_nodes/redqueen_admission_guard/__init__.py"
)
CHECK_PATH = Path(__file__).resolve().parents[1] / "check_admission_guard.py"


def load_guard():
    prompt_server = SimpleNamespace(
        app=web.Application(),
        routes=web.RouteTableDef(),
        prompt_queue=FakePromptQueue(),
    )
    server = types.ModuleType("server")
    server.PromptServer = SimpleNamespace(instance=prompt_server)
    spec = importlib.util.spec_from_file_location("test_redqueen_admission_guard", GUARD_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the ComfyUI admission guard")
    module = importlib.util.module_from_spec(spec)
    with (
        patch.dict(os.environ, {"REDQUEEN_SCHEDULER_SOCKET": "/tmp/test-scheduler.sock"}),
        patch.dict(sys.modules, {"server": server}),
    ):
        spec.loader.exec_module(module)
    return module, prompt_server


class FakePromptQueue:
    def __init__(self):
        self.running = []
        self.pending = []

    def get_current_queue_volatile(self):
        return self.running, self.pending


class FakeScheduler:
    def __init__(self):
        self.active = 0
        self.entered = 0
        self.exited = 0

    @asynccontextmanager
    async def lease(self, resource):
        if resource != "comfyui":
            raise AssertionError(f"unexpected resource: {resource}")
        self.active += 1
        self.entered += 1
        try:
            yield {}
        finally:
            self.active -= 1
            self.exited += 1


def load_check():
    spec = importlib.util.spec_from_file_location("test_check_admission_guard", CHECK_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the ComfyUI admission check")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AdmissionGuardTest(unittest.TestCase):
    def setUp(self):
        self.module, self.prompt_server = load_guard()
        self.scheduler = FakeScheduler()
        self.module.SCHEDULER = self.scheduler

    def test_install_registers_middleware_and_readiness_route(self):
        self.assertTrue(self.prompt_server.redqueen_admission_guard_installed)
        self.assertTrue(
            getattr(self.prompt_server.app.middlewares[0], "redqueen_admission_guard", False)
        )
        self.assertIn(
            "/redqueen/admission-guard",
            [route.path for route in self.prompt_server.routes],
        )

    def test_prompt_holds_shared_operation_lock_until_submission_finishes(self):
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "operations.lock"
            lock_path.touch(mode=0o600)
            middleware = self.module.prompt_admission_middleware(lock_path)
            request = SimpleNamespace(method="POST", path="/prompt")
            exclusive_was_blocked = []

            async def handler(_request):
                probe = os.open(lock_path, os.O_RDONLY)
                try:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    exclusive_was_blocked.append(True)
                finally:
                    os.close(probe)
                return web.Response(status=202)

            response = asyncio.run(middleware(request, handler))
            self.assertEqual(response.status, 202)
            self.assertEqual(exclusive_was_blocked, [True])
            probe = os.open(lock_path, os.O_RDONLY)
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(probe)

    def test_direct_prompt_holds_scheduler_lease_until_comfy_queue_drains(self):
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "operations.lock"
            lock_path.touch(mode=0o600)
            middleware = self.module.prompt_admission_middleware(lock_path)

            async def handler(_request):
                self.prompt_server.prompt_queue.running.append({"prompt_id": "job"})
                return web.json_response({"prompt_id": "job"})

            async def scenario():
                response = await middleware(
                    SimpleNamespace(method="POST", path="/prompt"), handler
                )
                self.assertEqual(response.status, 200)
                self.assertEqual(self.scheduler.active, 1)
                self.assertEqual(self.scheduler.exited, 0)
                self.prompt_server.prompt_queue.running.clear()
                for _ in range(20):
                    if self.scheduler.active == 0:
                        break
                    await asyncio.sleep(0.025)
                self.assertEqual(self.scheduler.active, 0)
                self.assertEqual(self.scheduler.entered, 1)
                self.assertEqual(self.scheduler.exited, 1)
                await asyncio.sleep(0)

            asyncio.run(scenario())

    def test_owned_image_prompt_uses_its_existing_broker_lease(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock_path = root / "operations.lock"
            lock_path.touch(mode=0o600)
            request_id = "a" * 32
            job_id = "12345678-1234-1234-1234-123456789abc"
            record = {
                "version": 1,
                "id": request_id,
                "request_id": request_id,
                "resource": "comfyui",
                "boot_id": self.module.boot_id(),
                "pid": os.getpid(),
                "process_start": self.module.process_start(os.getpid()),
                "phase": "active",
                "kind": "comfyui_job",
                "submitted": True,
                "job_id": job_id,
            }
            manifest = root / f"{request_id}.json"
            manifest.write_text(json.dumps(record))
            manifest.chmod(0o600)
            middleware = self.module.prompt_admission_middleware(
                lock_path,
                request_directory=root,
            )
            called = []

            async def handler(_request):
                called.append(True)
                return web.Response(status=202)

            response = asyncio.run(
                middleware(
                    SimpleNamespace(
                        method="POST",
                        path="/prompt",
                        headers={
                            "X-Redqueen-Request-ID": request_id,
                            "X-Redqueen-Job-ID": job_id,
                        },
                    ),
                    handler,
                )
            )

            self.assertEqual(response.status, 202)
            self.assertEqual(called, [True])
            self.assertEqual(self.scheduler.entered, 0)

    def test_invalid_internal_headers_fail_closed_without_a_nested_lease(self):
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "operations.lock"
            lock_path.touch(mode=0o600)
            called = []

            async def handler(_request):
                called.append(True)
                return web.Response(status=202)

            response = asyncio.run(
                self.module.prompt_admission_middleware(lock_path)(
                    SimpleNamespace(
                        method="POST",
                        path="/prompt",
                        headers={"X-Redqueen-Request-ID": "not-an-owned-request"},
                    ),
                    handler,
                )
            )

            self.assertEqual(response.status, 503)
            self.assertEqual(called, [])
            self.assertEqual(self.scheduler.entered, 0)

    def test_prompt_handler_oserror_propagates_and_releases_operation_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "operations.lock"
            lock_path.touch(mode=0o600)

            async def handler(_request):
                raise OSError("ComfyUI prompt storage failed")

            with self.assertRaisesRegex(OSError, "prompt storage failed"):
                asyncio.run(
                    self.module.prompt_admission_middleware(lock_path)(
                        SimpleNamespace(method="POST", path="/prompt"), handler
                    )
                )

            probe = os.open(lock_path, os.O_RDONLY)
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(probe)

    def test_maintenance_lock_rejects_direct_prompt(self):
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "operations.lock"
            lock_path.touch(mode=0o600)
            held = os.open(lock_path, os.O_RDONLY)
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                called = []

                async def handler(_request):
                    called.append(True)
                    return web.Response(status=202)

                response = asyncio.run(
                    self.module.prompt_admission_middleware(lock_path)(
                        SimpleNamespace(method="POST", path="/api/prompt"), handler
                    )
                )
                self.assertEqual(response.status, 503)
                self.assertEqual(called, [])
            finally:
                os.close(held)

    def test_non_prompt_routes_remain_available_during_maintenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "operations.lock"
            lock_path.touch(mode=0o600)
            held = os.open(lock_path, os.O_RDONLY)
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                async def handler(_request):
                    return web.Response(status=200)

                response = asyncio.run(
                    self.module.prompt_admission_middleware(lock_path)(
                        SimpleNamespace(method="GET", path="/queue"), handler
                    )
                )
                self.assertEqual(response.status, 200)
            finally:
                os.close(held)

    def test_missing_lock_file_fails_closed_for_prompt(self):
        with tempfile.TemporaryDirectory() as temporary:
            middleware = self.module.prompt_admission_middleware(
                Path(temporary) / "missing.lock"
            )

            async def handler(_request):
                self.fail("prompt handler ran without an operations lock")

            response = asyncio.run(
                middleware(SimpleNamespace(method="POST", path="/prompt"), handler)
            )
            self.assertEqual(response.status, 503)

    def test_pending_restore_marker_rejects_prompts_after_a_crash(self):
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "operations.lock"
            lock_path.touch(mode=0o600)
            restore_marker = Path(temporary) / "comfyui-restore.json"
            restore_marker.touch(mode=0o600)
            middleware = self.module.prompt_admission_middleware(
                lock_path, restore_marker
            )

            async def handler(_request):
                self.fail("prompt handler ran while a durable restore is pending")

            response = asyncio.run(
                middleware(SimpleNamespace(method="POST", path="/prompt"), handler)
            )
            self.assertEqual(response.status, 503)


class AdmissionCheckTest(unittest.TestCase):
    def test_readiness_requires_the_expected_guard_identity(self):
        module = load_check()
        module.ready = lambda: False
        with (
            patch.object(module.time, "monotonic", side_effect=[0, 1]),
            patch.object(module.time, "sleep"),
            self.assertRaisesRegex(RuntimeError, "failed to become ready"),
        ):
            module.wait_ready(timeout=0)

    def test_readiness_returns_only_after_the_guard_is_healthy(self):
        module = load_check()
        with patch.object(module, "ready", side_effect=[False, True]), patch.object(
            module.time, "sleep"
        ):
            module.wait_ready(timeout=1)


if __name__ == "__main__":
    unittest.main()
