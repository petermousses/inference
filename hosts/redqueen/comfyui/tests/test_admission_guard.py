"""Tests for the direct ComfyUI submission maintenance barrier."""

from __future__ import annotations

import asyncio
import fcntl
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from aiohttp import web


GUARD_PATH = (
    Path(__file__).resolve().parents[1]
    / "custom_nodes/redqueen_admission_guard/__init__.py"
)
CHECK_PATH = Path(__file__).resolve().parents[1] / "check_admission_guard.py"


def load_guard():
    prompt_server = SimpleNamespace(app=web.Application(), routes=web.RouteTableDef())
    server = types.ModuleType("server")
    server.PromptServer = SimpleNamespace(instance=prompt_server)
    spec = importlib.util.spec_from_file_location("test_redqueen_admission_guard", GUARD_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the ComfyUI admission guard")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"server": server}):
        spec.loader.exec_module(module)
    return module, prompt_server


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
