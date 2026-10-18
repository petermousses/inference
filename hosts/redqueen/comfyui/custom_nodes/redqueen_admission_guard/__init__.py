"""Reject direct ComfyUI prompt submissions during host maintenance."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
import stat

from aiohttp import web
from server import PromptServer


OPERATIONS_LOCK = Path("/srv/ai/model-router/run/operations.lock")
RESTORE_MARKER = Path("/srv/ai/model-router/run/comfyui-restore.json")
COMPONENT = "redqueen-prompt-admission"
VERSION = 1
PROMPT_PATHS = frozenset({"/prompt", "/api/prompt"})


def prompt_admission_middleware(
    lock_path: Path = OPERATIONS_LOCK,
    restore_marker: Path = RESTORE_MARKER,
):
    @web.middleware
    async def admission_guard(request, handler):
        if request.method != "POST" or request.path not in PROMPT_PATHS:
            return await handler(request)

        restore_descriptor = None
        try:
            restore_descriptor = os.open(
                restore_marker,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
            )
        except FileNotFoundError:
            pass
        except OSError:
            return web.json_response(
                {"error": "ComfyUI prompt admission is unavailable"},
                status=503,
                headers={"Retry-After": "1"},
            )
        else:
            try:
                info = os.fstat(restore_descriptor)
                if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid():
                    return web.json_response(
                        {"error": "ComfyUI prompt admission is unavailable"},
                        status=503,
                        headers={"Retry-After": "1"},
                    )
                return web.json_response(
                    {"error": "ComfyUI prompt admission is paused for model restore"},
                    status=503,
                    headers={"Retry-After": "1"},
                )
            finally:
                os.close(restore_descriptor)

        descriptor = None
        try:
            descriptor = os.open(
                lock_path,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
            )
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid():
                raise OSError("operations lock is not a regular ai-owned file")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                return web.json_response(
                    {"error": "ComfyUI prompt admission is paused for host maintenance"},
                    status=503,
                    headers={"Retry-After": "1"},
                )
            return await handler(request)
        except OSError:
            return web.json_response(
                {"error": "ComfyUI prompt admission is unavailable"},
                status=503,
                headers={"Retry-After": "1"},
            )
        finally:
            if descriptor is not None:
                os.close(descriptor)

    admission_guard.redqueen_admission_guard = True
    return admission_guard


def install(prompt_server) -> None:
    if prompt_server is None:
        raise RuntimeError("ComfyUI PromptServer is not initialized")
    if getattr(prompt_server, "redqueen_admission_guard_installed", False):
        raise RuntimeError("ComfyUI admission guard is already installed")
    middlewares = prompt_server.app.middlewares
    if any(getattr(item, "redqueen_admission_guard", False) for item in middlewares):
        raise RuntimeError("ComfyUI admission middleware is already installed")
    middlewares.insert(0, prompt_admission_middleware())

    @prompt_server.routes.get("/redqueen/admission-guard")
    async def admission_guard_status(_request):
        return web.json_response(
            {"component": COMPONENT, "version": VERSION, "pid": os.getpid()}
        )

    prompt_server.redqueen_admission_guard_installed = True


install(PromptServer.instance)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
