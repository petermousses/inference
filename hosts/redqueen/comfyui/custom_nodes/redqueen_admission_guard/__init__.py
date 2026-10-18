"""Serialize direct ComfyUI jobs with Redqueen's shared model scheduler."""

from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
from pathlib import Path
import re
import stat
import sys

from aiohttp import web
from server import PromptServer


_ADAPTERS_SOURCE = Path(__file__).resolve().parents[5] / "hosts/redqueen/adapters"
if _ADAPTERS_SOURCE.is_dir() and str(_ADAPTERS_SOURCE) not in sys.path:
    sys.path.insert(0, str(_ADAPTERS_SOURCE))

from redqueen_adapters.ownership import boot_id, process_start
from redqueen_adapters.scheduler_client import SchedulerClient, finish_shielded


OPERATIONS_LOCK = Path("/srv/ai/model-router/run/operations.lock")
RESTORE_MARKER = Path("/srv/ai/model-router/run/comfyui-restore.json")
SCHEDULER_SOCKET = os.environ.get("REDQUEEN_SCHEDULER_SOCKET", "")
REQUEST_DIRECTORY = (
    Path(SCHEDULER_SOCKET).parent / "requests"
    if SCHEDULER_SOCKET
    else Path("/srv/ai/model-router/run/requests")
)
SCHEDULER = SchedulerClient(SCHEDULER_SOCKET)
COMPONENT = "redqueen-prompt-admission"
VERSION = 1
PROMPT_PATHS = frozenset({"/prompt", "/api/prompt"})
REQUEST_ID = re.compile(r"[0-9a-f]{32}\Z")
JOB_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z")
MAX_REQUEST_RECORD = 64 * 1024
MONITOR_INTERVAL_S = 0.25
LOG = logging.getLogger("redqueen.comfyui.admission")
_MONITOR_TASKS: set[asyncio.Task] = set()


def _owned_brokered_job(request, directory: Path) -> bool | None:
    """Return None for an operator request, True for an owned API job, else False."""
    headers = getattr(request, "headers", {})
    request_id = headers.get("X-Redqueen-Request-ID")
    job_id = headers.get("X-Redqueen-Job-ID")
    if request_id is None and job_id is None:
        return None
    if (
        not isinstance(request_id, str)
        or not REQUEST_ID.fullmatch(request_id)
        or not isinstance(job_id, str)
        or not JOB_ID.fullmatch(job_id)
    ):
        return False

    descriptor = None
    try:
        descriptor = os.open(
            directory / f"{request_id}.json",
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
        )
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_nlink != 1
            or info.st_size > MAX_REQUEST_RECORD
        ):
            return False
        raw = os.read(descriptor, MAX_REQUEST_RECORD + 1)
    except OSError:
        return False
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if len(raw) > MAX_REQUEST_RECORD:
        return False
    try:
        record = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(record, dict):
        return False
    pid = record.get("pid")
    start = record.get("process_start")
    if (
        isinstance(record.get("version"), bool)
        or record.get("version") != 1
        or record.get("id") != request_id
        or record.get("request_id") != request_id
        or record.get("resource") != "comfyui"
        or record.get("kind") != "comfyui_job"
        or record.get("phase") != "active"
        or record.get("submitted") is not True
        or record.get("job_id") != job_id
        or record.get("boot_id") != boot_id()
        or isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid < 1
        or not isinstance(start, str)
        or not start
    ):
        return False
    try:
        return process_start(pid) == start
    except (OSError, ValueError, IndexError):
        return False


def _queue_busy(prompt_queue) -> bool | None:
    """Unknown queue state must retain the reservation."""
    snapshot = getattr(prompt_queue, "get_current_queue_volatile", None)
    if not callable(snapshot):
        return None
    try:
        state = snapshot()
    except Exception:
        return None
    if (
        not isinstance(state, (tuple, list))
        or len(state) != 2
        or not isinstance(state[0], (tuple, list))
        or not isinstance(state[1], (tuple, list))
    ):
        return None
    return bool(state[0] or state[1])


async def _exit_lease(lease) -> None:
    cleanup = asyncio.create_task(lease.__aexit__(None, None, None))
    await finish_shielded(cleanup)


async def _hold_until_queue_idle(lease, prompt_queue) -> None:
    try:
        while _queue_busy(prompt_queue) is not False:
            await asyncio.sleep(MONITOR_INTERVAL_S)
    finally:
        await _exit_lease(lease)


def _monitor_done(task: asyncio.Task) -> None:
    _MONITOR_TASKS.discard(task)
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except Exception:
        LOG.exception("failed to release a direct ComfyUI reservation")


def _transfer_to_queue_monitor(lease, prompt_queue) -> None:
    task = asyncio.create_task(_hold_until_queue_idle(lease, prompt_queue))
    _MONITOR_TASKS.add(task)
    task.add_done_callback(_monitor_done)


def prompt_admission_middleware(
    lock_path: Path = OPERATIONS_LOCK,
    restore_marker: Path = RESTORE_MARKER,
    *,
    scheduler_client=None,
    prompt_queue=None,
    request_directory: Path = REQUEST_DIRECTORY,
):
    @web.middleware
    async def admission_guard(request, handler):
        if request.method != "POST" or request.path not in PROMPT_PATHS:
            return await handler(request)

        prompt_queue_for_request = (
            prompt_queue
            if prompt_queue is not None
            else getattr(PromptServer.instance, "prompt_queue", None)
        )
        brokered = _owned_brokered_job(request, request_directory)
        if brokered is False:
            return web.json_response(
                {"error": "ComfyUI broker request ownership is invalid"},
                status=503,
                headers={"Retry-After": "1"},
            )

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

        lease = None
        if brokered is None:
            try:
                scheduler = SCHEDULER if scheduler_client is None else scheduler_client
                lease = scheduler.lease("comfyui")
                await lease.__aenter__()
            except Exception as error:
                LOG.warning("direct ComfyUI admission failed reason=%s", type(error).__name__)
                return web.json_response(
                    {"error": "ComfyUI is unavailable through the model scheduler"},
                    status=503,
                    headers={"Retry-After": "1"},
                )

        descriptor = None
        transferred = False
        try:
            try:
                descriptor = os.open(
                    lock_path,
                    os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
                )
                info = os.fstat(descriptor)
                if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid():
                    raise OSError("operations lock is not a regular ai-owned file")
                fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                if descriptor is not None:
                    os.close(descriptor)
                    descriptor = None
                return web.json_response(
                    {"error": "ComfyUI prompt admission is paused for host maintenance"},
                    status=503,
                    headers={"Retry-After": "1"},
                )
            except OSError:
                if descriptor is not None:
                    os.close(descriptor)
                    descriptor = None
                return web.json_response(
                    {"error": "ComfyUI prompt admission is unavailable"},
                    status=503,
                    headers={"Retry-After": "1"},
                )

            try:
                response = await handler(request)
            except BaseException:
                if lease is not None and _queue_busy(prompt_queue_for_request) is not False:
                    _transfer_to_queue_monitor(lease, prompt_queue_for_request)
                    transferred = True
                raise
            if lease is not None and _queue_busy(prompt_queue_for_request) is not False:
                _transfer_to_queue_monitor(lease, prompt_queue_for_request)
                transferred = True
            return response
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if lease is not None and not transferred:
                await _exit_lease(lease)

    admission_guard.redqueen_admission_guard = True
    return admission_guard


def install(prompt_server) -> None:
    if prompt_server is None:
        raise RuntimeError("ComfyUI PromptServer is not initialized")
    if getattr(prompt_server, "redqueen_admission_guard_installed", False):
        raise RuntimeError("ComfyUI admission guard is already installed")
    if not SCHEDULER.socket_path:
        raise RuntimeError("Redqueen scheduler socket is not configured")
    queue = getattr(prompt_server, "prompt_queue", None)
    if queue is None or not callable(getattr(queue, "get_current_queue_volatile", None)):
        raise RuntimeError("ComfyUI prompt queue inspection is unavailable")
    middlewares = prompt_server.app.middlewares
    if any(getattr(item, "redqueen_admission_guard", False) for item in middlewares):
        raise RuntimeError("ComfyUI admission middleware is already installed")
    middlewares.insert(0, prompt_admission_middleware(prompt_queue=queue))

    @prompt_server.routes.get("/redqueen/admission-guard")
    async def admission_guard_status(_request):
        return web.json_response(
            {"component": COMPONENT, "version": VERSION, "pid": os.getpid()}
        )

    prompt_server.redqueen_admission_guard_installed = True


install(PromptServer.instance)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
