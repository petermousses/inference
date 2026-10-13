"""Local admission leases; flock keeps active work protected across scheduler restarts."""

from __future__ import annotations

import asyncio
import fcntl
import os
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TypeVar

import aiohttp

from .common import AdmissionError

T = TypeVar("T")


async def finish_shielded(task: asyncio.Task[T]) -> T:
    """Drain owned work before propagating even repeated caller cancellation."""
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.cancelled():
                raise
            cancelled = True
        except Exception:
            if cancelled:
                raise asyncio.CancelledError from None
            raise
        else:
            if cancelled:
                raise asyncio.CancelledError
            return result


class ResourceLock:
    def __init__(self, directory: Path, resource: str, *, exclusive: bool = False):
        if not resource or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-._" for c in resource):
            raise ValueError("invalid resource name")
        self.path = directory / resource
        self.exclusive = exclusive
        self.fd: int | None = None

    def acquire(self) -> bool:
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, (fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return False
        self.fd = fd
        return True

    def close(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


class SchedulerClient:
    def __init__(self, socket_path: str = "", *, timeout: float = 135):
        self.socket_path = socket_path
        self.timeout = timeout
        self.slots: dict[str, asyncio.Semaphore] = {}

    async def ready(self) -> bool:
        if not self.socket_path:
            return True
        try:
            async with aiohttp.ClientSession(connector=aiohttp.UnixConnector(path=self.socket_path), timeout=aiohttp.ClientTimeout(total=3)) as session:
                async with session.get("http://scheduler/readyz") as response:
                    return response.status == 200
        except (aiohttp.ClientError, TimeoutError, OSError):
            return False

    @asynccontextmanager
    async def lease(self, resource: str):
        if not self.socket_path:
            yield
            return
        lock = ResourceLock(Path(self.socket_path).parent / "leases", resource)
        try:
            if not lock.acquire():
                raise AdmissionError(503, "model is changing state; retry shortly", "model_transition")
            connector = aiohttp.UnixConnector(path=self.socket_path)
            async with aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post("http://scheduler/admit", json={"resource": resource}) as response:
                    body = await response.json()
                    if response.status != 200:
                        error = body.get("error", {})
                        raise AdmissionError(response.status, error.get("message", "model capacity unavailable"), error.get("code", "capacity_unavailable"))
            yield
        except (aiohttp.ClientError, TimeoutError, OSError) as error:
            raise AdmissionError(503, "model scheduler is unavailable", "scheduler_unavailable") from error
        finally:
            lock.close()

    async def run(self, resource: str, operation: Callable[[], Awaitable[T]], *, concurrency: int = 1) -> T:
        slot = self.slots.setdefault(resource, asyncio.Semaphore(concurrency))
        if slot.locked():
            raise AdmissionError(429, "model is serving its maximum concurrent requests", "model_busy")
        async with slot, self.lease(resource):
            # A cancelled urllib/to_thread call can keep issuing GPU work. Keep
            # its lease until the entire operation, including worker threads, ends.
            task = asyncio.create_task(operation())
            return await finish_shielded(task)
