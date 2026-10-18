"""Local admission leases; flock keeps active work protected across scheduler restarts."""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TypeVar

import aiohttp

from .common import AdmissionError, decode_json, read_limited, retry_after_value
from .ownership import OwnedRequest

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


async def run_thread_shielded(operation: Callable[..., T], /, *args, **kwargs) -> T:
    """Do not let caller cancellation outlive synchronous work in a thread."""
    task = asyncio.create_task(asyncio.to_thread(operation, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await finish_shielded(task)
        except asyncio.CancelledError:
            pass
        raise


class ResourceLock:
    def __init__(self, directory: Path, resource: str, *, exclusive: bool = False):
        if not resource or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-._" for c in resource):
            raise ValueError("invalid resource name")
        self.path = directory / resource
        self.exclusive = exclusive
        self.fd: int | None = None

    def acquire(self) -> bool:
        if self.fd is not None:
            raise RuntimeError("resource lease is already acquired")
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, (fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return False
        except BaseException:
            os.close(fd)
            raise
        self.fd = fd
        return True

    def close(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


class SerializedExecution:
    """FIFO serialize GPU work across broker and adapter processes on this host."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.queue = directory / "execution-queue"

    async def _new_ticket(self) -> tuple[Path, int]:
        lock = ResourceLock(self.queue, "tickets", exclusive=True)
        while not lock.acquire():
            await asyncio.sleep(0.025)
        path: Path | None = None
        fd: int | None = None
        try:
            sequence_path = self.queue / "sequence"
            sequence_fd = os.open(
                sequence_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600
            )
            try:
                raw = os.read(sequence_fd, 64).decode("ascii").strip()
                sequence = int(raw or "0") + 1
                os.lseek(sequence_fd, 0, os.SEEK_SET)
                os.ftruncate(sequence_fd, 0)
                os.write(sequence_fd, str(sequence).encode("ascii"))
                os.fsync(sequence_fd)
            finally:
                os.close(sequence_fd)
            path = self.queue / f"{sequence:020d}-{uuid.uuid4().hex}.ticket"
            fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            return path, fd
        except BaseException:
            if fd is not None:
                os.close(fd)
            if path is not None:
                path.unlink(missing_ok=True)
            raise
        finally:
            lock.close()

    def _is_head(self, ticket: Path) -> bool:
        queue_lock = ResourceLock(self.queue, "tickets", exclusive=True)
        if not queue_lock.acquire():
            return False
        try:
            for entry in sorted(self.queue.glob("*.ticket")):
                if entry == ticket:
                    return True
                fd = os.open(entry, os.O_RDWR | os.O_NOFOLLOW)
                try:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        return False
                    entry.unlink(missing_ok=True)
                finally:
                    os.close(fd)
            return False
        finally:
            queue_lock.close()

    @asynccontextmanager
    async def hold(self):
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.queue.mkdir(mode=0o700, exist_ok=True)
        ticket, ticket_fd = await self._new_ticket()
        execution = ResourceLock(self.directory, "global-execution", exclusive=True)
        try:
            while not self._is_head(ticket) or not execution.acquire():
                await asyncio.sleep(0.025)
            ticket.unlink(missing_ok=True)
            yield
        finally:
            execution.close()
            os.close(ticket_fd)
            ticket.unlink(missing_ok=True)

    @asynccontextmanager
    async def hold_maintenance(self):
        """Hold a FIFO execution window and the host operations lock.

        Standalone GPU work must not bypass queued inference or overlap host
        deployment/recovery. Operations-lock contention fails closed instead
        of making later inference wait behind maintenance work.
        """
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.queue.mkdir(mode=0o700, exist_ok=True)
        ticket, ticket_fd = await self._new_ticket()
        execution = ResourceLock(self.directory, "global-execution", exclusive=True)
        operations = ResourceLock(self.directory.parent, "operations.lock", exclusive=True)
        try:
            while not self._is_head(ticket):
                await asyncio.sleep(0.025)
            while True:
                if not operations.acquire():
                    raise RuntimeError("host operations are busy; refusing standalone GPU work")
                if execution.acquire():
                    break
                operations.close()
                await asyncio.sleep(0.025)
            ticket.unlink(missing_ok=True)
            yield
        finally:
            execution.close()
            operations.close()
            os.close(ticket_fd)
            ticket.unlink(missing_ok=True)


class SchedulerClient:
    def __init__(self, socket_path: str = "", *, timeout: float | None = None):
        self.socket_path = socket_path
        # Retained argument for API compatibility. Admission may include an
        # arbitrarily long healthy model load; it has no response deadline.
        self.timeout = None
        self.slots: dict[str, asyncio.Semaphore] = {}
        self.generations: dict[str, str | None] = {}
        self.execution = (
            SerializedExecution(Path(socket_path).parent / "leases")
            if socket_path
            else None
        )

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
    async def lease(self, resource: str, *, exclusive: bool = False, wait_for_idle: bool = True):
        if not self.socket_path:
            yield {}
            return
        assert self.execution is not None
        async with self.execution.hold():
            async with self._lease(
                resource, exclusive=exclusive, wait_for_idle=wait_for_idle
            ) as admission:
                yield admission

    @asynccontextmanager
    async def _lease(
        self, resource: str, *, exclusive: bool = False, wait_for_idle: bool = True
    ):
        directory = Path(self.socket_path).parent
        (directory / "leases").mkdir(mode=0o700, parents=True, exist_ok=True)
        lock = ResourceLock(directory / "leases", resource, exclusive=exclusive)
        owned = None
        admitted = False
        try:
            try:
                if not lock.acquire():
                    raise AdmissionError(503, "model is changing state; retry shortly", "model_transition")
                owned = OwnedRequest(directory / "requests", resource).begin()
                connector = aiohttp.UnixConnector(path=self.socket_path)
                async with aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=None, sock_read=None, connect=10)) as session:
                    async with session.post(
                        "http://scheduler/admit",
                        json={"resource": resource, "wait_for_idle": wait_for_idle},
                    ) as response:
                        body = decode_json(await read_limited(response, 64 * 1024, check_status=False))
                        if not isinstance(body, dict):
                            raise AdmissionError(503, "invalid scheduler admission response", "scheduler_unavailable")
                        if response.status != 200:
                            error = body.get("error", {})
                            if not isinstance(error, dict):
                                raise AdmissionError(503, "invalid scheduler admission response", "scheduler_unavailable")
                            retry = response.headers.get("Retry-After")
                            raise AdmissionError(response.status, error.get("message", "model capacity unavailable"), error.get("code", "capacity_unavailable"), retry_after=retry_after_value(retry, default=None))
                        if not isinstance(body, dict) or body.get("ok") is not True:
                            raise AdmissionError(503, "invalid scheduler admission response", "scheduler_unavailable")
                        self.generations[resource] = body.get("generation")
            except (aiohttp.ClientError, TimeoutError, OSError, UnicodeDecodeError, ValueError) as error:
                raise AdmissionError(503, "model scheduler is unavailable", "scheduler_unavailable") from error
            admitted = True
            yield body
        except BaseException:
            if owned is not None and admitted:
                owned.detach()
            elif owned is not None:
                owned.close()
            raise
        else:
            if owned is not None:
                owned.close()
        finally:
            lock.close()

    async def run(self, resource: str, operation: Callable[[], Awaitable[T]], *, concurrency: int = 1) -> T:
        if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
            raise ValueError("concurrency must be a positive integer")
        async with self.lease(resource) as admission:
            profile_limit = admission.get("concurrency_limit", concurrency)
            if isinstance(profile_limit, bool) or not isinstance(profile_limit, int) or profile_limit < 1:
                raise AdmissionError(503, "scheduler returned an invalid concurrency limit", "scheduler_unavailable")
            slot = self.slots.setdefault(
                resource, asyncio.Semaphore(min(concurrency, profile_limit))
            )
            async with slot:
                task = asyncio.create_task(operation())
                try:
                    return await asyncio.shield(task)
                except asyncio.CancelledError:
                    # Operations now use cancellable async HTTP. Drain their owned
                    # cleanup before releasing the flock; record uncertain work.
                    task.cancel()
                    try:
                        await finish_shielded(task)
                    except asyncio.CancelledError:
                        pass
                    raise
