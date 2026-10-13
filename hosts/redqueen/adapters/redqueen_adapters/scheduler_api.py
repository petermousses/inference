"""Memory admission and idle-only LRU eviction for Redqueen's inference backends."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import aiohttp
from aiohttp import web

from .common import AdmissionError, admission_response, auth_middleware, safe_errors
from .memory_profile import MIB, verify_profiles
from .scheduler_client import ResourceLock, finish_shielded

LOG = logging.getLogger("redqueen.scheduler")
HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade"}
MODEL_ENDPOINTS = {"/v1/chat/completions", "/v1/completions", "/v1/responses", "/v1/embeddings", "/completion", "/tokenize", "/detokenize"}


@dataclass(frozen=True)
class Memory:
    gpu: int
    host: int

    def __add__(self, other: Memory) -> Memory:
        return Memory(self.gpu + other.gpu, self.host + other.host)

    def __sub__(self, other: Memory) -> Memory:
        return Memory(self.gpu - other.gpu, self.host - other.host)

    def fits(self, limit: Memory) -> bool:
        return self.gpu <= limit.gpu and self.host <= limit.host


def footprint(spec: dict) -> Memory:
    return Memory(spec["gpu_bytes"], spec["host_bytes"])


class Runtime:
    def __init__(self, policy: dict, resources: dict, session: aiohttp.ClientSession, api_key: str):
        self.policy, self.resources, self.session = policy, resources, session
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.url = policy["upstream_url"].rstrip("/")
        self.proxies: dict[str, str] = {}

    async def command(self, *arguments: str, timeout: float = 5) -> str:
        process = await asyncio.create_subprocess_exec("/usr/bin/systemctl", "--user", *arguments, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
        except BaseException:
            if process.returncode is None:
                process.kill()
            await process.wait()
            raise
        if process.returncode:
            LOG.error("systemctl failed exit=%s detail=%s", process.returncode, stderr.decode(errors="replace")[-512:].strip())
            raise RuntimeError("user service manager operation failed")
        return stdout.decode()

    async def healthy(self) -> bool:
        try:
            self.capacity()
            state = await self.command("show", "llama-swap.service", "-p", "ActiveState", "--value")
            if state.strip() != "active":
                return False
            async with self.session.get(f"{self.url}/health", timeout=aiohttp.ClientTimeout(total=2)) as response:
                return response.status == 200
        except (aiohttp.ClientError, TimeoutError, OSError, RuntimeError, ValueError):
            return False

    async def json(self, url: str) -> object:
        async with self.session.get(url, headers=self.headers, timeout=aiohttp.ClientTimeout(total=5)) as response:
            response.raise_for_status()
            return await response.json()

    async def snapshot(self) -> dict[str, str]:
        body = await self.json(f"{self.url}/running")
        states = {}
        self.proxies = {}
        for model in body["running"]:
            name = model["model"]
            if name not in self.resources:
                raise RuntimeError("unprofiled model is running")
            states[name] = model["state"]
            self.proxies[name] = model["proxy"]
        units = sorted({unit for spec in self.resources.values() for unit in spec.get("units", [])})
        output = await self.command("show", *units, "-p", "Id", "-p", "ActiveState") if units else ""
        active = {}
        for section in output.strip().split("\n\n"):
            fields = dict(line.split("=", 1) for line in section.splitlines() if "=" in line)
            if "Id" in fields:
                active[fields["Id"]] = fields.get("ActiveState", "unknown")
        for name, spec in self.resources.items():
            statuses = [active.get(unit, "unknown") for unit in spec.get("units", [])]
            if statuses and any(state not in {"inactive", "failed"} for state in statuses):
                states[name] = "ready" if all(state == "active" for state in statuses) else "starting"
        return states

    async def ready(self, name: str) -> bool:
        spec = self.resources[name]
        urls = [spec.get("health_url") or f"{self.proxies.get(name, '')}/health"]
        if spec.get("ready_url"):
            urls.append(spec["ready_url"])
        try:
            for url in urls:
                if not url.startswith("http://"):
                    return False
                async with self.session.get(url, headers=self.headers, timeout=aiohttp.ClientTimeout(total=2)) as response:
                    if response.status != 200:
                        return False
            return True
        except (aiohttp.ClientError, TimeoutError):
            return False

    async def busy(self, name: str) -> bool:
        spec = self.resources[name]
        try:
            if "queue_url" in spec:
                body = await self.json(spec["queue_url"])
                return bool(body["queue_running"] or body["queue_pending"])
            url = spec.get("slots_url") or f"{self.proxies[name]}/slots"
            slots = await self.json(url)
            return not isinstance(slots, list) or any(slot.get("is_processing", True) for slot in slots)
        except (aiohttp.ClientError, TimeoutError, KeyError, TypeError):
            return True

    async def start(self, name: str) -> None:
        spec = self.resources[name]
        if spec["kind"] == "text":
            async with self.session.get(f"{self.url}/upstream/{quote(name, safe='')}/health", headers=self.headers) as response:
                if response.status != 200:
                    raise RuntimeError("model failed to load")
                await response.read()
            return
        await self.command("start", *spec["units"], timeout=self.policy["load_timeout_s"])
        while not await self.ready(name):
            await asyncio.sleep(0.25)

    async def stop(self, name: str) -> None:
        spec = self.resources[name]
        if spec["kind"] == "text":
            async with self.session.post(f"{self.url}/api/models/unload/{quote(name, safe='')}", headers=self.headers) as response:
                response.raise_for_status()
                await response.read()
        else:
            # Stop API before encoder: the API's Requires= also stops it when
            # its encoder disappears, but neither unit is allowed to respawn.
            await self.command("stop", *spec["units"], timeout=self.policy["stop_timeout_s"])
        if name in await self.snapshot():
            raise RuntimeError("backend remained resident after stop")

    def capacity(self) -> tuple[Memory, Memory]:
        devices = list(Path("/sys/class/drm").glob("card*/device/mem_info_gtt_total"))
        if len(devices) != 1:
            raise RuntimeError("expected one Redqueen shared-memory GPU")
        gpu_total = int(devices[0].read_text())
        gpu_used = int(devices[0].with_name("mem_info_gtt_used").read_text())
        meminfo = {key: int(value.split()[0]) * 1024 for key, value in (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())}
        capacity = Memory(gpu_total - self.policy["gpu_reserve_mib"] * MIB, min(self.policy["host_budget_mib"] * MIB, meminfo["MemTotal"] - self.policy["host_reserve_mib"] * MIB))
        return capacity, Memory(gpu_used, meminfo["MemTotal"] - meminfo["MemAvailable"])


class Scheduler:
    def __init__(self, resources: dict, runtime: Runtime, lock_dir: Path, policy: dict, *, clock: Callable[[], float] = time.monotonic):
        self.resources, self.runtime, self.lock_dir, self.policy, self.clock = resources, runtime, lock_dir, policy, clock
        self.states: dict[str, str] = {}
        self.last_used: dict[str, float] = {}
        self.active: dict[str, int] = {}
        self.transition = asyncio.Lock()
        self.lock_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    async def refresh(self) -> None:
        self.states = await self.runtime.snapshot()
        for name in self.states:
            self.last_used.setdefault(name, self.clock())

    def reserved(self) -> Memory:
        return sum((footprint(self.resources[name]) for name in self.states), Memory(0, 0))

    async def admit(self, name: str) -> None:
        if name not in self.resources:
            raise AdmissionError(404, "model is not configured", "model_not_found")
        if self.transition.locked():
            # Existing warm work can proceed while another model loads. Probe
            # directly so a crashed warm model cannot autoload around the budget.
            if self.states.get(name) == "ready" and await self.runtime.ready(name):
                self.last_used[name] = self.clock()
                return
            raise AdmissionError(503, "another model is changing state; retry shortly", "model_transition")
        async with self.transition:
            victim_locks: list[tuple[str, ResourceLock]] = []
            loading = False
            try:
                await self.refresh()
                if name in self.states and self.states[name] != "ready":
                    raise AdmissionError(503, "model is changing state; retry shortly", "model_transition")
                capacity, usage = self.runtime.capacity()
                required = footprint(self.resources[name])
                if not required.fits(capacity):
                    raise AdmissionError(503, "model reservation exceeds this host's memory budget", "model_exceeds_budget")
                reserved = self.reserved()
                unknown = Memory(max(0, usage.gpu - reserved.gpu), max(0, usage.host - reserved.host))
                planned = reserved + unknown + (Memory(0, 0) if name in self.states else required)
                candidates = sorted((other for other in self.states if other != name), key=lambda other: (self.last_used[other], other))
                for victim in candidates:
                    if planned.fits(capacity):
                        break
                    lock = ResourceLock(self.lock_dir, victim, exclusive=True)
                    if not lock.acquire():
                        continue
                    if self.states[victim] != "ready" or await self.runtime.busy(victim):
                        lock.close()
                        continue
                    victim_locks.append((victim, lock))
                    planned = planned - footprint(self.resources[victim])
                if not planned.fits(capacity):
                    raise AdmissionError(503, "insufficient memory while other models are in use; retry later", "capacity_unavailable")
                for victim, _ in victim_locks:
                    LOG.info("evict model=%s last_request_age_s=%.1f", victim, self.clock() - self.last_used[victim])
                    await asyncio.wait_for(self.runtime.stop(victim), self.policy["stop_timeout_s"])
                    self.states.pop(victim, None)
                # Recheck real allocations after confirmed stops. Unknown GPU
                # users and delayed frees must never become imaginary capacity.
                capacity, usage = self.runtime.capacity()
                reserved = self.reserved()
                unknown = Memory(max(0, usage.gpu - reserved.gpu), max(0, usage.host - reserved.host))
                if not (reserved + unknown + (Memory(0, 0) if name in self.states else required)).fits(capacity):
                    raise AdmissionError(503, "memory is not yet available after eviction", "capacity_unavailable")
                if name not in self.states or not await self.runtime.ready(name):
                    loading = True
                    self.states[name] = "starting"
                    LOG.info("load model=%s gpu_mib=%.0f host_mib=%.0f context=%s", name, required.gpu / MIB, required.host / MIB, self.resources[name].get("context_tokens", 0))
                    await asyncio.wait_for(self.runtime.start(name), self.policy["load_timeout_s"])
                    self.states[name] = "ready"
                self.last_used[name] = self.clock()
            except AdmissionError:
                raise
            except (aiohttp.ClientError, TimeoutError, OSError, RuntimeError, ValueError, KeyError) as error:
                if loading:
                    try:
                        await asyncio.wait_for(self.runtime.stop(name), self.policy["stop_timeout_s"])
                        self.states.pop(name, None)
                    except Exception:
                        self.states[name] = "quarantined"
                LOG.error("admission failed model=%s reason=%s", name, type(error).__name__)
                raise AdmissionError(503, "model could not become ready within its load deadline", "model_load_failed") from error
            finally:
                for _, lock in victim_locks:
                    lock.close()

    @asynccontextmanager
    async def request(self, name: str):
        if name not in self.resources or self.resources[name]["kind"] != "text":
            raise AdmissionError(404, "model is not configured", "model_not_found")
        limit = self.resources[name].get("concurrency", 1)
        if self.active.get(name, 0) >= limit:
            raise AdmissionError(429, "model is serving its maximum concurrent requests", "model_busy")
        lock = ResourceLock(self.lock_dir, name)
        if not lock.acquire():
            raise AdmissionError(503, "model is being evicted; retry shortly", "model_transition")
        self.active[name] = self.active.get(name, 0) + 1
        try:
            # Finish admission cleanup even if its client disappears during a
            # slow load; otherwise an unaccounted process could keep loading.
            task = asyncio.create_task(self.admit(name))
            await finish_shielded(task)
            yield
        finally:
            self.active[name] -= 1
            lock.close()

    async def reap(self) -> None:
        if self.transition.locked():
            return
        async with self.transition:
            await self.refresh()
            for name in list(self.states):
                if self.clock() - self.last_used[name] < self.policy["idle_timeout_s"]:
                    continue
                lock = ResourceLock(self.lock_dir, name, exclusive=True)
                try:
                    if not lock.acquire() or await self.runtime.busy(name):
                        continue
                    await asyncio.wait_for(self.runtime.stop(name), self.policy["stop_timeout_s"])
                    self.states.pop(name, None)
                finally:
                    lock.close()


def create_apps(policy: dict, resources: dict, api_key: str, *, runtime_factory=Runtime) -> tuple[web.Application, web.Application]:
    public = web.Application(client_max_size=32 * MIB, middlewares=[safe_errors, auth_middleware(api_key)])
    local = web.Application(client_max_size=4096, middlewares=[safe_errors])
    state = {}

    async def lifecycle(_: web.Application):
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=policy["request_timeout_s"]), auto_decompress=False) as session:
            runtime = runtime_factory(policy, resources, session, api_key)
            scheduler = Scheduler(resources, runtime, Path(policy["socket_path"]).parent / "leases", policy)
            state.update(session=session, runtime=runtime, scheduler=scheduler)
            async def reaper():
                while True:
                    await asyncio.sleep(30)
                    try:
                        await scheduler.reap()
                    except (Exception, asyncio.CancelledError) as error:
                        if isinstance(error, asyncio.CancelledError):
                            raise
                        LOG.warning("idle cleanup deferred reason=%s", type(error).__name__)
            task = asyncio.create_task(reaper())
            try:
                yield
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    public.cleanup_ctx.append(lifecycle)

    async def health(_: web.Request):
        return web.json_response({"ok": True})

    async def ready(_: web.Request):
        ok = await state["runtime"].healthy()
        return web.json_response({"ok": ok}, status=200 if ok else 503)

    async def status(_: web.Request):
        scheduler = state["scheduler"]
        return web.json_response({"models": {name: {"state": value, "active_requests": scheduler.active.get(name, 0), "last_request_age_s": scheduler.clock() - scheduler.last_used.get(name, scheduler.clock()), **{key: resources[name][key] for key in ("gpu_bytes", "host_bytes", "context_tokens")}} for name, value in scheduler.states.items()}})

    async def admit(request: web.Request):
        body = await request.json()
        name = body.get("resource") if isinstance(body, dict) else None
        if not isinstance(name, str) or name not in resources or resources[name]["kind"] != "service":
            raise AdmissionError(404, "resource is not configured", "model_not_found")
        # The adapter holds a shared flock before requesting admission. It
        # remains the active-work signal even if this process restarts.
        lock = ResourceLock(state["scheduler"].lock_dir, name, exclusive=True)
        try:
            if lock.acquire():
                raise AdmissionError(400, "an active resource lease is required", "lease_required")
        finally:
            lock.close()
        task = asyncio.create_task(state["scheduler"].admit(name))
        await finish_shielded(task)
        return web.json_response({"ok": True})

    async def proxy(request: web.Request):
        path, model, body = request.path, None, None
        if path.startswith("/upstream/"):
            parts = path.split("/", 3)
            if len(parts) < 4:
                raise AdmissionError(404, "unknown inference route", "not_found")
            model = parts[2]
            if request.method not in {"GET", "POST"}:
                raise AdmissionError(405, "method is not allowed", "method_not_allowed")
        elif path in MODEL_ENDPOINTS and request.method == "POST":
            try:
                body = await request.json()
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise ValueError("request body must be valid JSON")
            if not isinstance(body, dict) or not isinstance(body.get("model"), str):
                raise ValueError("model must be a string")
            model = body["model"]
        elif request.method != "GET" or path not in {"/v1/models", "/models", "/health", "/running", "/metrics", "/logs"}:
            raise AdmissionError(404, "unknown inference route", "not_found")
        if model is not None:
            aliases = {alias: name for name, spec in resources.items() for alias in spec.get("aliases", [])}
            model = aliases.get(model, model)
        content_type = request.headers.get("Content-Type", "application/json")
        if body is not None:
            # Python and llama-swap's JSON parser disagree on duplicate keys;
            # non-JSON content types also let upstream query parameters route.
            # Forward exactly the canonical model whose memory we admitted.
            body["model"] = model
            data = json.dumps(body, allow_nan=False, ensure_ascii=False).encode()
            content_type = "application/json"
        else:
            data = await request.read()
        async def forward():
            response = None
            async def transfer():
                nonlocal response
                async with asyncio.timeout(policy["response_timeout_s"]):
                    upstream = await state["session"].request(request.method, f"{policy['upstream_url']}{request.rel_url}", data=data, headers={"Authorization": f"Bearer {api_key}", "Content-Type": content_type}, allow_redirects=False)
                    try:
                        first = await upstream.content.readany()
                    except BaseException:
                        upstream.close()
                        raise
                async with upstream:
                    headers = {key: value for key, value in upstream.headers.items() if key.lower() not in HOP_HEADERS and key.lower() != "content-length"}
                    response = web.StreamResponse(status=upstream.status, headers=headers)
                    await response.prepare(request)
                    if first:
                        await response.write(first)
                    while True:
                        async with asyncio.timeout(policy["response_timeout_s"]):
                            chunk = await upstream.content.readany()
                        if not chunk:
                            break
                        await response.write(chunk)
                    await response.write_eof()
                    return response
            try:
                # Include writes to slow clients: an upstream session timeout
                # ends at EOF and cannot bound a blocked downstream write.
                async with asyncio.timeout(policy["request_timeout_s"]):
                    return await transfer()
            except (TimeoutError, aiohttp.ClientError) as error:
                if response is not None and response.prepared:
                    try:
                        async with asyncio.timeout(min(1, policy["response_timeout_s"])):
                            if "text/event-stream" in response.headers.get("Content-Type", ""):
                                await response.write(b'data: {"error":{"message":"response deadline exceeded","type":"server_error","code":"upstream_timeout"}}\n\n')
                            await response.write_eof()
                    except (TimeoutError, ConnectionError, RuntimeError):
                        response.force_close()
                        if request.transport is not None:
                            request.transport.close()
                    return response
                raise AdmissionError(504 if isinstance(error, TimeoutError) else 503, "upstream did not respond within its deadline", "upstream_timeout") from error
        if model is None:
            return await forward()
        async with state["scheduler"].request(model):
            return await forward()

    public.router.add_get("/healthz", health)
    public.router.add_get("/readyz", ready)
    public.router.add_get("/scheduler", status)
    public.router.add_route("*", "/{path:.*}", proxy)
    local.router.add_post("/admit", admit)
    local.router.add_get("/healthz", health)
    local.router.add_get("/readyz", ready)
    return public, local


async def serve(policy: dict, resources: dict, api_key: str, host: str, port: int) -> None:
    public, local = create_apps(policy, resources, api_key)
    public_runner = web.AppRunner(public, handler_cancellation=True, access_log=None, shutdown_timeout=35)
    local_runner = web.AppRunner(local, handler_cancellation=True, access_log=None, shutdown_timeout=35)
    await public_runner.setup()
    await local_runner.setup()
    socket = Path(policy["socket_path"])
    if socket.exists():
        import stat
        info = socket.lstat()
        if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.geteuid():
            raise ValueError("refusing to replace a non-owned scheduler socket")
        socket.unlink()
    await web.UnixSite(local_runner, str(socket)).start()
    socket.chmod(0o600)
    await web.TCPSite(public_runner, host, port).start()
    try:
        await asyncio.Event().wait()
    finally:
        await local_runner.cleanup()
        await public_runner.cleanup()
        socket.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=Path("/srv/ai/model-router/scheduler.json"))
    parser.add_argument("--host", default="10.9.20.242")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text())
    profiles = json.loads(Path(policy["profile_path"]).read_text())
    verify_profiles(profiles)
    api_key = os.environ.get("QWEN_API_KEY", "")
    if not api_key:
        raise SystemExit("QWEN_API_KEY is required")
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    try:
        asyncio.run(serve(policy, profiles["resources"], api_key, args.host, args.port))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
