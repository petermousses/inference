"""Memory admission and idle-only LRU eviction for Redqueen's inference backends."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import re
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import aiohttp
from aiohttp import web

from .common import AdmissionError, IngressGate, UpstreamError, admission_response, auth_middleware, body_timeout, decode_json, read_json, read_limited, safe_errors
from .memory_profile import MIB, verify_profiles
from .scheduler_client import ResourceLock, SerializedExecution, finish_shielded
from .ownership import OwnedRequest, abandoned, boot_id, records, remove_record

LOG = logging.getLogger("redqueen.scheduler")
HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade"}
MODEL_ENDPOINTS = {"/v1/chat/completions", "/v1/completions", "/v1/responses", "/v1/embeddings", "/completion", "/tokenize", "/detokenize"}
OPENAI_REQUEST_SHAPES = {
    "/v1/chat/completions": "openai.chat",
    "/v1/completions": "openai.completion",
    "/v1/responses": "openai.responses",
    "/v1/embeddings": "openai.embeddings",
}
MULTIMODAL_REQUEST_ENDPOINTS = {
    "/completion",
    "/v1/completions",
    "/v1/chat/completions",
    "/v1/responses",
    "/v1/chat/completions/input_tokens",
    "/v1/responses/input_tokens",
    "/v1/messages/count_tokens",
    "/v1/messages",
    "/v1/audio/transcriptions",
}
LLAMA_CPP_REQUEST_ALIASES = {
    "/chat/completions": "/v1/chat/completions",
    "/completions": "/completion",
    "/responses": "/v1/responses",
    "/chat/completions/input_tokens": "/v1/chat/completions/input_tokens",
    "/responses/input_tokens": "/v1/responses/input_tokens",
    "/audio/transcriptions": "/v1/audio/transcriptions",
}
MULTIMODAL_REQUEST_SHAPES = {
    "openai.multimodal_chat",
    "openai.multimodal_completion",
    "openai.multimodal_responses",
    "openai.multimodal_audio",
    "openai.multimodal_video",
    "openai.multimodal_token_count",
    "openai.multimodal_file",
    "anthropic.multimodal_messages",
}
TOKEN_COUNT_ENDPOINTS = {
    "/v1/chat/completions/input_tokens",
    "/v1/responses/input_tokens",
    "/v1/messages/count_tokens",
}
AUDIO_TRANSCRIPTION_ENDPOINTS = {
    "/v1/audio/transcriptions",
}
NON_IMAGE_MEDIA_TYPES = {
    "audio",
    "audio_url",
    "input_audio",
    "file",
    "input_file",
    "document",
    "input_document",
    "input_video",
    "video",
    "video_url",
}
SSE_BOUNDARY = re.compile(rb"\r\n\r\n|\n\n|\r\r")
MAX_SSE_FRAME = 256 * 1024


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
    profile = spec.get("memory")
    if not isinstance(profile, dict):
        return Memory(spec["gpu_bytes"], spec["host_bytes"])
    total = Memory(0, 0)
    for component in ("mandatory", "reclaimable", "transient_peak"):
        value = profile.get(component)
        if not isinstance(value, dict):
            raise ValueError(f"invalid {component} memory profile")
        total += Memory(value["gpu_bytes"], value["host_bytes"])
    if total != Memory(spec["gpu_bytes"], spec["host_bytes"]):
        raise ValueError("worker memory components do not match its admission footprint")
    return total


def is_openai_worker(spec: dict) -> bool:
    """Whether the worker accepts requests through the public OpenAI gateway."""
    return spec.get("kind") == "text" or (
        spec.get("kind") == "service" and bool(spec.get("gateway_url"))
    )


def count_image_inputs(value: object) -> int:
    """Count OpenAI image parts and llama.cpp multimodal prompt items."""
    count = 0
    pending = [iter((value,))]
    while pending:
        try:
            item = next(pending[-1])
        except StopIteration:
            pending.pop()
            continue
        if isinstance(item, dict):
            content_type = item.get("type")
            if isinstance(content_type, str) and content_type in {"image", "image_url", "input_image"}:
                count += 1
            if "multimodal_data" in item:
                media = item["multimodal_data"]
                count += len(media) if isinstance(media, list) else 1
            pending.append(
                iter(
                    item[key]
                    for key in ("messages", "input", "content", "prompt")
                    if key in item
                )
            )
        elif isinstance(item, list):
            pending.append(iter(item))
    return count


def contains_image_input(value: object) -> bool:
    return count_image_inputs(value) > 0


def non_image_media_kind(value: object) -> str | None:
    """Find media inputs without an image admission profile, such as audio or video."""
    pending = [iter((value,))]
    while pending:
        try:
            item = next(pending[-1])
        except StopIteration:
            pending.pop()
            continue
        if isinstance(item, dict):
            content_type = item.get("type")
            if isinstance(content_type, str) and content_type in NON_IMAGE_MEDIA_TYPES:
                if "audio" in content_type:
                    return "audio"
                if "video" in content_type:
                    return "video"
                return "file"
            pending.append(
                iter(item[key] for key in ("messages", "input", "content", "prompt") if key in item)
            )
        elif isinstance(item, list):
            pending.append(iter(item))
    return None


def request_shape(path: str, body: object) -> str | None:
    shape = OPENAI_REQUEST_SHAPES.get(path)
    media_kind = non_image_media_kind(body)
    if path in TOKEN_COUNT_ENDPOINTS and (media_kind is not None or contains_image_input(body)):
        return "openai.multimodal_token_count"
    if path == "/v1/messages" and (media_kind is not None or contains_image_input(body)):
        return "anthropic.multimodal_messages"
    if path in {"/v1/chat/completions", "/v1/responses"} and media_kind is not None:
        return f"openai.multimodal_{media_kind}"
    if path == "/v1/chat/completions" and contains_image_input(body):
        return "openai.multimodal_chat"
    if path == "/v1/responses" and contains_image_input(body):
        return "openai.multimodal_responses"
    if path in {"/completion", "/v1/completions"} and count_image_inputs(body):
        return "openai.multimodal_completion"
    return shape


def capacity_limits(policy: dict, *, gpu_total: int, host_total: int) -> Memory:
    """Return the managed ceilings after preserving host availability headroom."""
    gpu_budget = policy["gpu_budget_mib"] * MIB
    host_budget = policy["host_budget_mib"] * MIB
    host_reserve = (
        policy["minimum_host_available_mib"] + policy["prefill_headroom_mib"]
    ) * MIB
    if min(gpu_budget, host_budget, gpu_total, host_total) <= 0:
        raise ValueError("memory budgets and discovered capacities must be positive")
    if min(policy["minimum_host_available_mib"], policy["prefill_headroom_mib"]) < 0:
        raise ValueError("memory headroom reserves must not be negative")
    if host_total <= host_reserve:
        raise ValueError("host memory is below the required availability reserve")
    return Memory(min(gpu_total, gpu_budget), min(host_budget, host_total - host_reserve))


class ExpertCacheResizeError(RuntimeError):
    """A worker rejected an expert-cache resize over HTTP."""

    def __init__(self, status_code: int):
        super().__init__("expert-cache resize was not accepted")
        self.status_code = status_code


class ExpertCacheStateError(RuntimeError):
    """The worker's cache state cannot safely support memory accounting."""

    def __init__(self, reason_code: str):
        super().__init__("expert-cache state is invalid or unsettled")
        self.reason_code = reason_code


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
            self.verify_configuration()
            self.capacity()
            units = sorted({"llama-swap.service", *(unit for spec in self.resources.values() for unit in spec.get("units", []))})
            loaded = (await self.command("show", *units, "-p", "LoadState", "--value")).split()
            if len(loaded) != len(units) or any(value != "loaded" for value in loaded):
                return False
            state = await self.command("show", "llama-swap.service", "-p", "ActiveState", "--value")
            if state.strip() != "active":
                return False
            async with self.session.get(f"{self.url}/health", timeout=aiohttp.ClientTimeout(total=2)) as response:
                return response.status == 200
        except (aiohttp.ClientError, TimeoutError, OSError, RuntimeError, ValueError):
            return False

    def verify_configuration(self) -> None:
        if self.policy.get("profile_path"):
            verify_profiles(json.loads(Path(self.policy["profile_path"]).read_text()))

    async def json(self, url: str, *, limit: int = 8 * 1024 * 1024) -> object:
        async with self.session.get(url, headers=self.headers, timeout=aiohttp.ClientTimeout(total=5)) as response:
            try:
                return decode_json(await read_limited(response, limit))
            except (ValueError, UnicodeDecodeError) as error:
                raise RuntimeError("invalid runtime control response") from error

    def expert_cache_url(self, name: str) -> str | None:
        spec = self.resources[name]
        url = spec.get("expert_cache_url")
        if url is None and spec.get("adaptive_cache"):
            proxy = self.proxies.get(name)
            base = proxy.rstrip("/") if proxy else f"{self.url}/upstream/{quote(name, safe='')}"
            url = base + "/expert-cache"
        return url

    async def expert_cache_info(
        self, name: str, *, allow_pending_backing_release: bool = False
    ) -> dict:
        url = self.expert_cache_url(name)
        if not url:
            raise ExpertCacheStateError("missing_endpoint")
        try:
            body = await self.json(url, limit=64 * 1024)
        except RuntimeError as error:
            raise ExpertCacheStateError("invalid_json_response") from error
        spec = self.resources[name]
        limits = spec.get("adaptive_cache")
        if not isinstance(body, dict) or not isinstance(limits, dict):
            raise ExpertCacheStateError("invalid_response_shape")
        minimum_capacity = body.get("minimum_capacity_bytes")
        maximum_capacity = body.get("maximum_capacity_bytes")
        # The fit planner rounds its safe maximum down to a representable
        # allocation, while the runtime reports its configured upper bound.
        # All scheduler targets remain capped at the profiled maximum below.
        if (
            isinstance(minimum_capacity, bool)
            or not isinstance(minimum_capacity, int)
            or minimum_capacity != limits["minimum_capacity_bytes"]
            or isinstance(maximum_capacity, bool)
            or not isinstance(maximum_capacity, int)
            or maximum_capacity < limits["maximum_capacity_bytes"]
        ):
            raise ExpertCacheStateError("profile_bounds_mismatch")
        capacity = body.get("capacity_bytes")
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ExpertCacheStateError("invalid_capacity")
        if not limits["minimum_capacity_bytes"] <= capacity <= limits["maximum_capacity_bytes"]:
            raise ExpertCacheStateError("capacity_out_of_bounds")
        pending_release = body.get("pending_backing_release")
        quarantined = body.get("quarantined")
        if type(pending_release) is not bool or type(quarantined) is not bool:
            raise ExpertCacheStateError("invalid_state_flags")
        if quarantined:
            raise ExpertCacheStateError("quarantined")
        if pending_release and not allow_pending_backing_release:
            raise ExpertCacheStateError("pending_backing_release")
        return body

    async def resize_expert_cache(self, name: str, capacity_bytes: int) -> dict:
        url = self.expert_cache_url(name)
        if not url:
            raise ExpertCacheStateError("missing_endpoint")
        timeout = aiohttp.ClientTimeout(total=None, connect=5, sock_connect=5, sock_read=None)
        async with self.session.post(
            url,
            headers=self.headers,
            json={"capacity_bytes": capacity_bytes},
            timeout=timeout,
        ) as response:
            try:
                body = decode_json(await read_limited(response, 64 * 1024, check_status=False))
            except (ValueError, UnicodeDecodeError) as error:
                raise ExpertCacheStateError("invalid_resize_response") from error
            if response.status != 200:
                raise ExpertCacheResizeError(response.status)
            if not isinstance(body, dict):
                raise ExpertCacheStateError("invalid_resize_response")
        info = await self.expert_cache_info(name)
        # Backends can report allocator-rounded capacity above the requested
        # byte target. The scheduler accounts for the reported capacity and
        # checks physical fit before dispatch; cache shrinks probe a lower
        # representable target when this overshoots.
        return info

    async def generation(self, name: str) -> str | None:
        units = self.resources[name].get("units", [])
        if not units:
            return None
        output = await self.command("show", *units, "-p", "Id", "-p", "MainPID", "-p", "ExecMainStartTimestampMonotonic")
        return boot_id() + ":" + output.strip()

    async def failed(self, name: str) -> bool:
        """Prove process loss independently of a slow/unreachable HTTP probe."""
        spec = self.resources[name]
        units = spec.get("units", [])
        if units:
            output = await self.command("show", *units, "-p", "ActiveState", "--value")
            return any(value in {"inactive", "failed"} for value in output.split())
        paths = spec.get("model_paths", [])
        if not paths:
            return False
        group = (await self.command("show", "llama-swap.service", "-p", "ControlGroup", "--value")).strip()
        if not group.startswith("/") or ".." in Path(group).parts:
            return False
        try:
            pids = (Path("/sys/fs/cgroup") / group.lstrip("/") / "cgroup.procs").read_text().split()
            for pid in pids:
                try:
                    arguments = Path(f"/proc/{int(pid)}/cmdline").read_bytes().split(b"\0")
                    if any(os.fsencode(path) in arguments for path in paths):
                        return False
                except FileNotFoundError:
                    continue
            return True
        except (OSError, ValueError):
            return False

    async def snapshot(self) -> dict[str, str]:
        body = await self.json(f"{self.url}/running")
        if not isinstance(body, dict) or not isinstance(body.get("running"), list):
            raise UpstreamError(502, "invalid model router response", "upstream_invalid_response")
        states = {}
        self.proxies = {}
        for model in body["running"]:
            if not isinstance(model, dict) or any(not isinstance(model.get(key), str) for key in ("model", "state", "proxy")):
                raise UpstreamError(502, "invalid model router response", "upstream_invalid_response")
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
            if self.resources[name].get("adaptive_cache"):
                # Pending allocator release is accounted as physical unknown
                # memory; readiness only rejects a quarantined or invalid worker.
                await self.expert_cache_info(name, allow_pending_backing_release=True)
            if spec.get("nodes_url"):
                nodes_url = spec["nodes_url"].rstrip("/")
                for node in spec["required_nodes"]:
                    # ComfyUI's unfiltered /object_info endpoint walks every
                    # installed node class. Probe the per-node route so a
                    # large extension catalog cannot exceed our bounded JSON
                    # decoder while the backend is starting.
                    node_info = await self.json(
                        f"{nodes_url}/{quote(node, safe='')}", limit=512 * 1024
                    )
                    if not isinstance(node_info, dict) or not isinstance(node_info.get(node), dict):
                        return False
            return True
        except (aiohttp.ClientError, TimeoutError, ValueError, TypeError, RuntimeError):
            return False

    async def busy(self, name: str) -> bool | None:
        spec = self.resources[name]
        try:
            if "queue_url" in spec:
                body = await self.json(spec["queue_url"])
                return bool(body["queue_running"] or body["queue_pending"])
            url = spec.get("slots_url") or f"{self.proxies[name]}/slots"
            slots = await self.json(url)
            if not isinstance(slots, list) or any(not isinstance(slot, dict) or not isinstance(slot.get("is_processing"), bool) for slot in slots):
                return None
            return any(slot["is_processing"] for slot in slots)
        except (aiohttp.ClientError, TimeoutError, KeyError, TypeError, AttributeError, RuntimeError, UpstreamError):
            return None

    async def start(self, name: str) -> None:
        spec = self.resources[name]
        if spec["kind"] == "text":
            async with self.session.get(f"{self.url}/upstream/{quote(name, safe='')}/health", headers=self.headers) as response:
                if response.status != 200:
                    raise RuntimeError("model failed to load")
                await read_limited(response, 64 * 1024)
            return
        # systemctl waits for ExecStartPost as part of the start job. ComfyUI's
        # admission-guard readiness probe can take 30 seconds after its 180s
        # unit start deadline, so the command timeout must exceed both.
        command_timeout = spec.get("start_command_timeout_s", 10)
        if (
            isinstance(command_timeout, bool)
            or not isinstance(command_timeout, (int, float))
            or command_timeout <= 0
        ):
            raise RuntimeError("backend start command timeout is invalid")
        startup_timeout = spec.get("startup_timeout_s", command_timeout + 30)
        if (
            isinstance(startup_timeout, bool)
            or not isinstance(startup_timeout, (int, float))
            or not math.isfinite(startup_timeout)
            or startup_timeout <= 0
        ):
            raise RuntimeError("backend startup timeout is invalid")
        deadline = time.monotonic() + startup_timeout
        await asyncio.wait_for(
            self.command(
                "start", *spec["units"],
                timeout=min(command_timeout, startup_timeout),
            ),
            timeout=startup_timeout,
        )
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("backend did not become ready before its startup deadline")
            if await asyncio.wait_for(self.ready(name), timeout=remaining):
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("backend did not become ready before its startup deadline")
            if await asyncio.wait_for(self.failed(name), timeout=remaining):
                raise RuntimeError("backend process exited during loading")
            await asyncio.sleep(min(0.25, max(0, deadline - time.monotonic())))

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
        capacity = capacity_limits(
            self.policy, gpu_total=gpu_total, host_total=meminfo["MemTotal"]
        )
        return capacity, Memory(gpu_used, meminfo["MemTotal"] - meminfo["MemAvailable"])


class Scheduler:
    def __init__(self, resources: dict, runtime: Runtime, lock_dir: Path, policy: dict, *, clock: Callable[[], float] = time.monotonic):
        self.resources, self.runtime, self.lock_dir, self.policy, self.clock = resources, runtime, lock_dir, policy, clock
        self.states: dict[str, str] = {}
        self.last_used: dict[str, float] = {}
        self.active: dict[str, int] = {}
        self.cache_allocations: dict[str, int] = {
            name: spec["adaptive_cache"]["minimum_capacity_bytes"]
            for name, spec in resources.items()
            if isinstance(spec.get("adaptive_cache"), dict)
        }
        self.cache_unready: set[str] = set()
        self.cache_fallback_caps: dict[str, int] = {}
        self.cache_fallback_demands: dict[str, tuple[str, ...]] = {}
        self.cache_fallback_targets: dict[str, int] = {}
        self.cache_rebalance_tasks: set[asyncio.Task] = set()
        self.transition = asyncio.Lock()
        self.execution = SerializedExecution(lock_dir)
        self.recovery: dict[str, dict] = {}
        self.lock_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    async def refresh(self) -> None:
        self.states = await self.runtime.snapshot()
        for name in self.states:
            self.last_used.setdefault(name, self.clock())
        for name, spec in self.resources.items():
            if not spec.get("adaptive_cache"):
                continue
            if name not in self.states:
                self.cache_allocations.pop(name, None)
                self.cache_unready.discard(name)
                self.cache_fallback_caps.pop(name, None)
                self.cache_fallback_demands.pop(name, None)
                self.cache_fallback_targets.pop(name, None)
                continue
            try:
                # Keep the logical capacity. Any physical bytes not returned by
                # the allocator remain in the usage-minus-reservation reserve.
                info = await self.runtime.expert_cache_info(
                    name, allow_pending_backing_release=True
                )
            except (aiohttp.ClientError, TimeoutError, OSError, RuntimeError, ValueError, KeyError):
                # If live cache ownership cannot be proven, reserve its full
                # profile and refuse to transfer that memory to another model.
                self.cache_allocations[name] = spec["adaptive_cache"]["maximum_capacity_bytes"]
                self.cache_unready.add(name)
            else:
                self.cache_allocations[name] = info["capacity_bytes"]
                self.cache_unready.discard(name)

    def reserved(self) -> Memory:
        return sum((self.reservation(name) for name in self.states), Memory(0, 0))

    def reservation(self, name: str, *, include_transient: bool | None = None) -> Memory:
        """Current reservation; adapter-only workers keep their legacy full peak."""
        spec = self.resources[name]
        profile = spec.get("memory")
        if not isinstance(profile, dict):
            return Memory(spec["gpu_bytes"], spec["host_bytes"])
        active = include_transient
        if active is None:
            active = self.active.get(name, 0) > 0 or not is_openai_worker(spec)
        mandatory = profile["mandatory"]
        result = Memory(mandatory["gpu_bytes"], mandatory["host_bytes"])
        adaptive = spec.get("adaptive_cache")
        if isinstance(adaptive, dict):
            minimum = adaptive["minimum_capacity_bytes"]
            capacity = self.cache_allocations.get(name, adaptive["maximum_capacity_bytes"])
            delta = math.ceil(max(0, capacity - minimum) * adaptive["reservation_multiplier"])
            result += Memory(delta, delta)
        else:
            reclaimable = profile["reclaimable"]
            result += Memory(reclaimable["gpu_bytes"], reclaimable["host_bytes"])
        if active:
            peak = profile["transient_peak"]
            result += Memory(peak["gpu_bytes"], peak["host_bytes"])
        return result

    def minimum_reservation(self, name: str, *, active: bool) -> Memory:
        """The admission floor; adaptive profiles include their minimum cache in mandatory."""
        spec = self.resources[name]
        profile = spec.get("memory")
        if not isinstance(profile, dict):
            return Memory(spec["gpu_bytes"], spec["host_bytes"])
        mandatory = profile["mandatory"]
        result = Memory(mandatory["gpu_bytes"], mandatory["host_bytes"])
        if not spec.get("adaptive_cache"):
            reclaimable = profile["reclaimable"]
            result += Memory(reclaimable["gpu_bytes"], reclaimable["host_bytes"])
        if active:
            peak = profile["transient_peak"]
            result += Memory(peak["gpu_bytes"], peak["host_bytes"])
        return result

    async def _rebalance_caches(
        self,
        incoming: str | None,
        capacity: Memory,
        unknown: Memory,
        *,
        reserve_incoming_startup: bool = False,
    ) -> bool:
        """Water-fill cache capacity equally across workers with current demand.

        Before loading a new worker, callers can hold resident adaptive caches
        at their minima until startup finishes and physical usage is measurable.
        """
        adaptive_names = [
            name for name in self.states
            if self.resources[name].get("adaptive_cache")
        ]
        if (
            incoming is not None
            and incoming not in self.states
            and self.resources[incoming].get("adaptive_cache")
        ):
            adaptive_names.append(incoming)
        if not adaptive_names:
            return True
        adjustable_names = [name for name in adaptive_names if name not in self.cache_unready]
        demand_names = [
            name for name in adjustable_names
            if self.active.get(name, 0) > 0 or name == incoming
        ]
        # With no in-flight request, keep the most recently used resident MoE
        # warm. A later request to another model will rebalance around it.
        if not demand_names and incoming is None:
            warm = [name for name in adjustable_names if name in self.states]
            if warm:
                demand_names = [max(warm, key=lambda name: (self.last_used.get(name, 0), name))]
        demand_signature = tuple(sorted(demand_names))
        for name in adjustable_names:
            if name not in demand_names:
                self.cache_fallback_caps.pop(name, None)
                self.cache_fallback_demands.pop(name, None)
                self.cache_fallback_targets.pop(name, None)
            else:
                if self.cache_fallback_demands.get(name) not in {
                    None,
                    demand_signature,
                }:
                    self.cache_fallback_caps.pop(name, None)
                    self.cache_fallback_targets.pop(name, None)
                self.cache_fallback_demands[name] = demand_signature

        def remember_fallback_cap(
            name: str, capacity_bytes: int, requested_bytes: int
        ) -> None:
            self.cache_fallback_caps[name] = capacity_bytes
            self.cache_fallback_demands[name] = demand_signature
            self.cache_fallback_targets[name] = requested_bytes

        def fixed_reservation() -> Memory:
            fixed = Memory(0, 0)
            for name in self.states:
                spec = self.resources[name]
                profile = spec.get("memory")
                if not isinstance(profile, dict):
                    fixed += Memory(spec["gpu_bytes"], spec["host_bytes"])
                    continue
                mandatory = profile["mandatory"]
                fixed += Memory(mandatory["gpu_bytes"], mandatory["host_bytes"])
                adaptive = spec.get("adaptive_cache")
                if isinstance(adaptive, dict) and name not in adjustable_names:
                    minimum = adaptive["minimum_capacity_bytes"]
                    allocated = self.cache_allocations.get(name, adaptive["maximum_capacity_bytes"])
                    extra = math.ceil(max(0, allocated - minimum) * adaptive["reservation_multiplier"])
                    fixed += Memory(extra, extra)
                elif not adaptive:
                    reclaimable = profile["reclaimable"]
                    fixed += Memory(reclaimable["gpu_bytes"], reclaimable["host_bytes"])
                if (
                    self.active.get(name, 0) > 0
                    or name == incoming
                    or not is_openai_worker(spec)
                ):
                    peak = profile["transient_peak"]
                    fixed += Memory(peak["gpu_bytes"], peak["host_bytes"])
            if incoming is not None and incoming not in self.states:
                fixed += self.minimum_reservation(incoming, active=True)
            return fixed

        def targets_for(
            extra_bytes: int, *, ignore_fallback_caps: bool = False
        ) -> dict[str, int]:
            if (
                reserve_incoming_startup
                and incoming is not None
                and incoming not in self.states
            ):
                return {
                    name: self.resources[name]["adaptive_cache"]["minimum_capacity_bytes"]
                    for name in adjustable_names
                }
            targets = {
                name: self.resources[name]["adaptive_cache"]["minimum_capacity_bytes"]
                for name in adjustable_names
                if name not in demand_names
            }
            remaining = extra_bytes
            uncapped = list(demand_names)
            while uncapped:
                share = remaining // len(uncapped)
                capped = []
                for name in uncapped:
                    limits = self.resources[name]["adaptive_cache"]
                    minimum = limits["minimum_capacity_bytes"]
                    maximum = min(
                        limits["maximum_capacity_bytes"],
                        limits["maximum_capacity_bytes"]
                        if ignore_fallback_caps
                        else self.cache_fallback_caps.get(
                            name, limits["maximum_capacity_bytes"]
                        ),
                    )
                    maximum_extra = math.ceil(
                        (maximum - minimum)
                        * limits["reservation_multiplier"]
                    )
                    if maximum_extra <= share:
                        targets[name] = maximum
                        remaining -= maximum_extra
                        capped.append(name)
                if not capped:
                    for name in uncapped:
                        limits = self.resources[name]["adaptive_cache"]
                        maximum = min(
                            limits["maximum_capacity_bytes"],
                            limits["maximum_capacity_bytes"]
                            if ignore_fallback_caps
                            else self.cache_fallback_caps.get(
                                name, limits["maximum_capacity_bytes"]
                            ),
                        )
                        targets[name] = limits["minimum_capacity_bytes"] + math.floor(
                            share / limits["reservation_multiplier"]
                        )
                        targets[name] = min(targets[name], maximum)
                    break
                uncapped = [name for name in uncapped if name not in capped]
            return targets

        def targets_for_demand(extra_bytes: int) -> dict[str, int]:
            unconstrained = targets_for(extra_bytes, ignore_fallback_caps=True)
            for name in demand_names:
                previous_target = self.cache_fallback_targets.get(name)
                if (
                    previous_target is not None
                    and unconstrained.get(name, 0) > previous_target
                ):
                    # More scheduler-verified headroom is now available than
                    # when this allocator ceiling was found; probe it again.
                    self.cache_fallback_caps.pop(name, None)
                    self.cache_fallback_targets.pop(name, None)
            return targets_for(extra_bytes)

        async def issue_resize(name: str, target: int) -> tuple[bool, dict]:
            task = asyncio.create_task(self.runtime.resize_expert_cache(name, target))
            try:
                info = await asyncio.shield(task)
            except asyncio.CancelledError:
                # The worker may finish a safe-point resize after its admission
                # caller disconnects. Drain it, then reconcile physical cache
                # ownership before allowing cancellation to release the lease.
                self.cache_allocations[name] = self.resources[name]["adaptive_cache"]["maximum_capacity_bytes"]
                self.cache_unready.add(name)
                try:
                    info = await finish_shielded(task)
                except asyncio.CancelledError:
                    try:
                        info = await self.runtime.expert_cache_info(name)
                    except (aiohttp.ClientError, TimeoutError, OSError, RuntimeError, ValueError, KeyError):
                        info = None
                except Exception:
                    try:
                        info = await self.runtime.expert_cache_info(name)
                    except (aiohttp.ClientError, TimeoutError, OSError, RuntimeError, ValueError, KeyError):
                        info = None
                if info is not None:
                    self.cache_allocations[name] = info["capacity_bytes"]
                    self.cache_unready.discard(name)
                raise
            except ExpertCacheResizeError as error:
                if error.status_code != 503:
                    self.cache_allocations[name] = self.resources[name]["adaptive_cache"]["maximum_capacity_bytes"]
                    self.cache_unready.add(name)
                    LOG.error("cache rebalance failed model=%s target_bytes=%d status=%d", name, target, error.status_code)
                    raise AdmissionError(503, "expert cache could not safely transfer its reservation", "cache_resize_failed") from error
                try:
                    # A rejected resize may have rolled back to the prior
                    # backing. Only retry after the worker proves its current
                    # allocation is settled and not quarantined.
                    info = await self.runtime.expert_cache_info(name)
                except (aiohttp.ClientError, TimeoutError, OSError, RuntimeError, ValueError, KeyError) as state_error:
                    self.cache_allocations[name] = self.resources[name]["adaptive_cache"]["maximum_capacity_bytes"]
                    self.cache_unready.add(name)
                    reason = getattr(state_error, "reason_code", type(state_error).__name__)
                    LOG.error("cache rebalance state unknown model=%s target_bytes=%d reason=%s", name, target, reason)
                    raise AdmissionError(503, "expert cache could not safely transfer its reservation", "cache_resize_failed") from state_error
                self.cache_allocations[name] = info["capacity_bytes"]
                self.cache_unready.discard(name)
                return False, info
            except (aiohttp.ClientError, TimeoutError, OSError, RuntimeError, ValueError, KeyError) as error:
                self.cache_allocations[name] = self.resources[name]["adaptive_cache"]["maximum_capacity_bytes"]
                self.cache_unready.add(name)
                reason = getattr(error, "reason_code", type(error).__name__)
                LOG.error("cache rebalance failed model=%s target_bytes=%d reason=%s", name, target, reason)
                raise AdmissionError(503, "expert cache could not safely transfer its reservation", "cache_resize_failed") from error
            self.cache_allocations[name] = info["capacity_bytes"]
            self.cache_unready.discard(name)

            return True, info

        def measured_fit() -> tuple[bool, Memory, Memory]:
            capacity_now, usage_now = self.runtime.capacity()
            reserved_now = self.reserved()
            unknown_now = Memory(
                max(0, usage_now.gpu - reserved_now.gpu),
                max(0, usage_now.host - reserved_now.host),
            )
            return (reserved_now + unknown_now).fits(capacity_now), capacity_now, unknown_now

        async def resize(name: str, target: int) -> tuple[bool, Memory, Memory]:
            limits = self.resources[name]["adaptive_cache"]
            initial = self.cache_allocations.get(name, limits["maximum_capacity_bytes"])
            accepted, info = await issue_resize(name, target)
            actual = info["capacity_bytes"]
            overshot = actual > target
            if overshot:
                # A successful HTTP response is not enough when allocator
                # granularity leaves more bytes reserved than the target.
                accepted = False
            fits, capacity_now, unknown_now = measured_fit()
            if accepted or (
                target < initial and actual <= target
            ) or (
                target > initial and actual == target
            ):
                if actual < target:
                    # The runtime rounds byte budgets down to whole expert
                    # slots. Remember that representable limit for this demand
                    # set so a later pass does not repeat the same resize.
                    remember_fallback_cap(
                        name,
                        min(self.cache_fallback_caps.get(name, actual), actual),
                        target,
                    )
                return fits, capacity_now, unknown_now

            # Keep a failed growth at a verified size; a failed shrink must
            # reach a smaller proven allocation before another model can
            # borrow those bytes.
            if target >= initial and not overshot:
                if target > initial and fits:
                    # The full safe budget can still exceed what the runtime
                    # allocator can represent. Search below that failed target
                    # while this scheduler owns the transition, accounting for
                    # the reported capacity and physical usage after every try.
                    lower = actual
                    upper = target
                    for _ in range(8):
                        candidate = lower + (upper - lower) // 2
                        if candidate <= lower or candidate >= upper:
                            break
                        candidate_accepted, candidate_info = await issue_resize(
                            name, candidate
                        )
                        actual = candidate_info["capacity_bytes"]
                        if actual > candidate:
                            candidate_accepted = False
                        fits, capacity_now, unknown_now = measured_fit()

                        if not candidate_accepted or not fits:
                            if actual > lower:
                                # Failed/oversized probes must not leave a
                                # larger reservation behind while search
                                # continues. Restore the last proven capacity.
                                _restored, restore_info = await issue_resize(
                                    name, lower
                                )
                                if restore_info["capacity_bytes"] != lower:
                                    self.cache_allocations[name] = limits[
                                        "maximum_capacity_bytes"
                                    ]
                                    self.cache_unready.add(name)
                                    raise AdmissionError(
                                        503,
                                        "expert cache could not safely restore its prior reservation",
                                        "cache_resize_failed",
                                    )
                                fits, capacity_now, unknown_now = measured_fit()
                            elif actual < lower:
                                # The worker reports a smaller settled backing;
                                # account for it before considering another try.
                                lower = actual
                            upper = candidate
                            if not fits:
                                remember_fallback_cap(name, lower, target)
                                LOG.warning(
                                    "cache resize capped model=%s requested_bytes=%d capacity_bytes=%d",
                                    name,
                                    target,
                                    lower,
                                )
                                return fits, capacity_now, unknown_now
                            continue

                        if actual > lower:
                            lower = actual
                        else:
                            # The allocator rounded this request to the current
                            # slot count, so no larger capacity is proven here.
                            upper = candidate

                    if self.cache_allocations.get(name) != lower:
                        _restored, restore_info = await issue_resize(name, lower)
                        if restore_info["capacity_bytes"] != lower:
                            self.cache_allocations[name] = limits[
                                "maximum_capacity_bytes"
                            ]
                            self.cache_unready.add(name)
                            raise AdmissionError(
                                503,
                                "expert cache could not safely restore its prior reservation",
                                "cache_resize_failed",
                            )
                    remember_fallback_cap(name, lower, target)
                    fits, capacity_now, unknown_now = measured_fit()
                    LOG.warning(
                        "cache resize capped model=%s requested_bytes=%d capacity_bytes=%d",
                        name,
                        target,
                        lower,
                    )
                    return fits, capacity_now, unknown_now

                remember_fallback_cap(name, actual, target)
                LOG.warning(
                    "cache resize capped model=%s requested_bytes=%d capacity_bytes=%d",
                    name,
                    target,
                    actual,
                )
                return fits, capacity_now, unknown_now

            minimum = limits["minimum_capacity_bytes"]
            lower = minimum
            upper = min(target, actual) if actual < initial else target
            candidate = minimum
            if candidate >= upper:
                LOG.error(
                    "cache shrink rejected model=%s minimum_bytes=%d requested_bytes=%d current_bytes=%d",
                    name,
                    minimum,
                    target,
                    actual,
                )
                raise AdmissionError(503, "expert cache could not safely transfer its reservation", "cache_resize_failed")

            accepted, info = await issue_resize(name, candidate)
            actual = info["capacity_bytes"]
            if actual > target:
                LOG.error(
                    "cache shrink cannot fit target model=%s minimum_bytes=%d requested_bytes=%d capacity_bytes=%d",
                    name, candidate, target, actual,
                )
                raise AdmissionError(503, "expert cache could not safely transfer its reservation", "cache_resize_failed")
            lower = actual

            # Refine the largest verified lower capacity with bounded probes.
            # The failed requested target is the upper bound.
            for _ in range(4):
                candidate = lower + (upper - lower) // 2
                if candidate <= lower or candidate >= upper:
                    break
                _accepted, info = await issue_resize(name, candidate)
                actual = info["capacity_bytes"]
                if lower < actual <= candidate:
                    lower = actual
                else:
                    upper = candidate

            if self.cache_allocations.get(name) != lower:
                _accepted, info = await issue_resize(name, lower)
                actual = info["capacity_bytes"]
                if actual > target:
                    raise AdmissionError(503, "expert cache could not safely transfer its reservation", "cache_resize_failed")
                lower = actual

            remember_fallback_cap(name, lower, target)
            fits, capacity_now, unknown_now = measured_fit()
            LOG.warning(
                "cache resize reduced model=%s requested_bytes=%d capacity_bytes=%d",
                name,
                target,
                lower,
            )
            return fits, capacity_now, unknown_now

        # Shrink, read actual usage again, and recompute before any expansion.
        # A freed logical cache that the allocator still owns stays in `unknown`
        # and cannot be spent on another worker.
        for _ in range(32):
            fixed = fixed_reservation()
            available = capacity - fixed - unknown
            minimum_fits = available.gpu >= 0 and available.host >= 0
            targets = targets_for_demand(max(0, min(available.gpu, available.host)))
            shrinking = [
                name for name in adjustable_names
                if name in self.states
                and self.cache_allocations.get(name, targets[name]) > targets[name]
            ]
            if not shrinking:
                # Recheck the current physical budget before making any growth.
                capacity_now, usage_now = self.runtime.capacity()
                reserved_now = self.reserved()
                unknown = Memory(
                    max(0, usage_now.gpu - reserved_now.gpu),
                    max(0, usage_now.host - reserved_now.host),
                )
                fixed = fixed_reservation()
                available = capacity_now - fixed - unknown
                minimum_fits = available.gpu >= 0 and available.host >= 0
                targets = targets_for_demand(max(0, min(available.gpu, available.host)))
                shrinking = [
                    name for name in adjustable_names
                    if name in self.states
                    and self.cache_allocations.get(name, targets[name]) > targets[name]
                ]
                if shrinking:
                    capacity, usage = capacity_now, usage_now
                    continue
                planned_extra = Memory(0, 0)
                for name in adjustable_names:
                    limits = self.resources[name]["adaptive_cache"]
                    extra = math.ceil(
                        max(0, targets[name] - limits["minimum_capacity_bytes"])
                        * limits["reservation_multiplier"]
                    )
                    planned_extra += Memory(extra, extra)
                if not (fixed + unknown + planned_extra).fits(capacity_now):
                    return False
                for name in sorted(targets):
                    if name not in self.states or name in self.cache_unready:
                        continue
                    current = self.cache_allocations.get(name, targets[name])
                    if targets[name] <= current:
                        continue
                    fits, capacity, unknown = await resize(name, targets[name])
                    if not fits:
                        return False
                return minimum_fits

            for name in sorted(shrinking):
                # A single shrink can leave the physical pool temporarily over
                # budget. Keep releasing every planned cache before deciding
                # whether the transfer can fit; the next pass will recompute
                # capacity from actual usage and refuse any unsafe growth.
                _fits, capacity, unknown = await resize(name, targets[name])
        return False

    async def admit(self, name: str, *, wait_for_idle: bool = False, wait_for_capacity: bool = False) -> None:
        if name not in self.resources:
            raise AdmissionError(404, "model is not configured", "model_not_found")
        operation = ResourceLock(self.lock_dir.parent, "operations.lock")
        if not operation.acquire():
            raise AdmissionError(503, "deployment or host recovery is in progress", "maintenance")
        try:
            for marker_name, reason in (
                ("maintenance.json", "host recovery has pending service work"),
                ("comfyui-restore.json", "ComfyUI restore is still pending"),
            ):
                marker = self.lock_dir.parent / marker_name
                if marker.exists() or marker.is_symlink():
                    raise AdmissionError(503, reason, "maintenance")
            if hasattr(self.runtime, "verify_configuration"):
                try:
                    self.runtime.verify_configuration()
                except (OSError, ValueError, KeyError) as error:
                    raise AdmissionError(503, "memory profiles require deployment validation", "configuration_unready", retry_after=None) from error
            while True:
                try:
                    await self._admit(name, wait_for_idle=wait_for_idle)
                    return
                except AdmissionError as error:
                    if not wait_for_capacity or error.code not in {"capacity_unavailable", "model_transition"}:
                        raise
                    # This waits for admission; no backend inference is retried.
                    await asyncio.sleep(0.1)
        finally:
            operation.close()

    async def _wait_for_idle_worker(self, name: str) -> None:
        """Protect this worker from overlap with its unmanaged backend jobs."""
        while True:
            await self.refresh()
            if name not in self.states:
                return
            if self.states[name] != "ready":
                raise AdmissionError(503, "backend activity is changing", "model_transition")
            busy = await self.runtime.busy(name)
            if busy is None:
                raise AdmissionError(503, "backend activity is unknown", "backend_health_uncertain")
            if not busy:
                return
            await asyncio.sleep(0.1)

    async def _admit(self, name: str, *, wait_for_idle: bool = False) -> None:
        recovery = self.recovery.get(name, {})
        if recovery.get("next_retry", 0) > self.clock():
            raise AdmissionError(503, "backend recovery is cooling down", "backend_recovering", retry_after=max(1, int(recovery["next_retry"] - self.clock())))
        if wait_for_idle:
            await self._wait_for_idle_worker(name)
        if self.transition.locked():
            if self.resources[name].get("adaptive_cache") or any(
                self.resources[resident].get("adaptive_cache")
                for resident in self.states
            ):
                raise AdmissionError(503, "another model is changing shared expert-cache reservations", "model_transition")
            # Existing warm work can proceed while another model loads. Probe
            # directly so a crashed warm model cannot autoload around the budget.
            if self.states.get(name) == "ready" and await self.runtime.ready(name):
                if is_openai_worker(self.resources[name]):
                    busy = await self.runtime.busy(name)
                    if busy is not False:
                        raise AdmissionError(429 if busy is True else 503, "backend work is still active" if busy is True else "backend activity is unknown", "model_busy" if busy is True else "backend_health_uncertain")
                self.last_used[name] = self.clock()
                return
            raise AdmissionError(503, "another model is changing state; retry shortly", "model_transition")
        async with self.transition:
            victim_locks: list[tuple[str, ResourceLock]] = []
            loading = False
            try:
                await self.refresh()
                if wait_for_idle:
                    await self._wait_for_idle_worker(name)
                if name in self.states and self.states[name] != "ready":
                    raise AdmissionError(503, "model is changing state; retry shortly", "model_transition")
                if name in self.states and not await self.runtime.ready(name):
                    # An existing lease may still own backend work. Recovery
                    # must use the idle reaper's exclusive lease and busy probe.
                    raise AdmissionError(503, "resident backend is not ready; retry later", "backend_unready")
                if name in self.states and is_openai_worker(self.resources[name]):
                    busy = await self.runtime.busy(name)
                    if busy is not False:
                        raise AdmissionError(429 if busy is True else 503, "backend work is still active" if busy is True else "backend activity is unknown", "model_busy" if busy is True else "backend_health_uncertain")
                capacity, usage = self.runtime.capacity()
                required = self.minimum_reservation(name, active=True)
                if not required.fits(capacity):
                    raise AdmissionError(503, "model reservation exceeds this host's memory budget", "model_exceeds_budget", retry_after=None)
                reserved = self.reserved()
                unknown = Memory(max(0, usage.gpu - reserved.gpu), max(0, usage.host - reserved.host))
                await self._rebalance_caches(
                    name,
                    capacity,
                    unknown,
                    reserve_incoming_startup=True,
                )
                capacity, usage = self.runtime.capacity()
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
                    if self.states[victim] != "ready" or await self.runtime.busy(victim) is not False:
                        lock.close()
                        continue
                    victim_locks.append((victim, lock))
                    planned = planned - self.reservation(victim)
                if not planned.fits(capacity):
                    raise AdmissionError(503, "insufficient memory while other models are in use; retry later", "capacity_unavailable")
                for victim, _ in victim_locks:
                    LOG.info("evict model=%s last_request_age_s=%.1f", victim, self.clock() - self.last_used[victim])
                    await asyncio.wait_for(self.runtime.stop(victim), self.policy["stop_timeout_s"])
                    self.states.pop(victim, None)
                    self.cache_allocations.pop(victim, None)
                    self.cache_unready.discard(victim)
                    self.cache_fallback_caps.pop(victim, None)
                    self.cache_fallback_demands.pop(victim, None)
                    self.cache_fallback_targets.pop(victim, None)
                # Recheck real allocations after confirmed stops. Unknown GPU
                # users and delayed frees must never become imaginary capacity.
                capacity, usage = self.runtime.capacity()
                reserved = self.reserved()
                unknown = Memory(max(0, usage.gpu - reserved.gpu), max(0, usage.host - reserved.host))
                if not (reserved + unknown + (Memory(0, 0) if name in self.states else required)).fits(capacity):
                    raise AdmissionError(503, "memory is not yet available after eviction", "capacity_unavailable")
                if name not in self.states:
                    loading = True
                    self.states[name] = "starting"
                    if self.resources[name].get("adaptive_cache"):
                        self.cache_allocations[name] = self.resources[name]["adaptive_cache"]["minimum_capacity_bytes"]
                    LOG.info("load model=%s gpu_mib=%.0f host_mib=%.0f context=%s", name, required.gpu / MIB, required.host / MIB, self.resources[name].get("context_tokens", 0))
                    await self.runtime.start(name)
                    self.states[name] = "ready"
                if self.resources[name].get("adaptive_cache"):
                    capacity, usage = self.runtime.capacity()
                    reserved = self.reserved()
                    unknown = Memory(max(0, usage.gpu - reserved.gpu), max(0, usage.host - reserved.host))
                    if not await self._rebalance_caches(name, capacity, unknown):
                        raise AdmissionError(503, "expert-cache allocation could not be safely rebalanced", "capacity_unavailable")
                    capacity, usage = self.runtime.capacity()
                    reserved = self.reserved()
                    unknown = Memory(max(0, usage.gpu - reserved.gpu), max(0, usage.host - reserved.host))
                    if not (reserved + unknown).fits(capacity):
                        raise AdmissionError(503, "memory is not yet available after expert-cache expansion", "capacity_unavailable")
                self.last_used[name] = self.clock()
                self.recovery.pop(name, None)
            except asyncio.CancelledError:
                if loading:
                    cleanup = asyncio.create_task(self._stop_failed_load(name))
                    await finish_shielded(cleanup)
                raise
            except UpstreamError as error:
                if not loading:
                    raise
                await self._stop_failed_load(name)
                LOG.error("admission failed model=%s reason=%s", name, type(error).__name__)
                raise AdmissionError(503, "model process could not become ready", "model_load_failed") from error
            except AdmissionError:
                if loading:
                    await self._stop_failed_load(name)
                raise
            except (aiohttp.ClientError, TimeoutError, OSError, RuntimeError, ValueError, KeyError) as error:
                if loading:
                    await self._stop_failed_load(name)
                LOG.error("admission failed model=%s reason=%s", name, type(error).__name__)
                raise AdmissionError(503, "model process could not become ready", "model_load_failed") from error
            finally:
                for _, lock in victim_locks:
                    lock.close()

    async def _stop_failed_load(self, name: str) -> None:
        try:
            await asyncio.wait_for(self.runtime.stop(name), self.policy["stop_timeout_s"])
            self.states.pop(name, None)
            self.cache_allocations.pop(name, None)
            self.cache_unready.discard(name)
            self.cache_fallback_caps.pop(name, None)
            self.cache_fallback_demands.pop(name, None)
            self.cache_fallback_targets.pop(name, None)
        except Exception:
            self.states[name] = "quarantined"
        failure = self.recovery.setdefault(name, {})
        attempts = failure.get("attempts", 0) + 1
        failure.update(attempts=attempts, state="quarantined" if name in self.states else "cooling-down", last_failure="model_load_failed", next_retry=self.clock() + min(60, 5 * 2 ** min(attempts - 1, 4)))

    async def _rebalance_after_request(self) -> None:
        """Return unused expert slots after demand falls, outside the request path."""
        operation = ResourceLock(self.lock_dir.parent, "operations.lock")
        if not operation.acquire():
            return
        try:
            if any(
                (self.lock_dir.parent / marker).exists()
                or (self.lock_dir.parent / marker).is_symlink()
                for marker in ("maintenance.json", "comfyui-restore.json")
            ):
                return
            async with self.transition:
                await self.refresh()
                if not any(
                    self.resources[name].get("adaptive_cache")
                    for name in self.states
                ):
                    return
                capacity, usage = self.runtime.capacity()
                reserved = self.reserved()
                unknown = Memory(
                    max(0, usage.gpu - reserved.gpu),
                    max(0, usage.host - reserved.host),
                )
                if not await self._rebalance_caches(None, capacity, unknown):
                    LOG.warning("post-request expert-cache rebalance did not fit current capacity")
        except asyncio.CancelledError:
            raise
        except Exception as error:
            LOG.warning("post-request expert-cache rebalance deferred reason=%s", type(error).__name__)
        finally:
            operation.close()

    def _schedule_cache_rebalance(self) -> None:
        if not any(spec.get("adaptive_cache") for spec in self.resources.values()):
            return
        task = asyncio.create_task(self._rebalance_after_request())
        self.cache_rebalance_tasks.add(task)
        task.add_done_callback(self.cache_rebalance_tasks.discard)

    @asynccontextmanager
    async def request(self, name: str):
        spec = self.resources.get(name)
        if spec is None or not is_openai_worker(spec):
            raise AdmissionError(404, "model is not configured", "model_not_found")
        execution_admission = (
            self.execution.hold()
            if spec.get("exclusive_execution") is True
            else self.execution.hold_worker(name)
        )
        rebalance_after = False
        try:
            async with execution_admission as execution:
                limit = self.resources[name].get(
                    "concurrency_limit", self.resources[name].get("concurrency", 1)
                )
                if self.active.get(name, 0) >= limit:
                    raise AdmissionError(429, "model is serving its maximum concurrent requests", "model_busy")
                lock = ResourceLock(self.lock_dir, name)
                if not lock.acquire():
                    raise AdmissionError(503, "model is being evicted; retry shortly", "model_transition")
                self.active[name] = self.active.get(name, 0) + 1
                owned = None
                admitted = False
                try:
                    owned = OwnedRequest(self.lock_dir.parent / "requests", name).begin()
                    # Finish admission cleanup even if its client disappears during a
                    # slow load; otherwise an unaccounted process could keep loading.
                    task = asyncio.create_task(self.admit(name, wait_for_idle=True, wait_for_capacity=True))
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        task.cancel()
                        try:
                            await finish_shielded(task)
                        except asyncio.CancelledError:
                            pass
                        raise
                    admitted = True
                    execution.ready()
                    yield
                except BaseException as error:
                    if owned is not None:
                        if admitted:
                            owned.detach()
                            if isinstance(error, asyncio.CancelledError):
                                await finish_shielded(
                                    asyncio.create_task(self._wait_until_idle(name))
                                )
                        else:
                            # Rejected admission never transferred work to this
                            # request. Keeping its record would grow diagnostics and
                            # disk use during ordinary capacity pressure.
                            owned.close()
                    raise
                else:
                    if owned is not None:
                        owned.close()
                finally:
                    self.active[name] -= 1
                    lock.close()
                    rebalance_after = True
        finally:
            if rebalance_after:
                self._schedule_cache_rebalance()

    async def _wait_until_idle(self, name: str) -> None:
        """Keep this worker's root and eviction lease until cancelled work drains."""
        while True:
            try:
                busy = await self.runtime.busy(name)
            except Exception:
                busy = None
            if busy is False:
                return
            await asyncio.sleep(0.1)

    async def reap(self) -> None:
        if self.transition.locked():
            return
        operation = ResourceLock(self.lock_dir.parent, "operations.lock")
        if not operation.acquire():
            return
        try:
            marker = self.lock_dir.parent / "maintenance.json"
            if marker.exists() or marker.is_symlink():
                return
            async with self.transition:
                await self._reap()
        finally:
            operation.close()

    async def _reap(self) -> None:
            await self.refresh()
            # No resident backend means an abandoned request cannot still own
            # model work. Keep Comfy job manifests so image reconciliation can
            # clean their files after its next generation starts.
            for name in self.resources:
                if name not in self.states:
                    for record in abandoned(self.lock_dir.parent / "requests", name):
                        if not record.get("job_id") and not record.get("client_id"):
                            remove_record(self.lock_dir.parent / "requests", record["id"])
            for name in list(self.states):
                lock = ResourceLock(self.lock_dir, name, exclusive=True)
                try:
                    if not lock.acquire():
                        continue
                    busy = await self.runtime.busy(name)
                    ready = self.states[name] == "ready" and await self.runtime.ready(name)
                    abandoned_work = abandoned(self.lock_dir.parent / "requests", name)
                    ambiguous_comfy_job = name == "comfyui" and any(
                        record.get("kind") == "comfyui_job"
                        and (record.get("upload_pending") is True or (record.get("submitted") is True and record.get("acknowledged") is False))
                        for record in abandoned_work
                    )
                    if ready:
                        self.recovery.pop(name, None)
                        if busy is False:
                            for record in abandoned_work:
                                if not record.get("job_id") and not record.get("client_id"):
                                    remove_record(self.lock_dir.parent / "requests", record["id"])
                        if busy is not False or (not ambiguous_comfy_job and self.clock() - self.last_used[name] < self.policy["idle_timeout_s"]):
                            continue
                    else:
                        failure = self.recovery.setdefault(name, {"attempts": 0})
                        failure["observations"] = failure.get("observations", 0) + 1
                        failure.update(state="recovering", last_failure="backend_unready")
                        if failure["observations"] < self.policy.get("recovery_failures", 3) or failure.get("next_retry", 0) > self.clock():
                            continue
                        failed = await self.runtime.failed(name) if hasattr(self.runtime, "failed") else False
                        if busy is True or (busy is None and not failed):
                            failure.update(state="quarantined", last_failure="backend_health_uncertain")
                            continue
                    await asyncio.wait_for(self.runtime.stop(name), self.policy["stop_timeout_s"])
                    self.states.pop(name, None)
                    self.cache_allocations.pop(name, None)
                    self.cache_unready.discard(name)
                    self.cache_fallback_caps.pop(name, None)
                    self.cache_fallback_demands.pop(name, None)
                    self.cache_fallback_targets.pop(name, None)
                    if not ready:
                        failure["attempts"] += 1
                        failure.update(state="cooling-down", next_retry=self.clock() + min(60, 5 * 2 ** min(failure["attempts"] - 1, 4)))
                        LOG.warning("backend_recovered model=%s action=stop_failed_process", name)
                except (Exception,) as error:
                    self.recovery[name] = {"state": "quarantined", "last_failure": type(error).__name__, "next_retry": self.clock() + 60}
                finally:
                    lock.close()


def create_apps(policy: dict, resources: dict, api_key: str, *, runtime_factory=Runtime) -> tuple[web.Application, web.Application]:
    request_limit = policy.get("public_request_limit", 4)
    if not isinstance(request_limit, int) or isinstance(request_limit, bool) or request_limit < 1:
        raise ValueError("public request limit must be a positive integer")
    public = web.Application(client_max_size=32 * MIB, middlewares=[safe_errors, auth_middleware(api_key), IngressGate(request_limit).middleware])
    local = web.Application(client_max_size=4096, middlewares=[safe_errors])
    state = {}

    async def lifecycle(_: web.Application):
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_read=None, connect=10)) as session:
            runtime = runtime_factory(policy, resources, session, api_key)
            scheduler = Scheduler(resources, runtime, Path(policy["socket_path"]).parent / "leases", policy)
            state.update(session=session, runtime=runtime, scheduler=scheduler)
            async def reaper():
                while True:
                    await asyncio.sleep(15)
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
                background = list(scheduler.cache_rebalance_tasks)
                for rebalance in background:
                    rebalance.cancel()
                await asyncio.gather(task, *background, return_exceptions=True)
    public.cleanup_ctx.append(lifecycle)

    async def health(_: web.Request):
        return web.json_response({"ok": True})

    async def ready(_: web.Request):
        ok = await state["runtime"].healthy()
        return web.json_response({"ok": ok}, status=200 if ok else 503)

    async def status(_: web.Request):
        scheduler = state["scheduler"]
        revision = "unknown"
        process = None
        try:
            process = await asyncio.create_subprocess_exec("git", "-C", str(Path(__file__).resolve().parents[4]), "rev-parse", "HEAD", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
            output, _ = await asyncio.wait_for(process.communicate(), 2)
            if process.returncode == 0:
                revision = output.decode().strip()
        except (OSError, UnicodeError, TimeoutError):
            pass
        finally:
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()
        diagnostics = {}
        for filename in ("boot-status.json", "supervisor.json", "recovery.json", "last-good.json"):
            path = scheduler.lock_dir.parent / filename
            try:
                if path.stat().st_size > 64 * 1024:
                    raise ValueError("oversized diagnostics")
                diagnostics[filename.removesuffix(".json")] = json.loads(path.read_text())
            except FileNotFoundError:
                diagnostics[filename.removesuffix(".json")] = {"state": "uninitialized"}
            except (OSError, ValueError):
                diagnostics[filename.removesuffix(".json")] = {"state": "invalid", "ready": False}
        return web.json_response({"models": {name: {"state": value, "active_requests": scheduler.active.get(name, 0), "last_request_age_s": scheduler.clock() - scheduler.last_used.get(name, scheduler.clock()), **{key: resources[name].get(key) for key in ("gpu_bytes", "host_bytes", "context_tokens", "memory", "artifact_identity", "artifact_fingerprint", "runtime_identity", "supported_request_shapes", "concurrency_limit")}} for name, value in scheduler.states.items()}, "recovery": {name: {**value, "retry_in_s": max(0, value.get("next_retry", 0) - scheduler.clock())} for name, value in scheduler.recovery.items()}, "ownership": [{key: record.get(key) for key in ("id", "resource", "pid", "phase", "boot_id")} for record in records(scheduler.lock_dir.parent / "requests")], "source": revision, "host": diagnostics})

    async def admit(request: web.Request):
        body = await read_json(request)
        name = body.get("resource") if isinstance(body, dict) else None
        wait_for_idle = body.get("wait_for_idle", True) if isinstance(body, dict) else None
        if not isinstance(name, str) or name not in resources or resources[name]["kind"] != "service":
            raise AdmissionError(404, "resource is not configured", "model_not_found")
        if not isinstance(wait_for_idle, bool):
            raise ValueError("wait_for_idle must be a boolean")
        # The adapter holds a shared flock before requesting admission. It
        # remains the active-work signal even if this process restarts.
        for directory in (state["scheduler"].lock_dir, state["scheduler"].lock_dir / "worker-slots"):
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            lock = ResourceLock(directory, name, exclusive=True)
            try:
                if lock.acquire():
                    raise AdmissionError(400, "active resource and worker operation leases are required", "lease_required")
            finally:
                lock.close()
        task = asyncio.create_task(state["scheduler"].admit(name, wait_for_idle=wait_for_idle, wait_for_capacity=True))
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            task.cancel()
            try:
                await finish_shielded(task)
            except asyncio.CancelledError:
                pass
            raise
        generation = await state["runtime"].generation(name) if hasattr(state["runtime"], "generation") else None
        return web.json_response({
            "ok": True,
            "generation": generation,
            "concurrency_limit": resources[name].get("concurrency_limit", 1),
        })

    async def proxy(request: web.Request):
        path, model, body = request.path, None, None
        upstream_path = None
        passthrough_data = None
        if path.startswith("/upstream/"):
            parts = path.split("/", 3)
            if len(parts) < 4:
                raise AdmissionError(404, "unknown inference route", "not_found")
            model = parts[2]
            upstream_path = "/" + parts[3]
            if request.method not in {"GET", "POST"}:
                raise web.HTTPMethodNotAllowed(request.method, {"GET", "POST"})
        elif path in MODEL_ENDPOINTS:
            if request.method != "POST":
                raise web.HTTPMethodNotAllowed(request.method, {"POST"})
            try:
                body = await read_json(request)
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise ValueError("request body must be valid JSON")
            if not isinstance(body, dict) or not isinstance(body.get("model"), str):
                raise ValueError("model must be a string")
            model = body["model"]
        elif path in {"/v1/models", "/models", "/health", "/running", "/metrics", "/logs"}:
            if request.method != "GET":
                raise web.HTTPMethodNotAllowed(request.method, {"GET"})
        else:
            raise AdmissionError(404, "unknown inference route", "not_found")
        if model is not None:
            aliases = {alias: name for name, spec in resources.items() for alias in spec.get("aliases", [])}
            model = aliases.get(model, model)
            if path.startswith("/upstream/") and resources.get(model, {}).get("gateway_url"):
                raise AdmissionError(404, "unknown inference route", "not_found")
            shape_path = upstream_path if path.startswith("/upstream/") else path
            shape_path = LLAMA_CPP_REQUEST_ALIASES.get(shape_path, shape_path)
            if shape_path in MULTIMODAL_REQUEST_ENDPOINTS and model in resources:
                if shape_path in AUDIO_TRANSCRIPTION_ENDPOINTS:
                    raise AdmissionError(
                        400,
                        "worker does not support this request shape",
                        "unsupported_request_shape",
                    )
                shape_body = body
                if shape_body is None:
                    try:
                        async with body_timeout():
                            passthrough_data = await request.read()
                        shape_body = decode_json(passthrough_data)
                    except ValueError as error:
                        raise ValueError("request body must be valid JSON") from error
                    if not isinstance(shape_body, dict):
                        raise ValueError("request body must be a JSON object")
                    # Forward the same duplicate-key interpretation used by
                    # the image guard; downstream parsers must not see a
                    # different body than admission inspected.
                    passthrough_data = json.dumps(
                        shape_body, allow_nan=False, ensure_ascii=False
                    ).encode()
                shape = request_shape(shape_path, shape_body)
                if shape in MULTIMODAL_REQUEST_SHAPES:
                    supported_shapes = resources[model].get("supported_request_shapes", [])
                    if shape not in supported_shapes:
                        raise AdmissionError(
                            400,
                            "worker does not support this request shape",
                            "unsupported_request_shape",
                        )
                    max_images = resources[model].get("max_image_inputs")
                    if max_images is not None and count_image_inputs(shape_body) > max_images:
                        raise AdmissionError(
                            400,
                            "request exceeds this model's image input limit",
                            "image_input_limit_exceeded",
                        )
        content_type = request.headers.get("Content-Type", "application/json")
        if body is not None:
            # Python and llama-swap's JSON parser disagree on duplicate keys;
            # non-JSON content types also let upstream query parameters route.
            # Forward exactly the canonical model whose memory we admitted.
            body["model"] = model
            data = json.dumps(body, allow_nan=False, ensure_ascii=False).encode()
            content_type = "application/json"
        else:
            async with body_timeout():
                data = passthrough_data if passthrough_data is not None else await request.read()
            if passthrough_data is not None:
                content_type = "application/json"
        upstream_url = policy["upstream_url"].rstrip("/")
        direct_spec = resources.get(model) if model is not None else None
        if path in MODEL_ENDPOINTS and direct_spec and direct_spec.get("gateway_url"):
            shape = request_shape(path, body)
            supported_shapes = direct_spec.get("supported_request_shapes", [])
            if shape is None or shape not in supported_shapes:
                raise AdmissionError(400, "worker does not support this request shape", "unsupported_request_shape")
            upstream_url = direct_spec["gateway_url"].rstrip("/")
        async def forward():
            response = None
            pending = None
            streaming = isinstance(body, dict) and body.get("stream") is True

            async def open_stream(headers=None, status=200):
                nonlocal response
                response = web.StreamResponse(status=status, headers=headers or {"Content-Type": "text/event-stream", "Cache-Control": "no-store"})
                response.headers["X-Request-ID"] = request.get("request_id", "")
                await response.prepare(request)
                return response

            async def wait_with_heartbeat(task):
                nonlocal response
                while not task.done():
                    done, _ = await asyncio.wait({task}, timeout=policy.get("heartbeat_s", 15))
                    if not done:
                        if response is None:
                            await open_stream()
                        await response.write(b": ping\n\n")
                return task.result()

            try:
                pending = asyncio.create_task(state["session"].request(request.method, f"{upstream_url}{request.rel_url}", data=data, headers={"Authorization": f"Bearer {api_key}", "Content-Type": content_type, "X-Request-ID": request.get("request_id", "")}, allow_redirects=False))
                upstream = await wait_with_heartbeat(pending) if streaming else await pending
                pending = None
                async with upstream:
                    headers = {key: value for key, value in upstream.headers.items() if key.lower() not in HOP_HEADERS and key.lower() not in {"content-length", "content-encoding"}}
                    headers["X-Request-ID"] = request.get("request_id", "")
                    if upstream.status >= 400:
                        await read_limited(upstream, 1024 * 1024)
                    if "text/event-stream" not in upstream.headers.get("Content-Type", ""):
                        payload = await read_limited(upstream, 32 * MIB, check_status=False)
                        if request.method == "POST":
                            try:
                                result = decode_json(payload)
                                if not isinstance(result, (dict, list)):
                                    raise ValueError("invalid response shape")
                            except (ValueError, UnicodeDecodeError) as error:
                                raise UpstreamError(502, "upstream returned invalid JSON", "upstream_invalid_response") from error
                        if response is not None:
                            raise UpstreamError(502, "upstream did not return a stream", "upstream_invalid_response")
                        return web.Response(body=payload, status=upstream.status, headers=headers)
                    if response is None:
                        await open_stream(headers, upstream.status)
                    fragment = b""
                    while True:
                        pending = asyncio.create_task(upstream.content.readany())
                        chunk = await wait_with_heartbeat(pending)
                        pending = None
                        if not chunk:
                            break
                        # Heartbeats must never split an upstream SSE event.
                        # Assemble bounded frames before sending any of their
                        # bytes so even a quiet mid-frame gap remains safe.
                        framed = fragment + chunk
                        offset = 0
                        for boundary in SSE_BOUNDARY.finditer(framed):
                            end = boundary.end()
                            if end - offset > MAX_SSE_FRAME:
                                raise UpstreamError(502, "upstream stream frame exceeds its byte limit", "upstream_invalid_response")
                            await response.write(framed[offset:end])
                            offset = end
                        fragment = framed[offset:]
                        if len(fragment) > MAX_SSE_FRAME:
                            raise UpstreamError(502, "upstream stream frame exceeds its byte limit", "upstream_invalid_response")
                    if fragment.strip():
                        raise UpstreamError(502, "upstream stream ended with an incomplete event", "upstream_invalid_response")
                    await response.write_eof()
                    return response
            except Exception as error:
                expected = isinstance(error, (TimeoutError, aiohttp.ClientError, AdmissionError, ConnectionError))
                failure = error if isinstance(error, AdmissionError) else AdmissionError(504 if isinstance(error, TimeoutError) else 502 if expected else 500, "upstream transport failed" if expected else "internal server error", "transport_timeout" if isinstance(error, TimeoutError) else "upstream_error" if expected else "internal_error", retry_after=None)
                if response is None or not response.prepared:
                    if not expected:
                        raise
                    raise failure from error
                if not expected:
                    LOG.error("stream_failed id=%s reason=%s", request.get("request_id", ""), type(error).__name__)
                try:
                    async with asyncio.timeout(1):
                        if "text/event-stream" in response.headers.get("Content-Type", ""):
                            frame = {"error": {"message": str(failure), "type": "server_error", "param": None, "code": failure.code}}
                            await response.write(b"data: " + json.dumps(frame).encode() + b"\n\n")
                        await response.write_eof()
                except (TimeoutError, ConnectionError, RuntimeError):
                    response.force_close()
                    if request.transport is not None:
                        request.transport.close()
                return response
            finally:
                if pending is not None:
                    pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)
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
