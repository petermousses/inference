from __future__ import annotations

import argparse
import asyncio
import base64
import copy
import contextvars
import io
import json
import os
import logging
import re
import secrets
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urlencode

import aiohttp
from aiohttp import web
from aiohttp.multipart import BodyPartReader
from PIL import Image, UnidentifiedImageError

from .common import AdmissionError, IngressGate, UpstreamError, auth_middleware, body_timeout, error_response, read_json, read_limited, safe_errors
from .ownership import OwnedRequest, abandoned, remove_record
from .scheduler_client import SchedulerClient, finish_shielded


MODEL = "qwen-image-2.1"
UNCENSORED_MODEL = "qwen-image-2.1-uncensored"
SUPPORTED_MODELS = (MODEL, UNCENSORED_MODEL)
UNCENSORED_DIFFUSION_MODEL = "qwen-image-2.1-UC-Q4_K_M.gguf"
UNCENSORED_TEXT_ENCODER = "qwen3vl_8b_int8_convrot.safetensors"
STANDARD_DIFFUSION_MODEL = "qwen_image_2.1_bf16.safetensors"
STANDARD_TEXT_ENCODER = "qwen3vl_8b_bf16.safetensors"
GENERATION_SIZES = {"512x512", "1024x1024", "2048x2048"}
EDIT_SIZES = {"512x512", "1024x1024"}
QUALITIES = {"low": 4, "medium": 16, "high": 25, "auto": 25}
BACKGROUNDS = {"auto", "opaque", "transparent"}
MAX_PROMPT_CHARS = 10_000
MAX_TEXT_FIELD_BYTES = 4 * MAX_PROMPT_CHARS
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_REFERENCE_DIMENSION = 4_096
MAX_REFERENCE_PIXELS = 16_777_216
MAX_CLEANUP_ENTRIES = 256
ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp"}
IMAGE_FORMATS = {"image/png": "PNG", "image/jpeg": "JPEG", "image/webp": "WEBP"}
LOG = logging.getLogger("redqueen.image")
_OWNED: contextvars.ContextVar[OwnedRequest | None] = contextvars.ContextVar("image_request", default=None)


@dataclass(frozen=True)
class ImageSettings:
    api_key: str
    comfy_url: str = "http://127.0.0.1:8189"
    workflow_dir: Path = Path("/srv/ai/adapters/workflows")
    comfy_state_dir: Path = Path("/srv/ai/comfyui/state")
    # Retained for callers using the old settings object; inference has no age limit.
    request_timeout_s: float | None = None
    connect_timeout_s: float = 10.0
    upload_timeout_s: float = 60.0
    cleanup_timeout_s: float = 10.0
    minimum_free_bytes: int = 1024**3
    ownership_directory: Path = Path("/srv/ai/model-router/run/requests")
    ingress_limit: int = 2
    poll_interval_s: float = 1.0
    max_active: int = 1
    max_waiting: int = 4
    scheduler_socket: str = ""

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("api key must not be empty")
        if self.max_active < 1 or self.max_waiting < 0:
            raise ValueError("invalid queue limits")
        if self.ingress_limit < 1 or self.minimum_free_bytes < 0 or min(self.connect_timeout_s, self.upload_timeout_s, self.cleanup_timeout_s, self.poll_interval_s) <= 0:
            raise ValueError("invalid image resource limits")


class ImageBackend(Protocol):
    async def generate(self, request: dict) -> bytes: ...
    async def edit(
        self, request: dict, images: list[tuple[str, bytes, str]]
    ) -> bytes: ...


class AdmissionGate:
    def __init__(self, active: int, waiting: int) -> None:
        self._capacity = active
        self._max_waiting = waiting
        self._active = 0
        self._waiting = 0
        self._condition = asyncio.Condition()

    @asynccontextmanager
    async def enter(self):
        async with self._condition:
            if self._active >= self._capacity:
                if self._waiting >= self._max_waiting:
                    raise RuntimeError("image queue is full")
                self._waiting += 1
                try:
                    await self._condition.wait_for(
                        lambda: self._active < self._capacity
                    )
                finally:
                    self._waiting -= 1
            self._active += 1
        try:
            yield
        finally:
            async with self._condition:
                self._active -= 1
                self._condition.notify(1)


def _text(value: object, field: str, *, maximum: int = MAX_PROMPT_CHARS) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    if len(value) > maximum:
        raise ValueError(f"{field} exceeds {maximum} characters")
    return value


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{field} must be an integer")
    if isinstance(value, str) and (
        not value or not value.isascii() or not value.isdecimal()
    ):
        raise ValueError(f"{field} must be an integer")
    parsed = int(value)
    if parsed < 0 or parsed > 2**63 - 1:
        raise ValueError(f"{field} must be between 0 and 2^63-1")
    return parsed


def normalize_request(payload: dict, *, edit: bool) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    allowed = {
        "model",
        "prompt",
        "n",
        "size",
        "quality",
        "background",
        "response_format",
        "seed",
        "user",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unsupported fields: {', '.join(unknown)}")
    model = payload.get("model", MODEL)
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"model must be one of {', '.join(SUPPORTED_MODELS)}")
    if _integer(payload.get("n", 1), "n") != 1:
        raise ValueError("only n=1 is supported")
    sizes = EDIT_SIZES if edit else GENERATION_SIZES
    size = payload.get("size", "1024x1024")
    if not isinstance(size, str) or size not in sizes:
        raise ValueError(f"size must be one of {', '.join(sorted(sizes))}")
    quality = payload.get("quality", "auto")
    if not isinstance(quality, str) or quality not in QUALITIES:
        raise ValueError(f"quality must be one of {', '.join(sorted(QUALITIES))}")
    background = payload.get("background", "auto")
    if not isinstance(background, str) or background not in BACKGROUNDS:
        raise ValueError(f"background must be one of {', '.join(sorted(BACKGROUNDS))}")
    if payload.get("response_format", "b64_json") != "b64_json":
        raise ValueError("only response_format=b64_json is supported")
    normalized = {
        "model": model,
        "prompt": _text(payload.get("prompt"), "prompt"),
        "size": size,
        "quality": quality,
        "background": background,
        "seed": _integer(payload.get("seed", secrets.randbits(63)), "seed"),
        "n": 1,
        "response_format": "b64_json",
    }
    if "user" in payload:
        normalized["user"] = _text(payload["user"], "user", maximum=256)
    return normalized


def _validate_reference(content_type: str, data: bytes) -> None:
    try:
        image = Image.open(io.BytesIO(data))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("reference image is invalid or corrupt") from error
    with image:
        width, height = image.size
        if image.format != IMAGE_FORMATS[content_type]:
            raise ValueError("reference image format does not match its content type")
        if (
            min(width, height) < 16
            or max(width, height) > MAX_REFERENCE_DIMENSION
            or width * height > MAX_REFERENCE_PIXELS
        ):
            raise ValueError(
                "reference image dimensions must be 16-4096 pixels per side and at most "
                f"{MAX_REFERENCE_PIXELS} pixels"
            )
        try:
            image.verify()
        except (OSError, SyntaxError, ValueError) as error:
            raise ValueError("reference image is invalid or corrupt") from error


def _normalize_output(data: bytes, request: dict) -> bytes:
    expected_size = tuple(int(value) for value in request["size"].split("x"))
    if not isinstance(data, bytes) or len(data) > MAX_IMAGE_BYTES:
        raise UpstreamError(502, "image backend returned an oversized image", "upstream_invalid_response")
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "PNG":
                raise UpstreamError(502, "image backend returned a non-PNG image", "upstream_invalid_response")
            if image.size != expected_size:
                raise UpstreamError(502, "image backend returned incorrect dimensions", "upstream_invalid_response")
            image.load()
            alpha = image.getchannel("A") if "A" in image.getbands() else None
            alpha_extrema = alpha.getextrema() if alpha is not None else (255, 255)
            if request["background"] == "opaque" and alpha is not None:
                rgba = image.convert("RGBA")
                flattened = Image.new("RGBA", image.size, (255, 255, 255, 255))
                flattened.alpha_composite(rgba)
                output = io.BytesIO()
                flattened.convert("RGB").save(output, format="PNG")
                data = output.getvalue()
    except (
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        ValueError,
        Image.DecompressionBombError,
    ) as error:
        raise UpstreamError(502, "image backend returned an invalid PNG", "upstream_invalid_response") from error
    if request["background"] == "transparent" and alpha_extrema[0] == 255:
        raise UpstreamError(502, "image backend failed to produce transparency", "upstream_invalid_response")
    return data


async def _read_bounded_part(
    part: BodyPartReader, maximum: int, description: str
) -> bytes:
    data = bytearray()
    while chunk := await part.read_chunk(size=64 * 1024):
        if len(data) + len(chunk) > maximum:
            raise ValueError(f"{description} exceeds {maximum} bytes")
        data.extend(chunk)
    return bytes(data)


async def parse_edit(request: web.Request) -> tuple[dict, list[tuple[str, bytes, str]]]:
    if not request.content_type.startswith("multipart/"):
        raise ValueError("image edits require multipart/form-data")
    fields: dict[str, object] = {}
    images: list[tuple[str, bytes, str]] = []
    reader = await request.multipart()
    async for part in reader:
        if not isinstance(part, BodyPartReader):
            raise ValueError("nested multipart fields are unsupported")
        if part.name == "mask":
            raise ValueError(
                "mask edits are not validated and are intentionally unsupported"
            )
        if part.name in {"image", "image[]"}:
            if len(images) >= 2:
                raise ValueError("at most two reference images are supported")
            content_type = part.headers.get("Content-Type", "").split(";", 1)[0].lower()
            if content_type not in ALLOWED_IMAGE_TYPES:
                raise ValueError("reference images must be PNG, JPEG, or WebP")
            data = await _read_bounded_part(part, MAX_IMAGE_BYTES, "reference image")
            if not data:
                raise ValueError("reference image must not be empty")
            await finish_shielded(asyncio.create_task(asyncio.to_thread(_validate_reference, content_type, data)))
            images.append(
                (part.filename or f"image-{len(images) + 1}", data, content_type)
            )
        elif part.name in {
            "model",
            "prompt",
            "n",
            "size",
            "quality",
            "background",
            "response_format",
            "seed",
            "user",
        }:
            if part.name in fields:
                raise ValueError(f"duplicate field: {part.name}")
            data = await _read_bounded_part(part, MAX_TEXT_FIELD_BYTES, part.name)
            try:
                fields[part.name] = data.decode(part.get_charset(default="utf-8"))
            except (LookupError, UnicodeDecodeError) as error:
                raise ValueError(f"{part.name} must be valid text") from error
        else:
            raise ValueError(f"unsupported multipart field: {part.name}")
    if not images:
        raise ValueError("at least one reference image is required")
    return normalize_request(fields, edit=True), images


class ComfyBackend:
    def __init__(self, settings: ImageSettings) -> None:
        self.settings = settings
        self.generation: str | None = None
        self._cleanup_pending: dict[str, dict] = {}
        self._session: aiohttp.ClientSession | None = None
        self._workflows = {
            "generate": self._load("qwen-image-2.1-t2i-smoke-api.json"),
            "transparent": self._load("qwen-image-2.1-transparency-smoke-api.json"),
            "edit": self._load("qwen-image-2.1-edit-smoke-api.json"),
            "multiref": self._load("qwen-image-2.1-multiref-smoke-api.json"),
        }

    def _load(self, name: str) -> dict:
        path = self.settings.workflow_dir / name
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _select_model(workflow: dict, model: str) -> None:
        if model == MODEL:
            return
        if model != UNCENSORED_MODEL:
            raise RuntimeError("image workflow received an unsupported model")

        diffusion_nodes = [
            node
            for node in workflow.values()
            if isinstance(node, dict) and node.get("class_type") == "UNETLoader"
        ]
        clip_nodes = [
            node
            for node in workflow.values()
            if isinstance(node, dict) and node.get("class_type") == "CLIPLoader"
        ]
        if len(diffusion_nodes) != 1:
            raise RuntimeError("expected exactly one fixed UNETLoader in the workflow")
        if len(clip_nodes) != 1:
            raise RuntimeError("expected exactly one fixed CLIPLoader in the workflow")

        diffusion_inputs = diffusion_nodes[0].get("inputs")
        clip_inputs = clip_nodes[0].get("inputs")
        if (
            not isinstance(diffusion_inputs, dict)
            or diffusion_inputs.get("unet_name") != STANDARD_DIFFUSION_MODEL
        ):
            raise RuntimeError("workflow has an unexpected diffusion model loader")
        if (
            not isinstance(clip_inputs, dict)
            or clip_inputs.get("clip_name") != STANDARD_TEXT_ENCODER
            or clip_inputs.get("type") != "qwen_image"
        ):
            raise RuntimeError("workflow has an unexpected Qwen Image text encoder")

        diffusion_nodes[0]["class_type"] = "UnetLoaderGGUF"
        diffusion_nodes[0]["inputs"] = {"unet_name": UNCENSORED_DIFFUSION_MODEL}
        clip_inputs["clip_name"] = UNCENSORED_TEXT_ENCODER

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()

    def _client(self) -> aiohttp.ClientSession:
        if self._session is None:
            timeout = aiohttp.ClientTimeout(
                total=None, connect=self.settings.connect_timeout_s,
                sock_connect=self.settings.connect_timeout_s, sock_read=None,
            )
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    def check_disk(self) -> None:
        try:
            for area in ("input", "output"):
                path = self.settings.comfy_state_dir / area
                if shutil.disk_usage(path).free < self.settings.minimum_free_bytes:
                    raise AdmissionError(503, "image storage has insufficient free space", "storage_pressure", retry_after=30)
        except OSError as error:
            raise AdmissionError(503, "image storage is unavailable", "storage_unavailable", retry_after=30) from error

    def has_abandoned(self) -> bool:
        return bool(self._cleanup_pending) or any(record.get("kind") == "comfyui_job" for record in abandoned(self.settings.ownership_directory, "comfyui"))

    @staticmethod
    def _decode_json(data: bytes) -> dict:
        try:
            result = json.loads(data)
            if not isinstance(result, dict):
                raise ValueError("response must be an object")
            return result
        except (ValueError, TypeError, UnicodeDecodeError, RecursionError) as error:
            raise UpstreamError(502, "image backend returned invalid metadata", "upstream_invalid_response") from error

    async def _json(self, method: str, path: str, **kwargs) -> dict:
        try:
            async with self._client().request(method, self.settings.comfy_url + path, **kwargs) as response:
                if response.status in {400, 422}:
                    raise UpstreamError(502, "image backend rejected its workflow", "image_backend_rejected")
                return self._decode_json(await read_limited(response, description="image backend"))
        except TimeoutError as error:
            raise UpstreamError(504, "image backend transport timed out", "transport_timeout", retry_after=None) from error
        except aiohttp.ClientError as error:
            raise UpstreamError(503, "image backend connection failed", "upstream_connection_failed", retry_after=1) from error

    async def _control(self, path: str, payload: dict) -> None:
        async with self._client().post(self.settings.comfy_url + path, json=payload) as response:
            await read_limited(response, description="image cleanup")

    def _clean_files(self, record: dict) -> bool:
        request_id = record.get("id")
        if not isinstance(request_id, str) or not re.fullmatch(r"[0-9a-f]{32}", request_id):
            return False
        visited = 0
        def remove(directory: Path) -> bool:
            nonlocal visited
            # scandir iterates without allocating a list of arbitrary size.
            # Retain the manifest and resume on a later pass at the limit.
            with os.scandir(directory) as entries:
                for entry in entries:
                    visited += 1
                    if visited > MAX_CLEANUP_ENTRIES:
                        return False
                    child = Path(entry.path)
                    if entry.is_dir(follow_symlinks=False):
                        if not remove(child):
                            return False
                    else:
                        child.unlink(missing_ok=True)
            directory.rmdir()
            return True
        try:
            for area in ("input", "output"):
                base = (self.settings.comfy_state_dir / area).resolve()
                directory = base / "api" / request_id
                if directory.is_symlink() or not directory.resolve().is_relative_to(base):
                    return False
                if not directory.exists():
                    continue
                if not remove(directory):
                    return False
            return True
        except OSError:
            return False

    async def _terminal(self, record: dict, generation: str | None) -> bool:
        if generation and record.get("backend_generation") and generation != record["backend_generation"]:
            return True
        if record.get("upload_pending"):
            return False
        if not record.get("submitted") or record.get("terminal"):
            return True
        job_id, client_id = record.get("job_id"), record.get("client_id")
        if not isinstance(job_id, str) or not isinstance(client_id, str):
            return False
        observed = bool(record.get("acknowledged"))
        while True:
            queue = await self._json("GET", "/queue")
            pending, running = queue.get("queue_pending"), queue.get("queue_running")
            if not isinstance(pending, list) or not isinstance(running, list):
                raise UpstreamError(502, "image backend returned an invalid queue", "upstream_invalid_response")
            own_pending, own_running = [], []
            for items, own in ((pending, own_pending), (running, own_running)):
                for item in items:
                    if not isinstance(item, list) or len(item) < 4 or not isinstance(item[1], str) or not isinstance(item[3], dict):
                        raise UpstreamError(502, "image backend returned an invalid queue", "upstream_invalid_response")
                    if item[1] == job_id:
                        if item[3].get("client_id") != client_id:
                            return False
                        own.append(item[1])
            if own_pending:
                observed = True
                await self._control("/queue", {"delete": own_pending})
            if own_running:
                observed = True
                await self._control("/interrupt", {"prompt_id": job_id})
            if not own_pending and not own_running:
                if observed:
                    return True
                history = await self._json("GET", f"/history/{job_id}")
                # A UUID supplied before submission gives us a stable lookup,
                # but an absent entry cannot disprove an in-flight submission.
                item = history.get(job_id)
                if not isinstance(item, dict) or not isinstance(item.get("status"), dict):
                    return False
                return item["status"].get("status_str") in ("success", "error")
            await asyncio.sleep(min(0.1, self.settings.poll_interval_s))

    async def _finish(self, owner: OwnedRequest) -> None:
        complete = False
        try:
            async with asyncio.timeout(self.settings.cleanup_timeout_s):
                complete = await self._terminal(owner.record, self.generation)
                if complete:
                    complete = self._clean_files(owner.record)
                    replaced = self.generation and owner.record.get("backend_generation") and self.generation != owner.record["backend_generation"]
                    if complete and owner.record.get("submitted") and not replaced:
                        await self._control("/history", {"delete": [owner.record["job_id"]]})
        except (AdmissionError, aiohttp.ClientError, OSError, TimeoutError, ValueError, TypeError) as error:
            LOG.warning("cleanup_deferred request=%s reason=%s", owner.id, type(error).__name__)
            complete = False
        try:
            if complete:
                owner.close()
                self._cleanup_pending.pop(owner.id, None)
            else:
                self._cleanup_pending[owner.id] = owner.record
                owner.detach()
        except OSError as error:
            self._cleanup_pending[owner.id] = {**owner.record, "phase": "abandoned"}
            LOG.warning("ownership_cleanup_deferred request=%s reason=%s", owner.id, type(error).__name__)

    @asynccontextmanager
    async def _owned_request(self):
        self.check_disk()
        request_id, job_id = secrets.token_hex(16), str(uuid.uuid4())
        try:
            owner = OwnedRequest(self.settings.ownership_directory, "comfyui", request_id, {
                "kind": "comfyui_job", "client_id": job_id, "job_id": job_id,
                "backend_generation": self.generation, "submitted": False,
                "acknowledged": False, "upload_pending": False,
                "input_paths": [], "output_paths": [],
                "output_prefix": f"api/{request_id}",
            }).begin()
        except OSError as error:
            raise AdmissionError(503, "image ownership storage is unavailable", "storage_unavailable", retry_after=30) from error
        token = _OWNED.set(owner)
        try:
            yield owner
        finally:
            try:
                await finish_shielded(asyncio.create_task(self._finish(owner)))
            finally:
                _OWNED.reset(token)

    async def reconcile(self, generation: str | None) -> None:
        """Called only while the adapter holds an exclusive Comfy resource lease."""
        pending = {record["id"]: record for record in abandoned(self.settings.ownership_directory, "comfyui") if record.get("kind") == "comfyui_job"}
        pending.update(self._cleanup_pending)
        try:
            async with asyncio.timeout(self.settings.cleanup_timeout_s):
                for record in pending.values():
                    if not await self._terminal(record, generation) or not self._clean_files(record):
                        raise AdmissionError(503, "abandoned image work cannot yet be reconciled", "cleanup_pending")
                    replaced = generation and record.get("backend_generation") and generation != record["backend_generation"]
                    if record.get("submitted") and not replaced:
                        await self._control("/history", {"delete": [record["job_id"]]})
                    remove_record(self.settings.ownership_directory, record["id"])
                    self._cleanup_pending.pop(record["id"], None)
        except (AdmissionError, aiohttp.ClientError, OSError, TimeoutError, ValueError, TypeError) as error:
            raise AdmissionError(503, "image cleanup is unavailable", "cleanup_pending") from error

    async def ready(self) -> bool:
        try:
            async with self._client().get(f"{self.settings.comfy_url}/system_stats", timeout=aiohttp.ClientTimeout(total=5)) as response:
                return response.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False

    async def generate(self, request: dict) -> bytes:
        async with self._owned_request() as owner:
            return await self._generate(request, owner.id)

    async def _generate(self, request: dict, request_id: str) -> bytes:
        workflow_key = (
            "transparent" if request["background"] == "transparent" else "generate"
        )
        workflow = copy.deepcopy(self._workflows[workflow_key])
        self._select_model(workflow, request["model"])
        width, height = (int(value) for value in request["size"].split("x"))
        prompt = request["prompt"]
        negative = ""
        if request["background"] == "transparent":
            prompt = (
                "This is an RGBA format image with transparency. "
                f"{prompt}. The image has an alpha channel and a fully transparent background. "
                "No floor, no backdrop, no border."
            )
            negative = "opaque background, white background, black background, checkerboard background"
        workflow["4"]["inputs"].update(
            prompt=prompt, negative_prompt=negative, resolution=max(width, height)
        )
        workflow["5"]["inputs"].update(width=width, height=height)
        workflow["7"]["inputs"].update(
            seed=request["seed"], steps=QUALITIES[request["quality"]]
        )
        workflow["9"]["inputs"]["filename_prefix"] = (
            f"api/{request_id}/{request['model']}"
        )
        return await self._run(workflow)

    async def edit(self, request: dict, images: list[tuple[str, bytes, str]]) -> bytes:
        async with self._owned_request() as owner:
            return await self._edit(request, images, owner.id)

    async def _edit(self, request: dict, images: list[tuple[str, bytes, str]], request_id: str) -> bytes:
        uploaded: list[str] = []
        try:
            for image in images:
                uploaded.append(await self._upload(request_id, *image))
            workflow_key = "edit" if len(uploaded) == 1 else "multiref"
            workflow = copy.deepcopy(self._workflows[workflow_key])
            self._select_model(workflow, request["model"])
            if workflow_key == "edit":
                encode_node, sampler_node, output_node = "5", "7", "9"
            else:
                encode_node, sampler_node, output_node = "6", "8", "10"
            for index, uploaded_name in enumerate(uploaded, start=1):
                workflow[str(index)]["inputs"]["image"] = uploaded_name
            resolution = int(request["size"].split("x", 1)[0])
            prompt = request["prompt"]
            negative = ""
            if request["background"] == "transparent":
                prompt += ", preserve the subject while making the background genuinely transparent"
                negative = (
                    "opaque background, white background, checkerboard background"
                )
            workflow[encode_node]["inputs"].update(
                prompt=prompt, negative_prompt=negative, resolution=resolution
            )
            workflow[sampler_node]["inputs"].update(
                seed=request["seed"], steps=QUALITIES[request["quality"]]
            )
            workflow[output_node]["inputs"]["filename_prefix"] = (
                f"api/{request_id}/{request['model']}"
            )
            return await self._run(workflow)
        finally:
            owner = _OWNED.get()
            if owner is None or not owner.record.get("submitted") or owner.record.get("terminal"):
                for uploaded_name in uploaded:
                    uploaded_path = Path(uploaded_name)
                    self._remove_file("input", str(uploaded_path.parent), uploaded_path.name)

    async def _upload(
        self, request_id: str, filename: str, data: bytes, content_type: str
    ) -> str:
        safe_suffix = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/webp": ".webp",
        }[content_type]
        safe_name = f"{secrets.token_hex(8)}{safe_suffix}"
        owner = _OWNED.get()
        if owner is None:
            raise RuntimeError("image upload requires request ownership")
        relative = f"api/{request_id}/{safe_name}"
        owner.update(input_paths=[*owner.record["input_paths"], relative], upload_pending=True)
        form = aiohttp.FormData()
        form.add_field("image", data, filename=safe_name, content_type=content_type)
        form.add_field("subfolder", f"api/{request_id}")
        form.add_field("type", "input")
        form.add_field("overwrite", "false")
        try:
            async with self._client().post(self.settings.comfy_url + "/upload/image", data=form, timeout=aiohttp.ClientTimeout(total=self.settings.upload_timeout_s, connect=self.settings.connect_timeout_s)) as response:
                # The pinned endpoint sends headers only after writing the file.
                # A lost acknowledgment before headers remains ambiguous.
                owner.update(upload_pending=False)
                payload = self._decode_json(await read_limited(response, description="image upload"))
        except TimeoutError as error:
            raise UpstreamError(504, "image upload timed out", "transport_timeout", retry_after=None) from error
        except aiohttp.ClientError as error:
            raise UpstreamError(503, "image upload connection failed", "upstream_connection_failed", retry_after=1) from error
        if (
            payload.get("name") != safe_name
            or payload.get("subfolder") != f"api/{request_id}"
        ):
            raise UpstreamError(502, "image backend returned an invalid upload descriptor", "upstream_invalid_response")
        return relative

    async def _run(self, workflow: dict) -> bytes:
        owner = _OWNED.get()
        if owner is None:
            raise RuntimeError("image submission requires request ownership")
        prompt_id = owner.record["job_id"]
        owner.update(submitted=True)
        try:
            async with self._client().post(self.settings.comfy_url + "/prompt", json={"prompt": workflow, "client_id": owner.record["client_id"], "prompt_id": prompt_id}) as response:
                if response.status in {400, 422}:
                    owner.update(submitted=False)
                    raise UpstreamError(502, "image backend rejected its workflow", "image_backend_rejected")
                payload = self._decode_json(await read_limited(response, description="image submission"))
        except TimeoutError as error:
            raise UpstreamError(504, "image backend transport timed out", "transport_timeout", retry_after=None) from error
        except aiohttp.ClientError as error:
            raise UpstreamError(503, "image backend connection failed", "upstream_connection_failed", retry_after=1) from error
        if payload.get("prompt_id") != prompt_id:
            raise UpstreamError(502, "image backend returned an invalid prompt id", "upstream_invalid_response")
        owner.update(acknowledged=True)
        while True:
            history = await self._json("GET", f"/history/{prompt_id}")
            if prompt_id not in history:
                await asyncio.sleep(self.settings.poll_interval_s)
                continue
            record = history[prompt_id]
            if not isinstance(record, dict) or not isinstance(record.get("status"), dict):
                raise UpstreamError(502, "image backend returned invalid job metadata", "upstream_invalid_response")
            status = record["status"].get("status_str")
            if not isinstance(status, str) or status not in {"success", "error"}:
                raise UpstreamError(502, "image backend returned invalid job status", "upstream_invalid_response")
            owner.update(terminal=True)
            if status != "success":
                raise UpstreamError(502, "image backend job failed", "image_job_failed")
            outputs = record.get("outputs")
            if not isinstance(outputs, dict):
                raise UpstreamError(502, "image backend returned invalid outputs", "upstream_invalid_response")
            descriptors = []
            for output in outputs.values():
                if not isinstance(output, dict) or not isinstance(output.get("images", []), list):
                    raise UpstreamError(502, "image backend returned invalid outputs", "upstream_invalid_response")
                descriptors.extend(output.get("images", []))
            if len(descriptors) != 1 or not isinstance(descriptors[0], dict):
                raise UpstreamError(502, "image backend did not return one image", "upstream_invalid_response")
            descriptor = descriptors[0]
            filename, subfolder = descriptor.get("filename"), descriptor.get("subfolder")
            if descriptor.get("type") != "output" or not isinstance(filename, str) or not filename or Path(filename).name != filename or not isinstance(subfolder, str):
                raise UpstreamError(502, "image backend returned an invalid output descriptor", "upstream_invalid_response")
            relative = Path(subfolder) / filename
            if not relative.is_relative_to(Path("api") / owner.id) or ".." in relative.parts:
                raise UpstreamError(502, "image output escaped its request directory", "upstream_invalid_response")
            owner.update(output_paths=[str(relative)])
            query = urlencode({"filename": filename, "subfolder": subfolder, "type": "output"})
            try:
                async with self._client().get(f"{self.settings.comfy_url}/view?{query}") as response:
                    data = await read_limited(response, limit=MAX_IMAGE_BYTES, description="image output")
            except TimeoutError as error:
                raise UpstreamError(504, "image output transport timed out", "transport_timeout", retry_after=None) from error
            except aiohttp.ClientError as error:
                raise UpstreamError(503, "image output connection failed", "upstream_connection_failed", retry_after=1) from error
            if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                raise UpstreamError(502, "image backend returned a non-PNG image", "upstream_invalid_response")
            return data

    def _remove_file(self, area: str, subfolder: str, filename: str) -> None:
        base = (self.settings.comfy_state_dir / area).resolve()
        candidate = (base / subfolder / filename).resolve()
        if not candidate.is_relative_to(base):
            return
        try:
            candidate.unlink(missing_ok=True)
            parent = candidate.parent
            while parent != base:
                parent.rmdir()
                parent = parent.parent
        except OSError:
            pass


def create_image_app(
    settings: ImageSettings, backend: ImageBackend | None = None
) -> web.Application:
    actual_backend = backend or ComfyBackend(settings)
    scheduler = SchedulerClient(settings.scheduler_socket)
    gate = AdmissionGate(settings.max_active, settings.max_waiting)
    app = web.Application(
        client_max_size=(2 * MAX_IMAGE_BYTES) + (1024 * 1024),
        middlewares=[safe_errors, auth_middleware(settings.api_key), IngressGate(limit=settings.ingress_limit).middleware],
    )

    async def recover() -> None:
        if isinstance(actual_backend, ComfyBackend) and actual_backend.has_abandoned():
            async with scheduler.lease("comfyui", exclusive=True) as admission:
                await actual_backend.reconcile(admission.get("generation"))

    async def maintenance(_: web.Application):
        async def reconcile_loop():
            while True:
                try:
                    await recover()
                except (AdmissionError, OSError) as error:
                    LOG.warning("reconciliation_deferred reason=%s", type(error).__name__)
                await asyncio.sleep(30)
        task = asyncio.create_task(reconcile_loop())
        try:
            yield
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    app.cleanup_ctx.append(maintenance)

    async def health(_: web.Request) -> web.Response:
        return web.json_response({"ok": True, "models": list(SUPPORTED_MODELS)})

    async def ready(_: web.Request) -> web.Response:
        if settings.scheduler_socket:
            ok = await scheduler.ready()
            return web.json_response({"ok": ok}, status=200 if ok else 503)
        check = getattr(actual_backend, "ready", None)
        if check is not None and not await check():
            return web.json_response({"ok": False}, status=503)
        return web.json_response({"ok": True})

    async def models(_: web.Request) -> web.Response:
        return web.json_response(
            {
                "object": "list",
                "data": [
                    {"id": model, "object": "model"} for model in SUPPORTED_MODELS
                ],
            }
        )

    async def run_job(
        operation: Callable[[], Coroutine[None, None, bytes]], normalized: dict
    ) -> web.Response:
        try:
            async with gate.enter():
                await recover()
                if isinstance(actual_backend, ComfyBackend):
                    actual_backend.check_disk()
                async def invoke():
                    if isinstance(actual_backend, ComfyBackend):
                        actual_backend.generation = scheduler.generations.get("comfyui")
                    return await operation()
                image = await scheduler.run("comfyui", invoke)
                image = await finish_shielded(asyncio.create_task(asyncio.to_thread(_normalize_output, image, normalized)))
        except RuntimeError as error:
            if str(error) == "image queue is full":
                response = error_response(429, str(error), "rate_limit_error")
                response.headers["Retry-After"] = "5"
                return response
            raise UpstreamError(502, "image backend failed", "upstream_error") from error
        except TimeoutError as error:
            raise UpstreamError(504, "image backend transport timed out", "transport_timeout", retry_after=None) from error
        except aiohttp.ClientError as error:
            raise UpstreamError(503, "image backend is unavailable", "upstream_connection_failed") from error
        return web.json_response(
            {
                "created": int(time.time()),
                "data": [{"b64_json": base64.b64encode(image).decode()}],
            },
            headers={"Cache-Control": "no-store"},
        )

    async def generations(request: web.Request) -> web.Response:
        payload = await read_json(request)
        normalized = normalize_request(payload, edit=False)
        return await run_job(lambda: actual_backend.generate(normalized), normalized)

    async def edits(request: web.Request) -> web.Response:
        async with body_timeout():
            normalized, images = await parse_edit(request)
        return await run_job(
            lambda: actual_backend.edit(normalized, images), normalized
        )

    async def cleanup(_: web.Application) -> None:
        close = getattr(actual_backend, "close", None)
        if close is not None:
            await close()

    app.router.add_get("/healthz", health)
    app.router.add_get("/readyz", ready)
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/images/generations", generations)
    app.router.add_post("/v1/images/edits", edits)
    app.on_cleanup.append(cleanup)
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="10.9.20.242")
    parser.add_argument("--port", type=int, default=8190)
    args = parser.parse_args()
    api_key = os.environ.get("REDQUEEN_IMAGE_API_KEY", "")
    if not api_key:
        raise SystemExit("REDQUEEN_IMAGE_API_KEY is required")
    scheduler_socket = os.environ.get("REDQUEEN_SCHEDULER_SOCKET", "")
    if not scheduler_socket:
        raise SystemExit("REDQUEEN_SCHEDULER_SOCKET is required")
    web.run_app(
        create_image_app(ImageSettings(api_key=api_key, scheduler_socket=scheduler_socket, max_waiting=0)), host=args.host, port=args.port, handler_cancellation=True
    )


if __name__ == "__main__":
    main()
