from __future__ import annotations

import argparse
import asyncio
import base64
import copy
import io
import json
import os
import secrets
import time
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

from .common import auth_middleware, error_response, safe_errors


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
ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp"}
IMAGE_FORMATS = {"image/png": "PNG", "image/jpeg": "JPEG", "image/webp": "WEBP"}


@dataclass(frozen=True)
class ImageSettings:
    api_key: str
    comfy_url: str = "http://127.0.0.1:8189"
    workflow_dir: Path = Path("/srv/ai/adapters/workflows")
    comfy_state_dir: Path = Path("/srv/ai/comfyui/state")
    request_timeout_s: float = 900.0
    poll_interval_s: float = 1.0
    max_active: int = 1
    max_waiting: int = 4

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("api key must not be empty")
        if self.max_active < 1 or self.max_waiting < 0:
            raise ValueError("invalid queue limits")


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
    if size not in sizes:
        raise ValueError(f"size must be one of {', '.join(sorted(sizes))}")
    quality = payload.get("quality", "auto")
    if quality not in QUALITIES:
        raise ValueError(f"quality must be one of {', '.join(sorted(QUALITIES))}")
    background = payload.get("background", "auto")
    if background not in BACKGROUNDS:
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
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            if image.format != "PNG":
                raise RuntimeError("image backend returned a non-PNG image")
            if image.size != expected_size:
                raise RuntimeError("image backend returned incorrect dimensions")
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
        raise RuntimeError("image backend returned an invalid PNG") from error
    if request["background"] == "transparent" and alpha_extrema[0] == 255:
        raise RuntimeError("image backend failed to produce transparency")
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
            await asyncio.to_thread(_validate_reference, content_type, data)
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
                total=self.settings.request_timeout_s, connect=5
            )
            self._session = aiohttp.ClientSession(
                timeout=timeout, raise_for_status=True
            )
        return self._session

    async def ready(self) -> bool:
        try:
            async with self._client().get(f"{self.settings.comfy_url}/system_stats"):
                return True
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False

    async def generate(self, request: dict) -> bytes:
        request_id = secrets.token_hex(16)
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
        request_id = secrets.token_hex(16)
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
            for uploaded_name in uploaded:
                uploaded_path = Path(uploaded_name)
                self._remove_file(
                    "input", str(uploaded_path.parent), uploaded_path.name
                )

    async def _upload(
        self, request_id: str, filename: str, data: bytes, content_type: str
    ) -> str:
        safe_suffix = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/webp": ".webp",
        }[content_type]
        safe_name = f"{secrets.token_hex(8)}{safe_suffix}"
        form = aiohttp.FormData()
        form.add_field("image", data, filename=safe_name, content_type=content_type)
        form.add_field("subfolder", f"api/{request_id}")
        form.add_field("type", "input")
        form.add_field("overwrite", "false")
        async with self._client().post(
            f"{self.settings.comfy_url}/upload/image", data=form
        ) as response:
            payload = await response.json()
        if (
            payload.get("name") != safe_name
            or payload.get("subfolder") != f"api/{request_id}"
        ):
            raise RuntimeError("ComfyUI returned an unexpected upload descriptor")
        return f"api/{request_id}/{safe_name}"

    async def _run(self, workflow: dict) -> bytes:
        client_id = secrets.token_hex(16)
        prompt_id: str | None = None
        try:
            async with self._client().post(
                f"{self.settings.comfy_url}/prompt",
                json={"prompt": workflow, "client_id": client_id},
            ) as response:
                payload = await response.json()
            prompt_id = payload.get("prompt_id")
            if not isinstance(prompt_id, str) or not prompt_id:
                raise RuntimeError("ComfyUI returned an invalid prompt id")
            deadline = (
                asyncio.get_running_loop().time() + self.settings.request_timeout_s
            )
            while asyncio.get_running_loop().time() < deadline:
                async with self._client().get(
                    f"{self.settings.comfy_url}/history/{prompt_id}"
                ) as response:
                    history = await response.json()
                if prompt_id in history:
                    record = history[prompt_id]
                    if record.get("status", {}).get("status_str") != "success":
                        raise RuntimeError("ComfyUI image job failed")
                    descriptors = [
                        image
                        for output in record.get("outputs", {}).values()
                        for image in output.get("images", [])
                    ]
                    if len(descriptors) != 1:
                        raise RuntimeError("ComfyUI did not return exactly one image")
                    descriptor = descriptors[0]
                    if descriptor.get("type") != "output":
                        raise RuntimeError("ComfyUI returned a non-output image")
                    filename = descriptor.get("filename")
                    subfolder = descriptor.get("subfolder")
                    if not isinstance(filename, str) or not isinstance(subfolder, str):
                        raise RuntimeError(
                            "ComfyUI returned an invalid output descriptor"
                        )
                    query = urlencode(
                        {"filename": filename, "subfolder": subfolder, "type": "output"}
                    )
                    try:
                        async with self._client().get(
                            f"{self.settings.comfy_url}/view?{query}"
                        ) as response:
                            data = await response.read()
                    finally:
                        self._remove_file("output", subfolder, filename)
                    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                        raise RuntimeError("ComfyUI returned a non-PNG image")
                    return data
                await asyncio.sleep(self.settings.poll_interval_s)
            raise TimeoutError("ComfyUI image job timed out")
        except asyncio.CancelledError:
            if prompt_id is not None:
                await asyncio.shield(self._cancel(prompt_id))
            raise
        except (TimeoutError, aiohttp.ClientError):
            if prompt_id is not None:
                await self._cancel(prompt_id)
            raise

    async def _cancel(self, prompt_id: str) -> None:
        try:
            async with self._client().get(
                f"{self.settings.comfy_url}/queue"
            ) as response:
                queue = await response.json()
            pending = {
                item[1] for item in queue.get("queue_pending", []) if len(item) > 1
            }
            running = {
                item[1] for item in queue.get("queue_running", []) if len(item) > 1
            }
            if prompt_id in pending:
                async with self._client().post(
                    f"{self.settings.comfy_url}/queue", json={"delete": [prompt_id]}
                ):
                    pass
            elif prompt_id in running:
                async with self._client().post(f"{self.settings.comfy_url}/interrupt"):
                    pass
        except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, TypeError):
            pass

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
    gate = AdmissionGate(settings.max_active, settings.max_waiting)
    app = web.Application(
        client_max_size=(2 * MAX_IMAGE_BYTES) + (1024 * 1024),
        middlewares=[safe_errors, auth_middleware(settings.api_key)],
    )

    async def health(_: web.Request) -> web.Response:
        return web.json_response({"ok": True, "models": list(SUPPORTED_MODELS)})

    async def ready(_: web.Request) -> web.Response:
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
                image = await operation()
                image = await asyncio.to_thread(_normalize_output, image, normalized)
        except RuntimeError as error:
            if str(error) == "image queue is full":
                response = error_response(429, str(error), "rate_limit_error")
                response.headers["Retry-After"] = "5"
                return response
            return error_response(502, "image backend failed", "upstream_error")
        except TimeoutError:
            return error_response(504, "image backend timed out", "upstream_error")
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return error_response(502, "image backend is unavailable", "upstream_error")
        return web.json_response(
            {
                "created": int(time.time()),
                "data": [{"b64_json": base64.b64encode(image).decode()}],
            },
            headers={"Cache-Control": "no-store"},
        )

    async def generations(request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except (json.JSONDecodeError, aiohttp.ContentTypeError):
            raise ValueError("request body must be valid JSON")
        normalized = normalize_request(payload, edit=False)
        return await run_job(lambda: actual_backend.generate(normalized), normalized)

    async def edits(request: web.Request) -> web.Response:
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
    web.run_app(
        create_image_app(ImageSettings(api_key=api_key)), host=args.host, port=args.port
    )


if __name__ == "__main__":
    main()
