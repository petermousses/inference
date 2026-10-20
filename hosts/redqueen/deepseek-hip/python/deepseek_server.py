"""Single-session OpenAI chat worker for DeepSeek-V4.1-Flash-1017 on Redqueen."""

from __future__ import annotations

import asyncio
import base64
import binascii
import codecs
import copy
import hmac
import io
import json
import logging
import math
import os
import queue
import re
import secrets
import sys
import threading
import time
import warnings
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace

sys_path_adapters = str(Path(__file__).resolve().parents[2] / "adapters")
if sys_path_adapters not in sys.path:
    sys.path.insert(0, sys_path_adapters)
from redqueen_adapters.scheduler_client import finish_shielded

from aiohttp import web

from checkpoint_mapping_audit import (
    _execute_pinned_model,
    _pinned_inference_imports,
    verify_pinned_inference_sources,
)
from cached_model import install_cached_moe
from checkpoint_reader import SafeTensorCheckpoint
from dense_checkpoint import DenseLoadMemoryEstimate, load_dense_checkpoint
from session_state import (
    SessionStateMemoryEstimate,
    load_session_state,
    plan_session_state,
)


LOG = logging.getLogger("redqueen.deepseek")
MAX_CONTEXT_TOKENS = 220_000
MAX_REQUEST_BYTES = 32 * 1024 * 1024
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
MAX_IMAGES = 8
STREAM_WRITE_TIMEOUT_S = 30
SUPPORTED_IMAGE_MIMES = {"image/png", "image/jpeg", "image/webp"}
DATA_URL = re.compile(r"^data:(image/(?:png|jpeg|webp));base64,(.*)$", re.DOTALL)


def _byte_level_alphabet() -> tuple[dict[int, str], dict[str, int]]:
    visible = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    codepoints = list(visible)
    extra = 0
    for value in range(256):
        if value not in visible:
            visible.append(value)
            codepoints.append(256 + extra)
            extra += 1
    encoder = {value: chr(codepoint) for value, codepoint in zip(visible, codepoints)}
    return encoder, {character: value for value, character in encoder.items()}


_BYTE_LEVEL_ENCODER, _BYTE_LEVEL_DECODER = _byte_level_alphabet()


class RequestError(ValueError):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code = status, code


class RequestCancelled(Exception):
    """The HTTP caller disconnected; the worker still drains and fences normally."""


def _error(status: int, code: str, message: str) -> web.Response:
    return web.json_response(
        {"error": {"message": message, "type": "invalid_request_error", "code": code}},
        status=status,
    )


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RequestError(400, "invalid_request", f"{name} must be a positive integer")
    return value


def _number(value: object, name: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RequestError(400, "invalid_request", f"{name} must be a finite number")
    try:
        value = float(value)
    except OverflowError as error:
        raise RequestError(400, "invalid_request", f"{name} must be a finite number") from error
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise RequestError(400, "invalid_request", f"{name} must be between {minimum:g} and {maximum:g}")
    return value


def _decode_image_url(value: object) -> bytes:
    if isinstance(value, dict):
        if set(value) - {"url", "detail"}:
            raise RequestError(400, "invalid_image", "image_url contains unsupported fields")
        detail = value.get("detail", "auto")
        if detail != "auto":
            raise RequestError(400, "unsupported_parameter", "only automatic image detail is supported")
        value = value.get("url")
    if not isinstance(value, str):
        raise RequestError(400, "invalid_image", "image_url must be an inline data URL")
    match = DATA_URL.fullmatch(value)
    if match is None or match[1] not in SUPPORTED_IMAGE_MIMES:
        raise RequestError(400, "invalid_image", "images must be inline PNG, JPEG, or WebP data URLs")
    payload = match[2]
    if len(payload) > ((MAX_IMAGE_BYTES + 2) // 3) * 4:
        raise RequestError(413, "image_too_large", "image exceeds the 20 MiB limit")
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as error:
        raise RequestError(400, "invalid_image", "image data URL contains invalid base64") from error
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise RequestError(413, "image_too_large", "image must be between 1 byte and 20 MiB")
    try:
        from PIL import Image

        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
            with Image.open(io.BytesIO(data)) as image:
                expected_format = {"image/png": "PNG", "image/jpeg": "JPEG", "image/webp": "WEBP"}[match[1]]
                if image.format != expected_format:
                    raise RequestError(400, "invalid_image", "image format does not match a supported format")
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise RequestError(413, "image_too_large", "image exceeds the 40 megapixel limit")
    except RequestError:
        raise
    except Exception as error:
        raise RequestError(400, "invalid_image", "image cannot be decoded") from error
    return data


def _validate_parameter_schema(
    schema: object,
    *,
    tool_index: int,
    path: str,
    depth: int,
    budget: list[int],
    dialect: str | None = None,
) -> None:
    def invalid(keyword: str) -> None:
        raise RequestError(
            400,
            "invalid_request",
            f"tools[{tool_index}].function.parameters{path} has an invalid {keyword}",
        )

    if depth > 32 or not isinstance(schema, (dict, bool)):
        invalid("schema structure")
    budget[0] += 1
    if budget[0] > 8192:
        invalid("schema size")
    if isinstance(schema, bool):
        return
    if depth == 0 and "$schema" in schema:
        dialect = schema["$schema"]

    schema_type = schema.get("type")
    valid_types = {"array", "boolean", "integer", "null", "number", "object", "string"}
    if "type" in schema:
        if isinstance(schema_type, str):
            if schema_type not in valid_types:
                invalid("type")
        elif (
            not isinstance(schema_type, list)
            or not schema_type
            or any(not isinstance(item, str) or item not in valid_types for item in schema_type)
            or len(set(schema_type)) != len(schema_type)
        ):
            invalid("type")

    required = schema.get("required")
    if "required" in schema and (
        not isinstance(required, list)
        or any(not isinstance(name, str) for name in required)
        or len(set(required)) != len(required)
    ):
        invalid("required")

    enum = schema.get("enum")
    if "enum" in schema and (not isinstance(enum, list) or not enum):
        invalid("enum")

    for keyword in ("title", "description", "$schema", "$id", "$ref", "$anchor", "pattern", "format"):
        if keyword in schema and not isinstance(schema[keyword], str):
            invalid(keyword)
    for keyword in ("deprecated", "readOnly", "writeOnly", "uniqueItems"):
        if keyword in schema and type(schema[keyword]) is not bool:
            invalid(keyword)
    for keyword in (
        "minimum", "maximum", "multipleOf",
        "minLength", "maxLength", "minItems", "maxItems", "minProperties", "maxProperties",
        "minContains", "maxContains",
    ):
        if keyword not in schema:
            continue
        value = schema[keyword]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            invalid(keyword)
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite or (keyword == "multipleOf" and value <= 0):
            invalid(keyword)
        if keyword.startswith(("min", "max")) and (not isinstance(value, int) or value < 0):
            invalid(keyword)

    draft4_schema = dialect == "http://json-schema.org/draft-04/schema#"
    for keyword, bound in (("exclusiveMinimum", "minimum"), ("exclusiveMaximum", "maximum")):
        if keyword not in schema:
            continue
        value = schema[keyword]
        if type(value) is bool:
            if (dialect is not None and not draft4_schema) or bound not in schema:
                invalid(keyword)
            continue
        if draft4_schema or isinstance(value, bool) or not isinstance(value, (int, float)):
            invalid(keyword)
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            invalid(keyword)

    for keyword in ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas"):
        if keyword not in schema:
            continue
        mapping = schema[keyword]
        if not isinstance(mapping, dict):
            invalid(keyword)
        for name, child in mapping.items():
            if not isinstance(name, str):
                invalid(keyword)
            _validate_parameter_schema(
                child,
                tool_index=tool_index,
                path=f"{path}.{keyword}.{name}",
                depth=depth + 1,
                budget=budget,
                dialect=dialect,
            )

    for keyword in (
        "items", "additionalItems", "additionalProperties", "unevaluatedItems",
        "unevaluatedProperties", "propertyNames", "contains", "not", "if", "then",
        "else", "contentSchema",
    ):
        if keyword not in schema:
            continue
        child = schema[keyword]
        if keyword in {
            "additionalItems", "additionalProperties", "unevaluatedItems", "unevaluatedProperties"
        } and type(child) is bool:
            continue
        if keyword == "items" and isinstance(child, list):
            for child_index, item_schema in enumerate(child):
                _validate_parameter_schema(
                    item_schema,
                    tool_index=tool_index,
                    path=f"{path}.items[{child_index}]",
                    depth=depth + 1,
                    budget=budget,
                    dialect=dialect,
                )
            continue
        _validate_parameter_schema(
            child,
            tool_index=tool_index,
            path=f"{path}.{keyword}",
            depth=depth + 1,
            budget=budget,
            dialect=dialect,
        )

    for keyword in ("allOf", "anyOf", "oneOf", "prefixItems"):
        if keyword not in schema:
            continue
        children = schema[keyword]
        if not isinstance(children, list) or not children:
            invalid(keyword)
        for child_index, child in enumerate(children):
            _validate_parameter_schema(
                child,
                tool_index=tool_index,
                path=f"{path}.{keyword}[{child_index}]",
                depth=depth + 1,
                budget=budget,
                dialect=dialect,
            )

    dependent_required = schema.get("dependentRequired")
    if "dependentRequired" in schema:
        if not isinstance(dependent_required, dict):
            invalid("dependentRequired")
        for name, names in dependent_required.items():
            if (
                not isinstance(name, str)
                or not isinstance(names, list)
                or any(not isinstance(required_name, str) for required_name in names)
                or len(set(names)) != len(names)
            ):
                invalid("dependentRequired")


def _validate_tools(tools: object) -> None:
    if tools is None:
        return
    if not isinstance(tools, list) or len(tools) > 128:
        raise RequestError(400, "invalid_request", "tools must be an array with at most 128 items")
    schema_budget = [0]
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict) or not isinstance(tool.get("function"), dict):
            raise RequestError(400, "invalid_request", f"tools[{index}] must contain a function object")
        function = tool["function"]
        if not isinstance(function.get("name"), str) or not function["name"]:
            raise RequestError(400, "invalid_request", f"tools[{index}].function.name must be a non-empty string")
        parameters = function.get("parameters")
        if "parameters" in function:
            if not isinstance(parameters, dict) or parameters.get("type", "object") != "object":
                raise RequestError(400, "invalid_request", f"tools[{index}].function.parameters must be an object schema")
            _validate_parameter_schema(
                parameters,
                tool_index=index,
                path="",
                depth=0,
                budget=schema_budget,
            )
        for description in (
            tool.get("description"),
            function.get("description"),
        ):
            if description is not None and not isinstance(description, str):
                raise RequestError(400, "invalid_request", f"tools[{index}] descriptions must be strings")
        namespaces = []
        for value in (tool.get("namespace"), function.get("namespace")):
            if value is not None and not isinstance(value, (str, dict)):
                raise RequestError(400, "invalid_request", f"tools[{index}] has an invalid namespace")
            if isinstance(value, dict):
                description = value.get("description")
                if description is not None and not isinstance(description, str):
                    raise RequestError(400, "invalid_request", f"tools[{index}] namespace description must be a string")
                value = value.get("name")
                if not isinstance(value, str):
                    raise RequestError(400, "invalid_request", f"tools[{index}] namespace.name must be a string")
            if value is not None and (not value or "::" in value):
                raise RequestError(400, "invalid_request", f"tools[{index}] namespace is invalid")
            namespaces.append(value)
        effective_namespace = namespaces[0] if namespaces[0] is not None else namespaces[1]
        prefix, separator, bare_name = function["name"].partition("::")
        if separator and (
            not prefix
            or not bare_name
            or "::" in bare_name
            or effective_namespace not in (None, prefix)
        ):
            raise RequestError(400, "invalid_request", f"tools[{index}].function.name has invalid namespace qualification")


def _sanitize_messages(messages: object, tools: object) -> tuple[list[dict], int]:
    if not isinstance(messages, list) or not messages or len(messages) > 4096:
        raise RequestError(400, "invalid_request", "messages must be a non-empty array of at most 4096 items")
    _validate_tools(tools)
    cleaned = copy.deepcopy(messages)
    images = 0
    for message in cleaned:
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise RequestError(400, "invalid_request", "each message must be an object with a role")
        if "content_blocks" in message:
            raise RequestError(400, "unsupported_parameter", "content_blocks are not accepted; use OpenAI message content")
        content = message.get("content")
        if not isinstance(content, list):
            if content is not None and not isinstance(content, str):
                raise RequestError(400, "invalid_request", "message content must be text or an array of content parts")
            continue
        for index, part in enumerate(content):
            if not isinstance(part, dict):
                raise RequestError(400, "invalid_request", "message content parts must be objects")
            if part.get("type") == "image_url":
                data = _decode_image_url(part.get("image_url"))
                content[index] = {"type": "image", "data": data}
                images += 1
                if images > MAX_IMAGES:
                    raise RequestError(400, "invalid_image", f"at most {MAX_IMAGES} images are supported per request")
            elif part.get("type") == "text" and isinstance(part.get("text"), str):
                continue
            else:
                raise RequestError(400, "invalid_request", "only text and inline image_url content parts are supported")
    if tools:
        if not cleaned:
            raise RequestError(400, "invalid_request", "tool definitions require at least one message")
        cleaned[0]["tools"] = tools
    return cleaned, images


def _request_options(body: dict) -> dict:
    allowed = {
        "model", "messages", "max_tokens", "max_completion_tokens", "temperature", "stream",
        "tools", "tool_choice", "reasoning_effort", "thinking_mode", "top_p", "n",
        "presence_penalty", "frequency_penalty", "logit_bias", "stop",
    }
    unsupported = sorted(set(body) - allowed)
    if unsupported:
        raise RequestError(400, "unsupported_parameter", f"unsupported request fields: {', '.join(unsupported[:8])}")
    if "model" in body and (not isinstance(body["model"], str) or not body["model"]):
        raise RequestError(400, "invalid_request", "model must be a non-empty string")
    if "max_tokens" in body and "max_completion_tokens" in body:
        raise RequestError(400, "invalid_request", "send only one of max_tokens and max_completion_tokens")
    max_tokens = _positive_integer(body.get("max_completion_tokens", body.get("max_tokens", 256)), "max_tokens")
    if max_tokens > MAX_CONTEXT_TOKENS:
        raise RequestError(400, "context_length_exceeded", "max_tokens exceeds the 220,000-token model limit")
    temperature = _number(body.get("temperature", 1.0), "temperature", minimum=0.0, maximum=2.0)
    top_p = body.get("top_p", 1)
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)):
        raise RequestError(400, "unsupported_parameter", "this worker supports temperature sampling and requires top_p=1")
    try:
        top_p = float(top_p)
    except OverflowError as error:
        raise RequestError(400, "unsupported_parameter", "this worker supports temperature sampling and requires top_p=1") from error
    if not math.isfinite(top_p) or top_p != 1:
        raise RequestError(400, "unsupported_parameter", "this worker supports temperature sampling and requires top_p=1")
    n = body.get("n", 1)
    if type(n) is not int or n != 1:
        raise RequestError(400, "unsupported_parameter", "this worker supports n=1")
    if body.get("presence_penalty", 0) != 0 or body.get("frequency_penalty", 0) != 0:
        raise RequestError(400, "unsupported_parameter", "presence and frequency penalties must be zero")
    if body.get("logit_bias") not in (None, {}):
        raise RequestError(400, "unsupported_parameter", "logit_bias is not supported")
    if body.get("stop") not in (None, [], ""):
        raise RequestError(400, "unsupported_parameter", "stop sequences are not supported")
    if "stream" in body and type(body["stream"]) is not bool:
        raise RequestError(400, "invalid_request", "stream must be a boolean")
    tools = body.get("tools")
    _validate_tools(tools)
    tool_choice = body.get("tool_choice", "auto")
    if tools and tool_choice not in (None, "auto"):
        raise RequestError(400, "unsupported_parameter", "tool_choice must be auto when tools are supplied")
    effort = body.get("reasoning_effort")
    valid_effort = (
        effort is None
        or isinstance(effort, str) and effort in {"low", "high", "max", "none"}
        or type(effort) is int and 1 <= effort <= 100
    )
    if not valid_effort:
        raise RequestError(400, "invalid_request", "reasoning_effort must be low, high, max, none, or an integer from 1 to 100")
    mode = body.get("thinking_mode")
    if mode is None:
        mode = "thinking" if effort not in (None, "none") else "chat"
    if not isinstance(mode, str) or mode not in {"chat", "thinking"}:
        raise RequestError(400, "invalid_request", "thinking_mode must be chat or thinking")
    if mode == "chat":
        effort = None
    elif effort == "none":
        raise RequestError(400, "invalid_request", "thinking_mode=thinking conflicts with reasoning_effort=none")
    elif effort is None:
        effort = "high"
    if body.get("stream", False) and (mode != "chat" or tools):
        raise RequestError(
            400,
            "unsupported_parameter",
            "streaming currently supports chat mode without tool definitions",
        )
    return {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": body.get("stream", False) is True,
        "thinking_mode": mode,
        "reasoning_effort": effort,
        "tools": tools,
    }


def _chunk(response_id: str, created: int, model: str, delta: dict, finish_reason=None) -> bytes:
    payload = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return b"data: " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode() + b"\n\n"


class _IncrementalTextDecoder:
    """Decode byte-level BPE tokens incrementally, retaining incomplete UTF-8 only."""

    def __init__(self, tokenizer):
        backend = getattr(tokenizer, "backend_tokenizer", None)
        decoder = getattr(backend, "decoder", None)
        if decoder is None or type(decoder).__name__ != "ByteLevel":
            raise RuntimeError("streaming requires the pinned byte-level tokenizer decoder")
        self.tokenizer = tokenizer
        self.added_tokens = getattr(tokenizer, "added_tokens_decoder", {})
        self.special_ids = set(getattr(tokenizer, "all_special_ids", ()))
        self.special_ids.update(
            int(token_id)
            for token_id, token in self.added_tokens.items()
            if getattr(token, "special", False)
        )
        self.utf8 = codecs.getincrementaldecoder("utf-8")("replace")

    def put(self, token_id: int) -> str:
        token_id = int(token_id)
        if token_id in self.special_ids:
            return ""
        piece = self.tokenizer.convert_ids_to_tokens(token_id)
        if not isinstance(piece, str):
            raise RuntimeError("pinned byte-level tokenizer returned a non-text token")
        if any(character not in _BYTE_LEVEL_DECODER for character in piece):
            if token_id not in self.added_tokens:
                raise RuntimeError("pinned tokenizer returned a non-byte token outside its added vocabulary")
            pending = self.utf8.decode(b"", final=True)
            self.utf8.reset()
            return pending + piece
        try:
            raw = bytes(_BYTE_LEVEL_DECODER[character] for character in piece)
        except KeyError as error:
            raise RuntimeError("pinned byte-level tokenizer returned a token outside its byte alphabet") from error
        return self.utf8.decode(raw, final=False)

    def finish(self) -> str:
        return self.utf8.decode(b"", final=True)


class DeepSeekEngine:
    """One resident official graph with fenced, request-scoped expert cache state."""

    def __init__(self, policy_path: Path, *, device_name: str = "cuda:0"):
        import kernel
        import torch

        self.kernel, self.torch = kernel, torch
        self.policy_path = policy_path
        self.policy = json.loads(policy_path.read_text())
        service = self.policy.get("services", {}).get("deepseek")
        if not isinstance(service, dict):
            raise RuntimeError("scheduler policy has no DeepSeek worker profile")
        self.service = service
        self.context_tokens = _positive_integer(service.get("context_tokens"), "context_tokens")
        if self.context_tokens != MAX_CONTEXT_TOKENS:
            raise RuntimeError("scheduler and worker context limits disagree")
        self.request_peak_bytes = _positive_integer(service.get("request_peak_mib"), "request_peak_mib") * 1024**2
        if os.environ.get("DS4_REQUEST_PEAK_MIB") != str(service.get("request_peak_mib")):
            raise RuntimeError("systemd request peak does not match scheduler.json")
        self.cache_capacity = _positive_integer(service.get("expert_cache_mib"), "expert_cache_mib") * 1024**2
        self.checkpoint_root = Path(service["artifact_root"]).resolve(strict=True)
        self.inference_dir = Path(service["inference_dir"]).resolve(strict=True)
        self.device = torch.device(device_name)
        torch.cuda.set_device(self.device)
        torch.set_num_threads(8)
        self.memory = {"active_leases": 0}
        self._active_requests = 0
        self._request_guard = asyncio.Lock()
        self._request_active = threading.Event()
        self._closed = False
        self._contexts = ExitStack()
        self._pinned_imports = None
        self._original_rmsnorm = None
        self._original_rope = None
        self._native_usage = {"rmsnorm": 0, "rope": 0}
        self._load_graph()

    def _fence(self) -> None:
        if self.memory["active_leases"] <= 0:
            raise RuntimeError("DeepSeek model work escaped its memory reservation")
        self.torch.cuda.synchronize(self.device)

    def _release_allocator(self) -> None:
        if self.memory["active_leases"] <= 0:
            raise RuntimeError("DeepSeek allocator release escaped its memory reservation")
        self.torch.cuda.empty_cache()

    def _load_graph(self) -> None:
        from run_dense_checkpoint_smoke import (
            _check_admission,
            _combined_startup_estimate,
            _verify_loaded_categories,
        )
        from checkpoint_forward_smoke import _drain_request_state
        from engram_forward import install_chunked_engram_forward
        from engram_rows import install_engram_row_reads
        from expert_cache import FP4ExpertCache
        from expert_weights import estimate_fp4_expert_memory, load_fp4_expert
        from tiled_hyper_connections import install_tiled_hyper_connections
        from tiled_indexer import install_tiled_indexer

        self._check_admission = _check_admission
        self._drain_request_state = _drain_request_state
        adapter_dir = Path(__file__).resolve().parent
        previous_path = sys.path[:]
        sys.path[:0] = [str(adapter_dir), str(self.inference_dir), str(self.inference_dir.parent / "encoding")]
        source_bytes = verify_pinned_inference_sources(self.inference_dir)
        self._pinned_imports = _pinned_inference_imports(source_bytes, self.inference_dir)
        self._pinned_imports.__enter__()
        try:
            from transformers import AutoTokenizer
            import encoding as encoding_module
            from run_dense_checkpoint_smoke import _verify_converted_samples

            self.encoding = encoding_module
            self.reference = _execute_pinned_model(source_bytes["model.py"], self.inference_dir / "model.py")
            self.image_processor = _execute_pinned_model(
                source_bytes["image_processor.py"], self.inference_dir / "image_processor.py"
            )
            config = json.loads(source_bytes["config.json"].decode("utf-8"))
            if not isinstance(config, dict):
                raise RuntimeError("pinned model config must be an object")
            previous_dtype = self.torch.get_default_dtype()
            try:
                self.torch.set_default_dtype(self.torch.bfloat16)
                self.args = self.reference["ModelArgs"](**config)
                self.args.max_batch_size = 1
                self.args.max_seq_len = self.context_tokens
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(self.checkpoint_root), local_files_only=True
                )
                with self.torch.device("meta"):
                    self.model = self.reference["Transformer"](self.args, self.tokenizer)
            finally:
                self.torch.set_default_dtype(previous_dtype)

            self.checkpoint = SafeTensorCheckpoint(
                self.checkpoint_root,
                expected_revision=self.kernel.REFERENCE_COMMIT,
            )
            session_plan = plan_session_state(self.model, self.args)

            class StartupReservation:
                def __init__(inner, estimate):
                    inner.estimate = estimate

                def __enter__(inner):
                    _check_admission(inner.estimate)
                    self.memory["active_leases"] += 1
                    return inner

                def __exit__(inner, exc_type, exc, traceback):
                    self.memory["active_leases"] -= 1
                    if self.memory["active_leases"] < 0:
                        raise RuntimeError("DeepSeek startup lease count underflow")
                    return False

            def admission(estimate):
                if isinstance(estimate, DenseLoadMemoryEstimate):
                    _check_admission(_combined_startup_estimate(estimate, session_plan.estimate))
                elif not isinstance(estimate, SessionStateMemoryEstimate):
                    raise TypeError(f"unsupported DeepSeek startup estimate: {type(estimate).__name__}")
                return StartupReservation(estimate)

            dense_state = load_dense_checkpoint(
                self.checkpoint,
                self.model,
                device=self.device,
                admission=admission,
                fence=self._fence,
                release_allocator=self._release_allocator,
            )
            _verify_loaded_categories(self.model, dense_state.plan, self.torch)
            _verify_converted_samples(
                self.checkpoint, self.model, dense_state.plan, self.device, self.torch
            )

            engram_hash_states = {}

            def reset_runtime_references():
                shared = self.reference["shared_attn"]
                for name in ("compress_kv", "index_k", "topk_idxs", "candidates"):
                    setattr(shared, name, None)
                for module in self.model.modules():
                    if module.__class__.__name__ == "Indexer":
                        module.freqs_cis = None

            def engram_hash_factory(module, model_args, target_device):
                if module.__class__.__name__ != "NgramHashState":
                    raise TypeError("session plan requested a non-pinned Engram hash module")
                with self.torch.device(target_device):
                    result = type(module)(model_args, module.layout, self.tokenizer)
                engram_hash_states[id(module)] = result
                return result

            self.reset_runtime_references = reset_runtime_references
            self.session_state = load_session_state(
                self.model,
                self.args,
                device=self.device,
                frequency_factory=self.reference["precompute_freqs_cis"],
                engram_hash_factory=engram_hash_factory,
                admission=admission,
                fence=self._fence,
                reset_runtime_references=reset_runtime_references,
                release_allocator=self._release_allocator,
            )
            engram_hash_states.clear()

            self.args.temperature = 1.0
            self.model.temperature = 1.0
            self.model.eval()
            self.cache = FP4ExpertCache(
                self.cache_capacity,
                estimate_resident_bytes=lambda key: estimate_fp4_expert_memory(
                    self.checkpoint,
                    key.layer_id,
                    key.expert_id,
                    mtp_layer=key.mtp_layer,
                ).resident_bytes,
                load=lambda key: load_fp4_expert(
                    self.checkpoint,
                    key.layer_id,
                    key.expert_id,
                    device=self.device,
                    mtp_layer=key.mtp_layer,
                ),
            )

            rmsnorm_class = self.reference["RMSNorm"]
            model_globals = rmsnorm_class.forward.__globals__
            self._original_rmsnorm = rmsnorm_class.forward
            self._original_rope = model_globals.get("apply_rotary_emb")
            if not callable(self._original_rope):
                raise RuntimeError("pinned model has no callable apply_rotary_emb")

            def native_rmsnorm(module, values):
                self._native_usage["rmsnorm"] += 1
                return self.kernel.rmsnorm(values, module.weight, module.eps)

            def native_rope(values, freqs_cis, inverse=False):
                self._native_usage["rope"] += 1
                return self.kernel.apply_rotary_emb(values, freqs_cis, inverse)

            rmsnorm_class.forward = native_rmsnorm
            model_globals["apply_rotary_emb"] = native_rope
            self.model_globals = model_globals
            self.rmsnorm_class = rmsnorm_class
            self._contexts.enter_context(
                install_tiled_indexer(self.reference, query_tile_rows=512)
            )
            self._contexts.enter_context(
                install_tiled_hyper_connections(self.reference, row_tile_rows=1024)
            )
            self._contexts.enter_context(install_engram_row_reads(self.model, self.checkpoint))
            self._contexts.enter_context(
                install_chunked_engram_forward(self.model, sequence_tile_rows=256)
            )
            self.installation = self._contexts.enter_context(
                install_cached_moe(self.model, self.cache, self.reference["linear"], fence=self._request_fence)
            )
            self.dense_state = dense_state
        except BaseException:
            try:
                self._contexts.close()
            except BaseException:
                LOG.exception("DeepSeek patch rollback failed during startup")
            if hasattr(self, "checkpoint"):
                self.checkpoint.close()
            if self._pinned_imports is not None:
                self._pinned_imports.__exit__(*sys.exc_info())
                self._pinned_imports = None
            sys.path[:] = previous_path
            raise
        sys.path[:] = previous_path
        os.environ["DS4_HIP_FP8_GEMM_IMPL"] = "exact"

    @contextmanager
    def _gpu_worker_context(self):
        torch = self.torch
        previous_device = torch.cuda.current_device()
        previous_default_device = torch.get_default_device()
        previous_default_dtype = torch.get_default_dtype()
        torch.cuda.set_device(self.device)
        torch.set_default_device(self.device)
        torch.set_default_dtype(torch.bfloat16)
        try:
            yield
        finally:
            torch.set_default_dtype(previous_default_dtype)
            torch.set_default_device(previous_default_device)
            torch.cuda.set_device(previous_device)

    def _request_fence(self) -> None:
        if not getattr(self, "_request_active", threading.Event()).is_set():
            raise RuntimeError("DeepSeek request fence ran outside its request reservation")
        self._fence()

    def _request_release_allocator(self) -> None:
        if not getattr(self, "_request_active", threading.Event()).is_set():
            raise RuntimeError("DeepSeek request allocator release escaped its reservation")
        self._release_allocator()

    def _request_reservation(self):
        engine = self

        class RequestReservation:
            async def __aenter__(inner):
                estimate = SimpleNamespace(
                    resident_device_bytes=engine.cache_capacity,
                    device_peak_bytes=engine.request_peak_bytes,
                    physical_peak_bytes=engine.request_peak_bytes,
                )
                engine._check_admission(estimate)
                torch = engine.torch
                torch.cuda.synchronize(engine.device)
                engine.memory["request_allocated_before"] = torch.cuda.memory_allocated(engine.device)
                engine.memory["request_reserved_before"] = torch.cuda.memory_reserved(engine.device)
                torch.cuda.reset_peak_memory_stats(engine.device)
                engine.memory["active_leases"] += 1
                engine._request_active.set()
                return inner

            async def __aexit__(inner, exc_type, exc, traceback):
                engine._request_active.clear()
                engine.memory["active_leases"] -= 1
                if engine.memory["active_leases"] < 0:
                    raise RuntimeError("DeepSeek request lease count underflow")
                return False

        return RequestReservation()

    def _prepare(self, messages: list[dict], options: dict):
        prompt, media = self.encoding.encode_messages(
            messages,
            thinking_mode=options["thinking_mode"],
            reasoning_effort=options["reasoning_effort"],
            return_multi_modal_data=True,
        )
        tokens, token_types, images = self.image_processor["prepare_vl_inputs"](
            prompt, media["images"], self.tokenizer, self.args
        )
        if not tokens:
            raise RequestError(400, "invalid_request", "prompt encoded to zero tokens")
        if len(tokens) + options["max_tokens"] > self.context_tokens:
            raise RequestError(
                400,
                "context_length_exceeded",
                f"prompt plus max_tokens exceeds the {self.context_tokens}-token context limit",
            )
        if any(isinstance(token, bool) or not isinstance(token, int) or not 0 <= token < self.args.vocab_size for token in tokens):
            raise RuntimeError("pinned tokenizer produced an invalid token id")
        if images is not None and (not isinstance(token_types, list) or len(token_types) != len(tokens)):
            raise RuntimeError("pinned image processor returned mismatched token metadata")
        return tokens, token_types, images

    def _parse_completion(self, token_ids: list[int], options: dict) -> dict:
        text = self.tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        parse_text = text
        eos_token = self.encoding.eos_token
        if not parse_text.endswith(eos_token):
            parse_text += eos_token
        try:
            message = self.encoding.parse_message_from_completion_text(
                parse_text, thinking_mode=options["thinking_mode"]
            )
        except (AssertionError, ValueError, KeyError, IndexError) as error:
            if options["tools"]:
                raise RuntimeError("model returned an incomplete or malformed tool call") from error
            if options["thinking_mode"] == "thinking":
                marker = self.encoding.thinking_end_token
                if marker in text:
                    reasoning, content = text.split(marker, 1)
                    message = {"content": content, "reasoning_content": reasoning, "tool_calls": []}
                else:
                    message = {"content": "", "reasoning_content": text, "tool_calls": []}
            else:
                message = {"content": text, "reasoning_content": "", "tool_calls": []}
        return message

    def _generate(
        self,
        body: dict,
        options: dict,
        cancel: threading.Event,
        stream_tokens: queue.Queue[int] | None = None,
    ) -> dict:
        result = None
        try:
            self.cache.resize_when_idle(self.cache_capacity)
            self.session_state.reset_for_new_request()
            self.reset_runtime_references()
            messages, image_count = _sanitize_messages(body.get("messages"), options["tools"])
            tokens, token_types, images = self._prepare(messages, options)
            torch = self.torch
            self.model.temperature = options["temperature"]
            self.args.temperature = options["temperature"]
            generated: list[int] = []
            stopped_on_eos = False
            with torch.inference_mode():
                input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
                model_kwargs = {}
                if images is not None:
                    model_kwargs["token_types"] = torch.tensor([token_types], dtype=torch.long, device=self.device)
                    model_kwargs["images"] = [images]
                output_ids, logits, main_hidden = self.model(
                    input_ids, start_pos=0, **model_kwargs
                )
                if output_ids.shape != (1,) or logits.shape != (1, self.args.vocab_size):
                    raise RuntimeError("pinned DeepSeek prefill returned an unexpected tensor shape")
                del logits, main_hidden, input_ids, model_kwargs
                next_token = int(output_ids[0].item())
                del output_ids
                for _index in range(options["max_tokens"]):
                    if cancel.is_set():
                        break
                    if next_token == self.tokenizer.eos_token_id:
                        stopped_on_eos = True
                        break
                    if not 0 <= next_token < self.args.vocab_size:
                        raise RuntimeError("pinned DeepSeek sampler produced an invalid token id")
                    generated.append(next_token)
                    if stream_tokens is not None:
                        while not cancel.is_set():
                            try:
                                stream_tokens.put(next_token, timeout=0.1)
                                break
                            except queue.Full:
                                continue
                    if cancel.is_set():
                        break
                    if len(generated) >= options["max_tokens"]:
                        break
                    current = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
                    output_ids, _logits, _main_hidden = self.model(
                        current,
                        start_pos=len(tokens) + len(generated) - 1,
                    )
                    next_token = int(output_ids[0].item())
                    del output_ids, _logits, _main_hidden, current
            if cancel.is_set():
                raise RequestCancelled("DeepSeek request was cancelled after draining GPU work")
            message = self._parse_completion(generated, options)
            result = {
                "prompt_tokens": len(tokens),
                "completion_tokens": len(generated),
                "image_count": image_count,
                "message": message,
                "finish_reason": "tool_calls" if message.get("tool_calls") else "stop" if stopped_on_eos else "length",
            }
        finally:
            self.model_globals["shared_attn"].candidates = None
            if not self.cache.stats.quarantined:
                self._drain_request_state(
                    cache=self.cache,
                    fence=self._request_fence,
                    reset_runtime_references=self.reset_runtime_references,
                    release_allocator=self._request_release_allocator,
                )
        if result is None:
            raise RuntimeError("DeepSeek generation produced no result")
        return result

    async def infer(self, request: web.Request, body: dict, options: dict) -> dict:
        if self._closed:
            raise RequestError(503, "model_unavailable", "DeepSeek worker is shutting down")
        if self._request_guard.locked():
            raise RequestError(429, "model_busy", "DeepSeek is serving its single active request")
        stream_tokens = queue.Queue(maxsize=64) if options["stream"] else None
        decoder = _IncrementalTextDecoder(self.tokenizer) if options["stream"] else None
        await self._request_guard.acquire()
        self._active_requests += 1
        cancel = threading.Event()
        task = asyncio.create_task(
            self.installation.run_reserved_sync(
                lambda: self._generate(body, options, cancel, stream_tokens),
                self._request_reservation(),
                worker_context=self._gpu_worker_context,
            )
        )
        response = None
        try:
            last_heartbeat = time.monotonic()
            response_id = "chatcmpl-" + secrets.token_urlsafe(18)
            created = int(time.time())
            model_name = body.get("model") if isinstance(body.get("model"), str) else "deepseek-v4.1-flash-1017"
            if options["stream"]:
                response = web.StreamResponse(
                    status=200,
                    headers={
                        "Content-Type": "text/event-stream",
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )
                try:
                    await asyncio.wait_for(response.prepare(request), timeout=STREAM_WRITE_TIMEOUT_S)
                except TimeoutError as error:
                    raise ConnectionResetError("DeepSeek streaming client stopped reading") from error
                await self._write_stream_bytes(
                    response, _chunk(response_id, created, model_name, {"role": "assistant"})
                )
            while not task.done():
                await asyncio.wait({task}, timeout=0.1 if decoder is not None else 1)
                if stream_tokens is not None and decoder is not None and response is not None:
                    await self._drain_stream_tokens(
                        stream_tokens, decoder, response, response_id, created, model_name
                    )
                transport = request.transport
                if transport is None or transport.is_closing():
                    cancel.set()
                    try:
                        await finish_shielded(task)
                    except BaseException:
                        pass
                    raise ConnectionResetError("DeepSeek client disconnected")
                if response is not None and time.monotonic() - last_heartbeat >= 15:
                    try:
                        await self._write_stream_bytes(response, b": keep-alive\n\n")
                        last_heartbeat = time.monotonic()
                    except (ConnectionError, OSError):
                        cancel.set()
                        try:
                            await finish_shielded(task)
                        except BaseException:
                            pass
                        raise ConnectionResetError("DeepSeek streaming client disconnected")
            if stream_tokens is not None and decoder is not None and response is not None:
                await self._drain_stream_tokens(
                    stream_tokens, decoder, response, response_id, created, model_name
                )
                await self._write_stream_text(
                    decoder.finish(), response, response_id, created, model_name
                )
            try:
                result = task.result()
            except RequestError as error:
                if response is None:
                    raise
                payload = {"error": {"message": str(error), "type": "invalid_request_error", "code": error.code}}
                await self._write_stream_bytes(response, b"data: " + json.dumps(payload, separators=(",", ":")).encode() + b"\n\n")
                await self._write_stream_bytes(response, b"data: [DONE]\n\n")
                await self._finish_stream(response)
                return response
            except Exception:
                if response is None:
                    raise
                LOG.exception("DeepSeek streaming request failed")
                payload = {"error": {"message": "DeepSeek inference failed", "type": "server_error", "code": "inference_failed"}}
                await self._write_stream_bytes(response, b"data: " + json.dumps(payload, separators=(",", ":")).encode() + b"\n\n")
                await self._write_stream_bytes(response, b"data: [DONE]\n\n")
                await self._finish_stream(response)
                return response
            result.update(id=response_id, created=created, model=model_name)
            if response is not None:
                return await self._write_stream_result(response, result)
            return result
        except asyncio.CancelledError:
            cancel.set()
            try:
                await finish_shielded(task)
            except BaseException:
                pass
            raise
        except ConnectionError:
            if response is None:
                raise
            cancel.set()
            try:
                await finish_shielded(task)
            except BaseException:
                pass
            transport = request.transport
            if transport is not None:
                transport.close()
            return response
        finally:
            if not task.done():
                cancel.set()
                try:
                    await finish_shielded(task)
                except BaseException:
                    pass
            self._active_requests -= 1
            self._request_guard.release()

    async def _drain_stream_tokens(self, tokens, decoder, response, response_id, created, model_name):
        while True:
            try:
                token = tokens.get_nowait()
            except queue.Empty:
                return
            text = decoder.put(token)
            if text:
                await self._write_stream_text(text, response, response_id, created, model_name)

    async def _write_stream_text(self, text, response, response_id, created, model_name):
        if text:
            await self._write_stream_bytes(
                response, _chunk(response_id, created, model_name, {"content": text})
            )

    async def _write_stream_bytes(self, response, payload: bytes) -> None:
        try:
            await asyncio.wait_for(response.write(payload), timeout=STREAM_WRITE_TIMEOUT_S)
        except TimeoutError as error:
            raise ConnectionResetError("DeepSeek streaming client stopped reading") from error

    async def _finish_stream(self, response) -> None:
        try:
            await asyncio.wait_for(response.write_eof(), timeout=STREAM_WRITE_TIMEOUT_S)
        except TimeoutError as error:
            raise ConnectionResetError("DeepSeek streaming client stopped reading") from error

    async def _write_stream_result(self, response: web.StreamResponse, result: dict) -> web.StreamResponse:
        message = result["message"]
        fields = []
        if message.get("tool_calls"):
            calls = []
            for index, call in enumerate(message["tool_calls"]):
                function = call.get("function", {})
                calls.append({
                    "index": index,
                    "id": "call_" + secrets.token_urlsafe(12),
                    "type": "function",
                    "function": {
                        "name": function.get("name", ""),
                        "arguments": function.get("arguments", ""),
                    },
                })
            fields.append({"tool_calls": calls})
        if not fields:
            fields.append({})
        for delta in fields:
            await self._write_stream_bytes(
                response, _chunk(result["id"], result["created"], result["model"], delta)
            )
        await self._write_stream_bytes(
            response,
            _chunk(
                result["id"], result["created"], result["model"], {},
                finish_reason=result["finish_reason"],
            )
        )
        await self._write_stream_bytes(response, b"data: [DONE]\n\n")
        await self._finish_stream(response)
        return response

    async def chat(self, request: web.Request) -> web.Response:
        try:
            body = await request.json(loads=json.loads)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            return _error(400, "invalid_json", "request body must be a JSON object")
        if not isinstance(body, dict):
            return _error(400, "invalid_request", "request body must be a JSON object")
        try:
            options = _request_options(body)
            response = await self.infer(request, body, options)
        except RequestError as error:
            return _error(error.status, error.code, str(error))
        except ConnectionResetError:
            return web.Response(status=499)
        except asyncio.CancelledError:
            raise
        except Exception:
            LOG.exception("DeepSeek request failed")
            return _error(503, "inference_failed", "DeepSeek inference failed; retry only after checking worker state")
        if isinstance(response, web.StreamResponse):
            return response
        message = response["message"]
        if message.get("tool_calls"):
            for call in message["tool_calls"]:
                call.setdefault("id", "call_" + secrets.token_urlsafe(12))
                call.setdefault("type", "function")
            message["tool_calls"] = [
                {key: call[key] for key in ("id", "type", "function")}
                for call in message["tool_calls"]
            ]
        usage = {
            "prompt_tokens": response["prompt_tokens"],
            "completion_tokens": response["completion_tokens"],
            "total_tokens": response["prompt_tokens"] + response["completion_tokens"],
        }
        answer = {
            "id": response["id"],
            "object": "chat.completion",
            "created": response["created"],
            "model": response["model"],
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": response["finish_reason"],
            }],
            "usage": usage,
        }
        if "image_count" in response:
            answer["usage"]["input_images"] = response["image_count"]
        return web.json_response(answer)

    async def close(self) -> None:
        if self._closed:
            return
        if self._active_requests:
            raise RuntimeError("cannot stop DeepSeek while a request still owns the worker")
        self._closed = True
        self.torch.cuda.synchronize(self.device)
        if not self.cache.stats.quarantined:
            if self.cache.stats.entries or self.cache.stats.leased_bytes or self.cache.stats.loading_bytes:
                self.cache.resize_when_idle(0, after_drop=self._release_allocator)
            self.session_state.close()
            self.dense_state.close()
            self.torch.cuda.synchronize(self.device)
            self.torch.cuda.empty_cache()
        self._contexts.close()
        self.checkpoint.close()
        self.rmsnorm_class.forward = self._original_rmsnorm
        self.model_globals["apply_rotary_emb"] = self._original_rope
        if self._pinned_imports is not None:
            self._pinned_imports.__exit__(None, None, None)
            self._pinned_imports = None


def create_app(engine: DeepSeekEngine, api_key: str) -> web.Application:
    if not isinstance(api_key, str) or not api_key:
        raise RuntimeError("QWEN_API_KEY is required")

    @web.middleware
    async def authenticate(request: web.Request, handler):
        value = request.headers.get("Authorization", "")
        if not hmac.compare_digest(value, f"Bearer {api_key}"):
            return _error(401, "unauthorized", "invalid bearer token")
        return await handler(request)

    app = web.Application(client_max_size=MAX_REQUEST_BYTES, middlewares=[authenticate])

    async def health(_request: web.Request):
        return web.json_response({"ok": not engine._closed, "model": "deepseek-v4.1-flash-1017"})

    async def slots(_request: web.Request):
        return web.json_response([{"id": 0, "is_processing": engine._active_requests > 0}])

    async def shutdown(_app: web.Application):
        await engine.close()

    app.router.add_get("/health", health)
    app.router.add_get("/slots", slots)
    app.router.add_post("/v1/chat/completions", engine.chat)
    app.on_cleanup.append(shutdown)
    return app


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    policy_path = Path(os.environ.get("DS4_SCHEDULER_POLICY", "/srv/ai/model-router/scheduler.json"))
    policy = json.loads(policy_path.read_text())
    service = policy.get("services", {}).get("deepseek")
    if not isinstance(service, dict):
        raise SystemExit("scheduler policy has no DeepSeek worker profile")
    api_key = os.environ.get("QWEN_API_KEY", "")
    bind = service.get("bind", "127.0.0.1")
    port = service.get("port", 8085)
    if not api_key or any(character.isspace() for character in api_key):
        raise SystemExit("QWEN_API_KEY is required")
    if bind != "127.0.0.1" or isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise SystemExit("DeepSeek worker must bind to loopback on a valid port")
    os.environ["DS4_HIP_FP8_GEMM_IMPL"] = "exact"
    engine = DeepSeekEngine(policy_path)
    web.run_app(create_app(engine, api_key), host=bind, port=port, access_log=None)


if __name__ == "__main__":
    main()
