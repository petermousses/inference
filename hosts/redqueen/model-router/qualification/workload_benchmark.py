#!/usr/bin/env python3
"""Run a frozen JSONL workload suite through Redqueen's OpenAI-compatible gateway.

The suite selects only routes already exposed by the Redqueen gateway. Results
are private run artifacts and must be written outside the source checkout.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import datetime as dt
import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


SCHEMA_VERSION = 1
MAX_CASES = 100_000
MAX_REQUEST_BYTES = 64 * 1024 * 1024
MAX_RESPONSE_BYTES = 64 * 1024 * 1024
MAX_IMAGE_BYTES = 32 * 1024 * 1024
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
FIELD_PATTERN = re.compile(r"^[A-Za-z0-9_.\[\]-]{1,64}$")
CONTENT_TYPE_PATTERN = re.compile(r"^[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+$")
ROUTES = {
    "text": {"/v1/chat/completions", "/v1/completions", "/v1/responses"},
    "prefill": {"/v1/chat/completions"},
    "multimodal": {"/v1/chat/completions", "/v1/responses"},
    "embeddings": {"/v1/embeddings"},
    "decisions": {"/v1/decisions", "/v1/systemone"},
    "clm": {"/v1/decisions", "/v1/systemone"},
    "images": {"/v1/images/generations", "/v1/images/edits"},
}


class SuiteError(ValueError):
    pass


class InvalidJSON(ValueError):
    pass


class ResponseTooLarge(ValueError):
    pass


SERIAL_WORKLOADS = {"clm", "images"}


def json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise InvalidJSON(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str):
    raise InvalidJSON(f"non-finite JSON number: {value}")


def strict_json_loads(value: str | bytes) -> object:
    try:
        return json.loads(value, object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise InvalidJSON("invalid JSON encoding") from error


def read_suite(path: Path) -> tuple[list[dict], str]:
    raw = path.read_bytes()
    cases: list[dict] = []
    seen: set[str] = set()
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = strict_json_loads(line)
        except InvalidJSON as error:
            raise SuiteError(f"line {line_number}: invalid JSON") from error
        if (
            not isinstance(case, dict)
            or isinstance(case.get("schema_version"), bool)
            or case.get("schema_version") != SCHEMA_VERSION
        ):
            raise SuiteError(f"line {line_number}: schema_version must be {SCHEMA_VERSION}")
        case_id = case.get("id")
        suite_id = case.get("suite_id")
        workload = case.get("workload")
        path_value = case.get("path")
        request = case.get("request")
        if not isinstance(case_id, str) or not ID_PATTERN.fullmatch(case_id) or case_id in seen:
            raise SuiteError(f"line {line_number}: id must be unique and match {ID_PATTERN.pattern}")
        if not isinstance(suite_id, str) or not ID_PATTERN.fullmatch(suite_id):
            raise SuiteError(f"line {line_number}: suite_id must be a stable identifier")
        if cases and suite_id != cases[0]["suite_id"]:
            raise SuiteError(f"line {line_number}: suite_id differs from the first case")
        if (
            not isinstance(workload, str)
            or workload not in ROUTES
            or not isinstance(path_value, str)
            or path_value not in ROUTES[workload]
        ):
            raise SuiteError(f"line {line_number}: unsupported workload/path pair")
        if not isinstance(request, dict):
            raise SuiteError(f"line {line_number}: request must be an object")
        if not isinstance(request.get("model"), str) or not request["model"]:
            raise SuiteError(f"line {line_number}: request.model must be a non-empty string")
        try:
            encoded = json_bytes(request)
        except (TypeError, ValueError) as error:
            raise SuiteError(f"line {line_number}: request must be finite JSON") from error
        if len(encoded) > MAX_REQUEST_BYTES:
            raise SuiteError(f"line {line_number}: request exceeds {MAX_REQUEST_BYTES} bytes")
        files = case.get("files", [])
        if not isinstance(files, list) or (files and path_value != "/v1/images/edits"):
            raise SuiteError(f"line {line_number}: files are supported only for image edits")
        if path_value == "/v1/images/edits" and not files:
            raise SuiteError(f"line {line_number}: image edits need at least one file")
        stream_value = request.get("stream")
        if stream_value is not None and not isinstance(stream_value, bool):
            raise SuiteError(f"line {line_number}: stream must be boolean")
        if stream_value is True and workload not in {"text", "prefill", "multimodal"}:
            raise SuiteError(f"line {line_number}: streaming is supported only for text generation")
        for item in files:
            if not isinstance(item, dict) or set(item) != {"field", "asset", "filename", "content_type", "sha256"}:
                raise SuiteError(f"line {line_number}: each file needs field, asset, filename, content_type, and sha256")
            if not all(isinstance(item[key], str) and item[key] for key in item):
                raise SuiteError(f"line {line_number}: file metadata must be non-empty strings")
            if not FIELD_PATTERN.fullmatch(item["field"]):
                raise SuiteError(f"line {line_number}: multipart field name is invalid")
            if (
                Path(item["filename"]).name != item["filename"]
                or any(character in item["filename"] for character in '\"\\\r\n')
            ):
                raise SuiteError(f"line {line_number}: multipart filename is invalid")
            if not CONTENT_TYPE_PATTERN.fullmatch(item["content_type"]):
                raise SuiteError(f"line {line_number}: multipart content_type is invalid")
            if not re.fullmatch(r"[a-f0-9]{64}", item["sha256"]):
                raise SuiteError(f"line {line_number}: multipart asset sha256 is invalid")
            if Path(item["asset"]).is_absolute() or ".." in Path(item["asset"]).parts:
                raise SuiteError(f"line {line_number}: file asset must stay inside the suite directory")
        score = case.get("scoring")
        if score is not None:
            validate_scoring(score, line_number)
        elif workload != "images":
            raise SuiteError(f"line {line_number}: non-image cases need a scoring rule")
        seen.add(case_id)
        cases.append(case)
        if len(cases) > MAX_CASES:
            raise SuiteError(f"suite exceeds {MAX_CASES} cases")
    if not cases:
        raise SuiteError("suite contains no cases")
    return cases, hashlib.sha256(raw).hexdigest()


def validate_scoring(score: object, line_number: int = 0) -> None:
    prefix = f"line {line_number}: " if line_number else ""
    if not isinstance(score, dict):
        raise SuiteError(prefix + "scoring must be an object")
    kind = score.get("kind")
    if kind == "exact":
        if not isinstance(score.get("pointer"), str) or not score["pointer"].startswith("/"):
            raise SuiteError(prefix + "exact scoring needs a JSON pointer")
        if re.search(r"~(?![01])", score["pointer"]):
            raise SuiteError(prefix + "exact scoring JSON pointer has an invalid escape")
        if "expected" not in score:
            raise SuiteError(prefix + "exact scoring needs expected")
        if any(not isinstance(score.get(flag, False), bool) for flag in ("casefold", "strip")):
            raise SuiteError(prefix + "casefold and strip must be booleans")
        try:
            json_bytes(score["expected"])
        except (TypeError, ValueError) as error:
            raise SuiteError(prefix + "exact expected value must be finite JSON") from error
    elif kind == "retrieval":
        relevant = score.get("relevant")
        if not isinstance(relevant, list) or not relevant:
            raise SuiteError(prefix + "retrieval scoring needs a non-empty relevant index list")
        if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in relevant):
            raise SuiteError(prefix + "relevant indices must be nonnegative integers")
        query_index = score.get("query_index", 0)
        if isinstance(query_index, bool) or not isinstance(query_index, int) or query_index < 0:
            raise SuiteError(prefix + "query_index must be a nonnegative integer")
        metric = score.get("metric", "mrr")
        if not isinstance(metric, str) or metric not in {"mrr", "recall_at_k", "ndcg_at_k"}:
            raise SuiteError(prefix + "retrieval metric must be mrr, recall_at_k, or ndcg_at_k")
        if metric != "mrr":
            k = score.get("k")
            if isinstance(k, bool) or not isinstance(k, int) or k < 1:
                raise SuiteError(prefix + "recall_at_k and ndcg_at_k need a positive k")
    else:
        raise SuiteError(prefix + "scoring kind must be exact or retrieval")


def json_pointer(value: object, pointer: str) -> object:
    current = value
    for raw_part in pointer.lstrip("/").split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            if part not in current:
                raise KeyError(pointer)
            current = current[part]
        elif isinstance(current, list):
            if not part.isascii() or not part.isdecimal():
                raise KeyError(pointer)
            index = int(part)
            if index >= len(current):
                raise KeyError(pointer)
            current = current[index]
        else:
            raise KeyError(pointer)
    return current


def vector(value: object) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError("embedding must be a non-empty numeric array")
    output: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            raise ValueError("embedding contains a non-finite or non-numeric value")
        output.append(float(item))
    return output


def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions do not match")
    left_scale = max(abs(value) for value in left)
    right_scale = max(abs(value) for value in right)
    if left_scale == 0 or right_scale == 0:
        raise ValueError("zero-length embedding cannot be ranked")
    left_norm = math.sqrt(sum((value / left_scale) ** 2 for value in left))
    right_norm = math.sqrt(sum((value / right_scale) ** 2 for value in right))
    dot = sum(
        (left_value / left_scale) * (right_value / right_scale)
        for left_value, right_value in zip(left, right, strict=True)
    )
    return dot / (left_norm * right_norm)


def numeric_timings(value: object) -> dict:
    if not isinstance(value, dict):
        return {}
    allowed = {"prompt_n", "prompt_ms", "predicted_n", "predicted_ms"}
    return {
        key: raw
        for key, raw in value.items()
        if key in allowed
        and isinstance(raw, (int, float))
        and not isinstance(raw, bool)
        and math.isfinite(raw)
        and raw >= 0
    }


def score_case(case: dict, response: object) -> dict | None:
    scoring = case.get("scoring")
    if scoring is None:
        return None
    if scoring["kind"] == "exact":
        actual = json_pointer(response, scoring["pointer"])
        expected = scoring["expected"]
        if isinstance(actual, str) and isinstance(expected, str):
            if scoring.get("strip", True):
                actual, expected = actual.strip(), expected.strip()
            if scoring.get("casefold", True):
                actual, expected = actual.casefold(), expected.casefold()
        same_type = type(actual) is type(expected) or (
            isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and isinstance(expected, (int, float)) and not isinstance(expected, bool)
        )
        return {"metric": "exact", "value": float(same_type and actual == expected)}

    data = response.get("data") if isinstance(response, dict) else None
    if not isinstance(data, list):
        raise ValueError("embedding response is missing data")
    requested_input = case["request"].get("input")
    if not isinstance(requested_input, list) or len(data) != len(requested_input):
        raise ValueError("embedding response count does not match the request")
    indexed = [item.get("index") if isinstance(item, dict) else None for item in data]
    if any(index is not None for index in indexed):
        if (
            any(isinstance(index, bool) or not isinstance(index, int) for index in indexed)
            or sorted(indexed) != list(range(len(data)))
        ):
            raise ValueError("embedding response indices must be unique and contiguous")
        data = [item for _, item in sorted(zip(indexed, data, strict=True), key=lambda pair: pair[0])]
    query_index = scoring.get("query_index", 0)
    if query_index >= len(data) or not isinstance(data[query_index], dict):
        raise ValueError("embedding response has no query vector")
    query = vector(data[query_index].get("embedding"))
    candidates: list[tuple[int, float]] = []
    relevant = set(scoring["relevant"])
    # Suite relevance indices are zero-based document positions after the query.
    document_position = 0
    for output_index, item in enumerate(data):
        if output_index == query_index:
            continue
        if not isinstance(item, dict):
            raise ValueError("embedding response data item is malformed")
        candidates.append((document_position, cosine(query, vector(item.get("embedding")))))
        document_position += 1
    if any(index >= len(candidates) for index in relevant):
        raise ValueError("relevant document index is outside the returned embedding batch")
    ranked = [index for index, _ in sorted(candidates, key=lambda pair: (-pair[1], pair[0]))]
    reciprocal_rank = next((1 / (rank + 1) for rank, index in enumerate(ranked) if index in relevant), 0.0)
    metric = scoring.get("metric", "mrr")
    if metric == "mrr":
        value = reciprocal_rank
    elif metric == "recall_at_k":
        value = float(bool(set(ranked[: scoring["k"]]) & relevant))
    else:
        value = sum(1 / math.log2(rank + 2) for rank, index in enumerate(ranked[: scoring["k"]]) if index in relevant)
        ideal = sum(1 / math.log2(rank + 2) for rank in range(min(len(relevant), scoring["k"])))
        value = value / ideal if ideal else 0.0
    return {"metric": metric, "value": float(value)}


def sanitized_url(value: str) -> str:
    parts = urlsplit(value)
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    if parts.port:
        host += f":{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def validate_origin(value: str) -> None:
    parts = urlsplit(value)
    try:
        port = parts.port
    except ValueError as error:
        raise ValueError("--url must include a valid port") from error
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or (port is not None and not 1 <= port <= 65535)
        or parts.path not in {"", "/"}
        or parts.query
        or parts.fragment
    ):
        raise ValueError("--url must be an HTTP(S) API origin without credentials, path, query, or fragment")


def endpoint_name(workload: str) -> str:
    if workload in {"decisions", "clm"}:
        return "decisions"
    if workload == "images":
        return "images"
    return "model"


def resolve_endpoints(
    workloads: set[str], *, model_url: str, decisions_url: str, images_url: str,
    model_key_env: str, decisions_key_env: str, images_key_env: str,
    environment: dict[str, str] | None = None,
) -> dict[str, tuple[str, str | None]]:
    environment = os.environ if environment is None else environment
    settings = {
        "model": (model_url, model_key_env),
        "decisions": (decisions_url, decisions_key_env),
        "images": (images_url, images_key_env),
    }
    endpoints = {}
    for name in {endpoint_name(workload) for workload in workloads}:
        url, key_name = settings[name]
        validate_origin(url)
        api_key = environment.get(key_name) if key_name else None
        if key_name and not api_key:
            raise ValueError(f"missing bearer key in environment variable {key_name}")
        endpoints[name] = (url, api_key)
    return endpoints


def ordered_model_groups(cases: list[dict], seed: int) -> list[tuple[str, list[dict]]]:
    groups: dict[str, list[dict]] = {}
    for case in cases:
        groups.setdefault(case["request"]["model"], []).append(case)
    rng = random.Random(seed)
    models = list(groups)
    rng.shuffle(models)
    ordered = []
    for model in models:
        rng.shuffle(groups[model])
        ordered.append((model, groups[model]))
    return ordered


def ordered_cases(cases: list[dict], seed: int) -> list[dict]:
    return [case for _, group in ordered_model_groups(cases, seed) for case in group]


def serial_group(case: dict) -> str | None:
    workload = case["workload"]
    if workload in SERIAL_WORKLOADS:
        return workload
    if workload == "decisions" and case["request"]["model"].startswith("clef-"):
        return "clef-decisions"
    return None


def sse_event_records(response, started: float) -> tuple[str, dict, float | None, dict]:
    text_parts: list[str] = []
    usage: dict = {}
    timings = {}
    first_content_s = None
    bytes_read = 0
    terminal = False
    while True:
        line = response.readline(MAX_RESPONSE_BYTES - bytes_read + 1)
        if not line:
            break
        bytes_read += len(line)
        if bytes_read > MAX_RESPONSE_BYTES:
            raise ResponseTooLarge("stream response exceeds size limit")
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload == b"[DONE]":
            terminal = True
            break
        if not payload:
            continue
        try:
            event = strict_json_loads(payload)
        except InvalidJSON as error:
            raise InvalidJSON("stream contains malformed JSON event") from error
        if not isinstance(event, dict):
            raise InvalidJSON("stream event must be a JSON object")
        if event.get("type") in {"error", "response.failed", "response.incomplete", "response.cancelled"} or "error" in event:
            raise ValueError("stream reported an inference error")
        if event.get("type") == "response.completed":
            terminal = True
        event_usage = event.get("usage")
        if isinstance(event_usage, dict):
            usage.update(event_usage)
        timings.update(numeric_timings(event.get("timings")))
        completed_response = event.get("response")
        if isinstance(completed_response, dict):
            completed_usage = completed_response.get("usage")
            if isinstance(completed_usage, dict):
                usage.update(completed_usage)
            timings.update(numeric_timings(completed_response.get("timings")))
        if event.get("type") in {"response.output_text.delta", "response.refusal.delta", "response.reasoning_summary_text.delta"} and isinstance(event.get("delta"), str):
            content_delta = event["delta"]
            generated = bool(content_delta)
            if event["type"] == "response.output_text.delta":
                delta_text = content_delta
            else:
                delta_text = ""
        else:
            delta_text = ""
            generated = False
            choices = event.get("choices")
            if isinstance(choices, list):
                for choice in choices:
                    if not isinstance(choice, dict):
                        continue
                    delta = choice.get("delta")
                    timings.update(numeric_timings(choice.get("timings")))
                    if choice.get("finish_reason") is not None:
                        terminal = True
                    if isinstance(delta, dict):
                        for field in ("content", "reasoning", "reasoning_content", "refusal"):
                            value = delta.get(field)
                            if isinstance(value, str):
                                generated = generated or bool(value)
                                if field == "content":
                                    delta_text += value
                        tool_calls = delta.get("tool_calls")
                        generated = generated or (
                            isinstance(tool_calls, list)
                            and any(
                                isinstance(call, dict)
                                and (
                                    call.get("id")
                                    or call.get("index") is not None
                                    or isinstance(call.get("function"), dict)
                                )
                                for call in tool_calls
                            )
                        )
                    if isinstance(choice.get("text"), str):
                        delta_text += choice["text"]
                        generated = generated or bool(choice["text"])
        if generated:
            if first_content_s is None:
                first_content_s = time.monotonic() - started
        if delta_text:
            text_parts.append(delta_text)
    if not terminal:
        raise ValueError("stream ended without a terminal event")
    return "".join(text_parts), usage, first_content_s, timings


def multipart_body(case: dict, suite_dir: Path) -> tuple[bytes, str]:
    boundary = "redqueen-" + uuid.uuid4().hex
    chunks: list[bytes] = []
    request_data = json_bytes(case["request"])
    fields = case["request"]
    for name, value in fields.items():
        if not isinstance(name, str) or not FIELD_PATTERN.fullmatch(name):
            raise SuiteError("multipart request field name is invalid")
        encoded = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, allow_nan=False)
        chunks.extend((
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
            encoded.encode("utf-8"),
            b"\r\n",
        ))
    total = len(request_data)
    for item in case.get("files", []):
        asset = (suite_dir / item["asset"]).resolve()
        if not asset.is_relative_to(suite_dir.resolve()) or not asset.is_file():
            raise SuiteError(f"image-edit asset is missing or escapes suite directory: {item['asset']}")
        if asset.stat().st_size > MAX_REQUEST_BYTES:
            raise SuiteError("image-edit asset exceeds size limit")
        data = asset.read_bytes()
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise SuiteError(f"image-edit asset failed sha256 validation: {item['asset']}")
        total += len(data)
        if total > MAX_REQUEST_BYTES:
            raise SuiteError("multipart request exceeds size limit")
        filename = item["filename"].replace('"', "")
        chunks.extend((
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{item["field"]}"; filename="{filename}"\r\n'.encode(),
            f'Content-Type: {item["content_type"]}\r\n\r\n'.encode(),
            data,
            b"\r\n",
        ))
    chunks.append(f"--{boundary}--\r\n".encode())
    body = b"".join(chunks)
    if len(body) > MAX_REQUEST_BYTES:
        raise SuiteError("multipart request exceeds size limit")
    return body, f"multipart/form-data; boundary={boundary}"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def invoke_case(
    *, case: dict, suite_dir: Path, origin: str, api_key: str | None,
    timeout_s: float, artifact_dir: Path | None, store_artifacts: bool, repeat: int,
) -> dict:
    url = origin.rstrip("/") + case["path"]
    request_data = case["request"]
    if case.get("files"):
        body, content_type = multipart_body(case, suite_dir)
    else:
        body, content_type = json_bytes(request_data), "application/json"
    headers = {"Content-Type": content_type, "Accept": "text/event-stream, application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    started = time.monotonic()
    status = None
    response = None
    response_headers_s = None
    ttft_s = None
    usage: dict = {}
    timings: dict = {}
    error_type = None
    try:
        opener = urllib.request.build_opener(NoRedirect, urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=timeout_s) as result:
            status = result.status
            response_headers_s = time.monotonic() - started
            content_type_header = result.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type_header == "text/event-stream":
                text_output, usage, ttft_s, timings = sse_event_records(result, started)
                response = {"text_output": text_output}
            else:
                raw = result.read(MAX_RESPONSE_BYTES + 1)
                if len(raw) > MAX_RESPONSE_BYTES:
                    raise ResponseTooLarge("response exceeds size limit")
                if not raw:
                    raise InvalidJSON("empty JSON response")
                response = strict_json_loads(raw)
                if not isinstance(response, dict):
                    raise InvalidJSON("JSON response must be an object")
                if "error" in response:
                    raise ValueError("JSON response reported an inference error")
                if isinstance(response, dict) and isinstance(response.get("usage"), dict):
                    usage = response["usage"]
                if isinstance(response, dict):
                    timings.update(numeric_timings(response.get("timings")))
                    choices = response.get("choices")
                    if isinstance(choices, list):
                        for choice in choices:
                            if isinstance(choice, dict):
                                timings.update(numeric_timings(choice.get("timings")))
    except urllib.error.HTTPError as error:
        status = error.code
        error_type = "http_error"
        error.close()
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        error_type = type(error).__name__
    except InvalidJSON:
        error_type = "invalid_json_response"
    except ResponseTooLarge:
        error_type = "response_too_large"
    except (ValueError, SuiteError) as error:
        error_type = type(error).__name__

    elapsed = time.monotonic() - started
    artifact = None
    if error_type is None and case["workload"] == "images":
        try:
            items = response.get("data") if isinstance(response, dict) else None
            if not isinstance(items, list) or len(items) != 1 or not isinstance(items[0], dict):
                raise ValueError("image response must contain exactly one data item")
            encoded = items[0].get("b64_json")
            if not isinstance(encoded, str):
                raise ValueError("image response is missing b64_json")
            image = base64.b64decode(encoded, validate=True)
            if not image or len(image) > MAX_IMAGE_BYTES or not image.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("image response is not a bounded PNG")
            if store_artifacts:
                if artifact_dir is None:
                    raise ValueError("image cases require --artifacts-dir")
                artifact_name = f"{case['id']}-r{repeat}.png"
                artifact_path = artifact_dir / artifact_name
                with artifact_path.open("xb") as stream:
                    os.chmod(artifact_path, 0o600)
                    stream.write(image)
                    stream.flush()
                    os.fsync(stream.fileno())
                artifact = {"path": str(artifact_path), "sha256": hashlib.sha256(image).hexdigest(), "bytes": len(image)}
            response = {**response, "data": [{key: value for key, value in items[0].items() if key != "b64_json"}]}
        except (ValueError, OSError) as error:
            error_type = "invalid_image_response"
            response = None
    record = {
        "schema_version": SCHEMA_VERSION,
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": platform.node(),
        "label": None,
        "endpoint": sanitized_url(url),
        "case_id": case["id"],
        "workload": case["workload"],
        "path": case["path"],
        "repeat": None,
        "status_code": status,
        "success": status is not None and 200 <= status < 300 and error_type is None,
        "error_type": error_type,
        "request_elapsed_s": elapsed,
        "response_headers_s": response_headers_s,
        "ttft_s": ttft_s,
        "usage": usage or None,
        "server_timings": timings or None,
        "artifact": artifact,
    }
    try:
        scored = score_case(case, response)
        record["quality"] = scored
    except (KeyError, TypeError, ValueError) as error:
        record["quality"] = None
        record["quality_error"] = type(error).__name__
    return record


def create_private_directory(path: Path, repository: Path) -> Path:
    resolved = path.resolve()
    if resolved.is_relative_to(repository):
        raise ValueError(f"run artifacts must be outside the repository: {resolved}")
    resolved.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.chmod(resolved, 0o700)
    return resolved


def source_revision(repository: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision if re.fullmatch(r"[a-f0-9]{40,64}", revision) else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path, help="versioned JSONL suite; see read_suite() schema")
    parser.add_argument("--url", required=True, help="HTTP(S) model gateway origin, without a path")
    parser.add_argument("--decisions-url", help="decision/CLM adapter origin; defaults to --url")
    parser.add_argument("--images-url", help="image adapter origin; defaults to --url")
    parser.add_argument("--label", required=True, help="model/runtime configuration identifier")
    parser.add_argument("--api-key-env", default="QWEN_API_KEY", help="bearer key environment variable; empty disables auth")
    parser.add_argument("--decisions-api-key-env", default="REDQUEEN_JEV_API_KEY", help="decision/CLM bearer key environment variable; empty disables auth")
    parser.add_argument("--images-api-key-env", default="REDQUEEN_IMAGE_API_KEY", help="image bearer key environment variable; empty disables auth")
    parser.add_argument("--output", type=Path, required=True, help="new JSONL result file outside the repository")
    parser.add_argument("--artifacts-dir", type=Path, help="new private directory for generated image outputs")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=2, help="simultaneous requests; image routes are limited to two ingress requests")
    parser.add_argument("--seed", type=int, default=0, help="seed for case-order randomization")
    parser.add_argument("--timeout-s", type=float, default=1800)
    args = parser.parse_args()
    if (
        args.repeats < 1
        or args.warmups < 0
        or not math.isfinite(args.timeout_s)
        or args.timeout_s <= 0
        or not 1 <= args.concurrency <= 4
    ):
        parser.error("repeats and timeout must be positive; warmups nonnegative; concurrency between one and four")
    if not ID_PATTERN.fullmatch(args.label):
        parser.error("label must contain only letters, numbers, dot, underscore, or hyphen")
    return args


def main() -> int:
    args = parse_args()
    repository = Path(__file__).resolve().parents[4]
    try:
        validate_origin(args.url)
        cases, suite_sha256 = read_suite(args.suite)
        has_images = any(case["workload"] == "images" for case in cases)
        if has_images and args.concurrency > 2:
            raise ValueError("image suites support at most two concurrent ingress requests")
        output_path = args.output.resolve()
        if output_path.is_relative_to(repository):
            raise ValueError("result file must be outside the repository")
        if output_path.exists():
            raise FileExistsError(f"result file already exists: {output_path}")
        endpoints = resolve_endpoints(
            {case["workload"] for case in cases},
            model_url=args.url,
            decisions_url=args.decisions_url or args.url,
            images_url=args.images_url or args.url,
            model_key_env=args.api_key_env,
            decisions_key_env=args.decisions_api_key_env,
            images_key_env=args.images_api_key_env,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if output_path.parent.stat().st_mode & 0o077:
            raise ValueError(f"result parent directory must be private (mode 0700): {output_path.parent}")
        artifact_dir = None
        if has_images:
            if args.artifacts_dir is None:
                raise ValueError("image cases require --artifacts-dir")
            artifact_dir = create_private_directory(args.artifacts_dir, repository)
        elif args.artifacts_dir is not None:
            artifact_dir = create_private_directory(args.artifacts_dir, repository)
        suite_dir = args.suite.resolve().parent
    except (OSError, ValueError, SuiteError) as error:
        print(f"cannot start benchmark: {error}", file=sys.stderr)
        return 2

    model_groups = ordered_model_groups(cases, args.seed)
    revision = source_revision(repository)
    def execute(case: dict, repeat: int, store_artifacts: bool) -> dict:
        endpoint_url, api_key = endpoints[endpoint_name(case["workload"])]
        return invoke_case(
            case=case, suite_dir=suite_dir, origin=endpoint_url, api_key=api_key,
            timeout_s=args.timeout_s, artifact_dir=artifact_dir,
            store_artifacts=store_artifacts, repeat=repeat,
        )

    try:
        with os.fdopen(os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8") as output:
            for _ in range(args.warmups):
                for _, group in model_groups:
                    workers = 1 if any(serial_group(case) is not None for case in group) else args.concurrency
                    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                        list(executor.map(
                            lambda case: execute(case, -1, False),
                            group,
                        ))
            for repeat in range(args.repeats):
                for _, group in model_groups:
                    workers = 1 if any(serial_group(case) is not None for case in group) else args.concurrency
                    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                        measured = executor.map(lambda case: execute(case, repeat, True), group)
                        for case, record in zip(group, measured, strict=True):
                            record.update({
                                "label": args.label,
                                "requested_model": case["request"]["model"],
                                "repeat": repeat,
                                "suite_id": case["suite_id"],
                                "suite_sha256": suite_sha256,
                                "order_seed": args.seed,
                                "concurrency": args.concurrency,
                                "effective_concurrency": workers,
                                "warmups": args.warmups,
                                "timeout_s": args.timeout_s,
                                "source_revision": revision,
                            })
                            output.write(json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n")
                            output.flush()
                            os.fsync(output.fileno())
    except (OSError, ValueError) as error:
        print(f"benchmark stopped: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
