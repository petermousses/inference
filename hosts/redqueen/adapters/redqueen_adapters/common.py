from __future__ import annotations

import asyncio
import hmac
import json
import logging
import re
import traceback
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

from aiohttp import web


LOG = logging.getLogger("redqueen.http")
JSON_MARKERS = re.compile(rb'["\\\[\]{},:]')


class AdmissionError(Exception):
    def __init__(self, status: int, message: str, code: str, retry_after: int | None = 1):
        super().__init__(message)
        self.status, self.code, self.retry_after = status, code, retry_after


class UpstreamError(AdmissionError):
    """An upstream fault, distinct from client input validation."""


def admission_response(error: AdmissionError) -> web.Response:
    response = error_response(error.status, str(error), "rate_limit_error" if error.status == 429 else "invalid_request_error" if 400 <= error.status < 500 else "server_error")
    body = json.loads(response.body)
    body["error"]["code"] = error.code
    response = web.json_response(body, status=error.status, headers={"Cache-Control": "no-store"})
    if error.status in {429, 503} and error.retry_after is not None:
        response.headers["Retry-After"] = str(error.retry_after)
    return response


def error_response(
    status: int, message: str, error_type: str = "invalid_request_error", *, code: str | None = None
) -> web.Response:
    return web.json_response(
        {
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": code or {400: "invalid_request", 401: "authentication_failed", 403: "access_denied", 404: "not_found", 405: "method_not_allowed", 408: "upload_timeout", 413: "body_too_large", 429: "model_busy", 500: "internal_error", 502: "upstream_error", 503: "backend_unavailable", 504: "transport_timeout"}.get(status, "http_error"),
            }
        },
        status=status,
        headers={"Cache-Control": "no-store"},
    )


def bearer_is_valid(request: web.Request, expected: str) -> bool:
    scheme, separator, supplied = request.headers.get("Authorization", "").partition(
        " "
    )
    return (
        separator == " "
        and scheme.lower() == "bearer"
        and hmac.compare_digest(supplied.encode("utf-8", "surrogatepass"), expected.encode("utf-8", "surrogatepass"))
    )


def auth_middleware(api_key: str) -> web.middleware:
    @web.middleware
    async def authenticate(
        request: web.Request,
        handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
    ) -> web.StreamResponse:
        if request.path in {"/healthz", "/readyz"}:
            return await handler(request)
        if not bearer_is_valid(request, api_key):
            return error_response(
                401, "invalid or missing bearer token", "authentication_error"
            )
        response = await handler(request)
        response.headers.setdefault("Cache-Control", "no-store")
        return response

    return authenticate


@web.middleware
async def safe_errors(
    request: web.Request,
    handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
) -> web.StreamResponse:
    supplied_id = request.headers.get("X-Request-ID", "")
    request_id = supplied_id if re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", supplied_id) else uuid.uuid4().hex
    request["request_id"] = request_id
    try:
        response = await handler(request)
    except AdmissionError as error:
        response = admission_response(error)
    except web.HTTPRequestEntityTooLarge:
        response = error_response(413, "request body is too large")
    except web.HTTPException as error:
        response = error_response(error.status, error.reason)
        for header in ("Allow", "WWW-Authenticate", "Retry-After"):
            if header in error.headers:
                response.headers[header] = error.headers[header]
    except ValueError as error:
        response = error_response(400, str(error))
    except asyncio.CancelledError:
        raise
    except Exception as error:
        frames = [(frame.filename, frame.lineno, frame.name) for frame in traceback.extract_tb(error.__traceback__)]
        LOG.error("request_failed id=%s reason=%s frames=%s", request_id, type(error).__name__, frames)
        response = error_response(500, "internal server error", "server_error")
    if not response.prepared:
        response.headers["X-Request-ID"] = request_id
    return response


@asynccontextmanager
async def body_timeout(seconds: float = 60):
    try:
        async with asyncio.timeout(seconds):
            yield
    except TimeoutError as error:
        raise AdmissionError(408, "request body could not be read", "upload_timeout", retry_after=None) from error


async def read_json(request: web.Request) -> object:
    async with body_timeout():
        raw = await request.read()
        # Bound object allocation before decoding. A byte cap alone still lets
        # millions of tiny JSON nodes exhaust the front service's memory.
        depth = markers = escaped_until = 0
        in_string = False
        for match in JSON_MARKERS.finditer(raw):
            position = match.start()
            if position < escaped_until:
                continue
            marker = raw[position]
            if in_string:
                if marker == 92:
                    escaped_until = position + 2
                elif marker == 34:
                    in_string = False
                continue
            if marker == 34:
                in_string = True
            elif marker in (91, 123):
                depth += 1
                if depth > 64:
                    raise ValueError("request JSON exceeds its nesting limit")
            elif marker in (93, 125):
                depth -= 1
            markers += 1
            if markers > 250_000:
                raise ValueError("request JSON exceeds its structure limit")
        try:
            return json.loads(raw)
        except (UnicodeDecodeError, ValueError, RecursionError):
            raise ValueError("request body must be valid JSON") from None


class IngressGate:
    """Reject excess handlers before they buffer or decode a request."""
    def __init__(self, limit: int = 2):
        self.limit, self.active = limit, 0

    @web.middleware
    async def middleware(self, request: web.Request, handler):
        if request.path in {"/healthz", "/readyz", "/scheduler", "/v1/models", "/models", "/health", "/running", "/metrics"}:
            return await handler(request)
        if self.active >= self.limit:
            raise AdmissionError(429, "request admission limit reached", "frontend_busy")
        self.active += 1
        try:
            response = await handler(request)
            if isinstance(response, web.Response) and not response.prepared:
                response.headers.setdefault("X-Request-ID", request.get("request_id", uuid.uuid4().hex))
                response.headers.setdefault("Cache-Control", "no-store")
                await response.prepare(request)
                # aiohttp normally transmits Response after middleware returns.
                # Keep the buffer's admission slot until the client drains it.
                await response.write_eof()
            return response
        finally:
            self.active -= 1


async def read_limited(response, limit: int = 8 * 1024 * 1024, *, description: str = "upstream", check_status: bool = True) -> bytes:
    data = bytearray()
    async for chunk in response.content.iter_chunked(64 * 1024):
        data.extend(chunk)
        if len(data) > limit:
            raise UpstreamError(502, f"{description} response exceeded its byte limit", "upstream_invalid_response")
    if check_status and response.status >= 400:
        status = response.status
        code = "upstream_error"
        try:
            candidate = json.loads(data).get("error", {}).get("code")
            if isinstance(candidate, str) and re.fullmatch(r"[a-zA-Z0-9_.-]{1,80}", candidate):
                code = candidate
        except (ValueError, AttributeError, TypeError):
            pass
        retry = response.headers.get("Retry-After", "1")
        retry_after = min(3600, int(retry)) if retry.isdecimal() and len(retry) <= 5 else 1
        if status in {429, 503}:
            raise UpstreamError(status, "upstream is temporarily unavailable", code if code != "upstream_error" else "model_busy" if status == 429 else "backend_unavailable", retry_after=retry_after)
        if status in {400, 422}:
            raise UpstreamError(400, "upstream rejected model input", code)
        raise UpstreamError(504 if status == 504 else 502, "upstream request failed", code)
    return bytes(data)
