from __future__ import annotations

import hmac
import json
from collections.abc import Awaitable, Callable

from aiohttp import web


class AdmissionError(Exception):
    def __init__(self, status: int, message: str, code: str):
        super().__init__(message)
        self.status, self.code = status, code


def admission_response(error: AdmissionError) -> web.Response:
    response = error_response(error.status, str(error), "rate_limit_error" if error.status == 429 else "server_error")
    body = json.loads(response.body)
    body["error"]["code"] = error.code
    response = web.json_response(body, status=error.status, headers={"Cache-Control": "no-store"})
    if error.status in {429, 503}:
        response.headers["Retry-After"] = "1"
    return response


def error_response(
    status: int, message: str, error_type: str = "invalid_request_error"
) -> web.Response:
    return web.json_response(
        {
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": None,
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
        and hmac.compare_digest(supplied, expected)
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
    try:
        return await handler(request)
    except AdmissionError as error:
        return admission_response(error)
    except web.HTTPRequestEntityTooLarge:
        return error_response(413, "request body is too large")
    except ValueError as error:
        return error_response(400, str(error))
