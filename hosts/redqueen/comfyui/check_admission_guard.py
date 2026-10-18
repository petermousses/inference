#!/usr/bin/env python3
"""Keep ComfyUI startup failed unless the Redqueen submission barrier loaded."""

from __future__ import annotations

import json
import time
from urllib.request import urlopen


URL = "http://127.0.0.1:8189/redqueen/admission-guard"
MAX_RESPONSE_BYTES = 4096


def ready() -> bool:
    try:
        with urlopen(URL, timeout=2) as response:
            body = response.read(MAX_RESPONSE_BYTES + 1)
    except (OSError, TimeoutError):
        return False
    if len(body) > MAX_RESPONSE_BYTES:
        return False
    try:
        status = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return (
        isinstance(status, dict)
        and status.get("component") == "redqueen-prompt-admission"
        and not isinstance(status.get("version"), bool)
        and status.get("version") == 1
        and isinstance(status.get("pid"), int)
        and not isinstance(status.get("pid"), bool)
        and status["pid"] > 0
    )


def wait_ready(timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while True:
        if ready():
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("ComfyUI admission guard failed to become ready")
        time.sleep(0.25)


if __name__ == "__main__":
    wait_ready()
