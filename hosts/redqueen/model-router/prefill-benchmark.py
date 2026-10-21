#!/usr/bin/env python3
"""Measure streamed chat TTFT and server-reported prefill timings from JSONL cases."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


def read_cases(path: Path) -> list[dict]:
    cases = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on line {line_number}") from error
        if not isinstance(case, dict):
            raise ValueError(f"line {line_number} must contain a JSON object")
        request = case.get("request", {key: value for key, value in case.items() if key != "id"})
        if not isinstance(request, dict) or not ("messages" in request or "prompt" in request):
            raise ValueError(f"line {line_number} needs request.messages or request.prompt")
        cases.append({"index": line_number, "request": request})
    if not cases:
        raise ValueError("input contains no benchmark cases")
    return cases


def numeric_timings(value: object) -> dict:
    if not isinstance(value, dict):
        return {}
    allowed = {"prompt_n", "prompt_ms", "predicted_n", "predicted_ms"}
    return {
        key: number
        for key, raw in value.items()
        if key in allowed and isinstance(raw, (int, float)) and not isinstance(raw, bool)
        for number in [raw]
    }


def has_generated_delta(delta: dict) -> bool:
    for field in ("content", "reasoning", "reasoning_content", "refusal"):
        value = delta.get(field)
        if isinstance(value, str) and value:
            return True
        if isinstance(value, list) and value:
            return True
    calls = delta.get("tool_calls")
    if isinstance(calls, list):
        for call in calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if any(call.get(key) for key in ("id", "type")):
                return True
            if isinstance(function, dict) and any(function.get(key) for key in ("name", "arguments")):
                return True
    return False


def input_chars(request: dict) -> int | None:
    if isinstance(request.get("prompt"), str):
        return len(request["prompt"])
    messages = request.get("messages")
    if isinstance(messages, list):
        return sum(len(json.dumps(message, ensure_ascii=False)) for message in messages)
    return None


def sanitized_endpoint(value: str) -> str:
    parts = urlsplit(value)
    hostname = parts.hostname
    if hostname is None:
        return urlunsplit((parts.scheme, "", parts.path, "", ""))
    host = f"[{hostname}]" if ":" in hostname else hostname
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def stream_case(
    *, url: str, api_key: str, model: str | None, request_data: dict,
    include_usage: bool, timeout_s: float,
) -> dict:
    body = dict(request_data)
    if model is not None:
        body["model"] = model
    if not isinstance(body.get("model"), str):
        raise ValueError("each request needs a model or --model")
    body["stream"] = True
    if include_usage:
        stream_options = body.get("stream_options", {})
        if not isinstance(stream_options, dict):
            raise ValueError("stream_options must be an object")
        body["stream_options"] = {**stream_options, "include_usage": True}

    request = urllib.request.Request(
        url,
        data=json.dumps(body, allow_nan=False, ensure_ascii=False).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    response_headers_s = None
    first_content_s = None
    response_usage = None
    server_timings = {}
    stream_chunks = 0
    status = None
    error_type = None

    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            status = response.status
            response_headers_s = time.monotonic() - started
            while True:
                line = response.readline()
                if not line:
                    break
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if not isinstance(event, dict):
                    continue
                usage = event.get("usage")
                if isinstance(usage, dict):
                    response_usage = {
                        key: value
                        for key, value in usage.items()
                        if key in {"prompt_tokens", "completion_tokens", "total_tokens"}
                        and isinstance(value, int)
                        and not isinstance(value, bool)
                    }
                server_timings.update(numeric_timings(event.get("timings")))
                choices = event.get("choices")
                if isinstance(choices, list):
                    for choice in choices:
                        if not isinstance(choice, dict):
                            continue
                        delta = choice.get("delta")
                        if isinstance(delta, dict):
                            if has_generated_delta(delta):
                                stream_chunks += 1
                                if first_content_s is None:
                                    first_content_s = time.monotonic() - started
                        server_timings.update(numeric_timings(choice.get("timings")))
    except urllib.error.HTTPError as error:
        status = error.code
        error_type = "http_error"
        error.close()
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        error_type = type(error).__name__

    completed = time.monotonic()
    elapsed = completed - started
    decode_elapsed = elapsed - first_content_s if first_content_s is not None else None
    completion_tokens = response_usage.get("completion_tokens") if response_usage else None
    decode_tokens_per_s = None
    if isinstance(completion_tokens, int) and decode_elapsed and decode_elapsed > 0:
        decode_tokens_per_s = completion_tokens / decode_elapsed
    return {
        "status_code": status,
        "error_type": error_type,
        "request_elapsed_s": elapsed,
        "response_headers_s": response_headers_s,
        "ttft_s": first_content_s,
        "stream_content_chunks": stream_chunks,
        "usage": response_usage,
        "decode_tokens_per_s": decode_tokens_per_s,
        "server_prefill_timings": server_timings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path, help="JSONL with one OpenAI chat request per line")
    parser.add_argument("--url", default="http://127.0.0.1:8081/v1/chat/completions")
    parser.add_argument("--model", help="override the model in every request")
    parser.add_argument("--label", required=True, help="short engine/configuration label for pairing runs")
    parser.add_argument("--api-key-env", default="QWEN_API_KEY", help="environment variable containing the bearer key")
    parser.add_argument("--output", type=Path, help="JSONL destination; stdout is the default")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=0, help="requests sent before recorded samples")
    parser.add_argument("--max-tokens", type=int, default=128, help="completion limit when a case omits one")
    parser.add_argument("--timeout-s", type=float, default=1800)
    parser.add_argument("--no-include-usage", action="store_true", help="do not request stream usage metadata")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0 or args.max_tokens < 1 or args.timeout_s <= 0:
        parser.error("repeats and max-tokens must be positive; warmups must be nonnegative; timeout must be positive")
    return args


def main() -> int:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"missing bearer key in environment variable {args.api_key_env}", file=sys.stderr)
        return 2
    try:
        cases = read_cases(args.cases)
    except (OSError, ValueError) as error:
        print(f"cannot load cases: {error}", file=sys.stderr)
        return 2

    output = args.output.open("w") if args.output else sys.stdout
    try:
        for repeat in range(-args.warmups, args.repeats):
            for case in cases:
                request_data = dict(case["request"])
                request_data.setdefault("max_tokens", args.max_tokens)
                measured = stream_case(
                    url=args.url,
                    api_key=api_key,
                    model=args.model,
                    request_data=request_data,
                    include_usage=not args.no_include_usage,
                    timeout_s=args.timeout_s,
                )
                if repeat < 0:
                    continue
                record = {
                    "schema_version": 1,
                    "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "host": platform.node(),
                    "label": args.label,
                    "endpoint": sanitized_endpoint(args.url),
                    "model": args.model or request_data.get("model"),
                    "case_index": case["index"],
                    "repeat": repeat,
                    "input_chars": input_chars(request_data),
                    "ttft_definition": "request start to first generated stream delta; includes gateway, queue, load, and prefill",
                    **measured,
                }
                output.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                output.flush()
    finally:
        if args.output:
            output.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
