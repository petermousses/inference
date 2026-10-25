#!/usr/bin/env python3
"""Run resident/SSD MoE logits parity under Redqueen's maintenance lease."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import uuid


GIB = 1024**3
MAX_GTT = 88 * GIB
MAX_PHYSICAL = 104 * GIB
MIN_HOST_AVAILABLE = 8 * GIB
PREFILL_HEADROOM = 5 * GIB
LEASE_DIRECTORY = Path("/srv/ai/model-router/run/leases")
RECOVERY_STATES = {"recovering", "quarantined"}
INFERENCE_SLICE = "ai-inference.slice"


def required_path(name: str, *, executable: bool = False) -> Path:
    raw = os.environ.get(name, "")
    if not raw:
        raise RuntimeError(f"set {name} to a verified host path")
    path = Path(raw).expanduser().resolve(strict=True)
    if not path.is_file() or (executable and not os.access(path, os.X_OK)):
        raise RuntimeError(f"{name} is not a usable regular file")
    return path


def positive_int(name: str, *, maximum: int = (1 << 63) - 1) -> int:
    raw = os.environ.get(name, "")
    if not re.fullmatch(r"[0-9]+", raw):
        raise RuntimeError(f"set {name} to an unsigned decimal integer")
    value = int(raw)
    if value <= 0 or value > maximum:
        raise RuntimeError(f"{name} is outside its allowed range")
    return value


def nonnegative_int(name: str, *, maximum: int = (1 << 63) - 1) -> int:
    raw = os.environ.get(name, "")
    if not re.fullmatch(r"[0-9]+", raw):
        raise RuntimeError(f"set {name} to an unsigned decimal integer")
    value = int(raw)
    if value < 0 or value > maximum:
        raise RuntimeError(f"{name} is outside its allowed range")
    return value


def source_root() -> Path:
    override = os.environ.get("REDQUEEN_SOURCE_ROOT")
    root = Path(override).expanduser() if override else Path(__file__).resolve().parents[2]
    root = root.resolve(strict=True)
    if not (root / "adapters").is_dir() or not (root / "deepseek-hip" / "python").is_dir():
        raise RuntimeError("REDQUEEN_SOURCE_ROOT must name the versioned hosts/redqueen source directory")
    return root


def memory_snapshot() -> tuple[int, int, int, int]:
    try:
        total_gtt = int(Path("/sys/class/drm/card0/device/mem_info_gtt_total").read_text())
        used_gtt = int(Path("/sys/class/drm/card0/device/mem_info_gtt_used").read_text())
        values = {}
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            fields = line.split()
            if fields and fields[0] in {"MemTotal:", "MemAvailable:"} and len(fields) >= 2:
                values[fields[0]] = int(fields[1]) * 1024
        total_host = values["MemTotal:"]
        available_host = values["MemAvailable:"]
    except (OSError, ValueError, KeyError) as error:
        raise RuntimeError("cannot read Redqueen host and GTT memory availability") from error
    if total_gtt <= 0 or used_gtt > total_gtt or available_host > total_host:
        raise RuntimeError("Redqueen memory availability values are inconsistent")
    return total_gtt - used_gtt, available_host, total_gtt, total_host


def check_memory(gtt_peak: int, host_peak: int, *, admission: bool) -> tuple[int, int]:
    free_gtt, available_host, total_gtt, total_host = memory_snapshot()
    used_gtt = total_gtt - free_gtt
    used_host = total_host - available_host
    projected_gtt = used_gtt + (gtt_peak if admission else 0)
    projected_host = used_host + (host_peak if admission else 0)
    projected_free_gtt = free_gtt - (gtt_peak if admission else 0)
    projected_available_host = available_host - (host_peak if admission else 0)
    if projected_gtt > MAX_GTT:
        raise RuntimeError("resident/streamed run would exceed the 88 GiB managed GTT ceiling")
    if projected_free_gtt < PREFILL_HEADROOM:
        raise RuntimeError("resident/streamed run would consume the 5 GiB GTT headroom")
    if projected_host > MAX_PHYSICAL:
        raise RuntimeError("resident/streamed run would exceed the 104 GiB managed physical ceiling")
    if projected_available_host < MIN_HOST_AVAILABLE + PREFILL_HEADROOM:
        raise RuntimeError("resident/streamed run would violate the 8 GiB host minimum plus 5 GiB headroom")
    return used_gtt, available_host


def start_qualified_process(
    command: list[str], *, stdout=None, stderr=None, env=None
) -> tuple[subprocess.Popen, str]:
    """Put standalone qualification workers under the inference slice cap."""
    unit = f"redqueen-moe-qualification-{os.getpid()}-{uuid.uuid4().hex}.scope"
    scoped = [
        "/usr/bin/systemd-run",
        "--user",
        "--scope",
        f"--slice={INFERENCE_SLICE}",
        f"--unit={unit}",
        "--quiet",
        "--",
        *command,
    ]
    return subprocess.Popen(scoped, stdout=stdout, stderr=stderr, env=env), unit


def stop_qualified_process(process: subprocess.Popen, unit: str) -> None:
    """Stop the whole transient scope before returning from a failed probe."""
    if process.poll() is not None:
        return
    try:
        stopped = subprocess.run(
            ["/usr/bin/systemctl", "--user", "stop", unit],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        stopped = None
    if stopped is None or stopped.returncode != 0:
        try:
            subprocess.run(
                [
                    "/usr/bin/systemctl",
                    "--user",
                    "kill",
                    "--kill-who=all",
                    "--signal=SIGKILL",
                    unit,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            pass
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def validate_host_state(run_directory: Path) -> None:
    for name in ("deployment.json", "maintenance.json", "comfyui-restore.json"):
        marker = run_directory / name
        if marker.exists() or marker.is_symlink():
            raise RuntimeError(f"host maintenance marker is present: {marker}")
    recovery = run_directory / "recovery.json"
    if recovery.exists() or recovery.is_symlink():
        try:
            state = json.loads(recovery.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeError("host recovery state is unreadable") from error
        if not isinstance(state, dict) or state.get("state") in RECOVERY_STATES:
            raise RuntimeError("host recovery is pending or quarantined")


def run_monitored(command: list[str], gtt_peak: int, host_peak: int, context_tokens: int) -> None:
    print("running", Path(command[0]).name, "with one MoE qualification process", flush=True)
    check_memory(gtt_peak, host_peak, admission=True)
    environment = os.environ.copy()
    environment["MOE_PROBE_CONTEXT_TOKENS"] = str(context_tokens)
    process, unit = start_qualified_process(command, env=environment)
    try:
        while True:
            try:
                return_code = process.wait(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                try:
                    check_memory(gtt_peak, host_peak, admission=False)
                except BaseException:
                    stop_qualified_process(process, unit)
                    raise
        if return_code:
            raise RuntimeError(f"qualification process exited with status {return_code}")
        check_memory(gtt_peak, host_peak, admission=False)
    except BaseException:
        stop_qualified_process(process, unit)
        raise


def log_tail(path: Path, limit: int = 16, max_bytes: int = 65536) -> str:
    try:
        with path.open("rb") as source:
            source.seek(0, os.SEEK_END)
            size = source.tell()
            source.seek(max(0, size - max_bytes))
            data = source.read()
        lines = data.decode("utf-8", errors="replace").splitlines()
        if size > max_bytes and lines:
            lines = lines[1:]
        return "\n".join(lines[-limit:])
    except OSError:
        return "<log unavailable>"


def run_concurrent_monitored(
    commands: list[list[str]],
    log_paths: list[Path],
    barrier: Path,
    gtt_peak: int,
    host_peak: int,
    context_tokens: int,
) -> list[dict[str, object]]:
    if len(commands) < 2 or len(commands) != len(log_paths):
        raise RuntimeError("concurrent qualification needs a command and log for every worker")
    print(f"starting {len(commands)} streamed MoE workers behind a decode barrier", flush=True)
    check_memory(gtt_peak, host_peak, admission=True)
    baseline_gtt, baseline_host_available = check_memory(gtt_peak, host_peak, admission=False)
    observed_peak_gtt = baseline_gtt
    observed_min_host_available = baseline_host_available
    processes: list[tuple[subprocess.Popen, str]] = []
    handles = []
    try:
        for command, log_path in zip(commands, log_paths, strict=True):
            handle = log_path.open("wb")
            handles.append(handle)
            environment = os.environ.copy()
            environment["MOE_PROBE_BARRIER"] = str(barrier)
            environment["MOE_PROBE_CONTEXT_TOKENS"] = str(context_tokens)
            processes.append(
                start_qualified_process(
                    command,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    env=environment,
                )
            )

        deadline = time.monotonic() + 900
        ready: set[int] = set()
        while len(ready) < len(processes):
            for index, ((process, _unit), log_path) in enumerate(zip(processes, log_paths, strict=True)):
                if index in ready:
                    continue
                if process.poll() is not None:
                    raise RuntimeError(
                        f"worker {index + 1} exited before decode barrier (status {process.returncode}):\n"
                        f"{log_tail(log_path)}"
                    )
                if any(
                    line == "READY"
                    for line in log_tail(log_path, limit=128, max_bytes=65536).splitlines()
                ):
                    ready.add(index)
            used_gtt, host_available = check_memory(gtt_peak, host_peak, admission=False)
            observed_peak_gtt = max(observed_peak_gtt, used_gtt)
            observed_min_host_available = min(observed_min_host_available, host_available)
            if time.monotonic() >= deadline:
                raise RuntimeError("streamed workers did not reach the decode barrier within 15 minutes")
            if len(ready) < len(processes):
                time.sleep(0.5)

        barrier.touch(exist_ok=False)
        print("all streamed workers are loaded; concurrent decode started", flush=True)
        deadline = time.monotonic() + 1800
        while any(process.poll() is None for process, _unit in processes):
            used_gtt, host_available = check_memory(gtt_peak, host_peak, admission=False)
            observed_peak_gtt = max(observed_peak_gtt, used_gtt)
            observed_min_host_available = min(observed_min_host_available, host_available)
            if time.monotonic() >= deadline:
                raise RuntimeError("concurrent streamed generation exceeded 30 minutes")
            time.sleep(0.5)
        for index, ((process, _unit), log_path) in enumerate(zip(processes, log_paths, strict=True)):
            if process.returncode:
                raise RuntimeError(
                    f"worker {index + 1} exited with status {process.returncode}:\n{log_tail(log_path)}"
                )
        used_gtt, host_available = check_memory(gtt_peak, host_peak, admission=False)
        observed_peak_gtt = max(observed_peak_gtt, used_gtt)
        observed_min_host_available = min(observed_min_host_available, host_available)

        results = []
        for index, log_path in enumerate(log_paths):
            result = None
            for line in reversed(log_tail(log_path, limit=65536, max_bytes=1 << 20).splitlines()):
                if line.startswith("{"):
                    result = json.loads(line)
                    break
            if not isinstance(result, dict):
                raise RuntimeError(f"worker {index + 1} emitted no final streaming stats: {log_tail(log_path)}")
            if (
                result.get("streamed") is not True
                or result.get("direct_io_verified") is not True
                or result.get("pending_reads") != 0
                or result.get("misses", 0) <= 0
                or result.get("context_tokens") != context_tokens
            ):
                raise RuntimeError(f"worker {index + 1} violated the stream completion contract: {result}")
            results.append(result)
        for index, result in enumerate(results, start=1):
            print(f"concurrent worker {index}: {json.dumps(result, sort_keys=True)}", flush=True)
        print(
            "concurrent observed memory: "
            + json.dumps(
                {
                    "baseline_gtt_used_bytes": baseline_gtt,
                    "peak_gtt_used_bytes": observed_peak_gtt,
                    "peak_gtt_increase_bytes": max(0, observed_peak_gtt - baseline_gtt),
                    "baseline_host_available_bytes": baseline_host_available,
                    "minimum_host_available_bytes": observed_min_host_available,
                    "host_available_decrease_bytes": max(
                        0, baseline_host_available - observed_min_host_available
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return results
    except BaseException:
        for process, unit in processes:
            if process.poll() is None:
                stop_qualified_process(process, unit)
        for index, log_path in enumerate(log_paths[:len(processes)], start=1):
            print(f"worker {index} log tail:\n{log_tail(log_path)}", file=sys.stderr, flush=True)
        raise
    finally:
        for handle in handles:
            handle.close()


def run_qualification(
    probe: Path,
    model: Path,
    gpu_layers: int,
    slots: int,
    budget: int,
    steps: int,
    gtt_peak: int,
    host_peak: int,
    comparer: Path,
    evict_idle: bool,
    stream_only: bool,
    concurrent_model: Path | None,
    context_tokens: int,
) -> None:
    source = source_root() / "deepseek-hip" / "python"
    sys.path.insert(0, str(source))
    try:
        from run_dense_checkpoint_smoke import _comfyui_smoke_window
    except ImportError as error:
        raise RuntimeError("the host ComfyUI isolation helper is unavailable") from error

    run_directory = LEASE_DIRECTORY.parent
    validate_host_state(run_directory)
    with _comfyui_smoke_window(run_directory / "comfyui-restore.json"):
        stopped = asyncio.run(stop_idle_text_workers()) if evict_idle else []
        try:
            check_memory(gtt_peak, host_peak, admission=True)
            with tempfile.TemporaryDirectory(prefix="moe-parity-") as temporary:
                resident_logits = Path(temporary) / "resident.logits"
                streamed_logits = Path(temporary) / "streamed.logits"
                common = [str(probe), str(model), str(gpu_layers)]
                if concurrent_model is not None:
                    if not stream_only:
                        raise RuntimeError("concurrent qualification requires --stream-only")
                    commands = [
                        common + [str(slots), str(budget), str(steps), str(streamed_logits)],
                        [
                            str(probe), str(concurrent_model), str(gpu_layers),
                            str(slots), str(budget), str(steps), str(Path(temporary) / "concurrent.logits"),
                        ],
                    ]
                    run_concurrent_monitored(
                        commands,
                        [Path(temporary) / "primary.log", Path(temporary) / "concurrent.log"],
                        Path(temporary) / "release-decode",
                        gtt_peak,
                        host_peak,
                        context_tokens,
                    )
                elif not stream_only:
                    run_monitored(
                        common + ["0", "0", str(steps), str(resident_logits)],
                        gtt_peak,
                        host_peak,
                        context_tokens,
                    )
                if concurrent_model is None:
                    run_monitored(
                        common + [str(slots), str(budget), str(steps), str(streamed_logits)],
                        gtt_peak,
                        host_peak,
                        context_tokens,
                    )
                if concurrent_model is not None:
                    print("concurrent streamed smoke passed", flush=True)
                elif stream_only:
                    print("stream-only smoke passed; no resident parity baseline was run", flush=True)
                else:
                    subprocess.run(
                        [sys.executable, str(comparer), str(resident_logits), str(streamed_logits)],
                        check=True,
                        timeout=30,
                    )
        finally:
            if stopped:
                asyncio.run(restore_idle_text_workers(stopped))


async def _idle_runtime():
    from aiohttp import ClientSession, ClientTimeout
    from redqueen_adapters.scheduler_api import Runtime

    policy_path = Path("/srv/ai/model-router/scheduler.json")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    profiles = json.loads(Path(policy["profile_path"]).read_text(encoding="utf-8"))
    api_key = os.environ.get("QWEN_API_KEY", "")
    if not api_key:
        raise RuntimeError("--evict-idle requires QWEN_API_KEY from the host service environment")
    session = ClientSession(timeout=ClientTimeout(total=None))
    runtime = Runtime(policy, profiles["resources"], session, api_key)
    return policy, session, runtime


async def stop_idle_text_workers() -> list[str]:
    policy, session, runtime = await _idle_runtime()
    try:
        states = await runtime.snapshot()
        targets = [
            name for name in states
            if runtime.resources[name].get("kind") == "text"
        ]
        for name in targets:
            if states[name] != "ready":
                raise RuntimeError(f"refusing to evict text worker in {states[name]} state: {name}")
            if await runtime.busy(name) is not False:
                raise RuntimeError(f"refusing to evict busy or unobservable text worker: {name}")
        stopped = []
        try:
            for name in targets:
                stopped.append(name)
                await asyncio.wait_for(runtime.stop(name), policy["stop_timeout_s"] + 10)
            return stopped
        except BaseException as error:
            try:
                await restore_idle_text_workers(stopped)
            except BaseException as restore_error:
                raise RuntimeError(
                    f"idle-worker eviction failed and prior workers could not all be restored: {restore_error}"
                ) from error
            raise
    finally:
        await session.close()


async def restore_idle_text_workers(names: list[str]) -> None:
    if not names:
        return
    policy, session, runtime = await _idle_runtime()
    failures = []
    try:
        for name in reversed(names):
            try:
                states = await runtime.snapshot()
                if name in states:
                    if states[name] != "ready":
                        raise RuntimeError(f"worker did not return to ready state: {name}")
                    continue
                await asyncio.wait_for(runtime.start(name), 900)
            except BaseException as error:
                failures.append(f"{name}: {type(error).__name__}")
    finally:
        await session.close()
    if failures:
        raise RuntimeError("failed to restore idle text workers: " + ", ".join(failures))


async def run_under_maintenance(args: argparse.Namespace) -> None:
    adapter_source = source_root() / "adapters"
    sys.path.insert(0, str(adapter_source))
    try:
        from redqueen_adapters.scheduler_client import SerializedExecution, run_thread_shielded
    except ImportError as error:
        raise RuntimeError("the versioned host scheduler lease implementation is unavailable") from error
    execution = SerializedExecution(LEASE_DIRECTORY)
    async with execution.hold_maintenance():
        validate_host_state(LEASE_DIRECTORY.parent)
        await run_thread_shielded(
            run_qualification,
            args.probe,
            args.model,
            args.gpu_layers,
            args.slots,
            args.budget,
            args.steps,
            args.gtt_peak,
            args.host_peak,
            args.comparer,
            args.evict_idle,
            args.stream_only,
            args.concurrent_model,
            args.context_tokens,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Set MOE_PROBE, MOE_MODEL, MOE_GPU_LAYERS, MOE_SLOTS, MOE_BUDGET_BYTES, "
            "MOE_FIT_GTT_PEAK_BYTES, and MOE_FIT_HOST_PEAK_BYTES. Use --context-tokens "
            "to select the allocation size exercised by each probe. Set "
            "REDQUEEN_SOURCE_ROOT only when running this script outside its checkout. Peak byte values "
            "must come from the pinned runtime's allocation-free fit planner."
        ),
    )
    parser.add_argument(
        "--stream-only",
        action="store_true",
        help="run only the SSD-backed generation smoke when a resident baseline cannot fit safely",
    )
    parser.add_argument(
        "--evict-idle",
        action="store_true",
        help="unload ready, idle llama-swap text workers under the maintenance lease and restore them afterward",
    )
    parser.add_argument(
        "--concurrent-model",
        type=Path,
        help="load and decode a second streamed model concurrently; requires --stream-only and combined fit peaks",
    )
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument(
        "--context-tokens",
        type=int,
        default=512,
        help="allocate this KV context in each probe, up to the 220000-token serving limit",
    )
    parser.add_argument("--gpu-layers", type=int, default=None)
    parser.add_argument("--slots", type=int, default=None)
    parser.add_argument("--budget-bytes", type=int, default=None)
    parser.add_argument("--gtt-peak-bytes", type=int, default=None)
    parser.add_argument("--host-peak-bytes", type=int, default=None)
    args = parser.parse_args()
    try:
        args.probe = required_path("MOE_PROBE", executable=True)
        args.model = required_path("MOE_MODEL")
        if args.concurrent_model is not None:
            args.concurrent_model = args.concurrent_model.expanduser().resolve(strict=True)
            if not args.concurrent_model.is_file() or args.concurrent_model == args.model:
                raise RuntimeError("--concurrent-model must be a different regular model file")
            if not args.stream_only:
                raise RuntimeError("--concurrent-model requires --stream-only")
        args.comparer = Path(__file__).with_name("compare-moe-logits.py").resolve(strict=True)
        args.gpu_layers = (
            positive_int("MOE_GPU_LAYERS") if args.gpu_layers is None else args.gpu_layers
        )
        args.slots = nonnegative_int("MOE_SLOTS") if args.slots is None else args.slots
        args.budget = nonnegative_int("MOE_BUDGET_BYTES") if args.budget_bytes is None else args.budget_bytes
        args.gtt_peak = (
            positive_int("MOE_FIT_GTT_PEAK_BYTES")
            if args.gtt_peak_bytes is None
            else args.gtt_peak_bytes
        )
        args.host_peak = (
            positive_int("MOE_FIT_HOST_PEAK_BYTES")
            if args.host_peak_bytes is None
            else args.host_peak_bytes
        )
        if args.steps <= 0 or args.steps > 256:
            raise RuntimeError("--steps must be in 1..256")
        if args.context_tokens <= 0 or args.context_tokens > 220000:
            raise RuntimeError("--context-tokens must be in 1..220000")
        if (
            args.gpu_layers <= 0
            or args.slots < 0
            or args.slots > (1 << 32) - 1
            or args.budget < 0
            or not (args.slots or args.budget)
        ):
            raise RuntimeError("GPU layers must be positive and streaming slots/budget must fit the native API")
        if not 0 < args.gtt_peak <= (1 << 64) - 1 or not 0 < args.host_peak <= (1 << 64) - 1:
            raise RuntimeError("fit planner peaks must be positive uint64 byte counts")
        if args.budget > args.gtt_peak:
            raise RuntimeError("stream cache budget cannot exceed the planned GTT peak")
        asyncio.run(run_under_maintenance(args))
    except (OSError, RuntimeError, ValueError, asyncio.CancelledError) as error:
        print(f"MoE qualification failed safely: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
