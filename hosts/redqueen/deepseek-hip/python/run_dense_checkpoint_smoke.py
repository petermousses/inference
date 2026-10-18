"""Load pinned non-routed 1017 weights on HIP under Redqueen memory ceilings."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import time
from urllib.request import urlopen

from checkpoint_mapping_audit import (
    _execute_pinned_model,
    _pinned_inference_imports,
    verify_pinned_inference_sources,
)
from checkpoint_forward_smoke import run_checkpoint_forward_smoke
from checkpoint_reader import SafeTensorCheckpoint
from dense_checkpoint import (
    DenseLoadMemoryEstimate,
    DenseTensorPlan,
    load_dense_checkpoint,
)
from session_state import SessionStateMemoryEstimate, load_session_state, plan_session_state
from torch_checkpoint import read_rows_to_device


_GIB = 1024**3
_MAX_MANAGED_GTT = 88 * _GIB
_MAX_MANAGED_PHYSICAL = 104 * _GIB
_MIN_HOST_AVAILABLE = 8 * _GIB
_PREFILL_HEADROOM = 5 * _GIB
_HOST_PREFILL_HEADROOM = 5 * _GIB
_GTT_RELEASE_TOLERANCE = 256 * 1024 * 1024
_SESSION_ACCOUNTING_TOLERANCE = 16 * 1024 * 1024
_COMFYUI_QUEUE_URL = "http://127.0.0.1:8189/queue"
_COMFYUI_QUEUE_LIMIT = 64 * 1024


def _comfyui_state() -> str:
    try:
        result = subprocess.run(
            [
                "/usr/bin/systemctl",
                "--user",
                "show",
                "comfyui.service",
                "--property=ActiveState",
                "--value",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("cannot inspect ComfyUI service state") from exc
    state = result.stdout.strip()
    if result.returncode or state not in {
        "active",
        "inactive",
        "failed",
        "activating",
        "deactivating",
    }:
        raise RuntimeError("ComfyUI service state is unknown")
    return state


def _comfyui_queue_idle() -> bool:
    try:
        with urlopen(_COMFYUI_QUEUE_URL, timeout=2) as response:
            body = response.read(_COMFYUI_QUEUE_LIMIT + 1)
    except (OSError, TimeoutError) as exc:
        raise RuntimeError("cannot verify the ComfyUI queue") from exc
    if len(body) > _COMFYUI_QUEUE_LIMIT:
        raise RuntimeError("ComfyUI queue response exceeded its size limit")
    try:
        queue = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("ComfyUI returned an invalid queue response") from exc
    if (
        not isinstance(queue, dict)
        or not isinstance(queue.get("queue_running"), list)
        or not isinstance(queue.get("queue_pending"), list)
    ):
        raise RuntimeError("ComfyUI returned an invalid queue response")
    return not queue["queue_running"] and not queue["queue_pending"]


def _comfyui_systemctl(action: str) -> None:
    try:
        subprocess.run(
            ["/usr/bin/systemctl", "--user", action, "comfyui.service"],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"could not {action} ComfyUI for the checkpoint smoke") from exc


def _wait_for_comfyui_state(expected: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        state = _comfyui_state()
        if state == expected or (expected == "inactive" and state == "failed"):
            return
        if state == "failed" or time.monotonic() >= deadline:
            raise RuntimeError(f"ComfyUI did not reach {expected} state")
        time.sleep(0.25)


@contextmanager
def _comfyui_smoke_window():
    """Stop an idle direct ComfyUI listener during checkpoint GPU allocation."""
    state = _comfyui_state()
    if state in {"inactive", "failed"}:
        yield
        return
    if state != "active":
        raise RuntimeError("ComfyUI is transitioning; refusing checkpoint GPU allocation")
    if not _comfyui_queue_idle():
        raise RuntimeError("ComfyUI has active or pending work; refusing checkpoint GPU allocation")
    try:
        _comfyui_systemctl("stop")
        _wait_for_comfyui_state("inactive", timeout=60)
        yield
    finally:
        _comfyui_systemctl("start")
        _wait_for_comfyui_state("active", timeout=180)


def _run_with_host_isolation(options: argparse.Namespace) -> int:
    """Build and run while scheduler, deployment, and ComfyUI work is isolated."""
    if options.forward_smoke and not os.environ.get("DS4_HIP_LIBRARY"):
        adapter_dir = Path(__file__).resolve().parent
        subprocess.run(
            [str(adapter_dir.parent / "build.sh")], check=True, timeout=600
        )
        build_directory = Path(
            os.environ.get("BUILD_DIR", str(adapter_dir.parent / "build"))
        ).expanduser()
        os.environ["DS4_HIP_LIBRARY"] = str(
            build_directory / "libredqueen_ds4_hip.so"
        )
    with _comfyui_smoke_window():
        return _run(options)


def _memory_snapshot() -> tuple[int, int, int]:
    try:
        gtt_total = int(Path("/sys/class/drm/card0/device/mem_info_gtt_total").read_text())
        gtt_used = int(Path("/sys/class/drm/card0/device/mem_info_gtt_used").read_text())
        memory = Path("/proc/meminfo").read_text(encoding="ascii")
    except (OSError, ValueError) as exc:
        raise RuntimeError("cannot read Redqueen GTT and host memory availability") from exc
    host_available_kib = None
    for line in memory.splitlines():
        if line.startswith("MemAvailable:"):
            fields = line.split()
            if len(fields) >= 2 and fields[1].isdigit():
                host_available_kib = int(fields[1])
            break
    if host_available_kib is None or gtt_used > gtt_total:
        raise RuntimeError("Redqueen memory availability values are invalid")
    return gtt_total - gtt_used, host_available_kib * 1024, gtt_total


def _check_admission(estimate) -> tuple[int, int, int]:
    if estimate.device_peak_bytes > _MAX_MANAGED_GTT:
        raise RuntimeError("model allocation exceeds the 88 GiB managed GTT ceiling")
    if estimate.physical_peak_bytes > _MAX_MANAGED_PHYSICAL:
        raise RuntimeError("model allocation exceeds the 104 GiB managed physical-memory ceiling")
    free_gtt, host_available, _gtt_total = _memory_snapshot()
    if free_gtt < estimate.device_peak_bytes + _PREFILL_HEADROOM:
        raise RuntimeError("model allocation would consume the 5 GiB prefill headroom")
    if host_available - estimate.physical_peak_bytes < _MIN_HOST_AVAILABLE + _HOST_PREFILL_HEADROOM:
        raise RuntimeError("model allocation would violate the 8 GiB host minimum plus 5 GiB prefill headroom")
    return free_gtt, host_available, _gtt_total


def _combined_startup_estimate(dense, session):
    return SimpleNamespace(
        resident_device_bytes=dense.resident_device_bytes + session.resident_device_bytes,
        device_peak_bytes=max(
            dense.device_peak_bytes,
            dense.resident_device_bytes + session.device_peak_bytes,
        ),
        physical_peak_bytes=max(
            dense.physical_peak_bytes,
            dense.resident_device_bytes + session.physical_peak_bytes,
        ),
    )


def _verify_loaded_categories(model, plan, torch) -> None:
    state = model.state_dict(keep_vars=True)
    for item in plan.tensors:
        if item.target_name not in state or state[item.target_name].is_meta:
            raise RuntimeError(f"dense tensor was not materialized: {item.target_name}")
    for name in plan.excluded_target_names:
        if name not in state or not state[name].is_meta:
            raise RuntimeError(f"excluded expert or Engram table was materialized: {name}")
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("the full dense load smoke requires a ROCm/HIP device")


def _verify_converted_samples(checkpoint, model, plan, device, torch) -> tuple[str, str]:
    parameter_names = {name for name, _parameter in model.named_parameters()}
    direct_item = next(
        (
            item
            for item in plan.tensors
            if item.operation == "copy" and item.shape and item.target_name in parameter_names
        ),
        None,
    )
    cast_item = next(
        (
            item
            for item in plan.tensors
            if item.operation == "cast" and item.shape and item.target_name in parameter_names
        ),
        None,
    )
    wo_a_item = next((item for item in plan.tensors if item.operation == "wo-a"), None)
    if direct_item is None or cast_item is None or wo_a_item is None:
        raise RuntimeError("dense plan lacks direct, cast, or wo_a tensors for sample verification")

    direct_info = checkpoint.tensors[direct_item.source_name]
    direct_source = read_rows_to_device(
        checkpoint,
        direct_item.source_name,
        [0],
        device=device,
        max_bytes=direct_info.row_bytes,
    )
    direct_loaded = model.get_parameter(direct_item.target_name)[0:1]
    if not torch.equal(direct_source, direct_loaded):
        raise AssertionError(f"direct tensor sample differs: {direct_item.target_name}")

    cast_info = checkpoint.tensors[cast_item.source_name]
    cast_source = read_rows_to_device(
        checkpoint,
        cast_item.source_name,
        [0],
        device=device,
        max_bytes=cast_info.row_bytes,
    )
    cast_loaded = model.get_parameter(cast_item.target_name)[0:1]
    if not torch.equal(cast_source.to(dtype=cast_loaded.dtype), cast_loaded):
        raise AssertionError(f"official dtype cast sample differs: {cast_item.target_name}")

    if wo_a_item.scale_source_name is None or wo_a_item.block_shape is None:
        raise RuntimeError("wo_a sample plan is missing its scale source")
    output_block, input_block = wo_a_item.block_shape
    weight_info = checkpoint.tensors[wo_a_item.source_name]
    scale_info = checkpoint.tensors[wo_a_item.scale_source_name]
    weight = read_rows_to_device(
        checkpoint,
        wo_a_item.source_name,
        range(output_block),
        device=device,
        max_bytes=output_block * weight_info.row_bytes,
    )
    scale = read_rows_to_device(
        checkpoint,
        wo_a_item.scale_source_name,
        [0],
        device=device,
        max_bytes=scale_info.row_bytes,
    )
    expected = (
        weight.unflatten(0, (-1, output_block))
        .unflatten(-1, (-1, input_block))
        .float()
        * scale[:, None, :, None].float()
    ).flatten(2, 3).flatten(0, 1).to(torch.bfloat16)
    actual = model.get_parameter(wo_a_item.target_name)[:output_block]
    torch.cuda.synchronize(device)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    return direct_item.target_name, wo_a_item.target_name


def _buffer_at(model, name: str):
    module_name, separator, attribute = name.rpartition(".")
    module = model.get_submodule(module_name) if separator else model
    return module._buffers[attribute]


def _verify_session_state(model, state, frequency_factory, device, torch) -> None:
    factory_cache_clear = getattr(frequency_factory, "cache_clear", None)
    if callable(factory_cache_clear):
        factory_cache_clear()
    try:
        for allocation in state.plan.allocations:
            values = [_buffer_at(model, name) for name in allocation.names]
            if not values or any(value.is_meta or value.device != device for value in values):
                raise RuntimeError(f"session buffer is missing from the HIP device: {allocation.names}")
            if allocation.kind == "freqs_cis":
                if any(value is not values[0] for value in values[1:]):
                    raise RuntimeError(f"shared rotary table was duplicated: {allocation.names}")
                with torch.device(device):
                    expected = frequency_factory(*allocation.frequency_key)
                if expected.device != device or not torch.equal(values[0], expected):
                    raise AssertionError(f"official rotary values differ: {allocation.names[0]}")
                del expected
            elif allocation.kind == "score_state":
                if not torch.isneginf(values[0]).all().item():
                    raise AssertionError(f"compressor score state was not initialized to -inf: {allocation.names[0]}")
            elif torch.count_nonzero(values[0]).item() != 0:
                raise AssertionError(f"session cache was not zero initialized: {allocation.names[0]}")
    finally:
        if callable(factory_cache_clear):
            factory_cache_clear()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load dense DeepSeek 1017 weights and a 220k-token session on Redqueen."
    )
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument(
        "--forward-smoke",
        action="store_true",
        help="run one checkpoint-backed native HIP prefill and DSpark decode step",
    )
    parser.add_argument(
        "--lease-directory",
        type=Path,
        default=Path(
            os.environ.get(
                "DS4_LEASE_DIRECTORY",
                "/srv/ai/model-router/run/leases",
            )
        ).expanduser(),
        help="scheduler lease directory used to drain brokered model work",
    )
    options = parser.parse_args()
    if options.estimate_only:
        return _run(options)

    adapter_source = Path(__file__).resolve().parents[2] / "adapters"
    sys.path.insert(0, str(adapter_source))
    try:
        from redqueen_adapters.scheduler_client import (
            SerializedExecution,
            run_thread_shielded,
        )
    except ImportError as exc:
        raise RuntimeError("host scheduler lease implementation is unavailable") from exc

    async def run_under_host_leases() -> int:
        execution = SerializedExecution(options.lease_directory)
        async with execution.hold_maintenance():
            run_directory = options.lease_directory.parent
            for name in ("deployment.json", "maintenance.json"):
                marker = run_directory / name
                if marker.exists() or marker.is_symlink():
                    raise RuntimeError(f"host maintenance marker is present: {marker}")
            recovery_path = run_directory / "recovery.json"
            if recovery_path.exists() or recovery_path.is_symlink():
                try:
                    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    raise RuntimeError("host recovery state is unreadable") from exc
                if not isinstance(recovery, dict) or recovery.get("state") in {
                    "recovering",
                    "quarantined",
                }:
                    raise RuntimeError("host recovery is pending or quarantined")
            print(
                "scheduler FIFO execution and host operations leases acquired; "
                "running isolated checkpoint smoke",
                flush=True,
            )
            return await run_thread_shielded(_run_with_host_isolation, options)

    return asyncio.run(run_under_host_leases())


def _run(options: argparse.Namespace) -> int:
    """Load, execute, and tear down the checkpoint smoke under its host leases."""

    inference_dir = Path(
        os.environ.get(
            "DS4_INFERENCE_DIR", "/srv/ai/models/deepseek-v4.1-flash-1017/inference"
        )
    ).expanduser()
    checkpoint_root = Path(
        os.environ.get("DS4_CHECKPOINT_DIR", str(inference_dir.parent))
    ).expanduser()
    adapter_dir = Path(__file__).resolve().parent
    source_bytes = verify_pinned_inference_sources(inference_dir)
    model_path = inference_dir / "model.py"

    previous_path = sys.path[:]
    previous_kernel = sys.modules.pop("kernel", None)
    sys.path[:0] = [str(adapter_dir), str(inference_dir)]
    device = None
    torch = None
    try:
        import kernel  # noqa: PLC0415

        sys.modules["kernel"] = kernel
        try:
            from transformers import AutoTokenizer
            import torch
        except ImportError as exc:
            raise RuntimeError(f"dense smoke dependency is unavailable: {exc.name}") from exc
        if not torch.cuda.is_available() or torch.version.hip is None:
            raise RuntimeError("the full dense load smoke requires Redqueen ROCm/HIP")
        device = torch.device("cuda:0")

        with _pinned_inference_imports(source_bytes, inference_dir):
            reference = _execute_pinned_model(source_bytes["model.py"], model_path)
            model_config = json.loads(source_bytes["config.json"].decode("utf-8"))
            if not isinstance(model_config, dict):
                raise RuntimeError("pinned model config must be an object")
            previous_dtype = torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.bfloat16)
                args = reference["ModelArgs"](**model_config)
                args.max_batch_size = 1
                args.max_seq_len = 220000
                tokenizer = AutoTokenizer.from_pretrained(
                    str(checkpoint_root), local_files_only=True
                )
                with torch.device("meta"):
                    model = reference["Transformer"](args, tokenizer)
            finally:
                torch.set_default_dtype(previous_dtype)

            with SafeTensorCheckpoint(
                checkpoint_root,
                expected_revision=kernel.REFERENCE_COMMIT,
            ) as checkpoint:
                memory = {
                    "active_leases": 0,
                    "free_gtt_before_load": None,
                    "session_allocated_before": None,
                    "session_reserved_before": None,
                }

                def admission(estimate):
                    component_free, component_host, component_total = _check_admission(estimate)
                    combined = None
                    if isinstance(estimate, DenseLoadMemoryEstimate):
                        session_plan = plan_session_state(model, args)
                        combined = _combined_startup_estimate(estimate, session_plan.estimate)
                        combined_free, combined_host, combined_total = _check_admission(combined)
                        summary = (
                            f"tensors={estimate.dense_tensor_count}; "
                            f"experts kept meta={estimate.routed_expert_target_bytes / _GIB:.2f} GiB; "
                            f"Engram tables kept meta={estimate.engram_table_target_bytes / _GIB:.2f} GiB"
                        )
                        prefix = "dense load"
                    elif isinstance(estimate, SessionStateMemoryEstimate):
                        summary = (
                            f"buffers={estimate.buffer_count}; rotary={estimate.rotary_bytes / _GIB:.3f} GiB; "
                            f"windows={estimate.window_kv_bytes / _GIB:.3f} GiB; "
                            f"compressed KV={estimate.compressed_kv_bytes / _GIB:.3f} GiB; "
                            f"index cache={estimate.index_cache_bytes / _GIB:.3f} GiB"
                        )
                        prefix = "220k session"
                    else:
                        raise TypeError(f"unsupported smoke reservation estimate: {type(estimate).__name__}")
                    print(
                        f"{prefix} admission: {summary}; "
                        f"resident={estimate.resident_device_bytes / _GIB:.2f} GiB; "
                        f"GTT peak={estimate.device_peak_bytes / _GIB:.2f} GiB "
                        f"(free={component_free / _GIB:.2f}/{component_total / _GIB:.2f} GiB); "
                        f"physical peak={estimate.physical_peak_bytes / _GIB:.2f} GiB; "
                        f"host available={component_host / _GIB:.2f} GiB",
                        flush=True,
                    )
                    if combined is not None:
                        print(
                            "combined dense plus 220k session estimate: "
                            f"resident={combined.resident_device_bytes / _GIB:.2f} GiB; "
                            f"GTT peak={combined.device_peak_bytes / _GIB:.2f} GiB "
                            f"(free={combined_free / _GIB:.2f}/{combined_total / _GIB:.2f} GiB); "
                            f"physical peak={combined.physical_peak_bytes / _GIB:.2f} GiB; "
                            f"host available={combined_host / _GIB:.2f} GiB",
                            flush=True,
                        )
                    if options.estimate_only:
                        raise SystemExit(0)

                    class SmokeReservation:
                        def __enter__(self):
                            free_gtt, _host_available, _gtt_total = _check_admission(estimate)
                            if isinstance(estimate, DenseLoadMemoryEstimate):
                                memory["free_gtt_before_load"] = free_gtt
                            elif isinstance(estimate, SessionStateMemoryEstimate):
                                torch.cuda.synchronize(device)
                                torch.cuda.reset_peak_memory_stats(device)
                                memory["session_allocated_before"] = torch.cuda.memory_allocated(device)
                                memory["session_reserved_before"] = torch.cuda.memory_reserved(device)
                            memory["active_leases"] += 1
                            return self

                        def __exit__(self, exc_type, exc, traceback):
                            memory["active_leases"] -= 1
                            if memory["active_leases"] < 0:
                                raise RuntimeError("smoke reservation lease count underflow")
                            return False

                    return SmokeReservation()

                def fence():
                    if memory["active_leases"] <= 0:
                        raise RuntimeError("model state was fenced outside memory admission")
                    torch.cuda.synchronize(device)

                def release_allocator():
                    if memory["active_leases"] <= 0:
                        raise RuntimeError("allocator release escaped memory admission")
                    torch.cuda.empty_cache()

                def progress(done: int, total: int, item: DenseTensorPlan):
                    if done == 1 or done % 64 == 0 or done == total:
                        print(
                            f"dense load progress: {done}/{total}; latest={item.target_name}",
                            flush=True,
                        )

                dense_state = None
                session_state = None
                try:
                    dense_state = load_dense_checkpoint(
                        checkpoint,
                        model,
                        device=device,
                        admission=admission,
                        fence=fence,
                        release_allocator=release_allocator,
                        progress=progress,
                    )
                    plan = dense_state.plan
                    _verify_loaded_categories(model, plan, torch)
                    direct_name, wo_a_name = _verify_converted_samples(
                        checkpoint, model, plan, device, torch
                    )
                    if model.max_seq_len != args.max_seq_len:
                        raise RuntimeError("meta model was not constructed for the 220k session limit")

                    def reset_runtime_references():
                        shared = reference["shared_attn"]
                        for name in ("compress_kv", "index_k", "topk_idxs", "candidates"):
                            setattr(shared, name, None)
                        for module in model.modules():
                            if module.__class__.__name__ == "Indexer":
                                module.freqs_cis = None

                    session_state = load_session_state(
                        model,
                        args,
                        device=device,
                        frequency_factory=reference["precompute_freqs_cis"],
                        admission=admission,
                        fence=fence,
                        reset_runtime_references=reset_runtime_references,
                        release_allocator=release_allocator,
                    )
                    session_plan = session_state.plan
                    session_allocated_before = memory["session_allocated_before"]
                    session_reserved_before = memory["session_reserved_before"]
                    if session_allocated_before is None or session_reserved_before is None:
                        raise RuntimeError("session allocation and allocator baselines were not captured")
                    torch.cuda.synchronize(device)
                    session_allocated_after = torch.cuda.memory_allocated(device)
                    actual_session_bytes = session_allocated_after - session_allocated_before
                    if abs(actual_session_bytes - session_plan.estimate.resident_device_bytes) > _SESSION_ACCOUNTING_TOLERANCE:
                        raise RuntimeError(
                            "session tensor allocations differ from the admitted resident estimate: "
                            f"actual={actual_session_bytes}, "
                            f"estimated={session_plan.estimate.resident_device_bytes}"
                        )
                    session_peak_bytes = torch.cuda.max_memory_allocated(device) - session_allocated_before
                    if session_peak_bytes > session_plan.estimate.device_peak_bytes + _SESSION_ACCOUNTING_TOLERANCE:
                        raise RuntimeError(
                            "session allocation peak exceeded the admitted estimate: "
                            f"actual={session_peak_bytes}, estimated={session_plan.estimate.device_peak_bytes}"
                        )
                    session_reserved_peak_bytes = (
                        torch.cuda.max_memory_reserved(device) - session_reserved_before
                    )
                    if session_reserved_peak_bytes > session_plan.estimate.device_peak_bytes + _SESSION_ACCOUNTING_TOLERANCE:
                        raise RuntimeError(
                            "session allocator-reserved peak exceeded the admitted estimate: "
                            f"actual={session_reserved_peak_bytes}, "
                            f"estimated={session_plan.estimate.device_peak_bytes}"
                        )
                    _verify_session_state(
                        model,
                        session_state,
                        reference["precompute_freqs_cis"],
                        device,
                        torch,
                    )
                    if options.forward_smoke:
                        run_checkpoint_forward_smoke(
                            kernel=kernel,
                            reference=reference,
                            model=model,
                            args=args,
                            tokenizer=tokenizer,
                            checkpoint=checkpoint,
                            device=device,
                            torch=torch,
                            memory=memory,
                            admission_check=_check_admission,
                            fence=fence,
                            release_allocator=release_allocator,
                        )
                    session_state.close()
                    session_allocated_after_close = torch.cuda.memory_allocated(device)
                    if abs(session_allocated_after_close - session_allocated_before) > _SESSION_ACCOUNTING_TOLERANCE:
                        raise RuntimeError("session close retained allocated CUDA tensor storage")
                    if any(
                        not container[attribute].is_meta
                        for allocation in session_plan.allocations
                        for container, attribute, _original in allocation.locations
                    ):
                        raise RuntimeError("session close did not restore buffers to meta state")
                    dense_state.close()
                    if not all(
                        tensor.is_meta for tensor in model.state_dict(keep_vars=True).values()
                    ):
                        raise RuntimeError("dense close did not restore checkpoint state to meta")
                    free_gtt_after, _, _ = _memory_snapshot()
                    free_gtt_before = memory["free_gtt_before_load"]
                    if free_gtt_before is None or (
                        free_gtt_after + _GTT_RELEASE_TOLERANCE < free_gtt_before
                    ):
                        raise RuntimeError(
                            "model-state close did not return GTT backing to the pre-load baseline: "
                            f"before={free_gtt_before}, after={free_gtt_after}"
                        )
                finally:
                    if session_state is not None and not session_state._closed and not session_state._quarantined:
                        session_state.close()
                    if dense_state is not None and not dense_state._closed and not dense_state._quarantined:
                        dense_state.close()

                print(
                    "pinned dense checkpoint and 220k session-state HIP smoke: PASS; "
                    f"hip={torch.version.hip}; loaded={plan.estimate.dense_tensor_count} tensors; "
                    f"resident={plan.estimate.resident_device_bytes / _GIB:.2f} GiB; "
                    f"session={session_plan.estimate.resident_device_bytes / _GIB:.3f} GiB; "
                    f"session allocation delta={actual_session_bytes / _GIB:.3f} GiB; "
                    f"session peak={session_peak_bytes / _GIB:.3f} GiB; "
                    f"allocator-reserved peak={session_reserved_peak_bytes / _GIB:.3f} GiB; "
                    f"direct={direct_name}; wo_a={wo_a_name}; "
                    "routed experts and Engram tables remained meta; all 220k session buffers matched "
                    "official initialization; close restored checkpoint/session state and returned GTT backing"
                )
    finally:
        sys.path[:] = previous_path
        sys.modules.pop("kernel", None)
        if previous_kernel is not None:
            sys.modules["kernel"] = previous_kernel

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
