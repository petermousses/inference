"""Load pinned non-routed 1017 weights on HIP under Redqueen memory ceilings."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from checkpoint_mapping_audit import (
    _execute_pinned_model,
    _pinned_inference_imports,
    verify_pinned_inference_sources,
)
from checkpoint_reader import SafeTensorCheckpoint
from dense_checkpoint import (
    DenseLoadMemoryEstimate,
    DenseTensorPlan,
    load_dense_checkpoint,
)
from torch_checkpoint import read_rows_to_device


_GIB = 1024**3
_MAX_MANAGED_GTT = 88 * _GIB
_MAX_MANAGED_PHYSICAL = 104 * _GIB
_MIN_HOST_AVAILABLE = 8 * _GIB
_PREFILL_HEADROOM = 5 * _GIB
_GTT_RELEASE_TOLERANCE = 256 * 1024 * 1024


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


def _check_admission(estimate: DenseLoadMemoryEstimate) -> tuple[int, int, int]:
    if estimate.device_peak_bytes > _MAX_MANAGED_GTT:
        raise RuntimeError("dense load exceeds the 88 GiB managed GTT ceiling")
    if estimate.physical_peak_bytes > _MAX_MANAGED_PHYSICAL:
        raise RuntimeError("dense load exceeds the 104 GiB managed physical-memory ceiling")
    free_gtt, host_available, _gtt_total = _memory_snapshot()
    if free_gtt < estimate.device_peak_bytes + _PREFILL_HEADROOM:
        raise RuntimeError("dense load would consume the 5 GiB prefill headroom")
    if host_available - estimate.physical_peak_bytes < _MIN_HOST_AVAILABLE:
        raise RuntimeError("dense load would leave less than 8 GiB host memory available")
    return free_gtt, host_available, _gtt_total


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load dense DeepSeek 1017 state on Redqueen under managed memory ceilings."
    )
    parser.add_argument("--estimate-only", action="store_true")
    options = parser.parse_args()

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
                args.max_seq_len = 1
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
                memory = {"active": False, "free_gtt_before_load": None}

                def admission(estimate: DenseLoadMemoryEstimate):
                    free_gtt, host_available, gtt_total = _check_admission(estimate)
                    print(
                        "dense load admission: "
                        f"tensors={estimate.dense_tensor_count}, "
                        f"resident={estimate.resident_device_bytes / _GIB:.2f} GiB, "
                        f"GTT peak={estimate.device_peak_bytes / _GIB:.2f} GiB "
                        f"(free={free_gtt / _GIB:.2f}/{gtt_total / _GIB:.2f} GiB), "
                        f"physical peak={estimate.physical_peak_bytes / _GIB:.2f} GiB, "
                        f"host available={host_available / _GIB:.2f} GiB, "
                        f"experts kept meta={estimate.routed_expert_target_bytes / _GIB:.2f} GiB, "
                        f"Engram tables kept meta={estimate.engram_table_target_bytes / _GIB:.2f} GiB",
                        flush=True,
                    )
                    if options.estimate_only:
                        raise SystemExit(0)

                    class SmokeReservation:
                        def __enter__(self):
                            free_gtt, _host_available, _gtt_total = _check_admission(estimate)
                            memory["free_gtt_before_load"] = free_gtt
                            memory["active"] = True
                            return self

                        def __exit__(self, exc_type, exc, traceback):
                            memory["active"] = False
                            return False

                    return SmokeReservation()

                def fence():
                    if not memory["active"]:
                        raise RuntimeError("dense state was fenced outside its memory admission")
                    torch.cuda.synchronize(device)

                def release_allocator():
                    if not memory["active"]:
                        raise RuntimeError("dense allocator release escaped its memory admission")
                    torch.cuda.empty_cache()

                def progress(done: int, total: int, item: DenseTensorPlan):
                    if done == 1 or done % 64 == 0 or done == total:
                        print(
                            f"dense load progress: {done}/{total}; latest={item.target_name}",
                            flush=True,
                        )

                state = None
                try:
                    state = load_dense_checkpoint(
                        checkpoint,
                        model,
                        device=device,
                        admission=admission,
                        fence=fence,
                        release_allocator=release_allocator,
                        progress=progress,
                    )
                    plan = state.plan
                    _verify_loaded_categories(model, plan, torch)
                    direct_name, wo_a_name = _verify_converted_samples(
                        checkpoint, model, plan, device, torch
                    )
                    state.close()
                    if not all(
                        tensor.is_meta for tensor in model.state_dict(keep_vars=True).values()
                    ):
                        raise RuntimeError("dense close did not restore the graph to meta state")
                    free_gtt_after, _, _ = _memory_snapshot()
                    free_gtt_before = memory["free_gtt_before_load"]
                    if free_gtt_before is None or (
                        free_gtt_after + _GTT_RELEASE_TOLERANCE < free_gtt_before
                    ):
                        raise RuntimeError(
                            "dense close did not return GTT backing to the pre-load baseline: "
                            f"before={free_gtt_before}, after={free_gtt_after}"
                        )
                finally:
                    if state is not None and not state._closed and not state._quarantined:
                        state.close()

                print(
                    "pinned dense checkpoint HIP smoke: PASS; "
                    f"hip={torch.version.hip}; loaded={plan.estimate.dense_tensor_count} tensors; "
                    f"resident={plan.estimate.resident_device_bytes / _GIB:.2f} GiB; "
                    f"direct={direct_name}; wo_a={wo_a_name}; "
                    "routed experts and Engram tables remained meta; close restored all checkpoint state "
                    "and returned GTT backing to baseline"
                )
    finally:
        sys.path[:] = previous_path
        sys.modules.pop("kernel", None)
        if previous_kernel is not None:
            sys.modules["kernel"] = previous_kernel

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
