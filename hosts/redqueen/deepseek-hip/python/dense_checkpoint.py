"""Admission-gated loading of non-routed DeepSeek 1017 checkpoint state."""

from __future__ import annotations

import gc
import inspect
import math
import threading
from dataclasses import dataclass
from typing import Callable, ContextManager

from checkpoint_mapping_audit import (
    AuditReport,
    MappingAuditError,
    TensorMetadata,
    _dtype_name,
    _official_checkpoint_cast,
    audit_metadata,
    collect_target_state,
    map_source_name,
)
from checkpoint_reader import (
    CheckpointError,
    SafeTensorCheckpoint,
    _MAX_READ_BYTES,
    _MAX_ROWS_PER_READ,
)
from torch_checkpoint import (
    read_rows_to_device,
    read_tensor_to_device,
)


_DEFAULT_CHUNK_BYTES = 64 * 1024 * 1024
_DEFAULT_WO_A_OUTPUT_BYTES = 8 * 1024 * 1024
_HOST_READ_HEADROOM_BYTES = 64 * 1024 * 1024
_DTYPE_BYTES = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "float8_e4m3fn": 1,
    "float8_e4m3fnuz": 1,
    "float8_e5m2": 1,
    "float8_e5m2fnuz": 1,
    "float8_e8m0fnu": 1,
    "float4_e2m1fn_x2": 1,
    "int16": 2,
    "float16": 2,
    "bfloat16": 2,
    "int32": 4,
    "float32": 4,
    "int64": 8,
    "float64": 8,
}
_LIVE_STATES_LOCK = threading.Lock()
_LIVE_STATES: dict[int, object] = {}
_QUARANTINED_LOADS: list[tuple[object, object, BaseException]] = []


@dataclass(frozen=True)
class DenseLoadMemoryEstimate:
    """Peak bytes needed while loading, including the final resident dense state."""

    resident_device_bytes: int
    device_peak_bytes: int
    host_transient_bytes: int
    physical_peak_bytes: int
    dense_tensor_count: int
    wo_a_tensor_count: int
    routed_expert_tensor_count: int
    engram_table_tensor_count: int
    routed_expert_target_bytes: int
    engram_table_target_bytes: int


@dataclass(frozen=True)
class DenseTensorPlan:
    """One checkpoint tensor selected for dense-state materialization."""

    target_name: str
    source_name: str
    operation: str
    shape: tuple[int, ...]
    target_dtype: str
    target_bytes: int
    source_bytes: int
    scale_source_name: str | None = None
    block_shape: tuple[int, int] | None = None


@dataclass(frozen=True)
class DenseCheckpointLoadPlan:
    """Validated checkpoint-to-graph selection and its pre-allocation estimate."""

    tensors: tuple[DenseTensorPlan, ...]
    excluded_target_names: tuple[str, ...]
    estimate: DenseLoadMemoryEstimate
    audit: AuditReport


def _numel(shape: tuple[int, ...]) -> int:
    return math.prod(shape)


def _tensor_nbytes(metadata: TensorMetadata) -> int:
    dtype = _dtype_name(metadata.dtype)
    try:
        element_bytes = _DTYPE_BYTES[dtype]
    except KeyError as exc:
        raise CheckpointError(f"unsupported dense target dtype: {dtype}") from exc
    return _numel(metadata.shape) * element_bytes


def _validate_chunk_limit(value: int, name: str) -> None:
    if type(value) is not int or not 0 < value <= _MAX_READ_BYTES:
        raise ValueError(f"{name} must be between 1 and {_MAX_READ_BYTES} bytes")


def _is_engram_table(name: str) -> bool:
    return ".engram.embed.weight" in name or ".engram.embed.scale" in name


def _plan_wo_a_block(
    target_name: str,
    target: TensorMetadata,
    weight_info,
    scale_info,
) -> tuple[int, int]:
    if (
        len(target.shape) != 2
        or len(weight_info.shape) != 2
        or len(scale_info.shape) != 2
        or weight_info.shape != target.shape
        or target.dtype != "bfloat16"
        or weight_info.dtype not in {"F8_E4M3", "F8_E4M3FN"}
        or scale_info.dtype != "F8_E8M0"
    ):
        raise CheckpointError(f"invalid pinned wo_a tensor metadata: {target_name}")
    if any(dimension <= 0 for dimension in (*target.shape, *scale_info.shape)):
        raise CheckpointError(f"wo_a dimensions must be positive: {target_name}")
    if any(
        target_dim % scale_dim
        for target_dim, scale_dim in zip(target.shape, scale_info.shape)
    ):
        raise CheckpointError(f"wo_a scales do not tile the target: {target_name}")
    block_shape = tuple(
        target_dim // scale_dim
        for target_dim, scale_dim in zip(target.shape, scale_info.shape)
    )
    if block_shape not in {(32, 32), (128, 128)}:
        raise CheckpointError(f"unsupported wo_a scale blocks {block_shape}: {target_name}")
    return block_shape


def _source_mappings(checkpoint: SafeTensorCheckpoint):
    ordinary: dict[str, str] = {}
    scales: dict[str, str] = {}
    for source_name in checkpoint.tensors:
        target_name, mode = map_source_name(source_name)
        if mode == "tied-alias":
            continue
        if mode == "wo-a-scale":
            scales[target_name.removesuffix(".scale") + ".weight"] = source_name
            continue
        if target_name is None:
            raise MappingAuditError(f"checkpoint tensor has no target mapping: {source_name}")
        if target_name in ordinary:
            raise MappingAuditError(f"multiple sources map to {target_name}")
        ordinary[target_name] = source_name
    return ordinary, scales


def plan_dense_checkpoint_load(
    checkpoint: SafeTensorCheckpoint,
    model: object,
    *,
    max_chunk_bytes: int = _DEFAULT_CHUNK_BYTES,
    max_wo_a_output_bytes: int = _DEFAULT_WO_A_OUTPUT_BYTES,
) -> DenseCheckpointLoadPlan:
    """Audit and estimate dense state before the caller admits any device allocations.

    Routed expert weights and the two Engram embedding tables remain on meta. Engram's
    WKV/Q/K state is ordinary dense state and is included. The physical peak includes the
    resident GTT-backed weights plus conservative checkpoint-read buffers.
    """
    if not isinstance(checkpoint, SafeTensorCheckpoint):
        raise TypeError("checkpoint must be a SafeTensorCheckpoint")
    if getattr(checkpoint, "_metadata_only", False):
        raise ValueError("dense checkpoint loading requires a payload-enabled checkpoint reader")
    _validate_chunk_limit(max_chunk_bytes, "max_chunk_bytes")
    _validate_chunk_limit(max_wo_a_output_bytes, "max_wo_a_output_bytes")

    target_state, _aliases = collect_target_state(model)
    audit = audit_metadata(checkpoint.tensors, target_state)
    if not audit.passed:
        preview = "; ".join(audit.errors[:8])
        raise MappingAuditError(f"checkpoint-to-graph audit failed: {preview}")

    ordinary_sources, scale_sources = _source_mappings(checkpoint)
    tensors: list[DenseTensorPlan] = []
    routed_count = 0
    engram_count = 0
    excluded_names: list[str] = []
    routed_bytes = 0
    engram_bytes = 0
    resident_bytes = 0
    copy_scratch_bytes = 0
    host_scratch_bytes = 0
    wo_a_scratch_bytes = 0
    wo_a_count = 0

    for target_name, target in target_state.items():
        target_bytes = _tensor_nbytes(target)
        if ".experts." in target_name:
            routed_count += 1
            routed_bytes += target_bytes
            excluded_names.append(target_name)
            continue
        if _is_engram_table(target_name):
            engram_count += 1
            engram_bytes += target_bytes
            excluded_names.append(target_name)
            continue

        source_name = ordinary_sources.get(target_name)
        if source_name is None:
            raise MappingAuditError(f"dense graph tensor has no checkpoint source: {target_name}")
        source = checkpoint.tensors[source_name]
        if target_name.endswith(".wo_a.weight"):
            scale_source_name = scale_sources.get(target_name)
            if scale_source_name is None:
                raise MappingAuditError(f"wo_a weight has no scale source: {target_name}")
            weight_info = checkpoint.tensors[source_name]
            scale_info = checkpoint.tensors[scale_source_name]
            block_shape = _plan_wo_a_block(target_name, target, weight_info, scale_info)
            output_per_scale_row = block_shape[0] * target.shape[1] * 2
            if output_per_scale_row > max_wo_a_output_bytes:
                raise CheckpointError(
                    f"one wo_a output block exceeds the load chunk limit: {target_name}"
                )
            scale_rows_per_chunk = max(
                1,
                min(
                    scale_info.shape[0],
                    max_wo_a_output_bytes // output_per_scale_row,
                    _MAX_ROWS_PER_READ // block_shape[0],
                ),
            )
            weight_chunk_bytes = (
                scale_rows_per_chunk * block_shape[0] * target.shape[1]
            )
            scale_chunk_bytes = scale_rows_per_chunk * scale_info.shape[1]
            wo_a_scratch_bytes = max(
                wo_a_scratch_bytes,
                11 * weight_chunk_bytes + 5 * scale_chunk_bytes,
            )
            host_scratch_bytes = max(
                host_scratch_bytes,
                2 * max(weight_chunk_bytes, scale_chunk_bytes) + _HOST_READ_HEADROOM_BYTES,
            )
            operation = "wo-a"
            scale_name = scale_source_name
            wo_a_count += 1
            block = block_shape
        else:
            source_dtype = _dtype_name(source.dtype)
            target_dtype = _dtype_name(target.dtype)
            if source_dtype != target_dtype and not _official_checkpoint_cast(
                source_name, target_name, source_dtype, target_dtype
            ):
                raise MappingAuditError(
                    f"unsupported dense dtype conversion {source_dtype} -> {target_dtype}: {target_name}"
                )
            if source.shape and source.row_bytes > max_chunk_bytes:
                raise CheckpointError(
                    f"one row of {source_name} ({source.row_bytes} bytes) exceeds the load chunk limit"
                )
            operation = "copy" if source_dtype == target_dtype else "cast"
            scale_name = None
            block = None
            if source.nbytes:
                if source.shape:
                    rows_per_chunk = max_chunk_bytes // source.row_bytes
                    chunk_bytes = min(
                        source.nbytes,
                        rows_per_chunk * source.row_bytes,
                        _MAX_ROWS_PER_READ * source.row_bytes,
                    )
                else:
                    chunk_bytes = min(source.nbytes, max_chunk_bytes)
                if operation == "cast":
                    elements = chunk_bytes // _DTYPE_BYTES[_dtype_name(source.dtype)]
                    cast_bytes = elements * _DTYPE_BYTES[target_dtype]
                    copy_scratch_bytes = max(
                        copy_scratch_bytes, chunk_bytes + cast_bytes
                    )
                else:
                    copy_scratch_bytes = max(copy_scratch_bytes, chunk_bytes)
                host_scratch_bytes = max(
                    host_scratch_bytes,
                    2 * chunk_bytes + _HOST_READ_HEADROOM_BYTES,
                )

        tensors.append(
            DenseTensorPlan(
                target_name=target_name,
                source_name=source_name,
                operation=operation,
                shape=target.shape,
                target_dtype=_dtype_name(target.dtype),
                target_bytes=target_bytes,
                source_bytes=source.nbytes,
                scale_source_name=scale_name,
                block_shape=block,
            )
        )
        resident_bytes += target_bytes

    if not tensors or resident_bytes <= 0:
        raise CheckpointError("checkpoint contains no dense model state to materialize")
    device_scratch = max(copy_scratch_bytes, wo_a_scratch_bytes)
    device_peak = resident_bytes + device_scratch
    estimate = DenseLoadMemoryEstimate(
        resident_device_bytes=resident_bytes,
        device_peak_bytes=device_peak,
        host_transient_bytes=host_scratch_bytes,
        physical_peak_bytes=resident_bytes + device_scratch + host_scratch_bytes,
        dense_tensor_count=len(tensors),
        wo_a_tensor_count=wo_a_count,
        routed_expert_tensor_count=routed_count,
        engram_table_tensor_count=engram_count,
        routed_expert_target_bytes=routed_bytes,
        engram_table_target_bytes=engram_bytes,
    )
    return DenseCheckpointLoadPlan(
        tensors=tuple(tensors),
        excluded_target_names=tuple(excluded_names),
        estimate=estimate,
        audit=audit,
    )


def _copy_cast_tensor(
    checkpoint: SafeTensorCheckpoint,
    source_name: str,
    destination,
    *,
    max_chunk_bytes: int,
    torch,
) -> None:
    info = checkpoint.tensors[source_name]
    if not info.shape:
        source = read_tensor_to_device(
            checkpoint, source_name, device=destination.device, max_bytes=max_chunk_bytes
        )
        destination.copy_(source.to(dtype=destination.dtype))
        del source
        return
    if not info.nbytes:
        return
    rows_per_chunk = min(
        _MAX_ROWS_PER_READ,
        max_chunk_bytes // info.row_bytes,
    )
    if rows_per_chunk < 1:
        raise CheckpointError(f"one row of {source_name} exceeds the load chunk limit")
    with torch.no_grad():
        for start in range(0, info.shape[0], rows_per_chunk):
            end = min(start + rows_per_chunk, info.shape[0])
            source = read_rows_to_device(
                checkpoint,
                source_name,
                range(start, end),
                device=destination.device,
                max_bytes=(end - start) * info.row_bytes,
            )
            converted = source.to(dtype=destination.dtype)
            destination[start:end].copy_(converted)
            del converted, source


def _copy_tensor(
    checkpoint: SafeTensorCheckpoint,
    source_name: str,
    destination,
    *,
    max_chunk_bytes: int,
    torch,
) -> None:
    """Copy with one live device read chunk so the admission estimate stays valid."""
    info = checkpoint.tensors[source_name]
    if not info.nbytes:
        checkpoint.read_tensor_bytes(source_name, max_bytes=max_chunk_bytes)
        return
    if not info.shape:
        chunk = read_tensor_to_device(
            checkpoint, source_name, device=destination.device, max_bytes=max_chunk_bytes
        )
        destination.copy_(chunk)
        del chunk
        return
    rows_per_chunk = min(_MAX_ROWS_PER_READ, max_chunk_bytes // info.row_bytes)
    if rows_per_chunk < 1:
        raise CheckpointError(f"one row of {source_name} exceeds the load chunk limit")
    with torch.no_grad():
        for start in range(0, info.shape[0], rows_per_chunk):
            end = min(start + rows_per_chunk, info.shape[0])
            chunk = read_rows_to_device(
                checkpoint,
                source_name,
                range(start, end),
                device=destination.device,
                max_bytes=(end - start) * info.row_bytes,
            )
            destination[start:end].copy_(chunk)
            del chunk


def _copy_wo_a(
    checkpoint: SafeTensorCheckpoint,
    item: DenseTensorPlan,
    destination,
    *,
    max_output_bytes: int,
    torch,
) -> None:
    if item.scale_source_name is None or item.block_shape is None:
        raise CheckpointError(f"wo_a plan is missing scale metadata: {item.target_name}")
    weight_info = checkpoint.tensors[item.source_name]
    scale_info = checkpoint.tensors[item.scale_source_name]
    output_block, input_block = item.block_shape
    output_per_scale_row = output_block * item.shape[1] * 2
    rows_per_chunk = max(
        1,
        min(
            scale_info.shape[0],
            max_output_bytes // output_per_scale_row,
            _MAX_ROWS_PER_READ // output_block,
        ),
    )
    if rows_per_chunk <= 0 or output_per_scale_row > max_output_bytes:
        raise CheckpointError(f"one wo_a output block exceeds the load chunk limit: {item.target_name}")
    with torch.no_grad():
        for start in range(0, scale_info.shape[0], rows_per_chunk):
            end = min(start + rows_per_chunk, scale_info.shape[0])
            weight_rows = range(start * output_block, end * output_block)
            weight_chunk_bytes = len(weight_rows) * weight_info.row_bytes
            scale_chunk_bytes = (end - start) * scale_info.row_bytes
            weight = read_rows_to_device(
                checkpoint,
                item.source_name,
                weight_rows,
                device=destination.device,
                max_bytes=weight_chunk_bytes,
            )
            scale = read_rows_to_device(
                checkpoint,
                item.scale_source_name,
                range(start, end),
                device=destination.device,
                max_bytes=scale_chunk_bytes,
            )
            converted = (
                weight.unflatten(0, (-1, output_block))
                .unflatten(-1, (-1, input_block))
                .float()
                * scale[:, None, :, None].float()
            ).flatten(2, 3).flatten(0, 1).to(torch.bfloat16)
            destination[start * output_block : end * output_block].copy_(converted)
            del converted, scale, weight


def _registered_locations(model: object) -> dict[int, list[tuple[str, object, str, object]]]:
    locations: dict[int, list[tuple[str, object, str, object]]] = {}
    for kind, iterator_name in (
        ("parameter", "named_parameters"),
        ("buffer", "named_buffers"),
    ):
        iterator = getattr(model, iterator_name, None)
        if not callable(iterator):
            raise TypeError(f"model must expose {iterator_name}")
        for full_name, tensor in iterator(remove_duplicate=False):
            if tensor is None:
                continue
            module_name, separator, attribute = full_name.rpartition(".")
            if not separator:
                module_name, attribute = "", full_name
            get_submodule = getattr(model, "get_submodule", None)
            if callable(get_submodule):
                module = get_submodule(module_name) if module_name else model
            else:
                module = model
                for part in module_name.split(".") if module_name else ():
                    module = getattr(module, part)
            container_name = "_parameters" if kind == "parameter" else "_buffers"
            container = getattr(module, container_name, None)
            if not isinstance(container, dict) or container.get(attribute) is not tensor:
                raise CheckpointError(f"cannot resolve registered model state: {full_name}")
            locations.setdefault(id(tensor), []).append((kind, container, attribute, tensor))
    return locations


def _call_sync(callback: Callable[[], object], name: str) -> None:
    result = _call_sync_result(callback, name)
    if result is not None:
        raise TypeError(f"{name} must return None")


def _call_sync_result(callback: Callable[[], object], name: str):
    result = callback()
    if inspect.isawaitable(result):
        close = getattr(result, "close", None)
        if callable(close):
            close()
        raise TypeError(f"{name} must complete synchronously")
    return result


def _call_progress(callback, completed: int, total: int, item: DenseTensorPlan) -> None:
    result = callback(completed, total, item)
    if inspect.isawaitable(result):
        close = getattr(result, "close", None)
        if callable(close):
            close()
        raise TypeError("progress callback must complete synchronously")
    if result is not None:
        raise TypeError("progress callback must return None")


def _quarantine_load(
    reservation,
    loaded: dict,
    error: BaseException,
    *,
    inflight: tuple[object, ...] = (),
) -> None:
    retained = {"loaded": loaded, "inflight": tuple(inflight)}
    with _LIVE_STATES_LOCK:
        _QUARANTINED_LOADS.append((reservation, retained, error))


class DenseCheckpointState:
    """Loaded dense parameters holding their admission lease until safely unloaded.

    The caller must drain every complete model execution and GPU operation before closing. The
    loader's fence handles queued copies; it cannot see a forward that is between other layers.
    """

    def __init__(
        self,
        model: object,
        plan: DenseCheckpointLoadPlan,
        reservation,
        restore_locations: list[tuple[str, object, str, object]],
        fence: Callable[[], object],
        release_allocator: Callable[[], object],
    ) -> None:
        self.model = model
        self.plan = plan
        self._reservation = reservation
        self._restore_locations = restore_locations
        self._fence = fence
        self._release_allocator = release_allocator
        self._closed = False
        self._quarantined = False
        self._close_lock = threading.Lock()
        with _LIVE_STATES_LOCK:
            _LIVE_STATES[id(self)] = self

    def __enter__(self) -> DenseCheckpointState:
        if self._closed or self._quarantined:
            raise RuntimeError("dense checkpoint state is not active")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        """Restore meta state, release allocator backing, then release the admission lease."""
        with self._close_lock:
            self._close_once()

    def _close_once(self) -> None:
        if self._closed:
            return
        if self._quarantined:
            raise RuntimeError("dense checkpoint state is quarantined; worker restart is required")
        try:
            _call_sync(self._fence, "GPU fence")
            for _kind, container, attribute, original in reversed(self._restore_locations):
                container[attribute] = original
            self._restore_locations.clear()
            gc.collect()
            _call_sync(self._release_allocator, "release_allocator")
            _call_sync(self._fence, "post-release GPU fence")
            _call_sync_result(
                lambda: self._reservation.__exit__(None, None, None),
                "admission context __exit__",
            )
        except BaseException as exc:
            self._quarantined = True
            with _LIVE_STATES_LOCK:
                _QUARANTINED_LOADS.append((self._reservation, self, exc))
            raise
        self._closed = True
        with _LIVE_STATES_LOCK:
            _LIVE_STATES.pop(id(self), None)


def load_dense_checkpoint(
    checkpoint: SafeTensorCheckpoint,
    model: object,
    *,
    device,
    admission: Callable[[DenseLoadMemoryEstimate], ContextManager[object]],
    fence: Callable[[], object],
    release_allocator: Callable[[], object],
    progress: Callable[[int, int, DenseTensorPlan], object] | None = None,
    max_chunk_bytes: int = _DEFAULT_CHUNK_BYTES,
    max_wo_a_output_bytes: int = _DEFAULT_WO_A_OUTPUT_BYTES,
) -> DenseCheckpointState:
    """Materialize audited dense checkpoint state only after memory admission.

    ``admission`` receives the complete peak estimate and returns a synchronous context manager.
    Its lease remains held for the loaded model lifetime and is released only by ``close`` after
    the caller drains model execution. Routed experts and Engram tables stay meta.
    """
    if not callable(admission):
        raise TypeError("admission must be callable")
    if not callable(fence) or not callable(release_allocator):
        raise TypeError("fence and release_allocator must be callable")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable when provided")
    plan = plan_dense_checkpoint_load(
        checkpoint,
        model,
        max_chunk_bytes=max_chunk_bytes,
        max_wo_a_output_bytes=max_wo_a_output_bytes,
    )
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for dense checkpoint loading") from exc
    target_device = torch.device(device)
    if target_device.type == "meta":
        raise ValueError("dense checkpoint destination must have allocated storage")

    state_dict = model.state_dict(keep_vars=True)
    locations_by_id = _registered_locations(model)
    for item in plan.tensors:
        target = state_dict.get(item.target_name)
        if target is None or not getattr(target, "is_meta", False):
            raise CheckpointError(f"dense graph tensor is missing or already materialized: {item.target_name}")
        locations = locations_by_id.get(id(target), [])
        if not locations:
            raise CheckpointError(f"dense graph tensor is not a registered parameter or buffer: {item.target_name}")
        if len({location[0] for location in locations}) != 1:
            raise CheckpointError(f"dense graph tensor is registered as both parameter and buffer: {item.target_name}")

    reservation = admission(plan.estimate)
    if not callable(getattr(reservation, "__enter__", None)) or not callable(
        getattr(reservation, "__exit__", None)
    ):
        raise TypeError("admission must return a synchronous context manager")
    _call_sync_result(reservation.__enter__, "admission context __enter__")

    loaded: dict[str, object] = {}
    assigned: list[tuple[str, object, str, object]] = []
    target = None
    destination = None
    locations = None
    replacement = None
    installed_state = None
    try:
        for index, item in enumerate(plan.tensors):
            target = state_dict[item.target_name]
            locations = locations_by_id.get(id(target), [])
            if not locations:
                raise CheckpointError(f"dense graph tensor is not a registered parameter or buffer: {item.target_name}")
            kinds = {location[0] for location in locations}
            if len(kinds) != 1:
                raise CheckpointError(f"dense graph tensor is registered as both parameter and buffer: {item.target_name}")
            destination = torch.empty(
                item.shape,
                dtype=target.dtype,
                device=target_device,
            )
            if item.operation == "wo-a":
                _copy_wo_a(
                    checkpoint,
                    item,
                    destination,
                    max_output_bytes=max_wo_a_output_bytes,
                    torch=torch,
                )
            elif item.operation == "copy":
                _copy_tensor(
                    checkpoint,
                    item.source_name,
                    destination,
                    max_chunk_bytes=max_chunk_bytes,
                    torch=torch,
                )
            elif item.operation == "cast":
                _copy_cast_tensor(
                    checkpoint,
                    item.source_name,
                    destination,
                    max_chunk_bytes=max_chunk_bytes,
                    torch=torch,
                )
            else:
                raise CheckpointError(f"unsupported dense load operation: {item.operation}")
            loaded[item.target_name] = (target, destination, locations)
            if progress is not None:
                _call_progress(progress, index + 1, len(plan.tensors), item)

        _call_sync(fence, "GPU load fence")
        for item in plan.tensors:
            target, destination, locations = loaded[item.target_name]
            if locations[0][0] == "parameter":
                replacement = torch.nn.Parameter(
                    destination,
                    requires_grad=target.requires_grad,
                )
            else:
                replacement = destination
            for _kind, container, attribute, original in locations:
                assigned.append((_kind, container, attribute, original))
                container[attribute] = replacement

        installed_state = model.state_dict(keep_vars=True)
        for item in plan.tensors:
            if installed_state[item.target_name].is_meta:
                raise CheckpointError(f"dense parameter remained meta after loading: {item.target_name}")
        for target_name in plan.excluded_target_names:
            if target_name in installed_state and (
                ".experts." in target_name or _is_engram_table(target_name)
            ) and not installed_state[target_name].is_meta:
                raise CheckpointError(f"excluded model state was materialized: {target_name}")
    except BaseException as load_error:
        cleanup_error = None
        try:
            _call_sync(fence, "failed-load GPU fence")
            for _kind, container, attribute, original in reversed(assigned):
                container[attribute] = original
            installed_state = None
            replacement = None
            destination = None
            target = None
            locations = None
            loaded.clear()
            gc.collect()
            _call_sync(release_allocator, "failed-load release_allocator")
            _call_sync(fence, "failed-load post-release GPU fence")
        except BaseException as exc:
            cleanup_error = exc
        if cleanup_error is not None:
            _quarantine_load(
                reservation,
                loaded,
                cleanup_error,
                inflight=(load_error, target, destination, locations, replacement, installed_state),
            )
            if hasattr(load_error, "add_note"):
                load_error.add_note(f"dense load cleanup also failed: {cleanup_error!r}; reservation retained")
            raise load_error from cleanup_error
        try:
            suppressed = _call_sync_result(
                lambda: reservation.__exit__(
                    type(load_error), load_error, load_error.__traceback__
                ),
                "admission context __exit__",
            )
        except BaseException as exc:
            _quarantine_load(reservation, loaded, exc)
            if hasattr(load_error, "add_note"):
                load_error.add_note(f"dense load reservation release failed: {exc!r}")
            raise load_error from exc
        if suppressed:
            raise RuntimeError("admission context managers may not suppress load failures") from load_error
        raise

    return DenseCheckpointState(
        model,
        plan,
        reservation,
        assigned,
        fence,
        release_allocator,
    )
