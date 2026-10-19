"""Admission-gated materialization of the pinned DeepSeek session buffers."""

from __future__ import annotations

import gc
import inspect
import math
import threading
import weakref
from dataclasses import dataclass
from typing import Callable, ContextManager


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
    "complex64": 8,
    "int64": 8,
    "float64": 8,
}
_FREQUENCY_FACTORY_LOCK = threading.Lock()
_MODEL_LIFECYCLE_LOCKS_LOCK = threading.Lock()
_MODEL_LIFECYCLE_LOCKS: weakref.WeakKeyDictionary[object, threading.Lock] = weakref.WeakKeyDictionary()
_LIVE_STATES_LOCK = threading.Lock()
_LIVE_STATES: dict[int, object] = {}
_QUARANTINED_STATES: list[tuple[object, object, BaseException]] = []


@dataclass(frozen=True)
class SessionStateMemoryEstimate:
    """Session state plus the largest transient used to build an official RoPE table."""

    resident_device_bytes: int
    device_peak_bytes: int
    physical_peak_bytes: int
    host_transient_bytes: int
    buffer_count: int
    window_kv_bytes: int
    compressed_kv_bytes: int
    index_cache_bytes: int
    compressor_tail_bytes: int
    rotary_bytes: int
    rotary_workspace_bytes: int
    engram_hash_bytes: int


@dataclass(frozen=True)
class SessionBufferAllocation:
    """One unique buffer allocation; names can include shared module aliases."""

    names: tuple[str, ...]
    kind: str
    shape: tuple[int, ...]
    dtype: object
    nbytes: int
    frequency_key: tuple[int | float, ...] | None
    locations: tuple[tuple[dict, str, object], ...]
    runtime_module: object | None = None
    runtime_attribute: str | None = None


@dataclass(frozen=True)
class SessionStatePlan:
    """Validated allocation plan derived from the official model's meta buffers."""

    allocations: tuple[SessionBufferAllocation, ...]
    estimate: SessionStateMemoryEstimate
    max_batch_size: int
    max_seq_len: int


def _positive_int(value, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _dtype_name(dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _nbytes(tensor) -> int:
    try:
        element_bytes = _DTYPE_BYTES[_dtype_name(tensor.dtype)]
    except KeyError as exc:
        raise ValueError(f"unsupported DeepSeek session buffer dtype: {tensor.dtype}") from exc
    return math.prod(tuple(tensor.shape)) * element_bytes


def _frequency_key(args, compress_ratio: int) -> tuple[int | float, ...]:
    if compress_ratio:
        original_seq_len = args.original_seq_len
        base = args.compress_rope_theta
    else:
        original_seq_len = 0
        base = args.rope_theta
    return (
        args.rope_head_dim,
        args.max_seq_len,
        original_seq_len,
        base,
        args.rope_factor,
        args.beta_fast,
        args.beta_slow,
    )


def _walk_registered_buffers(model):
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        raise TypeError("model must expose named_modules")
    try:
        modules = named_modules(remove_duplicate=False)
    except TypeError as exc:
        raise TypeError("model.named_modules must support remove_duplicate=False") from exc
    for module_name, module in modules:
        buffers = getattr(module, "_buffers", None)
        if not isinstance(buffers, dict):
            continue
        for attribute, tensor in buffers.items():
            if tensor is None:
                continue
            full_name = f"{module_name}.{attribute}" if module_name else attribute
            yield full_name, module, attribute, buffers, tensor


def _expected_shape(kind: str, module, args, batch: int, seq_len: int) -> tuple[int, ...]:
    if kind == "window_kv_cache":
        return (batch, args.window_size, args.head_dim)
    if kind == "compress_kv_cache":
        ratio = _positive_int(module.compress_ratio, "attention compress_ratio")
        if not module.is_kv_source:
            raise ValueError("compress_kv_cache is registered outside a KV source layer")
        return (batch, seq_len // ratio, args.head_dim)
    if kind == "k_cache":
        ratio = _positive_int(module.compress_ratio, "indexer compress_ratio")
        if not module.owns_k:
            raise ValueError("k_cache is registered outside an index-key owner")
        return (batch, seq_len // ratio, args.index_head_dim)
    if kind in {"kv_state", "score_state"}:
        ratio = _positive_int(module.compress_ratio, "compressor compress_ratio")
        if ratio <= 1:
            raise ValueError(f"{kind} is only valid for compression ratios above one")
        return (batch, ratio, args.head_dim)
    if kind == "freqs_cis":
        if not hasattr(module, "layer_id") or not hasattr(module, "compress_ratio"):
            raise ValueError("freqs_cis is registered outside an attention layer")
        return (seq_len, args.rope_head_dim // 2)
    if kind.startswith("engram_hash_"):
        if module.__class__.__name__ != "NgramHashState" or not hasattr(module, "layout"):
            raise ValueError(f"{kind} is registered outside the pinned NgramHashState")
        layout = module.layout
        layer_count = len(layout.layer_ids)
        if layer_count <= 0 or layout.max_ngram_size < 2 or layout.n_heads <= 0:
            raise ValueError("NgramHashState layout dimensions are invalid")
        attribute = kind.removeprefix("engram_hash_")
        if attribute == "primes":
            return (layer_count, layout.max_ngram_size - 1, layout.n_heads)
        if attribute == "offsets":
            return (layer_count, (layout.max_ngram_size - 1) * layout.n_heads)
        if attribute == "multipliers":
            return (layer_count, layout.max_ngram_size)
        if attribute == "token_map":
            return (_positive_int(getattr(args, "vocab_size", None), "vocab_size"),)
        if attribute == "cache":
            return (batch, seq_len)
        raise ValueError(f"unrecognized NgramHashState buffer: {attribute}")
    raise AssertionError(f"unknown session buffer kind: {kind}")


def _frequency_workspace_bytes(shape: tuple[int, ...]) -> int:
    # The official factory holds the float32 outer-product input and its ones_like input
    # while torch.polar creates complex64 output. Include int64 positions and 1 MiB slack.
    elements = math.prod(shape)
    return 16 * elements + 8 * shape[0] + 1024 * 1024


def plan_session_state(model: object, args: object) -> SessionStatePlan:
    """Size every registered 1017 session buffer without materializing its meta storage."""
    batch = _positive_int(getattr(args, "max_batch_size", None), "max_batch_size")
    seq_len = _positive_int(getattr(args, "max_seq_len", None), "max_seq_len")
    for name in (
        "window_size",
        "head_dim",
        "index_head_dim",
        "rope_head_dim",
        "rope_theta",
        "compress_rope_theta",
        "rope_factor",
        "beta_fast",
        "beta_slow",
        "original_seq_len",
    ):
        if not hasattr(args, name):
            raise ValueError(f"model args lack required session setting: {name}")
    if args.rope_head_dim <= 0 or args.rope_head_dim % 2:
        raise ValueError("rope_head_dim must be a positive even integer")

    groups: dict[tuple[object, ...], dict] = {}
    location_keys: set[tuple[int, str]] = set()
    recognized = {
        "window_kv_cache",
        "compress_kv_cache",
        "k_cache",
        "kv_state",
        "score_state",
        "freqs_cis",
    }
    engram_hash_buffers = {"primes", "offsets", "multipliers", "token_map", "cache"}
    for name, module, attribute, buffers, tensor in _walk_registered_buffers(model):
        if attribute in recognized:
            kind = attribute
        elif module.__class__.__name__ == "NgramHashState" and attribute in engram_hash_buffers:
            kind = f"engram_hash_{attribute}"
        else:
            continue
        if not getattr(tensor, "is_meta", False):
            raise ValueError(f"session buffer is already materialized: {name}")
        shape = tuple(tensor.shape)
        expected = _expected_shape(kind, module, args, batch, seq_len)
        if shape != expected:
            raise ValueError(f"unexpected {kind} shape at {name}: {shape} != {expected}")
        if kind.startswith("engram_hash_") and _dtype_name(tensor.dtype) != "int64":
            raise ValueError(f"NgramHashState buffer must be int64 at {name}")
        if kind == "freqs_cis" and _dtype_name(tensor.dtype) != "complex64":
            raise ValueError(f"official freqs_cis must be complex64 at {name}")
        if kind == "freqs_cis":
            key = _frequency_key(args, int(module.compress_ratio))
            group_key = (kind, key)
        else:
            group_key = (kind, id(tensor))
        group = groups.setdefault(
            group_key,
            {
                "kind": kind,
                "shape": shape,
                "dtype": tensor.dtype,
                "nbytes": _nbytes(tensor),
                "frequency_key": key if kind == "freqs_cis" else None,
                "names": [],
                "locations": [],
                "runtime_module": module if kind.startswith("engram_hash_") else None,
                "runtime_attribute": attribute if kind.startswith("engram_hash_") else None,
            },
        )
        if group["shape"] != shape or group["dtype"] != tensor.dtype:
            raise ValueError(f"inconsistent aliases for session buffer: {name}")
        if kind.startswith("engram_hash_") and group["runtime_module"] is not module:
            raise ValueError(f"NgramHashState aliases do not share one module at {name}")
        location_key = (id(buffers), attribute)
        if location_key not in location_keys:
            group["names"].append(name)
            group["locations"].append((buffers, attribute, tensor))
            location_keys.add(location_key)

    if not groups:
        raise ValueError("model contains no registered DeepSeek session buffers")
    for module_name, module in getattr(model, "named_modules")(remove_duplicate=False):
        if hasattr(module, "window_size") and hasattr(module, "layer_id"):
            for required in ("window_kv_cache", "freqs_cis"):
                if required not in getattr(module, "_buffers", {}):
                    full_name = f"{module_name}.{required}" if module_name else required
                    raise ValueError(f"attention session buffer is missing: {full_name}")
            if getattr(module, "is_kv_source", False):
                if "compress_kv_cache" not in module._buffers:
                    full_name = f"{module_name}.compress_kv_cache" if module_name else "compress_kv_cache"
                    raise ValueError(f"attention session buffer is missing: {full_name}")
        if hasattr(module, "owns_k") and module.owns_k and "k_cache" not in module._buffers:
            full_name = f"{module_name}.k_cache" if module_name else "k_cache"
            raise ValueError(f"indexer session buffer is missing: {full_name}")
        if module.__class__.__name__ == "Compressor" and module.compress_ratio > 1:
            for required in ("kv_state", "score_state"):
                if required not in module._buffers:
                    full_name = f"{module_name}.{required}" if module_name else required
                    raise ValueError(f"compressor session buffer is missing: {full_name}")

    allocations = tuple(
        SessionBufferAllocation(
            names=tuple(sorted(group["names"])),
            kind=group["kind"],
            shape=group["shape"],
            dtype=group["dtype"],
            nbytes=group["nbytes"],
            frequency_key=group["frequency_key"],
            locations=tuple(group["locations"]),
            runtime_module=group["runtime_module"],
            runtime_attribute=group["runtime_attribute"],
        )
        for group in sorted(
            groups.values(),
            key=lambda value: (value["kind"] == "freqs_cis", value["kind"], value["names"]),
        )
    )
    totals = {
        "window_kv_cache": 0,
        "compress_kv_cache": 0,
        "k_cache": 0,
        "kv_state": 0,
        "score_state": 0,
        "freqs_cis": 0,
        "engram_hash": 0,
    }
    for allocation in allocations:
        bucket = "engram_hash" if allocation.kind.startswith("engram_hash_") else allocation.kind
        totals[bucket] += allocation.nbytes
    workspace = max(
        (
            _frequency_workspace_bytes(allocation.shape)
            for allocation in allocations
            if allocation.kind == "freqs_cis"
        ),
        default=0,
    )
    resident = sum(allocation.nbytes for allocation in allocations)
    estimate = SessionStateMemoryEstimate(
        resident_device_bytes=resident,
        device_peak_bytes=resident + workspace,
        physical_peak_bytes=resident + workspace,
        host_transient_bytes=0,
        buffer_count=len(allocations),
        window_kv_bytes=totals["window_kv_cache"],
        compressed_kv_bytes=totals["compress_kv_cache"],
        index_cache_bytes=totals["k_cache"],
        compressor_tail_bytes=totals["kv_state"] + totals["score_state"],
        rotary_bytes=totals["freqs_cis"],
        rotary_workspace_bytes=workspace,
        engram_hash_bytes=totals["engram_hash"],
    )
    return SessionStatePlan(allocations, estimate, batch, seq_len)


def _call_sync_result(callback: Callable[[], object], name: str):
    result = callback()
    if inspect.isawaitable(result):
        close = getattr(result, "close", None)
        if callable(close):
            close()
        raise TypeError(f"{name} must complete synchronously")
    return result


def _call_sync(callback: Callable[[], object], name: str) -> None:
    result = _call_sync_result(callback, name)
    if result is not None:
        raise TypeError(f"{name} must return None")


def _clear_frequency_cache(factory) -> None:
    clear = getattr(factory, "cache_clear", None)
    if callable(clear):
        clear()


def _acquire_model_lifecycle_lock(model: object) -> threading.Lock:
    with _MODEL_LIFECYCLE_LOCKS_LOCK:
        try:
            lock = _MODEL_LIFECYCLE_LOCKS.get(model)
            if lock is None:
                lock = threading.Lock()
                _MODEL_LIFECYCLE_LOCKS[model] = lock
        except TypeError as exc:
            raise TypeError("model must support weak references and identity hashing") from exc
    if not lock.acquire(blocking=False):
        raise RuntimeError("DeepSeek model already has a session load or active session")
    return lock


def _make_frequency(factory, key, device, torch):
    with _FREQUENCY_FACTORY_LOCK:
        _clear_frequency_cache(factory)
        try:
            with torch.device(device):
                result = factory(*key)
        finally:
            _clear_frequency_cache(factory)
    if not isinstance(result, torch.Tensor):
        raise TypeError("official rotary factory did not return a torch.Tensor")
    if result.is_meta or result.device != torch.device(device):
        raise RuntimeError("official rotary factory did not materialize on the admitted device")
    if result.dtype != torch.complex64:
        raise RuntimeError(f"official rotary factory returned unexpected dtype: {result.dtype}")
    return result


class SessionState:
    """Materialized buffers that hold their admission lease until fenced teardown."""

    def __init__(self, model, plan, reservation, restore_locations, loaded, fence, reset, release, lifecycle_lock):
        self.model = model
        self.plan = plan
        self._reservation = reservation
        self._restore_locations = restore_locations
        self._loaded = loaded
        self._fence = fence
        self._reset_runtime_references = reset
        self._release_allocator = release
        self._lifecycle_lock = lifecycle_lock
        self._closed = False
        self._quarantined = False
        self._close_lock = threading.Lock()
        with _LIVE_STATES_LOCK:
            _LIVE_STATES[id(self)] = self

    def __enter__(self):
        if self._closed or self._quarantined:
            raise RuntimeError("DeepSeek session state is not active")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def reset_for_new_request(self):
        """Clear mutable sequence state before reusing this session for a new request."""
        with self._close_lock:
            if self._closed or self._quarantined:
                raise RuntimeError("DeepSeek session state is not active")
            try:
                _call_sync(self._fence, "pre-request reset GPU fence")
                reset_kinds = {
                    "window_kv_cache",
                    "compress_kv_cache",
                    "k_cache",
                    "kv_state",
                    "score_state",
                    "engram_hash_cache",
                }
                for allocation in self.plan.allocations:
                    if allocation.kind not in reset_kinds:
                        continue
                    values = [
                        container.get(attribute)
                        for container, attribute, _original in allocation.locations
                    ]
                    if not values or any(
                        value is None or getattr(value, "is_meta", True)
                        for value in values
                    ):
                        raise RuntimeError(
                            f"request reset found a missing session buffer: {allocation.names}"
                        )
                    if any(value is not values[0] for value in values[1:]):
                        raise RuntimeError(
                            f"request reset found split aliases: {allocation.names}"
                        )
                    if allocation.kind == "score_state":
                        values[0].fill_(float("-inf"))
                    else:
                        values[0].zero_()
                _call_sync(self._fence, "post-request reset GPU fence")
            except BaseException as exc:
                self._quarantined = True
                with _LIVE_STATES_LOCK:
                    _QUARANTINED_STATES.append((self._reservation, self, exc))
                raise

    def close(self):
        """Restore meta buffers only after runtime references and device work are drained."""
        with self._close_lock:
            if self._closed:
                return
            if self._quarantined:
                raise RuntimeError("DeepSeek session state is quarantined; worker restart is required")
            try:
                _call_sync(self._fence, "GPU fence")
                _call_sync(self._reset_runtime_references, "runtime reference reset")
                for container, attribute, original in reversed(self._restore_locations):
                    container[attribute] = original
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
                    _QUARANTINED_STATES.append((self._reservation, self, exc))
                raise
            self._restore_locations.clear()
            self._loaded.clear()
            self._closed = True
            with _LIVE_STATES_LOCK:
                _LIVE_STATES.pop(id(self), None)
            self._lifecycle_lock.release()


def _quarantine_failed_load(reservation, loaded, error, *, model, lifecycle_lock, inflight=()):
    retained = {
        "model": model,
        "lifecycle_lock": lifecycle_lock,
        "loaded": loaded,
        "inflight": tuple(inflight),
    }
    with _LIVE_STATES_LOCK:
        _QUARANTINED_STATES.append((reservation, retained, error))


def load_session_state(
    model: object,
    args: object,
    *,
    device,
    frequency_factory: Callable[..., object],
    engram_hash_factory: Callable[[object, object, object], object] | None = None,
    admission: Callable[[SessionStateMemoryEstimate], ContextManager[object]],
    fence: Callable[[], object],
    reset_runtime_references: Callable[[], object],
    release_allocator: Callable[[], object],
) -> SessionState:
    """Allocate official 1017 session buffers after admission and retain the lease until close.

    ``frequency_factory`` must be the pinned implementation's ``precompute_freqs_cis``.
    ``reset_runtime_references`` must clear the model module's shared-attention slots and any
    ordinary Python attributes that alias session buffers, such as Indexer.freqs_cis.
    The caller must stop all model work before closing the returned state.
    """
    if not callable(frequency_factory) or not callable(admission):
        raise TypeError("frequency_factory and admission must be callable")
    if engram_hash_factory is not None and not callable(engram_hash_factory):
        raise TypeError("engram_hash_factory must be callable")
    if not all(callable(callback) for callback in (fence, reset_runtime_references, release_allocator)):
        raise TypeError("fence, reset_runtime_references, and release_allocator must be callable")
    lifecycle_lock = _acquire_model_lifecycle_lock(model)
    try:
        plan = plan_session_state(model, args)
    except BaseException:
        lifecycle_lock.release()
        raise
    if (
        any(allocation.kind.startswith("engram_hash_") for allocation in plan.allocations)
        and engram_hash_factory is None
    ):
        lifecycle_lock.release()
        raise ValueError("NgramHashState buffers require the pinned Engram hash factory")
    try:
        import torch
    except ImportError as exc:
        lifecycle_lock.release()
        raise RuntimeError("PyTorch is required for DeepSeek session state") from exc
    try:
        target_device = torch.device(device)
        if target_device.type == "meta":
            raise ValueError("DeepSeek session destination must have allocated storage")
        reservation = admission(plan.estimate)
        if not callable(getattr(reservation, "__enter__", None)) or not callable(
            getattr(reservation, "__exit__", None)
        ):
            raise TypeError("admission must return a synchronous context manager")
        _call_sync_result(reservation.__enter__, "admission context __enter__")
    except BaseException:
        lifecycle_lock.release()
        raise

    assigned = []
    loaded = {}
    materialized_engram_states = {}
    target = None
    replacement = None
    allocation = None
    try:
        _call_sync(reset_runtime_references, "runtime reference reset")
        for allocation in plan.allocations:
            if allocation.kind == "freqs_cis":
                replacement = _make_frequency(
                    frequency_factory,
                    allocation.frequency_key,
                    target_device,
                    torch,
                )
                if tuple(replacement.shape) != allocation.shape:
                    raise RuntimeError(
                        f"official rotary shape mismatch for {allocation.names[0]}: "
                        f"{tuple(replacement.shape)} != {allocation.shape}"
                    )
            elif allocation.kind.startswith("engram_hash_"):
                module = allocation.runtime_module
                attribute = allocation.runtime_attribute
                if module is None or attribute is None or engram_hash_factory is None:
                    raise RuntimeError("NgramHashState allocation has no pinned runtime factory")
                module_key = id(module)
                candidate = materialized_engram_states.get(module_key)
                if candidate is None:
                    candidate = _call_sync_result(
                        lambda: engram_hash_factory(module, args, target_device),
                        "Engram hash state factory",
                    )
                    candidate_buffers = getattr(candidate, "_buffers", None)
                    if not isinstance(candidate_buffers, dict):
                        raise TypeError("Engram hash state factory must return a module with registered buffers")
                    materialized_engram_states[module_key] = candidate
                value = candidate._buffers.get(attribute)
                if not isinstance(value, torch.Tensor):
                    raise RuntimeError(f"Engram hash factory omitted buffer {attribute}")
                if (
                    value.is_meta
                    or value.device != target_device
                    or tuple(value.shape) != allocation.shape
                    or value.dtype != allocation.dtype
                ):
                    raise RuntimeError(f"Engram hash factory returned incompatible buffer {attribute}")
                replacement = value
            elif allocation.kind == "score_state":
                replacement = torch.full(
                    allocation.shape,
                    -torch.inf,
                    dtype=allocation.dtype,
                    device=target_device,
                )
            else:
                replacement = torch.zeros(
                    allocation.shape,
                    dtype=allocation.dtype,
                    device=target_device,
                )
            loaded[allocation.names[0]] = replacement
            for container, attribute, original in allocation.locations:
                assigned.append((container, attribute, original))
                container[attribute] = replacement
        _call_sync(fence, "GPU allocation fence")
        for allocation in plan.allocations:
            for container, attribute, _original in allocation.locations:
                value = container[attribute]
                if value.is_meta or value.device != target_device:
                    raise RuntimeError(f"session buffer was not materialized: {allocation.names[0]}")
        materialized_engram_states.clear()
        return SessionState(
            model,
            plan,
            reservation,
            assigned,
            loaded,
            fence,
            reset_runtime_references,
            release_allocator,
            lifecycle_lock,
        )
    except BaseException as load_error:
        cleanup_error = None
        try:
            _call_sync(fence, "failed-load GPU fence")
            _call_sync(reset_runtime_references, "failed-load runtime reference reset")
            for container, attribute, original in reversed(assigned):
                container[attribute] = original
            materialized_engram_states.clear()
            replacement = None
            target = None
            allocation = None
            loaded.clear()
            gc.collect()
            _call_sync(release_allocator, "failed-load release_allocator")
            _call_sync(fence, "failed-load post-release GPU fence")
        except BaseException as exc:
            cleanup_error = exc
        if cleanup_error is not None:
            _quarantine_failed_load(
                reservation,
                loaded,
                cleanup_error,
                model=model,
                lifecycle_lock=lifecycle_lock,
                inflight=(load_error, target, replacement, allocation),
            )
            if hasattr(load_error, "add_note"):
                load_error.add_note(
                    f"session-state cleanup also failed: {cleanup_error!r}; reservation retained"
                )
            raise load_error from cleanup_error
        try:
            suppressed = _call_sync_result(
                lambda: reservation.__exit__(
                    type(load_error), load_error, load_error.__traceback__
                ),
                "admission context __exit__",
            )
        except BaseException as exc:
            _quarantine_failed_load(
                reservation,
                loaded,
                exc,
                model=model,
                lifecycle_lock=lifecycle_lock,
                inflight=(load_error,),
            )
            if hasattr(load_error, "add_note"):
                load_error.add_note(f"session-state reservation release failed: {exc!r}")
            raise load_error from exc
        lifecycle_lock.release()
        if suppressed:
            raise RuntimeError("admission context managers may not suppress load failures") from load_error
        raise
