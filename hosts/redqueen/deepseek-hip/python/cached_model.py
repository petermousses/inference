"""Install cached routed experts into the pinned DeepSeek transformer graph."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, TypeVar

from expert_moe import execute_cached_moe

_T = TypeVar("_T")
_INSTALLATION_MARKER = "_redqueen_cached_moe_installation"


@dataclass(frozen=True)
class _ForwardPatch:
    moe: Any
    original: Callable
    installed: Callable


class CachedMoEInstallation:
    """Reversible per-instance MoE.forward patches for one transformer."""

    def __init__(
        self,
        patches: list[_ForwardPatch],
        cache: object,
        fence: Callable[[], None] | None,
    ) -> None:
        self._patches = patches
        self._cache = cache
        self._fence = fence
        self._execution_lock = asyncio.Lock()
        self._closed = False

    def __enter__(self) -> CachedMoEInstallation:
        if self._closed:
            raise RuntimeError("cached MoE installation is already closed")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        if self._execution_lock.locked():
            raise RuntimeError("cannot close cached MoE installation during model execution")
        for patch in self._patches:
            if getattr(patch.moe, _INSTALLATION_MARKER, None) is not self:
                raise RuntimeError("cached MoE installation ownership changed")
            if patch.moe.forward is not patch.installed:
                raise RuntimeError("MoE.forward changed while the cache was installed")
        for patch in self._patches:
            patch.moe.forward = patch.original
            delattr(patch.moe, _INSTALLATION_MARKER)
        self._closed = True

    async def run_reserved_sync(
        self,
        operation: Callable[[], _T],
        reservation,
        *,
        worker_context: Callable[[], object] | None = None,
    ) -> _T:
        """Serialize model execution and hold its broker lease through the GPU fence.

        The execution lock is acquired before memory admission, so a queued
        request does not reserve memory while another request uses the graph.
        The operation must consume or copy GPU results before returning, so
        the reservation covers all live request state. ``worker_context`` can
        install thread-local device/runtime state and must restore it on exit.
        """
        if self._closed:
            raise RuntimeError("cached MoE installation is already closed")
        if not callable(operation):
            raise TypeError("operation must be callable")
        if not callable(self._fence):
            raise ValueError("cached MoE installation requires a final GPU fence")
        if worker_context is not None and not callable(worker_context):
            raise TypeError("worker_context must be callable")
        if not callable(getattr(self._cache, "quarantine", None)):
            raise TypeError("cache must support quarantine")
        if not callable(getattr(reservation, "__aenter__", None)) or not callable(
            getattr(reservation, "__aexit__", None)
        ):
            raise TypeError("reservation must be an async context manager")

        async with self._execution_lock:
            if self._closed:
                raise RuntimeError("cached MoE installation is already closed")
            async with reservation:
                task = asyncio.create_task(
                    asyncio.to_thread(
                        _run_and_fence,
                        operation,
                        self._cache,
                        self._fence,
                        worker_context,
                    )
                )
                try:
                    return await asyncio.shield(task)
                except asyncio.CancelledError as cancellation:
                    # Do not cancel the worker: it must finish and fence while
                    # both the reservation and execution lock remain held.
                    while not task.done():
                        try:
                            await asyncio.shield(task)
                        except asyncio.CancelledError:
                            continue
                        except BaseException:
                            break
                    try:
                        task.result()
                    except BaseException as worker_error:
                        if hasattr(cancellation, "add_note"):
                            cancellation.add_note(
                                "GPU worker finished with "
                                f"{worker_error!r} while cancellation drained"
                            )
                    raise cancellation


def _blocks(value: object, name: str) -> list[object]:
    if value is None:
        raise TypeError(f"transformer.{name} must be an iterable of blocks")
    try:
        return list(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(f"transformer.{name} must be an iterable of blocks") from error


def _moe_from_block(block: object, name: str) -> object:
    moe = getattr(block, "ffn", None)
    if moe is None:
        raise TypeError(f"{name} block must expose an ffn MoE")
    for attribute in ("layer_id", "n_routed_experts", "gate", "shared_experts", "forward"):
        if not hasattr(moe, attribute):
            raise TypeError(f"{name} MoE must expose {attribute}")
    if not callable(moe.gate) or not callable(moe.shared_experts) or not callable(moe.forward):
        raise TypeError(f"{name} MoE gate, shared experts, and forward must be callable")
    if type(moe.layer_id) is not int or moe.layer_id < 0:
        raise ValueError(f"{name} MoE must declare a nonnegative integer layer id")
    if type(moe.n_routed_experts) is not int or moe.n_routed_experts < 1:
        raise ValueError(f"{name} MoE must declare a positive routed expert count")
    shared_limit = getattr(moe.shared_experts, "swiglu_limit", None)
    if (
        isinstance(shared_limit, bool)
        or not isinstance(shared_limit, (int, float))
        or not math.isfinite(shared_limit)
        or shared_limit < 0
    ):
        raise ValueError(f"{name} shared expert must declare a finite nonnegative swiglu_limit")
    return moe


def install_cached_moe(
    transformer: object,
    cache: object,
    linear: Callable,
    *,
    fence: Callable[[], None] | None = None,
) -> CachedMoEInstallation:
    """Route every backbone and MTP MoE through one lease-aware expert cache.

    Backbone keys keep their layer index under the ``layers`` namespace. MTP
    block indexes are rebased to their checkpoint-local index under ``mtp``.
    The installation is reversible and validates the entire graph before
    changing any module, so a malformed MTP block cannot leave half the model
    patched.
    """
    if not callable(linear):
        raise TypeError("linear must be callable")
    if fence is not None and not callable(fence):
        raise TypeError("fence must be callable")
    if not callable(getattr(cache, "acquire", None)) or not callable(
        getattr(cache, "quarantine", None)
    ):
        raise TypeError("cache must support acquire and quarantine")

    layers = _blocks(getattr(transformer, "layers", None), "layers")
    mtp_blocks = _blocks(getattr(transformer, "mtp", None), "mtp")
    targets: list[tuple[object, int, bool, str]] = []
    seen_moes: set[int] = set()
    for layer_index, block in enumerate(layers):
        name = f"layers[{layer_index}]"
        moe = _moe_from_block(block, name)
        if id(moe) in seen_moes:
            raise ValueError(f"{name} reuses an MoE already assigned to another block")
        seen_moes.add(id(moe))
        if moe.layer_id != layer_index:
            raise ValueError(f"{name} MoE layer id does not match its backbone index")
        targets.append((moe, layer_index, False, name))
    for mtp_index, block in enumerate(mtp_blocks):
        name = f"mtp[{mtp_index}]"
        moe = _moe_from_block(block, name)
        if id(moe) in seen_moes:
            raise ValueError(f"{name} reuses an MoE already assigned to another block")
        seen_moes.add(id(moe))
        if moe.layer_id != len(layers) + mtp_index:
            raise ValueError(f"{name} MoE layer id does not match its transformer index")
        targets.append((moe, mtp_index, True, name))

    for moe, _, _, name in targets:
        if hasattr(moe, _INSTALLATION_MARKER):
            raise RuntimeError(f"{name} already belongs to a cached MoE installation")

    patches: list[_ForwardPatch] = []
    for moe, cache_layer_id, mtp_layer, name in targets:
        original = moe.forward

        def cached_forward(
            activations,
            image_mask=None,
            *,
            _moe=moe,
            _layer_id=cache_layer_id,
            _mtp_layer=mtp_layer,
        ):
            return execute_cached_moe(
                cache,
                activations,
                _moe.gate,
                _moe.shared_experts,
                linear,
                layer_id=_layer_id,
                n_routed_experts=_moe.n_routed_experts,
                swiglu_limit=_moe.shared_experts.swiglu_limit,
                image_mask=image_mask,
                fence=fence,
                mtp_layer=_mtp_layer,
            )

        cached_forward.__name__ = f"cached_{name.replace('[', '_').replace(']', '')}_forward"
        patches.append(_ForwardPatch(moe, original, cached_forward))

    installation = CachedMoEInstallation(patches, cache, fence)
    marked: list[_ForwardPatch] = []
    patched: list[_ForwardPatch] = []
    try:
        for patch in patches:
            setattr(patch.moe, _INSTALLATION_MARKER, installation)
            marked.append(patch)
        for patch in patches:
            patch.moe.forward = patch.installed
            patched.append(patch)
    except BaseException:
        for patch in reversed(patched):
            if patch.moe.forward is patch.installed:
                patch.moe.forward = patch.original
        for patch in reversed(marked):
            if getattr(patch.moe, _INSTALLATION_MARKER, None) is installation:
                delattr(patch.moe, _INSTALLATION_MARKER)
        raise
    return installation


def _run_and_fence(
    operation: Callable[[], _T],
    cache: object,
    fence: Callable[[], None],
    worker_context: Callable[[], object] | None,
) -> _T:
    result: _T | None = None
    operation_error: BaseException | None = None
    try:
        context = worker_context() if worker_context is not None else nullcontext()
        with context:
            result = operation()
    except BaseException as error:
        operation_error = error

    try:
        fence()
    except BaseException as fence_error:
        try:
            cache.quarantine()
        except BaseException as quarantine_error:
            if hasattr(fence_error, "add_note"):
                fence_error.add_note(f"cache quarantine also failed: {quarantine_error!r}")
        if operation_error is not None:
            if hasattr(operation_error, "add_note"):
                operation_error.add_note("the final GPU fence also failed; worker cache was quarantined")
            raise operation_error from fence_error
        raise

    if operation_error is not None:
        raise operation_error
    return result  # type: ignore[return-value]
