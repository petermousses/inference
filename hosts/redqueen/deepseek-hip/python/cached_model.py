"""Install cached routed experts into the pinned DeepSeek transformer graph."""

from __future__ import annotations

import asyncio
import inspect
import math
import os
import threading
from concurrent.futures import Future as ConcurrentFuture
from collections.abc import Awaitable, Callable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, TypeVar

from expert_cache import ExpertCacheResize
from expert_moe import execute_cached_moe

_T = TypeVar("_T")
_INSTALLATION_MARKER = "_redqueen_cached_moe_installation"
_INSTALLATION_LOCK = threading.Lock()


class _UnsafeCallbackResult(TypeError):
    """An invalid async callback result could not be proven stopped."""


class _UndrainableCallbackResult(_UnsafeCallbackResult):
    """A callback may have left work active and needs worker teardown."""


_CALLBACK_DRAIN_TIMEOUT_SECONDS = 30
_WORKER_TEARDOWN_TIMEOUT_SECONDS = 30
_WORKER_TEARDOWN_FAILED_EXIT_CODE = 70


@dataclass(frozen=True)
class _WorkerResult:
    value: Any = None
    error: BaseException | None = None


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
        self._quarantine_state = threading.Event()

    @property
    def _quarantined(self) -> bool:
        return self._quarantine_state.is_set()

    def _mark_quarantined(self) -> None:
        self._quarantine_state.set()

    def __enter__(self) -> CachedMoEInstallation:
        if self._closed:
            raise RuntimeError("cached MoE installation is already closed")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if not self._quarantined:
            self.close()

    def close(self) -> None:
        with _INSTALLATION_LOCK:
            if self._closed:
                return
            if self._quarantined:
                raise RuntimeError(
                    "cannot close a cached MoE installation with an undrained callback"
                )
            if self._execution_lock.locked():
                raise RuntimeError("cannot close cached MoE installation during model work")
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
        worker_teardown: Callable[[BaseException], Awaitable[bool]] | None = None,
    ) -> _T:
        """Serialize model execution and hold its broker lease through the GPU fence.

        The execution lock is acquired before memory admission, so a queued
        request does not reserve memory while another request uses the graph.
        The operation must consume or copy GPU results before returning, so
        the reservation covers all live request state. It must not leave
        detached GPU or background work; any Future or awaitable it returns
        must represent all work and cleanup, and its callbacks must not start
        untracked work. ``worker_context`` can install thread-local device/runtime
        state and must restore it on exit. Its context-manager methods must be
        synchronous and return non-awaitable values; ``__exit__`` may not
        suppress model errors. ``worker_teardown`` must stop the isolated worker
        and confirm its GPU/background work and allocations are gone. It runs
        while the request reservation remains held. If omitted or unsuccessful,
        this process exits because no safe in-process recovery is possible.
        """
        if self._closed:
            raise RuntimeError("cached MoE installation is already closed")
        if self._quarantined:
            raise RuntimeError("cached MoE installation is quarantined; worker restart is required")
        if not callable(operation):
            raise TypeError("operation must be callable")
        if not callable(self._fence):
            raise ValueError("cached MoE installation requires a final GPU fence")
        if worker_context is not None and not callable(worker_context):
            raise TypeError("worker_context must be callable")
        _validate_sync_callable(operation, "operation")
        _validate_sync_callable(self._fence, "GPU fence")
        if worker_context is not None:
            _validate_sync_callable(worker_context, "worker_context")
        _validate_worker_teardown(worker_teardown)
        if not callable(getattr(self._cache, "quarantine", None)):
            raise TypeError("cache must support quarantine")
        if not callable(getattr(reservation, "__aenter__", None)) or not callable(
            getattr(reservation, "__aexit__", None)
        ):
            raise TypeError("reservation must be an async context manager")
        async with self._execution_lock:
            if self._closed:
                raise RuntimeError("cached MoE installation is already closed")
            if self._quarantined:
                raise RuntimeError("cached MoE installation is quarantined; worker restart is required")
            async with reservation:
                owner_loop = asyncio.get_running_loop()

                def teardown_worker(error: BaseException) -> None:
                    _teardown_worker_on_owner_loop(
                        owner_loop,
                        worker_teardown,
                        error,
                    )

                task = asyncio.create_task(
                    asyncio.to_thread(
                        _capture_sync_call,
                        _run_and_fence,
                        operation,
                        self._cache,
                        self._fence,
                        worker_context,
                        self._mark_quarantined,
                        teardown_worker,
                    )
                )
                try:
                    result = await asyncio.shield(task)
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
                        result = task.result()
                    except BaseException as worker_error:
                        if _cache_is_quarantined(self._cache):
                            self._mark_quarantined()
                        if hasattr(cancellation, "add_note"):
                            cancellation.add_note(
                                "GPU worker finished with "
                                f"{worker_error!r} while cancellation drained"
                            )
                    else:
                        if result.error is not None:
                            if _cache_is_quarantined(self._cache):
                                self._mark_quarantined()
                            if hasattr(cancellation, "add_note"):
                                cancellation.add_note(
                                    "GPU worker finished with "
                                    f"{result.error!r} while cancellation drained"
                                )
                    raise cancellation
                except BaseException:
                    if _cache_is_quarantined(self._cache):
                        self._mark_quarantined()
                    raise
                if result.error is not None:
                    if _cache_is_quarantined(self._cache):
                        self._mark_quarantined()
                    raise result.error
                return result.value  # type: ignore[return-value]

    async def resize_cache(
        self,
        capacity_bytes: int,
        *,
        reservation,
        release_allocator: Callable[[], None],
        worker_context: Callable[[], object] | None = None,
        worker_teardown: Callable[[BaseException], Awaitable[bool]] | None = None,
    ) -> ExpertCacheResize:
        """Shrink or grow the expert cache at a fenced, idle model boundary.

        The installation lock excludes model execution while the cache drops
        tensor references and the worker returns unused allocator blocks. The
        required reservation is entered after the installation lock and held
        across resize, allocator release, the confirming fence, and teardown.
        It must be the host broker's global execution/admission lease; this
        installation lock protects only this model graph. The returned byte
        count is logical cache payload only, so the broker must confirm actual
        host/GTT availability before admitting another worker. Allocator release
        runs only while backing release is pending, including after a shrink,
        eviction, or failed release callback. Fence and allocator callbacks
        must be synchronous and return ``None``. They
        must not detach GPU/background work; any Future or awaitable they return
        must represent all work and cleanup, with no untracked callback work.
        ``worker_context`` methods must be synchronous and return
        non-awaitable values; ``__exit__`` may not suppress resize errors.
        ``worker_teardown`` follows the same contract as
        ``run_reserved_sync`` and runs while the host broker still owns its
        resize/execution reservation.
        """
        if isinstance(capacity_bytes, bool) or type(capacity_bytes) is not int or capacity_bytes < 0:
            raise ValueError("cache capacity must be a nonnegative integer")
        if self._quarantined:
            raise RuntimeError("cached MoE installation is quarantined; worker restart is required")
        if not callable(self._fence):
            raise ValueError("cached MoE installation requires a final GPU fence")
        _validate_sync_callable(self._fence, "GPU fence")
        _validate_sync_callable(release_allocator, "release_allocator")
        if worker_context is not None:
            _validate_sync_callable(worker_context, "worker_context")
        _validate_worker_teardown(worker_teardown)
        if not callable(getattr(reservation, "__aenter__", None)) or not callable(
            getattr(reservation, "__aexit__", None)
        ):
            raise TypeError("reservation must be an async context manager")
        if not callable(getattr(self._cache, "resize_when_idle", None)):
            raise TypeError("cache must support resize_when_idle")
        if not callable(getattr(self._cache, "quarantine", None)):
            raise TypeError("cache must support quarantine")
        async with self._execution_lock, reservation:
            if self._closed:
                raise RuntimeError("cached MoE installation is already closed")
            if self._quarantined:
                raise RuntimeError("cached MoE installation is quarantined; worker restart is required")
            owner_loop = asyncio.get_running_loop()

            def teardown_worker(error: BaseException) -> None:
                _teardown_worker_on_owner_loop(
                    owner_loop,
                    worker_teardown,
                    error,
                )

            task = asyncio.create_task(
                asyncio.to_thread(
                    _capture_sync_call,
                    _resize_cache_sync,
                    self._cache,
                    capacity_bytes,
                    self._fence,
                    release_allocator,
                    worker_context,
                    self._mark_quarantined,
                    teardown_worker,
                )
            )
            try:
                result = await asyncio.shield(task)
            except asyncio.CancelledError as cancellation:
                # A cancelled control request must not release the execution
                # lock while the worker is dropping live tensor references.
                while not task.done():
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        continue
                    except BaseException:
                        break
                try:
                    result = task.result()
                except BaseException as worker_error:
                    if _cache_is_quarantined(self._cache):
                        self._mark_quarantined()
                    if hasattr(cancellation, "add_note"):
                        cancellation.add_note(
                            "cache resize worker finished with "
                            f"{worker_error!r} while cancellation drained"
                        )
                else:
                    if result.error is not None:
                        if _cache_is_quarantined(self._cache):
                            self._mark_quarantined()
                        if hasattr(cancellation, "add_note"):
                            cancellation.add_note(
                                "cache resize worker finished with "
                                f"{result.error!r} while cancellation drained"
                            )
                raise cancellation
            except BaseException:
                if _cache_is_quarantined(self._cache):
                    self._mark_quarantined()
                raise
            if result.error is not None:
                if _cache_is_quarantined(self._cache):
                    self._mark_quarantined()
                raise result.error
            return result.value  # type: ignore[return-value]


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
    after_moe: Callable[[], None] | None = None,
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
    if after_moe is not None and not callable(after_moe):
        raise TypeError("after_moe must be callable")
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

    with _INSTALLATION_LOCK:
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
                result = execute_cached_moe(
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
                    after_moe=after_moe,
                    mtp_layer=_mtp_layer,
                )
                return result

            cached_forward.__name__ = (
                f"cached_{name.replace('[', '_').replace(']', '')}_forward"
            )
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
    mark_unavailable: Callable[[], None],
    teardown_worker: Callable[[BaseException], None],
) -> _T:
    result: _T | None = None
    operation_error: BaseException | None = None
    try:
        context = (
            _call_sync(
                worker_context,
                "worker_context",
                allow_result=True,
            )
            if worker_context is not None
            else nullcontext()
        )
        result = _run_in_sync_context(
            context,
            lambda: _call_sync(
                operation,
                "operation",
                allow_result=True,
            ),
            "worker_context",
        )
    except BaseException as error:
        operation_error = error

    if isinstance(operation_error, _UnsafeCallbackResult):
        _quarantine_worker_state(cache, operation_error, mark_unavailable)

    try:
        _call_sync(fence, "GPU fence")
    except BaseException as fence_error:
        _quarantine_worker_state(cache, fence_error, mark_unavailable)
        if isinstance(operation_error, _UndrainableCallbackResult):
            teardown_worker(operation_error)
        else:
            teardown_worker(fence_error)
        if operation_error is not None:
            if hasattr(operation_error, "add_note"):
                operation_error.add_note(
                    "the final GPU fence also failed; installation was marked unavailable"
                )
            raise operation_error from fence_error
        raise

    if operation_error is not None:
        if isinstance(operation_error, _UndrainableCallbackResult):
            teardown_worker(operation_error)
        raise operation_error
    return result  # type: ignore[return-value]


def _resize_cache_sync(
    cache: object,
    capacity_bytes: int,
    fence: Callable[[], None],
    release_allocator: Callable[[], None],
    worker_context: Callable[[], object] | None,
    mark_unavailable: Callable[[], None],
    teardown_worker: Callable[[BaseException], None],
) -> ExpertCacheResize:
    try:
        context = (
            _call_sync(
                worker_context,
                "worker_context",
                allow_result=True,
            )
            if worker_context is not None
            else nullcontext()
        )
    except _UnsafeCallbackResult as callback_error:
        _quarantine_worker_state(cache, callback_error, mark_unavailable)
        if isinstance(callback_error, _UndrainableCallbackResult):
            teardown_worker(callback_error)
        raise

    def resize_inside_context() -> ExpertCacheResize:
        try:
            _call_sync(fence, "GPU fence")
        except BaseException as fence_error:
            _quarantine_worker_state(cache, fence_error, mark_unavailable)
            teardown_worker(fence_error)
            raise

        def release_and_fence() -> None:
            release_error: BaseException | None = None
            try:
                _call_sync(
                    release_allocator,
                    "release_allocator",
                )
            except BaseException as error:
                release_error = error
            try:
                _call_sync(fence, "GPU fence")
            except BaseException as fence_error:
                _quarantine_worker_state(cache, fence_error, mark_unavailable)
                teardown_worker(fence_error)
                if release_error is not None:
                    if hasattr(release_error, "add_note"):
                        release_error.add_note(
                            f"post-release GPU fence also failed: {fence_error!r}"
                        )
                    if isinstance(fence_error, _UndrainableCallbackResult) and not isinstance(
                        release_error,
                        _UndrainableCallbackResult,
                    ):
                        raise fence_error from release_error
                    raise release_error from fence_error
                raise
            if release_error is not None:
                raise release_error

        try:
            resized = cache.resize_when_idle(
                capacity_bytes,
                after_drop=release_and_fence,
            )
        except _UnsafeCallbackResult as callback_error:
            _quarantine_worker_state(cache, callback_error, mark_unavailable)
            raise
        if not isinstance(resized, ExpertCacheResize):
            raise TypeError("cache resize_when_idle must return ExpertCacheResize")

        stats = cache.stats
        if not isinstance(stats, type(resized.stats)) or not stats.within_capacity:
            raise RuntimeError("cache did not reach the requested capacity")
        if stats.pending_backing_release:
            raise RuntimeError("cache allocator backing release is still pending")
        if stats.quarantined:
            raise RuntimeError("cache was quarantined while resizing")
        if stats.leased_bytes or stats.loading_bytes or stats.releasing_bytes or stats.pending_loads:
            raise RuntimeError("cache resize completed before all expert activity stopped")
        return ExpertCacheResize(
            stats=stats,
            evicted_logical_bytes=resized.evicted_logical_bytes,
        )

    try:
        return _run_in_sync_context(
            context,
            resize_inside_context,
            "worker_context",
        )
    except _UnsafeCallbackResult as callback_error:
        _quarantine_worker_state(cache, callback_error, mark_unavailable)
        if isinstance(callback_error, _UndrainableCallbackResult):
            teardown_worker(callback_error)
        raise


def _run_in_sync_context(
    context: object,
    callback: Callable[[], _T],
    name: str,
) -> _T:
    context_type = type(context)
    enter = getattr(context_type, "__enter__", None)
    exit_context = getattr(context_type, "__exit__", None)
    if not callable(enter) or not callable(exit_context):
        raise _worker_context_failure(
            name,
            "did not return a synchronous context manager",
        )
    try:
        _validate_sync_callable(enter, f"{name}.__enter__")
        _validate_sync_callable(exit_context, f"{name}.__exit__")
    except TypeError as error:
        raise _worker_context_failure(
            name,
            "context-manager methods must be synchronous",
            error,
        ) from error

    enter_returned = False

    def enter_context() -> object:
        nonlocal enter_returned
        value = enter(context)
        enter_returned = True
        return value

    try:
        _call_sync(
            enter_context,
            f"{name}.__enter__",
            allow_result=True,
        )
    except BaseException as enter_error:
        if not enter_returned:
            if isinstance(enter_error, _UndrainableCallbackResult):
                raise
            context_error = _worker_context_failure(
                name,
                "__enter__ failed",
                enter_error,
            )
            raise context_error from enter_error
        try:
            _call_sync(
                lambda: exit_context(
                    context,
                    type(enter_error),
                    enter_error,
                    enter_error.__traceback__,
                ),
                f"{name}.__exit__",
                allow_result=True,
            )
        except BaseException as exit_error:
            if isinstance(enter_error, _UndrainableCallbackResult):
                if hasattr(enter_error, "add_note"):
                    enter_error.add_note(
                        f"{name}.__exit__ also failed after __enter__ returned: "
                        f"{exit_error!r}"
                    )
                raise enter_error from exit_error
            if isinstance(exit_error, _UndrainableCallbackResult):
                if hasattr(exit_error, "add_note"):
                    exit_error.add_note(
                        f"{name}.__enter__ also failed: {enter_error!r}"
                    )
                raise exit_error from enter_error
            context_error = _worker_context_failure(
                name,
                "context cleanup failed",
                enter_error,
                exit_error,
            )
            raise context_error from exit_error
        if isinstance(enter_error, _UndrainableCallbackResult):
            raise
        context_error = _worker_context_failure(
            name,
            "__enter__ failed",
            enter_error,
        )
        raise context_error from enter_error

    try:
        result = callback()
    except BaseException as operation_error:
        try:
            suppress = _call_sync(
                lambda: exit_context(
                    context,
                    type(operation_error),
                    operation_error,
                    operation_error.__traceback__,
                ),
                f"{name}.__exit__",
                allow_result=True,
            )
        except BaseException as exit_error:
            if isinstance(operation_error, _UndrainableCallbackResult):
                if hasattr(operation_error, "add_note"):
                    operation_error.add_note(
                        f"{name}.__exit__ also failed: {exit_error!r}"
                    )
                raise operation_error from exit_error
            if isinstance(exit_error, _UndrainableCallbackResult):
                if hasattr(exit_error, "add_note"):
                    exit_error.add_note(
                        f"{name} operation also failed: {operation_error!r}"
                    )
                raise exit_error from operation_error
            context_error = _worker_context_failure(
                name,
                "context cleanup failed",
                operation_error,
                exit_error,
            )
            raise context_error from exit_error
        if suppress:
            if hasattr(operation_error, "add_note"):
                operation_error.add_note(
                    f"{name}.__exit__ cannot suppress model or resize errors"
                )
        raise
    try:
        _call_sync(
            lambda: exit_context(context, None, None, None),
            f"{name}.__exit__",
            allow_result=True,
        )
    except _UndrainableCallbackResult:
        raise
    except BaseException as exit_error:
        raise _worker_context_failure(
            name,
            "context cleanup failed",
            exit_error,
        ) from exit_error
    return result


def _worker_context_failure(
    name: str,
    reason: str,
    primary_error: BaseException | None = None,
    secondary_error: BaseException | None = None,
) -> _UndrainableCallbackResult:
    error = _UndrainableCallbackResult(
        f"{name} {reason}; worker teardown is required"
    )
    if primary_error is not None and hasattr(error, "add_note"):
        error.add_note(f"context operation failed: {primary_error!r}")
    if secondary_error is not None and hasattr(error, "add_note"):
        error.add_note(f"context cleanup also failed: {secondary_error!r}")
    return error


def _quarantine_worker_state(
    cache: object,
    error: BaseException,
    mark_unavailable: Callable[[], None],
) -> None:
    mark_unavailable()
    try:
        cache.quarantine()
    except BaseException as quarantine_error:
        if hasattr(error, "add_note"):
            error.add_note(
                "cache quarantine failed; installation remains unavailable: "
                f"{quarantine_error!r}"
            )


def _cache_is_quarantined(cache: object) -> bool:
    if getattr(cache, "quarantined", False):
        return True
    return bool(getattr(getattr(cache, "stats", None), "quarantined", False))


def _validate_worker_teardown(
    callback: Callable[[BaseException], Awaitable[bool]] | None,
) -> None:
    if callback is None:
        return
    if not callable(callback):
        raise TypeError("worker_teardown must be an async callable")
    if not inspect.iscoroutinefunction(callback) and not inspect.iscoroutinefunction(
        getattr(callback, "__call__", None)
    ):
        raise TypeError("worker_teardown must be an async callable")


def _teardown_worker_on_owner_loop(
    owner_loop: asyncio.AbstractEventLoop,
    callback: Callable[[BaseException], Awaitable[bool]] | None,
    error: BaseException,
) -> None:
    if callback is None:
        _abort_process_after_worker_failure()

    async def teardown() -> bool:
        result = callback(error)
        if not inspect.isawaitable(result):
            raise TypeError("worker_teardown returned a non-awaitable result")
        return await asyncio.wait_for(
            result,
            timeout=_WORKER_TEARDOWN_TIMEOUT_SECONDS,
        )

    task = None
    teardown_coro = teardown()
    try:
        task = asyncio.run_coroutine_threadsafe(teardown_coro, owner_loop)
    except BaseException:
        teardown_coro.close()
        _abort_process_after_worker_failure()
    try:
        stopped = task.result(timeout=_WORKER_TEARDOWN_TIMEOUT_SECONDS + 1)
    except BaseException:
        if task is not None:
            task.cancel()
        _abort_process_after_worker_failure()
    if stopped is not True:
        _abort_process_after_worker_failure()


def _abort_process_after_worker_failure() -> None:
    """Process death is the only safe fallback for untracked native work."""
    os._exit(_WORKER_TEARDOWN_FAILED_EXIT_CODE)


def _capture_sync_call(callback: Callable[..., _T], *args: object) -> _WorkerResult:
    try:
        return _WorkerResult(value=callback(*args))
    except BaseException as error:
        return _WorkerResult(error=error)


def _validate_sync_callable(callback: Callable[..., object], name: str) -> None:
    if not callable(callback):
        raise TypeError(f"{name} must be callable")
    if inspect.iscoroutinefunction(callback) or inspect.iscoroutinefunction(
        getattr(callback, "__call__", None)
    ):
        raise TypeError(f"{name} must be synchronous")


def _call_sync(
    callback: Callable[..., _T],
    name: str,
    *,
    allow_result: bool = False,
) -> _T | None:
    result = callback()
    if result is None:
        return
    if isinstance(result, ConcurrentFuture):
        result.cancel()
        try:
            result.result(timeout=_CALLBACK_DRAIN_TIMEOUT_SECONDS)
        except TimeoutError as error:
            if not result.done():
                raise _UndrainableCallbackResult(
                    f"{name} returned a future that did not finish after cancellation; "
                    "worker teardown is required"
                ) from error
        except BaseException:
            pass
        raise _UnsafeCallbackResult(
            f"{name} returned a future; worker restart is required"
        )
    if isinstance(result, asyncio.Future):
        _drain_and_reject_asyncio_future(result, name)
        raise TypeError(f"{name} returned a future; callbacks must return None")
    if inspect.isawaitable(result):
        _reject_opaque_awaitable(result, name)
    if allow_result:
        return result
    raise _UnsafeCallbackResult(
        f"{name} returned a non-None result; worker restart is required"
    )


def _drain_and_reject_asyncio_future(future: asyncio.Future, name: str) -> None:
    loop = future.get_loop()
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None
    if running_loop is loop:
        raise _UndrainableCallbackResult(
            f"{name} returned a pending future on the current loop; "
            "worker restart is required"
        )

    completed = threading.Event()
    state: dict[str, bool | None] = {"was_pending": None}

    def consume_result() -> None:
        # Future state and result access must stay on its owning event loop.
        try:
            future.result()
        except BaseException:
            pass
        completed.set()

    def inspect_on_owner_loop() -> None:
        was_pending = not future.done()
        state["was_pending"] = was_pending
        if was_pending:
            future.add_done_callback(lambda _future: consume_result())
        else:
            consume_result()

    try:
        loop.call_soon_threadsafe(inspect_on_owner_loop)
    except RuntimeError:
        pass
    if not completed.wait(_CALLBACK_DRAIN_TIMEOUT_SECONDS):
        raise _UndrainableCallbackResult(
            f"{name} returned a future that did not finish before its deadline; "
            "worker teardown is required"
        )
    if state["was_pending"] is False:
        raise _UnsafeCallbackResult(
            f"{name} returned a completed asyncio future; worker restart is required"
        )
    raise _UnsafeCallbackResult(
        f"{name} returned a pending future; natural completion was drained, "
        "worker restart is required"
    )


def _reject_opaque_awaitable(awaitable: object, name: str) -> None:
    # A cold coroutine has not run and can be closed without moving its body
    # onto another thread. Other awaitables may already own active work; never
    # drive their protocol on the request loop or outside the HIP worker.
    if inspect.iscoroutine(awaitable) and inspect.getcoroutinestate(awaitable) == inspect.CORO_CREATED:
        awaitable.close()
    raise _UndrainableCallbackResult(
        f"{name} returned an opaque awaitable that cannot be drained safely; "
        "worker restart is required"
    )
