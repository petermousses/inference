"""Synchronous execution of one pinned DeepSeek FP4 SwiGLU expert."""

from __future__ import annotations

from collections.abc import Callable
import math

from expert_cache import FP4ExpertCache, FP4ExpertKey


def execute_fp4_expert(
    cache: FP4ExpertCache,
    key: FP4ExpertKey,
    activations,
    linear: Callable,
    *,
    swiglu_limit: float,
    routing_weights=None,
    fence: Callable[[], None] | None = None,
):
    """Run the reference 1017 expert formula while its cached weights are leased.

    ``linear`` is the pinned model's ``linear`` function. The lease is held until
    the selected stream is synchronized, including when a projection raises
    after enqueueing work. Pass ``fence`` when the linear implementation uses
    streams other than the activation's current stream. A failed fence means the
    worker must be treated as unhealthy and removed before it accepts more work.
    Its caller must retain the root memory reservation until the worker exits.
    """
    if not callable(linear):
        raise TypeError("linear must be callable")

    import torch
    import torch.nn.functional as functional

    if not isinstance(activations, torch.Tensor):
        raise TypeError("activations must be a torch.Tensor")
    if activations.ndim < 1:
        raise ValueError("activations must have rank at least one")
    if isinstance(swiglu_limit, bool) or not isinstance(swiglu_limit, (int, float)):
        raise TypeError("swiglu_limit must be a number from the pinned model config")
    if not math.isfinite(swiglu_limit) or swiglu_limit < 0:
        raise ValueError("swiglu_limit must be finite and nonnegative")
    if fence is None:
        if activations.is_cuda:
            fence = torch.cuda.current_stream(activations.device).synchronize
        else:
            fence = lambda: None
    elif not callable(fence):
        raise TypeError("fence must be callable")

    dtype = activations.dtype
    with cache.acquire(key) as expert:
        try:
            gate = linear(activations, expert.w1).float()
            up = linear(activations, expert.w3).float()
            if swiglu_limit > 0:
                up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
                gate = torch.clamp(gate, max=swiglu_limit)
            hidden = functional.silu(gate) * up
            if routing_weights is not None:
                hidden = routing_weights * hidden
            return linear(hidden.to(dtype), expert.w2)
        finally:
            # Native HIP calls enqueue work asynchronously. Keep the cache lease
            # until all three projections have finished using their weight views.
            try:
                fence()
            except BaseException:
                expert.quarantine()
                raise
