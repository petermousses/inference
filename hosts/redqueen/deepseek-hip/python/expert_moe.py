"""Apply pinned DeepSeek MoE routing through the cache-leased expert path."""

from __future__ import annotations

from collections.abc import Callable

from expert_cache import FP4ExpertCache, FP4ExpertKey
from expert_execution import execute_fp4_expert


def execute_cached_moe(
    cache: FP4ExpertCache,
    activations,
    gate: Callable,
    shared_expert: Callable,
    linear: Callable,
    *,
    layer_id: int,
    n_routed_experts: int,
    swiglu_limit: float,
    image_mask=None,
    fence: Callable[[], None] | None = None,
    mtp_layer: bool = False,
):
    """Run one single-rank MoE block using the official gate and shared expert.

    ``gate`` must implement the pinned ``Gate.forward`` contract and return its
    already-selected, already-weighted ``(weights, indices)``. This function
    does not recalculate scores, add correction bias, renormalize routes, or
    change expert selection. ``layer_id`` is the checkpoint-local layer id;
    set ``mtp_layer=True`` for MTP expert keys. This helper implements one rank,
    matching Redqueen's single-device worker. The caller must reserve the complete
    MoE operation and keep that reservation until this synchronous function has
    returned. A final stream fence covers the gate, shared expert, and output
    accumulation; a fence failure quarantines the cache and requires worker
    teardown before releasing the root reservation.
    """
    if not callable(gate) or not callable(shared_expert) or not callable(linear):
        raise TypeError("gate, shared_expert, and linear must be callable")
    if fence is not None and not callable(fence):
        raise TypeError("fence must be callable")
    if type(n_routed_experts) is not int or n_routed_experts <= 0:
        raise ValueError("n_routed_experts must be a positive integer")
    # Validate the layer id even when no token selects a routed expert.
    FP4ExpertKey(layer_id, 0, mtp_layer=mtp_layer)

    import torch

    if not isinstance(activations, torch.Tensor):
        raise TypeError("activations must be a torch.Tensor")
    if activations.ndim < 1:
        raise ValueError("activations must have rank at least one")
    if fence is None:
        if activations.is_cuda:
            fence = torch.cuda.current_stream(activations.device).synchronize
        else:
            fence = lambda: None
    shape = activations.size()
    flat = activations.view(-1, shape[-1])
    flat_image_mask = None
    if image_mask is not None:
        if not isinstance(image_mask, torch.Tensor):
            raise TypeError("image_mask must be a torch.Tensor")
        flat_image_mask = image_mask.flatten()
        if flat_image_mask.numel() != flat.shape[0]:
            raise ValueError("image_mask must contain one value per token")

    try:
        weights, indices = gate(flat, flat_image_mask)
        if not isinstance(weights, torch.Tensor) or not isinstance(indices, torch.Tensor):
            raise TypeError("gate must return tensor weights and indices")
        if weights.ndim != 2 or weights.shape != indices.shape:
            raise ValueError("gate weights and indices must have the same rank-two shape")
        if weights.shape[0] != flat.shape[0]:
            raise ValueError("gate outputs must contain one row per token")

        counts = torch.bincount(indices.flatten(), minlength=n_routed_experts).tolist()
        if len(counts) != n_routed_experts:
            raise ValueError("gate returned an expert index outside the configured range")
        output = torch.zeros_like(flat, dtype=torch.float32)
        for expert_id, count in enumerate(counts):
            if count == 0:
                continue
            token_indices, route_positions = torch.where(indices == expert_id)
            expert_output = execute_fp4_expert(
                cache,
                FP4ExpertKey(layer_id, expert_id, mtp_layer=mtp_layer),
                flat[token_indices],
                linear,
                swiglu_limit=swiglu_limit,
                routing_weights=weights[token_indices, route_positions, None],
                fence=fence,
            )
            output[token_indices] += expert_output

        shared_output = shared_expert(flat)
        if not isinstance(shared_output, torch.Tensor) or shared_output.shape != flat.shape:
            raise ValueError("shared expert output must match the flattened activation shape")
        output += shared_output
        return output.type_as(flat).view(shape)
    finally:
        try:
            fence()
        except BaseException:
            cache.quarantine()
            raise
