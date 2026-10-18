"""Bounded loading of one pinned DeepSeek 1017 FP4 routed expert."""

from __future__ import annotations

from dataclasses import dataclass

from checkpoint_reader import CheckpointError, SafeTensorCheckpoint
from torch_checkpoint import read_tensor_to_device


@dataclass(frozen=True)
class FP4ExpertMemoryEstimate:
    """Resident payload and the largest single-tensor transfer for one expert."""

    resident_bytes: int
    largest_tensor_bytes: int

    @property
    def conservative_load_peak_bytes(self) -> int:
        """Resident payload plus overlapping host and device copies of one tensor."""
        return self.resident_bytes + 2 * self.largest_tensor_bytes


@dataclass(frozen=True)
class LoadedFP4Expert:
    """Three FP4 weight tensors with scales attached as expected by the 1017 linear path."""

    layer_id: int
    expert_id: int
    mtp_layer: bool
    w1: object
    w2: object
    w3: object

    @property
    def resident_bytes(self) -> int:
        return sum(
            weight.nbytes + weight.scale.nbytes
            for weight in (self.w1, self.w2, self.w3)
        )


def _names(layer_id: int, expert_id: int, mtp_layer: bool) -> dict[str, tuple[str, str]]:
    for value, name in ((layer_id, "layer_id"), (expert_id, "expert_id")):
        if type(value) is not int:
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must be nonnegative")
    if type(mtp_layer) is not bool:
        raise TypeError("mtp_layer must be a bool")
    prefix = "mtp" if mtp_layer else "layers"
    base = f"{prefix}.{layer_id}.ffn.experts.{expert_id}"
    return {
        projection: (f"{base}.{projection}.weight", f"{base}.{projection}.scale")
        for projection in ("w1", "w2", "w3")
    }


def _validated_infos(
    checkpoint: SafeTensorCheckpoint,
    layer_id: int,
    expert_id: int,
    mtp_layer: bool,
):
    names = _names(layer_id, expert_id, mtp_layer)
    infos = {}
    for projection, (weight_name, scale_name) in names.items():
        try:
            weight = checkpoint.tensors[weight_name]
            scale = checkpoint.tensors[scale_name]
        except KeyError as exc:
            raise CheckpointError(f"checkpoint is missing expert tensor: {exc.args[0]}") from exc
        if weight.dtype != "I8":
            raise CheckpointError(f"{weight_name} must use packed I8 storage, got {weight.dtype}")
        if scale.dtype != "F8_E8M0":
            raise CheckpointError(f"{scale_name} must use E8M0 scales, got {scale.dtype}")
        if len(weight.shape) != 2 or len(scale.shape) != 2:
            raise CheckpointError(f"{weight_name} and {scale_name} must be rank two")
        output_dim, packed_input_dim = weight.shape
        input_dim = packed_input_dim * 2
        expected_scale_shape = (output_dim, input_dim // 32)
        if (
            output_dim == 0
            or input_dim == 0
            or output_dim % 32
            or input_dim % 32
            or scale.shape != expected_scale_shape
        ):
            raise CheckpointError(f"{weight_name} and {scale_name} do not match the 1017 FP4 block layout")
        infos[projection] = (weight_name, scale_name, weight, scale)

    w1 = infos["w1"][2]
    w2 = infos["w2"][2]
    w3 = infos["w3"][2]
    if w1.shape != w3.shape or w2.shape[1] * 2 != w1.shape[0] or w2.shape[0] != w1.shape[1] * 2:
        raise CheckpointError("expert projection shapes do not form a compatible SwiGLU block")
    return infos


def _estimate_memory(infos) -> FP4ExpertMemoryEstimate:
    tensors = [info for group in infos.values() for info in group[2:]]
    return FP4ExpertMemoryEstimate(
        resident_bytes=sum(info.nbytes for info in tensors),
        largest_tensor_bytes=max(info.nbytes for info in tensors),
    )


def estimate_fp4_expert_memory(
    checkpoint: SafeTensorCheckpoint,
    layer_id: int,
    expert_id: int,
    *,
    mtp_layer: bool = False,
) -> FP4ExpertMemoryEstimate:
    """Validate an expert and report resident bytes before allocating it."""
    return _estimate_memory(_validated_infos(checkpoint, layer_id, expert_id, mtp_layer))


def load_fp4_expert(
    checkpoint: SafeTensorCheckpoint,
    layer_id: int,
    expert_id: int,
    *,
    device,
    mtp_layer: bool = False,
) -> LoadedFP4Expert:
    """Load one routed expert's packed weights and scales onto ``device``.

    The caller must reserve ``conservative_load_peak_bytes`` plus checkpoint-reader
    transient headroom before calling. Keep the resident payload accounted while
    cached; on failure, account any memory still held by the device allocator until
    its backing is returned. If offloaded to a worker thread, retain the root
    reservation until that thread exits; cancelling an async waiter does not stop
    the load.
    """
    infos = _validated_infos(checkpoint, layer_id, expert_id, mtp_layer)
    estimate = _estimate_memory(infos)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to load FP4 experts") from exc
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if fp4_dtype is None:
        raise CheckpointError("PyTorch does not support the pinned FP4 expert format")

    loaded = {}
    for projection in ("w1", "w2", "w3"):
        weight_name, scale_name, weight_info, _ = infos[projection]
        packed_i8 = read_tensor_to_device(
            checkpoint,
            weight_name,
            device=device,
            max_bytes=max(64 * 1024 * 1024, weight_info.nbytes),
        )
        packed_weight = packed_i8.view(fp4_dtype)
        scale = read_tensor_to_device(
            checkpoint,
            scale_name,
            device=device,
            max_bytes=max(64 * 1024 * 1024, infos[projection][3].nbytes),
        )
        packed_weight.scale = scale
        loaded[projection] = packed_weight

    expert = LoadedFP4Expert(
        layer_id=layer_id,
        expert_id=expert_id,
        mtp_layer=mtp_layer,
        w1=loaded["w1"],
        w2=loaded["w2"],
        w3=loaded["w3"],
    )
    if expert.resident_bytes != estimate.resident_bytes:
        raise CheckpointError(
            f"loaded expert uses {expert.resident_bytes} bytes; expected {estimate.resident_bytes}"
        )
    return expert
