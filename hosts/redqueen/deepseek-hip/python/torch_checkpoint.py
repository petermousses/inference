"""Materialize bounded checkpoint tensors directly onto a PyTorch device.

Callers must reserve the live host and device allocations before materializing data.
During the copy, the raw host bytes and destination tensor overlap, so reserve at
least twice the requested output size in addition to the checkpoint reader's
transient read headroom. The returned device tensors remain caller-owned and must
stay accounted until released.
"""

from __future__ import annotations

import warnings
from typing import Sequence

from checkpoint_reader import CheckpointError, SafeTensorCheckpoint, _MAX_ROWS_PER_READ


_TORCH_DTYPES = {
    "BOOL": "bool",
    "U8": "uint8",
    "I8": "int8",
    "I16": "int16",
    "I32": "int32",
    "I64": "int64",
    "F16": "float16",
    "BF16": "bfloat16",
    "F32": "float32",
    "F64": "float64",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E4M3FN": "float8_e4m3fn",
    "F8_E4M3FNUZ": "float8_e4m3fnuz",
    "F8_E5M2": "float8_e5m2",
    "F8_E5M2FNUZ": "float8_e5m2fnuz",
    "F8_E8M0": "float8_e8m0fnu",
}


def _torch_dtype(dtype_name: str):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to materialize checkpoint tensors") from exc
    torch_name = _TORCH_DTYPES.get(dtype_name)
    dtype = getattr(torch, torch_name, None) if torch_name is not None else None
    if dtype is None:
        raise CheckpointError(f"PyTorch does not support checkpoint dtype: {dtype_name}")
    return torch, dtype


def _destination(dtype_name: str, device):
    torch, dtype = _torch_dtype(dtype_name)
    if device is None:
        raise ValueError("device must be specified")
    return torch, dtype, torch.device(device)


def _validate_tensor_read(checkpoint: SafeTensorCheckpoint, name: str, max_bytes: int):
    checkpoint._validate_limit(max_bytes)
    info = checkpoint.tensors[name]
    if info.nbytes > max_bytes:
        raise ValueError(f"tensor read of {info.nbytes} bytes exceeds limit of {max_bytes} bytes")
    return info


def _validate_rows(checkpoint: SafeTensorCheckpoint, name: str, rows, max_bytes: int):
    checkpoint._validate_limit(max_bytes)
    info = checkpoint.tensors[name]
    if not info.shape:
        raise ValueError(f"tensor {name} is scalar and has no rows")
    if type(rows) not in (list, tuple, range):
        raise TypeError("rows must be a built-in list, tuple, or range")
    if type(rows) is list:
        # Bound the snapshot even if a caller mutates the list concurrently.
        rows = rows[: _MAX_ROWS_PER_READ + 1]
    try:
        row_count = len(rows)
    except OverflowError as exc:
        raise ValueError(f"row read exceeds hard limit of {_MAX_ROWS_PER_READ} rows") from exc
    if row_count > _MAX_ROWS_PER_READ:
        raise ValueError(f"row read exceeds hard limit of {_MAX_ROWS_PER_READ} rows")
    output_size = row_count * info.row_bytes
    if output_size > max_bytes:
        raise ValueError(f"row read exceeds limit of {max_bytes} bytes")
    for row in rows:
        if type(row) is not int or row < 0 or row >= info.shape[0]:
            raise IndexError(f"row index out of range for {name}: {row!r}")
    return info, rows, row_count


def _to_device(raw: bytes, torch, dtype, shape: tuple[int, ...], target):
    element_size = torch.empty((), dtype=dtype).element_size()
    element_count = 1
    for dimension in shape:
        element_count *= dimension
    if len(raw) != element_count * element_size:
        raise CheckpointError("checkpoint bytes do not match the requested tensor shape and dtype")
    if not raw:
        return torch.empty(shape, dtype=dtype, device=target)

    # `raw` is immutable bytes. The CPU view is read-only by contract and is copied
    # before return, so no returned tensor can mutate the checkpoint buffer.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The given buffer is not writable.*",
            category=UserWarning,
        )
        cpu_view = torch.frombuffer(raw, dtype=dtype)
    return cpu_view.reshape(shape).to(device=target, non_blocking=False, copy=True)


def read_tensor_to_device(
    checkpoint: SafeTensorCheckpoint,
    name: str,
    *,
    device,
    max_bytes: int = 64 * 1024 * 1024,
):
    """Read one bounded tensor and return an independent tensor on ``device``."""
    info = _validate_tensor_read(checkpoint, name, max_bytes)
    torch, dtype, target = _destination(info.dtype, device)
    raw = checkpoint.read_tensor_bytes(name, max_bytes=max_bytes)
    return _to_device(raw, torch, dtype, info.shape, target)


def read_rows_to_device(
    checkpoint: SafeTensorCheckpoint,
    name: str,
    rows: Sequence[int],
    *,
    device,
    max_bytes: int = 64 * 1024 * 1024,
):
    """Read selected axis-zero rows in caller order and return them on ``device``.

    The reservation must cover ``2 * output_bytes`` for the overlapping raw host
    buffer and device tensor, plus the checkpoint reader's transient read headroom.
    """
    info, stable_rows, row_count = _validate_rows(checkpoint, name, rows, max_bytes)
    torch, dtype, target = _destination(info.dtype, device)
    raw = checkpoint.read_rows(name, stable_rows, max_bytes=max_bytes)
    return _to_device(raw, torch, dtype, (row_count, *info.shape[1:]), target)
