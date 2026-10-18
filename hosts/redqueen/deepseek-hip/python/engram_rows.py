"""Bounded, checkpoint-backed reads for the pinned DeepSeek 1017 Engram tables."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from checkpoint_reader import CheckpointError, SafeTensorCheckpoint, _MAX_ROWS_PER_READ
from torch_checkpoint import read_rows_to_device


_ENGRAM_BLOCK_SIZE = 32
_DEFAULT_MAX_INDICES_PER_BATCH = 32_768
_DEFAULT_MAX_ROWS_PER_BATCH = 16_384
_MAX_INDEX_BATCH = 65_536
_MAX_READ_HEADROOM = 2 * 1024 * 1024
_MAX_READ_BYTES = 64 * 1024 * 1024
_PATCH_MARKER = "_redqueen_engram_row_reader"
_INSTALLATION_LOCK = threading.Lock()


@dataclass(frozen=True)
class EngramLookupMemoryEstimate:
    """Conservative live lookup bytes; allocator-retained pools are accounted separately."""

    output_bytes: int
    host_peak_bytes: int
    device_peak_bytes: int


class EngramRowReader:
    """Read unique Engram rows in bounded batches and reproduce the official lookup math.

    The returned BF16 embedding output spans the complete input index tensor. The temporary
    unique-row tensors and checkpoint transfers are bounded by ``max_indices_per_batch`` and
    ``max_rows_per_batch``. Callers must retain the request's memory reservation through the
    final model GPU fence. This class intentionally does not cache rows across requests; any
    persistent cache needs a separately brokered byte limit.
    """

    def __init__(
        self,
        checkpoint: SafeTensorCheckpoint,
        layer_id: int,
        *,
        num_embeddings: int,
        embedding_dim: int,
        max_indices_per_batch: int = _DEFAULT_MAX_INDICES_PER_BATCH,
        max_rows_per_batch: int = _DEFAULT_MAX_ROWS_PER_BATCH,
    ) -> None:
        if not isinstance(checkpoint, SafeTensorCheckpoint):
            raise TypeError("checkpoint must be a SafeTensorCheckpoint")
        if getattr(checkpoint, "_metadata_only", False):
            raise ValueError("Engram row reads require a payload-enabled checkpoint reader")
        if type(layer_id) is not int or layer_id < 0:
            raise ValueError("layer_id must be a nonnegative integer")
        if type(num_embeddings) is not int or num_embeddings <= 0:
            raise ValueError("num_embeddings must be a positive integer")
        if type(embedding_dim) is not int or embedding_dim <= 0:
            raise ValueError("embedding_dim must be a positive integer")
        if embedding_dim % _ENGRAM_BLOCK_SIZE:
            raise ValueError("Engram embedding_dim must be divisible by the FP8 block size")
        if (
            type(max_indices_per_batch) is not int
            or not 0 < max_indices_per_batch <= _MAX_INDEX_BATCH
        ):
            raise ValueError(f"max_indices_per_batch must be between 1 and {_MAX_INDEX_BATCH}")
        if (
            type(max_rows_per_batch) is not int
            or not 0 < max_rows_per_batch <= _MAX_ROWS_PER_READ
        ):
            raise ValueError(f"max_rows_per_batch must be between 1 and {_MAX_ROWS_PER_READ}")

        prefix = f"layers.{layer_id}.engram.embed"
        self.weight_name = f"{prefix}.weight"
        self.scale_name = f"{prefix}.scale"
        try:
            self._weight = checkpoint.tensors[self.weight_name]
            self._scale = checkpoint.tensors[self.scale_name]
        except KeyError as exc:
            raise CheckpointError(f"checkpoint is missing Engram tensor: {exc.args[0]}") from exc
        expected_weight_shape = (num_embeddings, embedding_dim)
        expected_scale_shape = (num_embeddings, embedding_dim // _ENGRAM_BLOCK_SIZE)
        if self._weight.dtype not in {"F8_E4M3", "F8_E4M3FN"}:
            raise CheckpointError(
                f"{self.weight_name} must use FP8 E4M3 storage, got {self._weight.dtype}"
            )
        if self._weight.shape != expected_weight_shape:
            raise CheckpointError(
                f"{self.weight_name} has shape {self._weight.shape}; expected {expected_weight_shape}"
            )
        if self._scale.dtype != "F8_E8M0":
            raise CheckpointError(
                f"{self.scale_name} must use FP8 E8M0 storage, got {self._scale.dtype}"
            )
        if self._scale.shape != expected_scale_shape:
            raise CheckpointError(
                f"{self.scale_name} has shape {self._scale.shape}; expected {expected_scale_shape}"
            )
        if max_rows_per_batch * max(self._weight.row_bytes, self._scale.row_bytes) > _MAX_READ_BYTES:
            raise ValueError("max_rows_per_batch exceeds the bounded Engram row transfer size")

        self.layer_id = layer_id
        self._checkpoint = checkpoint
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.scale_blocks = embedding_dim // _ENGRAM_BLOCK_SIZE
        self.max_indices_per_batch = max_indices_per_batch
        self.max_rows_per_batch = max_rows_per_batch
        self._lookup_lock = threading.Lock()

    def estimate_memory(self, indices_shape: tuple[int, ...] | list[int]) -> EngramLookupMemoryEstimate:
        """Estimate output and bounded scratch before request admission or materialization."""
        if not isinstance(indices_shape, (tuple, list)) or not indices_shape:
            raise ValueError("indices_shape must be a nonempty tuple or list of dimensions")
        if any(type(dimension) is not int or dimension < 0 for dimension in indices_shape):
            raise ValueError("indices_shape dimensions must be nonnegative integers")
        index_count = math.prod(indices_shape)
        output_bytes = index_count * self.embedding_dim * 2  # BF16
        batch = min(index_count, self.max_indices_per_batch)
        rows = min(batch, self.max_rows_per_batch)
        row_bytes = self._weight.row_bytes + self._scale.row_bytes

        # Per index: CPU IDs/mask/unique/inverse/list storage, unique FP8 rows, gathered rows,
        # FP32 dequantization, BF16 conversion, and their scales. Add one returned row batch.
        host_peak = batch * 96 + 2 * rows * row_bytes + _MAX_READ_HEADROOM
        device_peak = (
            output_bytes
            + batch * (12 * self.embedding_dim + 6 * self.scale_blocks + 16)
            + rows * row_bytes
        )
        return EngramLookupMemoryEstimate(
            output_bytes=output_bytes,
            host_peak_bytes=host_peak,
            device_peak_bytes=device_peak,
        )

    def lookup(
        self,
        indices,
        *,
        vocab_start_idx: int = 0,
        vocab_end_idx: int | None = None,
    ):
        """Return BF16 embeddings with the official out-of-shard mask and scale math."""
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for Engram row lookup") from exc
        if not isinstance(indices, torch.Tensor):
            raise TypeError("indices must be a PyTorch tensor")
        if indices.dtype != torch.int64:
            raise TypeError("Engram indices must use torch.int64")
        if indices.device.type == "meta":
            raise ValueError("Engram indices must have allocated storage")
        if type(vocab_start_idx) is not int or type(vocab_end_idx) not in (int, type(None)):
            raise TypeError("Engram vocabulary bounds must be integers")
        if vocab_end_idx is None:
            vocab_end_idx = self.num_embeddings
        if vocab_start_idx != 0 or vocab_end_idx != self.num_embeddings:
            raise ValueError("the Redqueen Engram row reader supports the single-rank full table only")

        output_shape = (*tuple(indices.shape), self.embedding_dim)
        with torch.inference_mode(), self._lookup_lock:
            output = torch.zeros(output_shape, dtype=torch.bfloat16, device=indices.device)
            if indices.numel() == 0:
                return output
            flat_indices = indices.reshape(-1)
            flat_output = output.reshape(-1, self.embedding_dim)
            for start in range(0, flat_indices.numel(), self.max_indices_per_batch):
                end = min(start + self.max_indices_per_batch, flat_indices.numel())
                index_cpu = flat_indices[start:end].detach().to("cpu").contiguous()
                valid_cpu = (index_cpu >= 0) & (index_cpu < self.num_embeddings)
                if not bool(valid_cpu.any()):
                    continue
                valid_values = index_cpu[valid_cpu]
                unique_rows, inverse = torch.unique(
                    valid_values, sorted=True, return_inverse=True
                )
                unique_count = unique_rows.numel()
                device = indices.device
                weights = torch.empty(
                    (unique_count, self.embedding_dim),
                    dtype=torch.float8_e4m3fn,
                    device=device,
                )
                scales = torch.empty(
                    (unique_count, self.scale_blocks),
                    dtype=torch.float8_e8m0fnu,
                    device=device,
                )
                for row_start in range(0, unique_count, self.max_rows_per_batch):
                    row_end = min(row_start + self.max_rows_per_batch, unique_count)
                    rows = unique_rows[row_start:row_end].tolist()
                    weight_chunk = read_rows_to_device(
                        self._checkpoint,
                        self.weight_name,
                        rows,
                        device=device,
                        max_bytes=(row_end - row_start) * self._weight.row_bytes,
                    )
                    scale_chunk = read_rows_to_device(
                        self._checkpoint,
                        self.scale_name,
                        rows,
                        device=device,
                        max_bytes=(row_end - row_start) * self._scale.row_bytes,
                    )
                    weights[row_start:row_end].copy_(weight_chunk)
                    scales[row_start:row_end].copy_(scale_chunk)

                gather = inverse.to(device=device)
                gathered_weights = weights.index_select(0, gather)
                gathered_scales = scales.index_select(0, gather)
                values = (
                    gathered_weights.float()
                    .unflatten(-1, (-1, _ENGRAM_BLOCK_SIZE))
                    * gathered_scales.float().unsqueeze(-1)
                ).flatten(-2).to(torch.bfloat16)
                positions = torch.nonzero(valid_cpu, as_tuple=False).flatten().to(device=device)
                flat_output[start:end].index_copy_(0, positions, values)
            return output


@dataclass(frozen=True)
class _ForwardPatch:
    embed: object
    original: object
    installed: object
    reader: EngramRowReader
    marker: object


class EngramRowInstallation:
    """Reversible per-instance Engram.forward patches for one transformer.

    Before closing, the caller must drain complete transformer executions through its model-wide
    execution lease and fence their GPU work. This object waits only for calls already inside an
    installed Engram reader; it cannot see a forward that has not reached an Engram layer yet.
    """

    def __init__(self, patches: list[_ForwardPatch]) -> None:
        self._patches = patches
        self._closed = False
        self._closing = False
        self._condition = threading.Condition()
        self._active = 0

    @property
    def readers(self) -> Mapping[int, EngramRowReader]:
        return MappingProxyType({patch.reader.layer_id: patch.reader for patch in self._patches})

    def __enter__(self) -> EngramRowInstallation:
        if self._closed or self._closing:
            raise RuntimeError("Engram row installation is already closed")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _lookup(self, reader: EngramRowReader, embed: object, indices):
        with self._condition:
            if self._closed or self._closing:
                raise RuntimeError("Engram row installation is closed")
            self._active += 1
        try:
            return reader.lookup(
                indices,
                vocab_start_idx=embed.vocab_start_idx,
                vocab_end_idx=embed.vocab_end_idx,
            )
        finally:
            with self._condition:
                self._active -= 1
                if self._active == 0:
                    self._condition.notify_all()

    def close(self) -> None:
        """Restore original methods after the caller has drained model execution and GPU work."""
        with _INSTALLATION_LOCK:
            with self._condition:
                if self._closed:
                    return
                self._closing = True
                while self._active:
                    self._condition.wait()
                for patch in self._patches:
                    if getattr(patch.embed, _PATCH_MARKER, None) is not patch.marker:
                        raise RuntimeError("Engram row installation ownership changed")
                    if patch.embed.forward is not patch.installed:
                        raise RuntimeError("Engram.forward changed while row reads were installed")
                for patch in self._patches:
                    patch.embed.forward = patch.original
                    delattr(patch.embed, _PATCH_MARKER)
                self._closed = True


def install_engram_row_reads(
    transformer: object,
    checkpoint: SafeTensorCheckpoint,
    *,
    max_indices_per_batch: int = _DEFAULT_MAX_INDICES_PER_BATCH,
    max_rows_per_batch: int = _DEFAULT_MAX_ROWS_PER_BATCH,
) -> EngramRowInstallation:
    """Patch all pinned backbone Engram tables to read rows from the checkpoint on demand.

    The graph must have been constructed on the meta device so the enormous checkpoint tables
    are not materialized before this installation. Redqueen's first runtime supports one complete
    table per configured Engram layer; distributed/vocabulary-sharded execution is rejected.
    """
    blocks = getattr(transformer, "layers", None)
    if blocks is None:
        raise TypeError("transformer.layers must contain backbone blocks")
    try:
        blocks = tuple(blocks)
    except TypeError as exc:
        raise TypeError("transformer.layers must contain backbone blocks") from exc

    planned: list[tuple[object, EngramRowReader]] = []
    seen_layers: set[int] = set()
    seen_embeddings: set[int] = set()
    for block in blocks:
        engram = getattr(block, "engram", None)
        if engram is None:
            continue
        embed = getattr(engram, "embed", None)
        layer_id = getattr(block, "layer_id", None)
        if embed is None or type(layer_id) is not int or layer_id < 0:
            raise TypeError("Engram blocks must expose their nonnegative layer_id and embed module")
        if layer_id in seen_layers or id(embed) in seen_embeddings:
            raise ValueError("transformer contains duplicate Engram layer or embedding references")
        seen_layers.add(layer_id)
        seen_embeddings.add(id(embed))
        if getattr(embed, _PATCH_MARKER, None) is not None:
            raise RuntimeError("Engram row reads are already installed")
        for name in ("num_embeddings", "dim", "vocab_start_idx", "vocab_end_idx", "weight", "scale", "forward"):
            if not hasattr(embed, name):
                raise TypeError(f"Engram embedding must expose {name}")
        if not callable(embed.forward):
            raise TypeError("Engram embedding forward must be callable")
        if not getattr(embed.weight, "is_meta", False) or not getattr(embed.scale, "is_meta", False):
            raise ValueError("Engram weight and scale parameters must remain meta tensors")
        if embed.vocab_start_idx != 0 or embed.vocab_end_idx != embed.num_embeddings:
            raise ValueError("the Redqueen Engram row reader requires a single-rank full table")
        if tuple(embed.weight.shape) != (embed.num_embeddings, embed.dim):
            raise CheckpointError(f"Engram graph weight shape is invalid for layer {layer_id}")
        if tuple(embed.scale.shape) != (embed.num_embeddings, embed.dim // _ENGRAM_BLOCK_SIZE):
            raise CheckpointError(f"Engram graph scale shape is invalid for layer {layer_id}")
        reader = EngramRowReader(
            checkpoint,
            layer_id,
            num_embeddings=embed.num_embeddings,
            embedding_dim=embed.dim,
            max_indices_per_batch=max_indices_per_batch,
            max_rows_per_batch=max_rows_per_batch,
        )
        if str(embed.weight.dtype) != "torch.float8_e4m3fn":
            raise CheckpointError(f"Engram graph weight dtype is invalid for layer {layer_id}")
        if str(embed.scale.dtype) != "torch.float8_e8m0fnu":
            raise CheckpointError(f"Engram graph scale dtype is invalid for layer {layer_id}")
        planned.append((embed, reader))

    if not planned:
        raise ValueError("transformer has no Engram embeddings to install")
    patches: list[_ForwardPatch] = []
    installation = EngramRowInstallation(patches)
    try:
        with _INSTALLATION_LOCK:
            # Validate the complete patch set under the same lock used by close(). A concurrent
            # installer may have passed the earlier metadata checks while this installer held
            # the GIL, then patched these instances before reaching this critical section.
            # Reject that state before taking original callables or mutating any instance.
            for embed, _reader in planned:
                if getattr(embed, _PATCH_MARKER, None) is not None:
                    raise RuntimeError("Engram row reads are already installed")

            originals = [(embed, reader, embed.forward) for embed, reader in planned]
            for embed, _reader, original in originals:
                if not callable(original):
                    raise TypeError("Engram embedding forward must be callable")

            for embed, reader, original in originals:
                def installed(indices, *, _embed=embed, _reader=reader):
                    return installation._lookup(_reader, _embed, indices)

                marker = object()
                patch = _ForwardPatch(embed, original, installed, reader, marker)
                try:
                    setattr(embed, _PATCH_MARKER, marker)
                    embed.forward = installed
                except BaseException:
                    if getattr(embed, _PATCH_MARKER, None) is marker:
                        delattr(embed, _PATCH_MARKER)
                    if getattr(embed, "forward", None) is installed:
                        embed.forward = original
                    raise
                patches.append(patch)
    except BaseException:
        with _INSTALLATION_LOCK:
            for patch in reversed(patches):
                patch.embed.forward = patch.original
                if getattr(patch.embed, _PATCH_MARKER, None) is patch.marker:
                    delattr(patch.embed, _PATCH_MARKER)
        raise
    return installation
