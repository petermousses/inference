"""Bound temporary memory in the pinned Engram forward without changing its math."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from types import MethodType


_PATCH_MARKER = "_redqueen_chunked_engram_forward"
_INSTALL_LOCK = threading.Lock()
_DEFAULT_SEQUENCE_TILE_ROWS = 256


@dataclass(frozen=True)
class _ForwardPatch:
    module: object
    original: object
    installed: object
    marker: object


class ChunkedEngramForwardInstallation:
    """Reversibly tile Engram's pointwise work along tokens during inference.

    The full checkpoint embedding lookup and ``wkv`` projection stay unchanged. Only
    their float32 normalization, gating, and residual work is tiled, avoiding full
    float32 copies of the four-stream hidden state and key tensor.
    """

    def __init__(self, patches: list[_ForwardPatch]) -> None:
        self._patches = patches
        self._closed = False

    def __enter__(self) -> ChunkedEngramForwardInstallation:
        if self._closed:
            raise RuntimeError("Engram forward installation is already closed")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        with _INSTALL_LOCK:
            if self._closed:
                return
            for patch in reversed(self._patches):
                if getattr(patch.module, _PATCH_MARKER, None) is not patch.marker:
                    raise RuntimeError("Engram forward patch ownership changed")
                if patch.module.forward is not patch.installed:
                    raise RuntimeError("Engram.forward changed while tiled forward was installed")
            for patch in reversed(self._patches):
                patch.module.forward = patch.original
                delattr(patch.module, _PATCH_MARKER)
            self._closed = True


def _chunked_engram_forward(sequence_tile_rows: int):
    def forward(self, x, hash_ids, token_mask=None):
        import torch

        if torch.is_grad_enabled():
            raise RuntimeError("chunked Engram forward is restricted to inference")
        if x.ndim != 4:
            raise ValueError("Engram hidden state must have shape [batch, sequence, hc_mult, dim]")
        if hash_ids.ndim < 2 or hash_ids.shape[:2] != x.shape[:2]:
            raise ValueError("Engram hash IDs must match the hidden-state batch and sequence")
        if x.shape[-2:] != (self.hc_mult, self.dim):
            raise ValueError("Engram hidden-state dimensions do not match the module")
        if token_mask is not None and token_mask.shape != x.shape[:2]:
            raise ValueError("Engram token mask must match the hidden-state batch and sequence")

        # Keep the checkpoint's full embedding and projection operations intact. Tiling
        # starts after this point so GEMM shapes and their numerical results are unchanged.
        kv = self.wkv(self.embed(hash_ids).flatten(-2))
        key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)
        key = key.unflatten(-1, (self.hc_mult, self.dim))
        weight = self.q_weight.float() * self.k_weight.float()
        output = torch.empty_like(x)
        sequence_length = x.shape[1]

        for start in range(0, sequence_length, sequence_tile_rows):
            end = min(start + sequence_tile_rows, sequence_length)
            h_chunk = x[:, start:end].float()
            key_chunk = key[:, start:end].float()
            value_chunk = value[:, start:end].float().unsqueeze(-2)
            rstd = torch.rsqrt(h_chunk.square().mean(-1) + self.eps) * torch.rsqrt(
                key_chunk.square().mean(-1) + self.eps
            )
            dot = (h_chunk * weight * key_chunk).sum(-1) * rstd * self.dim**-0.5
            gate = torch.sigmoid(
                torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot)
            )
            if token_mask is not None:
                gate = gate.masked_fill(~token_mask[:, start:end].unsqueeze(-1), 0)
            output[:, start:end].copy_(
                (h_chunk + gate.unsqueeze(-1) * value_chunk).to(x.dtype)
            )
        return output

    return forward


def install_chunked_engram_forward(
    transformer: object,
    *,
    sequence_tile_rows: int = _DEFAULT_SEQUENCE_TILE_ROWS,
) -> ChunkedEngramForwardInstallation:
    """Install bounded-memory Engram residual math on each configured model layer."""
    if type(sequence_tile_rows) is not int or not 0 < sequence_tile_rows <= 4096:
        raise ValueError("sequence_tile_rows must be between 1 and 4096")
    layers = getattr(transformer, "layers", None)
    if layers is None or not hasattr(layers, "__iter__") or not hasattr(layers, "__len__"):
        raise TypeError("transformer layers must be a sized module sequence")

    planned = []
    seen = set()
    for layer in layers:
        module = getattr(layer, "engram", None)
        if module is None:
            continue
        if id(module) in seen:
            raise ValueError("transformer contains a duplicate Engram module")
        seen.add(id(module))
        for name in (
            "forward",
            "embed",
            "wkv",
            "hc_mult",
            "dim",
            "eps",
            "q_weight",
            "k_weight",
            "clamp_value",
        ):
            if not hasattr(module, name):
                raise TypeError(f"Engram module must expose {name}")
        if not callable(module.forward) or not callable(module.wkv) or not callable(module.embed):
            raise TypeError("Engram forward, projection, and embedding must be callable")
        if type(module.hc_mult) is not int or module.hc_mult <= 0:
            raise ValueError("Engram hc_mult must be a positive integer")
        if type(module.dim) is not int or module.dim <= 0:
            raise ValueError("Engram dim must be a positive integer")
        if getattr(module, _PATCH_MARKER, None) is not None:
            raise RuntimeError("chunked Engram forward is already installed")
        planned.append(module)
    if not planned:
        raise ValueError("transformer has no configured Engram modules")

    patches = []
    with _INSTALL_LOCK:
        # Another installer may have passed the optimistic preflight while waiting
        # for this lock. Recheck the whole set before replacing any forward method.
        for module in planned:
            if getattr(module, _PATCH_MARKER, None) is not None:
                raise RuntimeError("chunked Engram forward is already installed")
        for module in planned:
            original = module.forward
            installed = MethodType(_chunked_engram_forward(sequence_tile_rows), module)
            marker = object()
            setattr(module, _PATCH_MARKER, marker)
            try:
                module.forward = installed
            except BaseException:
                delattr(module, _PATCH_MARKER)
                for patch in reversed(patches):
                    patch.module.forward = patch.original
                    delattr(patch.module, _PATCH_MARKER)
                raise
            patches.append(_ForwardPatch(module, original, installed, marker))
    return ChunkedEngramForwardInstallation(patches)
