"""Native HIP dispatch for the pinned DeepSeek-V4.1-Flash-1017 graph.

This module intentionally implements only the released graph's UE8M0/FP4
formats. Unsupported formats raise instead of silently changing numerics.
"""

from __future__ import annotations

import ctypes
import math
import os
import operator
import threading
from pathlib import Path
from typing import Optional

import torch


REFERENCE_COMMIT = "1bae0e1088993b9b82de08f59de5be5a34e42170"
_INT_MAX = 2**31 - 1
_library_lock = threading.Lock()
_library_instance: ctypes.CDLL | None = None


def _e8m0_dtype():
    dtype = getattr(torch, "float8_e8m0fnu", None)
    if dtype is None:
        raise RuntimeError("the native DeepSeek bridge requires torch.float8_e8m0fnu")
    return dtype


def _fp4_dtype():
    dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if dtype is None:
        raise RuntimeError("the native DeepSeek bridge requires torch.float4_e2m1fn_x2")
    return dtype


def _library() -> ctypes.CDLL:
    global _library_instance
    if _library_instance is not None:
        return _library_instance
    with _library_lock:
        if _library_instance is not None:
            return _library_instance
        library_path = os.environ.get("DS4_HIP_LIBRARY")
        if not library_path:
            library_path = str(Path(__file__).resolve().parents[1] / "build" / "libredqueen_ds4_hip.so")
        if not Path(library_path).is_file():
            raise RuntimeError(
                "native DeepSeek HIP library not found; build hosts/redqueen/deepseek-hip/build.sh "
                "or set DS4_HIP_LIBRARY"
            )
        library = ctypes.CDLL(library_path)
        pointer = ctypes.c_void_p
        integer = ctypes.c_int
        floating = ctypes.c_float
        signatures = {
            "redqueen_ds4_quantize_activation": [pointer, pointer, pointer, integer, integer, integer, pointer],
            "redqueen_ds4_decode_e8m0": [pointer, pointer, integer, pointer],
            "redqueen_ds4_rmsnorm": [
                pointer,
                pointer,
                pointer,
                integer,
                integer,
                integer,
                integer,
                floating,
                pointer,
            ],
            "redqueen_ds4_rope_inplace": [
                pointer,
                pointer,
                integer,
                integer,
                integer,
                integer,
                integer,
                integer,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                pointer,
            ],
            "redqueen_ds4_fp4_quantize_activation": [
                pointer,
                pointer,
                pointer,
                integer,
                integer,
                integer,
                integer,
                integer,
                pointer,
            ],
            "redqueen_ds4_fp8_gemm": [
                pointer,
                pointer,
                pointer,
                pointer,
                pointer,
                integer,
                integer,
                integer,
                integer,
                integer,
                pointer,
            ],
            "redqueen_ds4_fp4_gemm": [
                pointer,
                pointer,
                pointer,
                pointer,
                pointer,
                integer,
                integer,
                integer,
                integer,
                integer,
                integer,
                pointer,
            ],
            "redqueen_ds4_sparse_attn": [
                pointer,
                pointer,
                pointer,
                pointer,
                pointer,
                integer,
                integer,
                integer,
                integer,
                integer,
                integer,
                floating,
                pointer,
            ],
            "redqueen_ds4_hc_split_sinkhorn": [
                pointer,
                pointer,
                pointer,
                pointer,
                pointer,
                pointer,
                integer,
                integer,
                integer,
                floating,
                pointer,
            ],
        }
        for name, argtypes in signatures.items():
            function = getattr(library, name)
            function.argtypes = argtypes
            function.restype = integer
        _library_instance = library
        return library


def _require_hip_tensor(tensor: torch.Tensor, name: str, dtype=None, device=None) -> torch.device:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if torch.version.hip is None or not tensor.is_cuda:
        raise RuntimeError(f"{name} must be a tensor on a ROCm device")
    if dtype is not None and tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if device is not None and tensor.device != device:
        raise ValueError(f"{name} is on {tensor.device}; expected {device}")
    return tensor.device


def _validate_same_device(first: torch.Tensor, **tensors: torch.Tensor) -> tuple[torch.device, torch.cuda.Stream]:
    device = _require_hip_tensor(first, "input")
    for name, tensor in tensors.items():
        _require_hip_tensor(tensor, name, device=device)
    stream = torch.cuda.current_stream(device)
    # Raw HIP launches bypass PyTorch's dispatcher; record the stream so the
    # caching allocator cannot recycle an input before its kernel completes.
    for tensor in (first, *tensors.values()):
        tensor.record_stream(stream)
    return device, stream


def _ptr(tensor: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(int(tensor.data_ptr()))


def _stream_ptr(stream: torch.cuda.Stream) -> ctypes.c_void_p:
    return ctypes.c_void_p(int(stream.cuda_stream))


def _check(status: int, operation: str) -> None:
    if status != 0:
        raise RuntimeError(f"{operation} failed with HIP status {status}")


def _c_int(value: int, name: str, *, positive: bool = True) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if value > _INT_MAX or value < (1 if positive else -_INT_MAX - 1):
        raise ValueError(f"{name} is outside the supported C integer range: {value}")
    return value


def _contiguous(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Apply the pinned model's RMSNorm formula through the native HIP kernel."""
    if x.ndim < 1 or weight.ndim != 1:
        raise ValueError("RMSNorm expects input rank >= 1 and a one-dimensional weight")
    device, stream = _validate_same_device(x, weight=weight)
    supported_dtypes = (torch.float32, torch.bfloat16)
    if x.dtype not in supported_dtypes or weight.dtype not in supported_dtypes:
        raise NotImplementedError("native RMSNorm supports float32 and bfloat16 input and weight")
    dim = _c_int(x.shape[-1], "RMSNorm dimension")
    rows = _c_int(x.numel() // dim if dim else 0, "RMSNorm row count")
    if dim == 0 or rows == 0 or weight.numel() != dim:
        raise ValueError("RMSNorm input and weight dimensions are incompatible or empty")
    if isinstance(eps, bool):
        raise TypeError("RMSNorm eps must be a finite nonnegative number")
    try:
        eps = float(eps)
    except (TypeError, ValueError) as exc:
        raise TypeError("RMSNorm eps must be a finite nonnegative number") from exc
    if not math.isfinite(eps) or eps < 0:
        raise ValueError("RMSNorm eps must be a finite nonnegative number")

    contiguous_x = _contiguous(x)
    contiguous_weight = _contiguous(weight)
    contiguous_x.record_stream(stream)
    contiguous_weight.record_stream(stream)
    output = torch.empty(x.shape, dtype=x.dtype, device=device)
    input_dtype = 0 if x.dtype == torch.float32 else 1
    weight_dtype = 0 if weight.dtype == torch.float32 else 1
    _check(
        _library().redqueen_ds4_rmsnorm(
            _ptr(contiguous_x),
            _ptr(contiguous_weight),
            _ptr(output),
            rows,
            dim,
            input_dtype,
            weight_dtype,
            eps,
            _stream_ptr(stream),
        ),
        "RMSNorm",
    )
    return output


def apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """Match the reference's in-place adjacent-pair RoPE operation."""
    if x.ndim not in (3, 4):
        raise ValueError("RoPE expects [batch, sequence, dim] or [batch, sequence, heads, dim]")
    device, stream = _validate_same_device(x, freqs_cis=freqs_cis)
    if x.dtype not in (torch.float32, torch.bfloat16):
        raise NotImplementedError("native RoPE supports float32 and bfloat16 input")
    if x.stride(-1) != 1:
        raise ValueError("native RoPE requires a contiguous rotary dimension")
    if freqs_cis.dtype != torch.complex64:
        raise TypeError("RoPE frequencies must be complex64")
    if not isinstance(inverse, bool):
        raise TypeError("inverse must be a bool")

    batch = _c_int(x.shape[0], "RoPE batch")
    sequence = _c_int(x.shape[1], "RoPE sequence")
    rotary_dim = _c_int(x.shape[-1], "RoPE dimension")
    heads = _c_int(x.shape[2] if x.ndim == 4 else 1, "RoPE heads")
    if rotary_dim % 2:
        raise ValueError("RoPE dimension must be even")
    if tuple(freqs_cis.shape) != (sequence, rotary_dim // 2):
        raise ValueError("RoPE frequencies must have shape [sequence, rotary_dim / 2]")

    frequencies = freqs_cis.resolve_conj()
    frequencies = _contiguous(frequencies)
    frequencies.record_stream(stream)
    if x.ndim == 3:
        batch_stride, sequence_stride, head_stride = x.stride(0), x.stride(1), 0
    else:
        batch_stride, sequence_stride, head_stride = x.stride(0), x.stride(1), x.stride(2)
    _check(
        _library().redqueen_ds4_rope_inplace(
            _ptr(x),
            _ptr(frequencies),
            batch,
            sequence,
            heads,
            rotary_dim,
            0 if x.dtype == torch.float32 else 1,
            int(inverse),
            batch_stride,
            sequence_stride,
            head_stride,
            _stream_ptr(stream),
        ),
        "RoPE",
    )
    return x


def _e8m0_to_float(scales: torch.Tensor) -> torch.Tensor:
    count = _c_int(scales.numel(), "scale count")
    output = torch.empty(scales.shape, dtype=torch.float32, device=scales.device)
    stream = torch.cuda.current_stream(scales.device)
    scales.record_stream(stream)
    _check(
        _library().redqueen_ds4_decode_e8m0(_ptr(scales), _ptr(output), count, _stream_ptr(stream)),
        "E8M0 scale decode",
    )
    return output


def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: Optional[str] = None,
    scale_dtype: torch.dtype = torch.float32,
    inplace: bool = False,
):
    """Match the 1017 activation quantizer for UE8M0 scales."""
    device = _require_hip_tensor(x, "x", dtype=torch.bfloat16)
    e8m0 = _e8m0_dtype()
    if scale_fmt != "ue8m0" or scale_dtype != e8m0:
        raise NotImplementedError("1017 native act_quant supports only scale_fmt='ue8m0' with E8M0 scales")
    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")
    block_size = _c_int(block_size, "block_size")
    if block_size not in (32, 128):
        raise ValueError("native act_quant supports block_size 32 or 128")
    cols = _c_int(x.shape[-1], "last dimension")
    if cols % block_size:
        raise ValueError("the last dimension must be divisible by block_size")

    # Keep the original allocation alive on this stream while contiguous()
    # may asynchronously copy from it. Recording only z is insufficient when
    # x is noncontiguous and the caller releases its reference immediately.
    stream = torch.cuda.current_stream(device)
    x.record_stream(stream)
    z = _contiguous(x)
    rows = _c_int(z.numel() // cols, "row count")
    if rows == 0:
        raise ValueError("empty activation tensors are not supported")
    output = torch.empty(z.shape, dtype=torch.float8_e4m3fn, device=device)
    scales = torch.empty((*z.shape[:-1], cols // block_size), dtype=e8m0, device=device)
    z.record_stream(stream)
    _check(
        _library().redqueen_ds4_quantize_activation(
            _ptr(z), _ptr(output), _ptr(scales), rows, cols, block_size, _stream_ptr(stream)
        ),
        "FP8 activation quantization",
    )
    if not inplace:
        return output, scales

    # The official inplace path dequantizes the quantized FP8 values back to
    # the input dtype before copying them into x.
    decoded_scales = _e8m0_to_float(scales).reshape(rows, -1).repeat_interleave(block_size, dim=-1)
    dequantized = (output.float().reshape(rows, cols) * decoded_scales).to(dtype=x.dtype).reshape(z.shape)
    x.copy_(dequantized)
    return x


def fp4_act_quant(
    x: torch.Tensor,
    block_size: int = 32,
    inplace: bool = False,
    scale_dtype: torch.dtype = torch.float8_e8m0fnu,
):
    """Match the 1017 packed FP4 activation quantizer."""
    device = _require_hip_tensor(x, "x", dtype=torch.bfloat16)
    e8m0 = _e8m0_dtype()
    e4m3 = torch.float8_e4m3fn
    if scale_dtype not in (e8m0, e4m3):
        raise NotImplementedError("native fp4_act_quant supports E8M0 or E4M3 scales")
    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")
    block_size = _c_int(block_size, "block_size")
    if block_size not in (16, 32):
        raise ValueError("native fp4_act_quant supports block_size 16 or 32")
    cols = _c_int(x.shape[-1], "last dimension")
    if cols % block_size:
        raise ValueError("the last dimension must be divisible by block_size")

    # contiguous() can enqueue a read from x on this stream. Retain x even if
    # the caller drops it after this wrapper returns.
    stream = torch.cuda.current_stream(device)
    x.record_stream(stream)
    z = _contiguous(x)
    rows = _c_int(z.numel() // cols, "row count")
    if rows == 0:
        raise ValueError("empty activation tensors are not supported")
    if inplace:
        output = z
    else:
        output = torch.empty((*z.shape[:-1], cols // 2), dtype=_fp4_dtype(), device=device)
    scales = torch.empty((*z.shape[:-1], cols // block_size), dtype=scale_dtype, device=device)
    z.record_stream(stream)
    scale_type = 0 if scale_dtype == e8m0 else 1
    _check(
        _library().redqueen_ds4_fp4_quantize_activation(
            _ptr(z),
            _ptr(output),
            _ptr(scales),
            rows,
            cols,
            block_size,
            scale_type,
            int(inplace),
            _stream_ptr(stream),
        ),
        "FP4 activation quantization",
    )
    if inplace:
        if output.data_ptr() != x.data_ptr():
            x.copy_(output)
        return x
    return output, scales


def _default_output_type() -> tuple[torch.dtype, int]:
    output_dtype = torch.get_default_dtype()
    if output_dtype == torch.float32:
        return output_dtype, 0
    if output_dtype == torch.bfloat16:
        return output_dtype, 1
    raise NotImplementedError(f"native GEMM supports float32 or bfloat16 output, not {output_dtype}")


def fp8_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
    block_size: int = 128,
) -> torch.Tensor:
    """Compute the pinned grouped FP8 GEMM using native HIP."""
    e8m0 = _e8m0_dtype()
    if scale_dtype != e8m0:
        raise NotImplementedError("native fp8_gemm supports only E8M0 scales")
    _require_hip_tensor(a, "a")
    _require_hip_tensor(b, "b")
    if a.ndim < 1 or b.ndim != 2:
        raise ValueError("a must have rank >= 1 and b must have rank 2")
    device, stream = _validate_same_device(a, a_s=a_s, b=b, b_s=b_s)
    _require_hip_tensor(a, "a", dtype=torch.float8_e4m3fn, device=device)
    _require_hip_tensor(b, "b", dtype=torch.float8_e4m3fn, device=device)
    _require_hip_tensor(a_s, "a_s", dtype=e8m0, device=device)
    _require_hip_tensor(b_s, "b_s", dtype=e8m0, device=device)
    if not all(t.is_contiguous() for t in (a, a_s, b, b_s)):
        raise ValueError("GEMM inputs and scales must be contiguous")
    block_size = _c_int(block_size, "block_size")
    if block_size not in (32, 128):
        raise ValueError("native fp8_gemm supports block_size 32 or 128")
    k = _c_int(a.shape[-1], "K")
    n = _c_int(b.shape[0], "N")
    m = _c_int(a.numel() // k, "M")
    if k % block_size or b.shape[1] != k or m == 0 or n == 0:
        raise ValueError("GEMM dimensions are incompatible or empty")
    if a_s.numel() != m * (k // block_size):
        raise ValueError("a_s has an incompatible shape")
    if tuple(b_s.shape) != ((n + block_size - 1) // block_size, k // block_size):
        raise ValueError("b_s has an incompatible shape")

    output_dtype, output_type = _default_output_type()
    output = torch.empty((*a.shape[:-1], n), dtype=output_dtype, device=device)
    _check(
        _library().redqueen_ds4_fp8_gemm(
            _ptr(a), _ptr(a_s), _ptr(b), _ptr(b_s), _ptr(output), m, n, k, block_size, output_type, _stream_ptr(stream)
        ),
        "FP8 GEMM",
    )
    return output


def fp4_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
    act_block_size: int = 128,
) -> torch.Tensor:
    """Compute the pinned FP8-by-packed-FP4 GEMM using native HIP."""
    e8m0 = _e8m0_dtype()
    if scale_dtype == e8m0:
        scale_type = 0
    elif scale_dtype == torch.float32:
        scale_type = 1
    else:
        raise NotImplementedError("native fp4_gemm supports E8M0 or float32 scales")
    _require_hip_tensor(a, "a")
    _require_hip_tensor(b, "b")
    if a.ndim < 1 or b.ndim != 2:
        raise ValueError("a must have rank >= 1 and b must have rank 2")
    device, stream = _validate_same_device(a, a_s=a_s, b=b, b_s=b_s)
    _require_hip_tensor(a, "a", dtype=torch.float8_e4m3fn, device=device)
    _require_hip_tensor(b, "b", dtype=_fp4_dtype(), device=device)
    _require_hip_tensor(a_s, "a_s", dtype=scale_dtype, device=device)
    _require_hip_tensor(b_s, "b_s", dtype=scale_dtype, device=device)
    if not all(t.is_contiguous() for t in (a, a_s, b, b_s)):
        raise ValueError("GEMM inputs and scales must be contiguous")
    act_block_size = _c_int(act_block_size, "act_block_size")
    if act_block_size not in (32, 128):
        raise ValueError("native fp4_gemm supports act_block_size 32 or 128")
    k = _c_int(a.shape[-1], "K")
    n = _c_int(b.shape[0], "N")
    m = _c_int(a.numel() // k, "M")
    if k % act_block_size or k % 32 or b.shape[1] * 2 != k or m == 0 or n == 0:
        raise ValueError("GEMM dimensions are incompatible or empty")
    if a_s.numel() != m * (k // act_block_size):
        raise ValueError("a_s has an incompatible shape")
    if tuple(b_s.shape) != (n, k // 32):
        raise ValueError("b_s has an incompatible shape")

    output_dtype, output_type = _default_output_type()
    output = torch.empty((*a.shape[:-1], n), dtype=output_dtype, device=device)
    _check(
        _library().redqueen_ds4_fp4_gemm(
            _ptr(a),
            _ptr(a_s),
            _ptr(b),
            _ptr(b_s),
            _ptr(output),
            m,
            n,
            k,
            act_block_size,
            scale_type,
            output_type,
            _stream_ptr(stream),
        ),
        "FP4 GEMM",
    )
    return output


def sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Run the official sparse-attention operation on the current HIP stream."""
    _require_hip_tensor(q, "q")
    _require_hip_tensor(kv, "kv")
    _require_hip_tensor(attn_sink, "attn_sink")
    _require_hip_tensor(topk_idxs, "topk_idxs")
    if q.ndim != 4 or kv.ndim != 3 or topk_idxs.ndim != 3 or attn_sink.ndim != 1:
        raise ValueError("expected q[B,Q,H,D], kv[B,K,D], sink[H], and indices[B,Q,T]")
    device, stream = _validate_same_device(q, kv=kv, attn_sink=attn_sink, topk_idxs=topk_idxs)
    _require_hip_tensor(q, "q", dtype=torch.bfloat16, device=device)
    _require_hip_tensor(kv, "kv", dtype=torch.bfloat16, device=device)
    _require_hip_tensor(attn_sink, "attn_sink", dtype=torch.float32, device=device)
    _require_hip_tensor(topk_idxs, "topk_idxs", dtype=torch.int32, device=device)
    batch, queries, heads, dim = map(int, q.shape)
    kv_batch, keys, kv_dim = map(int, kv.shape)
    if kv_batch != batch or kv_dim != dim or attn_sink.numel() != heads:
        raise ValueError("sparse-attention tensor dimensions are incompatible")
    if tuple(topk_idxs.shape[:2]) != (batch, queries):
        raise ValueError("topk_idxs has incompatible batch/query dimensions")
    scale = float(softmax_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("softmax_scale must be finite and positive")
    batch, queries, heads, dim, keys, topk = (
        _c_int(value, name)
        for value, name in (
            (batch, "batch"),
            (queries, "queries"),
            (heads, "heads"),
            (dim, "dim"),
            (keys, "keys"),
            (topk_idxs.shape[-1], "topk"),
        )
    )
    q_c, kv_c = _contiguous(q), _contiguous(kv)
    sink_c, indices_c = _contiguous(attn_sink), _contiguous(topk_idxs)
    output = torch.empty(q.shape, dtype=torch.bfloat16, device=device)
    _check(
        _library().redqueen_ds4_sparse_attn(
            _ptr(q_c),
            _ptr(kv_c),
            _ptr(sink_c),
            _ptr(indices_c),
            _ptr(output),
            batch,
            queries,
            heads,
            dim,
            keys,
            topk,
            ctypes.c_float(scale),
            _stream_ptr(stream),
        ),
        "sparse attention",
    )
    return output


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    """Run the pinned Hyper-Connections split and Sinkhorn operation."""
    if mixes.ndim != 3:
        raise ValueError("mixes must have shape [batch, sequence, (2 + hc_mult) * hc_mult]")
    device, stream = _validate_same_device(mixes, hc_scale=hc_scale, hc_base=hc_base)
    _require_hip_tensor(mixes, "mixes", dtype=torch.float32, device=device)
    _require_hip_tensor(hc_scale, "hc_scale", dtype=torch.float32, device=device)
    _require_hip_tensor(hc_base, "hc_base", dtype=torch.float32, device=device)
    hc_mult = _c_int(hc_mult, "hc_mult")
    sinkhorn_iters = _c_int(sinkhorn_iters, "sinkhorn_iters")
    if hc_mult > 16 or hc_scale.numel() != 3 or hc_base.numel() != (2 + hc_mult) * hc_mult:
        raise ValueError("hc_scale or hc_base has an incompatible size")
    if mixes.shape[-1] != (2 + hc_mult) * hc_mult:
        raise ValueError("mixes has an incompatible final dimension")
    eps = float(eps)
    if not math.isfinite(eps) or eps < 0:
        raise ValueError("eps must be finite and nonnegative")
    rows = _c_int(mixes.shape[0] * mixes.shape[1], "row count")
    if rows == 0:
        raise ValueError("empty mixes tensors are not supported")

    mixes_c, scale_c, base_c = _contiguous(mixes), _contiguous(hc_scale), _contiguous(hc_base)
    pre = torch.empty((*mixes.shape[:2], hc_mult), dtype=torch.float32, device=device)
    post = torch.empty_like(pre)
    comb = torch.empty((*mixes.shape[:2], hc_mult, hc_mult), dtype=torch.float32, device=device)
    _check(
        _library().redqueen_ds4_hc_split_sinkhorn(
            _ptr(mixes_c),
            _ptr(scale_c),
            _ptr(base_c),
            _ptr(pre),
            _ptr(post),
            _ptr(comb),
            rows,
            hc_mult,
            sinkhorn_iters,
            ctypes.c_float(eps),
            _stream_ptr(stream),
        ),
        "Hyper-Connections Sinkhorn",
    )
    return pre, post, comb
