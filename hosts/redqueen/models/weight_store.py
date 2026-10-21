#!/usr/bin/env python3
"""Build a source-backed, zero-copy tensor catalog for Redqueen model files."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import struct
import sys
import tempfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import BinaryIO


FORMAT = "redqueen.weight-store"
VERSION = 1
MAX_HEADER_BYTES = 256 * 1024 * 1024
MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024
MAX_STRING_BYTES = 64 * 1024 * 1024
MAX_ITEMS = 2_000_000
MAX_TENSOR_COUNT = 250_000
MAX_METADATA_COUNT = 250_000
MAX_TENSOR_DIMS = 64
MAX_SAFE_INTEGER = (1 << 48) - 1
ROOT = Path(__file__).resolve().parents[3]

# ggml block element and byte counts. Values follow ggml_type and the packed
# block declarations in upstream ggml; removed GGUF types intentionally fail.
GGML_BLOCKS: dict[int, tuple[str, int, int]] = {
    0: ("f32", 1, 4),
    1: ("f16", 1, 2),
    2: ("q4_0", 32, 18),
    3: ("q4_1", 32, 20),
    6: ("q5_0", 32, 22),
    7: ("q5_1", 32, 24),
    8: ("q8_0", 32, 34),
    9: ("q8_1", 32, 36),
    10: ("q2_k", 256, 84),
    11: ("q3_k", 256, 110),
    12: ("q4_k", 256, 144),
    13: ("q5_k", 256, 176),
    14: ("q6_k", 256, 210),
    15: ("q8_k", 256, 292),
    16: ("iq2_xxs", 256, 66),
    17: ("iq2_xs", 256, 74),
    18: ("iq3_xxs", 256, 98),
    19: ("iq1_s", 256, 50),
    20: ("iq4_nl", 32, 18),
    21: ("iq3_s", 256, 110),
    22: ("iq2_s", 256, 82),
    23: ("iq4_xs", 256, 136),
    24: ("i8", 1, 1),
    25: ("i16", 1, 2),
    26: ("i32", 1, 4),
    27: ("i64", 1, 8),
    28: ("f64", 1, 8),
    29: ("iq1_m", 256, 56),
    30: ("bf16", 1, 2),
    34: ("tq1_0", 256, 54),
    35: ("tq2_0", 256, 66),
    39: ("mxfp4", 32, 17),
    40: ("nvfp4", 64, 36),
    41: ("q1_0", 128, 18),
    42: ("q2_0", 64, 18),
}

GGUF_VALUE_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
SAFETENSORS_DTYPE_BITS = {
    "BOOL": 8,
    "U8": 8,
    "I8": 8,
    "F8_E4M3": 8,
    "F8_E5M2": 8,
    "F8_E5M2FNUZ": 8,
    "F8_E4M3FNUZ": 8,
    "F8_E8M0": 8,
    "F6_E2M3": 6,
    "F6_E3M2": 6,
    "U16": 16,
    "I16": 16,
    "F16": 16,
    "BF16": 16,
    "U32": 32,
    "I32": 32,
    "F32": 32,
    "U64": 64,
    "I64": 64,
    "F64": 64,
    "C64": 64,
    "F4": 4,
}


def _safe_posix_path(value: str, *, label: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or path.as_posix() != value
        or "\\" in value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"unsafe {label}: {value!r}")
    return path


class _GGUFReader:
    def __init__(self, stream: BinaryIO, file_size: int):
        self.stream = stream
        self.file_size = file_size

    def tell(self) -> int:
        return self.stream.tell()

    def read(self, size: int) -> bytes:
        if size < 0 or size > MAX_HEADER_BYTES or self.tell() + size > MAX_HEADER_BYTES:
            raise ValueError("GGUF header exceeds the configured size limit")
        data = self.stream.read(size)
        if len(data) != size:
            raise ValueError("truncated GGUF header")
        return data

    def string(self) -> str:
        length = struct.unpack("<Q", self.read(8))[0]
        if length > MAX_STRING_BYTES:
            raise ValueError("GGUF string exceeds the configured size limit")
        try:
            return self.read(length).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("GGUF string is not valid UTF-8") from error

    def skip(self, size: int) -> None:
        target = self.tell() + size
        if size < 0 or target > MAX_HEADER_BYTES or target > self.file_size:
            raise ValueError("invalid or oversized GGUF header value")
        self.stream.seek(target)

    def skip_string(self) -> None:
        length = struct.unpack("<Q", self.read(8))[0]
        self.skip(length)


def _gguf_scalar(reader: _GGUFReader, type_id: int):
    formats = {
        0: "<B",
        1: "<b",
        2: "<H",
        3: "<h",
        4: "<I",
        5: "<i",
        6: "<f",
        7: "<?",
        10: "<Q",
        11: "<q",
        12: "<d",
    }
    if type_id == 8:
        return reader.string()
    fmt = formats.get(type_id)
    if fmt is None:
        raise ValueError(f"unsupported GGUF metadata value type {type_id}")
    return struct.unpack(fmt, reader.read(struct.calcsize(fmt)))[0]


def _skip_gguf_value(reader: _GGUFReader, type_id: int) -> None:
    if type_id == 8:
        reader.skip_string()
        return
    if type_id != 9:
        size = GGUF_VALUE_SIZES.get(type_id)
        if size is None:
            raise ValueError(f"unsupported GGUF metadata value type {type_id}")
        reader.skip(size)
        return
    item_type = struct.unpack("<I", reader.read(4))[0]
    item_count = struct.unpack("<Q", reader.read(8))[0]
    if item_type == 9 or item_type not in (*GGUF_VALUE_SIZES, 8) or item_count > MAX_ITEMS:
        raise ValueError("invalid or oversized GGUF metadata array")
    if item_type == 8:
        for _ in range(item_count):
            reader.skip_string()
    else:
        reader.skip(item_count * GGUF_VALUE_SIZES[item_type])


def _gguf_tensor_storage(type_id: int, shape: list[int]) -> tuple[str, int, int]:
    info = GGML_BLOCKS.get(type_id)
    if info is None:
        raise ValueError(f"unsupported or removed GGML tensor type {type_id}")
    type_name, block_elements, block_bytes = info
    if not shape or any(dimension <= 0 for dimension in shape):
        raise ValueError("GGUF tensor dimensions must be positive")
    row_elements = shape[0]
    if row_elements % block_elements:
        raise ValueError(f"GGUF tensor row is not divisible by the {type_name} block size")
    row_bytes = (row_elements // block_elements) * block_bytes
    return type_name, block_elements, row_bytes * math.prod(shape[1:])


def parse_gguf(path: Path) -> dict:
    """Read GGUF tensor descriptors without reading or rewriting tensor data."""
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        reader = _GGUFReader(stream, file_size)
        if reader.read(4) != b"GGUF":
            raise ValueError(f"not a GGUF file: {path}")
        version = struct.unpack("<I", reader.read(4))[0]
        if version not in {2, 3}:
            raise ValueError(f"unsupported GGUF version {version}")
        tensor_count, metadata_count = struct.unpack("<QQ", reader.read(16))
        if tensor_count > MAX_TENSOR_COUNT or metadata_count > MAX_METADATA_COUNT:
            raise ValueError("GGUF tensor or metadata count exceeds the configured limit")

        metadata: dict[str, object] = {}
        metadata_keys = set()
        relevant = {"general.architecture", "general.alignment", "general.file_type"}
        for _ in range(metadata_count):
            key = reader.string()
            if key in metadata_keys:
                raise ValueError(f"duplicate GGUF metadata key {key!r}")
            metadata_keys.add(key)
            type_id = struct.unpack("<I", reader.read(4))[0]
            if key in relevant and type_id != 9:
                metadata[key] = _gguf_scalar(reader, type_id)
            elif key in relevant:
                raise ValueError(f"GGUF metadata {key!r} must be a scalar")
            else:
                _skip_gguf_value(reader, type_id)

        architecture = metadata.get("general.architecture")
        if architecture is not None and not isinstance(architecture, str):
            raise ValueError("GGUF general.architecture must be a string")
        alignment_value = metadata.get("general.alignment", 32)
        if isinstance(alignment_value, bool) or not isinstance(alignment_value, int):
            raise ValueError("GGUF general.alignment must be an integer")
        alignment = alignment_value
        if alignment < 1 or alignment > 1024 * 1024 or alignment & (alignment - 1):
            raise ValueError("GGUF general.alignment must be a bounded power of two")
        file_type = metadata.get("general.file_type")
        if file_type is not None and (isinstance(file_type, bool) or not isinstance(file_type, int)):
            raise ValueError("GGUF general.file_type must be an integer")

        tensors = []
        names = set()
        for _ in range(tensor_count):
            name = reader.string()
            if not name:
                raise ValueError("GGUF tensor names must not be empty")
            if name in names:
                raise ValueError(f"duplicate GGUF tensor name {name!r}")
            names.add(name)
            dimensions = struct.unpack("<I", reader.read(4))[0]
            if dimensions < 1 or dimensions > 16:
                raise ValueError(f"invalid GGUF dimension count for tensor {name!r}")
            shape = list(struct.unpack("<" + "Q" * dimensions, reader.read(8 * dimensions)))
            type_id = struct.unpack("<I", reader.read(4))[0]
            relative_offset = struct.unpack("<Q", reader.read(8))[0]
            type_name, block_elements, storage_bytes = _gguf_tensor_storage(type_id, shape)
            tensors.append(
                {
                    "name": name,
                    "shape": shape,
                    "ggml_type_id": type_id,
                    "ggml_type": type_name,
                    "block_elements": block_elements,
                    "storage_span_bytes": storage_bytes,
                    "relative_data_offset": relative_offset,
                }
            )

        header_end = reader.tell()
        aligned_data_start = ((header_end + alignment - 1) // alignment) * alignment
        if tensor_count:
            data_start = aligned_data_start
            if data_start > file_size:
                raise ValueError("GGUF aligned tensor data section starts past end of file")
        else:
            if file_size > aligned_data_start:
                raise ValueError("GGUF shard with no tensors contains unindexed trailing data")
            stream.seek(header_end)
            trailing_padding = stream.read(file_size - header_end)
            if any(trailing_padding):
                raise ValueError("GGUF metadata-only shard has nonzero alignment padding")
            data_start = aligned_data_start if file_size == aligned_data_start else header_end

    occupied = []
    for tensor in tensors:
        relative_offset = tensor.pop("relative_data_offset")
        absolute_offset = data_start + relative_offset
        end = absolute_offset + tensor["storage_span_bytes"]
        if absolute_offset < data_start or end > file_size or absolute_offset % alignment:
            raise ValueError(f"GGUF tensor {tensor['name']!r} has an invalid data offset or span")
        tensor["data_offset"] = absolute_offset
        tensor["data_offset_in_section"] = relative_offset
        occupied.append((absolute_offset, end, tensor["name"]))
    occupied.sort()
    for previous, current in zip(occupied, occupied[1:]):
        if previous[1] > current[0]:
            raise ValueError(f"GGUF tensors overlap: {previous[2]!r} and {current[2]!r}")

    return {
        "container": "gguf",
        "version": version,
        "architecture": architecture,
        "file_type": file_type,
        "alignment": alignment,
        "tensor_layout": "ggml-native",
        "dimension_order": "ggml",
        "data_start": data_start,
        "tensors": tensors,
    }


def _safetensors_expected_bytes(dtype: str, shape: list[int]) -> int:
    bits = SAFETENSORS_DTYPE_BITS.get(dtype)
    if bits is None:
        raise ValueError(f"unsupported SafeTensors dtype {dtype!r}")
    total_bits = math.prod(shape) * bits
    if total_bits % 8:
        raise ValueError(f"SafeTensors tensor dtype {dtype!r} and shape are not byte aligned")
    return total_bits // 8


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate SafeTensors JSON key {key!r}")
        result[key] = value
    return result


def parse_safetensors(path: Path) -> dict:
    """Read SafeTensors descriptors and verify every range against the file."""
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated SafeTensors length prefix")
        header_length = struct.unpack("<Q", prefix)[0]
        if header_length <= 0 or header_length > MAX_SAFETENSORS_HEADER_BYTES:
            raise ValueError("SafeTensors header length is invalid or exceeds the configured limit")
        data_start = 8 + header_length
        if data_start > file_size:
            raise ValueError("SafeTensors header extends past end of file")
        header_bytes = stream.read(header_length)
        if len(header_bytes) != header_length or not header_bytes.startswith(b"{"):
            raise ValueError("SafeTensors header must begin with a JSON object")
        try:
            header = json.loads(header_bytes, object_pairs_hook=_unique_json_object)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError("SafeTensors header is not valid JSON") from error
    if not isinstance(header, dict):
        raise ValueError("SafeTensors header must be a JSON object")
    if len(header) > MAX_TENSOR_COUNT:
        raise ValueError("SafeTensors tensor count exceeds the configured limit")
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items()):
        raise ValueError("SafeTensors __metadata__ must contain string keys and values")

    tensors = []
    intervals = []
    for name, item in header.items():
        if not isinstance(name, str) or not name or not isinstance(item, dict):
            raise ValueError("invalid SafeTensors tensor entry")
        if set(item) != {"dtype", "shape", "data_offsets"}:
            raise ValueError(f"SafeTensors descriptor for {name!r} has unknown or missing fields")
        dtype = item.get("dtype")
        shape = item.get("shape")
        offsets = item.get("data_offsets")
        if not isinstance(dtype, str) or not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"invalid SafeTensors descriptor for {name!r}")
        if len(shape) > MAX_TENSOR_DIMS or any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 0
            or dimension > MAX_SAFE_INTEGER
            for dimension in shape
        ):
            raise ValueError(f"invalid SafeTensors shape for {name!r}")
        if any(
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or offset < 0
            or offset > MAX_SAFE_INTEGER
            for offset in offsets
        ):
            raise ValueError(f"invalid SafeTensors offsets for {name!r}")
        start, end = offsets
        if start < 0 or end < start or end > file_size - data_start:
            raise ValueError(f"SafeTensors tensor {name!r} has offsets outside the payload")
        data_bytes = end - start
        expected_bytes = _safetensors_expected_bytes(dtype, shape)
        if data_bytes != expected_bytes:
            raise ValueError(f"SafeTensors tensor {name!r} byte span does not match dtype and shape")
        intervals.append((start, end, name))
        tensors.append(
            {
                "name": name,
                "dtype": dtype,
                "shape": shape,
                "data_offset": data_start + start,
                "data_bytes": data_bytes,
            }
        )
    intervals.sort()
    expected_start = 0
    for start, end, name in intervals:
        if start < expected_start:
            raise ValueError(f"SafeTensors tensor {name!r} overlaps a prior tensor")
        if start > expected_start:
            raise ValueError(f"SafeTensors tensor data contains a hole before {name!r}")
        expected_start = end
    if expected_start != file_size - data_start:
        raise ValueError("SafeTensors tensor metadata does not cover the complete data buffer")
    tensors.sort(key=lambda tensor: tensor["name"])
    return {
        "container": "safetensors",
        "tensor_layout": "safetensors-native",
        "dimension_order": "row-major",
        "header_bytes": header_length,
        "data_start": data_start,
        "metadata": metadata,
        "tensors": tensors,
    }


def _read_sha256_manifest(path: Path) -> dict[str, str]:
    if path.stat().st_size > MAX_HEADER_BYTES:
        raise ValueError(f"SHA256SUMS file exceeds the configured size limit: {path}")
    checksums = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        if len(line) < 67 or not re.fullmatch(r"[0-9a-fA-F]{64}", line[:64]) or line[64:66] not in {"  ", " *"}:
            raise ValueError(f"invalid SHA256SUMS entry at {path}:{line_number}")
        relative = line[66:]
        _safe_posix_path(relative, label="checksum path")
        if relative in checksums:
            raise ValueError(f"duplicate checksum path {relative!r} in {path}")
        checksums[relative] = line[:64].lower()
    return checksums


def _read_source_revision(path: Path) -> tuple[dict[str, str], str | None]:
    values = {}
    try:
        with path.open("rb") as stream:
            raw = stream.read(1024 * 1024 + 1)
    except FileNotFoundError:
        return values, None
    if len(raw) > 1024 * 1024:
        raise ValueError(f"SOURCE_REVISION exceeds the configured size limit: {path}")
    lines = raw.decode("utf-8").splitlines()
    for line in lines:
        key, separator, value = line.partition("=")
        if separator and key and key not in values:
            values[key] = value
    return dict(sorted(values.items())), hashlib.sha256(raw).hexdigest()


def _artifact_source(
    relative_path: PurePosixPath,
    metadata_root: Path,
    ref: dict,
    actual_size: int,
) -> tuple[str, dict]:
    metadata_name = relative_path.parts[0]
    metadata_dir = metadata_root / metadata_name
    try:
        metadata_dir = metadata_dir.resolve(strict=True)
        metadata_dir.relative_to(metadata_root)
    except FileNotFoundError:
        metadata_dir = metadata_root / metadata_name
    except ValueError as error:
        raise ValueError(f"model metadata directory resolves outside metadata root: {metadata_name}") from error
    checksums = {}
    checksum_path = metadata_dir / "SHA256SUMS"
    if checksum_path.is_file():
        try:
            resolved_checksum = checksum_path.resolve(strict=True)
            resolved_checksum.relative_to(metadata_root)
        except ValueError as error:
            raise ValueError(f"SHA256SUMS resolves outside metadata root: {checksum_path}") from error
        checksums = _read_sha256_manifest(resolved_checksum)
    artifact_relative = PurePosixPath(*relative_path.parts[1:]).as_posix()
    digest_from_metadata = checksums.get(artifact_relative)
    digest_from_inventory = ref.get("expected_sha256")
    if digest_from_inventory is not None:
        if not isinstance(digest_from_inventory, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest_from_inventory):
            raise ValueError(f"invalid pinned SHA-256 for {relative_path}")
        digest_from_inventory = digest_from_inventory.lower()
    if digest_from_metadata and digest_from_inventory and digest_from_metadata != digest_from_inventory:
        raise ValueError(f"tracked metadata and artifact inventory disagree for {relative_path}")
    expected_digest = digest_from_metadata or digest_from_inventory
    if expected_digest is None:
        raise ValueError(f"no pinned SHA-256 for model artifact {relative_path}")

    expected_size = ref.get("expected_size_bytes")
    if expected_size is not None and (
        isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size != actual_size
    ):
        raise ValueError(f"model artifact size disagrees with its inventory: {relative_path}")

    source_revision_path = metadata_dir / "SOURCE_REVISION"
    if source_revision_path.is_file():
        try:
            source_revision_path = source_revision_path.resolve(strict=True)
            source_revision_path.relative_to(metadata_root)
        except ValueError as error:
            raise ValueError(f"SOURCE_REVISION resolves outside metadata root: {metadata_dir}") from error
    source_revision, revision_sha256 = _read_source_revision(source_revision_path)
    provenance = {
        "metadata_directory": metadata_name if metadata_dir.is_dir() else None,
        "source_revision": source_revision or None,
        "source_revision_sha256": revision_sha256,
        "source_object_id": ref.get("source_object_id"),
        "digest_source": "metadata/SHA256SUMS" if digest_from_metadata else "artifact_inventory",
    }
    return expected_digest, provenance


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_store(
    artifact_refs: dict[str, dict],
    *,
    model_root: Path,
    metadata_root: Path,
    verify_hashes: bool = False,
    model_filter: set[str] | None = None,
) -> dict:
    model_root_input = Path(os.path.abspath(model_root))
    model_root = model_root_input.resolve(strict=True)
    metadata_root = metadata_root.resolve(strict=True)
    if not model_root.is_dir() or not metadata_root.is_dir():
        raise ValueError("model root and metadata root must be directories")

    filtered_refs = {}
    configured_models = set()
    for raw_path, item in artifact_refs.items():
        if not isinstance(raw_path, str) or not raw_path or not isinstance(item, dict):
            raise ValueError("invalid artifact reference entry")
        references = item.get("references")
        if not isinstance(references, list):
            raise ValueError(f"invalid model references for {raw_path}")
        for reference in references:
            if (
                not isinstance(reference, dict)
                or not isinstance(reference.get("model"), str)
                or not reference["model"]
            ):
                raise ValueError(f"invalid model reference for {raw_path}")
            configured_models.add(reference["model"])
        selected = [
            reference
            for reference in references
            if model_filter is None or reference["model"] in model_filter
        ]
        if selected:
            if item.get("path", raw_path) != raw_path:
                raise ValueError(f"artifact reference key and path disagree: {raw_path}")
            filtered_refs[raw_path] = {**item, "references": selected}
    if model_filter is not None:
        unknown_models = model_filter - configured_models
        if unknown_models:
            raise ValueError(f"unknown configured model or service id: {', '.join(sorted(unknown_models))}")
    if not filtered_refs:
        raise ValueError("no model artifacts selected for the weight catalog")

    artifacts = []
    model_refs: dict[str, list[dict]] = defaultdict(list)
    total_size = 0
    tensor_count = 0
    for raw_path, item in sorted(filtered_refs.items()):
        path = Path(raw_path)
        if not path.is_absolute():
            path = model_root_input / path
        path = Path(os.path.abspath(path))
        try:
            relative = PurePosixPath(path.relative_to(model_root_input).as_posix())
        except ValueError as error:
            raise ValueError(f"artifact path is outside the model root: {raw_path}") from error
        _safe_posix_path(relative.as_posix(), label="artifact path")
        resolved = path.resolve(strict=True)
        try:
            resolved.relative_to(model_root)
        except ValueError as error:
            raise ValueError(f"artifact resolves outside the model root: {raw_path}") from error
        if not resolved.is_file():
            raise ValueError(f"model artifact is not a regular file: {raw_path}")
        stat_result = resolved.stat()
        size_bytes = stat_result.st_size
        if item.get("expected_size_bytes") is not None and item["expected_size_bytes"] != size_bytes:
            raise ValueError(f"model artifact size disagrees with its inventory: {relative}")
        expected_digest, provenance = _artifact_source(relative, metadata_root, item, size_bytes)
        verified = False
        if verify_hashes:
            actual_digest = sha256_file(resolved)
            if actual_digest != expected_digest:
                raise ValueError(f"model artifact SHA-256 does not match its pinned manifest: {relative}")
            verified = True

        suffix = resolved.suffix.lower()
        try:
            if suffix == ".gguf":
                layout = parse_gguf(resolved)
            elif suffix == ".safetensors":
                layout = parse_safetensors(resolved)
            else:
                layout = {"container": "opaque", "tensor_layout": "opaque", "tensors": []}
        except ValueError as error:
            raise ValueError(f"invalid {suffix or 'opaque'} artifact {relative}: {error}") from error
        tensors = layout["tensors"]
        total_size += size_bytes
        tensor_count += len(tensors)
        artifact = {
            "path": relative.as_posix(),
            "size_bytes": size_bytes,
            "container": layout["container"],
            "tensor_layout": layout["tensor_layout"],
            "integrity": {
                "sha256": expected_digest,
                "verified_at_build": verified,
                "verification_method": "full_file_sha256" if verified else "pinned_manifest_only",
            },
            "conversion": {"mode": "identity-import", "payload_changed": False},
            "provenance": provenance,
            "references": sorted(
                (dict(reference) for reference in item["references"]),
                key=lambda reference: (reference.get("model", ""), reference.get("role", ""), reference.get("class", "")),
            ),
            "tensors": tensors,
        }
        for key in (
            "version",
            "architecture",
            "file_type",
            "alignment",
            "dimension_order",
            "data_start",
            "header_bytes",
            "metadata",
        ):
            if key in layout:
                artifact[key] = layout[key]
        for reference in artifact["references"]:
            model_id = reference["model"]
            model_refs[model_id].append(
                {"artifact": relative.as_posix(), "role": reference.get("role", "unspecified"), **({"class": reference["class"]} if "class" in reference else {})}
            )
        final_stat = resolved.stat()
        before_fingerprint = (
            stat_result.st_dev,
            stat_result.st_ino,
            stat_result.st_size,
            stat_result.st_mtime_ns,
            stat_result.st_ctime_ns,
        )
        after_fingerprint = (
            final_stat.st_dev,
            final_stat.st_ino,
            final_stat.st_size,
            final_stat.st_mtime_ns,
            final_stat.st_ctime_ns,
        )
        if before_fingerprint != after_fingerprint:
            raise ValueError(f"model artifact changed while cataloging: {relative}")
        artifacts.append(artifact)

    models = []
    for model_id, references in sorted(model_refs.items()):
        classes = sorted({reference["class"] for reference in references if "class" in reference})
        models.append(
            {
                "id": model_id,
                "workload_classes": classes,
                "artifacts": sorted(references, key=lambda ref: (ref["artifact"], ref["role"], ref.get("class", ""))),
            }
        )
    return {
        "format": FORMAT,
        "version": VERSION,
        "storage_mode": "source-backed",
        "model_root": str(model_root),
        "payload_policy": "original artifact bytes remain at their source paths; this catalog stores descriptors and references only",
        "models": models,
        "artifacts": artifacts,
        "summary": {
            "artifact_count": len(artifacts),
            "model_count": len(models),
            "tensor_count": tensor_count,
            "artifact_bytes": total_size,
            "fully_verified_artifacts": sum(item["integrity"]["verified_at_build"] for item in artifacts),
        },
    }


def write_atomic(path: Path, catalog: dict) -> None:
    """Durably replace a JSON catalog without following an existing symlink."""
    path = path.absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(catalog, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temporary_path = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_path, 0o644)
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def _load_preflight() -> object:
    path = ROOT / "hosts/redqueen/model-router/inference-preflight.py"
    spec = importlib.util.spec_from_file_location("redqueen_inference_preflight", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load artifact inventory helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_artifact_refs(router_path: Path, scheduler_path: Path) -> dict[str, dict]:
    try:
        import yaml
    except ModuleNotFoundError as error:
        raise RuntimeError("PyYAML is required to read the llama-swap configuration") from error
    try:
        router = yaml.safe_load(router_path.read_text(encoding="utf-8"))
        policy = json.loads(scheduler_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"cannot parse router or scheduler configuration: {error}") from error
    if not isinstance(router, dict) or not isinstance(policy, dict):
        raise ValueError("router and scheduler configurations must contain objects")
    preflight = _load_preflight()
    return preflight.collect_artifact_refs(router, policy, ROOT)


def _ensure_output_outside_model_root(output: Path, model_root: Path) -> None:
    output = output.absolute()
    parent = output.parent.resolve()
    candidate = parent / output.name
    try:
        candidate.relative_to(model_root.resolve(strict=True))
    except ValueError:
        return
    raise ValueError("catalog output must be outside the model root")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-config", type=Path, default=ROOT / "hosts/redqueen/model-router/config.yaml")
    parser.add_argument("--scheduler-config", type=Path, default=ROOT / "hosts/redqueen/model-router/scheduler.json")
    parser.add_argument("--model-root", type=Path, default=Path("/srv/ai/models"))
    parser.add_argument("--metadata-root", type=Path, default=ROOT / "hosts/redqueen/models/metadata")
    parser.add_argument("--output", required=True, type=Path, help="catalog JSON path outside --model-root")
    parser.add_argument("--model", action="append", dest="models", help="limit output to this configured model/service id; repeatable")
    parser.add_argument(
        "--verify-sha256",
        action="store_true",
        help="read every selected artifact fully and verify its pinned SHA-256; expensive for a large catalog",
    )
    args = parser.parse_args(argv)

    try:
        _ensure_output_outside_model_root(args.output, args.model_root)
        refs = _load_artifact_refs(args.router_config, args.scheduler_config)
        store = build_store(
            refs,
            model_root=args.model_root,
            metadata_root=args.metadata_root,
            verify_hashes=args.verify_sha256,
            model_filter=set(args.models) if args.models else None,
        )
        write_atomic(args.output, store)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"weight-store: {error}", file=sys.stderr)
        return 2
    print(json.dumps(store["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
