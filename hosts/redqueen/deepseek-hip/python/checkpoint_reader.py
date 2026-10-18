"""Metadata validation and bounded reads from the pinned DeepSeek safetensors."""

from __future__ import annotations

import json
import os
import struct
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_INDEX_FILE = "model.safetensors.index.json"
_REVISION_FILE = "SOURCE_REVISION"
_MAX_INDEX_BYTES = 64 * 1024 * 1024
_MAX_HEADER_BYTES = 128 * 1024 * 1024
_MAX_READ_BYTES = 256 * 1024 * 1024
_DEFAULT_READ_BYTES = 64 * 1024 * 1024
_MAX_ROWS_PER_READ = 65_536
_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E4M3FNUZ": 1,
    "F8_E5M2": 1,
    "F8_E5M2FNUZ": 1,
    "F8_E8M0": 1,
}


class CheckpointError(ValueError):
    """The checkpoint metadata does not describe a complete, pinned artifact."""


@dataclass(frozen=True)
class TensorInfo:
    name: str
    path: Path
    dtype: str
    shape: tuple[int, ...]
    absolute_offset: int
    nbytes: int

    @property
    def row_bytes(self) -> int:
        if not self.shape:
            raise ValueError(f"tensor {self.name} is scalar and has no rows")
        if self.shape[0] == 0:
            return 0
        return self.nbytes // self.shape[0]


def _json_no_duplicates(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path, *, limit: int) -> object:
    try:
        size = path.stat().st_size
        if size > limit:
            raise CheckpointError(f"metadata file exceeds {limit} byte limit: {path.name}")
        with path.open("rb") as source:
            raw = source.read(limit + 1)
        if len(raw) > limit:
            raise CheckpointError(f"metadata file exceeds {limit} byte limit: {path.name}")
        return json.loads(raw, object_pairs_hook=_json_no_duplicates)
    except CheckpointError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"cannot read checkpoint metadata {path.name}: {type(exc).__name__}") from exc


class SafeTensorCheckpoint:
    """Validate all shard headers, then serve explicitly bounded byte reads.

    Construction reads only each file's small JSON header. Tensor payloads stay on disk until
    ``read_tensor_bytes`` or ``read_rows`` is called. Row requests are coalesced into contiguous
    reads and preserve caller order and duplicates, making this suitable for future lazy expert and
    Engram paths without materializing a whole shard or table.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        expected_revision: str,
        index_filename: str = _INDEX_FILE,
    ) -> None:
        if not isinstance(expected_revision, str) or not expected_revision:
            raise ValueError("expected_revision must be a nonempty string")
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise CheckpointError("checkpoint root is not a directory")
        self._lock = threading.Lock()
        self._closed = False
        self.tensors: dict[str, TensorInfo] = {}
        self._file_identities: dict[Path, tuple[int, int, int, int, int]] = {}

        revision = self._read_revision()
        if revision != expected_revision:
            raise CheckpointError(
                f"checkpoint revision {revision!r} does not match expected {expected_revision!r}"
            )

        index_path = self._child_file(index_filename, "checkpoint index")
        document = _load_json(index_path, limit=_MAX_INDEX_BYTES)
        if not isinstance(document, dict) or not isinstance(document.get("weight_map"), dict):
            raise CheckpointError("checkpoint index has no weight_map object")
        weight_map = document["weight_map"]
        if not weight_map:
            raise CheckpointError("checkpoint index is empty")

        expected_by_file: dict[str, set[str]] = {}
        for name, filename in weight_map.items():
            if not isinstance(name, str) or not name:
                raise CheckpointError("checkpoint index contains an invalid tensor name")
            if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(".safetensors"):
                raise CheckpointError(f"checkpoint index contains an unsafe shard filename: {filename!r}")
            expected_by_file.setdefault(filename, set()).add(name)

        for filename, expected_names in expected_by_file.items():
            shard_path = self._child_file(filename, "safetensors shard")
            self._read_shard_header(shard_path, expected_names)

        metadata = document.get("metadata", {})
        if metadata is not None and not isinstance(metadata, dict):
            raise CheckpointError("checkpoint index metadata must be an object")
        indexed_total = sum(info.nbytes for info in self.tensors.values())
        if isinstance(metadata, dict) and "total_size" in metadata:
            declared_total = metadata["total_size"]
            if isinstance(declared_total, bool) or not isinstance(declared_total, int) or declared_total != indexed_total:
                raise CheckpointError("checkpoint index total_size does not match shard payloads")
        self.total_bytes = indexed_total
        self.tensor_count = len(self.tensors)

    def _child_file(self, filename: str, description: str) -> Path:
        path = self.root / filename
        try:
            resolved = path.resolve(strict=True)
            if resolved.parent != self.root or not resolved.is_file():
                raise CheckpointError(f"{description} is not a regular file inside the checkpoint root")
            return resolved
        except OSError as exc:
            raise CheckpointError(f"missing {description}: {filename}") from exc

    def _read_revision(self) -> str:
        path = self._child_file(_REVISION_FILE, "checkpoint source revision")
        try:
            if path.stat().st_size > 64 * 1024:
                raise CheckpointError("checkpoint source revision metadata is too large")
            fields = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    if key in fields:
                        raise CheckpointError(f"duplicate checkpoint revision field: {key}")
                    fields[key] = value
        except CheckpointError:
            raise
        except (OSError, UnicodeDecodeError) as exc:
            raise CheckpointError("cannot read checkpoint source revision") from exc
        revision = fields.get("revision")
        if not revision:
            raise CheckpointError("checkpoint source revision is missing")
        return revision

    def _read_shard_header(self, path: Path, expected_names: set[str]) -> None:
        fd = -1
        try:
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
            before = os.fstat(fd)
            file_size = before.st_size
            prefix = self._pread_exact(fd, 8, 0)
            header_size = struct.unpack("<Q", prefix)[0]
            if header_size == 0 or header_size > _MAX_HEADER_BYTES or header_size + 8 > file_size:
                raise CheckpointError(f"invalid safetensors header size in {path.name}")
            raw_header = self._pread_exact(fd, header_size, 8)
            after = os.fstat(fd)
            if self._file_identity(before) != self._file_identity(after):
                raise CheckpointError(f"safetensors shard changed while its header was read: {path.name}")
            self._file_identities[path] = self._file_identity(after)
        except CheckpointError:
            raise
        except OSError as exc:
            raise CheckpointError(f"cannot read safetensors shard header: {path.name}") from exc
        finally:
            if fd >= 0:
                os.close(fd)

        try:
            header = json.loads(raw_header, object_pairs_hook=_json_no_duplicates)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise CheckpointError(f"invalid safetensors header JSON in {path.name}") from exc
        if not isinstance(header, dict):
            raise CheckpointError(f"safetensors header is not an object: {path.name}")
        file_metadata = header.pop("__metadata__", None)
        if file_metadata is not None and (
            not isinstance(file_metadata, dict)
            or any(not isinstance(key, str) or not isinstance(value, str) for key, value in file_metadata.items())
        ):
            raise CheckpointError(f"invalid safetensors metadata in {path.name}")
        actual_names = set(header)
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)[:3]
            unexpected = sorted(actual_names - expected_names)[:3]
            raise CheckpointError(
                f"checkpoint index/header mismatch for {path.name}: missing={missing}, unexpected={unexpected}"
            )

        payload_size = file_size - 8 - header_size
        entries: list[tuple[int, int, str]] = []
        for name, metadata in header.items():
            if not isinstance(name, str) or not isinstance(metadata, dict):
                raise CheckpointError(f"invalid tensor entry in {path.name}")
            dtype = metadata.get("dtype")
            if not isinstance(dtype, str) or dtype not in _DTYPE_BYTES:
                raise CheckpointError(f"unsupported safetensors dtype for {name}: {dtype!r}")
            shape_value = metadata.get("shape")
            if not isinstance(shape_value, list) or any(
                isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in shape_value
            ):
                raise CheckpointError(f"invalid shape for safetensors tensor {name}")
            offsets = metadata.get("data_offsets")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(isinstance(offset, bool) or not isinstance(offset, int) for offset in offsets)
            ):
                raise CheckpointError(f"invalid data offsets for safetensors tensor {name}")
            start, end = offsets
            if start < 0 or end < start or end > payload_size:
                raise CheckpointError(f"safetensors tensor offsets exceed payload size: {name}")
            item_count = 1
            for dimension in shape_value:
                item_count *= dimension
            nbytes = item_count * _DTYPE_BYTES[dtype]
            if end - start != nbytes:
                raise CheckpointError(f"safetensors tensor size does not match dtype/shape: {name}")
            entries.append((start, end, name))
            self.tensors[name] = TensorInfo(
                name=name,
                path=path,
                dtype=dtype,
                shape=tuple(shape_value),
                absolute_offset=8 + header_size + start,
                nbytes=nbytes,
            )

        cursor = 0
        for start, end, name in sorted(entries):
            if start != cursor:
                raise CheckpointError(f"safetensors payload has a gap or overlap before {name}")
            cursor = end
        if cursor != payload_size:
            raise CheckpointError(f"safetensors payload has unindexed trailing bytes in {path.name}")

    @staticmethod
    def _validate_limit(max_bytes: int) -> None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("read limit must be a positive integer")
        if max_bytes > _MAX_READ_BYTES:
            raise ValueError(f"read limit cannot exceed hard limit of {_MAX_READ_BYTES} bytes")

    def _tensor(self, name: str) -> TensorInfo:
        with self._lock:
            if self._closed:
                raise RuntimeError("checkpoint reader is closed")
        try:
            return self.tensors[name]
        except KeyError as exc:
            raise KeyError(f"tensor is not in checkpoint index: {name}") from exc

    @staticmethod
    def _pread_exact(fd: int, nbytes: int, offset: int) -> bytes:
        result = bytearray(nbytes)
        view = memoryview(result)
        done = 0
        while done < nbytes:
            chunk = os.pread(fd, nbytes - done, offset + done)
            if not chunk:
                raise CheckpointError("unexpected end of safetensors payload")
            view[done : done + len(chunk)] = chunk
            done += len(chunk)
        return bytes(result)

    @staticmethod
    def _file_identity(stat_result: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            stat_result.st_dev,
            stat_result.st_ino,
            stat_result.st_size,
            stat_result.st_mtime_ns,
            stat_result.st_ctime_ns,
        )

    def _open_verified(self, info: TensorInfo) -> int:
        with self._lock:
            if self._closed:
                raise RuntimeError("checkpoint reader is closed")
        fd = os.open(info.path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        try:
            expected_identity = self._file_identities[info.path]
            if self._file_identity(os.fstat(fd)) != expected_identity:
                raise CheckpointError(f"checkpoint shard changed since metadata validation: {info.path.name}")
            return fd
        except BaseException:
            os.close(fd)
            raise

    def _check_unchanged(self, fd: int, info: TensorInfo) -> None:
        if self._file_identity(os.fstat(fd)) != self._file_identities[info.path]:
            raise CheckpointError(f"checkpoint shard changed during tensor read: {info.path.name}")

    def read_tensor_bytes(self, name: str, *, max_bytes: int = _DEFAULT_READ_BYTES) -> bytes:
        """Read one complete tensor only when it fits within the hard byte limit."""
        self._validate_limit(max_bytes)
        info = self._tensor(name)
        if info.nbytes > max_bytes:
            raise ValueError(f"tensor read of {info.nbytes} bytes exceeds limit of {max_bytes} bytes")
        if info.nbytes == 0:
            return b""
        fd = self._open_verified(info)
        try:
            result = self._pread_exact(fd, info.nbytes, info.absolute_offset)
            self._check_unchanged(fd, info)
            return result
        finally:
            os.close(fd)

    def read_rows(
        self,
        name: str,
        rows: Iterable[int],
        *,
        max_bytes: int = _DEFAULT_READ_BYTES,
    ) -> bytes:
        """Read bounded axis-zero rows, preserving order and duplicate indices."""
        self._validate_limit(max_bytes)
        info = self._tensor(name)
        if not info.shape:
            raise ValueError(f"tensor {name} is scalar and has no rows")
        row_count = info.shape[0]
        row_bytes = info.row_bytes
        positions: dict[int, list[int]] = {}
        requested_count = 0
        for position, row in enumerate(rows):
            if position >= _MAX_ROWS_PER_READ:
                raise ValueError(f"row read exceeds hard limit of {_MAX_ROWS_PER_READ} rows")
            if isinstance(row, bool) or not isinstance(row, int) or row < 0 or row >= row_count:
                raise IndexError(f"row index out of range for {name}: {row!r}")
            requested_count += 1
            if requested_count * row_bytes > max_bytes:
                raise ValueError(f"row read exceeds limit of {max_bytes} bytes")
            positions.setdefault(row, []).append(position)

        if not requested_count or not row_bytes:
            return b"" if not requested_count else bytes(requested_count * row_bytes)

        ordered_rows = sorted(positions)
        output = bytearray(requested_count * row_bytes)
        fd = self._open_verified(info)
        try:
            run_start = run_end = ordered_rows[0]
            for row in ordered_rows[1:] + [None]:
                if row is not None and row == run_end + 1:
                    run_end = row
                    continue
                run_count = run_end - run_start + 1
                run = self._pread_exact(
                    fd,
                    run_count * row_bytes,
                    info.absolute_offset + run_start * row_bytes,
                )
                for row_index in range(run_start, run_end + 1):
                    start = (row_index - run_start) * row_bytes
                    row_data = run[start : start + row_bytes]
                    for output_row in positions[row_index]:
                        output_start = output_row * row_bytes
                        output[output_start : output_start + row_bytes] = row_data
                if row is not None:
                    run_start = run_end = row
            self._check_unchanged(fd, info)
            return bytes(output)
        finally:
            os.close(fd)

    def close(self) -> None:
        with self._lock:
            self._closed = True

    def __enter__(self) -> SafeTensorCheckpoint:
        with self._lock:
            if self._closed:
                raise RuntimeError("checkpoint reader is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
