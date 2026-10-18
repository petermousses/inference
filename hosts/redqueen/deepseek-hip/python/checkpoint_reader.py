"""Metadata validation and bounded reads from the pinned DeepSeek safetensors."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import struct
import threading
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterator, Mapping, Sequence


_INDEX_FILE = "model.safetensors.index.json"
_REVISION_FILE = "SOURCE_REVISION"
_CHUNK_DIGESTS_FILE = "deepseek-v4.1-flash-1017.chunks.bin"
_HEADER_DIGESTS_FILE = "deepseek-v4.1-flash-1017.headers.tsv"
_CHUNK_SIZE = 1024 * 1024
_CHUNK_DIGEST_BYTES = 32
_MAX_INDEX_BYTES = 64 * 1024 * 1024
_MAX_HEADER_BYTES = 4 * 1024 * 1024
_MAX_READ_BYTES = 256 * 1024 * 1024
_DEFAULT_READ_BYTES = 64 * 1024 * 1024
_MAX_ROWS_PER_READ = 65_536
_READ_CHUNK_BYTES = 1024 * 1024
_ROW_INDEX_RESERVATION_BYTES = 16 * 1024 * 1024
_INDEX_VALIDATION_RESERVATION_BYTES = 256 * 1024 * 1024
_HEADER_VALIDATION_RESERVATION_BYTES = 64 * 1024 * 1024
_MAX_INFLIGHT_READ_BYTES = 1024 * 1024 * 1024
_CHUNK_MANIFEST_RESERVATION_BYTES = 32 * 1024 * 1024
_DEFAULT_HEADER_LOCK = threading.Lock()
_DEFAULT_CHUNK_DIGEST_LOCK = threading.Lock()
_DEFAULT_CHUNK_DIGEST_BYTES: bytes | None = None
_DEFAULT_CHUNK_DIGESTS_SHA256 = "0fcd23effcbdd2688f54170ed934fd226fd4714f499b91374b25cd898a5b9c6b"
_DEFAULT_HEADER_DIGESTS_SHA256 = "2f9a260e70ac175304b61ecc5498733b30cb576cbb3f99abd15e1596be7fcbc6"
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
    filename: str
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


def _file_identity(stat_result: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
        stat_result.st_ctime_ns,
    )


class ReadMemoryBudget:
    """Fair, weighted admission for transient checkpoint-read allocations."""

    def __init__(self, capacity_bytes: int) -> None:
        if isinstance(capacity_bytes, bool) or not isinstance(capacity_bytes, int) or capacity_bytes <= 0:
            raise ValueError("read memory budget must be a positive integer")
        self.capacity_bytes = capacity_bytes
        self._condition = threading.Condition()
        self._reserved_bytes = 0
        self._tickets: deque[object] = deque()
        self._thread_local = threading.local()

    @contextmanager
    def reserve(self, amount: int) -> Iterator[None]:
        if isinstance(amount, bool) or not isinstance(amount, int) or amount <= 0:
            raise ValueError("read memory reservation must be a positive integer")
        thread_reserved = getattr(self._thread_local, "reserved_bytes", 0)
        if thread_reserved:
            raise ValueError("read memory reservations cannot be nested")
        if amount > self.capacity_bytes:
            raise ValueError("read memory reservation exceeds the shared budget")
        with self._condition:
            ticket = object()
            self._tickets.append(ticket)
            try:
                while self._tickets[0] is not ticket or self._reserved_bytes + amount > self.capacity_bytes:
                    self._condition.wait()
                self._tickets.popleft()
                self._reserved_bytes += amount
                self._thread_local.reserved_bytes = amount
                self._condition.notify_all()
            except BaseException:
                self._tickets.remove(ticket)
                self._condition.notify_all()
                raise
        try:
            yield
        finally:
            with self._condition:
                self._reserved_bytes -= amount
                self._thread_local.reserved_bytes = thread_reserved
                self._condition.notify_all()


_DEFAULT_READ_MEMORY_BUDGET = ReadMemoryBudget(_MAX_INFLIGHT_READ_BYTES)


def _read_limited_fd(fd: int, *, limit: int, description: str, name: str) -> bytes:
    before = os.fstat(fd)
    try:
        if before.st_size > limit:
            raise CheckpointError(f"{description} exceeds {limit} byte limit: {name}")
        result = bytearray()
        while len(result) <= limit:
            chunk = os.read(fd, min(8 * 1024 * 1024, limit + 1 - len(result)))
            if not chunk:
                break
            result.extend(chunk)
        if len(result) > limit:
            raise CheckpointError(f"{description} exceeds {limit} byte limit: {name}")
        if _file_identity(os.fstat(fd)) != _file_identity(before):
            raise CheckpointError(f"{description} changed while it was read: {name}")
        return bytes(result)
    except OSError as exc:
        raise CheckpointError(f"cannot read {description}: {name}") from exc


def _read_regular_path(path: Path, *, limit: int, description: str) -> bytes:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise CheckpointError("platform does not support no-follow checkpoint metadata opens")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0) | nofollow
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise CheckpointError(f"cannot safely open {description}: {path}") from exc
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise CheckpointError(f"{description} is not a regular file: {path}")
        return _read_limited_fd(fd, limit=limit, description=description, name=str(path))
    finally:
        os.close(fd)


def _load_json_fd(fd: int, *, name: str, limit: int) -> object:
    try:
        raw = _read_limited_fd(fd, limit=limit, description="checkpoint metadata file", name=name)
        return json.loads(raw, object_pairs_hook=_json_no_duplicates)
    except CheckpointError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"cannot read checkpoint metadata {name}: {type(exc).__name__}") from exc


def _load_pinned_json_fd(
    fd: int, *, name: str, limit: int, expected: tuple[int, str]
) -> object:
    expected_size, expected_digest = expected
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size <= 0
        or not isinstance(expected_digest, str)
        or len(expected_digest) != 64
        or any(character not in "0123456789abcdef" for character in expected_digest)
    ):
        raise CheckpointError(f"invalid pinned file identity: {name}")
    raw = _read_limited_fd(fd, limit=limit, description="checkpoint metadata file", name=name)
    if len(raw) != expected_size or hashlib.sha256(raw).hexdigest() != expected_digest:
        raise CheckpointError(f"file does not match pinned SHA-256: {name}")
    try:
        return json.loads(raw, object_pairs_hook=_json_no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise CheckpointError(f"cannot read checkpoint metadata {name}: {type(exc).__name__}") from exc


class _LazyTensorMap(Mapping[str, TensorInfo]):
    def __init__(self, checkpoint: SafeTensorCheckpoint) -> None:
        self._checkpoint = checkpoint

    def __getitem__(self, name: str) -> TensorInfo:
        return self._checkpoint._tensor(name)

    def __iter__(self) -> Iterator[str]:
        with self._checkpoint._lock:
            if self._checkpoint._closed:
                raise RuntimeError("checkpoint reader is closed")
            return iter(self._checkpoint._weight_map)

    def __len__(self) -> int:
        return self._checkpoint.tensor_count


class SafeTensorCheckpoint:
    """Validate shard headers, then serve bounded reads from the pinned checkpoint.

    Construction authenticates the small index. Each shard's JSON header and every payload chunk
    used by a read are checked against versioned SHA-256 values. If ``verify_shard_digests`` is
    enabled, the complete shard SHA-256 is also checked before metadata is exposed. That full scan
    is intended for offline audits; chunk verification preserves lazy reads of large Engram tables.
    Row requests are coalesced into contiguous reads and preserve caller order and duplicates.
    Read-memory reservations cover transient buffers; callers must separately account for returned
    bytes they retain.

    ``metadata_only=True`` authenticates each safetensors prefix and header against the pinned
    header inventory without reading any payload bytes. Payload reads are disabled in that mode.

    ``expected_file_inventory`` defaults to the version-controlled model inventory beside this
    module. Its chunk digests default to the versioned binary chunk inventory beside it. Overrides
    must come from another trusted, pinned inventory.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        expected_revision: str,
        index_filename: str = _INDEX_FILE,
        expected_file_inventory: Mapping[str, tuple[int, str]] | None = None,
        expected_chunk_digest_bytes: bytes | None = None,
        expected_header_inventory: Mapping[str, tuple[int, str]] | None = None,
        read_memory_budget: ReadMemoryBudget | None = None,
        verify_shard_digests: bool = False,
        metadata_only: bool = False,
    ) -> None:
        if not isinstance(expected_revision, str) or not expected_revision:
            raise ValueError("expected_revision must be a nonempty string")
        if type(verify_shard_digests) is not bool:
            raise ValueError("verify_shard_digests must be a boolean")
        if type(metadata_only) is not bool:
            raise ValueError("metadata_only must be a boolean")
        if metadata_only and verify_shard_digests:
            raise ValueError("metadata-only checkpoint readers cannot verify full shard digests")
        self.root = Path(root).resolve(strict=True)
        self._lock = threading.Lock()
        self._closed = False
        self._root_fd = -1
        self._tensors: dict[str, TensorInfo] = {}
        self._weight_map: Mapping[str, str] = MappingProxyType({})
        self._names_by_shard: dict[str, set[str]] = {}
        self._expected_shards: frozenset[str] = frozenset()
        self._loaded_shards: set[str] = set()
        self._file_identities: dict[str, tuple[int, int, int, int, int]] = {}
        self._validated_shards: set[str] = set()
        self._shard_locks: dict[str, threading.Lock] = {}
        self._verified_chunks: dict[str, bytearray] = {}
        self._chunk_digest_offsets: dict[str, int] = {}
        self._read_memory_budget = read_memory_budget or _DEFAULT_READ_MEMORY_BUDGET
        inventory = self._load_default_file_inventory() if expected_file_inventory is None else {}
        if expected_file_inventory is not None:
            try:
                inventory = dict(expected_file_inventory)
            except (TypeError, ValueError) as exc:
                raise CheckpointError("pinned file inventory must be a mapping") from exc
            normalized_inventory: dict[str, tuple[int, str]] = {}
            for filename, identity in inventory.items():
                if (
                    not self._valid_child_name(filename)
                    or not isinstance(identity, (tuple, list))
                    or len(identity) != 2
                ):
                    raise CheckpointError(f"invalid pinned file inventory entry: {filename!r}")
                normalized_inventory[filename] = (identity[0], identity[1])
            inventory = normalized_inventory
        if expected_header_inventory is not None:
            try:
                header_inventory = dict(expected_header_inventory)
            except (TypeError, ValueError) as exc:
                raise CheckpointError("pinned header inventory must be a mapping") from exc
        elif metadata_only and expected_file_inventory is None:
            header_inventory = self._load_default_header_inventory(inventory)
        else:
            header_inventory = {}
        normalized_header_inventory: dict[str, tuple[int, str]] = {}
        for filename, identity in header_inventory.items():
            if (
                not self._valid_child_name(filename)
                or not isinstance(identity, (tuple, list))
                or len(identity) != 2
            ):
                raise CheckpointError(f"invalid pinned header inventory entry: {filename!r}")
            size, digest = identity
            if (
                filename not in inventory
                or not filename.endswith(".safetensors")
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size != inventory[filename][0]
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise CheckpointError(f"invalid pinned header identity: {filename!r}")
            normalized_header_inventory[filename] = (size, digest)
        expected_shards = {filename for filename in inventory if filename.endswith(".safetensors")}
        if metadata_only and set(normalized_header_inventory) != expected_shards:
            raise CheckpointError("pinned header inventory does not cover every safetensors shard")
        self._expected_header_inventory = MappingProxyType(normalized_header_inventory)
        self._metadata_only = metadata_only
        use_default_chunk_digests = (
            not metadata_only
            and expected_file_inventory is None
            and expected_chunk_digest_bytes is None
        )
        self._expected_file_inventory = MappingProxyType(inventory)
        if expected_chunk_digest_bytes is not None and type(expected_chunk_digest_bytes) is not bytes:
            raise CheckpointError("pinned chunk digests must be bytes")
        self._chunk_digest_bytes = (
            self._load_default_chunk_digests(inventory)
            if use_default_chunk_digests
            else expected_chunk_digest_bytes
        )
        self._verify_shard_digests = verify_shard_digests
        self.tensors: Mapping[str, TensorInfo] = _LazyTensorMap(self)
        self.total_bytes: int | None = None
        self.tensor_count = 0
        self._observed_tensor_bytes = 0

        try:
            self._root_fd = self._open_root_directory(self.root)
            self._initialize(expected_revision, index_filename)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _load_default_file_inventory() -> dict[str, tuple[int, str]]:
        inventory_path = Path(__file__).resolve().parents[2] / "models" / "deepseek-v4.1-flash-1017.files.tsv"
        try:
            raw = _read_regular_path(
                inventory_path, limit=8 * 1024 * 1024, description="pinned shard inventory"
            )
            lines = raw.decode("utf-8").splitlines()
        except CheckpointError:
            raise
        except (OSError, UnicodeDecodeError) as exc:
            raise CheckpointError(f"cannot read pinned shard inventory: {inventory_path}") from exc
        result: dict[str, tuple[int, str]] = {}
        for line_number, line in enumerate(lines, 1):
            fields = line.split("\t")
            if len(fields) != 4:
                raise CheckpointError(f"invalid pinned shard inventory row {line_number}")
            filename, size_text, digest, object_id = fields
            if not filename.endswith(".safetensors") and filename != _INDEX_FILE:
                continue
            if Path(filename).name != filename or filename in result:
                raise CheckpointError(f"invalid or duplicate pinned shard name at row {line_number}")
            try:
                size = int(size_text)
            except ValueError as exc:
                raise CheckpointError(f"invalid pinned shard size at row {line_number}") from exc
            if (
                size <= 0
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or not object_id
            ):
                raise CheckpointError(f"invalid pinned shard identity at row {line_number}")
            result[filename] = (size, digest)
        if not result:
            raise CheckpointError("pinned file inventory contains no safetensors shards")
        return result

    @staticmethod
    def _load_default_header_inventory(
        inventory: Mapping[str, tuple[int, str]],
    ) -> dict[str, tuple[int, str]]:
        manifest_path = Path(__file__).resolve().parents[2] / "models" / _HEADER_DIGESTS_FILE
        try:
            raw = _read_regular_path(
                manifest_path, limit=16 * 1024, description="pinned safetensors header inventory"
            )
        except CheckpointError as exc:
            raise CheckpointError(f"cannot read pinned safetensors header inventory: {manifest_path}") from exc
        if hashlib.sha256(raw).hexdigest() != _DEFAULT_HEADER_DIGESTS_SHA256:
            raise CheckpointError("pinned safetensors header inventory does not match its SHA-256")
        result: dict[str, tuple[int, str]] = {}
        try:
            lines = raw.decode("ascii").splitlines()
        except UnicodeDecodeError as exc:
            raise CheckpointError("pinned safetensors header inventory is not ASCII") from exc
        for line_number, line in enumerate(lines, 1):
            fields = line.split("\t")
            if len(fields) != 3:
                raise CheckpointError(f"invalid pinned safetensors header row {line_number}")
            filename, size_text, digest = fields
            if filename in result or filename not in inventory:
                raise CheckpointError(f"invalid or duplicate pinned header name at row {line_number}")
            try:
                size = int(size_text)
            except ValueError as exc:
                raise CheckpointError(f"invalid pinned header size at row {line_number}") from exc
            if (
                size != inventory[filename][0]
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise CheckpointError(f"invalid pinned header identity at row {line_number}")
            result[filename] = (size, digest)
        expected_shards = {filename for filename in inventory if filename.endswith(".safetensors")}
        if set(result) != expected_shards:
            raise CheckpointError("pinned safetensors header inventory does not cover every shard")
        return result

    def _load_default_chunk_digests(
        self, inventory: Mapping[str, tuple[int, str]]
    ) -> bytes:
        global _DEFAULT_CHUNK_DIGEST_BYTES
        with _DEFAULT_CHUNK_DIGEST_LOCK:
            if _DEFAULT_CHUNK_DIGEST_BYTES is not None:
                return _DEFAULT_CHUNK_DIGEST_BYTES
        with self._read_memory_budget.reserve(_CHUNK_MANIFEST_RESERVATION_BYTES):
            with _DEFAULT_CHUNK_DIGEST_LOCK:
                if _DEFAULT_CHUNK_DIGEST_BYTES is not None:
                    return _DEFAULT_CHUNK_DIGEST_BYTES
                shard_sizes = [
                    identity[0]
                    for filename, identity in sorted(inventory.items())
                    if filename.endswith(".safetensors")
                ]
                expected_chunks = sum((size + _CHUNK_SIZE - 1) // _CHUNK_SIZE for size in shard_sizes)
                expected_bytes = expected_chunks * _CHUNK_DIGEST_BYTES
                manifest_path = Path(__file__).resolve().parents[2] / "models" / _CHUNK_DIGESTS_FILE
                try:
                    raw = _read_regular_path(
                        manifest_path,
                        limit=expected_bytes,
                        description="pinned chunk digest file",
                    )
                except CheckpointError as exc:
                    raise CheckpointError(f"cannot read pinned chunk digest file: {manifest_path}") from exc
                if len(raw) != expected_bytes:
                    raise CheckpointError(
                        f"pinned chunk digest file has {len(raw)} bytes; expected {expected_bytes}"
                    )
                if hashlib.sha256(raw).hexdigest() != _DEFAULT_CHUNK_DIGESTS_SHA256:
                    raise CheckpointError("pinned chunk digest file does not match its SHA-256")
                _DEFAULT_CHUNK_DIGEST_BYTES = raw
                return raw

    @staticmethod
    def _open_root_directory(root: Path) -> int:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
        nofollow = getattr(os, "O_NOFOLLOW", None)
        if nofollow is None or not getattr(os, "O_DIRECTORY", 0):
            raise CheckpointError("platform does not support safe checkpoint directory opens")
        fd = -1
        try:
            fd = os.open(root, flags | nofollow)
            if not stat.S_ISDIR(os.fstat(fd).st_mode):
                raise CheckpointError("checkpoint root is not a directory")
            return fd
        except CheckpointError:
            if fd >= 0:
                os.close(fd)
            raise
        except OSError as exc:
            if fd >= 0:
                os.close(fd)
            raise CheckpointError("cannot safely open checkpoint root directory") from exc

    @staticmethod
    def _valid_child_name(filename: str) -> bool:
        return (
            isinstance(filename, str)
            and bool(filename)
            and filename not in {".", ".."}
            and Path(filename).name == filename
            and "/" not in filename
            and "\\" not in filename
            and "\x00" not in filename
        )

    def _open_child(self, filename: str, description: str) -> tuple[int, os.stat_result]:
        if not self._valid_child_name(filename):
            raise CheckpointError(f"unsafe {description} filename: {filename!r}")
        nofollow = getattr(os, "O_NOFOLLOW", None)
        if nofollow is None:
            raise CheckpointError("platform does not support no-follow checkpoint file opens")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0) | nofollow
        with self._lock:
            if self._closed or self._root_fd < 0:
                raise RuntimeError("checkpoint reader is closed")
            try:
                fd = os.open(filename, flags, dir_fd=self._root_fd)
            except OSError as exc:
                raise CheckpointError(f"cannot safely open {description}: {filename}") from exc
        try:
            file_stat = os.fstat(fd)
            if not stat.S_ISREG(file_stat.st_mode):
                raise CheckpointError(f"{description} is not a regular file: {filename}")
            return fd, file_stat
        except BaseException:
            os.close(fd)
            raise

    def _initialize(self, expected_revision: str, index_filename: str) -> None:
        revision = self._read_revision()
        if revision != expected_revision:
            raise CheckpointError(
                f"checkpoint revision {revision!r} does not match expected {expected_revision!r}"
            )

        with self._read_memory_budget.reserve(_INDEX_VALIDATION_RESERVATION_BYTES):
            with _DEFAULT_HEADER_LOCK:
                index_fd, _ = self._open_child(index_filename, "checkpoint index")
                try:
                    index_identity = self._expected_file_inventory.get(index_filename)
                    if index_identity is None:
                        raise CheckpointError(f"pinned file inventory is missing {index_filename}")
                    document = _load_pinned_json_fd(
                        index_fd,
                        name=index_filename,
                        limit=_MAX_INDEX_BYTES,
                        expected=index_identity,
                    )
                finally:
                    os.close(index_fd)
        if not isinstance(document, dict) or not isinstance(document.get("weight_map"), dict):
            raise CheckpointError("checkpoint index has no weight_map object")
        weight_map = document["weight_map"]
        if not weight_map:
            raise CheckpointError("checkpoint index is empty")

        expected_by_file: dict[str, set[str]] = {}
        for name, filename in weight_map.items():
            if not isinstance(name, str) or not name:
                raise CheckpointError("checkpoint index contains an invalid tensor name")
            if not self._valid_child_name(filename) or not filename.endswith(".safetensors"):
                raise CheckpointError(f"checkpoint index contains an unsafe shard filename: {filename!r}")
            expected_by_file.setdefault(filename, set()).add(name)

        expected_shards = {
            filename
            for filename in self._expected_file_inventory
            if filename.endswith(".safetensors")
        }
        if set(expected_by_file) != expected_shards:
            raise CheckpointError("pinned shard inventory does not exactly match checkpoint index shards")

        chunk_count = 0
        for filename in sorted(expected_by_file):
            expected_names = expected_by_file[filename]
            expected_identity = self._expected_file_inventory[filename]
            if not isinstance(expected_identity, tuple) or len(expected_identity) != 2:
                raise CheckpointError(f"invalid pinned shard identity: {filename}")
            expected_size, expected_digest = expected_identity
            if (
                isinstance(expected_size, bool)
                or not isinstance(expected_size, int)
                or expected_size <= 0
                or not isinstance(expected_digest, str)
                or len(expected_digest) != 64
                or any(character not in "0123456789abcdef" for character in expected_digest)
            ):
                raise CheckpointError(f"invalid pinned shard identity: {filename}")
            self._shard_locks[filename] = threading.Lock()
            self._names_by_shard[filename] = expected_names
            self._chunk_digest_offsets[filename] = chunk_count
            shard_chunk_count = (expected_size + _CHUNK_SIZE - 1) // _CHUNK_SIZE
            chunk_count += shard_chunk_count
            self._verified_chunks[filename] = bytearray((shard_chunk_count + 7) // 8)

        if self._chunk_digest_bytes is not None:
            expected_digest_bytes = chunk_count * _CHUNK_DIGEST_BYTES
            if len(self._chunk_digest_bytes) != expected_digest_bytes:
                raise CheckpointError(
                    f"pinned chunk digest size is {len(self._chunk_digest_bytes)} bytes; "
                    f"expected {expected_digest_bytes}"
                )
        elif not self._verify_shard_digests and not self._metadata_only:
            raise CheckpointError(
                "pinned chunk digests are required unless full shard SHA-256 verification is enabled"
            )

        metadata = document.get("metadata", {})
        if metadata is not None and not isinstance(metadata, dict):
            raise CheckpointError("checkpoint index metadata must be an object")
        indexed_total = None
        if isinstance(metadata, dict) and "total_size" in metadata:
            declared_total = metadata["total_size"]
            if isinstance(declared_total, bool) or not isinstance(declared_total, int) or declared_total < 0:
                raise CheckpointError("checkpoint index total_size is invalid")
            indexed_total = declared_total
        self.total_bytes = indexed_total
        self._weight_map = MappingProxyType(dict(weight_map))
        self._expected_shards = frozenset(expected_by_file)
        self.tensor_count = len(weight_map)

    def _verify_digest_fd(self, fd: int, filename: str, expected: tuple[int, str]) -> tuple[int, int, int, int, int]:
        expected_size, expected_digest = expected
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size <= 0
            or not isinstance(expected_digest, str)
            or len(expected_digest) != 64
            or any(character not in "0123456789abcdef" for character in expected_digest)
        ):
            raise CheckpointError(f"invalid pinned file identity: {filename}")
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
                raise CheckpointError(f"file size does not match pinned inventory: {filename}")
            digest = hashlib.sha256()
            offset = 0
            while offset < expected_size:
                chunk = os.pread(fd, min(_READ_CHUNK_BYTES, expected_size - offset), offset)
                if not chunk:
                    raise CheckpointError(f"file ended during digest verification: {filename}")
                digest.update(chunk)
                offset += len(chunk)
            after = os.fstat(fd)
        except OSError as exc:
            raise CheckpointError(f"cannot verify pinned file digest: {filename}") from exc
        if _file_identity(after) != _file_identity(before):
            raise CheckpointError(f"file changed during digest verification: {filename}")
        if digest.hexdigest() != expected_digest:
            raise CheckpointError(f"file does not match pinned SHA-256: {filename}")
        return _file_identity(after)

    def _read_verified_chunk(
        self,
        fd: int,
        filename: str,
        chunk_index: int,
        file_identity: tuple[int, int, int, int, int],
    ) -> bytes:
        expected_size, _ = self._expected_file_inventory[filename]
        chunk_count = (expected_size + _CHUNK_SIZE - 1) // _CHUNK_SIZE
        if chunk_index < 0 or chunk_index >= chunk_count:
            raise CheckpointError(f"chunk index is out of range for {filename}: {chunk_index}")
        if _file_identity(os.fstat(fd)) != file_identity:
            raise CheckpointError(f"checkpoint shard changed before chunk read: {filename}")
        offset = chunk_index * _CHUNK_SIZE
        chunk_length = min(_CHUNK_SIZE, expected_size - offset)
        raw = self._pread_exact(fd, chunk_length, offset)
        if not self._verify_shard_digests:
            if self._chunk_digest_bytes is None:
                raise CheckpointError("pinned chunk digests are unavailable")
            manifest_offset = (
                self._chunk_digest_offsets[filename] + chunk_index
            ) * _CHUNK_DIGEST_BYTES
            expected_digest = self._chunk_digest_bytes[
                manifest_offset : manifest_offset + _CHUNK_DIGEST_BYTES
            ]
            verified_byte = chunk_index >> 3
            verified_mask = 1 << (chunk_index & 7)
            with self._lock:
                already_verified = bool(self._verified_chunks[filename][verified_byte] & verified_mask)
            if not already_verified and hashlib.sha256(raw).digest() != expected_digest:
                raise CheckpointError(
                    f"chunk {chunk_index} does not match pinned SHA-256: {filename}"
                )
            if not already_verified:
                with self._lock:
                    self._verified_chunks[filename][verified_byte] |= verified_mask
        if _file_identity(os.fstat(fd)) != file_identity:
            raise CheckpointError(f"checkpoint shard changed during chunk read: {filename}")
        return raw

    def _read_verified_range(
        self,
        fd: int,
        filename: str,
        offset: int,
        nbytes: int,
        file_identity: tuple[int, int, int, int, int],
    ) -> bytes:
        expected_size, _ = self._expected_file_inventory[filename]
        if offset < 0 or nbytes < 0 or offset + nbytes > expected_size:
            raise CheckpointError(f"metadata range exceeds pinned file size: {filename}")
        result = bytearray(nbytes)
        copied = 0
        cursor = offset
        end = offset + nbytes
        while cursor < end:
            chunk_index = cursor // _CHUNK_SIZE
            chunk = self._read_verified_chunk(fd, filename, chunk_index, file_identity)
            chunk_offset = chunk_index * _CHUNK_SIZE
            source_offset = cursor - chunk_offset
            amount = min(end - cursor, len(chunk) - source_offset)
            result[copied : copied + amount] = memoryview(chunk)[source_offset : source_offset + amount]
            cursor += amount
            copied += amount
        return bytes(result)

    def _read_revision(self) -> str:
        filename = _REVISION_FILE
        fd, _ = self._open_child(filename, "checkpoint source revision")
        try:
            raw = _read_limited_fd(
                fd, limit=64 * 1024, description="checkpoint source revision", name=filename
            )
            fields = {}
            for line in raw.decode("utf-8").splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    if key in fields:
                        raise CheckpointError(f"duplicate checkpoint revision field: {key}")
                    fields[key] = value
        except CheckpointError:
            raise
        except (OSError, UnicodeDecodeError) as exc:
            raise CheckpointError("cannot read checkpoint source revision") from exc
        finally:
            os.close(fd)
        revision = fields.get("revision")
        if not revision:
            raise CheckpointError("checkpoint source revision is missing")
        return revision

    def _load_shard_header(self, filename: str) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("checkpoint reader is closed")
            if filename in self._loaded_shards:
                return
        with self._read_memory_budget.reserve(_HEADER_VALIDATION_RESERVATION_BYTES):
            with self._shard_locks[filename]:
                if filename in self._loaded_shards:
                    return
                with _DEFAULT_HEADER_LOCK:
                    if filename not in self._loaded_shards:
                        self._load_shard_header_locked(filename)

    def _load_shard_header_locked(self, filename: str) -> None:
        fd, shard_stat = self._open_child(filename, "safetensors shard")
        try:
            expected_size, _ = self._expected_file_inventory[filename]
            if shard_stat.st_size != expected_size:
                raise CheckpointError(f"file size does not match pinned inventory: {filename}")
            file_identity = (
                self._verify_digest_fd(fd, filename, self._expected_file_inventory[filename])
                if self._verify_shard_digests
                else _file_identity(shard_stat)
            )
            file_size = os.fstat(fd).st_size
            if self._metadata_only:
                prefix = self._pread_exact(fd, 8, 0)
            else:
                first_chunk = self._read_verified_chunk(fd, filename, 0, file_identity)
                prefix = first_chunk[:8]
            header_size = struct.unpack("<Q", prefix)[0]
            if header_size == 0 or header_size > _MAX_HEADER_BYTES or header_size + 8 > file_size:
                raise CheckpointError(f"invalid safetensors header size in {filename}")
            if self._metadata_only:
                raw_header = self._pread_exact(fd, header_size, 8)
                expected_header = self._expected_header_inventory[filename][1]
                if hashlib.sha256(prefix + raw_header).hexdigest() != expected_header:
                    raise CheckpointError(f"header does not match pinned SHA-256: {filename}")
            elif header_size + 8 <= len(first_chunk):
                raw_header = first_chunk[8 : 8 + header_size]
            else:
                raw_header = self._read_verified_range(
                    fd, filename, 8, header_size, file_identity
                )
            if _file_identity(os.fstat(fd)) != file_identity:
                raise CheckpointError(f"safetensors shard changed while its header was read: {filename}")
        except CheckpointError:
            raise
        except OSError as exc:
            raise CheckpointError(f"cannot read safetensors shard header: {filename}") from exc
        finally:
            os.close(fd)

        try:
            header = json.loads(raw_header, object_pairs_hook=_json_no_duplicates)
        except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as exc:
            raise CheckpointError(f"invalid safetensors header JSON in {filename}") from exc
        if not isinstance(header, dict):
            raise CheckpointError(f"safetensors header is not an object: {filename}")
        file_metadata = header.pop("__metadata__", None)
        if file_metadata is not None and (
            not isinstance(file_metadata, dict)
            or any(not isinstance(key, str) or not isinstance(value, str) for key, value in file_metadata.items())
        ):
            raise CheckpointError(f"invalid safetensors metadata in {filename}")
        expected_names = self._names_by_shard[filename]
        actual_names = set(header)
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)[:3]
            unexpected = sorted(actual_names - expected_names)[:3]
            raise CheckpointError(
                f"checkpoint index/header mismatch for {filename}: missing={missing}, unexpected={unexpected}"
            )

        payload_size = file_size - 8 - header_size
        entries: list[tuple[int, int, str]] = []
        parsed_tensors: dict[str, TensorInfo] = {}
        for name, metadata in header.items():
            if not isinstance(name, str) or not isinstance(metadata, dict):
                raise CheckpointError(f"invalid tensor entry in {filename}")
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
            parsed_tensors[name] = TensorInfo(
                name=name,
                    filename=filename,
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
            raise CheckpointError(f"safetensors payload has unindexed trailing bytes in {filename}")

        with self._lock:
            observed_total = self._observed_tensor_bytes + sum(
                info.nbytes for info in parsed_tensors.values()
            )
            if (
                self.total_bytes is not None
                and len(self._loaded_shards) + 1 == len(self._expected_shards)
                and observed_total != self.total_bytes
            ):
                raise CheckpointError("checkpoint index total_size does not match pinned shard payloads")
            self._tensors.update(parsed_tensors)
            self._observed_tensor_bytes = observed_total
            self._file_identities[filename] = file_identity
            self._validated_shards.add(filename)
            self._loaded_shards.add(filename)

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
            filename = self._weight_map.get(name)
        if filename is None:
            raise KeyError(f"tensor is not in checkpoint index: {name}")
        self._load_shard_header(filename)
        with self._lock:
            if self._closed:
                raise RuntimeError("checkpoint reader is closed")
        try:
            return self._tensors[name]
        except KeyError as exc:
            raise CheckpointError(f"checkpoint shard omitted indexed tensor: {name}") from exc

    @staticmethod
    def _pread_exact(fd: int, nbytes: int, offset: int) -> bytes:
        result = bytearray(nbytes)
        SafeTensorCheckpoint._pread_into(fd, memoryview(result), offset)
        return bytes(result)

    @staticmethod
    def _pread_into(fd: int, destination: memoryview, offset: int) -> None:
        done = 0
        while done < len(destination):
            chunk = os.pread(fd, min(_READ_CHUNK_BYTES, len(destination) - done), offset + done)
            if not chunk:
                raise CheckpointError("unexpected end of safetensors payload")
            destination[done : done + len(chunk)] = chunk
            done += len(chunk)

    def _open_verified(self, info: TensorInfo) -> int:
        if self._metadata_only:
            raise CheckpointError("checkpoint payload reads are disabled in metadata-only mode")
        filename = info.filename
        fd, file_stat = self._open_child(filename, "checkpoint shard")
        try:
            with self._lock:
                expected_identity = self._file_identities.get(filename)
                is_validated = filename in self._validated_shards
            if not is_validated or expected_identity is None:
                raise CheckpointError(f"checkpoint shard metadata was not validated: {filename}")
            if _file_identity(file_stat) != expected_identity:
                raise CheckpointError(f"checkpoint shard changed since metadata validation: {filename}")
            return fd
        except BaseException:
            os.close(fd)
            raise

    def _check_unchanged(self, fd: int, info: TensorInfo) -> None:
        if _file_identity(os.fstat(fd)) != self._file_identities[info.filename]:
            raise CheckpointError(f"checkpoint shard changed during tensor read: {info.filename}")

    def read_tensor_bytes(self, name: str, *, max_bytes: int = _DEFAULT_READ_BYTES) -> bytes:
        """Read one complete tensor only when it fits within the hard byte limit."""
        self._validate_limit(max_bytes)
        info = self._tensor(name)
        if info.nbytes > max_bytes:
            raise ValueError(f"tensor read of {info.nbytes} bytes exceeds limit of {max_bytes} bytes")
        reservation = max(2 * _CHUNK_SIZE, 2 * info.nbytes + 2 * _CHUNK_SIZE)
        with self._read_memory_budget.reserve(reservation):
            fd = self._open_verified(info)
            try:
                if info.nbytes == 0:
                    self._check_unchanged(fd, info)
                    return b""
                result = bytearray(info.nbytes)
                cursor = info.absolute_offset
                result_offset = 0
                end = cursor + info.nbytes
                while cursor < end:
                    chunk_index = cursor // _CHUNK_SIZE
                    chunk = self._read_verified_chunk(
                        fd, info.filename, chunk_index, self._file_identities[info.filename]
                    )
                    chunk_offset = chunk_index * _CHUNK_SIZE
                    source_offset = cursor - chunk_offset
                    amount = min(end - cursor, len(chunk) - source_offset)
                    result[result_offset : result_offset + amount] = memoryview(chunk)[
                        source_offset : source_offset + amount
                    ]
                    cursor += amount
                    result_offset += amount
                self._check_unchanged(fd, info)
                return bytes(result)
            finally:
                os.close(fd)

    def read_rows(
        self,
        name: str,
        rows: Sequence[int],
        *,
        max_bytes: int = _DEFAULT_READ_BYTES,
    ) -> bytes:
        """Read bounded axis-zero rows from a list, tuple, or range."""
        self._validate_limit(max_bytes)
        info = self._tensor(name)
        if not info.shape:
            raise ValueError(f"tensor {name} is scalar and has no rows")
        row_count = info.shape[0]
        row_bytes = info.row_bytes
        if type(rows) not in (list, tuple, range):
            raise TypeError("rows must be a built-in list, tuple, or range")
        if type(rows) is list:
            # Bound the snapshot even if a caller mutates the list concurrently.
            rows = rows[: _MAX_ROWS_PER_READ + 1]
        try:
            requested_count = len(rows)
        except OverflowError as exc:
            raise ValueError(f"row read exceeds hard limit of {_MAX_ROWS_PER_READ} rows") from exc
        if requested_count > _MAX_ROWS_PER_READ:
            raise ValueError(f"row read exceeds hard limit of {_MAX_ROWS_PER_READ} rows")
        if not requested_count:
            return b""
        output_size = requested_count * row_bytes
        if output_size > max_bytes:
            raise ValueError(f"row read exceeds limit of {max_bytes} bytes")
        reservation = _ROW_INDEX_RESERVATION_BYTES + 2 * output_size + 2 * _CHUNK_SIZE
        with self._read_memory_budget.reserve(reservation):
            positions: dict[int, list[int]] = {}
            for position, row in enumerate(rows):
                if type(row) is not int or row < 0 or row >= row_count:
                    raise IndexError(f"row index out of range for {name}: {row!r}")
                positions.setdefault(row, []).append(position)

            output = bytearray(output_size)
            fd = self._open_verified(info)
            try:
                if row_bytes:
                    cached_chunk_index = -1
                    cached_chunk = b""
                    for row_index in sorted(positions):
                        row_file_start = info.absolute_offset + row_index * row_bytes
                        row_file_end = row_file_start + row_bytes
                        cursor = row_file_start
                        while cursor < row_file_end:
                            chunk_index = cursor // _CHUNK_SIZE
                            if chunk_index != cached_chunk_index:
                                cached_chunk = self._read_verified_chunk(
                                    fd,
                                    info.filename,
                                    chunk_index,
                                    self._file_identities[info.filename],
                                )
                                cached_chunk_index = chunk_index
                            chunk_file_start = chunk_index * _CHUNK_SIZE
                            source_start = cursor - chunk_file_start
                            amount = min(row_file_end - cursor, len(cached_chunk) - source_start)
                            if amount <= 0:
                                raise CheckpointError(
                                    f"invalid chunk range while reading tensor rows: {info.filename}"
                                )
                            source_end = source_start + amount
                            row_data_offset = cursor - row_file_start
                            chunk_view = memoryview(cached_chunk)
                            for output_row in positions[row_index]:
                                output_start = output_row * row_bytes + row_data_offset
                                output[output_start : output_start + amount] = chunk_view[
                                    source_start:source_end
                                ]
                            cursor += amount
                self._check_unchanged(fd, info)
                return bytes(output)
            finally:
                os.close(fd)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            root_fd, self._root_fd = self._root_fd, -1
        if root_fd >= 0:
            os.close(root_fd)

    def __enter__(self) -> SafeTensorCheckpoint:
        with self._lock:
            if self._closed:
                raise RuntimeError("checkpoint reader is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
