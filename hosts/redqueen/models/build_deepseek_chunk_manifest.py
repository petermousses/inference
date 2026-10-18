#!/usr/bin/env python3
"""Build pinned DeepSeek shard chunk hashes after verifying the full file inventory."""

from __future__ import annotations

import argparse
import hashlib
import os
import secrets
import stat
import sys
from pathlib import Path


CHUNK_BYTES = 1024 * 1024
MAX_INVENTORY_BYTES = 8 * 1024 * 1024


def load_inventory(path: Path) -> list[tuple[str, int, str]]:
    result = []
    seen = set()
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise RuntimeError("platform does not support no-follow inventory opens")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0) | nofollow
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_INVENTORY_BYTES:
            raise ValueError("inventory must be a regular file no larger than 8 MiB")
        raw = bytearray()
        while len(raw) <= MAX_INVENTORY_BYTES:
            part = os.read(fd, min(1024 * 1024, MAX_INVENTORY_BYTES + 1 - len(raw)))
            if not part:
                break
            raw.extend(part)
        after = os.fstat(fd)
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        if identity_before != identity_after or len(raw) > MAX_INVENTORY_BYTES:
            raise ValueError("inventory changed while it was read or exceeds 8 MiB")
        lines = bytes(raw).decode("utf-8").splitlines()
    finally:
        os.close(fd)
    for line_number, line in enumerate(lines, 1):
        fields = line.split("\t")
        if len(fields) != 4:
            raise ValueError(f"invalid inventory row {line_number}")
        filename, size_text, digest, source_oid = fields
        if not filename.endswith(".safetensors"):
            continue
        if (
            Path(filename).name != filename
            or filename in {".", ".."}
            or "\\" in filename
            or "\x00" in filename
            or filename in seen
        ):
            raise ValueError(f"invalid or duplicate shard name at row {line_number}")
        if not size_text.isdecimal() or len(digest) != 64:
            raise ValueError(f"invalid shard identity at row {line_number}")
        if any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"invalid SHA-256 at row {line_number}")
        if len(source_oid) != 40 or any(character not in "0123456789abcdef" for character in source_oid):
            raise ValueError(f"invalid source object id at row {line_number}")
        result.append((filename, int(size_text), digest))
        seen.add(filename)
    if not result:
        raise ValueError("inventory contains no safetensors shards")
    return sorted(result)


def build_manifest(root: Path, inventory_path: Path, output_path: Path) -> tuple[int, str]:
    root = root.resolve(strict=True)
    inventory_path = inventory_path.resolve(strict=True)
    resolved_parent = output_path.parent.resolve(strict=True)
    resolved_output = resolved_parent / output_path.name
    output_entry_paths = (
        Path(os.path.abspath(output_path)),
        resolved_output,
    )
    for output_entry in output_entry_paths:
        try:
            output_entry.relative_to(root)
        except ValueError:
            continue
        raise ValueError("output must be outside the checkpoint directory")
    if resolved_output in {inventory_path, Path(__file__).resolve()}:
        raise ValueError("output must not replace the inventory or manifest builder")
    shards = load_inventory(inventory_path)
    if not output_path.name or output_path.name in {".", ".."}:
        raise ValueError("output must name a file")

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    if not getattr(os, "O_DIRECTORY", 0) or not getattr(os, "O_NOFOLLOW", 0):
        raise RuntimeError("platform does not support safe output directory opens")
    output_dir_fd = os.open(resolved_parent, directory_flags)
    root_fd = -1
    temp_fd = -1
    temp_name = None
    total_chunks = 0
    try:
        root_fd = os.open(root, directory_flags)
        root_stat = os.fstat(root_fd)
        current_fd = os.dup(output_dir_fd)
        try:
            for _ in range(1024):
                current_stat = os.fstat(current_fd)
                if (current_stat.st_dev, current_stat.st_ino) == (root_stat.st_dev, root_stat.st_ino):
                    raise ValueError("output must be outside the checkpoint directory")
                parent_fd = os.open("..", directory_flags, dir_fd=current_fd)
                parent_stat = os.fstat(parent_fd)
                if (parent_stat.st_dev, parent_stat.st_ino) == (current_stat.st_dev, current_stat.st_ino):
                    os.close(parent_fd)
                    break
                os.close(current_fd)
                current_fd = parent_fd
            else:
                raise ValueError("output directory ancestry exceeds safety limit")
        finally:
            os.close(current_fd)

        for _ in range(16):
            candidate = f".{output_path.name}.{secrets.token_hex(16)}.tmp"
            try:
                temp_fd = os.open(
                    candidate,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                    dir_fd=output_dir_fd,
                )
                temp_name = candidate
                break
            except FileExistsError:
                continue
        else:
            raise FileExistsError("could not allocate a unique temporary manifest")

        output_hash = hashlib.sha256()
        with os.fdopen(temp_fd, "wb") as destination:
            temp_fd = -1
            for filename, expected_size, expected_digest in shards:
                nofollow = getattr(os, "O_NOFOLLOW", None)
                if nofollow is None:
                    raise RuntimeError("platform does not support no-follow checkpoint file opens")
                flags = (
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NONBLOCK", 0)
                    | nofollow
                )
                fd = os.open(filename, flags, dir_fd=root_fd)
                try:
                    before = os.fstat(fd)
                    if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
                        raise ValueError(f"checkpoint file size/type differs from inventory: {filename}")
                    file_hash = hashlib.sha256()
                    observed = 0
                    while observed < expected_size:
                        chunk_length = min(CHUNK_BYTES, expected_size - observed)
                        chunk_buffer = bytearray(chunk_length)
                        filled = 0
                        while filled < chunk_length:
                            part = os.read(fd, chunk_length - filled)
                            if not part:
                                raise ValueError(f"checkpoint file ended before expected size: {filename}")
                            chunk_buffer[filled : filled + len(part)] = part
                            filled += len(part)
                        chunk = bytes(chunk_buffer)
                        file_hash.update(chunk)
                        chunk_digest = hashlib.sha256(chunk).digest()
                        destination.write(chunk_digest)
                        output_hash.update(chunk_digest)
                        observed += len(chunk)
                        total_chunks += 1
                    after = os.fstat(fd)
                    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                    if identity_before != identity_after:
                        raise ValueError(f"checkpoint file changed while hashing: {filename}")
                    if file_hash.hexdigest() != expected_digest:
                        raise ValueError(f"checkpoint SHA-256 differs from inventory: {filename}")
                finally:
                    os.close(fd)
                print(f"verified {filename} ({expected_size} bytes)", file=sys.stderr)
            destination.flush()
            os.fchmod(destination.fileno(), 0o644)
            os.fsync(destination.fileno())
        os.replace(
            temp_name,
            output_path.name,
            src_dir_fd=output_dir_fd,
            dst_dir_fd=output_dir_fd,
        )
        temp_name = None
        os.fsync(output_dir_fd)
    finally:
        if temp_fd >= 0:
            os.close(temp_fd)
        if temp_name is not None:
            try:
                os.unlink(temp_name, dir_fd=output_dir_fd)
            except FileNotFoundError:
                pass
        if root_fd >= 0:
            os.close(root_fd)
        os.close(output_dir_fd)

    return total_chunks, output_hash.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_root", type=Path)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    count, digest = build_manifest(args.checkpoint_root, args.inventory, args.output)
    print(f"chunks={count} bytes={count * 32} sha256={digest}")


if __name__ == "__main__":
    main()
