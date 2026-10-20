"""Replace the linked DeepSeek revision marker with a verified regular file."""

from __future__ import annotations

import argparse
import os
import stat
import sys
import tempfile
from pathlib import Path


MAX_REVISION_BYTES = 64 * 1024
REVISION_MODE = 0o440


def materialize(source: Path, target: Path) -> bool:
    source = Path(source)
    target = Path(target)
    source_stat = source.lstat()
    if (
        not stat.S_ISREG(source_stat.st_mode)
        or source_stat.st_size < 1
        or source_stat.st_size > MAX_REVISION_BYTES
    ):
        raise ValueError("source revision must be a small regular file")
    source = source.resolve(strict=True)
    data = source.read_bytes()
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        target_stat = target.lstat()
    except FileNotFoundError:
        target_stat = None
    if target_stat is not None:
        if stat.S_ISLNK(target_stat.st_mode):
            try:
                resolved_target = target.resolve(strict=True)
            except OSError as error:
                raise ValueError("existing revision symlink is broken") from error
            if resolved_target != source:
                raise ValueError("existing revision symlink does not point to tracked metadata")
        elif stat.S_ISREG(target_stat.st_mode):
            if target_stat.st_size != len(data) or target.read_bytes() != data:
                raise ValueError("existing regular revision differs from tracked metadata")
            target.chmod(REVISION_MODE, follow_symlinks=False)
            return False
        else:
            raise ValueError("existing revision path is not a regular file or tracked symlink")

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, REVISION_MODE)
        with os.fdopen(fd, "wb") as output:
            fd = -1
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        temporary.unlink(missing_ok=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    try:
        materialize(args.source, args.target)
    except (OSError, ValueError) as error:
        print(f"cannot materialize DeepSeek source revision: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
