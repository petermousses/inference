"""Persistent ownership evidence. Request age never expires a live owner."""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path


def boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except FileNotFoundError:
        # Portable fixtures; production Linux always exposes the kernel boot ID.
        import subprocess
        return subprocess.check_output(["sysctl", "-n", "kern.boottime"], text=True).strip()


def process_start(pid: int) -> str | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[19]
    except FileNotFoundError:
        if Path("/proc").exists():
            return None
        import subprocess
        result = subprocess.run(["ps", "-p", str(pid), "-o", "lstart="], capture_output=True, text=True)
        return result.stdout.strip() or None


def atomic_json(path: Path, body: dict) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(body, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


class OwnedRequest:
    def __init__(self, directory: Path, resource: str, request_id: str | None = None, job: dict | None = None):
        self.directory, self.resource = Path(directory), resource
        self.id = request_id or uuid.uuid4().hex
        if not self.id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for c in self.id) or self.id in {".", ".."}:
            raise ValueError("invalid ownership id")
        if job and job.keys() & {"version", "id", "request_id", "resource", "boot_id", "pid", "process_start"}:
            raise ValueError("cannot override request ownership")
        self.path = self.directory / f"{self.id}.json"
        self.record = {"version": 1, "id": self.id, "request_id": self.id, "resource": resource, "boot_id": boot_id(), "pid": os.getpid(), "process_start": process_start(os.getpid()), "phase": "active", **(job or {})}

    def begin(self):
        atomic_json(self.path, self.record)
        return self

    def update(self, **fields):
        if fields.keys() & {"version", "id", "request_id", "resource", "boot_id", "pid", "process_start"}:
            raise ValueError("cannot change request ownership")
        self.record.update(fields)
        atomic_json(self.path, self.record)

    def close(self):
        self.path.unlink(missing_ok=True)

    def detach(self):
        """Record that the caller ended while owned backend work still exists."""
        self.update(phase="abandoned")


def records(directory: Path, resource: str | None = None) -> list[dict]:
    result = []
    for path in Path(directory).glob("*.json"):
        if path.is_symlink():
            continue
        try:
            if path.stat().st_size > 64 * 1024:
                continue
            body = json.loads(path.read_text())
            if not isinstance(body, dict) or body.get("version") != 1 or path.name != f"{body.get('id')}.json":
                continue
            if resource is None or body.get("resource") == resource:
                result.append(body)
        except (OSError, ValueError):
            continue
    return result


def abandoned(directory: Path, resource: str) -> list[dict]:
    result = []
    current_boot = boot_id()
    for body in records(directory, resource):
        pid = body.get("pid")
        if not isinstance(pid, int) or isinstance(pid, bool) or pid < 1 or not body.get("process_start"):
            continue
        try:
            if body.get("phase") == "abandoned" or body.get("boot_id") != current_boot or process_start(pid) != body["process_start"]:
                result.append(body)
        except (OSError, ValueError, IndexError):
            # Uncertain process visibility cannot authorize cancellation.
            continue
    return result


def remove_record(directory: Path, request_id: str) -> None:
    if request_id and all(c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for c in request_id) and request_id not in {".", ".."}:
        (Path(directory) / f"{request_id}.json").unlink(missing_ok=True)
