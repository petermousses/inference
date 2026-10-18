#!/usr/bin/env python3
"""Dependency-free recovery helper, copied outside the mutable source checkout.

No candidate application import is allowed here: this program must still run
when a new checkout, application dependency, or model configuration is broken.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from contextvars import ContextVar
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import stat
import subprocess
import tempfile
import time
import urllib.error
import urllib.request


FRONT_URLS = {
    "llama-swap.service": "http://127.0.0.1:8084/health",
    "redqueen-scheduler.service": "http://10.9.20.242:8081/healthz",
    "qwen-image-adapter.service": "http://10.9.20.242:8190/healthz",
    "jevk5-adapter.service": "http://10.9.20.242:8191/healthz",
}
BACKENDS = ("comfyui.service", "jevk5.service", "clm-api.service", "clm-encoder.service", "comfyui-control.service", "qwen38.service")
SUPPORT = ("redqueen-preflight.service", "redqueen-supervisor.service")
ALL_UNITS = tuple(FRONT_URLS) + BACKENDS + SUPPORT
_OPERATION_FD: ContextVar[int | None] = ContextVar("redqueen_operation_fd", default=None)


def inherited_operation_fds() -> tuple[int, ...]:
    descriptor = _OPERATION_FD.get()
    return (descriptor,) if descriptor is not None else ()


def command(*arguments: str, check: bool = True, timeout: float = 35):
    process = subprocess.Popen(arguments, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, start_new_session=True,
                               pass_fds=inherited_operation_fds())
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as original:
        # A shell's children may own inherited maintenance locks or pipes.
        # Stopping only the shell leaves recovery blocked indefinitely. These
        # groups contain control tools, never systemd-managed inference jobs.
        def stop_group(sig):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
        stop_group(signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            stop_group(signal.SIGKILL)
            try:
                stdout, stderr = process.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                for stream in (process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()
                raise subprocess.TimeoutExpired(arguments, timeout, original.output, original.stderr) from original
        raise subprocess.TimeoutExpired(arguments, timeout, stdout, stderr) from original
    result = subprocess.CompletedProcess(arguments, process.returncode, stdout, stderr)
    if check:
        result.check_returncode()
    return result


def systemctl(*arguments: str, check: bool = True):
    return command("/usr/bin/systemctl", "--user", *arguments, check=check)


def atomic_write(path: Path, body: bytes, mode: int = 0o600) -> None:
    """Commit a durable file before making its name visible."""
    path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            os.fchmod(stream.fileno(), mode)
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_json(path: Path, document: dict) -> None:
    atomic_write(path, (json.dumps(document, indent=2) + "\n").encode())


def durable_unlink(path: Path) -> None:
    path.unlink(missing_ok=True)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class MaintenanceBusy(RuntimeError):
    pass


@contextmanager
def operation_lock(directory: Path, *, wait: bool = False):
    directory.mkdir(mode=0o750, parents=True, exist_ok=True)
    descriptor = os.open(directory / "operations.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError as error:
            raise MaintenanceBusy("deployment or recovery is in progress") from error
        token = _OPERATION_FD.set(descriptor)
        try:
            yield descriptor
        finally:
            _OPERATION_FD.reset(token)
    finally:
        os.close(descriptor)


def leases_held(directory: Path) -> bool:
    """Only flock ownership matters; file timestamps never expire work."""
    for path in sorted((directory / "leases").glob("*")):
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            # Queue tickets can be removed while a recovery scan is underway.
            continue
        if stat.S_ISDIR(mode):
            # SerializedExecution stores its FIFO tickets in this child
            # directory. It is not itself a request lease.
            continue
        if not stat.S_ISREG(mode):
            raise RuntimeError(f"unexpected non-regular lease entry: {path}")
        try:
            descriptor = os.open(
                path, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK
            )
        except FileNotFoundError:
            continue
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise RuntimeError(f"lease entry changed type while scanning: {path}")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
        finally:
            os.close(descriptor)
    return False


def boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return "unknown"


def begin_journal(directory: Path, repo: Path, previous: str, enabled: dict,
                  active: list[str], profile_path: Path, backup: Path | None,
                  preflight_definition: dict | None = None) -> dict:
    document = {
        "version": 1, "phase": "prepared", "repo": str(repo), "previous": previous,
        "enabled": enabled, "active": active, "profile_path": str(profile_path),
        "profile_backup": str(backup) if backup is not None else None,
        "pid": os.getpid(), "boot_id": boot_id(), "started_at": time.time(),
        "preflight_definition": preflight_definition,
    }
    atomic_json(directory / "deployment.json", document)
    return document


def deployment_phase(directory: Path, document: dict, phase: str) -> None:
    document["phase"] = phase
    atomic_json(directory / "deployment.json", document)


def finish_deployment(directory: Path, repo: Path, revision: str) -> None:
    atomic_json(directory / "last-good.json", {
        "version": 1, "repo": str(repo), "revision": revision,
        "completed_at": time.time(), "boot_id": boot_id(),
    })
    durable_unlink(directory / "deployment.json")


def install_helper(source: Path, target: Path, *, bootstrap_only: bool = False) -> None:
    """Install a regular copy; application linker symlinks must never own this."""
    if bootstrap_only and target.is_file() and not target.is_symlink():
        return
    if target.is_symlink():
        raise RuntimeError(f"recovery helper must be a regular file: {target}")
    body = source.read_bytes()
    compile(body, str(source), "exec")
    atomic_write(target, body, 0o750)


def preflight_unit_body(root: Path) -> bytes:
    return ("[Unit]\n"
            "Description=Recover interrupted Redqueen deployments and validate unattended startup\n"
            "After=network-online.target\nWants=network-online.target\n"
            "Before=llama-swap.service redqueen-scheduler.service qwen-image-adapter.service jevk5-adapter.service default.target\n"
            "\n[Service]\nType=oneshot\nRemainAfterExit=yes\n"
            "Environment=PYTHONDONTWRITEBYTECODE=1\nEnvironment=XDG_RUNTIME_DIR=%t\n"
            "Environment=DBUS_SESSION_BUS_ADDRESS=unix:path=%t/bus\n"
            f"ExecStart=/usr/bin/python3 {root}/model-router/libexec/host_reliability.py boot\n"
            "TimeoutStartSec=240\nUMask=0077\nNoNewPrivileges=true\n"
            "\n[Install]\nWantedBy=default.target\n").encode()


def preflight_snapshot() -> dict:
    path = Path.home() / ".config/systemd/user/redqueen-preflight.service"
    if path.is_symlink():
        return {"kind": "symlink", "target": os.readlink(path)}
    if path.exists():
        if not path.is_file() or path.stat().st_size > 64 * 1024:
            raise RuntimeError("existing preflight unit is not a bounded regular unit file")
        return {"kind": "file", "body": path.read_text()}
    return {"kind": "absent"}


def install_preflight_unit(root: Path, *, source: Path | None = None, bootstrap_only: bool = False) -> None:
    path = Path.home() / ".config/systemd/user/redqueen-preflight.service"
    if bootstrap_only and path.is_file() and not path.is_symlink():
        body = path.read_text()
        if f"ExecStart=/usr/bin/python3 {root}/model-router/libexec/host_reliability.py boot" not in body:
            raise RuntimeError("existing preflight unit is not the independent recovery unit")
        return
    body = source.read_bytes() if source is not None else preflight_unit_body(root)
    atomic_write(path, body, 0o644)


def bootstrap_preflight(root: Path) -> None:
    """Wire independent boot recovery before the first mutable git pull."""
    install_preflight_unit(root, bootstrap_only=True)
    systemctl("daemon-reload")
    systemctl("enable", "redqueen-preflight.service")


def restore_preflight(definition: dict | None) -> None:
    if definition is None:
        return
    path = Path.home() / ".config/systemd/user/redqueen-preflight.service"
    kind = definition.get("kind")
    if kind == "absent":
        for target in (path, path.parent / "default.target.wants/redqueen-preflight.service"):
            if target.exists() or target.is_symlink():
                durable_unlink(target)
    elif kind == "file":
        atomic_write(path, definition["body"].encode(), 0o644)
    elif kind == "symlink":
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(definition["target"])
        os.replace(temporary, path)
    else:
        raise RuntimeError("deployment journal contains an invalid preflight definition")


def safe_directory(path: Path, mode: int = 0o750) -> None:
    if path.is_symlink():
        raise RuntimeError(f"refusing symlink for transient directory: {path}")
    path.mkdir(mode=mode, parents=True, exist_ok=True)
    if not path.is_dir() or path.stat().st_uid != os.geteuid():
        raise RuntimeError(f"transient directory must belong to ai: {path}")
    if mode == 0o700:
        path.chmod(mode)


def transient_directories(root: Path) -> None:
    safe_directory(root / "model-router/run")
    safe_directory(root / "model-router/run/leases", 0o700)
    safe_directory(root / "model-router/run/requests", 0o700)
    for relative in ("input", "output", "temp", "user/default/workflows"):
        safe_directory(root / "comfyui/state" / relative)


def profile_errors(path: Path) -> list[str]:
    try:
        document = json.loads(path.read_text())
        if (
            document.get("version") != 1
            or document.get("worker_profile_version") != 1
            or not document.get("resources")
            or not document.get("sources")
        ):
            return ["memory profiles are invalid; rebuild with deploy-scheduler.sh"]
        for name, profile in document["resources"].items():
            if not isinstance(name, str) or not name or not isinstance(profile, dict):
                return ["memory profile resource is invalid; rebuild with deploy-scheduler.sh"]
            memory = profile.get("memory")
            if not isinstance(memory, dict):
                return [f"memory profile is missing worker contract for {name}"]
            totals = {"gpu_bytes": 0, "host_bytes": 0}
            for component in ("mandatory", "reclaimable", "transient_peak"):
                values = memory.get(component)
                if not isinstance(values, dict):
                    return [f"memory profile has invalid {component} memory for {name}"]
                for field in totals:
                    amount = values.get(field)
                    if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
                        return [f"memory profile has invalid {component} memory for {name}"]
                    totals[field] += amount
            if any(
                isinstance(profile.get(field), bool)
                or not isinstance(profile.get(field), int)
                or profile.get(field) != amount
                for field, amount in totals.items()
            ):
                return [f"memory profile footprint does not match its components for {name}"]
            fingerprint = profile.get("artifact_fingerprint")
            identities = profile.get("artifact_identity")
            shapes = profile.get("supported_request_shapes")
            concurrency = profile.get("concurrency_limit")
            if (
                not isinstance(identities, list)
                or not identities
                or not isinstance(fingerprint, str)
                or len(fingerprint) != 64
                or any(char not in "0123456789abcdef" for char in fingerprint)
            ):
                return [f"memory profile has invalid artifact identity for {name}"]
            artifact_names = []
            for identity in identities:
                if not isinstance(identity, dict):
                    return [f"memory profile has invalid artifact identity for {name}"]
                model = identity.get("model")
                repositories, revisions = identity.get("repositories"), identity.get("revisions")
                manifest_hash = identity.get("manifest_sha256")
                if (
                    not isinstance(model, str) or not model
                    or not isinstance(repositories, dict) or not repositories
                    or not isinstance(revisions, dict) or not revisions
                    or any(not isinstance(value, str) or not value for value in (*repositories.values(), *revisions.values()))
                    or not isinstance(manifest_hash, str) or len(manifest_hash) != 64
                    or any(char not in "0123456789abcdef" for char in manifest_hash)
                ):
                    return [f"memory profile has invalid artifact identity for {name}"]
                artifact_names.append(model)
            if artifact_names != sorted(set(artifact_names)):
                return [f"memory profile has duplicate artifact identities for {name}"]
            expected_fingerprint = hashlib.sha256(
                json.dumps(identities, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if expected_fingerprint != fingerprint:
                return [f"memory profile artifact fingerprint does not match for {name}"]
            runtime_identity = profile.get("runtime_identity")
            if not isinstance(runtime_identity, dict):
                return [f"memory profile has no runtime identity for {name}"]
            if "path" in runtime_identity:
                if (
                    not isinstance(runtime_identity.get("path"), str)
                    or isinstance(runtime_identity.get("size"), bool)
                    or not isinstance(runtime_identity.get("size"), int)
                    or runtime_identity["size"] < 0
                    or isinstance(runtime_identity.get("mtime_ns"), bool)
                    or not isinstance(runtime_identity.get("mtime_ns"), int)
                ):
                    return [f"memory profile has invalid runtime identity for {name}"]
            else:
                units = runtime_identity.get("service_units")
                if not isinstance(units, list) or not units or any(not isinstance(unit, str) or not unit for unit in units):
                    return [f"memory profile has invalid service runtime identity for {name}"]
            if (
                not isinstance(shapes, list)
                or not shapes
                or any(not isinstance(shape, str) or not shape for shape in shapes)
                or len(set(shapes)) != len(shapes)
            ):
                return [f"memory profile has invalid request shapes for {name}"]
            if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
                return [f"memory profile has invalid concurrency for {name}"]
        for expected in document["sources"]:
            source = Path(expected["path"])
            stat = source.stat()
            actual = {"path": str(source), "size": stat.st_size}
            if "sha256" in expected:
                digest = hashlib.sha256()
                with source.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                actual["sha256"] = digest.hexdigest()
            else:
                actual["mtime_ns"] = stat.st_mtime_ns
            if actual != expected:
                return [f"memory profiles are stale: {source}; rebuild with deploy-scheduler.sh"]
        return []
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        return [f"memory profile validation failed: {type(error).__name__}"]


def front_requirements(root: Path, unit: str) -> list[Path]:
    router = [root / "model-router/config.yaml", root / "secrets/llama-swap.env"]
    return {
        "llama-swap.service": router + [root / "runtimes/llama-swap-v260-unbounded/bin/llama-swap", root / "secrets/qwen38-api-keys"],
        "redqueen-scheduler.service": router + [root / "model-router/venv/bin/python", root / "model-router/run/memory-profiles.json", root / "adapters/source/redqueen_adapters/scheduler_api.py"],
        "qwen-image-adapter.service": [root / "adapters/venv/bin/python", root / "adapters/source/redqueen_adapters/image_api.py", root / "secrets/redqueen-adapters.env"],
        "jevk5-adapter.service": [root / "adapters/venv/bin/python", root / "adapters/source/redqueen_adapters/jev_api.py", root / "secrets/redqueen-adapters.env"],
    }[unit]


def credential_errors(path: Path, required: tuple[str, ...]) -> list[str]:
    """Validate EnvironmentFile literals without ever executing or logging them."""
    try:
        info = path.stat()
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
            return [f"{path}: credentials must belong to ai and have mode 0600"]
        if info.st_size > 64 * 1024:
            return [f"{path}: credential file exceeds 64 KiB"]
        values = {}
        for number, line in enumerate(path.read_text().splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            name, separator, literal = line.partition("=")
            name = name.strip()
            if not separator or not name or not (name[0].isalpha() or name[0] == "_") or not all(c.isalnum() or c == "_" for c in name):
                return [f"{path}: invalid credential declaration on line {number}"]
            try:
                tokens = shlex.split(literal, comments=False, posix=True)
            except ValueError:
                return [f"{path}: invalid quoting on line {number}"]
            if len(tokens) > 1:
                return [f"{path}: invalid credential value on line {number}"]
            values[name] = tokens[0] if tokens else ""
        return [f"{path}: missing or empty {name}" for name in required if not values.get(name, "").strip()]
    except (OSError, UnicodeError):
        return [f"{path}: credential file is missing or unreadable"]


def preflight(root: Path, *, check_dependencies: bool = True) -> list[str]:
    errors = []
    try:
        transient_directories(root)
    except (OSError, RuntimeError) as error:
        errors.append(str(error))
    for unit in FRONT_URLS:
        for path in front_requirements(root, unit):
            if not path.is_file() or not os.access(path, os.R_OK) or path.stat().st_size == 0:
                errors.append(f"{unit}: missing or unreadable {path}")
        try:
            result = systemctl("is-enabled", unit, check=False)
            if result.returncode != 0 or result.stdout.strip() not in {"enabled", "enabled-runtime"}:
                errors.append(f"{unit}: enable the unit for unattended startup")
        except (OSError, subprocess.SubprocessError):
            errors.append(f"{unit}: user service manager is unavailable")
    for unit in BACKENDS:
        try:
            if systemctl("show", unit, "--property=LoadState", "--value", check=False).stdout.strip() != "loaded":
                errors.append(f"{unit}: unit definition is missing; run link-runtime.sh")
            if systemctl("is-enabled", unit, check=False).stdout.strip() in {"enabled", "enabled-runtime"}:
                errors.append(f"{unit}: disable its boot start; model backends require scheduler admission")
        except (OSError, subprocess.SubprocessError):
            errors.append(f"{unit}: cannot inspect model backend unit")
    try:
        if command("loginctl", "show-user", str(os.geteuid()), "--property=Linger", "--value").stdout.strip() != "yes":
            errors.append("user lingering is disabled; run sudo loginctl enable-linger ai")
    except (OSError, subprocess.SubprocessError):
        errors.append("cannot verify user lingering")
    errors.extend(profile_errors(root / "model-router/run/memory-profiles.json"))
    errors.extend(credential_errors(root / "secrets/llama-swap.env", ("QWEN_API_KEY",)))
    errors.extend(credential_errors(root / "secrets/redqueen-adapters.env", ("REDQUEEN_IMAGE_API_KEY", "REDQUEEN_JEV_API_KEY", "CLM_API_KEY", "QWEN_API_KEY")))
    for relative in ("comfyui/source/main.py", "comfyui/venv/bin/python", "clm/venv/bin/clm-serve"):
        if not (root / relative).is_file():
            errors.append(f"model runtime is missing: {root / relative}")
    if check_dependencies:
        # Check isolated dependency pins in their own interpreters, without
        # importing any code from the mutable application checkout.
        program = ("import importlib.metadata as m,pathlib,sys; "
                   "pins=pathlib.Path(sys.argv[1]).read_text().splitlines(); "
                   "assert all(m.version(p.split('==')[0])==p.split('==')[1] "
                   "for p in pins if p.strip() and not p.startswith('#'))")
        for relative in ("model-router", "adapters"):
            try:
                command(str(root / relative / "venv/bin/python"), "-c", program,
                        str(root / relative / "requirements.lock"), timeout=15)
            except (OSError, subprocess.SubprocessError):
                errors.append(f"{relative}: runtime dependencies do not match requirements.lock")
    atomic_json(root / "model-router/run/boot-status.json", {
        "checked_at": time.time(), "boot_id": boot_id(), "ready": not errors, "errors": errors,
    })
    return errors


def restore_journal(root: Path, document: dict) -> None:
    """Restore using only stdlib/git/systemd and the last good linker."""
    if document.get("version") != 1:
        raise RuntimeError("unsupported deployment journal")
    repo = Path(document["repo"])
    if repo != root / "inference":
        raise RuntimeError("deployment journal points outside the inference checkout")
    previous = document["previous"]
    if not isinstance(previous, str) or len(previous) != 40 or any(c not in "0123456789abcdef" for c in previous):
        raise RuntimeError("deployment journal has an invalid rollback revision")
    if not set(document["enabled"]).issubset(ALL_UNITS) or not set(document["active"]).issubset(ALL_UNITS):
        raise RuntimeError("deployment journal contains unmanaged units")
    profile = Path(document["profile_path"])
    backup = Path(document["profile_backup"]) if document["profile_backup"] else None
    directory = root / "model-router/run"
    if profile.parent != directory or (backup is not None and backup.parent != directory):
        raise RuntimeError("deployment journal profile paths are invalid")
    if command("git", "-C", str(repo), "status", "--porcelain").stdout.strip():
        raise RuntimeError("interrupted deployment checkout is dirty; manual recovery required")
    command("git", "-C", str(repo), "cat-file", "-e", f"{previous}^{{commit}}")
    # Do not discard an irreplaceable profile backup after stopping services.
    old_profile = backup.read_bytes() if backup is not None else None
    # Recovery may run inside either support service itself. Never ask its
    # service manager to stop the process currently restoring the host.
    systemctl("stop", *FRONT_URLS, *BACKENDS)
    command("git", "-C", str(repo), "switch", "--detach", previous)
    for unit in ("redqueen-scheduler.service", *SUPPORT):
        if not (repo / "hosts/redqueen/systemd" / unit).exists():
            for path in (Path.home() / ".config/systemd/user" / unit, Path.home() / ".config/systemd/user/default.target.wants" / unit):
                if path.is_symlink():
                    path.unlink()
    linker = repo / "hosts/redqueen/link-runtime.sh"
    command("bash", str(linker), timeout=180)
    restore_preflight(document.get("preflight_definition"))
    if old_profile is None:
        durable_unlink(profile)
    else:
        atomic_write(profile, old_profile)
    for unit, enabled in document["enabled"].items():
        if (repo / "hosts/redqueen/systemd" / unit).exists() or (unit == "redqueen-preflight.service" and (document.get("preflight_definition") or {}).get("kind") in {"file", "symlink"}):
            systemctl("enable" if enabled else "disable", unit)
    command("bash", str(linker), timeout=180)
    restore_preflight(document.get("preflight_definition"))
    systemctl("daemon-reload")
    if document["active"]:
        # A preflight unit may itself be performing this boot recovery. Queue
        # starts instead of waiting for its own After= dependency to finish.
        systemctl("start", "--no-block", *document["active"])
    atomic_json(directory / "recovery.json", {
        "state": "rolled_back", "revision": previous, "at": time.time(), "boot_id": boot_id(),
    })
    atomic_json(directory / "last-good.json", {
        "version": 1, "repo": str(repo), "revision": previous, "rolled_back_at": time.time(), "boot_id": boot_id(),
    })
    durable_unlink(directory / "deployment.json")


def recover_pending(root: Path) -> bool:
    directory = root / "model-router/run"
    journal = directory / "deployment.json"
    if not journal.exists():
        return False
    with operation_lock(directory):
        if not journal.exists():
            return False
        if leases_held(directory):
            raise MaintenanceBusy("interrupted deployment recovery deferred for active requests")
        document = json.loads(journal.read_text())
        atomic_json(directory / "recovery.json", {"state": "recovering", "at": time.time()})
        try:
            restore_journal(root, document)
        except Exception as error:
            atomic_json(directory / "recovery.json", {
                "state": "quarantined", "at": time.time(), "reason": str(error),
            })
            raise
    return True


def front_alive(item: tuple[str, str]) -> bool:
    unit, url = item
    try:
        state = systemctl("is-active", unit, check=False)
        if state.returncode != 0:
            return False
        # Never use readiness here: missing GPU/model/router dependencies
        # cannot be fixed by restarting an otherwise responsive HTTP server.
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError, subprocess.SubprocessError):
        return False


def supervise_once(root: Path, state: dict, *, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    directory = root / "model-router/run"
    try:
        recover_pending(root)
        state.pop("recovery_error", None)
    except MaintenanceBusy:
        pass
    except Exception as error:
        state["recovery_error"] = str(error)
        # A failed rollback must not be followed by blindly restarting the
        # candidate services. Leave the journal intact for operator diagnosis.
        atomic_json(directory / "supervisor.json", state)
        return state
    with ThreadPoolExecutor(max_workers=len(FRONT_URLS)) as executor:
        checks = list(executor.map(front_alive, FRONT_URLS.items()))
    maintenance_path = directory / "maintenance.json"
    maintenance = None
    if maintenance_path.exists():
        maintenance = json.loads(maintenance_path.read_text())
        unit = maintenance.get("unit")
        if unit not in FRONT_URLS:
            raise RuntimeError("maintenance marker contains an unmanaged unit")
        # A timed-out systemctl client does not cancel its queued systemd job.
        # Keep new admissions blocked until that exact front is responsive and
        # the job has completed, even across supervisor crashes/reboots.
        alive = checks[list(FRONT_URLS).index(unit)]
        try:
            with operation_lock(directory):
                job = systemctl("show", unit, "--property=Job", "--value")
                finished = False
                if job.stdout.strip() in {"", "0"}:
                    finished = alive
                    if not alive:
                        details = systemctl("show", unit, "--property=ActiveState", "--property=MainPID")
                        properties = dict(line.split("=", 1) for line in details.stdout.splitlines() if "=" in line)
                        # A boot discards old jobs. A definitively stopped
                        # front cannot later execute that abandoned restart;
                        # release global maintenance and recover it normally.
                        finished = properties.get("ActiveState") in {"inactive", "failed"} and properties.get("MainPID") == "0"
                if finished:
                    durable_unlink(maintenance_path)
                    maintenance = None
        except (MaintenanceBusy, OSError, subprocess.SubprocessError):
            pass
    state["maintenance"] = maintenance
    for (unit, _), alive in zip(FRONT_URLS.items(), checks):
        info = state.setdefault("fronts", {}).setdefault(unit, {"failures": 0, "next_retry": 0})
        info["checked_at"] = now
        if alive:
            info.update(failures=0, state="ready", reason=None)
            continue
        info["failures"] += 1
        if now < info["next_retry"]:
            continue
        info["state"] = "unhealthy"
        if info["failures"] < 3:
            continue
        if maintenance is not None and maintenance["unit"] != unit:
            info.update(state="deferred", reason="waiting for another front's pending recovery job")
            continue
        try:
            with operation_lock(directory):
                if (directory / "deployment.json").exists():
                    raise MaintenanceBusy("interrupted deployment must recover before fronts restart")
                if leases_held(directory):
                    raise MaintenanceBusy("recovery deferred for active inference leases")
                missing = [str(path) for path in front_requirements(root, unit) if not path.is_file()]
                if missing:
                    info.update(state="quarantined", reason="missing runtime assets: " + ", ".join(missing), next_retry=now + 60)
                    continue
                maintenance = {"unit": unit, "state": "recovering", "reason": "three consecutive liveness failures", "boot_id": boot_id(), "at": now}
                atomic_json(maintenance_path, maintenance)
                systemctl("reset-failed", unit, check=False)
                systemctl("restart", unit)
                info.update(state="recovering", reason="three consecutive liveness failures", next_retry=now + 60, failures=0)
        except MaintenanceBusy as error:
            info.update(state="deferred", reason=str(error))
        except (OSError, subprocess.SubprocessError) as error:
            info.update(state="cooling_down", reason=f"service control failed: {type(error).__name__}", next_retry=now + 60)
    state["checked_at"] = now
    state["boot_id"] = boot_id()
    state["maintenance"] = maintenance
    atomic_json(directory / "supervisor.json", state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("boot", "recover", "supervise", "install", "unit"))
    parser.add_argument("--root", type=Path, default=Path("/srv/ai"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--bootstrap-only", action="store_true")
    args = parser.parse_args()
    if os.geteuid() == 0:
        raise SystemExit("run this as ai")
    root = args.root
    transient_directories(root)
    if args.action == "install":
        if args.source is None:
            parser.error("install requires --source")
        install_helper(args.source, root / "model-router/libexec/host_reliability.py", bootstrap_only=args.bootstrap_only)
    elif args.action == "unit":
        install_preflight_unit(root, source=args.source, bootstrap_only=args.bootstrap_only)
    elif args.action in {"boot", "recover"}:
        try:
            recover_pending(root)
        except MaintenanceBusy as error:
            print(str(error), flush=True)
            # A live deployment owns recovery; never block its service starts.
            return
        if args.action == "boot":
            errors = preflight(root)
            if errors:
                raise SystemExit("boot preflight: " + "; ".join(errors))
    else:
        state = {}
        while True:
            started = time.monotonic()
            try:
                state = supervise_once(root, state)
            except Exception as error:
                print(f"recovery supervisor failed: {type(error).__name__}: {error}", flush=True)
            time.sleep(max(0, 15 - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
