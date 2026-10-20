"""Apply the checked-out scheduler revision after an idle admission window."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import io
import json
import os
import shlex
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath

import aiohttp
import yaml

from .memory_profile import build_profiles, file_signature, verify_profiles, weight_paths
from .scheduler_api import Runtime
from .scheduler_client import ResourceLock
from . import host_reliability

FRONTS = ["qwen-image-adapter.service", "jevk5-adapter.service", "redqueen-scheduler.service", "llama-swap.service", "comfyui-control.service"]
MANAGED = ["comfyui.service", "deepseek-hip.service", "jevk5.service", "clm-api.service", "clm-encoder.service", "comfyui-control.service", "qwen38.service"]
SUPPORT = ["redqueen-preflight.service", "redqueen-supervisor.service"]
GENERATED_PROFILE_FIELDS = {
    "weights_bytes", "model_paths", "context_tokens", "planner", "gpu_bytes",
    "host_bytes", "memory", "artifact_identity", "artifact_fingerprint",
    "runtime_identity", "supported_request_shapes", "concurrency_limit",
    "max_image_inputs", "multimodal_workspace_mib",
}


def run(*arguments: str, check: bool = True, timeout: float = 180) -> subprocess.CompletedProcess:
    return host_reliability.command(*arguments, check=check, timeout=timeout)


def systemctl(*arguments: str, check: bool = True):
    return run("/usr/bin/systemctl", "--user", *arguments, check=check, timeout=35)


async def drain(policy: dict, profiles: dict, api_key: str, deadline: float) -> list[ResourceLock]:
    try:
        # This bounds the maintenance drain, not inference. A slow health or
        # slots probe cannot stretch the 60 second deployment window.
        async with asyncio.timeout_at(deadline):
            return await _drain(policy, profiles, api_key, deadline)
    except TimeoutError as error:
        raise RuntimeError("active inference did not drain within 60 seconds; deployment was not applied") from error


async def _drain(policy: dict, profiles: dict, api_key: str, deadline: float) -> list[ResourceLock]:
    directory = Path(policy["socket_path"]).parent / "leases"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    profiled_units = {
        unit
        for profile in profiles["resources"].values()
        for unit in (
            profile.get("units", [])
            + profile.get("runtime_identity", {}).get("service_units", [])
        )
    }
    for unit in MANAGED:
        if unit == "comfyui-control.service":
            continue
        state = systemctl("show", unit, "--property=ActiveState", "--value", check=False).stdout.strip()
        if state not in {"", "inactive", "failed", "not-found"} and unit not in profiled_units:
            raise RuntimeError(f"active managed worker {unit} is missing from the deployed memory profile")
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
        probe_policy = dict(policy)
        try:
            async with session.get(f"{policy['upstream_url']}/health") as response:
                response.raise_for_status()
        except (aiohttp.ClientError, TimeoutError):
            probe_policy["upstream_url"] = "http://10.9.20.242:8081"
        runtime = Runtime(probe_policy, profiles["resources"], session, api_key)
        while time.monotonic() < deadline:
            locks = []
            try:
                for name in sorted(profiles["resources"]):
                    lock = ResourceLock(directory, name, exclusive=True)
                    if not lock.acquire():
                        break
                    locks.append(lock)
                if len(locks) == len(profiles["resources"]):
                    # Backend probes also cover the first migration, whose old
                    # adapters have no leases yet, and work draining after EOF.
                    states = await runtime.snapshot()
                    if all([state == "ready" and await runtime.busy(name) is False for name, state in states.items()]):
                        return locks
            except BaseException:
                for lock in locks:
                    lock.close()
                raise
            for lock in locks:
                lock.close()
            await asyncio.sleep(0.5)
    raise RuntimeError("active inference did not drain within 60 seconds; deployment was not applied")


async def healthy(policy: dict) -> None:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2)) as session:
        for _ in range(40):
            try:
                for url in (f"{policy['upstream_url']}/health", "http://10.9.20.242:8081/readyz", "http://10.9.20.242:8190/readyz", "http://10.9.20.242:8191/readyz"):
                    async with session.get(url) as response:
                        response.raise_for_status()
                return
            except (aiohttp.ClientError, TimeoutError):
                await asyncio.sleep(0.25)
        raise RuntimeError("new inference services did not become healthy")


def restore(repo: Path, previous: str, enabled: dict, active: list[str], profile_backup: Path | None,
            profile_path: Path, preflight_definition: dict | None = None, branch: str | None = None,
            profile_backup_sha256: str | None = None) -> None:
    backup_bytes = profile_backup.read_bytes() if profile_backup is not None else None
    if profile_backup_sha256 is not None and (
        backup_bytes is None
        or hashlib.sha256(backup_bytes).hexdigest() != profile_backup_sha256
    ):
        raise RuntimeError("deployment profile backup failed its journal checksum; manual recovery required")
    loaded = [unit for unit in sorted(set(FRONTS + MANAGED + SUPPORT))
              if systemctl("show", unit, "--property=LoadState", "--value", check=False).stdout.strip() == "loaded"]
    if loaded:
        systemctl("stop", *loaded)
    run("git", "-C", str(repo), "switch", "--detach", previous)
    if branch:
        run("git", "-C", str(repo), "switch", "--force-create", branch, previous)
    # The first migration adds this unit. Remove only its deployment-owned
    # links if the previous revision predates it.
    for unit in ("redqueen-scheduler.service", *SUPPORT):
        if not (repo / "hosts/redqueen/systemd" / unit).exists():
            for path in (Path.home() / ".config/systemd/user" / unit, Path.home() / ".config/systemd/user/default.target.wants" / unit):
                if path.is_symlink():
                    path.unlink()
    run("bash", str(repo / "hosts/redqueen/link-runtime.sh"))
    host_reliability.restore_preflight(preflight_definition)
    if backup_bytes is not None:
        host_reliability.atomic_write(profile_path, backup_bytes)
    else:
        profile_path.unlink(missing_ok=True)
    for unit, was_enabled in enabled.items():
        if (repo / "hosts/redqueen/systemd" / unit).exists() or (unit == "redqueen-preflight.service" and (preflight_definition or {}).get("kind") in {"file", "symlink"}):
            systemctl("enable" if was_enabled else "disable", unit)
    # disable also removes the manually linked definitions; restore those
    # without recreating boot-target wants links.
    run("bash", str(repo / "hosts/redqueen/link-runtime.sh"))
    host_reliability.restore_preflight(preflight_definition)
    systemctl("daemon-reload")
    if active:
        systemctl("start", *active)


def validate(source: Path, *, fresh_profiler: bool = False) -> tuple[dict, dict]:
    policy_path, router_path = source / "model-router/scheduler.json", source / "model-router/config.yaml"
    policy = json.loads(policy_path.read_text())
    profiler = build_profiles
    if fresh_profiler:
        # --update can execute a verified staged deployment module during the
        # first migration. Profile the checked-out candidate with its own
        # canonical __file__, never a staging path that will later disappear.
        profiler_path = source / "adapters/redqueen_adapters/memory_profile.py"
        specification = importlib.util.spec_from_file_location("redqueen_adapters._candidate_memory_profile", profiler_path)
        if specification is None or specification.loader is None:
            raise RuntimeError("candidate memory profiler cannot be loaded")
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        profiler = module.build_profiles
    profiles = profiler(router_path, policy_path)
    run("/srv/ai/runtimes/llama-swap-v260-unbounded/bin/llama-swap", "-config", str(router_path), "-validate")
    run("systemd-analyze", "--user", "verify", *(str(path) for path in (source / "systemd").glob("*.service")))
    print(f"validated {len(profiles['resources'])} memory reservations", flush=True)
    return policy, profiles


def lock_directory_for_policy(policy: dict) -> tuple[Path, Path]:
    profile_value, socket_value = policy.get("profile_path"), policy.get("socket_path")
    if (
        not isinstance(profile_value, str) or not Path(profile_value).is_absolute()
        or not isinstance(socket_value, str) or not Path(socket_value).is_absolute()
    ):
        raise RuntimeError("scheduler profile_path and socket_path must be absolute")
    profile_path = Path(profile_value).resolve()
    directory = profile_path.parent
    if Path(socket_value).resolve().parent != directory:
        raise RuntimeError("scheduler socket and memory profile must share the locked directory")
    return profile_path, directory


def verify_lock_directory(policy: dict, directory: Path) -> Path:
    profile_path, policy_directory = lock_directory_for_policy(policy)
    if policy_directory != directory.resolve():
        raise RuntimeError("scheduler paths differ from the held deployment lock directory")
    return profile_path


def rebuild_last_good_resources(repo: Path, revision: str) -> tuple[dict, dict]:
    """Regenerate admission data and source identities from pinned last-good code."""
    result = subprocess.run(
        ["git", "-C", str(repo), "archive", "--format=tar", revision, "hosts/redqueen"],
        capture_output=True, check=False, timeout=120,
    )
    if result.returncode:
        raise RuntimeError("could not read last-good profiler source")
    with tempfile.TemporaryDirectory(prefix="redqueen-last-good-profile-") as temporary:
        root = Path(temporary).resolve()
        with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:") as archive:
            for member in archive.getmembers():
                relative = PurePosixPath(member.name)
                if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
                    raise RuntimeError("last-good source archive contains an unsafe path")
                target = root.joinpath(*relative.parts)
                if member.isdir():
                    target.mkdir(mode=0o700, parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                    source = archive.extractfile(member)
                    if source is None:
                        raise RuntimeError("last-good source archive is incomplete")
                    target.write_bytes(source.read())
        source_root = root / "hosts/redqueen"
        router_path = source_root / "model-router/config.yaml"
        policy_path = source_root / "model-router/scheduler.json"
        original_policy_bytes = policy_path.read_bytes()
        policy = json.loads(policy_path.read_text())

        def archived_path(value: str) -> str:
            path = Path(value)
            try:
                relative = path.relative_to(repo)
            except ValueError as error:
                raise RuntimeError("last-good profile input is outside the inference checkout") from error
            if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
                raise RuntimeError("last-good profile input has an unsafe source path")
            archived = root / relative
            if not archived.is_file():
                raise RuntimeError(f"last-good profile input is missing from source: {relative}")
            return str(archived)

        policy["artifact_metadata_root"] = str(source_root / "models/metadata")
        for service in policy["services"].values():
            for field in ("command_unit", "profile_script", "artifact_inventory"):
                if field in service:
                    service[field] = archived_path(service[field])
        policy_path.write_text(json.dumps(policy))

        profiler_path = source_root / "adapters/redqueen_adapters/memory_profile.py"
        specification = importlib.util.spec_from_file_location(
            f"_last_good_memory_profile_{revision}", profiler_path,
        )
        if specification is None or specification.loader is None:
            raise RuntimeError("last-good memory profiler cannot be loaded")
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        document = module.build_profiles(router_path, policy_path)
        policy_source = next(
            (source for source in document["sources"] if source.get("path") == str(policy_path)),
            None,
        )
        if policy_source is None:
            raise RuntimeError("last-good profiler omitted its scheduler source identity")
        policy_source["size"] = len(original_policy_bytes)
        policy_source["sha256"] = hashlib.sha256(original_policy_bytes).hexdigest()
        source_signatures = {}
        for source in document["sources"]:
            source_path = Path(source["path"])
            try:
                relative = source_path.relative_to(root)
            except ValueError:
                normalized_path = source_path
            else:
                if any(part in {"", ".", ".."} for part in relative.parts):
                    raise RuntimeError("last-good profiler emitted an unsafe source path")
                normalized_path = repo / relative
            normalized = {**source, "path": str(normalized_path)}
            if normalized["path"] in source_signatures:
                raise RuntimeError("last-good profiler emitted duplicate source identities")
            source_signatures[normalized["path"]] = normalized
        return document["resources"], source_signatures


def verify_legacy_profile_payload(
    repo: Path, revision: str, router: dict, scheduler: dict, profiles: dict,
) -> None:
    rebuilt, expected_sources = rebuild_last_good_resources(repo, revision)
    if set(rebuilt) != set(profiles["resources"]):
        raise RuntimeError("legacy memory profile cannot be reproduced from last-good source")
    actual_sources = {source["path"]: source for source in profiles["sources"]}
    if actual_sources != expected_sources:
        raise RuntimeError("legacy memory profile source identities differ from regenerated last-good inputs")
    for name, resource in profiles["resources"].items():
        expected = rebuilt[name]
        service = scheduler["services"].get(name)
        for field in GENERATED_PROFILE_FIELDS:
            expected_value = expected.get(field)
            if field == "planner" and isinstance(service, dict) and "profile_script" in service:
                expected_value = service["profile_script"]
            if resource.get(field) != expected_value:
                raise RuntimeError(f"legacy memory profile {name} does not match regenerated {field}")
        if name in router["models"]:
            model = router["models"][name]
            if (
                resource.get("kind") != "text"
                or resource.get("aliases", []) != model.get("aliases", [])
                or resource.get("concurrency") != (model.get("concurrencyLimit", 1) or 1)
                or resource.get("concurrency_limit") != (model.get("concurrencyLimit", 1) or 1)
            ):
                raise RuntimeError(f"legacy model profile {name} does not match last-good configuration")
        elif resource.get("kind") != "service":
            raise RuntimeError(f"legacy service profile {name} does not match last-good configuration")


def verify_deployed_profiles(
    repo: Path, head: str, profile_path: Path, profile_bytes: bytes, profiles: dict,
    marker_bytes: bytes | None = None,
) -> tuple[str, dict, dict, bool]:
    """Prove the drain snapshot was written by the last successful revision."""
    marker_path = profile_path.parent / "last-good.json"
    try:
        marker = json.loads(marker_bytes if marker_bytes is not None else marker_path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("--update requires a readable last-good deployment marker") from error
    revision = marker.get("revision") if isinstance(marker, dict) else None
    if (
        not isinstance(marker, dict)
        or isinstance(marker.get("version"), bool)
        or marker.get("version") != 1
        or marker.get("repo") != str(repo)
        or not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise RuntimeError("--update requires a valid last-good deployment marker")
    profile_sha256 = hashlib.sha256(profile_bytes).hexdigest()
    if "profile_sha256" in marker and (
        not isinstance(marker["profile_sha256"], str)
        or len(marker["profile_sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in marker["profile_sha256"])
        or marker["profile_sha256"] != profile_sha256
    ):
        raise RuntimeError("deployed memory profile differs from the last-good snapshot")
    run("git", "-C", str(repo), "rev-parse", "--verify", f"{revision}^{{commit}}")
    if run("git", "-C", str(repo), "merge-base", "--is-ancestor", revision, head, check=False).returncode:
        raise RuntimeError("last-good deployment is not an ancestor of the current checkout")

    required_sources = {
        "hosts/redqueen/model-router/config.yaml",
        "hosts/redqueen/model-router/scheduler.json",
        "hosts/redqueen/adapters/redqueen_adapters/memory_profile.py",
    }
    verified_sources = set()
    source_contents = {}
    for source in profiles["sources"]:
        path = Path(source["path"])
        try:
            relative = path.relative_to(repo)
        except ValueError:
            if file_signature(path, content="sha256" in source) != source:
                raise RuntimeError(f"deployed profile source changed since last-good: {path}")
            continue
        if "sha256" not in source or any(part in {".", ".."} for part in relative.parts):
            raise RuntimeError(f"deployed profile has no content identity for repository source {relative}")
        result = run(
            "git", "-C", str(repo), "show", f"{revision}:{relative.as_posix()}",
            check=False, timeout=30,
        )
        if result.returncode or hashlib.sha256(result.stdout.encode("utf-8")).hexdigest() != source["sha256"]:
            raise RuntimeError(f"deployed profile does not match last-good source {relative}")
        verified_sources.add(relative.as_posix())
        source_contents[relative.as_posix()] = result.stdout
    if not required_sources.issubset(verified_sources):
        raise RuntimeError("deployed profile is missing required last-good source identities")
    try:
        router = yaml.safe_load(source_contents["hosts/redqueen/model-router/config.yaml"])
        scheduler = json.loads(source_contents["hosts/redqueen/model-router/scheduler.json"])
        if not isinstance(router, dict) or not isinstance(router.get("models"), dict):
            raise ValueError("invalid last-good router model table")
        if not isinstance(scheduler, dict) or not isinstance(scheduler.get("services"), dict):
            raise ValueError("invalid last-good scheduler service table")
        expected_resources = set(router["models"]) | set(scheduler["services"])
    except (AttributeError, KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        raise RuntimeError("last-good model configuration cannot be read") from error
    if set(profiles["resources"]) != expected_resources:
        raise RuntimeError("deployed memory profile resource set differs from last-good model configuration")
    if verify_lock_directory(scheduler, profile_path.parent) != profile_path.resolve():
        raise RuntimeError("last-good scheduler paths differ from the deployed profile")
    for name, service in scheduler["services"].items():
        resource = profiles["resources"][name]
        if not isinstance(service, dict) or resource.get("kind") != "service":
            raise RuntimeError(f"deployed service profile {name} differs from last-good configuration")
        if set(resource) - (set(service) | GENERATED_PROFILE_FIELDS | {"kind"}):
            raise RuntimeError(f"deployed service profile {name} has undeclared fields")
        if any(resource.get(key) != value for key, value in service.items()):
            raise RuntimeError(f"deployed service profile {name} differs from last-good configuration")
    for name, model in router["models"].items():
        resource = profiles["resources"][name]
        if not isinstance(model, dict) or resource.get("kind") != "text":
            raise RuntimeError(f"deployed model profile {name} differs from last-good configuration")
        model_metadata = {"kind", "concurrency", "aliases", "concurrency_limit"}
        if set(resource) - (GENERATED_PROFILE_FIELDS | model_metadata):
            raise RuntimeError(f"deployed model profile {name} has undeclared fields")
        concurrency = model.get("concurrencyLimit", 1) or 1
        if (
            resource.get("concurrency") != concurrency
            or resource.get("concurrency_limit") != concurrency
            or resource.get("aliases", []) != model.get("aliases", [])
        ):
            raise RuntimeError(f"deployed model profile {name} differs from last-good configuration")
        try:
            argv = shlex.split(model["cmd"])
            model_path = argv[argv.index("--model") + 1]
        except (KeyError, ValueError, IndexError, TypeError) as error:
            raise RuntimeError(f"last-good model command for {name} is invalid") from error
        if resource.get("model_paths") != [str(path) for path in weight_paths(Path(model_path))]:
            raise RuntimeError(f"deployed model paths for {name} differ from last-good configuration")
        runtime_identity = resource.get("runtime_identity", {})
        if not isinstance(runtime_identity, dict) or runtime_identity.get("path") != str(Path(argv[0])):
            raise RuntimeError(f"deployed runtime for {name} differs from last-good configuration")
    return revision, scheduler, router, "profile_sha256" in marker


def deploy(repo: Path, previous: str | None, *, check_only: bool = False, update: bool = False) -> None:
    if os.geteuid() == 0:
        raise RuntimeError("run this as ai")
    if update and check_only:
        raise RuntimeError("--update and --check cannot be combined")
    repo = repo.resolve()
    source = repo / "hosts/redqueen"
    if check_only:
        if run("git", "-C", str(repo), "status", "--porcelain").stdout.strip():
            raise RuntimeError("deployment checkout must be clean")
        head = run("git", "-C", str(repo), "rev-parse", "HEAD").stdout.strip()
        previous = previous or head
        run("git", "-C", str(repo), "merge-base", "--is-ancestor", previous, head)
        policy, _ = validate(source)
        lock_directory_for_policy(policy)
        return

    # The policy read here only selects the lock directory. Re-read it after
    # acquiring the lock before capturing any deployment state.
    policy_hint = json.loads((source / "model-router/scheduler.json").read_text())
    profile_path, directory = lock_directory_for_policy(policy_hint)

    def update_state() -> tuple[str, bytes, bytes, dict, str, dict, dict, bool]:
        if run("git", "-C", str(repo), "status", "--porcelain").stdout.strip():
            raise RuntimeError("deployment checkout must be clean")
        if run("git", "-C", str(repo), "branch", "--show-current").stdout.strip() != "develop":
            raise RuntimeError("--update requires the develop branch")
        if run("git", "-C", str(repo), "rev-parse", "--abbrev-ref", "@{upstream}").stdout.strip() != "origin/develop":
            raise RuntimeError("--update requires tracking origin/develop")
        head = run("git", "-C", str(repo), "rev-parse", "HEAD").stdout.strip()
        try:
            profile_bytes = profile_path.read_bytes()
            marker_bytes = (directory / "last-good.json").read_bytes()
            profiles = json.loads(profile_bytes)
            if not isinstance(profiles, dict):
                raise ValueError("memory profile root is not an object")
            verify_profiles(profiles, verify_sources=False)
            revision, scheduler, router, has_digest = verify_deployed_profiles(
                repo, head, profile_path, profile_bytes, profiles, marker_bytes,
            )
        except (OSError, json.JSONDecodeError, AttributeError, KeyError, TypeError, ValueError) as error:
            raise RuntimeError("--update requires a valid last-good memory profile snapshot") from error
        return head, profile_bytes, marker_bytes, profiles, revision, scheduler, router, has_digest

    if update:
        with host_reliability.operation_lock(directory):
            policy = json.loads((source / "model-router/scheduler.json").read_text())
            if verify_lock_directory(policy, directory) != profile_path:
                raise RuntimeError("scheduler paths changed while acquiring the deployment lock")
            snapshot = update_state()
        head, profile_bytes, marker_bytes, profiles, previous, policy, router, has_digest = snapshot

        if not has_digest:
            # Rebuild every admission field against the pinned source without
            # holding the request lock. The exact snapshot is rechecked below.
            verify_legacy_profile_payload(repo, previous, router, policy, profiles)

        with host_reliability.operation_lock(directory):
            current_policy = json.loads((source / "model-router/scheduler.json").read_text())
            if verify_lock_directory(current_policy, directory) != profile_path:
                raise RuntimeError("scheduler paths changed while acquiring the deployment lock")
            current = update_state()
            if current != snapshot:
                raise RuntimeError("last-good deployment changed during profile verification; retry update")
            if not has_digest:
                marker = json.loads(marker_bytes)
                marker["profile_sha256"] = hashlib.sha256(profile_bytes).hexdigest()
                host_reliability.atomic_json(directory / "last-good.json", marker)
            rollback_profile_bytes = profile_bytes
            print(
                f"using {len(profiles['resources'])} profiles from last-good {previous[:12]} to drain current workers",
                flush=True,
            )
            branch = run("git", "-C", str(repo), "branch", "--show-current").stdout.strip() or None
            if systemctl("is-active", "qwen38.service", check=False).returncode == 0:
                raise RuntimeError("legacy qwen38.service is active; drain and stop it explicitly before deployment")
            _activate(
                repo, previous, source, policy, profiles, profile_path,
                branch=branch, update=True, rollback_profile_bytes=rollback_profile_bytes,
            )
        return

    with host_reliability.operation_lock(directory):
        if run("git", "-C", str(repo), "status", "--porcelain").stdout.strip():
            raise RuntimeError("deployment checkout must be clean")
        policy = json.loads((source / "model-router/scheduler.json").read_text())
        if verify_lock_directory(policy, directory) != profile_path:
            raise RuntimeError("scheduler paths changed while acquiring the deployment lock")
        head = run("git", "-C", str(repo), "rev-parse", "HEAD").stdout.strip()
        previous = previous or head
        run("git", "-C", str(repo), "merge-base", "--is-ancestor", previous, head)
        policy, profiles = validate(source)
        if verify_lock_directory(policy, directory) != profile_path:
            raise RuntimeError("validated scheduler paths differ from the held lock directory")
        branch = run("git", "-C", str(repo), "branch", "--show-current").stdout.strip() or None
        if systemctl("is-active", "qwen38.service", check=False).returncode == 0:
            raise RuntimeError("legacy qwen38.service is active; drain and stop it explicitly before deployment")
        _activate(repo, previous, source, policy, profiles, profile_path, branch=branch, update=False)


def _activate(repo: Path, previous: str, source: Path, policy: dict, profiles: dict,
              profile_path: Path, *, branch: str | None, update: bool,
              rollback_profile_bytes: bytes | None = None) -> None:
    directory = profile_path.parent
    if (directory / "deployment.json").exists():
        raise RuntimeError("an interrupted deployment needs recovery before another activation")
    api_key = os.environ["QWEN_API_KEY"]
    locks = asyncio.run(drain(policy, profiles, api_key, time.monotonic() + 60))
    journal = None
    try:
        units = sorted(set(FRONTS + MANAGED + SUPPORT))
        enabled = {unit: systemctl("is-enabled", unit, check=False).returncode == 0 for unit in units}
        active = [unit for unit in units if systemctl("is-active", unit, check=False).returncode == 0]
        backup = None
        if rollback_profile_bytes is not None:
            if not profile_path.is_file() or profile_path.read_bytes() != rollback_profile_bytes:
                raise RuntimeError("last-good memory profile changed after validation; deployment aborted")
        if profile_path.exists():
            backup = profile_path.with_name(f"memory-profiles.previous.{time.time_ns()}.json")
            backup_bytes = rollback_profile_bytes if rollback_profile_bytes is not None else profile_path.read_bytes()
            host_reliability.atomic_write(backup, backup_bytes)
        host_reliability.install_helper(Path(host_reliability.__file__),
                                        directory.parent / "libexec/host_reliability.py",
                                        bootstrap_only=True)
        # This fsynced journal precedes both git pull and service mutation.
        # SIGKILL/power loss recovery uses the installed stdlib-only helper.
        previous_sha = run("git", "-C", str(repo), "rev-parse", "--verify", f"{previous}^{{commit}}").stdout.strip() or previous
        definition = host_reliability.preflight_snapshot()
        journal = host_reliability.begin_journal(
            directory, repo, previous_sha, enabled, active, profile_path, backup, definition, branch=branch,
        )
        if update:
            host_reliability.bootstrap_preflight(directory.parent.parent)
            host_reliability.deployment_phase(directory, journal, "updating_source")
            run("git", "-C", str(repo), "pull", "--ff-only", "origin", "develop")
            run("bash", str(source / "model-router/create-venv.sh"))
            policy, profiles = validate(source, fresh_profiler=True)
            if verify_lock_directory(policy, directory) != profile_path:
                raise RuntimeError("scheduler path migration requires an explicit deployment migration")
            # New resources did not exist in the old scheduler. Hold their
            # exclusive locks too before activation, without releasing old ones.
            old_names = {lock.path.name for lock in locks}
            for name in sorted(set(profiles["resources"]) - old_names):
                lock = ResourceLock(directory / "leases", name, exclusive=True)
                if not lock.acquire():
                    raise RuntimeError("new model resource is in use; deployment aborted")
                locks.append(lock)
        host_reliability.deployment_phase(directory, journal, "activating")
        print("idle window acquired; applying user units", flush=True)
        running_fronts = [unit for unit in active if unit in FRONTS]
        if running_fronts:
            systemctl("stop", *running_fronts)
        running_backends = [unit for unit in active if unit in MANAGED and unit not in FRONTS]
        if running_backends:
            systemctl("stop", *running_backends)
        run("bash", str(source / "link-runtime.sh"))
        host_reliability.atomic_json(profile_path, profiles)
        systemctl("disable", *MANAGED)
        run("bash", str(source / "link-runtime.sh"))
        systemctl("enable", "llama-swap.service", "redqueen-scheduler.service", "qwen-image-adapter.service", "jevk5-adapter.service")
        systemctl("start", "llama-swap.service", "redqueen-scheduler.service", "qwen-image-adapter.service", "jevk5-adapter.service")
        if (source / "systemd/redqueen-preflight.service").exists():
            errors = host_reliability.preflight(directory.parent.parent)
            if errors:
                raise RuntimeError("runtime preflight failed: " + "; ".join(errors))
        asyncio.run(healthy(policy))
        for unit in ("redqueen-preflight.service", "redqueen-supervisor.service"):
            if (source / "systemd" / unit).exists():
                systemctl("enable", unit)
        if (source / "systemd/redqueen-supervisor.service").exists():
            systemctl("start", "redqueen-supervisor.service")
        helper = source / "adapters/redqueen_adapters/host_reliability.py"
        if helper.is_file():
            host_reliability.install_helper(helper, profile_path.parent.parent / "libexec/host_reliability.py")
            host_reliability.install_preflight_unit(directory.parent.parent, source=source / "systemd/redqueen-preflight.service")
            systemctl("daemon-reload")
        revision = run("git", "-C", str(repo), "rev-parse", "HEAD").stdout.strip()
        host_reliability.finish_deployment(directory, repo, revision, profile_path)
    except BaseException:
        if journal is not None:
            print(f"deployment failed; restoring {previous}", flush=True)
            restore(
                repo, previous, enabled, active, backup, profile_path,
                (journal or {}).get("preflight_definition"), branch=branch,
                profile_backup_sha256=(journal or {}).get("profile_backup_sha256"),
            )
            marker = {"version": 1, "repo": str(repo), "revision": previous, "rolled_back_at": time.time()}
            if profile_path.is_file():
                marker["profile_sha256"] = hashlib.sha256(profile_path.read_bytes()).hexdigest()
            host_reliability.atomic_json(directory / "last-good.json", marker)
            host_reliability.durable_unlink(directory / "deployment.json")
        raise
    finally:
        for lock in locks:
            lock.close()
    print(f"deployed {run('git', '-C', str(repo), 'rev-parse', 'HEAD').stdout.strip()}; APIs healthy", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollback-ref")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--update", action="store_true", help="drain, journal, pull origin/develop, validate and activate under one lock")
    parser.add_argument("--repo", type=Path, help="deployment checkout (supports a verified staged helper during first migration)")
    args = parser.parse_args()
    repo = args.repo.resolve() if args.repo is not None else Path(__file__).resolve().parents[4]
    deploy(repo, args.rollback_ref, check_only=args.check, update=args.update)


if __name__ == "__main__":
    main()
