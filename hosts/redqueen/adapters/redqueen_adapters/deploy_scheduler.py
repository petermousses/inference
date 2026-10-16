"""Apply the checked-out scheduler revision after an idle admission window."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

import aiohttp

from .memory_profile import build_profiles
from .scheduler_api import Runtime
from .scheduler_client import ResourceLock
from . import host_reliability

FRONTS = ["qwen-image-adapter.service", "jevk5-adapter.service", "redqueen-scheduler.service", "llama-swap.service", "comfyui-control.service"]
MANAGED = ["comfyui.service", "jevk5.service", "clm-api.service", "clm-encoder.service", "comfyui-control.service", "qwen38.service"]
SUPPORT = ["redqueen-preflight.service", "redqueen-supervisor.service"]


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


def restore(repo: Path, previous: str, enabled: dict, active: list[str], profile_backup: Path | None, profile_path: Path, preflight_definition: dict | None = None) -> None:
    loaded = [unit for unit in sorted(set(FRONTS + MANAGED + SUPPORT))
              if systemctl("show", unit, "--property=LoadState", "--value", check=False).stdout.strip() == "loaded"]
    if loaded:
        systemctl("stop", *loaded)
    run("git", "-C", str(repo), "switch", "--detach", previous)
    # The first migration adds this unit. Remove only its deployment-owned
    # links if the previous revision predates it.
    for unit in ("redqueen-scheduler.service", *SUPPORT):
        if not (repo / "hosts/redqueen/systemd" / unit).exists():
            for path in (Path.home() / ".config/systemd/user" / unit, Path.home() / ".config/systemd/user/default.target.wants" / unit):
                if path.is_symlink():
                    path.unlink()
    run("bash", str(repo / "hosts/redqueen/link-runtime.sh"))
    host_reliability.restore_preflight(preflight_definition)
    if profile_backup is not None:
        host_reliability.atomic_write(profile_path, profile_backup.read_bytes())
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


def deploy(repo: Path, previous: str, *, check_only: bool = False, update: bool = False) -> None:
    if os.geteuid() == 0:
        raise RuntimeError("run this as ai")
    if run("git", "-C", str(repo), "status", "--porcelain").stdout.strip():
        raise RuntimeError("deployment checkout must be clean")
    if update and check_only:
        raise RuntimeError("--update and --check cannot be combined")
    if update:
        if run("git", "-C", str(repo), "branch", "--show-current").stdout.strip() != "develop":
            raise RuntimeError("--update requires the develop branch")
        if run("git", "-C", str(repo), "rev-parse", "--abbrev-ref", "@{upstream}").stdout.strip() != "origin/develop":
            raise RuntimeError("--update requires tracking origin/develop")
        previous = run("git", "-C", str(repo), "rev-parse", "HEAD").stdout.strip()
    run("git", "-C", str(repo), "merge-base", "--is-ancestor", previous, "HEAD")
    source = repo / "hosts/redqueen"
    policy, profiles = validate(source)
    profile_path = Path(policy["profile_path"])
    if check_only:
        return
    with host_reliability.operation_lock(profile_path.parent):
        if systemctl("is-active", "qwen38.service", check=False).returncode == 0:
            raise RuntimeError("legacy qwen38.service is active; drain and stop it explicitly before deployment")
        _activate(repo, previous, source, policy, profiles, profile_path, update=update)


def _activate(repo: Path, previous: str, source: Path, policy: dict, profiles: dict,
              profile_path: Path, *, update: bool) -> None:
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
        if profile_path.exists():
            backup = profile_path.with_name(f"memory-profiles.previous.{time.time_ns()}.json")
            host_reliability.atomic_write(backup, profile_path.read_bytes())
        host_reliability.install_helper(Path(host_reliability.__file__),
                                        directory.parent / "libexec/host_reliability.py",
                                        bootstrap_only=True)
        # This fsynced journal precedes both git pull and service mutation.
        # SIGKILL/power loss recovery uses the installed stdlib-only helper.
        previous_sha = run("git", "-C", str(repo), "rev-parse", "--verify", f"{previous}^{{commit}}").stdout.strip() or previous
        definition = host_reliability.preflight_snapshot()
        journal = host_reliability.begin_journal(directory, repo, previous_sha, enabled, active, profile_path, backup, definition)
        if update:
            host_reliability.bootstrap_preflight(directory.parent.parent)
            host_reliability.deployment_phase(directory, journal, "updating_source")
            run("git", "-C", str(repo), "pull", "--ff-only", "origin", "develop")
            run("bash", str(source / "model-router/create-venv.sh"))
            policy, profiles = validate(source, fresh_profiler=True)
            if Path(policy["profile_path"]) != profile_path:
                raise RuntimeError("profile path migration requires an explicit deployment migration")
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
        host_reliability.finish_deployment(directory, repo, revision)
    except BaseException:
        if journal is not None:
            print(f"deployment failed; restoring {previous}", flush=True)
            restore(repo, previous, enabled, active, backup, profile_path, (journal or {}).get("preflight_definition"))
            host_reliability.atomic_json(directory / "last-good.json", {"version": 1, "repo": str(repo), "revision": previous, "rolled_back_at": time.time()})
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
    previous = args.rollback_ref or run("git", "-C", str(repo), "rev-parse", "HEAD").stdout.strip()
    deploy(repo, previous, check_only=args.check, update=args.update)


if __name__ == "__main__":
    main()
