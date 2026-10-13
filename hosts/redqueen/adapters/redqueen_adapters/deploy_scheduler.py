"""Apply the checked-out scheduler revision after an idle admission window."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import aiohttp

from .memory_profile import build_profiles
from .scheduler_api import Runtime
from .scheduler_client import ResourceLock

FRONTS = ["qwen-image-adapter.service", "jevk5-adapter.service", "redqueen-scheduler.service", "llama-swap.service", "comfyui-control.service"]
MANAGED = ["comfyui.service", "jevk5.service", "clm-api.service", "clm-encoder.service", "comfyui-control.service"]


def run(*arguments: str, check: bool = True, timeout: float = 180) -> subprocess.CompletedProcess:
    return subprocess.run(arguments, check=check, timeout=timeout, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def systemctl(*arguments: str, check: bool = True):
    return run("/usr/bin/systemctl", "--user", *arguments, check=check)


async def drain(policy: dict, profiles: dict, api_key: str, deadline: float) -> list[ResourceLock]:
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
                    if all([state == "ready" and not await runtime.busy(name) for name, state in states.items()]):
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


def restore(repo: Path, previous: str, enabled: dict, active: list[str], profile_backup: Path | None, profile_path: Path) -> None:
    systemctl("stop", *FRONTS, *MANAGED, check=False)
    run("git", "-C", str(repo), "switch", "--detach", previous)
    # The first migration adds this unit. Remove only its deployment-owned
    # links if the previous revision predates it.
    if not (repo / "hosts/redqueen/systemd/redqueen-scheduler.service").exists():
        for path in (Path.home() / ".config/systemd/user/redqueen-scheduler.service", Path.home() / ".config/systemd/user/default.target.wants/redqueen-scheduler.service"):
            if path.is_symlink():
                path.unlink()
    run("bash", str(repo / "hosts/redqueen/link-runtime.sh"))
    if profile_backup is not None:
        shutil.copyfile(profile_backup, profile_path)
        profile_path.chmod(0o600)
    else:
        profile_path.unlink(missing_ok=True)
    for unit, was_enabled in enabled.items():
        if (repo / "hosts/redqueen/systemd" / unit).exists():
            systemctl("enable" if was_enabled else "disable", unit)
    systemctl("daemon-reload")
    if active:
        systemctl("start", *active)


def deploy(repo: Path, previous: str, *, check_only: bool = False) -> None:
    if os.geteuid() == 0:
        raise RuntimeError("run this as ai")
    if run("git", "-C", str(repo), "status", "--porcelain").stdout.strip():
        raise RuntimeError("deployment checkout must be clean")
    run("git", "-C", str(repo), "merge-base", "--is-ancestor", previous, "HEAD")
    source = repo / "hosts/redqueen"
    policy_path, router_path = source / "model-router/scheduler.json", source / "model-router/config.yaml"
    policy = json.loads(policy_path.read_text())
    profile_path = Path(policy["profile_path"])
    profile_path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    profiles = build_profiles(router_path, policy_path)
    run("/srv/ai/bin/llama-swap", "-config", str(router_path), "-validate")
    run("systemd-analyze", "--user", "verify", *(str(path) for path in (source / "systemd").glob("*.service")))
    print(f"validated {len(profiles['resources'])} memory reservations", flush=True)
    if check_only:
        return
    api_key = os.environ["QWEN_API_KEY"]
    locks = asyncio.run(drain(policy, profiles, api_key, time.monotonic() + 60))
    units = sorted(set(FRONTS + MANAGED))
    enabled = {unit: systemctl("is-enabled", unit, check=False).returncode == 0 for unit in units}
    active = [unit for unit in units if systemctl("is-active", unit, check=False).returncode == 0]
    backup = None
    if profile_path.exists():
        backup = profile_path.with_name(f"memory-profiles.previous.{time.time_ns()}.json")
        shutil.copyfile(profile_path, backup)
        backup.chmod(0o600)
    try:
        print("idle window acquired; applying user units", flush=True)
        running_fronts = [unit for unit in active if unit in FRONTS]
        if running_fronts:
            systemctl("stop", *running_fronts)
        run("bash", str(source / "link-runtime.sh"))
        temporary = profile_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(profiles, indent=2))
        temporary.chmod(0o600)
        temporary.replace(profile_path)
        systemctl("disable", *MANAGED)
        systemctl("enable", "llama-swap.service", "redqueen-scheduler.service", "qwen-image-adapter.service", "jevk5-adapter.service")
        systemctl("start", "llama-swap.service", "redqueen-scheduler.service", "qwen-image-adapter.service", "jevk5-adapter.service")
        asyncio.run(healthy(policy))
    except BaseException:
        print(f"deployment failed; restoring {previous}", flush=True)
        restore(repo, previous, enabled, active, backup, profile_path)
        raise
    finally:
        for lock in locks:
            lock.close()
    print(f"deployed {run('git', '-C', str(repo), 'rev-parse', 'HEAD').stdout.strip()}; APIs healthy", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollback-ref", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[4]
    deploy(repo, args.rollback_ref, check_only=args.check)


if __name__ == "__main__":
    main()
