"""Build host-local memory reservations using llama.cpp's no-allocation planner."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
from pathlib import Path

import yaml

MIB = 1024**2
MEMORY_FLAGS = {
    "--model", "--ctx-size", "--cache-type-k", "--cache-type-v", "--gpu-layers",
    "--parallel", "--batch-size", "--ubatch-size", "--flash-attn",
}


def file_signature(path: Path, *, content: bool = False) -> dict:
    stat = path.stat()
    result = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if content:
        result["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        result.pop("mtime_ns")
    return result


def verify_profiles(document: dict) -> None:
    if document.get("version") != 1 or not document.get("resources"):
        raise ValueError("invalid memory profile document")
    for expected in document["sources"]:
        actual = file_signature(Path(expected["path"]), content="sha256" in expected)
        if actual != expected:
            raise ValueError(f"memory profile is stale: {expected['path']}; rebuild profiles")


def weight_paths(path: Path) -> list[Path]:
    match = re.match(r"^(.*)-00001-of-(\d{5})\.gguf$", path.name)
    if match:
        return [path.with_name(f"{match[1]}-{index:05d}-of-{match[2]}.gguf") for index in range(1, int(match[2]) + 1)]
    return [path]


def memory_rows(output: str) -> dict[str, tuple[int, int, int]]:
    rows = {}
    for line in output.splitlines():
        match = re.fullmatch(r"(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s*", line)
        if match:
            rows[match[1]] = tuple((int(value) + 1) * MIB for value in match.groups()[1:])
    if "Host" not in rows or not any(name.startswith("ROCm") for name in rows):
        raise ValueError("memory estimator did not report Host and ROCm memory")
    return rows


def build_profiles(router_path: Path, policy_path: Path) -> dict:
    policy = json.loads(policy_path.read_text())
    router = yaml.safe_load(router_path.read_text())
    sources = {str(path): file_signature(path, content=True) for path in (router_path, policy_path)}
    resources = {}

    def add_source(path: Path, *, content: bool = False) -> None:
        sources[str(path)] = file_signature(path, content=content)

    def profile_command(name: str, command: str, metadata: dict) -> dict:
        argv = shlex.split(command)
        runtime = Path(argv[0])
        add_source(runtime)
        estimator = runtime.with_name("llama-fit-params")
        if not estimator.is_file():
            estimator = Path(policy["estimator"])
        add_source(estimator)
        arguments, flags = [], {}
        for index, argument in enumerate(argv[:-1]):
            if argument in MEMORY_FLAGS:
                flags[argument] = argv[index + 1]
                arguments.extend((argument, argv[index + 1]))
        if "--model" not in flags or int(flags.get("--ctx-size", 0)) <= 0:
            raise ValueError(f"{name} needs an explicit model and context size")
        weights = weight_paths(Path(flags["--model"]))
        extras = [Path(argv[argv.index("--mmproj") + 1])] if "--mmproj" in argv else []
        host_weights = [Path(path) for path in metadata.get("host_weights", [])]
        for path in weights + extras + host_weights:
            add_source(path)
        result = subprocess.run(
            [str(estimator), "--fit-print", "on", *arguments],
            capture_output=True, text=True, timeout=60, check=False,
        )
        if result.returncode:
            raise ValueError(f"{name} memory estimation failed: {result.stderr[-1000:]}")
        rows = memory_rows(result.stdout)
        gpu = sum(sum(values) for device, values in rows.items() if device != "Host")
        total = sum(sum(values) for values in rows.values())
        extra_bytes = sum(path.stat().st_size for path in extras)
        host_bytes = sum(path.stat().st_size for path in host_weights)
        # Include all shards, even tensors the runtime might lazily map or ignore.
        weight_bytes = sum(path.stat().st_size for path in weights)
        total = max(total, weight_bytes) + extra_bytes + host_bytes
        margin, overhead = policy["estimate_margin"], policy["model_overhead_mib"] * MIB
        return {
            **metadata, "weights_bytes": weight_bytes + extra_bytes + host_bytes,
            "model_paths": [str(path) for path in weights],
            "context_tokens": int(flags["--ctx-size"]), "planner": str(estimator),
            "gpu_bytes": math.ceil((gpu + extra_bytes) * margin) + overhead,
            "host_bytes": math.ceil(total * margin) + overhead,
        }

    for name, model in router["models"].items():
        resources[name] = profile_command(name, model["cmd"], {
            "kind": "text", "concurrency": model.get("concurrencyLimit", 1) or 1,
            "aliases": model.get("aliases", []),
        })
    for name, service in policy["services"].items():
        metadata = {"kind": "service", **service}
        if "command_unit" in service:
            unit = Path(service["command_unit"])
            add_source(unit, content=True)
            match = re.search(r"^ExecStart=(.+)$", unit.read_text(), re.MULTILINE)
            if not match:
                raise ValueError(f"missing ExecStart in {unit}")
            resources[name] = profile_command(name, match[1], metadata)
        else:
            weights = [Path(path) for path in service["weights"]]
            for path in weights:
                add_source(path)
            size = sum(path.stat().st_size for path in weights)
            reservation = math.ceil(size * policy["estimate_margin"]) + service["workspace_mib"] * MIB
            resources[name] = {**metadata, "weights_bytes": size, "context_tokens": 0, "gpu_bytes": reservation, "host_bytes": reservation}
    return {"version": 1, "sources": list(sources.values()), "resources": resources}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=Path("/srv/ai/model-router/scheduler.json"))
    parser.add_argument("--router", type=Path, default=Path("/srv/ai/model-router/config.yaml"))
    parser.add_argument("--output", type=Path, default=Path("/srv/ai/model-router/run/memory-profiles.json"))
    args = parser.parse_args()
    document = build_profiles(args.router, args.policy)
    temporary = args.output.with_name(args.output.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("x") as target:
            json.dump(document, target, indent=2)
            target.flush()
            os.fsync(target.fileno())
        os.chmod(temporary, 0o600)
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    for name, spec in document["resources"].items():
        print(f"{name}: GPU {spec['gpu_bytes'] / MIB:.0f} MiB, host {spec['host_bytes'] / MIB:.0f} MiB, context {spec['context_tokens']}")


if __name__ == "__main__":
    main()
