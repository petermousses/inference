"""Build host-local memory reservations using llama.cpp's no-allocation planner."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import math
import os
import re
import shlex
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

import yaml

MIB = 1024**2
MEMORY_FLAGS = {
    "--model", "--ctx-size", "--cache-type-k", "--cache-type-v", "--gpu-layers",
    "--parallel", "--batch-size", "--ubatch-size", "--flash-attn",
}


def _loopback_http_url(value: object, *, path: str | None) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return False
    if (
        parsed.scheme != "http"
        or hostname is None
        or port is None
        or port == 0
        or (
            path == "" and parsed.path not in {"", "/"}
        )
        or (path not in {None, ""} and parsed.path != path)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        return False
    try:
        loopback = ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        loopback = hostname.lower() == "localhost"
    return loopback


def file_signature(path: Path, *, content: bool = False) -> dict:
    stat = path.stat()
    result = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if content:
        result["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        result.pop("mtime_ns")
    return result


def verify_profiles(document: dict) -> None:
    if document.get("version") != 1 or document.get("worker_profile_version") != 1 or not document.get("resources"):
        raise ValueError("invalid memory profile document")
    for name, profile in document["resources"].items():
        if not isinstance(name, str) or not name or not isinstance(profile, dict):
            raise ValueError("invalid memory profile resource")
        memory = profile.get("memory")
        if not isinstance(memory, dict):
            raise ValueError(f"{name} has no worker memory contract")
        totals = {"gpu_bytes": 0, "host_bytes": 0}
        for component in ("mandatory", "reclaimable", "transient_peak"):
            value = memory.get(component)
            if not isinstance(value, dict):
                raise ValueError(f"{name} has an invalid {component} memory profile")
            for field in totals:
                amount = value.get(field)
                if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
                    raise ValueError(f"{name} has an invalid {component} memory value")
                totals[field] += amount
        if any(
            isinstance(profile.get(field), bool)
            or not isinstance(profile.get(field), int)
            or profile.get(field) != amount
            for field, amount in totals.items()
        ):
            raise ValueError(f"{name} memory contract does not match its admission footprint")
        identities = profile.get("artifact_identity")
        fingerprint = profile.get("artifact_fingerprint")
        if (
            not isinstance(identities, list)
            or not identities
            or not isinstance(fingerprint, str)
            or not re.fullmatch(r"[0-9a-f]{64}", fingerprint)
        ):
            raise ValueError(f"{name} has no valid artifact identity")
        artifact_names = []
        for identity in identities:
            if not isinstance(identity, dict):
                raise ValueError(f"{name} has an invalid artifact identity")
            model = identity.get("model")
            repositories, revisions = identity.get("repositories"), identity.get("revisions")
            manifest_hash = identity.get("manifest_sha256")
            if (
                not isinstance(model, str) or not model
                or not isinstance(repositories, dict) or not repositories
                or not isinstance(revisions, dict) or not revisions
                or any(not isinstance(value, str) or not value for value in (*repositories.values(), *revisions.values()))
                or not isinstance(manifest_hash, str)
                or not re.fullmatch(r"[0-9a-f]{64}", manifest_hash)
            ):
                raise ValueError(f"{name} has an invalid artifact identity")
            artifact_names.append(model)
        if artifact_names != sorted(set(artifact_names)):
            raise ValueError(f"{name} has duplicate or unordered artifact identities")
        if fingerprint_artifacts(identities) != fingerprint:
            raise ValueError(f"{name} artifact fingerprint does not match its identity")
        runtime_identity = profile.get("runtime_identity")
        if not isinstance(runtime_identity, dict):
            raise ValueError(f"{name} has no runtime identity")
        if "path" in runtime_identity:
            if (
                not isinstance(runtime_identity.get("path"), str)
                or isinstance(runtime_identity.get("size"), bool)
                or not isinstance(runtime_identity.get("size"), int)
                or runtime_identity["size"] < 0
                or isinstance(runtime_identity.get("mtime_ns"), bool)
                or not isinstance(runtime_identity.get("mtime_ns"), int)
            ):
                raise ValueError(f"{name} has an invalid runtime identity")
        else:
            units = runtime_identity.get("service_units")
            if not isinstance(units, list) or not units or any(not isinstance(unit, str) or not unit for unit in units):
                raise ValueError(f"{name} has an invalid service runtime identity")
        shapes = profile.get("supported_request_shapes")
        if (
            not isinstance(shapes, list)
            or not shapes
            or any(not isinstance(shape, str) or not shape for shape in shapes)
            or len(set(shapes)) != len(shapes)
        ):
            raise ValueError(f"{name} has invalid supported request shapes")
        if "gateway_url" in profile:
            openai_shapes = {
                "openai.chat", "openai.completion", "openai.responses",
                "openai.embeddings", "openai.multimodal_chat",
                "openai.multimodal_responses",
            }
            if (
                profile.get("kind") != "service"
                or not _loopback_http_url(profile["gateway_url"], path="")
                or not _loopback_http_url(profile.get("slots_url"), path="/slots")
                or not _loopback_http_url(profile.get("health_url"), path=None)
                or not any(shape in openai_shapes for shape in shapes)
            ):
                raise ValueError(f"{name} has an invalid direct OpenAI service contract")
        concurrency = profile.get("concurrency_limit")
        if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
            raise ValueError(f"{name} has an invalid concurrency limit")
    sources = document.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("invalid memory profile sources")
    for expected in sources:
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


def artifact_identity(paths: list[Path], policy: dict, add_source) -> tuple[list[dict], str]:
    """Bind a worker profile to tracked model revisions and checksum manifests."""
    model_root = Path(policy["model_root"]).resolve()
    metadata_root = Path(policy["artifact_metadata_root"])
    artifacts: dict[str, dict] = {}
    manifests: dict[str, dict[str, str]] = {}
    for path in paths:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(model_root)
        except ValueError as error:
            raise ValueError(f"artifact is outside the configured model root: {path}") from error
        if not relative.parts:
            raise ValueError(f"artifact does not name a model directory: {path}")
        model_name = relative.parts[0]
        if model_name not in artifacts:
            metadata = metadata_root / model_name
            revision_path, manifest_path = metadata / "SOURCE_REVISION", metadata / "SHA256SUMS"
            if not revision_path.is_file() or not manifest_path.is_file():
                raise ValueError(f"{model_name} needs tracked SOURCE_REVISION and SHA256SUMS metadata")
            add_source(revision_path, content=True)
            add_source(manifest_path, content=True)
            revision_data = {}
            for line in revision_path.read_text().splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    revision_data[key.strip()] = value.strip()
            repositories = {key: value for key, value in revision_data.items() if key.endswith("repository")}
            revisions = {key: value for key, value in revision_data.items() if key.endswith("revision")}
            if not repositories or not revisions:
                raise ValueError(f"{model_name} has no pinned repository revisions")
            manifest = {}
            for line in manifest_path.read_text().splitlines():
                fields = line.split(maxsplit=1)
                if len(fields) == 2 and re.fullmatch(r"[0-9a-fA-F]{64}", fields[0]):
                    manifest[fields[1].lstrip("* ")] = fields[0].lower()
            manifests[model_name] = manifest
            artifacts[model_name] = {
                "model": model_name,
                "repositories": repositories,
                "revisions": revisions,
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            }
        artifact_path = Path(*relative.parts[1:]).as_posix()
        if artifact_path not in manifests[model_name]:
            raise ValueError(f"{path} is absent from the tracked {model_name} checksum manifest")
    identities = [artifacts[name] for name in sorted(artifacts)]
    return identities, fingerprint_artifacts(identities)


def fingerprint_artifacts(identities: list[dict]) -> str:
    return hashlib.sha256(
        json.dumps(identities, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def memory_contract(
    mandatory: tuple[int, int],
    *,
    reclaimable: tuple[int, int] = (0, 0),
    transient_peak: tuple[int, int] = (0, 0),
) -> dict:
    components = {
        "mandatory": {"gpu_bytes": mandatory[0], "host_bytes": mandatory[1]},
        "reclaimable": {"gpu_bytes": reclaimable[0], "host_bytes": reclaimable[1]},
        "transient_peak": {"gpu_bytes": transient_peak[0], "host_bytes": transient_peak[1]},
    }
    return {
        **components,
        "gpu_bytes": sum(value["gpu_bytes"] for value in components.values()),
        "host_bytes": sum(value["host_bytes"] for value in components.values()),
    }


def command_request_shapes(name: str, argv: list[str]) -> list[str]:
    shapes = ["openai.chat", "openai.completion", "openai.responses"]
    if "--embedding" in argv:
        shapes.append("openai.embeddings")
    if "--mmproj" in argv:
        shapes.append("openai.multimodal_chat")
    if name == "clef" or name.startswith("clef-"):
        shapes.append("systemone.decision")
    return shapes


def build_profiles(router_path: Path, policy_path: Path) -> dict:
    policy = json.loads(policy_path.read_text())
    router = yaml.safe_load(router_path.read_text())
    sources = {str(path): file_signature(path, content=True) for path in (router_path, policy_path, Path(__file__).resolve())}
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
        artifacts, artifact_fingerprint = artifact_identity(
            weights + extras + host_weights, policy, add_source
        )
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
        host_weight_bytes = sum(path.stat().st_size for path in host_weights)
        # Include all shards, even tensors the runtime might lazily map or ignore.
        weight_bytes = sum(path.stat().st_size for path in weights)
        total = max(total, weight_bytes) + extra_bytes + host_weight_bytes
        margin, overhead = policy["estimate_margin"], policy["model_overhead_mib"] * MIB
        gpu_bytes = math.ceil((gpu + extra_bytes) * margin) + overhead
        host_bytes = math.ceil(total * margin) + overhead
        return {
            **metadata, "weights_bytes": weight_bytes + extra_bytes + host_weight_bytes,
            "model_paths": [str(path) for path in weights],
            "context_tokens": int(flags["--ctx-size"]), "planner": str(estimator),
            "gpu_bytes": gpu_bytes, "host_bytes": host_bytes,
            "memory": memory_contract((gpu_bytes, host_bytes)),
            "artifact_identity": artifacts,
            "artifact_fingerprint": artifact_fingerprint,
            "runtime_identity": file_signature(runtime),
            "supported_request_shapes": metadata.get(
                "supported_request_shapes", command_request_shapes(name, argv)
            ),
            "concurrency_limit": metadata["concurrency_limit"],
        }

    for name, model in router["models"].items():
        resources[name] = profile_command(name, model["cmd"], {
            "kind": "text", "concurrency": model.get("concurrencyLimit", 1) or 1,
            "aliases": model.get("aliases", []),
            "concurrency_limit": model.get("concurrencyLimit", 1) or 1,
        })
    for name, service in policy["services"].items():
        metadata = {"kind": "service", **service}
        unit_root = Path(__file__).resolve().parents[2] / "systemd"
        for unit_name in service.get("units", []):
            unit_path = unit_root / unit_name
            add_source(unit_path, content=True)
        if "command_unit" in service:
            unit = Path(service["command_unit"])
            add_source(unit, content=True)
            match = re.search(r"^ExecStart=(.+)$", unit.read_text(), re.MULTILINE)
            if not match:
                raise ValueError(f"missing ExecStart in {unit}")
            resources[name] = profile_command(name, match[1], metadata)
        else:
            weights = [Path(path) for path in service["weights"]]
            artifacts, artifact_fingerprint = artifact_identity(weights, policy, add_source)
            for path in weights:
                add_source(path)
            size = sum(path.stat().st_size for path in weights)
            workspace = service["workspace_mib"] * MIB
            resident = math.ceil(size * policy["estimate_margin"])
            memory = memory_contract((resident, resident), transient_peak=(workspace, workspace))
            resources[name] = {
                **metadata, "weights_bytes": size, "context_tokens": 0,
                "gpu_bytes": memory["gpu_bytes"], "host_bytes": memory["host_bytes"],
                "memory": memory, "artifact_identity": artifacts,
                "artifact_fingerprint": artifact_fingerprint,
                "runtime_identity": {"service_units": service.get("units", [])},
                "supported_request_shapes": service["supported_request_shapes"],
                "concurrency_limit": service["concurrency_limit"],
            }
    document = {
        "version": 1, "worker_profile_version": 1,
        "sources": list(sources.values()), "resources": resources,
    }
    verify_profiles(document)
    return document


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
        print(f"{name}: GPU {spec['gpu_bytes'] / MIB:.0f} MiB, host {spec['host_bytes'] / MIB:.0f} MiB, context {spec['context_tokens']}, artifacts {spec['artifact_fingerprint'][:12]}")


if __name__ == "__main__":
    main()
