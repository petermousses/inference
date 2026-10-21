#!/usr/bin/env python3
"""Capture Redqueen's hardware, storage, and configured weight inventory."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import yaml

MIB = 1024**2
ROOT = Path(__file__).resolve().parents[3]
MODEL_FLAGS = {"--model", "--mmproj", "--model-draft", "--model-vocoder"}
SHARDED_GGUF = re.compile(r"^(.*)-00001-of-(\d{5})\.gguf$")


def read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except (FileNotFoundError, PermissionError, OSError, UnicodeError):
        return None


def read_int(path: Path) -> int | None:
    value = read_text(path)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def linux_memory() -> dict:
    values: dict[str, int] = {}
    try:
        lines = Path("/proc/meminfo").read_text().splitlines()
    except OSError:
        lines = []
    for line in lines:
        key, separator, value = line.partition(":")
        if not separator:
            continue
        match = re.match(r"\s*(\d+)\s*kB\s*$", value)
        if match:
            values[key] = int(match[1]) * 1024
    return {
        "total_bytes": values.get("MemTotal"),
        "available_bytes": values.get("MemAvailable"),
        "swap_total_bytes": values.get("SwapTotal"),
        "swap_free_bytes": values.get("SwapFree"),
    }


def pci_devices() -> list[dict]:
    devices = []
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
        device = card / "device"
        if not device.exists():
            continue
        entry = {
            "drm_card": card.name,
            "vendor_id": read_text(device / "vendor"),
            "device_id": read_text(device / "device"),
            "product_name": read_text(device / "product_name"),
            "vram_total_bytes": read_int(device / "mem_info_vram_total"),
            "vram_used_bytes": read_int(device / "mem_info_vram_used"),
            "gtt_total_bytes": read_int(device / "mem_info_gtt_total"),
            "gtt_used_bytes": read_int(device / "mem_info_gtt_used"),
        }
        devices.append(entry)
    return devices


def kfd_topology() -> list[dict]:
    nodes = []
    for properties in sorted(Path("/sys/class/kfd/kfd/topology/nodes").glob("*/properties")):
        values = {}
        try:
            lines = properties.read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            fields = line.split(maxsplit=1)
            if len(fields) != 2:
                continue
            key, value = fields
            if key not in {"name", "vendor_id", "device_id", "gfx_target_version", "local_mem_size", "simd_count", "array_count"}:
                continue
            try:
                values[key] = int(value)
            except ValueError:
                values[key] = value
        nodes.append({"node": properties.parent.name, **values})
    return nodes


def decode_mount_field(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)


def mounted_filesystems() -> list[dict]:
    mounts = []
    try:
        lines = Path("/proc/self/mounts").read_text().splitlines()
    except OSError:
        lines = []
    for line in lines:
        fields = line.split()
        if len(fields) < 4:
            continue
        source, mount, fs_type = (decode_mount_field(value) for value in fields[:3])
        source = re.sub(r"(?i)(?:[^/@:]+:[^/@]*)@", "<redacted>@", source)
        mounts.append({"source": source, "mountpoint": mount, "filesystem": fs_type})
    return sorted(mounts, key=lambda item: len(item["mountpoint"]), reverse=True)


def nearest_existing(path: Path) -> Path | None:
    candidate = path
    while candidate != candidate.parent:
        if candidate.exists():
            return candidate
        candidate = candidate.parent
    return candidate if candidate.exists() else None


def storage_for(path: Path, mounts: list[dict]) -> dict | None:
    probe = nearest_existing(path)
    if probe is None:
        return None
    try:
        info = os.stat(probe)
    except OSError:
        return None
    resolved = probe.resolve()
    mount = next(
        (
            item
            for item in mounts
            if resolved == Path(item["mountpoint"])
            or Path(item["mountpoint"]) in resolved.parents
        ),
        None,
    )
    if mount is None:
        return {"mountpoint": None, "source": None, "filesystem": None}

    major, minor = os.major(info.st_dev), os.minor(info.st_dev)
    sys_device = Path("/sys/dev/block") / f"{major}:{minor}"
    resolved_sys = sys_device.resolve() if sys_device.exists() else None
    candidates = [resolved_sys] if resolved_sys else []
    if resolved_sys:
        candidates.extend(resolved_sys.parents)
    rotational = None
    device_model = None
    device_name = None
    for candidate in candidates:
        if rotational is None:
            rotational = read_int(candidate / "queue/rotational")
        if device_model is None:
            device_model = read_text(candidate / "device/model") or read_text(candidate / "model")
        if device_name is None:
            device_name = read_text(candidate / "device/name") or read_text(candidate / "name")
    usage = None
    try:
        disk = shutil.disk_usage(mount["mountpoint"])
        usage = {"total_bytes": disk.total, "free_bytes": disk.free, "used_bytes": disk.used}
    except OSError:
        pass
    return {
        "source": mount["source"],
        "mountpoint": mount["mountpoint"],
        "filesystem": mount["filesystem"],
        "device_major_minor": f"{major}:{minor}",
        "device_model": device_model,
        "device_name": device_name,
        "rotational": bool(rotational) if rotational is not None else None,
        "usage": usage,
    }


def resolve_config_path(value: str, root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        return root / path
    if path.exists():
        return path
    marker = "/hosts/redqueen/"
    if marker in str(path):
        return root / "hosts" / "redqueen" / str(path).split(marker, 1)[1]
    return path


def flag_values(argv: list[str]) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    index = 0
    while index < len(argv):
        argument = argv[index]
        flag, separator, inline_value = argument.partition("=")
        if separator and flag in MODEL_FLAGS and inline_value and not inline_value.startswith("$"):
            values.setdefault(flag, []).append(inline_value)
            index += 1
            continue
        if argument in MODEL_FLAGS and index + 1 < len(argv):
            value = argv[index + 1]
            if not value.startswith("$"):
                values.setdefault(argument, []).append(value)
            index += 2
            continue
        index += 1
    return values


def expand_weight_path(path: Path) -> list[Path]:
    match = SHARDED_GGUF.match(path.name)
    if not match:
        return [path]
    count = int(match[2])
    return [path.with_name(f"{match[1]}-{index:05d}-of-{count:05d}.gguf") for index in range(1, count + 1)]


def collect_artifact_refs(router: dict, policy: dict, root: Path) -> dict[str, dict]:
    artifacts: dict[str, dict] = {}

    def add(model: str, role: str, raw_path: str, workload_class: str | None = None) -> None:
        if not isinstance(raw_path, str) or not raw_path or "$" in raw_path:
            return
        path = resolve_config_path(raw_path, root)
        for shard in expand_weight_path(path):
            key = str(shard)
            item = artifacts.setdefault(key, {"path": key, "references": []})
            reference = {"model": model, "role": role}
            if workload_class is not None:
                reference["class"] = workload_class
            if reference not in item["references"]:
                item["references"].append(reference)

    for name, model in router.get("models", {}).items():
        if not isinstance(model, dict) or not isinstance(model.get("cmd"), str):
            continue
        argv = shlex.split(model["cmd"])
        flags = flag_values(argv)
        model_class = "embedding" if "--embedding" in argv else "text"
        for flag, paths in flags.items():
            role = "primary" if flag in {"--model", "--model-draft"} else "processor_or_auxiliary"
            for raw_path in paths:
                add(name, role, raw_path, model_class)
        for raw_path in model.get("weights", []):
            add(name, "auxiliary", raw_path, model_class)

    for name, service in policy.get("services", {}).items():
        if not isinstance(service, dict):
            continue
        service_class = {"jevk5": "decisions", "clm": "clm", "comfyui": "images"}.get(name, "service")
        for role in ("weights", "host_weights"):
            for raw_path in service.get(role, []):
                add(name, role, raw_path, service_class)
        unit_value = service.get("command_unit")
        if not isinstance(unit_value, str):
            continue
        unit_path = resolve_config_path(unit_value, root)
        try:
            unit_text = unit_path.read_text()
        except OSError:
            continue
        for command in re.findall(r"^ExecStart=(.+)$", unit_text, re.MULTILINE):
            flags = flag_values(shlex.split(command))
            for paths in flags.values():
                for raw_path in paths:
                    add(name, "primary", raw_path, service_class)

    return artifacts


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while chunk := stream.read(8 * MIB):
            digest.update(chunk)
    return digest.hexdigest()


def host_details() -> dict:
    cpu = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip().lower() in {"model name", "hardware"}:
                cpu = value.strip()
                break
    except OSError:
        pass
    return {
        "hostname": platform.node(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "cpu": cpu,
        "memory": linux_memory(),
        "gpu_devices": pci_devices(),
        "kfd_topology": kfd_topology(),
        "tools": {name: shutil.which(name) for name in ("rocminfo", "rocm-smi", "hipcc", "fio")},
    }


def fio_read_probe(path: Path, mode: str, size_bytes: int) -> dict:
    fio = shutil.which("fio")
    if fio is None:
        return {"status": "unavailable", "reason": "fio_not_installed"}
    if not path.is_file():
        return {"status": "skipped", "reason": "artifact_too_small_or_missing"}
    block_size = "1m" if mode == "read" else "256k"
    block_bytes = MIB if mode == "read" else 256 * 1024
    try:
        file_size = path.stat().st_size
    except OSError:
        return {"status": "skipped", "reason": "artifact_unreadable"}
    size_bytes = min(size_bytes, file_size)
    size_bytes -= size_bytes % block_bytes
    if size_bytes < block_bytes:
        return {"status": "skipped", "reason": "artifact_too_small_for_aligned_direct_io"}
    offset_bytes = ((file_size - size_bytes) // 2) // block_bytes * block_bytes
    command = [
        fio,
        "--name=redqueen-preflight",
        f"--filename={path}",
        f"--rw={mode}",
        f"--bs={block_size}",
        f"--size={size_bytes}",
        f"--offset={offset_bytes}",
        "--ioengine=psync",
        "--iodepth=1",
        "--numjobs=1",
        "--direct=1",
        "--output-format=json",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=300, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"status": "failed", "reason": type(error).__name__}
    if result.returncode:
        return {"status": "failed", "returncode": result.returncode, "detail": result.stderr[-1024:].strip()}
    try:
        payload = json.loads(result.stdout)
        read = payload["jobs"][0]["read"]
    except (ValueError, KeyError, IndexError, TypeError):
        return {"status": "failed", "reason": "invalid_fio_output"}
    return {
        "status": "ok",
        "mode": mode,
        "block_size": block_size,
        "sample_bytes": size_bytes,
        "sample_offset_bytes": offset_bytes,
        "bytes_per_second": read.get("bw_bytes", int(read["bw"] * 1024) if "bw" in read else None),
        "iops": read.get("iops"),
        "io_bytes": read.get("io_bytes"),
        "runtime_ms": read.get("runtime"),
        "direct_io": True,
        "note": "bypasses the page cache; device and controller caches may still affect results",
    }


def collect(args: argparse.Namespace) -> dict:
    router_path = Path(args.router_config).resolve()
    policy_path = Path(args.scheduler_config).resolve()
    router = yaml.safe_load(router_path.read_text())
    policy = json.loads(policy_path.read_text())
    artifacts = collect_artifact_refs(router, policy, ROOT)
    mounts = mounted_filesystems()
    storage_by_mount: dict[str, dict] = {}
    entries = []

    for artifact in sorted(artifacts.values(), key=lambda item: item["path"]):
        path = Path(artifact["path"])
        try:
            info = path.stat()
            file_entry = {
                **artifact,
                "exists": stat.S_ISREG(info.st_mode),
                "size_bytes": info.st_size,
                "mtime_ns": info.st_mtime_ns,
            }
            if args.sha256 and stat.S_ISREG(info.st_mode):
                file_entry["sha256"] = sha256_file(path)
        except FileNotFoundError:
            file_entry = {**artifact, "exists": False, "size_bytes": None, "mtime_ns": None}
        except OSError as error:
            file_entry = {**artifact, "exists": None, "error": type(error).__name__}
        file_entry["storage"] = storage_for(path, mounts)
        if file_entry.get("exists") and file_entry["storage"] and file_entry["storage"].get("mountpoint"):
            storage_by_mount.setdefault(file_entry["storage"]["mountpoint"], file_entry["storage"])
        entries.append(file_entry)

    storage = []
    for mountpoint, device in sorted(storage_by_mount.items()):
        entry = {**device, "io_probe": None}
        if args.benchmark_io:
            candidates = [
                item
                for item in entries
                if item.get("exists") and item.get("storage", {}).get("mountpoint") == mountpoint
            ]
            sample = max(candidates, key=lambda item: item["size_bytes"], default=None)
            if sample is None:
                entry["io_probe"] = {"status": "skipped", "reason": "no_artifact_on_mount"}
            else:
                sample_path = Path(sample["path"])
                sample_bytes = min(args.io_sample_mib * MIB, sample["size_bytes"])
                entry["probe_artifact"] = sample["path"]
                entry["io_probe"] = {
                    "sequential_read": fio_read_probe(sample_path, "read", sample_bytes),
                    "random_read": fio_read_probe(sample_path, "randread", sample_bytes),
                }
        storage.append(entry)

    missing = [item["path"] for item in entries if item.get("exists") is False]
    return {
        "schema_version": 1,
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": host_details(),
        "memory_policy": {
            key: policy[key]
            for key in ("gpu_reserve_mib", "host_reserve_mib", "host_budget_mib", "estimate_margin", "model_overhead_mib")
            if key in policy
        },
        "sources": {
            "router_config": str(router_path),
            "scheduler_config": str(policy_path),
        },
        "summary": {
            "artifact_count": len(entries),
            "missing_artifacts": len(missing),
            "missing_paths": missing,
            "configured_model_count": len(router.get("models", {})),
            "configured_service_count": len(policy.get("services", {})),
        },
        "artifacts": entries,
        "storage": storage,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-config", type=Path, default=ROOT / "hosts/redqueen/model-router/config.yaml")
    parser.add_argument("--scheduler-config", type=Path, default=ROOT / "hosts/redqueen/model-router/scheduler.json")
    parser.add_argument("--output", type=Path, help="write JSON atomically to this path; stdout is the default")
    parser.add_argument("--sha256", action="store_true", help="hash every configured artifact; this reads all weight bytes")
    parser.add_argument("--benchmark-io", action="store_true", help="run read-only fio direct-I/O probes on artifact mounts")
    parser.add_argument("--io-sample-mib", type=int, default=256, help="maximum bytes per fio workload and mount (default: 256 MiB)")
    args = parser.parse_args()
    if args.io_sample_mib < 1:
        parser.error("--io-sample-mib must be positive")
    args.router_config = args.router_config.resolve()
    args.scheduler_config = args.scheduler_config.resolve()
    return args


def main() -> int:
    args = parse_args()
    try:
        result = collect(args)
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"preflight failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
        return 0
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(rendered)
        os.chmod(temporary, 0o640)
        os.replace(temporary, output)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
