"""Metadata-only audit for the pinned 1017 checkpoint-to-graph mapping.

The source-key rules mirror the official inference/convert.py conversion with
model-parallel size one.  This module does not read tensor payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import importlib.util
import json
import os
import re
import stat
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from types import ModuleType
from typing import Iterator, Mapping


_SPECIAL_FINAL_COMPONENTS = ("hc", "attn_sink", "tie2eid", "tid2eid", "ape", "image_")
_DTYPE_NAMES = {
    "BOOL": "bool",
    "U8": "uint8",
    "I8": "int8",
    "I16": "int16",
    "I32": "int32",
    "I64": "int64",
    "F16": "float16",
    "BF16": "bfloat16",
    "F32": "float32",
    "F64": "float64",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E4M3FN": "float8_e4m3fn",
    "F8_E4M3FNUZ": "float8_e4m3fnuz",
    "F8_E5M2": "float8_e5m2",
    "F8_E5M2FNUZ": "float8_e5m2fnuz",
    "F8_E8M0": "float8_e8m0fnu",
}
_TIED_MTP_ALIASES = {"embed.weight", "head.weight"}
_BF16_TO_FP32_TARGETS = (
    re.compile(r"vision\.blocks\.\d+\.norm[12]\.weight\Z"),
    re.compile(r"layers\.\d+\.attn\.compressor\.w(?:kv|gate)\.weight\Z"),
    re.compile(r"mtp\.\d+\.markov_head\.head\.weight\Z"),
    re.compile(r"mtp\.\d+\.confidence_head\.proj\.weight\Z"),
)


def _official_checkpoint_cast(_source_name: str, target_name: str, source_dtype: str, target_dtype: str) -> bool:
    """BF16 tensors copied into these explicitly-FP32 upstream parameters."""
    if source_dtype != "bfloat16" or target_dtype != "float32":
        return False
    return target_name == "vision.norm.weight" or target_name == "head.weight" or any(
        pattern.fullmatch(target_name) for pattern in _BF16_TO_FP32_TARGETS
    )


@dataclass(frozen=True)
class TensorMetadata:
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class AuditReport:
    source_tensors: int
    target_tensors: int
    mapped_tensors: int
    tied_aliases: int
    consumed_wo_a_scales: int
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.errors


class MappingAuditError(RuntimeError):
    """The pinned checkpoint does not match the official inference graph."""


def _read_pinned_source(path: Path, *, limit: int) -> bytes:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise MappingAuditError("platform does not support no-follow inference source opens")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0) | nofollow
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise MappingAuditError(f"cannot safely open pinned inference file: {path}") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise MappingAuditError(f"pinned inference file is not a bounded regular file: {path}")
        chunks = []
        total = 0
        while True:
            chunk = os.read(fd, min(1024 * 1024, limit + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > limit:
                raise MappingAuditError(f"pinned inference file exceeds {limit} bytes: {path}")
        after = os.fstat(fd)
        identity = lambda result: (
            result.st_dev, result.st_ino, result.st_size, result.st_mtime_ns, result.st_ctime_ns
        )
        if identity(before) != identity(after):
            raise MappingAuditError(f"pinned inference file changed while read: {path}")
        return b"".join(chunks)
    except OSError as exc:
        raise MappingAuditError(f"cannot read pinned inference file: {path}") from exc
    finally:
        os.close(fd)


def verify_pinned_inference_sources(
    inference_dir: Path, inventory_path: Path | None = None
) -> dict[str, bytes]:
    """Verify the complete official inference snapshot before executing its graph."""
    try:
        root = inference_dir.resolve(strict=True)
    except OSError as exc:
        raise MappingAuditError(f"pinned inference source directory is unavailable: {inference_dir}") from exc
    if not root.is_dir():
        raise MappingAuditError(f"pinned inference source path is not a directory: {root}")
    if inventory_path is None:
        inventory_path = Path(__file__).resolve().parents[2] / "models" / "deepseek-v4.1-flash-1017.files.tsv"
    raw_inventory = _read_pinned_source(inventory_path, limit=8 * 1024 * 1024)
    expected: dict[str, tuple[int, str]] = {}
    try:
        lines = raw_inventory.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise MappingAuditError("pinned model file inventory is not UTF-8") from exc
    for line_number, line in enumerate(lines, 1):
        fields = line.split("\t")
        if len(fields) != 4:
            raise MappingAuditError(f"invalid pinned model inventory row {line_number}")
        filename, size_text, digest, object_id = fields
        if not filename.startswith("inference/"):
            continue
        relative = PurePosixPath(filename.removeprefix("inference/"))
        if (
            relative.is_absolute()
            or not relative.parts
            or "\\" in filename
            or relative.as_posix() != filename.removeprefix("inference/")
            or any(part in {"", ".", ".."} for part in relative.parts)
            or filename in expected
        ):
            raise MappingAuditError(f"invalid or duplicate pinned inference path at row {line_number}")
        try:
            size = int(size_text)
        except ValueError as exc:
            raise MappingAuditError(f"invalid pinned inference file size at row {line_number}") from exc
        if (
            size < 0
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not object_id
        ):
            raise MappingAuditError(f"invalid pinned inference file identity at row {line_number}")
        expected[str(relative)] = (size, digest)
    required = {"model.py", "config.json"}
    if not required <= expected.keys():
        raise MappingAuditError("pinned model inventory lacks inference/model.py or inference/config.json")

    source_bytes: dict[str, bytes] = {}
    for relative, (expected_size, expected_digest) in expected.items():
        path = root.joinpath(*PurePosixPath(relative).parts)
        current = root
        for part in PurePosixPath(relative).parts:
            current = current / part
            if current.is_symlink():
                raise MappingAuditError(f"pinned inference path contains a symlink: {current}")
        raw = _read_pinned_source(path, limit=16 * 1024 * 1024)
        if len(raw) != expected_size or hashlib.sha256(raw).hexdigest() != expected_digest:
            raise MappingAuditError(f"inference file does not match pinned SHA-256: {relative}")
        source_bytes[relative] = raw

    actual_files: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts:
            continue
        if path.is_symlink():
            raise MappingAuditError(f"inference source tree contains an unpinned symlink: {path}")
        if path.is_file():
            actual_files.add(relative.as_posix())
    if actual_files != expected.keys():
        raise MappingAuditError("inference source tree has missing or unpinned files")
    return source_bytes


class _PinnedSourceLoader(importlib.abc.Loader):
    def __init__(self, module_name: str, path: Path, source: bytes, *, is_package: bool) -> None:
        self.module_name = module_name
        self.path = path
        self.source = source
        self.package = is_package

    def create_module(self, spec):
        return None

    def exec_module(self, module: ModuleType) -> None:
        module.__file__ = str(self.path)
        module.__cached__ = None
        module.__package__ = self.module_name if self.package else self.module_name.rpartition(".")[0]
        exec(compile(self.source, str(self.path), "exec"), module.__dict__)


class _PinnedInferenceFinder(importlib.abc.MetaPathFinder):
    def __init__(self, source_bytes: Mapping[str, bytes], inference_dir: Path) -> None:
        self._modules: dict[str, _PinnedSourceLoader] = {}
        for relative, source in source_bytes.items():
            source_path = PurePosixPath(relative)
            if source_path.suffix != ".py":
                continue
            package = source_path.name == "__init__.py"
            module_parts = source_path.parts[:-1] if package else (*source_path.parts[:-1], source_path.stem)
            if not module_parts:
                continue
            module_name = ".".join(module_parts)
            if module_name == "kernel":
                continue
            if module_name in self._modules:
                raise MappingAuditError(f"pinned inference modules have a duplicate name: {module_name}")
            self._modules[module_name] = _PinnedSourceLoader(
                module_name,
                inference_dir.joinpath(*source_path.parts),
                source,
                is_package=package,
            )

    def find_spec(self, fullname: str, path=None, target=None):
        loader = self._modules.get(fullname)
        if loader is None:
            return None
        return importlib.util.spec_from_loader(
            fullname,
            loader,
            origin=str(loader.path),
            is_package=loader.package,
        )


@contextmanager
def _pinned_inference_imports(
    source_bytes: Mapping[str, bytes], inference_dir: Path
) -> Iterator[None]:
    """Route pinned upstream module imports through their verified in-memory source bytes."""
    finder = _PinnedInferenceFinder(source_bytes, inference_dir)
    previous = {name: sys.modules.pop(name, None) for name in finder._modules}
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
        for name, old_module in previous.items():
            sys.modules.pop(name, None)
            if old_module is not None:
                sys.modules[name] = old_module


def _execute_pinned_model(source: bytes, path: Path) -> dict[str, object]:
    module_name = "redqueen_checkpoint_mapping_reference"
    old_module = sys.modules.pop(module_name, None)
    module = ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = ""
    module.__cached__ = None
    sys.modules[module_name] = module
    try:
        exec(compile(source, str(path), "exec"), module.__dict__)
        return dict(module.__dict__)
    finally:
        sys.modules.pop(module_name, None)
        if old_module is not None:
            sys.modules[module_name] = old_module


def map_source_name(source_name: str) -> tuple[str | None, str | None]:
    """Return the converted target name and special handling, if any.

    The return mode is ``None`` for ordinary tensors, ``tied-alias`` for the
    two MTP aliases skipped by convert.py, and ``wo-a-scale`` for the scale
    consumed while converting an attention ``wo_a.weight`` to BF16.
    """
    name = source_name
    if name.startswith("model."):
        name = name[len("model.") :]

    if name.startswith("mtp.") and name.split(".", 2)[-1] in _TIED_MTP_ALIASES:
        return name.split(".", 2)[-1], "tied-alias"

    name = name.replace("self_attn", "attn")
    if not name.startswith("vision."):
        name = name.replace("mlp", "ffn")
    name = name.replace("weight_scale_inv", "scale")
    name = name.replace("e_score_correction_bias", "bias")

    if any(component in name for component in _SPECIAL_FINAL_COMPONENTS):
        key = name.split(".")[-1]
    else:
        parts = name.split(".")
        if len(parts) < 2:
            return name, None
        key = parts[-2]
    if name.endswith(".wo_a.scale"):
        return name, "wo-a-scale"
    return name, None


def _dtype_name(dtype: str) -> str:
    value = str(dtype)
    if value.startswith("torch."):
        value = value[6:]
    return _DTYPE_NAMES.get(value, value)


def metadata_from_tensor(tensor: object) -> TensorMetadata:
    """Extract shape and dtype without materializing or copying tensor data."""
    shape = tuple(int(dimension) for dimension in tensor.shape)  # type: ignore[attr-defined]
    return TensorMetadata(shape=shape, dtype=_dtype_name(str(tensor.dtype)))  # type: ignore[attr-defined]


def collect_target_state(model: object) -> tuple[dict[str, TensorMetadata], frozenset[str]]:
    """Collect persistent model state, omitting only official tied MTP aliases."""
    materialized = []
    for name, tensor in model.named_parameters(remove_duplicate=False):  # type: ignore[attr-defined]
        if not tensor.is_meta:
            materialized.append(name)
    for name, tensor in model.named_buffers(remove_duplicate=False):  # type: ignore[attr-defined]
        if not tensor.is_meta:
            materialized.append(name)
    if materialized:
        raise MappingAuditError(
            "upstream model allocated non-meta tensors during audit: " + ", ".join(materialized[:8])
        )

    state = model.state_dict(keep_vars=True)  # type: ignore[attr-defined]
    targets: dict[str, TensorMetadata] = {}
    aliases: set[str] = set()
    for name, tensor in state.items():
        parts = name.split(".", 2)
        if len(parts) == 3 and parts[0] == "mtp" and parts[2] in _TIED_MTP_ALIASES:
            canonical = parts[2]
            if canonical not in state or tensor is not state[canonical]:
                raise MappingAuditError(f"upstream graph declares an untied MTP alias: {name}")
            aliases.add(name)
            continue
        targets[name] = metadata_from_tensor(tensor)
    return targets, frozenset(aliases)


def _metadata(value: object) -> TensorMetadata:
    shape = tuple(int(dimension) for dimension in value.shape)  # type: ignore[attr-defined]
    return TensorMetadata(shape=shape, dtype=_dtype_name(value.dtype))  # type: ignore[attr-defined]


def _compare(
    source_name: str,
    source: TensorMetadata,
    target_name: str,
    target: TensorMetadata,
    *,
    allow_fp4_pack: bool = False,
) -> list[str]:
    errors = []
    if source.shape != target.shape:
        errors.append(
            f"shape mismatch {source_name} -> {target_name}: source {source.shape}, target {target.shape}"
        )
    source_dtype = _dtype_name(source.dtype)
    target_dtype = _dtype_name(target.dtype)
    if source_dtype != target_dtype:
        fp4_source = (
            allow_fp4_pack
            and source_dtype == "int8"
            and target_dtype == "float4_e2m1fn_x2"
        )
        if not fp4_source and not _official_checkpoint_cast(
            source_name, target_name, source_dtype, target_dtype
        ):
            errors.append(
                f"dtype mismatch {source_name} -> {target_name}: source {source_dtype}, target {target_dtype}"
            )
    return errors


def _validate_wo_a_scale(
    weight_name: str,
    weight: TensorMetadata,
    scale_name: str,
    scale: TensorMetadata,
    target: TensorMetadata,
) -> list[str]:
    errors = []
    if weight.shape != target.shape:
        errors.append(
            f"shape mismatch {weight_name} -> {weight_name}: source {weight.shape}, target {target.shape}"
        )
    if _dtype_name(target.dtype) != "bfloat16":
        errors.append(f"wo_a target must be BF16 after official dequantization: {weight_name}")
    if _dtype_name(weight.dtype) != "float8_e4m3fn":
        errors.append(f"wo_a source must be FP8 E4M3: {weight_name} has {_dtype_name(weight.dtype)}")
    if _dtype_name(scale.dtype) != "float8_e8m0fnu":
        errors.append(f"wo_a scale must be FP8 E8M0: {scale_name} has {_dtype_name(scale.dtype)}")
    if len(weight.shape) != 2 or len(scale.shape) != 2:
        errors.append(f"wo_a weight and scale must be rank two: {weight_name}, {scale_name}")
        return errors
    if any(dimension <= 0 for dimension in (*weight.shape, *scale.shape)):
        errors.append(f"wo_a weight and scale dimensions must be positive: {weight_name}, {scale_name}")
        return errors
    if any(weight_dim % scale_dim for weight_dim, scale_dim in zip(weight.shape, scale.shape)):
        errors.append(f"wo_a scale must tile the weight exactly: {weight_name}, {scale_name}")
        return errors
    block_shape = tuple(weight_dim // scale_dim for weight_dim, scale_dim in zip(weight.shape, scale.shape))
    if block_shape not in {(32, 32), (128, 128)}:
        errors.append(f"unsupported wo_a quantization blocks {block_shape}: {weight_name}")
    return errors


def audit_metadata(
    source_tensors: Mapping[str, object],
    target_tensors: Mapping[str, TensorMetadata],
) -> AuditReport:
    """Compare checkpoint headers with graph state; never calls a payload reader."""
    errors: list[str] = []
    mapped: dict[str, str] = {}
    aliases = 0
    tied_sources: list[tuple[str, TensorMetadata, str]] = []
    wo_a_scales: dict[str, tuple[str, TensorMetadata]] = {}
    ordinary_sources: dict[str, tuple[str, TensorMetadata, str | None]] = {}

    for source_name, raw_metadata in source_tensors.items():
        source = _metadata(raw_metadata)
        target_name, mode = map_source_name(source_name)
        if target_name is None:
            errors.append(f"converter skipped an unrecognized source tensor: {source_name}")
            continue
        if mode == "wo-a-scale":
            weight_name = target_name.removesuffix(".scale") + ".weight"
            if weight_name in wo_a_scales:
                errors.append(f"duplicate wo_a scale sources for {weight_name}")
            else:
                wo_a_scales[weight_name] = (source_name, source)
            continue
        if mode == "tied-alias":
            tied_sources.append((source_name, source, target_name))
            continue
        if target_name in ordinary_sources:
            errors.append(
                f"duplicate source mapping to {target_name}: {ordinary_sources[target_name][0]} and {source_name}"
            )
            continue
        ordinary_sources[target_name] = (source_name, source, mode)

    for target_name, (source_name, source, mode) in ordinary_sources.items():
        target = target_tensors.get(target_name)
        if target is None:
            if mode == "tied-alias":
                errors.append(f"tied MTP alias has no canonical graph tensor: {source_name} -> {target_name}")
            else:
                errors.append(f"unexpected converted tensor: {source_name} -> {target_name}")
            continue
        allow_fp4_pack = ".experts." in target_name and target_name.endswith(".weight")
        if target_name.endswith(".wo_a.weight"):
            if source.shape != target.shape:
                errors.append(
                    f"shape mismatch {source_name} -> {target_name}: source {source.shape}, target {target.shape}"
                )
        else:
            errors.extend(
                _compare(source_name, source, target_name, target, allow_fp4_pack=allow_fp4_pack)
            )
        mapped[target_name] = source_name

    for source_name, source, target_name in tied_sources:
        target = target_tensors.get(target_name)
        if target is None:
            errors.append(f"tied MTP alias has no canonical graph tensor: {source_name} -> {target_name}")
            continue
        aliases += 1
        errors.extend(_compare(source_name, source, target_name, target))

    for target_name in sorted(target_tensors):
        if target_name not in mapped:
            errors.append(f"missing checkpoint tensor for graph state: {target_name}")

    for target_name, (scale_source_name, scale) in wo_a_scales.items():
        target = target_tensors.get(target_name)
        source_entry = ordinary_sources.get(target_name)
        if target is None:
            errors.append(f"wo_a scale has no graph weight: {scale_source_name} -> {target_name}")
        elif source_entry is None:
            errors.append(f"wo_a scale has no checkpoint weight: {scale_source_name} -> {target_name}")
        else:
            weight_source_name, weight, mode = source_entry
            if mode == "tied-alias":
                errors.append(f"wo_a weight cannot be a tied alias: {weight_source_name}")
            errors.extend(
                _validate_wo_a_scale(
                    weight_source_name, weight, scale_source_name, scale, target
                )
            )

    for target_name, (source_name, source, mode) in ordinary_sources.items():
        if target_name.endswith(".wo_a.weight") and mode != "tied-alias" and target_name not in wo_a_scales:
            errors.append(f"wo_a weight has no checkpoint scale: {source_name}")

    return AuditReport(
        source_tensors=len(source_tensors),
        target_tensors=len(target_tensors),
        mapped_tensors=len(mapped),
        tied_aliases=aliases,
        consumed_wo_a_scales=len(wo_a_scales),
        errors=tuple(errors),
    )


def _read_revision(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise MappingAuditError(f"cannot read checkpoint revision marker: {path}") from exc
    fields = dict(line.split("=", 1) for line in lines if "=" in line)
    revision = fields.get("revision")
    if not revision:
        raise MappingAuditError(f"checkpoint revision marker has no revision: {path}")
    return revision


def run_audit(checkpoint_root: Path, inference_dir: Path, adapter_dir: Path) -> AuditReport:
    """Build the pinned official graph on meta and audit real checkpoint headers."""
    source_bytes = verify_pinned_inference_sources(inference_dir)
    model_path = inference_dir / "model.py"
    expected_revision = _read_revision(checkpoint_root / "SOURCE_REVISION")

    previous_path = sys.path[:]
    previous_kernel = sys.modules.pop("kernel", None)
    sys.path[:0] = [str(adapter_dir), str(inference_dir)]
    try:
        import kernel  # noqa: PLC0415

        if expected_revision != kernel.REFERENCE_COMMIT:
            raise MappingAuditError(
                f"checkpoint revision {expected_revision!r} does not match upstream graph pin "
                f"{kernel.REFERENCE_COMMIT!r}"
            )
        sys.modules["kernel"] = kernel

        try:
            from transformers import AutoTokenizer
            from checkpoint_reader import SafeTensorCheckpoint
        except ImportError as exc:
            raise MappingAuditError(f"audit dependency is unavailable: {exc.name}") from exc

        with _pinned_inference_imports(source_bytes, inference_dir):
            reference = _execute_pinned_model(source_bytes["model.py"], model_path)
            try:
                model_config = json.loads(source_bytes["config.json"].decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise MappingAuditError(f"cannot parse pinned upstream model config: {exc}") from exc
            if not isinstance(model_config, dict):
                raise MappingAuditError("pinned upstream model config is not an object")

            torch = reference["torch"]
            previous_default_dtype = torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.bfloat16)
                args = reference["ModelArgs"](**model_config)
                # These only size non-persistent runtime caches, not trained parameters.
                args.max_batch_size = 1
                args.max_seq_len = 1
                tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_root), local_files_only=True)
                with torch.device("meta"):
                    model = reference["Transformer"](args, tokenizer)
                targets, _ = collect_target_state(model)

                with SafeTensorCheckpoint(
                    checkpoint_root, expected_revision=expected_revision, metadata_only=True
                ) as checkpoint:
                    report = audit_metadata(checkpoint.tensors, targets)
            finally:
                torch.set_default_dtype(previous_default_dtype)
    finally:
        sys.path[:] = previous_path
        sys.modules.pop("kernel", None)
        if previous_kernel is not None:
            sys.modules["kernel"] = previous_kernel

    return report


def main(argv: list[str] | None = None) -> int:
    default_inference_dir = Path(
        os.environ.get(
            "DS4_INFERENCE_DIR", "/srv/ai/models/deepseek-v4.1-flash-1017/inference"
        )
    ).expanduser()
    parser = argparse.ArgumentParser(
        description="Audit pinned DeepSeek checkpoint tensor metadata against the official graph."
    )
    parser.add_argument(
        "--inference-dir", type=Path, default=default_inference_dir,
        help="pinned official inference source directory",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(os.environ.get("DS4_CHECKPOINT_DIR", str(default_inference_dir.parent))).expanduser(),
        help="pinned checkpoint directory (weights are not read)",
    )
    args = parser.parse_args(argv)
    adapter_dir = Path(__file__).resolve().parent
    inference_dir = args.inference_dir.expanduser()
    checkpoint_root = args.checkpoint_dir.expanduser()
    report = run_audit(checkpoint_root, inference_dir, adapter_dir)
    print(
        "metadata-only checkpoint mapping audit: "
        f"{'PASS' if report.passed else 'FAIL'}; "
        f"source={report.source_tensors}, target={report.target_tensors}, "
        f"mapped={report.mapped_tensors}, tied_aliases={report.tied_aliases}, "
        f"wo_a_scales_consumed={report.consumed_wo_a_scales}"
    )
    for error in report.errors[:30]:
        print(f"error: {error}")
    if len(report.errors) > 30:
        print(f"error: … and {len(report.errors) - 30} more")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
