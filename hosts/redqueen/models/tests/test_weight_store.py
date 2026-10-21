from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    "redqueen_weight_store", ROOT / "hosts/redqueen/models/weight_store.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load weight_store.py")
weight_store = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = weight_store
SPEC.loader.exec_module(weight_store)


def gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def make_gguf(*, tensor_type: int = 0, alignment: int = 64, tensor_offset: int = 0) -> bytes:
    metadata = b"".join(
        (
            gguf_string("general.architecture")
            + struct.pack("<I", 8)
            + gguf_string("llama"),
            gguf_string("general.alignment")
            + struct.pack("<II", 4, alignment),
            gguf_string("general.file_type")
            + struct.pack("<II", 4, 15),
            gguf_string("tokenizer.ggml.tokens")
            + struct.pack("<IIQ", 9, 8, 2)
            + gguf_string("alpha")
            + gguf_string("beta"),
        )
    )
    tensor_info = (
        gguf_string("token_embd.weight")
        + struct.pack("<IQQIQ", 2, 2, 2, tensor_type, tensor_offset)
    )
    header = b"GGUF" + struct.pack("<IQQ", 3, 1, 4) + metadata + tensor_info
    data_start = (len(header) + alignment - 1) // alignment * alignment
    return header + (b"\0" * (data_start - len(header))) + bytes(range(16))


class TensorHeaderTests(unittest.TestCase):
    def test_safetensors_records_exact_dtype_shape_and_data_range(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.safetensors"
            header = {
                "__metadata__": {"format": "pt"},
                "embedding": {"dtype": "F16", "shape": [2, 2], "data_offsets": [0, 8]},
                "norm": {"dtype": "F32", "shape": [1], "data_offsets": [8, 12]},
            }
            encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
            encoded += b" " * ((-len(encoded)) % 8)
            path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(12))

            result = weight_store.parse_safetensors(path)

            self.assertEqual(result["tensor_layout"], "safetensors-native")
            self.assertEqual(result["tensors"][0]["name"], "embedding")
            self.assertEqual(result["tensors"][0]["dtype"], "F16")
            self.assertEqual(result["tensors"][0]["shape"], [2, 2])
            self.assertEqual(result["tensors"][0]["data_offset"], 8 + len(encoded))
            self.assertEqual(result["tensors"][0]["data_bytes"], 8)

    def test_safetensors_rejects_overlapping_offsets(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.safetensors"
            header = {
                "first": {"dtype": "U8", "shape": [8], "data_offsets": [0, 8]},
                "second": {"dtype": "U8", "shape": [4], "data_offsets": [4, 8]},
            }
            encoded = json.dumps(header).encode("utf-8")
            path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(8))

            with self.assertRaisesRegex(ValueError, "overlap"):
                weight_store.parse_safetensors(path)

    def test_safetensors_rejects_unindexed_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "hole.safetensors"
            header = {
                "first": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]},
                "second": {"dtype": "U8", "shape": [3], "data_offsets": [5, 8]},
            }
            encoded = json.dumps(header).encode("utf-8")
            path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(8))

            with self.assertRaisesRegex(ValueError, "hole"):
                weight_store.parse_safetensors(path)

    def test_safetensors_rejects_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "duplicate.safetensors"
            header = b'{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}'
            path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(1))

            with self.assertRaisesRegex(ValueError, "duplicate"):
                weight_store.parse_safetensors(path)

    def test_gguf_records_native_metadata_and_aligned_tensor_span(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.gguf"
            path.write_bytes(make_gguf(alignment=64))

            result = weight_store.parse_gguf(path)

            self.assertEqual(result["version"], 3)
            self.assertEqual(result["architecture"], "llama")
            self.assertEqual(result["file_type"], 15)
            self.assertEqual(result["alignment"], 64)
            self.assertEqual(result["tensor_layout"], "ggml-native")
            self.assertEqual(result["tensors"][0]["shape"], [2, 2])
            self.assertEqual(result["tensors"][0]["ggml_type_id"], 0)
            self.assertEqual(result["tensors"][0]["data_offset"], result["data_start"])
            self.assertEqual(result["tensors"][0]["storage_span_bytes"], 16)

    def test_gguf_rejects_tensor_offsets_outside_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.gguf"
            path.write_bytes(make_gguf(tensor_offset=1_000_000))

            with self.assertRaisesRegex(ValueError, "outside|span"):
                weight_store.parse_gguf(path)


class WeightStoreTests(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[Path, Path, dict[str, dict], bytes]:
        model_root = root / "models"
        model_dir = model_root / "toy"
        metadata_dir = root / "metadata" / "toy"
        model_dir.mkdir(parents=True)
        metadata_dir.mkdir(parents=True)
        path = model_dir / "model.safetensors"
        payload = bytes(range(16))
        header = {"weight": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}
        encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
        encoded += b" " * ((-len(encoded)) % 8)
        file_data = struct.pack("<Q", len(encoded)) + encoded + payload
        path.write_bytes(file_data)
        digest = hashlib.sha256(file_data).hexdigest()
        (metadata_dir / "SOURCE_REVISION").write_text(
            "weights_repository=org/toy\nweights_revision=0123456789abcdef\n"
        )
        (metadata_dir / "SHA256SUMS").write_text(f"{digest}  model.safetensors\n")
        refs = {
            str(path): {
                "path": str(path),
                "references": [{"model": "toy-model", "role": "primary", "class": "text"}],
            }
        }
        return model_root, root / "metadata", refs, file_data

    def test_catalog_imports_pinned_artifact_without_rewriting_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_root, metadata_root, refs, original = self.fixture(root)

            catalog = weight_store.build_store(
                refs, model_root=model_root, metadata_root=metadata_root, verify_hashes=True
            )

            self.assertEqual(catalog["format"], "redqueen.weight-store")
            self.assertEqual(catalog["version"], 1)
            self.assertEqual(catalog["storage_mode"], "source-backed")
            artifact = catalog["artifacts"][0]
            self.assertEqual(artifact["path"], "toy/model.safetensors")
            self.assertEqual(artifact["integrity"]["sha256"], hashlib.sha256(original).hexdigest())
            self.assertTrue(artifact["integrity"]["verified_at_build"])
            self.assertEqual(artifact["conversion"]["mode"], "identity-import")
            self.assertEqual(artifact["conversion"]["payload_changed"], False)
            self.assertEqual(catalog["models"][0]["id"], "toy-model")
            self.assertEqual(catalog["summary"]["tensor_count"], 1)
            self.assertEqual((model_root / artifact["path"]).read_bytes(), original)

    def test_catalog_rejects_artifacts_outside_model_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_root, metadata_root, refs, _ = self.fixture(root)
            path = root / "outside.safetensors"
            path.write_bytes(b"outside")
            refs[str(path)] = {"path": str(path), "references": [{"model": "toy-model", "role": "primary"}]}

            with self.assertRaisesRegex(ValueError, "outside the model root"):
                weight_store.build_store(refs, model_root=model_root, metadata_root=metadata_root)

    def test_catalog_rejects_symlink_that_escapes_model_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_root, metadata_root, refs, _ = self.fixture(root)
            outside = root / "outside.safetensors"
            outside.write_bytes(b"outside")
            link = model_root / "toy" / "escape.safetensors"
            link.symlink_to(outside)
            refs[str(link)] = {"path": str(link), "references": [{"model": "toy-model", "role": "primary"}]}

            with self.assertRaisesRegex(ValueError, "resolves outside the model root"):
                weight_store.build_store(refs, model_root=model_root, metadata_root=metadata_root)

    def test_catalog_does_not_guess_checksum_from_same_basename(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_root, metadata_root, refs, original = self.fixture(root)
            del refs[next(iter(refs))]
            nested = model_root / "toy" / "weights" / "model.safetensors"
            nested.parent.mkdir()
            nested.write_bytes(original)
            other_digest = hashlib.sha256(b"different model payload").hexdigest()
            (metadata_root / "toy" / "SHA256SUMS").write_text(
                f"{other_digest}  another-directory/model.safetensors\n"
            )
            refs[str(nested)] = {
                "path": str(nested),
                "references": [{"model": "toy-model", "role": "primary"}],
            }

            with self.assertRaisesRegex(ValueError, "no pinned SHA-256"):
                weight_store.build_store(refs, model_root=model_root, metadata_root=metadata_root)

    def test_atomic_writer_replaces_symlink_without_touching_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.json"
            target.write_text("keep")
            output = root / "catalog.json"
            output.symlink_to(target)

            weight_store.write_atomic(output, {"format": "redqueen.weight-store", "version": 1})

            self.assertEqual(target.read_text(), "keep")
            self.assertFalse(output.is_symlink())
            self.assertEqual(json.loads(output.read_text())["version"], 1)


if __name__ == "__main__":
    unittest.main()
