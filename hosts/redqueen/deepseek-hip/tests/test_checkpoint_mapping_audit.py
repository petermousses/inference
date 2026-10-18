from __future__ import annotations

import hashlib
import importlib
import os
import py_compile
import sys
import tempfile
import unittest
from pathlib import Path

from checkpoint_mapping_audit import (
    MappingAuditError,
    TensorMetadata,
    _pinned_inference_imports,
    audit_metadata,
    map_source_name,
    verify_pinned_inference_sources,
)


class CheckpointMappingAuditTest(unittest.TestCase):
    def pinned_source_tree(self, root: Path) -> tuple[Path, Path]:
        inference = root / "inference"
        inference.mkdir()
        source_files = {
            "model.py": b"class Transformer: pass\n",
            "config.json": b"{}\n",
            "vision.py": b"class Vision: pass\nVALUE = 'pinned'\n",
        }
        rows = []
        for name, raw in source_files.items():
            (inference / name).write_bytes(raw)
            rows.append(
                "\t".join(
                    (
                        f"inference/{name}",
                        str(len(raw)),
                        hashlib.sha256(raw).hexdigest(),
                        "fixture-object-id",
                    )
                )
            )
        inventory = root / "files.tsv"
        inventory.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return inference, inventory

    def test_verifies_every_file_in_the_pinned_inference_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            inference, inventory = self.pinned_source_tree(Path(directory))
            verify_pinned_inference_sources(inference, inventory)

    def test_rejects_changed_upstream_graph_source(self):
        with tempfile.TemporaryDirectory() as directory:
            inference, inventory = self.pinned_source_tree(Path(directory))
            (inference / "vision.py").write_text("class Changed: pass\n", encoding="utf-8")
            with self.assertRaisesRegex(MappingAuditError, "does not match pinned SHA-256"):
                verify_pinned_inference_sources(inference, inventory)

    def test_rejects_unpinned_python_source_in_inference_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            inference, inventory = self.pinned_source_tree(Path(directory))
            (inference / "extra.py").write_text("raise RuntimeError()\n", encoding="utf-8")
            with self.assertRaisesRegex(MappingAuditError, "missing or unpinned files"):
                verify_pinned_inference_sources(inference, inventory)

    def test_pinned_importer_ignores_same_size_same_timestamp_bytecode(self):
        with tempfile.TemporaryDirectory() as directory:
            inference, inventory = self.pinned_source_tree(Path(directory))
            source_path = inference / "vision.py"
            pinned_source = source_path.read_bytes()
            original_stat = source_path.stat()
            tampered_source = pinned_source.replace(b"pinned", b"tamper")
            self.assertEqual(len(tampered_source), len(pinned_source))
            source_path.write_bytes(tampered_source)
            os.utime(source_path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
            cache_dir = inference / "__pycache__"
            cache_dir.mkdir()
            py_compile.compile(
                str(source_path),
                cfile=str(cache_dir / f"vision.{sys.implementation.cache_tag}.pyc"),
                doraise=True,
                invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
            )
            source_path.write_bytes(pinned_source)
            os.utime(source_path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

            verified_sources = verify_pinned_inference_sources(inference, inventory)
            old_path = sys.path[:]
            try:
                sys.path.insert(0, str(inference))
                cached_module = importlib.import_module("vision")
                self.assertEqual(cached_module.VALUE, "tamper")
                sys.modules.pop("vision", None)
                with _pinned_inference_imports(verified_sources, inference):
                    module = importlib.import_module("vision")
                    self.assertEqual(module.VALUE, "pinned")
            finally:
                sys.modules.pop("vision", None)
                sys.path[:] = old_path

    def test_official_renames_and_tied_aliases(self):
        self.assertEqual(
            map_source_name("model.layers.2.self_attn.wq_b.weight"),
            ("layers.2.attn.wq_b.weight", None),
        )
        self.assertEqual(
            map_source_name("model.layers.2.mlp.experts.7.w1.weight"),
            ("layers.2.ffn.experts.7.w1.weight", None),
        )
        self.assertEqual(
            map_source_name("mtp.1.embed.weight"), ("embed.weight", "tied-alias")
        )
        self.assertEqual(
            map_source_name("mtp.2.head.weight"), ("head.weight", "tied-alias")
        )
        self.assertEqual(
            map_source_name("layers.0.ffn.gate.weight_scale_inv"),
            ("layers.0.ffn.gate.scale", None),
        )
        self.assertEqual(
            map_source_name("layers.3.ffn.gate.e_score_correction_bias"),
            ("layers.3.ffn.gate.bias", None),
        )

    def test_vision_and_engram_names_follow_graph_without_mlp_rewrite(self):
        self.assertEqual(
            map_source_name("vision.blocks.0.mlp.w1.weight"),
            ("vision.blocks.0.mlp.w1.weight", None),
        )
        self.assertEqual(
            map_source_name("layers.14.engram.embed.weight"),
            ("layers.14.engram.embed.weight", None),
        )

    def test_audits_backbone_mtp_vision_engram_fp4_and_wo_a(self):
        targets = {
            "embed.weight": TensorMetadata((8, 4), "bfloat16"),
            "head.weight": TensorMetadata((8, 4), "bfloat16"),
            "layers.4.attn.wo_a.weight": TensorMetadata((64, 64), "bfloat16"),
            "layers.1.engram.embed.weight": TensorMetadata((16, 8), "float8_e4m3fn"),
            "layers.0.ffn.experts.0.w1.weight": TensorMetadata((32, 16), "float4_e2m1fn_x2"),
            "layers.0.ffn.experts.0.w1.scale": TensorMetadata((32, 1), "float8_e8m0fnu"),
            "mtp.0.ffn.experts.0.w1.weight": TensorMetadata((32, 16), "float4_e2m1fn_x2"),
            "vision.blocks.0.mlp.w1.weight": TensorMetadata((32, 16), "bfloat16"),
        }
        source = {
            "embed.weight": TensorMetadata((8, 4), "BF16"),
            "head.weight": TensorMetadata((8, 4), "BF16"),
            "mtp.0.embed.weight": TensorMetadata((8, 4), "BF16"),
            "layers.4.attn.wo_a.weight": TensorMetadata((64, 64), "F8_E4M3"),
            "layers.4.attn.wo_a.scale": TensorMetadata((2, 2), "F8_E8M0"),
            "layers.1.engram.embed.weight": TensorMetadata((16, 8), "F8_E4M3"),
            "layers.0.ffn.experts.0.w1.weight": TensorMetadata((32, 16), "I8"),
            "layers.0.ffn.experts.0.w1.scale": TensorMetadata((32, 1), "F8_E8M0"),
            "mtp.0.ffn.experts.0.w1.weight": TensorMetadata((32, 16), "I8"),
            "vision.blocks.0.mlp.w1.weight": TensorMetadata((32, 16), "BF16"),
        }
        report = audit_metadata(source, targets)
        self.assertTrue(report.passed, report.errors)
        self.assertEqual(report.source_tensors, 10)
        self.assertEqual(report.target_tensors, 8)
        self.assertEqual(report.mapped_tensors, 8)
        self.assertEqual(report.tied_aliases, 1)
        self.assertEqual(report.consumed_wo_a_scales, 1)

    def test_reports_missing_unexpected_duplicate_and_shape_dtype_errors(self):
        targets = {
            "embed.weight": TensorMetadata((8, 4), "bfloat16"),
            "head.weight": TensorMetadata((8, 4), "bfloat16"),
        }
        source = {
            "embed.weight": TensorMetadata((8, 4), "BF16"),
            "model.embed.weight": TensorMetadata((8, 4), "BF16"),
            "unexpected.weight": TensorMetadata((4, 4), "BF16"),
            "head.weight": TensorMetadata((7, 4), "F16"),
        }
        report = audit_metadata(source, targets)
        self.assertFalse(report.passed)
        self.assertTrue(any("duplicate source mapping" in error for error in report.errors))
        self.assertTrue(any("unexpected converted tensor" in error for error in report.errors))
        self.assertTrue(any("shape mismatch" in error for error in report.errors))
        self.assertTrue(any("dtype mismatch" in error for error in report.errors))

    def test_tied_alias_must_match_the_backbone_shape_and_dtype(self):
        targets = {"embed.weight": TensorMetadata((8, 4), "bfloat16")}
        source = {
            "embed.weight": TensorMetadata((8, 4), "BF16"),
            "mtp.0.embed.weight": TensorMetadata((7, 4), "BF16"),
        }
        report = audit_metadata(source, targets)
        self.assertFalse(report.passed)
        self.assertTrue(any("shape mismatch" in error for error in report.errors))

    def test_wo_a_requires_exact_supported_blocks_and_a_matching_scale(self):
        targets = {"layers.0.attn.wo_a.weight": TensorMetadata((64, 64), "bfloat16")}
        bad_scale = {
            "layers.0.attn.wo_a.weight": TensorMetadata((64, 64), "F8_E4M3"),
            "layers.0.attn.wo_a.scale": TensorMetadata((2, 4), "F8_E8M0"),
        }
        report = audit_metadata(bad_scale, targets)
        self.assertFalse(report.passed)
        self.assertTrue(any("unsupported wo_a quantization blocks" in error for error in report.errors))

        missing_scale = {
            "layers.0.attn.wo_a.weight": TensorMetadata((64, 64), "F8_E4M3")
        }
        report = audit_metadata(missing_scale, targets)
        self.assertTrue(any("no checkpoint scale" in error for error in report.errors))

    def test_int8_packing_exception_is_limited_to_fp4_expert_weights(self):
        targets = {
            "layers.0.ffn.experts.0.w1.weight": TensorMetadata((32, 16), "float4_e2m1fn_x2"),
            "layers.0.attn.wq_a.weight": TensorMetadata((32, 16), "float8_e4m3fn"),
        }
        source = {
            "layers.0.ffn.experts.0.w1.weight": TensorMetadata((32, 16), "I8"),
            "layers.0.attn.wq_a.weight": TensorMetadata((32, 16), "I8"),
        }
        report = audit_metadata(source, targets)
        self.assertFalse(report.passed)
        self.assertTrue(any("dtype mismatch" in error and "wq_a" in error for error in report.errors))

    def test_only_official_bf16_to_fp32_parameter_paths_allow_checkpoint_casts(self):
        targets = {
            "vision.blocks.0.norm1.weight": TensorMetadata((8,), "float32"),
            "vision.norm.weight": TensorMetadata((8,), "float32"),
            "layers.2.attn.compressor.wkv.weight": TensorMetadata((8, 4), "float32"),
            "head.weight": TensorMetadata((16, 4), "float32"),
            "mtp.2.markov_head.head.weight": TensorMetadata((16, 4), "float32"),
            "mtp.2.confidence_head.proj.weight": TensorMetadata((1, 4), "float32"),
            "layers.0.norm.weight": TensorMetadata((8,), "float32"),
        }
        source = {
            name: TensorMetadata(metadata.shape, "BF16")
            for name, metadata in targets.items()
        }
        report = audit_metadata(source, targets)
        self.assertFalse(report.passed)
        self.assertEqual(len(report.errors), 1)
        self.assertIn("layers.0.norm.weight", report.errors[0])


if __name__ == "__main__":
    unittest.main()
