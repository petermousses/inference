"""Chunk-hash inventory generation for the pinned DeepSeek checkpoint."""

import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "models"))
import build_deepseek_chunk_manifest as manifest_builder
from build_deepseek_chunk_manifest import CHUNK_BYTES, build_manifest


class ChunkManifestTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / "checkpoint"
        self.checkpoint.mkdir()
        self.output = self.root / "chunks.bin"
        self.inventory = self.root / "files.tsv"
        self.shard = self.checkpoint / "model-00001-of-00001.safetensors"
        self.raw = bytes(index % 251 for index in range(CHUNK_BYTES + 17))
        self.shard.write_bytes(self.raw)
        self.inventory.write_text(
            f"{self.shard.name}\t{len(self.raw)}\t{hashlib.sha256(self.raw).hexdigest()}\t{'a' * 40}\n"
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_builds_fixed_size_chunk_hashes_after_full_file_verification(self):
        count, digest = build_manifest(self.checkpoint, self.inventory, self.output)
        expected = b"".join(
            hashlib.sha256(self.raw[offset : offset + CHUNK_BYTES]).digest()
            for offset in range(0, len(self.raw), CHUNK_BYTES)
        )
        self.assertEqual(count, 2)
        self.assertEqual(self.output.read_bytes(), expected)
        self.assertEqual(digest, hashlib.sha256(expected).hexdigest())

    def test_rejects_file_tampering_without_replacing_existing_manifest(self):
        self.output.write_bytes(b"previous manifest")
        damaged = bytearray(self.raw)
        damaged[-1] ^= 1
        self.shard.write_bytes(damaged)
        with self.assertRaisesRegex(ValueError, "SHA-256 differs from inventory"):
            build_manifest(self.checkpoint, self.inventory, self.output)
        self.assertEqual(self.output.read_bytes(), b"previous manifest")

    def test_refuses_to_replace_checkpoint_or_inventory_files(self):
        with self.assertRaisesRegex(ValueError, "outside the checkpoint directory"):
            build_manifest(self.checkpoint, self.inventory, self.shard)
        self.assertEqual(self.shard.read_bytes(), self.raw)

        with self.assertRaisesRegex(ValueError, "inventory or manifest builder"):
            build_manifest(self.checkpoint, self.inventory, self.inventory)

    def test_refuses_symlink_output_entry_inside_checkpoint(self):
        outside = self.root / "outside.bin"
        outside.write_bytes(b"keep this file")
        symlink_output = self.checkpoint / "generated.bin"
        symlink_output.symlink_to(outside)

        with self.assertRaisesRegex(ValueError, "outside the checkpoint directory"):
            build_manifest(self.checkpoint, self.inventory, symlink_output)

        self.assertTrue(symlink_output.is_symlink())
        self.assertEqual(outside.read_bytes(), b"keep this file")

    def test_pins_output_directory_if_its_symlink_is_swapped(self):
        outside = self.root / "outside"
        outside.mkdir()
        alias = self.root / "output-parent"
        alias.symlink_to(outside, target_is_directory=True)
        output = alias / "sentinel.bin"
        checkpoint_sentinel = self.checkpoint / output.name
        checkpoint_sentinel.write_bytes(b"must remain unchanged")

        def swap_parent(_):
            alias.unlink()
            alias.symlink_to(self.checkpoint, target_is_directory=True)
            return "a" * 32

        with patch.object(manifest_builder.secrets, "token_hex", side_effect=swap_parent):
            count, digest = build_manifest(self.checkpoint, self.inventory, output)

        expected = b"".join(
            hashlib.sha256(self.raw[offset : offset + CHUNK_BYTES]).digest()
            for offset in range(0, len(self.raw), CHUNK_BYTES)
        )
        self.assertEqual((count, digest), (2, hashlib.sha256(expected).hexdigest()))
        self.assertEqual((outside / output.name).read_bytes(), expected)
        self.assertEqual(checkpoint_sentinel.read_bytes(), b"must remain unchanged")

    def test_closes_every_directory_descriptor_from_ancestry_walk(self):
        real_open = os.open
        real_close = os.close
        ancestry_fds = []
        closed_fds = set()

        def tracked_open(path, flags, *args, **kwargs):
            fd = real_open(path, flags, *args, **kwargs)
            if path == ".." and "dir_fd" in kwargs:
                ancestry_fds.append(fd)
            return fd

        def tracked_close(fd):
            if fd in ancestry_fds:
                closed_fds.add(fd)
            return real_close(fd)

        with (
            patch.object(manifest_builder.os, "open", side_effect=tracked_open),
            patch.object(manifest_builder.os, "close", side_effect=tracked_close),
        ):
            build_manifest(self.checkpoint, self.inventory, self.output)

        self.assertTrue(ancestry_fds)
        self.assertEqual(set(ancestry_fds), closed_fds)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "requires named pipes")
    def test_rejects_fifo_shards_without_blocking(self):
        self.shard.unlink()
        os.mkfifo(self.shard)
        script = (
            "import sys; from pathlib import Path; "
            f"sys.path.insert(0, {str(Path(__file__).resolve().parents[2] / 'models')!r}); "
            "from build_deepseek_chunk_manifest import build_manifest; "
            "build_manifest(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))"
        )
        result = subprocess.run(
            [sys.executable, "-c", script, str(self.checkpoint), str(self.inventory), str(self.output)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("checkpoint file size/type differs", result.stderr)


if __name__ == "__main__":
    unittest.main()
