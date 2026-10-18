"""Bounded metadata and row reads for the pinned DeepSeek safetensors checkpoint."""

import concurrent.futures
import json
import struct
import tempfile
import unittest
from pathlib import Path

from checkpoint_reader import CheckpointError, SafeTensorCheckpoint


def write_shard(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header = {}
    data = bytearray()
    for name, (dtype, shape, payload) in tensors.items():
        start = len(data)
        data.extend(payload)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [start, len(data)]}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)


def checkpoint(root: Path, shards: dict[str, dict]) -> None:
    weight_map = {}
    for filename, tensors in shards.items():
        write_shard(root / filename, tensors)
        weight_map.update({name: filename for name in tensors})
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": sum(
            len(payload) for tensors in shards.values() for _, _, payload in tensors.values()
        )}, "weight_map": weight_map})
    )
    (root / "SOURCE_REVISION").write_text("revision=abc123\n")


class SafeTensorCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        checkpoint(
            self.root,
            {
                "part-1.safetensors": {
                    "table": ("U8", [5, 2], b"abcdefghij"),
                    "scalar": ("U8", [3], b"xyz"),
                },
                "part-2.safetensors": {"other": ("BF16", [1], b"\x00\x00")},
            },
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_reads_rows_in_requested_order_with_duplicates(self):
        with SafeTensorCheckpoint(self.root, expected_revision="abc123") as source:
            self.assertEqual(source.tensor_count, 3)
            self.assertEqual(source.read_rows("table", [4, 1, 2, 1]), b"ijcdefcd")
            self.assertEqual(source.read_tensor_bytes("scalar"), b"xyz")

    def test_reads_concurrently_from_shared_shard_descriptors(self):
        with SafeTensorCheckpoint(self.root, expected_revision="abc123") as source:
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(lambda _: source.read_rows("table", [3, 0]), range(32)))
        self.assertEqual(results, [b"ghab"] * 32)

    def test_rejects_revision_mismatch(self):
        with self.assertRaisesRegex(CheckpointError, "revision"):
            SafeTensorCheckpoint(self.root, expected_revision="wrong")

    def test_rejects_index_to_header_mismatch(self):
        index_path = self.root / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        index["weight_map"]["missing"] = "part-1.safetensors"
        index_path.write_text(json.dumps(index))
        with self.assertRaisesRegex(CheckpointError, "index.*header|header.*index"):
            SafeTensorCheckpoint(self.root, expected_revision="abc123")

    def test_rejects_out_of_bounds_tensor_offsets(self):
        path = self.root / "part-1.safetensors"
        raw = path.read_bytes()
        header_size = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8 : 8 + header_size])
        header["table"]["data_offsets"] = [0, 11]
        encoded = json.dumps(header, separators=(",", ":")).encode()
        path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + raw[8 + header_size :])
        with self.assertRaisesRegex(CheckpointError, "offset|payload|size"):
            SafeTensorCheckpoint(self.root, expected_revision="abc123")

    def test_rejects_reads_over_the_explicit_memory_limit(self):
        with SafeTensorCheckpoint(self.root, expected_revision="abc123") as source:
            with self.assertRaisesRegex(ValueError, "limit"):
                source.read_tensor_bytes("table", max_bytes=9)
            with self.assertRaisesRegex(ValueError, "limit"):
                source.read_rows("table", [0, 1, 2, 3, 4], max_bytes=9)
            with self.assertRaisesRegex(ValueError, "hard limit"):
                source.read_tensor_bytes("scalar", max_bytes=256 * 1024 * 1024 + 1)

    def test_rejects_shard_replacement_after_metadata_validation(self):
        with SafeTensorCheckpoint(self.root, expected_revision="abc123") as source:
            path = self.root / "part-1.safetensors"
            path.write_bytes(path.read_bytes())
            with self.assertRaisesRegex(CheckpointError, "changed since metadata validation"):
                source.read_tensor_bytes("scalar")

    def test_rejects_row_batches_over_the_hard_count_limit(self):
        with SafeTensorCheckpoint(self.root, expected_revision="abc123") as source:
            with self.assertRaisesRegex(ValueError, "hard limit.*rows"):
                source.read_rows("table", (0 for _ in range(65_537)))

    def test_rejects_use_after_close(self):
        source = SafeTensorCheckpoint(self.root, expected_revision="abc123")
        source.close()
        source.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            source.read_tensor_bytes("scalar")


if __name__ == "__main__":
    unittest.main()
