"""Bounded metadata and row reads for the pinned DeepSeek safetensors checkpoint."""

import concurrent.futures
import hashlib
import json
import os
import struct
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import checkpoint_reader
from checkpoint_reader import CheckpointError, ReadMemoryBudget, SafeTensorCheckpoint

CHUNK_SIZE = 1024 * 1024


def write_shard(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header = {}
    data = bytearray()
    for name, (dtype, shape, payload) in tensors.items():
        start = len(data)
        data.extend(payload)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [start, len(data)]}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)


def checkpoint(root: Path, shards: dict[str, dict]) -> dict[str, tuple[int, str]]:
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
    filenames = [*shards, "model.safetensors.index.json"]
    return {
        filename: (path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
        for filename in filenames
        for path in [root / filename]
    }


def chunk_digest_bytes(root: Path, inventory: dict[str, tuple[int, str]]) -> bytes:
    digests = []
    for filename in sorted(name for name in inventory if name.endswith(".safetensors")):
        raw = (root / filename).read_bytes()
        digests.extend(
            hashlib.sha256(raw[offset : offset + CHUNK_SIZE]).digest()
            for offset in range(0, len(raw), CHUNK_SIZE)
        )
    return b"".join(digests)


class SafeTensorCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.inventory = checkpoint(
            self.root,
            {
                "part-1.safetensors": {
                    "table": ("U8", [5, 2], b"abcdefghij"),
                    "scalar": ("U8", [3], b"xyz"),
                },
                "part-2.safetensors": {"other": ("BF16", [1], b"\x00\x00")},
            },
        )

    def reader(self, *, verify_shard_digests=True):
        return SafeTensorCheckpoint(
            self.root,
            expected_revision="abc123",
            expected_file_inventory=self.inventory,
            verify_shard_digests=verify_shard_digests,
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_reads_rows_in_requested_order_with_duplicates(self):
        with self.reader() as source:
            self.assertEqual(source.tensor_count, 3)
            self.assertEqual(source.read_rows("table", [4, 1, 2, 1]), b"ijcdefcd")
            self.assertEqual(source.read_tensor_bytes("scalar"), b"xyz")

    def test_read_rows_rejects_unbounded_lazy_iterables(self):
        with self.reader() as source:
            with self.assertRaisesRegex(TypeError, "built-in list, tuple, or range"):
                source.read_rows("table", (row for row in (1, 2)))

    def test_rows_are_snapshotted_before_waiting_for_memory_admission(self):
        entered = threading.Event()
        release = threading.Event()

        class BlockingBudget:
            def reserve(self, amount):
                class Reservation:
                    def __enter__(self_inner):
                        entered.set()
                        if not release.wait(timeout=2):
                            raise TimeoutError("test did not release row read")

                    def __exit__(self_inner, exc_type, exc_value, traceback):
                        return False

                return Reservation()

        with self.reader() as source:
            source.tensors["table"]
            source._read_memory_budget = BlockingBudget()
            rows = [4, 1]
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                result = pool.submit(source.read_rows, "table", rows)
                self.assertTrue(entered.wait(timeout=2))
                rows[:] = [0, 0]
                release.set()
                self.assertEqual(result.result(timeout=2), b"ijcd")

    def test_reads_row_runs_across_internal_chunk_boundaries(self):
        root = self.root / "wide"
        root.mkdir()
        row_size = 600_000
        rows = [bytes([value]) * row_size for value in (17, 31, 47)]
        inventory = checkpoint(
            root,
            {"wide.safetensors": {"wide": ("U8", [len(rows), row_size], b"".join(rows))}},
        )
        with SafeTensorCheckpoint(
            root,
            expected_revision="abc123",
            expected_file_inventory=inventory,
            expected_chunk_digest_bytes=chunk_digest_bytes(root, inventory),
        ) as source:
            self.assertEqual(source.read_rows("wide", [2, 1, 0]), rows[2] + rows[1] + rows[0])

    def test_reads_each_touched_chunk_once_for_disjoint_rows(self):
        root = self.root / "sparse"
        root.mkdir()
        rows = [bytes([value % 251]) * 4096 for value in range(256)]
        inventory = checkpoint(
            root,
            {"sparse.safetensors": {"sparse": ("U8", [len(rows), len(rows[0])], b"".join(rows))}},
        )
        with SafeTensorCheckpoint(
            root,
            expected_revision="abc123",
            expected_file_inventory=inventory,
            expected_chunk_digest_bytes=chunk_digest_bytes(root, inventory),
        ) as source:
            source.tensors["sparse"]
            with patch.object(source, "_read_verified_chunk", wraps=source._read_verified_chunk) as read_chunk:
                self.assertEqual(source.read_rows("sparse", [0, 100]), rows[0] + rows[100])
            self.assertEqual(read_chunk.call_count, 1)

    def test_reads_concurrently_from_one_checkpoint_reader(self):
        with self.reader() as source:
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(lambda _: source.read_rows("table", [3, 0]), range(32)))
        self.assertEqual(results, [b"ghab"] * 32)

    def test_default_shard_access_does_not_scan_the_full_file(self):
        with patch.object(SafeTensorCheckpoint, "_verify_digest_fd", side_effect=AssertionError):
            source = SafeTensorCheckpoint(
                self.root,
                expected_revision="abc123",
                expected_file_inventory=self.inventory,
                expected_chunk_digest_bytes=chunk_digest_bytes(self.root, self.inventory),
            )
            with source:
                self.assertEqual(source.read_tensor_bytes("scalar"), b"xyz")
                self.assertEqual(source.read_tensor_bytes("other"), b"\x00\x00")

    def test_chunk_hash_rejects_same_size_payload_tampering(self):
        expected_chunks = chunk_digest_bytes(self.root, self.inventory)
        path = self.root / "part-1.safetensors"
        raw = bytearray(path.read_bytes())
        raw[-1] ^= 1
        path.write_bytes(raw)
        source = SafeTensorCheckpoint(
            self.root,
            expected_revision="abc123",
            expected_file_inventory=self.inventory,
            expected_chunk_digest_bytes=expected_chunks,
        )
        with source:
            with self.assertRaisesRegex(CheckpointError, "chunk 0.*pinned SHA-256"):
                source.read_tensor_bytes("scalar")

    def test_exposes_immutable_tensor_metadata(self):
        with self.reader() as source:
            with self.assertRaises(TypeError):
                source.tensors["table"] = source.tensors["scalar"]

    def test_rejects_revision_mismatch(self):
        with self.assertRaisesRegex(CheckpointError, "revision"):
            SafeTensorCheckpoint(
                self.root, expected_revision="wrong", expected_file_inventory=self.inventory
            )

    def test_rejects_tampered_index_against_pinned_digest(self):
        index_path = self.root / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        index["metadata"]["total_size"] += 1
        index_path.write_text(json.dumps(index))
        with self.assertRaisesRegex(CheckpointError, "pinned SHA-256"):
            self.reader()

    def test_hashes_the_exact_index_bytes_that_it_parses(self):
        read_limited = checkpoint_reader._read_limited_fd

        def mutate_index_bytes(fd, *, limit, description, name):
            raw = read_limited(fd, limit=limit, description=description, name=name)
            if name == "model.safetensors.index.json":
                document = json.loads(raw)
                document["metadata"]["total_size"] += 1
                return json.dumps(document).encode()
            return raw

        with patch.object(checkpoint_reader, "_read_limited_fd", side_effect=mutate_index_bytes):
            with self.assertRaisesRegex(CheckpointError, "pinned SHA-256"):
                self.reader()

    def test_rejects_index_to_header_mismatch(self):
        index_path = self.root / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        index["weight_map"]["missing"] = "part-1.safetensors"
        index_path.write_text(json.dumps(index))
        inventory = dict(self.inventory)
        index_stat = index_path.stat()
        inventory[index_path.name] = (index_stat.st_size, hashlib.sha256(index_path.read_bytes()).hexdigest())
        with SafeTensorCheckpoint(
            self.root,
            expected_revision="abc123",
            expected_file_inventory=inventory,
            verify_shard_digests=True,
        ) as source:
            with self.assertRaisesRegex(CheckpointError, "index.*header|header.*index"):
                source.tensors["missing"]

    def test_rejects_out_of_bounds_tensor_offsets(self):
        path = self.root / "part-1.safetensors"
        raw = path.read_bytes()
        header_size = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8 : 8 + header_size])
        header["table"]["data_offsets"] = [0, 11]
        encoded = json.dumps(header, separators=(",", ":")).encode()
        path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + raw[8 + header_size :])
        inventory = dict(self.inventory)
        shard_stat = path.stat()
        inventory[path.name] = (shard_stat.st_size, hashlib.sha256(path.read_bytes()).hexdigest())
        with SafeTensorCheckpoint(
            self.root,
            expected_revision="abc123",
            expected_file_inventory=inventory,
            verify_shard_digests=True,
        ) as source:
            with self.assertRaisesRegex(CheckpointError, "offset|payload|size"):
                source.tensors["table"]

    def test_rejects_reads_over_the_explicit_memory_limit(self):
        with self.reader() as source:
            with self.assertRaisesRegex(ValueError, "limit"):
                source.read_tensor_bytes("table", max_bytes=9)
            with self.assertRaisesRegex(ValueError, "limit"):
                source.read_rows("table", [0, 1, 2, 3, 4], max_bytes=9)
            with self.assertRaisesRegex(ValueError, "hard limit"):
                source.read_tensor_bytes("scalar", max_bytes=256 * 1024 * 1024 + 1)

    def test_rejects_shard_replacement_after_metadata_validation(self):
        with self.reader() as source:
            source.tensors["scalar"]
            path = self.root / "part-1.safetensors"
            path.write_bytes(path.read_bytes())
            with self.assertRaisesRegex(CheckpointError, "changed since metadata validation"):
                source.read_tensor_bytes("scalar")

    def test_rejects_row_batches_over_the_hard_count_limit(self):
        with self.reader() as source:
            with self.assertRaisesRegex(ValueError, "hard limit.*rows"):
                source.read_rows("table", range(65_537))

    def test_rejects_use_after_close(self):
        source = self.reader()
        source.close()
        source.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            source.read_tensor_bytes("scalar")

    def test_rejects_symlinked_shard_inside_checkpoint(self):
        path = self.root / "part-1.safetensors"
        with tempfile.TemporaryDirectory() as outside_dir:
            outside = Path(outside_dir) / "outside.safetensors"
            outside.write_bytes(path.read_bytes())
            path.unlink()
            path.symlink_to(outside)
            with self.reader() as source:
                with self.assertRaisesRegex(CheckpointError, "safely open"):
                    source.read_tensor_bytes("scalar")

    def test_rejects_shard_symlink_replacement_before_read(self):
        with tempfile.TemporaryDirectory() as outside_dir:
            outside = Path(outside_dir) / "outside.safetensors"
            outside.write_bytes((self.root / "part-1.safetensors").read_bytes())
            with self.reader() as source:
                path = self.root / "part-1.safetensors"
                path.unlink()
                path.symlink_to(outside)
                with self.assertRaisesRegex(CheckpointError, "safely open"):
                    source.read_tensor_bytes("scalar")

    def test_rejects_payload_tampering_against_pinned_digest(self):
        path = self.root / "part-1.safetensors"
        raw = bytearray(path.read_bytes())
        raw[-1] ^= 1
        path.write_bytes(raw)
        with self.reader(verify_shard_digests=True) as source:
            with self.assertRaisesRegex(CheckpointError, "pinned SHA-256"):
                source.read_tensor_bytes("scalar")

    def test_read_memory_budget_bounds_aggregate_reservations(self):
        budget = ReadMemoryBudget(10)
        entered = threading.Event()
        release = threading.Event()
        acquired = threading.Event()

        def hold_eight_bytes():
            with budget.reserve(8):
                entered.set()
                release.wait(timeout=2)

        def request_five_bytes():
            with budget.reserve(5):
                acquired.set()

        first = threading.Thread(target=hold_eight_bytes)
        second = threading.Thread(target=request_five_bytes)
        first.start()
        self.assertTrue(entered.wait(timeout=2))
        second.start()
        try:
            self.assertFalse(acquired.wait(timeout=0.05))
        finally:
            release.set()
        first.join(timeout=2)
        second.join(timeout=2)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertTrue(acquired.is_set())

    def test_read_memory_budget_rejects_nested_overcommit_without_deadlocking(self):
        budget = ReadMemoryBudget(10)
        with budget.reserve(8):
            with self.assertRaisesRegex(ValueError, "cannot be nested"):
                with budget.reserve(5):
                    pass

    def test_loads_all_shard_hashes_from_versioned_inventory(self):
        inventory = SafeTensorCheckpoint._load_default_file_inventory()
        self.assertEqual(len([name for name in inventory if name.endswith(".safetensors")]), 48)
        self.assertIn("model-00001-of-00048.safetensors", inventory)
        self.assertIn("model.safetensors.index.json", inventory)

    def test_loads_pinned_chunk_digest_manifest(self):
        loader = object.__new__(SafeTensorCheckpoint)
        loader._read_memory_budget = ReadMemoryBudget(64 * 1024 * 1024)
        inventory = SafeTensorCheckpoint._load_default_file_inventory()
        digests = loader._load_default_chunk_digests(inventory)
        expected_chunks = sum(
            (size + CHUNK_SIZE - 1) // CHUNK_SIZE
            for name, (size, _) in inventory.items()
            if name.endswith(".safetensors")
        )
        self.assertEqual(len(digests), expected_chunks * 32)
        self.assertEqual(
            hashlib.sha256(digests).hexdigest(),
            checkpoint_reader._DEFAULT_CHUNK_DIGESTS_SHA256,
        )

    def test_support_file_reader_rejects_symlinks_and_fifos(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            regular = root / "regular"
            regular.write_bytes(b"ok")
            symlink = root / "symlink"
            symlink.symlink_to(regular)
            with self.assertRaisesRegex(CheckpointError, "safely open"):
                checkpoint_reader._read_regular_path(
                    symlink, limit=10, description="test support file"
                )

            fifo = root / "fifo"
            fifo_path = getattr(os, "mkfifo", None)
            if fifo_path is not None:
                fifo_path(fifo)
                with self.assertRaisesRegex(CheckpointError, "regular file"):
                    checkpoint_reader._read_regular_path(
                        fifo, limit=10, description="test support file"
                    )


if __name__ == "__main__":
    unittest.main()
