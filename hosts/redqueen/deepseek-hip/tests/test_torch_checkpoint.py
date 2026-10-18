"""Bounded PyTorch tensor materialization from the pinned checkpoint reader."""

import hashlib
import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path

import torch_checkpoint
from checkpoint_reader import SafeTensorCheckpoint
from test_checkpoint_reader import chunk_digest_bytes, checkpoint

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
try:
    import torch
except ImportError:
    torch = None


class CheckpointFixture:
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.values = struct.pack("<6h", 11, 12, 21, 22, 31, 32)
        self.inventory = checkpoint(
            self.root,
            {
                "weights.safetensors": {
                    "table": ("I16", [3, 2], self.values),
                    "empty_rows": ("U8", [4, 0], b""),
                    "scalar": ("I16", [], struct.pack("<h", 42)),
                }
            },
        )
        self.reader = SafeTensorCheckpoint(
            self.root,
            expected_revision="abc123",
            expected_file_inventory=self.inventory,
            expected_chunk_digest_bytes=chunk_digest_bytes(self.root, self.inventory),
        )

    def tearDown(self):
        self.reader.close()
        self.temp.cleanup()


@unittest.skipUnless(TORCH_AVAILABLE, "requires PyTorch")
class TorchCheckpointTest(CheckpointFixture, unittest.TestCase):
    def test_reads_full_tensor_as_independent_typed_device_tensor(self):
        result = torch_checkpoint.read_tensor_to_device(self.reader, "table", device="cpu")
        self.assertEqual(result.dtype, torch.int16)
        self.assertEqual(tuple(result.shape), (3, 2))
        self.assertEqual(result.tolist(), [[11, 12], [21, 22], [31, 32]])

        result[0, 0] = -1
        again = torch_checkpoint.read_tensor_to_device(self.reader, "table", device="cpu")
        self.assertEqual(again.tolist(), [[11, 12], [21, 22], [31, 32]])

    def test_reads_duplicate_rows_in_requested_order(self):
        result = torch_checkpoint.read_rows_to_device(
            self.reader, "table", [2, 0, 2], device="cpu"
        )
        self.assertEqual(result.dtype, torch.int16)
        self.assertEqual(tuple(result.shape), (3, 2))
        self.assertEqual(result.tolist(), [[31, 32], [11, 12], [31, 32]])

    def test_materializes_empty_rows_with_the_requested_shape(self):
        result = torch_checkpoint.read_rows_to_device(
            self.reader, "empty_rows", [3, 1], device="cpu"
        )
        self.assertEqual(result.dtype, torch.uint8)
        self.assertEqual(tuple(result.shape), (2, 0))
        self.assertEqual(result.numel(), 0)

    def test_does_not_mutate_the_source_checkpoint(self):
        before = hashlib.sha256((self.root / "weights.safetensors").read_bytes()).digest()
        result = torch_checkpoint.read_rows_to_device(
            self.reader, "table", [1], device="cpu"
        )
        result.fill_(0)
        after = hashlib.sha256((self.root / "weights.safetensors").read_bytes()).digest()
        self.assertEqual(after, before)

    def test_copies_large_tensor_in_bounded_row_chunks_into_existing_storage(self):
        destination = torch.full((3, 2), -1, dtype=torch.int16)
        pointer = destination.data_ptr()
        original = torch_checkpoint.read_rows_to_device
        reads = []

        def record_read(checkpoint, name, rows, **kwargs):
            reads.append((rows.start, rows.stop, kwargs["max_bytes"]))
            return original(checkpoint, name, rows, **kwargs)

        with unittest.mock.patch.object(
            torch_checkpoint, "read_rows_to_device", side_effect=record_read
        ):
            result = torch_checkpoint.copy_tensor_to_device(
                self.reader, "table", destination, max_chunk_bytes=8
            )

        self.assertIs(result, destination)
        self.assertEqual(destination.data_ptr(), pointer)
        self.assertEqual(destination.tolist(), [[11, 12], [21, 22], [31, 32]])
        self.assertEqual(reads, [(0, 2, 8), (2, 3, 8)])

    def test_copies_scalar_and_empty_tensor(self):
        scalar = torch.empty((), dtype=torch.int16)
        empty = torch.empty((4, 0), dtype=torch.uint8)
        self.assertIs(
            torch_checkpoint.copy_tensor_to_device(self.reader, "scalar", scalar), scalar
        )
        self.assertEqual(scalar.item(), 42)
        self.assertIs(
            torch_checkpoint.copy_tensor_to_device(self.reader, "empty_rows", empty), empty
        )

    def test_rejects_destination_shape_and_row_chunks_that_cannot_fit(self):
        wrong_shape = torch.empty((2, 2), dtype=torch.int16)
        with self.assertRaises(ValueError):
            torch_checkpoint.copy_tensor_to_device(self.reader, "table", wrong_shape)

        destination = torch.full((3, 2), -1, dtype=torch.int16)
        with unittest.mock.patch.object(torch_checkpoint, "read_rows_to_device") as read_rows:
            with self.assertRaises(ValueError):
                torch_checkpoint.copy_tensor_to_device(
                    self.reader, "table", destination, max_chunk_bytes=3
                )
        read_rows.assert_not_called()


class TorchCheckpointValidationTest(CheckpointFixture, unittest.TestCase):
    def test_invalid_tensor_limit_fails_before_pytorch_or_device_resolution(self):
        with unittest.mock.patch.object(
            torch_checkpoint,
            "_destination",
            side_effect=AssertionError("destination resolved before size validation"),
        ):
            with self.assertRaises(ValueError):
                torch_checkpoint.read_tensor_to_device(
                    self.reader, "table", device="invalid-device", max_bytes=0
                )

    def test_invalid_rows_fail_before_pytorch_or_device_resolution(self):
        cases = (
            (iter([0]), TypeError),
            (range(0, 65_537), ValueError),
            ([3], IndexError),
        )
        for rows, error in cases:
            with self.subTest(rows=type(rows).__name__, error=error.__name__):
                with unittest.mock.patch.object(
                    torch_checkpoint,
                    "_destination",
                    side_effect=AssertionError("destination resolved before row validation"),
                ):
                    with self.assertRaises(error):
                        torch_checkpoint.read_rows_to_device(
                            self.reader, "table", rows, device="invalid-device"
                        )


if __name__ == "__main__":
    unittest.main()
