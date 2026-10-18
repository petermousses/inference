"""Exact, bounded Engram table row reads from the pinned checkpoint."""

from __future__ import annotations

import concurrent.futures
import importlib.util
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import engram_rows
from checkpoint_reader import CheckpointError, SafeTensorCheckpoint
from engram_rows import EngramRowReader, install_engram_row_reads
from test_checkpoint_reader import chunk_digest_bytes, checkpoint, header_inventory

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
try:
    import torch
except ImportError:
    torch = None

HAS_ENGRAM_DTYPES = TORCH_AVAILABLE and all(
    hasattr(torch, name) for name in ("float8_e4m3fn", "float8_e8m0fnu", "bfloat16")
)


@unittest.skipUnless(HAS_ENGRAM_DTYPES, "requires PyTorch FP8 E4M3, E8M0, and BF16 dtypes")
class EngramRowReaderTest(unittest.TestCase):
    rows = 19
    dim = 64
    block_size = 32

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.table_bytes: dict[int, tuple[bytes, bytes]] = {}
        tensors = {}
        for layer_id, offset in ((1, 0), (14, 5)):
            values = (
                torch.arange(self.rows * self.dim, dtype=torch.float32)
                .reshape(self.rows, self.dim)
                .add(offset)
                .remainder(15)
                .sub(7)
                .div(4)
                .to(torch.float8_e4m3fn)
            )
            scales = torch.ones(
                (self.rows, self.dim // self.block_size), dtype=torch.float8_e8m0fnu
            )
            weight_bytes = values.view(torch.uint8).numpy().tobytes()
            scale_bytes = scales.view(torch.uint8).numpy().tobytes()
            self.table_bytes[layer_id] = (weight_bytes, scale_bytes)
            prefix = f"layers.{layer_id}.engram.embed"
            tensors[f"{prefix}.weight"] = ("F8_E4M3", [self.rows, self.dim], weight_bytes)
            tensors[f"{prefix}.scale"] = (
                "F8_E8M0",
                [self.rows, self.dim // self.block_size],
                scale_bytes,
            )
        self.inventory = checkpoint(self.root, {"weights.safetensors": tensors})
        self.reader = SafeTensorCheckpoint(
            self.root,
            expected_revision="abc123",
            expected_file_inventory=self.inventory,
            expected_chunk_digest_bytes=chunk_digest_bytes(self.root, self.inventory),
        )

    def tearDown(self):
        self.reader.close()
        self.temp.cleanup()

    def _reference(self, layer_id: int, indices: torch.Tensor) -> torch.Tensor:
        weight_bytes, scale_bytes = self.table_bytes[layer_id]
        weight = torch.frombuffer(weight_bytes, dtype=torch.float8_e4m3fn).reshape(
            self.rows, self.dim
        )
        scales = torch.frombuffer(scale_bytes, dtype=torch.float8_e8m0fnu).reshape(
            self.rows, self.dim // self.block_size
        )
        mask = (indices < 0) | (indices >= self.rows)
        local = indices.masked_fill(mask, 0)
        values = torch.nn.functional.embedding(local, weight)
        row_scales = torch.nn.functional.embedding(local, scales)
        result = (
            values.float().unflatten(-1, (-1, self.block_size))
            * row_scales.float().unsqueeze(-1)
        ).flatten(-2).to(torch.bfloat16)
        return result.masked_fill(mask.unsqueeze(-1), 0)

    def _meta_transformer(self, layer_ids):
        fixture = self

        class MetaEmbedding:
            def __init__(self):
                self.num_embeddings = fixture.rows
                self.dim = fixture.dim
                self.vocab_start_idx = 0
                self.vocab_end_idx = fixture.rows
                self.weight = torch.empty(
                    (fixture.rows, fixture.dim), dtype=torch.float8_e4m3fn, device="meta"
                )
                self.scale = torch.empty(
                    (fixture.rows, fixture.dim // fixture.block_size),
                    dtype=torch.float8_e8m0fnu,
                    device="meta",
                )

            def forward(self, indices):
                raise AssertionError("full Engram table path was used")

        class Block:
            def __init__(self, layer_id):
                self.layer_id = layer_id
                self.engram = SimpleNamespace(embed=MetaEmbedding())

        return SimpleNamespace(layers=[Block(layer_id) for layer_id in layer_ids]), MetaEmbedding

    def test_lookup_matches_official_embedding_with_duplicates_and_invalid_ids(self):
        indices = torch.tensor(
            [[[0, 3, 3, 18], [7, -1, 19, 2]], [[8, 8, 1, 0], [18, 4, 4, 4]]],
            dtype=torch.int64,
        )
        reader = EngramRowReader(
            self.reader,
            1,
            num_embeddings=self.rows,
            embedding_dim=self.dim,
            max_indices_per_batch=5,
            max_rows_per_batch=3,
        )
        actual = reader.lookup(indices)
        torch.testing.assert_close(actual, self._reference(1, indices), rtol=0, atol=0)

    def test_lookup_reads_only_bounded_unique_row_batches(self):
        indices = torch.tensor([[[1, 1, 2, 2, 3, 4, 4, 4]]], dtype=torch.int64)
        requested = []
        original = engram_rows.read_rows_to_device

        def record_rows(checkpoint_reader, name, rows, **kwargs):
            requested.append((name, tuple(rows)))
            return original(checkpoint_reader, name, rows, **kwargs)

        reader = EngramRowReader(
            self.reader,
            1,
            num_embeddings=self.rows,
            embedding_dim=self.dim,
            max_indices_per_batch=4,
            max_rows_per_batch=2,
        )
        with patch("engram_rows.read_rows_to_device", side_effect=record_rows):
            reader.lookup(indices)

        self.assertTrue(requested)
        self.assertTrue(all(len(rows) <= 2 for _, rows in requested))
        self.assertEqual(
            [name for name, _ in requested],
            [
                "layers.1.engram.embed.weight",
                "layers.1.engram.embed.scale",
                "layers.1.engram.embed.weight",
                "layers.1.engram.embed.scale",
            ],
        )
        self.assertEqual(
            [row for name, rows in requested if name.endswith(".weight") for row in rows],
            [1, 2, 3, 4],
        )

    def test_out_of_range_only_lookup_returns_zeros_without_checkpoint_payload_reads(self):
        indices = torch.tensor([[[-1, self.rows, self.rows + 1]]], dtype=torch.int64)
        reader = EngramRowReader(
            self.reader,
            1,
            num_embeddings=self.rows,
            embedding_dim=self.dim,
        )
        with patch("engram_rows.read_rows_to_device", side_effect=AssertionError("unexpected read")):
            actual = reader.lookup(indices)
        self.assertTrue(torch.equal(actual, torch.zeros_like(actual)))

    def test_estimate_accounts_for_full_output_and_bounded_transfer_scratch(self):
        reader = EngramRowReader(
            self.reader,
            1,
            num_embeddings=self.rows,
            embedding_dim=self.dim,
            max_indices_per_batch=5,
            max_rows_per_batch=3,
        )
        estimate = reader.estimate_memory((2, 3, 4))
        self.assertEqual(estimate.output_bytes, 2 * 3 * 4 * self.dim * 2)
        self.assertGreater(estimate.device_peak_bytes, estimate.output_bytes)
        self.assertGreater(estimate.host_peak_bytes, 0)

    def test_rejects_invalid_table_metadata_before_reading(self):
        with self.assertRaises(CheckpointError):
            EngramRowReader(
                self.reader,
                1,
                num_embeddings=self.rows,
                embedding_dim=self.dim + 32,
            )

    def test_rejects_metadata_only_checkpoint_reader(self):
        metadata_reader = SafeTensorCheckpoint(
            self.root,
            expected_revision="abc123",
            expected_file_inventory=self.inventory,
            expected_header_inventory=header_inventory(self.root, self.inventory),
            metadata_only=True,
        )
        try:
            with self.assertRaisesRegex(ValueError, "payload-enabled"):
                EngramRowReader(
                    metadata_reader,
                    1,
                    num_embeddings=self.rows,
                    embedding_dim=self.dim,
                )
        finally:
            metadata_reader.close()

    def test_installs_and_restores_per_layer_row_readers(self):
        transformer, MetaEmbedding = self._meta_transformer([1, 14])
        indices = torch.tensor([[[2, 5, 7]]], dtype=torch.int64)
        with install_engram_row_reads(
            transformer, self.reader, max_indices_per_batch=2, max_rows_per_batch=2
        ) as installation:
            self.assertEqual(len(installation.readers), 2)
            actual = transformer.layers[1].engram.embed.forward(indices)
            torch.testing.assert_close(actual, self._reference(14, indices), rtol=0, atol=0)
            with self.assertRaisesRegex(RuntimeError, "already installed"):
                install_engram_row_reads(transformer, self.reader)
        self.assertIs(
            transformer.layers[1].engram.embed.forward.__func__, MetaEmbedding.forward
        )
        with self.assertRaisesRegex(AssertionError, "full Engram table"):
            transformer.layers[1].engram.embed.forward(indices)

    def test_close_waits_for_active_row_reads_before_restoring_forward(self):
        transformer, MetaEmbedding = self._meta_transformer([1])
        embed = transformer.layers[0].engram.embed
        entered_read = threading.Event()
        release_read = threading.Event()
        original_read = engram_rows.read_rows_to_device

        def blocking_read(checkpoint_reader, name, rows, **kwargs):
            entered_read.set()
            if not release_read.wait(timeout=5):
                raise TimeoutError("test did not release Engram row read")
            return original_read(checkpoint_reader, name, rows, **kwargs)

        with install_engram_row_reads(transformer, self.reader, max_rows_per_batch=2) as installation:
            with patch("engram_rows.read_rows_to_device", side_effect=blocking_read):
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    lookup = pool.submit(
                        embed.forward, torch.tensor([[[1, 2, 3]]], dtype=torch.int64)
                    )
                    self.assertTrue(entered_read.wait(timeout=2))
                    waiting_to_close = threading.Event()
                    original_wait = installation._condition.wait

                    def tracked_wait(timeout=None):
                        waiting_to_close.set()
                        return original_wait(timeout)

                    with patch.object(
                        installation._condition, "wait", side_effect=tracked_wait
                    ):
                        closing = pool.submit(installation.close)
                        self.assertTrue(waiting_to_close.wait(timeout=2))
                        self.assertFalse(closing.done())
                        release_read.set()
                        lookup.result(timeout=5)
                        closing.result(timeout=5)

        self.assertIs(embed.forward.__func__, MetaEmbedding.forward)

    def test_duplicate_engram_layers_fail_before_any_patch(self):
        transformer, MetaEmbedding = self._meta_transformer([1, 1])
        with self.assertRaisesRegex(ValueError, "duplicate Engram"):
            install_engram_row_reads(transformer, self.reader)
        for block in transformer.layers:
            embed = block.engram.embed
            self.assertIs(embed.forward.__func__, MetaEmbedding.forward)
            self.assertFalse(hasattr(embed, "_redqueen_engram_row_reader"))


if __name__ == "__main__":
    unittest.main()
