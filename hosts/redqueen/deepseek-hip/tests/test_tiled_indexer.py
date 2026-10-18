"""Tiled exact-index selection parity against the pinned eager tensor expression."""

import unittest

import torch

from tiled_indexer import tiled_indexer_selection


def reference_candidate_blocks(logits, compress_lens, topk_blocks, block_size):
    width = logits.size(-1)
    scores = torch.nn.functional.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device) == last, torch.inf)
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def expand_candidate_ids(candidate_ids, key_count, block_size):
    num_blocks = (key_count + block_size - 1) // block_size
    key_positions = torch.arange(key_count, device=candidate_ids.device).view(
        1, 1, 1, key_count
    )
    block_ids = candidate_ids.to(torch.int32).unsqueeze(-1)
    valid_blocks = (block_ids >= 0) & (block_ids < num_blocks)
    block_starts = block_ids * block_size
    return (
        valid_blocks
        & (key_positions >= block_starts)
        & (key_positions < block_starts + block_size)
    ).any(dim=2)


def eager_reference(
    query,
    keys,
    head_weights,
    *,
    valid_positions,
    topk,
    offset,
    candidate_blocks=None,
    candidate_block_size=None,
    candidate_topk_blocks=None,
):
    batch, query_count, _heads, _dim = query.shape
    score_heads = torch.einsum("bshd,btd->bsht", query, keys)
    score_heads.relu_()
    scores = (score_heads * head_weights.unsqueeze(-1)).sum(dim=2)
    key_positions = torch.arange(keys.size(1), device=query.device)
    if isinstance(valid_positions, int):
        lens = valid_positions
        scores.masked_fill_(key_positions >= lens, -torch.inf)
        lens_for_mask = lens
    else:
        lens = valid_positions.reshape(query_count, 1)
        scores.masked_fill_(key_positions.view(1, 1, -1) >= lens.view(1, query_count, 1), -torch.inf)
        lens_for_mask = lens

    candidates = None
    if candidate_topk_blocks is not None:
        candidates = reference_candidate_blocks(
            scores,
            lens_for_mask,
            candidate_topk_blocks,
            candidate_block_size,
        )
    elif candidate_blocks is not None:
        scores.masked_fill_(~candidate_blocks, -torch.inf)

    selected = scores.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
    if isinstance(valid_positions, int):
        valid = selected < lens_for_mask
    else:
        valid = selected < lens_for_mask.view(1, query_count, 1)
    selected = torch.where(valid, selected + offset, -1).to(torch.int32)
    return selected, candidates


class TiledIndexerSelectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.version.hip is None:
            raise RuntimeError("requires a ROCm GPU; refusing to report a skipped self-test as success")
        cls.device = torch.device("cuda")

    def inputs(self, query_count=197, key_count=101):
        torch.manual_seed(271828)
        query = torch.randn(
            (1, query_count, 4, 16), dtype=torch.bfloat16, device=self.device
        )
        keys = torch.randn((1, key_count, 16), dtype=torch.bfloat16, device=self.device)
        weights = torch.rand((1, query_count, 4), dtype=torch.bfloat16, device=self.device)
        return query, keys, weights

    def test_tiled_candidate_source_matches_eager_scores_and_indices(self):
        query, keys, weights = self.inputs()
        valid = torch.arange(1, query.size(1) + 1, device=self.device) // 2
        # Exercise a partial first block and a partial final candidate block.
        valid.clamp_(min=1, max=keys.size(1))
        expected_indices, expected_candidates = eager_reference(
            query,
            keys,
            weights,
            valid_positions=valid,
            topk=17,
            offset=128,
            candidate_block_size=8,
            candidate_topk_blocks=5,
        )
        actual_indices, actual_candidates = tiled_indexer_selection(
            query,
            keys,
            weights,
            valid_positions=valid,
            topk=17,
            offset=128,
            query_tile_rows=32,
            candidate_block_size=8,
            candidate_topk_blocks=5,
            candidate_selector=reference_candidate_blocks,
        )
        self.assertTrue(torch.equal(actual_indices, expected_indices))
        decoded_candidates = expand_candidate_ids(actual_candidates, keys.size(1), 8)
        self.assertTrue(torch.equal(decoded_candidates, expected_candidates))

    def test_tiled_consumer_matches_eager_with_compact_candidate_blocks(self):
        query, keys, weights = self.inputs(query_count=129, key_count=113)
        valid = torch.arange(1, query.size(1) + 1, device=self.device).clamp(max=keys.size(1))
        source_indices, source_ids = tiled_indexer_selection(
            query,
            keys,
            weights,
            valid_positions=valid,
            topk=23,
            offset=0,
            query_tile_rows=32,
            candidate_block_size=8,
            candidate_topk_blocks=6,
            candidate_selector=reference_candidate_blocks,
        )
        self.assertEqual(source_indices.shape, (1, query.size(1), 23))
        source_mask = expand_candidate_ids(source_ids, keys.size(1), 8)

        consumer_query, consumer_keys, consumer_weights = self.inputs(query_count=129, key_count=113)
        expected_indices, _ = eager_reference(
            consumer_query,
            consumer_keys,
            consumer_weights,
            valid_positions=valid,
            topk=19,
            offset=256,
            candidate_blocks=source_mask,
        )
        actual_indices, unused = tiled_indexer_selection(
            consumer_query,
            consumer_keys,
            consumer_weights,
            valid_positions=valid,
            topk=19,
            offset=256,
            query_tile_rows=32,
            candidate_blocks=source_ids,
            candidate_block_size=8,
            candidate_topk_blocks=6,
        )
        self.assertIsNone(unused)
        self.assertTrue(torch.equal(actual_indices, expected_indices))

    def test_candidate_consumer_preserves_full_axis_tie_order(self):
        query_count, key_count, block_size = 31, 101, 8
        query = torch.zeros(
            (1, query_count, 4, 16), dtype=torch.bfloat16, device=self.device
        )
        keys = torch.randn((1, key_count, 16), dtype=torch.bfloat16, device=self.device)
        weights = torch.ones((1, query_count, 4), dtype=torch.bfloat16, device=self.device)
        chosen_blocks = torch.tensor(
            [0, 2, 4, 6, 8, 10], dtype=torch.int16, device=self.device
        ).view(1, 1, -1).expand(1, query_count, -1).contiguous()
        candidate_mask = expand_candidate_ids(chosen_blocks, key_count, block_size)

        expected, _ = eager_reference(
            query,
            keys,
            weights,
            valid_positions=key_count,
            topk=31,
            offset=0,
            candidate_blocks=candidate_mask,
        )
        actual, _ = tiled_indexer_selection(
            query,
            keys,
            weights,
            valid_positions=key_count,
            topk=31,
            offset=0,
            query_tile_rows=32,
            candidate_blocks=chosen_blocks,
            candidate_block_size=block_size,
            candidate_topk_blocks=6,
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_query_and_weight_projections_run_per_tile(self):
        torch.manual_seed(314159)
        query_source = torch.randn((1, 65, 24), dtype=torch.bfloat16, device=self.device)
        weight_source = torch.randn((1, 65, 20), dtype=torch.bfloat16, device=self.device)
        query_weight = torch.randn((64, 24), dtype=torch.bfloat16, device=self.device)
        weight_weight = torch.randn((4, 20), dtype=torch.bfloat16, device=self.device)
        keys = torch.randn((1, 137, 16), dtype=torch.bfloat16, device=self.device)
        projected_query = torch.nn.functional.linear(query_source, query_weight).unflatten(
            -1, (4, 16)
        )
        projected_weights = torch.nn.functional.linear(weight_source, weight_weight)
        expected, _ = eager_reference(
            projected_query,
            keys,
            projected_weights,
            valid_positions=keys.size(1),
            topk=23,
            offset=0,
        )

        query_tile_sizes = []
        weight_tile_sizes = []

        def project_query(values, first, last):
            query_tile_sizes.append(last - first)
            return torch.nn.functional.linear(values, query_weight).unflatten(-1, (4, 16))

        def project_weights(values, first, last):
            weight_tile_sizes.append(last - first)
            return torch.nn.functional.linear(values, weight_weight)

        actual, _ = tiled_indexer_selection(
            query_source,
            keys,
            weight_source,
            query_heads=4,
            query_projection=project_query,
            head_weights_projection=project_weights,
            valid_positions=keys.size(1),
            topk=23,
            offset=0,
            query_tile_rows=32,
        )
        self.assertEqual(query_tile_sizes, [32, 32, 1])
        self.assertEqual(weight_tile_sizes, [32, 32, 1])
        self.assertTrue(torch.equal(actual, expected))

    def test_decode_scalar_valid_length_matches_eager(self):
        query, keys, weights = self.inputs(query_count=7, key_count=101)
        expected, _ = eager_reference(
            query,
            keys,
            weights,
            valid_positions=83,
            topk=31,
            offset=128,
        )
        actual, candidates = tiled_indexer_selection(
            query,
            keys,
            weights,
            valid_positions=83,
            topk=31,
            offset=128,
            query_tile_rows=3,
        )
        self.assertIsNone(candidates)
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
