"""Memory-bounded exact index selection for the pinned 1017 graph."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable


def tiled_indexer_selection(
    query,
    keys,
    head_weights,
    *,
    query_heads: int | None = None,
    query_projection: Callable | None = None,
    head_weights_projection: Callable | None = None,
    valid_positions,
    topk: int,
    offset: int,
    query_tile_rows: int,
    candidate_blocks=None,
    candidate_block_size: int | None = None,
    candidate_selector: Callable | None = None,
    candidate_topk_blocks: int | None = None,
):
    """Apply the pinned score, mask, candidate, and top-k order in bounded query tiles.

    The pinned eager implementation materializes ``[batch, queries, heads, keys]``
    float32 scores. This keeps that same tensor's lifetime to ``query_tile_rows``
    queries, then emits only the selected indices. Candidate sources store compact
    block IDs; consumers expand only the current block mask while retaining the
    pinned full-key score axis and top-k tie behavior.
    """
    torch = _torch_for(query)
    if keys.ndim != 3:
        raise ValueError("index keys must have shape [B,T,D]")
    if query_projection is None:
        if query.ndim != 4:
            raise ValueError("index query must have shape [B,S,H,D]")
        batch, query_count, heads, dim = query.shape
    else:
        if query.ndim != 3 or type(query_heads) is not int or query_heads <= 0:
            raise ValueError("projected queries require source [B,S,D] and a positive head count")
        batch, query_count, _ = query.shape
        heads, dim = query_heads, keys.shape[2]
    if keys.shape[0] != batch or keys.shape[2] != dim:
        raise ValueError("index selection query and key dimensions do not match")
    if head_weights.ndim != 3 or head_weights.shape[:2] != (batch, query_count):
        raise ValueError("index weight source must have shape [B,S,D]")
    if head_weights_projection is None:
        if head_weights.shape != (batch, query_count, heads):
            raise ValueError("index selection head weights do not match query dimensions")
    if type(query_tile_rows) is not int or query_tile_rows <= 0:
        raise ValueError("query_tile_rows must be a positive integer")
    if type(topk) is not int or not 0 < topk <= keys.shape[1]:
        raise ValueError("topk must be in [1, number of compressed keys]")
    if type(offset) is not int or offset < 0:
        raise ValueError("offset must be a nonnegative integer")

    if isinstance(valid_positions, int):
        if not 0 <= valid_positions <= keys.shape[1]:
            raise ValueError("valid_positions is outside the compressed-key range")
        valid_vector = None
    else:
        if tuple(valid_positions.shape) not in {(query_count,), (query_count, 1)}:
            raise ValueError("valid_positions must have one entry per query")
        valid_vector = valid_positions.reshape(query_count, 1)

    if candidate_topk_blocks is not None and (
        type(candidate_topk_blocks) is not int or candidate_topk_blocks <= 0
    ):
        raise ValueError("candidate_topk_blocks must be positive")
    if candidate_blocks is not None:
        if candidate_selector is not None:
            raise ValueError("one index selection cannot both create and consume candidate blocks")
        if type(candidate_block_size) is not int or candidate_block_size <= 0:
            raise ValueError("candidate_block_size is required when consuming candidate blocks")
        expected_blocks = (keys.shape[1] + candidate_block_size - 1) // candidate_block_size
        if candidate_topk_blocks is None:
            raise ValueError("candidate_topk_blocks is required when consuming candidate blocks")
        candidate_slots = min(candidate_topk_blocks, expected_blocks)
        if tuple(candidate_blocks.shape) != (batch, query_count, candidate_slots):
            raise ValueError("candidate block IDs have an unexpected shape")
        if candidate_blocks.dtype != torch.int16:
            raise ValueError("candidate block IDs must use int16 storage")
        if topk > candidate_slots * candidate_block_size:
            raise ValueError("candidate block slots cannot supply the requested top-k")
    if candidate_selector is not None:
        if type(candidate_block_size) is not int or candidate_block_size <= 0:
            raise ValueError("candidate_block_size is required when selecting candidates")
        if type(candidate_topk_blocks) is not int or candidate_topk_blocks <= 0:
            raise ValueError("candidate_topk_blocks must be positive")
        candidate_count = (keys.shape[1] + candidate_block_size - 1) // candidate_block_size
        if candidate_count > torch.iinfo(torch.int16).max:
            raise ValueError("candidate block IDs exceed the 220k-profile int16 range")
        candidate_slots = min(candidate_topk_blocks, candidate_count)
        selected_candidate_blocks = torch.empty(
            (batch, query_count, candidate_slots), dtype=torch.int16, device=query.device
        )
    else:
        selected_candidate_blocks = None

    selected = torch.empty(
        (batch, query_count, topk), dtype=torch.int32, device=query.device
    )
    key_positions = torch.arange(keys.shape[1], device=query.device)
    for first in range(0, query_count, query_tile_rows):
        last = min(first + query_tile_rows, query_count)
        query_source_tile = query[:, first:last]
        if query_projection is None:
            query_tile = query_source_tile
        else:
            query_tile = query_projection(query_source_tile, first, last)
        if tuple(query_tile.shape) != (batch, last - first, heads, dim):
            raise ValueError("query projection returned an unexpected tile shape")
        weight_source_tile = head_weights[:, first:last]
        if head_weights_projection is None:
            weight_tile = weight_source_tile
        else:
            weight_tile = head_weights_projection(weight_source_tile, first, last)
        if tuple(weight_tile.shape) != (batch, last - first, heads):
            raise ValueError("head-weight projection returned an unexpected tile shape")
        score_heads = torch.einsum("bshd,btd->bsht", query_tile, keys)
        score_heads.relu_()
        score_heads.mul_(weight_tile.unsqueeze(-1))
        scores = score_heads.sum(dim=2)

        if valid_vector is None:
            lens_for_rows = valid_positions
            scores.masked_fill_(key_positions >= lens_for_rows, float("-inf"))
        else:
            lens_for_rows = valid_vector[first:last]
            scores.masked_fill_(
                key_positions.view(1, 1, -1)
                >= lens_for_rows.view(1, last - first, 1),
                float("-inf"),
            )

        candidate_block_mask = None
        if candidate_blocks is not None:
            block_ids = candidate_blocks[:, first:last].to(torch.int32)
            num_blocks = (keys.shape[1] + candidate_block_size - 1) // candidate_block_size
            valid_block_ids = (block_ids >= 0) & (block_ids < num_blocks)
            invalid_destinations = num_blocks + torch.arange(
                block_ids.shape[-1], device=query.device
            ).view(1, 1, -1)
            destinations = torch.where(valid_block_ids, block_ids, invalid_destinations)
            block_mask_with_scratch = torch.zeros(
                (batch, last - first, num_blocks + block_ids.shape[-1]),
                dtype=torch.bool,
                device=query.device,
            )
            block_mask_with_scratch.scatter_(
                -1, destinations.to(torch.int64), valid_block_ids
            )
            candidate_block_mask = block_mask_with_scratch[..., :num_blocks]
            dense_candidate_mask = candidate_block_mask.repeat_interleave(
                candidate_block_size, dim=-1
            )[..., : keys.shape[1]]
            scores.masked_fill_(~dense_candidate_mask, float("-inf"))

        if candidate_selector is not None:
            dense_blocks = candidate_selector(
                scores,
                lens_for_rows,
                candidate_topk_blocks,
                candidate_block_size,
            )
            if dense_blocks.shape != (batch, last - first, keys.shape[1]):
                raise ValueError("candidate selector returned an unexpected dense mask shape")
            block_mask = dense_blocks[..., ::candidate_block_size]
            block_ids = torch.arange(block_mask.size(-1), device=query.device).view(
                1, 1, -1
            ).expand(batch, last - first, -1)
            sentinel = torch.iinfo(torch.int16).max
            ordered_ids = torch.where(block_mask, block_ids, sentinel).topk(
                selected_candidate_blocks.size(-1), dim=-1, largest=False, sorted=True
            ).values
            selected_candidate_blocks[:, first:last] = ordered_ids.to(torch.int16)

        local = scores.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        if valid_vector is None:
            valid = local < lens_for_rows
        else:
            valid = local < lens_for_rows.view(1, last - first, 1)
        selected[:, first:last] = torch.where(valid, local + offset, -1).to(torch.int32)

        # PyTorch evaluates the next tile's einsum before replacing these names.
        # Drop the large outputs now so the caching allocator can reuse their
        # storage instead of overlapping two score tiles.
        del (
            query_source_tile,
            query_tile,
            weight_source_tile,
            weight_tile,
            score_heads,
            scores,
            local,
            lens_for_rows,
            valid,
        )
        if candidate_blocks is not None:
            del (
                block_ids,
                valid_block_ids,
                invalid_destinations,
                destinations,
                block_mask_with_scratch,
                candidate_block_mask,
                dense_candidate_mask,
            )
        if candidate_selector is not None:
            del dense_blocks, block_mask, block_ids, sentinel, ordered_ids

    return selected, selected_candidate_blocks


@contextmanager
def install_tiled_indexer(reference: dict, *, query_tile_rows: int):
    """Temporarily replace the pinned Indexer.forward with the tiled equivalent."""
    indexer_type = reference.get("Indexer")
    if indexer_type is None or not callable(getattr(indexer_type, "forward", None)):
        raise TypeError("pinned model reference lacks Indexer.forward")
    original_forward = indexer_type.forward
    model_globals = original_forward.__globals__
    required = (
        "apply_rotary_emb",
        "fp4_act_quant",
        "shared_attn",
        "select_candidate_blocks",
    )
    if any(name not in model_globals for name in required):
        raise RuntimeError("pinned indexer globals do not match the verified 1017 graph")

    def forward(self, x, qr, latent, start_pos, offset):
        batch, query_count, _ = x.size()
        if query_count <= query_tile_rows:
            return original_forward(self, x, qr, latent, start_pos, offset)
        ratio = self.compress_ratio
        end_pos = start_pos + query_count
        if self.freqs_cis is None:
            raise AssertionError("indexer rotary table was not attached")

        if self.owns_k and latent is not None:
            freqs = (
                self.freqs_cis[: query_count - query_count % ratio : ratio]
                if start_pos == 0
                else self.freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
            )
            keys_to_store = self.k_norm(self.wk(latent))
            model_globals["apply_rotary_emb"](keys_to_store[..., -self.rope_head_dim :], freqs)
            model_globals["fp4_act_quant"](keys_to_store, model_globals["fp4_block_size"], True)
            first_key = start_pos // ratio
            self.k_cache[:batch, first_key : first_key + keys_to_store.size(1)] = keys_to_store
        if self.owns_k:
            model_globals["shared_attn"].index_k = self.k_cache

        index_keys = model_globals["shared_attn"].index_k[:batch, : end_pos // ratio]

        def project_query(source_tile, first, last):
            projected = self.wq_b(source_tile).unflatten(
                -1, (self.n_local_heads, self.index_head_dim)
            )
            model_globals["apply_rotary_emb"](
                projected[..., -self.rope_head_dim :],
                self.freqs_cis[start_pos + first : start_pos + last],
            )
            model_globals["fp4_act_quant"](
                projected, model_globals["fp4_block_size"], True
            )
            return projected

        def project_weights(source_tile, _first, _last):
            return self.weights_proj(source_tile) * (
                self.softmax_scale * self.n_heads**-0.5
            )

        if start_pos == 0:
            torch = _torch_for(x)
            valid_positions = torch.arange(1, query_count + 1, device=x.device) // ratio
        else:
            valid_positions = end_pos // ratio

        existing_candidates = (
            model_globals["shared_attn"].candidates if self.uses_candidates else None
        )
        selected, candidates = tiled_indexer_selection(
            qr,
            index_keys,
            x,
            query_heads=self.n_local_heads,
            query_projection=project_query,
            head_weights_projection=project_weights,
            valid_positions=valid_positions,
            topk=min(self.index_topk, end_pos // ratio),
            offset=offset,
            query_tile_rows=query_tile_rows,
            candidate_blocks=existing_candidates,
            candidate_block_size=(
                self.candidate_block_size if self.uses_candidates or self.is_candidate_source else None
            ),
            candidate_selector=(
                model_globals["select_candidate_blocks"] if self.is_candidate_source else None
            ),
            candidate_topk_blocks=(
                self.candidate_topk_blocks if self.uses_candidates or self.is_candidate_source else None
            ),
        )
        if self.is_candidate_source:
            model_globals["shared_attn"].candidates = candidates
        return selected

    indexer_type.forward = forward
    try:
        yield
    finally:
        indexer_type.forward = original_forward


def _torch_for(tensor):
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("index selection inputs must be tensors")
    return torch
