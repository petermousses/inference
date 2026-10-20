"""Sequence-tiled exact Hyper-Connections operations for the pinned 1017 graph."""

from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def install_tiled_hyper_connections(reference: dict, *, row_tile_rows: int):
    """Tile the pinned Block HC operations to bound their temporary tensors.

    The eager ``hc_post`` broadcasts to ``[B,S,hc,hc,D]``. Tiling across S
    preserves each row's expression and reduction order while bounding that
    intermediate. ``hc_mixes`` and ``hc_pre`` use the same row tiling to avoid
    full-sequence float32 copies during long prefills.
    """
    block_type = reference.get("Block")
    if block_type is None:
        raise TypeError("pinned model reference lacks Block")
    methods = ("hc_mixes", "hc_pre", "hc_post")
    originals = {name: getattr(block_type, name, None) for name in methods}
    if any(not callable(method) for method in originals.values()):
        raise TypeError("pinned Block lacks a callable HC method")
    if type(row_tile_rows) is not int or row_tile_rows <= 0:
        raise ValueError("row_tile_rows must be a positive integer")

    model_globals = originals["hc_mixes"].__globals__
    if any(name not in model_globals for name in ("F", "torch", "hc_split_sinkhorn")):
        raise RuntimeError("pinned Block globals do not match the verified 1017 graph")

    def hc_mixes(self, x, hc_fn, hc_scale, hc_base):
        batch, sequence, hc_mult, _dim = x.shape
        if sequence <= row_tile_rows:
            return originals["hc_mixes"](self, x, hc_fn, hc_scale, hc_base)
        if hc_mult != self.hc_mult:
            raise ValueError("HC input copy count differs from the pinned Block")

        torch = model_globals["torch"]
        functional = model_globals["F"]
        mixes = torch.empty(
            (batch, sequence, (2 + hc_mult) * hc_mult),
            dtype=torch.float32,
            device=x.device,
        )
        for first in range(0, sequence, row_tile_rows):
            last = min(first + row_tile_rows, sequence)
            flattened = x[:, first:last].flatten(2).float()
            rsqrt = torch.rsqrt(
                flattened.square().mean(-1, keepdim=True) + self.norm_eps
            )
            projected = functional.linear(flattened, hc_fn) * rsqrt
            mixes[:, first:last] = projected
            del flattened, rsqrt, projected

        result = model_globals["hc_split_sinkhorn"](
            mixes,
            hc_scale,
            hc_base,
            hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )
        del mixes
        return result

    def hc_pre(self, x, pre_mix):
        batch, sequence, hc_mult, dim = x.shape
        if sequence <= row_tile_rows:
            return originals["hc_pre"](self, x, pre_mix)
        output = model_globals["torch"].empty(
            (batch, sequence, dim), dtype=x.dtype, device=x.device
        )
        for first in range(0, sequence, row_tile_rows):
            last = min(first + row_tile_rows, sequence)
            x_tile = x[:, first:last]
            mix_tile = pre_mix[:, first:last]
            weighted = mix_tile.unsqueeze(-1) * x_tile.float()
            output[:, first:last] = weighted.sum(dim=2).to(x.dtype)
            del x_tile, mix_tile, weighted
        return output

    def hc_post(self, x, residual, post, comb):
        batch, sequence, hc_mult, dim = residual.shape
        if sequence <= row_tile_rows:
            return originals["hc_post"](self, x, residual, post, comb)
        if hc_mult != self.hc_mult or x.shape != (batch, sequence, dim):
            raise ValueError("HC post inputs differ from the pinned Block dimensions")

        output = model_globals["torch"].empty_like(residual)
        for first in range(0, sequence, row_tile_rows):
            last = min(first + row_tile_rows, sequence)
            x_tile = x[:, first:last]
            residual_tile = residual[:, first:last]
            post_tile = post[:, first:last]
            comb_tile = comb[:, first:last]
            projected = post_tile.unsqueeze(-1) * x_tile.unsqueeze(-2)
            combined = (
                comb_tile.unsqueeze(-1) * residual_tile.unsqueeze(-2)
            ).sum(dim=2)
            output_tile = (projected + combined).type_as(x)
            output[:, first:last] = output_tile
            del (
                x_tile,
                residual_tile,
                post_tile,
                comb_tile,
                projected,
                combined,
                output_tile,
            )
        return output

    block_type.hc_mixes = hc_mixes
    block_type.hc_pre = hc_pre
    block_type.hc_post = hc_post
    try:
        yield
    finally:
        for name, original in originals.items():
            setattr(block_type, name, original)
