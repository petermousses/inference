"""Parity tests for sequence-tiled 1017 Hyper-Connections operations."""

import unittest

import torch
import torch.nn.functional as F

from tiled_hyper_connections import install_tiled_hyper_connections


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps):
    del hc_scale, hc_base, sinkhorn_iters, eps
    pre, post, comb = mixes.split((hc_mult, hc_mult, hc_mult * hc_mult), dim=-1)
    return pre.tanh(), post.tanh(), comb.tanh().unflatten(-1, (hc_mult, hc_mult))


class FakeBlock:
    hc_mult = 4
    norm_eps = 1e-6
    hc_sinkhorn_iters = 3
    hc_eps = 1e-6

    def hc_mixes(self, x, hc_fn, hc_scale, hc_base):
        flattened = x.flatten(2).float()
        rsqrt = torch.rsqrt(flattened.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(flattened, hc_fn) * rsqrt
        return hc_split_sinkhorn(
            mixes,
            hc_scale,
            hc_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )

    def hc_pre(self, x, pre_mix):
        return (pre_mix.unsqueeze(-1) * x.float()).sum(dim=2).to(x.dtype)

    def hc_post(self, x, residual, post, comb):
        return (
            post.unsqueeze(-1) * x.unsqueeze(-2)
            + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        ).type_as(x)


class TiledHyperConnectionsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.version.hip is None:
            raise RuntimeError("requires a ROCm GPU; refusing to report a skipped self-test as success")
        cls.device = torch.device("cuda")

    def test_tiled_hc_methods_match_eager_bitwise(self):
        torch.manual_seed(161803)
        block = FakeBlock()
        x = torch.randn((1, 11, 4, 8), dtype=torch.bfloat16, device=self.device)
        hc_fn = torch.randn((24, 32), dtype=torch.float32, device=self.device)
        hc_scale = torch.randn((3,), dtype=torch.float32, device=self.device)
        hc_base = torch.randn((24,), dtype=torch.float32, device=self.device)
        sublayer = torch.randn((1, 11, 8), dtype=torch.bfloat16, device=self.device)

        pre, post, comb = block.hc_mixes(x, hc_fn, hc_scale, hc_base)
        expected_pre = block.hc_pre(x, pre)
        expected_post = block.hc_post(sublayer, x, post, comb)
        original_methods = (FakeBlock.hc_mixes, FakeBlock.hc_pre, FakeBlock.hc_post)

        with install_tiled_hyper_connections({"Block": FakeBlock}, row_tile_rows=3):
            actual_mixes = block.hc_mixes(x, hc_fn, hc_scale, hc_base)
            actual_pre = block.hc_pre(x, actual_mixes[0])
            actual_post = block.hc_post(sublayer, x, actual_mixes[1], actual_mixes[2])

        for actual, expected in zip(actual_mixes, (pre, post, comb)):
            self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(actual_pre, expected_pre))
        self.assertTrue(torch.equal(actual_post, expected_post))
        self.assertEqual(
            (FakeBlock.hc_mixes, FakeBlock.hc_pre, FakeBlock.hc_post), original_methods
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
