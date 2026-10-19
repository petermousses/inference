"""Memory-bounded Engram residual math matches the pinned reference exactly."""

from __future__ import annotations

import copy
import importlib.util
import threading
import unittest
from types import SimpleNamespace

import engram_forward
from engram_forward import install_chunked_engram_forward


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(TORCH_AVAILABLE, "requires PyTorch")
class ChunkedEngramForwardTest(unittest.TestCase):
    def _engram(self):
        torch.manual_seed(41017)

        class Engram(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.hc_mult = 4
                self.dim = 12
                self.eps = 1e-6
                self.clamp_value = 1e-6
                self.embed = torch.nn.Embedding(23, 5)
                self.wkv = torch.nn.Linear(7 * 5, 5 * self.dim, bias=True)
                self.q_weight = torch.nn.Parameter(torch.randn(self.hc_mult, self.dim))
                self.k_weight = torch.nn.Parameter(torch.randn(self.hc_mult, self.dim))

            def forward(self, x, hash_ids, token_mask=None):
                kv = self.wkv(self.embed(hash_ids).flatten(-2))
                key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)
                key = key.float().unflatten(-1, (self.hc_mult, self.dim))
                weight = self.q_weight.float() * self.k_weight.float()
                h, eps = x.float(), self.eps
                rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(
                    key.square().mean(-1) + eps
                )
                dot = (h * weight * key).sum(-1) * rstd * self.dim**-0.5
                gate = torch.sigmoid(
                    torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot)
                )
                if token_mask is not None:
                    gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
                return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)

        return Engram()

    def test_chunked_forward_is_bitwise_identical_for_multiple_tiles_and_masks(self):
        reference = self._engram()
        for sequence_length in (1, 7, 19):
            for tile_rows in (1, 3, 8):
                with self.subTest(sequence_length=sequence_length, tile_rows=tile_rows):
                    candidate = copy.deepcopy(reference)
                    transformer = SimpleNamespace(layers=[SimpleNamespace(engram=candidate)])
                    x = torch.randn(2, sequence_length, 4, 12).to(torch.bfloat16)
                    ids = torch.randint(0, 23, (2, sequence_length, 7))
                    mask = torch.rand(2, sequence_length) > 0.25
                    with torch.inference_mode():
                        expected = candidate(x, ids, mask)
                        with install_chunked_engram_forward(
                            transformer, sequence_tile_rows=tile_rows
                        ):
                            actual = candidate(x, ids, mask)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertIsNotNone(candidate.forward)
                    self.assertFalse(hasattr(candidate, engram_forward._PATCH_MARKER))

    def test_installation_rejects_duplicate_engram_modules(self):
        engram = self._engram()
        transformer = SimpleNamespace(
            layers=[SimpleNamespace(engram=engram), SimpleNamespace(engram=engram)]
        )
        with self.assertRaisesRegex(ValueError, "duplicate Engram"):
            install_chunked_engram_forward(transformer)

    def test_concurrent_installations_do_not_wrap_or_strand_a_patch(self):
        barrier = threading.Barrier(2)
        arrival_lock = threading.Lock()
        arrived = set()

        class EngramStub:
            def __init__(self):
                self.forward = lambda *args: None
                self.embed = lambda *args: None
                self.wkv = lambda *args: None
                self.hc_mult = 1
                self.dim = 1
                self.eps = 1e-6
                self.q_weight = None
                self.k_weight = None
                self.clamp_value = 1e-6

            def __getattr__(self, name):
                if name == engram_forward._PATCH_MARKER:
                    thread_id = threading.get_ident()
                    with arrival_lock:
                        first_lookup = thread_id not in arrived
                        arrived.add(thread_id)
                    if first_lookup:
                        barrier.wait(timeout=5)
                raise AttributeError(name)

        module = EngramStub()
        original = module.forward
        transformer = SimpleNamespace(layers=[SimpleNamespace(engram=module)])
        results = []
        result_lock = threading.Lock()

        def install_and_hold():
            try:
                result = install_chunked_engram_forward(transformer)
            except BaseException as exc:
                result = exc
            with result_lock:
                results.append(result)

        threads = [threading.Thread(target=install_and_hold) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertTrue(all(not thread.is_alive() for thread in threads))

        installations = [result for result in results if not isinstance(result, BaseException)]
        failures = [result for result in results if isinstance(result, BaseException)]
        self.assertEqual(len(installations), 1)
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], RuntimeError)
        installations[0].close()
        self.assertIs(module.forward, original)
        self.assertNotIn(engram_forward._PATCH_MARKER, module.__dict__)

    def test_close_preflights_all_ownership_before_restoring_any_module(self):
        first = self._engram()
        second = self._engram()
        transformer = SimpleNamespace(
            layers=[SimpleNamespace(engram=first), SimpleNamespace(engram=second)]
        )
        installation = install_chunked_engram_forward(transformer)
        first_installed = first.forward
        second_installed = second.forward
        first.forward = lambda *args: None

        with self.assertRaisesRegex(RuntimeError, "forward changed"):
            installation.close()
        self.assertIs(second.forward, second_installed)
        self.assertTrue(hasattr(second, engram_forward._PATCH_MARKER))
        self.assertTrue(hasattr(first, engram_forward._PATCH_MARKER))
        first.forward = first_installed
        installation.close()
        self.assertFalse(hasattr(first, engram_forward._PATCH_MARKER))
        self.assertFalse(hasattr(second, engram_forward._PATCH_MARKER))

    def test_inference_only_patch_rejects_grad_enabled_execution(self):
        engram = self._engram()
        transformer = SimpleNamespace(layers=[SimpleNamespace(engram=engram)])
        x = torch.randn(1, 2, 4, 12, dtype=torch.bfloat16)
        ids = torch.randint(0, 23, (1, 2, 7))
        with install_chunked_engram_forward(transformer):
            with self.assertRaisesRegex(RuntimeError, "restricted to inference"):
                engram(x, ids)


if __name__ == "__main__":
    unittest.main()
