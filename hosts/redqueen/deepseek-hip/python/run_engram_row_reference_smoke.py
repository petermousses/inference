"""Compare bounded Engram row reads with the pinned 1017 graph on live headers/data."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from checkpoint_mapping_audit import (
    _execute_pinned_model,
    _pinned_inference_imports,
    verify_pinned_inference_sources,
)
from checkpoint_reader import SafeTensorCheckpoint
from engram_rows import install_engram_row_reads
from torch_checkpoint import read_rows_to_device


def _compare_with_checkpoint(checkpoint, reader, embed, torch) -> None:
    indices = torch.tensor(
        [[[-1, 0, 1, 2, reader.num_embeddings // 2, reader.num_embeddings - 1,
           reader.num_embeddings]]],
        dtype=torch.int64,
    )
    actual = embed(indices)
    flat = indices.reshape(-1)
    mask = (flat < 0) | (flat >= reader.num_embeddings)
    valid = flat[~mask]
    unique_rows, inverse = torch.unique(valid, sorted=True, return_inverse=True)
    rows = unique_rows.tolist()
    weights = read_rows_to_device(
        checkpoint, reader.weight_name, rows, device="cpu"
    )
    scales = read_rows_to_device(
        checkpoint, reader.scale_name, rows, device="cpu"
    )
    expected_values = (
        weights.index_select(0, inverse).float().unflatten(-1, (-1, 32))
        * scales.index_select(0, inverse).float().unsqueeze(-1)
    ).flatten(-2).to(torch.bfloat16)
    expected = torch.zeros(
        (flat.numel(), reader.embedding_dim), dtype=torch.bfloat16
    )
    expected[~mask] = expected_values
    expected = expected.reshape(*indices.shape, reader.embedding_dim)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def main() -> int:
    inference_dir = Path(
        os.environ.get(
            "DS4_INFERENCE_DIR", "/srv/ai/models/deepseek-v4.1-flash-1017/inference"
        )
    ).expanduser()
    checkpoint_root = Path(
        os.environ.get("DS4_CHECKPOINT_DIR", str(inference_dir.parent))
    ).expanduser()
    adapter_dir = Path(__file__).resolve().parent
    source_bytes = verify_pinned_inference_sources(inference_dir)
    model_path = inference_dir / "model.py"

    previous_path = sys.path[:]
    previous_kernel = sys.modules.pop("kernel", None)
    sys.path[:0] = [str(adapter_dir), str(inference_dir)]
    try:
        import kernel  # noqa: PLC0415

        sys.modules["kernel"] = kernel
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(f"Engram smoke dependency is unavailable: {exc.name}") from exc
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for the Engram row smoke") from exc

        with _pinned_inference_imports(source_bytes, inference_dir):
            reference = _execute_pinned_model(source_bytes["model.py"], model_path)
            try:
                model_config = json.loads(source_bytes["config.json"].decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"cannot parse pinned model config: {exc}") from exc
            if not isinstance(model_config, dict):
                raise RuntimeError("pinned model config must be an object")
            previous_dtype = torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.bfloat16)
                args = reference["ModelArgs"](**model_config)
                args.max_batch_size = 1
                args.max_seq_len = 1
                tokenizer = AutoTokenizer.from_pretrained(
                    str(checkpoint_root), local_files_only=True
                )
                with torch.device("meta"):
                    transformer = reference["Transformer"](args, tokenizer)
            finally:
                torch.set_default_dtype(previous_dtype)

            with SafeTensorCheckpoint(
                checkpoint_root,
                expected_revision=kernel.REFERENCE_COMMIT,
            ) as checkpoint:
                with install_engram_row_reads(transformer, checkpoint) as installation:
                    readers = installation.readers
                    if set(readers) != {1, 14}:
                        raise RuntimeError(f"unexpected pinned Engram layers: {sorted(readers)}")
                    for layer_id, reader in readers.items():
                        embed = transformer.layers[layer_id].engram.embed
                        _compare_with_checkpoint(checkpoint, reader, embed.forward, torch)
                    if any(
                        not transformer.layers[layer_id].engram.embed.weight.is_meta
                        or not transformer.layers[layer_id].engram.embed.scale.is_meta
                        for layer_id in readers
                    ):
                        raise RuntimeError("Engram tables were materialized during row smoke")
    finally:
        sys.path[:] = previous_path
        sys.modules.pop("kernel", None)
        if previous_kernel is not None:
            sys.modules["kernel"] = previous_kernel

    print("pinned Engram row reference smoke: PASS; layers=1,14; table weights remained meta")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
