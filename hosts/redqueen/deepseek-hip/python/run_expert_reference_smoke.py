"""Compare cached native FP4 expert execution with pinned 1017 Expert.forward."""

from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from checkpoint_reader import SafeTensorCheckpoint
from expert_cache import FP4ExpertCache, FP4ExpertKey
from expert_execution import execute_fp4_expert
from expert_weights import estimate_fp4_expert_memory, load_fp4_expert


def main() -> None:
    inference_dir = Path(
        os.environ.get(
            "DS4_INFERENCE_DIR", "/srv/ai/models/deepseek-v4.1-flash-1017/inference"
        )
    ).expanduser()
    model_path = inference_dir / "model.py"
    if not model_path.is_file():
        raise SystemExit(f"official DeepSeek inference/model.py not found: {model_path}")
    config_path = inference_dir / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"pinned DeepSeek inference/config.json not found: {config_path}")
    model_config = json.loads(config_path.read_text())
    swiglu_limit = model_config.get("swiglu_limit")
    if (
        isinstance(swiglu_limit, bool)
        or not isinstance(swiglu_limit, (int, float))
        or swiglu_limit <= 0
    ):
        raise SystemExit("pinned 1017 config must declare a positive swiglu_limit")

    adapter_dir = Path(__file__).resolve().parent
    checkpoint_root = inference_dir.parent
    revision_path = checkpoint_root / "SOURCE_REVISION"
    if not revision_path.is_file():
        raise SystemExit(f"pinned checkpoint source metadata not found: {revision_path}")
    revision_fields = dict(
        line.split("=", 1)
        for line in revision_path.read_text().splitlines()
        if "=" in line
    )

    # The official graph imports this module by name. Resolve it to the native
    # HIP bridge before exposing the pinned reference directory.
    sys.path[:0] = [str(adapter_dir), str(inference_dir)]
    import kernel  # noqa: PLC0415

    if revision_fields.get("revision") != kernel.REFERENCE_COMMIT:
        raise SystemExit(
            f"DeepSeek source revision {revision_fields.get('revision')!r} does not match "
            f"native bridge pin {kernel.REFERENCE_COMMIT}"
        )
    sys.modules["kernel"] = kernel
    reference = runpy.run_path(str(model_path), run_name="redqueen_expert_reference")

    if not torch.cuda.is_available() or torch.version.hip is None:
        raise SystemExit("the expert reference smoke requires a ROCm GPU")
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    key = FP4ExpertKey(layer_id=0, expert_id=0)

    with SafeTensorCheckpoint(
        checkpoint_root, expected_revision=kernel.REFERENCE_COMMIT
    ) as checkpoint:
        estimate = estimate_fp4_expert_memory(
            checkpoint, key.layer_id, key.expert_id, mtp_layer=key.mtp_layer
        )
        cache = FP4ExpertCache(
            estimate.resident_bytes,
            estimate_resident_bytes=lambda requested: estimate_fp4_expert_memory(
                checkpoint,
                requested.layer_id,
                requested.expert_id,
                mtp_layer=requested.mtp_layer,
            ).resident_bytes,
            load=lambda requested: load_fp4_expert(
                checkpoint,
                requested.layer_id,
                requested.expert_id,
                device=device,
                mtp_layer=requested.mtp_layer,
            ),
        )
        with cache.acquire(key) as expert:
            dim = expert.w1.shape[1] * 2
            activations = torch.randn((2, dim), device=device, dtype=torch.bfloat16)
            routing_weights = torch.tensor(
                [[0.25], [0.8]], device=device, dtype=torch.float32
            )

            class Projection:
                def __init__(self, weight):
                    self.weight = weight

                def __call__(self, values):
                    return reference["linear"](values, self.weight)

            reference_expert = SimpleNamespace(
                w1=Projection(expert.w1),
                w2=Projection(expert.w2),
                w3=Projection(expert.w3),
                swiglu_limit=swiglu_limit,
            )
            expected = reference["Expert"].forward(
                reference_expert, activations, routing_weights
            )
            torch.cuda.current_stream(device).synchronize()

        actual = execute_fp4_expert(
            cache,
            key,
            activations,
            reference["linear"],
            swiglu_limit=swiglu_limit,
            routing_weights=routing_weights,
        )
        torch.cuda.synchronize(device)
        if not torch.equal(actual, expected):
            max_difference = (actual.float() - expected.float()).abs().max().item()
            raise AssertionError(
                f"cached expert differs from pinned Expert.forward; max_abs={max_difference}"
            )
        if cache.stats.leased_bytes != 0:
            raise AssertionError("expert lease remained active after the GPU fence")
        print(
            "cached expert matches pinned Expert.forward:",
            f"revision={kernel.REFERENCE_COMMIT}",
            f"device={torch.cuda.get_device_name(device)}",
            f"expert_bytes={estimate.resident_bytes}",
            f"shape={tuple(actual.shape)}",
            f"exact={torch.equal(actual, expected)}",
        )


if __name__ == "__main__":
    main()
