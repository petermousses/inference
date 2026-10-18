"""Compare cached routed MoE execution with the pinned 1017 MoE.forward."""

from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path

import torch

from expert_cache import FP4ExpertCache, FP4ExpertKey
from expert_moe import execute_cached_moe
from expert_weights import LoadedFP4Expert


def main() -> None:
    inference_dir = Path(
        os.environ.get(
            "DS4_INFERENCE_DIR", "/srv/ai/models/deepseek-v4.1-flash-1017/inference"
        )
    ).expanduser()
    model_path = inference_dir / "model.py"
    config_path = inference_dir / "config.json"
    if not model_path.is_file() or not config_path.is_file():
        raise SystemExit(f"pinned DeepSeek inference source is missing from {inference_dir}")
    model_config = json.loads(config_path.read_text())
    checkpoint_root = inference_dir.parent
    revision_path = checkpoint_root / "SOURCE_REVISION"
    if not revision_path.is_file():
        raise SystemExit(f"pinned checkpoint source metadata not found: {revision_path}")
    revision_fields = dict(
        line.split("=", 1)
        for line in revision_path.read_text().splitlines()
        if "=" in line
    )

    adapter_dir = Path(__file__).resolve().parent
    sys.path[:0] = [str(adapter_dir), str(inference_dir)]
    import kernel  # noqa: PLC0415

    if revision_fields.get("revision") != kernel.REFERENCE_COMMIT:
        raise SystemExit(
            f"DeepSeek source revision {revision_fields.get('revision')!r} does not match "
            f"native bridge pin {kernel.REFERENCE_COMMIT}"
        )
    reference = runpy.run_path(str(model_path), run_name="redqueen_moe_reference")
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise SystemExit("the MoE reference smoke requires a ROCm GPU")

    torch.manual_seed(0)
    device = torch.device("cuda:0")
    args = reference["ModelArgs"](
        dim=32,
        moe_inter_dim=32,
        n_layers=1,
        n_routed_experts=4,
        n_activated_experts=2,
        n_shared_experts=1,
        score_func=model_config["score_func"],
        route_scale=model_config["route_scale"],
        swiglu_limit=model_config["swiglu_limit"],
        dtype=model_config["dtype"],
        expert_dtype="fp4",
        vision_n_layers=1,
    )
    moe = reference["MoE"](0, args).to(device)
    with torch.no_grad():
        moe.gate.weight.zero_()
        moe.gate.bias.copy_(torch.tensor([3.0, 2.0, 1.0, 0.0], device=device))
        moe.gate.bias_vl.copy_(torch.tensor([0.0, 1.0, 2.0, 3.0], device=device))
        for expert in moe.experts:
            for projection in (expert.w1, expert.w2, expert.w3):
                packed = torch.randint(
                    0,
                    256,
                    projection.weight.shape,
                    dtype=torch.uint8,
                    device=device,
                ).view(torch.int8)
                packed = packed.view(torch.float4_e2m1fn_x2)
                projection.weight.copy_(packed)
                projection.scale.fill_(1.0)
        for parameter in moe.shared_experts.parameters():
            if parameter.dtype == torch.float8_e4m3fn:
                values = torch.randn(
                    parameter.shape, device=device, dtype=torch.float32
                ).mul_(0.5)
                parameter.copy_(values.to(torch.float8_e4m3fn))
            elif parameter.dtype == torch.float8_e8m0fnu:
                parameter.fill_(1.0)
            else:
                parameter.uniform_(-0.5, 0.5)

    keys = [FP4ExpertKey(0, expert_id) for expert_id in range(args.n_routed_experts)]
    experts = {}
    for key in keys:
        source = moe.experts[key.expert_id]
        weights = []
        for projection in (source.w1, source.w2, source.w3):
            weight = projection.weight.detach()
            weight.scale = projection.scale.detach()
            weights.append(weight)
        experts[key] = LoadedFP4Expert(
            layer_id=key.layer_id,
            expert_id=key.expert_id,
            mtp_layer=key.mtp_layer,
            w1=weights[0],
            w2=weights[1],
            w3=weights[2],
        )

    expert_bytes = experts[keys[0]].resident_bytes
    loaded_expert_ids = []

    def load_expert(key):
        loaded_expert_ids.append(key.expert_id)
        return experts[key]

    cache = FP4ExpertCache(
        expert_bytes,
        estimate_resident_bytes=lambda key: experts[key].resident_bytes,
        load=load_expert,
    )
    activations = torch.randn((2, 3, args.dim), device=device, dtype=torch.bfloat16) * 4
    image_mask = torch.tensor(
        [[False, True, False], [True, False, True]], device=device, dtype=torch.bool
    )
    expected = moe(activations, image_mask)
    torch.cuda.current_stream(device).synchronize()
    actual = execute_cached_moe(
        cache,
        activations,
        moe.gate,
        moe.shared_experts,
        reference["linear"],
        layer_id=0,
        n_routed_experts=args.n_routed_experts,
        swiglu_limit=args.swiglu_limit,
        image_mask=image_mask,
    )
    torch.cuda.synchronize(device)
    if not torch.equal(actual, expected):
        max_difference = (actual.float() - expected.float()).abs().max().item()
        raise AssertionError(
            f"cached MoE differs from pinned MoE.forward; max_abs={max_difference}"
        )
    if cache.stats.leased_bytes != 0:
        raise AssertionError("expert lease remained active after routed MoE completion")
    if sorted(loaded_expert_ids) != list(range(args.n_routed_experts)):
        raise AssertionError(
            "image-aware route smoke did not exercise every configured expert"
        )
    print(
        "cached MoE matches pinned MoE.forward:",
        f"revision={kernel.REFERENCE_COMMIT}",
        f"device={torch.cuda.get_device_name(device)}",
        f"tokens={activations.shape[0] * activations.shape[1]}",
        f"expert_loads={len(loaded_expert_ids)}",
        f"expert_bytes={expert_bytes}",
        f"exact={torch.equal(actual, expected)}",
    )


if __name__ == "__main__":
    main()
