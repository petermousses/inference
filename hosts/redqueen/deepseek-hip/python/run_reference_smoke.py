"""Compare cached native MoE execution with the pinned full-model graph."""

from __future__ import annotations

import asyncio
import os
import runpy
import sys
import threading
from pathlib import Path


from cached_model import install_cached_moe, run_reserved_sync
from expert_cache import FP4ExpertCache, FP4ExpertKey
from expert_weights import LoadedFP4Expert


def initialize_smoke_parameters(model, torch, device) -> None:
    """Give the uncheckpointed reference graph finite, reproducible weights."""
    fp4_dtype = torch.float4_e2m1fn_x2
    scale_dtype = torch.float8_e8m0fnu
    fp8_dtype = torch.float8_e4m3fn
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.dtype == fp4_dtype:
                packed = torch.randint(
                    0,
                    256,
                    parameter.shape,
                    dtype=torch.uint8,
                    device=device,
                ).view(torch.int8)
                parameter.copy_(packed.view(fp4_dtype))
            elif parameter.dtype == scale_dtype:
                parameter.fill_(1.0)
            elif parameter.dtype == fp8_dtype:
                values = torch.randn(
                    parameter.shape, dtype=torch.float32, device=device
                ).mul_(0.05)
                parameter.copy_(values.to(fp8_dtype))
            elif parameter.is_floating_point():
                values = torch.randn(
                    parameter.shape, dtype=torch.float32, device=device
                ).mul_(0.02)
                parameter.copy_(values.to(parameter.dtype))


def main() -> None:
    inference_dir = Path(
        os.environ.get("DS4_INFERENCE_DIR", "/srv/ai/models/deepseek-v4.1-flash-1017/inference")
    ).expanduser()
    model_path = inference_dir / "model.py"
    if not model_path.is_file():
        raise SystemExit(f"official DeepSeek inference/model.py not found: {model_path}")

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
    # Import our kernel module before exposing the reference directory. The
    # reference's own kernel.py imports TileLang and is intentionally bypassed.
    sys.path[:0] = [str(adapter_dir), str(inference_dir)]
    import kernel  # noqa: PLC0415

    if revision_fields.get("revision") != kernel.REFERENCE_COMMIT:
        raise SystemExit(
            f"DeepSeek source revision {revision_fields.get('revision')!r} does not match "
            f"native bridge pin {kernel.REFERENCE_COMMIT}"
        )

    sys.modules["kernel"] = kernel
    reference = runpy.run_path(str(model_path), run_name="redqueen_deepseek_reference")
    usage = {"rmsnorm": 0, "rope": 0}
    # run_path returns a shallow globals copy; function objects still resolve
    # names through their original module dictionary.
    rmsnorm_class = reference["RMSNorm"]
    model_globals = rmsnorm_class.forward.__globals__
    if model_globals.get("RMSNorm") is not rmsnorm_class:
        raise RuntimeError("could not resolve pinned model module globals")
    if "apply_rotary_emb" not in model_globals:
        raise RuntimeError("pinned model module has no apply_rotary_emb global")

    def native_rmsnorm(self, x):
        usage["rmsnorm"] += 1
        return kernel.rmsnorm(x, self.weight, self.eps)

    def native_rope(x, freqs_cis, inverse=False):
        usage["rope"] += 1
        return kernel.apply_rotary_emb(x, freqs_cis, inverse)

    rmsnorm_class.forward = native_rmsnorm
    model_globals["apply_rotary_emb"] = native_rope

    # Run the same small random graph exercise as the pinned model.py main
    # block, after swapping only RMSNorm and RoPE for their native HIP paths.
    torch = reference["torch"]
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    args = reference["ModelArgs"](
        dspark_block_size=6,
        dspark_target_layer_ids=(3, 4),
        temperature=0,
    )
    tokens = torch.randint(0, args.vocab_size, (2, 150))
    model = reference["Transformer"](args)
    device = torch.device("cuda")
    initialize_smoke_parameters(model, torch, device)

    def run_graph():
        snapshots = []

        def snapshot(output_ids, logits, main_hidden):
            return tuple(
                None if value is None else value.detach().cpu().clone()
                for value in (output_ids, logits, main_hidden)
            )

        output_ids, logits, main_hidden = model(tokens[:, :128])
        snapshots.append(snapshot(output_ids, logits, main_hidden))
        model.forward_spec(output_ids, main_hidden)
        for position in range(128, 150):
            output_ids, logits, main_hidden = model(
                tokens[:, position : position + 1], position
            )
            result = model.forward_spec(output_ids, main_hidden, position)
            if result is None:
                raise RuntimeError("pinned graph did not return a DSpark step")
            snapshots.append(snapshot(output_ids, logits, main_hidden))
            output_ids, _, _ = result
        return snapshots

    expected = run_graph()

    def moe_for(key):
        if key.mtp_layer:
            return model.mtp[key.layer_id].ffn
        return model.layers[key.layer_id].ffn

    def expert_for(key):
        moe = moe_for(key)
        expert = moe.experts[key.expert_id]
        if expert is None:
            raise RuntimeError(f"synthetic model has no expert for {key}")
        return LoadedFP4Expert(
            layer_id=key.layer_id,
            expert_id=key.expert_id,
            mtp_layer=key.mtp_layer,
            w1=expert.w1.weight,
            w2=expert.w2.weight,
            w3=expert.w3.weight,
        )

    cache = FP4ExpertCache(
        1,
        estimate_resident_bytes=lambda key: expert_for(key).resident_bytes,
        load=expert_for,
    )
    cache.resize(
        max(
            expert_for(FP4ExpertKey(layer_id, 0, mtp_layer=mtp_layer)).resident_bytes
            for mtp_layer, blocks in ((False, model.layers), (True, model.mtp))
            for layer_id in range(len(blocks))
        )
    )
    reservation_active = threading.Event()

    class SmokeReservation:
        async def __aenter__(self):
            reservation_active.set()

        async def __aexit__(self, exc_type, exc, traceback):
            reservation_active.clear()

    def request_fence():
        if not reservation_active.is_set():
            raise RuntimeError("GPU fence ran outside the request memory reservation")
        torch.cuda.synchronize(device)

    with install_cached_moe(
        model, cache, reference["linear"], fence=request_fence
    ):
        actual = asyncio.run(
            run_reserved_sync(
                lambda: (torch.cuda.set_device(device), run_graph())[1],
                SmokeReservation(),
                cache=cache,
                fence=request_fence,
            )
        )

    if len(actual) != len(expected) or any(
        not torch.equal(expected_tensor, actual_tensor)
        for expected_step, actual_step in zip(expected, actual)
        for expected_tensor, actual_tensor in zip(expected_step, actual_step)
        if expected_tensor is not None
    ):
        raise AssertionError("cached full-model graph differs from the pinned MoE path")
    stats = cache.stats
    if stats.leased_bytes or stats.loading_bytes or stats.pending_loads:
        raise RuntimeError("cached transformer returned with an outstanding expert lease")
    if not stats.within_capacity or stats.entries > 1:
        raise RuntimeError("single-expert cache exceeded its configured capacity")
    if not usage["rmsnorm"] or not usage["rope"]:
        raise RuntimeError(f"native graph overrides were not exercised: {usage}")
    print(
        "cached native reference graph passed:",
        f"RMSNorm={usage['rmsnorm']}",
        f"RoPE={usage['rope']}",
        f"cache_entries={stats.entries}",
        f"cache_capacity_bytes={stats.capacity_bytes}",
        "exact=True",
    )


if __name__ == "__main__":
    main()
