"""Run a checkpoint-backed 1017 prefill and DSpark step on the native HIP path."""

from __future__ import annotations

import asyncio
import threading
from contextlib import contextmanager
from types import SimpleNamespace

from cached_model import install_cached_moe
from engram_rows import install_engram_row_reads
from expert_cache import FP4ExpertCache, FP4ExpertKey
from expert_weights import estimate_fp4_expert_memory, load_fp4_expert


_GIB = 1024**3
_FORWARD_PEAK_BUDGET = 8 * _GIB
_EXPERT_CACHE_CAPACITY = 1 * _GIB


def run_checkpoint_forward_smoke(
    *,
    kernel,
    reference,
    image_processor,
    model,
    args,
    tokenizer,
    checkpoint,
    device,
    torch,
    memory,
    admission_check,
    fence,
    release_allocator,
) -> None:
    """Exercise the full pinned graph with real tokens and one DSpark step.

    Dense weights and the 220k session state must already be resident. The
    request reserves an additional 8 GiB peak before any expert is loaded.
    """
    encoded = tokenizer.encode(
        "The quick brown fox jumps over the lazy dog.", add_special_tokens=False
    )
    if len(encoded) < 4:
        raise RuntimeError("the pinned tokenizer produced too few tokens to exercise Engram")
    text_token_ids = [int(value) for value in encoded[:4]]
    if any(not 0 <= token_id < args.vocab_size for token_id in text_token_ids):
        raise RuntimeError("the pinned tokenizer produced an invalid smoke token")
    if model.vision is None or model.aligner is None or not args.vision_enabled:
        raise RuntimeError("the pinned checkpoint graph has no enabled vision tower")

    image = image_processor["Image"].new("RGB", (544, 544), color=(37, 113, 197))
    with image_processor["io"].BytesIO() as image_file:
        image.save(image_file, format="PNG")
        image_bytes = image_file.getvalue()
    patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = image_processor["load_image"](
        {"data": image_bytes}, args
    )
    image_types = image_processor["image_token_types"](n_llm_h, n_llm_w)
    image_prefix = [
        int(value)
        for value in tokenizer.encode("Describe this image:", add_special_tokens=False)
    ]
    image_token_ids = [int(args.image_token_id)] * int(image_types.numel())
    image_token_ids = image_prefix + image_token_ids
    image_token_types = [int(image_processor["TEXT"])] * len(image_prefix) + [
        int(value) for value in image_types.tolist()
    ]
    image_input = image_processor["ImageInput"](
        start=len(image_prefix),
        patches=patches,
        n_vit_h=n_vit_h,
        n_vit_w=n_vit_w,
        types=image_types,
    )
    if len(image_token_ids) != len(image_token_types) or not image_token_ids:
        raise RuntimeError("pinned image preprocessing returned mismatched token metadata")

    model.temperature = 0
    model.eval()
    loaded_experts: set[FP4ExpertKey] = set()
    engram_lookups: dict[int, int] = {}

    def estimate_expert(key: FP4ExpertKey) -> int:
        return estimate_fp4_expert_memory(
            checkpoint,
            key.layer_id,
            key.expert_id,
            mtp_layer=key.mtp_layer,
        ).resident_bytes

    def load_expert(key: FP4ExpertKey):
        loaded_experts.add(key)
        return load_fp4_expert(
            checkpoint,
            key.layer_id,
            key.expert_id,
            device=device,
            mtp_layer=key.mtp_layer,
        )

    cache = FP4ExpertCache(
        _EXPERT_CACHE_CAPACITY,
        estimate_resident_bytes=estimate_expert,
        load=load_expert,
    )
    request_estimate = SimpleNamespace(
        resident_device_bytes=_EXPERT_CACHE_CAPACITY,
        device_peak_bytes=_FORWARD_PEAK_BUDGET,
        physical_peak_bytes=_FORWARD_PEAK_BUDGET,
    )
    request_active = threading.Event()

    class RequestReservation:
        def __init__(self, name: str):
            self.name = name

        async def __aenter__(self):
            free_gtt, host_available, gtt_total = admission_check(request_estimate)
            torch.cuda.synchronize(device)
            memory["forward_allocated_before"] = torch.cuda.memory_allocated(device)
            memory["forward_reserved_before"] = torch.cuda.memory_reserved(device)
            torch.cuda.reset_peak_memory_stats(device)
            memory["active_leases"] += 1
            request_active.set()
            print(
                f"single-request {self.name} forward admission: "
                f"GTT peak budget={_FORWARD_PEAK_BUDGET / _GIB:.1f} GiB "
                f"(free={free_gtt / _GIB:.2f}/{gtt_total / _GIB:.2f} GiB); "
                f"host available={host_available / _GIB:.2f} GiB; "
                f"expert cache cap={_EXPERT_CACHE_CAPACITY / _GIB:.1f} GiB",
                flush=True,
            )
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            request_active.clear()
            memory["active_leases"] -= 1
            if memory["active_leases"] < 0:
                raise RuntimeError("smoke reservation lease count underflow")
            return False

    def request_fence():
        if not request_active.is_set():
            raise RuntimeError("model work or GPU fence escaped its request reservation")
        fence()

    def request_release_allocator():
        if not request_active.is_set():
            raise RuntimeError("expert backing release escaped its request reservation")
        release_allocator()

    @contextmanager
    def gpu_worker_context():
        previous_device = torch.cuda.current_device()
        previous_default_device = torch.get_default_device()
        previous_default_dtype = torch.get_default_dtype()
        torch.cuda.set_device(device)
        torch.set_default_device(device)
        torch.set_default_dtype(torch.bfloat16)
        try:
            yield
        finally:
            torch.set_default_dtype(previous_default_dtype)
            torch.set_default_device(previous_default_device)
            torch.cuda.set_device(previous_device)

    rmsnorm_class = reference["RMSNorm"]
    model_globals = rmsnorm_class.forward.__globals__
    if model_globals.get("RMSNorm") is not rmsnorm_class:
        raise RuntimeError("could not resolve pinned model globals for native RMSNorm")
    original_rmsnorm = rmsnorm_class.forward
    original_rope = model_globals.get("apply_rotary_emb")
    if not callable(original_rope):
        raise RuntimeError("pinned model has no callable apply_rotary_emb")
    native_usage = {"rmsnorm": 0, "rope": 0}

    def native_rmsnorm(self, values):
        native_usage["rmsnorm"] += 1
        return kernel.rmsnorm(values, self.weight, self.eps)

    def native_rope(values, freqs_cis, inverse=False):
        native_usage["rope"] += 1
        return kernel.apply_rotary_emb(values, freqs_cis, inverse)

    def run_one_request(name, token_ids, *, token_types=None, images=None, image_token_count=0):
        input_token_count = len(token_ids)
        try:
            # The prior request drops its experts and shrinks the logical cap to
            # zero. Restore the admitted cache budget before this request loads
            # any routed experts.
            cache.resize_when_idle(_EXPERT_CACHE_CAPACITY)
            with torch.inference_mode():
                input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
                model_kwargs = {}
                if token_types is not None:
                    model_kwargs["token_types"] = torch.tensor(
                        [token_types], dtype=torch.long, device=device
                    )
                    model_kwargs["images"] = images
                output_ids, logits, main_hidden = model(input_ids, start_pos=0, **model_kwargs)
                if output_ids.shape != (1,):
                    raise RuntimeError(
                        f"pinned prefill returned output_ids shape {tuple(output_ids.shape)}; expected (1,)"
                    )
                if logits.shape != (1, args.vocab_size):
                    raise RuntimeError(
                        f"pinned prefill returned logits shape {tuple(logits.shape)}"
                    )
                if (
                    main_hidden is None
                    or main_hidden.ndim != 3
                    or main_hidden.shape[:2] != (1, input_token_count)
                ):
                    raise RuntimeError("pinned prefill did not return the configured DSpark hidden state")
                if not torch.isfinite(logits).all().item() or not torch.isfinite(main_hidden).all().item():
                    raise RuntimeError("pinned prefill returned non-finite logits or hidden state")

                speculative = model.forward_spec(
                    output_ids, main_hidden[:, -1:], start_pos=input_token_count
                )
                if speculative is None or len(speculative) != 3:
                    raise RuntimeError("pinned DSpark graph did not return its decode step")
                spec_ids, spec_logits, confidence = speculative
                if spec_ids.shape != (1, args.dspark_block_size + 1) or spec_logits.shape != (
                    1,
                    args.dspark_block_size,
                    args.vocab_size,
                ):
                    raise RuntimeError("pinned DSpark graph returned unexpected output shapes")
                if not torch.isfinite(spec_logits).all().item():
                    raise RuntimeError("pinned DSpark graph returned non-finite logits")
                if confidence is not None and torch.is_floating_point(confidence):
                    if not torch.isfinite(confidence).all().item():
                        raise RuntimeError("pinned DSpark graph returned non-finite confidence")

                result = {
                    "name": name,
                    "input_token_count": input_token_count,
                    "main_token_id": int(output_ids[0].item()),
                    "speculative_token_id": int(spec_ids[0, -1].item()),
                    "image_token_count": image_token_count,
                    "native_usage": dict(native_usage),
                }
            request_fence()
            cache.resize_when_idle(0, after_drop=request_release_allocator)
            request_fence()
            torch.cuda.synchronize(device)
            result["allocated_peak"] = (
                torch.cuda.max_memory_allocated(device)
                - memory["forward_allocated_before"]
            )
            result["reserved_peak"] = (
                torch.cuda.max_memory_reserved(device)
                - memory["forward_reserved_before"]
            )
            if result["allocated_peak"] > _FORWARD_PEAK_BUDGET:
                raise RuntimeError(
                    f"single-request {name} allocated peak {result['allocated_peak']} exceeds its "
                    f"{_FORWARD_PEAK_BUDGET}-byte admission budget"
                )
            if result["reserved_peak"] > _FORWARD_PEAK_BUDGET:
                raise RuntimeError(
                    f"single-request {name} allocator peak {result['reserved_peak']} exceeds its "
                    f"{_FORWARD_PEAK_BUDGET}-byte admission budget"
                )
            return result
        except BaseException:
            if not cache.stats.quarantined:
                request_fence()
                cache.resize_when_idle(0, after_drop=request_release_allocator)
                request_fence()
            raise

    rmsnorm_class.forward = native_rmsnorm
    model_globals["apply_rotary_emb"] = native_rope
    text_result = image_result = None
    try:
        with install_engram_row_reads(model, checkpoint) as engram_installation:
            original_lookups = {}
            for layer_id, reader in engram_installation.readers.items():
                engram_lookups[layer_id] = 0
                original_lookup = reader.lookup
                original_lookups[layer_id] = original_lookup

                def counted_lookup(*lookup_args, _lookup=original_lookup, _layer=layer_id, **lookup_kwargs):
                    engram_lookups[_layer] += 1
                    return _lookup(*lookup_args, **lookup_kwargs)

                reader.lookup = counted_lookup
            try:
                with install_cached_moe(
                    model, cache, reference["linear"], fence=request_fence
                ) as installation:
                    async def run_cases():
                        text = await installation.run_reserved_sync(
                            lambda: run_one_request("text", text_token_ids),
                            RequestReservation("text"),
                            worker_context=gpu_worker_context,
                        )
                        image = await installation.run_reserved_sync(
                            lambda: run_one_request(
                                "image",
                                image_token_ids,
                                token_types=image_token_types,
                                images=[[image_input]],
                                image_token_count=len(image_types),
                            ),
                            RequestReservation("image"),
                            worker_context=gpu_worker_context,
                        )
                        return text, image

                    text_result, image_result = asyncio.run(run_cases())
            finally:
                for layer_id, reader in engram_installation.readers.items():
                    reader.lookup = original_lookups[layer_id]
        if text_result is None or image_result is None:
            raise RuntimeError("forward memory baselines were not captured")
        allocated_peak = max(text_result["allocated_peak"], image_result["allocated_peak"])
        reserved_peak = max(text_result["reserved_peak"], image_result["reserved_peak"])
        stats = cache.stats
        if stats.resident_bytes or stats.leased_bytes or stats.loading_bytes or stats.entries:
            raise RuntimeError("forward smoke retained routed expert allocations")
        backbone_layers = {key.layer_id for key in loaded_experts if not key.mtp_layer}
        mtp_layers = {key.layer_id for key in loaded_experts if key.mtp_layer}
        if backbone_layers != set(range(args.n_layers)):
            raise RuntimeError(f"forward missed backbone expert layers: {sorted(backbone_layers)}")
        if mtp_layers != set(range(args.n_mtp_layers)):
            raise RuntimeError(f"forward missed MTP expert layers: {sorted(mtp_layers)}")
        if any(count == 0 for count in engram_lookups.values()):
            raise RuntimeError(f"checkpoint-backed Engram readers were not exercised: {engram_lookups}")
        if not all(image_result["native_usage"].values()):
            raise RuntimeError(
                f"native RMSNorm/RoPE were not exercised: {image_result['native_usage']}"
            )
        print(
            "pinned 1017 checkpoint-backed native HIP text and image forward: PASS; "
            f"text_tokens={text_result['input_token_count']}; "
            f"text_main_token={text_result['main_token_id']}; "
            f"text_dspark_token={text_result['speculative_token_id']}; "
            f"image_tokens={image_result['input_token_count']}; "
            f"image_span_tokens={image_result['image_token_count']}; "
            f"image_grid={n_vit_h}x{n_vit_w} ViT / {n_llm_h}x{n_llm_w} LLM; "
            f"image_main_token={image_result['main_token_id']}; "
            f"image_dspark_token={image_result['speculative_token_id']}; "
            f"expert_cache_misses={len(loaded_experts)}; "
            f"allocated_peak={allocated_peak / _GIB:.3f} GiB; "
            f"allocator_peak={reserved_peak / _GIB:.3f} GiB; "
            f"RMSNorm={image_result['native_usage']['rmsnorm']}; "
            f"RoPE={image_result['native_usage']['rope']}; "
            f"Engram_layer_lookups={engram_lookups}; "
            "routed/MTP experts used checkpoint-backed readers",
            flush=True,
        )
    finally:
        if rmsnorm_class.forward is not native_rmsnorm:
            raise RuntimeError("pinned RMSNorm forward changed during the checkpoint smoke")
        if model_globals.get("apply_rotary_emb") is not native_rope:
            raise RuntimeError("pinned RoPE function changed during the checkpoint smoke")
        rmsnorm_class.forward = original_rmsnorm
        model_globals["apply_rotary_emb"] = original_rope
