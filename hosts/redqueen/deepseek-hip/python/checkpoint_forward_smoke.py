"""Run a checkpoint-backed 1017 prefill and DSpark step on the native HIP path."""

from __future__ import annotations

import asyncio
import gc
import os
import threading
from contextlib import contextmanager
from types import SimpleNamespace

from cached_model import install_cached_moe
from engram_rows import install_engram_row_reads
from engram_forward import install_chunked_engram_forward
from expert_cache import FP4ExpertCache, FP4ExpertKey
from expert_weights import estimate_fp4_expert_memory, load_fp4_expert
from tiled_indexer import install_tiled_indexer
from tiled_hyper_connections import install_tiled_hyper_connections


_GIB = 1024**3
_FORWARD_PEAK_BUDGET = 8 * _GIB
_EXPERT_CACHE_CAPACITY = 1 * _GIB
_INDEXER_QUERY_TILE_ROWS = 512
_HC_QUERY_TILE_ROWS = 1_024
_BOUNDARY_TOTAL_TOKENS = 220_000
_BOUNDARY_PROMPT_TOKENS = _BOUNDARY_TOTAL_TOKENS - 1
_FP8_GEMM_IMPL_ENV = "DS4_HIP_FP8_GEMM_IMPL"


def _drain_request_state(*, cache, fence, reset_runtime_references, release_allocator):
    """Return request backing while its reservation is still held."""
    fence()
    reset_runtime_references()
    gc.collect()
    released = False

    def release():
        nonlocal released
        release_allocator()
        released = True

    cache.resize_when_idle(0, after_drop=release)
    if not released:
        # Scratch can exist even when no expert was loaded by the request.
        release_allocator()
    fence()


class _SystemMemorySampler:
    """Track host availability and GTT free space while one request is active."""

    def __init__(self, snapshot, *, interval: float = 0.5):
        self._snapshot = snapshot
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started = False
        self._samples = 0
        self._min_free_gtt = None
        self._min_host_available = None
        self._max_sampled_used_gtt = None
        self._gtt_total = None
        self._error = None

    def _sample(self):
        try:
            free_gtt, host_available, gtt_total = self._snapshot()
        except BaseException as exc:
            self._error = exc
            self._stop.set()
            return
        self._samples += 1
        used_gtt = gtt_total - free_gtt
        self._min_free_gtt = (
            free_gtt
            if self._min_free_gtt is None
            else min(self._min_free_gtt, free_gtt)
        )
        self._min_host_available = (
            host_available
            if self._min_host_available is None
            else min(self._min_host_available, host_available)
        )
        self._max_sampled_used_gtt = (
            used_gtt
            if self._max_sampled_used_gtt is None
            else max(self._max_sampled_used_gtt, used_gtt)
        )
        self._gtt_total = gtt_total

    def _run(self):
        while not self._stop.wait(self._interval):
            self._sample()

    def start(self):
        self._sample()
        if self._error is not None:
            raise RuntimeError("could not sample host memory before boundary prefill") from self._error
        self._started = True
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._started:
            self._thread.join()
        self._sample()
        if self._error is not None:
            raise RuntimeError("host memory sampling failed during boundary prefill") from self._error
        if (
            not self._samples
            or self._min_free_gtt is None
            or self._min_host_available is None
            or self._max_sampled_used_gtt is None
            or self._gtt_total is None
        ):
            raise RuntimeError("host memory sampler collected no samples")
        return {
            "samples": self._samples,
            "min_free_gtt": self._min_free_gtt,
            "min_host_available": self._min_host_available,
            "max_sampled_used_gtt": self._max_sampled_used_gtt,
            "gtt_total": self._gtt_total,
        }


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
    system_memory_snapshot,
    boundary_peak_budget,
    boundary_admission_check,
    boundary_memory_check,
    fence,
    release_allocator,
    reset_for_new_request,
    reset_runtime_references,
    include_short_cases: bool = True,
    include_images: bool = True,
    include_boundary: bool = False,
    fp8_gemm_impl: str = "exact",
    compare_optimized_gemm: bool = False,
) -> None:
    """Exercise the pinned graph, including an optional exact 220k-token request.

    Dense weights and the 220k session state must already be resident. Short
    requests reserve an 8 GiB peak; the 220k request reserves the remaining
    capacity under the managed-memory ceilings and required headroom.
    """
    if not include_short_cases and not include_boundary:
        raise ValueError("at least one checkpoint forward case must be enabled")
    if include_images and not include_short_cases:
        raise ValueError("the image case requires the short checkpoint forwards")
    if not callable(reset_for_new_request):
        raise TypeError("checkpoint requests require a session-state reset callback")
    if not callable(reset_runtime_references):
        raise TypeError("checkpoint requests require a runtime-reference reset callback")
    if fp8_gemm_impl not in {"exact", "fp32-grouped-matmul"}:
        raise ValueError(f"unsupported FP8 GEMM profile: {fp8_gemm_impl!r}")
    if compare_optimized_gemm and not include_short_cases:
        raise ValueError("GEMM comparison requires short checkpoint forwards")
    seed_tokens = [
        int(value)
        for value in tokenizer.encode(
            "The quick brown fox jumps over the lazy dog.", add_special_tokens=False
        )
    ]
    if not seed_tokens:
        raise RuntimeError("the pinned tokenizer produced no boundary-prefill seed tokens")
    if any(not 0 <= token_id < args.vocab_size for token_id in seed_tokens):
        raise RuntimeError("the pinned tokenizer produced an invalid boundary-prefill seed token")

    text_token_ids = seed_tokens[:4]
    image_result_metadata = None
    if include_short_cases:
        if len(text_token_ids) < 4:
            raise RuntimeError("the pinned tokenizer produced too few tokens to exercise Engram")
        if include_images:
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
            image_result_metadata = (n_vit_h, n_vit_w, n_llm_h, n_llm_w)
    if include_boundary:
        if not callable(system_memory_snapshot):
            raise TypeError("boundary prefill requires a system memory snapshot callback")
        if (
            not callable(boundary_peak_budget)
            or not callable(boundary_admission_check)
            or not callable(boundary_memory_check)
        ):
            raise TypeError("boundary prefill requires memory-budget admission callbacks")

    model.temperature = 0
    for mtp_layer in model.mtp:
        mtp_layer.temperature = 0
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
    request_active = threading.Event()

    class RequestReservation:
        def __init__(
            self,
            name: str,
            peak_budget: int = _FORWARD_PEAK_BUDGET,
            *,
            admission_check_override=None,
        ):
            self.name = name
            self.peak_budget = peak_budget
            self.admission_check_override = admission_check_override

        async def __aenter__(self):
            if callable(self.peak_budget):
                self.peak_budget = self.peak_budget()
            if not isinstance(self.peak_budget, int) or self.peak_budget <= 0:
                raise RuntimeError("request reservation has no positive peak budget")
            estimate = SimpleNamespace(
                resident_device_bytes=_EXPERT_CACHE_CAPACITY,
                device_peak_bytes=self.peak_budget,
                physical_peak_bytes=self.peak_budget,
            )
            check_admission = self.admission_check_override or admission_check
            free_gtt, host_available, gtt_total = check_admission(estimate)
            torch.cuda.synchronize(device)
            memory[f"{self.name}_allocated_before"] = torch.cuda.memory_allocated(device)
            memory[f"{self.name}_reserved_before"] = torch.cuda.memory_reserved(device)
            torch.cuda.reset_peak_memory_stats(device)
            memory["active_leases"] += 1
            request_active.set()
            print(
                f"single-request {self.name} forward admission: "
                f"GTT peak budget={self.peak_budget / _GIB:.1f} GiB "
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

    def run_one_request(
        name,
        token_ids,
        *,
        token_types=None,
        images=None,
        image_token_count=0,
        run_dspark=True,
        peak_budget=_FORWARD_PEAK_BUDGET,
        capture_logits=False,
    ):
        input_token_count = len(token_ids)
        is_boundary = name == "boundary-220k"
        sampler = _SystemMemorySampler(system_memory_snapshot) if is_boundary else None
        layer_hooks = []
        try:
            # The prior request drops its experts and shrinks the logical cap to
            # zero. Restore the admitted cache budget before this request loads
            # any routed experts.
            cache.resize_when_idle(_EXPERT_CACHE_CAPACITY)
            reset_for_new_request()
            reset_runtime_references()
            if sampler is not None:
                sampler.start()
            if is_boundary:
                for layer_id, layer in enumerate(model.layers):
                    def report_layer(_module, _inputs, _layer_id=layer_id):
                        print(
                            f"220k text prefill layer {_layer_id + 1}/{args.n_layers}",
                            flush=True,
                        )

                    layer_hooks.append(layer.register_forward_pre_hook(report_layer))
            with torch.inference_mode():
                input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
                model_kwargs = {}
                if token_types is not None:
                    model_kwargs["token_types"] = torch.tensor(
                        [token_types], dtype=torch.long, device=device
                    )
                    model_kwargs["images"] = images
                try:
                    output_ids, logits, main_hidden = model(
                        input_ids, start_pos=0, **model_kwargs
                    )
                finally:
                    for handle in layer_hooks:
                        handle.remove()
                    layer_hooks.clear()
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
                if not torch.isfinite(logits).all().item():
                    raise RuntimeError("pinned prefill returned non-finite logits or hidden state")
                if run_dspark:
                    if not torch.isfinite(main_hidden).all().item():
                        raise RuntimeError("pinned prefill returned non-finite DSpark hidden state")
                else:
                    for first in range(0, input_token_count, 4_096):
                        last = min(first + 4_096, input_token_count)
                        if not torch.isfinite(main_hidden[:, first:last]).all().item():
                            raise RuntimeError(
                                "pinned boundary prefill returned non-finite DSpark hidden state"
                            )

                speculative = confidence = spec_ids = spec_logits = None
                if run_dspark:
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
                elif input_token_count + 1 != _BOUNDARY_TOTAL_TOKENS:
                    raise RuntimeError("boundary prefill must sample exactly the 220,000th token")

                # Candidate masks are per-prefill scratch, not session state.
                model_globals["shared_attn"].candidates = None

                result = {
                    "name": name,
                    "input_token_count": input_token_count,
                    "main_token_id": int(output_ids[0].item()),
                    "speculative_token_id": (
                        int(spec_ids[0, -1].item()) if spec_ids is not None else None
                    ),
                    "total_context_tokens": input_token_count + 1,
                    "image_token_count": image_token_count,
                    "native_usage": dict(native_usage),
                }
                if capture_logits:
                    result["logits_cpu"] = logits.detach().float().cpu()
                    result["main_hidden_cpu"] = main_hidden.detach().float().cpu()
                    if spec_logits is not None:
                        result["dspark_logits_cpu"] = spec_logits.detach().float().cpu()
                del main_hidden, logits, output_ids, input_ids
                speculative = confidence = spec_ids = spec_logits = None
                model_kwargs.clear()
            _drain_request_state(
                cache=cache,
                fence=request_fence,
                reset_runtime_references=reset_runtime_references,
                release_allocator=request_release_allocator,
            )
            torch.cuda.synchronize(device)
            if sampler is not None:
                result["system_memory"] = sampler.stop()
                sampler = None
                boundary_memory_check(result["system_memory"])
            result["allocated_peak"] = (
                torch.cuda.max_memory_allocated(device)
                - memory[f"{name}_allocated_before"]
            )
            result["reserved_peak"] = (
                torch.cuda.max_memory_reserved(device)
                - memory[f"{name}_reserved_before"]
            )
            memory_diagnostic = ""
            if is_boundary:
                system_memory = result["system_memory"]
                memory_diagnostic = (
                    f"; min_free_gtt={system_memory['min_free_gtt']} bytes"
                    f"; min_host_available={system_memory['min_host_available']} bytes"
                    f"; max_sampled_used_gtt={system_memory['max_sampled_used_gtt']} bytes"
                )
            if result["allocated_peak"] > peak_budget:
                raise RuntimeError(
                    f"single-request {name} allocated peak {result['allocated_peak']} exceeds its "
                    f"{peak_budget}-byte admission budget{memory_diagnostic}"
                )
            if result["reserved_peak"] > peak_budget:
                raise RuntimeError(
                    f"single-request {name} allocator peak {result['reserved_peak']} exceeds its "
                    f"{peak_budget}-byte admission budget{memory_diagnostic}"
                )
            return result
        except BaseException:
            for handle in layer_hooks:
                handle.remove()
            if sampler is not None:
                try:
                    sampler.stop()
                except BaseException:
                    pass
            model_globals["shared_attn"].candidates = None
            if not cache.stats.quarantined:
                _drain_request_state(
                    cache=cache,
                    fence=request_fence,
                    reset_runtime_references=reset_runtime_references,
                    release_allocator=request_release_allocator,
                )
            raise

    rmsnorm_class.forward = native_rmsnorm
    model_globals["apply_rotary_emb"] = native_rope
    previous_fp8_gemm_impl = os.environ.get(_FP8_GEMM_IMPL_ENV)
    text_result = image_result = boundary_result = None
    try:
        os.environ[_FP8_GEMM_IMPL_ENV] = fp8_gemm_impl
        with install_tiled_indexer(
            reference, query_tile_rows=_INDEXER_QUERY_TILE_ROWS
        ), install_tiled_hyper_connections(
            reference, row_tile_rows=_HC_QUERY_TILE_ROWS
        ), install_engram_row_reads(
            model, checkpoint
        ) as engram_installation, install_chunked_engram_forward(
            model, sequence_tile_rows=256
        ):
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
                        text = image = boundary = None
                        comparisons = {}
                        if include_short_cases:
                            if compare_optimized_gemm:
                                os.environ[_FP8_GEMM_IMPL_ENV] = "exact"
                            text = await installation.run_reserved_sync(
                                lambda: run_one_request(
                                    "text",
                                    text_token_ids,
                                    capture_logits=compare_optimized_gemm,
                                ),
                                RequestReservation("text"),
                                worker_context=gpu_worker_context,
                            )
                            if include_images:
                                image = await installation.run_reserved_sync(
                                    lambda: run_one_request(
                                        "image",
                                        image_token_ids,
                                        token_types=image_token_types,
                                        images=[[image_input]],
                                        image_token_count=len(image_types),
                                        capture_logits=compare_optimized_gemm,
                                    ),
                                    RequestReservation("image"),
                                    worker_context=gpu_worker_context,
                                )
                            if compare_optimized_gemm:
                                os.environ[_FP8_GEMM_IMPL_ENV] = fp8_gemm_impl
                                optimized_name = fp8_gemm_impl
                                text_fast = await installation.run_reserved_sync(
                                    lambda: run_one_request(
                                        f"text-{optimized_name}",
                                        text_token_ids,
                                        capture_logits=True,
                                    ),
                                    RequestReservation(f"text-{optimized_name}"),
                                    worker_context=gpu_worker_context,
                                )
                                image_fast = None
                                if include_images:
                                    image_fast = await installation.run_reserved_sync(
                                        lambda: run_one_request(
                                            f"image-{optimized_name}",
                                            image_token_ids,
                                            token_types=image_token_types,
                                            images=[[image_input]],
                                            image_token_count=len(image_types),
                                            capture_logits=True,
                                        ),
                                        RequestReservation(f"image-{optimized_name}"),
                                        worker_context=gpu_worker_context,
                                    )

                                def compare_logits(name, exact, optimized):
                                    tensor_results = {}
                                    for label, key in (
                                        ("main_logits", "logits_cpu"),
                                        ("main_hidden", "main_hidden_cpu"),
                                        ("dspark_logits", "dspark_logits_cpu"),
                                    ):
                                        exact_tensor = exact[key]
                                        optimized_tensor = optimized[key]
                                        if exact_tensor.shape != optimized_tensor.shape:
                                            raise RuntimeError(
                                                f"{name} {label} tensors have different shapes"
                                            )
                                        differences = (exact_tensor - optimized_tensor).abs()
                                        relative = differences / exact_tensor.abs().clamp_min(1.0)
                                        tensor_results[label] = {
                                            "max_abs": float(differences.max().item()),
                                            "mean_abs": float(differences.mean().item()),
                                            "p99_relative": float(
                                                torch.quantile(relative.flatten(), 0.99).item()
                                            ),
                                            "bitwise_match": torch.equal(
                                                exact_tensor, optimized_tensor
                                            ),
                                        }
                                    return {
                                        **tensor_results,
                                        "main_token_match": exact["main_token_id"]
                                        == optimized["main_token_id"],
                                        "dspark_token_match": exact["speculative_token_id"]
                                        == optimized["speculative_token_id"],
                                    }

                                comparisons = {"text": compare_logits("text", text, text_fast)}
                                if include_images:
                                    comparisons["image"] = compare_logits(
                                        "image", image, image_fast
                                    )
                                failures = {
                                    name: result
                                    for name, result in comparisons.items()
                                    if not result["main_token_match"]
                                    or not result["dspark_token_match"]
                                    or not result["main_logits"]["bitwise_match"]
                                    or not result["main_hidden"]["bitwise_match"]
                                    or not result["dspark_logits"]["bitwise_match"]
                                }
                                if failures:
                                    raise RuntimeError(
                                        f"optimized FP8 GEMM failed exact output parity: {failures}"
                                    )
                                del text_fast, image_fast
                                os.environ[_FP8_GEMM_IMPL_ENV] = fp8_gemm_impl
                        if include_boundary:
                            repeat_count = (
                                _BOUNDARY_PROMPT_TOKENS + len(seed_tokens) - 1
                            ) // len(seed_tokens)
                            boundary_tokens = (
                                seed_tokens * repeat_count
                            )[:_BOUNDARY_PROMPT_TOKENS]
                            if len(boundary_tokens) != _BOUNDARY_PROMPT_TOKENS:
                                raise RuntimeError("failed to construct the 220k boundary prompt")
                            print(
                                "starting exact-context boundary request: "
                                f"prompt={len(boundary_tokens)} tokens; one sampled output; "
                                f"total={len(boundary_tokens) + 1}",
                                flush=True,
                            )
                            boundary_reservation = RequestReservation(
                                "boundary-220k",
                                boundary_peak_budget,
                                admission_check_override=boundary_admission_check,
                            )
                            boundary = await installation.run_reserved_sync(
                                lambda: run_one_request(
                                    "boundary-220k",
                                    boundary_tokens,
                                    run_dspark=False,
                                    peak_budget=boundary_reservation.peak_budget,
                                ),
                                boundary_reservation,
                                worker_context=gpu_worker_context,
                            )
                        return text, image, boundary, comparisons

                    (
                        text_result,
                        image_result,
                        boundary_result,
                        fp8_comparisons,
                    ) = asyncio.run(run_cases())
            finally:
                for layer_id, reader in engram_installation.readers.items():
                    reader.lookup = original_lookups[layer_id]
        request_results = [
            result
            for result in (text_result, image_result, boundary_result)
            if result is not None
        ]
        if not request_results:
            raise RuntimeError("forward memory baselines were not captured")
        allocated_peak = max(result["allocated_peak"] for result in request_results)
        reserved_peak = max(result["reserved_peak"] for result in request_results)
        stats = cache.stats
        if stats.resident_bytes or stats.leased_bytes or stats.loading_bytes or stats.entries:
            raise RuntimeError("forward smoke retained routed expert allocations")
        backbone_layers = {key.layer_id for key in loaded_experts if not key.mtp_layer}
        mtp_layers = {key.layer_id for key in loaded_experts if key.mtp_layer}
        if backbone_layers != set(range(args.n_layers)):
            raise RuntimeError(f"forward missed backbone expert layers: {sorted(backbone_layers)}")
        expected_mtp_layers = set(range(args.n_mtp_layers)) if include_short_cases else set()
        if mtp_layers != expected_mtp_layers:
            raise RuntimeError(f"forward missed MTP expert layers: {sorted(mtp_layers)}")
        if any(count == 0 for count in engram_lookups.values()):
            raise RuntimeError(f"checkpoint-backed Engram readers were not exercised: {engram_lookups}")
        native_result = boundary_result or image_result or text_result
        if not any(native_result["native_usage"].values()):
            raise RuntimeError(
                f"native RMSNorm/RoPE were not exercised: {native_result['native_usage']}"
            )
        if include_short_cases and image_result is not None:
            n_vit_h, n_vit_w, n_llm_h, n_llm_w = image_result_metadata
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
        elif include_short_cases:
            print(
                "pinned 1017 checkpoint-backed native HIP text forward: PASS; "
                f"text_tokens={text_result['input_token_count']}; "
                f"text_main_token={text_result['main_token_id']}; "
                f"text_dspark_token={text_result['speculative_token_id']}; "
                f"expert_cache_misses={len(loaded_experts)}; "
                f"allocated_peak={allocated_peak / _GIB:.3f} GiB; "
                f"allocator_peak={reserved_peak / _GIB:.3f} GiB; "
                f"RMSNorm={text_result['native_usage']['rmsnorm']}; "
                f"RoPE={text_result['native_usage']['rope']}; "
                f"Engram_layer_lookups={engram_lookups}; "
                "routed/MTP experts used checkpoint-backed readers",
                flush=True,
            )
        if compare_optimized_gemm:
            print(
                f"{fp8_gemm_impl} rocBLAS FP8-input profile vs exact native HIP FP8: "
                f"text={fp8_comparisons['text']}; "
                f"image={fp8_comparisons.get('image', 'skipped')}",
                flush=True,
            )
        if boundary_result is not None:
            system_memory = boundary_result["system_memory"]
            print(
                "pinned 1017 native HIP 220k-context prefill: PASS; "
                f"prompt_tokens={boundary_result['input_token_count']}; "
                f"sampled_output_tokens=1; total_context_tokens={boundary_result['total_context_tokens']}; "
                f"main_token={boundary_result['main_token_id']}; "
                f"allocated_peak={boundary_result['allocated_peak'] / _GIB:.3f} GiB; "
                f"allocator_peak={boundary_result['reserved_peak'] / _GIB:.3f} GiB; "
                f"min_free_gtt={system_memory['min_free_gtt'] / _GIB:.2f} GiB; "
                f"min_host_available={system_memory['min_host_available'] / _GIB:.2f} GiB; "
                f"max_sampled_used_gtt={system_memory['max_sampled_used_gtt'] / _GIB:.2f} GiB; "
                f"system_samples={system_memory['samples']}; "
                f"expert_cache_misses={len(loaded_experts)}; "
                f"RMSNorm={boundary_result['native_usage']['rmsnorm']}; "
                f"RoPE={boundary_result['native_usage']['rope']}; "
                f"Engram_layer_lookups={engram_lookups}",
                flush=True,
            )
    finally:
        if previous_fp8_gemm_impl is None:
            os.environ.pop(_FP8_GEMM_IMPL_ENV, None)
        else:
            os.environ[_FP8_GEMM_IMPL_ENV] = previous_fp8_gemm_impl
        if rmsnorm_class.forward is not native_rmsnorm:
            raise RuntimeError("pinned RMSNorm forward changed during the checkpoint smoke")
        if model_globals.get("apply_rotary_emb") is not native_rope:
            raise RuntimeError("pinned RoPE function changed during the checkpoint smoke")
        rmsnorm_class.forward = original_rmsnorm
        model_globals["apply_rotary_emb"] = original_rope
