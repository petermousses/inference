"""Run the pinned reference model's small built-in graph smoke test."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


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
    model_globals = reference["Transformer"].forward.__globals__
    rmsnorm_class = model_globals["RMSNorm"]

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
    args = reference["ModelArgs"](dspark_block_size=6, dspark_target_layer_ids=(3, 4))
    tokens = torch.randint(0, args.vocab_size, (2, 150))
    model = reference["Transformer"](args)
    output_ids, _, main_hidden = model(tokens[:, :128])
    model.forward_spec(output_ids, main_hidden)
    for position in range(128, 150):
        output_ids, _, main_hidden = model(tokens[:, position : position + 1], position)
        result = model.forward_spec(output_ids, main_hidden, position)
        if result is None:
            raise RuntimeError("pinned graph did not return a DSpark step")
        output_ids, _, _ = result
    torch.cuda.synchronize()
    if not usage["rmsnorm"] or not usage["rope"]:
        raise RuntimeError(f"native graph overrides were not exercised: {usage}")
    print(f"native reference graph passed: RMSNorm={usage['rmsnorm']}, RoPE={usage['rope']}")


if __name__ == "__main__":
    main()
