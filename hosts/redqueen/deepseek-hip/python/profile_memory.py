"""Print allocation-free resident memory estimates for the pinned 1017 graph."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from checkpoint_mapping_audit import (
    _execute_pinned_model,
    _pinned_inference_imports,
    verify_pinned_inference_sources,
)
from checkpoint_reader import SafeTensorCheckpoint
from dense_checkpoint import plan_dense_checkpoint_load
from session_state import plan_session_state


def profile(checkpoint_root: Path, inference_dir: Path) -> dict[str, int]:
    adapter_dir = Path(__file__).resolve().parent
    source_bytes = verify_pinned_inference_sources(inference_dir)
    previous_path = sys.path[:]
    previous_kernel = sys.modules.pop("kernel", None)
    sys.path[:0] = [str(adapter_dir), str(inference_dir), str(inference_dir.parent / "encoding")]
    try:
        import kernel
        import torch
        from transformers import AutoTokenizer

        sys.modules["kernel"] = kernel
        with _pinned_inference_imports(source_bytes, inference_dir):
            reference = _execute_pinned_model(source_bytes["model.py"], inference_dir / "model.py")
            config = json.loads(source_bytes["config.json"].decode("utf-8"))
            if not isinstance(config, dict):
                raise RuntimeError("pinned model config must be an object")
            previous_dtype = torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.bfloat16)
                args = reference["ModelArgs"](**config)
                args.max_batch_size = 1
                args.max_seq_len = 220000
                tokenizer = AutoTokenizer.from_pretrained(
                    str(checkpoint_root), local_files_only=True
                )
                with torch.device("meta"):
                    model = reference["Transformer"](args, tokenizer)
            finally:
                torch.set_default_dtype(previous_dtype)

            with SafeTensorCheckpoint(
                checkpoint_root,
                expected_revision=kernel.REFERENCE_COMMIT,
            ) as checkpoint:
                dense = plan_dense_checkpoint_load(checkpoint, model)
                session = plan_session_state(model, args).estimate
                resident = dense.estimate.resident_device_bytes + session.resident_device_bytes
                if resident <= 0:
                    raise RuntimeError("pinned model returned an empty resident memory estimate")
                return {
                    "resident_device_bytes": resident,
                    "resident_host_bytes": resident,
                }
    finally:
        sys.path[:] = previous_path
        sys.modules.pop("kernel", None)
        if previous_kernel is not None:
            sys.modules["kernel"] = previous_kernel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--inference-dir", type=Path, required=True)
    args = parser.parse_args()
    result = profile(args.checkpoint_root, args.inference_dir)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
