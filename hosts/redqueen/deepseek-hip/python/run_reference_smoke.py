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
    # Import our kernel module before exposing the reference directory. The
    # reference's own kernel.py imports TileLang and is intentionally bypassed.
    sys.path[:0] = [str(adapter_dir), str(inference_dir)]
    import kernel  # noqa: PLC0415

    sys.modules["kernel"] = kernel
    sys.argv[0] = str(model_path)
    runpy.run_path(str(model_path), run_name="__main__")


if __name__ == "__main__":
    main()
