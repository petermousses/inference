#!/usr/bin/env python3
"""Compare resident and streamed probe logits bit-for-bit."""

from __future__ import annotations

import argparse
import math
import struct
import sys
from pathlib import Path


HEADER = struct.Struct("=III")
MAGIC = 0x4D4F4531


def read_logits(path: Path) -> tuple[int, int, bytes, tuple[float, ...]]:
    data = path.read_bytes()
    if len(data) < HEADER.size:
        raise ValueError(f"{path}: truncated header")
    magic, vocab, steps = HEADER.unpack_from(data)
    if magic != MAGIC or vocab == 0 or steps == 0:
        raise ValueError(f"{path}: invalid probe header")
    expected = HEADER.size + vocab * steps * 4
    if len(data) != expected:
        raise ValueError(f"{path}: expected {expected} bytes, found {len(data)}")
    payload = data[HEADER.size:]
    values = struct.unpack(f"={vocab * steps}f", payload)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{path}: logits contain NaN or infinity")
    return vocab, steps, payload, values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("resident", type=Path)
    parser.add_argument("streamed", type=Path)
    args = parser.parse_args()
    try:
        resident_vocab, resident_steps, resident_bytes, resident = read_logits(args.resident)
        stream_vocab, stream_steps, stream_bytes, streamed = read_logits(args.streamed)
        if (resident_vocab, resident_steps) != (stream_vocab, stream_steps):
            raise ValueError("resident and streamed files have different shapes")
        if resident_bytes != stream_bytes:
            differing_byte = next(
                i for i, pair in enumerate(zip(resident_bytes, stream_bytes)) if pair[0] != pair[1]
            )
            index = differing_byte // 4
            step, token = divmod(index, resident_vocab)
            max_abs = max(abs(a - b) for a, b in zip(resident, streamed))
            raise ValueError(
                f"logits differ at step={step} token={token}; "
                f"resident={resident[index]:.9g} streamed={streamed[index]:.9g}; "
                f"max_abs={max_abs:.9g}"
            )
        print(
            f"exact logits parity: {resident_steps} steps, vocab={resident_vocab}, "
            f"values={len(resident)}"
        )
        return 0
    except (OSError, ValueError, struct.error) as error:
        print(f"parity check failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
