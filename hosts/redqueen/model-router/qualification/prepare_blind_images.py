#!/usr/bin/env python3
"""Create blinded A/B image pairs from two workload benchmark result files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

from compare_workloads import read_results, validate_results_against_suite
from workload_benchmark import MAX_IMAGE_BYTES, SCHEMA_VERSION


def result_image(record: dict, repository: Path) -> bytes:
    artifact = record.get("artifact")
    if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
        raise ValueError(f"missing image artifact for {record.get('case_id')}")
    path = Path(artifact["path"]).resolve()
    if path.is_relative_to(repository) or not path.is_file():
        raise ValueError("image artifact is missing or inside the repository")
    if path.stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError(f"image artifact exceeds the size limit: {record.get('case_id')}")
    image = path.read_bytes()
    if (
        not image
        or len(image) > MAX_IMAGE_BYTES
        or not image.startswith(b"\x89PNG\r\n\x1a\n")
        or hashlib.sha256(image).hexdigest() != artifact.get("sha256")
    ):
        raise ValueError(f"image artifact failed integrity validation: {record.get('case_id')}")
    return image


def write_private(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def prepare(reference_path: Path, candidate_path: Path, suite_path: Path,
            output_dir: Path, key_path: Path, *, seed: int) -> int:
    repository = Path(__file__).resolve().parents[4]
    reference, ref_suite, _, ref_suite_id, ref_settings = read_results(reference_path)
    candidate, cand_suite, _, cand_suite_id, cand_settings = read_results(candidate_path)
    if ref_suite != cand_suite:
        raise ValueError("reference and candidate use different frozen suites")
    if ref_suite_id != cand_suite_id:
        raise ValueError("reference and candidate use different suite ids")
    if ref_settings != cand_settings:
        raise ValueError("reference and candidate use different benchmark settings")
    validate_results_against_suite(suite_path, reference, ref_suite, ref_suite_id)
    validate_results_against_suite(suite_path, candidate, cand_suite, cand_suite_id)
    if set(reference) != set(candidate):
        raise ValueError("reference and candidate result keys differ")
    pairs = []
    for key in sorted(reference):
        left, right = reference[key], candidate[key]
        if key[1] != 0 or left.get("workload") != "images" or right.get("workload") != "images":
            continue
        if not left.get("success") or not right.get("success"):
            raise ValueError(f"image request failed for {key}")
        if left.get("path") != right.get("path") or left.get("requested_model") != right.get("requested_model"):
            raise ValueError(f"image route or requested model differs for {key}")
        if left.get("effective_concurrency") != right.get("effective_concurrency"):
            raise ValueError(f"image effective concurrency differs for {key}")
        pairs.append((key, left, right))
    if not pairs:
        raise ValueError("no paired image results found")
    image_pairs = [
        (key, result_image(left, repository), result_image(right, repository))
        for key, left, right in pairs
    ]

    destination = output_dir.resolve()
    key_destination = key_path.resolve()
    if destination.is_relative_to(repository) or key_destination.is_relative_to(repository):
        raise ValueError("blind images and key must be stored outside the repository")
    if key_destination.is_relative_to(destination):
        raise ValueError("blind key must be stored outside the image directory")
    if destination.exists() or key_destination.exists():
        raise FileExistsError("blind image directory and key path must be new")
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.chmod(destination, 0o700)
    key_destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    rng = random.Random(seed)
    key = {}
    template = []
    for (case_id, repeat), reference_image, candidate_image in image_pairs:
        pair_id = hashlib.sha256(
            f"{ref_suite}\0{case_id}\0{repeat}\0{seed}".encode("utf-8")
        ).hexdigest()[:16]
        if rng.getrandbits(1):
            alias_a, alias_b = "candidate", "reference"
            image_a, image_b = candidate_image, reference_image
        else:
            alias_a, alias_b = "reference", "candidate"
            image_a, image_b = reference_image, candidate_image
        write_private(destination / f"{pair_id}-A.png", image_a)
        write_private(destination / f"{pair_id}-B.png", image_b)
        model = reference[(case_id, repeat)]["requested_model"]
        key[pair_id] = {"A": alias_a, "B": alias_b, "model": model}
        template.append({"pair_id": pair_id, "preference": None})
    write_private(
        destination / "ratings-template.jsonl",
        ("\n".join(json.dumps(row, sort_keys=True) for row in template) + "\n").encode(),
    )
    write_private(
        key_destination,
        (json.dumps({
            "schema_version": SCHEMA_VERSION,
            "suite_id": ref_suite_id,
            "suite_sha256": ref_suite,
            "pairs": key,
        }, sort_keys=True, indent=2) + "\n").encode(),
    )
    return len(pairs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--suite", type=Path, required=True, help="exact JSONL suite used for both runs")
    parser.add_argument("--output-dir", type=Path, required=True, help="new private directory for A/B PNGs and ratings-template.jsonl")
    parser.add_argument("--key-output", type=Path, required=True, help="new private blinding key file, outside --output-dir")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        count = prepare(args.reference, args.candidate, args.suite, args.output_dir, args.key_output, seed=args.seed)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"cannot prepare blind image pairs: {error}", file=sys.stderr)
        return 2
    print(f"prepared {count} blind image pairs in {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
