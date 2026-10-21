#!/usr/bin/env python3
"""Paired bootstrap analysis for workload_benchmark.py result files."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from pathlib import Path

from workload_benchmark import ROUTES, SCHEMA_VERSION, read_suite, strict_json_loads


MIN_PAIRED_CASES = 8


def read_results(path: Path) -> tuple[dict[tuple[str, int], dict], str, str, str, dict]:
    records: dict[tuple[str, int], dict] = {}
    suite_hash = None
    label = None
    suite_id = None
    settings = None
    effective_by_model: dict[str, int] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = strict_json_loads(line)
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from error
        if (
            not isinstance(record, dict)
            or isinstance(record.get("schema_version"), bool)
            or record.get("schema_version") != SCHEMA_VERSION
        ):
            raise ValueError(f"{path}:{line_number}: unsupported result schema")
        case_id = record.get("case_id")
        repeat = record.get("repeat")
        if (
            not isinstance(case_id, str)
            or isinstance(repeat, bool)
            or not isinstance(repeat, int)
            or repeat < 0
        ):
            raise ValueError(f"{path}:{line_number}: invalid case_id/repeat")
        key = (case_id, repeat)
        if key in records:
            raise ValueError(f"{path}:{line_number}: duplicate result {key}")
        digest = record.get("suite_sha256")
        current_label = record.get("label")
        current_suite_id = record.get("suite_id")
        if (
            not isinstance(digest, str)
            or not re.fullmatch(r"[a-f0-9]{64}", digest)
            or not isinstance(current_label, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", current_label)
            or not isinstance(current_suite_id, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", current_suite_id)
        ):
            raise ValueError(f"{path}:{line_number}: missing suite hash, label, or suite id")
        if suite_hash is not None and digest != suite_hash:
            raise ValueError(f"{path}:{line_number}: mixed suite hashes")
        if label is not None and current_label != label:
            raise ValueError(f"{path}:{line_number}: mixed labels")
        if suite_id is not None and current_suite_id != suite_id:
            raise ValueError(f"{path}:{line_number}: mixed suite ids")
        suite_hash, label, suite_id = digest, current_label, current_suite_id
        latency = record.get("request_elapsed_s")
        if isinstance(latency, bool) or not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency < 0:
            raise ValueError(f"{path}:{line_number}: invalid request latency")
        workload = record.get("workload")
        route = record.get("path")
        requested_model = record.get("requested_model")
        if (
            not isinstance(workload, str)
            or workload not in ROUTES
            or not isinstance(route, str)
            or route not in ROUTES[workload]
            or not isinstance(requested_model, str)
            or not requested_model
        ):
            raise ValueError(f"{path}:{line_number}: unsupported workload, route, or requested model")
        success = record.get("success")
        status_code = record.get("status_code")
        error_type = record.get("error_type")
        if (
            not isinstance(success, bool)
            or (status_code is not None and (
                isinstance(status_code, bool) or not isinstance(status_code, int) or not 100 <= status_code <= 599
            ))
            or (error_type is not None and not isinstance(error_type, str))
            or success != (
                status_code is not None and 200 <= status_code < 300 and error_type is None
            )
        ):
            raise ValueError(f"{path}:{line_number}: inconsistent HTTP status, error, or success fields")
        effective_concurrency = record.get("effective_concurrency")
        if (
            isinstance(effective_concurrency, bool)
            or not isinstance(effective_concurrency, int)
            or not 1 <= effective_concurrency <= 4
        ):
            raise ValueError(f"{path}:{line_number}: invalid effective concurrency")
        for metric in ("response_headers_s", "ttft_s"):
            value = record.get(metric)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0
            ):
                raise ValueError(f"{path}:{line_number}: invalid {metric}")
        timings = record.get("server_timings")
        if timings is not None and (
            not isinstance(timings, dict)
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0
                for value in timings.values()
            )
        ):
            raise ValueError(f"{path}:{line_number}: invalid server timings")
        current_settings = {
            "order_seed": record.get("order_seed"),
            "concurrency": record.get("concurrency"),
            "warmups": record.get("warmups"),
            "timeout_s": record.get("timeout_s"),
        }
        if (
            isinstance(current_settings["order_seed"], bool)
            or not isinstance(current_settings["order_seed"], int)
            or isinstance(current_settings["concurrency"], bool)
            or not isinstance(current_settings["concurrency"], int)
            or not 1 <= current_settings["concurrency"] <= 4
            or isinstance(current_settings["warmups"], bool)
            or not isinstance(current_settings["warmups"], int)
            or current_settings["warmups"] < 0
            or isinstance(current_settings["timeout_s"], bool)
            or not isinstance(current_settings["timeout_s"], (int, float))
            or not math.isfinite(current_settings["timeout_s"])
            or current_settings["timeout_s"] <= 0
        ):
            raise ValueError(f"{path}:{line_number}: invalid benchmark settings")
        if settings is not None and current_settings != settings:
            raise ValueError(f"{path}:{line_number}: mixed benchmark settings")
        if effective_concurrency > current_settings["concurrency"]:
            raise ValueError(f"{path}:{line_number}: effective concurrency exceeds the requested limit")
        prior_effective = effective_by_model.setdefault(requested_model, effective_concurrency)
        if prior_effective != effective_concurrency:
            raise ValueError(f"{path}:{line_number}: mixed effective concurrency for {requested_model}")
        settings = current_settings
        records[key] = record
    if not records or suite_hash is None or label is None or suite_id is None or settings is None:
        raise ValueError(f"{path}: no benchmark results")
    repeats_by_case: dict[str, set[int]] = {}
    for case_id, repeat in records:
        repeats_by_case.setdefault(case_id, set()).add(repeat)
    expected_repeats = set(range(max(max(values) for values in repeats_by_case.values()) + 1))
    if any(values != expected_repeats for values in repeats_by_case.values()):
        raise ValueError(f"{path}: repeats are incomplete or inconsistent across cases")
    return records, suite_hash, label, suite_id, settings


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires observations")
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def bootstrap_median_ci(values: list[float], *, iterations: int, seed: int) -> list[float] | None:
    if len(values) < MIN_PAIRED_CASES:
        return None
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        sample = [values[rng.randrange(len(values))] for _ in values]
        samples.append(percentile(sample, 0.5))
    return [percentile(samples, 0.025), percentile(samples, 0.975)]


def numeric_quality(record: dict) -> tuple[str, float] | None:
    quality = record.get("quality")
    if not isinstance(quality, dict):
        return None
    metric, value = quality.get("metric"), quality.get("value")
    if not isinstance(metric, str) or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return metric, float(value)


def server_metric(record: dict, metric: str) -> float | None:
    if metric == "ttft_s":
        value = record.get(metric)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return None
        return float(value)
    timings = record.get("server_timings")
    if not isinstance(timings, dict):
        return None
    value = timings.get(metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def median_record_metric(records: list[dict], metric: str) -> float | None:
    values = [value for record in records if (value := server_metric(record, metric)) is not None]
    return percentile(values, 0.5) if values else None


def median_throughput(records: list[dict], count_metric: str, time_metric: str) -> float | None:
    values = []
    for record in records:
        count = server_metric(record, count_metric)
        elapsed_ms = server_metric(record, time_metric)
        if count is not None and elapsed_ms is not None and count > 0 and elapsed_ms > 0:
            values.append(count * 1000 / elapsed_ms)
    return percentile(values, 0.5) if values else None


def summarize_workload(
    workload: str,
    pairs: list[tuple[tuple[str, int], dict, dict]],
    *,
    iterations: int,
    seed: int,
) -> dict:
    by_case: dict[str, list[tuple[dict, dict]]] = {}
    failures = 0
    reference_records = []
    candidate_records = []
    for key, reference, candidate in pairs:
        if not reference.get("success") or not candidate.get("success"):
            failures += 1
        by_case.setdefault(key[0], []).append((reference, candidate))
        reference_records.append(reference)
        candidate_records.append(candidate)

    latency_improvements = []
    reference_latency = []
    candidate_latency = []
    quality_differences = []
    quality_metrics: set[str] = set()
    quality_cases = 0
    for case_id in sorted(by_case):
        observations = by_case[case_id]
        reference_case = [float(left["request_elapsed_s"]) for left, _ in observations]
        candidate_case = [float(right["request_elapsed_s"]) for _, right in observations]
        median_reference = percentile(reference_case, 0.5)
        median_candidate = percentile(candidate_case, 0.5)
        reference_latency.append(median_reference)
        candidate_latency.append(median_candidate)
        if median_reference > 0:
            latency_improvements.append(1 - median_candidate / median_reference)

        diffs = []
        metrics = set()
        for left, right in observations:
            left_score, right_score = numeric_quality(left), numeric_quality(right)
            if left_score is None or right_score is None or left_score[0] != right_score[0]:
                continue
            metrics.add(left_score[0])
            diffs.append(right_score[1] - left_score[1])
        if len(metrics) == 1 and diffs:
            quality_metrics.update(metrics)
            quality_differences.append(percentile(diffs, 0.5))
            quality_cases += 1

    latency_median = percentile(latency_improvements, 0.5) if latency_improvements else None
    latency_ci = bootstrap_median_ci(latency_improvements, iterations=iterations, seed=seed)
    if failures:
        latency_status = "failures"
    elif latency_ci is None:
        latency_status = "insufficient_pairs"
    elif latency_ci[0] >= 0.20:
        latency_status = "pass"
    elif latency_ci[1] < 0.20:
        latency_status = "fail"
    else:
        latency_status = "inconclusive"

    quality_ci = bootstrap_median_ci(quality_differences, iterations=iterations, seed=seed + 1)
    if not quality_differences:
        quality_status = "not_scored"
    elif quality_cases != len(by_case) or len(quality_metrics) != 1:
        quality_status = "incomplete_pairs"
    elif quality_ci is None:
        quality_status = "insufficient_pairs"
    elif quality_ci[1] < 0:
        quality_status = "regression"
    else:
        quality_status = "no_supported_regression"

    return {
        "paired_cases": len(by_case),
        "paired_requests": len(pairs),
        "effective_concurrency": reference_records[0]["effective_concurrency"],
        "request_failures": failures,
        "reference_median_latency_s": percentile(reference_latency, 0.5),
        "candidate_median_latency_s": percentile(candidate_latency, 0.5),
        "reference_median_ttft_s": median_record_metric(reference_records, "ttft_s"),
        "candidate_median_ttft_s": median_record_metric(candidate_records, "ttft_s"),
        "reference_median_server_prefill_ms": median_record_metric(reference_records, "prompt_ms"),
        "candidate_median_server_prefill_ms": median_record_metric(candidate_records, "prompt_ms"),
        "reference_median_server_prefill_tokens_per_s": median_throughput(reference_records, "prompt_n", "prompt_ms"),
        "candidate_median_server_prefill_tokens_per_s": median_throughput(candidate_records, "prompt_n", "prompt_ms"),
        "reference_median_server_decode_tokens_per_s": median_throughput(reference_records, "predicted_n", "predicted_ms"),
        "candidate_median_server_decode_tokens_per_s": median_throughput(candidate_records, "predicted_n", "predicted_ms"),
        "paired_median_latency_improvement": latency_median,
        "latency_improvement_ci95": latency_ci,
        "latency_gate": latency_status,
        "quality_metric": next(iter(quality_metrics)) if len(quality_metrics) == 1 else None,
        "quality_scored_cases": quality_cases,
        "paired_median_quality_delta": percentile(quality_differences, 0.5) if quality_differences else None,
        "quality_delta_ci95": quality_ci,
        "quality_gate": quality_status,
    }


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float] | None:
    if trials == 0:
        return None
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def read_image_ratings(
    ratings_path: Path, key_path: Path, expected_suite: str, expected_suite_id: str
) -> dict:
    key_document = strict_json_loads(key_path.read_text(encoding="utf-8"))
    if (
        not isinstance(key_document, dict)
        or isinstance(key_document.get("schema_version"), bool)
        or key_document.get("schema_version") != SCHEMA_VERSION
        or key_document.get("suite_sha256") != expected_suite
        or key_document.get("suite_id") != expected_suite_id
        or not isinstance(key_document.get("pairs"), dict)
    ):
        raise ValueError("image key does not match the frozen benchmark suite")
    key = key_document["pairs"]
    if not isinstance(key, dict) or not key:
        raise ValueError("image key must be a non-empty object")
    ratings: dict[str, dict[str, str]] = {}
    rated_pair_ids = set()
    for line_number, line in enumerate(ratings_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = strict_json_loads(line)
        except ValueError as error:
            raise ValueError(f"{ratings_path}:{line_number}: invalid JSON") from error
        if not isinstance(record, dict) or not isinstance(record.get("pair_id"), str):
            raise ValueError(f"{ratings_path}:{line_number}: invalid rating record")
        pair_id = record["pair_id"]
        preference = record.get("preference")
        if (
            pair_id in rated_pair_ids
            or pair_id not in key
            or not isinstance(preference, str)
            or preference not in {"A", "B", "tie"}
        ):
            raise ValueError(f"{ratings_path}:{line_number}: duplicate, unknown, or invalid image preference")
        pair_key = key[pair_id]
        if (
            not isinstance(pair_key, dict)
            or set(pair_key) != {"A", "B", "model"}
            or set(value for name, value in pair_key.items() if name in {"A", "B"}) != {"reference", "candidate"}
            or not isinstance(pair_key.get("model"), str)
            or not pair_key["model"]
        ):
            raise ValueError(f"{key_path}: invalid blind key for {pair_id}")
        winner = "tie" if preference == "tie" else pair_key[preference]
        ratings.setdefault(pair_key["model"], {})[pair_id] = winner
        rated_pair_ids.add(pair_id)
    if rated_pair_ids != set(key):
        raise ValueError("image ratings must contain exactly one preference for every blinded pair")
    return ratings


def image_preference_summary(ratings: dict[str, str]) -> dict:
    candidate_wins = sum(value == "candidate" for value in ratings.values())
    reference_wins = sum(value == "reference" for value in ratings.values())
    ties = sum(value == "tie" for value in ratings.values())
    decisive = candidate_wins + reference_wins
    ci = wilson_interval(candidate_wins, decisive)
    if len(ratings) < 20:
        status = "insufficient_pairs"
    elif ci is None:
        status = "no_supported_regression"
    elif ci[1] < 0.5:
        status = "regression"
    else:
        status = "no_supported_regression"
    return {
        "rated_pairs": len(ratings),
        "candidate_wins": candidate_wins,
        "reference_wins": reference_wins,
        "ties": ties,
        "candidate_share_of_decisive_preferences": candidate_wins / decisive if decisive else None,
        "candidate_preference_ci95": ci,
        "quality_gate": status,
    }


def validate_results_against_suite(
    suite_path: Path, records: dict[tuple[str, int], dict], expected_hash: str, expected_id: str
) -> None:
    cases, actual_hash = read_suite(suite_path)
    suite_id = cases[0]["suite_id"]
    if actual_hash != expected_hash or suite_id != expected_id:
        raise ValueError("result files do not match the supplied frozen suite")
    expected = {
        case["id"]: (case["workload"], case["path"], case["request"].get("model"))
        for case in cases
    }
    actual = {}
    for (case_id, _), record in records.items():
        identity = (record["workload"], record["path"], record["requested_model"])
        if case_id in actual and actual[case_id] != identity:
            raise ValueError(f"result route or requested model changed between repeats for {case_id}")
        actual[case_id] = identity
    if set(actual) != set(expected):
        raise ValueError("results do not contain exactly one complete run of every suite case")
    for case_id, identity in expected.items():
        if actual[case_id] != identity:
            raise ValueError(f"result route or requested model does not match the frozen suite for {case_id}")


def analyze(reference_path: Path, candidate_path: Path, *, suite_path: Path, iterations: int, seed: int,
            image_ratings: Path | None = None, image_key: Path | None = None) -> dict:
    reference, ref_suite, ref_label, ref_suite_id, ref_settings = read_results(reference_path)
    candidate, cand_suite, cand_label, cand_suite_id, cand_settings = read_results(candidate_path)
    if ref_suite != cand_suite:
        raise ValueError("reference and candidate use different frozen suites")
    if ref_suite_id != cand_suite_id:
        raise ValueError("reference and candidate use different suite ids")
    if ref_settings != cand_settings:
        raise ValueError("reference and candidate use different benchmark settings")
    validate_results_against_suite(suite_path, reference, ref_suite, ref_suite_id)
    validate_results_against_suite(suite_path, candidate, cand_suite, cand_suite_id)
    if set(reference) != set(candidate):
        missing = len(set(reference) - set(candidate))
        extra = len(set(candidate) - set(reference))
        raise ValueError(f"paired result keys differ: {missing} missing, {extra} extra")
    pairs_by_route: dict[tuple[str, str, str], list[tuple[tuple[str, int], dict, dict]]] = {}
    for key in sorted(reference):
        left, right = reference[key], candidate[key]
        if (
            left.get("workload") != right.get("workload")
            or left.get("path") != right.get("path")
            or left.get("requested_model") != right.get("requested_model")
            or left.get("effective_concurrency") != right.get("effective_concurrency")
        ):
            raise ValueError(f"paired result route or concurrency mismatch for {key}")
        route_key = (left["workload"], left["requested_model"], left["path"])
        pairs_by_route.setdefault(route_key, []).append((key, left, right))
    workloads = {}
    for index, ((workload, model, route), pairs) in enumerate(sorted(pairs_by_route.items())):
        workloads.setdefault(workload, {}).setdefault("models", {}).setdefault(model, {}).setdefault("routes", {})[
            route
        ] = summarize_workload(
            workload, pairs, iterations=iterations, seed=seed + index * 101
        )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "suite_sha256": ref_suite,
        "suite_id": ref_suite_id,
        "reference_label": ref_label,
        "candidate_label": cand_label,
        "bootstrap_iterations": iterations,
        "seed": seed,
        "benchmark_settings": ref_settings,
        "workloads": workloads,
    }
    if image_ratings is not None or image_key is not None:
        if image_ratings is None or image_key is None:
            raise ValueError("--image-ratings and --image-key must be used together")
        if "images" not in summary["workloads"]:
            raise ValueError("image ratings were provided but the suite contains no images")
        summary["workloads"].setdefault("images", {})
        ratings_by_model = read_image_ratings(image_ratings, image_key, ref_suite, ref_suite_id)
        image_models = summary["workloads"]["images"]["models"]
        if set(ratings_by_model) != set(image_models):
            raise ValueError("image ratings do not cover exactly the models in the image suite")
        for model, ratings in ratings_by_model.items():
            image_models[model]["image_preference"] = image_preference_summary(ratings)
            image_models[model]["quality_gate"] = image_models[model]["image_preference"]["quality_gate"]
    elif "images" in summary["workloads"]:
        for model_result in summary["workloads"]["images"]["models"].values():
            model_result["image_preference"] = None
            model_result["quality_gate"] = "needs_blinded_ratings"
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--suite", type=Path, required=True, help="exact JSONL suite used for both runs")
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-ratings", type=Path)
    parser.add_argument("--image-key", type=Path)
    parser.add_argument("--output", type=Path, help="new JSON report path; stdout is the default")
    args = parser.parse_args()
    if args.bootstrap_iterations < 100:
        parser.error("bootstrap-iterations must be at least 100")
    try:
        summary = analyze(
            args.reference, args.candidate, suite_path=args.suite, iterations=args.bootstrap_iterations,
            seed=args.seed, image_ratings=args.image_ratings, image_key=args.image_key,
        )
        output = json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        if args.output:
            repository = Path(__file__).resolve().parents[4]
            destination = args.output.resolve()
            if destination.is_relative_to(repository):
                raise ValueError("reports must be written outside the repository")
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(output)
                stream.flush()
                os.fsync(stream.fileno())
        else:
            sys.stdout.write(output)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"cannot compare benchmark results: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
