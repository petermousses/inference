from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


QUALIFICATION = Path(__file__).resolve().parents[1] / "qualification"
sys.path.insert(0, str(QUALIFICATION))

import compare_workloads
import prepare_blind_images
import make_workload_suite
import workload_benchmark


class WorkloadBenchmarkTest(unittest.TestCase):
    def suite_file(self, root: Path, rows: list[dict]) -> Path:
        path = root / "suite.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    def case(self, *, case_id: str = "text-1", workload: str = "text",
             route: str = "/v1/chat/completions", scoring: dict | None = None) -> dict:
        return {
            "schema_version": 1,
            "suite_id": "test-suite-v1",
            "id": case_id,
            "workload": workload,
            "path": route,
            "request": {"model": "test-model", "messages": [{"role": "user", "content": "2 + 2?"}]},
            "scoring": scoring or {"kind": "exact", "pointer": "/text_output", "expected": "4"},
        }

    def test_suite_is_versioned_unique_and_route_scoped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [self.case(), self.case(case_id="embed-1", workload="embeddings", route="/v1/embeddings",
                                            scoring={"kind": "retrieval", "relevant": [0]})]
            path = self.suite_file(root, rows)
            cases, digest = workload_benchmark.read_suite(path)
            self.assertEqual(len(cases), 2)
            self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
            with self.assertRaisesRegex(workload_benchmark.SuiteError, "unsupported workload/path"):
                self.suite_file(root, [self.case(route="/v1/admin")])
                workload_benchmark.read_suite(path.with_name("suite.jsonl"))

    def test_workloads_select_the_redqueen_front_end_family(self):
        self.assertEqual(workload_benchmark.endpoint_name("text"), "model")
        self.assertEqual(workload_benchmark.endpoint_name("prefill"), "model")
        self.assertEqual(workload_benchmark.endpoint_name("embeddings"), "model")
        self.assertEqual(workload_benchmark.endpoint_name("multimodal"), "model")
        self.assertEqual(workload_benchmark.endpoint_name("decisions"), "decisions")
        self.assertEqual(workload_benchmark.endpoint_name("clm"), "decisions")
        self.assertEqual(workload_benchmark.endpoint_name("images"), "images")
        endpoints = workload_benchmark.resolve_endpoints(
            {"text", "decisions", "images"},
            model_url="http://model:8081", decisions_url="http://decision:8191",
            images_url="http://image:8190", model_key_env="MODEL_KEY",
            decisions_key_env="DECISION_KEY", images_key_env="IMAGE_KEY",
            environment={"MODEL_KEY": "model-secret", "DECISION_KEY": "decision-secret", "IMAGE_KEY": "image-secret"},
        )
        self.assertEqual(set(endpoints), {"model", "decisions", "images"})
        self.assertEqual(endpoints["decisions"][0], "http://decision:8191")
        self.assertEqual(endpoints["images"][0], "http://image:8190")
        self.assertEqual(endpoints["model"][1], "model-secret")
        self.assertEqual(endpoints["decisions"][1], "decision-secret")
        self.assertEqual(endpoints["images"][1], "image-secret")
        with self.assertRaisesRegex(ValueError, "valid port"):
            workload_benchmark.validate_origin("http://model:99999")
        self.assertIsNone(workload_benchmark.serial_group(self.case()))
        self.assertEqual(
            workload_benchmark.serial_group({"workload": "decisions", "request": {"model": "clef-flash-q4"}}),
            "clef-decisions",
        )
        self.assertEqual(
            workload_benchmark.serial_group({"workload": "clm", "request": {"model": "clm-v0.1-8b"}}),
            "clm",
        )
        cases = [
            {"request": {"model": model}}
            for model in ("one", "two", "one", "three", "two")
        ]
        ordered = workload_benchmark.ordered_cases(cases, 9)
        ordered_models = [case["request"]["model"] for case in ordered]
        self.assertEqual(len(ordered), len(cases))
        self.assertEqual(len({model for model in ordered_models}), 3)
        self.assertEqual(sum(left != right for left, right in zip(ordered_models, ordered_models[1:])), 2)
        self.assertEqual(workload_benchmark.ordered_cases(cases, 9), ordered)

    def test_generated_suite_covers_each_workload_and_verifies_edit_asset(self):
        with tempfile.TemporaryDirectory() as temporary:
            suite_path, digest, count = make_workload_suite.make_suite(Path(temporary) / "suite")
            cases, actual_digest = workload_benchmark.read_suite(suite_path)
            self.assertEqual(digest, actual_digest)
            self.assertEqual(count, len(cases))
            self.assertEqual({case["suite_id"] for case in cases}, {"redqueen-workload-v2"})
            self.assertEqual({case["workload"] for case in cases}, {
                "text", "prefill", "embeddings", "decisions", "clm", "multimodal", "images",
            })
            config = Path(__file__).resolve().parents[2] / "model-router" / "config.yaml"
            configured_models = set(re.findall(r"^\s+--alias\s+(\S+)$", config.read_text(), re.MULTILINE))
            self.assertEqual(configured_models, set(make_workload_suite.MODEL_ALIASES))
            self.assertEqual(
                {case["request"]["model"] for case in cases if case["workload"] == "text"},
                set(make_workload_suite.TEXT_MODELS),
            )
            self.assertEqual(
                {
                    model: {case["path"] for case in cases if case["workload"] == "text" and case["request"]["model"] == model}
                    for model in make_workload_suite.TEXT_MODELS
                },
                {
                    model: ({"/v1/chat/completions"} if model in make_workload_suite.OPENAI_SERVICE_MODELS
                            else set(workload_benchmark.ROUTES["text"]))
                    for model in make_workload_suite.TEXT_MODELS
                },
            )
            for model in make_workload_suite.TEXT_MODELS:
                route_counts = {
                    route: sum(
                        case["workload"] == "text"
                        and case["request"]["model"] == model
                        and case["path"] == route
                        for case in cases
                    )
                    for route in workload_benchmark.ROUTES["text"]
                }
                expected_routes = (
                    {"/v1/chat/completions": 8}
                    if model in make_workload_suite.OPENAI_SERVICE_MODELS
                    else {route: 8 for route in workload_benchmark.ROUTES["text"]}
                )
                self.assertEqual(
                    {route: count for route, count in route_counts.items() if count}, expected_routes
                )
            self.assertEqual(
                {case["request"]["model"] for case in cases if case["workload"] == "prefill"},
                set(make_workload_suite.TEXT_MODELS),
            )
            self.assertTrue(all(
                len(case["request"]["messages"][0]["content"]) >= 12_000
                for case in cases if case["workload"] == "prefill"
            ))
            self.assertEqual(
                {case["request"]["model"] for case in cases if case["workload"] == "multimodal"},
                set(make_workload_suite.MULTIMODAL_MODELS),
            )
            self.assertEqual(
                {case["request"]["model"] for case in cases if case["workload"] == "images"},
                set(make_workload_suite.IMAGE_MODELS),
            )
            scheduler = json.loads((config.parent / "scheduler.json").read_text())
            self.assertEqual(
                scheduler["services"]["deepseek"]["supported_request_shapes"],
                ["openai.chat", "openai.multimodal_chat"],
            )
            self.assertEqual(
                set(scheduler["multimodal_profiles"]) | {"deepseek"},
                set(make_workload_suite.MULTIMODAL_MODELS),
            )
            jev_api = (Path(__file__).resolve().parents[2] / "adapters" / "redqueen_adapters" / "jev_api.py").read_text()
            clef_families = jev_api.split("CLEF_MODEL_FAMILIES = {", 1)[1].split("}", 1)[0]
            self.assertEqual(
                set(re.findall(r'^\s*"([^"]+)":', clef_families, re.MULTILINE)),
                set(make_workload_suite.CLEF_MODELS),
            )
            image_api = (Path(__file__).resolve().parents[2] / "adapters" / "redqueen_adapters" / "image_api.py").read_text()
            image_models = tuple(re.findall(r'^(?:MODEL|UNCENSORED_MODEL) = "([^"]+)"$', image_api, re.MULTILINE))
            self.assertEqual(image_models, make_workload_suite.IMAGE_MODELS)
            edit = next(case for case in cases if case["path"] == "/v1/images/edits")
            body, content_type = workload_benchmark.multipart_body(edit, suite_path.parent)
            self.assertIn(b"image[]", body)
            self.assertTrue(content_type.startswith("multipart/form-data; boundary="))
            decisions = [case for case in cases if case["workload"] == "decisions"]
            expected_decision_models = {make_workload_suite.DECISION_MODEL, *make_workload_suite.CLEF_MODELS}
            self.assertEqual({case["request"]["model"] for case in decisions}, expected_decision_models)
            for model in expected_decision_models:
                model_cases = [case for case in decisions if case["request"]["model"] == model]
                self.assertEqual({case["path"] for case in model_cases}, {"/v1/systemone", "/v1/decisions"})
                expected = make_workload_suite.DECISION_EXPECTED
                if model in make_workload_suite.CLEF_MODELS:
                    expected = expected[:16]
                self.assertEqual(len(model_cases), len(expected))
                self.assertEqual([case["scoring"]["expected"] for case in model_cases], expected)

    def test_exact_scoring_is_type_aware_and_json_pointer_handles_escapes(self):
        case = self.case(scoring={"kind": "exact", "pointer": "/answers/a~1b", "expected": "yes"})
        self.assertEqual(workload_benchmark.score_case(case, {"answers": {"a/b": "yes"}})["value"], 1.0)
        numeric = self.case(scoring={"kind": "exact", "pointer": "/value", "expected": 1})
        self.assertEqual(workload_benchmark.score_case(numeric, {"value": True})["value"], 0.0)
        self.assertEqual(workload_benchmark.score_case(numeric, {"value": 1.0})["value"], 1.0)

    def test_retrieval_scoring_uses_cosine_and_zero_based_document_positions(self):
        case = self.case(
            workload="embeddings", route="/v1/embeddings",
            scoring={"kind": "retrieval", "relevant": [1], "metric": "mrr"},
        )
        case["request"]["input"] = ["query", "document zero", "document one"]
        response = {"data": [
            {"embedding": [1, 0]},
            {"embedding": [0, 1]},
            {"embedding": [0.9, 0.1]},
        ]}
        self.assertEqual(workload_benchmark.score_case(case, response)["value"], 1.0)
        response["data"][2]["embedding"] = [0, 1]
        self.assertEqual(workload_benchmark.score_case(case, response)["value"], 0.5)
        with self.assertRaisesRegex(ValueError, "count does not match"):
            workload_benchmark.score_case(case, {"data": response["data"][:2]})

    def test_retrieval_scoring_rejects_unhashable_metric(self):
        with self.assertRaisesRegex(workload_benchmark.SuiteError, "retrieval metric"):
            workload_benchmark.validate_scoring({"kind": "retrieval", "relevant": [0], "metric": []})

    def test_cosine_handles_large_finite_embedding_values(self):
        self.assertAlmostEqual(workload_benchmark.cosine([1e308, 1e308], [1e308, 1e308]), 1.0)

    def test_server_timings_drop_negative_values(self):
        self.assertEqual(
            workload_benchmark.numeric_timings({"prompt_n": 10, "prompt_ms": -1, "predicted_ms": 0}),
            {"prompt_n": 10, "predicted_ms": 0},
        )

    def test_sse_reader_collects_streamed_text_usage_and_ttft(self):
        events = (
            b'data: {"choices":[{"delta":{"content":"4"}}]}\n\n'
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":1}}\n\n'
            b"data: [DONE]\n\n"
        )
        text, usage, ttft, timings = workload_benchmark.sse_event_records(io.BytesIO(events), 0.0)
        self.assertEqual(text, "4")
        self.assertEqual(usage, {"completion_tokens": 1})
        self.assertGreaterEqual(ttft, 0)
        self.assertEqual(timings, {})

    def test_sse_reader_rejects_invalid_or_failed_events(self):
        for payload, message in (
            (b"data: not-json\n\n", "malformed JSON"),
            (b'data: {"type":"response.failed"}\n\n', "inference error"),
            (b'data: {"error":{"message":"bad"}}\n\n', "inference error"),
            (b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n', "terminal event"),
        ):
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, message):
                workload_benchmark.sse_event_records(io.BytesIO(payload), 0.0)
        response_events = (
            b'data: {"type":"response.output_text.delta","delta":"4"}\n\n'
            b'data: {"type":"response.completed","response":{"usage":{"input_tokens":8,"output_tokens":1}}}\n\n'
            b"data: [DONE]\n\n"
        )
        text, usage, ttft, timings = workload_benchmark.sse_event_records(io.BytesIO(response_events), 0.0)
        self.assertEqual(text, "4")
        self.assertEqual(usage, {"input_tokens": 8, "output_tokens": 1})
        self.assertGreaterEqual(ttft, 0)
        self.assertEqual(timings, {})

    def test_invoke_records_latency_and_never_follows_redirects(self):
        class Handler(BaseHTTPRequestHandler):
            authorization = None

            def log_message(self, *args):
                pass

            def do_POST(self):
                self.server.authorization = self.headers.get("Authorization")
                self.send_response(302)
                self.send_header("Location", "http://127.0.0.1:1/leak")
                self.end_headers()

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            record = workload_benchmark.invoke_case(
                case=self.case(), suite_dir=Path("."), origin=f"http://127.0.0.1:{server.server_port}",
                api_key="private-token", timeout_s=1, artifact_dir=None, store_artifacts=False, repeat=0,
            )
            self.assertEqual(server.authorization, "Bearer private-token")
            self.assertEqual(record["status_code"], 302)
            self.assertFalse(record["success"])
            self.assertEqual(record["error_type"], "http_error")
            self.assertNotIn("private-token", json.dumps(record))
            self.assertGreaterEqual(record["request_elapsed_s"], 0)
        finally:
            server.shutdown()
            server.server_close()
            worker.join()

    def test_model_output_is_scored_but_not_persisted(self):
        output = "private generated content"

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.dumps({"choices": [{"message": {"content": output}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            case = self.case(scoring={
                "kind": "exact", "pointer": "/choices/0/message/content", "expected": output,
            })
            record = workload_benchmark.invoke_case(
                case=case, suite_dir=Path("."), origin=f"http://127.0.0.1:{server.server_port}",
                api_key=None, timeout_s=1, artifact_dir=None, store_artifacts=False, repeat=0,
            )
            self.assertEqual(record["quality"], {"metric": "exact", "value": 1.0})
            self.assertNotIn("response", record)
            self.assertNotIn(output, json.dumps(record))
        finally:
            server.shutdown()
            server.server_close()
            worker.join()

    def test_image_artifacts_are_private_and_repeats_do_not_collide(self):
        png = b"\x89PNG\r\n\x1a\nfixture"
        encoded = base64.b64encode(png).decode()
        case = self.case(case_id="image-1", workload="images", route="/v1/images/generations", scoring=None)
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.dumps({"data": [{"b64_json": encoded}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                artifacts = Path(temporary) / "artifacts"
                artifacts.mkdir(mode=0o700)
                for repeat in (0, 1):
                    record = workload_benchmark.invoke_case(
                        case=case, suite_dir=Path("."), origin=f"http://127.0.0.1:{server.server_port}",
                        api_key=None, timeout_s=1, artifact_dir=artifacts, store_artifacts=True, repeat=repeat,
                    )
                    self.assertTrue(record["success"])
                    self.assertEqual(Path(record["artifact"]["path"]).read_bytes(), png)
                    self.assertEqual(Path(record["artifact"]["path"]).stat().st_mode & 0o777, 0o600)
                    self.assertNotIn("response", record)
                    self.assertNotIn("b64_json", json.dumps(record))
        finally:
            server.shutdown()
            server.server_close()
            worker.join()

    def comparison_suite(self, root: Path, workload: str, count: int) -> Path:
        path = "/v1/images/generations" if workload == "images" else "/v1/chat/completions"
        rows = []
        for index in range(count):
            row = {
                "schema_version": 1, "suite_id": "test-suite-v1", "id": f"{workload}-{index}",
                "workload": workload, "path": path, "request": {"model": "test-model"},
            }
            if workload != "images":
                row["scoring"] = {"kind": "exact", "pointer": "/value", "expected": 1}
            rows.append(row)
        return self.suite_file(root, rows)

    def make_results(self, root: Path, label: str, *, suite_hash: str, latency: float,
                     quality: float, workload: str = "text", count: int = 4) -> Path:
        path = root / f"{label}.jsonl"
        records = []
        for index in range(count):
            records.append({
                "schema_version": 1, "suite_id": "test-suite-v1", "case_id": f"{workload}-{index}", "repeat": 0,
                "suite_sha256": suite_hash, "label": label, "workload": workload,
                "order_seed": 0, "concurrency": 1, "effective_concurrency": 1, "warmups": 0, "timeout_s": 1800,
                "path": "/v1/chat/completions", "requested_model": "test-model", "status_code": 200,
                "error_type": None, "success": True,
                "request_elapsed_s": latency,
                "quality": {"metric": "exact", "value": quality},
            })
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
        return path

    def test_paired_bootstrap_enforces_latency_and_quality_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            suite = self.comparison_suite(root, "text", 8)
            suite_hash = hashlib.sha256(suite.read_bytes()).hexdigest()
            reference = self.make_results(root, "reference", suite_hash=suite_hash, latency=1.0, quality=1.0, count=8)
            candidate = self.make_results(root, "candidate", suite_hash=suite_hash, latency=0.7, quality=1.0, count=8)
            result = compare_workloads.analyze(reference, candidate, suite_path=suite, iterations=200, seed=4)
            workload = result["workloads"]["text"]["models"]["test-model"]["routes"]["/v1/chat/completions"]
            self.assertEqual(workload["latency_gate"], "pass")
            self.assertEqual(workload["quality_gate"], "no_supported_regression")

            candidate.write_text(candidate.read_text().replace('"value": 1.0', '"value": 0.0'), encoding="utf-8")
            result = compare_workloads.analyze(reference, candidate, suite_path=suite, iterations=200, seed=4)
            self.assertEqual(
                result["workloads"]["text"]["models"]["test-model"]["routes"]["/v1/chat/completions"]["quality_gate"],
                "regression",
            )

            altered = json.loads(candidate.read_text(encoding="utf-8").splitlines()[0])
            altered["requested_model"] = "other-model"
            candidate.write_text(json.dumps(altered) + "\n" + "\n".join(candidate.read_text(encoding="utf-8").splitlines()[1:]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requested model"):
                compare_workloads.analyze(reference, candidate, suite_path=suite, iterations=200, seed=4)

    def test_blinded_image_pair_key_and_ratings_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ref_image, cand_image = root / "ref.png", root / "cand.png"
            ref_bytes = b"\x89PNG\r\n\x1a\nreference"
            cand_bytes = b"\x89PNG\r\n\x1a\ncandidate"
            ref_image.write_bytes(ref_bytes)
            cand_image.write_bytes(cand_bytes)
            suite = self.comparison_suite(root, "images", 1)
            suite_hash = hashlib.sha256(suite.read_bytes()).hexdigest()

            def image_record(label: str, image: Path, data: bytes) -> dict:
                return {
                    "schema_version": 1, "suite_id": "test-suite-v1", "case_id": "images-0", "repeat": 0,
                    "suite_sha256": suite_hash, "label": label, "workload": "images",
                    "order_seed": 0, "concurrency": 1, "effective_concurrency": 1, "warmups": 0, "timeout_s": 1800,
                    "path": "/v1/images/generations", "requested_model": "test-model", "status_code": 200,
                    "error_type": None, "success": True, "request_elapsed_s": 1.0,
                    "quality": None,
                    "artifact": {"path": str(image), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)},
                }

            reference, candidate = root / "reference.jsonl", root / "candidate.jsonl"
            reference.write_text(json.dumps(image_record("old", ref_image, ref_bytes)) + "\n", encoding="utf-8")
            candidate.write_text(json.dumps(image_record("new", cand_image, cand_bytes)) + "\n", encoding="utf-8")
            blind_dir, key_path = root / "blind", root / "key.json"
            self.assertEqual(prepare_blind_images.prepare(reference, candidate, suite, blind_dir, key_path, seed=1), 1)
            self.assertEqual(len(list(blind_dir.glob("*-A.png"))), 1)
            key = json.loads(key_path.read_text(encoding="utf-8"))
            pair_id = next(iter(key["pairs"]))
            ratings = root / "ratings.jsonl"
            ratings.write_text(json.dumps({"pair_id": pair_id, "preference": "A"}) + "\n", encoding="utf-8")
            scored = compare_workloads.read_image_ratings(ratings, key_path, suite_hash, "test-suite-v1")
            self.assertEqual(set(scored), {"test-model"})
            self.assertEqual(set(scored["test-model"]), {pair_id})
            self.assertEqual(compare_workloads.image_preference_summary(scored["test-model"])["rated_pairs"], 1)
            summary = compare_workloads.analyze(
                reference, candidate, suite_path=suite, iterations=200, seed=2,
                image_ratings=ratings, image_key=key_path,
            )
            self.assertEqual(
                summary["workloads"]["images"]["models"]["test-model"]["quality_gate"],
                "insufficient_pairs",
            )

    def test_blind_image_quality_uses_one_independent_pair_per_case(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ref_image, cand_image = root / "ref.png", root / "cand.png"
            ref_bytes = b"\x89PNG\r\n\x1a\nreference"
            cand_bytes = b"\x89PNG\r\n\x1a\ncandidate"
            ref_image.write_bytes(ref_bytes)
            cand_image.write_bytes(cand_bytes)
            suite = self.comparison_suite(root, "images", 1)
            suite_hash = hashlib.sha256(suite.read_bytes()).hexdigest()

            def image_record(label: str, image: Path, data: bytes, repeat: int) -> dict:
                record = {
                    "schema_version": 1, "suite_id": "test-suite-v1", "case_id": "images-0", "repeat": repeat,
                    "suite_sha256": suite_hash, "label": label, "workload": "images",
                    "order_seed": 0, "concurrency": 1, "effective_concurrency": 1, "warmups": 0, "timeout_s": 1800,
                    "path": "/v1/images/generations", "requested_model": "test-model", "status_code": 200,
                    "error_type": None, "success": True, "request_elapsed_s": 1.0,
                    "quality": None,
                    "artifact": {"path": str(image), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)},
                }
                return record

            reference, candidate = root / "reference.jsonl", root / "candidate.jsonl"
            reference.write_text("".join(
                json.dumps(image_record("old", ref_image, ref_bytes, repeat)) + "\n" for repeat in (0, 1)
            ), encoding="utf-8")
            candidate.write_text("".join(
                json.dumps(image_record("new", cand_image, cand_bytes, repeat)) + "\n" for repeat in (0, 1)
            ), encoding="utf-8")
            blind_dir, key_path = root / "blind", root / "key.json"
            self.assertEqual(prepare_blind_images.prepare(reference, candidate, suite, blind_dir, key_path, seed=1), 1)
            key = json.loads(key_path.read_text(encoding="utf-8"))
            self.assertEqual(len(key["pairs"]), 1)


if __name__ == "__main__":
    unittest.main()
