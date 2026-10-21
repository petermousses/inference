from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[4]


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


preflight = load_script("inference_preflight", ROOT / "hosts/redqueen/model-router/inference-preflight.py")
benchmark = load_script("prefill_benchmark", ROOT / "hosts/redqueen/model-router/prefill-benchmark.py")


class InferencePreflightTests(unittest.TestCase):
    def test_pci_inventory_ignores_display_connector_entries(self):
        with tempfile.TemporaryDirectory() as temporary:
            drm_root = Path(temporary)
            for name in ("card0", "card0-DP-1"):
                device = drm_root / name / "device"
                device.mkdir(parents=True)
                (device / "vendor").write_text("0x1002\n")
                (device / "device").write_text("0x744c\n")

            devices = preflight.pci_devices(drm_root)

            self.assertEqual([item["drm_card"] for item in devices], ["card0"])

    def test_collects_gguf_shards_and_manifest_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = root / "weights"
            model_file = weights / "inference" / "model.safetensors"
            model_file.parent.mkdir(parents=True)
            model_file.write_bytes(b"model")
            digest = hashlib.sha256(b"model").hexdigest()
            manifest = root / "files.tsv"
            manifest.write_text(f"inference/model.safetensors\t5\t{digest}\t{'a' * 40}\n")
            router = {
                "models": {
                    "text": {"cmd": "llama-server --model /models/text-00001-of-00002.gguf --mmproj=/models/vision.gguf"}
                }
            }
            policy = {
                "services": {
                    "deepseek": {"artifact_root": str(weights), "artifact_inventory": str(manifest)}
                }
            }

            artifacts = preflight.collect_artifact_refs(router, policy, root)

            self.assertEqual(len(artifacts["/models/text-00002-of-00002.gguf"]["references"]), 1)
            self.assertEqual(artifacts[str(model_file)]["expected_size_bytes"], 5)
            self.assertEqual(artifacts[str(model_file)]["expected_sha256"], digest)

    def test_inventory_rejects_paths_outside_artifact_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "files.tsv"
            manifest.write_text(f"../escape.safetensors\t1\t{'0' * 64}\t{'a' * 40}\n")
            policy = {
                "services": {
                    "deepseek": {"artifact_root": str(root / "weights"), "artifact_inventory": str(manifest)}
                }
            }
            with self.assertRaisesRegex(ValueError, "unsafe artifact path"):
                preflight.collect_artifact_refs({}, policy, root)

    def test_collect_reports_scheduler_memory_budgets_and_manifest_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = root / "weights"
            model_file = weights / "inference" / "model.safetensors"
            model_file.parent.mkdir(parents=True)
            model_file.write_bytes(b"model")
            digest = hashlib.sha256(b"model").hexdigest()
            manifest = root / "files.tsv"
            manifest.write_text(f"inference/model.safetensors\t5\t{digest}\t{'a' * 40}\n")
            router_path = root / "router.yaml"
            router_path.write_text("models: {}\n")
            scheduler_path = root / "scheduler.json"
            scheduler_path.write_text(
                json.dumps(
                    {
                        "gpu_budget_mib": 10,
                        "host_budget_mib": 20,
                        "minimum_host_available_mib": 3,
                        "prefill_headroom_mib": 4,
                        "estimate_margin": 1.08,
                        "model_overhead_mib": 5,
                        "moe_stream_overhead_mib": 6,
                        "services": {
                            "deepseek": {
                                "artifact_root": str(weights),
                                "artifact_inventory": str(manifest),
                            }
                        },
                    }
                )
            )
            args = argparse.Namespace(
                router_config=router_path,
                scheduler_config=scheduler_path,
                sha256=True,
                benchmark_io=False,
                io_sample_mib=1,
            )

            fake_yaml = type(
                "Yaml",
                (),
                {"safe_load": staticmethod(lambda _: {"models": {}}), "YAMLError": ValueError},
            )()
            with patch.object(preflight, "yaml", fake_yaml), patch.object(
                preflight, "host_details", return_value={}
            ), patch.object(preflight, "mounted_filesystems", return_value=[]):
                result = preflight.collect(args)

            self.assertEqual(result["memory_policy"]["gpu_budget_mib"], 10)
            self.assertEqual(result["memory_policy"]["prefill_headroom_mib"], 4)
            artifact = next(item for item in result["artifacts"] if item["path"] == str(model_file))
            self.assertTrue(artifact["size_matches_manifest"])
            self.assertTrue(artifact["sha256_matches_manifest"])

    def test_atomic_output_replaces_symlink_without_touching_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.write_text("keep")
            output = root / "report.json"
            output.symlink_to(target)

            preflight.write_atomic(output, "report")

            self.assertEqual(target.read_text(), "keep")
            self.assertFalse(output.is_symlink())
            self.assertEqual(output.read_text(), "report")


class PrefillBenchmarkTests(unittest.TestCase):
    def test_cli_defaults_to_redqueen_scheduler_listener(self):
        with patch.object(sys, "argv", ["prefill-benchmark.py", "cases.jsonl", "--label", "test"]):
            args = benchmark.parse_args()

        self.assertEqual(args.url, "http://10.9.20.242:8081/v1/chat/completions")

    def test_reads_cases_lazily_and_prepares_chat_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cases.jsonl"
            path.write_text(
                '{"id":"first","request":{"model":"one","messages":[{"role":"user","content":"hi"}]}}\n'
                '{"model":"two","prompt":"hello"}\n'
            )

            cases = benchmark.read_cases(path)
            first = next(cases)
            self.assertEqual(first["index"], 1)
            self.assertEqual(first["request"]["model"], "one")
            second = next(cases)
            self.assertEqual(second["index"], 2)
            self.assertEqual(benchmark.prepare_request(second["request"], model="override", include_usage=True)["model"], "override")

    def test_ttft_ignores_tool_call_metadata_without_generated_text(self):
        self.assertFalse(benchmark.has_generated_delta({"tool_calls": [{"id": "call_1", "type": "function"}]}))
        self.assertTrue(
            benchmark.has_generated_delta(
                {"tool_calls": [{"id": "call_1", "type": "function", "function": {"arguments": "{"}}]}
            )
        )

    def test_rejects_requests_without_a_model_before_any_request_is_sent(self):
        with self.assertRaisesRegex(ValueError, "needs a model"):
            benchmark.prepare_request({"prompt": "hello"}, model=None, include_usage=False)

    def test_stream_measurement_uses_first_generated_delta_and_usage(self):
        class FakeResponse:
            status = 200

            def __init__(self):
                self.events = iter(
                    [
                        {"choices": [{"delta": {"tool_calls": [{"id": "call_1", "type": "function"}]}}]},
                        {"choices": [{"delta": {"content": "ready"}}], "timings": {"prompt_n": 4, "prompt_ms": 2.5}},
                        {"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}},
                        "[DONE]",
                    ]
                )

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def readline(self):
                event = next(self.events, None)
                if event is None:
                    return b""
                if isinstance(event, dict) and event.get("choices") and event["choices"][0].get("delta", {}).get("content"):
                    time.sleep(0.05)
                payload = event if isinstance(event, str) else json.dumps(event)
                return f"data: {payload}\n\n".encode()

        observed = {}

        def fake_urlopen(request, timeout):
            observed["request"] = request
            observed["timeout"] = timeout
            return FakeResponse()

        with patch.object(benchmark.urllib.request, "urlopen", side_effect=fake_urlopen):
            result = benchmark.stream_case(
                url="http://127.0.0.1:8081/v1/chat/completions",
                api_key="fixture-key",
                model=None,
                request_data={"model": "fixture", "messages": [{"role": "user", "content": "hi"}]},
                include_usage=True,
                timeout_s=2,
            )

        self.assertEqual(observed["request"].get_header("Authorization"), "Bearer fixture-key")
        request_body = json.loads(observed["request"].data)
        self.assertTrue(request_body["stream"])
        self.assertTrue(request_body["stream_options"]["include_usage"])
        self.assertGreater(result["ttft_s"], 0.04)
        self.assertEqual(result["usage"]["completion_tokens"], 1)
        self.assertEqual(result["server_prefill_timings"]["prompt_n"], 4)


if __name__ == "__main__":
    unittest.main()
