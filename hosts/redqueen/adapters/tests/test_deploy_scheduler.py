from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import yaml

from redqueen_adapters import deploy_scheduler
from redqueen_adapters.memory_profile import fingerprint_artifacts


class DeployedProfileVerificationTests(unittest.TestCase):
    def test_accepts_generated_multimodal_model_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "inference"
            profile_directory = root / "run"
            profile_directory.mkdir()
            profile_path = profile_directory / "memory-profiles.json"
            model_path = "/srv/ai/models/vision/vision-00001-of-00002.gguf"
            runtime_path = "/srv/ai/runtimes/llama-server"
            router = {
                "models": {
                    "vision": {
                        "cmd": f"{runtime_path} --model {model_path}",
                        "concurrencyLimit": 1,
                        "aliases": [],
                    }
                }
            }
            scheduler = {
                "profile_path": str(profile_path),
                "socket_path": str(profile_directory / "scheduler.sock"),
                "services": {},
            }
            contents = {
                "hosts/redqueen/model-router/config.yaml": yaml.safe_dump(router),
                "hosts/redqueen/model-router/scheduler.json": json.dumps(scheduler) + "\n",
                "hosts/redqueen/adapters/redqueen_adapters/memory_profile.py": "# fixture\n",
            }
            for relative, content in contents.items():
                path = repo / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")

            subprocess.run(["git", "-C", str(repo), "init", "--quiet"], check=True)
            subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
            subprocess.run(
                [
                    "git", "-C", str(repo), "-c", "user.name=test",
                    "-c", "user.email=test@example.invalid", "commit", "--quiet", "-m", "fixture",
                ],
                check=True,
            )
            revision = subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            sources = [
                {
                    "path": str(repo / relative),
                    "sha256": hashlib.sha256((repo / relative).read_bytes()).hexdigest(),
                }
                for relative in contents
            ]
            artifact_identity = [
                {
                    "model": "vision",
                    "repositories": {"repository": "local/vision"},
                    "revisions": {"revision": "test-revision"},
                    "manifest_sha256": "a" * 64,
                }
            ]
            zero_memory = {"gpu_bytes": 0, "host_bytes": 0}
            profiles = {
                "version": 1,
                "worker_profile_version": 1,
                "resources": {
                    "vision": {
                        "kind": "text",
                        "concurrency": 1,
                        "concurrency_limit": 1,
                        "aliases": [],
                        "weights_bytes": 0,
                        "context_tokens": 4096,
                        "planner": "/srv/ai/runtimes/llama-fit-params",
                        "gpu_bytes": 0,
                        "host_bytes": 0,
                        "memory": {
                            "mandatory": zero_memory.copy(),
                            "reclaimable": zero_memory.copy(),
                            "transient_peak": zero_memory.copy(),
                        },
                        "artifact_identity": artifact_identity,
                        "artifact_fingerprint": fingerprint_artifacts(artifact_identity),
                        "model_paths": [
                            "/srv/ai/models/vision/vision-00001-of-00002.gguf",
                            "/srv/ai/models/vision/vision-00002-of-00002.gguf",
                        ],
                        "runtime_identity": {
                            "path": runtime_path,
                            "size": 0,
                            "mtime_ns": 0,
                        },
                        "supported_request_shapes": ["openai.multimodal_chat"],
                        "max_image_inputs": 1,
                        "multimodal_workspace_mib": 4096,
                    }
                },
                "sources": sources,
            }
            marker = {
                "version": 1,
                "repo": str(repo),
                "revision": revision,
            }

            def run_git(*arguments: str, check: bool = True, timeout: float = 180):
                return subprocess.run(
                    arguments,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )

            deploy_scheduler.verify_profiles(profiles, verify_sources=False)
            profile_bytes = json.dumps(profiles, sort_keys=True).encode("utf-8")
            marker["profile_sha256"] = hashlib.sha256(profile_bytes).hexdigest()
            with patch.object(deploy_scheduler, "run", side_effect=run_git):
                verified = deploy_scheduler.verify_deployed_profiles(
                    repo,
                    revision,
                    profile_path,
                    profile_bytes,
                    profiles,
                    json.dumps(marker).encode("utf-8"),
                )

            self.assertEqual(verified[0], revision)
            self.assertTrue(verified[3])

            tampered = deepcopy(profiles)
            tampered["resources"]["vision"]["unexpected"] = True
            tampered_bytes = json.dumps(tampered, sort_keys=True).encode("utf-8")
            marker["profile_sha256"] = hashlib.sha256(tampered_bytes).hexdigest()
            with patch.object(deploy_scheduler, "run", side_effect=run_git):
                with self.assertRaisesRegex(RuntimeError, "has undeclared fields"):
                    deploy_scheduler.verify_deployed_profiles(
                        repo,
                        revision,
                        profile_path,
                        tampered_bytes,
                        tampered,
                        json.dumps(marker).encode("utf-8"),
                    )

            tampered = deepcopy(profiles)
            tampered["resources"]["vision"]["max_image_inputs"] = True
            with self.assertRaisesRegex(ValueError, "invalid max_image_inputs"):
                deploy_scheduler.verify_profiles(tampered, verify_sources=False)


if __name__ == "__main__":
    unittest.main()
