"""Dependency-free boot and deployment fault fixtures."""

from pathlib import Path
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

from redqueen_adapters import host_reliability as host


ROOT = Path(__file__).resolve().parents[2]


class UnitPolicyTest(unittest.TestCase):
    def test_fronts_keep_retrying_without_start_limit_exhaustion(self):
        for name in ("redqueen-scheduler", "qwen-image-adapter", "jevk5-adapter", "llama-swap"):
            with self.subTest(name=name):
                body = (ROOT / "systemd" / f"{name}.service").read_text()
                self.assertIn("StartLimitIntervalSec=0", body)
                self.assertIn("Restart=always", body)
                self.assertIn("RestartSec=5", body)
                self.assertIn("RestartSteps=5", body)
                self.assertIn("RestartMaxDelaySec=60", body)

    def test_model_backends_never_restart_outside_admission(self):
        for name in ("comfyui", "jevk5", "clm-api", "clm-encoder", "qwen38"):
            with self.subTest(name=name):
                self.assertIn("Restart=no", (ROOT / "systemd" / f"{name}.service").read_text())

    def test_preflight_runs_before_fronts_and_recovery_uses_regular_copy(self):
        for name in host.FRONT_URLS:
            self.assertIn("After=redqueen-preflight.service", (ROOT / "systemd" / name).read_text())
        comfy_unit = (ROOT / "systemd/comfyui.service").read_text()
        after = next(line for line in comfy_unit.splitlines() if line.startswith("After="))
        self.assertIn("redqueen-preflight.service", after.split("=", 1)[1].split())
        service = (ROOT / "systemd/redqueen-supervisor.service").read_text()
        self.assertIn("/srv/ai/model-router/libexec/host_reliability.py", service)
        self.assertNotIn("/srv/ai/adapters/source", service)
        for name in ("qwen-image-adapter", "jevk5-adapter"):
            self.assertIn("ReadWritePaths=/srv/ai/model-router/run/requests", (ROOT / "systemd" / f"{name}.service").read_text())
        self.assertEqual(host.preflight_unit_body(Path("/srv/ai")), (ROOT / "systemd/redqueen-preflight.service").read_bytes())

    def test_transient_setup_creates_private_regular_operations_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            host.transient_directories(root)
            lock = root / "model-router/run/operations.lock"
            self.assertTrue(lock.is_file())
            self.assertEqual(lock.stat().st_mode & 0o777, 0o600)
            lock.unlink()
            lock.symlink_to(root / "outside")
            with self.assertRaises(OSError):
                host.transient_directories(root)

    def test_comfyui_requires_the_prompt_admission_guard_before_startup_succeeds(self):
        unit = (ROOT / "systemd/comfyui.service").read_text()
        self.assertIn(
            "ExecStartPost=/srv/ai/comfyui/venv/bin/python "
            "/srv/ai/config/hosts/redqueen/comfyui/check_admission_guard.py",
            unit,
        )

    def test_every_native_model_uses_an_unbounded_socket_timeout(self):
        config = (ROOT / "model-router/config.yaml").read_text()
        self.assertEqual(config.count("    cmd: >-"), config.count("      --timeout -1"))
        self.assertIn("healthCheckTimeout: 0", config)
        for name in ("jevk5", "clm-encoder", "qwen38"):
            self.assertIn("--timeout -1", (ROOT / "systemd" / f"{name}.service").read_text())

    def test_router_installer_verifies_the_tracked_patch(self):
        patch_path = ROOT / "model-router/patches/llama-swap-v260-unbounded-load.patch"
        digest = hashlib.sha256(patch_path.read_bytes()).hexdigest()
        self.assertIn(f"readonly patch_sha256={digest}", (ROOT / "model-router/install-llama-swap-runtime.sh").read_text())


class HostFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.home = self.root / "home"
        home_patch = patch.object(Path, "home", return_value=self.home)
        home_patch.start()
        self.addCleanup(home_patch.stop)
        self.directory = self.root / "model-router/run"
        host.transient_directories(self.root)
        for unit in host.FRONT_URLS:
            for path in host.front_requirements(self.root, unit):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture")
        source = self.root / "models/weights"
        source.parent.mkdir()
        source.write_bytes(b"weights")
        artifact_identity = [{
            "model": "fixture", "repositories": {"repository": "test/fixture"},
            "revisions": {"revision": "1"}, "manifest_sha256": "f" * 64,
        }]
        artifact_fingerprint = hashlib.sha256(
            json.dumps(artifact_identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.profiles = {
            "version": 1, "worker_profile_version": 1, "resources": {"fixture": {
                "gpu_bytes": 1, "host_bytes": 1,
                "memory": {
                    "mandatory": {"gpu_bytes": 1, "host_bytes": 1},
                    "reclaimable": {"gpu_bytes": 0, "host_bytes": 0},
                    "transient_peak": {"gpu_bytes": 0, "host_bytes": 0},
                },
                "artifact_identity": artifact_identity,
                "artifact_fingerprint": artifact_fingerprint,
                "runtime_identity": {"path": "/usr/bin/llama-server", "size": 1, "mtime_ns": 1},
                "supported_request_shapes": ["openai.chat"],
                "concurrency_limit": 1,
            }},
            "sources": [{"path": str(source), "size": source.stat().st_size,
                         "mtime_ns": source.stat().st_mtime_ns}],
        }
        (self.directory / "memory-profiles.json").write_text(json.dumps(self.profiles))

    def result(self, *args, **kwargs):
        output = "enabled\n"
        if args[0] == "loginctl":
            output = "yes\n"
        elif args[0] == "show":
            output = "0\n" if "--property=Job" in args else "loaded\n"
        elif args[0] == "is-enabled" and args[1] in host.BACKENDS:
            output = "linked\n"
        return subprocess.CompletedProcess(args, 0, output, "")


class RecoveryTest(HostFixture):
    def test_first_update_wires_independent_boot_recovery_before_source_can_change(self):
        helper = self.root / "model-router/libexec/host_reliability.py"
        host.install_helper(Path(host.__file__), helper)
        definition = host.preflight_snapshot()
        self.assertEqual(definition["kind"], "absent")
        host.begin_journal(self.directory, self.root / "inference", "1" * 40,
                           {"redqueen-preflight.service": False}, [],
                           self.directory / "memory-profiles.json", None, definition)
        unit = self.home / ".config/systemd/user/redqueen-preflight.service"
        wants = unit.parent / "default.target.wants/redqueen-preflight.service"
        def control(*args, **kwargs):
            self.assertTrue((self.directory / "deployment.json").exists())
            if args[:2] == ("enable", "redqueen-preflight.service"):
                wants.parent.mkdir()
                wants.symlink_to(unit)
            return self.result(*args, **kwargs)
        with patch.object(host, "systemctl", side_effect=control):
            host.bootstrap_preflight(self.root)
        self.assertTrue(wants.is_symlink())
        self.assertTrue(unit.is_file())
        self.assertFalse(unit.is_symlink())
        self.assertIn(str(helper), unit.read_text())
        before = next(line for line in unit.read_text().splitlines() if line.startswith("Before="))
        dependencies = before.split("=", 1)[1].split()
        self.assertIn("comfyui.service", dependencies)
        self.assertIn("llama-swap.service", dependencies)
        self.assertIn("default.target", dependencies)
        # A partial/corrupt candidate checkout cannot remove either dependency
        # used by this already enabled boot job.
        candidate = self.root / "inference"
        candidate.mkdir()
        (candidate / "broken.py").write_text("raise ImportError()\n")
        self.assertTrue(helper.is_file())
        self.assertTrue(unit.is_file())
        host.restore_preflight(definition)
        self.assertFalse(unit.exists())
        self.assertFalse(wants.exists())

    def test_failed_update_restores_the_prior_regular_preflight_definition(self):
        unit = self.home / ".config/systemd/user/redqueen-preflight.service"
        unit.parent.mkdir(parents=True)
        unit.write_text("previous independent unit")
        definition = host.preflight_snapshot()
        host.install_preflight_unit(self.root)
        self.assertNotEqual(unit.read_text(), "previous independent unit")
        host.restore_preflight(definition)
        self.assertEqual(unit.read_text(), "previous independent unit")
        self.assertFalse(unit.is_symlink())

    def test_credentials_require_keys_and_private_permissions_without_logging_values(self):
        path = self.root / "secrets/llama-swap.env"
        secret = "sk-private-do-not-log"
        path.write_text(f"export QWEN_API_KEY='{secret}'\n")
        path.chmod(0o600)
        self.assertEqual(host.credential_errors(path, ("QWEN_API_KEY",)), [])
        path.write_text(f"OTHER='{secret}'\nQWEN_API_KEY=''\n")
        errors = host.credential_errors(path, ("QWEN_API_KEY",))
        self.assertIn("missing or empty QWEN_API_KEY", errors[0])
        self.assertNotIn(secret, " ".join(errors))
        path.chmod(0o644)
        self.assertIn("0600", host.credential_errors(path, ("QWEN_API_KEY",))[0])

    def test_credential_parsing_never_executes_shell_values(self):
        path = self.root / "secrets/llama-swap.env"
        marker = self.root / "must-not-exist"
        path.write_text(f'QWEN_API_KEY="$(touch {marker})"\n')
        path.chmod(0o600)
        host.credential_errors(path, ("QWEN_API_KEY",))
        self.assertFalse(marker.exists())

    def test_atomic_write_failure_keeps_the_previous_file(self):
        path = self.directory / "fixture.json"
        path.write_bytes(b"previous")
        with patch.object(host.os, "fsync", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                host.atomic_write(path, b"candidate")
        self.assertEqual(path.read_bytes(), b"previous")
        self.assertEqual(list(self.directory.glob(".fixture.json.*")), [])

    def test_missing_profile_backup_does_not_stop_running_services(self):
        document = {"version": 1, "repo": str(self.root / "inference"), "previous": "1" * 40,
                    "enabled": {}, "active": [], "profile_path": str(self.directory / "memory-profiles.json"),
                    "profile_backup": str(self.directory / "missing-backup.json")}
        with patch.object(host, "command", return_value=subprocess.CompletedProcess([], 0, "", "")), patch.object(host, "systemctl") as control:
            with self.assertRaises(FileNotFoundError):
                host.restore_journal(self.root, document)
        control.assert_not_called()

    def test_global_lock_rejects_parallel_deployment_or_recovery(self):
        with host.operation_lock(self.directory):
            with self.assertRaises(host.MaintenanceBusy):
                with host.operation_lock(self.directory):
                    self.fail("a second operation entered the maintenance window")
        with host.operation_lock(self.directory):
            pass

    def test_control_subprocess_inherits_the_operation_lock(self):
        with host.operation_lock(self.directory) as descriptor:
            host.command(sys.executable, "-c", "import fcntl,sys; fcntl.fcntl(int(sys.argv[1]),fcntl.F_GETFD)", str(descriptor))

    def test_running_child_retains_lock_after_deployer_context_exits(self):
        with host.operation_lock(self.directory):
            child = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.buffer.read(1)"],
                                     stdin=subprocess.PIPE, pass_fds=host.inherited_operation_fds())
        try:
            with self.assertRaises(host.MaintenanceBusy):
                with host.operation_lock(self.directory):
                    self.fail("recovery raced a child left by the deployer")
        finally:
            child.communicate(b"x", timeout=5)
        with host.operation_lock(self.directory):
            pass

    def test_control_timeout_reaps_descendants_holding_the_maintenance_lock(self):
        with host.operation_lock(self.directory) as descriptor:
            program = "import os,subprocess,sys,time; child=subprocess.Popen(['sleep','60'],pass_fds=(int(sys.argv[1]),)); print('spawned',child.pid,flush=True); time.sleep(60)"
            with self.assertRaises(subprocess.TimeoutExpired) as failure:
                host.command(sys.executable, "-c", program, str(descriptor), timeout=1)
            self.assertIn("spawned", failure.exception.stdout)
        # A live grandchild would still own the inherited flock here.
        with host.operation_lock(self.directory):
            pass

    def test_lease_age_never_releases_active_work(self):
        path = self.directory / "leases/glm"
        with path.open("w") as stream:
            fcntl.flock(stream, fcntl.LOCK_SH)
            os.utime(path, (1, 1))
            self.assertTrue(host.leases_held(self.directory))
        self.assertFalse(host.leases_held(self.directory))

    def test_execution_queue_directory_is_not_a_request_lease(self):
        queue = self.directory / "leases/execution-queue"
        queue.mkdir()
        self.assertFalse(host.leases_held(self.directory))
        with (self.directory / "leases/glm").open("w") as stream:
            fcntl.flock(stream, fcntl.LOCK_SH)
            self.assertTrue(host.leases_held(self.directory))

    def test_non_regular_lease_entry_fails_closed(self):
        path = self.directory / "leases/unexpected"
        path.symlink_to(self.directory / "leases/glm")
        with self.assertRaisesRegex(RuntimeError, "non-regular lease entry"):
            host.leases_held(self.directory)

    def test_interrupted_deployment_defers_rollback_for_protected_requests(self):
        journal = self.directory / "deployment.json"
        journal.write_text("{}")
        with (self.directory / "leases/image").open("w") as stream:
            fcntl.flock(stream, fcntl.LOCK_SH)
            with patch.object(host, "restore_journal") as restore:
                with self.assertRaisesRegex(host.MaintenanceBusy, "active requests"):
                    host.recover_pending(self.root)
        restore.assert_not_called()
        self.assertTrue(journal.exists())

    def test_installed_helper_survives_bad_candidate_source(self):
        source, target = self.root / "helper.py", self.root / "libexec/helper.py"
        source.write_text("version = 'good'\n")
        host.install_helper(source, target)
        source.write_text("this candidate has invalid python syntax\n")
        host.install_helper(source, target, bootstrap_only=True)
        self.assertFalse(target.is_symlink())
        self.assertEqual(target.read_text(), "version = 'good'\n")
        with self.assertRaises(SyntaxError):
            host.install_helper(source, target)
        self.assertEqual(target.read_text(), "version = 'good'\n")

    def test_boot_creates_transient_dirs_and_reports_missing_dependency(self):
        missing = self.root / "adapters/venv/bin/python"
        missing.unlink()
        with patch.object(host, "systemctl", side_effect=self.result), patch.object(host, "command", side_effect=self.result):
            errors = host.preflight(self.root)
        self.assertTrue(any(str(missing) in error for error in errors))
        status = json.loads((self.directory / "boot-status.json").read_text())
        self.assertFalse(status["ready"])
        self.assertTrue((self.root / "comfyui/state/temp").is_dir())

    def test_stale_profile_fails_closed_and_valid_profile_passes(self):
        path = self.directory / "memory-profiles.json"
        self.assertEqual(host.profile_errors(path), [])
        Path(self.profiles["sources"][0]["path"]).write_bytes(b"different weights")
        self.assertIn("stale", host.profile_errors(path)[0])

    def test_invalid_worker_memory_contract_fails_preflight(self):
        path = self.directory / "memory-profiles.json"
        self.profiles["resources"]["fixture"]["memory"]["mandatory"]["gpu_bytes"] = 2
        path.write_text(json.dumps(self.profiles))
        self.assertIn("does not match", host.profile_errors(path)[0])

    def test_pending_journal_rolls_back_even_when_candidate_imports_and_linker_are_broken(self):
        repo = self.root / "inference"
        repo.mkdir()
        def git(*args):
            return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
        git("init", "-q")
        git("config", "user.email", "fixture@example.invalid")
        git("config", "user.name", "fixture")
        source = repo / "hosts/redqueen"
        (source / "systemd").mkdir(parents=True)
        for unit in host.ALL_UNITS:
            (source / "systemd" / unit).write_text("previous unit")
        linker = source / "link-runtime.sh"
        marker = self.root / "linked-previous"
        linker.write_text(f"#!/bin/sh\nprintf good > '{marker}'\n")
        (repo / "broken_candidate.py").write_text("previous = True\n")
        git("add", ".")
        git("commit", "-qm", "last good")
        previous = git("rev-parse", "HEAD")
        linker.write_text("#!/bin/sh\nexit 1\n")
        (repo / "broken_candidate.py").write_text("raise ImportError('bad candidate')\n")
        git("add", ".")
        git("commit", "-qm", "bad candidate")
        profile = self.directory / "memory-profiles.json"
        backup = self.directory / "previous.json"
        backup.write_bytes(b"previous profiles")
        host.begin_journal(self.directory, repo, previous,
                           {"llama-swap.service": True, "comfyui.service": False},
                           ["llama-swap.service"], profile, backup)
        with patch.object(host, "systemctl", side_effect=self.result) as control:
            self.assertTrue(host.recover_pending(self.root))
        self.assertEqual(git("rev-parse", "HEAD"), previous)
        self.assertEqual(marker.read_text(), "good")
        self.assertEqual(profile.read_bytes(), b"previous profiles")
        self.assertFalse((self.directory / "deployment.json").exists())
        control.assert_any_call("enable", "llama-swap.service")
        control.assert_any_call("disable", "comfyui.service")
        control.assert_any_call("start", "--no-block", "llama-swap.service")
        self.assertEqual(json.loads((self.directory / "recovery.json").read_text())["state"], "rolled_back")

    def test_control_failure_keeps_pending_journal_for_retry(self):
        journal = self.directory / "deployment.json"
        journal.write_text('{"version":1}')
        with patch.object(host, "restore_journal", side_effect=subprocess.TimeoutExpired("systemctl", 35)):
            with self.assertRaises(subprocess.TimeoutExpired):
                host.recover_pending(self.root)
        self.assertTrue(journal.exists())
        self.assertEqual(json.loads((self.directory / "recovery.json").read_text())["state"], "quarantined")


class SupervisorTest(HostFixture):
    def supervise(self, state, now, control):
        with patch.object(host, "front_alive", return_value=False), patch.object(host, "systemctl", side_effect=control):
            return host.supervise_once(self.root, state, now=now)

    def test_restart_after_three_failures_then_cooldown_without_storm(self):
        control = Mock(side_effect=self.result)
        state = {}
        for now in (0, 15):
            state = self.supervise(state, now, control)
            self.assertEqual(control.call_count, 0)
        state = self.supervise(state, 30, control)
        self.assertEqual(sum(call.args[0] == "restart" for call in control.call_args_list), 1)
        for now in (45, 60, 75):
            state = self.supervise(state, now, control)
        self.assertEqual(sum(call.args[0] == "restart" for call in control.call_args_list), 1)
        self.assertEqual(state["fronts"]["llama-swap.service"]["next_retry"], 90)

    def test_healthy_liveness_does_not_use_dependency_readiness(self):
        with patch.object(host, "front_alive", return_value=True), patch.object(host, "systemctl") as control:
            state = host.supervise_once(self.root, {}, now=0)
        self.assertTrue(all(info["state"] == "ready" for info in state["fronts"].values()))
        control.assert_not_called()
        self.assertTrue(all(url.endswith(("/healthz", "/health")) for url in host.FRONT_URLS.values()))

    def test_supervisor_restores_comfyui_from_a_crash_persistent_marker(self):
        marker = self.directory / "comfyui-restore.json"
        host.atomic_json(marker, {
            "version": 1, "unit": "comfyui.service", "was_active": True, "pid": 123,
        })
        def control(*args, **kwargs):
            if args[:2] == ("show", "comfyui.service"):
                return subprocess.CompletedProcess(args, 0, "active\n", "")
            return self.result(*args, **kwargs)

        with patch.object(host, "front_alive", return_value=True), patch.object(
            host, "systemctl", side_effect=control
        ), patch.object(host, "comfyui_guard_ready", return_value=True):
            state = host.supervise_once(self.root, {}, now=0)
        self.assertFalse(marker.exists())
        self.assertNotIn("comfyui_restore", state)

    def test_supervisor_starts_stopped_comfyui_before_clearing_restore_marker(self):
        marker = self.directory / "comfyui-restore.json"
        host.atomic_json(marker, {
            "version": 1, "unit": "comfyui.service", "was_active": True, "pid": 123,
        })
        active_states = iter(("inactive\n", "active\n"))

        def control(*args, **kwargs):
            if args[0] == "show":
                return subprocess.CompletedProcess(args, 0, next(active_states), "")
            return subprocess.CompletedProcess(args, 0, "", "")

        with patch.object(host, "systemctl", side_effect=control) as calls, patch.object(
            host, "comfyui_guard_ready", return_value=True
        ):
            state = {}
            host.restore_comfyui_if_requested(self.root, state, now=0)
        self.assertFalse(marker.exists())
        self.assertNotIn("comfyui_restore", state)
        self.assertTrue(any(call.args[:2] == ("start", "comfyui.service") for call in calls.call_args_list))

    def test_comfyui_restore_waits_for_inference_leases(self):
        marker = self.directory / "comfyui-restore.json"
        host.atomic_json(marker, {
            "version": 1, "unit": "comfyui.service", "was_active": True, "pid": 123,
        })
        with (self.directory / "leases/active-request").open("w") as active:
            fcntl.flock(active, fcntl.LOCK_SH)
            control = Mock(side_effect=self.result)
            state = {}
            with patch.object(host, "systemctl", side_effect=control):
                host.restore_comfyui_if_requested(self.root, state, now=0)
        self.assertTrue(marker.exists())
        self.assertEqual(state["comfyui_restore"]["state"], "deferred")
        control.assert_not_called()

    def test_active_model_defers_every_restart_without_expiring_lease(self):
        control = Mock(side_effect=self.result)
        with (self.directory / "leases/slow-prefill").open("w") as stream:
            fcntl.flock(stream, fcntl.LOCK_SH)
            state = {}
            for now in (0, 15, 30, 1000000):
                state = self.supervise(state, now, control)
        control.assert_not_called()
        self.assertTrue(all(info["state"] == "deferred" for info in state["fronts"].values()))
        self.assertIn("active inference", state["fronts"]["llama-swap.service"]["reason"])

    def test_parallel_maintenance_defers_recovery(self):
        control = Mock(side_effect=self.result)
        state = {}
        with host.operation_lock(self.directory):
            for now in (0, 15, 30):
                state = self.supervise(state, now, control)
        control.assert_not_called()
        self.assertTrue(all(info["state"] == "deferred" for info in state["fronts"].values()))

    def test_absent_runtime_is_quarantined_until_it_returns(self):
        (self.root / "adapters/venv/bin/python").unlink()
        control = Mock(side_effect=self.result)
        state = {}
        with patch.object(host, "front_alive", side_effect=lambda item: item[0] != "qwen-image-adapter.service"), patch.object(host, "systemctl", side_effect=control):
            for now in (0, 15, 30):
                state = host.supervise_once(self.root, state, now=now)
        info = state["fronts"]["qwen-image-adapter.service"]
        self.assertEqual(info["state"], "quarantined")
        self.assertIn("missing runtime", info["reason"])
        self.assertNotIn(unittest.mock.call("restart", "qwen-image-adapter.service"), control.call_args_list)

    def test_systemctl_failure_is_bounded_and_retried_after_cooldown(self):
        control = Mock(side_effect=subprocess.TimeoutExpired("systemctl", 35))
        state = {}
        for now in (0, 15, 30):
            state = self.supervise(state, now, control)
        self.assertEqual(state["fronts"]["llama-swap.service"]["state"], "cooling_down")
        self.assertTrue((self.directory / "maintenance.json").exists())
        calls = sum(call.args[0] == "restart" for call in control.call_args_list)
        self.supervise(state, 45, control)
        self.assertEqual(sum(call.args[0] == "restart" for call in control.call_args_list), calls)

    def test_queued_systemd_job_keeps_maintenance_until_front_is_healthy_and_job_finished(self):
        host.atomic_json(self.directory / "maintenance.json", {"unit": "llama-swap.service", "state": "recovering"})
        def pending(*args, **kwargs):
            return subprocess.CompletedProcess(args, 0, "1234", "")
        with patch.object(host, "front_alive", return_value=True), patch.object(host, "systemctl", side_effect=pending):
            state = host.supervise_once(self.root, {}, now=0)
        self.assertTrue((self.directory / "maintenance.json").exists())
        self.assertIsNotNone(state["maintenance"])
        with patch.object(host, "front_alive", return_value=True), patch.object(host, "systemctl", side_effect=self.result):
            state = host.supervise_once(self.root, state, now=15)
        self.assertFalse((self.directory / "maintenance.json").exists())
        self.assertIsNone(state["maintenance"])

    def test_uncertain_loading_front_does_not_clear_a_finished_job_marker(self):
        host.atomic_json(self.directory / "maintenance.json", {"unit": "llama-swap.service", "state": "recovering"})
        def activating(*args, **kwargs):
            output = "0" if "--property=Job" in args else "ActiveState=activating\nMainPID=0\n"
            return subprocess.CompletedProcess(args, 0, output, "")
        with patch.object(host, "front_alive", return_value=False), patch.object(host, "systemctl", side_effect=activating) as control:
            state = host.supervise_once(self.root, {}, now=0)
        self.assertTrue((self.directory / "maintenance.json").exists())
        self.assertFalse(any(call.args[0] == "restart" for call in control.call_args_list))

    def test_rebooted_definitely_dead_front_releases_stale_maintenance(self):
        host.atomic_json(self.directory / "maintenance.json", {"unit": "llama-swap.service", "state": "recovering", "boot_id": "previous-boot"})
        def stopped(*args, **kwargs):
            output = "0" if "--property=Job" in args else "ActiveState=inactive\nMainPID=0\n"
            return subprocess.CompletedProcess(args, 0, output, "")
        with patch.object(host, "front_alive", return_value=False), patch.object(host, "systemctl", side_effect=stopped):
            state = host.supervise_once(self.root, {}, now=0)
        self.assertFalse((self.directory / "maintenance.json").exists())
        self.assertIsNone(state["maintenance"])


class DeploymentSafetyTest(unittest.TestCase):
    def setUp(self):
        from redqueen_adapters import deploy_scheduler
        self.deploy = deploy_scheduler
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name)
        self.home = self.repo / "home"
        home_patch = patch.object(Path, "home", return_value=self.home)
        home_patch.start()
        self.addCleanup(home_patch.stop)
        self.source = self.repo / "hosts/redqueen"
        (self.source / "model-router").mkdir(parents=True)
        (self.source / "systemd").mkdir()
        for unit in self.deploy.FRONTS + self.deploy.MANAGED:
            (self.source / "systemd" / unit).write_text("fixture")
        self.directory = self.repo / "run"
        self.directory.mkdir()
        self.profile = self.directory / "profiles.json"
        self.profile.write_text("old profiles")
        self.policy = {"profile_path": str(self.profile)}
        (self.source / "model-router/scheduler.json").write_text(json.dumps(self.policy))
        self.previous, self.candidate = "1" * 40, "2" * 40
        self.pulled = False
        self.controls = []
        self.lease = Mock(path=self.directory / "leases/a")

    def run_command(self, *args, **kwargs):
        output = ""
        if args[0] == "git":
            if args[3:5] == ("branch", "--show-current"):
                output = "develop"
            elif args[3:6] == ("rev-parse", "--abbrev-ref", "@{upstream}"):
                output = "origin/develop"
            elif args[3:5] == ("rev-parse", "HEAD"):
                output = self.candidate if self.pulled else self.previous
            elif args[3:5] == ("rev-parse", "--verify"):
                output = self.previous
            elif args[3] == "pull":
                journal = json.loads((self.directory / "deployment.json").read_text())
                self.assertEqual(journal["phase"], "updating_source")
                self.assertEqual(journal["previous"], self.previous)
                self.assertEqual(Path(journal["profile_backup"]).read_text(), "old profiles")
                self.assertTrue((self.repo / "libexec/host_reliability.py").is_file())
                unit = self.home / ".config/systemd/user/redqueen-preflight.service"
                self.assertTrue(unit.is_file())
                self.assertFalse(unit.is_symlink())
                self.assertIn(("enable", "redqueen-preflight.service"), self.controls)
                self.pulled = True
            elif args[3:5] == ("switch", "--detach"):
                self.pulled = False
        return subprocess.CompletedProcess(args, 0, output, "")

    def control(self, *args, **kwargs):
        self.controls.append(args)
        output, code = "", 0
        if args[0] in {"is-active", "is-enabled"}:
            code = 0 if args[1] == "llama-swap.service" else 1
        elif args[0] == "show":
            output = "loaded"
        return subprocess.CompletedProcess(args, code, output, "")

    def execute(self, *, drain=None, validate=None, update=True, check_only=False, control=None):
        with patch.object(self.deploy.os, "geteuid", return_value=1002), patch.dict("os.environ", {"QWEN_API_KEY": "fixture"}), patch.object(self.deploy, "run", side_effect=self.run_command), patch.object(self.deploy, "systemctl", side_effect=control or self.control), patch.object(host, "systemctl", side_effect=control or self.control), patch.object(self.deploy, "validate", side_effect=validate or [(self.policy, {"resources": {"a": {}}}), (self.policy, {"resources": {"a": {}}})]), patch.object(self.deploy, "drain", new=drain or AsyncMock(return_value=[self.lease])), patch.object(self.deploy, "healthy", new=AsyncMock()):
            self.deploy.deploy(self.repo, self.previous, update=update, check_only=check_only)

    def test_update_journals_and_installs_independent_recovery_before_git_pull(self):
        self.execute()
        self.assertTrue(self.pulled)
        self.assertFalse((self.directory / "deployment.json").exists())
        self.assertEqual(json.loads((self.directory / "last-good.json").read_text())["revision"], self.candidate)
        self.lease.close.assert_called_once()

    def test_updated_profiles_use_the_canonical_candidate_profiler(self):
        profiler = self.source / "adapters/redqueen_adapters/memory_profile.py"
        profiler.parent.mkdir(parents=True)
        profiler.write_text("def build_profiles(router, policy):\n    return {'resources': {'a': {}}, 'profiler_path': __file__}\n")
        with patch.object(self.deploy, "run", side_effect=self.run_command), patch.object(self.deploy, "build_profiles", side_effect=AssertionError("temporary staged profiler used")):
            _, document = self.deploy.validate(self.source, fresh_profiler=True)
        self.assertEqual(Path(document["profiler_path"]), profiler)

    def test_disk_full_before_journaling_does_not_update_source_or_stop_services(self):
        with patch.object(host, "atomic_write", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.execute()
        self.assertFalse(self.pulled)
        self.assertFalse(any(args[0] == "stop" for args in self.controls))
        self.assertEqual(self.profile.read_text(), "old profiles")
        self.lease.close.assert_called_once()

    def test_failed_drain_never_pulls_or_stops_active_work(self):
        with self.assertRaisesRegex(RuntimeError, "did not drain"):
            self.execute(drain=AsyncMock(side_effect=RuntimeError("active inference did not drain")))
        self.assertFalse(self.pulled)
        self.assertFalse(any(args[0] == "stop" for args in self.controls))

        self.assertEqual(self.profile.read_text(), "old profiles")
        self.assertFalse((self.directory / "deployment.json").exists())
        with host.operation_lock(self.directory):
            pass

    def test_bad_candidate_after_pull_restores_source_profiles_and_units(self):
        with self.assertRaisesRegex(ValueError, "broken candidate"):
            self.execute(validate=[(self.policy, {"resources": {"a": {}}}), ValueError("broken candidate")])
        self.assertFalse(self.pulled)
        self.assertEqual(self.profile.read_text(), "old profiles")
        self.assertIn(("start", "llama-swap.service"), self.controls)
        self.assertFalse((self.directory / "deployment.json").exists())
        self.lease.close.assert_called_once()

    def test_check_does_not_mutate_services_journal_or_approved_profiles(self):
        self.execute(update=False, check_only=True)
        self.assertEqual(self.controls, [])
        self.assertFalse((self.directory / "operations.lock").exists())
        self.assertFalse((self.directory / "deployment.json").exists())
        self.assertEqual(self.profile.read_text(), "old profiles")

    def test_state_capture_failure_releases_drained_leases(self):
        def fail(*args, **kwargs):
            if args[0] == "is-enabled":
                raise subprocess.TimeoutExpired("systemctl", 35)
            return self.control(*args, **kwargs)
        with self.assertRaises(subprocess.TimeoutExpired):
            self.execute(control=fail)
        self.lease.close.assert_called_once()
        self.assertFalse(self.pulled)
        self.assertFalse((self.directory / "deployment.json").exists())

    def test_active_legacy_backend_is_not_killed_without_request_ownership(self):
        def legacy(*args, **kwargs):
            if args[:2] == ("is-active", "qwen38.service"):
                return subprocess.CompletedProcess(args, 0, "active", "")
            return self.control(*args, **kwargs)
        with self.assertRaisesRegex(RuntimeError, "legacy qwen38"):
            self.execute(control=legacy)
        self.assertFalse(self.pulled)
        self.assertFalse(any(args[0] == "stop" for args in self.controls))


class DrainSafetyTest(unittest.IsolatedAsyncioTestCase):
    async def test_unknown_backend_activity_never_counts_as_idle(self):
        from redqueen_adapters import deploy_scheduler as deploy
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            policy = {"socket_path": str(directory / "scheduler.sock"), "upstream_url": "http://fixture"}
            runtime = Mock(snapshot=AsyncMock(return_value={"a": "ready"}), busy=AsyncMock(return_value=None))
            response = Mock()
            request = Mock(__aenter__=AsyncMock(return_value=response), __aexit__=AsyncMock(return_value=False))
            session = Mock(get=Mock(return_value=request))
            context = Mock(__aenter__=AsyncMock(return_value=session), __aexit__=AsyncMock(return_value=False))
            with patch.object(deploy, "Runtime", return_value=runtime), patch.object(deploy.aiohttp, "ClientSession", return_value=context), patch.object(deploy, "time", Mock(monotonic=Mock(side_effect=[0, 1]))), patch.object(deploy.asyncio, "sleep", new=AsyncMock()):
                with self.assertRaisesRegex(RuntimeError, "did not drain"):
                    await deploy.drain(policy, {"resources": {"a": {}}}, "fixture", 0.5)
            runtime.busy.assert_awaited_once_with("a")
            # Failed drain must close every exclusive resource lock.
            with (directory / "leases/a").open("r+") as stream:
                fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)


if __name__ == "__main__":
    unittest.main()
