import importlib.util
import json
from pathlib import Path
import signal
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location("safe_run", Path(__file__).parents[1] / "safe_run.py")
sr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sr)


class SafeRunTests(unittest.TestCase):
    def args(self, *options):
        return sr.parse_args([*options, "--", "python3", "train.py"])

    def healthy(self):
        return {"collected_at": time.time(), "admission": {"blocked": False, "essential_telemetry_missing": []}, "alerts": []}

    def test_memory_units_and_limits(self):
        self.assertEqual(sr.parse_size("1.5GiB"), int(1.5 * sr.GIB))
        self.assertEqual(sr.parse_size("512M"), 512 * sr.MIB)
        for options in [("--memory", "23G"), ("--cpu", "17"), ("--name", "x;touch_bad")]:
            with self.subTest(options=options), self.assertRaises(SystemExit):
                self.args(*options)

    def test_command_is_argv_without_shell_interpolation(self):
        argument = "$(touch /tmp/should-not-exist); echo danger"
        args = sr.parse_args(["--", "printf", "%s", argument])
        result = sr.build_command(args, "wg-test.scope")
        self.assertEqual(result[-4:], ["--", "printf", "%s", argument])
        self.assertIn("--property=CPUQuota=800%", result)
        self.assertIn(f"--property=MemoryMax={12 * sr.GIB}", result)

    def test_missing_command_separator_is_rejected(self):
        with self.assertRaises(SystemExit):
            sr.parse_args(["echo", "hi"])

    def test_health_stale_future_missing_and_critical(self):
        sr.check_health(self.healthy(), time.time())
        for change in ({"collected_at": time.time() - 121}, {"collected_at": time.time() + 60}, {"admission": {}}, {"alerts": [{"severity": "critical"}]}):
            status = self.healthy()
            status.update(change)
            with self.subTest(change=change), self.assertRaises(sr.AdmissionError):
                sr.check_health(status, time.time())

    def test_gpu_telemetry_missing_does_not_block_cpu(self):
        status = self.healthy()
        status["admission"]["essential_telemetry_missing"] = ["gpu"]
        sr.check_health(status, time.time(), gpu=False)
        with self.assertRaises(sr.AdmissionError):
            sr.check_health(status, time.time(), gpu=True)

    def test_blocked_admission_is_fail_closed(self):
        status = self.healthy()
        status["admission"] = {"blocked": True, "reasons": ["thermal"]}
        with self.assertRaisesRegex(sr.AdmissionError, "thermal"):
            sr.check_health(status, time.time())

    def test_gpu_specific_admission_blocks_only_gpu(self):
        status = self.healthy()
        status["admission"].update(gpu_blocked=True, gpu_reasons=["GPU hot"])
        status["alerts"] = [{"severity": "critical", "key": "gpu_temperature_0", "message": "GPU hot"}]
        sr.check_health(status, time.time(), gpu=False)
        with self.assertRaisesRegex(sr.AdmissionError, "GPU hot"):
            sr.check_health(status, time.time(), gpu=True)

    def test_gpu_xid_critical_alert_still_blocks_cpu(self):
        status = self.healthy()
        status["alerts"] = [{"severity": "critical", "key": "kernel_gpu_xid"}]
        with self.assertRaises(sr.AdmissionError):
            sr.check_health(status, time.time(), gpu=False)

    def test_memory_reserve_and_concurrent_budget(self):
        g = sr.GIB
        sr.check_memory(8*g, 22*g, 4*g, 12*g)
        for available, current, reserved in [(13*g, 0, 0), (20*g, 15*g, 0), (20*g, 0, 15*g)]:
            with self.subTest(values=(available,current,reserved)), self.assertRaises(sr.AdmissionError):
                sr.check_memory(8*g, available, current, reserved)

    def test_idle_existing_reservation_cannot_consume_host_reserve(self):
        # First job reserved 12GiB but has not allocated it; host has 21GiB free.
        # A second 8GiB job must leave room for first job's growth plus 6GiB.
        with self.assertRaises(sr.AdmissionError):
            sr.check_memory(8*sr.GIB, 21*sr.GIB, 0, 12*sr.GIB)

    def test_gpu_busy_compute_memory_temperature_and_bad_values(self):
        for response in ["12000, 21, 50", "4000, 0, 50", "12000, 0, 82", "N/A, 0, 50", "nan, 0, 50"]:
            with self.subTest(response=response), patch.object(sr, "run_query", return_value=response), self.assertRaises(sr.AdmissionError):
                sr.gpu_preflight(4*sr.GIB)
        with patch.object(sr, "run_query", side_effect=["12000, 0, 50", "1234\n"]), self.assertRaises(sr.AdmissionError):
            sr.gpu_preflight(4*sr.GIB)
        with patch.object(sr, "run_query", side_effect=["12000, 0, 50", ""]):
            sr.gpu_preflight(4*sr.GIB)

    def test_fallback_does_not_claim_nonexistent_quota(self):
        with patch.object(sr.shutil, "which", return_value="/usr/bin/ionice"):
            result = sr.build_command(self.args(), "wg-test.scope", {"cpu_quota": False, "io_weight": False})
        self.assertFalse(any("CPUQuota=" in x or "IOWeight=" in x for x in result))
        self.assertEqual(result[-7:], ["--", "/usr/bin/ionice", "-c", "3", "--", "python3", "train.py"])

    def test_orphan_scope_reservation_is_not_removed(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            record = directory / "orphan.json"
            record.write_text(json.dumps({"pid": 99999999, "start_time": "1", "unit": "wg-orphan.scope", "memory": 8*sr.GIB, "gpu": True}))
            with patch.object(sr, "process_identity", return_value=None), patch.object(sr, "unit_properties", return_value={"LoadState": "loaded", "ActiveState": "active"}):
                self.assertEqual(len(sr.active_reservations(directory)), 1)
            self.assertTrue(record.exists())
            with patch.object(sr, "process_identity", return_value=None), patch.object(sr, "unit_properties", return_value={"LoadState": "not-found", "ActiveState": "inactive"}):
                self.assertEqual(sr.active_reservations(directory), [])
            self.assertFalse(record.exists())

    def test_signal_exit_status_matches_shell(self):
        self.assertEqual(sr.exit_code(7), 7)
        self.assertEqual(sr.exit_code(-signal.SIGTERM), 143)
        self.assertEqual(sr.exit_code(-signal.SIGINT), 130)

    def test_supervisor_keeps_scope_until_background_children_exit(self):
        process = Mock()
        process.wait.return_value = 7
        states = [{"LoadState": "loaded", "ActiveState": "active"}, {"LoadState": "not-found", "ActiveState": "inactive"}]
        with patch.object(sr, "unit_properties", side_effect=states), patch.object(sr.time, "sleep") as sleep:
            self.assertEqual(sr.supervise(process, "wg-test.scope"), 7)
        sleep.assert_called_once_with(1)

    def test_private_wrapper_sets_only_own_group_before_exec(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            group = root / sr.SLICE / "wg-test.scope"
            group.mkdir(parents=True)
            maximum = 256 * sr.MIB
            for name, value in {"memory.max": maximum, "memory.high": int(maximum*.8), "memory.swap.max": sr.GIB, "pids.max": 2048, "memory.oom.group": 0}.items():
                (group / name).write_text(str(value))
            (group.parent / "memory.oom.group").write_text("0")
            original_read = Path.read_text
            def fake_read(path, *args, **kwargs):
                if str(path) == "/proc/self/cgroup":
                    return f"0::/{sr.SLICE}/wg-test.scope\n"
                return original_read(path, *args, **kwargs)
            with patch.object(sr, "CGROUP_ROOT", root), patch.object(Path, "read_text", fake_read), patch.object(sr.os, "execvp") as execute:
                sr.exec_workload(["wg-test.scope", str(maximum), "--", "echo", "ok"])
                execute.assert_called_once_with("echo", ["echo", "ok"])
                self.assertEqual((group / "memory.oom.group").read_text().strip(), "1")
                self.assertEqual((group.parent / "memory.oom.group").read_text(), "0")
                with self.assertRaises(sr.AdmissionError):
                    sr.exec_workload(["wg-other.scope", str(maximum), "--", "echo", "ok"])
                (group / "memory.max").write_text("max")
                with self.assertRaises(sr.AdmissionError):
                    sr.exec_workload(["wg-test.scope", str(maximum), "--", "echo", "ok"])

    def test_dry_run_never_queries_or_starts_work(self):
        with patch.object(sr, "run_query", side_effect=AssertionError("unexpected query")), patch.object(sr.subprocess, "Popen", side_effect=AssertionError("unexpected launch")):
            self.assertEqual(sr.main(["--dry-run", "--", "echo", "hello"]), 0)


if __name__ == "__main__":
    unittest.main()
