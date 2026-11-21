import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("monitor", Path(__file__).parents[1] / "monitor.py")
monitor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(monitor)


class AlertTests(unittest.TestCase):
    def setUp(self):
        self.state = {}
        self.condition = {"cpu_temperature": {"severity": "warning", "message": "hot", "immediate": False}}

    def sample(self, conditions, at=1000):
        return monitor.update_alerts(self.state, conditions, monitor.DEFAULTS, at)

    def test_debounce_recovery_and_cooldown(self):
        self.assertEqual(self.sample(self.condition)[0], [])
        self.assertEqual(self.sample(self.condition, 1030)[0], [])
        active, events = self.sample(self.condition, 1060)
        self.assertEqual(len(active), 1)
        self.assertEqual(events[0]["kind"], "activated")
        self.assertEqual(self.sample(self.condition, 1090)[1], [])
        self.assertEqual(self.sample(self.condition, 1960)[1][0]["kind"], "reminder")
        self.assertEqual(len(self.sample({}, 1990)[0]), 1)
        self.assertEqual(len(self.sample({}, 2020)[0]), 1)
        active, events = self.sample({}, 2050)
        self.assertEqual(active, [])
        self.assertEqual(events[0]["kind"], "recovered")

    def test_critical_escalates_immediately_and_needs_full_recovery(self):
        for _ in range(3):
            self.sample(self.condition)
        self.condition["cpu_temperature"]["severity"] = "critical"
        active, events = self.sample(self.condition, 1010)
        self.assertEqual(events[0]["kind"], "escalated")
        self.assertTrue(monitor.admission_status(active)["blocked"])
        self.condition["cpu_temperature"]["severity"] = "warning"
        active, _ = self.sample(self.condition, 1040)
        self.assertEqual(active[0]["severity"], "critical")

    def test_explicit_hardware_alarm_immediate(self):
        self.condition["cpu_temperature"]["immediate"] = True
        self.assertEqual(len(self.sample(self.condition)[0]), 1)

    def test_missing_telemetry_debounced_and_gpu_scoped(self):
        missing = {"telemetry_gpu": {"severity": "warning", "message": "missing"}}
        for _ in range(2):
            self.assertEqual(self.sample(missing)[0], [])
        active, _ = self.sample(missing)
        self.assertFalse(monitor.admission_status(active)["blocked"])
        self.assertEqual(monitor.admission_status(active)["essential_telemetry_missing"], ["gpu"])
        missing["telemetry_memory"] = {"severity": "warning", "message": "missing"}
        for _ in range(3):
            active, _ = self.sample(missing)
        self.assertTrue(monitor.admission_status(active)["blocked"])

    def test_bad_samples_reset_after_healthy_sample(self):
        self.sample(self.condition)
        self.sample(self.condition)
        self.sample({})
        self.assertEqual(self.sample(self.condition)[0], [])

    def test_missing_sensor_does_not_claim_health_recovered(self):
        for _ in range(3):
            self.sample(self.condition)
        keys = monitor.unavailable_condition_keys(self.state, ["cpu"])
        for _ in range(4):
            active, events = monitor.update_alerts(self.state, {}, monitor.DEFAULTS, 1100, keys)
            self.assertEqual(len(active), 1)
            self.assertEqual(events, [])

    def test_raw_admission_blocks_before_notification_debounce(self):
        cases = [("cpu_temperature", "warning"), ("memory_available", "warning"),
                 ("nvme_temperature_nvme0", "critical"), ("psi_io", "critical"),
                 ("disk_space_/", "critical")]
        for key, severity in cases:
            with self.subTest(key=key):
                raw = {key: {"severity": severity, "message": "test"}}
                active, _ = monitor.update_alerts({}, raw, monitor.DEFAULTS, 1000)
                self.assertEqual(active, [])
                self.assertTrue(monitor.admission_status(active, raw)["blocked"])
        self.assertFalse(monitor.admission_status([], {"nvme_temperature_nvme0": {
            "severity": "warning", "message": "warm"}})["blocked"])

    def test_immediate_missing_telemetry_and_gpu_admission_scope(self):
        for source in ("cpu", "memory"):
            status = monitor.admission_status([], {}, [source])
            self.assertTrue(status["blocked"])
            self.assertEqual(status["essential_telemetry_missing"], [source])
        status = monitor.admission_status([], {}, ["gpu"])
        self.assertFalse(status["blocked"])
        self.assertTrue(status["gpu_blocked"])
        status = monitor.admission_status([], {"gpu_temperature_0": {"severity": "warning"}})
        self.assertFalse(status["blocked"])
        self.assertTrue(status["gpu_blocked"])


class CollectionTests(unittest.TestCase):
    def test_cursor_parsing_does_not_expose_message(self):
        cursor, faults = monitor.parse_journal('\n'.join([
            json.dumps({"__CURSOR": "s=abc", "MESSAGE": "NVRM: Xid (PCI:0000): 109 private filename"}),
            "not JSON", "-- cursor: s=def"]))
        self.assertEqual(cursor, "s=def")
        self.assertEqual(faults, ["gpu_xid"])

    def test_initial_journal_fault_is_baselined(self):
        output = json.dumps({"__CURSOR": "s=abc", "MESSAGE": "NVRM: Xid 109"})
        state = {}
        with mock.patch.object(monitor, "run_command", return_value=(0, output)):
            self.assertEqual(monitor.journal_sample(state, "boot-a"), ([], None))
            self.assertEqual(state["journal"]["cursor"], "s=abc")
            self.assertEqual(monitor.journal_sample(state, "boot-a"), (["gpu_xid"], None))

    def test_known_reboot_reports_early_boot_fault_and_pages_forward(self):
        state = {"journal": {"cursor": "old-boot-cursor", "boot_id": "boot-a"}}
        output = json.dumps({"__CURSOR": "boot-b-page-one", "MESSAGE": "NVRM: Xid 109"})
        with mock.patch.object(monitor, "journal_boot_head", return_value=(0, output)):
            self.assertEqual(monitor.journal_sample(state, "boot-b"), (["gpu_xid"], None))
        self.assertEqual(state["journal"], {"cursor": "boot-b-page-one", "boot_id": "boot-b"})
        with mock.patch.object(monitor, "run_command", return_value=(0, "-- cursor: boot-b-page-two")) as run:
            self.assertEqual(monitor.journal_sample(state, "boot-b"), ([], None))
        self.assertIn("boot-b-page-one", run.call_args.args[0])

    def test_lost_journal_cursor_rebaselines_with_visible_gap(self):
        state = {"journal": {"cursor": "old", "boot_id": "a"}}
        with mock.patch.object(monitor, "run_command", side_effect=[
                (1, ""), (0, json.dumps({"__CURSOR": "new"}))]):
            self.assertEqual(monitor.journal_sample(state, "a"), ([], "kernel_journal_unavailable"))
        self.assertEqual(state["journal"]["cursor"], "new")

    def test_secondary_nvme_sensor_is_visibility_only(self):
        metrics = {"nvme": [{"device": "nvme0", "composite_temp_c": 55,
                             "sensors": [{"label": "Sensor 2", "temperature_c": 84}],
                             "hardware_alarm": False, "manufacturer_limits_c": [84]}]}
        self.assertEqual(monitor.build_conditions(metrics, [], [], {}, monitor.DEFAULTS), {})
        metrics["nvme"][0]["hardware_alarm"] = True
        condition = monitor.build_conditions(metrics, [], [], {}, monitor.DEFAULTS)
        self.assertTrue(condition["nvme_temperature_nvme0"]["immediate"])

    def test_psi_parser(self):
        parsed = monitor.parse_psi("some avg10=1.2 avg60=4.5 avg300=0.2 total=400\nfull avg10=0 avg60=0 avg300=0 total=10")
        self.assertEqual(parsed["some"]["avg60"], 4.5)

    def test_core_only_throttle_increase_is_detected(self):
        previous = {"cpu": {"package_throttle_count": 0, "core_throttle_count_sum": 1}}
        current = {"cpu": {"package_throttle_count": 0, "core_throttle_count_sum": 2}}
        conditions = monitor.build_conditions(current, [], [], previous, monitor.DEFAULTS)
        self.assertTrue(conditions["cpu_thermal_throttle"]["immediate"])

    def test_test_notification_title_is_unambiguous(self):
        with mock.patch.object(monitor, "run_command", return_value=(0, "")) as run:
            monitor.notify({"kind": "test", "severity": "normal", "message": "test"}, {}, 1000)
        self.assertIn("工作站安全監控：測試通知", run.call_args.args[0])

    def test_many_events_use_one_notification_timeout_budget(self):
        events = [{"key": str(index), "kind": "activated", "severity": "warning", "message": "warm"}
                  for index in range(10)]
        events[-1]["severity"] = "critical"
        with mock.patch.object(monitor, "notify", return_value=False) as notify:
            monitor.notify_events(events, {}, 1000)
        self.assertEqual(notify.call_count, 1)
        self.assertEqual(notify.call_args.args[0]["severity"], "critical")
        self.assertTrue(all(not event["notification_accepted"] for event in events))
        self.assertTrue(all(event["notification_batched"] for event in events))


class PersistenceTests(unittest.TestCase):
    def test_atomic_json_and_bounded_rotation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "latest.json"
            monitor.atomic_json(path, {"example": "測試"})
            self.assertEqual(monitor.read_json(path, {}), {"example": "測試"})
            log = Path(temporary) / "history.jsonl"
            for i in range(20):
                monitor.append_log(log, {"index": i, "value": "x" * 30}, limit=100, backups=3)
            self.assertEqual(len(list(Path(temporary).glob("history.jsonl*"))), 4)
            self.assertEqual(json.loads(log.read_text())["index"], 19)

    def test_partial_config_preserves_thresholds(self):
        config = monitor.merge_dict(monitor.DEFAULTS, {"thresholds": {"cpu_temp_c": {"warning": 82}}})
        self.assertEqual(config["thresholds"]["cpu_temp_c"], {"warning": 82, "critical": 95})


if __name__ == "__main__":
    unittest.main()
