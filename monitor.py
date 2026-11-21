#!/usr/bin/env python3
"""Read-only workstation health sampling and desktop alerts (Python stdlib).

latest.json schema version 1: collected_at is UNIX seconds; boot_id is the Linux
boot identifier; metrics has memory, cpu, gpus, disks, nvme and psi; alerts holds
active {key, severity, message} entries. telemetry_errors contains short source
names, never raw journal messages. admission.blocked applies immediately to CPU
heat, low memory, critical storage/pressure and missing CPU/memory telemetry;
gpu_blocked/gpu_reasons scope GPU heat and missing GPU telemetry to GPU work.
essential_telemetry_missing lists current or not-yet-recovered telemetry gaps.
pending_conditions exposes observations still awaiting notification debounce.
Notification status records command acceptance only: notify-send cannot prove
that a person saw a notification. No component changes hardware or kills jobs.
"""

import argparse
import copy
import fcntl
import glob
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time


DEFAULTS = {
    "sample_interval_seconds": 30,
    "alerts": {"consecutive_samples": 3, "recovery_samples": 3,
               "cooldown_seconds": 900},
    "thresholds": {
        "cpu_temp_c": {"warning": 85, "critical": 95},
        "gpu_temp_c": {"warning": 80, "critical": 88},
        "nvme_temp_c": {"warning": 70, "critical": 80},
        "memory_available_bytes": {"warning": 4 * 1024**3, "critical": 2 * 1024**3},
        "disk_used_percent": {"warning": 85, "critical": 92},
        "psi_cpu_avg60": {"warning": 20, "critical": 40},
        "psi_memory_avg60": {"warning": 5, "critical": 15},
        "psi_io_avg60": {"warning": 10, "critical": 25},
    },
}
LOG_LIMIT = 5 * 1024 * 1024
SEVERITIES = {"warning": 1, "critical": 2}
FAULT_PATTERNS = {
    "gpu_xid": re.compile(r"NVRM:.*Xid\b", re.I),
    "machine_check": re.compile(r"\b(?:Machine check|Uncorrected hardware error|Hardware Error|MCE:.*(?:error|failure))\b", re.I),
    "io_fault": re.compile(r"\b(?:I/O error|critical medium error|EXT4-fs error|BTRFS.*(?:corrupt|error)|nvme.*(?:fatal|controller is down))\b", re.I),
}
FAULT_MESSAGES = {
    "gpu_xid": "核心回報 NVIDIA Xid 錯誤；請檢查 GPU 工作與驅動紀錄。",
    "machine_check": "核心回報硬體／機器檢查錯誤；請檢查硬體錯誤紀錄。",
    "io_fault": "核心回報儲存裝置或檔案系統錯誤；請檢查磁碟健康與備份。",
}


def merge_dict(base, update):
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def read_json(path, fallback):
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else copy.deepcopy(fallback)
    except (OSError, ValueError):
        return copy.deepcopy(fallback)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def append_log(path, value, limit=LOG_LIMIT, backups=3):
    path = Path(path)
    line = json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists() and path.stat().st_size + len(line.encode()) > limit:
        oldest = Path(str(path) + "." + str(backups))
        if oldest.exists():
            oldest.unlink()
        for index in range(backups - 1, 0, -1):
            source = Path(str(path) + "." + str(index))
            if source.exists():
                source.replace(str(path) + "." + str(index + 1))
        path.replace(str(path) + ".1")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
        handle.write(line)


def read_text(path):
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def number(path, divisor=1):
    try:
        return float(read_text(path)) / divisor
    except (TypeError, ValueError):
        return None


def run_command(argv, timeout=5):
    try:
        result = subprocess.run(argv, capture_output=True, text=True,
                                timeout=timeout, check=False)
        return result.returncode, result.stdout
    except (OSError, subprocess.TimeoutExpired):
        return None, ""


def collect_memory(path="/proc/meminfo"):
    text = read_text(path)
    if text is None:
        return None
    parsed = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                parsed[parts[0].rstrip(":")] = int(parts[1]) * 1024
            except ValueError:
                pass
    if not all(key in parsed for key in ("MemTotal", "MemAvailable")):
        return None
    return {"total_bytes": parsed["MemTotal"], "available_bytes": parsed["MemAvailable"],
            "swap_total_bytes": parsed.get("SwapTotal"),
            "swap_free_bytes": parsed.get("SwapFree")}


def parse_psi(text):
    values = {}
    for line in (text or "").splitlines():
        parts = line.split()
        if not parts:
            continue
        parsed = {}
        for pair in parts[1:]:
            if "=" in pair:
                key, value = pair.split("=", 1)
                try:
                    parsed[key] = float(value) if key != "total" else int(value)
                except ValueError:
                    pass
        if "avg60" in parsed:
            values[parts[0]] = parsed
    return values


def collect_temperatures():
    cpu_temps, nvme = [], []
    for directory in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        name = read_text(directory / "name")
        if name in ("coretemp", "k10temp", "zenpower"):
            for label_path in directory.glob("temp*_label"):
                label = read_text(label_path) or ""
                if label.startswith("Package id") or label in ("Tctl", "Tdie"):
                    value = number(str(label_path).replace("_label", "_input"), 1000)
                    if value is not None and 0 <= value <= 150:
                        cpu_temps.append(value)
        if name == "nvme":
            identifier = next((p.name for p in directory.resolve().parents
                               if re.fullmatch(r"nvme\d+", p.name)), directory.name)
            entry = {"device": identifier, "composite_temp_c": None, "sensors": [],
                     "hardware_alarm": False, "manufacturer_limits_c": []}
            for input_path in sorted(directory.glob("temp*_input")):
                prefix = str(input_path).removesuffix("_input")
                label = read_text(prefix + "_label") or input_path.stem
                value = number(input_path, 1000)
                entry["sensors"].append({"label": label, "temperature_c": value})
                if label.lower() == "composite" or input_path.name == "temp1_input":
                    entry["composite_temp_c"] = value
                    entry["hardware_alarm"] = any(number(prefix + suffix) == 1 for suffix in
                                                  ("_alarm", "_crit_alarm", "_max_alarm"))
                    entry["manufacturer_limits_c"] = [value for suffix in ("_max", "_crit")
                                                       if (value := number(prefix + suffix, 1000))
                                                       is not None and 20 < value < 150]
            nvme.append(entry)
    package_counts = [number(p) for p in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/thermal_throttle/package_throttle_count")]
    core_counts = [number(p) for p in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/thermal_throttle/core_throttle_count")]
    cpu = {"package_temp_c": max(cpu_temps) if cpu_temps else None,
           "package_throttle_count": max((x for x in package_counts if x is not None), default=None),
           "core_throttle_count_sum": sum(x for x in core_counts if x is not None) if core_counts else None}
    return cpu, nvme


def gpu_expected():
    return Path("/proc/driver/nvidia/gpus").exists() or any(
        read_text(p) == "0x10de" for p in Path("/sys/bus/pci/devices").glob("*/vendor"))


def collect_gpus():
    fields = "index,temperature.gpu,memory.used,memory.total,utilization.gpu,power.draw,power.limit"
    code, output = run_command(["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits"])
    if code != 0:
        return None
    gpus = []
    for row in output.splitlines():
        parts = [x.strip() for x in row.split(",")]
        if len(parts) != 7:
            continue
        parsed = []
        for part in parts:
            try:
                parsed.append(float(part))
            except ValueError:
                parsed.append(None)
        if parsed[0] is None:
            continue
        gpus.append(dict(zip(("index", "temperature_c", "memory_used_mib", "memory_total_mib",
                              "utilization_percent", "power_draw_w", "power_limit_w"), parsed)))
        gpus[-1]["index"] = int(gpus[-1]["index"])
    return gpus or None


def parse_journal(output):
    """Parse only cursors and bounded fault categories; do not retain log text."""
    cursor, faults = None, set()
    for line in output.splitlines():
        if line.startswith("-- cursor: "):
            cursor = line[len("-- cursor: "):].strip()
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if not isinstance(entry, dict):
            continue
        if isinstance(entry.get("__CURSOR"), str):
            cursor = entry["__CURSOR"]
        message = entry.get("MESSAGE", "")
        if isinstance(message, str):
            for key, pattern in FAULT_PATTERNS.items():
                if pattern.search(message[:16384]):
                    faults.add(key)
    return cursor, sorted(faults)


def journal_boot_head():
    """Read the first bounded page after a known reboot, even without old logs.

    Piping to head avoids reading an entire boot into memory. Only these helper
    children are stopped on timeout; no workload process is inspected or touched.
    The last JSON entry supplies the persistent cursor for the next page.
    """
    producer = consumer = None
    started = time.monotonic()
    try:
        producer = subprocess.Popen(["journalctl", "-k", "-b", "--no-pager", "--output=json", "--no-tail"],
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        consumer = subprocess.Popen(["head", "-n", "500"], stdin=producer.stdout,
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        producer.stdout.close()
        output, _ = consumer.communicate(timeout=5)
        producer.wait(timeout=max(0.1, 5 - (time.monotonic() - started)))
        return consumer.returncode, output
    except (OSError, subprocess.TimeoutExpired):
        return None, ""
    finally:
        for process in (consumer, producer):
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=1)


def journal_sample(state, boot_id):
    journal = state.setdefault("journal", {})
    known_reboot = bool(journal.get("boot_id")) and journal["boot_id"] != boot_id
    baseline = not journal.get("cursor") and not known_reboot
    command = ["journalctl", "-k", "-b", "--no-pager", "--output=json", "--show-cursor"]
    command += ["-n", "1"] if baseline or known_reboot else ["--after-cursor", journal["cursor"], "-n", "500"]
    code, output = journal_boot_head() if known_reboot else run_command(command)
    if code != 0:
        # A vacuumed cursor can become invalid: rebaseline, but surface the gap.
        if not baseline and not known_reboot:
            code, output = run_command(command[:6] + ["-n", "1"])
            cursor, _ = parse_journal(output)
            if code == 0 and cursor:
                journal.update({"cursor": cursor, "boot_id": boot_id})
        return [], "kernel_journal_unavailable"
    cursor, faults = parse_journal(output)
    if not cursor:
        return [], "kernel_journal_unavailable"
    journal.update({"cursor": cursor, "boot_id": boot_id})
    return ([] if baseline else faults), None


def collect_metrics(state, boot_id):
    errors = []
    memory = collect_memory()
    if memory is None:
        errors.append("memory")
    cpu, nvme = collect_temperatures()
    if cpu["package_temp_c"] is None:
        errors.append("cpu")
    expected = gpu_expected()
    gpus = collect_gpus() if expected else []
    if expected and (not gpus or any(g["temperature_c"] is None or
                                   g["memory_used_mib"] is None or g["memory_total_mib"] is None for g in gpus)):
        errors.append("gpu")
    psi = {}
    for source in ("cpu", "memory", "io"):
        psi[source] = parse_psi(read_text("/proc/pressure/" + source))
        if "some" not in psi[source]:
            errors.append("psi_" + source)
    disks = []
    for mount in ("/", "/media/jonathan/DATA"):
        if mount != "/" and not os.path.ismount(mount):
            continue
        try:
            stat = os.statvfs(mount)
            total, available = stat.f_blocks * stat.f_frsize, stat.f_bavail * stat.f_frsize
            disks.append({"mount": mount, "total_bytes": total, "available_bytes": available,
                          "used_percent": round(100 * (1 - available / total), 2) if total else 100})
        except OSError:
            errors.append("disk_" + mount)
    for disk in nvme:
        if disk["composite_temp_c"] is None:
            errors.append("nvme_" + disk["device"])
    faults, error = journal_sample(state, boot_id)
    if error:
        errors.append(error)
    return {"memory": memory or {}, "cpu": cpu, "gpus": gpus or [], "gpu_expected": expected,
            "nvme": nvme, "disks": disks, "psi": psi}, sorted(set(errors)), faults


def threshold_condition(value, thresholds, reverse=False):
    if value is None:
        return None
    if (value <= thresholds["critical"] if reverse else value >= thresholds["critical"]):
        return "critical"
    if (value <= thresholds["warning"] if reverse else value >= thresholds["warning"]):
        return "warning"
    return None


def build_conditions(metrics, errors, faults, previous_metrics, config):
    conditions = {}
    thresholds = config["thresholds"]

    def add(key, value, threshold, message, reverse=False):
        severity = threshold_condition(value, threshold, reverse)
        if severity:
            conditions[key] = {"severity": severity, "message": message, "immediate": False}

    cpu = metrics.get("cpu", {})
    temperature = cpu.get("package_temp_c")
    add("cpu_temperature", temperature, thresholds["cpu_temp_c"], f"CPU 封裝溫度 {temperature}°C。")
    old_count = previous_metrics.get("cpu", {}).get("package_throttle_count")
    count = cpu.get("package_throttle_count")
    old_core_count = previous_metrics.get("cpu", {}).get("core_throttle_count_sum")
    core_count = cpu.get("core_throttle_count_sum")
    package_increased = old_count is not None and count is not None and count > old_count
    cores_increased = old_core_count is not None and core_count is not None and core_count > old_core_count
    if package_increased or cores_increased:
        conditions["cpu_thermal_throttle"] = {"severity": "warning", "immediate": True,
                                                "message": "CPU 熱降頻計數增加，散熱保護已介入。"}
    available = metrics.get("memory", {}).get("available_bytes")
    add("memory_available", available, thresholds["memory_available_bytes"],
        f"可用記憶體剩餘 {available / 1024**3:.2f} GiB。" if available is not None else "記憶體資料不可用。", True)
    for gpu in metrics.get("gpus", []):
        add("gpu_temperature_" + str(gpu["index"]), gpu.get("temperature_c"), thresholds["gpu_temp_c"],
            f"GPU {gpu['index']} 溫度 {gpu.get('temperature_c')}°C。")
    for disk in metrics.get("nvme", []):
        limits = disk.get("manufacturer_limits_c", [])
        limit = min(limits) if limits else thresholds["nvme_temp_c"]["critical"]
        effective = {"critical": min(limit, thresholds["nvme_temp_c"]["critical"]),
                     "warning": min(limit - 10, thresholds["nvme_temp_c"]["warning"])}
        key = "nvme_temperature_" + disk["device"]
        add(key, disk.get("composite_temp_c"), effective,
            f"NVMe {disk['device']} 綜合溫度 {disk.get('composite_temp_c')}°C。")
        if disk.get("hardware_alarm"):
            conditions[key] = {"severity": "critical", "immediate": True,
                               "message": f"NVMe {disk['device']} 回報硬體溫度警報。"}
    for disk in metrics.get("disks", []):
        add("disk_space_" + disk["mount"], disk.get("used_percent"), thresholds["disk_used_percent"],
            f"檔案系統 {disk['mount']} 已用 {disk.get('used_percent')}%。")
    for source, label in (("cpu", "CPU"), ("memory", "記憶體"), ("io", "磁碟 I/O")):
        value = metrics.get("psi", {}).get(source, {}).get("some", {}).get("avg60")
        add("psi_" + source, value, thresholds["psi_" + source + "_avg60"],
            f"{label} 資源等待壓力 avg60 為 {value}%。")
    for source in errors:
        conditions["telemetry_" + source] = {"severity": "warning", "immediate": False,
                                             "message": f"監測資料持續不可用：{source}。"}
    for fault in faults:
        conditions["kernel_" + fault] = {"severity": "critical", "immediate": True,
                                        "message": FAULT_MESSAGES[fault]}
    return conditions


def unavailable_condition_keys(state, errors):
    """Missing samples cannot establish recovery of an earlier health alert."""
    keys = set()
    for key in state.get("conditions", {}):
        if (("cpu" in errors and key in ("cpu_temperature", "cpu_thermal_throttle"))
                or ("memory" in errors and key == "memory_available")
                or ("gpu" in errors and key.startswith("gpu_temperature_"))
                or (key.startswith("psi_") and key in errors)
                or (key.startswith("disk_space_") and "disk_" + key.removeprefix("disk_space_") in errors)
                or (key.startswith("nvme_temperature_") and "nvme_" + key.removeprefix("nvme_temperature_") in errors)
                or (key.startswith("kernel_") and "kernel_journal_unavailable" in errors)):
            keys.add(key)
    return keys


def update_alerts(state, conditions, config, now, unavailable_keys=()):
    """Debounce activation; require healthy recovery; escalate without cooldown."""
    records = state.setdefault("conditions", {})
    consecutive = max(1, int(config["alerts"]["consecutive_samples"]))
    recovery = max(1, int(config["alerts"]["recovery_samples"]))
    cooldown = max(900, int(config["alerts"]["cooldown_seconds"]))
    events = []
    for key in sorted(set(records) | set(conditions)):
        record = records.setdefault(key, {"bad_samples": 0, "healthy_samples": 0,
                                          "active": False, "last_notified_at": 0})
        if key in unavailable_keys:
            record["bad_samples"] = 0
            record["healthy_samples"] = 0
            continue
        condition = conditions.get(key)
        if condition:
            record["bad_samples"] += 1
            record["healthy_samples"] = 0
            severity = condition["severity"]
            escalation = record.get("active") and SEVERITIES[severity] > SEVERITIES[record["severity"]]
            ready = condition.get("immediate", False) or record["bad_samples"] >= consecutive
            if ready or escalation:
                newly_active = not record["active"]
                # Do not lower an active critical alert until healthy recovery.
                if newly_active or SEVERITIES[severity] >= SEVERITIES[record["severity"]]:
                    record.update({"severity": severity, "message": condition["message"]})
                record["active"] = True
                if newly_active or escalation or now - record["last_notified_at"] >= cooldown:
                    kind = "activated" if newly_active else "escalated" if escalation else "reminder"
                    events.append({"key": key, "kind": kind, "severity": record["severity"],
                                   "message": record["message"], "at": now})
                    record["last_notified_at"] = now
        else:
            record["bad_samples"] = 0
            record["healthy_samples"] += 1
            if record["active"] and record["healthy_samples"] >= recovery:
                record["active"] = False
                events.append({"key": key, "kind": "recovered", "severity": "normal", "at": now,
                               "message": ("未再觀察到新的核心錯誤；先前事件仍須檢查。" if key.startswith("kernel_")
                                           else "監測條件已恢復正常：" + key + "。")})
            if not record["active"] and record["healthy_samples"] >= recovery:
                del records[key]
    active = [{"key": key, "severity": record["severity"], "message": record["message"]}
              for key, record in sorted(records.items()) if record["active"]]
    return active, events


def notify(event, state, now):
    title = ("工作站安全監控：測試通知" if event["kind"] == "test" else
             "工作站安全監控：已恢復" if event["kind"] == "recovered" else "工作站安全監控：警報")
    critical = event["severity"] == "critical"
    command = ["notify-send", "--app-name=Workstation Guard", "--urgency=" + ("critical" if critical else "normal"),
               "--expire-time=" + ("0" if critical else "15000"), "--", title, event["message"][:500]]
    code, _ = run_command(command, timeout=3)
    delivery = state.setdefault("notifications", {})
    accepted = code == 0
    delivery.update({"last_attempt_at": now, "last_result": "accepted_by_command" if accepted else "command_failed",
                     "consecutive_failures": 0 if accepted else min(100000, delivery.get("consecutive_failures", 0) + 1),
                     "delivery_limit": "notify-send 成功僅表示通知命令被接受，無法確認桌面顯示或使用者已讀。"})
    if accepted:
        delivery["last_success_at"] = now
    return accepted


def admission_status(active, conditions=None, errors=()):
    """Admission is conservative immediately; notifications debounce separately."""
    conditions = conditions or {}
    missing = {entry["key"].removeprefix("telemetry_") for entry in active
               if entry["key"] in ("telemetry_cpu", "telemetry_memory", "telemetry_gpu")}
    missing.update(source for source in errors if source in ("cpu", "memory", "gpu"))
    reasons, gpu_reasons = set(), set()
    for entry in active:
        if entry["severity"] == "critical":
            (gpu_reasons if entry["key"].startswith("gpu_temperature_") else reasons).add(entry["key"])
    for key, condition in conditions.items():
        if key.startswith("gpu_temperature_"):
            gpu_reasons.add(key)
        elif key in ("cpu_temperature", "memory_available") or condition["severity"] == "critical":
            reasons.add(key)
    reasons.update("telemetry_" + source for source in missing if source in ("cpu", "memory"))
    if "gpu" in missing:
        gpu_reasons.add("telemetry_gpu")
    return {"blocked": bool(reasons), "reasons": sorted(set(reasons)),
            "essential_telemetry_missing": sorted(missing),
            "gpu_blocked": bool(gpu_reasons), "gpu_reasons": sorted(gpu_reasons)}


def notify_events(events, state, now):
    """One bounded notification call per sample, even during a fault burst."""
    if not events:
        return
    if len(events) == 1:
        summary = events[0]
    else:
        ordered = sorted(events, key=lambda item: SEVERITIES.get(item["severity"], 0), reverse=True)
        severity = ordered[0]["severity"]
        summary = {"key": "sample_summary", "severity": severity, "at": now,
                   "kind": "recovered" if all(event["kind"] == "recovered" for event in events) else "batch",
                   "message": f"本次有 {len(events)} 項監測通知：\n" + "\n".join(event["message"] for event in ordered[:5])}
    accepted = notify(summary, state, now)
    for event in events:
        event["notification_accepted"] = accepted
        event["notification_batched"] = len(events) > 1


def once(config, state_dir):
    now = time.time()
    state = read_json(state_dir / "state.json", {})
    boot_id = read_text("/proc/sys/kernel/random/boot_id") or "unknown"
    previous = read_json(state_dir / "latest.json", {}).get("metrics", {})
    if state.get("boot_id") != boot_id:
        state["conditions"] = {}
        previous = {}
    # Long sampling gaps invalidate consecutive-sample claims.
    gap_limit = max(120, int(config["sample_interval_seconds"]) * 3)
    if now - state.get("last_sample_at", now) > gap_limit:
        for record in state.get("conditions", {}).values():
            record["bad_samples"] = 0
            record["healthy_samples"] = 0
    state["boot_id"] = boot_id
    metrics, errors, faults = collect_metrics(state, boot_id)
    conditions = build_conditions(metrics, errors, faults, previous, config)
    active, events = update_alerts(state, conditions, config, now,
                                  unavailable_condition_keys(state, errors))
    notify_events(events, state, now)
    for event in events:
        append_log(state_dir / "alerts.jsonl", event)
    state["last_sample_at"] = now
    active_keys = {entry["key"] for entry in active}
    pending = [{"key": key, "severity": condition["severity"], "message": condition["message"]}
               for key, condition in sorted(conditions.items()) if key not in active_keys]
    latest = {"schema_version": 1, "collected_at": now, "boot_id": boot_id,
              "metrics": metrics, "alerts": active, "telemetry_errors": errors,
              "pending_conditions": pending,
              "admission": admission_status(active, conditions, errors), "notifications": state.get("notifications", {
                  "last_result": "not_tested", "delivery_limit": "尚未送出通知；桌面通知需要使用者登入圖形工作階段。"})}
    atomic_json(state_dir / "state.json", state)
    atomic_json(state_dir / "latest.json", latest)
    append_log(state_dir / "history.jsonl", latest)
    return latest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--once", action="store_true")
    action.add_argument("--status", action="store_true")
    action.add_argument("--test-alert", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print status as JSON")
    parser.add_argument("--config", type=Path, default=Path.home() / ".config/workstation-guard/config.json")
    parser.add_argument("--state-dir", type=Path, default=Path.home() / ".local/state/workstation-guard")
    args = parser.parse_args(argv)
    if args.status:
        latest = read_json(args.state_dir / "latest.json", {})
        if not latest:
            print("尚無監測資料。")
            return 1
        latest["age_seconds"] = round(max(0, time.time() - latest.get("collected_at", 0)), 1)
        if args.json:
            print(json.dumps(latest, ensure_ascii=False, indent=2))
        else:
            print(f"資料時間距今 {latest['age_seconds']} 秒；作用中警報 {len(latest.get('alerts', []))} 筆。")
            print("新工作啟動：" + ("暫緩" if latest["admission"]["blocked"] else "可接受"))
            if latest["admission"].get("gpu_blocked"):
                print("GPU 新工作：暫緩（" + ", ".join(latest["admission"].get("gpu_reasons", [])) + "）。")
            if latest["admission"]["blocked"]:
                print("准入限制：" + ", ".join(latest["admission"]["reasons"]))
            for entry in latest.get("alerts", []):
                print(f"[{entry['severity']}] {entry['message']}")
            if latest.get("telemetry_errors"):
                print("資料缺口：" + ", ".join(latest["telemetry_errors"]))
            notification = latest.get("notifications", {})
            print("通知狀態：" + notification.get("last_result", "not_tested"))
            print(notification.get("delivery_limit", ""))
        return 0
    args.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(args.state_dir / "monitor.lock", os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(descriptor, "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        if args.test_alert:
            state = read_json(args.state_dir / "state.json", {})
            now = time.time()
            event = {"kind": "test", "key": "notification_test", "severity": "normal", "at": now,
                     "message": "通知管道測試；目前工作不受影響，這不是硬體故障警報。"}
            accepted = notify(event, state, now)
            event["notification_accepted"] = accepted
            atomic_json(args.state_dir / "state.json", state)
            append_log(args.state_dir / "alerts.jsonl", event)
            latest = read_json(args.state_dir / "latest.json", {})
            if latest:
                latest["notifications"] = state["notifications"]
                atomic_json(args.state_dir / "latest.json", latest)
            print("通知命令已接受；無法確認使用者已讀。" if accepted else "通知命令未成功，請檢查圖形工作階段。")
            return 0 if accepted else 1
        config = merge_dict(DEFAULTS, read_json(args.config, {}))
        once(config, args.state_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
