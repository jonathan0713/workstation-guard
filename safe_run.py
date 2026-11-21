#!/usr/bin/env python3
"""Admit only NEW workloads into the workstation's user resource slice."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid

GIB = 1024 ** 3
MIB = 1024 ** 2
SLICE = "workstation-workloads.slice"
AGGREGATE_MAX = 22 * GIB
RESERVE = 6 * GIB
STATE = Path.home() / ".local/state/workstation-guard"
CONFIG = Path.home() / ".config/workstation-guard/config.json"
CGROUP_ROOT = Path("/sys/fs/cgroup")


class AdmissionError(RuntimeError):
    """A new workload cannot safely be admitted."""


def parse_size(value: str) -> int:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?)(?:i?B)?", value, re.I)
    if not match:
        raise argparse.ArgumentTypeError("容量格式須為 512M、4G、12GiB 等（二進位單位）。")
    number, suffix = match.groups()
    size = int(float(number) * 1024 ** ("KMGT".find(suffix.upper()) + 1 if suffix else 0))
    if size < 256 * MIB:
        raise argparse.ArgumentTypeError("容量至少須為 256MiB。")
    return size


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="安全啟動新工作：獨立 systemd scope、記憶體上限、CPU 限制及健康預檢。",
        epilog=("用法：safe-run --cpu 8 --memory 12G -- python3 train.py\n"
                "GPU：safe-run --gpu --gpu-memory 4G --memory 10G -- python3 train.py\n"
                "只套用新工作；不搬移、不停止現有程序。--memory 是最壞情況預算，"
                "啟動時可用 RAM 須至少為此預算加 6GiB；所有受管理工作預算合計最多 22GiB。\n"
                "達到 MemoryHigh 會回收記憶體；達到 MemoryMax 可能由核心終止此新工作。"
                "GPU 預算是啟動預檢，並非顯存硬上限。監控告警不會自動殺掉工作。"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cpu", type=int, default=8, help="CPU 數／時間配額，整數 1–16（預設 8）")
    parser.add_argument("--memory", type=parse_size, default=12 * GIB, help="RAM 硬上限，最多 22G（預設 12G）")
    parser.add_argument("--name", default="job", help="工作名稱：英數、底線、連字號，最多 32 字元")
    parser.add_argument("--gpu", action="store_true", help="獨佔一個受管理 GPU 工作名額（GPU 0）")
    parser.add_argument("--gpu-memory", type=parse_size, default=4 * GIB, help="GPU 顯存需求預檢（預設 4G，另留 1G）")
    parser.add_argument("--dry-run", action="store_true", help="只驗證參數並列出預計命令；不做健康驗證或啟動")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- 後接程式及參數，不經 shell 解譯")
    args = parser.parse_args(argv)
    if not 1 <= args.cpu <= 16:
        parser.error("--cpu 須為 1–16。")
    if args.memory > AGGREGATE_MAX:
        parser.error("--memory 不得超過 22GiB。")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}", args.name):
        parser.error("--name 須為 1–32 個英數、底線或連字號，且以英數開頭。")
    if not args.command or args.command[0] != "--" or len(args.command) < 2:
        parser.error("請以 -- 分隔程式，例如：safe-run --memory 8G -- python3 task.py")
    args.command = args.command[1:]
    return args


def check_health(status: dict, now: float, gpu: bool = False):
    timestamp = status.get("collected_at")
    if not isinstance(timestamp, (float, int)) or not math.isfinite(timestamp) or not -5 <= now - timestamp <= 120:
        raise AdmissionError("監控資料缺失或超過 120 秒；請先檢查 workstation-guard.service。")
    admission = status.get("admission")
    if not isinstance(admission, dict) or not isinstance(admission.get("blocked"), bool):
        raise AdmissionError("監控資料缺少有效 admission 狀態。")
    if admission["blocked"]:
        reasons = admission.get("reasons", [])
        raise AdmissionError("健康檢查暫停新工作：" + "；".join(map(str, reasons)))
    if gpu and admission.get("gpu_blocked"):
        raise AdmissionError("GPU 健康檢查暫停新工作：" + "；".join(map(str, admission.get("gpu_reasons", []))))
    if any(
        a.get("severity") == "critical"
        and (gpu or not str(a.get("key", "")).startswith("gpu_temperature_"))
        for a in status.get("alerts", []) if isinstance(a, dict)
    ):
        raise AdmissionError("仍有未解除的 critical 告警，暫停新工作。")
    missing = admission.get("essential_telemetry_missing", [])
    if any(key in missing for key in ("cpu", "memory")) or (gpu and "gpu" in missing):
        raise AdmissionError("必要硬體遙測缺失，無法確認新工作的安全條件。")
    # Refuse status data from an earlier boot, even after a wall-clock adjustment.
    boot = Path("/proc/sys/kernel/random/boot_id")
    if status.get("boot_id") and boot.exists() and status["boot_id"] != boot.read_text().strip():
        raise AdmissionError("監控資料來自先前開機，請等待新樣本。")


def available_memory(path=Path("/proc/meminfo")):
    for line in path.read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise AdmissionError("無法讀取 MemAvailable。")


def check_memory(requested, available, current, reserved=0, limit=AGGREGATE_MAX):
    # Existing admitted jobs may grow to their reserved caps after this check.
    # Account for that unused budget so they cannot consume the host's reserve.
    unused_reserved = max(0, reserved - current)
    if requested + unused_reserved + RESERVE > available:
        raise AdmissionError(f"RAM 餘裕不足：可用 {available / GIB:.1f}GiB，新工作上限 {requested / GIB:.1f}GiB，既有工作未用預算 {unused_reserved / GIB:.1f}GiB，須另留 6GiB。")
    if current + requested > limit:
        raise AdmissionError("資源群組目前用量加新工作上限將超過 22GiB。")
    if reserved + requested > limit:
        raise AdmissionError("現有受管理工作的記憶體預算加新預算將超過 22GiB；請降低 --memory 或等待工作完成。")


def run_query(command, timeout=5):
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AdmissionError(f"查詢失敗：{command[0]}：{exc}") from exc
    if result.returncode != 0:
        raise AdmissionError(f"查詢失敗：{command[0]}：{result.stderr.strip() or result.stdout.strip()}")
    return result.stdout


def unit_properties(unit):
    command = ["systemctl", "--user", "show", unit, "--property=LoadState", "--property=ActiveState", "--property=ControlGroup"]
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AdmissionError(f"systemd 使用者管理器無法連線：{exc}") from exc
    props = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if result.returncode and props.get("LoadState") != "not-found":
        raise AdmissionError(f"無法查詢 systemd scope：{result.stderr.strip()}")
    if not props.get("LoadState") or not props.get("ActiveState"):
        raise AdmissionError("systemd scope 狀態不完整。")
    return props


def slice_capabilities():
    props = unit_properties(SLICE)
    if props.get("LoadState") != "loaded" or props.get("ActiveState") != "active" or not props.get("ControlGroup"):
        raise AdmissionError("workstation-workloads.slice 尚未載入或啟用；請先完成安裝。")
    group = CGROUP_ROOT / props["ControlGroup"].lstrip("/")
    try:
        max_memory = int((group / "memory.max").read_text().strip())
        current = int((group / "memory.current").read_text().strip())
    except (OSError, ValueError) as exc:
        raise AdmissionError("slice 缺少可驗證的記憶體硬上限，拒絕啟動。") from exc
    if max_memory > AGGREGATE_MAX or max_memory <= 0:
        raise AdmissionError("slice 的記憶體硬上限不是有效的 ≤22GiB 設定。")
    cpu_path = group / "cpu.max"
    cpu_quota = False
    if cpu_path.exists():
        fields = cpu_path.read_text().split()
        cpu_quota = len(fields) == 2 and fields[0] != "max" and int(fields[0]) / int(fields[1]) <= 16
    return {"group": group, "limit": max_memory, "current": current, "cpu_quota": cpu_quota, "io_weight": (group / "io.weight").exists()}


def gpu_preflight(required):
    output = run_query(["nvidia-smi", "-i", "0", "--query-gpu=memory.free,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"])
    try:
        free, utilization, temperature = (float(x.strip()) for x in output.strip().split(","))
        if not all(math.isfinite(n) for n in (free, utilization, temperature)):
            raise ValueError("non-finite GPU metrics")
    except ValueError as exc:
        raise AdmissionError("無法解析 GPU 0 即時狀態。") from exc
    if temperature >= 80:
        raise AdmissionError(f"GPU 0 溫度 {temperature:g}°C 偏高，暫停新 GPU 工作。")
    if utilization > 20:
        raise AdmissionError(f"GPU 0 已忙碌（{utilization:g}%）；不干擾既有 GPU 工作。")
    if free * MIB < required + GIB:
        raise AdmissionError(f"GPU 0 可用顯存僅 {free / 1024:.1f}GiB，無法提供工作預算及 1GiB 餘裕。")
    processes = run_query(["nvidia-smi", "-i", "0", "--query-compute-apps=pid", "--format=csv,noheader,nounits"])
    if processes.strip():
        raise AdmissionError("GPU 0 已有運算程序；不干擾既有工作。")


def cpu_pool():
    allowed = set(os.sched_getaffinity(0))
    if CONFIG.exists():
        try:
            configured = json.loads(CONFIG.read_text()).get("cpu_pool")
        except (OSError, ValueError) as exc:
            raise AdmissionError("無法解析 workstation-guard/config.json。") from exc
        if configured is not None:
            if not isinstance(configured, list) or not configured or any(type(n) is not int or n < 0 for n in configured):
                raise AdmissionError("config.json 的 cpu_pool 必須是非空 CPU 編號清單。")
            pool = list(dict.fromkeys(configured))[:16]
            if not set(pool) <= allowed:
                raise AdmissionError("設定的 CPU pool 不在此程序可用 CPU 範圍；請從一般終端機執行。")
            return pool
    def frequency(number):
        try:
            return int(Path(f"/sys/devices/system/cpu/cpu{number}/cpufreq/cpuinfo_max_freq").read_text())
        except (OSError, ValueError):
            return 2 ** 63
    return sorted(allowed, key=lambda n: (frequency(n), n))[:16]


def build_command(args, unit, capabilities=None):
    command = ["systemd-run", "--user", "--scope", "--slice=" + SLICE, "--unit=" + unit, "--collect", "--quiet"]
    properties = {"MemoryHigh": int(args.memory * .8), "MemoryMax": args.memory, "MemorySwapMax": GIB, "TasksMax": 2048}
    if capabilities is None or capabilities["cpu_quota"]:
        properties.update(CPUQuota=f"{args.cpu * 100}%", CPUWeight=50)
    if capabilities is None or capabilities["io_weight"]:
        properties["IOWeight"] = 50
    command.extend(f"--property={key}={value}" for key, value in properties.items())
    workload = list(args.command)
    if capabilities is not None and not capabilities["io_weight"]:
        ionice = shutil.which("ionice")
        if ionice:
            workload = [ionice, "-c", "3", "--"] + workload
    # The small in-scope wrapper verifies memory controls before exec, and makes
    # cgroup OOM affect this new workload group instead of leaving orphan workers.
    wrapper = [sys.executable, str(Path(__file__).resolve()), "--_workload", unit, str(args.memory), "--"]
    return command + ["--"] + wrapper + workload


def exec_workload(argv):
    """Private entry point called by systemd-run after placement in a new scope."""
    if len(argv) < 4 or argv[2] != "--" or not re.fullmatch(r"wg-[A-Za-z0-9_-]+\.scope", argv[0]):
        raise AdmissionError("內部 scope 參數錯誤。")
    unit, maximum = argv[0], int(argv[1])
    unified = next((line.split(":", 2)[2] for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::")), "")
    group = CGROUP_ROOT / unified.lstrip("/")
    if group.name != unit or group.parent.name != SLICE:
        raise AdmissionError("工作尚未進入其專用資源群組，拒絕執行。")
    for name, expected in (("memory.max", maximum), ("memory.high", int(maximum * .8)), ("memory.swap.max", GIB), ("pids.max", 2048)):
        actual = (group / name).read_text().strip()
        if actual == "max" or int(actual) > expected:
            raise AdmissionError(f"新 scope 的 {name} 未套用預期上限，拒絕執行。")
    (group / "memory.oom.group").write_text("1\n")
    if (group / "memory.oom.group").read_text().strip() != "1":
        raise AdmissionError("無法設定新 scope 的群組 OOM 保護，拒絕執行。")
    os.execvp(argv[3], argv[3:])


def process_identity(pid):
    try:
        # comm can include spaces or parentheses; fields after the last ')' are stable.
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def active_reservations(directory):
    active = []
    for path in directory.glob("*.json"):
        try:
            item = json.loads(path.read_text())
            pid = item["pid"]
            start = item["start_time"]
            if not isinstance(item["memory"], int) or not re.fullmatch(r"wg-[A-Za-z0-9_-]+\.scope", item["unit"]):
                raise ValueError("invalid reservation")
        except (OSError, KeyError, ValueError, TypeError) as exc:
            raise AdmissionError(f"工作預算紀錄無法驗證：{path}") from exc
        alive = start is not None and process_identity(pid) == start
        if not alive and item.get("child_start_time") is not None:
            alive = process_identity(item["child_pid"]) == item["child_start_time"]
        if not alive:
            # An orphaned scope can outlive the launcher: never erase it on PID alone.
            props = unit_properties(item["unit"])
            alive = props.get("ActiveState") not in ("inactive", "failed") and props.get("LoadState") != "not-found"
        if alive:
            active.append(item)
        else:
            path.unlink()
    return active


@contextlib.contextmanager
def locked_file(path, nonblocking=False):
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        except BlockingIOError as exc:
            raise AdmissionError("另一個受管理 GPU 工作仍在執行。") from exc
        yield fd
    finally:
        os.close(fd)


def exit_code(returncode):
    return 128 - returncode if returncode < 0 else returncode


def supervise(process, unit):
    old_handlers = {}
    def forward(signum, _frame):
        # The foreground tty already delivers SIGINT to this shared process group.
        # Forwarding also supports kill(launcher_pid); never escalate to SIGKILL.
        if process.poll() is None:
            try:
                process.send_signal(signum)
            except ProcessLookupError:
                pass
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        old_handlers[signum] = signal.signal(signum, forward)
    try:
        code = process.wait()
        # Background descendants retain the reservation and GPU lock until the scope ends.
        warned = False
        while True:
            try:
                props = unit_properties(unit)
                if props.get("LoadState") == "not-found" or props.get("ActiveState") in ("inactive", "failed"):
                    break
            except AdmissionError:
                if not warned:
                    print("safe-run：暫時無法確認 scope 已清空；保留工作預算與 GPU 鎖，等待 systemd 恢復。", file=sys.stderr)
                    warned = True
            time.sleep(1)
        return exit_code(code)
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


def main(argv=None):
    args = parse_args(argv)
    unit = f"wg-{args.name}-{uuid.uuid4().hex[:10]}.scope"
    if args.dry_run:
        print("僅參數驗證：尚未檢查健康狀態、實際 controller 或資源餘裕；此命令顯示 CPU/IO controller 可用時的設定。")
        print(shlex.join(build_command(args, unit)))
        return 0
    reservation_path = None
    process = None
    try:
        STATE.mkdir(parents=True, exist_ok=True, mode=0o700)
        reservations = STATE / "reservations"
        reservations.mkdir(exist_ok=True, mode=0o700)
        with contextlib.ExitStack() as stack:
            if args.gpu:
                stack.enter_context(locked_file(STATE / "gpu.lock", nonblocking=True))
            with locked_file(STATE / "admission.lock"):
                try:
                    health = json.loads((STATE / "latest.json").read_text())
                except (OSError, ValueError) as exc:
                    raise AdmissionError("尚無有效監控資料；請檢查 workstation-guard.service。") from exc
                check_health(health, time.time(), args.gpu)
                capabilities = slice_capabilities()
                active = active_reservations(reservations)
                if args.gpu and any(item.get("gpu") for item in active):
                    raise AdmissionError("另一個受管理 GPU scope 仍在執行。")
                check_memory(args.memory, available_memory(), capabilities["current"], sum(item["memory"] for item in active), capabilities["limit"])
                if args.gpu:
                    gpu_preflight(args.gpu_memory)
                selected = None
                if not capabilities["cpu_quota"]:
                    pool = cpu_pool()
                    if args.cpu > len(pool):
                        raise AdmissionError(f"可用共用 CPU pool 僅 {len(pool)} 個，請降低 --cpu。")
                    selected = pool[:args.cpu]
                    print(f"safe-run：CPU controller 未委派，使用 CPU affinity {','.join(map(str, selected))} + nice 10；無 CPU 時間硬配額。", file=sys.stderr)
                if not capabilities["io_weight"]:
                    print("safe-run：I/O controller 未委派，使用 ionice idle（核心／磁碟排程器可能不支援）；無 I/O 硬配額。", file=sys.stderr)
                command = build_command(args, unit, capabilities)
                # This hook runs only in the new child, never in this launcher or existing jobs.
                def child_setup():
                    if selected is not None:
                        os.sched_setaffinity(0, selected)
                        os.nice(max(0, 10 - os.getpriority(os.PRIO_PROCESS, 0)))
                reservation_path = reservations / f"{unit}.json"
                reservation = {"unit": unit, "pid": os.getpid(), "start_time": process_identity(os.getpid()), "memory": args.memory, "gpu": args.gpu, "created_at": time.time()}
                reservation_path.write_text(json.dumps(reservation))
                try:
                    process = subprocess.Popen(command, preexec_fn=child_setup)
                    reservation.update(child_pid=process.pid, child_start_time=process_identity(process.pid))
                    reservation_path.write_text(json.dumps(reservation))
                except (OSError, subprocess.SubprocessError) as exc:
                    reservation_path.unlink(missing_ok=True)
                    reservation_path = None
                    raise AdmissionError(f"無法啟動新 scope：{exc}") from exc
            print(f"safe-run：{unit}；RAM 上限 {args.memory / GIB:.2f}GiB、回收門檻 {args.memory * .8 / GIB:.2f}GiB。", file=sys.stderr)
            result = supervise(process, unit)
            reservation_path.unlink(missing_ok=True)
            return result
    except (AdmissionError, OSError, ValueError) as exc:
        print(f"safe-run：拒絕啟動新工作：{exc}", file=sys.stderr)
        return 75


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--_workload":
        try:
            exec_workload(sys.argv[2:])
        except (AdmissionError, OSError, ValueError) as error:
            print(f"safe-run：新 scope 驗證失敗，尚未執行工作：{error}", file=sys.stderr)
            raise SystemExit(75)
    else:
        raise SystemExit(main())
