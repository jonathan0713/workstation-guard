#!/usr/bin/env python3
"""Bounded monitor-service failure notification; never touches workload processes."""
import json
from pathlib import Path
import subprocess
import time

state = Path.home() / '.local/state/workstation-guard/failure-notification.json'
now = time.time()
try:
    previous = json.loads(state.read_text()).get('accepted_at', 0)
except (OSError, ValueError):
    previous = 0
if now - previous >= 900:
    try:
        result = subprocess.run([
            '/usr/bin/notify-send', '--app-name=Workstation-Guard',
            '--urgency=critical', '--expire-time=0',
            '系統監控未完成',
            '健康監控服務失敗。資料過期後將拒絕新工作；現有工作不受影響。'
            '請執行 systemctl --user status workstation-guard.service。',
        ], timeout=3, check=False)
        if result.returncode == 0:
            state.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            state.write_text(json.dumps({'accepted_at': now}) + '\n')
    except (OSError, subprocess.TimeoutExpired) as exc:
        print('Desktop failure notification unavailable:', type(exc).__name__)
