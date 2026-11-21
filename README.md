# workstation-guard

用 systemd user scope 管理 Linux 工作負載的 CPU、RAM 與 I/O，並監控主機溫度、記憶體、磁碟使用量與壓力。GPU 預檢需要 NVIDIA 的 `nvidia-smi`。監控支援桌面通知，並將最近狀態與歷史寫到使用者目錄。

## 安裝

需要 Python 3、systemd user manager 和 cgroup v2。執行以下指令可安裝到使用者目錄。

```sh
install -d "$HOME/.local/share/workstation-guard" "$HOME/.config/systemd/user"
install -m 644 monitor.py safe_run.py failure_notify.py "$HOME/.local/share/workstation-guard/"
install -m 644 units/* "$HOME/.config/systemd/user/"
systemctl --user daemon-reload
systemctl --user enable --now workstation-guard.timer
```

需要自訂警報門檻或 CPU pool 時，把 `config.example.json` 複製到 `~/.config/workstation-guard/config.json` 後編輯。預設會從目前程序可用的 CPU 中選擇最多 16 個。Slice 的記憶體上限可依主機容量調整。

## 使用

```sh
./health_cli.sh
./health_cli.sh --json
./safe_run_cli.sh --cpu 4 --memory 4G -- python3 task.py
./safe_run_cli.sh --gpu --gpu-memory 4G --cpu 4 --memory 4G -- python3 train.py
```

直接啟動的工作會維持原本的資源設定。可透過 `python3 -m unittest discover -s tests` 執行測試。
