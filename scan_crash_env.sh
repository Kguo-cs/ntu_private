#!/usr/bin/env bash
set -euo pipefail

OUT="crash_scan_$(hostname)_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

echo "[INFO] Saving logs to $OUT"

run_cmd() {
    local name="$1"
    shift
    echo "[INFO] Running: $name"
    {
        echo "===== $name ====="
        echo "\$ $*"
        "$@"
    } > "$OUT/$name.txt" 2>&1 || true
}

run_shell() {
    local name="$1"
    shift
    echo "[INFO] Running: $name"
    {
        echo "===== $name ====="
        echo "\$ $*"
        bash -lc "$*"
    } > "$OUT/$name.txt" 2>&1 || true
}

echo "===== BASIC =====" > "$OUT/basic.txt"
{
    date
    hostname
    whoami
    uname -a
    uptime
    echo
    echo "===== OS ====="
    cat /etc/os-release || true
    echo
    echo "===== LIMITS ====="
    ulimit -a
} >> "$OUT/basic.txt" 2>&1 || true

run_cmd "disk_df" df -h
run_cmd "inode_df" df -ih
run_cmd "memory_free" free -h
run_cmd "lsblk" lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINT,MODEL
run_cmd "mounts" mount

run_shell "dev_shm" "df -h /dev/shm; ls -alh /dev/shm | head -100"
run_shell "tmp_space" "df -h /tmp /var/tmp 2>/dev/null || true"

run_shell "python_env" '
which python || true
python -V || true
which pip || true
pip -V || true
which conda || true
conda info 2>/dev/null || true
conda list 2>/dev/null | egrep -i "torch|cuda|cudnn|numpy|opencv|cv2|h5py|lmdb|pyarrow|scipy|pandas|lightning|hydra|wandb|numba|mkl" || true
pip list 2>/dev/null | egrep -i "torch|cuda|cudnn|numpy|opencv|cv2|h5py|lmdb|pyarrow|scipy|pandas|lightning|hydra|wandb|numba|mkl" || true
'

run_shell "torch_check" '
python - <<PY
import os, sys, platform
print("python:", sys.version)
print("platform:", platform.platform())
print("PYTHONFAULTHANDLER:", os.environ.get("PYTHONFAULTHANDLER"))
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda version:", torch.version.cuda)
    print("cudnn:", torch.backends.cudnn.version())
    print("num devices:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
        print(torch.cuda.get_device_properties(i))
    import torch.multiprocessing as mp
    print("mp sharing strategy:", mp.get_sharing_strategy())
except Exception as e:
    print("torch import/check failed:", repr(e))
PY
'

if command -v nvidia-smi >/dev/null 2>&1; then
    run_cmd "nvidia_smi" nvidia-smi
    run_cmd "nvidia_smi_q" nvidia-smi -q
    run_shell "nvidia_query" '
    nvidia-smi --query-gpu=index,name,driver_version,pstate,temperature.gpu,utilization.gpu,memory.total,memory.used,memory.free,power.draw,power.limit --format=csv,noheader,nounits || true
    '
else
    echo "nvidia-smi not found" > "$OUT/nvidia_smi.txt"
fi

run_shell "dmesg_errors" '
dmesg -T 2>/dev/null | egrep -i "segfault|oom|out of memory|killed process|nvrm|xid|cuda|gpu|pcie|aer|mce|machine check|hardware error|edac|ecc|nvme|i/o error|ext4|xfs|btrfs|reset|thermal|thrott" | tail -500 || true
'

run_shell "journal_errors" '
journalctl -k --no-pager -n 2000 2>/dev/null | egrep -i "segfault|oom|out of memory|killed process|nvrm|xid|cuda|gpu|pcie|aer|mce|machine check|hardware error|edac|ecc|nvme|i/o error|ext4|xfs|btrfs|reset|thermal|thrott" | tail -500 || true
'

run_shell "coredump" '
coredumpctl list --no-pager 2>/dev/null | tail -100 || true
'

run_shell "recent_kernel_tail" '
dmesg -T 2>/dev/null | tail -300 || true
'

run_shell "process_limits" '
echo "pid max:"; cat /proc/sys/kernel/pid_max || true
echo "threads max:"; cat /proc/sys/kernel/threads-max || true
echo "file max:"; cat /proc/sys/fs/file-max || true
'

tar -czf "${OUT}.tar.gz" "$OUT"

echo
echo "[DONE] Scan saved to: ${OUT}.tar.gz"
echo
echo "重点看这些文件："
echo "  $OUT/dmesg_errors.txt"
echo "  $OUT/journal_errors.txt"
echo "  $OUT/nvidia_smi_q.txt"
echo "  $OUT/dev_shm.txt"
echo "  $OUT/torch_check.txt"