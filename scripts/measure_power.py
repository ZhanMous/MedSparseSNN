#!/usr/bin/env python3
"""
简单的功耗测量脚本（基于 pynvml）
用法示例：
  python scripts/measure_power.py --cmd "python train.py --model SNN"
或者：
  python scripts/measure_power.py --pid 12345

脚本会定期采样 GPU 功耗并将结果保存到 outputs/csv/power_measurements.csv
"""
import time
import argparse
import csv
import subprocess

try:
    import pynvml
except Exception:
    pynvml = None


def monitor_pid(pid, interval=0.1, device_index=0, out_csv='outputs/csv/power_measurements.csv'):
    if pynvml is None:
        raise RuntimeError('pynvml is required for power measurement (pip install nvidia-ml-py3)')

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

    samples = []
    try:
        while True:
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            except pynvml.NVMLError:
                power_mw = 0
            timestamp = time.time()
            samples.append({'ts': timestamp, 'power_w': power_mw / 1000.0})
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        pynvml.nvmlShutdown()

    # save
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['ts', 'power_w'])
        writer.writeheader()
        writer.writerows(samples)


def monitor_command(cmd, interval=0.1, device_index=0, out_csv='outputs/csv/power_measurements.csv'):
    # start subprocess
    p = subprocess.Popen(cmd, shell=True)
    if pynvml is None:
        print('pynvml not available, waiting for process to finish without power measurement')
        p.wait()
        return

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

    samples = []
    try:
        while p.poll() is None:
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            except pynvml.NVMLError:
                power_mw = 0
            timestamp = time.time()
            samples.append({'ts': timestamp, 'power_w': power_mw / 1000.0})
            time.sleep(interval)
    finally:
        pynvml.nvmlShutdown()

    # save
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['ts', 'power_w'])
        writer.writeheader()
        writer.writerows(samples)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--cmd', type=str, help='要运行并测量功耗的命令（字符串）')
    group.add_argument('--pid', type=int, help='要监控的进程 PID')
    parser.add_argument('--interval', type=float, default=0.1, help='采样间隔（秒）')
    parser.add_argument('--device', type=int, default=0, help='GPU 设备索引')
    parser.add_argument('--out', type=str, default='outputs/csv/power_measurements.csv', help='输出 CSV 路径')

    args = parser.parse_args()

    if args.cmd:
        monitor_command(args.cmd, interval=args.interval, device_index=args.device, out_csv=args.out)
    else:
        print('监控 PID 需要手动停止（Ctrl-C）')
        monitor_pid(args.pid, interval=args.interval, device_index=args.device, out_csv=args.out)


if __name__ == '__main__':
    main()
