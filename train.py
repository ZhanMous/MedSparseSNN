# -*- coding: utf-8 -*-
"""
训练脚本
- 5次独立重复实验
- 支持 SNN / DenseSNN / ANN
- 记录 test_acc / MIA_acc / 训练时间 / 功耗 / 延迟
- 性能优化：混合精度训练、梯度累积、多进程数据加载
"""

import os
import time
import random
import argparse
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from spikingjelly.activation_based import functional
from data.dataloader import get_medmnist_loaders, resolve_dataset_info
from models import SNN, DenseSNN, ANN
import numpy as np
import csv
import warnings
warnings.filterwarnings('ignore')

try:
    import pynvml
except Exception:
    pynvml = None

# 超参数
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
T = 6
V_THRESHOLD = 1.0
ACCUMULATION_STEPS = 1  # 梯度累积步数
GRADIENT_CLIP = 1.0     # 梯度裁剪

# 保存目录
OUTPUT_ROOT = 'outputs'
CHECKPOINT_DIR = os.path.join(OUTPUT_ROOT, 'checkpoints')
CSV_DIR = os.path.join(OUTPUT_ROOT, 'csv')

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

DEFAULT_MODELS = ['SNN', 'DenseSNN', 'ANN']
DEFAULT_SEEDS = [42, 43, 44, 45, 46]

def set_seed(seed, deterministic=False):
    """设置随机种子；若 deterministic=True 则启用严格可复现设置"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # 默认：允许小幅波动以换取性能
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

# 计算参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def reset_model_state(model_name, model):
    if model_name == 'SNN':
        functional.reset_net(model)
    elif model_name == 'DenseSNN' and hasattr(model, 'reset'):
        model.reset()


def build_model(model_name, num_classes, in_channels, T, v_threshold):
    if model_name == 'SNN':
        return SNN(in_channels=in_channels, num_classes=num_classes, T=T, v_threshold=v_threshold)
    if model_name == 'DenseSNN':
        return DenseSNN(in_channels=in_channels, num_classes=num_classes, T=T, v_threshold=v_threshold)
    if model_name == 'ANN':
        return ANN(in_channels=in_channels, num_classes=num_classes)
    raise ValueError(f"Unknown model: {model_name}")


def evaluate_model(model, model_name, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for data, targets in data_loader:
            data = data.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            reset_model_state(model_name, model)
            outputs = model(data)

            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total_samples += targets.size(0)
            total_correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / max(len(data_loader), 1)
    avg_acc = 100.0 * total_correct / max(total_samples, 1)
    return avg_loss, avg_acc


def measure_efficiency(model, model_name, data_loader, device, max_batches=10):
    """测量真实推理延迟，并在可用时采样 GPU 功耗。"""
    latencies_ms = []
    power_samples_w = []
    model.eval()

    nvml_handle = None
    if device.type == 'cuda' and pynvml is not None:
        try:
            pynvml.nvmlInit()
            nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device.index or 0)
        except Exception:
            nvml_handle = None

    try:
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(data_loader):
                if batch_idx >= max_batches:
                    break

                data = data.to(device, non_blocking=True)
                reset_model_state(model_name, model)

                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                start_time = time.perf_counter()
                _ = model(data)
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                latencies_ms.append(elapsed_ms / data.size(0))

                if nvml_handle is not None:
                    try:
                        power_samples_w.append(pynvml.nvmlDeviceGetPowerUsage(nvml_handle) / 1000.0)
                    except Exception:
                        pass
    finally:
        if nvml_handle is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    avg_latency_ms = float(np.mean(latencies_ms)) if latencies_ms else None
    avg_power_w = float(np.mean(power_samples_w)) if power_samples_w else None
    return avg_power_w, avg_latency_ms


def format_summary_metric(mean_value, std_value, unit=''):
    if mean_value is None or std_value is None:
        return 'N/A'
    suffix = unit if unit else ''
    return f"{mean_value:.2f} ± {std_value:.2f}{suffix}"

# 训练一个模型
def train_model(
    model_name,
    seed,
    dataset_flag='bloodmnist',
    deterministic=False,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    encoding='direct',
    augment=True,
    T_value=T,
    output_prefix=None,
):
    # 设置随机种子
    set_seed(seed, deterministic=deterministic)

    dataset_flag, _, _, num_classes, in_channels = resolve_dataset_info(dataset_flag)
    
    # 加载数据
    train_loader, val_loader, test_loader, _ = get_medmnist_loaders(
        dataset_flag=dataset_flag,
        batch_size=batch_size,
        mode='snn' if model_name in ['SNN', 'DenseSNN'] else 'ann',
        T=T_value,
        encoding=encoding,
        augment=augment,
        seed=seed,
    )
    
    # 初始化模型
    model = build_model(
        model_name,
        num_classes=num_classes,
        in_channels=in_channels,
        T=T_value,
        v_threshold=V_THRESHOLD,
    )
    
    # 计算参数量
    params = count_parameters(model)
    print(f"{model_name} 参数数量: {params} = {params/1e6:.3f}M")
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[MedSparseSNN] 使用设备: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}, 显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
    model = model.to(device)
    
    # 优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    
    # 混合精度训练
    scaler = GradScaler() if torch.cuda.is_available() else None
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 训练记录
    start_time = time.time()
    best_val_acc = 0.0
    best_epoch = 0
    best_state_dict = copy.deepcopy(model.state_dict())
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        optimizer.zero_grad()
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            # 混合精度前向传播
            if torch.cuda.is_available() and scaler is not None:
                with autocast():
                    reset_model_state(model_name, model)
                    outputs = model(data)
                    
                    loss = criterion(outputs, targets) / ACCUMULATION_STEPS
                
                # 反向传播
                scaler.scale(loss).backward()
                
                # 梯度累积
                if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                    # 梯度裁剪
                    if GRADIENT_CLIP > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                # CPU 模式
                reset_model_state(model_name, model)
                outputs = model(data)
                
                loss = criterion(outputs, targets) / ACCUMULATION_STEPS
                loss.backward()
                
                if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                    if GRADIENT_CLIP > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                    optimizer.step()
                    optimizer.zero_grad()
            
            train_loss += loss.item() * ACCUMULATION_STEPS
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # 更新学习率
        scheduler.step()
        
        train_acc = 100. * correct / total
        val_loss, val_acc = evaluate_model(model, model_name, val_loader, criterion, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state_dict = copy.deepcopy(model.state_dict())

            checkpoint_name = output_prefix or dataset_flag
            model_path = os.path.join(CHECKPOINT_DIR, f'{checkpoint_name}_{model_name}_T{T_value}_seed{seed}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'dataset': dataset_flag,
                'encoding': encoding,
            }, model_path)
        
        lr = optimizer.param_groups[0]['lr']
        print(
            f"{dataset_flag} | {model_name} | seed={seed} | epoch {epoch+1}/{epochs}: "
            f"Train Loss {train_loss / max(len(train_loader), 1):.4f}, Train Acc {train_acc:.2f}%, "
            f"Val Loss {val_loss:.4f}, Val Acc {val_acc:.2f}%, Best Val {best_val_acc:.2f}% (epoch {best_epoch}), "
            f"LR {lr:.6f}"
        )
    
    training_time = time.time() - start_time

    model.load_state_dict(best_state_dict)
    _, test_acc = evaluate_model(model, model_name, test_loader, criterion, device)

    power, latency = measure_efficiency(model, model_name, test_loader, device)
    
    return {
        'dataset': dataset_flag,
        'model': model_name,
        'seed': seed,
        'encoding': encoding,
        'augment': augment,
        'T': T_value,
        'epochs': epochs,
        'best_epoch': best_epoch,
        'val_acc': best_val_acc,
        'test_acc': test_acc,
        'training_time': training_time,
        'power': power,
        'latency': latency,
        'params': params,
    }


def summarize_results(results, output_prefix):
    summary = []
    for model, rows in results.items():
        acc_values = [row['test_acc'] for row in rows]
        val_values = [row['val_acc'] for row in rows]
        time_values = [row['training_time'] for row in rows]
        power_values = [row['power'] for row in rows if row['power'] is not None]
        latency_values = [row['latency'] for row in rows if row['latency'] is not None]
        params = rows[0]['params'] if rows else 0

        summary.append({
            'dataset': rows[0]['dataset'],
            'model': model,
            'encoding': rows[0]['encoding'],
            'augment': rows[0]['augment'],
            'T': rows[0]['T'],
            'epochs': rows[0]['epochs'],
            'repeats': len(rows),
            'val_acc': format_summary_metric(float(np.mean(val_values)), float(np.std(val_values))),
            'test_acc': format_summary_metric(float(np.mean(acc_values)), float(np.std(acc_values))),
            'training_time': format_summary_metric(float(np.mean(time_values)), float(np.std(time_values)), 's'),
            'power': format_summary_metric(float(np.mean(power_values)), float(np.std(power_values)), 'W') if power_values else 'N/A',
            'latency': format_summary_metric(float(np.mean(latency_values)), float(np.std(latency_values)), 'ms/sample') if latency_values else 'N/A',
            'params': f"{params} ({params/1e6:.3f}M)",
            'output_prefix': output_prefix,
        })

    return summary


def build_seed_list(repeats, seeds=None):
    if seeds:
        return [int(seed.strip()) for seed in seeds.split(',') if seed.strip()]
    if repeats <= len(DEFAULT_SEEDS):
        return DEFAULT_SEEDS[:repeats]
    return list(range(42, 42 + repeats))


def run_experiments(
    models=None,
    dataset_flag='bloodmnist',
    deterministic=False,
    repeats=5,
    seeds=None,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    encoding='direct',
    augment=True,
    T_value=T,
    output_prefix=None,
):
    models = models or DEFAULT_MODELS
    seed_list = build_seed_list(repeats, seeds=seeds)
    output_prefix = output_prefix or dataset_flag

    results = {model: [] for model in models}

    for model in models:
        print(f"\n{'=' * 60}")
        print(f"  数据集={dataset_flag} | 训练 {model}")
        print(f"{'=' * 60}")
        for index, seed in enumerate(seed_list, start=1):
            print(f"\n--- 第 {index} 次重复实验 (seed={seed}) ---")
            result = train_model(
                model,
                seed,
                dataset_flag=dataset_flag,
                deterministic=deterministic,
                epochs=epochs,
                batch_size=batch_size,
                encoding=encoding,
                augment=augment,
                T_value=T_value,
                output_prefix=output_prefix,
            )
            results[model].append(result)

    detailed_rows = [row for rows in results.values() for row in rows]
    detailed_path = os.path.join(CSV_DIR, f'training_runs_{output_prefix}.csv')
    with open(detailed_path, 'w', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=['dataset', 'model', 'seed', 'encoding', 'augment', 'T', 'epochs', 'best_epoch', 'val_acc', 'test_acc', 'training_time', 'power', 'latency', 'params']
        )
        writer.writeheader()
        writer.writerows(detailed_rows)

    summary = summarize_results(results, output_prefix)
    summary_path = os.path.join(CSV_DIR, f'training_summary_{output_prefix}.csv')
    with open(summary_path, 'w', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=['dataset', 'model', 'encoding', 'augment', 'T', 'epochs', 'repeats', 'val_acc', 'test_acc', 'training_time', 'power', 'latency', 'params', 'output_prefix']
        )
        writer.writeheader()
        writer.writerows(summary)

    print(f"\n{'=' * 60}")
    print("  训练完成")
    print(f"{'=' * 60}")
    print(f"详细结果: {detailed_path}")
    print(f"汇总结果: {summary_path}")

    for item in summary:
        print(f"\n{item['model']}:")
        print(f"  验证准确率: {item['val_acc']}%")
        print(f"  测试准确率: {item['test_acc']}%")
        print(f"  训练时间: {item['training_time']}")
        print(f"  功耗: {item['power']}")
        print(f"  延迟: {item['latency']}")
        print(f"  参数量: {item['params']}")

    return detailed_path, summary_path

def main(args):
    models = DEFAULT_MODELS
    if args.models:
        models = [name.strip() for name in args.models.split(',') if name.strip()]
    elif args.start_from:
        start_idx = DEFAULT_MODELS.index(args.start_from)
        models = DEFAULT_MODELS[start_idx:]

    run_experiments(
        models=models,
        dataset_flag=args.dataset,
        deterministic=args.deterministic,
        repeats=args.repeats,
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        encoding=args.encoding,
        augment=not args.no_augment,
        T_value=args.timesteps,
        output_prefix=args.output_prefix,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('start_from', nargs='?', default=None, choices=['SNN', 'DenseSNN', 'ANN'], help='从哪个模型开始训练')
    parser.add_argument('--deterministic', action='store_true', help='启用严格可复现的 cudnn 设置')
    parser.add_argument('--dataset', default='bloodmnist', help='MedMNIST 数据集标识，例如 bloodmnist 或 pathmnist')
    parser.add_argument('--models', default=None, help='以逗号分隔的模型列表，例如 SNN,DenseSNN,ANN')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='训练轮数')
    parser.add_argument('--repeats', type=int, default=5, help='重复实验次数')
    parser.add_argument('--seeds', default=None, help='逗号分隔的种子列表，优先级高于 repeats')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='批次大小')
    parser.add_argument('--encoding', choices=['direct', 'poisson'], default='direct', help='SNN 输入编码方式')
    parser.add_argument('--timesteps', type=int, default=T, help='SNN 时间步数')
    parser.add_argument('--no-augment', action='store_true', help='关闭训练集数据增强')
    parser.add_argument('--output-prefix', default=None, help='输出文件名前缀')
    args = parser.parse_args()
    main(args)
