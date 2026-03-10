#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""分析任意 MedMNIST 最终对比实验的稀疏性、理论 MAC 节省与单样本能耗。"""

import argparse
import csv
import os
import sys

import numpy as np
import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
CSV_DIR = os.path.join(OUTPUT_DIR, 'csv')
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from calculate_flops import calculate_ann_flops
from data.dataloader import get_medmnist_loaders, resolve_dataset_info
from models import ANN, FixedPLIFNode, NonSparsePLIF, PLIFNode
from train import build_model, reset_model_state


def read_csv_rows(path):
    with open(path, 'r', newline='') as handle:
        return list(csv.DictReader(handle))


def load_checkpoint_model(model_name, checkpoint_path, dataset_flag, timesteps, device):
    _, _, _, num_classes, in_channels = resolve_dataset_info(dataset_flag)
    model = build_model(model_name, num_classes=num_classes, in_channels=in_channels, T=timesteps, v_threshold=1.0)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def collect_spike_rate(model, model_name, data_loader, device, max_batches=20):
    tracked_types = (PLIFNode, FixedPLIFNode, NonSparsePLIF)
    totals = {'nonzero': 0, 'elements': 0}

    def hook(_, __, output):
        if torch.is_tensor(output):
            totals['nonzero'] += int((output != 0).sum().item())
            totals['elements'] += int(output.numel())

    handles = []
    for module in model.modules():
        if isinstance(module, tracked_types):
            handles.append(module.register_forward_hook(hook))

    try:
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(data_loader):
                if batch_idx >= max_batches:
                    break
                data = data.to(device, non_blocking=True)
                reset_model_state(model_name, model)
                _ = model(data)
    finally:
        for handle in handles:
            handle.remove()

    if totals['elements'] == 0:
        return None
    return totals['nonzero'] / totals['elements']


def format_metric(values, precision=4, unit=''):
    if not values:
        return 'N/A'
    mean = float(np.mean(values))
    std = float(np.std(values))
    suffix = unit if unit else ''
    return f"{mean:.{precision}f} ± {std:.{precision}f}{suffix}"


def main(args):
    detailed_training_path = os.path.join(CSV_DIR, f'training_runs_{args.training_prefix}.csv')
    if not os.path.exists(detailed_training_path):
        raise FileNotFoundError(f'Missing training runs file: {detailed_training_path}')

    training_rows = read_csv_rows(detailed_training_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, _, num_classes, in_channels = resolve_dataset_info(args.dataset)
    ann_flops, ann_macs = calculate_ann_flops(ANN(in_channels=in_channels, num_classes=num_classes))

    evaluation_rows = []
    for row in training_rows:
        model_name = row['model']
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{args.training_prefix}_{model_name}_T{row['T']}_seed{row['seed']}.pth")

        spike_rate = None
        theoretical_macs_saving = 0.0
        if model_name in {'SNN', 'DenseSNN'}:
            model = load_checkpoint_model(model_name, checkpoint_path, args.dataset, int(row['T']), device)
            _, _, test_loader, _ = get_medmnist_loaders(
                dataset_flag=args.dataset,
                batch_size=args.batch_size,
                mode='snn',
                T=int(row['T']),
                encoding=row['encoding'],
                augment=str(row['augment']) == 'True',
                seed=int(row['seed']),
            )
            spike_rate = collect_spike_rate(model, model_name, test_loader, device, max_batches=args.max_batches)
            if model_name == 'SNN' and spike_rate is not None:
                theoretical_macs_saving = 1.0 - spike_rate

        power = float(row['power']) if row['power'] else None
        latency = float(row['latency']) if row['latency'] else None
        energy_mj = power * latency if power is not None and latency is not None else None

        evaluation_rows.append({
            'dataset': args.dataset,
            'model': model_name,
            'seed': row['seed'],
            'encoding': row['encoding'],
            'augment': row['augment'],
            'T': row['T'],
            'test_acc': float(row['test_acc']),
            'power_w': power,
            'latency_ms_per_sample': latency,
            'energy_mj_per_sample': energy_mj,
            'spike_rate': spike_rate,
            'theoretical_macs_saving': theoretical_macs_saving,
            'ann_reference_macs': ann_macs,
            'ann_reference_flops': ann_flops,
        })

    detailed_out = os.path.join(CSV_DIR, f'medmnist_privacy_efficiency_runs_{args.training_prefix}.csv')
    with open(detailed_out, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['dataset', 'model', 'seed', 'encoding', 'augment', 'T', 'test_acc', 'power_w', 'latency_ms_per_sample', 'energy_mj_per_sample', 'spike_rate', 'theoretical_macs_saving', 'ann_reference_macs', 'ann_reference_flops'])
        writer.writeheader()
        writer.writerows(evaluation_rows)

    summary_out = os.path.join(CSV_DIR, f'medmnist_privacy_efficiency_summary_{args.training_prefix}.csv')
    with open(summary_out, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['dataset', 'model', 'repeats', 'test_acc', 'power_w', 'latency_ms_per_sample', 'energy_mj_per_sample', 'spike_rate', 'theoretical_macs_saving'])
        for model_name in ['SNN', 'DenseSNN', 'ANN']:
            rows = [row for row in evaluation_rows if row['model'] == model_name]
            writer.writerow([
                args.dataset,
                model_name,
                len(rows),
                format_metric([row['test_acc'] for row in rows], precision=2),
                format_metric([row['power_w'] for row in rows if row['power_w'] is not None], precision=2, unit='W'),
                format_metric([row['latency_ms_per_sample'] for row in rows if row['latency_ms_per_sample'] is not None], precision=2, unit='ms/sample'),
                format_metric([row['energy_mj_per_sample'] for row in rows if row['energy_mj_per_sample'] is not None], precision=2, unit='mJ'),
                format_metric([row['spike_rate'] for row in rows if row['spike_rate'] is not None], precision=4),
                format_metric([100.0 * row['theoretical_macs_saving'] for row in rows if row['theoretical_macs_saving'] is not None], precision=2, unit='%'),
            ])

    print(f'Wrote {detailed_out}')
    print(f'Wrote {summary_out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--training-prefix', required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-batches', type=int, default=20)
    main(parser.parse_args())