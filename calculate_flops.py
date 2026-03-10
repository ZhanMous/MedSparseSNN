# -*- coding: utf-8 -*-
"""
计算理论有效操作数（FLOPs/MACs）
比较SNN/DenseSNN/ANN的理论计算量
"""

import torch
import torch.nn as nn
from models import SNN, DenseSNN, ANN
import csv
import os

OUTPUT_ROOT = 'outputs'
CSV_DIR = os.path.join(OUTPUT_ROOT, 'csv')
os.makedirs(CSV_DIR, exist_ok=True)


def count_conv2d_macs(layer, input_shape):
    """计算Conv2d层的MACs"""
    in_channels = layer.in_channels
    out_channels = layer.out_channels
    kernel_h, kernel_w = layer.kernel_size
    stride_h, stride_w = layer.stride
    pad_h, pad_w = layer.padding
    batch_size = input_shape[0]
    
    output_h = (input_shape[2] + 2 * pad_h - kernel_h) // stride_h + 1
    output_w = (input_shape[3] + 2 * pad_w - kernel_w) // stride_w + 1
    
    macs = batch_size * out_channels * output_h * output_w * in_channels * kernel_h * kernel_w
    if layer.bias is not None:
        macs += batch_size * out_channels * output_h * output_w
    return macs, (batch_size, out_channels, output_h, output_w)


def count_linear_macs(layer, input_shape):
    """计算Linear层的MACs"""
    in_features = layer.in_features
    out_features = layer.out_features
    batch_size = input_shape[0]
    
    macs = batch_size * in_features * out_features
    if layer.bias is not None:
        macs += batch_size * out_features
    return macs, (batch_size, out_features)


def calculate_ann_flops(model, input_shape=(1, 3, 28, 28)):
    """计算ANN的理论FLOPs"""
    total_macs = 0
    current_shape = input_shape
    
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            macs, current_shape = count_conv2d_macs(layer, current_shape)
            total_macs += macs
        elif isinstance(layer, nn.Linear):
            macs, current_shape = count_linear_macs(layer, current_shape)
            total_macs += macs
        elif isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AdaptiveAvgPool2d):
            if isinstance(layer, nn.MaxPool2d):
                kernel_size = layer.kernel_size
                stride = layer.stride
                padding = layer.padding
                if isinstance(kernel_size, int):
                    kernel_h, kernel_w = kernel_size, kernel_size
                else:
                    kernel_h, kernel_w = kernel_size
                if isinstance(stride, int):
                    stride_h, stride_w = stride, stride
                else:
                    stride_h, stride_w = stride
                if isinstance(padding, int):
                    pad_h, pad_w = padding, padding
                else:
                    pad_h, pad_w = padding if padding else (0, 0)
                output_h = (current_shape[2] + 2 * pad_h - kernel_h) // stride_h + 1
                output_w = (current_shape[3] + 2 * pad_w - kernel_w) // stride_w + 1
                current_shape = (current_shape[0], current_shape[1], output_h, output_w)
            elif isinstance(layer, nn.AdaptiveAvgPool2d):
                output_size = layer.output_size
                if isinstance(output_size, int):
                    output_size = (output_size, output_size)
                current_shape = (current_shape[0], current_shape[1], output_size[0], output_size[1])
    
    flops = 2 * total_macs  # 1 MAC ≈ 2 FLOPs
    return flops, total_macs


def calculate_snn_theoretical_flops(ann_total_macs, spike_rate=0.003, T=6):
    """
    计算SNN的理论有效操作数
    
    假设：
    - ANN计算所有MAC操作
    - SNN仅计算发放脉冲对应的MAC操作
    - spike_rate: 平均神经元发放率（本实验为0.003）
    - T: 时间步数
    """
    # SNN需要计算T个时间步，但只有spike_rate比例的神经元发放脉冲
    snn_total_macs = ann_total_macs * T
    snn_effective_macs = ann_total_macs * T * spike_rate
    snn_total_flops = 2 * snn_total_macs
    snn_effective_flops = 2 * snn_effective_macs
    
    # 与ANN比较的节省率（忽略时间步的话）
    theoretical_MACs_saving = 1 - spike_rate
    
    return {
        'ANN_total_MACs': ann_total_macs,
        'ANN_total_FLOPs': 2 * ann_total_macs,
        'SNN_total_MACs': snn_total_macs,
        'SNN_total_FLOPs': snn_total_flops,
        'SNN_effective_MACs': snn_effective_macs,
        'SNN_effective_FLOPs': snn_effective_flops,
        'spike_rate': spike_rate,
        'theoretical_MACs_saving': theoretical_MACs_saving
    }


def main():
    print("=" * 60)
    print("计算理论有效操作数（FLOPs/MACs）")
    print("=" * 60)
    
    results = []
    
    print("\n1. 计算ANN的理论操作数...")
    ann_model = ANN()
    ann_flops, ann_macs = calculate_ann_flops(ann_model)
    print(f"   ANN 理论 MACs: {ann_macs:,}")
    print(f"   ANN 理论 FLOPs: {ann_flops:,}")
    
    print("\n2. 计算SNN的理论有效操作数...")
    spike_rate = 0.003  # 来自实验结果
    T = 6
    flops_results = calculate_snn_theoretical_flops(ann_macs, spike_rate=spike_rate, T=T)
    
    print(f"   平均神经元发放率: {spike_rate:.3f}")
    print(f"   SNN 总 MACs (T={T}): {flops_results['SNN_total_MACs']:,.0f}")
    print(f"   SNN 有效 MACs: {flops_results['SNN_effective_MACs']:,.0f}")
    print(f"   理论 MACs 节省: {flops_results['theoretical_MACs_saving']*100:.1f}%")
    
    results.append({
        'Model': 'ANN',
        'Total_MACs': ann_macs,
        'Total_FLOPs': ann_flops,
        'Effective_MACs': ann_macs,
        'Spike_Rate': 1.0,
        'MACs_Saving': '0.0%'
    })
    
    results.append({
        'Model': 'SNN (Sparse)',
        'Total_MACs': flops_results['SNN_total_MACs'],
        'Total_FLOPs': flops_results['SNN_total_FLOPs'],
        'Effective_MACs': flops_results['SNN_effective_MACs'],
        'Spike_Rate': spike_rate,
        'MACs_Saving': f"{flops_results['theoretical_MACs_saving']*100:.1f}%"
    })
    
    results.append({
        'Model': 'DenseSNN',
        'Total_MACs': flops_results['SNN_total_MACs'],
        'Total_FLOPs': flops_results['SNN_total_FLOPs'],
        'Effective_MACs': flops_results['SNN_total_MACs'],
        'Spike_Rate': spike_rate,
        'MACs_Saving': '0.0%'
    })
    
    csv_path = os.path.join(CSV_DIR, 'theoretical_flops.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Model', 'Total_MACs', 'Total_FLOPs', 
                                                 'Effective_MACs', 'Spike_Rate', 'MACs_Saving'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n结果已保存到: {csv_path}")
    
    print("\n" + "=" * 60)
    print("理论有效操作数计算完成")
    print("=" * 60)
    print("\n结论：")
    print(f"  - SNN 由于平均神经元发放率仅为 {spike_rate:.3f}，")
    print(f"  - 理论上可节省 {flops_results['theoretical_MACs_saving']*100:.1f}% 的有效 MAC 操作")
    print(f"  - 在专用神经形态芯片上可充分发挥这一稀疏计算优势")


if __name__ == '__main__':
    main()
