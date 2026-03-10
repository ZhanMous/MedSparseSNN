#!/usr/bin/env python3

import csv
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
CSV_DIR = os.path.join(OUTPUT_DIR, 'csv')
FIGURE_DIR = os.path.join(OUTPUT_DIR, 'figures')

os.makedirs(FIGURE_DIR, exist_ok=True)


plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.figsize': (6.8, 4.2),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'ANN': '#4C78A8',
    'SNN': '#F58518',
    'DenseSNN': '#54A24B',
    'Spiking Transformer': '#E45756',
}


def read_csv(path):
    with open(path, 'r', newline='') as handle:
        return list(csv.DictReader(handle))


def get_value(row, *keys):
    for key in keys:
        if key in row:
            return row[key]
    lowered = {key.lower(): value for key, value in row.items()}
    for key in keys:
        if key.lower() in lowered:
            return lowered[key.lower()]
    raise KeyError(keys[0])


def normalize_model_name(name):
    text = str(name).strip()
    if text == 'SNN (Sparse)':
        return 'SNN'
    return text


def parse_mean_std(text):
    if text is None:
        return None, None
    raw = str(text).strip()
    if not raw or raw.upper() == 'N/A':
        return None, None
    match = re.match(r'\s*([0-9.]+)\s*±\s*([0-9.]+)', raw)
    if match:
        return float(match.group(1)), float(match.group(2))
    match = re.match(r'\s*([0-9.]+)', raw)
    if match:
        return float(match.group(1)), 0.0
    return None, None


def save_figure(fig, name):
    path = os.path.join(FIGURE_DIR, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'Saved {path}')


def plot_model_performance():
    rows = read_csv(os.path.join(CSV_DIR, 'training_summary.csv'))
    models = ['ANN', 'SNN', 'DenseSNN']
    acc_means = []
    acc_stds = []
    for model in models:
        row = next(item for item in rows if item['model'] == model)
        mean, std = parse_mean_std(row['test_acc'])
        acc_means.append(mean)
        acc_stds.append(std)

    fig, ax = plt.subplots()
    bars = ax.bar(models, acc_means, yerr=acc_stds, capsize=4, color=[COLORS[m] for m in models])
    for bar, value in zip(bars, acc_means):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.4, f'{value:.2f}', ha='center', va='bottom')
    ax.set_ylabel('Test accuracy (%)')
    ax.set_ylim(88, 100)
    save_figure(fig, 'model_performance.png')


def plot_sparsity_vs_mia():
    rows = read_csv(os.path.join(CSV_DIR, 'ablation_results.csv'))
    sparsity = []
    mia = []
    thresholds = []
    for row in rows:
        s_mean, _ = parse_mean_std(row['Sparsity'])
        m_mean, _ = parse_mean_std(row['MIA Accuracy'])
        if s_mean is None or m_mean is None:
            continue
        thresholds.append(row['v_threshold'])
        sparsity.append(s_mean)
        mia.append(m_mean)

    fig, ax = plt.subplots()
    ax.plot(sparsity, mia, marker='o', linewidth=1.8, color=COLORS['SNN'])
    for x_value, y_value, label in zip(sparsity, mia, thresholds):
        ax.annotate(f'v={label}', (x_value, y_value), textcoords='offset points', xytext=(4, 5))
    ax.set_xlabel('Global sparsity')
    ax.set_ylabel('MIA accuracy')
    ax.set_ylim(0.48, 0.60)
    save_figure(fig, 'sparsity_vs_mia.png')


def plot_power_latency():
    rows = read_csv(os.path.join(CSV_DIR, 'power_results.csv'))
    fig, ax = plt.subplots()
    for row in rows:
        model = row['Model']
        if model == 'SNN (Sparse)':
            label = 'SNN'
        else:
            label = model
        latency, latency_std = parse_mean_std(row['Latency (ms)'])
        power, power_std = parse_mean_std(row['Dynamic Power (W)'])
        if latency is None or power is None:
            continue
        ax.errorbar(latency, power, xerr=latency_std, yerr=power_std, fmt='o', ms=7, capsize=3, label=label, color=COLORS.get(label, '#333333'))
        ax.annotate(label, (latency, power), textcoords='offset points', xytext=(5, 4))
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Dynamic power (W)')
    save_figure(fig, 'power_latency.png')


def plot_cross_dataset_tradeoff():
    blood_train = read_csv(os.path.join(CSV_DIR, 'training_summary.csv'))
    blood_mia = read_csv(os.path.join(CSV_DIR, 'mia_results.csv'))
    path_train = read_csv(os.path.join(CSV_DIR, 'training_summary_pathology_final_compare.csv'))
    path_mia = read_csv(os.path.join(CSV_DIR, 'mia_results_pathology_final_compare.csv'))
    derma_train = read_csv(os.path.join(CSV_DIR, 'training_summary_dermamnist_final_compare.csv'))
    derma_mia = read_csv(os.path.join(CSV_DIR, 'mia_results_dermamnist_final_compare.csv'))

    datasets = [
        ('BloodMNIST', blood_train, blood_mia),
        ('PathMNIST', path_train, path_mia),
        ('DermaMNIST', derma_train, derma_mia),
    ]
    models = ['ANN', 'SNN', 'DenseSNN']

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    x = np.arange(len(datasets))
    width = 0.22

    for idx, model in enumerate(models):
        acc_vals = []
        mia_vals = []
        for _, train_rows, mia_rows in datasets:
            train_row = next(item for item in train_rows if normalize_model_name(item['model']) == model)
            mia_row = next(item for item in mia_rows if normalize_model_name(get_value(item, 'model', 'Model')) == model)
            acc_mean, _ = parse_mean_std(train_row['test_acc'])
            mia_mean, _ = parse_mean_std(get_value(mia_row, 'accuracy', 'MIA Accuracy'))
            acc_vals.append(acc_mean)
            mia_vals.append(mia_mean)
        axes[0].bar(x + (idx - 1) * width, acc_vals, width=width, label=model, color=COLORS[model])
        axes[1].bar(x + (idx - 1) * width, mia_vals, width=width, label=model, color=COLORS[model])

    axes[0].set_xticks(x)
    axes[0].set_xticklabels([name for name, _, _ in datasets])
    axes[0].set_ylabel('Test accuracy (%)')
    axes[0].set_ylim(55, 100)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels([name for name, _, _ in datasets])
    axes[1].set_ylabel('MIA accuracy')
    axes[1].set_ylim(0.45, 0.66)
    axes[1].legend(frameon=False, loc='upper left')

    save_figure(fig, 'cross_dataset_tradeoff.png')


def plot_transformer_sparsity_vs_mia():
    rows = read_csv(os.path.join(CSV_DIR, 'p1_spiking_transformer_ablation.csv'))
    sparsity = []
    mia = []
    thresholds = []
    for row in rows:
        s_mean, _ = parse_mean_std(row['sparsity'])
        m_mean, _ = parse_mean_std(row['mia_acc'])
        if s_mean is None or m_mean is None:
            continue
        thresholds.append(row['v_threshold'])
        sparsity.append(s_mean)
        mia.append(m_mean)

    fig, ax = plt.subplots()
    ax.plot(sparsity, mia, marker='s', linewidth=1.8, color=COLORS['Spiking Transformer'])
    for x_value, y_value, label in zip(sparsity, mia, thresholds):
        ax.annotate(f'v={label}', (x_value, y_value), textcoords='offset points', xytext=(4, 5))
    ax.set_xlabel('Global sparsity')
    ax.set_ylabel('MIA accuracy')
    ax.set_ylim(0.49, 0.61)
    save_figure(fig, 'transformer_sparsity_vs_mia.png')


def plot_transformer_comparison():
    base_train = read_csv(os.path.join(CSV_DIR, 'training_summary.csv'))
    base_mia = read_csv(os.path.join(CSV_DIR, 'mia_results.csv'))
    transformer_rows = read_csv(os.path.join(CSV_DIR, 'p1_spiking_transformer_ablation.csv'))
    transformer_best = next(row for row in transformer_rows if row['v_threshold'] == '1.0')

    models = ['ANN', 'SNN', 'DenseSNN', 'Spiking Transformer']
    acc_values = []
    mia_values = []
    for model in models:
        if model == 'Spiking Transformer':
            acc_mean, _ = parse_mean_std(transformer_best['test_acc'])
            mia_mean, _ = parse_mean_std(transformer_best['mia_acc'])
        else:
            train_row = next(item for item in base_train if normalize_model_name(item['model']) == model)
            mia_row = next(item for item in base_mia if normalize_model_name(get_value(item, 'Model', 'model')) == model)
            acc_mean, _ = parse_mean_std(train_row['test_acc'])
            mia_mean, _ = parse_mean_std(get_value(mia_row, 'MIA Accuracy', 'accuracy'))
        acc_values.append(acc_mean)
        mia_values.append(mia_mean)

    fig, ax1 = plt.subplots()
    x = np.arange(len(models))
    width = 0.38
    ax1.bar(x - width / 2, acc_values, width=width, color=[COLORS[m] for m in models], alpha=0.9)
    ax1.set_ylabel('Test accuracy (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['ANN', 'SNN', 'DenseSNN', 'Transformer'])
    ax1.set_ylim(88, 100)

    ax2 = ax1.twinx()
    ax2.plot(x + width / 2, mia_values, color='#222222', marker='o', linewidth=1.6)
    ax2.set_ylabel('MIA accuracy')
    ax2.set_ylim(0.48, 0.66)

    save_figure(fig, 'spiking_transformer_comparison.png')


def main():
    plot_model_performance()
    plot_sparsity_vs_mia()
    plot_power_latency()
    plot_cross_dataset_tradeoff()
    plot_transformer_sparsity_vs_mia()
    plot_transformer_comparison()


if __name__ == '__main__':
    main()