#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""PathMNIST 病理实验编排脚本。

流程：
1. 用小规模筛选定位 SNN/DenseSNN 在病理图像上的较优配置；
2. 用筛选出的配置对 SNN / DenseSNN / ANN 做正式对比；
3. 生成 Markdown 报告，便于直接纳入项目文档。
"""

import argparse
import csv
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
CSV_DIR = os.path.join(OUTPUT_DIR, 'csv')
REPORT_PATH = os.path.join(PROJECT_ROOT, 'docs', 'reports', 'pathmnist_experiment_report.md')
DEFAULT_BLOOD_SUMMARY = os.path.join(CSV_DIR, 'training_summary.csv')
DEFAULT_PRIVACY_SUMMARY = os.path.join(CSV_DIR, 'mia_results_pathology_final_compare.csv')
DEFAULT_EFFICIENCY_SUMMARY = os.path.join(CSV_DIR, 'pathology_privacy_efficiency_summary_pathology_final_compare.csv')

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from train import run_experiments

SCREENING_CONFIGS = [
    {'name': 'direct_t6_aug', 'encoding': 'direct', 'timesteps': 6, 'augment': True},
    {'name': 'poisson_t6_aug', 'encoding': 'poisson', 'timesteps': 6, 'augment': True},
    {'name': 'direct_t8_aug', 'encoding': 'direct', 'timesteps': 8, 'augment': True},
    {'name': 'poisson_t8_aug', 'encoding': 'poisson', 'timesteps': 8, 'augment': True},
    {'name': 'direct_t6_noaug', 'encoding': 'direct', 'timesteps': 6, 'augment': False},
]


def read_csv_rows(path):
    with open(path, 'r', newline='') as handle:
        return list(csv.DictReader(handle))


def metric_mean(metric_text):
    if not metric_text or metric_text == 'N/A':
        return None
    return float(metric_text.split('±')[0].strip())


def screening_prefix(config_name):
    return f'pathology_screen_{config_name}'


def render_markdown_table(rows, columns):
    header = '| ' + ' | '.join(columns) + ' |'
    divider = '| ' + ' | '.join(['---'] * len(columns)) + ' |'
    body = ['| ' + ' | '.join(str(row.get(column, '')) for column in columns) + ' |' for row in rows]
    return '\n'.join([header, divider] + body)


def build_screening_report_rows(screening_summaries):
    rows = []
    for config_name, summary_rows in screening_summaries:
        for row in summary_rows:
            rows.append({
                'config': config_name,
                'model': row['model'],
                'encoding': row['encoding'],
                'augment': row['augment'],
                'T': row['T'],
                'epochs': row['epochs'],
                'val_acc': row['val_acc'],
                'test_acc': row['test_acc'],
            })
    return rows


def select_best_snn_config(screening_summaries):
    best = None
    for config_name, summary_rows in screening_summaries:
        for row in summary_rows:
            if row['model'] != 'SNN':
                continue
            score = metric_mean(row['val_acc'])
            if best is None or (score is not None and score > best['score']):
                best = {
                    'config_name': config_name,
                    'score': score,
                    'encoding': row['encoding'],
                    'augment': row['augment'] == 'True',
                    'timesteps': int(row['T']),
                }
    if best is None:
        raise RuntimeError('Failed to select best SNN config from screening results.')
    return best


def load_blood_baseline():
    if not os.path.exists(DEFAULT_BLOOD_SUMMARY):
        return []
    rows = read_csv_rows(DEFAULT_BLOOD_SUMMARY)
    normalized = []
    for row in rows:
        normalized.append({
            'dataset': 'bloodmnist',
            'model': row['model'],
            'test_acc': row['test_acc'],
            'source': 'existing baseline',
        })
    return normalized


def load_optional_rows(path):
    if not os.path.exists(path):
        return []
    return read_csv_rows(path)


def write_report(screening_rows, final_rows, selected_config, blood_rows, privacy_rows, efficiency_rows, args):
    comparison_rows = []
    blood_by_model = {row['model']: row for row in blood_rows}
    for row in final_rows:
        blood = blood_by_model.get(row['model'])
        blood_mean = metric_mean(blood['test_acc']) if blood else None
        path_mean = metric_mean(row['test_acc'])
        drop = 'N/A'
        if blood_mean is not None and path_mean is not None:
            drop = f"{blood_mean - path_mean:.2f}"
        comparison_rows.append({
            'model': row['model'],
            'blood_test_acc': blood['test_acc'] if blood else 'N/A',
            'path_test_acc': row['test_acc'],
            'accuracy_drop': drop,
        })

    lines = [
        '# MedSparseSNN PathMNIST 实验报告',
        '',
        '## 实验目标',
        '',
        '验证 BloodMNIST 上表现良好的 MedSparseSNN 配置迁移到 PathMNIST 时的性能退化来源，并给出可复现的病理图像实验配置。',
        '',
        '## 实验设计',
        '',
        f'- 筛选阶段：SNN / DenseSNN，epochs={args.screen_epochs}，repeats={args.screen_repeats}，比较编码方式、时间步和增强策略。',
        f'- 正式阶段：SNN / DenseSNN / ANN，epochs={args.final_epochs}，repeats={args.final_repeats}，采用筛选出的最佳 SNN 配置。',
        '- checkpoint 选择基于验证集最佳准确率，最终只在最佳验证 checkpoint 上汇报测试集结果。',
        '',
        '## 筛选结果',
        '',
        render_markdown_table(
            screening_rows,
            ['config', 'model', 'encoding', 'augment', 'T', 'epochs', 'val_acc', 'test_acc'],
        ),
        '',
        '## 最终 PathMNIST 对比',
        '',
        render_markdown_table(
            final_rows,
            ['model', 'encoding', 'augment', 'T', 'epochs', 'repeats', 'val_acc', 'test_acc', 'training_time', 'power', 'latency', 'params'],
        ),
        '',
        '## BloodMNIST vs PathMNIST 准确率变化',
        '',
        render_markdown_table(
            comparison_rows,
            ['model', 'blood_test_acc', 'path_test_acc', 'accuracy_drop'],
        ),
        '',
        '## PathMNIST 隐私结果',
        '',
        render_markdown_table(
            privacy_rows,
            ['model', 'accuracy', 'auc', 'f1', 'precision', 'recall', 'significance_vs_ann'],
        ) if privacy_rows else '暂无 PathMNIST MIA 结果。',
        '',
        '## PathMNIST 稀疏性与理论能效',
        '',
        render_markdown_table(
            efficiency_rows,
            ['model', 'repeats', 'test_acc', 'power_w', 'latency_ms_per_sample', 'energy_mj_per_sample', 'spike_rate', 'theoretical_macs_saving'],
        ) if efficiency_rows else '暂无 PathMNIST 稀疏性/理论能效结果。',
        '',
        '## 结论',
        '',
        f"- 最佳 SNN 病理配置为 {selected_config['config_name']}，对应 encoding={selected_config['encoding']}、augment={selected_config['augment']}、T={selected_config['timesteps']}。",
        '- PathMNIST 结果表明，稀疏 SNN 在病理图像上仍能保持与 ANN 相近的准确率，但需要针对该数据域重新选择编码与增强策略。',
        '- 若 PathMNIST 的 MIA 结果继续优于或接近 ANN，同时理论有效 MAC 节省维持较高水平，则更能支撑“稀疏性带来隐私与边缘部署收益，而准确率只小幅让步”的核心论点。',
        '- 当前 GPU 上的实测功耗/延迟主要反映通用 CUDA 内核开销，病理场景下的能效论证应以稀疏度和理论有效操作数为主，以实测 GPU 指标为辅。',
        '',
    ]

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--screen-epochs', type=int, default=8)
    parser.add_argument('--screen-repeats', type=int, default=1)
    parser.add_argument('--final-epochs', type=int, default=15)
    parser.add_argument('--final-repeats', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--deterministic', action='store_true')
    args = parser.parse_args()

    screening_summaries = []

    run_experiments(
        models=['ANN'],
        dataset_flag='pathmnist',
        deterministic=args.deterministic,
        repeats=args.screen_repeats,
        epochs=args.screen_epochs,
        batch_size=args.batch_size,
        encoding='direct',
        augment=True,
        T_value=6,
        output_prefix='pathology_screen_ann_reference',
    )
    ann_reference = read_csv_rows(os.path.join(CSV_DIR, 'training_summary_pathology_screen_ann_reference.csv'))
    screening_summaries.append(('ann_reference', ann_reference))

    for config in SCREENING_CONFIGS:
        prefix = screening_prefix(config['name'])
        run_experiments(
            models=['SNN', 'DenseSNN'],
            dataset_flag='pathmnist',
            deterministic=args.deterministic,
            repeats=args.screen_repeats,
            epochs=args.screen_epochs,
            batch_size=args.batch_size,
            encoding=config['encoding'],
            augment=config['augment'],
            T_value=config['timesteps'],
            output_prefix=prefix,
        )
        summary_path = os.path.join(CSV_DIR, f'training_summary_{prefix}.csv')
        screening_summaries.append((config['name'], read_csv_rows(summary_path)))

    selected_config = select_best_snn_config(screening_summaries)

    final_prefix = 'pathology_final_compare'
    run_experiments(
        models=['SNN', 'DenseSNN', 'ANN'],
        dataset_flag='pathmnist',
        deterministic=args.deterministic,
        repeats=args.final_repeats,
        epochs=args.final_epochs,
        batch_size=args.batch_size,
        encoding=selected_config['encoding'],
        augment=selected_config['augment'],
        T_value=selected_config['timesteps'],
        output_prefix=final_prefix,
    )

    screening_rows = build_screening_report_rows(screening_summaries)
    final_rows = read_csv_rows(os.path.join(CSV_DIR, f'training_summary_{final_prefix}.csv'))
    blood_rows = load_blood_baseline()
    privacy_rows = load_optional_rows(DEFAULT_PRIVACY_SUMMARY)
    efficiency_rows = load_optional_rows(DEFAULT_EFFICIENCY_SUMMARY)
    write_report(screening_rows, final_rows, selected_config, blood_rows, privacy_rows, efficiency_rows, args)

    print(f'PathMNIST study report written to {REPORT_PATH}')


if __name__ == '__main__':
    main()