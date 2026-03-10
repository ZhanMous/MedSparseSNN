#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""通用 MedMNIST 数据集实验编排脚本。"""

import argparse
import csv
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
CSV_DIR = os.path.join(OUTPUT_DIR, 'csv')
REPORT_DIR = os.path.join(PROJECT_ROOT, 'docs', 'reports')
DEFAULT_BLOOD_SUMMARY = os.path.join(CSV_DIR, 'training_summary.csv')

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
    return [{'dataset': 'bloodmnist', 'model': row['model'], 'test_acc': row['test_acc']} for row in rows]


def load_optional_rows(path):
    if not os.path.exists(path):
        return []
    return read_csv_rows(path)


def default_report_name(dataset_flag):
    return os.path.join('docs', 'reports', f'{dataset_flag}_experiment_report.md')


def final_prefix(dataset_flag):
    return f'{dataset_flag}_final_compare'


def screening_prefix(dataset_flag, config_name):
    return f'{dataset_flag}_screen_{config_name}'


def write_report(report_path, dataset_flag, screening_rows, final_rows, selected_config, blood_rows, privacy_rows, efficiency_rows, args):
    comparison_rows = []
    blood_by_model = {row['model']: row for row in blood_rows}
    for row in final_rows:
        blood = blood_by_model.get(row['model'])
        blood_mean = metric_mean(blood['test_acc']) if blood else None
        dataset_mean = metric_mean(row['test_acc'])
        drop = 'N/A'
        if blood_mean is not None and dataset_mean is not None:
            drop = f"{blood_mean - dataset_mean:.2f}"
        comparison_rows.append({
            'model': row['model'],
            'blood_test_acc': blood['test_acc'] if blood else 'N/A',
            'dataset_test_acc': row['test_acc'],
            'accuracy_drop': drop,
        })

    lines = [
        f'# MedSparseSNN {dataset_flag} 实验报告',
        '',
        '## 实验目标',
        '',
        f'验证 BloodMNIST 上表现良好的 MedSparseSNN 配置迁移到 {dataset_flag} 时的性能退化来源，并给出可复现的实验配置。',
        '',
        '## 实验设计',
        '',
        f'- 筛选阶段：SNN / DenseSNN，epochs={args.screen_epochs}，repeats={args.screen_repeats}，比较编码方式、时间步和增强策略。',
        f'- 正式阶段：SNN / DenseSNN / ANN，epochs={args.final_epochs}，repeats={args.final_repeats}，采用筛选出的最佳 SNN 配置。',
        '- checkpoint 选择基于验证集最佳准确率，最终只在最佳验证 checkpoint 上汇报测试集结果。',
        '',
        '## 筛选结果',
        '',
        render_markdown_table(screening_rows, ['config', 'model', 'encoding', 'augment', 'T', 'epochs', 'val_acc', 'test_acc']),
        '',
        f'## 最终 {dataset_flag} 对比',
        '',
        render_markdown_table(final_rows, ['model', 'encoding', 'augment', 'T', 'epochs', 'repeats', 'val_acc', 'test_acc', 'training_time', 'power', 'latency', 'params']),
        '',
        f'## BloodMNIST vs {dataset_flag} 准确率变化',
        '',
        render_markdown_table(comparison_rows, ['model', 'blood_test_acc', 'dataset_test_acc', 'accuracy_drop']),
        '',
        f'## {dataset_flag} 隐私结果',
        '',
        render_markdown_table(privacy_rows, ['model', 'accuracy', 'auc', 'f1', 'precision', 'recall', 'significance_vs_ann']) if privacy_rows else f'暂无 {dataset_flag} MIA 结果。',
        '',
        f'## {dataset_flag} 稀疏性与理论能效',
        '',
        render_markdown_table(efficiency_rows, ['model', 'repeats', 'test_acc', 'power_w', 'latency_ms_per_sample', 'energy_mj_per_sample', 'spike_rate', 'theoretical_macs_saving']) if efficiency_rows else f'暂无 {dataset_flag} 稀疏性/理论能效结果。',
        '',
        '## 结论',
        '',
        f"- 最佳 SNN 配置为 {selected_config['config_name']}，对应 encoding={selected_config['encoding']}、augment={selected_config['augment']}、T={selected_config['timesteps']}。",
        f'- {dataset_flag} 的结果可作为 BloodMNIST 与 PathMNIST 之外的第三个外部验证点，用于评估稀疏性优势的跨域稳定性。',
        '- 需要联合准确率、MIA 指标与理论有效 MAC 节省一起判断该数据域是否支撑核心论点，而不是只看单一指标。',
        '',
    ]

    with open(report_path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--screen-epochs', type=int, default=8)
    parser.add_argument('--screen-repeats', type=int, default=1)
    parser.add_argument('--final-epochs', type=int, default=15)
    parser.add_argument('--final-repeats', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--report-name', default=None)
    parser.add_argument('--skip-training', action='store_true')
    args = parser.parse_args()

    dataset_flag = args.dataset.lower()
    report_name = args.report_name or default_report_name(dataset_flag)
    report_path = report_name if os.path.isabs(report_name) else os.path.join(PROJECT_ROOT, report_name)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    final_output_prefix = final_prefix(dataset_flag)

    screening_summaries = []
    ann_prefix = screening_prefix(dataset_flag, 'ann_reference')
    if not args.skip_training:
        run_experiments(
            models=['ANN'],
            dataset_flag=dataset_flag,
            deterministic=args.deterministic,
            repeats=args.screen_repeats,
            epochs=args.screen_epochs,
            batch_size=args.batch_size,
            encoding='direct',
            augment=True,
            T_value=6,
            output_prefix=ann_prefix,
        )
    screening_summaries.append(('ann_reference', read_csv_rows(os.path.join(CSV_DIR, f'training_summary_{ann_prefix}.csv'))))

    for config in SCREENING_CONFIGS:
        prefix = screening_prefix(dataset_flag, config['name'])
        if not args.skip_training:
            run_experiments(
                models=['SNN', 'DenseSNN'],
                dataset_flag=dataset_flag,
                deterministic=args.deterministic,
                repeats=args.screen_repeats,
                epochs=args.screen_epochs,
                batch_size=args.batch_size,
                encoding=config['encoding'],
                augment=config['augment'],
                T_value=config['timesteps'],
                output_prefix=prefix,
            )
        screening_summaries.append((config['name'], read_csv_rows(os.path.join(CSV_DIR, f'training_summary_{prefix}.csv'))))

    selected_config = select_best_snn_config(screening_summaries)
    if not args.skip_training:
        run_experiments(
            models=['SNN', 'DenseSNN', 'ANN'],
            dataset_flag=dataset_flag,
            deterministic=args.deterministic,
            repeats=args.final_repeats,
            epochs=args.final_epochs,
            batch_size=args.batch_size,
            encoding=selected_config['encoding'],
            augment=selected_config['augment'],
            T_value=selected_config['timesteps'],
            output_prefix=final_output_prefix,
        )

    screening_rows = build_screening_report_rows(screening_summaries)
    final_rows = read_csv_rows(os.path.join(CSV_DIR, f'training_summary_{final_output_prefix}.csv'))
    blood_rows = load_blood_baseline()
    privacy_rows = load_optional_rows(os.path.join(CSV_DIR, f'mia_results_{final_output_prefix}.csv'))
    efficiency_rows = load_optional_rows(os.path.join(CSV_DIR, f'medmnist_privacy_efficiency_summary_{final_output_prefix}.csv'))
    write_report(report_path, dataset_flag, screening_rows, final_rows, selected_config, blood_rows, privacy_rows, efficiency_rows, args)
    print(f'Wrote report to {report_path}')


if __name__ == '__main__':
    main()