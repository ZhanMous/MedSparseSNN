# -*- coding: utf-8 -*-
"""
MIA 攻击脚本
- 支持 BloodMNIST / PathMNIST 等 MedMNIST 分类任务
- 使用影子模型 + Logistic Regression 的黑盒攻击
- 输入特征：max confidence + entropy + confidence margin
"""

import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from spikingjelly.activation_based import functional
from torch.utils.data import DataLoader, Subset

from data.dataloader import get_medmnist_loaders, resolve_dataset_info
from train import build_model


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def reset_model_state(model_name, model):
    if model_name == 'SNN':
        functional.reset_net(model)
    elif model_name == 'DenseSNN' and hasattr(model, 'reset'):
        model.reset()


def t_test(data1, data2):
    import scipy.stats as stats
    t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
    return t_stat, p_value


def get_significance_label(p_value):
    if p_value < 0.01:
        return '**'
    if p_value < 0.05:
        return '*'
    return ''


BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
T = 6
V_THRESHOLD = 1.0
NUM_SHADOW_MODELS = 5
NUM_REPEATS = 5

OUTPUT_ROOT = 'outputs'
CSV_DIR = os.path.join(OUTPUT_ROOT, 'csv')
os.makedirs(CSV_DIR, exist_ok=True)


def get_loaders_for_model(model_name, dataset_flag, batch_size, timesteps, encoding, augment, seed=None):
    mode = 'snn' if model_name in ['SNN', 'DenseSNN'] else 'ann'
    return get_medmnist_loaders(
        dataset_flag=dataset_flag,
        batch_size=batch_size,
        mode=mode,
        T=timesteps,
        encoding=encoding,
        augment=augment,
        seed=seed,
    )


def compute_entropy(probabilities):
    return -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)


def compute_confidence_margin(probabilities):
    top2 = torch.topk(probabilities, k=2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def train_shadow_model(model_name, seed, dataset_flag, batch_size, epochs, timesteps, encoding, augment):
    set_seed(seed)

    train_loader, _, test_loader, _ = get_loaders_for_model(
        model_name,
        dataset_flag=dataset_flag,
        batch_size=batch_size,
        timesteps=timesteps,
        encoding=encoding,
        augment=augment,
        seed=seed,
    )
    _, _, _, num_classes, in_channels = resolve_dataset_info(dataset_flag)

    train_dataset = train_loader.dataset
    test_dataset = test_loader.dataset

    n_train = len(train_dataset)
    n_member = min(n_train // 2, len(test_dataset))
    member_indices = np.random.choice(n_train, n_member, replace=False)
    non_member_indices = np.random.choice(len(test_dataset), n_member, replace=False)

    member_loader = DataLoader(Subset(train_dataset, member_indices), batch_size=batch_size, shuffle=True)
    non_member_loader = DataLoader(Subset(test_dataset, non_member_indices), batch_size=batch_size, shuffle=False)

    model = build_model(
        model_name,
        num_classes=num_classes,
        in_channels=in_channels,
        T=timesteps,
        v_threshold=V_THRESHOLD,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        for data, targets in member_loader:
            data = data.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            reset_model_state(model_name, model)
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    return model, member_loader, non_member_loader


def extract_features(model, model_name, data_loader):
    device = next(model.parameters()).device
    features = []
    labels = []

    model.eval()
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            reset_model_state(model_name, model)
            outputs = model(data)
            probabilities = nn.functional.softmax(outputs, dim=1)

            max_conf = probabilities.max(dim=1)[0].cpu().numpy()
            entropy = compute_entropy(probabilities).cpu().numpy()
            margin = compute_confidence_margin(probabilities).cpu().numpy()

            batch_features = np.column_stack((max_conf, entropy, margin))
            features.append(batch_features)
            labels.append(np.ones(len(batch_features)))

    return np.vstack(features), np.concatenate(labels)


def run_mia_attack(model_name, dataset_flag, batch_size, epochs, num_shadow_models, timesteps, encoding, augment):
    print(f"\n=== 执行 {dataset_flag} / {model_name} 的 MIA 攻击 ===")

    all_features = []
    all_labels = []

    for shadow_idx in range(num_shadow_models):
        print(f"\n--- 训练第 {shadow_idx + 1} 个影子模型 ---")
        shadow_model, member_loader, non_member_loader = train_shadow_model(
            model_name,
            seed=shadow_idx,
            dataset_flag=dataset_flag,
            batch_size=batch_size,
            epochs=epochs,
            timesteps=timesteps,
            encoding=encoding,
            augment=augment,
        )

        member_features, member_labels = extract_features(shadow_model, model_name, member_loader)
        non_member_features, non_member_labels = extract_features(shadow_model, model_name, non_member_loader)
        non_member_labels = np.zeros(len(non_member_labels))

        all_features.append(np.vstack((member_features, non_member_features)))
        all_labels.append(np.concatenate((member_labels, non_member_labels)))

    X = np.vstack(all_features)
    y = np.concatenate(all_labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    attack_model = LogisticRegression(max_iter=1000)
    attack_model.fit(X_train, y_train)

    y_pred = attack_model.predict(X_test)
    y_pred_proba = attack_model.predict_proba(X_test)[:, 1]

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba),
        'f1': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred),
    }


def summarize_results(all_mia_results, output_prefix, dataset_flag, repeats, num_shadow_models, epochs, timesteps, encoding, augment):
    summary_results = {}
    for model in all_mia_results:
        summary_results[model] = {}
        for metric in all_mia_results[model]:
            mean_val = np.mean(all_mia_results[model][metric])
            std_val = np.std(all_mia_results[model][metric])
            summary_results[model][metric] = {'mean': mean_val, 'std': std_val}

    significance = {}
    if 'ANN' in all_mia_results:
        ann_accuracy_values = all_mia_results['ANN']['accuracy']
        for model in all_mia_results:
            if model == 'ANN':
                continue
            t_stat, p_value = t_test(ann_accuracy_values, all_mia_results[model]['accuracy'])
            significance[model] = {'t_stat': t_stat, 'p_value': p_value, 'label': get_significance_label(p_value)}

    detailed_path = os.path.join(CSV_DIR, f'mia_runs_{output_prefix}.csv')
    with open(detailed_path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['dataset', 'model', 'repeat', 'epochs', 'shadow_models', 'encoding', 'augment', 'T', 'accuracy', 'auc', 'f1', 'precision', 'recall'])
        for model, metrics in all_mia_results.items():
            for repeat_idx in range(len(metrics['accuracy'])):
                writer.writerow([
                    dataset_flag,
                    model,
                    repeat_idx + 1,
                    epochs,
                    num_shadow_models,
                    encoding,
                    augment,
                    timesteps,
                    metrics['accuracy'][repeat_idx],
                    metrics['auc'][repeat_idx],
                    metrics['f1'][repeat_idx],
                    metrics['precision'][repeat_idx],
                    metrics['recall'][repeat_idx],
                ])

    summary_path = os.path.join(CSV_DIR, f'mia_results_{output_prefix}.csv')
    with open(summary_path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['dataset', 'model', 'epochs', 'repeats', 'shadow_models', 'encoding', 'augment', 'T', 'accuracy', 'auc', 'f1', 'precision', 'recall', 'significance_vs_ann'])
        for model in summary_results:
            sig_label = significance.get(model, {}).get('label', '')
            writer.writerow([
                dataset_flag,
                model,
                epochs,
                repeats,
                num_shadow_models,
                encoding,
                augment,
                timesteps,
                f"{summary_results[model]['accuracy']['mean']:.4f} ± {summary_results[model]['accuracy']['std']:.4f}",
                f"{summary_results[model]['auc']['mean']:.4f} ± {summary_results[model]['auc']['std']:.4f}",
                f"{summary_results[model]['f1']['mean']:.4f} ± {summary_results[model]['f1']['std']:.4f}",
                f"{summary_results[model]['precision']['mean']:.4f} ± {summary_results[model]['precision']['std']:.4f}",
                f"{summary_results[model]['recall']['mean']:.4f} ± {summary_results[model]['recall']['std']:.4f}",
                sig_label,
            ])

    return summary_results, significance, detailed_path, summary_path


def main(args):
    models = [name.strip() for name in args.models.split(',') if name.strip()]
    all_mia_results = {}

    for model in models:
        all_mia_results[model] = {'accuracy': [], 'auc': [], 'f1': [], 'precision': [], 'recall': []}
        print(f"\n=== 执行 {args.dataset} / {model} 的 MIA 攻击 ({args.repeats} 次重复) ===")
        for repeat in range(args.repeats):
            print(f"\n--- 第 {repeat + 1} 次重复实验 ---")
            metrics = run_mia_attack(
                model,
                dataset_flag=args.dataset,
                batch_size=args.batch_size,
                epochs=args.epochs,
                num_shadow_models=args.shadow_models,
                timesteps=args.timesteps,
                encoding=args.encoding,
                augment=not args.no_augment,
            )
            for metric in metrics:
                all_mia_results[model][metric].append(metrics[metric])

    output_prefix = args.output_prefix or args.dataset
    summary_results, significance, detailed_path, summary_path = summarize_results(
        all_mia_results,
        output_prefix=output_prefix,
        dataset_flag=args.dataset,
        repeats=args.repeats,
        num_shadow_models=args.shadow_models,
        epochs=args.epochs,
        timesteps=args.timesteps,
        encoding=args.encoding,
        augment=not args.no_augment,
    )

    print("\n=== MIA 攻击完成 ===")
    print(f"详细结果已保存到 {detailed_path}")
    print(f"汇总结果已保存到 {summary_path}")

    print("\nMIA 攻击完整指标 (均值 ± 标准差):")
    for model in summary_results:
        sig_label = significance.get(model, {}).get('label', '')
        print(f"\n  {model} {sig_label}:")
        for metric in ['accuracy', 'auc', 'f1', 'precision', 'recall']:
            print(f"    {metric}: {summary_results[model][metric]['mean']:.4f} ± {summary_results[model][metric]['std']:.4f}")

    if significance:
        print("\n统计检验结果 (双侧 t 检验 vs ANN, 仅对比 Accuracy):")
        for model in significance:
            print(f"  {model}: t={significance[model]['t_stat']:.4f}, p={significance[model]['p_value']:.4f} {significance[model]['label']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='bloodmnist', help='MedMNIST 数据集标识，例如 bloodmnist 或 pathmnist')
    parser.add_argument('--models', default='SNN,DenseSNN,ANN', help='逗号分隔的模型列表')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='每个影子模型训练轮数')
    parser.add_argument('--repeats', type=int, default=NUM_REPEATS, help='重复实验次数')
    parser.add_argument('--shadow-models', type=int, default=NUM_SHADOW_MODELS, help='影子模型数量')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='批次大小')
    parser.add_argument('--timesteps', type=int, default=T, help='SNN 时间步数')
    parser.add_argument('--encoding', choices=['direct', 'poisson'], default='direct', help='SNN 输入编码方式')
    parser.add_argument('--no-augment', action='store_true', help='关闭训练增强')
    parser.add_argument('--output-prefix', default=None, help='输出文件名前缀')
    main(parser.parse_args())
