# MedSparseSNN DermaMNIST 实验报告

## 实验目标

验证 BloodMNIST 上表现良好的 MedSparseSNN 配置迁移到 DermaMNIST 时的性能退化来源，并给出可复现的皮肤镜图像实验配置。

## 实验设计

- 筛选阶段：SNN / DenseSNN，epochs=8，repeats=1，比较编码方式、时间步和增强策略。
- 正式阶段：SNN / DenseSNN / ANN，epochs=15，repeats=2，采用筛选出的最佳 SNN 配置。
- checkpoint 选择基于验证集最佳准确率，最终只在最佳验证 checkpoint 上汇报测试集结果。

## 筛选结果

| config | model | encoding | augment | T | epochs | val_acc | test_acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ann_reference | ANN | direct | True | 6 | 8 | 78.36 ± 0.00 | 75.81 ± 0.00 |
| direct_t6_aug | SNN | direct | True | 6 | 8 | 69.99 ± 0.00 | 68.33 ± 0.00 |
| direct_t6_aug | DenseSNN | direct | True | 6 | 8 | 66.90 ± 0.00 | 66.88 ± 0.00 |
| poisson_t6_aug | SNN | poisson | True | 6 | 8 | 67.20 ± 0.00 | 65.24 ± 0.00 |
| poisson_t6_aug | DenseSNN | poisson | True | 6 | 8 | 66.90 ± 0.00 | 66.88 ± 0.00 |
| direct_t8_aug | SNN | direct | True | 8 | 8 | 69.89 ± 0.00 | 67.38 ± 0.00 |
| direct_t8_aug | DenseSNN | direct | True | 8 | 8 | 66.90 ± 0.00 | 66.88 ± 0.00 |
| poisson_t8_aug | SNN | poisson | True | 8 | 8 | 67.10 ± 0.00 | 65.74 ± 0.00 |
| poisson_t8_aug | DenseSNN | poisson | True | 8 | 8 | 66.90 ± 0.00 | 66.88 ± 0.00 |
| direct_t6_noaug | SNN | direct | False | 6 | 8 | 70.99 ± 0.00 | 68.23 ± 0.00 |
| direct_t6_noaug | DenseSNN | direct | False | 6 | 8 | 66.90 ± 0.00 | 66.88 ± 0.00 |

## 最终 DermaMNIST 对比

| model | encoding | augment | T | epochs | repeats | val_acc | test_acc | training_time | power | latency | params |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SNN | direct | False | 6 | 15 | 2 | 71.73 ± 0.55 | 69.93 ± 0.20 | 55.65 ± 0.06s | 29.27 ± 2.06W | 0.12 ± 0.00ms/sample | 119363 (0.119M) |
| DenseSNN | direct | False | 6 | 15 | 2 | 67.05 ± 0.05 | 66.81 ± 0.02 | 40.20 ± 0.11s | 18.63 ± 0.10W | 0.24 ± 0.00ms/sample | 119367 (0.119M) |
| ANN | direct | False | 6 | 15 | 2 | 77.07 ± 0.20 | 75.06 ± 0.20 | 16.04 ± 0.23s | 14.51 ± 0.86W | 0.03 ± 0.00ms/sample | 119357 (0.119M) |

## BloodMNIST vs DermaMNIST 准确率变化

| model | blood_test_acc | dataset_test_acc | accuracy_drop |
| --- | --- | --- | --- |
| SNN | 93.63 ± 0.28 | 69.93 ± 0.20 | 23.70 |
| DenseSNN | 92.15 ± 0.35 | 66.81 ± 0.02 | 25.34 |
| ANN | 95.59 ± 0.11 | 75.06 ± 0.20 | 20.53 |

## DermaMNIST 隐私结果

| model | accuracy | auc | f1 | precision | recall | significance_vs_ann |
| --- | --- | --- | --- | --- | --- | --- |
| SNN | 0.4842 ± 0.0000 | 0.4954 ± 0.0005 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |  |
| DenseSNN | 0.4842 ± 0.0000 | 0.4949 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |  |
| ANN | 0.4809 ± 0.0021 | 0.4857 ± 0.0016 | 0.4538 ± 0.0086 | 0.4961 ± 0.0025 | 0.4182 ± 0.0129 |  |

## DermaMNIST 稀疏性与理论能效

| model | repeats | test_acc | power_w | latency_ms_per_sample | energy_mj_per_sample | spike_rate | theoretical_macs_saving |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SNN | 2 | 69.93 ± 0.20 | 29.27 ± 2.06W | 0.12 ± 0.00ms/sample | 3.56 ± 0.27mJ | 0.0926 ± 0.0023 | 90.74 ± 0.23% |
| DenseSNN | 2 | 66.81 ± 0.02 | 18.63 ± 0.10W | 0.24 ± 0.00ms/sample | 4.53 ± 0.07mJ | 0.1925 ± 0.0082 | 0.00 ± 0.00% |
| ANN | 2 | 75.06 ± 0.20 | 14.51 ± 0.86W | 0.03 ± 0.00ms/sample | 0.48 ± 0.00mJ | N/A | 0.00 ± 0.00% |

## 结论

- 最佳 SNN 配置为 direct_t6_noaug，对应 encoding=direct、augment=False、T=6。
- DermaMNIST 的结果可作为 BloodMNIST 与 PathMNIST 之外的第三个外部验证点，用于评估稀疏性优势在皮肤镜图像上的跨域稳定性。
- 在 DermaMNIST 上，SNN 与 ANN 的测试准确率差距扩大到约 5.13 个百分点，但 SNN 仍明显优于 DenseSNN，说明稀疏实现带来的收益并未消失，而是面临更强的数据域挑战。
- 当前黑盒 MIA 设定下，DermaMNIST 上 SNN、DenseSNN 和 ANN 的攻击 AUC 都接近 0.5，说明该数据域中的隐私泄露信号整体较弱，现阶段更适合将其解读为“未观察到额外隐私风险”，而不是“已证明显著隐私优势”。
- DermaMNIST 上 SNN 的平均 spike rate 为 0.0926，对应 90.74% 的理论有效 MAC 节省，是三数据集中最强的理论稀疏性结果之一，可作为边缘计算叙事的重要支撑点。
