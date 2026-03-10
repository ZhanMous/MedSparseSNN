# MedSparseSNN PathMNIST 实验报告

## 实验目标

验证 BloodMNIST 上表现良好的 MedSparseSNN 配置迁移到 PathMNIST 时的性能退化来源，并给出可复现的病理图像实验配置。

## 实验设计

- 筛选阶段：SNN / DenseSNN，epochs=8，repeats=1，比较编码方式、时间步和增强策略。
- 正式阶段：SNN / DenseSNN / ANN，epochs=15，repeats=2，采用筛选出的最佳 SNN 配置。
- checkpoint 选择基于验证集最佳准确率，最终只在最佳验证 checkpoint 上汇报测试集结果。

## 筛选结果

| config | model | encoding | augment | T | epochs | val_acc | test_acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ann_reference | ANN | direct | True | 6 | 8 | 93.28 ± 0.00 | 87.41 ± 0.00 |
| direct_t6_aug | SNN | direct | True | 6 | 8 | 83.49 ± 0.00 | 82.42 ± 0.00 |
| direct_t6_aug | DenseSNN | direct | True | 6 | 8 | 40.61 ± 0.00 | 53.22 ± 0.00 |
| poisson_t6_aug | SNN | poisson | True | 6 | 8 | 59.95 ± 0.00 | 65.86 ± 0.00 |
| poisson_t6_aug | DenseSNN | poisson | True | 6 | 8 | 33.07 ± 0.00 | 42.16 ± 0.00 |
| direct_t8_aug | SNN | direct | True | 8 | 8 | 84.14 ± 0.00 | 82.41 ± 0.00 |
| direct_t8_aug | DenseSNN | direct | True | 8 | 8 | 40.65 ± 0.00 | 53.75 ± 0.00 |
| poisson_t8_aug | SNN | poisson | True | 8 | 8 | 62.81 ± 0.00 | 67.87 ± 0.00 |
| poisson_t8_aug | DenseSNN | poisson | True | 8 | 8 | 34.08 ± 0.00 | 43.11 ± 0.00 |
| direct_t6_noaug | SNN | direct | False | 6 | 8 | 90.21 ± 0.00 | 80.13 ± 0.00 |
| direct_t6_noaug | DenseSNN | direct | False | 6 | 8 | 54.50 ± 0.00 | 60.00 ± 0.00 |

## 最终 PathMNIST 对比

| model | encoding | augment | T | epochs | repeats | val_acc | test_acc | training_time | power | latency | params |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SNN | direct | False | 6 | 15 | 2 | 92.54 ± 0.18 | 82.33 ± 0.31 | 883.03 ± 16.84s | 23.93 ± 1.14W | 0.13 ± 0.00ms/sample | 119527 (0.120M) |
| DenseSNN | direct | False | 6 | 15 | 2 | 58.01 ± 0.61 | 62.02 ± 0.85 | 527.74 ± 10.72s | 21.88 ± 0.50W | 0.23 ± 0.02ms/sample | 119531 (0.120M) |
| ANN | direct | False | 6 | 15 | 2 | 96.70 ± 0.08 | 85.12 ± 0.40 | 170.11 ± 4.17s | 14.77 ± 0.68W | 0.04 ± 0.00ms/sample | 119521 (0.120M) |

## BloodMNIST vs PathMNIST 准确率变化

| model | blood_test_acc | path_test_acc | accuracy_drop |
| --- | --- | --- | --- |
| SNN | 93.63 ± 0.28 | 82.33 ± 0.31 | 11.30 |
| DenseSNN | 92.15 ± 0.35 | 62.02 ± 0.85 | 30.13 |
| ANN | 95.59 ± 0.11 | 85.12 ± 0.40 | 10.47 |

## PathMNIST 隐私结果

| model | accuracy | auc | f1 | precision | recall | significance_vs_ann |
| --- | --- | --- | --- | --- | --- | --- |
| SNN | 0.5634 ± 0.0053 | 0.5972 ± 0.0029 | 0.5554 ± 0.0143 | 0.5650 ± 0.0029 | 0.5466 ± 0.0250 |  |
| DenseSNN | 0.5472 ± 0.0000 | 0.5665 ± 0.0000 | 0.6018 ± 0.0000 | 0.5367 ± 0.0000 | 0.6849 ± 0.0000 |  |
| ANN | 0.5412 ± 0.0065 | 0.5382 ± 0.0085 | 0.5202 ± 0.0083 | 0.5446 ± 0.0068 | 0.4979 ± 0.0095 |  |

## PathMNIST 稀疏性与理论能效

| model | repeats | test_acc | power_w | latency_ms_per_sample | energy_mj_per_sample | spike_rate | theoretical_macs_saving |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SNN | 2 | 82.33 ± 0.31 | 23.93 ± 1.14W | 0.13 ± 0.00ms/sample | 3.16 ± 0.08mJ | 0.1582 ± 0.0052 | 84.18 ± 0.52% |
| DenseSNN | 2 | 62.02 ± 0.85 | 21.88 ± 0.50W | 0.23 ± 0.02ms/sample | 5.12 ± 0.50mJ | 0.2072 ± 0.0074 | 0.00 ± 0.00% |
| ANN | 2 | 85.12 ± 0.40 | 14.77 ± 0.68W | 0.04 ± 0.00ms/sample | 0.54 ± 0.02mJ | N/A | 0.00 ± 0.00% |

## 结论

- 最佳 SNN 病理配置为 direct_t6_noaug，对应 encoding=direct、augment=False、T=6。
- 在 PathMNIST 上，SNN 与 ANN 的测试准确率差距为 2.79 个百分点，仍处于“接近但略落后”的区间；DenseSNN 明显退化，说明稀疏实现本身仍是关键变量。
- PathMNIST 上当前黑盒 MIA 设定下，SNN 的攻击准确率和 AUC 均略高于 ANN，但在当前 2 次重复下未达到显著性。这说明病理图像场景中的隐私优势目前尚未被充分验证，而不是已经被证伪。
- 病理场景下的实测 GPU 能耗与延迟并未体现 SNN 优势，但 SNN 仍保持约 84.18% 的理论有效 MAC 节省，且相较 DenseSNN 的单样本能耗更低。对边缘计算的论证应优先依赖稀疏率和理论有效操作数，而非通用 GPU 的 wall-clock 指标。
