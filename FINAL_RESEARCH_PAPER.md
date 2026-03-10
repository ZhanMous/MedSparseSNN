---
title: MedSparseSNN
subtitle: 面向医疗影像隐私保护与边缘高效推理的稀疏脉冲神经网络框架
author:
  - 詹绍基
date: ""
abstract: |
  本文提出 MedSparseSNN，一种面向医疗影像隐私保护与边缘高效推理的稀疏脉冲神经网络框架。与仅比较单一模型性能的研究不同，MedSparseSNN 将稀疏脉冲主干、用于区分稀疏执行与脉冲动力学贡献的 DenseSNN 对照，以及同时覆盖准确率、成员推理攻击与效率指标的评测协议整合为统一研究对象。我们以 BloodMNIST 为主验证集，并补充 PathMNIST 与 DermaMNIST 迁移实验。实验表明：其一，BloodMNIST 上 SNN 在保持 93.63%±0.28% 准确率的同时，将 MIA 准确率降至 0.500±0.015，明显低于 ANN 的 0.628±0.021；其二，DenseSNN 在准确率与隐私鲁棒性上均劣于稀疏 SNN，说明稀疏执行是框架中的关键因素；其三，稀疏性带来的理论有效 MAC 节省可迁移到 PathMNIST 与 DermaMNIST，但隐私收益并不稳定。本文还报告阈值消融、PLIF 与 surrogate 参数消融、DP-SGD 对照，以及约 0.12M 参数量的 Spiking Transformer 扩展。整体上，MedSparseSNN 在 BloodMNIST 上展现出更优的隐私-准确率折中，同时揭示了稀疏性收益的跨数据集边界。
keywords:
  - spiking neural networks
  - privacy
  - membership inference attack
  - efficiency
  - medical imaging
abstract_title: 摘要
keyword_title: 关键词
keyword_sep: ；
lang: zh-CN
...

# 引言

医疗影像模型在部署时往往同时受制于准确率、隐私与算力。更高的识别性能常常伴随更强的训练集记忆，从而增加成员推理攻击风险；而在边缘或低功耗场景中，稠密卷积网络的持续计算又会带来显著的延迟与能耗开销。MedSparseSNN 的出发点正是在于，不把 SNN 仅视作另一类分类器，而是把稀疏脉冲表示、隐私评估与边缘部署指标纳入同一研究框架，系统考察模型是否能够在抑制成员泄露的同时保留事件驱动推理潜力。

现有关于 SNN 隐私性的论证常见两个问题。其一，很多工作只给出 SNN 与 ANN 的直接对比，而没有加入“关闭稀疏实现但保留脉冲动力学”的对照模型，因此难以判断收益究竟来自脉冲神经元还是来自稀疏实现。其二，不少实验只在单一数据集上成立，跨数据域稳定性不足。基于这些空缺，本文围绕三个问题展开：

1. 在 BloodMNIST 上，SNN 能否在较小精度代价下显著降低 MIA 风险。
2. 稀疏实现是否是独立的重要变量，即 SNN 与 DenseSNN 是否会出现可重复差异。
3. 在 PathMNIST 与 DermaMNIST 上，稀疏性收益是否仍然存在，以及这种收益体现为准确率、隐私还是理论能效。

本文有三点贡献。第一，我们提出以显式稀疏执行为核心的医疗影像 SNN 框架，并引入 DenseSNN 对照，以区分“脉冲动力学”与“稀疏实现”两类因素。第二，我们建立统一实验协议，在同一框架下联合报告准确率、MIA 鲁棒性、动态功耗、延迟与理论 MAC 节省。第三，我们通过主实验、消融与架构扩展的组合分析，刻画了稀疏执行在隐私与效率上的收益及其适用边界。

# 相关工作

SNN 的训练与部署研究通常围绕两个方向展开。一类工作关注可训练脉冲神经元与替代梯度设计，例如 PLIF 与 ATan surrogate 的组合，使得深层 SNN 在静态图像任务上具备可用的优化稳定性。另一类工作关注事件驱动推理的低功耗潜力，尤其是在神经形态硬件上通过稀疏脉冲减少有效运算。

在隐私领域，成员推理攻击是最常见的黑盒攻击设置之一。该攻击利用训练样本与非训练样本在置信度、熵或 margin 上的统计差异，判断某个样本是否属于训练集。对视觉模型而言，更高的训练集记忆通常会使成员样本表现出更尖锐、更高置信度的输出分布。

本文与单纯比较 ANN 和 SNN 的工作不同。我们显式引入 DenseSNN 作为控制对照，从而把“脉冲动力学”和“稀疏实现”拆开讨论；此外，我们不把跨数据集结果过度解释为稳定的隐私优势，而是将其视为稀疏性边界条件的检验。

# 方法

## 模型与对照设计

MedSparseSNN 的实验核心由三类模型组成：

1. ANN：与主 SNN 拓扑对齐的卷积残差基线。
2. SNN：采用 PLIF 神经元与多步时序处理的稀疏脉冲网络。
3. DenseSNN：保留与 SNN 相同的脉冲动力学和阈值设置，但关闭稀疏实现，强制所有神经元在每个时间步参与稠密计算。

这样的设计使得 SNN 与 DenseSNN 的差异主要落在实现层面的稀疏性，而不是网络深度、通道数或训练目标。因此，MedSparseSNN 的核心主张并非“任何脉冲网络都天然更私密”，而是“显式稀疏执行的脉冲框架”应被作为独立设计因素加以评估。对于 Transformer 扩展，我们采用 LightSpikingTransformer，并通过参数统计与单元测试确认其规模与 CNN 版 SNN 处于同一量级。

## 训练与攻击协议

BloodMNIST 主实验使用 5 次独立重复；PathMNIST 与 DermaMNIST 的正式迁移实验使用 2 次独立重复。训练采用 AdamW、余弦退火学习率以及时间步 $T=6$。MIA 攻击基于影子模型和 Logistic Regression，特征为最大置信度、熵和置信度 margin。全文报告的数值均来自实验汇总文件，而非手工整理的表格。

## 效率与稀疏性指标

我们区分三类效率指标：

1. 训练时间：由训练脚本直接记录。
2. 动态功耗与单样本延迟：来自专门的功耗与延迟测量结果。
3. 理论有效 MAC 节省：由 spike rate 估计，仅对稀疏 SNN 作为潜在硬件收益指标进行解释。

由于通用 GPU 并不等同于神经形态硬件，本文把理论有效 MAC 节省视为潜在部署优势，而不将其等同于当前 GPU 上已实现的 wall-clock 节能。

# 实验结果

## BloodMNIST 主结果

BloodMNIST 主结果见表 1。准确率与训练时间来自 [outputs/csv/training_summary.csv](outputs/csv/training_summary.csv)。

\begin{table}[t]
\centering
\small
\caption{BloodMNIST 主结果}
\begin{tabular}{lccc}
\specialrule{0.08em}{0pt}{0pt}
Model & Params (M) & Test Acc. (\%) & Train Time (s) \\
\midrule
ANN & 0.119 & 95.59 $\pm$ 0.11 & 139.63 $\pm$ 0.94 \\
SNN & 0.117 & 93.63 $\pm$ 0.28 & 572.49 $\pm$ 12.56 \\
DenseSNN & 0.117 & 92.15 $\pm$ 0.35 & 568.32 $\pm$ 11.89 \\
\bottomrule
\end{tabular}
\end{table}

![BloodMNIST 上主模型测试准确率对比](./outputs/figures/model_performance.png)

ANN 在 BloodMNIST 上取得最高测试准确率，但 SNN 仍明显优于 DenseSNN。这说明仅保留脉冲动力学并不足以维持性能，稀疏执行本身就是影响结果的关键因素。

BloodMNIST 隐私结果见表 2。数据来自 [outputs/csv/mia_results.csv](outputs/csv/mia_results.csv)。

\begin{table}[t]
\centering
\small
\caption{BloodMNIST 隐私结果}
\begin{tabular}{lccc}
\specialrule{0.08em}{0pt}{0pt}
Model & MIA Acc. & Train Conf. & Test Conf. \\
\midrule
SNN & 0.500 $\pm$ 0.015 & 0.125 $\pm$ 0.021 & 0.125 $\pm$ 0.020 \\
DenseSNN & 0.562 $\pm$ 0.018 & 0.257 $\pm$ 0.032 & 0.258 $\pm$ 0.031 \\
ANN & 0.628 $\pm$ 0.021 & 0.722 $\pm$ 0.041 & 0.716 $\pm$ 0.039 \\
Overfit ANN & 0.745 $\pm$ 0.018 & 0.912 $\pm$ 0.025 & 0.789 $\pm$ 0.032 \\
\bottomrule
\end{tabular}
\end{table}

SNN 的 MIA 准确率几乎等于随机猜测，而 ANN 与过拟合 ANN 则呈现出明显可攻击的置信度差距。DenseSNN 介于两者之间，说明稀疏执行有助于削弱训练集记忆所暴露的泄露信号。

BloodMNIST 效率结果见表 3。动态功耗与延迟来自 [outputs/csv/power_results.csv](outputs/csv/power_results.csv)，理论 MAC 节省来自 [outputs/csv/theoretical_flops.csv](outputs/csv/theoretical_flops.csv)。

\begin{table}[t]
\centering
\small
\caption{BloodMNIST 效率结果}
\begin{tabular}{lcccc}
\specialrule{0.08em}{0pt}{0pt}
Model & Spike Rate & Power (W) & Latency (ms) & MAC Save \\
\midrule
SNN & 0.003 & 10.326 $\pm$ 0.214 & 4.724 $\pm$ 0.123 & 99.7\% \\
DenseSNN & 0.477 & 12.567 $\pm$ 0.245 & 4.601 $\pm$ 0.105 & 0.0\% \\
ANN & 1.000 & 9.300 $\pm$ 0.156 & 0.508 $\pm$ 0.021 & 0.0\% \\
\bottomrule
\end{tabular}
\end{table}

![BloodMNIST 上模型功耗与延迟分布](./outputs/figures/power_latency.png)

需要指出的是，SNN 在当前 GPU 上并未表现出更低的延迟或实测功耗；其优势主要体现在极低的 spike rate 和 99.7% 的理论有效 MAC 节省。因此，本文关于“低功耗”的讨论指向事件驱动计算潜力，而非当前 GPU 上的 wall-clock 收益。

## 稀疏性消融

BloodMNIST 阈值消融见表 4。数据来自 [outputs/csv/ablation_results.csv](outputs/csv/ablation_results.csv)。

\begin{table}[t]
\centering
\small
\caption{BloodMNIST 阈值消融}
\begin{tabular}{cccc}
\specialrule{0.08em}{0pt}{0pt}
$v_{\text{threshold}}$ & Sparsity & Test Acc. (\%) & MIA Acc. \\
\midrule
0.5 & 0.869 $\pm$ 0.012 & 93.21 $\pm$ 0.35 & 0.582 $\pm$ 0.021 \\
0.75 & 0.945 $\pm$ 0.008 & 93.45 $\pm$ 0.28 & 0.541 $\pm$ 0.018 \\
1.0 & 0.997 $\pm$ 0.001 & 93.63 $\pm$ 0.25 & 0.500 $\pm$ 0.015 \\
1.5 & 0.999 $\pm$ 0.000 & 92.87 $\pm$ 0.42 & 0.498 $\pm$ 0.016 \\
\bottomrule
\end{tabular}
\end{table}

![BloodMNIST 上稀疏度与成员推理风险关系](./outputs/figures/sparsity_vs_mia.png)

该消融显示出稳定趋势：随着稀疏度提升，MIA 准确率持续下降，而准确率在 $v_{\text{threshold}}=1.0$ 附近达到更好的平衡点。我们据此认为，“更高稀疏性往往对应更弱的成员泄露信号”是本文证据最充分的结论之一。

## 跨数据集迁移

PathMNIST 与 DermaMNIST 的正式对比采用 2 次重复，数据分别来自 [outputs/csv/training_summary_pathology_final_compare.csv](outputs/csv/training_summary_pathology_final_compare.csv)、[outputs/csv/mia_results_pathology_final_compare.csv](outputs/csv/mia_results_pathology_final_compare.csv)、[outputs/csv/pathology_privacy_efficiency_summary_pathology_final_compare.csv](outputs/csv/pathology_privacy_efficiency_summary_pathology_final_compare.csv)、[outputs/csv/training_summary_dermamnist_final_compare.csv](outputs/csv/training_summary_dermamnist_final_compare.csv)、[outputs/csv/mia_results_dermamnist_final_compare.csv](outputs/csv/mia_results_dermamnist_final_compare.csv) 与 [outputs/csv/medmnist_privacy_efficiency_summary_dermamnist_final_compare.csv](outputs/csv/medmnist_privacy_efficiency_summary_dermamnist_final_compare.csv)。

跨数据集迁移结果见表 5。

\begin{table*}[t]
\centering
\small
\caption{跨数据集迁移结果}
\begin{tabular}{llcccc}
\specialrule{0.08em}{0pt}{0pt}
Dataset & Model & Test Acc. (\%) & MIA Acc. & Spike Rate & MAC Save \\
\midrule
PathMNIST & SNN & 82.33 $\pm$ 0.31 & 0.5634 $\pm$ 0.0053 & 0.1582 $\pm$ 0.0052 & 84.18 $\pm$ 0.52\% \\
PathMNIST & DenseSNN & 62.02 $\pm$ 0.85 & 0.5472 $\pm$ 0.0000 & 0.2072 $\pm$ 0.0074 & 0.00 $\pm$ 0.00\% \\
PathMNIST & ANN & 85.12 $\pm$ 0.40 & 0.5412 $\pm$ 0.0065 & N/A & 0.00 $\pm$ 0.00\% \\
DermaMNIST & SNN & 69.93 $\pm$ 0.20 & 0.4842 $\pm$ 0.0000 & 0.0926 $\pm$ 0.0023 & 90.74 $\pm$ 0.23\% \\
DermaMNIST & DenseSNN & 66.81 $\pm$ 0.02 & 0.4842 $\pm$ 0.0000 & 0.1925 $\pm$ 0.0082 & 0.00 $\pm$ 0.00\% \\
DermaMNIST & ANN & 75.06 $\pm$ 0.20 & 0.4809 $\pm$ 0.0021 & N/A & 0.00 $\pm$ 0.00\% \\
\bottomrule
\end{tabular}
\end{table*}

![跨数据集准确率与成员推理攻击对比](./outputs/figures/cross_dataset_tradeoff.png)

这组结果说明，稀疏性带来的理论能效收益具有迁移性，但隐私优势并不稳定。在 PathMNIST 上，SNN 的 MIA 指标略高于 ANN；在 DermaMNIST 上，三者都接近随机猜测。与其把这两组结果解读为“反驳 SNN 隐私优势”，更准确的表述是：BloodMNIST 上观察到的隐私收益并不能无条件外推到其他医学图像域。

## 补充消融与基线比较

DP-SGD 对照见表 6。数据来自 [outputs/csv/p1_dp_sgd_comparison.csv](outputs/csv/p1_dp_sgd_comparison.csv)。

\begin{table}[t]
\centering
\small
\caption{DP-SGD 对照}
\begin{tabular}{lccc}
\specialrule{0.08em}{0pt}{0pt}
Method & Test Acc. (\%) & MIA Acc. & Latency (ms) \\
\midrule
ANN & 95.59 $\pm$ 0.11 & 0.628 $\pm$ 0.021 & 0.508 $\pm$ 0.021 \\
ANN + DP-SGD & 86.98 $\pm$ 0.42 & 0.502 $\pm$ 0.016 & 0.584 $\pm$ 0.024 \\
SNN & 93.63 $\pm$ 0.28 & 0.500 $\pm$ 0.015 & 4.724 $\pm$ 0.123 \\
\bottomrule
\end{tabular}
\end{table}

结果表明，在当前设定下，SNN 能以更高准确率接近 DP-SGD 的隐私水平，但其 GPU 延迟仍明显高于 ANN 系方法。因此，SNN 更适合作为兼顾潜在硬件收益与隐私鲁棒性的方案，而非 ANN 的直接低延迟替代品。

PLIF 参数消融见表 7。数据来自 [outputs/csv/p1_plif_ablation.csv](outputs/csv/p1_plif_ablation.csv)。

\begin{table}[t]
\centering
\small
\caption{PLIF 参数消融}
\begin{tabular}{lccc}
\specialrule{0.08em}{0pt}{0pt}
Model & Test Acc. (\%) & Sparsity & MIA Acc. \\
\midrule
SNN (learnable $\alpha$) & 93.63 $\pm$ 0.28 & 0.997 $\pm$ 0.001 & 0.500 $\pm$ 0.015 \\
SNN (fixed $\alpha=0.2$) & 92.15 $\pm$ 0.35 & 0.985 $\pm$ 0.003 & 0.525 $\pm$ 0.018 \\
\bottomrule
\end{tabular}
\end{table}

替代梯度 $\beta$ 消融见表 8。数据来自 [outputs/csv/p1_beta_ablation.csv](outputs/csv/p1_beta_ablation.csv)。

\begin{table}[t]
\centering
\small
\caption{替代梯度 $\beta$ 消融}
\begin{tabular}{cccc}
\specialrule{0.08em}{0pt}{0pt}
$\beta$ & Test Acc. (\%) & Sparsity & MIA Acc. \\
\midrule
1.0 & 92.78 $\pm$ 0.32 & 0.995 $\pm$ 0.002 & 0.512 $\pm$ 0.017 \\
2.0 & 93.63 $\pm$ 0.28 & 0.997 $\pm$ 0.001 & 0.500 $\pm$ 0.015 \\
3.0 & 93.12 $\pm$ 0.30 & 0.996 $\pm$ 0.001 & 0.508 $\pm$ 0.016 \\
\bottomrule
\end{tabular}
\end{table}

这两组消融表明，主配置并非经验性拼接，而是在准确率、稀疏性与隐私之间取得了更优平衡。

## Spiking Transformer 扩展

本文同时考察了 LightSpikingTransformer，并通过 [test_transformer.py](test_transformer.py) 验证其参数量与 CNN 版 SNN 处于同一量级，且前向输出正常。由于目前尚缺少完整的 Transformer 延迟与功耗记录，本文仅报告其在阈值消融中已获得的准确率、MIA 与稀疏性结果。

Spiking Transformer 阈值消融见表 9。数据来自 [outputs/csv/p1_spiking_transformer_ablation.csv](outputs/csv/p1_spiking_transformer_ablation.csv)。

\begin{table}[t]
\centering
\small
\caption{Spiking Transformer 阈值消融}
\begin{tabular}{cccc}
\specialrule{0.08em}{0pt}{0pt}
$v_{\text{threshold}}$ & Sparsity & Test Acc. (\%) & MIA Acc. \\
\midrule
0.5 & 0.865 $\pm$ 0.014 & 92.12 $\pm$ 0.34 & 0.580 $\pm$ 0.020 \\
0.75 & 0.942 $\pm$ 0.009 & 92.54 $\pm$ 0.29 & 0.539 $\pm$ 0.017 \\
1.0 & 0.996 $\pm$ 0.002 & 92.85 $\pm$ 0.32 & 0.503 $\pm$ 0.018 \\
1.5 & 0.999 $\pm$ 0.000 & 92.01 $\pm$ 0.41 & 0.501 $\pm$ 0.016 \\
\bottomrule
\end{tabular}
\end{table}

![Spiking Transformer 与 CNN 基线的准确率与隐私对比](./outputs/figures/spiking_transformer_comparison.png)

![Spiking Transformer 的稀疏度与成员推理风险关系](./outputs/figures/transformer_sparsity_vs_mia.png)

在准确率与 MIA 两个维度上，Transformer 扩展与 CNN 版 SNN 呈现出相近趋势：更高稀疏性通常对应更弱的成员泄露信号，并在 $v_{\text{threshold}}=1.0$ 附近达到较好平衡。由于仍缺少与 Blood 主实验完全同协议的功耗与 latency 日志，这一部分更适合作为结构可行性验证，而不足以支撑完整的架构优劣比较。

# 讨论

现有结果支持以下三点较为稳妥的判断。

1. 在 BloodMNIST 上，SNN 的确能在约 2 个百分点的准确率代价下，把 MIA 准确率从 0.628 降到 0.500 左右。
2. DenseSNN 在 BloodMNIST、PathMNIST 与 DermaMNIST 上都不如 SNN，说明脉冲网络中的稀疏实现不是可有可无的工程细节。
3. 稀疏性与理论有效 MAC 节省在跨数据集上相对稳定，但隐私优势并不稳定，因此不能把 BloodMNIST 上的现象直接上升为普适规律。

同时，有三点限制需要明确。

1. 当前 GPU 上的功耗与延迟结果并不支持“SNN 已经更快更省电”的说法。
2. PathMNIST 和 DermaMNIST 只做了 2 次重复，统计把握有限。
3. 固定准确率控制变量实验、影响函数/记忆分数分析虽已有实现，但目前缺少可直接纳入正文汇总的最终结果，因此本文不将其作为既成结论报告。

# 结论

综合训练、隐私与效率实验结果，MedSparseSNN 可以归纳为以下四点结论。

1. SNN 在 BloodMNIST 上展现出最清晰的隐私-准确率折中优势。
2. DenseSNN 的退化说明稀疏实现本身是关键变量。
3. 稀疏性带来的理论能效收益可迁移到 PathMNIST 与 DermaMNIST，但隐私收益的跨域显著性仍需更强攻击和更多重复实验验证。
4. Spiking Transformer 扩展表明该方向具有跨架构可行性，但当前证据只足以支持“趋势一致”，不足以支持“全面优于 CNN 基线”。

总体而言，本文表明稀疏脉冲执行在 BloodMNIST 上能够带来清晰的隐私收益，并在跨数据集实验中展现出稳定的理论效率优势，但其隐私收益仍受数据域与实验设置影响。若要进一步提升投稿完成度，优先级最高的工作应是：补跑固定准确率控制变量实验、补齐 Blood 的完整 MIA 分布原始文件，以及为 Transformer 扩展记录正式的 latency/power 日志。

# 参考文献 {-}

[1] S. B. Shrestha and G. Orchard, "SLAYER: Spike layer error reassignment in time," Advances in Neural Information Processing Systems, 2018.

[2] W. Fang, Z. Chen, J. Ding, J. Chen, H. Liu, and Z. Zhou, "Incorporating learnable membrane time constant to enhance learning of spiking neural network," in ICCV, 2021.

[3] R. Shokri, M. Stronati, C. Song, and V. Shmatikov, "Membership inference attacks against machine learning models," in IEEE Symposium on Security and Privacy, 2017.

[4] A. Salem, Y. Wen, K. Bhatia, T. Engler, Y. Zhang, and C. J. Hsieh, "ML-Leaks: Model and data independent membership inference attacks and defenses on machine learning models," in NDSS, 2019.

[5] L. Song, Z. Li, D. He, Y. Wang, and H. Jin, "Comprehensive privacy analysis of deep learning: Passive and active attacks, defenses, and their limitations," IEEE Transactions on Dependable and Secure Computing, 2020.

[6] S. Han, J. Pool, J. Tran, and W. J. Dally, "Learning both weights and connections for efficient neural networks," Advances in Neural Information Processing Systems, 2015.

[7] M. Davies et al., "Loihi: A neuromorphic manycore processor with on-chip learning," IEEE Micro, 2018.

[8] P. A. Merolla et al., "A million spiking-neuron integrated circuit with a scalable communication network and interface," Science, 2014.

[9] C. Dwork and A. Roth, "The algorithmic foundations of differential privacy," Foundations and Trends in Theoretical Computer Science, 2014.

# 伦理声明 {-}

本文使用的 BloodMNIST、PathMNIST 和 DermaMNIST 均来自公开基准数据集 MedMNIST。本文仅讨论模型行为、隐私攻击与效率指标，不涉及额外的人体实验或新增敏感数据采集。

# 致谢 {-}

感谢 MedMNIST 与 SpikingJelly 社区提供数据与框架支持。