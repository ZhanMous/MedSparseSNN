MedSparseSNN — 复试演讲稿（中英双语，约 2 分钟）

【中文部分】

引言
大家好，我叫（你的名字）。今天我汇报的工作题目是《MedSparseSNN：面向隐私保护与边缘高效推理的稀疏脉冲医学影像分类研究》。本工作关注的问题是：在医学影像分类任务上，稀疏脉冲神经网络（SNN）是否可以在保证准确率的同时带来更好的能效与隐私鲁棒性。

研究动机与贡献
近年来 SNN 在能耗建模上具有潜力，但真实硬件优势和隐私风险尚未被系统化比较。本工作有三点主要贡献：
- 设计并实现了可比较的三路基线（SNN、DenseSNN、ANN）在相同拓扑下的对照实验；
- 在 BloodMNIST 数据集上进行了多次重复实验，并同时测量准确率、成员推理攻击（MIA）成功率、理论 FLOPs 与实测功耗；
- 提出使用 DenseSNN（关闭稀疏优化）作为消融对照，量化稀疏计算带来的效益。

方法要点
模型采用 MS-ResNet 风格的轻量拓扑与可学习 PLIF 神经元，时间步长 T=6。图像通过 Poisson 泊松编码转换为脉冲序列用于 SNN；对照的 ANN 使用相同的空间拓扑但无时间维度。训练使用 AdamW、混合精度与多次不同随机种子重复以提高结果稳定性。

主要结果（概述）
在 BloodMNIST 上，ANN 的平均测试准确率约为 95.6%，SNN 为 93.6%，DenseSNN 为 92.1%。SNN 在理论 FLOPs 与稀疏计算下显示出能效优势；DenseSNN 证明了稀疏性本身带来的计算节省。关于隐私，MIA 实验显示不同模型在泄露风险上有差别（详见论文表格）。

可复现性与工程实践
仓库提供 `run_demo.sh`、`environment.yml`、smoke tests 以及生成图表的脚本，便于在面试后复现关键结果。需要注意的是，功耗测量依赖 `pynvml` 与 NVIDIA 驱动，且部分外部项目对 PyTorch/transformers 版本敏感，建议使用容器化环境以保证一致性。

局限性与未来方向
当前工作在单一数据集与小模型上验证，未来可以扩展到更大医学影像集并在真实 SNN 加速器或节能芯片上验证功耗收益。此外，进一步研究 SNN 在隐私保护策略（如差分隐私）下的行为也是有价值的方向。

结束语与答疑准备
谢谢聆听，接下来我可以展示一张图或表格。如果有问题，我准备回答关于 PLIF 选择、能效度量方法、DenseSNN 设计动机与复现步骤等问题。

【English Section】

Introduction
Hello, my name is (your name). The project I present is "MedSparseSNN: Sparse Spiking Neural Networks for Privacy-Aware and Edge-Efficient Medical Image Classification." The core question is whether sparse SNNs can achieve comparable accuracy while improving energy efficiency and offering different privacy robustness characteristics on medical image tasks.

Motivation and Contributions
SNNs promise low-energy inference but lack systematic comparisons in terms of accuracy, privacy leakage, and practical energy measurements. Our contributions are threefold:
- We implement three comparable baselines (SNN, DenseSNN, ANN) with the same topology for fair comparison;
- We conduct repeated experiments on BloodMNIST, measuring test accuracy, membership inference attack (MIA) success, theoretical FLOPs, and empirical power consumption;
- We propose DenseSNN (disabling sparse optimization) as an ablation to quantify benefits from sparsity.

Method Highlights
We use an MS-ResNet-like lightweight topology with PLIF neurons (learnable time-constant) and T=6 time steps. Images are encoded with Poisson spike encoding for SNN; ANN uses the same spatial topology without the temporal dimension. Training uses AdamW, mixed precision, and multiple seeds to ensure stable measurements.

Key Results (summary)
On BloodMNIST, ANN achieves ~95.6% test accuracy, SNN ~93.6%, and DenseSNN ~92.1%. SNN demonstrates energy-efficiency advantages in theoretical FLOPs and through sparse computation; DenseSNN validates that these gains come from sparsity rather than topology. MIA experiments reveal model-dependent privacy leakage patterns (see tables in the repo/paper).

Reproducibility and Practical Notes
The repository includes `run_demo.sh`, `environment.yml`, smoke tests, and plotting scripts to reproduce core results. Note that power measurement requires `pynvml` and NVIDIA drivers; some external dependencies are version-sensitive—using a container is recommended for reproducibility.

Limitations & Future Work
This study is limited to a single dataset and relatively small models. Future work includes scaling to larger medical datasets, evaluating on SNN accelerators, and integrating privacy-preserving training (e.g., DP-SGD) to study trade-offs.

Closing and Q&A
Thank you. I can walk through a figure or table now. I am ready to answer questions on PLIF choice, energy measurement methodology, DenseSNN rationale, or how to run the provided demo.

---
文件已保存: [docs/presentations/presentation_script.md](docs/presentations/presentation_script.md)
