**MedSparseSNN — 1 分钟口头稿**

您好，我是（姓名），我的项目叫 MedSparseSNN，目标是评估稀疏脉冲神经网络在医学影像分类上的可行性与能效。我们在 BloodMNIST 上比较三类模型——SNN、关闭稀疏的 DenseSNN 和常规 ANN，采用 MS-ResNet 风格网络与 PLIF 神经元，并用 Poisson 编码处理输入。评估维度包括 test accuracy、对成员推断攻击（MIA）的鲁棒性、理论 FLOPs 以及实测 GPU 功耗。主要结论：SNN 在准确率上可与 ANN 持平，但在理论 FLOPs 与稀疏计算场景下展现出能效优势；DenseSNN 作为消融基线验证稀疏带来的收益与实现开销。工作附带完整代码与复现脚本，便于延展与现场讨论。谢谢。

**常见问答（简短回答）**

- **为什么选 PLIF？**: PLIF 可学习膜电位时间常数，使神经元动力学更灵活，利于收敛与表现。
- **如何衡量能效？**: 结合理论 FLOPs 与实测 GPU 功耗和推理延迟，提供任务级别能效对比。
- **DenseSNN 的作用？**: 作为消融基线，区分稀疏结构带来的计算/功耗收益与实现开销。
- **MIA 实验关键设置？**: 保持攻击模型一致、成员/非成员划分稳定，重复多次并报告均值与置信区间。

---
文件: [docs/presentations/presentation_one_slide.md](docs/presentations/presentation_one_slide.md)
