# MedSparseSNN

MedSparseSNN 是一个面向医疗影像隐私保护与边缘高效推理的研究型代码库，聚焦比较稀疏 SNN、非稀疏对照模型与 ANN 在准确率、成员推理攻击鲁棒性和理论能效上的差异。当前仓库同时保留论文材料、复现实验脚本和历史归档，因此更适合作为研究仓库使用，而不是通用 Python 包。

## 当前可运行入口

- train.py：训练 SNN、DenseSNN、ANN，并生成训练摘要。
- mia_attack.py：执行成员推理攻击实验。
- calculate_flops.py：计算理论计算量。
- scripts/run_medmnist_study.py：运行 DermaMNIST、PathMNIST 等 MedMNIST 数据集的筛选与正式对比。
- scripts/run_pathology_study.py：运行 PathMNIST 专项实验并生成报告。
- scripts/analyze_medmnist_privacy_efficiency.py：汇总 spike rate、理论有效 MAC 节省和单样本能耗。
- test_transformer.py 与 tests/test_smoke.py：轻量测试与一致性检查。

## 环境准备

推荐直接使用仓库中锁定的依赖版本。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果使用 Conda：

```bash
conda env create -f environment.yml
conda activate medsparsesnn
```

## 快速开始

训练默认主实验：

```bash
python train.py
```

运行指定模型或数据集：

```bash
python train.py DenseSNN
python train.py --dataset pathmnist --epochs 15 --repeats 2 --output-prefix pathmnist_baseline
```

运行隐私攻击：

```bash
python mia_attack.py
```

运行完整跨数据集实验：

```bash
python scripts/run_medmnist_study.py --dataset pathmnist --screen-epochs 8 --final-epochs 15 --final-repeats 2
python scripts/run_medmnist_study.py --dataset dermamnist --screen-epochs 8 --final-epochs 15 --final-repeats 2
```

实验报告默认写入 docs/reports，CSV、模型权重和图像等运行产物继续写入 outputs。

运行测试：

```bash
pytest
```

## 仓库结构

```text
.
├── config.py
├── models.py
├── train.py
├── mia_attack.py
├── calculate_flops.py
├── data/
├── scripts/
├── tests/
├── outputs/
├── docs/
│   ├── reports/
│   ├── presentations/
│   └── submission/
├── archive/
│   └── simulated_examples/
├── paper/
├── FINAL_RESEARCH_PAPER.md
└── FINAL_RESEARCH_PAPER_EN.md
```

## 目录约定

- outputs：训练产物、CSV 汇总、图像和模型权重。
- outputs/checkpoints：模型权重与实验 checkpoint。
- outputs/csv：训练摘要、MIA 结果、效率统计等结构化结果。
- outputs/figures：论文和分析图表。
- outputs/paper-build：论文 PDF 与构建中间文件。
- outputs/tables：自动生成的 LaTeX 表格。
- docs/reports：自动生成或人工整理的实验报告。
- docs/presentations：答辩和展示材料。
- docs/submission：补充材料、回复审稿意见等投稿文档。
- archive/simulated_examples：历史脚本与旧版材料，不作为当前主流程入口。

## 仓库重命名

当前工作区目录名仍是 HemoSparse，这是为了避免在活跃编辑会话中直接移动根目录导致路径失效。若需要在文件系统层面完成最终重命名，可执行 [scripts/rename_repo.sh](scripts/rename_repo.sh)。

## 已知限制

- 功耗测量依赖 NVIDIA NVML；若本机不可用，训练摘要中的 power 字段会显示为 N/A。
- 多个实验脚本仍然保持研究脚本风格，尚未抽象为稳定库接口。
- outputs 中已有结果文件属于历史实验产物，不保证和当前代码完全同步。

## 致谢

- MedMNIST：https://medmnist.com/
- SpikingJelly：https://github.com/fangwei123456/spikingjelly
- PyTorch：https://pytorch.org/
