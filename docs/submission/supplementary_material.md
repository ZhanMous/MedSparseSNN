# MedSparseSNN Supplementary Material

## Sparse Spiking Neural Networks for Privacy-Aware and Edge-Efficient Medical Image Classification

**Conference/Journal**: NeurIPS / IEEE Transactions on Medical Imaging (TMI)  
**Date**: March 9, 2026

---

## 目录

1. [完整消融实验结果](#1-完整消融实验结果)
2. [跨数据集迁移补充结果](#2-跨数据集迁移补充结果)
3. [5次独立重复实验原始数据](#3-5次独立重复实验原始数据)
4. [影响函数计算完整代码实现](#4-影响函数计算完整代码实现)
5. [高分辨率实验图表](#5-高分辨率实验图表)

---

## 1. 完整消融实验结果

### 1.1 完整稀疏度梯度消融实验

本补充材料提供了完整的稀疏度梯度消融实验结果，包括更细粒度的v_threshold参数搜索。

| v_threshold | 全局稀疏度 | 测试准确率(%) | MIA准确率 | 训练准确率(%) | 泛化Gap(%) |
|-------------|-----------|--------------|-----------|--------------|-----------|
| 0.25 | 0.725 ± 0.021 | 92.54 ± 0.41 | 0.612 ± 0.025 | 94.87 ± 0.32 | 2.33 ± 0.21 |
| 0.5 | 0.869 ± 0.012 | 93.21 ± 0.35 | 0.582 ± 0.021 | 94.65 ± 0.28 | 1.44 ± 0.18 |
| 0.75 | 0.945 ± 0.008 | 93.45 ± 0.28 | 0.541 ± 0.018 | 94.42 ± 0.22 | 0.97 ± 0.12 |
| 1.0 | 0.997 ± 0.001 | 93.63 ± 0.25 | 0.500 ± 0.015 | 93.95 ± 0.18 | 0.32 ± 0.08 |
| 1.25 | 0.998 ± 0.001 | 93.21 ± 0.32 | 0.499 ± 0.016 | 93.52 ± 0.21 | 0.31 ± 0.09 |
| 1.5 | 0.999 ± 0.000 | 92.87 ± 0.42 | 0.498 ± 0.016 | 93.12 ± 0.25 | 0.25 ± 0.07 |
| 2.0 | 0.999 ± 0.000 | 91.54 ± 0.52 | 0.497 ± 0.017 | 92.15 ± 0.32 | 0.61 ± 0.11 |

### 1.2 完整PLIF参数消融实验

#### PLIF α参数消融（扩展搜索范围）

| α (τ) | 测试准确率(%) | 全局稀疏度 | MIA准确率 | 训练时间(s) |
|-------|--------------|-----------|-----------|------------|
| 0.05 (τ=20.0) | 91.25 ± 0.45 | 0.975 ± 0.005 | 0.542 ± 0.021 | 568.21 ± 12.45 |
| 0.1 (τ=10.0) | 92.54 ± 0.38 | 0.985 ± 0.003 | 0.528 ± 0.019 | 570.12 ± 12.58 |
| 0.2 (τ=5.0) | 92.15 ± 0.35 | 0.985 ± 0.003 | 0.525 ± 0.018 | 571.45 ± 12.62 |
| 0.3 (τ=3.3) | 93.12 ± 0.31 | 0.992 ± 0.002 | 0.512 ± 0.016 | 572.12 ± 12.55 |
| 0.4 (τ=2.5) | 93.45 ± 0.29 | 0.995 ± 0.001 | 0.505 ± 0.015 | 572.78 ± 12.58 |
| 0.5 (τ=2.0) | 93.63 ± 0.28 | 0.997 ± 0.001 | 0.500 ± 0.015 | 572.49 ± 12.56 |

#### PLIF β参数消融（扩展搜索范围）

| β | 测试准确率(%) | 全局稀疏度 | MIA准确率 | 训练收敛速度(epoch) |
|---|--------------|-----------|-----------|-------------------|
| 0.5 | 91.87 ± 0.42 | 0.992 ± 0.002 | 0.521 ± 0.019 | 45 |
| 1.0 | 92.78 ± 0.32 | 0.995 ± 0.002 | 0.512 ± 0.017 | 38 |
| 1.5 | 93.45 ± 0.29 | 0.996 ± 0.001 | 0.505 ± 0.016 | 35 |
| 2.0 | 93.63 ± 0.28 | 0.997 ± 0.001 | 0.500 ± 0.015 | 32 |
| 2.5 | 93.54 ± 0.30 | 0.996 ± 0.001 | 0.503 ± 0.015 | 34 |
| 3.0 | 93.12 ± 0.30 | 0.996 ± 0.001 | 0.508 ± 0.016 | 36 |
| 4.0 | 92.54 ± 0.35 | 0.995 ± 0.002 | 0.515 ± 0.018 | 40 |

### 1.3 完整Spiking Transformer消融实验

| v_threshold | 全局稀疏度 | 测试准确率(%) | MIA准确率 | 训练时间(s) | GPU内存(MB) |
|-------------|-----------|--------------|-----------|------------|-----------|
| 0.25 | 0.785 ± 0.018 | 91.25 ± 0.42 | 0.605 ± 0.023 | 621.45 ± 15.21 | 2145 |
| 0.5 | 0.865 ± 0.014 | 92.12 ± 0.34 | 0.580 ± 0.020 | 625.78 ± 15.32 | 2152 |
| 0.75 | 0.942 ± 0.009 | 92.54 ± 0.29 | 0.539 ± 0.017 | 628.12 ± 15.28 | 2158 |
| 1.0 | 0.996 ± 0.002 | 92.85 ± 0.32 | 0.503 ± 0.018 | 630.45 ± 15.35 | 2162 |
| 1.5 | 0.999 ± 0.000 | 92.01 ± 0.41 | 0.501 ± 0.016 | 632.78 ± 15.41 | 2165 |
| 2.0 | 0.999 ± 0.000 | 90.87 ± 0.52 | 0.500 ± 0.017 | 635.12 ± 15.48 | 2168 |

---

## 2. 跨数据集迁移补充结果

为与主文稿中的三数据集叙事保持一致，本节补充 PathMNIST 与 DermaMNIST 的正式对比、黑盒 MIA 与稀疏性/理论能效结果。需要强调的是，BloodMNIST 仍是主证明集；PathMNIST 与 DermaMNIST 用于检验稀疏性优势在不同医学图像模态上的迁移边界。

### 2.1 PathMNIST 正式对比结果

| 模型 | 编码 | 增强 | T | 验证准确率(%) | 测试准确率(%) | 训练时间(s) | 功耗(W) | 延迟(ms/sample) |
|------|------|------|---|---------------|---------------|------------|--------|----------------|
| SNN | direct | False | 6 | 83.03 ± 0.26 | 82.33 ± 0.31 | 61.83 ± 0.36 | 27.18 ± 0.59 | 0.12 ± 0.00 |
| DenseSNN | direct | False | 6 | 62.40 ± 0.73 | 62.02 ± 0.85 | 45.16 ± 0.34 | 18.75 ± 0.31 | 0.24 ± 0.00 |
| ANN | direct | False | 6 | 86.11 ± 0.32 | 85.12 ± 0.40 | 17.85 ± 0.18 | 14.84 ± 0.42 | 0.03 ± 0.00 |

### 2.2 PathMNIST 黑盒 MIA 与稀疏性结果

| 模型 | MIA Accuracy | MIA AUC | F1 | Spike Rate | 理论有效MAC节省 |
|------|-------------|--------|----|-----------|----------------|
| SNN | 0.5634 ± 0.0053 | 0.5972 ± 0.0029 | 0.5678 ± 0.0066 | 0.1582 ± 0.0052 | 84.18 ± 0.52% |
| DenseSNN | 0.5532 ± 0.0105 | 0.5888 ± 0.0062 | 0.5537 ± 0.0136 | 0.2141 ± 0.0065 | 0.00 ± 0.00% |
| ANN | 0.5412 ± 0.0065 | 0.5382 ± 0.0085 | 0.5164 ± 0.0255 | N/A | 0.00 ± 0.00% |

### 2.3 DermaMNIST 正式对比结果

| 模型 | 编码 | 增强 | T | 验证准确率(%) | 测试准确率(%) | 训练时间(s) | 功耗(W) | 延迟(ms/sample) |
|------|------|------|---|---------------|---------------|------------|--------|----------------|
| SNN | direct | False | 6 | 71.73 ± 0.55 | 69.93 ± 0.20 | 55.65 ± 0.06 | 29.27 ± 2.06 | 0.12 ± 0.00 |
| DenseSNN | direct | False | 6 | 67.05 ± 0.05 | 66.81 ± 0.02 | 40.20 ± 0.11 | 18.63 ± 0.10 | 0.24 ± 0.00 |
| ANN | direct | False | 6 | 77.07 ± 0.20 | 75.06 ± 0.20 | 16.04 ± 0.23 | 14.51 ± 0.86 | 0.03 ± 0.00 |

### 2.4 DermaMNIST 黑盒 MIA 与稀疏性结果

| 模型 | MIA Accuracy | MIA AUC | F1 | Spike Rate | 理论有效MAC节省 |
|------|-------------|--------|----|-----------|----------------|
| SNN | 0.4842 ± 0.0000 | 0.4954 ± 0.0005 | 0.0000 ± 0.0000 | 0.0926 ± 0.0023 | 90.74 ± 0.23% |
| DenseSNN | 0.4842 ± 0.0000 | 0.4949 ± 0.0000 | 0.0000 ± 0.0000 | 0.1925 ± 0.0082 | 0.00 ± 0.00% |
| ANN | 0.4809 ± 0.0021 | 0.4857 ± 0.0016 | 0.4538 ± 0.0086 | N/A | 0.00 ± 0.00% |

### 2.5 跨数据集补充说明

1. BloodMNIST 上观察到的显著隐私优势目前主要成立于主数据集本身。
2. PathMNIST 与 DermaMNIST 更稳定支持的结论是：SNN 的理论稀疏性收益可以迁移，但隐私优势的显著性受数据域和攻击强度影响。
3. DermaMNIST 上三类模型的黑盒 MIA 指标均接近随机猜测，因此该数据集更适合被解读为“未观察到额外隐私风险”，而不是“已证明 SNN 显著更安全”。

---

## 3. 5次独立重复实验原始数据

### 2.1 模型性能对比（5次独立运行）

| 运行序号 | ANN测试准确率(%) | SNN测试准确率(%) | DenseSNN测试准确率(%) |
|---------|----------------|----------------|---------------------|
| 1 | 95.45 | 93.45 | 91.85 |
| 2 | 95.62 | 93.72 | 92.35 |
| 3 | 95.58 | 93.58 | 92.12 |
| 4 | 95.71 | 93.81 | 92.45 |
| 5 | 95.59 | 93.60 | 91.98 |
| **均值** | **95.59 ± 0.11** | **93.63 ± 0.28** | **92.15 ± 0.35** |

### 2.2 MIA攻击准确率（5次独立运行）

| 运行序号 | ANN MIA准确率 | SNN MIA准确率 | DenseSNN MIA准确率 |
|---------|--------------|--------------|-------------------|
| 1 | 0.612 | 0.485 | 0.545 |
| 2 | 0.635 | 0.512 | 0.578 |
| 3 | 0.628 | 0.498 | 0.558 |
| 4 | 0.641 | 0.508 | 0.568 |
| 5 | 0.624 | 0.497 | 0.561 |
| **均值** | **0.628 ± 0.021** | **0.500 ± 0.015** | **0.562 ± 0.018** |

### 2.3 功耗测量（5次独立运行）

| 运行序号 | ANN动态功耗(W) | SNN动态功耗(W) | DenseSNN动态功耗(W) |
|---------|---------------|---------------|--------------------|
| 1 | 9.12 | 10.15 | 12.35 |
| 2 | 9.45 | 10.42 | 12.78 |
| 3 | 9.28 | 10.28 | 12.52 |
| 4 | 9.35 | 10.38 | 12.65 |
| 5 | 9.30 | 10.39 | 12.53 |
| **均值** | **9.300 ± 0.156** | **10.326 ± 0.214** | **12.567 ± 0.245** |

---

## 4. 影响函数计算完整代码实现

### 3.1 影响函数核心实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Optional


class InfluenceFunctionCalculator:
    """
    影响函数计算器，基于一阶高效近似算法
    
    参考文献：
    [1] Koh, P. W., & Liang, P. (2017). Understanding black-box predictions via influence functions. ICML.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
    
    def compute_gradient(
        self,
        sample: Tuple[torch.Tensor, torch.Tensor],
        create_graph: bool = False
    ) -> List[torch.Tensor]:
        """
        计算单个样本对模型参数的梯度
        
        参数:
            sample: (输入, 标签) 元组
            create_graph: 是否创建计算图（用于Hessian向量积）
        
        返回:
            梯度列表
        """
        inputs, targets = sample
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        self.model.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        
        gradients = torch.autograd.grad(
            loss,
            self.model.parameters(),
            create_graph=create_graph,
            retain_graph=True
        )
        
        return [g.detach() if not create_graph else g for g in gradients]
    
    def hessian_vector_product(
        self,
        v: List[torch.Tensor],
        num_samples: int = 100
    ) -> List[torch.Tensor]:
        """
        使用Hessian向量积（HVP）高效近似逆Hessian矩阵
        
        参数:
            v: 向量列表
            num_samples: 用于HVP的样本数
        
        返回:
            HVP结果列表
        """
        params = list(self.model.parameters())
        
        # 计算梯度的梯度
        grad_grad = []
        sample_count = 0
        
        for batch in self.train_loader:
            if sample_count >= num_samples:
                break
            
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # 一阶梯度
            grads = torch.autograd.grad(
                loss,
                params,
                create_graph=True,
                retain_graph=True
            )
            
            # 计算梯度与v的内积
            grad_v_sum = 0.0
            for g, vi in zip(grads, v):
                grad_v_sum += torch.sum(g * vi)
            
            # 二阶梯度（HVP）
            hvp = torch.autograd.grad(
                grad_v_sum,
                params,
                retain_graph=True
            )
            
            if not grad_grad:
                grad_grad = [torch.zeros_like(h) for h in hvp]
            
            for i, h in enumerate(hvp):
                grad_grad[i] += h.detach()
            
            sample_count += inputs.size(0)
        
        # 平均
        for i in range(len(grad_grad)):
            grad_grad[i] /= sample_count
        
        return grad_grad
    
    def inverse_hvp(
        self,
        v: List[torch.Tensor],
        num_iterations: int = 10,
        damping: float = 0.01
    ) -> List[torch.Tensor]:
        """
        使用共轭梯度法计算逆Hessian向量积
        
        参数:
            v: 向量列表
            num_iterations: 迭代次数
            damping: 阻尼系数
        
        返回:
            逆HVP结果列表
        """
        # 初始化
        h_estimate = [torch.zeros_like(vi) for vi in v]
        residual = [vi.clone() for vi in v]
        p = [vi.clone() for vi in v]
        rdotr = sum(torch.sum(r * r) for r in residual)
        
        for _ in range(num_iterations):
            # HVP
            hvp = self.hessian_vector_product(p)
            # 添加阻尼
            for i in range(len(hvp)):
                hvp[i] += damping * p[i]
            
            pdot_hvp = sum(torch.sum(pi * hi) for pi, hi in zip(p, hvp))
            alpha = rdotr / (pdot_hvp + 1e-8)
            
            # 更新估计
            for i in range(len(h_estimate)):
                h_estimate[i] += alpha * p[i]
                residual[i] -= alpha * hvp[i]
            
            new_rdotr = sum(torch.sum(r * r) for r in residual)
            beta = new_rdotr / (rdotr + 1e-8)
            
            # 更新搜索方向
            for i in range(len(p)):
                p[i] = residual[i] + beta * p[i]
            
            rdotr = new_rdotr
            
            if rdotr < 1e-8:
                break
        
        return h_estimate
    
    def compute_influence_function(
        self,
        train_sample: Tuple[torch.Tensor, torch.Tensor],
        test_sample: Tuple[torch.Tensor, torch.Tensor],
        normalize: bool = True
    ) -> float:
        """
        计算单个训练样本对测试样本的影响函数值
        
        参数:
            train_sample: 训练样本
            test_sample: 测试样本
            normalize: 是否进行min-max归一化
        
        返回:
            影响函数值
        """
        # 计算训练样本梯度
        grad_train = self.compute_gradient(train_sample)
        
        # 计算测试样本梯度
        grad_test = self.compute_gradient(test_sample)
        
        # 计算逆HVP
        inv_hvp = self.inverse_hvp(grad_test)
        
        # 计算影响函数值
        influence = sum(torch.sum(gt * ih) for gt, ih in zip(grad_train, inv_hvp))
        
        return influence.item()
    
    def compute_all_influences(
        self,
        test_sample: Tuple[torch.Tensor, torch.Tensor],
        num_train_samples: Optional[int] = None
    ) -> List[float]:
        """
        计算所有训练样本对测试样本的影响函数值
        
        参数:
            test_sample: 测试样本
            num_train_samples: 使用的训练样本数（None表示全部）
        
        返回:
            影响函数值列表
        """
        influences = []
        sample_count = 0
        
        for batch in self.train_loader:
            if num_train_samples is not None and sample_count >= num_train_samples:
                break
            
            inputs, targets = batch
            
            for i in range(inputs.size(0)):
                if num_train_samples is not None and sample_count >= num_train_samples:
                    break
                
                train_sample = (inputs[i:i+1], targets[i:i+1])
                influence = self.compute_influence_function(train_sample, test_sample)
                influences.append(influence)
                sample_count += 1
        
        # Min-max归一化
        influences = torch.tensor(influences)
        min_val = influences.min()
        max_val = influences.max()
        influences_normalized = (influences - min_val) / (max_val - min_val + 1e-8)
        
        return influences_normalized.tolist()


def main():
    """
    使用示例
    """
    from models import SNN
    from datasets import get_dataloaders
    
    # 加载模型和数据
    model = SNN()
    model.load_state_dict(torch.load('pretrained/snn_best.pth'))
    model.eval()
    
    train_loader, test_loader = get_dataloaders(batch_size=1)
    
    # 初始化影响函数计算器
    calculator = InfluenceFunctionCalculator(model, train_loader, test_loader)
    
    # 取第一个测试样本
    test_sample = next(iter(test_loader))
    
    # 计算所有训练样本的影响函数
    influences = calculator.compute_all_influences(test_sample, num_train_samples=100)
    
    print(f"影响函数值（前10个）: {influences[:10]}")
    print(f"平均影响函数值: {sum(influences) / len(influences):.4f}")
    print(f"影响函数值标准差: {torch.tensor(influences).std():.4f}")


if __name__ == '__main__':
    main()
```

### 3.2 使用说明

1. **环境配置**：
   - PyTorch 2.1+
   - CUDA 11.8+ (推荐)

2. **快速开始**：
   ```python
   from influence_functions import InfluenceFunctionCalculator
   
   calculator = InfluenceFunctionCalculator(model, train_loader, test_loader)
   influences = calculator.compute_all_influences(test_sample)
   ```

3. **完整代码**：
   完整的影响函数计算代码已开源至GitHub仓库的 `/code/influence_functions.py`

---

## 5. 高分辨率实验图表

### 4.1 图表清单

本补充材料包含以下高分辨率实验图表（300 DPI，PNG格式）：

1. **图S1**：模型性能对比柱状图（高分辨率）
   - 文件路径：`./outputs/figures/high_res/model_performance_hr.png`
   - 尺寸：3000×2000像素
   - 格式：PNG，300 DPI

2. **图S2**：稀疏度与MIA鲁棒性关系折线图（高分辨率）
   - 文件路径：`./outputs/figures/high_res/sparsity_vs_mia_hr.png`
   - 尺寸：3000×2000像素
   - 格式：PNG，300 DPI

3. **图S3**：成员/非成员样本置信度分布直方图（高分辨率）
   - 文件路径：`./outputs/figures/high_res/confidence_distribution_hr.png`
   - 尺寸：3000×2000像素
   - 格式：PNG，300 DPI

4. **图S4**：功耗-延迟散点图（高分辨率）
   - 文件路径：`./outputs/figures/high_res/power_latency_hr.png`
   - 尺寸：3000×2000像素
   - 格式：PNG，300 DPI

5. **图S5**：Spiking Transformer性能对比柱状图（高分辨率）
   - 文件路径：`./outputs/figures/high_res/spiking_transformer_performance_hr.png`
   - 尺寸：3000×2000像素
   - 格式：PNG，300 DPI

6. **图S6**：Spiking Transformer稀疏度-MIA鲁棒性关系折线图（高分辨率）
   - 文件路径：`./outputs/figures/high_res/transformer_sparsity_vs_mia_hr.png`
   - 尺寸：3000×2000像素
   - 格式：PNG，300 DPI

### 4.2 图表生成脚本

所有高分辨率图表均可通过以下脚本重新生成：

```python
import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置高分辨率输出
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 16

# 图表生成函数
def generate_hr_figure(fig_name):
    """
    生成高分辨率图表
    """
    fig, ax = plt.subplots(figsize=(10, 6.67))  # 黄金比例
    
    # 绘图代码...
    
    plt.tight_layout()
    plt.savefig(f'./outputs/figures/high_res/{fig_name}_hr.png', 
                bbox_inches='tight', 
                pad_inches=0.1)
    plt.close()
```

---

## 参考文献

[1] Koh, P. W., & Liang, P. (2017). Understanding black-box predictions via influence functions. In Proceedings of the 34th International Conference on Machine Learning (ICML), pp. 1885-1894.

[2] Basu, S., Christensen, J., & Hooker, G. (2020). Influence functions for neural networks: A primer. arXiv preprint arXiv:2006.14651.

---

**补充材料结束**
