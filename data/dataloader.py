# -*- coding: utf-8 -*-
"""
MedMNIST 数据加载与脉冲编码工具。
- 适配 SpikingJelly 多步模式 [T, N, C, H, W]
- 支持 BloodMNIST / PathMNIST 等 MedMNIST 分类任务
- 为病理图像提供更适合纹理分类的增强策略
"""

import os
import sys
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import medmnist
from medmnist import INFO

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_ROOT, DATA_FLAG, DEFAULT_T, IN_CHANNELS,
    DEFAULT_BATCH_SIZE, NUM_WORKERS, PIN_MEMORY,
    get_adaptive_batch_size
)


PATHOLOGY_DATASETS = {'pathmnist'}


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


# ============================================================
# 泊松脉冲编码器（SpikingJelly 风格）
# ============================================================
class PoissonEncoder:
    """
    泊松脉冲编码：将像素值（归一化到[0,1]）作为发放概率，
    生成 T 个时间步的脉冲序列。
    输入: [N, C, H, W] (值域 [0,1])
    输出: [T, N, C, H, W] (二值脉冲 0/1)
    """
    def __init__(self, T: int):
        self.T = T

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, C, H, W] 或 [C, H, W]，值域 [0, 1]
        返回: [T, ...] 的二值脉冲张量
        """
        # 扩展时间维度
        if x.dim() == 3:  # 单样本 [C, H, W]
            shape = (self.T,) + x.shape
        else:  # batch [N, C, H, W]
            shape = (self.T,) + x.shape

        # 泊松采样：以像素值为概率生成脉冲
        rand = torch.rand(shape, device=x.device)
        spikes = (rand < x.unsqueeze(0)).float()
        return spikes


# ============================================================
# 脉冲编码数据集包装器
# ============================================================
class SpikeEncodedDataset(Dataset):
    """
    将 MedMNIST 数据集包装为泊松脉冲编码输出。
    SNN模式：返回 [T, C, H, W] 脉冲序列 或 图像序列
    ANN模式：返回 [C, H, W] 原始图像
    """
    def __init__(self, base_dataset, T: int, mode='snn', encoding='poisson'):
        """
        Args:
            base_dataset: MedMNIST 数据集实例
            T: 时间步数
            mode: 'snn' 返回序列, 'ann' 返回原始图像
            encoding: 'poisson' 泊松编码, 'direct' 直接编码 (复制图像)
        """
        self.base_dataset = base_dataset
        self.T = T
        self.mode = mode
        self.encoding = encoding
        self.encoder = PoissonEncoder(T)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        label = label.squeeze().long() if isinstance(label, torch.Tensor) else int(label.squeeze())

        if self.mode == 'snn':
            if self.encoding == 'poisson':
                # 泊松编码: [C, H, W] → [T, C, H, W]
                spikes = self.encoder(img)
                return spikes, label
            else:
                # 直接编码 (Direct Coding): 直接复制 T 份图像
                # [C, H, W] → [T, C, H, W]
                direct_input = img.unsqueeze(0).repeat(self.T, 1, 1, 1)
                return direct_input, label
        else:
            # ANN模式：直接返回原始图像
            return img, label


# ============================================================
# 数据集元信息
# ============================================================
def resolve_dataset_info(dataset_flag=None):
    dataset_flag = (dataset_flag or DATA_FLAG).lower()
    if dataset_flag not in INFO:
        raise ValueError(f"Unsupported MedMNIST dataset: {dataset_flag}")

    info = INFO[dataset_flag]
    data_class = getattr(medmnist, info['python_class'])
    num_classes = len(info['label'])
    in_channels = int(info.get('n_channels', IN_CHANNELS))
    return dataset_flag, info, data_class, num_classes, in_channels


def build_transforms(dataset_flag, augment=True):
    if not augment:
        transform = transforms.Compose([transforms.ToTensor()])
        return transform, transform

    if dataset_flag in PATHOLOGY_DATASETS:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return train_transform, test_transform


# ============================================================
# 数据加载器（SNN / ANN 双模式）
# ============================================================
def get_medmnist_loaders(
    batch_size=None,
    T=DEFAULT_T,
    mode='snn',
    encoding='direct',
    augment=True,
    seed=None,
    num_workers=None,
    pin_memory=None,
    dataset_flag=None,
    img_size=28,
):
    """
    加载 MedMNIST 分类数据集，返回 DataLoader。

    Args:
        batch_size: 批次大小，None 则自适应检测
        T: SNN 时间步数
        mode: 'snn' 脉冲突发 / 'ann' 原始图像
        encoding: 'direct' (推荐) / 'poisson'
        augment: 是否启用数据增强（训练集）

    Returns:
        train_loader, val_loader, test_loader, info
    """
    if batch_size is None:
        batch_size = get_adaptive_batch_size()

    if num_workers is None:
        num_workers = NUM_WORKERS

    if pin_memory is None:
        pin_memory = PIN_MEMORY

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    dataset_flag, info, DataClass, _, _ = resolve_dataset_info(dataset_flag)

    # ---- 数据变换 ----
    train_transform, test_transform = build_transforms(dataset_flag, augment=augment)

    # ---- 加载数据集 ----
    os.makedirs(DATA_ROOT, exist_ok=True)

    train_dataset = DataClass(split='train', transform=train_transform,
                              download=True, root=DATA_ROOT, size=img_size)
    val_dataset = DataClass(split='val', transform=test_transform,
                            download=True, root=DATA_ROOT, size=img_size)
    test_dataset = DataClass(split='test', transform=test_transform,
                             download=True, root=DATA_ROOT, size=img_size)

    # ---- 脉冲编码包装 ----
    train_dataset = SpikeEncodedDataset(train_dataset, T=T, mode=mode, encoding=encoding)
    val_dataset = SpikeEncodedDataset(val_dataset, T=T, mode=mode, encoding=encoding)
    test_dataset = SpikeEncodedDataset(test_dataset, T=T, mode=mode, encoding=encoding)

    # ---- DataLoader（4070优化）----
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        worker_init_fn=_seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=generator,
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=generator,
    )

    n_train = len(train_dataset)
    n_val = len(val_dataset)
    n_test = len(test_dataset)
    print(f"[MedSparseSNN] {dataset_flag} 加载完成 | 模式={mode} | T={T} | encoding={encoding}")
    print(f"  训练集: {n_train} | 验证集: {n_val} | 测试集: {n_test}")
    print(f"  batch_size={batch_size} | num_workers={num_workers} | pin_memory={pin_memory}")
    print(f"  类别: {list(info['label'].values())}")

    return train_loader, val_loader, test_loader, info


def get_blood_mnist_loaders(**kwargs):
    kwargs.setdefault('dataset_flag', 'bloodmnist')
    return get_medmnist_loaders(**kwargs)


# ============================================================
# 独立测试
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("测试 SNN 模式（泊松编码）")
    print("=" * 60)
    train_loader, val_loader, test_loader, info = get_blood_mnist_loaders(
        batch_size=4, T=10, mode='snn'
    )
    spikes, labels = next(iter(train_loader))
    print(f"  SNN 输入 shape: {spikes.shape}")  # [T, N, C, H, W]
    print(f"  标签 shape: {labels.shape}")
    print(f"  脉冲发放率: {spikes.mean().item():.4f}")

    print("\n" + "=" * 60)
    print("测试 ANN 模式（原始图像）")
    print("=" * 60)
    train_loader_ann, _, _, _ = get_blood_mnist_loaders(
        batch_size=4, T=10, mode='ann'
    )
    imgs, labels = next(iter(train_loader_ann))
    print(f"  ANN 输入 shape: {imgs.shape}")  # [N, C, H, W]
    print(f"  标签 shape: {labels.shape}")
    print(f"  像素值范围: [{imgs.min():.2f}, {imgs.max():.2f}]")
