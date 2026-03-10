# -*- coding: utf-8 -*-
"""
MedSparseSNN 全局配置文件
针对现代GPU架构 (8GB GDDR6) 专属优化
包含混合精度(AMP)、自适应Batch Size调整与GPU功耗实测逻辑
"""

import os
import torch
import matplotlib
matplotlib.use('Agg')  # 非GUI后端，适合服务器环境

# ===========================================
# 🧪 实验配置
# ===========================================
DATA_FLAG = 'bloodmnist'  # 数据集标识
DATA_ROOT = './data'  # 数据根目录
PROJECT_NAME = 'MedSparseSNN'  # 项目名称
SEED = 42  # 随机种子
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 主设备
T = 6  # 时间步长
EPOCHS = 50  # 训练轮数
INIT_LR = 1e-3  # 初始学习率
BATCH_SIZE = 64  # 默认批次大小
WEIGHT_DECAY = 1e-4  # 权重衰减
STEP_SIZE = 20  # 学习率衰减步长
GAMMA = 0.1  # 学习率衰减系数
DEFAULT_T = T  # 保持向后兼容
DEFAULT_BATCH_SIZE = BATCH_SIZE  # 保持向后兼容
NUM_EPOCHS = EPOCHS  # 保持向后兼容
IN_CHANNELS = 3  # 输入通道数

# SNN特定的学习率
SNN_LR = 1e-3
SNN_ADAMW_LR = 1e-3
ANN_LR = 1e-3
DENSESNN_LR = 1e-3
MOMENTUM = 0.9  # 保持向后兼容

# LIF神经元参数
LIF_TAU = 2.0
LIF_V_THRESHOLD = 1.0

# 时间步相关参数
MAX_T = 20  # 最大时间步
MIN_BATCH_SIZE = 8
MAX_BATCH_SIZE = 32

# 分类任务参数
NUM_CLASSES = 8
IMG_SIZE = 28

# 设置GPU信息提示
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[{PROJECT_NAME}] 使用GPU: {gpu_name} ({gpu_mem:.1f} GB)")
else:
    print(f"[{PROJECT_NAME}] 警告: CUDA不可用，使用CPU（训练将极慢）")

# ============================================================
# 3. 数据集配置
# ============================================================
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'raw')

# ===========================================
# 📊 输出路径配置
# ===========================================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(ROOT_DIR, 'outputs')
MODELS_DIR = os.path.join(OUTPUTS_DIR, 'models')
RESULTS_DIR = os.path.join(OUTPUTS_DIR, 'results')
FIGURES_DIR = os.path.join(OUTPUTS_DIR, 'figures')
CHECKPOINT_DIR = os.path.join(OUTPUTS_DIR, 'checkpoints')  # 检查点目录

# 创建输出目录
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(os.path.join(FIGURES_DIR, 'academic'), exist_ok=True)
os.makedirs(os.path.join(FIGURES_DIR, 'data'), exist_ok=True)
os.makedirs(os.path.join(OUTPUTS_DIR, 'checkpoints'), exist_ok=True)

# ===========================================
# 📈 可视化配置
# ===========================================
FIG_SIZE = (10, 6)  # 图表默认尺寸
FIG_DPI = 300  # 图表分辨率
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']  # 颜色主题
MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']  # 标记样式
LABELS = ['SNN', 'DenseSNN', 'ANN']  # 模型标签

# ===========================================
# ⚡ 性能优化配置
# ===========================================
USE_AMP = True  # 是否启用混合精度训练
PIN_MEMORY = True  # DataLoader是否启用pin_memory
NUM_WORKERS = 4  # DataLoader进程数
GRAD_CLIP_NORM = 1.0  # 梯度裁剪阈值

# ===========================================
# 🔋 功耗测量配置
# ===========================================
MEASURE_POWER = True  # 是否测量功耗 (需要pynvml支持)
POWER_DEVICE_ID = 0  # GPU设备ID
POWER_WARMUP_ITERS = 10  # 功耗测量预热迭代数
POWER_TEST_ITERS = 100  # 功耗测量测试迭代数

def set_seed(seed=SEED):
    """设置全局随机种子"""
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_adaptive_batch_size():
    """获取自适应批次大小"""
    return BATCH_SIZE