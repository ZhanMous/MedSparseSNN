# -*- coding: utf-8 -*-
"""
MedSparseSNN 模型定义
包含三个对等结构的模型：SNN、DenseSNN、ANN
"""

import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, surrogate, functional

class PLIFNode(neuron.LIFNode):
    """PLIF 神经元，α 为可学习参数"""
    def __init__(self, tau=2.0, v_threshold=1.0, surrogate_function=None, step_mode='m'):
        super().__init__(tau=tau, v_threshold=v_threshold, surrogate_function=surrogate_function, step_mode=step_mode)
        self.alpha = nn.Parameter(torch.tensor(1.0 / tau))
    
    def forward(self, x):
        self.tau = 1.0 / self.alpha
        return super().forward(x)

class FixedPLIFNode(neuron.LIFNode):
    """固定PLIF神经元，α固定为0.2（tau=5.0），不可学习"""
    def __init__(self, tau=5.0, v_threshold=1.0, surrogate_function=None, step_mode='m'):
        super().__init__(tau=tau, v_threshold=v_threshold, surrogate_function=surrogate_function, step_mode=step_mode)
        self.alpha = torch.tensor(0.2)
    
    def forward(self, x):
        self.tau = 5.0
        return super().forward(x)

class SpikingResBlock(nn.Module):
    """脉冲残差块"""
    def __init__(self, in_channels, out_channels, stride=1, v_threshold=1.0, step_mode='m'):
        super().__init__()
        self.step_mode = step_mode
        surrogate_function = surrogate.ATan()
        
        self.conv_bn_lif = nn.Sequential(
            layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, step_mode=step_mode),
            layer.BatchNorm2d(out_channels, step_mode=step_mode),
            PLIFNode(v_threshold=v_threshold, surrogate_function=surrogate_function, step_mode=step_mode),
            layer.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, step_mode=step_mode),
            layer.BatchNorm2d(out_channels, step_mode=step_mode)
        )
        self.lif = PLIFNode(v_threshold=v_threshold, surrogate_function=surrogate_function, step_mode=step_mode)
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                layer.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False, step_mode=step_mode),
                layer.BatchNorm2d(out_channels, step_mode=step_mode)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv_bn_lif(x)
        out += self.shortcut(x)
        out = self.lif(out)
        return out

class FixedSpikingResBlock(nn.Module):
    """固定PLIF参数的脉冲残差块"""
    def __init__(self, in_channels, out_channels, stride=1, v_threshold=1.0, step_mode='m'):
        super().__init__()
        self.step_mode = step_mode
        surrogate_function = surrogate.ATan()
        
        self.conv_bn_lif = nn.Sequential(
            layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, step_mode=step_mode),
            layer.BatchNorm2d(out_channels, step_mode=step_mode),
            FixedPLIFNode(v_threshold=v_threshold, surrogate_function=surrogate_function, step_mode=step_mode),
            layer.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, step_mode=step_mode),
            layer.BatchNorm2d(out_channels, step_mode=step_mode)
        )
        self.lif = FixedPLIFNode(v_threshold=v_threshold, surrogate_function=surrogate_function, step_mode=step_mode)
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                layer.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False, step_mode=step_mode),
                layer.BatchNorm2d(out_channels, step_mode=step_mode)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv_bn_lif(x)
        out += self.shortcut(x)
        out = self.lif(out)
        return out

class SNN(nn.Module):
    """SNN 模型
    - MS-ResNet 结构
    - PLIF 神经元
    - 时间步 T=6
    - 保持稀疏计算优化
    """
    def __init__(self, in_channels=3, num_classes=8, T=6, v_threshold=1.0):
        super().__init__()
        self.T = T
        
        surrogate_function = surrogate.ATan()
        
        self.stem = nn.Sequential(
            layer.Conv2d(in_channels, 20, kernel_size=3, padding=1, bias=False, step_mode='m'),
            layer.BatchNorm2d(20, step_mode='m'),
            PLIFNode(v_threshold=v_threshold, surrogate_function=surrogate_function, step_mode='m'),
            layer.MaxPool2d(2, 2, step_mode='m')
        )
        
        self.layer1 = SpikingResBlock(20, 41, stride=2, v_threshold=v_threshold, step_mode='m')
        self.layer2 = SpikingResBlock(41, 82, stride=2, v_threshold=v_threshold, step_mode='m')
        
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1), step_mode='m')
        self.flatten = layer.Flatten(step_mode='m')
        
        self.classifier = nn.Sequential(
            layer.Linear(82, num_classes, bias=False, step_mode='m'),
            PLIFNode(v_threshold=v_threshold, surrogate_function=surrogate_function, step_mode='m')
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        if x.dim() == 4:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        elif x.dim() == 5:
            x = x.transpose(0, 1)
            
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        
        return x.mean(dim=0)

class SNN_FixedAlpha(nn.Module):
    """SNN模型，PLIF参数固定为α=0.2（tau=5.0）
    - MS-ResNet结构
    - 固定PLIF神经元
    - 时间步T=6
    - 保持稀疏计算优化
    """
    def __init__(self, in_channels=3, num_classes=8, T=6, v_threshold=1.0):
        super().__init__()
        self.T = T
        
        surrogate_function = surrogate.ATan()
        
        self.stem = nn.Sequential(
            layer.Conv2d(in_channels, 20, kernel_size=3, padding=1, bias=False, step_mode='m'),
            layer.BatchNorm2d(20, step_mode='m'),
            FixedPLIFNode(v_threshold=v_threshold, surrogate_function=surrogate_function, step_mode='m'),
            layer.MaxPool2d(2, 2, step_mode='m')
        )
        
        self.layer1 = FixedSpikingResBlock(20, 41, stride=2, v_threshold=v_threshold, step_mode='m')
        self.layer2 = FixedSpikingResBlock(41, 82, stride=2, v_threshold=v_threshold, step_mode='m')
        
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1), step_mode='m')
        self.flatten = layer.Flatten(step_mode='m')
        
        self.classifier = nn.Sequential(
            layer.Linear(82, num_classes, bias=False, step_mode='m'),
            FixedPLIFNode(v_threshold=v_threshold, surrogate_function=surrogate_function, step_mode='m')
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        if x.dim() == 4:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        elif x.dim() == 5:
            x = x.transpose(0, 1)
            
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        
        return x.mean(dim=0)

class NonSparsePLIF(nn.Module):
    """无稀疏计算的PLIF：仅关闭稀疏优化，但保持相同的脉冲发放动力学
    与标准PLIF完全相同的发放行为，但强制使用稠密计算"""
    def __init__(self, tau=2.0, v_threshold=1.0, reset_mode='zero'):
        super().__init__()
        self.decay = nn.Parameter(torch.tensor(1.0 - 1.0 / tau))
        self.v_threshold = v_threshold
        self.reset_mode = reset_mode
        self.alpha = nn.Parameter(torch.tensor(1.0 / tau))
        self.v = None
    
    def forward(self, x_seq: torch.Tensor):
        T, N, C, H, W = x_seq.shape
        
        self.decay.data = 1.0 - self.alpha.data
        
        if self.v is None or self.v.shape != (N, C, H, W):
            self.v = torch.zeros((N, C, H, W), device=x_seq.device)
        
        spike_seq = []
        for t in range(T):
            x = x_seq[t]
            
            self.v = self.v * self.decay + x
            
            spike = (self.v >= self.v_threshold).float()
            
            if self.reset_mode == 'zero':
                self.v = self.v * (1 - spike)
            
            spike_seq.append(spike.clone())
        
        return torch.stack(spike_seq, dim=0)
    
    def reset(self):
        self.v = None

class NonSparseSpikingResBlock(nn.Module):
    """无稀疏脉冲残差块"""
    def __init__(self, in_channels, out_channels, stride=1, v_threshold=1.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.lif1 = NonSparsePLIF(v_threshold=v_threshold)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.lif2 = NonSparsePLIF(v_threshold=v_threshold)
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x_seq: torch.Tensor):
        T, N, C, H, W = x_seq.shape
        
        out_seq = []
        for t in range(T):
            x = x_seq[t]
            
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.lif1(out.unsqueeze(0))[0]
            
            out = self.conv2(out)
            out = self.bn2(out)
            
            shortcut = self.shortcut(x)
            out += shortcut
            
            out = self.lif2(out.unsqueeze(0))[0]
            out_seq.append(out)
        
        return torch.stack(out_seq, dim=0)

class DenseSNN(nn.Module):
    """DenseSNN 模型
    - 与 SNN 完全相同的拓扑结构和参数量
    - 唯一区别：完全关闭稀疏计算，强制所有神经元参与计算
    - 使用普通 PyTorch 层（无 SpikingJelly 稀疏优化）+ NonSparsePLIF
    - 用于与 SNN 对照，评估稀疏计算的效果
    """
    def __init__(self, in_channels=3, num_classes=8, T=6, v_threshold=1.0):
        super().__init__()
        self.T = T
        self.v_threshold = v_threshold
        
        self.conv1 = nn.Conv2d(in_channels, 20, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(20)
        self.lif1 = NonSparsePLIF(v_threshold=v_threshold)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.layer1 = NonSparseSpikingResBlock(20, 41, stride=2, v_threshold=v_threshold)
        self.layer2 = NonSparseSpikingResBlock(41, 82, stride=2, v_threshold=v_threshold)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(82, num_classes, bias=False)
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor):
        if x.dim() == 4:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        elif x.dim() == 5:
            x = x.transpose(0, 1)
        
        T, N, C, H, W = x.shape
        
        out_seq = []
        for t in range(T):
            xt = x[t]
            
            xt = self.conv1(xt)
            xt = self.bn1(xt)
            xt = self.lif1(xt.unsqueeze(0))[0]
            xt = self.pool1(xt)
            
            xt = xt.unsqueeze(0)
            xt = self.layer1(xt)[0]
            
            xt = xt.unsqueeze(0)
            xt = self.layer2(xt)[0]
            
            xt = self.avgpool(xt)
            xt = self.flatten(xt)
            out_seq.append(xt)
        
        out = torch.stack(out_seq, dim=0).mean(dim=0)
        out = self.classifier(out)
        return out
    
    def reset(self):
        for m in self.modules():
            if isinstance(m, NonSparsePLIF):
                m.reset()

class ANN(nn.Module):
    """ANN 模型
    - 同拓扑结构
    - ReLU 激活，无时间维度
    """
    def __init__(self, in_channels=3, num_classes=8):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 20, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        class ResBlock(nn.Module):
            def __init__(self, in_channels, out_channels, stride=1):
                super().__init__()
                self.conv_bn_relu = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
                if stride != 1 or in_channels != out_channels:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(out_channels)
                    )
                else:
                    self.shortcut = nn.Identity()
            def forward(self, x):
                out = self.conv_bn_relu(x)
                out += self.shortcut(x)
                out = nn.ReLU(inplace=True)(out)
                return out
        
        self.layer1 = ResBlock(20, 41, stride=2)
        self.layer2 = ResBlock(41, 82, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        self.classifier = nn.Sequential(
            nn.Linear(82, num_classes, bias=False)
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        if x.dim() == 5:
            x = x.mean(dim=0)
            
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


class SpikingMultiHeadAttention(nn.Module):
    """脉冲多头注意力模块"""
    def __init__(self, dim, num_heads, v_threshold=1.0, step_mode='m'):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim
        
        surrogate_function = surrogate.ATan()
        
        self.qkv = layer.Linear(dim, dim * 3, bias=False, step_mode=step_mode)
        self.proj = layer.Linear(dim, dim, bias=False, step_mode=step_mode)
        self.lif = PLIFNode(v_threshold=v_threshold, surrogate_function=surrogate_function, step_mode=step_mode)
    
    def forward(self, x):
        T, N, L, D = x.shape
        
        qkv = self.qkv(x)
        qkv = qkv.reshape(T, N, L, 3, self.num_heads, self.head_dim).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).permute(0, 1, 3, 2, 4).reshape(T, N, L, D)
        
        x = self.proj(x)
        x = self.lif(x)
        return x


class SpikingFeedForward(nn.Module):
    """脉冲前馈网络模块"""
    def __init__(self, dim, hidden_dim, v_threshold=1.0, step_mode='m'):
        super().__init__()
        surrogate_function = surrogate.ATan()
        
        self.fc1 = layer.Linear(dim, hidden_dim, bias=False, step_mode=step_mode)
        self.lif1 = PLIFNode(v_threshold=v_threshold, surrogate_function=surrogate_function, step_mode=step_mode)
        self.fc2 = layer.Linear(hidden_dim, dim, bias=False, step_mode=step_mode)
        self.lif2 = PLIFNode(v_threshold=v_threshold, surrogate_function=surrogate_function, step_mode=step_mode)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.lif1(x)
        x = self.fc2(x)
        x = self.lif2(x)
        return x


class SpikingTransformerBlock(nn.Module):
    """脉冲Transformer块"""
    def __init__(self, dim, num_heads, hidden_dim, v_threshold=1.0, step_mode='m'):
        super().__init__()
        self.dim = dim
        surrogate_function = surrogate.ATan()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpikingMultiHeadAttention(dim, num_heads, v_threshold, step_mode)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = SpikingFeedForward(dim, hidden_dim, v_threshold, step_mode)
    
    def forward(self, x):
        T, N, L, D = x.shape
        x_norm = self.norm1(x.reshape(T*N, L, D)).reshape(T, N, L, D)
        x = x + self.attn(x_norm)
        x_norm = self.norm2(x.reshape(T*N, L, D)).reshape(T, N, L, D)
        x = x + self.ff(x_norm)
        return x


class LightSpikingTransformer(nn.Module):
    """轻量化Spiking Transformer模型
    - Patch=4×4
    - Transformer Block=2层
    - 隐藏维度=68
    - 注意力头数=2
    - FFN中间维度=136
    - 总参数量≈0.117M（与现有SNN-CNN完全一致）
    - PLIF神经元，时间步T=6
    - 保持稀疏计算优化
    """
    def __init__(self, in_channels=3, num_classes=8, T=6, v_threshold=1.0, img_size=28, patch_size=4):
        super().__init__()
        self.T = T
        self.patch_size = patch_size
        
        surrogate_function = surrogate.ATan()
        
        num_patches = (img_size // patch_size) ** 2
        embed_dim = 68
        
        self.stem_conv1 = nn.Sequential(
            layer.Conv2d(in_channels, 36, kernel_size=3, padding=1, bias=False, step_mode='m'),
            layer.BatchNorm2d(36, step_mode='m'),
            PLIFNode(v_threshold=v_threshold, surrogate_function=surrogate_function, step_mode='m')
        )
        
        self.patch_embed = layer.Conv2d(36, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False, step_mode='m')
        self.patch_norm = layer.BatchNorm2d(embed_dim, step_mode='m')
        self.patch_lif = PLIFNode(v_threshold=v_threshold, surrogate_function=surrogate_function, step_mode='m')
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.blocks = nn.ModuleList([
            SpikingTransformerBlock(embed_dim, 2, 136, v_threshold, step_mode='m')
            for _ in range(2)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = layer.Linear(embed_dim, num_classes, bias=False, step_mode='m')
        self.head_lif = PLIFNode(v_threshold=v_threshold, surrogate_function=surrogate_function, step_mode='m')
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, layer.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor):
        if x.dim() == 4:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        elif x.dim() == 5:
            x = x.transpose(0, 1)
        
        x = self.stem_conv1(x)
        
        x = self.patch_embed(x)
        x = self.patch_norm(x)
        x = self.patch_lif(x)
        
        T, N, C, H, W = x.shape
        x = x.flatten(3).permute(0, 1, 3, 2)
        
        pos_embed = self.pos_embed.unsqueeze(0).repeat(T, N, 1, 1)
        x = x + pos_embed
        
        for block in self.blocks:
            x = block(x)
        
        T, N, L, D = x.shape
        x = self.norm(x.reshape(T*N, L, D)).reshape(T, N, L, D)
        x = x.mean(dim=2)
        x = self.head(x)
        x = self.head_lif(x)
        
        return x.mean(dim=0)