"""
PCG CNN 分类模型模块
====================
Torre-Cruz et al. 2023 Section 5

实现论文对比的 6 种 CNN 架构：
  0. UjaNet         - 论文自研架构（Table 3），推荐首选
  1. LeNet5         - 低复杂度基准（论文发现性能接近复杂架构）
  2. AlexNet        - 经典 5 层 CNN
  3. VGG16          - 深度 CNN，迁移学习
  4. ResNet50       - 残差网络，迁移学习
  5. GoogLeNet      - Inception 结构，迁移学习

全部适配单通道灰度输入（H矩阵作为图像），输出二分类（正常/异常）。

输入格式: (batch, 1, K, T) = (batch, 1, 128, 256)  [默认]
输出格式: (batch, 2)  logits（未经 softmax）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# 0. UjaNet（论文自研架构，Table 3）
# ---------------------------------------------------------------------------

class UjaNet(nn.Module):
    """
    Torre-Cruz et al. 2023 论文提出的 UjaNet 架构（Table 3）。
    专为单通道 PCG H 矩阵特征设计的轻量二分类 CNN。

    结构（完整复现 Table 3）：
      Layer 0: Conv2D(5×5, 16 filters, LeakyReLU) → MaxPool2D(2×2)
      Layer 2: Conv2D(5×5, 32 filters, LeakyReLU) → MaxPool2D(2×2)
      Layer 4: Flatten
      Layer 5: Dense(100, LeakyReLU) + Dropout(0.5)
      Layer 6: Dense( 50, LeakyReLU) + Dropout(0.5)
      Layer 7: Dense(num_classes)
    论文原版最后一层用 Dense(1, Sigmoid)，此处统一为 Dense(2) 以兼容
    CrossEntropyLoss，分类结果等价。
    """

    def __init__(self, input_size: tuple = (128, 256), num_classes: int = 2):
        super().__init__()
        K, T = input_size
        self.lrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.conv1 = nn.Conv2d(1,  16, kernel_size=5)   # no padding — faithful to paper
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Dynamic flattened size computation
        h1 = (K - 4) // 2       # after conv1(no pad) + pool1
        w1 = (T - 4) // 2
        h2 = max((h1 - 4) // 2, 1)  # after conv2(no pad) + pool2
        w2 = max((w1 - 4) // 2, 1)
        self._flat_size = 32 * h2 * w2

        self.fc1  = nn.Linear(self._flat_size, 100)
        self.drop1 = nn.Dropout(0.5)
        self.fc2  = nn.Linear(100, 50)
        self.drop2 = nn.Dropout(0.5)
        self.fc3  = nn.Linear(50, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.lrelu(self.conv1(x)))
        x = self.pool2(self.lrelu(self.conv2(x)))
        x = x.flatten(1)
        x = self.drop1(self.lrelu(self.fc1(x)))
        x = self.drop2(self.lrelu(self.fc2(x)))
        return self.fc3(x)


# ---------------------------------------------------------------------------
# 1. LeNet-5（适配 ONMF H 矩阵尺寸）
# ---------------------------------------------------------------------------

class LeNet5(nn.Module):
    """
    改编自 LeCun 1998 LeNet-5，适配单通道 (1, K, T) 输入。
    论文发现：此低复杂度模型 + ONMF H 特征 ≈ 复杂架构（避免过拟合小数据集）。

    网络结构：
      Conv1(1→6, 5×5, pad=2) → Tanh → AvgPool(2×2)
      Conv2(6→16, 5×5)       → Tanh → AvgPool(2×2)
      Flatten → FC(→120) → Tanh → FC(→84) → Tanh → FC(→num_classes)
    """

    def __init__(self, input_size=(128, 256), num_classes=2):
        super().__init__()
        K, T = input_size

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 动态计算全连接层输入尺寸
        # After conv1+pool1: floor(K/2), floor(T/2)
        # After conv2+pool2: floor((floor(K/2)-4)/2), floor((floor(T/2)-4)/2)
        h1 = K // 2
        w1 = T // 2
        h2 = (h1 - 4) // 2
        w2 = (w1 - 4) // 2
        self._flat_size = 16 * max(h2, 1) * max(w2, 1)

        self.fc1 = nn.Linear(self._flat_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool1(torch.tanh(self.conv1(x)))
        x = self.pool2(torch.tanh(self.conv2(x)))
        x = x.flatten(1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)


# ---------------------------------------------------------------------------
# 2-5. 基于 torchvision 的迁移学习架构
# ---------------------------------------------------------------------------

def _build_pretrained(arch: str, pretrained: bool, num_classes: int) -> nn.Module:
    """
    内部工厂：将 torchvision 标准架构改为单通道输入 + 二分类输出。
    使用 weights= API（torchvision >= 0.13），向下兼容旧版 pretrained= 参数。
    """
    try:
        import torchvision.models as M

        # 统一使用 weights= API 避免 FutureWarning
        def _load(model_fn, weights_cls):
            if pretrained:
                try:
                    w = weights_cls.DEFAULT
                    return model_fn(weights=w)
                except Exception:
                    return model_fn(pretrained=True)
            else:
                try:
                    return model_fn(weights=None)
                except Exception:
                    return model_fn(pretrained=False)

        if arch == 'alexnet':
            model = _load(M.alexnet, M.AlexNet_Weights)
            # 替换第一卷积层支持单通道输入
            model.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
            # 替换分类头
            model.classifier[-1] = nn.Linear(4096, num_classes)

        elif arch == 'vgg16':
            model = _load(M.vgg16, M.VGG16_Weights)
            model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
            model.classifier[-1] = nn.Linear(4096, num_classes)

        elif arch == 'resnet50':
            model = _load(M.resnet50, M.ResNet50_Weights)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            model.fc = nn.Linear(2048, num_classes)

        elif arch == 'googlenet':
            # GoogLeNet 关闭辅助分类器
            try:
                model = M.googlenet(weights=M.GoogLeNet_Weights.DEFAULT if pretrained else None,
                                    aux_logits=False)
            except Exception:
                model = M.googlenet(pretrained=pretrained, aux_logits=False)
            model.conv1.conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            model.fc = nn.Linear(1024, num_classes)

        else:
            raise ValueError(f"不支持的架构: {arch}。可选: alexnet, vgg16, resnet50, googlenet")

    except ImportError:
        raise ImportError(
            "需要 torchvision 来使用预训练网络。\n"
            "安装: pip install torchvision  或  conda install torchvision"
        )

    return model


# ---------------------------------------------------------------------------
# 统一模型工厂
# ---------------------------------------------------------------------------

SUPPORTED_ARCHS = ['ujanet', 'lenet5', 'alexnet', 'vgg16', 'resnet50', 'googlenet']


def build_model(
    arch: str = 'lenet5',
    num_classes: int = 2,
    input_size: tuple = (128, 256),
    pretrained: bool = True,
    dropout: float = 0.0,
) -> nn.Module:
    """
    构建 CNN 分类模型。

    Parameters
    ----------
    arch        : 'lenet5' | 'alexnet' | 'vgg16' | 'resnet50' | 'googlenet'
    num_classes : 输出类别数（论文为 2：正常/异常）
    input_size  : (K, T) = (128, 256) 默认
    pretrained  : 是否加载 ImageNet 预训练权重（仅 alexnet/vgg16/resnet50/googlenet）
    dropout     : 在最终分类层前插入 Dropout（0.0=不使用）

    Returns
    -------
    nn.Module
    """
    arch = arch.lower()
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(f"arch 必须是 {SUPPORTED_ARCHS} 之一，收到: '{arch}'")

    if arch == 'ujanet':
        model = UjaNet(input_size=input_size, num_classes=num_classes)
    elif arch == 'lenet5':
        model = LeNet5(input_size=input_size, num_classes=num_classes)
    else:
        model = _build_pretrained(arch, pretrained=pretrained, num_classes=num_classes)

    # 可选：在分类头前插入 Dropout
    if dropout > 0.0 and arch != 'lenet5':
        model = _wrap_with_dropout(model, arch, dropout, num_classes)

    return model


def _wrap_with_dropout(model: nn.Module, arch: str,
                       dropout: float, num_classes: int) -> nn.Module:
    """在模型分类头前插入 Dropout。"""
    if arch == 'alexnet':
        in_feats = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            *list(model.classifier.children())[:-1],
            nn.Dropout(dropout),
            nn.Linear(in_feats, num_classes)
        )
    elif arch == 'vgg16':
        in_feats = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            *list(model.classifier.children())[:-1],
            nn.Dropout(dropout),
            nn.Linear(in_feats, num_classes)
        )
    elif arch == 'resnet50':
        in_feats = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, num_classes)
        )
    elif arch == 'googlenet':
        in_feats = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, num_classes)
        )
    return model


# ---------------------------------------------------------------------------
# 模型信息工具
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """统计可训练参数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module, input_size=(1, 1, 128, 256), device='cpu') -> None:
    """打印模型参数量及一次前向传播的输出形状。"""
    model = model.to(device)
    x = torch.zeros(input_size).to(device)
    with torch.no_grad():
        out = model(x)
    n_params = count_parameters(model)
    print(f"模型: {model.__class__.__name__}")
    print(f"输入: {tuple(x.shape)}  →  输出: {tuple(out.shape)}")
    print(f"可训练参数: {n_params:,}  ({n_params/1e6:.2f} M)")
