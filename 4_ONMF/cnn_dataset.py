"""
PCG ONMF CNN 数据集模块
=======================
Torre-Cruz et al. 2023 Section 4 & 5

支持两种数据来源：
  1. 直接从 .npy 文件夹加载（已由 ONMF 流水线生成的 *_H.npy + *_W.npy）
  2. 从 CSV/WAV 原始文件实时计算 ONMF 特征（批量模式）

支持的标签格式：
  - 文件名包含 'normal' / 'abnormal' / 'murmur' 关键字（大小写不敏感）
  - 与标签 CSV（两列: filename, label）配合使用
  - PhysioNet 2016 格式: REFERENCE.csv（第1列文件名, 第2列 -1/1）

H 矩阵输入调整：
  H 原始尺寸 (K, T) 随录音长度 T 变化。
  为满足 CNN 固定输入需求，统一调整为 (K, target_T)：
    - target_T 默认 256（约 2 秒 @ 4096Hz + hop=32）
    - 调整方式: bilinear 插值（保留时频结构）
"""

import os
import re
import warnings
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# ---------------------------------------------------------------------------
# 标签推断规则
# ---------------------------------------------------------------------------

# 前缀匹配优先（AS001 → AS → 异常），词边界匹配兜底
_LABEL_PATTERNS = [
    # 以 N 开头且后接数字 → 正常（如 N001，避免匹配 NOR/MVP 等）
    (re.compile(r'^N\d', re.I), 0),
    # 标准关键词
    (re.compile(r'\bnormal\b',   re.I), 0),
    (re.compile(r'\babnormal\b', re.I), 1),
    (re.compile(r'\bmurmur\b',   re.I), 1),
    # 瓣膜病前缀：AS/MR/MS/MVP/AR/VSD/HCM (开头+字母数字)
    (re.compile(r'^(AS|MR|MS|MVP|AR|VSD|HCM|ASD|PDA)\d', re.I), 1),
    # 兜底词边界
    (re.compile(r'\b(mr|ms|ar|as|mvp|vsd|hcm)\b', re.I), 1),
]


def _infer_label_from_name(filename: str) -> Optional[int]:
    """
    从文件名推断标签。
    返回 0（正常）、1（异常）或 None（无法判断）。
    """
    stem = Path(filename).stem
    for pattern, label in _LABEL_PATTERNS:
        if pattern.search(stem):
            return label
    return None


def load_label_csv(csv_path: str, filename_col: int = 0, label_col: int = 1,
                   normal_value: Union[str, int] = 1,
                   abnormal_value: Union[str, int] = -1) -> Dict[str, int]:
    """
    从 CSV 文件加载文件名→标签映射。

    支持格式：
      - PhysioNet 2016 REFERENCE.csv: 文件名列(0), 标签列(1),  1=正常, -1=异常
      - 自定义: 任意两列, normal_value / abnormal_value 指定正常标记符

    Returns
    -------
    dict: {stem -> label(0/1)}  label: 0=正常, 1=异常
    """
    import pandas as pd

    # 带 header 读取（label_generator 生成的 CSV 带 header）
    df = pd.read_csv(csv_path)
    # 兼容：若读出列名是数字（无 header），则重新无 header 读取
    try:
        int(df.columns[label_col])
        df = pd.read_csv(csv_path, header=None)
    except (ValueError, TypeError):
        pass  # 列名是字符串，带 header，正常

    # 收集全部数值标签，自动判断格式
    numeric_vals = set()
    for _, row in df.iterrows():
        try:
            numeric_vals.add(int(float(str(row.iloc[label_col]).strip())))
        except (ValueError, TypeError):
            pass

    # 若出现 -1 → PhysioNet 格式（1=正常, -1=异常）；否则 0/1 格式
    is_physionet = (-1 in numeric_vals)

    mapping = {}
    for _, row in df.iterrows():
        fname = str(row.iloc[filename_col]).strip()
        if fname.lower() in ('filename', 'file', 'name', 'stem'):
            continue
        stem = Path(fname).stem
        raw  = str(row.iloc[label_col]).strip().lower()

        try:
            v = int(float(raw))
        except (ValueError, TypeError):
            v = raw  # 字符串

        if v in ('normal', 'nor'):
            mapping[stem] = 0
        elif v in ('abnormal', 'murmur'):
            mapping[stem] = 1
        elif is_physionet:
            if v == 1:    mapping[stem] = 0  # PhysioNet: 1=正常
            elif v == -1: mapping[stem] = 1  # PhysioNet: -1=异常
        else:
            if v == 0:   mapping[stem] = 0   # 标准: 0=正常
            elif v == 1: mapping[stem] = 1   # 标准: 1=异常

    return mapping


# ---------------------------------------------------------------------------
# 核心 Dataset
# ---------------------------------------------------------------------------

class ONMFDataset(Dataset):
    """
    从已生成的 ONMF 特征文件（*_H.npy 或 *_W.npy）构建 PyTorch Dataset。

    Parameters
    ----------
    feature_dir   : 包含 *_H.npy 或 *_W.npy 文件的目录
    label_source  : 标签来源:
                    'filename' → 从文件名关键字推断
                    str(路径)  → CSV 标签文件路径
    feature_type  : 'H'（时域激活，CNN主要输入）或 'W'（频域基底）
    target_size   : (K, T) 固定输入尺寸，默认 (128, 256)
    transform     : 可选，数据增强 callable(tensor) -> tensor
    """

    def __init__(
        self,
        feature_dir: str,
        label_source: Union[str, Dict[str, int]] = 'filename',
        feature_type: str = 'H',
        target_size: Tuple[int, int] = (128, 256),
        transform=None,
    ):
        self.feature_dir = Path(feature_dir)
        self.feature_type = feature_type.upper()
        self.target_size = target_size
        self.transform = transform

        # 扫描特征文件
        pattern = f'*_{self.feature_type}.npy'
        files = sorted(self.feature_dir.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"在 {feature_dir} 中未找到 {pattern} 文件。\n"
                f"请先运行 ONMF 预处理并保存特征矩阵。"
            )

        # 加载标签映射
        if isinstance(label_source, dict):
            label_map = label_source
        elif label_source == 'filename':
            label_map = None   # 逐文件推断
        elif os.path.isfile(label_source):
            label_map = load_label_csv(label_source)
        else:
            raise ValueError(f"label_source 无效: {label_source}")

        # 构建样本列表
        self.samples: List[Tuple[Path, int]] = []
        skipped = 0
        for fp in files:
            stem = fp.stem.replace(f'_{self.feature_type}', '')
            if label_map is not None:
                label = label_map.get(stem)
                if label is None:
                    label = _infer_label_from_name(stem)
            else:
                label = _infer_label_from_name(stem)

            if label is None:
                warnings.warn(
                    f"无法确定标签，跳过: {fp.name}\n"
                    f"  提示: 文件名应包含 'normal'/'abnormal' 关键字，"
                    f"或提供标签CSV文件。"
                )
                skipped += 1
                continue
            self.samples.append((fp, int(label)))

        if not self.samples:
            raise ValueError(
                f"没有找到任何有效标签的样本。\n"
                f"请检查文件名或提供标签CSV。跳过了 {skipped} 个文件。"
            )

        labels = [s[1] for s in self.samples]
        n_normal   = labels.count(0)
        n_abnormal = labels.count(1)
        print(f"[ONMFDataset] {feature_type}矩阵, "
              f"共 {len(self.samples)} 个样本: "
              f"正常={n_normal}, 异常={n_abnormal}"
              + (f", 已跳过{skipped}个" if skipped else ""))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        fp, label = self.samples[idx]
        feat = np.load(fp).astype(np.float32)

        # 统一确保是 2D: (K, T) 或 (F, K)（W矩阵 shape 不同）
        if feat.ndim == 1:
            feat = feat.reshape(1, -1)

        # 调整为目标尺寸 (K, T) → (1, K, T) → interpolate → (1, K, T_target)
        x = torch.from_numpy(feat).unsqueeze(0).unsqueeze(0)  # (1, 1, K, T)
        if x.shape[2] != self.target_size[0] or x.shape[3] != self.target_size[1]:
            x = torch.nn.functional.interpolate(
                x, size=self.target_size,
                mode='bilinear', align_corners=False
            )
        x = x.squeeze(0)   # (1, K, T_target)

        # 可选归一化：将输入归一化到 [0, 1]
        x_min, x_max = x.min(), x.max()
        if x_max > x_min:
            x = (x - x_min) / (x_max - x_min)

        if self.transform is not None:
            x = self.transform(x)

        return x, label

    def get_labels(self) -> List[int]:
        return [s[1] for s in self.samples]

    def get_class_weights(self) -> torch.Tensor:
        """
        计算每个类别的权重，用于 WeightedRandomSampler。
        权重 = N_total / (N_classes * N_class_i)
        """
        labels = self.get_labels()
        n = len(labels)
        classes = sorted(set(labels))
        weights = torch.zeros(n)
        for c in classes:
            n_c = labels.count(c)
            for i, lbl in enumerate(labels):
                if lbl == c:
                    weights[i] = n / (len(classes) * n_c)
        return weights


# ---------------------------------------------------------------------------
# 实时计算数据集（从原始 CSV/WAV 文件）
# ---------------------------------------------------------------------------

class ONMFLiveDataset(Dataset):
    """
    从原始 PCG 文件（CSV/WAV）实时计算 ONMF 特征并返回张量。
    适用于推理阶段或小数据集直接训练。

    此 Dataset 每次 __getitem__ 都重新运行 ONMF，速度较慢但无需预生成。
    建议搭配 DataLoader(num_workers=0) 使用，避免多进程资源竞争问题。
    """

    def __init__(
        self,
        file_paths: List[str],
        labels: List[int],
        onmf_params: Optional[dict] = None,
        target_size: Tuple[int, int] = (128, 256),
    ):
        assert len(file_paths) == len(labels)
        self.file_paths = file_paths
        self.labels = labels
        self.target_size = target_size
        self.onmf_params = onmf_params or {}

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        from data_loader import load_pcg_file
        from onmf_preprocessing import preprocess_pcg_to_onmf

        fp = self.file_paths[idx]
        label = self.labels[idx]

        data = load_pcg_file(fp)
        signal = data['signal']
        fs = data['fs']

        result = preprocess_pcg_to_onmf(signal, fs=fs, **self.onmf_params)
        H = result['H'].astype(np.float32)

        x = torch.from_numpy(H).unsqueeze(0).unsqueeze(0)
        if x.shape[2] != self.target_size[0] or x.shape[3] != self.target_size[1]:
            x = torch.nn.functional.interpolate(
                x, size=self.target_size,
                mode='bilinear', align_corners=False
            )
        x = x.squeeze(0)
        x_min, x_max = x.min(), x.max()
        if x_max > x_min:
            x = (x - x_min) / (x_max - x_min)
        return x, label


# ---------------------------------------------------------------------------
# DataLoader 工厂函数
# ---------------------------------------------------------------------------

def make_dataloaders(
    feature_dir: str,
    label_source: Union[str, Dict[str, int]] = 'filename',
    feature_type: str = 'H',
    target_size: Tuple[int, int] = (128, 256),
    train_ratio: float = 0.7,
    val_ratio:   float = 0.15,
    batch_size:  int = 16,
    num_workers: int = 0,
    balance_train: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, ONMFDataset]:
    """
    创建训练/验证/测试 DataLoader。

    Parameters
    ----------
    feature_dir   : *_H.npy / *_W.npy 所在目录
    label_source  : 'filename', CSV路径, 或 {stem: label} 字典
    feature_type  : 'H' 或 'W'
    target_size   : CNN 输入尺寸 (K, T)
    train_ratio   : 训练集比例
    val_ratio     : 验证集比例（剩余为测试集）
    batch_size    : mini-batch 大小
    num_workers   : DataLoader 工作进程数
    balance_train : 是否用 WeightedRandomSampler 平衡训练集类别
    seed          : 随机种子

    Returns
    -------
    train_loader, val_loader, test_loader, full_dataset
    """
    dataset = ONMFDataset(feature_dir, label_source, feature_type, target_size)

    n = len(dataset)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n).tolist()

    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx   = indices[n_train: n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    from torch.utils.data import Subset
    train_ds = Subset(dataset, train_idx)
    val_ds   = Subset(dataset, val_idx)
    test_ds  = Subset(dataset, test_idx)

    # 类别平衡采样
    if balance_train:
        all_labels = dataset.get_labels()
        train_labels = [all_labels[i] for i in train_idx]
        n_train_total = len(train_labels)
        classes = sorted(set(train_labels))
        sample_weights = []
        for lbl in train_labels:
            n_c = train_labels.count(lbl)
            w = n_train_total / (len(classes) * n_c)
            sample_weights.append(w)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=n_train_total,
            replacement=True
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  sampler=sampler, num_workers=num_workers,
                                  pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers,
                                  pin_memory=True)

    val_loader  = DataLoader(val_ds,  batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    print(f"[DataLoader] 训练={len(train_ds)}, 验证={len(val_ds)}, 测试={len(test_ds)}")
    return train_loader, val_loader, test_loader, dataset
