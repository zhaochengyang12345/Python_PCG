# 1_HeartSound_Segmentation_Springer

基于 Springer HSMM 模型的心音**交互式分割查看器**。支持对 WAV / CSV 格式的 PCG 信号进行预处理、自动分割（S1 / 收缩期 / S2 / 舒张期），并提供图形界面完成人工标注与批量导出。（主要是筛选采集的瓣膜数据是否可以入组！）

---

## 目录结构

```
1_HeartSound_Segmentation_Springer/
├── main.py                  # 主程序入口（含全部用户配置）
├── data/
│   └── example_data.mat     # Springer 模型训练数据示例
├── analytics/               # 包络提取算法
│   ├── hilbert_envelope.py
│   ├── homomorphic_envelope.py
│   └── wavelet_envelope.py
├── models/                  # Springer HSMM 核心模型
│   ├── band_pi_matrices.py
│   ├── segmentation_algorithm.py
│   └── viterbi_algorithm.py
└── utils/                   # 信号预处理工具
    ├── preprocessing.py
    └── tools.py
```

---

## 环境依赖

```
numpy
scipy
pandas
matplotlib
PyWavelets
tkinter  # Python 标准库，无需额外安装
```

---

## 快速开始

```bash
python main.py
```

程序启动后会弹出文件夹选择对话框，选择包含 WAV 或 CSV（需含 `pcg` 列）文件的目录即可。

---

## 主要配置项（`main.py` 顶部）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `TRAIN_DATA_PATH` | `data/example_data.mat` | Springer 训练数据路径 |
| `TRAIN_COUNT` | `5` | 参与训练的录音数 |
| `FALLBACK_FS` | `8000` | CSV 无时间列时的默认采样率（Hz）|
| `CSV_PCG_COLUMN` | `'pcg'` | CSV 中 PCG 信号的列名 |
| `ENABLE_SEGMENTATION` | `True` | 是否启用 Springer 分割 |
| `PROCESSING_FS` | `1000` | 内部处理采样率（与训练一致勿修改）|
| `LOWCUT / HIGHCUT` | `20 / 400` | 带通滤波截止频率（Hz）|
| `WAVELET_DENOISE` | `True` | 是否启用小波软阈值去噪 |
| `VIEW_WINDOW` | `8.0` | 界面初始可见时间窗（秒）|

---

## 界面操作

| 操作 | 说明 |
|---|---|
| `←` / `→` 方向键 | 切换上一个 / 下一个文件 |
| 键盘 `1` | 标注为 ✓ 入组 |
| 键盘 `2` | 标注为 ✗ 排除 |
| 键盘 `3` | 标注为 `?` 待定 |
| 界面按钮"批量导出" | 将指定类别的文件复制到目标文件夹 |

标注结果自动持久化保存为 `.label_annotations.json`，下次打开同一目录时自动恢复。

---

## 处理流程

```
原始 PCG 信号
    ↓ Hampel 滤波（去除脉冲伪影）
    ↓ 陷波滤波（50 Hz / 100 Hz 工频干扰）
    ↓ 带通滤波（20–400 Hz）
    ↓ 小波软阈值去噪（可选）
    ↓ 幅度归一化
    ↓ Springer HSMM 分割
    → 标注 S1 / 收缩期 / S2 / 舒张期
```

---

## 输出

- 界面实时展示分割结果波形图
- 若 `SAVE_PLOT = True`，分割结果图保存为 PNG 至 `output/` 子目录
- 标注文件：同输入目录下的 `.label_annotations.json`
