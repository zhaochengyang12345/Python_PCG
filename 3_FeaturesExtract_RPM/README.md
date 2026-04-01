# 3_FeaturesExtract_RPM

基于**相对位置矩阵（RPM）+ 深度卷积网络迁移学习**的 PCG 特征提取流水线。将 CSV 心音信号转化为 RPM 图像后，使用预训练 CNN 提取高维抽象特征，适合作为深度学习分类器的输入。

---

## 处理流程

```
CSV（心音间期序列）
    ↓ Step 1: 1_generate_pcgintervals.py
Series/*.xlsx（RR / S1 / S2 / 收缩期 / 舒张期 间期时序）
    ↓ Step 2: 2_RPM.py
RPM/*.png（相对位置矩阵图像）
    ↓ Step 3: 3_inceptionv3.py / 3_mobilenet.py / 3_resnet50.py / 3_vgg16.py
Result/<模型名>/*_features.xlsx（CNN 深度特征）
```

---

## 目录结构

```
3_FeaturesExtract_RPM/
├── 1_generate_pcgintervals.py  # Step 1：提取心音间期
├── 2_RPM.py                    # Step 2：生成 RPM 图像
├── 3_inceptionv3.py            # Step 3：InceptionV3 特征提取（2048 维）
├── 3_mobilenet.py              # Step 3：MobileNet 特征提取（1024 维）
├── 3_resnet50.py               # Step 3：ResNet50 特征提取（2048 维）
├── 3_vgg16.py                  # Step 3：VGG16 特征提取（512 维）
├── CSV/                        # 输入：原始心音 CSV 文件
├── Series/                     # 中间产物：间期时序 xlsx
├── RPM/                        # 中间产物：RPM 图像 png
├── Result/                     # 输出：深度特征 xlsx
│   ├── InceptionV3/
│   ├── MobileNet/
│   ├── ResNet50/
│   └── VGG16/
└── springer_algo/              # Springer 分割算法（Step 1 依赖）
```

---

## 环境依赖

```bash
pip install numpy scipy pandas openpyxl matplotlib keras tensorflow
```

> 要求 TensorFlow ≥ 2.16（Keras 3.x）

---

## 使用步骤

### Step 1 — 提取心音间期

将原始 CSV 文件放入 `CSV/` 目录，每个文件需含 `pcg`（或 `PCG`）列。

```bash
python 1_generate_pcgintervals.py
```

- 读取 `CSV/` 下所有 `.csv` 文件
- 使用 Springer HSMM 算法分割每个心动周期
- 输出到 `Series/`，文件名格式：`pcgintervals_<原文件名>.xlsx`

每个输出文件包含以下列：

| 列名 | 含义 |
|---|---|
| `RR` | RR 间期（秒）|
| `IntS1` | S1 持续时长（秒）|
| `IntS2` | S2 持续时长（秒）|
| `IntSys` | 收缩期时长（秒）|
| `IntDia` | 舒张期时长（秒）|

**关键配置**（脚本顶部）：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `AUDIO_FS` | `1000` | Springer 算法目标采样率（Hz），勿修改 |
| `DEFAULT_FS` | `8000` | CSV 无采样率列时的默认原始采样率 |

---

### Step 2 — 生成 RPM 图像

```bash
python 2_RPM.py
```

- 读取 `Series/` 下所有 `.xlsx` 文件（取第二列作为时序信号）
- 对每条时序进行 z-score 归一化 → PAA 压缩 → 构造相对位置矩阵
- 以 `jet` 伪彩色保存为 PNG 图像至 `RPM/`
- 命名格式：`<原文件名>_rpm_<列名>.png`

**RPM 原理**：

$$M_{ij} = x_i - x_j$$

矩阵元素为时序中任意两时刻的幅值差，经 min-max 归一化到 $[0, 255]$。RPM 将时序的节律特性编码为二维纹理图案，使 CNN 能捕捉心动周期的动态规律。

---

### Step 3 — 提取深度特征

以 InceptionV3 为例（其他三个网络用法完全相同）：

```bash
python 3_inceptionv3.py
```

- 读取 `RPM/` 下所有 `.png` 图像
- 加载 ImageNet 预训练权重（首次运行会自动下载）
- 去掉分类头，接 GlobalAveragePooling 输出特征向量
- 结果保存至 `Result/InceptionV3/inceptionv3_features.xlsx`

| 脚本 | 输入尺寸 | 输出维数 | 输出文件 |
|---|---|---|---|
| `3_inceptionv3.py` | 139×139 | 2048 | `inceptionv3_features.xlsx` |
| `3_mobilenet.py` | 128×128 | 1024 | `mobilenet_features.xlsx` |
| `3_resnet50.py` | 256×256 | 2048 | `resnet50_features.xlsx` |
| `3_vgg16.py` | 128×128 | 512 | `vgg16_features.xlsx` |

输出 Excel 第一列为 `filename`（图像文件名），其余各列为特征值。

---

## CSV 文件格式要求

```
pcg,（可选）fs
0.012,-
-0.034,-
0.078,-
...
```

- `pcg` 列必须存在（大小写均可）
- 若含 `fs` / `Fs` / `sampling_rate` 列，自动读取采样率；否则使用 `DEFAULT_FS`（8000 Hz）
- NaN 值会自动进行线性插值填充

---

## 注意事项

- Step 1 在短录音（< 2 个心动周期）上可能无法分割，脚本会自动跳过并打印提示
- Step 3 首次运行需要联网下载 ImageNet 预训练权重（~20–100 MB）
- 多个 Step 3 脚本可以并行运行，互不干扰
