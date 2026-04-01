# PCG 心音杂音孤立数据集处理工具

将自建心音图（PCG）长录音处理成与公开数据集风格一致的标注短片段，支持三步式交互处理流程与可视化 GUI。

---

## 主要功能

### 三步处理流程

| 步骤 | 功能 | 适用场景 |
|------|------|---------|
| **Step 1** | 将长录音按心动周期切段（每段 2–6 个完整周期） | 统一片段时长，对齐 PhysioNet / CirCor 数据集格式 |
| **Step 2** | 按疾病类型保留杂音区段（含 S1/S2 边界） | 屏蔽无关心动时段，突显杂音期相 |
| **Step 3** | 剔除 S1/S2，提取纯杂音 | 生成可直接用于机器学习的孤立杂音片段 |

### 支持疾病

| 代码 | 疾病 | 杂音期相 | 典型频率范围 |
|------|------|---------|------------|
| AS | 主动脉瓣狭窄 | 收缩期（喷射型） | 100–400 Hz |
| AR | 主动脉瓣反流 | 舒张期（递减型） | 200–500 Hz |
| MR | 二尖瓣反流 | 收缩期（全收缩期） | 150–400 Hz |
| MS | 二尖瓣狭窄 | 舒张期（隆隆样） | 30–120 Hz |
| MVP | 二尖瓣脱垂 | 收缩期（喀喇音+渐增型） | 100–400 Hz |
| N | 正常 | — | — |

疾病代码从**文件名或父目录名**中自动识别（大小写不限），例如 `D:\data\AS\patient001.wav` 或 `patient_AR_001.wav` 均可正确识别。

### 其他功能

- **实时波形预览**：三个子图分别展示原始滤波波形、Step 2 掩膜结果、Step 3 纯杂音，四种状态背景色直观标注 S1/收缩期/S2/舒张期
- **疾病参考侧栏**：选中片段后自动显示该疾病的杂音期相、性质、频率范围、额外心音及最佳听诊位置
- **手动区段覆盖**：通过勾选 S1/收缩期/S2/舒张期复选框自定义保留区段，覆盖自动映射
- **手动分段编辑器**：针对质量较差的数据，在独立弹窗中通过拖拽选区手动修正 S1/S2 边界，支持 50 步撤销
- **批量保存**：将全部片段以 WAV + JSON 元数据形式一键保存

---

## 输出结果

### 音频文件

每个处理片段保存为 16-bit WAV，命名规则：

```
{源文件名}_{疾病代码}_{后缀}_{序号}_{周期数}cyc.wav
```

| 步骤 | 后缀 | 示例 |
|------|------|------|
| Step 1 | `seg` | `patient001_AS_seg_001_3cyc.wav` |
| Step 2 | `masked` | `patient001_AS_masked_001_3cyc.wav` |
| Step 3 | `murmur` | `patient001_AS_murmur_001_3cyc.wav` |

### 元数据文件

同目录下生成 `segments_metadata.json`：

```json
[
  {
    "index": 1,
    "source_file": "D:/data/AS/patient001.wav",
    "disease": "AS",
    "cycle_count": 3,
    "fs": 4000,
    "duration_s": 2.415,
    "step": 3
  }
]
```

---

## 环境依赖

```
numpy
scipy
pandas
matplotlib
PyWavelets
tkinter      # Python 标准库，无需额外安装
```

安装依赖：

```bash
pip install numpy scipy pandas matplotlib PyWavelets
```

---

## 快速开始

### 1. 启动程序

```bash
python pcg_processor.py
```

### 2. 准备数据

将 WAV 或 CSV 文件按疾病代码组织到子目录中：

```
data/
├── AS/
│   ├── patient001.wav
│   └── patient002.wav
├── AR/
│   └── patient003.wav
└── MR/
    └── patient004.csv   # 含 pcg 列，支持 8000 Hz CSV
```

也可以将疾病代码直接写在文件名中（如 `record_AS_20240101.wav`）。

### 3. 操作流程

**Step 1 — 分割心动周期**

1. 点击"浏览…"选择数据文件夹，点击"扫描文件"
2. 选择 **Step1 分割心动周期**，点击"▶ 运行处理"
3. 处理完成后在左侧列表选择片段，右侧显示分割波形预览
4. Step 1 输出可直接批量保存（点击"批量保存"）

**Step 2 — 疾病掩膜**

1. 选择 **Step2 疾病掩膜**，点击"▶ 运行处理"
2. 中间栏自动显示当前疾病的临床参考信息
3. 如需自定义保留区段，勾选"启用手动模式"后选择状态，点击"重新应用到当前片段"

**Step 3 — 纯杂音提取**

1. 选择 **Step3 剔除S1/S2**，点击"▶ 运行处理"
2. 右侧三个子图对比展示原始信号、掩膜信号、纯杂音
3. 批量保存即可得到孤立杂音数据集

### 4. 手动校正分段（可选）

当 Springer 自动分割效果不理想时：

1. 在左侧列表选中需要校正的片段
2. 点击右侧"✏ 手动调整分段"
3. 在弹窗中点击顶部状态按钮选色（如 S1），在波形上**拖拽**选区，松手后自动填充
4. 支持撤销（↩）、重置、缩放/平移导航
5. 点击"✓ 确认应用"，主窗口 Step 2/3 结果自动更新

---

## 项目结构

```
5_Isolated_Murmur_PCG_Dataset/
├── pcg_processor.py         # 主程序（含全部处理逻辑与 GUI）
├── report.md                # 技术报告
├── README.md                # 本文件
└── springer_hsmm/           # Springer HSMM 算法子包
    ├── main.py
    ├── data/
    │   └── example_data.mat # 792 条训练记录（自动加载前 5 条）
    ├── models/
    │   ├── segmentation_algorithm.py
    │   ├── viterbi_algorithm.py
    │   └── band_pi_matrices.py
    ├── analytics/
    │   ├── hilbert_envelope.py
    │   ├── homomorphic_envelope.py
    │   └── wavelet_envelope.py
    └── utils/
        ├── preprocessing.py
        └── tools.py
```

---

## 技术说明

- **分割算法**：Springer 隐半马尔可夫模型（HSMM），状态 1=S1、2=收缩期、3=S2、4=舒张期
- **预处理链**：Hampel 滤波 → 50/100 Hz 陷波 → 20–400 Hz 带通 → 幅度归一化
- **WAV 采样率**：4000 Hz（不符时自动重采样）
- **CSV 采样率**：自动推断，默认 8000 Hz
- **训练记录数**：最少 5 条（`band_pi_matrices` 硬性要求）
- **后台处理**：子线程 + `queue.Queue` 轮询，GUI 全程响应
