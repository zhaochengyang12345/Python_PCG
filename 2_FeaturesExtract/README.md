# 2_FeaturesExtract

基于 Springer HSMM 分割结果的 PCG **手工特征提取**工具。从 CSV 格式的心音信号中提取具有明确物理含义的时域、频域、能量、熵、峰度、循环平稳等多维特征，输出为 Excel 表格，可直接用于下游分类器训练。

> 移植自 PhysioNet/CinC 2016 Challenge MATLAB 官方代码。

---

## 目录结构

```
2_FeaturesExtract/
├── main_python.py              # 主程序入口（含配置）
├── challenge_python.py         # 心音分类/特征提取主函数
├── features.xlsx               # 输出：提取的特征表
├── answers.txt                 # 输出：每条记录分类结果
├── TEST/                       # 输入：CSV 文件 + RECORDS 列表
│   └── RECORDS                 # 记录名列表（每行一个文件名，无扩展名）
├── Springer_B_matrix.mat       # Springer HSMM 模型参数
├── Springer_pi_vector.mat
├── Springer_total_obs_distribution.mat
├── springer_lib/               # Springer 算法支撑库
└── requirements.txt
```

---

## 环境依赖

```bash
pip install numpy scipy pandas openpyxl soundfile PyWavelets
```

---

## 快速开始

### 1. 准备输入数据

将 CSV 文件放入 `TEST/` 目录。每个 CSV 文件需包含：
- `pcg` 列（或 `PCG`）：PCG 信号采样值
- 可选 `fs` / `Fs` / `sampling_rate` 列：采样率（不含时默认 1000 Hz）

在 `TEST/RECORDS` 文件中列出所有要处理的记录名（每行一个，不含 `.csv` 扩展名）：

```
patient_001
patient_002
patient_003
```

### 2. 配置特征模式

打开 `main_python.py` 修改顶部配置：

```python
FEATURE_MODE = 358          # 提取的标准特征数量
USE_WAVELET_FEATURE = True  # 是否追加 64 维小波能量特征
```

| `FEATURE_MODE` | 包含特征类型 | 维数 |
|---|---|---|
| `48` | 时间间隔特征 | 48 |
| `82` | 时间 + 频谱 | 82 |
| `131` | 时间 + 能量 + 频谱 + 峰度 + 循环 | 131 |
| `350` | 标准完整版（不含最后 8 项）| 350 |
| `358` | 全部标准特征 | 358 |

`USE_WAVELET_FEATURE = True` 时在上述维数基础上再追加 64 维小波能量特征。

### 3. 运行

```bash
python main_python.py
```

---

## 输出

| 文件 | 内容 |
|---|---|
| `features.xlsx` | 特征矩阵，每行对应一条记录，列名为特征名称（含序号）|
| `answers.txt` | 每条记录分类结果：`1`=异常 / `-1`=正常 / `0`=不确定 |

---

## 特征说明

所有特征均**有明确物理含义**，按类别分组如下：

### 时间间隔特征（16 维）
| 特征名 | 含义 |
|---|---|
| `m_RR` / `sd_RR` | RR 间期均值 / 标准差 |
| `m_IntS1` / `sd_IntS1` | S1 持续时间均值 / 标准差 |
| `m_IntS2` / `sd_IntS2` | S2 持续时间均值 / 标准差 |
| `m_IntSys` / `sd_IntSys` | 收缩期时长均值 / 标准差 |
| `m_IntDia` / `sd_IntDia` | 舒张期时长均值 / 标准差 |
| `m_Ratio_SysRR` 等 | 各阶段时长与 RR 间期的比值 |

### 能量特征（约 20 维）
各心音阶段（S1 / 收缩期 / S2 / 舒张期）的归一化能量及其比值。

### 频谱特征（约 60 维）
各阶段在不同频率子带上的功率谱密度均值（`mSpectrum_S1_*` 等）。

### 峰度特征（8 维）
各阶段包络峭度（`mean_s1_kurtosis` 等），反映瞬态冲击强度。

### 循环平稳特征（约 60 维）
心动周期与收缩/舒张期的循环功率谱（`spectrum_cyclePeriod_*` 等）。

### 熵特征（12 维）
收缩期与舒张期的样本熵 / 模糊熵 / 距离熵，反映信号复杂度。

### 小波能量特征（64 维，可选）
db4 小波 4 层分解各尺度的能量，捕捉瞬时频率成分。
