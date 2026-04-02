# Springer 算法改进与 S3/S4 心音检测相关文献整理

---

## 一、Springer 原始算法论文

### 1. Logistic Regression-HSMM-Based Heart Sound Segmentation

- **作者**：David B. Springer, Lionel Tarassenko, Gari D. Clifford
- **期刊**：IEEE Transactions on Biomedical Engineering
- **年份**：2016，Vol. 63, No. 4, pp. 822–832
- **DOI / 链接**：[https://doi.org/10.1109/TBME.2015.2475278](https://doi.org/10.1109/TBME.2015.2475278)
- **主要工作**：
  - 提出了基于逻辑回归（Logistic Regression）与隐半马尔可夫模型（HSMM）相结合的心音分割算法，即"Springer 算法"。
  - 采用 4 状态 HSMM（S1 → 收缩期 → S2 → 舒张期），利用同态包络、小波包络和 Hilbert 包络三种特征，通过逻辑回归训练观测概率模型。
  - 在 PhysioNet/CinC 2016 Challenge 数据集上取得了当时最优的心音分割性能。
  - 提供了开源 MATLAB 实现，是后续大量研究的基准算法。

---

### 2. An Open Access Database for the Evaluation of Heart Sound Algorithms

- **作者**：Chengyu Liu, David Springer, Qiao Li, et al.
- **期刊**：Physiological Measurement
- **年份**：2016，Vol. 37, No. 12, p. 2181
- **DOI / 链接**：[https://doi.org/10.1088/0967-3334/37/12/2181](https://doi.org/10.1088/0967-3334/37/12/2181)
- **主要工作**：
  - 发布了 PhysioNet/CinC 2016 Challenge 公开心音数据库，包含来自正常人和心脏病患者的大量 PCG 录音。
  - 数据覆盖多种采集设备、多个采集部位和噪声环境，具有代表性。
  - 该数据集已成为心音分割、分类算法的基准评测平台，后续包括 Springer 算法在内的大量工作均在此数据集上验证。

---

## 二、Springer 算法改进方向

### 3. Adaptive Sojourn Time HSMM for Heart Sound Segmentation

- **作者**：Jorge Oliveira, Francesco Renna, Tiago Mantadelis, Miguel T. Coimbra
- **期刊**：IEEE Journal of Biomedical and Health Informatics
- **年份**：2019，Vol. 23, No. 2, pp. 642–649
- **DOI / 链接**：[https://doi.org/10.1109/JBHI.2018.2841197](https://doi.org/10.1109/JBHI.2018.2841197)
- **主要工作**：
  - 针对 Springer 算法中固定停留时间（Sojourn Time）分布导致的分割误差问题，提出了自适应停留时间 HSMM。
  - 根据输入信号的心率动态调整各状态的持续时间分布，提升了对心率变异性较大信号的分割鲁棒性。
  - 在 CirCor DigiScope 数据集的半监督标注流程中被用作三种参考算法之一。

---

### 4. Deep Convolutional Neural Networks for Heart Sound Segmentation

- **作者**：Francesco Renna, Jorge H. Oliveira, Miguel T. Coimbra
- **期刊**：IEEE Journal of Biomedical and Health Informatics
- **年份**：2019，Vol. 23, No. 6, pp. 2435–2445
- **DOI / 链接**：[https://doi.org/10.1109/JBHI.2019.2894222](https://doi.org/10.1109/JBHI.2019.2894222)
- **主要工作**：
  - 提出用深度卷积神经网络（DCNN）直接对 PCG 信号进行端到端的心音分割，替代传统 HSMM 框架。
  - 网络无需手工设计特征，直接从原始信号或时频图中学习 S1/S2 检测的判别特征。
  - 同样被用于 CirCor DigiScope 数据集标注的参考算法之一，与 Springer 算法和自适应 HSMM 构成三方对比。

---

## 三、S3/S4 心音独立检测研究

### 5. Detection and Boundary Identification of Phonocardiogram Sounds Using an Expert Frequency-Energy Based Metric

- **作者**：H. Naseri, M. R. Homaeinezhad
- **期刊**：Annals of Biomedical Engineering
- **年份**：2013，Vol. 41, No. 2, pp. 279–292
- **DOI / 链接**：[https://doi.org/10.1007/s10439-012-0645-x](https://doi.org/10.1007/s10439-012-0645-x)
- **PubMed**：[https://pubmed.ncbi.nlm.nih.gov/22956159/](https://pubmed.ncbi.nlm.nih.gov/22956159/)
- **主要工作**：
  - 提出基于频率-能量联合判决统计量（Decision Statistic, DS）的 PCG 心音检测与边界识别方法。
  - 采用两阶段策略：第一阶段检测 S1 和 S2；第二阶段从剔除 S1/S2 后的残差信号中检测偶发性 **S3 和 S4**。
  - 对来自多种瓣膜疾病患者的 52 分钟 PCG 信号进行了测试，S1/S2/S3/S4 四类心音的平均灵敏度 Se = 99.00%，阳性预测值 PPV = 98.60%。
  - 为 S3/S4 心音的自动检测提供了重要的技术参考。

---

### 6. An Automated Tool for Localization of Heart Sound Components S1, S2, S3 and S4 in Pulmonary Sounds Using Hilbert Transform and Heron's Formula

- **作者**：Ashok Mondal, Parthasarathi Bhattacharya, Goutam Saha
- **期刊**：SpringerPlus
- **年份**：2013，Vol. 2, p. 512
- **DOI / 链接**：[https://doi.org/10.1186/2193-1801-2-512](https://doi.org/10.1186/2193-1801-2-512)
- **PubMed**：[https://pubmed.ncbi.nlm.nih.gov/24255827/](https://pubmed.ncbi.nlm.nih.gov/24255827/)
- **主要工作**：
  - 针对肺音（Lung Sound）中心音干扰的消除需求，提出了一种同时定位 S1、S2、**S3、S4** 四种心音成分的自动化算法。
  - 方法基于 Hilbert 变换提取包络，并结合 Heron 三角面积公式设定自适应阈值来区分心音段与非心音段。
  - 相比基于奇异谱分析（SSA）的对比方法，在准确率（ACC）、检测错误率（DER）、假阴性率（FNR）和执行时间（ET）方面均有改善。

---

## 四、大规模数据集与综合评测

### 7. The CirCor DigiScope Dataset: From Murmur Detection to Murmur Classification

- **作者**：Jorge H. Oliveira, Francesco Renna, Paulo Costa, et al.（包括 Gari D. Clifford）
- **期刊**：IEEE Journal of Biomedical and Health Informatics
- **年份**：2021
- **DOI / 链接**：[https://doi.org/10.1109/JBHI.2021.3137048](https://doi.org/10.1109/JBHI.2021.3137048)
- **数据集**：[https://physionet.org/content/circor-heart-sound/1.0.3/](https://physionet.org/content/circor-heart-sound/1.0.3/)
- **主要工作**：
  - 发布迄今**最大儿科 PCG 公开数据集**（CirCor DigiScope），来自巴西 1568 名 0–21 岁受试者，共 5272 条录音，总时长超 33.5 小时。
  - 采用 Springer 算法、自适应 HSMM 算法（文献 3）和 DCNN 算法（文献 4）三方投票的半监督方案完成 S1/S2 分割标注，专家对分歧处进行人工校正。
  - 提供了杂音的精细标注（时序、形态、音调、强度、音质、听诊位置等），支持杂音检测和分类两类任务。
  - 被用于 **George B. Moody PhysioNet Challenge 2022**，推动了新型分割与诊断算法的研究。

---

## 五、综述与背景参考

### 8. Heart Sound Segmentation Using Bidirectional LSTMs with Attention（参考）

- **作者**：Titus de Silva Fernando, et al.
- **期刊**：IEEE Journal of Biomedical and Health Informatics
- **年份**：2020
- **链接**：[https://doi.org/10.1109/JBHI.2019.2949516](https://doi.org/10.1109/JBHI.2019.2949516)
- **主要工作**：
  - 提出基于双向 LSTM + 注意力机制的 PCG 心音分割框架，以序列标注方式直接输出 S1/收缩期/S2/舒张期的逐帧标签。
  - 由于 LSTM 天然适合时序依赖建模，在含噪声和病理信号上的鲁棒性优于传统 HSMM 方法。
  - 代表了 Springer 算法从统计模型向深度学习演进的重要一步。

---

## 总结

| 编号 | 论文简称 | 方法类型 | 是否涉及 S3/S4 | 核心贡献 |
|------|---------|---------|--------------|---------|
| 1 | Springer 2016 | LR-HSMM | 否（4状态） | 原始 Springer 算法 |
| 2 | Liu 2016 | 数据集 | 否 | PhysioNet 2016 评测库 |
| 3 | Oliveira 2019a | 自适应 HSMM | 否 | 停留时间自适应改进 |
| 4 | Renna 2019 | 深度 CNN | 否 | 端到端深度学习分割 |
| 5 | Naseri 2013 | 频率-能量判决 | **是** | 两阶段检测 S1~S4 |
| 6 | Mondal 2013 | Hilbert + Heron | **是** | 四成分同时定位 |
| 7 | Oliveira 2021 | 数据集 | 否（S1/S2） | 最大儿科数据集 + 2022 Challenge |
| 8 | Fernando 2020 | BiLSTM + 注意力 | 否 | 深度序列模型分割 |
