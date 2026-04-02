# 多瓣膜性心脏病 PCG 研究文献整理

> **核心问题**：现有公开数据集和研究是否主要集中在单瓣膜疾病？是否存在提升空间？
>
> **结论**：是的，绝大多数现有研究和数据集仅支持**单瓣膜、单一疾病类别**的识别，将多瓣膜共病（Multi-valvular Disease, MVD）纳入系统性研究的工作极少，是明确的研究空白。

---

## 一、现有公开数据集对比（单瓣膜 vs 多瓣膜）

| 数据集 | 年份 | 标注粒度 | 是否支持多标签/多瓣膜 | 链接 |
|--------|------|---------|---------------------|------|
| PhysioNet/CinC 2016 | 2016 | 正常 / 异常（二分类） | **否** | https://physionet.org/content/hss/1.0/ |
| PASCAL Challenge 2011 | 2011 | 正常/杂音/异位/伪影 | **否** | http://www.peterjbentley.com/heartchallenge/ |
| Yaseen Khan's Dataset | 2018 | 5类（AS/MR/MS/MVP/正常） | **否（单标签）** | https://github.com/yaseen21khan/Classification-of-Heart-Sound-Signal |
| CirCor DigiScope | 2021 | 杂音时序/形态/音调/强度 | **否（S1/S2为主）** | https://physionet.org/content/circor-heart-sound/1.0.3/ |
| Heart Sounds Shenzhen (HSS) | 2019 | 正常 / 异常 | **否** | — |
| HeartWave | 2023 | 9类心脏疾病 | **部分支持** | https://doi.org/10.1109/ACCESS.2023.3325749 |
| **BMD-HS（BUET）** | **2024/2026** | **6类含多发病 MD 类，多标签** | ✅ **是** | https://github.com/mHealthBuet/BMD-HS-Dataset |

---

## 二、涉及多瓣膜/多疾病的研究文献

---

### 1. BUET Multi-disease Heart Sound Dataset: A Comprehensive Auscultation Dataset for Developing Computer-Aided Diagnostic Systems

- **作者**：Shams Nafisa Ali, Afia Zahin, Samiul Based Shuvo, Taufiq Hasan 等（孟加拉国工程技术大学 BUET）
- **期刊**：Computer Methods and Programs in Biomedicine Update
- **年份**：2026（arXiv 预印本 2024 年 9 月）
- **论文链接**：https://doi.org/10.1016/j.cmpbup.2026.100237
- **arXiv**：https://arxiv.org/abs/2409.00724
- **数据集**：https://github.com/mHealthBuet/BMD-HS-Dataset
- **主要研究内容**：
  - 发布了 **BMD-HS 数据集**，共 864 条 20 秒 PCG 录音，来自孟加拉国心血管病研究所（NICVD）108 名受试者，所有诊断均经**超声心动图确认**。
  - 数据集设计了**六个类别**：正常（N）、主动脉狭窄（AS）、主动脉反流（AR）、二尖瓣狭窄（MS）、二尖瓣反流（MR），以及**多发病（MD）**。
  - **最大亮点**：采用**多标签标注（multi-label annotation）**，可同时标记患者的多种共存疾病，例如同时患有 AS 和 AR。而现有大多数数据集均为单标签。
  - 每名受试者在四个听诊部位（主动脉位、二尖瓣位、肺动脉位、三尖瓣位）和两种体位（坐位、仰卧位）下各采集 8 条录音，支持多部位联合诊断研究。
  - 提供了年龄、性别、吸烟习惯、居住地等人口学元数据，为社会流行病学相关性研究提供支持。
  - MD 类（多发病）患者共 50 例，占全部病例的最大比例，分析表明：约 36% 的多发病患者合并重度 AS，20% 合并重度 MS，14% 合并重度 MR。
  - 基准实验（Mel 频谱 + CNN + 元数据融合）在测试集上 macro F1 = 0.80，ICBHI score = 0.94。

---

### 2. Aortic Stenosis with Other Concomitant Valvular Disease: Aortic Regurgitation, Mitral Regurgitation, Mitral Stenosis, or Tricuspid Regurgitation

- **作者**：P. Unger, C. Tribouilloy
- **期刊**：Cardiology Clinics
- **年份**：2020，Vol. 38, No. 1, pp. 33–46
- **论文链接**：https://doi.org/10.1016/j.ccl.2019.09.004
- **PubMed**：检索词 Unger Tribouilloy 2020 concomitant valvular
- **主要研究内容**：
  - 这是一篇临床综述，系统地阐述了**主动脉狭窄（AS）合并其他瓣膜疾病**的临床评估和处理策略，是理解多瓣膜疾病病理机制的重要背景文献。
  - 详细讨论了 AS 合并 AR、MR、MS 和三尖瓣反流（TR）各种组合的流行病学、血流动力学影响、超声评估方法和治疗决策。
  - 强调临床上多瓣膜共病极为常见，但当前诊断标准和指南主要针对单瓣膜疾病制定，对共病状态的处理尚无明确循证依据。
  - **对 AI-PCG 研究的启示**：这类文献说明多瓣膜共病在临床上的普遍性和复杂性，也指出了 PCG 信号中多瓣膜共鸣叠加（如 AS 的收缩期杂音 + MR 的全收缩期杂音）给自动算法带来的识别难度。

---

### 3. Transfer Learning Models for Detecting Six Categories of Phonocardiogram Recordings

- **作者**：Miao Wang, Bo Guo, Yanrui Hu, Zhaobo Zhao, Chengyu Liu, Hong Tang
- **期刊**：Journal of Cardiovascular Development and Disease (JCDD)
- **年份**：2022，Vol. 9, No. 3, p. 86
- **DOI / 链接**：https://doi.org/10.3390/jcdd9030086
- **PubMed**：https://pubmed.ncbi.nlm.nih.gov/35323634/
- **主要研究内容**：
  - 提出了基于迁移学习的 PCG **六类**分类方法，类别包括：正常、AS、MR、MS、二尖瓣脱垂（MVP）和二尖瓣反流+主动脉瓣反流共病（MR+AR）。
  - **明确引入了一个多瓣膜共病类别（MR+AR）**，这在当时较为少见，使分类问题从传统单标签扩展到包含至少一种多病组合的场景。
  - 使用预训练的 VGG16、InceptionV3 等迁移学习网络对 PCG 的时频图（Mel 频谱图、MFCC 等）进行分类。
  - 数据来源于 Yaseen Khan 的开放数据集，并对多病类别进行了补充标注。
  - 实验结果表明，多病联合类别的分类准确率相对较低（与单类别相比），作者指出"多病信号之间的频域叠加是分类困难的主要原因"，间接揭示了多瓣膜疾病检测的核心挑战。

---

### 4. An Efficient and Robust Phonocardiography (PCG)-Based Valvular Heart Diseases (VHD) Detection Framework Using Vision Transformer (ViT)

- **作者**：Sonain Jamil, Arunabha M. Roy
- **期刊**：Computers in Biology and Medicine
- **年份**：2023，Vol. 158, p. 106734
- **DOI / 链接**：https://doi.org/10.1016/j.compbiomed.2023.106734
- **PubMed**：https://pubmed.ncbi.nlm.nih.gov/36989745/
- **主要研究内容**：
  - 提出基于 **Vision Transformer（ViT）**的瓣膜性心脏病（VHD）自动检测框架，输入为 PCG 信号的连续小波变换（CWT）时频图。
  - 分类类别涵盖：正常、AS、AR、MS、MR 五类，属于**单瓣膜、多类别**分类任务。
  - ViT 模型通过自注意力机制捕获时频图中的长程时频依赖关系，相比 CNN 在噪声鲁棒性上有一定优势。
  - 论文中讨论了不同瓣膜疾病在时频图中的频率特征重叠问题，指出 MS 的低频舒张期杂音与 AR 的早舒张期杂音在频域上的混叠是误分的主要来源之一。
  - **局限性**：该框架仍为单标签分类，无法处理实际临床中常见的多瓣膜共病情况，作者在结论部分明确将"扩展至多标签多疾病场景"列为未来工作方向。

---

### 5. HeartWave: A Multiclass Dataset of Heart Sounds for Cardiovascular Diseases Detection

- **作者**：S. Alrabie, A. Barnawi（King Abdulaziz University & National Heart Institute）
- **期刊**：IEEE Access
- **年份**：2023，Vol. 11, pp. 118722–118736
- **DOI / 链接**：https://doi.org/10.1109/ACCESS.2023.3325749
- **主要研究内容**：
  - 发布了 **HeartWave 数据集**，来自沙特阿拉伯和埃及四家医院，共 1353 条录音，包含 **9 个类别**，涵盖了较丰富的瓣膜相关疾病。
  - 9 类别包括：正常、AS、AR、MS、MR、二尖瓣脱垂（MVP）、心力衰竭（HF）、冠心病（CAD）及肺动脉高压（PAH）。
  - 与 BMD-HS 不同，HeartWave 仍采用**单标签方案**，即每条录音只属于一个类别，不支持多病共存标注。
  - 数据集特点：规模较大、疾病种类多样、来源于真实临床环境，但缺乏多标签能力和超声验证。
  - 基准实验结合了 CNN、LSTM 等多种模型，在完整 9 类任务上整体准确率约为 85%。

---

### 6. Automated Classification of Valvular Heart Diseases Using FBSE-EWT and PSR Based Geometrical Features

- **作者**：Sibghatullah I. Khan, Saeed Mian Qaisar, Ram Bilas Pachori（印度 IIT Indore 等）
- **期刊**：Biomedical Signal Processing and Control
- **年份**：2022，Vol. 73
- **DOI / 链接**：https://doi.org/10.1016/j.bspc.2021.103385
- **ScienceDirect**：https://www.sciencedirect.com/science/article/pii/S1746809421010429
- **主要研究内容**：
  - 提出结合**Fourier-Bessel 级数展开经验小波变换（FBSE-EWT）**和**相空间重构（PSR）**几何特征的自动瓣膜心脏病分类方法，实现 AS/MR/MS/MVP/正常五类分类。
  - 属于单标签分类，但其特征提取方法（频带分解 + 相空间轨迹几何描述）对于多病混叠信号的特征分离具有潜在的可扩展性。
  - 分类精度较高（准确率 > 97%），但测试数据来自理想化的 Yaseen Khan 数据集，真实临床适用性有待验证。
  - **对多瓣膜研究的参考价值**：EWT 分解方法可为多病混叠信号的频带分离提供技术基础。

---

### 7. Transfer Learning Based Heart Valve Disease Classification from Phonocardiogram Signal

- **作者**：Arnab Maity, Akanksha Pathak, Goutam Saha（印度 IIT Kharagpur）
- **期刊**：Biomedical Signal Processing and Control
- **年份**：2023，Vol. 86
- **DOI / 链接**：https://doi.org/10.1016/j.bspc.2023.104975
- **ScienceDirect**：https://www.sciencedirect.com/science/article/pii/S1746809423002380
- **主要研究内容**：
  - 提出基于预训练 CNN（VGG、ResNet、DenseNet 等）迁移学习的 PCG 瓣膜病分类方法，覆盖 AS/AR/MS/MR/正常五类。
  - 使用 Synchrosqueezing Transform（SST）时频表示作为输入，比传统 STFT/Mel 谱对非平稳心音信号具有更好的时频分辨率。
  - 属于单标签分类，作者在讨论中指出数据集缺乏多病共存样本是主要局限，并建议未来研究收集真实临床多病数据。

---

## 三、研究现状综合分析

### 现有研究的主要局限

```
1. 标注粒度不足
   - 大多数数据集（PhysioNet 2016、PASCAL 等）仅二分类（正常/异常）
   - 即使细粒度数据集（Yaseen Khan、CirCor）也是单标签，不支持多病标注

2. 多瓣膜共病几乎被忽略
   - 目前仅 BMD-HS 数据集（2024/2026）通过多标签系统显式支持多发病（MD）类别
   - 仅 Wang 等 2022 年的论文中引入了 MR+AR 共病组合类别

3. 模型架构尚未针对多标签优化
   - 现有大量模型仍为 softmax 单标签输出，无法同时预测多种疾病
   - 多标签损失函数（Binary Cross-Entropy）、阈值优化、标签相关性建模等问题均未充分研究

4. 真实临床环境下缺乏验证
   - 多数研究使用相同来源数据集（Yaseen Khan 的理想化数据）
   - 信号质量参差不齐条件下多病共病的鲁棒检测未经充分测试
```

### 核心技术挑战

| 挑战 | 具体描述 |
|------|---------|
| 频域叠加 | AS 收缩期杂音 + MR 全收缩期杂音在相同时间窗内共存，频率成分重叠 |
| 时域混叠 | AR 早舒张期杂音 + MS 舒张期杂音均出现在舒张期，边界难以分离 |
| 标签相关性 | AS 患者常合并 AR（因主动脉根部病变），MS 患者常合并 MR（因风湿病因）|
| 数据稀缺 | 多病共存患者在公开数据集中极少，类别不平衡严重 |
| 评估指标 | 传统准确率不适用多标签任务，需使用 macro F1、subset accuracy、Hamming loss 等 |

### 可能的研究切入点

1. **多标签分类架构**：将 Springer HSMM 分割 + 多标签 CNN/Transformer 结合，先分割再对每心动周期进行多标签分类
2. **多部位联合诊断**：利用四个听诊部位（主动脉位、二尖瓣位等）的信号互补性，联合推断多瓣膜状态
3. **弱监督 / 半监督学习**：利用有限的多病标注数据，结合丰富的单病数据进行知识迁移
4. **生成式数据增强**：生成多病混叠的合成 PCG 信号，缓解数据稀缺问题（可参考 GAN/Diffusion 方法）
5. **特征解耦**：在时频域上分离来自不同瓣膜位置的声学特征（依赖多通道/多部位录音）

---

## 四、参考文献汇总

| 编号 | 第一作者 | 年份 | 多病支持 | 链接 |
|------|---------|------|---------|------|
| 1 | Ali (BMD-HS) | 2024/2026 | ✅ 多标签 | https://doi.org/10.1016/j.cmpbup.2026.100237 |
| 2 | Unger (综述) | 2020 | 临床背景 | https://doi.org/10.1016/j.ccl.2019.09.004 |
| 3 | Wang (迁移学习6类) | 2022 | 部分（MR+AR） | https://doi.org/10.3390/jcdd9030086 |
| 4 | Jamil (ViT-VHD) | 2023 | ❌ 单标签 | https://doi.org/10.1016/j.compbiomed.2023.106734 |
| 5 | Alrabie (HeartWave) | 2023 | ❌ 单标签9类 | https://doi.org/10.1109/ACCESS.2023.3325749 |
| 6 | Khan (FBSE-EWT) | 2022 | ❌ 单标签 | https://doi.org/10.1016/j.bspc.2021.103385 |
| 7 | Maity (迁移学习SST) | 2023 | ❌ 单标签 | https://doi.org/10.1016/j.bspc.2023.104975 |
