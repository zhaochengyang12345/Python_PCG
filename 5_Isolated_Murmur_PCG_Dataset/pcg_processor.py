# -*- coding: utf-8 -*-
"""
PCG 心音数据集处理工具
=====================
将自建 WAV/CSV 数据集按照公开数据集风格处理，支持三步操作：
  Step1 — 将长录音分割为含 2-6 个完整心动周期的短片段，并批量保存
  Step2 — 按疾病类型，仅保留生理意义上有杂音的时段（收缩/舒张期掩膜）
  Step3 — 在 Step2 基础上进一步剔除 S1/S2，只保留纯杂音区段

用法：直接运行，弹出 GUI；亦可通过命令行 `python pcg_processor.py`
"""

import os
import sys
import json
import warnings
import threading
import queue

import numpy as np
import scipy.io.wavfile as wav_io
from scipy.signal import butter, filtfilt

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib import rcParams

warnings.filterwarnings("ignore")

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

# ── 将 springer_hsmm 包加入路径 ─────────────────────────────────
_SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
_SPRINGER_DIR    = os.path.join(_SCRIPT_DIR, "springer_hsmm")
sys.path.insert(0, _SPRINGER_DIR)
sys.path.insert(0, _SCRIPT_DIR)

from springer_hsmm.utils.preprocessing import (
    load_wav_file, load_csv_file, scan_folder,
    preprocess_for_display, resample_to_fs, load_mat_data,
)
from springer_hsmm.models.segmentation_algorithm import (
    train_segmentation_algorithm, predict_segmentation,
)

# ================================================================
#  配置
# ================================================================
TRAIN_DATA_PATH  = os.path.join(_SPRINGER_DIR, "data", "example_data.mat")
TRAIN_COUNT      = 5
TRAIN_AUDIO_FS   = 1000

WAV_INPUT_FS     = 4000
CSV_INPUT_FS     = 8000
CSV_PCG_COLUMN   = "pcg"

FEATURES_FS      = 50
PROCESSING_FS    = 1000

# 预处理参数
LOWCUT           = 20.0
HIGHCUT          = 400.0
FILTER_ORDER     = 4
CLIP_PERCENTILE  = 99.0
HAMPEL_WIN       = 40
HAMPEL_SIGMA     = 2.5
NOTCH_FREQS      = [50.0, 100.0]

# 分割参数
MIN_CYCLES       = 2      # 每段最少完整心动周期数
MAX_CYCLES       = 6      # 每段最多完整心动周期数

# 显示颜色
STATE_COLORS = {1: "#e05c7a", 2: "#fab387", 3: "#f5a623", 4: "#89b4fa"}
STATE_LABELS = {1: "S1", 2: "收缩期", 3: "S2", 4: "舒张期"}

# ================================================================
#  疾病-杂音区段映射（基于生理学）
#  值为 Springer 状态列表：1=S1, 2=收缩期, 3=S2, 4=舒张期
#
#  Step2 保留范围：包含边界 S1/S2，让用户看到完整上下文
#  Step3 在此基础上再剔除 S1(1)/S2(3)，只剩纯杂音
# ================================================================
# 根据生理书：
#   AS  : 收缩期喷射性杂音（100-400 Hz）
#   AR  : 舒张期高调递减型吹风样杂音（200-500 Hz）+ 重度时中晚期隆隆样
#   MR  : 全收缩期/泛收缩期吹风样杂音（150-500 Hz）
#   MS  : 舒张期中晚期隆隆样杂音（30-200 Hz）
#   MVP : 收缩中期喀喇音 + 收缩晚期渐增型（100-400 Hz）
#
#  Step2 包含 S1/S2 以保证上下文完整；[1,2,3] = S1+收缩期+S2
DISEASE_MURMUR_STATES = {
    "AS":  [1, 2, 3],       # 收缩期（含边界 S1/S2）；S4 偶发，需手动启用舒张期
    "AR":  [1, 3, 4],       # 舒张期（含边界 S1/S2）
    "MR":  [1, 2, 3],       # 收缩期（含边界 S1/S2）；S3 偶发，需手动启用舒张期
    "MS":  [1, 3, 4],       # 舒张期（含边界 S1/S2）
    "MVP": [1, 2, 3],       # 收缩期（含边界 S1/S2）
    # 正常 — 保留全部
    "N":   [1, 2, 3, 4],
}

# ── 疾病参考信息（显示在 GUI 侧栏）────────────────────────────
DISEASE_INFO = {
    "AS": {
        "name": "主动脉瓣狭窄 AS",
        "murmur_phase": "收缩期",
        "murmur_type": "喷射型/递增递减型",
        "freq_range": "100–400 Hz（部分可延至 350–400 Hz）",
        "extra_sounds": "① 可有 S4（S1 前 100ms，20–70 Hz）\n② 先天性二叶瓣可有射血喀喇音（约 100–300 Hz）\n③ 重症时可见反常分裂 S2/A2 减弱\n⚠ S4 偶发（重度约 30–40%），默认不保留舒张期；\n  若本录音确认有 S4，请勾选手动模式并启用【舒张期】",
        "best_position": "主动脉瓣区（胸骨右缘第2肋间）",
        "keep_states": [1, 2, 3],
    },
    "AR": {
        "name": "主动脉瓣反流 AR",
        "murmur_phase": "舒张期",
        "murmur_type": "① 典型高调、吹风样、递减型舒张早期杂音\n② 重度时舒张中晚期隆隆样杂音（Austin Flint 约 40–120 Hz）",
        "freq_range": "① 约 200–500 Hz\n② 约 40–120 Hz",
        "extra_sounds": "① 急性或重度可有 S3（S2 后 120–180ms，20–50 Hz）\n② 重度时 A2 可减弱",
        "best_position": "主动脉瓣第二听诊区（胸骨左缘第3-4肋间）",
        "keep_states": [1, 3, 4],
    },
    "MR": {
        "name": "二尖瓣反流 MR",
        "murmur_phase": "收缩期",
        "murmur_type": "全收缩期/泛收缩期吹风样杂音",
        "freq_range": "约 150–400 Hz（部分可到 500 Hz）",
        "extra_sounds": "① S3（S2 后 120–180ms，约 20–50 Hz）\n② S1 可减弱\n⚠ S3 偶发（中重度约 40–60%），默认不保留舒张期；\n  若本录音确认有 S3，请勾选手动模式并启用【舒张期】",
        "best_position": "心尖部（左第5肋间锁骨中线内侧）",
        "keep_states": [1, 2, 3],
    },
    "MS": {
        "name": "二尖瓣狭窄 MS",
        "murmur_phase": "舒张期",
        "murmur_type": "舒张中晚期隆隆样杂音，常伴舒张晚期增强",
        "freq_range": "30–120 Hz（部分可到 40–200 Hz）",
        "extra_sounds": "① 开瓣音 OS（S2 后 60–120ms，80–250 Hz）\n② 常有 S1 增强（约 30–100 Hz）",
        "best_position": "心尖部（左第5肋间锁骨中线内侧）",
        "keep_states": [1, 3, 4],
    },
    "MVP": {
        "name": "二尖瓣脱垂 MVP",
        "murmur_phase": "收缩期",
        "murmur_type": "收缩中期喀喇音（100–300 Hz）+ 收缩晚期渐增型（150–400 Hz）",
        "freq_range": "100–400 Hz",
        "extra_sounds": "/",
        "best_position": "心尖部（左第5肋间锁骨中线内侧）",
        "keep_states": [1, 2, 3],
    },
    "N": {
        "name": "正常 N",
        "murmur_phase": "无杂音",
        "murmur_type": "/",
        "freq_range": "/",
        "extra_sounds": "/",
        "best_position": "/",
        "keep_states": [1, 2, 3, 4],
    },
}

# 如无法识别疾病标签，默认保留全部周期内容
_DEFAULT_MURMUR_STATES = [1, 2, 3, 4]

# ================================================================
#  全局 Springer 模型（延迟加载）
# ================================================================
_springer_model = None
_model_lock     = threading.Lock()


def _ensure_model():
    global _springer_model
    with _model_lock:
        if _springer_model is not None:
            return _springer_model
        print("[模型] 正在从训练数据加载 Springer HSMM 模型…")
        audio_data, annotations = load_mat_data(TRAIN_DATA_PATH)
        # band_pi_matrices 硬编码需要 5 条记录
        n = max(5, min(TRAIN_COUNT, len(audio_data)))
        if len(audio_data) < 5:
            raise RuntimeError(
                f"训练数据只有 {len(audio_data)} 条，至少需要 5 条。"
            )
        train_recs  = [audio_data[i] for i in range(n)]
        train_annot = annotations[:n]
        _springer_model = train_segmentation_algorithm(
            train_recs, train_annot, FEATURES_FS, TRAIN_AUDIO_FS
        )
        print("[模型] 加载完成。")
        return _springer_model


# ================================================================
#  信号加载工具
# ================================================================
def load_signal(path):
    """
    加载 WAV 或 CSV 文件，返回 (signal_float64, fs)。
    CSV 始终使用 8000 Hz（除非文件内嵌时间列可推断）。
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        sig, fs = load_wav_file(path)
        if fs != WAV_INPUT_FS:
            sig = resample_to_fs(sig, fs, WAV_INPUT_FS)
        return sig, WAV_INPUT_FS
    elif ext == ".csv":
        sig, detected_fs = load_csv_file(path, CSV_PCG_COLUMN)
        fs = detected_fs if detected_fs else CSV_INPUT_FS
        return sig, fs
    else:
        raise ValueError(f"不支持的文件格式: {path}")


def _detect_disease_from_path(path):
    """
    从文件名或上级目录名中提取疾病标签（大小写不敏感）。
    返回 DISEASE_MURMUR_STATES 中的键（字符串），找不到返回 None。
    """
    tokens = path.replace("\\", "/").upper().split("/")
    for token in reversed(tokens):
        for key in DISEASE_MURMUR_STATES:
            if key in token:
                return key
    return None


# ================================================================
#  Step 1：分割为完整心动周期片段
# ================================================================
def segment_into_cycles(signal, fs, model, min_cycles=MIN_CYCLES, max_cycles=MAX_CYCLES):
    """
    使用 Springer 模型将长录音切割成含 min_cycles~max_cycles 个完整
    心动周期的片段。只保留以 S1 开头、以 S1/S2 之前结束的完整周期。

    返回：list of dict {
        'signal': ndarray,
        'fs': int,
        'states': ndarray,          # 与 signal 等长，值 1/2/3/4
        'cycle_count': int,
    }
    """
    # 运行分割
    states = predict_segmentation(
        signal, FEATURES_FS, fs,
        model["pi_vector"], model["model"], model["total_obs_distribution"],
        processing_fs=PROCESSING_FS,
    )
    states = np.array(states, dtype=int)

    # 找到所有 S1 起始位置（状态从非1变为1的跳变沿）
    s1_onsets = _find_state_onsets(states, target_state=1)

    if len(s1_onsets) < 2:
        return []

    segments = []
    i = 0
    while i < len(s1_onsets):
        for j in range(i + max_cycles, i + min_cycles - 1, -1):
            if j < len(s1_onsets):
                start = s1_onsets[i]
                end   = s1_onsets[j]   # 以下一个 S1 起始作为边界（不含）
                seg_cycles = j - i
                seg_sig    = signal[start:end]
                seg_states = states[start:end]
                if len(seg_sig) > 0:
                    segments.append({
                        "signal":      seg_sig,
                        "fs":          fs,
                        "states":      seg_states,
                        "cycle_count": seg_cycles,
                        "start_sample": start,
                        "end_sample":   end,
                    })
                i = j
                break
        else:
            # 剩余不足 min_cycles，跳过
            break

    return segments


def _find_state_onsets(states, target_state=1):
    """返回状态序列中 target_state 的所有起始（上升沿）样本索引。"""
    onsets = []
    in_state = False
    for idx, s in enumerate(states):
        if s == target_state and not in_state:
            onsets.append(idx)
            in_state = True
        elif s != target_state:
            in_state = False
    return onsets


def detect_extra_sounds(signal, states, fs,
                        s3_window_ms=(80, 220),
                        s4_window_ms=(50, 150),
                        energy_threshold_ratio=2.5):
    """
    在 Springer 4 状态分割结果基础上，用离散小波变换（DWT）在固定
    时间窗内检测 S3（舒张早期）和 S4（收缩前期）。

    算法依据
    --------
    S3/S4 的主要能量集中在 20–70 Hz 低频段。利用 DWT 将信号分解至
    多个尺度，选取对应 20–70 Hz 的细节系数层（随 fs 自动选取），
    在各心动周期的固定时间窗内计算小波域能量，与同周期舒张中段
    的参考能量做比值检验：

      - S3：S2 结束后 80–220 ms 内的小波能量 ÷ 舒张中段参考能量
      - S4：S1 开始前 50–150 ms 内的小波能量 ÷ 舒张中段参考能量

    比值 ≥ energy_threshold_ratio 时判定为候选。

    DWT 层级选择（db4 小波）
    -----------------------
    层级 d 对应频率区间 [fs/2^(d+1), fs/2^d]，选取使区间覆盖
    20–70 Hz 最多的层级（通常为同时覆盖该范围的 1–2 层）：
      fs=4000: level 5 → 62.5–125 Hz，level 6 → 31.25–62.5 Hz  ← 主要选 6
      fs=8000: level 6 → 62.5–125 Hz，level 7 → 31.25–62.5 Hz  ← 主要选 7

    参数
    ----
    signal                 : 原始信号，float64
    states                 : 与 signal 等长的状态序列（1/2/3/4）
    fs                     : 采样率（Hz）
    s3_window_ms           : S3 相对 S2 结束的搜索时间窗（ms）
    s4_window_ms           : S4 相对 S1 开始的反向搜索时间窗（ms）
    energy_threshold_ratio : 目标窗 / 参考窗 小波能量比阈值

    返回
    ----
    dict {
        "s3_candidates": [(start_sample, end_sample, energy_ratio), ...],
        "s4_candidates": [(start_sample, end_sample, energy_ratio), ...],
    }
    """
    import pywt
    import math

    WAVELET = "db4"

    # ── 选取覆盖 20–70 Hz 的 DWT 细节层 ─────────────────────────
    # 层级 d: 频率上界 = fs/2^d, 下界 = fs/2^(d+1)
    # 选取所有与 [20, 70] 有交集的层级
    def _target_levels(fs, f_lo=20.0, f_hi=70.0):
        levels = []
        max_level = pywt.dwt_max_level(4096, WAVELET)  # 足够大的上限
        for d in range(1, max_level + 1):
            band_hi = fs / (2 ** d)
            band_lo = fs / (2 ** (d + 1))
            if band_lo < f_hi and band_hi > f_lo:
                levels.append(d)
        return levels if levels else [6]

    target_levels = _target_levels(fs)
    max_level = max(target_levels)

    # DWT 分解
    coeffs = pywt.wavedec(signal, WAVELET, level=max_level)
    # coeffs[0] = 近似系数 cA_N
    # coeffs[k] = 第 k 层细节系数 cD_k（k=1 为最高频）

    # 将目标层细节系数在时域重建（仅保留这些层，其余置零）
    coeffs_filtered = [np.zeros_like(c) for c in coeffs]
    for d in target_levels:
        if d < len(coeffs):
            coeffs_filtered[d] = coeffs[d].copy()
    sig_wavelet = pywt.waverec(coeffs_filtered, WAVELET)
    # 长度可能比原始信号多 1，对齐截断
    sig_wavelet = sig_wavelet[:len(signal)]

    # ── 辅助：小波子带的样本能量（均方值） ──────────────────────
    def _wavelet_energy(arr):
        return float(np.mean(arr ** 2)) + 1e-20

    states = np.asarray(states, dtype=int)
    n = len(states)

    # 找所有 S2 结束位置（状态 3→4）
    s2_ends = [i for i in range(1, n) if states[i - 1] == 3 and states[i] == 4]
    # 找所有 S1 开始位置
    s1_starts = _find_state_onsets(states, target_state=1)

    s3_candidates = []
    s4_candidates = []

    w_s3_lo = int(s3_window_ms[0] * fs / 1000)
    w_s3_hi = int(s3_window_ms[1] * fs / 1000)
    w_s4_lo = int(s4_window_ms[0] * fs / 1000)
    w_s4_hi = int(s4_window_ms[1] * fs / 1000)

    # ── S3 检测：S2 结束后的固定窗 ──────────────────────────────
    for s2e in s2_ends:
        win_start = s2e + w_s3_lo
        win_end   = s2e + w_s3_hi
        if win_end >= n:
            continue
        # 参考：舒张中段（S2结束后 300–600 ms），S3 窗口之后的静息区
        ref_start = s2e + int(0.30 * fs)
        ref_end   = s2e + int(0.60 * fs)
        if ref_end >= n:
            continue
        ref_energy = _wavelet_energy(sig_wavelet[ref_start:ref_end])
        win_energy = _wavelet_energy(sig_wavelet[win_start:win_end])
        ratio = win_energy / ref_energy
        if ratio >= energy_threshold_ratio:
            s3_candidates.append((win_start, win_end, round(float(ratio), 1)))

    # ── S4 检测：S1 开始前的固定窗 ──────────────────────────────
    for s1s in s1_starts:
        win_end   = s1s - w_s4_lo
        win_start = s1s - w_s4_hi
        if win_start < 0 or win_end <= win_start:
            continue
        # 参考：该窗口再往前 150 ms 的舒张静息区
        ref_start = win_start - int(0.15 * fs)
        ref_end   = win_start
        if ref_start < 0:
            continue
        ref_energy = _wavelet_energy(sig_wavelet[ref_start:ref_end])
        win_energy = _wavelet_energy(sig_wavelet[win_start:win_end])
        ratio = win_energy / ref_energy
        if ratio >= energy_threshold_ratio:
            s4_candidates.append((win_start, win_end, round(float(ratio), 1)))

    return {
        "s3_candidates": s3_candidates,
        "s4_candidates": s4_candidates,
    }


# ================================================================
#  Step 2：按疾病保留有杂音的区段（含 S1/S2 边界，保留上下文）
# ================================================================
def apply_disease_mask(signal, states, disease_key, custom_states=None):
    """
    根据疾病将信号中不关注的时段置零，仅保留 murmur 相关时段。
    Step2 保留范围包含边界 S1/S2，让用户看到完整上下文。
    Step3 才剔除 S1/S2，使两步结果有明显区别。

    custom_states: 若用户手动选择了要保留的状态列表，优先使用。

    返回 masked_signal (ndarray, 与输入等长)
    """
    if custom_states is not None:
        keep_states = custom_states
    elif disease_key is None:
        return signal.copy()
    else:
        info = DISEASE_INFO.get(disease_key)
        keep_states = info["keep_states"] if info else _DEFAULT_MURMUR_STATES

    mask = np.zeros(len(signal), dtype=bool)
    for ms in keep_states:
        mask |= (states == ms)

    masked = signal.copy()
    masked[~mask] = 0.0
    return masked


# ================================================================
#  Step 3：剔除 S1 / S2（在 Step2 基础上再将 S1/S2 置零）
# ================================================================
def remove_s1_s2(signal, states):
    """
    将状态为 S1(1) 和 S2(3) 的采样点置零，保留收缩/舒张期部分。

    返回 filtered_signal (ndarray, 与输入等长)
    """
    out = signal.copy()
    out[(states == 1) | (states == 3)] = 0.0
    return out


# ================================================================
#  批量处理入口
# ================================================================
def process_file_step1(path, model):
    """
    加载单个文件，运行 Step1 分割，返回片段列表。
    每个元素：dict{signal, fs, states, cycle_count, start_sample, end_sample}
    """
    sig, fs = load_signal(path)
    return segment_into_cycles(sig, fs, model)


def process_file_step2(path, model, custom_states=None):
    """
    加载文件 → Step1 分割 → Step2 疾病掩膜（含S1/S2边界）。
    返回 list of dict，增加字段 masked_signal, disease_key。
    custom_states: 用户手动指定要保留的状态列表（覆盖自动检测）。
    """
    disease = _detect_disease_from_path(path)
    segs    = process_file_step1(path, model)
    for seg in segs:
        seg["disease_key"]    = disease
        seg["masked_signal"]  = apply_disease_mask(
            seg["signal"], seg["states"], disease, custom_states=custom_states
        )
    return segs, disease


def process_file_step3(path, model, custom_states=None):
    """
    加载文件 → Step1 → Step2 → Step3（剔除S1/S2，只剩纯杂音）。
    返回 list of dict，增加字段 murmur_only_signal。
    """
    segs, disease = process_file_step2(path, model, custom_states=custom_states)
    for seg in segs:
        seg["murmur_only_signal"] = remove_s1_s2(
            seg["masked_signal"], seg["states"]
        )
    return segs, disease


# ================================================================
#  交互式分段编辑器 (Toplevel 对话框)
# ================================================================
class SegEditorDialog(tk.Toplevel):
    """
    在独立弹窗中对单个片段的状态序列进行手动校正。

    操作方式
    --------
    1. 点击顶部状态按钮，选择目标状态（S1 / 收缩期 / S2 / 舒张期）
    2. 在波形上 **点击并拖拽** 选定时间区间，松手后自动填充所选状态
    3. 可多次操作，支持撤销(↩)和重置
    4. 点击"确认应用"将修正后的 states 写回片段并刷新主窗口
    """

    _BTN_COLORS = {1: "#e05c7a", 2: "#fab387", 3: "#f5a623", 4: "#89b4fa"}

    def __init__(self, parent, path, seg, on_apply_cb):
        super().__init__(parent)
        self.transient(parent)
        self.title(f"分段编辑器 — {os.path.basename(path)}")
        self.geometry("1100x580")
        self.resizable(True, True)

        self._path        = path
        self._seg         = seg
        self._on_apply_cb = on_apply_cb
        self._fs          = seg["fs"]
        self._sig         = seg["signal"]
        self._states      = np.array(seg["states"], dtype=int).copy()
        self._orig_states = self._states.copy()
        self._history     = []           # undo 栈，最多 50 步

        self._active_state = tk.IntVar(value=1)
        self._hint_var     = tk.StringVar(
            value="提示：先点击状态按钮选色，再在波形上拖拽选区，松手后自动填充")
        self._sel_start    = None
        self._rb_span      = None
        self._cid_press    = None
        self._cid_motion   = None
        self._cid_release  = None

        self._build_ui()
        self._connect_events()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.grab_set()

    # ── UI ──────────────────────────────────────────────────────
    def _build_ui(self):
        tb = ttk.Frame(self, padding=(6, 4))
        tb.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(tb, text="标记为：", font=("Microsoft YaHei", 9)).pack(side=tk.LEFT)
        for sid, lbl in [(1, "S1"), (2, "收缩期"), (3, "S2"), (4, "舒张期")]:
            color = self._BTN_COLORS[sid]
            tk.Radiobutton(
                tb, text=f"  {lbl}  ",
                variable=self._active_state, value=sid,
                bg=color, selectcolor=color, activebackground=color,
                relief=tk.RAISED, indicatoron=False,
                padx=6, pady=3,
                font=("Microsoft YaHei", 9, "bold"),
            ).pack(side=tk.LEFT, padx=3)

        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(tb, text="↩ 撤销", command=self._undo).pack(side=tk.LEFT, padx=2)
        ttk.Button(tb, text="↺ 重置", command=self._reset).pack(side=tk.LEFT, padx=2)
        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Label(tb, textvariable=self._hint_var,
                  foreground="#555", font=("Microsoft YaHei", 8)).pack(side=tk.LEFT)

        ttk.Button(tb, text="✗ 取消",
                   command=self._on_close).pack(side=tk.RIGHT, padx=2)
        ttk.Button(tb, text="✓ 确认应用",
                   command=self._apply).pack(side=tk.RIGHT, padx=4)

        fig_frame = ttk.Frame(self)
        fig_frame.pack(fill=tk.BOTH, expand=True)

        self._fig_e, self._ax_e = plt.subplots(
            1, 1, figsize=(13, 5), facecolor="#fafafa"
        )
        self._fig_e.subplots_adjust(left=0.05, right=0.99, top=0.90, bottom=0.10)
        self._ax_e.set_facecolor("#fafafa")

        self._canvas_e = FigureCanvasTkAgg(self._fig_e, master=fig_frame)
        self._canvas_e.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._nav = NavigationToolbar2Tk(
            self._canvas_e, fig_frame, pack_toolbar=False)
        self._nav.update()
        self._nav.pack(side=tk.BOTTOM, fill=tk.X)

        self._redraw()

    # ── 绘图 ────────────────────────────────────────────────────
    def _redraw(self):
        ax = self._ax_e
        ax.cla()
        ax.set_facecolor("#fafafa")

        t = np.linspace(0, (len(self._sig) - 1) / self._fs, len(self._sig))
        _, sig_d = preprocess_for_display(
            self._sig, self._fs,
            lowcut=LOWCUT, highcut=HIGHCUT,
            filter_order=FILTER_ORDER, clip_pct=CLIP_PERCENTILE,
            hampel_win=max(4, int(self._fs * 0.002)),
            hampel_sigma=HAMPEL_SIGMA, notch_freqs=NOTCH_FREQS,
            apply_wavelet_denoise=False,
        )
        scale = np.percentile(np.abs(sig_d), 99) or 1.0
        sig_n = np.clip(sig_d / scale, -1, 1)

        _draw_state_shading(ax, t, self._states, alpha=0.38)
        ax.plot(t, sig_n, color="#1a73e8", linewidth=0.55, alpha=0.92)
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("归一化幅度")
        ax.set_ylim(-1.35, 1.55)
        ax.grid(True, alpha=0.3, linewidth=0.4)
        _add_state_legend(ax)

        stats = {s: int(np.sum(self._states == s)) / self._fs
                 for s in [1, 2, 3, 4]}
        stat_str = "  ".join(
            f"{STATE_LABELS[s]}: {stats[s]:.2f}s" for s in [1, 2, 3, 4])
        ax.set_title(
            f"{os.path.basename(self._path)}  |  分段编辑模式 — 拖拽波形选区后松手填充\n{stat_str}",
            fontsize=8.5
        )
        self._rb_span = None
        self._canvas_e.draw_idle()

    # ── 鼠标事件 ─────────────────────────────────────────────────
    def _connect_events(self):
        self._cid_press   = self._canvas_e.mpl_connect(
            "button_press_event",   self._on_press)
        self._cid_motion  = self._canvas_e.mpl_connect(
            "motion_notify_event",  self._on_motion)
        self._cid_release = self._canvas_e.mpl_connect(
            "button_release_event", self._on_release)

    def _disconnect_events(self):
        for cid in (self._cid_press, self._cid_motion, self._cid_release):
            if cid is not None:
                try:
                    self._canvas_e.mpl_disconnect(cid)
                except Exception:
                    pass

    def _nav_active(self):
        return bool(self._nav and self._nav.mode)

    def _on_press(self, event):
        if event.inaxes != self._ax_e or event.button != 1 or self._nav_active():
            return
        self._sel_start = event.xdata
        if self._rb_span is not None:
            try:
                self._rb_span.remove()
            except Exception:
                pass
            self._rb_span = None

    def _on_motion(self, event):
        if self._sel_start is None or self._nav_active():
            return
        if event.inaxes != self._ax_e or event.xdata is None:
            return
        x0, x1 = self._sel_start, event.xdata
        if self._rb_span is not None:
            try:
                self._rb_span.remove()
            except Exception:
                pass
        color = self._BTN_COLORS.get(self._active_state.get(), "#cccccc")
        self._rb_span = self._ax_e.axvspan(
            min(x0, x1), max(x0, x1), color=color, alpha=0.45, zorder=5
        )
        self._canvas_e.draw_idle()

    def _on_release(self, event):
        if self._sel_start is None or self._nav_active():
            self._sel_start = None
            return
        x0 = self._sel_start
        x1 = event.xdata if (event.xdata is not None) else x0
        self._sel_start = None

        t_start, t_end = min(x0, x1), max(x0, x1)
        # 选区过短（< 5 个样本）忽略
        if (t_end - t_start) * self._fs < 5:
            if self._rb_span is not None:
                try:
                    self._rb_span.remove()
                except Exception:
                    pass
                self._rb_span = None
                self._canvas_e.draw_idle()
            return

        n  = len(self._states)
        i0 = max(0, int(t_start * self._fs))
        i1 = min(n, int(t_end   * self._fs) + 1)

        self._history.append(self._states.copy())
        if len(self._history) > 50:
            self._history.pop(0)

        target = self._active_state.get()
        self._states[i0:i1] = target
        self._hint_var.set(
            f"✔ {t_start:.3f}s – {t_end:.3f}s → {STATE_LABELS[target]}"
            f"  （{len(self._history)} 步可撤销）"
        )
        self._redraw()

    # ── 撤销 / 重置 / 确认 ──────────────────────────────────────
    def _undo(self):
        if not self._history:
            self._hint_var.set("没有可撤销的操作。")
            return
        self._states = self._history.pop()
        self._hint_var.set(f"已撤销，剩余 {len(self._history)} 步可撤销。")
        self._redraw()

    def _reset(self):
        if not messagebox.askyesno(
                "确认重置", "重置为 Springer 原始分段结果？", parent=self):
            return
        self._history.append(self._states.copy())
        self._states = self._orig_states.copy()
        self._hint_var.set("已重置为原始分段结果。")
        self._redraw()

    def _apply(self):
        self._on_apply_cb(self._states.copy())
        self._on_close()

    def _on_close(self):
        self._disconnect_events()
        try:
            plt.close(self._fig_e)
        except Exception:
            pass
        self.grab_release()
        self.destroy()


# ================================================================
#  GUI 应用
# ================================================================
class PCGProcessorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PCG 心音数据集处理工具")
        self.geometry("1280x820")
        self.resizable(True, True)
        self.configure(bg="#f5f5f5")

        # 共享状态
        self._source_folder = tk.StringVar(value="")
        self._save_folder   = tk.StringVar(value="")
        self._step_var      = tk.IntVar(value=1)
        self._file_list     = []       # 扫描到的所有文件路径
        self._segments      = []       # 处理后的片段列表 [(path, seg), ...]
        self._sel_seg_idx   = -1       # 当前查看的片段索引
        self._model         = None
        self._status_var    = tk.StringVar(value="就绪")
        self._progress_var  = tk.DoubleVar(value=0.0)
        self._result_queue  = queue.Queue()
        # 手动区段选择（CheckVar for each state 1-4）
        self._save_step_var  = tk.IntVar(value=1)   # 批量保存时保存哪一步的输出
        self._manual_mode   = tk.BooleanVar(value=False)
        self._keep_s1       = tk.BooleanVar(value=True)
        self._keep_sys      = tk.BooleanVar(value=True)
        self._keep_s2       = tk.BooleanVar(value=True)
        self._keep_dia      = tk.BooleanVar(value=False)

        self._build_ui()
        self._poll_queue()

    # ────────────────────────────────────────────────────────────
    #  UI 构建
    # ────────────────────────────────────────────────────────────
    def _build_ui(self):
        # === 顶部控制面板 ===
        ctrl = ttk.LabelFrame(self, text="控制面板", padding=8)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(8, 4))

        # 文件夹选择
        row0 = ttk.Frame(ctrl)
        row0.pack(fill=tk.X, pady=2)
        ttk.Label(row0, text="数据文件夹：", width=12).pack(side=tk.LEFT)
        ttk.Entry(row0, textvariable=self._source_folder, width=60).pack(side=tk.LEFT, padx=4)
        ttk.Button(row0, text="浏览…", command=self._browse_source).pack(side=tk.LEFT)
        ttk.Button(row0, text="扫描文件", command=self._scan_files).pack(side=tk.LEFT, padx=6)

        # Step 选择
        row1 = ttk.Frame(ctrl)
        row1.pack(fill=tk.X, pady=4)
        ttk.Label(row1, text="处理步骤：", width=12).pack(side=tk.LEFT)
        for val, lbl in [
            (1, "Step1  分割心动周期"),
            (2, "Step2  疾病掩膜（保留杂音区段，含S1/S2边界）"),
            (3, "Step3  剔除S1/S2（仅保留纯杂音）"),
        ]:
            ttk.Radiobutton(
                row1, text=lbl, variable=self._step_var, value=val
            ).pack(side=tk.LEFT, padx=8)

        # 运行按钮
        row2 = ttk.Frame(ctrl)
        row2.pack(fill=tk.X, pady=2)
        self._run_btn = ttk.Button(row2, text="▶  运行处理", command=self._run_processing)
        self._run_btn.pack(side=tk.LEFT)
        self._progress = ttk.Progressbar(
            row2, variable=self._progress_var, maximum=100, length=300
        )
        self._progress.pack(side=tk.LEFT, padx=10)
        ttk.Label(row2, textvariable=self._status_var, foreground="#555").pack(side=tk.LEFT)

        # === 主体区域：左侧列表 + 中间疾病参考 + 右侧波形 ===
        body = ttk.Frame(self)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        # -- 左侧片段列表 --
        left = ttk.LabelFrame(body, text="片段列表", width=300, padding=4)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 4))
        left.pack_propagate(False)

        # 列表工具栏
        lbar = ttk.Frame(left)
        lbar.pack(fill=tk.X, pady=(0, 2))
        ttk.Button(lbar, text="清除列表", command=self._clear_list).pack(side=tk.LEFT)

        # 保存区（先 pack 底部控件，再 pack 列表框，保证保存区始终可见）
        save_frame = ttk.LabelFrame(left, text="批量保存", padding=4)
        save_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(4, 0))

        save_row = ttk.Frame(save_frame)
        save_row.pack(fill=tk.X)
        ttk.Label(save_row, text="保存至：").pack(side=tk.LEFT)
        ttk.Entry(save_row, textvariable=self._save_folder, width=14).pack(side=tk.LEFT, padx=2)
        ttk.Button(save_row, text="…", width=2, command=self._browse_save).pack(side=tk.LEFT)
        ttk.Button(save_row, text="批量保存", command=self._batch_save).pack(side=tk.LEFT, padx=4)

        save_step_row = ttk.Frame(save_frame)
        save_step_row.pack(fill=tk.X, pady=(3, 0))
        ttk.Label(save_step_row, text="保存步骤：").pack(side=tk.LEFT)
        for val, lbl in [
            (1, "Step1"),
            (2, "Step2"),
            (3, "Step3"),
        ]:
            ttk.Radiobutton(
                save_step_row, text=lbl,
                variable=self._save_step_var, value=val
            ).pack(side=tk.LEFT, padx=4)

        # 列表框（放在保存区之后 pack，expand=True 占用剩余空间）
        list_frame = ttk.Frame(left)
        list_frame.pack(fill=tk.BOTH, expand=True)

        self._seg_listbox = tk.Listbox(
            list_frame, selectmode=tk.SINGLE, font=("Consolas", 9),
            activestyle="dotbox", width=38
        )
        sb = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self._seg_listbox.yview)
        self._seg_listbox.configure(yscrollcommand=sb.set)
        self._seg_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._seg_listbox.bind("<<ListboxSelect>>", self._on_seg_select)

        # -- 中间：疾病参考 + 手动区段选择 --
        mid = ttk.Frame(body, width=240)
        mid.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 4))
        mid.pack_propagate(False)

        # 疾病参考信息框
        info_frame = ttk.LabelFrame(mid, text="杂音参考", padding=6)
        info_frame.pack(fill=tk.BOTH, expand=True)
        self._info_text = tk.Text(
            info_frame, wrap=tk.WORD, font=("Microsoft YaHei", 8),
            state=tk.DISABLED, bg="#f9f9f9", relief=tk.FLAT,
            height=18
        )
        info_sb = ttk.Scrollbar(info_frame, orient=tk.VERTICAL,
                                command=self._info_text.yview)
        self._info_text.configure(yscrollcommand=info_sb.set)
        self._info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        info_sb.pack(side=tk.RIGHT, fill=tk.Y)

        # 手动区段选择
        manual_frame = ttk.LabelFrame(mid, text="手动选择保留区段（Step2/3）", padding=6)
        manual_frame.pack(fill=tk.X, pady=(6, 0))
        ttk.Checkbutton(manual_frame, text="启用手动模式",
                        variable=self._manual_mode,
                        command=self._on_manual_toggle).pack(anchor=tk.W)
        self._cb_s1  = ttk.Checkbutton(manual_frame, text="S1（状态1）",
                                        variable=self._keep_s1)
        self._cb_sys = ttk.Checkbutton(manual_frame, text="收缩期（状态2）",
                                        variable=self._keep_sys)
        self._cb_s2  = ttk.Checkbutton(manual_frame, text="S2（状态3）",
                                        variable=self._keep_s2)
        self._cb_dia = ttk.Checkbutton(manual_frame, text="舒张期（状态4）",
                                        variable=self._keep_dia)
        for cb in (self._cb_s1, self._cb_sys, self._cb_s2, self._cb_dia):
            cb.pack(anchor=tk.W, padx=8)
        self._on_manual_toggle()  # 初始化禁用状态
        ttk.Button(manual_frame, text="重新应用到当前片段",
                   command=self._reapply_manual).pack(pady=(4, 0))

        # -- 右侧波形区域 --
        right = ttk.LabelFrame(body, text="波形预览", padding=4)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 编辑工具栏（波形区域顶部）
        edit_bar = ttk.Frame(right)
        edit_bar.pack(side=tk.TOP, fill=tk.X, pady=(0, 3))
        ttk.Button(
            edit_bar, text="✏  手动调整分段",
            command=self._open_seg_editor
        ).pack(side=tk.LEFT)
        ttk.Label(
            edit_bar,
            text="  在新窗口中拖拽选区手动修正 S1/S2 位置，适用于质量较差的数据",
            foreground="#666", font=("Microsoft YaHei", 8)
        ).pack(side=tk.LEFT)

        self._fig, self._axes = plt.subplots(
            3, 1, figsize=(10, 7),
            gridspec_kw={"height_ratios": [3, 2, 2]},
            facecolor="#fafafa",
        )
        self._fig.subplots_adjust(hspace=0.40, left=0.08, right=0.97)
        for ax in self._axes:
            ax.set_facecolor("#fafafa")

        self._canvas = FigureCanvasTkAgg(self._fig, master=right)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self._canvas, right, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    # ────────────────────────────────────────────────────────────
    #  文件夹/扫描
    # ────────────────────────────────────────────────────────────
    def _browse_source(self):
        d = filedialog.askdirectory(title="选择数据文件夹")
        if d:
            self._source_folder.set(d)

    def _browse_save(self):
        d = filedialog.askdirectory(title="选择保存文件夹")
        if d:
            self._save_folder.set(d)

    def _scan_files(self):
        folder = self._source_folder.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showwarning("提示", "请先选择有效的数据文件夹！")
            return
        files = scan_folder(folder)
        self._file_list = files
        self._status_var.set(f"已扫描到 {len(files)} 个文件")
        # 刷新列表
        self._seg_listbox.delete(0, tk.END)
        for f in files:
            self._seg_listbox.insert(tk.END, f"[文件] {os.path.basename(f)}")
        self._segments = []

    def _clear_list(self):
        """清除列表和处理结果。"""
        self._seg_listbox.delete(0, tk.END)
        self._file_list = []
        self._segments  = []
        self._sel_seg_idx = -1
        self._status_var.set("列表已清除")
        for ax in self._axes:
            ax.cla()
            ax.set_facecolor("#fafafa")
        self._canvas.draw_idle()

    # ────────────────────────────────────────────────────────────
    #  运行处理（后台线程）
    # ────────────────────────────────────────────────────────────
    def _run_processing(self):
        if not self._file_list:
            messagebox.showwarning("提示", "请先扫描文件夹！")
            return
        step = self._step_var.get()
        self._run_btn.configure(state=tk.DISABLED)
        self._progress_var.set(0)
        self._status_var.set("正在加载模型…")
        t = threading.Thread(target=self._worker, args=(step,), daemon=True)
        t.start()

    def _get_custom_states(self):
        """若启用手动模式，返回用户勾选的状态列表，否则返回 None。"""
        if not self._manual_mode.get():
            return None
        states = []
        if self._keep_s1.get():  states.append(1)
        if self._keep_sys.get(): states.append(2)
        if self._keep_s2.get():  states.append(3)
        if self._keep_dia.get(): states.append(4)
        return states if states else [1, 2, 3, 4]

    def _on_manual_toggle(self):
        """切换手动模式时启用/禁用 CheckButton。"""
        state = tk.NORMAL if self._manual_mode.get() else tk.DISABLED
        for cb in (self._cb_s1, self._cb_sys, self._cb_s2, self._cb_dia):
            cb.configure(state=state)

    def _reapply_manual(self):
        """在当前选中片段上手动重新应用掩膜并刷新波形。"""
        if self._sel_seg_idx < 0 or self._sel_seg_idx >= len(self._segments):
            return
        step = getattr(self, "_step_done", self._step_var.get())
        if step < 2:
            messagebox.showinfo("提示", "手动模式仅对 Step2/Step3 有效。")
            return
        path, seg = self._segments[self._sel_seg_idx]
        custom = self._get_custom_states()
        seg["masked_signal"] = apply_disease_mask(
            seg["signal"], seg["states"], seg.get("disease_key"), custom_states=custom
        )
        if step >= 3:
            seg["murmur_only_signal"] = remove_s1_s2(
                seg["masked_signal"], seg["states"]
            )
        self._draw_segment(path, seg)

    # ────────────────────────────────────────────────────────────
    #  分段手动调整编辑器
    # ────────────────────────────────────────────────────────────
    def _open_seg_editor(self):
        """打开分段编辑器弹窗（针对当前选中片段）。"""
        if self._sel_seg_idx < 0 or self._sel_seg_idx >= len(self._segments):
            messagebox.showwarning(
                "提示", "请先在左侧列表中选择一个片段，再使用手动调整分段功能。")
            return
        path, seg = self._segments[self._sel_seg_idx]
        SegEditorDialog(self, path, seg, self._on_seg_editor_apply)

    def _on_seg_editor_apply(self, new_states):
        """
        分段编辑器确认后的回调。
        用修正后的 states 更新片段，重新计算 Step2/3 掩膜，刷新主窗口波形。
        """
        if self._sel_seg_idx < 0 or self._sel_seg_idx >= len(self._segments):
            return
        path, seg = self._segments[self._sel_seg_idx]
        seg["states"] = new_states
        step   = getattr(self, "_step_done", self._step_var.get())
        custom = self._get_custom_states()
        if step >= 2:
            seg["masked_signal"] = apply_disease_mask(
                seg["signal"], seg["states"],
                seg.get("disease_key"), custom_states=custom
            )
        if step >= 3:
            seg["murmur_only_signal"] = remove_s1_s2(
                seg["masked_signal"], seg["states"]
            )
        self._draw_segment(path, seg)
        self._status_var.set(f"✓ 片段 #{self._sel_seg_idx + 1} 分段已手动更新")

    def _worker(self, step):
        try:
            # 加载模型
            model = _ensure_model()
            self._result_queue.put(("status", "模型就绪，开始处理…"))
            custom = self._get_custom_states()

            segments = []
            total = len(self._file_list)
            for i, path in enumerate(self._file_list):
                try:
                    if step == 1:
                        segs = process_file_step1(path, model)
                        for seg in segs:
                            segments.append((path, seg))
                    elif step == 2:
                        segs, disease = process_file_step2(path, model, custom_states=custom)
                        for seg in segs:
                            segments.append((path, seg))
                    else:
                        segs, disease = process_file_step3(path, model, custom_states=custom)
                        for seg in segs:
                            segments.append((path, seg))
                except Exception as e:
                    print(f"[警告] 处理 {path} 时出错: {e}")

                pct = (i + 1) / total * 100
                self._result_queue.put(("progress", pct))
                self._result_queue.put(("status", f"处理中… {i+1}/{total}"))

            self._result_queue.put(("done", (step, segments)))
        except Exception as e:
            self._result_queue.put(("error", str(e)))

    def _poll_queue(self):
        try:
            while True:
                msg = self._result_queue.get_nowait()
                kind, data = msg
                if kind == "status":
                    self._status_var.set(data)
                elif kind == "progress":
                    self._progress_var.set(data)
                elif kind == "done":
                    self._on_processing_done(*data)
                elif kind == "error":
                    messagebox.showerror("处理出错", data)
                    self._run_btn.configure(state=tk.NORMAL)
                    self._status_var.set("处理失败")
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _on_processing_done(self, step, segments):
        self._segments = segments
        self._step_done = step
        self._run_btn.configure(state=tk.NORMAL)
        self._status_var.set(f"完成！共 {len(segments)} 个片段（Step{step}）")
        self._progress_var.set(100)
        # 刷新列表
        self._seg_listbox.delete(0, tk.END)
        for idx, (path, seg) in enumerate(segments):
            fname     = os.path.basename(path)
            cycles    = seg.get("cycle_count", "?")
            disease   = seg.get("disease_key", "?")
            dur_s     = len(seg["signal"]) / seg["fs"]
            label     = f"#{idx+1:03d} | {fname} | {disease} | {cycles}周期 | {dur_s:.1f}s"
            self._seg_listbox.insert(tk.END, label)

    # ────────────────────────────────────────────────────────────
    #  片段选择 & 波形展示
    # ────────────────────────────────────────────────────────────
    def _on_seg_select(self, event):
        sel = self._seg_listbox.curselection()
        if not sel:
            return
        if not self._segments:
            return
        idx = sel[0]
        if idx >= len(self._segments):
            return
        self._sel_seg_idx = idx
        path, seg = self._segments[idx]
        # 更新右侧疾病参考信息
        disease = seg.get("disease_key")
        self._update_disease_info(disease)
        self._draw_segment(path, seg)

    def _update_disease_info(self, disease_key):
        """更新中间栏的疾病参考文本，并同步手动模式默认勾选。"""
        info = DISEASE_INFO.get(disease_key) if disease_key else None
        self._info_text.configure(state=tk.NORMAL)
        self._info_text.delete("1.0", tk.END)
        if info:
            lines = [
                f"疾病：{info['name']}",
                "",
                f"杂音期相：{info['murmur_phase']}",
                "",
                f"杂音性质：",
                f"  {info['murmur_type']}",
                "",
                f"频率范围：",
                f"  {info['freq_range']}",
                "",
                f"额外心音：",
            ]
            for line in lines:
                self._info_text.insert(tk.END, line + "\n")
            for extra_line in info['extra_sounds'].split("\n"):
                self._info_text.insert(tk.END, f"  {extra_line}\n")
            self._info_text.insert(tk.END, "\n")
            self._info_text.insert(tk.END, f"最佳听诊位：\n  {info['best_position']}\n")
            self._info_text.insert(tk.END, "\n")
            ks = info["keep_states"]
            phase_map = {1: "S1", 2: "收缩期", 3: "S2", 4: "舒张期"}
            keep_str = "、".join(phase_map[s] for s in ks)
            self._info_text.insert(tk.END, f"[自动保留区段]\n  {keep_str}\n")
            # 同步手动默认勾选
            if not self._manual_mode.get():
                self._keep_s1.set(1 in ks)
                self._keep_sys.set(2 in ks)
                self._keep_s2.set(3 in ks)
                self._keep_dia.set(4 in ks)
        else:
            self._info_text.insert(tk.END, "未识别到疾病标签。\n\n"
                "请确保文件名或上级目录名中包含\n"
                "AS / AR / MR / MS / MVP / N\n"
                "（大小写不限）。")
        self._info_text.configure(state=tk.DISABLED)

    def _draw_segment(self, path, seg):
        step = getattr(self, "_step_done", self._step_var.get())

        for ax in self._axes:
            ax.cla()
            ax.set_visible(True)
            ax.set_facecolor("#fafafa")

        sig    = seg["signal"]
        states = seg["states"]
        fs     = seg["fs"]
        t      = np.linspace(0, (len(sig) - 1) / fs, len(sig))

        # ── 对原始信号做显示预处理（Hampel + 陷波 + 带通）────────
        _, sig_disp = preprocess_for_display(
            sig, fs,
            lowcut=LOWCUT, highcut=HIGHCUT,
            filter_order=FILTER_ORDER,
            clip_pct=CLIP_PERCENTILE,
            hampel_win=max(4, int(fs * 0.002)),
            hampel_sigma=HAMPEL_SIGMA,
            notch_freqs=NOTCH_FREQS,
            apply_wavelet_denoise=False,   # 显示层不做小波去噪，保留杂音形态
        )
        scale = np.percentile(np.abs(sig_disp), 99) or 1.0
        sig_n = np.clip(sig_disp / scale, -1, 1)

        # --- 子图1：滤波后波形 + 状态着色 ---
        ax0 = self._axes[0]
        _draw_state_shading(ax0, t, states, alpha=0.25)
        ax0.plot(t, sig_n, color="#1a73e8", linewidth=0.55, alpha=0.92)
        ax0.set_ylabel("归一化幅度")
        title   = os.path.basename(path)
        disease = seg.get("disease_key", "N/A")

        # ── 额外心音候选标注（S3/S4 规则检测）────────────────────
        extra = detect_extra_sounds(sig, states, fs)
        s3_hits = extra["s3_candidates"]
        s4_hits = extra["s4_candidates"]
        for (i0, i1, ratio) in s3_hits:
            if i1 < len(t):
                ax0.axvspan(t[i0], t[i1], color="#a855f7", alpha=0.45, zorder=4)
                ax0.text(t[(i0 + i1) // 2], 1.15, f"S3?\n({ratio:.1f}x)",
                         ha="center", va="bottom", fontsize=6.5,
                         color="#7c3aed", fontweight="bold")
        for (i0, i1, ratio) in s4_hits:
            if i1 < len(t):
                ax0.axvspan(t[i0], t[i1], color="#06b6d4", alpha=0.45, zorder=4)
                ax0.text(t[(i0 + i1) // 2], 1.15, f"S4?\n({ratio:.1f}x)",
                         ha="center", va="bottom", fontsize=6.5,
                         color="#0284c7", fontweight="bold")

        extra_note = ""
        if s3_hits:
            extra_note += f"  ⚠ 检测到 {len(s3_hits)} 处 S3 候选"
        if s4_hits:
            extra_note += f"  ⚠ 检测到 {len(s4_hits)} 处 S4 候选"
        ax0.set_title(
            f"{title}  |  疾病: {disease}  |  {seg.get('cycle_count','?')}个完整周期  —  Step{step}{extra_note}",
            fontsize=9,
        )
        ax0.set_ylim(-1.2, 1.55)
        ax0.grid(True, alpha=0.3, linewidth=0.4)
        _add_state_legend(ax0)

        # --- 子图2：Step2 疾病掩膜信号（含S1/S2边界，滤波后）---
        ax1 = self._axes[1]
        if step >= 2 and "masked_signal" in seg:
            msig = seg["masked_signal"]
            # 对掩膜信号做带通滤波（跳过置零区域的影响直接用scale归一化）
            _, msig_disp = preprocess_for_display(
                msig, fs,
                lowcut=LOWCUT, highcut=HIGHCUT,
                filter_order=FILTER_ORDER,
                clip_pct=99.5,
                hampel_win=max(4, int(fs * 0.002)),
                hampel_sigma=HAMPEL_SIGMA,
                notch_freqs=NOTCH_FREQS,
                apply_wavelet_denoise=False,
            )
            msig_n = np.clip(msig_disp / scale, -1, 1)
            _draw_state_shading(ax1, t, states, alpha=0.2)
            ax1.plot(t, msig_n, color="#2a9d5c", linewidth=0.55, alpha=0.92)
            ax1.set_ylabel("留存杂音区段")
            ax1.set_title(
                f"Step2：保留 {_murmur_state_names(disease)} 区段（含S1/S2边界）",
                fontsize=9
            )
            ax1.set_ylim(-1.2, 1.2)
            ax1.grid(True, alpha=0.3, linewidth=0.4)
        else:
            ax1.text(0.5, 0.5, "Step2 未运行", ha="center", va="center",
                     transform=ax1.transAxes, color="#aaa")

        # --- 子图3：Step3 纯杂音（剔除S1/S2，滤波后）---
        ax2 = self._axes[2]
        if step >= 3 and "murmur_only_signal" in seg:
            msig2 = seg["murmur_only_signal"]
            _, msig2_disp = preprocess_for_display(
                msig2, fs,
                lowcut=LOWCUT, highcut=HIGHCUT,
                filter_order=FILTER_ORDER,
                clip_pct=99.5,
                hampel_win=max(4, int(fs * 0.002)),
                hampel_sigma=HAMPEL_SIGMA,
                notch_freqs=NOTCH_FREQS,
                apply_wavelet_denoise=False,
            )
            msig2_n = np.clip(msig2_disp / scale, -1, 1)
            _draw_state_shading(ax2, t, states, alpha=0.15)
            ax2.plot(t, msig2_n, color="#e05c7a", linewidth=0.55, alpha=0.92)
            ax2.set_ylabel("纯杂音")
            ax2.set_title("Step3：剔除 S1/S2 后的纯杂音区段", fontsize=9)
            ax2.set_ylim(-1.2, 1.2)
            ax2.set_xlabel("时间 (s)")
            ax2.grid(True, alpha=0.3, linewidth=0.4)
        else:
            if step < 3:
                ax2.text(0.5, 0.5, "Step3 未运行", ha="center", va="center",
                         transform=ax2.transAxes, color="#aaa")

        if step == 1:
            ax1.set_visible(False)
            ax2.set_visible(False)
            # Step1 只有一个子图，让它充满
            self._axes[0].set_xlabel("时间 (s)")
        elif step == 2:
            ax1.set_visible(True)
            ax2.set_visible(False)
            ax1.set_xlabel("时间 (s)")
        else:
            ax1.set_visible(True)
            ax2.set_visible(True)

        self._fig.tight_layout(pad=1.2)
        self._canvas.draw_idle()

    # ────────────────────────────────────────────────────────────
    #  批量保存
    # ────────────────────────────────────────────────────────────
    def _batch_save(self):
        if not self._segments:
            messagebox.showwarning("提示", "尚无处理结果，请先运行处理！")
            return

        step = self._save_step_var.get()

        # 检查所选 step 的数据是否已生成
        step_key = {1: None, 2: "masked_signal", 3: "murmur_only_signal"}
        required_key = step_key[step]
        if required_key and not self._segments[0][1].get(required_key) is not None:
            # 更精确：检查第一个片段中是否存在该键
            if required_key not in self._segments[0][1]:
                step_name = {1: "Step1", 2: "Step2", 3: "Step3"}[step]
                messagebox.showwarning(
                    "数据不存在",
                    f"当前片段尚未完成 {step_name} 处理，请先运行对应步骤后再保存。"
                )
                return

        save_dir = self._save_folder.get().strip()
        if not save_dir:
            save_dir = filedialog.askdirectory(title="选择保存文件夹")
            if not save_dir:
                return
            self._save_folder.set(save_dir)

        os.makedirs(save_dir, exist_ok=True)

        suffix_map = {1: "seg", 2: "masked", 3: "murmur"}
        suffix = suffix_map[step]
        saved = 0
        skipped = 0

        for idx, (path, seg) in enumerate(self._segments):
            base    = os.path.splitext(os.path.basename(path))[0]
            disease = seg.get("disease_key", "UNK") or "UNK"
            cycles  = seg.get("cycle_count", 0)
            seg_idx = idx + 1

            if step == 1:
                out_sig = seg["signal"]
            elif step == 2:
                if "masked_signal" not in seg:
                    skipped += 1
                    continue
                out_sig = seg["masked_signal"]
            else:
                if "murmur_only_signal" not in seg:
                    skipped += 1
                    continue
                out_sig = seg["murmur_only_signal"]

            out_name = f"{base}_{disease}_{suffix}_{seg_idx:03d}_{cycles}cyc.wav"
            out_path = os.path.join(save_dir, out_name)
            _save_wav(out_sig, seg["fs"], out_path)
            saved += 1

        # 保存 metadata JSON
        meta = []
        for idx, (path, seg) in enumerate(self._segments):
            if step == 2 and "masked_signal" not in seg:
                continue
            if step == 3 and "murmur_only_signal" not in seg:
                continue
            meta.append({
                "index":        idx + 1,
                "source_file":  path,
                "disease":      seg.get("disease_key"),
                "cycle_count":  seg.get("cycle_count"),
                "fs":           seg["fs"],
                "duration_s":   round(len(seg["signal"]) / seg["fs"], 3),
                "step":         step,
            })
        meta_path = os.path.join(save_dir, "segments_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        msg = f"已保存 {saved} 个片段（Step{step}）至：\n{save_dir}"
        if skipped:
            msg += f"\n\n（{skipped} 个片段因未完成 Step{step} 处理而跳过）"
        messagebox.showinfo("保存完成", msg)


# ================================================================
#  辅助绘图函数
# ================================================================
def _draw_state_shading(ax, t, states, alpha=0.2):
    """在坐标轴上按 Springer 状态绘制背景色块。"""
    states = np.asarray(states, dtype=int)
    changes = np.where(np.diff(states) != 0)[0] + 1
    starts  = np.concatenate([[0], changes])
    ends    = np.concatenate([changes, [len(states)]])
    for s, e in zip(starts, ends):
        if e > len(t):
            e = len(t)
        st = states[s]
        color = STATE_COLORS.get(st, "#cccccc")
        ax.axvspan(t[s], t[min(e, len(t) - 1)], color=color, alpha=alpha)


def _add_state_legend(ax):
    """添加状态图例。"""
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(color=STATE_COLORS[k], alpha=0.5, label=STATE_LABELS[k])
        for k in sorted(STATE_COLORS)
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.7)


def _murmur_state_names(disease_key):
    """返回疾病对应的杂音区段中文名称。"""
    mapping = {
        "AS":  "收缩期",
        "AR":  "舒张期",
        "MR":  "收缩期",
        "MS":  "舒张期",
        "MVP": "收缩期",
        "N":   "收缩期 + 舒张期",
    }
    return mapping.get(disease_key, "收缩期 + 舒张期")


def _save_wav(signal, fs, path):
    """将 float64 信号保存为 16-bit WAV。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arr = np.clip(signal, -1.0, 1.0)
    arr = (arr * 32767).astype(np.int16)
    wav_io.write(path, int(fs), arr)


# ================================================================
#  入口
# ================================================================
if __name__ == "__main__":
    app = PCGProcessorApp()
    app.mainloop()