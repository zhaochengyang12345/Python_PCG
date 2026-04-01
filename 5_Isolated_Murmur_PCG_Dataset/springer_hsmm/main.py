"""
Springer HSMM 心音分割算法的交互式查看器。
功能：
    - 从指定文件夹加载 WAV 和 CSV 文件（CSV 需包含 PCG 列）
    - 对每个文件进行预处理（Hampel + 陷波 + 带通 + 归一化）
    - 可选：使用 Springer HSMM 模型进行分割预测
    - 显示预处理后的波形和分割结果（S1 / 收缩期 / S2 / 舒张期）
    - 导航：上一个 / 下一个文件，键盘 ← → 键
    - 数据筛选：为每条记录标注 ✓入组 / ✗排除 / ?待定（键盘 1/2/3 快捷键）
    - 批量导出：将指定类别的文件复制到目标文件夹
    - 标注持久化：自动保存/加载 .label_annotations.json
"""
# -*- coding: utf-8 -*-
# ================================================================
#                       用户配置区
# ================================================================

# ── 训练数据 ─────────────────────────────────────────────────
# 训练用 .mat 文件路径（相对于本脚本）
TRAIN_DATA_PATH = 'data/example_data.mat'
# 参与训练的录音数量
TRAIN_COUNT     = 5
# 训练 .mat 数据的采样率（Hz）
TRAIN_AUDIO_FS  = 1000

# ── 输入文件 ──────────────────────────────────────────────────
# 当前采集约定：听诊器导出的 WAV 为 4000 Hz，CSV 为 8000 Hz。
# 为避免文件头或时间列不一致带来的歧义，这里按文件类型固定采样率。
WAV_INPUT_FS    = 4000
CSV_INPUT_FS    = 8000
# CSV 文件中 PCG 信号所在的列名
CSV_PCG_COLUMN  = 'pcg'

# ── 分割设置 ──────────────────────────────────────────────────
# 设为 False 可跳过 Springer 分割（隐藏折线、允许短文件）
ENABLE_SEGMENTATION = True
# 特征提取帧率 — 除非重新训练模型，否则不要修改
FEATURES_FS     = 50
# 内部处理采样率：信号在进入 Springer 流水线前重采样至此。
# 保持 1000 Hz（与训练一致）可获得约8× 速度提升。
PROCESSING_FS   = 1000

# ── 显示预处理参数 ────────────────────────────────────────────
LOWCUT          = 20.0    # 带通滤波低截止频率（Hz）
HIGHCUT         = 400.0   # 带通滤波高截止频率（Hz）— 心音主要能量在 20–400 Hz
FILTER_ORDER    = 4       # Butterworth 滤波器阶数
CLIP_PERCENTILE = 99.0    # 百分位截断阈值（去除极端伪影）
HAMPEL_WIN      = 40      # Hampel 半窗长（采样点数）
HAMPEL_SIGMA    = 2.5     # Hampel 离群点判定阈值（MAD 的倍数）
NOTCH_FREQS     = [50.0, 100.0]   # 陷波频率列表（Hz）；设为 [] 可关闭

# ── 小波去噪参数（针对瓣膜疾病高杂音数据）────────────────────
WAVELET_DENOISE           = True   # 启用小波软阈值去噪（公开数据库可设 False）
WAVELET_NAME              = 'db6'  # 小波基：db6 与 S1/S2 形态最匹配
WAVELET_LEVEL             = 5      # 分解层数（1000 Hz 采样率下推荐 5）
WAVELET_THRESHOLD_FACTOR  = 0.4   # 阈值系数：0.3 保守 / 0.4 均衡 / 0.6 激进

# ── 界面显示参数 ──────────────────────────────────────────────
VIEW_WINDOW     = 3    # 初始可见时间窗宽度（秒）
MAX_DISPLAY_DURATION = 0  # 截断信号至此长度（秒）；0 表示显示完整信号
LINE_WIDTH      = 0.5     # 波形线宽（越细在密集信号上越清晰）
LINE_ALPHA      = 0.90
DISPLAY_BINS    = 4000    # 绘图前抜取至此点数（保持线条纤细）
COLOR_WAV       = "#1a73e8"   # WAV 文件波形颜色（深蓝，白底可见）
COLOR_CSV       = "#2a9d5c"   # CSV 文件波形颜色（深绿，白底可见）
WINDOW_SIZE     = "1200x720"  # 初始窗口大小

# ── 输出设置 ──────────────────────────────────────────────────
SAVE_PLOT       = False   # 是否将分割结果图保存为 PNG 至 output/ 子目录
OUTPUT_SUBDIR   = 'output'

# ================================================================
import os, sys, json, shutil, warnings
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import kurtosis as sp_kurtosis
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib import rcParams

warnings.filterwarnings("ignore")

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from utils.preprocessing import (load_mat_data, load_wav_file, load_csv_file,
                                  scan_folder, preprocess_for_display,
                                  resample_to_fs)
from models.segmentation_algorithm import train_segmentation_algorithm, predict_segmentation

# 各状态颜色（白色背景下可见）
STATE_COLORS = {1: '#e05c7a', 2: '#fab387', 3: '#f5a623', 4: '#89b4fa'}
STATE_LABELS = {1: 'S1', 2: '收缩期', 3: 'S2', 4: '舒张期'}


# ════════════════════════════════════════════════════════════════
#  训练模型
# ════════════════════════════════════════════════════════════════

print("=" * 60)
print("Springer HSMM 心音分割算法")
print("=" * 60)

if ENABLE_SEGMENTATION:
    print(f"\n正在使用 {TRAIN_DATA_PATH} 中的 {TRAIN_COUNT} 条录音进行训练...")
    train_data_path = os.path.join(_SCRIPT_DIR, TRAIN_DATA_PATH)
    if not os.path.isfile(train_data_path):
        print(f"ERROR: Training data not found: {train_data_path}")
        sys.exit(1)
    audio_data, annotations = load_mat_data(train_data_path)
    _info = train_segmentation_algorithm(
        audio_data[:TRAIN_COUNT], annotations[:TRAIN_COUNT],
        FEATURES_FS, TRAIN_AUDIO_FS
    )
    print("训练完成。\n")
else:
    print("分割功能已禁用 — 跳过训练。\n")
    _info = None


# ════════════════════════════════════════════════════════════════
#  分割查看器
# ════════════════════════════════════════════════════════════════

class SegmentationViewer(tk.Tk):
    """
    带 Springer 分割叠加层的交互式 PCG 波形查看器。

    每个文件单面板显示：
      ① PCG 波形（经 Hampel + 陷波 + 带通 + 归一化预处理后）
      ② 分割折线（S1 / 收缩期 / S2 / 舒张期，右 Y 轴叠加）

    导航：点击“上一个 / 下一个”按鈕或按键盘 ← → 键。
    """

    # 标注类别常量
    LABEL_ACCEPT  = '✓'   # 入组
    LABEL_REJECT  = '✗'   # 排除
    LABEL_UNSURE  = '?'   # 待定
    LABEL_COLORS  = {'✓': '#2e7d32', '✗': '#c62828', '?': '#e65100', None: '#888888'}
    LABEL_NAMES   = {'✓': '入组', '✗': '排除', '?': '待定', None: '未标注'}

    def __init__(self, model_info):
        super().__init__()
        self.model_info  = model_info
        self.audio_files: list = []
        self.current_idx: int  = 0
        self.cache: dict       = {}   # 路径 → (time_axis, display_sig, pred_states, fs, rel_path)
        self._duration          = 0.0
        self.labels: dict      = {}   # 路径 → '✓' / '✗' / '?' / None
        self.quality_cache: dict = {}  # 路径 → quality metrics dict
        self.base_folder: str  = ''
        self._list_visible: bool       = False
        self._list_btns: list          = []    # 文件列表中的按钮控件
        self._list_canvas: tk.Canvas   = None  # type: ignore
        self._list_panel: tk.Frame     = None  # type: ignore

        self.title("心音波形分段查看器")
        self.geometry(WINDOW_SIZE)
        self.configure(bg="#f8f9fa")
        self.resizable(True, True)
        self._build_ui()

    # ── 界面构建 ────────────────────────────────────────────
    def _build_ui(self):
        btn = dict(bg="#e0e0e0", fg="#222222",
                   activebackground="#c8c8c8", activeforeground="#222222",
                   relief="flat", font=("Segoe UI", 10),
                   padx=12, pady=6, cursor="hand2")

        # 顶部工具栏
        top = tk.Frame(self, bg="#f0f0f0", pady=6)
        top.pack(fill="x", side="top")

        tk.Button(top, text="📂  选择文件夹",
                  command=self.open_folder, **btn).pack(side="left", padx=(12, 6))
        self.lbl_count = tk.Label(top, text="未加载文件",
                                  bg="#f0f0f0", fg="#555555",
                                  font=("Segoe UI", 9))
        self.lbl_count.pack(side="left", padx=8)

        self._btn_list_toggle = tk.Button(
            top, text="📋  文件列表",
            command=self._toggle_list_panel,
            bg="#e0e0e0", fg="#222222",
            activebackground="#c8c8c8", activeforeground="#222222",
            relief="flat", font=("Segoe UI", 10),
            padx=12, pady=6, cursor="hand2")
        self._btn_list_toggle.pack(side="right", padx=(6, 12))

        # 图形区域 + 右侧文件列表面板（水平排列）
        self._center_frame = tk.Frame(self, bg="#f8f9fa")
        self._center_frame.pack(fill="both", expand=True, padx=(16, 0), pady=(8, 4))

        fig_frame = tk.Frame(self._center_frame, bg="#f8f9fa")
        fig_frame.pack(side="left", fill="both", expand=True)

        # 右侧文件列表面板（初始隐藏）
        self._list_panel = tk.Frame(self._center_frame, bg="#f0f2f5", width=180)
        self._list_panel.pack_propagate(False)
        # 不在此处pack，由 _toggle_list_panel 控制
        self._build_list_panel(self._list_panel)

        self.fig, self.ax_wave = plt.subplots(1, 1, figsize=(11, 6))
        self.ax_step = self.ax_wave.twinx()   # 右轴折线（分割状态）
        self.fig.patch.set_facecolor("#ffffff")
        self.ax_wave.set_facecolor("#ffffff")
        self._style_axes()

        self.canvas = FigureCanvasTkAgg(self.fig, master=fig_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        nav_bar = NavigationToolbar2Tk(self.canvas, fig_frame)
        nav_bar.update()
        nav_bar.config(bg="#f0f0f0")

        # 时间滚动条
        sc_frame = tk.Frame(self, bg="#f0f0f0", pady=4)
        sc_frame.pack(fill="x", padx=16)

        tk.Label(sc_frame, text="时间位置", bg="#f0f0f0", fg="#555555",
                 font=("Segoe UI", 8)).pack(side="left", padx=(0, 6))
        self.scrollbar = tk.Scale(
            sc_frame, orient="horizontal",
            from_=0.0, to=0.0, resolution=0.1,
            bg="#e0e0e0", fg="#222222", troughcolor="#f0f0f0",
            highlightthickness=0, bd=0, sliderrelief="flat",
            activebackground="#1a73e8", font=("Segoe UI", 8),
            command=self._on_scroll)
        self.scrollbar.pack(fill="x", expand=True, side="left")

        # ── 质控指标栏 ─────────────────────────────────────────
        qa_outer = tk.Frame(self, bg="#f4f6fb", pady=4)
        qa_outer.pack(fill="x", padx=16)
        tk.Label(qa_outer, text="质控指标：", bg="#f4f6fb", fg="#444",
                 font=("Segoe UI", 8, "bold")).pack(side="left", padx=(0, 6))

        # 名称 → 参考标准描述
        _QA_REFS = {
            "周期性":  "参考 ≥0.40",
            "时长":   "参考 ≥8 s",
            "削波率":  "参考 <0.5%",
            "峰度":   "参考 4∼50",
            "SNR":    "参考 ≥10 dB",
            "频带占比": "参考 ≥60%",
            "心率":   "参考 40–150 bpm",
            "节律性":  "参考 CV<20%",
        }
        self._qa_badges: dict = {}   # 指标名 → (外框Frame, 值Label)
        for name, ref_str in _QA_REFS.items():
            cell = tk.Frame(qa_outer, bg="#eeeeee", padx=0, pady=0)
            cell.pack(side="left", padx=3)
            tk.Label(cell, text=name, bg="#eeeeee", fg="#555",
                     font=("Segoe UI", 7, "bold")).pack(side="top", padx=8, pady=(3, 0))
            val_lbl = tk.Label(cell, text="—", bg="#eeeeee", fg="#888",
                               font=("Segoe UI", 9, "bold"))
            val_lbl.pack(side="top", padx=8, pady=(1, 0))
            tk.Label(cell, text=ref_str, bg="#eeeeee", fg="#999",
                     font=("Segoe UI", 6)).pack(side="top", padx=8, pady=(0, 3))
            self._qa_badges[name] = (cell, val_lbl)

        # ── 标注栏 ──────────────────────────────────────────
        label_bar = tk.Frame(self, bg="#eef2f7", pady=6)
        label_bar.pack(fill="x", side="bottom")

        tk.Label(label_bar, text="标注：", bg="#eef2f7", fg="#333333",
                 font=("Segoe UI", 10, "bold")).pack(side="left", padx=(16, 4))

        self._btn_accept = tk.Button(
            label_bar, text="✓  入组",
            command=lambda: self._set_label(self.LABEL_ACCEPT),
            bg="#e8f5e9", fg="#2e7d32", activebackground="#2e7d32",
            activeforeground="#ffffff", relief="flat",
            font=("Segoe UI", 10, "bold"), padx=14, pady=5, cursor="hand2")
        self._btn_accept.pack(side="left", padx=4)

        self._btn_reject = tk.Button(
            label_bar, text="✗  排除",
            command=lambda: self._set_label(self.LABEL_REJECT),
            bg="#ffebee", fg="#c62828", activebackground="#c62828",
            activeforeground="#ffffff", relief="flat",
            font=("Segoe UI", 10, "bold"), padx=14, pady=5, cursor="hand2")
        self._btn_reject.pack(side="left", padx=4)

        self._btn_unsure = tk.Button(
            label_bar, text="?  待定",
            command=lambda: self._set_label(self.LABEL_UNSURE),
            bg="#fff3e0", fg="#e65100", activebackground="#e65100",
            activeforeground="#ffffff", relief="flat",
            font=("Segoe UI", 10, "bold"), padx=14, pady=5, cursor="hand2")
        self._btn_unsure.pack(side="left", padx=4)

        self._lbl_cur_label = tk.Label(
            label_bar, text="当前：未标注",
            bg="#eef2f7", fg="#888888",
            font=("Segoe UI", 10, "italic"))
        self._lbl_cur_label.pack(side="left", padx=16)

        self._lbl_stats = tk.Label(
            label_bar, text="",
            bg="#eef2f7", fg="#555555",
            font=("Segoe UI", 9))
        self._lbl_stats.pack(side="left", padx=8)

        tk.Button(
            label_bar, text="📤  批量导出…",
            command=self.export_dialog,
            bg="#e3f2fd", fg="#1565c0", activebackground="#1565c0",
            activeforeground="#ffffff", relief="flat",
            font=("Segoe UI", 10), padx=14, pady=5, cursor="hand2"
        ).pack(side="right", padx=(4, 16))

        tk.Button(
            label_bar, text="💾  保存标注",
            command=self._save_labels,
            bg="#e8f5e9", fg="#1b5e20", activebackground="#1b5e20",
            activeforeground="#ffffff", relief="flat",
            font=("Segoe UI", 10), padx=14, pady=5, cursor="hand2"
        ).pack(side="right", padx=4)

        # 底部导航栏
        bot = tk.Frame(self, bg="#f8f9fa", pady=8)
        bot.pack(fill="x", side="bottom")

        tk.Button(bot, text="◀  上一个", command=self.show_prev, **btn).pack(side="left", padx=(16, 6))
        tk.Button(bot, text="下一个  ▶", command=self.show_next, **btn).pack(side="left", padx=6)

        self.lbl_index = tk.Label(bot, text="",
                                  bg="#f8f9fa", fg="#555555",
                                  font=("Segoe UI", 9))
        self.lbl_index.pack(side="left", padx=12)

        self.bind("<Left>",  lambda e: self.show_prev())
        self.bind("<Right>", lambda e: self.show_next())
        # 快捷键：1=入组  2=排除  3=待定
        self.bind("1", lambda e: self._set_label(self.LABEL_ACCEPT))
        self.bind("2", lambda e: self._set_label(self.LABEL_REJECT))
        self.bind("3", lambda e: self._set_label(self.LABEL_UNSURE))

    def _style_axes(self):
        for ax in (self.ax_wave, self.ax_step):
            ax.tick_params(colors="#333333", labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#bbbbbb")
            ax.xaxis.label.set_color("#333333")
            ax.yaxis.label.set_color("#333333")

    # ── 滚动条 ───────────────────────────────────────────────
    def _on_scroll(self, val):
        if self._duration <= 0:
            return
        start = float(val)
        end   = min(start + VIEW_WINDOW, self._duration)
        self.ax_wave.set_xlim(start, end)
        self.canvas.draw_idle()

    # ── 文件夹加载 ─────────────────────────────────────────────
    def open_folder(self):
        folder = filedialog.askdirectory(title="选择包含 PCG 录音的文件夹 (.wav / .csv)")
        if not folder:
            return

        files = scan_folder(folder)
        if not files:
            messagebox.showwarning("未找到文件",
                                   f"所选文件夹下未找到 .wav 或 .csv 文件\n路径：{folder}")
            return

        self.base_folder = folder
        self.audio_files = files
        self.current_idx = 0
        self.cache.clear()
        self.quality_cache.clear()
        self.labels.clear()
        self._load_labels()   # 尝试读取已有标注
        errors = []

        for i, path in enumerate(files):
            fname = os.path.basename(path)
            self.lbl_count.config(text=f"处理中 {i+1}/{len(files)}: {fname}")
            self.update()
            try:
                self.cache[path] = self._process_file(path, folder)
            except Exception as exc:
                errors.append(f"{fname}: {exc}")
                self.cache[path] = None

        ok = len(files) - len(errors)
        self.lbl_count.config(text=f"共 {len(files)} 个文件（成功 {ok} / 失败 {len(errors)}）")

        if errors:
            messagebox.showwarning(
                "部分文件处理失败",
                "以下文件无法处理，已跳过：\n\n" + "\n".join(errors[:10]) +
                (f"\n...共 {len(errors)} 个" if len(errors) > 10 else ""))

        # 跳转到第一个有效文件
        for i, p in enumerate(files):
            if self.cache.get(p) is not None:
                self.current_idx = i
                break

        self.plot_current()

    def _process_file(self, path, base_folder):
        """加载文件 → 显示预处理 → 分割（可选）。返回缓存条目。"""
        ext = os.path.splitext(path)[1].lower()

        if ext == ".wav":
            # 按当前采集约定固定为 4000 Hz；若文件头不一致，仅作提示。
            raw, native_fs = load_wav_file(path)
            fs = WAV_INPUT_FS
            status = "OK" if native_fs == WAV_INPUT_FS else "MISMATCH"
            print(
                f"[WAV FS] {status} | file={os.path.basename(path)} | "
                f"actual={native_fs} Hz | configured={WAV_INPUT_FS} Hz"
            )
            if native_fs != WAV_INPUT_FS:
                print(
                    f"WARNING: WAV header fs={native_fs} Hz, "
                    f"but configured input fs={WAV_INPUT_FS} Hz for {path}"
                )
        else:
            # 按当前采集约定固定为 8000 Hz；忽略 CSV 时间列自动推断结果。
            raw, detected_fs = load_csv_file(path, CSV_PCG_COLUMN)
            fs = CSV_INPUT_FS
            detected_text = f"{detected_fs} Hz" if detected_fs else "unknown"
            status = "OK" if detected_fs == CSV_INPUT_FS else "MISMATCH"
            if detected_fs is None:
                status = "UNKNOWN"
            print(
                f"[CSV FS] {status} | file={os.path.basename(path)} | "
                f"detected={detected_text} | configured={CSV_INPUT_FS} Hz"
            )
            if detected_fs and detected_fs != CSV_INPUT_FS:
                print(
                    f"WARNING: CSV detected fs={detected_fs} Hz, "
                    f"but configured input fs={CSV_INPUT_FS} Hz for {path}"
                )

        # 信号长度检查：分割至少需要 2 秒
        if ENABLE_SEGMENTATION and len(raw) < fs * 2:
            raise ValueError(
                f"信号过短（{len(raw)/fs:.2f} s，分割最少需要 2.0 s；"
                f"可将 ENABLE_SEGMENTATION 设为 False 以查看短文件）")

        # 按 MAX_DISPLAY_DURATION 截断信号（若已设置）
        if MAX_DISPLAY_DURATION > 0:
            raw = raw[:int(MAX_DISPLAY_DURATION * fs)]

        # 显示预处理（Hampel + 陷波 + 小波去噪 + 带通 + 归一化）
        time_axis, disp_sig = preprocess_for_display(
            raw, fs,
            lowcut=LOWCUT, highcut=HIGHCUT, filter_order=FILTER_ORDER,
            clip_pct=CLIP_PERCENTILE,
            hampel_win=HAMPEL_WIN, hampel_sigma=HAMPEL_SIGMA,
            notch_freqs=NOTCH_FREQS,
            apply_wavelet_denoise=WAVELET_DENOISE,
            wavelet=WAVELET_NAME,
            wavelet_level=WAVELET_LEVEL,
            wavelet_threshold=WAVELET_THRESHOLD_FACTOR)

        # Springer 分割（可选）
        if ENABLE_SEGMENTATION:
            pred_states = predict_segmentation(
                raw, FEATURES_FS, fs,
                self.model_info['pi_vector'],
                self.model_info['model'],
                self.model_info['total_obs_distribution'],
                processing_fs=PROCESSING_FS)
        else:
            pred_states = None

        rel = os.path.relpath(path, base_folder)

        # 计算质控指标（使用预处理后的显示信号）
        quality = SegmentationViewer._compute_quality(disp_sig, fs, pred_states)
        self.quality_cache[path] = quality

        # 抜取至 DISPLAY_BINS 点，使 plot() 绘出纤细清晰的线条
        n      = len(disp_sig)
        stride = max(1, n // DISPLAY_BINS)
        t_dec  = time_axis[::stride]
        y_dec  = disp_sig[::stride]

        return (time_axis, t_dec, y_dec, pred_states, fs, rel)

    # ── 文件导航 ─────────────────────────────────────────────
    def show_prev(self):
        if not self.audio_files:
            return
        self.current_idx = (self.current_idx - 1) % len(self.audio_files)
        self.plot_current()

    def show_next(self):
        if not self.audio_files:
            return
        self.current_idx = (self.current_idx + 1) % len(self.audio_files)
        self.plot_current()

    # ── 绘图 ────────────────────────────────────────────────────
    def plot_current(self):
        if not self.audio_files:
            return
        path = self.audio_files[self.current_idx]
        cached = self.cache.get(path)
        if cached is None:
            messagebox.showerror("读取失败", f"该文件处理失败，已跳过：\n{path}")
            return

        time_axis, t_dec, y_dec, pred_states, fs, rel = cached
        fname = os.path.basename(path)
        ext   = os.path.splitext(fname)[1].upper()
        color = COLOR_WAV if ext == ".WAV" else COLOR_CSV

        # ── Waveform panel ──────────────────────────────────────
        self.ax_wave.cla()
        self.ax_step.cla()
        self.ax_wave.set_facecolor("#ffffff")
        self.ax_wave.plot(t_dec, y_dec, color=color,
                          linewidth=LINE_WIDTH, alpha=LINE_ALPHA)
        self.ax_wave.set_ylabel("归一化幅度", fontsize=8, color="#333333")
        self.ax_wave.set_ylim(-1.12, 1.12)
        self.ax_wave.axhline(0, color="#cccccc", linewidth=0.4, linestyle="--")
        self.ax_wave.grid(True, color="#e8e8e8", linewidth=0.35, linestyle=":")
        self.ax_wave.set_title(fname, color="#222222",
                                fontsize=11, fontweight="bold", pad=8)

        # 按状态绘制分割折线
        if pred_states is not None:
            ts_seg = np.linspace(0, (len(pred_states) - 1) / fs, len(pred_states))
            changes = np.where(np.diff(pred_states) != 0)[0] + 1
            starts  = np.concatenate([[0], changes])
            ends    = np.concatenate([changes, [len(pred_states)]])
            # 在波形图顶部标注 S1 / S2 文字
            for s, e in zip(starts, ends):
                sv = int(pred_states[s])
                t0, t1 = ts_seg[s], ts_seg[min(e, len(ts_seg)-1)]
                if sv in (1, 3) and t1 - t0 > 0.05:
                    self.ax_wave.text(
                    (t0 + t1) / 2, 0.97, STATE_LABELS[sv],
                    transform=self.ax_wave.get_xaxis_transform(),
                    ha='center', va='top', fontsize=7,
                    color=STATE_COLORS[sv], fontweight='bold', clip_on=True)
            # 分割折线（右轴）
            for i, (s, e) in enumerate(zip(starts, ends)):
                sv  = int(pred_states[s])
                t0  = ts_seg[s]
                t1  = ts_seg[min(e, len(ts_seg) - 1)]
                clr = STATE_COLORS.get(sv, '#888888')
                self.ax_step.plot([t0, t1], [sv, sv],
                                  color=clr, linewidth=2.2, solid_capstyle='butt',
                                  zorder=3)
                if i + 1 < len(starts):
                    sv_next  = int(pred_states[starts[i + 1]])
                    clr_next = STATE_COLORS.get(sv_next, '#888888')
                    self.ax_step.plot([t1, t1], [sv, sv_next],
                                      color=clr_next, linewidth=1.2, linestyle='--', zorder=3)
            self.ax_step.set_yticks([1, 2, 3, 4])
            self.ax_step.set_yticklabels(['S1', 'Sys', 'S2', 'Dia'], fontsize=7)
            self.ax_step.set_ylim(-5, 5)
            self.ax_step.set_xlim(self.ax_wave.get_xlim())
            self.ax_step.grid(False)
        else:
            self.ax_step.set_yticks([])
            self.ax_step.set_visible(False)

        # Annotation
        notch_str = ("/".join(f"{int(f)}" for f in NOTCH_FREQS) + " Hz陷波") if NOTCH_FREQS else "无陷波"
        self.ax_wave.annotate(
            f"格式: {ext}  |  采样率: {fs} Hz  |  时长: {time_axis[-1]:.2f} s  |  "
            f"预处理: Hampel + {notch_str} + 带通 {LOWCUT:.0f}–{HIGHCUT:.0f} Hz",
            xy=(0.01, 0.02), xycoords="axes fraction",
            fontsize=7, color="#888888", ha="left", va="bottom")

        # ── Styling & scrollbar ──────────────────────────────────
        self.ax_wave.set_xlabel("时间 (s)", fontsize=8, color="#333333")
        self._style_axes()
        self._duration = float(time_axis[-1])
        xlim_end = min(VIEW_WINDOW, self._duration)
        self.ax_wave.set_xlim(0, xlim_end)

        scroll_max = max(0.0, self._duration - VIEW_WINDOW)
        self.scrollbar.config(from_=0.0, to=round(scroll_max, 1))
        self.scrollbar.set(0.0)

        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()

        self.lbl_index.config(
            text=f"{self.current_idx + 1} / {len(self.audio_files)}  |  {rel}")
        self.title(f"心音波形分段查看器 — {fname}")
        self._update_label_ui()
        self._update_quality_bar(self.quality_cache.get(path))

        # 可选：保存图片
        if SAVE_PLOT:
            base_folder = self.base_folder or os.path.dirname(self.audio_files[0])
            sub_rel    = os.path.relpath(os.path.dirname(path), base_folder)
            sub_out    = os.path.join(base_folder, OUTPUT_SUBDIR, sub_rel)
            os.makedirs(sub_out, exist_ok=True)
            save_path = os.path.join(sub_out,
                os.path.splitext(fname)[0] + "_segmentation.png")
            self.fig.savefig(save_path, dpi=100, bbox_inches="tight",
                             facecolor="#ffffff")
            print(f"已保存：{save_path}")

    # ── 质控指标计算 ────────────────────────────────────────────
    @staticmethod
    def _compute_quality(sig: np.ndarray, fs: int,
                         pred_states) -> dict:
        """
        计算 PCG 心音信号的质控指标。
        返回 dict，每个指标包含：value(数值)、text(显示文字)、level('good'/'warn'/'bad'/'na')
        """
        result = {}
        eps = 1e-12

        # ─── 0. 预计算 Welch PSD（复用于频带占比）────────────────────
        try:
            _nperseg  = min(len(sig), 2048)
            _freqs, _psd = sp_signal.welch(sig, fs=fs, nperseg=_nperseg)
        except Exception:
            _freqs, _psd = None, None

        # ─── 1. 周期性 / 包络自相关峰值 ────────────────────────────
        # 使用 Shannon 能量包络的归一化自相关，寻找 40-150 bpm 区间内最大峰值。
        # 不受杂音幅度影响，不会把 MR/AS 全收缩期杂音误判为噪声。
        try:
            se       = -(sig ** 2) * np.log(sig ** 2 + eps)   # Shannon 能量
            win_sm   = max(1, int(fs * 0.025))                 # 25ms 平滑窗
            kernel   = np.ones(win_sm) / win_sm
            env      = np.convolve(se, kernel, mode='same')
            env      = np.maximum(env, 0.0)
            env_norm = env - env.mean()
            var_env  = float(np.var(env_norm)) + eps
            # FFT 自相关 O(n log n)，避免 np.correlate full 的 O(n²) 卡顿
            n_ac    = len(env_norm)
            fft_env = np.fft.rfft(env_norm, n=2 * n_ac)
            ac      = np.fft.irfft(fft_env * np.conj(fft_env))[:n_ac]
            ac      = ac / (var_env * n_ac)                    # 归一化到 [-1,1]
            lag_min  = int(fs * 0.40)                          # 150 bpm → 0.40s
            lag_max  = min(int(fs * 1.50), len(ac) - 1)       # 40 bpm  → 1.50s
            if lag_max > lag_min:
                peak_val = float(np.max(ac[lag_min:lag_max + 1]))
                level = 'good' if peak_val >= 0.40 else ('warn' if peak_val >= 0.20 else 'bad')
                result['周期性'] = dict(value=peak_val,
                                       text=f"{peak_val:.2f}", level=level)
            else:
                result['周期性'] = dict(value=None, text='太短', level='na')
        except Exception:
            result['周期性'] = dict(value=None, text='N/A', level='na')

        # ─── 2. 时长 ────────────────────────────────────────────
        try:
            dur = len(sig) / fs
            level = 'good' if dur >= 8 else ('warn' if dur >= 5 else 'bad')
            result['时长'] = dict(value=dur, text=f"{dur:.1f} s", level=level)
        except Exception:
            result['时长'] = dict(value=None, text='N/A', level='na')

        # ─── 3. 削波率 ──────────────────────────────────────────
        # 归一化信号中幅值 ≥ 0.99 的样本比例
        try:
            clip_ratio = np.mean(np.abs(sig) >= 0.99) * 100
            level = 'good' if clip_ratio < 0.5 else ('warn' if clip_ratio < 2.0 else 'bad')
            result['削波率'] = dict(value=clip_ratio,
                                   text=f"{clip_ratio:.2f}%", level=level)
        except Exception:
            result['削波率'] = dict(value=None, text='N/A', level='na')

        # ─── 4. 峰度 ────────────────────────────────────────────
        # 心音信号应呈冲击性，峰度高于纯噪音(=3)
        try:
            kurt = float(sp_kurtosis(sig, fisher=False))  # excess=False → 正态=3
            level = 'good' if 4 <= kurt <= 50 else ('warn' if 3 <= kurt <= 80 else 'bad')
            result['峰度'] = dict(value=kurt, text=f"{kurt:.1f}", level=level)
        except Exception:
            result['峰度'] = dict(value=None, text='N/A', level='na')

        # ─── 5. SNR (dB) —— 基于帧能量分布─────────────────────
        # 注意：MR/AS 全收缩期杂音会拉低此指标，它反映的是「设备增益匹配」而非信号同质量。
        # 低于 5 dB 且周期性正常 → 提示增益小/传感器接触不良，而非直接判为差质。
        try:
            frame_len  = max(1, int(fs * 0.05))
            n_frames   = len(sig) // frame_len
            frames     = sig[:n_frames * frame_len].reshape(n_frames, frame_len)
            rms_frames = np.sqrt(np.mean(frames ** 2, axis=1))
            rms_sorted = np.sort(rms_frames)
            noise_rms  = np.mean(rms_sorted[:max(1, int(n_frames * 0.30))]) + eps
            sig_rms    = np.mean(rms_sorted[int(n_frames * 0.60):]) + eps
            snr_db     = 20 * np.log10(sig_rms / noise_rms)
            level = 'good' if snr_db >= 10 else ('warn' if snr_db >= 5 else 'bad')
            result['SNR'] = dict(value=snr_db, text=f"{snr_db:.1f} dB", level=level)
        except Exception:
            result['SNR'] = dict(value=None, text='N/A', level='na')

        # ─── 6. 心率 BPM（需 Springer 分割）──────────────────────
        # 仅在分割结果可用时计算
        try:
            if pred_states is not None and len(pred_states) > 0:
                states = np.array(pred_states, dtype=int)
                # S1 onset: 前一帧不是1，当前帧是1
                s1_onsets = np.where((states[1:] == 1) & (states[:-1] != 1))[0] + 1
                if len(s1_onsets) >= 2:
                    ts_s1 = s1_onsets / fs
                    rr    = np.diff(ts_s1)          # R-R 间期（秒）
                    hr    = 60.0 / np.mean(rr)
                    level = 'good' if 50 <= hr <= 120 else ('warn' if 40 <= hr <= 150 else 'bad')
                    result['心率'] = dict(value=hr, text=f"{hr:.0f} bpm", level=level)
                else:
                    result['心率'] = dict(value=None,
                                         text='检测不足', level='na')
            else:
                result['心率'] = dict(value=None, text='无分割', level='na')
        except Exception:
            result['心率'] = dict(value=None, text='N/A', level='na')

        # ─── 7. 节律规律性（需 Springer 分割）────────────────────
        # RR 间期的变异系数 CV = std/mean，越小越规律
        try:
            if pred_states is not None and len(pred_states) > 0:
                states = np.array(pred_states, dtype=int)
                s1_onsets = np.where((states[1:] == 1) & (states[:-1] != 1))[0] + 1
                if len(s1_onsets) >= 3:
                    rr = np.diff(s1_onsets / fs)
                    cv = np.std(rr) / (np.mean(rr) + eps) * 100
                    level = 'good' if cv < 20 else ('warn' if cv < 40 else 'bad')
                    result['节律性'] = dict(value=cv,
                                           text=f"CV {cv:.1f}%", level=level)
                else:
                    result['节律性'] = dict(value=None,
                                           text='数据不足', level='na')
            else:
                result['节律性'] = dict(value=None, text='无分割', level='na')
        except Exception:
            result['节律性'] = dict(value=None, text='N/A', level='na')

        # ─── 8. 频带能量占比 (20–800 Hz)
        # 心音与瓣膜病杂音的主要能量集中在 20-800Hz。
        # <40%：大量基线漂移（呼吸/移动）或极高频摩擦声干扰严重。
        # 使用 Welch PSD 避免短时异常值影响。
        try:
            if _freqs is not None:
                mask_band  = (_freqs >= 20) & (_freqs <= 800)
                total_pwr  = float(np.trapz(_psd, _freqs)) + eps
                band_pwr   = float(np.trapz(_psd[mask_band], _freqs[mask_band]))
                bp_ratio   = band_pwr / total_pwr
                level = 'good' if bp_ratio >= 0.60 else ('warn' if bp_ratio >= 0.40 else 'bad')
                result['频带占比'] = dict(value=bp_ratio,
                                        text=f"{bp_ratio*100:.1f}%", level=level)
            else:
                result['频带占比'] = dict(value=None, text='N/A', level='na')
        except Exception:
            result['频带占比'] = dict(value=None, text='N/A', level='na')

        return result

    # 指标颜色映射
    _QA_COLORS = {
        'good': ('#e8f5e9', '#2e7d32'),   # 绿底深绿字
        'warn': ('#fff8e1', '#e65100'),   # 黄底橙字
        'bad':  ('#ffebee', '#c62828'),   # 红底深红字
        'na':   ('#f5f5f5', '#aaaaaa'),   # 灰底灰字
    }
    # 硬性剔除指标（任一为 bad 时入组应谨慎）
    _QA_CORE = ('周期性', '时长', '削波率', '频带占比')

    def _update_quality_bar(self, quality):
        """根据 quality dict 刷新质控指标标签颜色与数值。"""
        if quality is None:
            for name, (cell, lbl) in self._qa_badges.items():
                cell.config(bg="#eeeeee")
                lbl.config(text="—", bg="#eeeeee", fg="#aaaaaa")
                for child in cell.winfo_children():
                    child.config(bg="#eeeeee")
            return

        for name, (cell, lbl) in self._qa_badges.items():
            m    = quality.get(name, {})
            lvl  = m.get('level', 'na')
            text = m.get('text', '—')
            bg, fg = self._QA_COLORS.get(lvl, self._QA_COLORS['na'])
            cell.config(bg=bg)
            lbl.config(text=text, bg=bg, fg=fg)
            for child in cell.winfo_children():
                child.config(bg=bg)

    # ── 文件列表面板 ────────────────────────────────────────────
    def _build_list_panel(self, parent: tk.Frame):
        """在 parent 内构建可滚动的文件编号列表。"""
        # 标题行
        hdr = tk.Frame(parent, bg="#dde3ed", pady=6)
        hdr.pack(fill="x")
        tk.Label(hdr, text="文件列表", bg="#dde3ed", fg="#222",
                 font=("Segoe UI", 9, "bold")).pack(side="left", padx=10)

        # 图例
        legend = tk.Frame(parent, bg="#f0f2f5", pady=3)
        legend.pack(fill="x")
        for sym, color in [("✓", "#2e7d32"), ("✗", "#c62828"),
                           ("?", "#e65100"), ("○", "#aaaaaa")]:
            tk.Label(legend, text=sym, bg="#f0f2f5", fg=color,
                     font=("Segoe UI", 8, "bold")).pack(side="left", padx=(6, 0))

        # 可滚动区域
        wrapper = tk.Frame(parent, bg="#f0f2f5")
        wrapper.pack(fill="both", expand=True)

        self._list_canvas = tk.Canvas(wrapper, bg="#f0f2f5",
                                      highlightthickness=0, bd=0)
        vsb = tk.Scrollbar(wrapper, orient="vertical",
                           command=self._list_canvas.yview)
        self._list_canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._list_canvas.pack(side="left", fill="both", expand=True)

        self._list_inner = tk.Frame(self._list_canvas, bg="#f0f2f5")
        self._list_canvas_win = self._list_canvas.create_window(
            (0, 0), window=self._list_inner, anchor="nw")

        self._list_inner.bind("<Configure>", self._on_list_inner_configure)
        self._list_canvas.bind("<Configure>", self._on_list_canvas_configure)
        self._list_canvas.bind("<MouseWheel>",
            lambda e: self._list_canvas.yview_scroll(-1*(e.delta//120), "units"))

    def _on_list_inner_configure(self, _event=None):
        self._list_canvas.configure(scrollregion=self._list_canvas.bbox("all"))

    def _on_list_canvas_configure(self, event):
        self._list_canvas.itemconfig(self._list_canvas_win, width=event.width)

    def _toggle_list_panel(self):
        """显示 / 隐藏右侧文件列表面板。"""
        if self._list_visible:
            self._list_panel.pack_forget()
            self._list_visible = False
            self._btn_list_toggle.config(relief="flat", bg="#e0e0e0")
        else:
            self._list_panel.pack(side="right", fill="y", padx=(6, 16))
            self._list_visible = True
            self._btn_list_toggle.config(relief="sunken", bg="#c8d8f0")
            self._refresh_list_panel()

    # 颜色方案（idle / active）
    _LIST_IDLE   = {"✓": ("#e8f5e9", "#2e7d32"), "✗": ("#ffebee", "#c62828"),
                    "?": ("#fff3e0", "#e65100"),  None: ("#f5f5f5", "#999999")}
    _LIST_ACTIVE = {"✓": ("#2e7d32", "#ffffff"), "✗": ("#c62828", "#ffffff"),
                    "?": ("#e65100", "#ffffff"),  None: ("#1a73e8", "#ffffff")}

    def _refresh_list_panel(self):
        """根据最新文件列表和标注状态，重建列表按钮网格。"""
        if not self._list_visible:
            return
        for w in self._list_inner.winfo_children():
            w.destroy()
        self._list_btns.clear()

        COLS = 4
        for i, path in enumerate(self.audio_files):
            lbl    = self.labels.get(path)
            is_cur = (i == self.current_idx)
            if is_cur:
                bg, fg = self._LIST_ACTIVE.get(lbl, ("#1a73e8", "#ffffff"))
                relief = "sunken"
            else:
                bg, fg = self._LIST_IDLE.get(lbl, ("#f5f5f5", "#999999"))
                relief = "flat"

            sym  = lbl if lbl else "○"
            text = f"{sym}\n{i+1}"

            b = tk.Button(
                self._list_inner,
                text=text,
                font=("Segoe UI", 8, "bold"),
                bg=bg, fg=fg,
                activebackground="#1a73e8", activeforeground="#ffffff",
                relief=relief, bd=1,
                width=4, height=2,
                cursor="hand2",
                command=lambda idx=i: self._jump_to(idx))
            row, col = divmod(i, COLS)
            b.grid(row=row, column=col, padx=2, pady=2, sticky="nsew")
            self._list_btns.append(b)

        for c in range(COLS):
            self._list_inner.columnconfigure(c, weight=1)

        if self.audio_files:
            self.after(50, self._scroll_list_to_current)

    def _scroll_list_to_current(self):
        """将列表滚动到当前文件按钮可见处。"""
        if not self.audio_files or not self._list_btns:
            return
        COLS = 4
        row        = self.current_idx // COLS
        total_rows = (len(self.audio_files) - 1) // COLS + 1
        if total_rows <= 1:
            return
        frac = row / total_rows
        self._list_canvas.yview_moveto(max(0.0, frac - 0.1))

    def _jump_to(self, idx: int):
        """跳转到指定索引的文件，刷新图形和列表。"""
        if idx < 0 or idx >= len(self.audio_files):
            return
        self.current_idx = idx
        self.plot_current()

    # ── 标注方法 ─────────────────────────────────────────────
    def _set_label(self, label: str):
        """为当前文件设置标注，并刷新界面。"""
        if not self.audio_files:
            return
        path = self.audio_files[self.current_idx]
        # 再次点击同一标注则取消
        if self.labels.get(path) == label:
            self.labels[path] = None
        else:
            self.labels[path] = label
        self._update_label_ui()

    def _update_label_ui(self):
        """根据当前文件标注状态刷新按钮高亮与统计信息。"""
        if not self.audio_files:
            return
        path    = self.audio_files[self.current_idx]
        current = self.labels.get(path)

        # 高亮当前选中的标注按钮
        styles = {
            self.LABEL_ACCEPT: (self._btn_accept, '#2e7d32', '#e8f5e9'),
            self.LABEL_REJECT: (self._btn_reject, '#c62828', '#ffebee'),
            self.LABEL_UNSURE: (self._btn_unsure, '#e65100', '#fff3e0'),
        }
        for lbl, (btn, active_bg, idle_bg) in styles.items():
            if current == lbl:
                btn.config(bg=active_bg, fg='#ffffff', relief='sunken')
            else:
                btn.config(bg=idle_bg, fg=active_bg, relief='flat')

        # 当前标注文字
        name  = self.LABEL_NAMES.get(current, '未标注')
        color = self.LABEL_COLORS.get(current, '#888888')
        self._lbl_cur_label.config(text=f"当前：{name}", fg=color)

        # 统计
        total   = len(self.audio_files)
        n_acc   = sum(1 for v in self.labels.values() if v == self.LABEL_ACCEPT)
        n_rej   = sum(1 for v in self.labels.values() if v == self.LABEL_REJECT)
        n_uns   = sum(1 for v in self.labels.values() if v == self.LABEL_UNSURE)
        n_none  = total - n_acc - n_rej - n_uns
        self._lbl_stats.config(
            text=f"✓ {n_acc}   ✗ {n_rej}   ? {n_uns}   未标注 {n_none} / 共 {total}")
        self._refresh_list_panel()

    # ── 标注持久化 ───────────────────────────────────────────
    def _labels_path(self) -> str:
        """标注 JSON 文件的路径（与数据文件夹同级）。"""
        return os.path.join(self.base_folder, '.label_annotations.json') if self.base_folder else ''

    def _save_labels(self):
        """将当前标注保存到 JSON 文件。"""
        if not self.base_folder:
            messagebox.showwarning('未加载文件夹', '请先选择文件夹再保存标注。')
            return
        data = {os.path.relpath(k, self.base_folder): v
                for k, v in self.labels.items() if v is not None}
        path = self._labels_path()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        messagebox.showinfo('保存成功', f'标注已保存至：\n{path}')

    def _load_labels(self):
        """从 JSON 文件加载标注（如存在）。"""
        path = self._labels_path()
        if not path or not os.path.isfile(path):
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for rel, v in data.items():
                abs_path = os.path.normpath(os.path.join(self.base_folder, rel))
                if abs_path in self.audio_files or any(
                        os.path.normpath(p) == abs_path for p in self.audio_files):
                    # 找到匹配的绝对路径
                    for p in self.audio_files:
                        if os.path.normpath(p) == abs_path:
                            self.labels[p] = v
                            break
            print(f'已加载 {len(self.labels)} 条标注记录。')
        except Exception as e:
            print(f'加载标注失败：{e}')

    # ── 批量导出 ─────────────────────────────────────────────
    def export_dialog(self):
        """弹出导出对话框，将指定类别的文件复制到目标文件夹。"""
        if not self.audio_files:
            messagebox.showwarning('未加载文件', '请先选择文件夹并加载数据。')
            return

        dlg = tk.Toplevel(self)
        dlg.title('批量导出')
        dlg.geometry('460x320')
        dlg.configure(bg='#f8f9fa')
        dlg.resizable(False, False)
        dlg.grab_set()

        # ── 选择要导出的类别 ──
        tk.Label(dlg, text='选择要导出的类别（可多选）：',
                 bg='#f8f9fa', fg='#222', font=('Segoe UI', 10, 'bold')
                 ).pack(anchor='w', padx=20, pady=(18, 6))

        n_acc  = sum(1 for v in self.labels.values() if v == self.LABEL_ACCEPT)
        n_rej  = sum(1 for v in self.labels.values() if v == self.LABEL_REJECT)
        n_uns  = sum(1 for v in self.labels.values() if v == self.LABEL_UNSURE)
        n_none = len(self.audio_files) - n_acc - n_rej - n_uns

        var_acc  = tk.BooleanVar(value=True)
        var_rej  = tk.BooleanVar(value=False)
        var_uns  = tk.BooleanVar(value=False)
        var_none = tk.BooleanVar(value=False)

        ck_cfg = dict(bg='#f8f9fa', font=('Segoe UI', 10), activebackground='#f8f9fa')
        tk.Checkbutton(dlg, text=f'✓  入组   （{n_acc} 个文件）',
                       variable=var_acc, fg='#2e7d32', selectcolor='#e8f5e9', **ck_cfg
                       ).pack(anchor='w', padx=36, pady=3)
        tk.Checkbutton(dlg, text=f'✗  排除   （{n_rej} 个文件）',
                       variable=var_rej, fg='#c62828', selectcolor='#ffebee', **ck_cfg
                       ).pack(anchor='w', padx=36, pady=3)
        tk.Checkbutton(dlg, text=f'?   待定   （{n_uns} 个文件）',
                       variable=var_uns, fg='#e65100', selectcolor='#fff3e0', **ck_cfg
                       ).pack(anchor='w', padx=36, pady=3)
        tk.Checkbutton(dlg, text=f'—  未标注 （{n_none} 个文件）',
                       variable=var_none, fg='#888888', selectcolor='#eeeeee', **ck_cfg
                       ).pack(anchor='w', padx=36, pady=3)

        # ── 目标文件夹 ──
        tk.Label(dlg, text='导出到文件夹：',
                 bg='#f8f9fa', fg='#222', font=('Segoe UI', 10, 'bold')
                 ).pack(anchor='w', padx=20, pady=(14, 2))

        dest_var = tk.StringVar()
        row = tk.Frame(dlg, bg='#f8f9fa')
        row.pack(fill='x', padx=20)
        dest_entry = tk.Entry(row, textvariable=dest_var, font=('Segoe UI', 9),
                              relief='flat', bg='#eeeeee', fg='#222', width=38)
        dest_entry.pack(side='left', padx=(0, 6), ipady=4)
        tk.Button(row, text='浏览…',
                  command=lambda: dest_var.set(
                      filedialog.askdirectory(title='选择导出目标文件夹') or dest_var.get()),
                  bg='#e0e0e0', relief='flat', font=('Segoe UI', 9), padx=8, pady=4
                  ).pack(side='left')

        # ── 保留相对路径选项 ──
        keep_struct = tk.BooleanVar(value=True)
        tk.Checkbutton(dlg, text='保留原始文件夹层级结构',
                       variable=keep_struct, bg='#f8f9fa',
                       font=('Segoe UI', 9), activebackground='#f8f9fa'
                       ).pack(anchor='w', padx=20, pady=(8, 0))

        # ── 确认按钮 ──
        def do_export():
            dest = dest_var.get().strip()
            if not dest:
                messagebox.showwarning('未选择目标', '请先选择导出目标文件夹。', parent=dlg)
                return
            want = set()
            if var_acc.get():  want.add(self.LABEL_ACCEPT)
            if var_rej.get():  want.add(self.LABEL_REJECT)
            if var_uns.get():  want.add(self.LABEL_UNSURE)
            if var_none.get(): want.add(None)
            if not want:
                messagebox.showwarning('未选类别', '请至少勾选一个类别。', parent=dlg)
                return

            to_copy = [p for p in self.audio_files
                       if self.labels.get(p) in want]
            if not to_copy:
                messagebox.showinfo('无文件', '所选类别下没有文件。', parent=dlg)
                return

            ok, fail = 0, []
            for src in to_copy:
                try:
                    if keep_struct.get() and self.base_folder:
                        rel  = os.path.relpath(src, self.base_folder)
                        dst  = os.path.join(dest, rel)
                    else:
                        dst  = os.path.join(dest, os.path.basename(src))
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
                    ok += 1
                except Exception as e:
                    fail.append(f'{os.path.basename(src)}: {e}')

            msg = f'成功复制 {ok} 个文件到：\n{dest}'
            if fail:
                msg += f'\n\n失败 {len(fail)} 个：\n' + '\n'.join(fail[:5])
                if len(fail) > 5:
                    msg += f'\n...共 {len(fail)} 个'
            messagebox.showinfo('导出完成', msg, parent=dlg)
            if not fail:
                dlg.destroy()

        tk.Button(dlg, text='▶  开始导出', command=do_export,
                  bg='#1565c0', fg='#ffffff', activebackground='#0d47a1',
                  activeforeground='#ffffff', relief='flat',
                  font=('Segoe UI', 10, 'bold'), padx=20, pady=7, cursor='hand2'
                  ).pack(side='bottom', pady=(0, 18))


# ════════════════════════════════════════════════════════════════
#  启动查看器
# ════════════════════════════════════════════════════════════════

app = SegmentationViewer(_info)
app.mainloop()
