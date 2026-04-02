"""
ECG 波形批量绘图工具
- 弹出对话框选择文件夹
- 读取文件夹下所有 CSV 文件（单列，无表头）
- 逐个绘制波形图并保存为 PNG，同时支持交互翻页查看
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


# ── 全局样式 ────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#1e1e2e",
    "axes.facecolor":   "#1e1e2e",
    "axes.edgecolor":   "#555577",
    "axes.labelcolor":  "#cdd6f4",
    "xtick.color":      "#cdd6f4",
    "ytick.color":      "#cdd6f4",
    "text.color":       "#cdd6f4",
    "grid.color":       "#313244",
    "grid.linewidth":   0.6,
    "lines.linewidth":  1.2,
    "lines.color":      "#89dceb",
    "font.family":      "DejaVu Sans",
})


# ── 读取 CSV ────────────────────────────────────────────────
def load_csv(filepath):
    """
    读取单列 ECG CSV 文件。
    自动跳过非数字表头行，兼容有无表头两种情况。
    """
    try:
        df = pd.read_csv(filepath, header=None)
        # 取第一列
        col = df.iloc[:, 0]
        # 转数值，无法转换的变为 NaN 后丢弃（跳过文字表头）
        col = pd.to_numeric(col, errors="coerce").dropna()
        if col.empty:
            return None, f"文件 {os.path.basename(filepath)} 无有效数值数据"
        return col.values.astype(np.float32), None
    except Exception as e:
        return None, str(e)


# ── 主 GUI 窗口 ─────────────────────────────────────────────
class ECGViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ECG 波形查看器")
        self.geometry("1100x680")
        self.configure(bg="#1e1e2e")
        self.resizable(True, True)

        self.files = []          # 全部 csv 路径
        self.current_idx = 0    # 当前显示索引
        self.save_dir = ""       # 保存目录

        self._build_ui()

    # ── 构建界面 ──────────────────────────────────────────
    def _build_ui(self):
        # 顶部工具栏
        bar = tk.Frame(self, bg="#313244", pady=6)
        bar.pack(fill=tk.X, side=tk.TOP)

        btn_style = dict(bg="#585b70", fg="#cdd6f4", relief=tk.FLAT,
                         padx=12, pady=4, cursor="hand2",
                         activebackground="#7f849c", activeforeground="#ffffff")

        tk.Button(bar, text="📂 选择文件夹", command=self._select_folder,
                  **btn_style).pack(side=tk.LEFT, padx=8)

        tk.Button(bar, text="💾 保存全部 PNG", command=self._save_all,
                  **btn_style).pack(side=tk.LEFT, padx=4)

        tk.Button(bar, text="◀ 上一个", command=self._prev,
                  **btn_style).pack(side=tk.LEFT, padx=4)

        tk.Button(bar, text="▶ 下一个", command=self._next,
                  **btn_style).pack(side=tk.LEFT, padx=4)

        # 文件名标签
        self.lbl_file = tk.Label(bar, text="未选择文件夹",
                                  bg="#313244", fg="#a6e3a1",
                                  font=("Consolas", 11))
        self.lbl_file.pack(side=tk.LEFT, padx=16)

        # 进度标签
        self.lbl_progress = tk.Label(bar, text="",
                                      bg="#313244", fg="#fab387",
                                      font=("Consolas", 10))
        self.lbl_progress.pack(side=tk.RIGHT, padx=16)

        # 进度条（保存时用）
        self.pbar = ttk.Progressbar(self, orient=tk.HORIZONTAL,
                                     mode="determinate", length=200)

        # 绘图区
        self.fig, self.ax = plt.subplots(figsize=(10, 4.5))
        self.fig.tight_layout(pad=2.5)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 4))

        toolbar_frame = tk.Frame(self, bg="#1e1e2e")
        toolbar_frame.pack(fill=tk.X)
        NavigationToolbar2Tk(self.canvas, toolbar_frame)

    # ── 选择文件夹 ─────────────────────────────────────────
    def _select_folder(self):
        folder = filedialog.askdirectory(title="选择包含 ECG CSV 文件的文件夹")
        if not folder:
            return
        self.save_dir = folder
        # 递归收集所有 csv
        csv_files = []
        for root, _, files in os.walk(folder):
            for f in sorted(files):
                if f.lower().endswith(".csv"):
                    csv_files.append(os.path.join(root, f))

        if not csv_files:
            messagebox.showwarning("提示", "所选文件夹下未找到 CSV 文件")
            return

        self.files = csv_files
        self.current_idx = 0
        self.lbl_progress.config(text=f"共 {len(self.files)} 个文件")
        self._plot_current()

    # ── 绘制当前文件 ───────────────────────────────────────
    def _plot_current(self):
        if not self.files:
            return

        filepath = self.files[self.current_idx]
        filename = os.path.basename(filepath)
        signal, err = load_csv(filepath)

        self.ax.cla()
        self.ax.grid(True, axis="both")

        if err:
            self.ax.text(0.5, 0.5, f"读取失败:\n{err}",
                         ha="center", va="center",
                         transform=self.ax.transAxes, fontsize=12, color="#f38ba8")
        else:
            n = len(signal)
            # 尝试按 500 Hz 构建时间轴
            fs = 500
            t = np.arange(n) / fs
            self.ax.plot(t, signal, color="#89dceb", linewidth=0.9)
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Amplitude")
            self.ax.set_xlim([t[0], t[-1]])
            # 标记最大最小值
            idx_max = np.argmax(signal)
            idx_min = np.argmin(signal)
            self.ax.annotate(f"max={signal[idx_max]:.0f}",
                              xy=(t[idx_max], signal[idx_max]),
                              xytext=(10, 10), textcoords="offset points",
                              color="#a6e3a1", fontsize=8,
                              arrowprops=dict(arrowstyle="->", color="#a6e3a1", lw=0.8))
            self.ax.annotate(f"min={signal[idx_min]:.0f}",
                              xy=(t[idx_min], signal[idx_min]),
                              xytext=(10, -18), textcoords="offset points",
                              color="#f38ba8", fontsize=8,
                              arrowprops=dict(arrowstyle="->", color="#f38ba8", lw=0.8))

        self.ax.set_title(
            f"{filename}  [{self.current_idx + 1} / {len(self.files)}]",
            pad=8, fontsize=12, color="#cba6f7"
        )
        self.fig.tight_layout(pad=2.5)
        self.canvas.draw()

        short = filename if len(filename) <= 50 else "…" + filename[-47:]
        self.lbl_file.config(text=short)
        self.lbl_progress.config(
            text=f"{self.current_idx + 1} / {len(self.files)} 个文件"
        )

    # ── 翻页 ───────────────────────────────────────────────
    def _prev(self):
        if not self.files:
            return
        self.current_idx = (self.current_idx - 1) % len(self.files)
        self._plot_current()

    def _next(self):
        if not self.files:
            return
        self.current_idx = (self.current_idx + 1) % len(self.files)
        self._plot_current()

    # ── 键盘左右翻页 ──────────────────────────────────────
    def _bind_keys(self):
        self.bind("<Left>",  lambda e: self._prev())
        self.bind("<Right>", lambda e: self._next())

    # ── 保存全部为 PNG ─────────────────────────────────────
    def _save_all(self):
        if not self.files:
            messagebox.showwarning("提示", "请先选择文件夹")
            return

        out_dir = os.path.join(self.save_dir, "ecg_plots")
        os.makedirs(out_dir, exist_ok=True)

        self.pbar.config(maximum=len(self.files), value=0)
        self.pbar.pack(fill=tk.X, padx=10, pady=4)
        self.update_idletasks()

        fig_s, ax_s = plt.subplots(figsize=(12, 4))
        fig_s.patch.set_facecolor("#1e1e2e")
        ax_s.set_facecolor("#1e1e2e")

        saved = 0
        errors = []
        for i, filepath in enumerate(self.files):
            signal, err = load_csv(filepath)
            ax_s.cla()
            ax_s.set_facecolor("#1e1e2e")
            ax_s.grid(True, color="#313244", linewidth=0.6)
            ax_s.tick_params(colors="#cdd6f4")
            ax_s.spines[:].set_color("#555577")

            name = os.path.basename(filepath)
            if err:
                errors.append(f"{name}: {err}")
            else:
                n = len(signal)
                t = np.arange(n) / 500
                ax_s.plot(t, signal, color="#89dceb", linewidth=0.8)
                ax_s.set_xlabel("Time (s)", color="#cdd6f4")
                ax_s.set_ylabel("Amplitude", color="#cdd6f4")
                ax_s.set_xlim([t[0], t[-1]])
                ax_s.set_title(name, color="#cba6f7", fontsize=11)

            png_name = os.path.splitext(name)[0] + ".png"
            # 保持子文件夹结构
            rel = os.path.relpath(filepath, self.save_dir)
            out_path = os.path.join(out_dir,
                                    os.path.splitext(rel)[0] + ".png")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            fig_s.tight_layout(pad=2)
            fig_s.savefig(out_path, dpi=150, facecolor="#1e1e2e")
            saved += 1

            self.pbar["value"] = i + 1
            self.lbl_progress.config(
                text=f"保存中… {i + 1}/{len(self.files)}"
            )
            self.update_idletasks()

        plt.close(fig_s)
        self.pbar.pack_forget()
        self.lbl_progress.config(text=f"{len(self.files)} / {len(self.files)} 个文件")

        msg = f"✅ 已保存 {saved} 张图像\n📁 输出目录：{out_dir}"
        if errors:
            msg += f"\n\n⚠️ {len(errors)} 个文件读取失败：\n" + "\n".join(errors[:5])
        messagebox.showinfo("保存完成", msg)


# ── 入口 ────────────────────────────────────────────────────
if __name__ == "__main__":
    app = ECGViewer()
    app._bind_keys()
    # 启动时可直接弹出选择框
    app.after(300, app._select_folder)
    app.mainloop()
