"""
ONMF PCG预处理  —  交互式GUI主程序
用户点击运行，弹出界面，选择文件夹后自动处理。

依赖：tkinter (Python标准库), numpy, scipy, pandas, matplotlib
"""

import os
import sys
import threading
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# 中文字体
def _setup_chinese_font():
    candidates = ['Microsoft YaHei', 'SimHei', 'SimSun', 'STSong']
    available = {f.name for f in _fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            return
    import os
    font_path = r'C:\Windows\Fonts\msyh.ttc'
    if os.path.exists(font_path):
        prop = _fm.FontProperties(fname=font_path)
        plt.rcParams['font.sans-serif'] = [prop.get_name(), 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
_setup_chinese_font()

# 本地模块
from data_loader import find_audio_files, load_pcg_file
from onmf_preprocessing import preprocess_pcg_to_onmf
from visualization import (plot_all_results, plot_feature_matrix,
                           plot_figure4_style, plot_figure3_style,
                           plot_w_bases, plot_clean_view)


# ---------------------------------------------------------------------------
# 全局状态
# ---------------------------------------------------------------------------
_processed_results = {}   # filepath -> result dict
_current_file = None       # 当前显示的文件路径


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _log(widget: scrolledtext.ScrolledText, msg: str):
    """向日志文本框追加一行。"""
    widget.config(state=tk.NORMAL)
    widget.insert(tk.END, msg + '\n')
    widget.see(tk.END)
    widget.config(state=tk.DISABLED)


def _set_status(var: tk.StringVar, msg: str):
    var.set(msg)


# ---------------------------------------------------------------------------
# 核心处理线程
# ---------------------------------------------------------------------------

def _process_files_thread(folder, params, log_widget, status_var,
                           file_listbox, run_btn, app_root):
    """在后台线程中执行文件处理。"""
    try:
        _set_status(status_var, '正在扫描文件夹...')
        csv_files = find_audio_files(folder)

        if not csv_files:
            _log(log_widget, '未在所选文件夹中找到 CSV 或 WAV 文件。')
            _set_status(status_var, '完成：未找到文件')
            app_root.after(0, lambda: run_btn.config(state=tk.NORMAL))
            return

        _log(log_widget, f'找到 {len(csv_files)} 个文件。')

        for i, fp in enumerate(csv_files, 1):
            rel = os.path.relpath(fp, folder)
            _set_status(status_var, f'处理 {i}/{len(csv_files)}: {rel}')
            _log(log_widget, f'\n[{i}/{len(csv_files)}] 处理: {rel}')

            try:
                # 加载数据（CSV 或 WAV）
                data = load_pcg_file(
                    fp,
                    pcg_col=params['pcg_col'],
                    default_fs=params['fs'],
                    interp_method='linear'
                )
                if data['n_missing'] > 0:
                    _log(log_widget,
                         f'  插值补全: {data["n_missing"]} 个缺失值')

                signal = data['signal']
                fs = data['fs']

                if len(signal) < 256:
                    _log(log_widget, f'  警告：信号太短（{len(signal)}点），跳过')
                    continue

                # 预处理 (Algorithm 1 严格遵循论文)
                # window_size_ms 在函数内部在降采样后换算，避免用输入fs预计算导致窗口翻倍
                result = preprocess_pcg_to_onmf(
                    signal,
                    fs=fs,
                    r=params['r'],
                    low_hz=params['low_hz'],
                    high_hz=params['high_hz'],
                    window_size_ms=params['window_ms'],
                    overlap_ratio=params['overlap'],
                    max_iter=params['max_iter'],
                    use_temporal=params['use_temporal'],
                )
                result['fs'] = result.get('fs_processed', fs)   # 使用降采样后的实际fs（4096Hz）
                result['filename'] = data['filename']
                result['filepath'] = fp

                _processed_results[fp] = result

                feat = result['feature']
                _log(log_widget,
                     f'  完成: 特征矩阵形状={feat.shape}, '
                     f'ONMF迭代={len(result["errors"])}次, '
                     f'最终误差={result["errors"][-1]:.4f}')

                # 在主线程更新列表框
                app_root.after(0, lambda r=rel, p=fp: _add_to_listbox(
                    file_listbox, r, p))

            except Exception as e:
                _log(log_widget, f'  错误: {e}')
                if params.get('verbose'):
                    _log(log_widget, traceback.format_exc())

        _set_status(status_var, f'全部完成：{len(_processed_results)} 个文件已处理')
        _log(log_widget, '\n====== 处理完成 ======')

    except Exception as e:
        _log(log_widget, f'\n严重错误: {e}\n{traceback.format_exc()}')
        _set_status(status_var, '处理出错')
    finally:
        app_root.after(0, lambda: run_btn.config(state=tk.NORMAL))


def _add_to_listbox(listbox, display_name, filepath):
    listbox.insert(tk.END, display_name)
    listbox.result_paths = getattr(listbox, 'result_paths', [])
    listbox.result_paths.append(filepath)


# ---------------------------------------------------------------------------
# 可视化回调
# ---------------------------------------------------------------------------

def _show_visualization(listbox, vis_type_var, log_widget):
    """在独立窗口中显示选中文件的可视化结果。"""
    sel = listbox.curselection()
    if not sel:
        messagebox.showinfo('提示', '请先在列表中选择一个文件')
        return

    idx = sel[0]
    paths = getattr(listbox, 'result_paths', [])
    if idx >= len(paths):
        return

    fp = paths[idx]
    result = _processed_results.get(fp)
    if result is None:
        messagebox.showerror('错误', '该文件尚未处理完成')
        return

    vis_type = vis_type_var.get()
    fs = result.get('fs', 4000.0)
    filename = result.get('filename', os.path.basename(fp))

    # 关闭现有matplotlib窗口，避免堆积
    plt.close('all')

    if vis_type == 'clean':
        plot_clean_view(result, filename=filename, fs=fs, show=True)
    elif vis_type == 'fig4':
        plot_figure4_style(result, filename=filename, show=True)
    elif vis_type == 'fig3':
        plot_figure3_style(result, filename=filename, show=True)
    elif vis_type == 'w_bases':
        plot_w_bases(result, filename=filename, show=True)
    elif vis_type == 'all':
        plot_all_results(result, filename=filename, fs=fs, show=True)
    elif vis_type == 'feature':
        feat = result['feature']
        is_temporal = feat.shape[0] < feat.shape[1]
        lrow = '基底索引 K' if is_temporal else '频率帧索引'
        lcol = '时间帧索引' if is_temporal else '基底索引 K'
        title = f'CNN输入特征矩阵 ("{"H-时域" if is_temporal else "W-频域"}")  |  {filename}'
        plot_feature_matrix(feat, title=title,
                            label_row=lrow, label_col=lcol, show=True)


# _plot_stft_only 和 _plot_wh 已被 visualization.py 中的新函数取代。
# 保留占位以防旧代码引用。
def _plot_stft_only(result, filename, fs):
    plot_figure3_style(result, filename=filename, show=True)


def _plot_wh(result, filename):
    plot_figure4_style(result, filename=filename, show=True)


def _save_results(listbox, log_widget, output_dir_var):
    """将选中文件的特征矩阵保存为npy文件。"""
    sel = listbox.curselection()
    targets = []
    if sel:
        paths = getattr(listbox, 'result_paths', [])
        targets = [(paths[i], _processed_results.get(paths[i]))
                   for i in sel if i < len(paths)]
    else:
        targets = [(fp, r) for fp, r in _processed_results.items()]

    if not targets:
        messagebox.showinfo('提示', '没有可保存的结果')
        return

    out_dir = output_dir_var.get()
    if not out_dir:
        out_dir = filedialog.askdirectory(title='选择保存目录')
        if not out_dir:
            return
        output_dir_var.set(out_dir)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    saved = 0
    for fp, result in targets:
        if result is None:
            continue
        stem = Path(fp).stem
        feat_path = os.path.join(out_dir, f'{stem}_feature.npy')
        np.save(feat_path, result['feature'])
        w_path = os.path.join(out_dir, f'{stem}_W.npy')
        h_path = os.path.join(out_dir, f'{stem}_H.npy')
        np.save(w_path, result['W'])
        np.save(h_path, result['H'])
        saved += 1

    _log(log_widget, f'\n已保存 {saved} 个文件的特征矩阵到: {out_dir}')
    messagebox.showinfo('保存完成', f'已保存 {saved} 个特征矩阵到:\n{out_dir}')


# ---------------------------------------------------------------------------
# 主GUI构建
# ---------------------------------------------------------------------------

def build_gui():
    root = tk.Tk()
    root.title('ONMF PCG 预处理工具  —  Torre-Cruz et al. 2023')
    root.geometry('1050x750')
    root.resizable(True, True)

    # ---- 顶部标题 ----
    header = tk.Label(root,
                      text='ONMF PCG 预处理工具',
                      font=('Microsoft YaHei', 14, 'bold'),
                      fg='#1a5276', pady=8)
    header.pack(fill=tk.X)

    sub_header = tk.Label(root,
                          text='Algorithm 1: 幅度归一化 → STFT 抖幅谱   |   频带截取 B_A=[200-700Hz]   |   ONMF分解 (K=128, I=60)   |   输出 H/W 特征',
                          font=('Microsoft YaHei', 9),
                          fg='#555')
    sub_header.pack()

    ttk.Separator(root, orient='horizontal').pack(fill=tk.X, pady=4)

    # ---- 主布局 ----
    main_frame = ttk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

    left_frame = ttk.Frame(main_frame, width=340)
    left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))
    left_frame.pack_propagate(False)

    right_frame = ttk.Frame(main_frame)
    right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # ================================================================
    # 左侧：参数面板
    # ================================================================
    param_lf = ttk.LabelFrame(left_frame, text='处理参数', padding=8)
    param_lf.pack(fill=tk.X, pady=(0, 6))

    def _make_row(parent, label, row, default, width=8, hint=None):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', pady=2)
        var = tk.StringVar(value=str(default))
        entry = ttk.Entry(parent, textvariable=var, width=width)
        entry.grid(row=row, column=1, sticky='w', padx=4)
        if hint:
            ttk.Label(parent, text=hint, foreground='gray', font=('', 8)).grid(
                row=row, column=2, sticky='w')
        return var

    pcg_col_var   = _make_row(param_lf, 'PCG列名:',      0, 'pcg',   hint='CSV列名')
    fs_var        = _make_row(param_lf, '采样率(Hz):',   1, 4096,   hint='WAV自动识别')
    r_var         = _make_row(param_lf, 'ONMF秩 K:',     2, 128,    hint='论文最优K=128')
    low_var       = _make_row(param_lf, '低截止(Hz):',   3, 200,    hint='B_A下限')
    high_var      = _make_row(param_lf, '高截止(Hz):',   4, 700,    hint='B_A上限')
    win_var       = _make_row(param_lf, 'STFT窗(ms):',   5, 31.25, hint='128样本@4096Hz')
    overlap_var   = _make_row(param_lf, 'STFT重叠:',     6, 0.75,   hint='75%重叠=25%hop')
    maxiter_var   = _make_row(param_lf, '最大迭代:',     7, 60,     hint='论文I=60')

    # 频带预设按钮（Table 2）
    ttk.Label(param_lf, text='频带预设:').grid(row=8, column=0, sticky='w', pady=(4, 2))
    band_frame = ttk.Frame(param_lf)
    band_frame.grid(row=8, column=1, columnspan=2, sticky='w')

    BAND_PRESETS = {
        'B_A\n200-700': (200, 700),   # 最优（论文结论）
        'B_C\n20-700':  (20,  700),
        'B_N\n20-200':  (20,  200),
        'B_F\n20-2048': (20, 2048),
    }
    for label, (lo, hi) in BAND_PRESETS.items():
        ttk.Button(
            band_frame, text=label, width=7,
            command=lambda l=lo, h=hi: (low_var.set(str(l)), high_var.set(str(h)))
        ).pack(side=tk.LEFT, padx=2)

    # 特征类型选择
    ttk.Label(param_lf, text='特征类型:').grid(row=9, column=0, sticky='w', pady=2)
    feature_var = tk.StringVar(value='temporal')
    frame_feat = ttk.Frame(param_lf)
    frame_feat.grid(row=9, column=1, columnspan=2, sticky='w')
    ttk.Radiobutton(frame_feat, text='H（时域）', variable=feature_var,
                    value='temporal').pack(side=tk.LEFT)
    ttk.Radiobutton(frame_feat, text='W（频域）', variable=feature_var,
                    value='spectral').pack(side=tk.LEFT, padx=4)

    # ---- 文件夹选择 ----
    folder_lf = ttk.LabelFrame(left_frame, text='输入/输出', padding=8)
    folder_lf.pack(fill=tk.X, pady=(0, 6))

    folder_var = tk.StringVar()
    output_dir_var = tk.StringVar()

    def _browse_folder():
        d = filedialog.askdirectory(title='选择包含CSV文件的文件夹')
        if d:
            folder_var.set(d)

    def _browse_output():
        d = filedialog.askdirectory(title='选择结果保存文件夹')
        if d:
            output_dir_var.set(d)

    ttk.Label(folder_lf, text='输入文件夹:').grid(row=0, column=0, sticky='w')
    folder_entry = ttk.Entry(folder_lf, textvariable=folder_var, width=28)
    folder_entry.grid(row=0, column=1, sticky='ew', padx=(4, 0))
    ttk.Button(folder_lf, text='浏览', command=_browse_folder, width=6).grid(
        row=0, column=2, padx=(4, 0))

    ttk.Label(folder_lf, text='输出目录:').grid(row=1, column=0, sticky='w', pady=(4, 0))
    output_entry = ttk.Entry(folder_lf, textvariable=output_dir_var, width=28)
    output_entry.grid(row=1, column=1, sticky='ew', padx=(4, 0))
    ttk.Button(folder_lf, text='浏览', command=_browse_output, width=6).grid(
        row=1, column=2, padx=(4, 0))

    folder_lf.columnconfigure(1, weight=1)

    # ---- 运行按钮 ----
    status_var = tk.StringVar(value='就绪')
    run_btn = ttk.Button(
        left_frame, text='▶  开始处理',
        style='Accent.TButton'
    )
    run_btn.pack(fill=tk.X, pady=4)

    ttk.Label(left_frame, textvariable=status_var,
              foreground='#1a5276', font=('', 9)).pack()

    # ---- 可视化控件 ----
    vis_lf = ttk.LabelFrame(left_frame, text='可视化', padding=8)
    vis_lf.pack(fill=tk.X, pady=(4, 0))

    vis_type_var = tk.StringVar(value='fig4')
    vis_options = [
        ('简洁视图（PCG+谱+H）★', 'clean'),
        ('Figure 4风格（W+H）',   'fig4'),
        ('Figure 3风格（幅度谱）', 'fig3'),
        ('W基底频谱曲线',          'w_bases'),
        ('综合结果（6图）',        'all'),
        ('CNN输入特征矩阵',        'feature'),
    ]
    for text, val in vis_options:
        ttk.Radiobutton(vis_lf, text=text,
                        variable=vis_type_var, value=val).pack(anchor='w')

    # ================================================================
    # 右侧：已处理文件列表 + 日志
    # ================================================================
    list_lf = ttk.LabelFrame(right_frame, text='已处理文件（点击选择后可视化/保存）', padding=4)
    list_lf.pack(fill=tk.BOTH, expand=True, pady=(0, 4))

    list_scroll = ttk.Scrollbar(list_lf, orient=tk.VERTICAL)
    file_listbox = tk.Listbox(list_lf, selectmode=tk.EXTENDED,
                              yscrollcommand=list_scroll.set,
                              font=('Consolas', 9), height=10)
    list_scroll.config(command=file_listbox.yview)
    list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    file_listbox.pack(fill=tk.BOTH, expand=True)
    file_listbox.result_paths = []

    btn_row = ttk.Frame(right_frame)
    btn_row.pack(fill=tk.X, pady=(0, 4))

    vis_btn = ttk.Button(
        btn_row, text='查看可视化',
        command=lambda: _show_visualization(file_listbox, vis_type_var, log_widget)
    )
    vis_btn.pack(side=tk.LEFT, padx=(0, 6))

    save_btn = ttk.Button(
        btn_row, text='保存特征矩阵(.npy)',
        command=lambda: _save_results(file_listbox, log_widget, output_dir_var)
    )
    save_btn.pack(side=tk.LEFT)

    clear_btn = ttk.Button(
        btn_row, text='清空列表',
        command=lambda: [file_listbox.delete(0, tk.END),
                         _processed_results.clear(),
                         file_listbox.__setattr__('result_paths', [])]
    )
    clear_btn.pack(side=tk.RIGHT)

    log_lf = ttk.LabelFrame(right_frame, text='处理日志', padding=4)
    log_lf.pack(fill=tk.BOTH, expand=True)

    log_widget = scrolledtext.ScrolledText(log_lf, state=tk.DISABLED,
                                           font=('Consolas', 8),
                                           bg='#1e1e1e', fg='#d4d4d4',
                                           insertbackground='white', height=12)
    log_widget.pack(fill=tk.BOTH, expand=True)

    # ================================================================
    # 运行按钮回调
    # ================================================================
    def _on_run():
        folder = folder_var.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror('错误', '请先选择有效的CSV文件夹')
            return

        # 解析参数
        try:
            params = {
                'pcg_col':     pcg_col_var.get().strip() or 'pcg',
                'fs':          float(fs_var.get()),
                'r':           int(r_var.get()),
                'low_hz':      float(low_var.get()),
                'high_hz':     float(high_var.get()),
                'window_ms':   float(win_var.get()),
                'overlap':     float(overlap_var.get()),
                'max_iter':    int(maxiter_var.get()),
                'use_temporal': feature_var.get() == 'temporal',
                'verbose':     False,
            }
        except ValueError as e:
            messagebox.showerror('参数错误', f'参数格式错误: {e}')
            return

        run_btn.config(state=tk.DISABLED)
        _log(log_widget, f'开始处理文件夹: {folder}')
        _log(log_widget, f'参数: {params}')

        t = threading.Thread(
            target=_process_files_thread,
            args=(folder, params, log_widget, status_var,
                  file_listbox, run_btn, root),
            daemon=True
        )
        t.start()

    run_btn.config(command=_on_run)

    # ---- 底部说明 ----
    ttk.Separator(root, orient='horizontal').pack(fill=tk.X, padx=10)
    footer = ttk.Label(
        root,
        text='Torre-Cruz et al. 2023 | Algorithm 1: PCG → 幅度归一化 → STFT → 谱归一化 → B_A频带截取 → ONMF (K=128, I=60) → W(9频域)/H(时域) → CNN',
        font=('', 8), foreground='gray'
    )
    footer.pack(pady=3)

    return root


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    root = build_gui()
    root.mainloop()


if __name__ == '__main__':
    main()
