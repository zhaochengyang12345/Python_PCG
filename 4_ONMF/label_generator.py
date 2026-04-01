"""
PCG 标签生成工具
================
根据 *_H.npy 文件名前缀自动推断标签，支持可视化确认和手动调整后导出 CSV。

使用规则（可在界面中调整）：
  前缀 N   → 0（正常）
  前缀 AS/MR/MS/MVP/AR 等 → 1（异常）

输出 CSV 格式（与 PhysioNet REFERENCE.csv 兼容）：
  列1: 文件名 stem（不含扩展名和 _H/_W 后缀）
  列2: 标签（0=正常, 1=异常）
"""

import os
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm


# ── 中文字体 ──────────────────────────────────────────────────────────────
def _setup_font():
    for font in ['Microsoft YaHei', 'SimHei']:
        if font in {f.name for f in _fm.fontManager.ttflist}:
            plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            return
_setup_font()


# ── 默认规则（前缀 → 标签） ───────────────────────────────────────────────
DEFAULT_ABNORMAL = {'AS', 'MR', 'MS', 'MVP', 'AR', 'VSD', 'HCM', 'ASD', 'PDA'}
DEFAULT_NORMAL   = {'N', 'NORMAL', 'NOR'}


def detect_prefix(stem: str) -> str:
    """从文件名 stem 中提取字母前缀（如 AS001 → AS）。"""
    m = re.match(r'^([A-Za-z]+)', stem)
    return m.group(1).upper() if m else ''


def infer_label(prefix: str, normal_set: set, abnormal_set: set) -> int | None:
    p = prefix.upper()
    if p in normal_set:   return 0
    if p in abnormal_set: return 1
    return None


def scan_directory(npy_dir: str):
    """
    扫描目录，返回所有 *_H.npy 文件的 stem 列表和前缀统计。
    stem: 去掉 _H.npy 的文件名，如 AS001
    """
    npy_dir = Path(npy_dir)
    files = sorted(npy_dir.glob('*_H.npy'))
    stems = [f.stem.replace('_H', '') for f in files]
    prefixes = [detect_prefix(s) for s in stems]
    prefix_counts = Counter(prefixes)
    return stems, prefixes, prefix_counts


# ──────────────────────────────────────────────────────────────────────────────
# 主 GUI
# ──────────────────────────────────────────────────────────────────────────────

class LabelGeneratorGUI:

    def __init__(self, root):
        self.root = root
        self.root.title('PCG 标签生成工具')
        self.root.geometry('1000x680')
        self.root.resizable(True, True)

        self.npy_dir = tk.StringVar()
        self.prefix_labels: dict[str, tk.IntVar] = {}   # prefix → IntVar(0/1)
        self.stems   = []
        self.prefixes = []
        self.prefix_counts = Counter()

        self._build_ui()

    # ── UI 构建 ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = self.root

        # 标题
        tk.Label(root, text='PCG 标签生成工具',
                 font=('Microsoft YaHei', 14, 'bold'),
                 fg='#1a5276', pady=8).pack(fill=tk.X)
        tk.Label(root,
                 text='根据文件名前缀自动分配标签，可视化确认后导出 CSV',
                 font=('Microsoft YaHei', 9), fg='#555').pack()
        ttk.Separator(root, orient='horizontal').pack(fill=tk.X, pady=4)

        main = ttk.Frame(root)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        left = ttk.Frame(main, width=350)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))
        left.pack_propagate(False)

        right = ttk.Frame(main)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ── 左侧：目录 + 规则 ─────────────────────────────────────────────

        # 目录选择
        dir_lf = ttk.LabelFrame(left, text='特征目录', padding=8)
        dir_lf.pack(fill=tk.X, pady=(0, 6))

        ttk.Entry(dir_lf, textvariable=self.npy_dir, width=28).pack(
            side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(dir_lf, text='浏览', width=6,
                   command=self._browse).pack(side=tk.LEFT, padx=(4, 0))

        ttk.Button(left, text='🔍  扫描目录',
                   command=self._scan).pack(fill=tk.X, pady=(0, 6))

        # 前缀-标签规则表
        self.rule_lf = ttk.LabelFrame(left, text='前缀 → 标签规则（点击切换）', padding=8)
        self.rule_lf.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(self.rule_lf,
                  text='（扫描后自动填充）',
                  foreground='gray').pack()

        # 统计摘要
        self.stat_var = tk.StringVar(value='请先扫描目录')
        ttk.Label(left, textvariable=self.stat_var,
                  foreground='#1a5276', font=('Consolas', 9),
                  wraplength=330, justify='left').pack(anchor='w', pady=4)

        # 导出按钮
        ttk.Button(left, text='💾  导出标签 CSV',
                   command=self._export).pack(fill=tk.X, pady=2)
        ttk.Button(left, text='👁  查看随机样本（H矩阵）',
                   command=self._preview_samples).pack(fill=tk.X, pady=2)

        # ── 右侧：文件列表 ──────────────────────────────────────────────────
        list_lf = ttk.LabelFrame(right, text='文件列表（双击查看H矩阵）', padding=4)
        list_lf.pack(fill=tk.BOTH, expand=True)

        cols = ('stem', 'prefix', 'label')
        self.tree = ttk.Treeview(list_lf, columns=cols, show='headings',
                                 selectmode='extended')
        self.tree.heading('stem',   text='文件名（stem）')
        self.tree.heading('prefix', text='前缀')
        self.tree.heading('label',  text='标签（0=正常, 1=异常）')
        self.tree.column('stem',   width=280)
        self.tree.column('prefix', width=80,  anchor='center')
        self.tree.column('label',  width=120, anchor='center')
        self.tree.tag_configure('normal',   background='#e8f5e9')
        self.tree.tag_configure('abnormal', background='#fce4ec')
        self.tree.tag_configure('unknown',  background='#fff9c4')

        vsb = ttk.Scrollbar(list_lf, orient='vertical',
                            command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind('<Double-1>', self._on_double_click)

        # 右键菜单：批量修改标签
        self.ctx_menu = tk.Menu(root, tearoff=0)
        self.ctx_menu.add_command(label='标记为 0（正常）',
                                  command=lambda: self._set_selected_label(0))
        self.ctx_menu.add_command(label='标记为 1（异常）',
                                  command=lambda: self._set_selected_label(1))
        self.tree.bind('<Button-3>', self._on_right_click)

        # 底部说明
        ttk.Separator(root, orient='horizontal').pack(fill=tk.X, padx=10)
        ttk.Label(root,
                  text='提示: 右键可批量修改选中行的标签 | 双击行查看该文件 H 矩阵',
                  font=('', 8), foreground='gray').pack(pady=3)

    # ── 目录浏览 ─────────────────────────────────────────────────────────────

    def _browse(self):
        d = filedialog.askdirectory(title='选择含 *_H.npy 的目录')
        if d:
            self.npy_dir.set(d)
            self._scan()

    # ── 扫描目录 ─────────────────────────────────────────────────────────────

    def _scan(self):
        npy_dir = self.npy_dir.get().strip()
        if not npy_dir or not os.path.isdir(npy_dir):
            messagebox.showerror('错误', '请先选择有效的特征目录')
            return

        self.stems, self.prefixes, self.prefix_counts = scan_directory(npy_dir)

        if not self.stems:
            messagebox.showwarning('提示', '目录中未找到 *_H.npy 文件')
            return

        # 重建规则面板
        for w in self.rule_lf.winfo_children():
            w.destroy()

        self.prefix_labels.clear()
        for prefix, count in sorted(self.prefix_counts.items()):
            # 自动推断初始标签
            if prefix in DEFAULT_NORMAL:
                init = 0
            elif prefix in DEFAULT_ABNORMAL:
                init = 1
            else:
                init = 1   # 不认识的前缀默认为异常，用户可手动改

            var = tk.IntVar(value=init)
            self.prefix_labels[prefix] = var

            row = ttk.Frame(self.rule_lf)
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=f'{prefix}  ({count}个)',
                      width=16).pack(side=tk.LEFT)
            ttk.Radiobutton(row, text='0 正常', variable=var, value=0,
                            command=self._refresh_tree).pack(side=tk.LEFT)
            ttk.Radiobutton(row, text='1 异常', variable=var, value=1,
                            command=self._refresh_tree).pack(side=tk.LEFT, padx=6)

        self._refresh_tree()

    # ── 刷新文件列表 ─────────────────────────────────────────────────────────

    def _refresh_tree(self):
        self.tree.delete(*self.tree.get_children())

        n0 = n1 = n_unk = 0
        for stem, prefix in zip(self.stems, self.prefixes):
            lbl_var = self.prefix_labels.get(prefix)
            if lbl_var is not None:
                lbl = lbl_var.get()
                lbl_str = f'{lbl}  ({"正常" if lbl == 0 else "异常"})'
                tag = 'normal' if lbl == 0 else 'abnormal'
                if lbl == 0: n0 += 1
                else: n1 += 1
            else:
                lbl_str = '?  (未知前缀)'
                tag = 'unknown'
                n_unk += 1

            self.tree.insert('', 'end', iid=stem,
                             values=(stem, prefix, lbl_str), tags=(tag,))

        total = len(self.stems)
        self.stat_var.set(
            f'共 {total} 个文件\n'
            f'  正常 (0): {n0}  ({n0/total*100:.1f}%)\n'
            f'  异常 (1): {n1}  ({n1/total*100:.1f}%)\n'
            + (f'  未知前缀: {n_unk}\n' if n_unk else '') +
            f'\n未知前缀请右键手动标注'
        )

    # ── 双击查看 H 矩阵 ───────────────────────────────────────────────────────

    def _on_double_click(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        stem = sel[0]
        npy_path = Path(self.npy_dir.get()) / f'{stem}_H.npy'
        if not npy_path.exists():
            messagebox.showwarning('提示', f'文件不存在: {npy_path}')
            return
        self._show_h_matrix(npy_path, stem)

    def _show_h_matrix(self, npy_path, stem):
        H = np.load(str(npy_path)).astype(np.float32)
        plt.close('all')
        fig, ax = plt.subplots(figsize=(10, 4))
        h_vmin = float(np.percentile(H, 75))
        h_vmax = float(np.percentile(H, 99.5))
        if h_vmax <= h_vmin:
            h_vmin, h_vmax = 0.0, float(H.max()) or 1.0
        ax.imshow(H, aspect='auto', origin='lower',
                  cmap='hot', interpolation='nearest',
                  vmin=h_vmin, vmax=h_vmax)
        prefix = detect_prefix(stem)
        lbl_var = self.prefix_labels.get(prefix)
        lbl_text = f"标签: {'0 正常' if lbl_var and lbl_var.get()==0 else '1 异常'}"
        ax.set_title(f'H 矩阵  |  {stem}  |  {lbl_text}  |  shape={H.shape}')
        ax.set_xlabel('时间帧')
        ax.set_ylabel('基底 k')
        plt.tight_layout()
        plt.show()

    # ── 预览随机样本 ──────────────────────────────────────────────────────────

    def _preview_samples(self):
        if not self.stems:
            messagebox.showinfo('提示', '请先扫描目录')
            return
        npy_dir = Path(self.npy_dir.get())
        rng = np.random.default_rng()
        idxs = rng.choice(len(self.stems), min(8, len(self.stems)), replace=False)
        cols = 4
        rows = (len(idxs) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3))
        axes = np.array(axes).flatten()
        fig.suptitle('随机样本 H 矩阵预览', fontsize=12)
        for i, idx in enumerate(idxs):
            stem = self.stems[idx]
            prefix = self.prefixes[idx]
            lbl_var = self.prefix_labels.get(prefix)
            lbl = lbl_var.get() if lbl_var else -1
            npy_path = npy_dir / f'{stem}_H.npy'
            ax = axes[i]
            if npy_path.exists():
                H = np.load(str(npy_path)).astype(np.float32)
                h_vmin = float(np.percentile(H, 75))
                h_vmax = float(np.percentile(H, 99.5))
                if h_vmax <= h_vmin: h_vmin, h_vmax = 0.0, float(H.max()) or 1.0
                ax.imshow(H, aspect='auto', origin='lower', cmap='hot',
                          interpolation='nearest', vmin=h_vmin, vmax=h_vmax)
                color = '#388E3C' if lbl == 0 else '#C62828'
                lbl_txt = '正常' if lbl == 0 else '异常'
                ax.set_title(f'{stem}\n[{lbl_txt}]', fontsize=8, color=color)
            else:
                ax.text(0.5, 0.5, '文件不存在', ha='center', transform=ax.transAxes)
            ax.axis('off')
        for i in range(len(idxs), len(axes)):
            axes[i].axis('off')
        plt.tight_layout()
        plt.close('all')
        plt.show()

    # ── 右键批量改标签 ────────────────────────────────────────────────────────

    def _on_right_click(self, event):
        self.ctx_menu.post(event.x_root, event.y_root)

    def _set_selected_label(self, new_label: int):
        sel = self.tree.selection()
        if not sel:
            return
        # 找出选中行的前缀，批量设置
        affected_prefixes = set()
        stems_to_update = []
        for stem in sel:
            prefix = self.tree.set(stem, 'prefix')
            stems_to_update.append((stem, prefix))
            affected_prefixes.add(prefix)

        # 更新前缀规则
        for prefix in affected_prefixes:
            if prefix in self.prefix_labels:
                self.prefix_labels[prefix].set(new_label)

        # 直接刷新 treeview 中受影响行（同前缀的全部行都会刷新）
        self._refresh_tree()

    # ── 导出 CSV ──────────────────────────────────────────────────────────────

    def _export(self):
        if not self.stems:
            messagebox.showinfo('提示', '请先扫描目录')
            return

        # 检查是否有未知标签
        unknowns = [s for s, p in zip(self.stems, self.prefixes)
                    if p not in self.prefix_labels]
        if unknowns:
            if not messagebox.askyesno(
                '未知标签',
                f'有 {len(unknowns)} 个文件前缀未设置规则，将被跳过（不导出标签）。\n继续吗？'
            ):
                return

        # 默认保存到特征目录下的 REFERENCE.csv
        default_path = str(Path(self.npy_dir.get()) / 'REFERENCE.csv')
        save_path = filedialog.asksaveasfilename(
            title='保存标签CSV',
            initialfile='REFERENCE.csv',
            initialdir=self.npy_dir.get(),
            defaultextension='.csv',
            filetypes=[('CSV文件', '*.csv'), ('所有文件', '*.*')]
        )
        if not save_path:
            return

        rows = []
        n0 = n1 = n_skip = 0
        for stem, prefix in zip(self.stems, self.prefixes):
            lbl_var = self.prefix_labels.get(prefix)
            if lbl_var is None:
                n_skip += 1
                continue
            lbl = lbl_var.get()
            rows.append(f'{stem},{lbl}')
            if lbl == 0: n0 += 1
            else: n1 += 1

        with open(save_path, 'w', encoding='utf-8', newline='') as f:
            f.write('filename,label\n')
            f.write('\n'.join(rows))
            if rows:
                f.write('\n')

        msg = (f'已导出 {len(rows)} 条标签\n'
               f'  正常 (0): {n0}\n'
               f'  异常 (1): {n1}\n'
               + (f'  跳过未知: {n_skip}\n' if n_skip else '') +
               f'\n保存位置:\n{save_path}')
        messagebox.showinfo('导出成功', msg)

        # 显示 CNN 使用提示
        self._show_usage_hint(save_path)

    def _show_usage_hint(self, csv_path):
        hint = (
            f'标签CSV已生成，在 CNN 训练界面中：\n\n'
            f'  → 标签CSV路径填写：\n    {csv_path}\n\n'
            f'注意：CSV 第1列为文件名（如 AS001），\n'
            f'第2列为标签（0=正常，1=异常）。\n'
            f'cnn_dataset.py 将自动匹配 *_H.npy 文件。'
        )
        messagebox.showinfo('使用提示', hint)


# ──────────────────────────────────────────────────────────────────────────────
# 命令行快速生成（无GUI）
# ──────────────────────────────────────────────────────────────────────────────

def auto_generate_labels(npy_dir: str, output_csv: str = None,
                         normal_prefixes: set = None,
                         abnormal_prefixes: set = None,
                         verbose: bool = True) -> str:
    """
    命令行模式：自动扫描目录，按前缀规则生成标签 CSV。

    Parameters
    ----------
    npy_dir           : 含 *_H.npy 的目录
    output_csv        : 输出路径，默认 npy_dir/REFERENCE.csv
    normal_prefixes   : 正常类前缀集合，默认 DEFAULT_NORMAL
    abnormal_prefixes : 异常类前缀集合，默认 DEFAULT_ABNORMAL
    verbose           : 是否打印统计

    Returns
    -------
    output_csv 路径
    """
    n_set = normal_prefixes   or DEFAULT_NORMAL
    a_set = abnormal_prefixes or DEFAULT_ABNORMAL

    stems, prefixes, counts = scan_directory(npy_dir)
    if not stems:
        raise FileNotFoundError(f'未找到 *_H.npy 文件: {npy_dir}')

    if output_csv is None:
        output_csv = str(Path(npy_dir) / 'REFERENCE.csv')

    rows, n0, n1, n_unk = [], 0, 0, 0
    for stem, prefix in zip(stems, prefixes):
        lbl = infer_label(prefix, n_set, a_set)
        if lbl is None:
            n_unk += 1
            if verbose:
                print(f'  [未知前缀] {stem}  前缀={prefix}')
            continue
        rows.append(f'{stem},{lbl}')
        if lbl == 0: n0 += 1
        else: n1 += 1

    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        f.write('filename,label\n')
        f.write('\n'.join(rows) + '\n')

    if verbose:
        print(f'标签CSV已生成: {output_csv}')
        print(f'  正常 (0): {n0}')
        print(f'  异常 (1): {n1}')
        if n_unk:
            print(f'  跳过未知: {n_unk}')

    return output_csv


# ──────────────────────────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import sys

    # 命令行模式: python label_generator.py <npy_dir> [output.csv]
    if len(sys.argv) >= 2 and '--gui' not in sys.argv:
        npy_dir = sys.argv[1]
        out = sys.argv[2] if len(sys.argv) >= 3 else None
        auto_generate_labels(npy_dir, out)
        return

    # GUI 模式
    root = tk.Tk()
    app = LabelGeneratorGUI(root)

    # 若命令行传入目录则自动扫描
    if len(sys.argv) >= 2:
        app.npy_dir.set(sys.argv[1])
        root.after(200, app._scan)

    root.mainloop()


if __name__ == '__main__':
    main()
