"""
可视化工具：将ONMF预处理结果可视化，匹配论文 Figure 3 和 Figure 4 风格。

论文可视化规范（Torre-Cruz et al. 2023）：
  Figure 3: 线性幅度谱（Magnitude Spectrogram），非dB，时间(s)xHz
  Figure 4: W矩阵（频域基底，频率Hz x K）和H矩阵（时序激活，K x 时间s）
    - H行归一化便于观察周期规律（正常心音有规律，异常心音无规律）
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# 中文字体配置
def _setup_chinese_font():
    import matplotlib.font_manager as fm
    import sys
    candidates = ['Microsoft YaHei', 'SimHei', 'SimSun', 'STSong']
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            return
    if sys.platform == 'win32':
        import os
        fp = r'C:\Windows\Fonts\msyh.ttc'
        if os.path.exists(fp):
            prop = fm.FontProperties(fname=fp)
            plt.rcParams['font.sans-serif'] = [prop.get_name(), 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False

_setup_chinese_font()

# 论文Figure风格配色方案（论文Figure 4使用 hot：黑→红→橙→黄）
CMAP_SPEC = 'hot'   # 幅度谱热力图
CMAP_ONMF = 'hot'  # ONMF矩阵（W & H）


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _set_time_ticks(ax, t_arr, n_ticks=6):
    """为imshow坐标轴设置时间(s)刻度标签。"""
    n = len(t_arr)
    positions = np.linspace(0, n - 1, min(n_ticks, n), dtype=int)
    ax.set_xticks(positions)
    ax.set_xticklabels([f'{t_arr[p]:.1f}' for p in positions])


def _set_freq_ticks(ax, f_arr, n_ticks=6):
    """为imshow坐标轴设置频率(Hz)刻度标签。"""
    n = len(f_arr)
    positions = np.linspace(0, n - 1, min(n_ticks, n), dtype=int)
    ax.set_yticks(positions)
    ax.set_yticklabels([f'{f_arr[p]:.0f}' for p in positions])


def _row_normalize(M):
    """每行除以行最大值（用于H矩阵可视化，突出时序模式）。"""
    row_max = M.max(axis=1, keepdims=True)
    return M / np.maximum(row_max, 1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# 主综合图：对应论文 Figure 3 + Figure 4 风格
# ─────────────────────────────────────────────────────────────────────────────

def plot_all_results(result: dict,
                     filename: str = '',
                     fs: float = 4096.0,
                     show: bool = True,
                     save_path: str = None):
    """
    综合可视化（Figure 3 + Figure 4 风格），6个子图：

      行1：  带通滤波时域信号（全宽）
      行2左： 幅度谱 Magnitude Spectrogram（Figure 3，线性幅度）
      行2中： ONMF收敛误差曲线
      行2右： W 频域基底矩阵（Figure 4 A/C）
      行3左+中： H 时序激活矩阵（Figure 4 B/D，行归一化）
      行3右：  ONMF重建幅度谱 W@H（线性幅度）

    关键说明：
      - 幅度谱用线性值（非dB），与论文Figure 3一致
      - H矩阵行归一化：正常心音呈规律周期条纹，异常心音显示不规律激活
        这是论文的核心发现，也是CNN分类的依据
    """
    # 兼容新旧结果结构: 新版本返回 'pcg'，旧版本返回 'filtered'
    pcg_signal = result.get('pcg', result.get('filtered', None))
    S_band   = result['S_band']       # 已max归一化，线性幅度
    f_band   = result['f_band']
    f_full   = result.get('f_full', f_band)
    t_stft   = result['t']
    W        = result['W']
    H        = result['H']
    errors   = result['errors']

    if pcg_signal is not None:
        t_signal = np.arange(len(pcg_signal)) / fs
    band_str = f'{f_band[0]:.0f}-{f_band[-1]:.0f} Hz'
    K = W.shape[1]

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(
        f'ONMF 预处理结果  |  文件: {filename}\n'
        f'频带: [{band_str}]  |  K={K} 基底  |  STFT: {result.get("n_fft", "?")}点FFT',
        fontsize=12, fontweight='bold', y=0.98
    )
    gs = GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.40)

    # ── 行1：时域信号 ──────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    if pcg_signal is not None:
        ax1.plot(t_signal, pcg_signal, linewidth=0.5, color='#2196F3', alpha=0.85)
        ax1.set_xlim(t_signal[0], t_signal[-1])
    else:
        ax1.text(0.5, 0.5, '时域信号不可用', ha='center', transform=ax1.transAxes)
    preproc_desc = result.get('preproc_desc', '')
    ax1.set_title(f'PCG 信号  [{preproc_desc}]' if preproc_desc else 'PCG 信号（降采样+幅度归一化）', fontsize=10)
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('归一化幅度')
    ax1.grid(True, alpha=0.2, linestyle='--')

    # ── 行2左：幅度谱（Figure 3风格，线性幅度）──────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    im2 = ax2.imshow(
        S_band, aspect='auto', origin='lower',
        cmap=CMAP_SPEC, interpolation='nearest',
        vmin=0, vmax=np.percentile(S_band, 99)   # 去除极端值影响
    )
    ax2.set_title('幅度谱（Figure 3，线性，非dB）', fontsize=9)
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('频率 (Hz)')
    _set_time_ticks(ax2, t_stft)
    _set_freq_ticks(ax2, f_band)
    fig.colorbar(im2, ax=ax2, label='归一化幅度', shrink=0.9)

    # ── 行2中：ONMF收敛曲线 ──────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.semilogy(errors, color='#E53935', linewidth=1.8)
    ax3.set_title(f'ONMF 收敛曲线（{len(errors)}次）', fontsize=9)
    ax3.set_xlabel('迭代次数')
    ax3.set_ylabel('相对重建误差')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.text(len(errors) * 0.55, errors[-1] * 2,
             f'最终误差:\n{errors[-1]:.4f}', fontsize=8, color='#E53935')

    # ── 行2右：W 频域基底（Figure 4 A/C 风格）————W 嵌入全频轴 0-1000Hz ──────
    ax4 = fig.add_subplot(gs[1, 2])
    W_full4 = np.zeros((len(f_full), K))
    bm4 = (f_full >= f_band[0]) & (f_full <= f_band[-1])
    W_full4[bm4, :] = W
    w4_vmax = np.percentile(W_full4[bm4, :], 99) if W.max() > 0 else 1.0
    im4 = ax4.imshow(
        W_full4, aspect='auto', origin='lower',
        cmap=CMAP_ONMF, interpolation='nearest', vmin=0, vmax=w4_vmax,
        extent=[0, K, 0, float(f_full[-1])]
    )
    ax4.set_ylim(0, min(1000, float(f_full[-1])))
    ax4.set_title(
        f'W 频域基底（Figure 4 A/C）\n'
        f'[{W.shape[0]} Hz-bins × {K} 基底]  正常/异常W相似',
        fontsize=9
    )
    ax4.set_xlabel('基底索引 k')
    ax4.set_ylabel('Frequency(Hz)')
    k_ticks = np.linspace(0, K - 1, min(8, K), dtype=int)
    ax4.set_xticks(k_ticks)
    ax4.set_xticklabels([str(k) for k in k_ticks])
    fig.colorbar(im4, ax=ax4, shrink=0.9)

    # ── 行3左+中：H 时序激活（Figure 4 B/D风格，原始值）──────────────────────
    ax5 = fig.add_subplot(gs[2, :2])
    # vmin 设为 75 百分位：心跳间隔的低激活背景 → 黑色（与论文 Figure 4 一致）
    # vmax 设为 99.5 百分位：只有真正的心跳峰值才达到最亮
    h_vmin = float(np.percentile(H, 75))
    h_vmax = float(np.percentile(H, 99.5))
    if h_vmax <= h_vmin:
        h_vmin, h_vmax = 0.0, float(H.max()) or 1.0
    im5 = ax5.imshow(
        H, aspect='auto', origin='upper',
        cmap=CMAP_ONMF, interpolation='nearest',
        vmin=h_vmin, vmax=h_vmax
    )
    ax5.set_title(
        f'H 时序激活（Figure 4 B/D）→ CNN 主要输入特征（原始值）\n'
        f'[{K}基底 × {H.shape[1]}时间帧]'
        f'  正常心音=规律周期竖线  |  异常心音=不规律间距',
        fontsize=9
    )
    ax5.set_xlabel('时间 (s)')
    ax5.set_ylabel('基底索引 k')
    _set_time_ticks(ax5, t_stft)
    k_yticks = np.linspace(0, K - 1, min(8, K), dtype=int)
    ax5.set_yticks(k_yticks)
    ax5.set_yticklabels([str(k) for k in k_yticks])
    fig.colorbar(im5, ax=ax5, label='归一化激活强度', shrink=0.85)

    # ── 行3右：重建谱 W@H（线性幅度）──────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    V_recon = W @ H
    im6 = ax6.imshow(
        V_recon, aspect='auto', origin='lower',
        cmap=CMAP_SPEC, interpolation='nearest', vmin=0
    )
    ax6.set_title('重建幅度谱 W@H（线性幅度）', fontsize=9)
    ax6.set_xlabel('时间 (s)')
    ax6.set_ylabel('频率 (Hz)')
    _set_time_ticks(ax6, t_stft)
    _set_freq_ticks(ax6, f_band)
    fig.colorbar(im6, ax=ax6, shrink=0.9)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 严格复现：W（左）+ H（右）
# ─────────────────────────────────────────────────────────────────────────────

def plot_figure4_style(result: dict,
                       filename: str = '',
                       show: bool = True,
                       save_path: str = None):
    """
    严格对应论文 Figure 4：W矩阵（左）+ H矩阵（右，原始值）。

    论文发现：
      - W对正常/异常信号几乎相同（频谱模板相似）→ 频域特征区分度低
      - H差异显著：正常=周期竖线（心跳瞬间多基底同时激活）；异常=不规律 → CNN分类依据
    关键：H 显示原始值（非行归一化）。心跳瞬间多个基底同时激活 → 整列亮竖线；
    非心跳帧 → 接近零（暗色背景）。行归一化会放大背景噪声产生水平条纹，与论文不符。
    """
    W      = result['W']
    H      = result['H']
    f_band = result['f_band']
    f_full = result.get('f_full', f_band)   # 全频率轴 (0-Nyquist)
    t_stft = result['t']
    K      = W.shape[1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    band_str = f'{f_band[0]:.0f}-{f_band[-1]:.0f} Hz'
    fig.suptitle(
        f'ONMF W & H 矩阵（Figure 4 风格）  |  {filename}\n'
        f'频带: [{band_str}]  K={K}基底',
        fontsize=12, fontweight='bold'
    )

    # W 矩阵（Figure 4 A或C）
    # 将 W 嵌入全频轴 (0-Nyquist),  B_A 带外填零，与论文 Figure 4 Y 轴 0-1000Hz 一致
    ax = axes[0]
    W_full = np.zeros((len(f_full), K))
    band_mask = (f_full >= f_band[0]) & (f_full <= f_band[-1])
    W_full[band_mask, :] = W
    w_vmax = np.percentile(W_full[band_mask, :], 99) if W.max() > 0 else 1.0
    f_nyq = float(f_full[-1])
    im = ax.imshow(W_full, aspect='auto', origin='lower',
                   cmap=CMAP_ONMF, interpolation='nearest', vmin=0, vmax=w_vmax,
                   extent=[0, K, 0, f_nyq])
    ax.set_ylim(0, min(1000, f_nyq))  # 与论文 Figure 4 Y 轴 0-1000Hz 一致
    ax.set_title(
        f'W 频域基底  [{W.shape[0]} Hz-bins × {K}]\n'
        f'（论文：正常/异常W外观相似，频域特征区分度低）',
        fontsize=10
    )
    ax.set_xlabel('Bases(k)')
    ax.set_ylabel('Frequency(Hz)')
    k_ticks = np.linspace(0, K, min(10, K + 1), dtype=int)
    ax.set_xticks(k_ticks)
    ax.set_xticklabels([str(k) for k in k_ticks])
    fig.colorbar(im, ax=ax, label='基底幅度')

    # H 矩阵（Figure 4 B或D）——原始值，百分位对比度拉伸
    # vmin=75%ile → 背景黑色；vmax=99.5%ile → 只有心跳峰值达到最亮（论文薄线效果）
    ax2 = axes[1]
    t_lo, t_hi = float(t_stft[0]), float(t_stft[-1])
    h_vmin = float(np.percentile(H, 75))
    h_vmax = float(np.percentile(H, 99.5))
    if h_vmax <= h_vmin:
        h_vmin, h_vmax = 0.0, float(H.max()) or 1.0
    im2 = ax2.imshow(H, aspect='auto', origin='upper',
                     cmap=CMAP_ONMF, interpolation='nearest',
                     vmin=h_vmin, vmax=h_vmax,
                     extent=[t_lo, t_hi, K, 0])
    ax2.set_title(
        f'H 时序激活（原始值）  [{K} × {H.shape[1]}帧]\n'
        f'（正常→规律竖线；异常→不规律间距 ← CNN分类的核心特征）',
        fontsize=10
    )
    ax2.set_xlabel('Time(seconds)')
    ax2.set_ylabel('Bases(k)')
    k_step = max(1, K // 6)
    k_yticks = list(range(0, K, k_step)) + [K]
    ax2.set_yticks(k_yticks)
    ax2.set_yticklabels([str(k) for k in k_yticks])
    fig.colorbar(im2, ax=ax2, label='激活强度')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 严格复现：线性幅度谱（全频带 + 各频带对比）
# ─────────────────────────────────────────────────────────────────────────────

def plot_figure3_style(result: dict,
                       filename: str = '',
                       show: bool = True,
                       save_path: str = None):
    """
    对应论文 Figure 3：幅度谱（线性，非dB）对比 —— 全频带 vs 截取频带。

    Figure 3 描述：
      'Magnitude spectrogram associated with normal/abnormal heart sounds,
      analysing spectral bands B_C, B_N, B_A'
    本图显示当前处理结果的全频带谱与截取频带谱（对应Table 2其中一个频带）。
    """
    S_full = result.get('S_full', result['S_band'])
    f_full = result.get('f_full', result['f_band'])
    S_band = result['S_band']
    f_band = result['f_band']
    t_stft = result['t']
    band_str = f'{f_band[0]:.0f}-{f_band[-1]:.0f} Hz'

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f'幅度谱 Magnitude Spectrogram（Figure 3 风格）  |  {filename}',
        fontsize=12, fontweight='bold'
    )

    # 全频带谱
    ax1 = axes[0]
    clip_val = np.percentile(S_full, 99)
    im1 = ax1.imshow(S_full, aspect='auto', origin='lower',
                     cmap=CMAP_SPEC, interpolation='nearest',
                     vmin=0, vmax=clip_val)
    n_full = len(f_full)
    t_idx = np.linspace(0, len(t_stft) - 1, 6, dtype=int)
    ax1.set_xticks(t_idx)
    ax1.set_xticklabels([f'{t_stft[i]:.1f}' for i in t_idx])
    f_idx = np.linspace(0, n_full - 1, 7, dtype=int)
    ax1.set_yticks(f_idx)
    ax1.set_yticklabels([f'{f_full[i]:.0f}' for i in f_idx])
    ax1.set_title(f'B_F 全频带 [0-{f_full[-1]:.0f} Hz]\n（已最大值归一化，线性幅度）', fontsize=10)
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('频率 (Hz)')
    fig.colorbar(im1, ax=ax1, label='归一化幅度', shrink=0.95)

    # 截取频带谱
    ax2 = axes[1]
    clip_val2 = np.percentile(S_band, 99)
    im2 = ax2.imshow(S_band, aspect='auto', origin='lower',
                     cmap=CMAP_SPEC, interpolation='nearest',
                     vmin=0, vmax=clip_val2)
    _set_time_ticks(ax2, t_stft)
    _set_freq_ticks(ax2, f_band)
    ax2.set_title(f'频带截取 [{band_str}]\n（Table 2对应频带，ONMF输入）', fontsize=10)
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('频率 (Hz)')
    fig.colorbar(im2, ax=ax2, label='归一化幅度', shrink=0.95)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# W基底频谱曲线图
# ─────────────────────────────────────────────────────────────────────────────

def plot_w_bases(result: dict,
                 filename: str = '',
                 max_show: int = 32,
                 show: bool = True,
                 save_path: str = None):
    """将W各列（频谱基底）以频谱曲线 + 热力图双图显示。"""
    W      = result['W']
    f_band = result['f_band']
    K      = W.shape[1]
    show_k = min(K, max_show)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f'W 频谱基底分析  |  {filename}', fontsize=12)

    # 叠加频谱曲线
    ax1 = axes[0]
    cmap_l = plt.cm.tab20
    for i in range(show_k):
        c = cmap_l(i / max(show_k - 1, 1))
        ax1.plot(f_band, W[:, i], lw=0.9, alpha=0.7, color=c,
                 label=f'k={i}' if show_k <= 8 else None)
    ax1.set_title(f'W各列频谱曲线（前{show_k}/{K}个基底）\n每列=一个频谱基向量', fontsize=10)
    ax1.set_xlabel('频率 (Hz)')
    ax1.set_ylabel('归一化幅度')
    ax1.grid(True, alpha=0.25)
    if show_k <= 8:
        ax1.legend(fontsize=8)

    # 热力图
    ax2 = axes[1]
    im = ax2.imshow(W[:, :show_k], aspect='auto', origin='lower',
                    cmap=CMAP_ONMF, interpolation='nearest', vmin=0)
    ax2.set_title(f'W 热力图  [{W.shape[0]} Hz-bins × {show_k}]', fontsize=10)
    ax2.set_xlabel('基底索引 k')
    ax2.set_ylabel('频率 (Hz)')
    _set_freq_ticks(ax2, f_band)
    fig.colorbar(im, ax=ax2)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 兼容接口
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_matrix(feature: np.ndarray,
                        title: str = 'ONMF特征矩阵（CNN输入）',
                        label_row: str = '基底索引 k',
                        label_col: str = '时间帧',
                        show: bool = True,
                        save_path: str = None):
    """单独可视化CNN输入特征矩阵（W或H），热力图+曲线双图。"""
    vis = _row_normalize(feature)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=12)

    im = axes[0].imshow(vis, aspect='auto', cmap=CMAP_ONMF,
                        origin='lower', vmin=0, vmax=1)
    axes[0].set_title(f'热力图（行归一化）  [{feature.shape[0]}×{feature.shape[1]}]')
    axes[0].set_xlabel(label_col)
    axes[0].set_ylabel(label_row)
    fig.colorbar(im, ax=axes[0])

    for i in range(min(feature.shape[0], 16)):
        axes[1].plot(vis[i], lw=0.8, alpha=0.7, label=f'k={i}')
    axes[1].set_title('各基底激活曲线（行归一化）')
    axes[1].set_xlabel(label_col)
    axes[1].set_ylabel('归一化激活')
    axes[1].grid(True, alpha=0.3)
    if feature.shape[0] <= 8:
        axes[1].legend(fontsize=7)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 简洁视图：PCG时域 + 幅度谱 + H时序激活（无色条）
# ─────────────────────────────────────────────────────────────────────────────

def plot_clean_view(result: dict,
                   filename: str = '',
                   fs: float = 4096.0,
                   show: bool = True,
                   save_path: str = None):
    """
    简洁三图视图（无色条）：
      上：PCG 时域信号（全宽）
      左下：幅度谱 S_band（Figure 3 风格，线性幅度）
      右下：H 时序激活矩阵（Figure 4 B/D 风格，细线效果）
    """
    from matplotlib.gridspec import GridSpec

    pcg_signal = result.get('pcg', result.get('filtered', None))
    S_band = result['S_band']
    f_band = result['f_band']
    H      = result['H']
    t_stft = result['t']
    K      = H.shape[0]
    band_str = f'{f_band[0]:.0f}-{f_band[-1]:.0f} Hz'

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(
        f'ONMF 预处理结果  |  {filename}  |  频带 [{band_str}]  K={K}',
        fontsize=12, fontweight='bold', y=0.99
    )
    gs = GridSpec(2, 2, figure=fig,
                  height_ratios=[1, 2],
                  hspace=0.40, wspace=0.25)

    # ── 上：PCG 时域信号（全宽）
    ax1 = fig.add_subplot(gs[0, :])
    if pcg_signal is not None:
        t_sig = np.arange(len(pcg_signal)) / fs
        ax1.plot(t_sig, pcg_signal, lw=0.6, color='#2196F3', alpha=0.9)
        ax1.set_xlim(t_sig[0], t_sig[-1])
    else:
        ax1.text(0.5, 0.5, '时域信号不可用', ha='center', transform=ax1.transAxes)
    preproc_desc = result.get('preproc_desc', '')
    ax1_title = f'PCG  [{preproc_desc}]' if preproc_desc else 'PCG'
    ax1.set_title(ax1_title, fontsize=10)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.2, linestyle='--')
    ax1.spines[['top', 'right']].set_visible(False)

    # ── 左下：幅度谱（Figure 3 风格）
    ax2 = fig.add_subplot(gs[1, 0])
    s_vmax = float(np.percentile(S_band, 99))
    ax2.imshow(
        S_band, aspect='auto', origin='lower',
        cmap=CMAP_SPEC, interpolation='nearest',
        vmin=0, vmax=s_vmax
    )
    ax2.set_title(f'幅度频谱图\n[{band_str}]', fontsize=10)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Frequency (Hz)')
    _set_time_ticks(ax2, t_stft)
    _set_freq_ticks(ax2, f_band)
    ax2.spines[['top', 'right']].set_visible(False)

    # ── 右下：H 时序激活（行归一化热力图，与 CNN 输入一致）
    ax3 = fig.add_subplot(gs[1, 1])
    H_vis = _row_normalize(H)          # 每行独立归一化到 [0, 1]
    t_lo, t_hi = float(t_stft[0]), float(t_stft[-1])
    ax3.imshow(
        H_vis, aspect='auto', origin='lower',
        cmap=CMAP_ONMF, interpolation='nearest',
        vmin=0, vmax=1,
        extent=[t_lo, t_hi, 0, K]
    )
    ax3.set_title(
        f'激活矩阵H [{K} × {H.shape[1]} frames]\n',
        fontsize=10
    )
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Bases (k)')
    k_step = max(1, K // 6)
    k_ticks = list(range(0, K + 1, k_step))
    ax3.set_yticks(k_ticks)
    ax3.set_yticklabels([str(k) for k in k_ticks])
    ax3.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig
