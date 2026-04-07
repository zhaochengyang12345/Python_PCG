"""
ONMF (Orthogonal Non-negative Matrix Factorization) PCG Preprocessing
Torre-Cruz et al. 2023 - Algorithm 1 strict implementation

论文参数 (Section 4.2, Table 2):
  fs_target = 4096 Hz  (数据先降采样到此频率)
  STFT: N=128 samples, hop=32 samples (25% hop), n_fft=1024 (4Hz resolution)
  Frequency band B_A = [200, 700] Hz  (optimal, Table 2)
  ONMF rank K=128, iterations I=60
  Init: random positive  (Algorithm 1 Step 4)
"""

import warnings
import numpy as np
from scipy import signal as scipy_signal


# ---------------------------------------------------------------------------
# 论文 Table 2 四种频带定义
# ---------------------------------------------------------------------------
FREQ_BANDS = {
    'B_F': (20, 2048),   # 全频带
    'B_C': (20,  700),   # 正常+异常综合带
    'B_N': (20,  200),   # 正常心音主能量带
    'B_A': (200, 700),   # 异常心音主能量带 (论文最优)
}

# 论文固定参数
TARGET_FS   = 4096.0   # 目标处理采样率 (Hz)
WIN_SAMPLES = 128      # STFT 窗口样本数  (31.25 ms @ 4096 Hz)
HOP_SAMPLES = 32       # STFT Hop 样本数  (25% hop, 7.81 ms)
N_FFT       = 1024     # FFT 点数 (零填充, 4 Hz 频率分辨率)
K_DEFAULT   = 128      # ONMF 秩 K
I_DEFAULT   = 60       # ONMF 迭代次数


# ---------------------------------------------------------------------------
# 内部工具函数
# ---------------------------------------------------------------------------

def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return p


def _normalize_columns(W: np.ndarray, eps: float = 1e-10):
    norms = np.linalg.norm(W, axis=0, keepdims=True)
    return W / np.maximum(norms, eps), norms.flatten()


# ---------------------------------------------------------------------------
# 时域信号预处理：Hampel 去刺 + 工频陷波 + Butterworth 带通
# ---------------------------------------------------------------------------

def hampel_filter(x: np.ndarray, half_window: int = 50,
                  n_sigma: float = 3.0) -> np.ndarray:
    """
    Hampel 滑动窗口尖峰检测与替换（向量化实现）。

    half_window : 滑动窗口半径（样本数），默认 50 ≈ 12 ms @ 4096 Hz
    n_sigma     : 判断阈值（n × 1.4826 × MAD），默认 3σ

    注意：此处用 median_filter(|x - med|) 作为逐窗 MAD 的快速近似，
    与精确逐点 Hampel 效果相当，速度快约 100×。
    """
    from scipy.ndimage import median_filter
    x = x.copy()
    k = 1.4826          # 正态分布下 MAD → 标准差一致估计量
    win = 2 * half_window + 1
    med = median_filter(x, size=win, mode='nearest')
    mad = k * median_filter(np.abs(x - med), size=win, mode='nearest')
    spikes = (mad > 0) & (np.abs(x - med) > n_sigma * mad)
    x[spikes] = med[spikes]
    return x


def apply_notch_filters(x: np.ndarray, fs: float,
                        freqs=(50.0, 100.0), Q: float = 30.0) -> np.ndarray:
    """工频陷波：去除 50 Hz 及其谐波（电网干扰）。"""
    nyq = fs / 2.0
    for f0 in freqs:
        if 0 < f0 < nyq:
            b, a = scipy_signal.iirnotch(f0, Q, fs)
            x = scipy_signal.filtfilt(b, a, x)
    return x


def apply_bandpass_filter(x: np.ndarray, fs: float,
                          low_hz: float = 20.0, high_hz: float = 400.0,
                          order: int = 4) -> np.ndarray:
    """4 阶 Butterworth 带通滤波，去除基线漂移与高频噪声。"""
    nyq = fs / 2.0
    low  = max(low_hz, 0.5) / nyq
    high = min(high_hz, nyq * 0.99) / nyq
    if low >= high:
        return x
    b, a = scipy_signal.butter(order, [low, high], btype='band')
    return scipy_signal.filtfilt(b, a, x)


# ---------------------------------------------------------------------------
# Stage 1: STFT + 频带截取
# ---------------------------------------------------------------------------

def compute_stft(pcg: np.ndarray, fs: float,
                 win_samples: int = WIN_SAMPLES,
                 hop_samples: int = HOP_SAMPLES,
                 n_fft: int = N_FFT):
    """
    Algorithm 1 Step 2: 计算 STFT 幅度谱
    窗函数: Hamming, 与论文一致
    """
    noverlap = win_samples - hop_samples
    f, t, Zxx = scipy_signal.stft(
        pcg, fs=fs,
        window='hamming',
        nperseg=win_samples,
        noverlap=noverlap,
        nfft=n_fft,
        return_onesided=True
    )
    return f, t, np.abs(Zxx)


def apply_freq_band_mask(f: np.ndarray, S: np.ndarray,
                         low_hz: float, high_hz: float):
    """
    Algorithm 1 Step 3: 在谱域截取频带 (非时域滤波)
    对应论文 Table 2 的 B_A / B_C / B_N / B_F
    """
    mask = (f >= low_hz) & (f <= high_hz)
    if not np.any(mask):
        raise ValueError(
            f"频带 [{low_hz}, {high_hz}] Hz 无有效频率点。"
            f"STFT 范围: [{f[0]:.1f}, {f[-1]:.1f}] Hz"
        )
    return f[mask], S[mask, :]


# ---------------------------------------------------------------------------
# Stage 2: ONMF (Orthogonal Non-negative Matrix Factorization)
# ---------------------------------------------------------------------------

def onmf(V: np.ndarray, r: int,
         max_iter: int = I_DEFAULT,
         tol: float = 1e-6,
         random_state: int = 42):
    """
    Algorithm 1 Steps 4-7 (Torre-Cruz et al. 2023):
      V ≈ W @ H,  W≥0, H≥0, W^T W ≈ I (W 列正交)

    Step 4: 随机正数初始化 W, H
    Step 5: 更新 W  (乘法 + QR 正交化 + 非负投影)
    Step 6: 更新 H  (标准乘法更新)
    Step 7: 重复至收敛

    参考: Choi 2008; Torre-Cruz et al. 2023 Eq.(4)-(5)
    """
    rng = np.random.default_rng(random_state)
    n, m = V.shape

    r_eff = min(r, n, m)
    if r_eff < r:
        warnings.warn(
            f"秩 r={r} 超过矩阵维度 min({n},{m})={r_eff}, 已自动调整。"
            f"提示: 增大 n_fft 或扩展频带可增加频率 bins。"
        )
        r = r_eff

    # Step 4: 随机正数初始化 (论文 Algorithm 1)
    W = rng.random((n, r)) + 1e-6
    H = rng.random((r, m)) + 1e-6
    W, _ = _normalize_columns(W)

    eps = 1e-10
    V_norm = np.linalg.norm(V, 'fro') + eps
    errors = []

    for _ in range(max_iter):
        # Step 6: 更新 H
        WtV  = W.T @ V
        WtWH = (W.T @ W) @ H
        H = H * (WtV / (WtWH + eps))
        H = np.maximum(H, eps)

        # Step 5: 更新 W (乘法)
        VHt  = V @ H.T
        WHHt = W @ (H @ H.T)
        W = W * (VHt / (WHHt + eps))
        W = np.maximum(W, eps)

        # Step 5: QR 正交化 + 非负投影
        Q, R = np.linalg.qr(W)
        signs = np.sign(np.diag(R))
        signs[signs == 0] = 1.0
        W_orth = np.maximum(Q * signs[np.newaxis, :], 0.0)

        # 死亡基底恢复: 回退到归一化前的 W
        dead = np.all(W_orth < eps, axis=0)
        if dead.any():
            W_fb, _ = _normalize_columns(W)
            W_orth[:, dead] = W_fb[:, dead]

        W, _ = _normalize_columns(W_orth)

        # Step 7: 检查收敛
        error = np.linalg.norm(V - W @ H, 'fro') / V_norm
        errors.append(float(error))
        if len(errors) > 1:
            if abs(errors[-2] - error) / (abs(errors[-2]) + eps) < tol:
                break

    return W, H, errors


# ---------------------------------------------------------------------------
# 完整预处理流水线 (Algorithm 1)
# ---------------------------------------------------------------------------

def preprocess_pcg_to_onmf(
    pcg: np.ndarray,
    fs: float = TARGET_FS,
    r: int = K_DEFAULT,
    low_hz: float = 200.0,
    high_hz: float = 700.0,
    win_samples: int = WIN_SAMPLES,
    hop_samples: int = HOP_SAMPLES,
    n_fft: int = N_FFT,
    max_iter: int = I_DEFAULT,
    use_temporal: bool = True,
    random_state: int = 42,
    resample_to: float = TARGET_FS,
    # 兼容旧接口
    window_size_ms: float = None,
    overlap_ratio: float = None,
    hop_ratio: float = None,
    # 时域预处理参数
    apply_hampel: bool = True,
    hampel_half_window: int = 50,
    hampel_sigma: float = 3.0,
    apply_notch: bool = True,
    notch_freqs: tuple = (50.0, 100.0),
    notch_Q: float = 30.0,
    apply_time_bandpass: bool = True,
    bp_low_hz: float = 20.0,
    bp_high_hz: float = 400.0,
    bp_order: int = 4,
    # 以下参数保留签名但忽略 (已移除)
    normalize_signal: bool = True,
    denoise: bool = False,
    spike_sigma: float = 5.0,
    sparsity: float = 0.0,
    spec_subtract: bool = False,
    noise_percentile: float = 10.0,
    over_sub: float = 1.0,
) -> dict:
    """
    CNN 输入前的完整预处理流水线 (Algorithm 1, Torre-Cruz et al. 2023):

      PCG
        → (降采样到 4096 Hz)
        → Hampel 尖峰去除（可选，默认开启）
        → 工频陷波 50/100 Hz（可选，默认开启）
        → 带通滤波 20–400 Hz（可选，默认开启）
        → 幅度归一化
        → STFT 幅度谱  [N=128, hop=32, n_fft=1024, Hamming]
        → 谱最大值归一化  X ← X / max(X)
        → 频域频带截取  B_A=[200,700] Hz
        → 随机初始化 ONMF (K=128, I=60)
        → 返回 W (频谱基底) 和 H (时序激活)

    Parameters
    ----------
    pcg          : 1D PCG 信号
    fs           : 输入采样率 (Hz)
    r            : ONMF 秩 K, 论文最优 128
    low_hz       : 频带下限 Hz  (B_A: 200)
    high_hz      : 频带上限 Hz  (B_A: 700)
    win_samples  : STFT 窗口样本数 (在目标 fs 下), 论文 N=128
    hop_samples  : STFT hop 样本数, 论文 32 (25% hop)
    n_fft        : FFT 点数, 论文 1024
    max_iter     : ONMF 最大迭代次数, 论文 I=60
    use_temporal : True→输出 H (时域激活); False→输出 W (频域基底)
    resample_to  : 目标处理采样率, None=跳过降采样

    Returns
    -------
    dict:
      'W'          : (n_band_freq × r)  频谱基底, 列正交非负
      'H'          : (r × n_frames)     时序激活, 非负
      'feature'    : CNN 输入特征 (H 或 W)
      'S_full'     : 全频带归一化幅度谱
      'f_full'     : 全频带频率轴 (Hz)
      'S_band'     : 截取频带幅度谱
      'f_band'     : 截取频带频率轴 (Hz)
      't'          : 时间轴 (s)
      'errors'     : ONMF 重建误差列表
      'win_samples': 实际窗口样本数
      'hop_samples': 实际 hop 样本数
      'n_fft'      : 实际 FFT 点数
      'fs_processed': 实际处理采样率
    """
    pcg = np.asarray(pcg, dtype=np.float64)

    # 降采样到目标采样率 (论文 Section 4.2)
    if resample_to is not None and abs(fs - resample_to) > 1.0:
        n_new = int(round(len(pcg) * resample_to / fs))
        pcg = scipy_signal.resample(pcg, n_new)
        fs = resample_to

    # 兼容旧接口: 在降采样后计算 win_samples / hop_samples
    if window_size_ms is not None:
        win_samples = max(1, int(round(fs * window_size_ms / 1000.0)))
    if overlap_ratio is not None:
        hop_samples = max(1, int(round(win_samples * (1.0 - overlap_ratio))))
    elif hop_ratio is not None:
        hop_samples = max(1, int(round(win_samples * hop_ratio)))

    if n_fft is None:
        n_fft = _next_pow2(max(int(fs / 4.0), win_samples))

    # 时域预处理（降采样后、幅度归一化前）
    preproc_tags = []
    if apply_hampel:
        pcg = hampel_filter(pcg, half_window=hampel_half_window, n_sigma=hampel_sigma)
        preproc_tags.append(f'Hampel({hampel_sigma}σ)')
    if apply_notch:
        pcg = apply_notch_filters(pcg, fs, freqs=notch_freqs, Q=notch_Q)
        preproc_tags.append(f'陷波{list(notch_freqs)}Hz')
    if apply_time_bandpass:
        pcg = apply_bandpass_filter(pcg, fs, low_hz=bp_low_hz, high_hz=bp_high_hz,
                                    order=bp_order)
        preproc_tags.append(f'带通{bp_low_hz:.0f}-{bp_high_hz:.0f}Hz')

    # Algorithm 1 Step 1: 幅度归一化
    mx = np.max(np.abs(pcg))
    if mx > 0:
        pcg = pcg / mx

    # Algorithm 1 Step 2: STFT 幅度谱
    f_full, t, S_full = compute_stft(pcg, fs, win_samples, hop_samples, n_fft)

    # Algorithm 1 Step 2 (cont.): 谱最大值归一化  X ← X / max(X)
    s_max = S_full.max()
    S_norm = S_full / s_max if s_max > 0 else S_full.copy()

    # Algorithm 1 Step 3: 频域频带截取
    f_band, S_band = apply_freq_band_mask(f_full, S_norm, low_hz, high_hz)
    S_band = np.maximum(S_band, 0.0)

    # Algorithm 1 Steps 4-7: ONMF
    W, H, errors = onmf(S_band, r=r, max_iter=max_iter, random_state=random_state)

    return {
        'W':           W,
        'H':           H,
        'feature':     H if use_temporal else W,
        'pcg':         pcg,          # 降采样+预处理+幅度归一化后的时域信号
        'S_full':      S_norm,
        'f_full':      f_full,
        'S_band':      S_band,
        'f_band':      f_band,
        't':           t,
        'errors':      errors,
        'win_samples': win_samples,
        'hop_samples': hop_samples,
        'n_fft':       n_fft,
        'fs_processed': fs,
        'preproc_desc': ' + '.join(preproc_tags) if preproc_tags else '仅归一化',
    }
