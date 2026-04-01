import os
from math import gcd

from scipy.io import loadmat, wavfile
from scipy.signal import resample, resample_poly, butter, filtfilt, medfilt, iirnotch
from scipy.interpolate import PchipInterpolator
import numpy as np
import pandas as pd
import pywt


# ── MAT training data ────────────────────────────────────────────

def load_mat_data(file_name):
    from scipy.io import loadmat as _loadmat
    mat = _loadmat(file_name)
    annotations = mat['example_data']['example_annotations'][0][0]
    audio_signal = mat['example_data']['example_audio_data'][0][0]
    return audio_signal[0], annotations


# ── Resampling ───────────────────────────────────────────────────

def resample_to_fs(signal, orig_fs, target_fs):
    """Fast polyphase resampling — O(n), much faster than FFT resample."""
    if int(orig_fs) == int(target_fs):
        return np.array(signal, dtype=float)
    g = gcd(int(orig_fs), int(target_fs))
    return resample_poly(signal, int(target_fs) // g, int(orig_fs) // g)


# ── Spike / artefact removal ────────────────────────────────────

def hampel_filter(data, win=20, sigma=2.5):
    """
    Vectorised Hampel identifier: sliding-window median + MAD.
    Replaces outliers with the local median.  No Python for-loop.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    k = 1.4826
    n = len(data)
    w = 2 * win + 1
    padded  = np.pad(data, win, mode="edge")
    windows = sliding_window_view(padded, w)
    med     = np.median(windows, axis=1)
    mad     = np.median(np.abs(windows - med[:, None]), axis=1)
    outlier = np.abs(data - med) > sigma * k * mad
    out     = data.copy()
    out[outlier] = med[outlier]
    return out


def percentile_clip(data, pct=99.0):
    """Winsorise — clip extreme values to [pct_low, pct_high]."""
    lo = np.percentile(data, 100.0 - pct)
    hi = np.percentile(data, pct)
    return np.clip(data, lo, hi)


def notch_filter(signal, fs, freqs=(50.0, 100.0), Q=30.0):
    """
    Apply one IIR notch filter per frequency in *freqs*.
    Typical use: remove 50 Hz power-line interference and its 2nd harmonic.
    Frequencies outside (0, fs/2) are silently skipped.
    """
    data = signal.copy()
    for f0 in freqs:
        if 0 < f0 < fs / 2.0:
            b, a = iirnotch(f0, Q, fs)
            data = filtfilt(b, a, data)
    return data


def schmidt_spike_removal(original_signal, fs, max_iter=20):
    """Original Springer spike removal (used inside Springer feature pipeline)."""
    window_size    = int(np.round(fs / 2))
    trailing_samples = np.mod(len(original_signal), window_size)

    if trailing_samples:
        sample_frames = np.reshape(original_signal[0:-trailing_samples], (-1, window_size))
    else:
        sample_frames = np.reshape(original_signal, (-1, window_size))
    maa = np.max(np.abs(sample_frames), axis=1)

    for _ in range(max_iter):
        if not len(maa[maa > 3 * np.median(maa)]):
            break
        win_num        = np.argmax(maa)
        spike_position = np.argmax(np.abs(sample_frames[win_num]))
        zero_crossings = np.where(np.abs(np.diff(np.sign(sample_frames[win_num]))) > 1)[0]

        spike_start = zero_crossings[:spike_position]
        spike_start = spike_start[-1] if len(spike_start) else 0

        spike_end = zero_crossings[spike_position:]
        spike_end = spike_end.min() if len(spike_end) else window_size - 1

        sample_frames[win_num, spike_start:spike_end] = 0.0001
        maa = np.max(np.abs(sample_frames), axis=1)

    return np.concatenate((sample_frames.flatten(), original_signal[sample_frames.size:]))


# ── Wavelet soft-threshold denoising ────────────────────────────

def wavelet_denoise(signal, wavelet='db6', level=5, threshold_factor=0.4):
    """
    小波软阈值去噪 —— 专为 PCG 瓣膜杂音设计。

    原理：S1/S2 是瞬态能量，集中在低频近似系数；持续性杂音能量扩散在
    高频细节系数中。对细节系数施加基于 MAD 的软阈值可有效抑制杂音，
    同时保留 S1/S2 的形态和幅度。

    Parameters
    ----------
    signal           : 1-D float array
    wavelet          : 小波基，'db6' 与 S1/S2 形态最匹配
    level            : 分解层数；1000 Hz 采样率下第5层对应 ~15–30 Hz
    threshold_factor : 阈值缩放系数（0.3–0.6），越大去噪越强

    Returns
    -------
    去噪后的 1-D float array（与输入等长）
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    # 基于最细节层 MAD 估计信号噪声标准差（Donoho 估计器）
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    thr = threshold_factor * sigma * np.sqrt(2 * np.log(max(len(signal), 2)))
    # 仅对细节系数（indices 1..level）施加软阈值；低频近似完整保留
    coeffs_thr = [coeffs[0]] + [
        pywt.threshold(c, thr, mode='soft') for c in coeffs[1:]
    ]
    denoised = pywt.waverec(coeffs_thr, wavelet)
    return denoised[:len(signal)]


# ── Display preprocessing pipeline ──────────────────────────────

def preprocess_for_display(signal, fs, lowcut=20.0, highcut=400.0,
                            filter_order=4, clip_pct=99.0,
                            hampel_win=40, hampel_sigma=2.5,
                            notch_freqs=(50.0, 100.0), notch_Q=30.0,
                            apply_wavelet_denoise=True,
                            wavelet='db6', wavelet_level=5,
                            wavelet_threshold=0.4):
    """
    Preprocessing pipeline for waveform display (NOT for Springer segmentation).

    Steps:
      1. Percentile clip      — remove extreme motion artefacts
      2. Hampel filter        — adaptive spike removal
      3. 3-point median       — single-sample impulse rejection
      4. Notch filter         — remove power-line interference (50/100 Hz)
      5. Wavelet soft-thresh  — suppress continuous murmur noise [NEW]
      6. Bandpass filter      — keep cardiac band (lowcut–highcut Hz)
      7. Robust normalise     — scale by 99th-percentile amplitude → [-1, 1]

    Parameters
    ----------
    apply_wavelet_denoise : bool  启用小波软阈值去噪（推荐对瓣膜病数据开启）
    wavelet               : str   小波基（默认 'db6'，与 S1/S2 形态匹配）
    wavelet_level         : int   分解层数（1000 Hz 下 5 层 ≈ 15–30 Hz 低频）
    wavelet_threshold     : float 阈值系数（0.3 保守 → 0.6 积极去噪）

    Returns (time_axis, signal_display)
    """
    data = signal.astype(float).copy()
    data = percentile_clip(data, clip_pct)
    win  = max(5, int(round(fs * 0.0025)))
    data = hampel_filter(data, win=win, sigma=hampel_sigma)
    data = medfilt(data, kernel_size=3)
    # Notch filter (power-line removal)
    if notch_freqs:
        data = notch_filter(data, fs, freqs=notch_freqs, Q=notch_Q)
    # Wavelet soft-threshold denoising (suppresses continuous valve murmurs)
    if apply_wavelet_denoise:
        data = wavelet_denoise(data,
                               wavelet=wavelet,
                               level=wavelet_level,
                               threshold_factor=wavelet_threshold)
    # Bandpass
    nyq  = 0.5 * fs
    lo   = max(lowcut  / nyq, 0.001)
    hi   = min(highcut / nyq, 0.999)
    b, a = butter(filter_order, [lo, hi], btype="band")
    data = filtfilt(b, a, data)
    # Robust normalise
    scale = np.percentile(np.abs(data), 99.0)
    if scale > 0:
        data = np.clip(data / scale, -1.0, 1.0)
    time_axis = np.linspace(0, (len(data) - 1) / fs, len(data))
    return time_axis, data


# ── File loaders (return raw signal at target_fs, no display preproc) ──

def load_wav_file(path):
    """
    Load a WAV file. Handles int16 / int32 / uint8 / float dtypes.
    Merges multi-channel to mono.  Returns (signal_float64, native_fs).
    No resampling is applied here — the caller decides if resampling is needed.
    """
    native_fs, raw = wavfile.read(path)

    if raw.dtype == np.int16:
        data = raw.astype(np.float64) / 32768.0
    elif raw.dtype == np.int32:
        data = raw.astype(np.float64) / 2147483648.0
    elif raw.dtype == np.uint8:
        data = (raw.astype(np.float64) - 128.0) / 128.0
    else:
        data = raw.astype(np.float64)

    if data.ndim > 1:
        data = data.mean(axis=1)

    return data, int(native_fs)


def load_csv_file(path, pcg_col='pcg'):
    """
    Load a CSV file.
    - Auto-detects sampling rate from time/timestamp/t column (if present).
    - Handles NaN: >50% missing → ValueError; boundary fill + PCHIP for interior.
    Returns (signal_array, detected_fs).
    """
    df = pd.read_csv(path)
    # normalise column names
    col_map = {c: c.strip().lower() for c in df.columns}
    df.rename(columns=col_map, inplace=True)

    norm_col = pcg_col.strip().lower()
    if norm_col not in df.columns:
        raise ValueError(
            f"Column '{pcg_col}' not found in {path}. "
            f"Available: {list(df.columns)}"
        )

    raw = df[norm_col].values.astype(np.float64)

    # ── 自动检测采样率 ───────────────────────────────────────
    detected_fs = None
    for time_col in ("time", "timestamp", "t"):
        if time_col in df.columns:
            t = pd.to_numeric(df[time_col], errors="coerce").dropna().values
            if len(t) > 1:
                dt = float(np.median(np.diff(t)))
                if dt > 0:
                    detected_fs = int(round(1.0 / dt)) if dt < 1 else int(round(1000.0 / dt))
            break

    # ── NaN handling ────────────────────────────────────────────
    nan_mask  = np.isnan(raw)
    nan_ratio = nan_mask.sum() / len(raw)
    if nan_ratio > 0.5:
        raise ValueError(
            f"pcg column has {nan_ratio:.1%} NaN values (>50%); data quality too poor."
        )

    if nan_mask.any():
        nan_count = int(nan_mask.sum())
        print(f"    [CSV] 正在对 '{norm_col}' 列中 {nan_count} 个 NaN 値进行 PCHIP 插值...")
        idx       = np.arange(len(raw))
        valid_idx = idx[~nan_mask]
        valid_val = raw[~nan_mask]
        fill      = raw.copy()
        # 边界填充
        if nan_mask[0]:
            fill[:valid_idx[0]] = valid_val[0]
        if nan_mask[-1]:
            fill[valid_idx[-1] + 1:] = valid_val[-1]
        # 内部 PCHIP 插值
        still_nan = np.isnan(fill)
        if still_nan.any():
            pchip = PchipInterpolator(valid_idx, valid_val, extrapolate=False)
            fill[still_nan] = pchip(idx[still_nan])
        raw = fill

    return raw, detected_fs


# ── 文件夹扫描器 ────────────────────────────────────────────────

def scan_folder(folder):
    """递归扫描文件夹，返回所有 .wav 和 .csv 文件路径（已排序）。"""
    found = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith('.wav') or f.lower().endswith('.csv'):
                found.append(os.path.join(root, f))
    found.sort()
    return found


# ── Springer internal helpers (used inside tools.py) ────────────

def normalize(input_signal):
    return (input_signal - np.mean(input_signal)) / np.std(input_signal)


def high_pass_filter(input_signal, order, cutoff, sampling_rate):
    b, a = butter(order, cutoff / (sampling_rate / 2), 'highpass', output='ba')
    return filtfilt(b, a, input_signal)


def low_pass_filter(input_signal, order, cutoff, sampling_rate):
    b, a = butter(order, cutoff / (sampling_rate / 2), 'lowpass', output='ba')
    return filtfilt(b, a, input_signal)


def downsample(envelope, feature_fs, sampling_rate):
    number_of_samples = int(np.round(len(envelope) * float(feature_fs / sampling_rate)))
    return resample(envelope, number_of_samples)
