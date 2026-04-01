from scipy.signal import hilbert, butter, filtfilt
import numpy as np


def hilbert_envelope(input_signal, sampling_rate, plot=None,
                     enhance_s1s2=True, s1s2_band=(20, 150)):
    """
    计算 PCG 信号的 Hilbert 包络。

    Parameters
    ----------
    enhance_s1s2 : bool
        True（默认）时先对信号做 S1/S2 主能量频段带通滤波再求包络。
        这样可避免持续性杂音的高频成分抬高包络基线，使 S1/S2 峰值更突出，
        对瓣膜病高杂音数据的分割鲁棒性更好。
        False 时保持原始行为（直接求 Hilbert 包络）。
    s1s2_band    : tuple(float, float)
        S1/S2 主能量频带（Hz），默认 20–150 Hz。
        如需保留更多高频成分可调高上限（如 200 Hz）。
    """
    if enhance_s1s2:
        nyq = 0.5 * float(sampling_rate)
        lo  = max(s1s2_band[0] / nyq, 0.001)
        hi  = min(s1s2_band[1] / nyq, 0.999)
        b, a = butter(4, [lo, hi], btype='band')
        signal_for_env = filtfilt(b, a, input_signal)
    else:
        signal_for_env = input_signal

    hilbert_env = np.abs(hilbert(signal_for_env))

    if plot:
        plot(hilbert_env, sampling_rate)

    return hilbert_env
