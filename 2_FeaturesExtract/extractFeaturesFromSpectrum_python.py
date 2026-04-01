"""
从频谱提取特征 - Python版本
从MATLAB代码转译

提取S1、S2、舒张期、收缩期的频谱特征
"""

import numpy as np
from scipy.fft import fft


def extractFeaturesFromSpectrum(assigned_states, pcg_signal):
    """
    从频谱提取特征（82个特征）
    
    参数:
    assigned_states: 分配给声音记录的状态值数组
    pcg_signal: PCG信号
    
    返回:
    featureSpectrum: 频谱特征向量（82维）
    """
    
    nfft = 100
    
    # 找到状态变化的位置
    indx = np.where(np.abs(np.diff(assigned_states)) > 0)[0]
    
    if assigned_states[0] > 0:
        state_map = {4: 1, 3: 2, 2: 3, 1: 4}
        K = state_map.get(assigned_states[0], 1)
    else:
        state_map = {4: 1, 3: 2, 2: 3, 1: 0}
        K = state_map.get(assigned_states[indx[0]+1], 0) + 1
    
    indx2 = indx[K-1:]
    rem = len(indx2) % 4
    if rem > 0:
        indx2 = indx2[:-rem]
    
    A = indx2.reshape(-1, 4)
    
    # 初始化频谱矩阵
    Spectrum_Sys = []
    Spectrum_Dia = []
    Spectrum_S1 = []
    Spectrum_S2 = []
    
    for i in range(len(A) - 1):
        try:
            signal_Sys = pcg_signal[A[i, 1]:A[i, 2]]
            signal_Dia = pcg_signal[A[i, 3]:A[i+1, 0]]
            signal_S1 = pcg_signal[A[i, 0]:A[i, 1]]
            signal_S2 = pcg_signal[A[i, 2]:A[i, 3]]
            
            if len(signal_Sys) > 0:
                Spectrum_Sys.append(np.abs(fft(signal_Sys, nfft)))
            if len(signal_Dia) > 0:
                Spectrum_Dia.append(np.abs(fft(signal_Dia, nfft)))
            if len(signal_S1) > 0:
                Spectrum_S1.append(np.abs(fft(signal_S1, nfft)))
            if len(signal_S2) > 0:
                Spectrum_S2.append(np.abs(fft(signal_S2, nfft)))
        except:
            continue
    
    # 计算平均频谱
    if len(Spectrum_Sys) > 0:
        mSpectrum_Sys = np.mean(Spectrum_Sys, axis=0)
    else:
        mSpectrum_Sys = np.zeros(nfft)
    
    if len(Spectrum_Dia) > 0:
        mSpectrum_Dia = np.mean(Spectrum_Dia, axis=0)
    else:
        mSpectrum_Dia = np.zeros(nfft)
    
    if len(Spectrum_S1) > 0:
        mSpectrum_S1 = np.mean(Spectrum_S1, axis=0)
    else:
        mSpectrum_S1 = np.zeros(nfft)
    
    if len(Spectrum_S2) > 0:
        mSpectrum_S2 = np.mean(Spectrum_S2, axis=0)
    else:
        mSpectrum_S2 = np.zeros(nfft)
    
    # 提取特定频率的特征
    # S1: 2-13 (12个), S2: 2-13 (12个), Sys: 2-30 (29个), Dia: 2-30 (29个)
    # 总共: 12+12+29+29 = 82个特征
    featureSpectrum = np.concatenate([
        mSpectrum_S1[1:13],   # 索引1-12对应MATLAB的2-13
        mSpectrum_S2[1:13],
        mSpectrum_Sys[1:30],
        mSpectrum_Dia[1:30]
    ])
    
    return featureSpectrum
