"""
获取PSD特征 (Springer HMM) - Python版本
从MATLAB代码转译

基于PSD的心音分割特征提取

Developed by David Springer in the paper:
D. Springer et al., "Logistic Regression-HSMM-based Heart Sound 
Segmentation," IEEE Trans. Biomed. Eng., In Press, 2015.
"""

import numpy as np
from scipy import signal


def get_PSD_feature_Springer_HMM(data, sampling_frequency, 
                                 frequency_limit_low, frequency_limit_high,
                                 figures=False):
    """
    获取PSD特征
    
    参数:
    data: 音频波形
    sampling_frequency: 采样频率
    frequency_limit_low: 要分析的频率范围的下限
    frequency_limit_high: 频率范围的上限
    figures: （可选）布尔变量，用于显示图形
    
    返回:
    psd: 最大和最小限制之间的最大PSD值数组，
         重采样到与原始数据相同的大小
    """
    
    # 找到信号的频谱图
    nperseg = int(sampling_frequency / 40)
    noverlap = int(sampling_frequency / 80)
    nfft = int(sampling_frequency / 2)
    
    F, T, P = signal.spectrogram(data, 
                                 fs=sampling_frequency,
                                 nperseg=nperseg,
                                 noverlap=noverlap,
                                 nfft=nfft)
    
    if figures:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.pcolormesh(T, F, 10*np.log10(P + 1e-10), shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title('Spectrogram')
        plt.colorbar()
        plt.show()
    
    # 找到频率范围的索引
    low_limit_position = np.argmin(np.abs(F - frequency_limit_low))
    high_limit_position = np.argmin(np.abs(F - frequency_limit_high))
    
    # 找到感兴趣频率范围内的平均PSD
    psd = np.mean(P[low_limit_position:high_limit_position+1, :], axis=0)
    
    if figures:
        import matplotlib.pyplot as plt
        t4 = np.arange(len(psd)) / sampling_frequency
        t3 = np.arange(len(data)) / sampling_frequency
        
        plt.figure()
        plt.plot(t3, (data - np.mean(data)) / np.std(data), 'c', label='Audio Data')
        plt.plot(t4, (psd - np.mean(psd)) / np.std(psd), 'k', label='PSD Feature')
        plt.legend()
        plt.title('PSD Feature')
        plt.show()
    
    return psd
