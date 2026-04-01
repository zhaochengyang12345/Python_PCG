"""
Hilbert包络 - Python版本
从MATLAB代码转译

此函数找到信号的Hilbert包络

Developed by David Springer for comparison purposes in the paper:
D. Springer et al., "Logistic Regression-HSMM-based Heart Sound Segmentation," 
IEEE Trans. Biomed. Eng., In Press, 2015.
"""

import numpy as np
from scipy.signal import hilbert


def Hilbert_Envelope(input_signal, sampling_frequency, figures=False):
    """
    使用Hilbert变换找到信号的包络
    
    参数:
    input_signal: 原始信号
    sampling_frequency: 信号的采样频率
    figures: （可选）布尔变量，显示原始和包络信号的图形
    
    返回:
    hilbert_envelope: 原始信号的Hilbert包络
    """
    
    # 使用Hilbert变换找到信号的包络
    hilbert_envelope = np.abs(hilbert(input_signal))
    
    if figures:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(input_signal, label='Original Signal')
        plt.plot(hilbert_envelope, 'r', label='Hilbert Envelope')
        plt.legend()
        plt.title('Hilbert Envelope')
        plt.show()
    
    return hilbert_envelope
