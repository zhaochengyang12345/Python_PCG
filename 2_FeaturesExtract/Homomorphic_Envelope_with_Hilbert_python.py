"""
同态包络（使用Hilbert变换） - Python版本
从MATLAB代码转译

使用以下出版物中描述的方法找到信号的同态包络

Developed by David Springer for comparison purposes in the paper:
D. Springer et al., "Logistic Regression-HSMM-based Heart Sound 
Segmentation," IEEE Trans. Biomed. Eng., In Press, 2015.
"""

import numpy as np
from scipy.signal import butter, filtfilt, hilbert


def Homomorphic_Envelope_with_Hilbert(input_signal, sampling_frequency, 
                                      lpf_frequency=8, figures=False):
    """
    使用Hilbert变换找到信号的同态包络
    
    参数:
    input_signal: 原始（一维）信号
    sampling_frequency: 信号的采样频率（Hz）
    lpf_frequency: 包络提取中使用的低通滤波器的截止频率
                   （默认 = 8 Hz，如Schmidt的出版物中所述）
    figures: （可选）布尔变量，控制是否显示原始信号和提取包络的图形
    
    返回:
    homomorphic_envelope: 原始信号的同态包络（未归一化）
    """
    
    # 8Hz, 1阶, Butterworth 低通滤波器
    B_low, A_low = butter(1, 2 * lpf_frequency / sampling_frequency, 'low')
    
    # 计算同态包络
    # exp(LPF(log(abs(hilbert(signal)))))
    analytic_signal = hilbert(input_signal)
    envelope = np.abs(analytic_signal)
    log_envelope = np.log(envelope + 1e-10)  # 添加小常数避免log(0)
    filtered_log = filtfilt(B_low, A_low, log_envelope)
    homomorphic_envelope = np.exp(filtered_log)
    
    # 移除第一个样本中的虚假尖峰
    homomorphic_envelope[0] = homomorphic_envelope[1]
    
    if figures:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(input_signal, label='Original Signal')
        plt.plot(homomorphic_envelope, 'r', label='Homomorphic Envelope')
        plt.legend()
        plt.title('Homomorphic Envelope')
        plt.show()
    
    return homomorphic_envelope
