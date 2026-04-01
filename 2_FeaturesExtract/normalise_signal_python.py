"""
信号标准化 - Python版本
从MATLAB代码转译

此函数减去均值并除以标准差来标准化一维信号，用于机器学习应用

Developed by David Springer for the paper:
D. Springer et al., "Logistic Regression-HSMM-based Heart Sound
Segmentation," IEEE Trans. Biomed. Eng., In Press, 2015.
"""

import numpy as np


def normalise_signal(signal):
    """
    标准化信号（z-score标准化）
    
    参数:
    signal: 原始信号（一维数组）
    
    返回:
    normalised_signal: 标准化后的信号（减去均值，除以标准差）
    """
    
    mean_of_signal = np.mean(signal)
    standard_deviation = np.std(signal)
    
    # 避免除以零
    if standard_deviation == 0:
        standard_deviation = 1.0
    
    normalised_signal = (signal - mean_of_signal) / standard_deviation
    
    return normalised_signal
