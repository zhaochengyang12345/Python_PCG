"""
自定义小波滤波 - Python版本
从MATLAB代码转译

小波滤波31.25-250Hz
提取小波分解的第2、3、4层细节重构滤波后的信号
"""

import numpy as np
import pywt


def my_wavelet(original_signal):
    """
    使用db6小波进行5层小波分解，提取第2、3、4层细节系数重构的滤波信号
    
    参数:
    original_signal: 原始信号
    
    返回:
    wavelet_filtered_signal: 滤波后的信号（31.25-250Hz）
    """
    
    # 用'db6'小波进行5层的小波分解
    coeffs = pywt.wavedec(original_signal, 'db6', level=5)
    
    # coeffs = [cA5, cD5, cD4, cD3, cD2, cD1]
    # 重构各层细节系数
    # 使用upcoef来重构单个细节层（类似MATLAB的wrcoef）
    
    # 第2层细节系数（cD2）
    # upcoef第一个参数：'d'表示细节，'a'表示近似
    # 第二个参数：系数
    # 第三个参数：小波类型
    # 第四个参数：重构的层数
    # 第五个参数：原始信号长度
    d2 = pywt.upcoef('d', coeffs[-2], 'db6', level=2, take=len(original_signal))
    
    # 第3层细节系数（cD3）
    d3 = pywt.upcoef('d', coeffs[-3], 'db6', level=3, take=len(original_signal))
    
    # 第4层细节系数（cD4）
    d4 = pywt.upcoef('d', coeffs[-4], 'db6', level=4, take=len(original_signal))
    
    # 合并第2、3、4层
    wavelet_filtered_signal = d2 + d3 + d4
    
    return wavelet_filtered_signal
