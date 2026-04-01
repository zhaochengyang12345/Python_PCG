"""
离散小波变换 - Python版本
从MATLAB代码转译

使用指定的小波在N层找到信号X的离散小波变换

Developed by David Springer for comparison purposes in the paper:
D. Springer et al., "Logistic Regression-HSMM-based Heart Sound 
Segmentation," IEEE Trans. Biomed. Eng., In Press, 2015.
"""

import numpy as np
import pywt


def getDWT(X, N, Name):
    """
    找到离散小波变换
    
    参数:
    X: 原始信号
    N: 分解层数
    Name: 使用的小波名称
    
    返回:
    cD: N行矩阵，包含N层的细节系数
    cA: 同样包含近似系数
    """
    
    # Morlet小波不支持DWT，因此执行CWT
    if Name == 'morl':
        # PyWavelets中使用cwt
        scales = np.arange(1, N+1)
        c, frequencies = pywt.cwt(X, scales, 'morl')
        cD = c
        cA = c
        return cD, cA
    
    # 执行小波分解
    coeffs = pywt.wavedec(X, Name, level=N)
    
    # coeffs = [cA_n, cD_n, cD_n-1, ..., cD_1]
    len_X = len(X)
    cD = np.zeros((N, len_X))
    cA = np.zeros((N, len_X))
    
    # 重构每一层的细节和近似系数
    for k in range(1, N+1):
        # 重构细节系数
        # 使用upcoef重构单层（类似MATLAB的wrcoef）
        # coeffs列表: [cA_N, cD_N, cD_N-1, ..., cD_1]
        # 细节系数cD_k在索引k的位置
        d = pywt.upcoef('d', coeffs[k], Name, level=k)
        
        # 调整长度以匹配原始信号
        if len(d) > len_X:
            d = d[:len_X]
        elif len(d) < len_X:
            d = np.pad(d, (0, len_X - len(d)), 'constant')
        cD[k-1, :] = d
        
        # 重构近似系数
        # 近似系数cA_N在索引0的位置
        # 重构到第k层的近似需要从cA_N开始
        a = pywt.upcoef('a', coeffs[0], Name, level=N-k+1)
        
        # 调整长度以匹配原始信号
        if len(a) > len_X:
            a = a[:len_X]
        elif len(a) < len_X:
            a = np.pad(a, (0, len_X - len(a)), 'constant')
        cA[k-1, :] = a
    
    # 将接近零的值设置为零
    cD[np.abs(cD) < np.sqrt(np.finfo(float).eps)] = 0
    cA[np.abs(cA) < np.sqrt(np.finfo(float).eps)] = 0
    
    return cD, cA
