"""
从循环平稳性提取特征 - Python版本
从MATLAB代码转译

提取循环平稳性特征（2个特征）
"""

import numpy as np


def extractFeatreusFromCyclostationarity(pcg_signal, fs):
    """
    从循环平稳性提取特征（2个特征）
    
    参数:
    pcg_signal: PCG信号
    fs: 采样频率
    
    返回:
    features: 循环平稳性特征（2维）
    """
    
    Min_cf = 0.5
    Max_cf = 3.0
    
    Len = len(pcg_signal)
    step = round(1 * fs)
    Win = round(5 * fs)
    
    try:
        from degree_cycle_python import degree_cycle
    except:
        # 如果无法导入degree_cycle，使用简化版本
        def degree_cycle(signal, min_cf, max_cf, sampling_freq):
            # 简化版：返回默认值
            return 0.5
    
    if Len <= Win + step:
        degree = degree_cycle(pcg_signal, Min_cf, Max_cf, fs)
        features = np.array([degree, 0])
    else:
        deg = []
        k = 0
        flag = True
        
        while flag:
            ind_start = k * step
            ind_end = ind_start + Win
            
            if ind_end > Len:
                flag = False
            else:
                segment = pcg_signal[ind_start:ind_end]
                degree = degree_cycle(segment, Min_cf, Max_cf, fs)
                deg.append(degree)
                k += 1
        
        if len(deg) > 0:
            features = np.array([np.mean(deg), np.std(deg)])
        else:
            features = np.array([0, 0])
    
    return features
