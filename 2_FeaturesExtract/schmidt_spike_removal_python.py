"""
Schmidt尖峰去除 - Python版本
从MATLAB代码转译

此函数移除信号中的尖峰，如Schmidt等人在以下论文中所述：
Schmidt, S. E., Holst-Hansen, C., Graff, C., Toft, E., & Struijk, J. J. (2010). 
Segmentation of heart sound recordings by a duration-dependent hidden Markov model.

Developed by David Springer for comparison purposes in the paper:
D. Springer et al., "Logistic Regression-HSMM-based Heart Sound Segmentation," 
IEEE Trans. Biomed. Eng., In Press, 2015.
"""

import numpy as np


def schmidt_spike_removal(original_signal, fs):
    """
    移除信号中的尖峰
    
    参数:
    original_signal: 原始（一维）音频信号数组
    fs: 采样频率（Hz）
    
    返回:
    despiked_signal: 移除尖峰后的音频信号
    """
    
    # 找到窗口大小（500 ms）
    windowsize = round(fs / 2)
    
    # 找到不是整数倍窗口的尾部样本
    trailingsamples = len(original_signal) % windowsize
    
    # 将信号重塑为多个窗口
    if trailingsamples > 0:
        signal_to_reshape = original_signal[:-trailingsamples]
    else:
        signal_to_reshape = original_signal
    
    num_windows = len(signal_to_reshape) // windowsize
    sampleframes = signal_to_reshape.reshape(windowsize, num_windows, order='F')  # Fortran顺序（列优先）
    
    # 找到MAAs（最大绝对幅度）
    MAAs = np.max(np.abs(sampleframes), axis=0)
    
    # 当仍有样本大于3倍中位数时，移除这些尖峰
    while np.any(MAAs > np.median(MAAs) * 3):
        
        # 找到具有最大MAA的窗口
        window_num = np.argmax(MAAs)
        
        # 找到尖峰在该窗口内的位置
        spike_position = np.argmax(np.abs(sampleframes[:, window_num]))
        
        # 找到过零点（符号改变）
        sign_changes = np.diff(np.sign(sampleframes[:, window_num]))
        zero_crossings = np.abs(sign_changes) > 1
        zero_crossings = np.append(zero_crossings, False)
        
        # 找到尖峰的开始，找尖峰位置之前的最后一个过零点
        spike_start_indices = np.where(zero_crossings[:spike_position])[0]
        if len(spike_start_indices) > 0:
            spike_start = spike_start_indices[-1]
        else:
            spike_start = 0
        
        # 找到尖峰的结束，找尖峰位置之后的第一个过零点
        zero_crossings[:spike_position+1] = False
        spike_end_indices = np.where(zero_crossings)[0]
        if len(spike_end_indices) > 0:
            spike_end = spike_end_indices[0]
        else:
            spike_end = windowsize - 1
        
        # 设置为接近零的小值
        sampleframes[spike_start:spike_end+1, window_num] = 0.0001
        
        # 重新计算MAAs
        MAAs = np.max(np.abs(sampleframes), axis=0)
    
    # 重塑回一维信号
    despiked_signal = sampleframes.reshape(-1, order='F')
    
    # 将尾部样本添加回信号
    if trailingsamples > 0:
        despiked_signal = np.concatenate([despiked_signal, original_signal[-trailingsamples:]])
    
    return despiked_signal
