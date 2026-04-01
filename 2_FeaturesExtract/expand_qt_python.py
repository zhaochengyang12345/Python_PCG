"""
扩展状态序列 - Python版本
从MATLAB代码转译

将派生的HMM状态扩展到更高的采样频率的函数

Developed by David Springer for comparison purposes in the paper:
D. Springer et al., "Logistic Regression-HSMM-based Heart Sound 
Segmentation," IEEE Trans. Biomed. Eng., In Press, 2015.
"""

import numpy as np


def expand_qt(original_qt, old_fs, new_fs, new_length):
    """
    将状态序列扩展到新的采样频率和长度
    
    参数:
    original_qt: 从HMM派生的原始状态
    old_fs: original_qt的原始采样频率
    new_fs: 所需的采样频率
    new_length: qt信号的所需长度
    
    返回:
    expanded_qt: 扩展到新FS和长度的qt
    """
    
    original_qt = np.array(original_qt).flatten()
    expanded_qt = np.zeros(new_length)
    
    # 找到状态变化的索引
    indeces_of_changes = np.where(np.diff(original_qt) != 0)[0]
    indeces_of_changes = np.append(indeces_of_changes, len(original_qt) - 1)
    
    start_index = 0
    for i in range(len(indeces_of_changes)):
        end_index = indeces_of_changes[i]
        
        # 找到中点
        mid_point = round((end_index - start_index) / 2) + start_index
        
        # 获取中点的值
        if mid_point < len(original_qt):
            value_at_mid_point = original_qt[mid_point]
        else:
            value_at_mid_point = original_qt[-1]
        
        # 计算扩展后的索引
        expanded_start_index = round((start_index / old_fs) * new_fs)
        expanded_end_index = round((end_index / old_fs) * new_fs)
        
        if expanded_end_index >= new_length:
            expanded_end_index = new_length - 1
        
        # 填充扩展后的qt
        if expanded_start_index < new_length:
            expanded_qt[expanded_start_index:expanded_end_index+1] = value_at_mid_point
        
        start_index = end_index
    
    return expanded_qt
