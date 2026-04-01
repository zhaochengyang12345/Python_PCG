"""
模糊熵 - Python版本
从MATLAB代码转译

计算时间序列的模糊熵（FuzzyEn）

参考文献:
Chen et al. "Characterization of surface EMG signal based on fuzzy entropy"
DOI: 10.1109/TNSRE.2007.897025

PROJECT: Research Master in signal theory and bioengineering - University of Valladolid
DATE: 11/10/2014
VERSION: 1.0
AUTHOR: Jesús Monge Álvarez (原始MATLAB版本)
"""

import numpy as np


def FuzzyEn(series, dim, r, n):
    """
    计算时间序列的模糊熵
    
    参数:
    series: 时间序列
    dim: SampEn算法中使用的嵌入维度
    r: 模糊指数函数的宽度
    n: 模糊指数函数的步长
    
    返回:
    FuzzyEn: 模糊熵值
    """
    
    # 检查输入参数
    assert len(series) > 0, '必须提供时间序列（第一个输入）'
    assert dim is not None, '必须提供嵌入维度（第二个输入）'
    assert r is not None, '必须提供模糊指数函数的宽度r（第三个输入）'
    assert n is not None, '必须提供模糊指数函数的步长n（第四个输入）'
    
    # 标准化输入时间序列
    series = (series - np.mean(series)) / np.std(series)
    N = len(series)
    phi = np.zeros(2)
    
    for j in range(2):
        m = dim + j  # m是每次迭代使用的嵌入维度
        
        # 预定义变量以提高计算效率
        patterns = np.zeros((m, N - m + 1))
        
        # 首先，我们组成模式
        # 矩阵'patterns'的列将是长度为'm'的(N-m+1)个模式
        if m == 1:  # 如果嵌入维度为1，每个样本就是一个模式
            patterns = series.reshape(1, -1)
        else:  # 否则，我们构建长度为'm'的模式
            for i in range(m):
                patterns[i, :] = series[i:N-m+i+1]
        
        # 我们从每个模式中减去其基线
        for i in range(N - m + 1):
            patterns[:, i] = patterns[:, i] - np.mean(patterns[:, i])
        
        # 此循环遍历矩阵'patterns'的列
        aux = np.zeros(N - m)
        for i in range(N - m):
            # 其次，我们计算当前模式与其余模式之间的最大绝对距离
            if m == 1:
                dist = np.abs(patterns - patterns[:, i:i+1])
            else:
                dist = np.max(np.abs(patterns - patterns[:, i:i+1]), axis=0)
            
            # 第三，我们获得相似度
            simi = np.exp((-1) * (dist**n) / r)
            
            # 我们对当前模式的所有相似度求平均
            aux[i] = (np.sum(simi) - 1) / (N - m - 1)  # 减1以避免自比较
        
        # 最后，我们获得'phi'参数作为前'N-m'个平均相似度的平均值
        phi[j] = np.sum(aux) / (N - m)
    
    # 避免除以零或取对数为负数
    if phi[1] > 0 and phi[0] > 0:
        FuzzyEn = np.log(phi[0]) - np.log(phi[1])
    else:
        FuzzyEn = 0
    
    return FuzzyEn
