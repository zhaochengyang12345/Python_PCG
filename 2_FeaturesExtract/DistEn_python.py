"""
分布熵 - Python版本
从MATLAB代码转译

参考文献:
Detection of Coupling in Short Physiological Series by a Joint Distribution Entropy Method

Author: Koke Yao (原始MATLAB版本)
Date: 2018/08/02
"""

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import skew


def DistEn(datain, m, tau):
    """
    计算分布熵
    
    参数:
    datain: 输入数据
    m: 嵌入维度
    tau: 时间延迟
    
    返回:
    DistEn: 分布熵值
    """
    
    N = len(datain)  # 信号长度
    
    # 重新缩放到[0, 1]
    datain_min = np.min(datain)
    datain_max = np.max(datain)
    if datain_max > datain_min:
        datain = (datain - datain_min) / (datain_max - datain_min)
    else:
        datain = np.zeros_like(datain)
    
    # 构造Hankel矩阵
    indices = []
    for i in range(N - m * tau + 1):
        row = []
        for j in range(m):
            row.append(i + j * tau)
        indices.append(row)
    
    indices = np.array(indices)
    
    try:
        xm = datain[indices]
    except:
        return 0
    
    if m == 1:
        xm = xm.flatten()
    
    # 计算切比雪夫距离
    if m == 1:
        ChebXM = pdist(xm.reshape(-1, 1), 'chebyshev')
    else:
        ChebXM = pdist(xm, 'chebyshev')
    
    nb = len(ChebXM)
    
    if nb < 2:
        return 0
    
    # 计算最优箱数B
    sigma = np.sqrt(6 * (nb - 2) / ((nb + 1) * (nb + 3)))
    skewness = skew(ChebXM)
    
    try:
        B = np.ceil(1 + np.log2(nb) + np.log2(1 + np.abs(skewness) / sigma))
    except:
        B = 10
    
    # 确保B是有效的正整数
    if not np.isfinite(B) or B < 1:
        B = 10  # 使用默认值
    B = max(1, int(round(B)))  # 确保至少为1且是整数
    
    # 直方图统计计数
    BinCounts, _ = np.histogram(ChebXM, bins=B)
    
    # 计算概率
    p = (BinCounts + 1) / (nb + B)
    
    # 计算分布熵
    p = p[p > 0]  # 只保留非零概率
    DistEn = -np.sum(p * np.log2(p)) / np.log2(B)
    
    return DistEn
