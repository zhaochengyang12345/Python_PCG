"""
小波能量特征提取 - Python版本
从MATLAB代码转译

提取64维小波能量特征
"""

import numpy as np
import pywt


def PCG_Wavelet_Efeature(ECG):
    """
    提取PCG小波包能量特征（64维）
    
    参数:
    ECG: PCG信号
    
    返回:
    wp_feature: 64维小波能量特征
    """
    
    # 归一化信号到[-1, 1]
    x = ECG.copy()
    x_min, x_max = np.min(x), np.max(x)
    if x_max > x_min:
        x = 2 * (x - x_min) / (x_max - x_min) - 1
    else:
        x = np.zeros_like(x)
    
    # 小波包分解（6层，使用db6小波）
    wp = pywt.WaveletPacket(data=x, wavelet='db6', mode='symmetric', maxlevel=6)
    
    # 获取第6层的所有节点
    # 在PyWavelets中，第6层有2^6=64个节点
    # 节点命名：第6层的节点用6位二进制字符串表示，'a'表示0，'d'表示1
    # 例如：'aaaaaa', 'aaaaad', 'aaaada', ...
    
    # 提取64个节点的能量
    e = []
    nodes_level6 = [node.path for node in wp.get_level(6, 'freq')]
    
    for i in range(64):
        try:
            # 将索引i转换为节点路径
            # i的二进制表示决定路径：0='a', 1='d'
            path = ''
            temp_i = i
            for _ in range(6):
                if temp_i % 2 == 0:
                    path = 'a' + path
                else:
                    path = 'd' + path
                temp_i = temp_i // 2
            
            # 获取节点数据
            rcfs = wp[path].data
            # 计算2-范数的平方（能量）
            energy = np.sum(rcfs**2)
            e.append(energy)
        except Exception as ex:
            # 如果节点不存在，能量为0
            e.append(0)
    
    e = np.array(e)
    
    # 归一化能量（可选，根据需要）
    total = np.sum(e)
    if total > 0:
        p = e / total
    else:
        p = e
    
    # 可以计算熵等其他特征（原代码中注释掉了）
    # Entropy = -np.sum(p[p>0] * np.log(p[p>0]))
    
    # 返回小波系数能量（64维）
    wp_feature = e
    
    return wp_feature
