"""
从峰度提取特征 - Python版本
从MATLAB代码转译

计算S1、S2、收缩期、舒张期的峰度特征
"""

import numpy as np


def extractFeaturesFromKurtosis(assigned_states, pcg_signal):
    """
    从峰度提取特征（8个特征）
    
    参数:
    assigned_states: 分配给声音记录的状态值数组
    pcg_signal: PCG信号
    
    返回:
    features: 峰度特征向量（8维）
    """
    
    # 找到状态变化的位置
    indx = np.where(np.abs(np.diff(assigned_states)) > 0)[0]
    
    if assigned_states[0] > 0:
        state_map = {4: 1, 3: 2, 2: 3, 1: 4}
        K = state_map.get(assigned_states[0], 1)
    else:
        state_map = {4: 1, 3: 2, 2: 3, 1: 0}
        K = state_map.get(assigned_states[indx[0]+1], 0) + 1
    
    indx2 = indx[K-1:]
    rem = len(indx2) % 4
    if rem > 0:
        indx2 = indx2[:-rem]
    
    A = indx2.reshape(-1, 4)
    
    # 初始化峰度列表
    s1_kur = []
    s2_kur = []
    sys_kur = []
    dia_kur = []
    
    for cn in range(len(A) - 1):
        try:
            # S1峰度
            s1 = pcg_signal[A[cn, 0]:A[cn, 1]]
            if len(s1) > 0:
                s1m4 = np.sum(s1**4) / len(s1)
                s1m2 = (np.sum(s1**2) / len(s1))**2
                if s1m2 > 0:
                    s1_kur.append(s1m4 / s1m2)
                else:
                    s1_kur.append(0)
            
            # S2峰度
            s2 = pcg_signal[A[cn, 2]:A[cn, 3]]
            if len(s2) > 0:
                s2m4 = np.sum(s2**4) / len(s2)
                s2m2 = (np.sum(s2**2) / len(s2))**2
                if s2m2 > 0:
                    s2_kur.append(s2m4 / s2m2)
                else:
                    s2_kur.append(0)
            
            # 收缩期峰度
            sys = pcg_signal[A[cn, 1]:A[cn, 2]]
            if len(sys) > 0:
                sysm4 = np.sum(sys**4) / len(sys)
                sysm2 = (np.sum(sys**2) / len(sys))**2
                if sysm2 > 0:
                    sys_kur.append(sysm4 / sysm2)
                else:
                    sys_kur.append(0)
            
            # 舒张期峰度
            dia = pcg_signal[A[cn, 3]:A[cn+1, 0]]
            if len(dia) > 0:
                diam4 = np.sum(dia**4) / len(dia)
                diam2 = (np.sum(dia**2) / len(dia))**2
                if diam2 > 0:
                    dia_kur.append(diam4 / diam2)
                else:
                    dia_kur.append(0)
        except:
            continue
    
    # 计算统计量
    features = np.array([
        np.mean(s1_kur) if len(s1_kur) > 0 else 0,
        np.std(s1_kur) if len(s1_kur) > 0 else 0,
        np.mean(s2_kur) if len(s2_kur) > 0 else 0,
        np.std(s2_kur) if len(s2_kur) > 0 else 0,
        np.mean(sys_kur) if len(sys_kur) > 0 else 0,
        np.std(sys_kur) if len(sys_kur) > 0 else 0,
        np.mean(dia_kur) if len(dia_kur) > 0 else 0,
        np.std(dia_kur) if len(dia_kur) > 0 else 0
    ])
    
    return features
