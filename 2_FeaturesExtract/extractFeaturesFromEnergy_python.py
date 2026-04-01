"""
从能量提取特征 - Python版本
从MATLAB代码转译

计算心音能量与非心音能量比值，均值，标准差
"""

import numpy as np


def extractFeaturesFromEnergy(assigned_states, pcg_signal):
    """
    从能量提取特征（20个特征）
    
    参数:
    assigned_states: 分配给声音记录的状态值数组
    pcg_signal: PCG信号
    
    返回:
    featuresEnergy: 能量特征向量（20维）
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
    
    # 初始化特征列表
    ratio_energy_SysTotal = []
    ratio_energy_DiaTotal = []
    ratio_energy_S1Total = []
    ratio_energy_S2Total = []
    ratio_energy_HsTotal = []
    ratio_energy_S1ToSys = []
    ratio_energy_S1ToDia = []
    ratio_energy_S2ToSys = []
    ratio_energy_S2ToDia = []
    ratio_energy_DiaToSys = []
    
    for i in range(len(A) - 1):
        try:
            signal_Sys = pcg_signal[A[i, 1]:A[i, 2]]
            signal_Dia = pcg_signal[A[i, 3]:A[i+1, 0]]
            signal_S1 = pcg_signal[A[i, 0]:A[i, 1]]
            signal_S2 = pcg_signal[A[i, 2]:A[i, 3]]
            signal_Cycle = pcg_signal[A[i, 0]:A[i+1, 0]]
            
            energy_Sys = np.sum(signal_Sys**2)
            energy_Dia = np.sum(signal_Dia**2)
            energy_S1 = np.sum(signal_S1**2)
            energy_S2 = np.sum(signal_S2**2)
            energy_Cycle = np.sum(signal_Cycle**2)
            
            if energy_Cycle > 0:
                ratio_energy_SysTotal.append(energy_Sys / energy_Cycle)
                ratio_energy_DiaTotal.append(energy_Dia / energy_Cycle)
                ratio_energy_S1Total.append(energy_S1 / energy_Cycle)
                ratio_energy_S2Total.append(energy_S2 / energy_Cycle)
                ratio_energy_HsTotal.append((energy_S1 + energy_S2) / energy_Cycle)
            
            if energy_Sys > 0:
                ratio_energy_S1ToSys.append(energy_S1 / energy_Sys)
                ratio_energy_S2ToSys.append(energy_S2 / energy_Sys)
            
            if energy_Dia > 0:
                ratio_energy_S1ToDia.append(energy_S1 / energy_Dia)
                ratio_energy_S2ToDia.append(energy_S2 / energy_Dia)
            
            if energy_Sys > 0:
                ratio_energy_DiaToSys.append(energy_Dia / energy_Sys)
        except:
            continue
    
    # 计算统计量
    m_energy_SysTotal = np.mean(ratio_energy_SysTotal) if len(ratio_energy_SysTotal) > 0 else 0
    sd_energy_SysTotal = np.std(ratio_energy_SysTotal) if len(ratio_energy_SysTotal) > 0 else 0
    m_energy_DiaTotal = np.mean(ratio_energy_DiaTotal) if len(ratio_energy_DiaTotal) > 0 else 0
    sd_energy_DiaTotal = np.std(ratio_energy_DiaTotal) if len(ratio_energy_DiaTotal) > 0 else 0
    m_energy_S1Total = np.mean(ratio_energy_S1Total) if len(ratio_energy_S1Total) > 0 else 0
    sd_energy_S1Total = np.std(ratio_energy_S1Total) if len(ratio_energy_S1Total) > 0 else 0
    m_energy_S2Total = np.mean(ratio_energy_S2Total) if len(ratio_energy_S2Total) > 0 else 0
    sd_energy_S2Total = np.std(ratio_energy_S2Total) if len(ratio_energy_S2Total) > 0 else 0
    m_energy_HsTotal = np.mean(ratio_energy_HsTotal) if len(ratio_energy_HsTotal) > 0 else 0
    sd_energy_HsTotal = np.std(ratio_energy_HsTotal) if len(ratio_energy_HsTotal) > 0 else 0
    m_energy_S1ToSys = np.mean(ratio_energy_S1ToSys) if len(ratio_energy_S1ToSys) > 0 else 0
    sd_energy_S1ToSys = np.std(ratio_energy_S1ToSys) if len(ratio_energy_S1ToSys) > 0 else 0
    m_energy_S1ToDia = np.mean(ratio_energy_S1ToDia) if len(ratio_energy_S1ToDia) > 0 else 0
    sd_energy_S1ToDia = np.std(ratio_energy_S1ToDia) if len(ratio_energy_S1ToDia) > 0 else 0
    m_energy_S2ToSys = np.mean(ratio_energy_S2ToSys) if len(ratio_energy_S2ToSys) > 0 else 0
    sd_energy_S2ToSys = np.std(ratio_energy_S2ToSys) if len(ratio_energy_S2ToSys) > 0 else 0
    m_energy_S2ToDia = np.mean(ratio_energy_S2ToDia) if len(ratio_energy_S2ToDia) > 0 else 0
    sd_energy_S2ToDia = np.std(ratio_energy_S2ToDia) if len(ratio_energy_S2ToDia) > 0 else 0
    m_energy_DiaToSys = np.mean(ratio_energy_DiaToSys) if len(ratio_energy_DiaToSys) > 0 else 0
    sd_energy_DiaToSys = np.std(ratio_energy_DiaToSys) if len(ratio_energy_DiaToSys) > 0 else 0
    
    featuresEnergy = np.array([
        m_energy_SysTotal, sd_energy_SysTotal, m_energy_DiaTotal, sd_energy_DiaTotal,
        m_energy_S1Total, sd_energy_S1Total, m_energy_S2Total, sd_energy_S2Total,
        m_energy_HsTotal, sd_energy_HsTotal, m_energy_S1ToSys, sd_energy_S1ToSys,
        m_energy_S1ToDia, sd_energy_S1ToDia, m_energy_S2ToSys, sd_energy_S2ToSys,
        m_energy_S2ToDia, sd_energy_S2ToDia, m_energy_DiaToSys, sd_energy_DiaToSys
    ])
    
    return featuresEnergy
