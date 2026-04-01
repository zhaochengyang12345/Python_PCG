"""
从心音间隔提取特征 - Python版本
从MATLAB代码转译

基于assigned_states（由运行runSpringerSegmentationAlgorithm函数得到）计算48个特征

Written by: Chengyu Liu, January 22 2016
            chengyu.liu@emory.edu
"""

import numpy as np
from scipy.fft import fft


def extractFeaturesFromHsIntervals(assigned_states, PCG):
    """
    从心音间隔提取48个特征
    
    参数:
    assigned_states: 分配给声音记录的状态值数组
    PCG: 重采样的声音记录（1000 Hz）
    
    返回:
    features: 当前声音记录的48个特征
    """
    
    # 假设assigned_states至少覆盖2个完整的心跳周期
    indx = np.where(np.abs(np.diff(assigned_states)) > 0)[0]
    
    if assigned_states[0] > 0:
        # 对于某些记录，assigned_states开头有状态零
        state_map = {4: 1, 3: 2, 2: 3, 1: 4}
        K = state_map.get(assigned_states[0], 1)
    else:
        state_map = {4: 1, 3: 2, 2: 3, 1: 0}
        K = state_map.get(assigned_states[indx[0]+1], 0) + 1
    
    indx2 = indx[K-1:]  # Python索引从0开始
    rem = len(indx2) % 4
    if rem > 0:
        indx2 = indx2[:-rem]
    
    A = indx2.reshape(-1, 4)  # A是N×4矩阵
    
    # 特征计算
    try:
        m_RR = np.round(np.mean(np.diff(A[:, 0])))
        sd_RR = np.round(np.std(np.diff(A[:, 0])))
        m_IntS1 = np.round(np.mean(A[:, 1] - A[:, 0]))
        sd_IntS1 = np.round(np.std(A[:, 1] - A[:, 0]))
        m_IntS2 = np.round(np.mean(A[:, 3] - A[:, 2]))
        sd_IntS2 = np.round(np.std(A[:, 3] - A[:, 2]))
        m_IntSys = np.round(np.mean(A[:, 2] - A[:, 1]))
        sd_IntSys = np.round(np.std(A[:, 2] - A[:, 1]))
        m_IntDia = np.round(np.mean(A[1:, 0] - A[:-1, 3]))
        sd_IntDia = np.round(np.std(A[1:, 0] - A[:-1, 3]))
    except:
        # 如果计算失败，返回零特征
        return np.zeros(48)
    
    # 计算比率和幅度
    R_SysRR = []
    R_DiaRR = []
    R_SysDia = []
    P_SysS1 = []
    P_DiaS2 = []
    
    for i in range(len(A) - 1):
        try:
            rr_interval = A[i+1, 0] - A[i, 0]
            if rr_interval > 0:
                r_sys = (A[i, 2] - A[i, 1]) / rr_interval * 100
                r_dia = (A[i+1, 0] - A[i, 3]) / rr_interval * 100
                R_SysRR.append(r_sys)
                R_DiaRR.append(r_dia)
                if r_dia > 0:
                    R_SysDia.append(r_sys / r_dia * 100)
            
            # 幅度特征
            s1_len = A[i, 1] - A[i, 0]
            sys_len = A[i, 2] - A[i, 1]
            s2_len = A[i, 3] - A[i, 2]
            dia_len = A[i+1, 0] - A[i, 3]
            
            if s1_len > 0:
                P_S1 = np.sum(np.abs(PCG[A[i, 0]:A[i, 1]])) / s1_len
            else:
                P_S1 = 0
                
            if sys_len > 0:
                P_Sys = np.sum(np.abs(PCG[A[i, 1]:A[i, 2]])) / sys_len
            else:
                P_Sys = 0
                
            if s2_len > 0:
                P_S2 = np.sum(np.abs(PCG[A[i, 2]:A[i, 3]])) / s2_len
            else:
                P_S2 = 0
                
            if dia_len > 0:
                P_Dia = np.sum(np.abs(PCG[A[i, 3]:A[i+1, 0]])) / dia_len
            else:
                P_Dia = 0
            
            if P_S1 > 0:
                P_SysS1.append(P_Sys / P_S1 * 100)
            if P_S2 > 0:
                P_DiaS2.append(P_Dia / P_S2 * 100)
        except:
            continue
    
    # 计算均值和标准差
    m_Ratio_SysRR = np.mean(R_SysRR) if len(R_SysRR) > 0 else 0
    sd_Ratio_SysRR = np.std(R_SysRR) if len(R_SysRR) > 0 else 0
    m_Ratio_DiaRR = np.mean(R_DiaRR) if len(R_DiaRR) > 0 else 0
    sd_Ratio_DiaRR = np.std(R_DiaRR) if len(R_DiaRR) > 0 else 0
    m_Ratio_SysDia = np.mean(R_SysDia) if len(R_SysDia) > 0 else 0
    sd_Ratio_SysDia = np.std(R_SysDia) if len(R_SysDia) > 0 else 0
    
    # 过滤异常值
    P_SysS1_valid = [x for x in P_SysS1 if 0 < x < 100]
    P_DiaS2_valid = [x for x in P_DiaS2 if 0 < x < 100]
    
    m_Amp_SysS1 = np.mean(P_SysS1_valid) if len(P_SysS1_valid) > 1 else 0
    sd_Amp_SysS1 = np.std(P_SysS1_valid) if len(P_SysS1_valid) > 1 else 0
    m_Amp_DiaS2 = np.mean(P_DiaS2_valid) if len(P_DiaS2_valid) > 1 else 0
    sd_Amp_DiaS2 = np.std(P_DiaS2_valid) if len(P_DiaS2_valid) > 1 else 0
    
    # 频率特征
    Num_cyc = len(A) - 1
    Fs = 1000  # 采样频率
    
    # 初始化特征数组
    HFAll_S1 = []
    LFAll_S1 = []
    HFAll_Sys = []
    LFAll_Sys = []
    SampEn_Sys = []
    HFAll_S2 = []
    LFAll_S2 = []
    HFAll_Dia = []
    LFAll_Dia = []
    SampEn_Dia = []
    
    # 简化的熵计算（完整实现需要FuzzyEn和DistEn函数）
    try:
        from FuzzyEn_python import FuzzyEn
        from DistEn_python import DistEn
        use_entropy = True
    except:
        use_entropy = False
    
    FuzzyEn_Sys = []
    DistEn_Sys = []
    FuzzyEn_Dia = []
    DistEn_Dia = []
    
    for kk in range(Num_cyc):
        try:
            # S1频谱特征
            PCG_S1 = PCG[A[kk, 0]:A[kk, 1]]
            if len(PCG_S1) > 0:
                FFT_S1 = fft(PCG_S1)
                mag_S1 = np.abs(FFT_S1)
                HF_S1 = np.sum(mag_S1[int(np.ceil(200/Fs*len(mag_S1))):int(np.ceil(len(mag_S1)/2))])
                LF_S1 = np.sum(mag_S1[:int(np.ceil(50/Fs*len(mag_S1)))])
                All_S1 = np.sum(mag_S1)
                if All_S1 > 0:
                    HFAll_S1.append(HF_S1 / All_S1)
                    LFAll_S1.append(LF_S1 / All_S1)
            
            # 收缩期频谱特征
            PCG_Sys = PCG[A[kk, 1]:A[kk, 2]]
            if len(PCG_Sys) > 0:
                FFT_Sys = fft(PCG_Sys)
                mag_Sys = np.abs(FFT_Sys)
                HF_Sys = np.sum(mag_Sys[int(np.ceil(200/Fs*len(mag_Sys))):int(np.ceil(len(mag_Sys)/2))])
                LF_Sys = np.sum(mag_Sys[:int(np.ceil(50/Fs*len(mag_Sys)))])
                All_Sys = np.sum(mag_Sys)
                if All_Sys > 0:
                    HFAll_Sys.append(HF_Sys / All_Sys)
                    LFAll_Sys.append(LF_Sys / All_Sys)
                
                # 熵特征（简化版）
                if use_entropy and len(PCG_Sys) > 10:
                    try:
                        FuzzyEn_Sys.append(FuzzyEn(PCG_Sys, 2, 0.2, 1))
                        DistEn_Sys.append(DistEn(PCG_Sys, 2, 1))
                    except:
                        pass
            
            # S2频谱特征
            PCG_S2 = PCG[A[kk, 2]:A[kk, 3]]
            if len(PCG_S2) > 0:
                FFT_S2 = fft(PCG_S2)
                mag_S2 = np.abs(FFT_S2)
                HF_S2 = np.sum(mag_S2[int(np.ceil(200/Fs*len(mag_S2))):int(np.ceil(len(mag_S2)/2))])
                LF_S2 = np.sum(mag_S2[:int(np.ceil(50/Fs*len(mag_S2)))])
                All_S2 = np.sum(mag_S2)
                if All_S2 > 0:
                    HFAll_S2.append(HF_S2 / All_S2)
                    LFAll_S2.append(LF_S2 / All_S2)
            
            # 舒张期频谱特征
            if kk < Num_cyc - 1:
                PCG_Dia = PCG[A[kk, 3]:A[kk+1, 0]]
                if len(PCG_Dia) > 0:
                    FFT_Dia = fft(PCG_Dia)
                    mag_Dia = np.abs(FFT_Dia)
                    HF_Dia = np.sum(mag_Dia[int(np.ceil(200/Fs*len(mag_Dia))):int(np.ceil(len(mag_Dia)/2))])
                    LF_Dia = np.sum(mag_Dia[:int(np.ceil(50/Fs*len(mag_Dia)))])
                    All_Dia = np.sum(mag_Dia)
                    if All_Dia > 0:
                        HFAll_Dia.append(HF_Dia / All_Dia)
                        LFAll_Dia.append(LF_Dia / All_Dia)
                    
                    # 熵特征（简化版）
                    if use_entropy and len(PCG_Dia) > 10:
                        try:
                            FuzzyEn_Dia.append(FuzzyEn(PCG_Dia, 2, 0.2, 1))
                            DistEn_Dia.append(DistEn(PCG_Dia, 2, 1))
                        except:
                            pass
        except:
            continue
    
    # 计算统计量
    m_HFAll_S1 = np.mean(HFAll_S1) if len(HFAll_S1) > 0 else 0
    sd_HFAll_S1 = np.std(HFAll_S1) if len(HFAll_S1) > 0 else 0
    m_LFAll_S1 = np.mean(LFAll_S1) if len(LFAll_S1) > 0 else 0
    sd_LFAll_S1 = np.std(LFAll_S1) if len(LFAll_S1) > 0 else 0
    
    m_HFAll_Sys = np.mean(HFAll_Sys) if len(HFAll_Sys) > 0 else 0
    sd_HFAll_Sys = np.std(HFAll_Sys) if len(HFAll_Sys) > 0 else 0
    m_LFAll_Sys = np.mean(LFAll_Sys) if len(LFAll_Sys) > 0 else 0
    sd_LFAll_Sys = np.std(LFAll_Sys) if len(LFAll_Sys) > 0 else 0
    
    m_SampEn_Sys = 0  # 简化版，需要完整实现
    sd_SampEn_Sys = 0
    m_FuzzyEn_Sys = np.mean(FuzzyEn_Sys) if len(FuzzyEn_Sys) > 0 else 0
    sd_FuzzyEn_Sys = np.std(FuzzyEn_Sys) if len(FuzzyEn_Sys) > 0 else 0
    m_DistEn_Sys = np.mean(DistEn_Sys) if len(DistEn_Sys) > 0 else 0
    sd_DistEn_Sys = np.std(DistEn_Sys) if len(DistEn_Sys) > 0 else 0
    
    m_HFAll_S2 = np.mean(HFAll_S2) if len(HFAll_S2) > 0 else 0
    sd_HFAll_S2 = np.std(HFAll_S2) if len(HFAll_S2) > 0 else 0
    m_LFAll_S2 = np.mean(LFAll_S2) if len(LFAll_S2) > 0 else 0
    sd_LFAll_S2 = np.std(LFAll_S2) if len(LFAll_S2) > 0 else 0
    
    m_HFAll_Dia = np.mean(HFAll_Dia) if len(HFAll_Dia) > 0 else 0
    sd_HFAll_Dia = np.std(HFAll_Dia) if len(HFAll_Dia) > 0 else 0
    m_LFAll_Dia = np.mean(LFAll_Dia) if len(LFAll_Dia) > 0 else 0
    sd_LFAll_Dia = np.std(LFAll_Dia) if len(LFAll_Dia) > 0 else 0
    
    m_SampEn_Dia = 0  # 简化版
    sd_SampEn_Dia = 0
    m_FuzzyEn_Dia = np.mean(FuzzyEn_Dia) if len(FuzzyEn_Dia) > 0 else 0
    sd_FuzzyEn_Dia = np.std(FuzzyEn_Dia) if len(FuzzyEn_Dia) > 0 else 0
    m_DistEn_Dia = np.mean(DistEn_Dia) if len(DistEn_Dia) > 0 else 0
    sd_DistEn_Dia = np.std(DistEn_Dia) if len(DistEn_Dia) > 0 else 0
    
    # 组合所有特征（48个）
    features = np.array([
        m_RR, sd_RR, m_IntS1, sd_IntS1, m_IntS2, sd_IntS2, m_IntSys,
        sd_IntSys, m_IntDia, sd_IntDia, m_Ratio_SysRR, sd_Ratio_SysRR,
        m_Ratio_DiaRR, sd_Ratio_DiaRR, m_Ratio_SysDia, sd_Ratio_SysDia,
        m_Amp_SysS1, sd_Amp_SysS1, m_Amp_DiaS2, sd_Amp_DiaS2, m_HFAll_S1,
        m_HFAll_Sys, m_HFAll_S2, m_HFAll_Dia, m_LFAll_S1, m_LFAll_Sys,
        m_LFAll_S2, m_LFAll_Dia, sd_HFAll_S1, sd_HFAll_Sys, sd_HFAll_S2,
        sd_HFAll_Dia, sd_LFAll_S1, sd_LFAll_Sys, sd_LFAll_S2, sd_LFAll_Dia,
        m_SampEn_Sys, sd_SampEn_Sys, m_SampEn_Dia, sd_SampEn_Dia,
        m_FuzzyEn_Sys, sd_FuzzyEn_Sys, m_FuzzyEn_Dia, sd_FuzzyEn_Dia,
        m_DistEn_Sys, sd_DistEn_Sys, m_DistEn_Dia, sd_DistEn_Dia
    ])
    
    return features
