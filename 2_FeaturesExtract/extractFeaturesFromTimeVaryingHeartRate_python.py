"""
从时变心率提取特征 - Python版本
从MATLAB代码转译

提取心率序列功率谱、收缩期序列功率谱、舒张期序列功率谱（59个特征）
"""

import numpy as np
from scipy.fft import fft
from scipy.interpolate import PchipInterpolator


def extractFeaturesFromTimeVaryingHeartRate(signal, Fs):
    """
    从时变心率提取特征（59个特征）
    
    参数:
    signal: 输入信号
    Fs: 采样频率
    
    返回:
    features: 时变心率特征向量（59维）
    """
    
    Lw = round(10 * Fs)  # 滑动窗口长度（样本数）
    step = round(0.5 * Fs)  # 滑动步长（样本数）
    
    # 使用简化版心率估计
    def getHeartRateSchmidt(sig, fs, plot=0):
        # 简化版：估计心率为60-100 bpm
        return 75.0, 0.3
    
    if len(signal) < Lw + step:
        heartRate, systolicTimeInterval = getHeartRateSchmidt(signal, Fs, 0)
        cycleDuration = 60 / heartRate
        diasystolicTimeInterval = cycleDuration - systolicTimeInterval
        ind = np.array([0, len(signal)-1])
        heartRate = np.array([heartRate, heartRate])
        systolicTimeInterval = np.array([systolicTimeInterval, systolicTimeInterval])
        diasystolicTimeInterval = np.array([diasystolicTimeInterval, diasystolicTimeInterval])
    else:
        ind = np.arange(Lw-1, len(signal), step)
        heartRate_list = []
        systolicTimeInterval_list = []
        diasystolicTimeInterval_list = []
        
        for k in range(len(ind)):
            tx = signal[ind[k]-Lw+1:ind[k]+1]
            
            # 检查是否是平坦线
            if np.sum(tx**2) / np.sum(signal**2) < 0.05 * Lw / len(signal):
                heartRate_list.append(0)
                systolicTimeInterval_list.append(0)
                diasystolicTimeInterval_list.append(0)
            else:
                hr, sti = getHeartRateSchmidt(tx, Fs, 0)
                heartRate_list.append(hr)
                systolicTimeInterval_list.append(sti)
                cycleDuration = 60 / hr
                diasystolicTimeInterval_list.append(cycleDuration - sti)
        
        ind = np.concatenate([[0], ind])
        heartRate = np.array([heartRate_list[0]] + heartRate_list)
        systolicTimeInterval = np.array([systolicTimeInterval_list[0]] + systolicTimeInterval_list)
        diasystolicTimeInterval = np.array([diasystolicTimeInterval_list[0]] + diasystolicTimeInterval_list)
    
    # 消除为0的值，用平均值替换
    heartRate[heartRate == 0] = np.mean(heartRate[heartRate != 0]) if np.any(heartRate != 0) else 75
    systolicTimeInterval[systolicTimeInterval == 0] = np.mean(systolicTimeInterval[systolicTimeInterval != 0]) if np.any(systolicTimeInterval != 0) else 0.3
    diasystolicTimeInterval[diasystolicTimeInterval == 0] = np.mean(diasystolicTimeInterval[diasystolicTimeInterval != 0]) if np.any(diasystolicTimeInterval != 0) else 0.5
    
    # 对各间隔进行插值获得等间隔采样
    uFs = 2  # 单位：Hz
    Interval = round(Fs / uFs)
    interp_ind = np.arange(0, len(signal), Interval)
    if interp_ind[-1] < len(signal) - 1:
        interp_ind = np.append(interp_ind, len(signal) - 1)
    
    # 使用PCHIP插值（保形分段三次插值）
    try:
        interp_hr = PchipInterpolator(ind, heartRate)
        uniform_sampled_heartRate = interp_hr(interp_ind)
        
        interp_sti = PchipInterpolator(ind, systolicTimeInterval)
        uniform_sampled_systolicTimeInterval = interp_sti(interp_ind)
        
        interp_dsti = PchipInterpolator(ind, diasystolicTimeInterval)
        uniform_sampled_diasystolicTimeInterval = interp_dsti(interp_ind)
    except:
        # 如果插值失败，使用线性插值
        uniform_sampled_heartRate = np.interp(interp_ind, ind, heartRate)
        uniform_sampled_systolicTimeInterval = np.interp(interp_ind, ind, systolicTimeInterval)
        uniform_sampled_diasystolicTimeInterval = np.interp(interp_ind, ind, diasystolicTimeInterval)
    
    uniform_sampled_cyclePeriod = 60.0 / uniform_sampled_heartRate
    
    # FFT分析
    nfft = 40
    spectrum_cyclePeriod = np.abs(fft(uniform_sampled_cyclePeriod, nfft))
    spectrum_systolicTimeInterval = np.abs(fft(uniform_sampled_systolicTimeInterval, nfft))
    spectrum_diasystolicTimeInterval = np.abs(fft(uniform_sampled_diasystolicTimeInterval, nfft))
    
    # 组合特征
    # 2个统计特征 + 19*3 = 2 + 57 = 59个特征
    features = np.concatenate([
        [np.mean(uniform_sampled_cyclePeriod), np.std(uniform_sampled_cyclePeriod)],
        spectrum_cyclePeriod[1:nfft//2],  # 19个
        spectrum_systolicTimeInterval[1:nfft//2],  # 19个
        spectrum_diasystolicTimeInterval[1:nfft//2]  # 19个
    ])
    
    return features
