"""
心音分类主函数 - Python版本
从MATLAB代码转译

Sample entry for the 2016 PhysioNet/CinC Challenge.

Written by: Chengyu Liu, February 21 2016
            chengyu.liu@emory.edu
"""

import numpy as np
from scipy.signal import butter, filtfilt, resample
from scipy.io import loadmat
from pathlib import Path
import os

from load_pcg_data_python import load_pcg_data
from default_Springer_HSMM_options_python import default_Springer_HSMM_options
from normalise_signal_python import normalise_signal
from classifyFromHsIntervals_python import classifyFromHsIntervals


def challenge(recordName, use_wavelet_feature=True):
    """
    心音分类主函数
    
    参数:
    recordName: 字符串，指定要处理的记录名
    use_wavelet_feature: 布尔值，是否提取小波能量特征（默认True）
    
    返回:
    classifyResult: 整数值
                        1 = 异常记录
                       -1 = 正常记录
                        0 = 不确定（噪声过多）
    features: 提取的特征向量
    """
    
    # 加载训练好的Springer HSMM模型参数矩阵
    # 这些参数使用MIT心音数据库的409个心音记录训练（记录a0001-a0409）
    from scipy.io import loadmat
    import os
    
    # 获取当前脚本的父目录（即原始MATLAB代码所在目录）
    current_dir = Path(__file__).parent.parent
    
    # 加载B_matrix
    B_mat_path = current_dir / 'Springer_B_matrix.mat'
    B_mat_data = loadmat(str(B_mat_path))
    # B_matrix是一个(1,4)的cell array，每个cell包含一个状态的逻辑回归系数
    B_mat_cells = B_mat_data['Springer_B_matrix']
    Springer_B_matrix = [B_mat_cells[0, i].flatten() for i in range(4)]
    
    # 加载pi_vector
    pi_mat_path = current_dir / 'Springer_pi_vector.mat'
    pi_mat_data = loadmat(str(pi_mat_path))
    Springer_pi_vector = pi_mat_data['Springer_pi_vector'].flatten()
    
    # 加载total_obs_distribution
    obs_mat_path = current_dir / 'Springer_total_obs_distribution.mat'
    obs_mat_data = loadmat(str(obs_mat_path))
    obs_cells = obs_mat_data['Springer_total_obs_distribution']
    # total_obs_distribution是一个(2,1)的cell array：[mean; cov]
    # mean应该是(4,)向量，cov应该是(4,4)矩阵
    Springer_total_obs_distribution = [
        obs_cells[0, 0].flatten(),  # mean: flatten to 1D array
        obs_cells[1, 0]   # covariance: keep as 2D array
    ]
    
    # 加载数据并重采样
    springer_options = default_Springer_HSMM_options()
    
    # 使用新的数据加载函数，支持WAV、CSV和TXT格式
    PCG, Fs1 = load_pcg_data(recordName)
    
    # 检查信号长度
    if len(PCG) <= round(2 * Fs1):
        features = np.zeros(318)
        classifyResult = 0
        return classifyResult, features
    
    # 检查信号中的零值比例
    if np.sum(PCG == 0) >= round(0.5 * len(PCG)):  # 超过一半的序列是零
        PCG = PCG + 0.01 * np.random.randn(len(PCG))
    
    # 重采样到1000 Hz
    num_samples = int(len(PCG) * springer_options['audio_Fs'] / Fs1)
    PCG_resampled = resample(PCG, num_samples)
    
    # 带通滤波（40-120 Hz）
    b, a = butter(5, [2*40/springer_options['audio_Fs'], 
                      2*120/springer_options['audio_Fs']], 
                  btype='band')
    PCG_band = filtfilt(b, a, PCG_resampled)
    PCG_band[np.isnan(PCG_band)] = 0
    PCG_band = normalise_signal(PCG_band)
    
    # 运行Springer分割算法获取assigned_states
    from runSpringerSegmentationAlgorithm_python import runSpringerSegmentationAlgorithm
    assigned_states = runSpringerSegmentationAlgorithm(
        PCG_resampled, springer_options['audio_Fs'],
        Springer_B_matrix, Springer_pi_vector,
        Springer_total_obs_distribution, False
    )
    
    # 从心音间隔提取特征
    try:
        from extractFeaturesFromHsIntervals_python import extractFeaturesFromHsIntervals
        featuresIntervals = extractFeaturesFromHsIntervals(assigned_states, PCG_resampled)
    except:
        featuresIntervals = np.zeros(48)
    
    # 从能量提取特征
    try:
        from extractFeaturesFromEnergy_python import extractFeaturesFromEnergy
        featuresEnergy = extractFeaturesFromEnergy(assigned_states, PCG_resampled)
    except:
        featuresEnergy = np.zeros(20)
    
    # 从频谱提取特征
    try:
        from extractFeaturesFromSpectrum_python import extractFeaturesFromSpectrum
        featuresSpectrum = extractFeaturesFromSpectrum(assigned_states, PCG_resampled)
    except:
        featuresSpectrum = np.zeros(82)
    
    # 从时变心率提取特征
    try:
        from extractFeaturesFromTimeVaryingHeartRate_python import extractFeaturesFromTimeVaryingHeartRate
        featuresHeartRate = extractFeaturesFromTimeVaryingHeartRate(PCG_band, springer_options['audio_Fs'])
    except:
        featuresHeartRate = np.zeros(59)
    
    # 从峰度提取特征
    try:
        from extractFeaturesFromKurtosis_python import extractFeaturesFromKurtosis
        featuresKurtosis = extractFeaturesFromKurtosis(assigned_states, PCG_resampled)
    except:
        featuresKurtosis = np.zeros(8)
    
    # 从循环平稳性提取特征
    try:
        from extractFeatreusFromCyclostationarity_python import extractFeatreusFromCyclostationarity
        featuresCyclostationarity = extractFeatreusFromCyclostationarity(PCG_resampled, springer_options['audio_Fs'])
    except:
        featuresCyclostationarity = np.zeros(2)
    
    # 根据参数决定是否提取小波能量特征
    if use_wavelet_feature:
        # 提取64维小波能量特征
        from PCG_Wavelet_Efeature_python import PCG_Wavelet_Efeature
        wp_feature = PCG_Wavelet_Efeature(PCG_resampled)
        
        # 生成标准特征（358维）
        standard_features = np.concatenate([
            featuresIntervals, featuresEnergy, featuresSpectrum,
            featuresHeartRate, featuresKurtosis, featuresCyclostationarity,
            featuresSpectrum**2, featuresHeartRate[2:]**2
        ])
        standard_features = standard_features[:358]  # 截取到358维
        
        # 追加小波特征（64维）
        features = np.concatenate([standard_features, wp_feature])
        # 最终特征：358 + 64 = 422维
    else:
        # 标准特征（不含小波）- 358维
        standard_features = np.concatenate([
            featuresIntervals, featuresEnergy, featuresSpectrum,
            featuresHeartRate, featuresKurtosis, featuresCyclostationarity,
            featuresSpectrum**2, featuresHeartRate[2:]**2
        ])
        features = standard_features[:358]
    
    # 运行分类函数获取最终分类结果
    classifyResult = classifyFromHsIntervals(features)
    
    return classifyResult, features
