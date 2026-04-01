"""
运行Springer分割算法 - Python版本
使用现成的Springer-Segmentation-Python实现（完整替换）

Developed for use in the paper:
D. Springer et al., "Logistic Regression-HSMM-based Heart Sound 
Segmentation," IEEE Trans. Biomed. Eng., In Press, 2015.
"""

import numpy as np
import sys
from pathlib import Path

# 添加springer_lib到路径
springer_lib_path = Path(__file__).parent / 'springer_lib'
if str(springer_lib_path) not in sys.path:
    sys.path.insert(0, str(springer_lib_path))

# 使用用户提供的Springer实现
from viterbi import viterbi_segment
from duration_distributions import DataDistribution
from extract_features import get_default_features
from heart_rate import get_heart_rate
from expand_qt_python import expand_qt
from normalise_signal_python import normalise_signal


def runSpringerSegmentationAlgorithm(audio_data, Fs, B_matrix, pi_vector, 
                                    total_observation_distribution, figures=False):
    """
    使用Springer分割算法为PCG记录分配状态（使用用户提供的完整实现）
    
    参数:
    audio_data: PCG记录的音频数据
    Fs: 音频记录的采样频率（重采样后的频率为1000Hz）
    B_matrix: HMM的观测矩阵（逻辑回归模型列表，每个状态一个）
    pi_vector: 初始状态分布
    total_observation_distribution: 所有数据的观测概率 [mean, cov]
    figures: （可选）布尔变量，用于显示图形
    
    返回:
    assigned_states: 分配给原始audio_data的状态值数组（原始采样频率）
    """
    
    # 使用用户提供的特征提取函数（来自extract_features.py）
    featuresFs = 50  # 特征采样频率
    PCG_Features = get_default_features(audio_data, Fs)
    
    # 使用用户提供的心率估计函数（来自heart_rate.py）
    heart_rates, systolic_time_intervals = get_heart_rate(audio_data, Fs, multiple_rates=False)
    heartRate = heart_rates[0]
    systolicTimeInterval = systolic_time_intervals[0]
    
    # 将B_matrix转换为模型对象列表
    # B_matrix是一个列表，每个元素是一个状态的逻辑回归系数
    class SimpleLogisticModel:
        def __init__(self, coef):
            self.coef = coef
        
        def predict_proba(self, X):
            """预测类别概率"""
            # 添加截距项
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
            
            # 逻辑回归: p = 1 / (1 + exp(-X*coef))
            z = np.dot(X_with_intercept, self.coef)
            # 数值稳定性
            z = np.clip(z, -500, 500)
            prob_class1 = 1 / (1 + np.exp(-z))
            prob_class0 = 1 - prob_class1
            
            # 返回 [P(class=0), P(class=1)]
            return np.column_stack([prob_class0, prob_class1])
    
    # 创建模型列表
    models = [SimpleLogisticModel(B_matrix[i]) for i in range(4)]
    
    # 创建DataDistribution实例（使用默认prior）
    data_distribution = DataDistribution(data=None, features_frequency=featuresFs)
    
    # 使用现成的viterbi_segment函数
    # 返回 (delta, psi, assigned_states)
    _, _, qt = viterbi_segment(
        observation_sequence=PCG_Features,
        models=models,
        total_obs_distribution=total_observation_distribution,
        distribution=data_distribution,  # 传递实例而不是类
        heart_rate=heartRate,
        systolic_time=systolicTimeInterval,
        recording_frequency=featuresFs
    )
    
    # 扩展到原始采样频率
    assigned_states = expand_qt(qt, featuresFs, Fs, len(audio_data))
    
    if figures:
        import matplotlib.pyplot as plt
        plt.figure()
        t1 = np.arange(len(audio_data)) / Fs
        plt.plot(t1, normalise_signal(audio_data), 'k', label='Audio data')
        plt.plot(t1, assigned_states, 'r--', label='Derived states')
        plt.legend()
        plt.title('Derived state sequence')
        plt.show()
    
    return assigned_states
