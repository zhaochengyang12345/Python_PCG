"""
Springer HSMM算法的默认选项 - Python版本
从MATLAB代码转译

Developed for use in the paper:
D. Springer et al., "Logistic Regression-HSMM-based Heart Sound 
Segmentation," IEEE Trans. Biomed. Eng., In Press, 2015.
"""


def default_Springer_HSMM_options():
    """
    返回Springer分割算法使用的默认选项
    
    返回:
    springer_options: 包含算法选项的字典
    """
    
    springer_options = {
        # 提取信号特征的采样频率
        'audio_Fs': 1000,
        
        # 下采样频率（在Springer论文中设置为50）
        'audio_segmentation_Fs': 50,
        
        # S1和S2定位的容差（秒）
        'segmentation_tolerance': 0.1,
        
        # 是否使用mex代码
        'use_mex': True,
        
        # 是否使用小波函数
        'include_wavelet_feature': True
    }
    
    return springer_options
