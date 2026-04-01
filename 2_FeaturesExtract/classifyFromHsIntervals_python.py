"""
心音分类函数 - Python版本
从MATLAB代码转译

此函数使用获取的特征对心音记录进行分类

Written by: Chengyu Liu, January 22 2016
            chengyu.liu@emory.edu
"""

import numpy as np
from scipy.special import expit  # logistic sigmoid function


def classifyFromHsIntervals(features):
    """
    使用获取的特征对心音记录进行分类
    
    参数:
    features: 获取的特征向量
    
    返回:
    classifyResult: 分类结果
                    1 = 异常记录
                   -1 = 正常记录
                    0 = 不确定的记录（噪声过多）
    
    说明:
    以下分类规则（公式）基于在平衡训练数据库上使用逻辑回归模型
    从20个特征中进行特征选择的结果。
    您可以基于获取的特征或您生成的特征或您认为有用的其他信息
    构建更准确的分类规则
    """
    
    # 处理不同特征数量的情况
    num_features = len(features)
    if num_features < 350:
        # 如果特征数少于350，用0填充到350以匹配B矩阵
        features = np.pad(features, (0, 350 - num_features), 'constant', constant_values=0)
    elif num_features > 350:
        # 如果特征数多于350，只取前350个
        features = features[:350]
    
    # B矩阵由在验证集上训练逻辑回归模型获得
    # 强烈建议在所有训练集上重新训练逻辑回归模型以更新B矩阵，以获得更准确的分类结果
    
    # 基于350个特征的B矩阵（全1，需要用实际训练的权重替换）
    B = np.ones(351)  # 包括截距项
    
    # 计算预测值（logistic回归）
    # predictor = 1 / (1 + exp(-(B[0] + sum(B[1:] * features))))
    z = B[0] + np.dot(B[1:], features)
    predictor = expit(z)  # sigmoid函数
    
    # 分类
    # 我们只提供正常/异常分类，如果您认为当前记录噪声太大，可以设置classifyResult=0
    # 评分函数会考虑classifyResult=0的情况
    thr = 0.5  # 分类阈值，thr>0.5表示异常记录
    if predictor > thr:
        classifyResult = 1
    else:
        classifyResult = -1
    
    return classifyResult
