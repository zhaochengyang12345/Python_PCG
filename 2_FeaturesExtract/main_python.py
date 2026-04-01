"""
PCG特征提取主程序 - Python版本
从MATLAB代码转译
"""

import os
import numpy as np
import pandas as pd
import time
from pathlib import Path
from challenge_python import challenge

# ========== 特征提取配置 ==========
FEATURE_MODE = 358  # 修改此值来选择要提取的特征数量

# 小波能量特征开关 (True=开启64维小波特征, False=不开启)
USE_WAVELET_FEATURE = True  # 修改为True以启用小波能量特征
# ====================================

# 获取脚本所在目录
script_dir = Path(__file__).parent
data_dir = script_dir / 'TEST'

# 将配置设置为全局变量
globals()['GLOBAL_FEATURE_MODE'] = FEATURE_MODE
globals()['GLOBAL_USE_WAVELET'] = USE_WAVELET_FEATURE

# 读取记录列表
records_file = data_dir / 'RECORDS'
if records_file.exists():
    with open(records_file, 'r', encoding='utf-8') as f:
        RECORDS = [line.strip() for line in f if line.strip()]
else:
    print(f'找不到文件: {records_file}. 正在从目录读取文件...')
    # 从目录读取所有csv文件
    RECORDS = [f.stem for f in data_dir.glob('*.csv')]

if not RECORDS:
    raise FileNotFoundError(f'在 {data_dir} 中未找到记录文件')

# 运行验证集并获取评分结果
classifyResult = np.zeros(len(RECORDS), dtype=int)
total_time = 0

# 显示配置信息
print('\n========== 特征提取配置 ==========')
print(f'标准特征数量: {FEATURE_MODE}')
if USE_WAVELET_FEATURE:
    print('小波能量特征: 开启 (+64维)')
    print(f'实际总特征数: {min(FEATURE_MODE, 358) + 64}')
else:
    print('小波能量特征: 关闭')
    print(f'实际总特征数: {FEATURE_MODE}')

if FEATURE_MODE == 48:
    print('说明: 仅时间间隔特征')
elif FEATURE_MODE == 82:
    print('说明: 时间特征 + 频谱特征')
elif FEATURE_MODE == 131:
    print('说明: 时间 + 能量 + 频谱 + 峰度 + 循环特征')
elif FEATURE_MODE == 350:
    print('说明: 标准版本（包含平方项，不含最后8特征）')
elif FEATURE_MODE == 358:
    print('说明: 完整版本（包含所有特征）')
else:
    raise ValueError(f'不支持的特征数量: {FEATURE_MODE}. 可选择 48, 82, 131, 350 或 358')

print('=================================\n')

# 打开答案文件
answers_file = script_dir / 'answers.txt'
fid = open(answers_file, 'w', encoding='utf-8')

# 完整的特征名称列表(358个)
full_feature_names = ['m_RR','sd_RR','m_IntS1','sd_IntS1','m_IntS2','sd_IntS2','m_IntSys','sd_IntSys','m_IntDia','sd_IntDia','m_Ratio_SysRR','sd_Ratio_SysRR','m_Ratio_DiaRR','sd_Ratio_DiaRR','m_Ratio_SysDia','sd_Ratio_SysDia','m_Amp_SysS1','sd_Amp_SysS1','m_Amp_DiaS2','sd_Amp_DiaS2','m_HFAll_S1','m_HFAll_Sys','m_HFAll_S2','m_HFAll_Dia','m_LFAll_S1','m_LFAll_Sys','m_LFAll_S2','m_LFAll_Dia','sd_HFAll_S1','sd_HFAll_Sys','sd_HFAll_S2','sd_HFAll_Dia','sd_LFAll_S1','sd_LFAll_Sys','sd_LFAll_S2','sd_LFAll_Dia','m_SampEn_Sys','sd_SampEn_Sys','m_SampEn_Dia','sd_SampEn_Dia','m_FuzzyEn_Sys','sd_FuzzyEn_Sys','m_FuzzyEn_Dia','sd_FuzzyEn_Dia','m_DistEn_Sys','sd_DistEn_Sys','m_DistEn_Dia','sd_DistEn_Dia','m_energy_SysTotal','sd_energy_SysTotal','m_energy_DiaTotal','sd_energy_DiaTotal','m_energy_S1Total','sd_energy_S1Total','m_energy_S2Total','sd_energy_S2Total','m_energy_HsTotal','sd_energy_HsTotal','m_energy_S1ToSys','sd_energy_S1ToSys','m_energy_S1ToDia','sd_energy_S1ToDia','m_energy_S2ToSys','sd_energy_S2ToSys','m_energy_S2ToDia','sd_energy_S2ToDia','m_energy_DiaToSys','sd_energy_DiaToSys','mSpectrum_S1_2','mSpectrum_S1_3','mSpectrum_S1_4','mSpectrum_S1_5','mSpectrum_S1_6','mSpectrum_S1_7','mSpectrum_S1_8','mSpectrum_S1_9','mSpectrum_S1_10','mSpectrum_S1_11','mSpectrum_S1_12','mSpectrum_S1_13','mSpectrum_S2_2','mSpectrum_S2_3','mSpectrum_S2_4','mSpectrum_S2_5','mSpectrum_S2_6','mSpectrum_S2_7','mSpectrum_S2_8','mSpectrum_S2_9','mSpectrum_S2_10','mSpectrum_S2_11','mSpectrum_S2_12','mSpectrum_S2_13','mSpectrum_Sys_2','mSpectrum_Sys_3','mSpectrum_Sys_4','mSpectrum_Sys_5','mSpectrum_Sys_6','mSpectrum_Sys_7','mSpectrum_Sys_8','mSpectrum_Sys_9','mSpectrum_Sys_10','mSpectrum_Sys_11','mSpectrum_Sys_12','mSpectrum_Sys_13','mSpectrum_Sys_14','mSpectrum_Sys_15','mSpectrum_Sys_16','mSpectrum_Sys_17','mSpectrum_Sys_18','mSpectrum_Sys_19','mSpectrum_Sys_20','mSpectrum_Sys_21','mSpectrum_Sys_22','mSpectrum_Sys_23','mSpectrum_Sys_24','mSpectrum_Sys_25','mSpectrum_Sys_26','mSpectrum_Sys_27','mSpectrum_Sys_28','mSpectrum_Sys_29','mSpectrum_Sys_30','mSpectrum_Dia_2','mSpectrum_Dia_3','mSpectrum_Dia_4','mSpectrum_Dia_5','mSpectrum_Dia_6','mSpectrum_Dia_7','mSpectrum_Dia_8','mSpectrum_Dia_9','mSpectrum_Dia_10','mSpectrum_Dia_11','mSpectrum_Dia_12','mSpectrum_Dia_13','mSpectrum_Dia_14','mSpectrum_Dia_15','mSpectrum_Dia_16','mSpectrum_Dia_17','mSpectrum_Dia_18','mSpectrum_Dia_19','mSpectrum_Dia_20','mSpectrum_Dia_21','mSpectrum_Dia_22','mSpectrum_Dia_23','mSpectrum_Dia_24','mSpectrum_Dia_25','mSpectrum_Dia_26','mSpectrum_Dia_27','mSpectrum_Dia_28','mSpectrum_Dia_29','mSpectrum_Dia_30','mean_cyclePeriod','std_cyclePeriod','spectrum_cyclePeriod_2','spectrum_cyclePeriod_3','spectrum_cyclePeriod_4','spectrum_cyclePeriod_5','spectrum_cyclePeriod_6','spectrum_cyclePeriod_7','spectrum_cyclePeriod_8','spectrum_cyclePeriod_9','spectrum_cyclePeriod_10','spectrum_cyclePeriod_11','spectrum_cyclePeriod_12','spectrum_cyclePeriod_13','spectrum_cyclePeriod_14','spectrum_cyclePeriod_15','spectrum_cyclePeriod_16','spectrum_cyclePeriod_17','spectrum_cyclePeriod_18','spectrum_cyclePeriod_19','spectrum_cyclePeriod_20','spectrum_systolic_2','spectrum_systolic_3','spectrum_systolic_4','spectrum_systolic_5','spectrum_systolic_6','spectrum_systolic_7','spectrum_systolic_8','spectrum_systolic_9','spectrum_systolic_10','spectrum_systolic_11','spectrum_systolic_12','spectrum_systolic_13','spectrum_systolic_14','spectrum_systolic_15','spectrum_systolic_16','spectrum_systolic_17','spectrum_systolic_18','spectrum_systolic_19','spectrum_systolic_20','spectrum_diastolic_2','spectrum_diastolic_3','spectrum_diastolic_4','spectrum_diastolic_5','spectrum_diastolic_6','spectrum_diastolic_7','spectrum_diastolic_8','spectrum_diastolic_9','spectrum_diastolic_10','spectrum_diastolic_11','spectrum_diastolic_12','spectrum_diastolic_13','spectrum_diastolic_14','spectrum_diastolic_15','spectrum_diastolic_16','spectrum_diastolic_17','spectrum_diastolic_18','spectrum_diastolic_19','spectrum_diastolic_20','mean_s1_kurtosis','std_s1_kurtosis','mean_s2_kurtosis','std_s2_kurtosis','mean_sys_kurtosis','std_sys_kurtosis','mean_dia_kurtosis','std_dia_kurtosis','mean_cyclostationarity','std_cyclostationarity','mSpectrum_S1_2_sq','mSpectrum_S1_3_sq','mSpectrum_S1_4_sq','mSpectrum_S1_5_sq','mSpectrum_S1_6_sq','mSpectrum_S1_7_sq','mSpectrum_S1_8_sq','mSpectrum_S1_9_sq','mSpectrum_S1_10_sq','mSpectrum_S1_11_sq','mSpectrum_S1_12_sq','mSpectrum_S1_13_sq','mSpectrum_S2_2_sq','mSpectrum_S2_3_sq','mSpectrum_S2_4_sq','mSpectrum_S2_5_sq','mSpectrum_S2_6_sq','mSpectrum_S2_7_sq','mSpectrum_S2_8_sq','mSpectrum_S2_9_sq','mSpectrum_S2_10_sq','mSpectrum_S2_11_sq','mSpectrum_S2_12_sq','mSpectrum_S2_13_sq','mSpectrum_Sys_2_sq','mSpectrum_Sys_3_sq','mSpectrum_Sys_4_sq','mSpectrum_Sys_5_sq','mSpectrum_Sys_6_sq','mSpectrum_Sys_7_sq','mSpectrum_Sys_8_sq','mSpectrum_Sys_9_sq','mSpectrum_Sys_10_sq','mSpectrum_Sys_11_sq','mSpectrum_Sys_12_sq','mSpectrum_Sys_13_sq','mSpectrum_Sys_14_sq','mSpectrum_Sys_15_sq','mSpectrum_Sys_16_sq','mSpectrum_Sys_17_sq','mSpectrum_Sys_18_sq','mSpectrum_Sys_19_sq','mSpectrum_Sys_20_sq','mSpectrum_Sys_21_sq','mSpectrum_Sys_22_sq','mSpectrum_Sys_23_sq','mSpectrum_Sys_24_sq','mSpectrum_Sys_25_sq','mSpectrum_Sys_26_sq','mSpectrum_Sys_27_sq','mSpectrum_Sys_28_sq','mSpectrum_Sys_29_sq','mSpectrum_Sys_30_sq','mSpectrum_Dia_2_sq','mSpectrum_Dia_3_sq','mSpectrum_Dia_4_sq','mSpectrum_Dia_5_sq','mSpectrum_Dia_6_sq','mSpectrum_Dia_7_sq','mSpectrum_Dia_8_sq','mSpectrum_Dia_9_sq','mSpectrum_Dia_10_sq','mSpectrum_Dia_11_sq','mSpectrum_Dia_12_sq','mSpectrum_Dia_13_sq','mSpectrum_Dia_14_sq','mSpectrum_Dia_15_sq','mSpectrum_Dia_16_sq','mSpectrum_Dia_17_sq','mSpectrum_Dia_18_sq','mSpectrum_Dia_19_sq','mSpectrum_Dia_20_sq','mSpectrum_Dia_21_sq','mSpectrum_Dia_22_sq','mSpectrum_Dia_23_sq','mSpectrum_Dia_24_sq','mSpectrum_Dia_25_sq','mSpectrum_Dia_26_sq','mSpectrum_Dia_27_sq','mSpectrum_Dia_28_sq','mSpectrum_Dia_29_sq','mSpectrum_Dia_30_sq','spectrum_cyclePeriod_4_sq','spectrum_cyclePeriod_5_sq','spectrum_cyclePeriod_6_sq','spectrum_cyclePeriod_7_sq','spectrum_cyclePeriod_8_sq','spectrum_cyclePeriod_9_sq','spectrum_cyclePeriod_10_sq','spectrum_cyclePeriod_11_sq','spectrum_cyclePeriod_12_sq','spectrum_cyclePeriod_13_sq','spectrum_cyclePeriod_14_sq','spectrum_cyclePeriod_15_sq','spectrum_cyclePeriod_16_sq','spectrum_cyclePeriod_17_sq','spectrum_cyclePeriod_18_sq','spectrum_cyclePeriod_19_sq','spectrum_cyclePeriod_20_sq','spectrum_cyclePeriod_21_sq','spectrum_cyclePeriod_22_sq','spectrum_systolic_4_sq','spectrum_systolic_5_sq','spectrum_systolic_6_sq','spectrum_systolic_7_sq','spectrum_systolic_8_sq','spectrum_systolic_9_sq','spectrum_systolic_10_sq','spectrum_systolic_11_sq','spectrum_systolic_12_sq','spectrum_systolic_13_sq','spectrum_systolic_14_sq','spectrum_systolic_15_sq','spectrum_systolic_16_sq','spectrum_systolic_17_sq','spectrum_systolic_18_sq','spectrum_systolic_19_sq','spectrum_systolic_20_sq','spectrum_systolic_21_sq','spectrum_systolic_22_sq','spectrum_diastolic_4_sq','spectrum_diastolic_5_sq','spectrum_diastolic_6_sq','spectrum_diastolic_7_sq','spectrum_diastolic_8_sq','spectrum_diastolic_9_sq','spectrum_diastolic_10_sq','spectrum_diastolic_11_sq','spectrum_diastolic_12_sq','spectrum_diastolic_13_sq','spectrum_diastolic_14_sq','spectrum_diastolic_15_sq','spectrum_diastolic_16_sq','spectrum_diastolic_17_sq','spectrum_diastolic_18_sq','spectrum_diastolic_19_sq','spectrum_diastolic_20_sq','spectrum_diastolic_21_sq','spectrum_diastolic_22_sq']

# 根据FEATURE_MODE选择标准特征（不受小波开关影响），并添加序号
selected_features = full_feature_names[:min(FEATURE_MODE, len(full_feature_names))]
feature_names = [f'{i+1}.{name}' for i, name in enumerate(selected_features)]
features_length = len(feature_names)

# 如果开启小波特征，额外添加64维小波特征名称（带序号）
if USE_WAVELET_FEATURE:
    wavelet_feature_names = [f'{features_length + i + 1}.wavelet_energy_{i+1}' for i in range(64)]
    feature_names = feature_names + wavelet_feature_names
    features_length = features_length + 64  # 额外增加64维

# 初始化特征矩阵
features_1 = np.zeros((len(RECORDS), features_length))
filenames_list = []

# 处理每条记录
for i, fname in enumerate(RECORDS):
    filenames_list.append(fname)
    start_time = time.time()
    
    # 调用challenge函数
    record_path = str(data_dir / fname)
    classifyResult[i], features_all = challenge(record_path, use_wavelet_feature=USE_WAVELET_FEATURE)
    
    # 将分类结果写入answers.txt文件
    fid.write(f'{fname},{classifyResult[i]}\n')
    
    # 收集特征数据（根据实际提取的特征长度）
    actual_feature_length = min(len(features_all), features_length)
    current_features = features_all[:actual_feature_length]
    features_1[i, :actual_feature_length] = current_features
    
    elapsed_time = time.time() - start_time
    total_time += elapsed_time
    print(f'---已处理 {i+1}/{len(RECORDS)} 条记录.')

fclose = fid.close()

# 保存到Excel文件，包含表头和文件名列
output_data = pd.DataFrame(features_1, columns=feature_names)
output_data.insert(0, 'Filename', filenames_list)
output_data.to_excel(script_dir / 'features.xlsx', index=False)

average_time = total_time / len(RECORDS)
print(f'\n验证集生成完成.')
print(f'  总时间 = {total_time:.2f}秒')
print(f'  平均时间 = {average_time:.2f}秒')

print(f'\n答案文件已创建为 answers.txt.')
print(f'处理完成.')
if USE_WAVELET_FEATURE:
    print(f'特征已保存到 features.xlsx，共 {features_length} 个特征.')
    print(f'  (包含 {min(FEATURE_MODE, 358)} 标准特征 + 64 小波特征)')
else:
    print(f'特征已保存到 features.xlsx，共 {features_length} 个特征.')
