"""
加载PCG数据 - Python版本
从MATLAB代码转译
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import wavfile
import soundfile as sf


def load_pcg_data(recordName):
    """
    加载PCG数据，支持WAV和CSV两种格式
    
    参数:
    recordName: 文件路径（不含扩展名）
    
    返回:
    PCG: 心音信号数据
    Fs: 采样频率
    
    支持的格式：
    - .wav: 音频文件
    - .csv: CSV文件，需包含'pcg'列，第一行为表头
    """
    
    # 默认采样频率
    DEFAULT_FS = 8000
    
    record_path = Path(recordName)
    
    # 尝试读取WAV文件
    wav_file = record_path.with_suffix('.wav')
    if wav_file.exists():
        try:
            # 使用soundfile读取（支持更多格式）
            PCG, Fs = sf.read(str(wav_file))
            # 如果是立体声，取第一个声道
            if len(PCG.shape) > 1:
                PCG = PCG[:, 0]
            return PCG, Fs
        except:
            # 使用scipy.io.wavfile作为备选
            Fs, PCG = wavfile.read(str(wav_file))
            # 转换为浮点数
            if PCG.dtype == np.int16:
                PCG = PCG.astype(np.float32) / 32768.0
            elif PCG.dtype == np.int32:
                PCG = PCG.astype(np.float32) / 2147483648.0
            # 如果是立体声，取第一个声道
            if len(PCG.shape) > 1:
                PCG = PCG[:, 0]
            return PCG, Fs
    
    # 尝试读取CSV文件
    csv_file = record_path.with_suffix('.csv')
    if csv_file.exists():
        try:
            # 使用pandas读取CSV文件
            data_table = pd.read_csv(csv_file)
            
            # 检查是否存在'pcg'列
            if 'pcg' in data_table.columns:
                PCG = data_table['pcg'].values
            elif 'PCG' in data_table.columns:
                PCG = data_table['PCG'].values
            else:
                raise ValueError("CSV文件中未找到'pcg'或'PCG'列")
            
            # 转换为浮点数
            PCG = PCG.astype(np.float64)
            
            # 处理空值（NaN）：使用线性插值
            if np.any(np.isnan(PCG)):
                nan_count = np.sum(np.isnan(PCG))
                print(f'检测到 {nan_count} 个空值，使用插值补全...')
                
                # 找到非NaN的索引
                valid_idx = np.where(~np.isnan(PCG))[0]
                nan_idx = np.where(np.isnan(PCG))[0]
                
                if len(valid_idx) > 1:
                    # 使用线性插值填充NaN值
                    PCG[nan_idx] = np.interp(nan_idx, valid_idx, PCG[valid_idx])
                elif len(valid_idx) == 1:
                    # 如果只有一个有效值，用该值填充所有NaN
                    PCG[nan_idx] = PCG[valid_idx[0]]
                else:
                    # 如果全是NaN，用0填充
                    PCG[:] = 0
                    print('警告：所有值均为空，已用0填充')
            
            # 检查是否有采样频率列
            if 'fs' in data_table.columns:
                Fs = float(data_table['fs'].iloc[0])
            elif 'Fs' in data_table.columns:
                Fs = float(data_table['Fs'].iloc[0])
            elif 'sampling_rate' in data_table.columns:
                Fs = float(data_table['sampling_rate'].iloc[0])
            else:
                # 使用默认采样频率
                Fs = DEFAULT_FS
                print(f'未找到采样频率信息，使用默认值: {DEFAULT_FS} Hz')
            
            # 确保PCG是一维数组
            if len(PCG.shape) > 1:
                PCG = PCG.flatten()
            
            return PCG, Fs
            
        except Exception as e:
            raise IOError(f'读取CSV文件失败: {str(e)}')
    
    # 尝试读取TXT文件（兼容原有格式）
    txt_file = record_path.with_suffix('.txt')
    if txt_file.exists():
        PCG = np.loadtxt(str(txt_file))
        Fs = DEFAULT_FS
        return PCG, Fs
    
    # 如果所有格式都不存在
    raise FileNotFoundError(f'未找到文件: {recordName} (.wav, .csv, 或 .txt)')
