"""
PCG数据读取工具：支持 CSV 和 WAV 文件，CSV 需有 pcg 列，WAV 自动识别采样率。
支持插值补全缺失值，可选尖峰去除预处理。
"""

import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.io import wavfile
from pathlib import Path


SUPPORTED_EXTS = {'.csv', '.wav'}


def find_csv_files(folder: str) -> list:
    """
    递归查找文件夹及所有子文件夹下的 CSV 和 WAV 文件。
    保留函数名以兼容旧接口。
    """
    return find_audio_files(folder)


def find_audio_files(folder: str) -> list:
    """
    递归查找文件夹及所有子文件夹下的 CSV 和 WAV 文件。

    Parameters
    ----------
    folder : 目标文件夹路径

    Returns
    -------
    sorted list of absolute file paths (str)
    """
    folder = Path(folder)
    files = []
    for ext in SUPPORTED_EXTS:
        files.extend(folder.rglob(f'*{ext}'))
    return sorted([str(f) for f in files])



def load_pcg_from_wav(filepath: str) -> dict:
    """
    从 WAV 文件加载 PCG 信号，自动识别采样率。
    多通道取第一通道。

    Returns
    -------
    dict: signal, fs, n_missing(=0), filename, filepath
    """
    try:
        fs, data = wavfile.read(filepath)
    except Exception as e:
        raise IOError(f"无法读取WAV文件 {filepath}: {e}")

    # 多通道取第一通道
    if data.ndim > 1:
        data = data[:, 0]

    # 转换为 float64 并归一化到 [-1, 1]
    if data.dtype == np.int16:
        signal = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        signal = data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.uint8:
        signal = (data.astype(np.float64) - 128.0) / 128.0
    else:
        signal = data.astype(np.float64)

    return {
        'signal':    signal,
        'fs':        float(fs),
        'n_missing': 0,
        'filename':  os.path.basename(filepath),
        'filepath':  filepath,
    }


def load_pcg_file(filepath: str,
                  pcg_col: str = 'pcg',
                  default_fs: float = 8000.0,
                  interp_method: str = 'linear') -> dict:
    """
    统一加载接口：根据文件属名自动分彺 CSV 和 WAV。
    """
    ext = Path(filepath).suffix.lower()
    if ext == '.wav':
        return load_pcg_from_wav(filepath)
    else:
        return load_pcg_from_csv(filepath, pcg_col=pcg_col,
                                 default_fs=default_fs,
                                 interp_method=interp_method)


def interpolate_pcg(series: pd.Series, method: str = 'linear') -> np.ndarray:
    """
    对PCG列进行缺失值插值补全。
    - 首尾缺失用最近有效值填充（外推）
    - 内部缺失用线性插值填充

    Parameters
    ----------
    series : PCG列（可能含NaN）
    method : 插值方法，'linear' 或 'cubic'

    Returns
    -------
    np.ndarray (float64)，无NaN
    """
    values = series.values.astype(float)
    n = len(values)
    valid_mask = ~np.isnan(values)

    if not np.any(valid_mask):
        raise ValueError("PCG列全部为NaN，无法插值")

    # 若无缺失值，直接返回
    if np.all(valid_mask):
        return values

    valid_idx = np.where(valid_mask)[0]
    valid_vals = values[valid_mask]

    # 内部插值
    interp_fn = interp1d(
        valid_idx, valid_vals,
        kind=method,
        bounds_error=False,
        fill_value=(valid_vals[0], valid_vals[-1])  # 首尾外推用端点值
    )
    all_idx = np.arange(n)
    interpolated = interp_fn(all_idx)
    return interpolated.astype(np.float64)


def load_pcg_from_csv(filepath: str,
                      pcg_col: str = 'pcg',
                      fs_col: str = None,
                      default_fs: float = 4000.0,
                      interp_method: str = 'linear') -> dict:
    """
    从单个CSV文件中加载PCG数据。

    Parameters
    ----------
    filepath     : CSV文件路径
    pcg_col      : PCG数据所在列名（大小写不敏感）
    fs_col       : 采样率列名（若CSV中有）
    default_fs   : 默认采样率（若CSV中无采样率信息）
    interp_method: 插值方法

    Returns
    -------
    dict:
      'signal'       : PCG信号 np.ndarray
      'fs'           : 采样率 float
      'n_missing'    : 插值填补的缺失值数量
      'filename'     : 文件名
      'filepath'     : 完整路径
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise IOError(f"无法读取文件 {filepath}: {e}")

    # 列名大小写不敏感匹配
    col_map = {c.lower().strip(): c for c in df.columns}
    pcg_col_lower = pcg_col.lower().strip()

    if pcg_col_lower not in col_map:
        available = list(df.columns)
        raise KeyError(
            f"文件 {os.path.basename(filepath)} 中未找到列 '{pcg_col}'。"
            f"可用列：{available}"
        )

    actual_col = col_map[pcg_col_lower]
    series = df[actual_col]

    n_missing = int(series.isna().sum())
    signal = interpolate_pcg(series, method=interp_method)

    # 尝试获取采样率
    fs = default_fs
    if fs_col is not None:
        fs_col_lower = fs_col.lower().strip()
        if fs_col_lower in col_map:
            fs_vals = df[col_map[fs_col_lower]].dropna()
            if len(fs_vals) > 0:
                fs = float(fs_vals.iloc[0])

    return {
        'signal': signal,
        'fs': fs,
        'n_missing': n_missing,
        'filename': os.path.basename(filepath),
        'filepath': filepath,
    }


def load_all_pcg_from_folder(folder: str,
                              pcg_col: str = 'pcg',
                              fs_col: str = None,
                              default_fs: float = 4000.0,
                              interp_method: str = 'linear') -> list:
    """
    递归读取文件夹下所有CSV文件的PCG数据。

    Returns
    -------
    list of dict（结构同 load_pcg_from_csv），成功读取的文件
    list of (filepath, error_msg)，读取失败的文件
    """
    csv_files = find_csv_files(folder)
    results = []
    errors = []
    for fp in csv_files:
        try:
            data = load_pcg_from_csv(fp, pcg_col, fs_col, default_fs, interp_method)
            results.append(data)
        except Exception as e:
            errors.append((fp, str(e)))
    return results, errors
