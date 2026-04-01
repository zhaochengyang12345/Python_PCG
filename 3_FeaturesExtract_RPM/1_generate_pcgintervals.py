"""
PCG 心音间期特征批量提取脚本
从用户选择的文件夹中遍历所有 CSV 文件，提取心跳周期间期（RR, IntS1, IntS2, IntSys, IntDia）
输出到 2_FeaturesExtract_RPM/Series/ 目录，文件名格式：pcgintervals_<原文件名>.xlsx

路径结构（相对于本脚本所在目录 2_FeaturesExtract_RPM/）：
  ./springer_algo/   ← Springer 算法代码及模型文件
  ./CSV/             ← 输入目录（放置待处理的 CSV 文件）
  ./Series/          ← 输出目录
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import resample
from scipy.io import loadmat

# ========== 路径（相对于本脚本文件位置，无需修改） ==========
SCRIPT_DIR = Path(__file__).resolve().parent   # 2_FeaturesExtract_RPM/
PCG_DIR    = SCRIPT_DIR / "springer_algo"      # Springer 算法目录
INPUT_DIR  = SCRIPT_DIR / "CSV"               # 输入 CSV 目录
OUTPUT_DIR = SCRIPT_DIR / "Series"             # 输出到同目录下的 Series/
# ============================================================

AUDIO_FS   = 1000   # Springer 算法目标采样率
DEFAULT_FS = 8000   # CSV 文件默认原始采样率

# 将 PCG 特征提取程序目录加入 Python 路径（必须在导入 Springer 模块之前执行）
sys.path.insert(0, str(PCG_DIR))
from runSpringerSegmentationAlgorithm_python import runSpringerSegmentationAlgorithm  # type: ignore


def collect_csv_files(folder: Path):
    """递归收集文件夹及所有子文件夹下的 CSV 文件"""
    return sorted(folder.rglob("*.csv"))


def load_and_preprocess(csv_path: Path, target_fs: int = AUDIO_FS, source_fs: int = DEFAULT_FS):
    """加载 CSV，插值 NaN，重采样到 target_fs Hz"""
    df = pd.read_csv(csv_path)

    # 提取 pcg 列
    if 'pcg' in df.columns:
        pcg = df['pcg'].values.astype(np.float64)
    elif 'PCG' in df.columns:
        pcg = df['PCG'].values.astype(np.float64)
    else:
        raise ValueError(f"CSV 文件 '{csv_path.name}' 中未找到 'pcg' 列")

    # 线性插值填充 NaN
    nan_mask = np.isnan(pcg)
    if nan_mask.any():
        nan_count = nan_mask.sum()
        print(f"  检测到 {nan_count} 个空值，使用线性插值补全...")
        valid_idx = np.where(~nan_mask)[0]
        nan_idx = np.where(nan_mask)[0]
        if len(valid_idx) > 1:
            pcg[nan_idx] = np.interp(nan_idx, valid_idx, pcg[valid_idx])
        elif len(valid_idx) == 1:
            pcg[nan_idx] = pcg[valid_idx[0]]
        else:
            pcg[:] = 0.0

    # 检查是否有采样率列，否则使用默认值
    for col in ('fs', 'Fs', 'sampling_rate'):
        if col in df.columns:
            source_fs = float(df[col].iloc[0])
            break

    # 重采样
    num_samples = int(len(pcg) * target_fs / source_fs)
    pcg_resampled = resample(pcg, num_samples)
    print(f"  采样率 {source_fs:.0f}→{target_fs} Hz，重采样后 {num_samples} 个样本")
    return pcg_resampled


def process_csv_file(csv_path: Path, springer_params: dict) -> bool:
    """
    处理单个 CSV 文件，生成对应的间期 Excel 文件到 OUTPUT_DIR
    返回 True 表示成功，False 表示跳过/失败
    """
    print(f"\n[处理] {csv_path}")

    # 输出文件名：pcgintervals_<原文件名>.xlsx（保持子目录结构可选；此处展开到同一 Series 目录）
    out_name = f"pcgintervals_{csv_path.stem}.xlsx"
    out_path = OUTPUT_DIR / out_name

    try:
        # 1. 加载并预处理
        pcg_resampled = load_and_preprocess(csv_path)

        # 2. 有效性检查
        if len(pcg_resampled) <= round(2 * AUDIO_FS):
            print(f"  警告: 信号过短（{len(pcg_resampled)} 样本），跳过")
            return False

        if np.sum(pcg_resampled == 0) >= round(0.5 * len(pcg_resampled)):
            pcg_resampled = pcg_resampled + 0.01 * np.random.randn(len(pcg_resampled))

        # 3. Springer 分割
        print(f"  运行 Springer 分割...")
        assigned_states = run_springer_segmentation_with_params(pcg_resampled, springer_params)

        # 4. 提取间期
        df_intervals = extract_per_cycle_intervals(assigned_states)
        print(f"  提取 {len(df_intervals)} 个心跳周期")

        # 5. 保存
        df_intervals.to_excel(out_path, index=False)
        print(f"  已保存 → {out_path.relative_to(SCRIPT_DIR)}")
        return True

    except Exception as e:
        print(f"  错误: {e}")
        return False


def load_springer_params() -> dict:
    """预加载 Springer 模型参数（避免每个文件重复读取 mat 文件）"""
    B_data = loadmat(str(PCG_DIR / "Springer_B_matrix.mat"))
    B_cells = B_data['Springer_B_matrix']
    springer_B = [B_cells[0, i].flatten() for i in range(4)]

    pi_data = loadmat(str(PCG_DIR / "Springer_pi_vector.mat"))
    springer_pi = pi_data['Springer_pi_vector'].flatten()

    obs_data = loadmat(str(PCG_DIR / "Springer_total_obs_distribution.mat"))
    obs_cells = obs_data['Springer_total_obs_distribution']
    springer_obs = [obs_cells[0, 0].flatten(), obs_cells[1, 0]]

    return {'B': springer_B, 'pi': springer_pi, 'obs': springer_obs}


def run_springer_segmentation_with_params(pcg_resampled, params: dict, audio_fs: int = AUDIO_FS):
    """使用已加载的参数运行 Springer 分割（避免重复 loadmat）"""
    return runSpringerSegmentationAlgorithm(
        pcg_resampled, audio_fs, params['B'], params['pi'], params['obs'], False
    )


def run_springer_segmentation(pcg_resampled, audio_fs: int = AUDIO_FS):
    """加载 Springer 模型参数，运行分割算法"""
    params = load_springer_params()
    return run_springer_segmentation_with_params(pcg_resampled, params, audio_fs)


def extract_per_cycle_intervals(assigned_states):
    """
    从 assigned_states 提取每个心跳周期的时间间期（单位：ms，1000Hz采样时即为毫秒）
    返回 DataFrame，列为 RR, IntS1, IntS2, IntSys, IntDia
    """
    indx = np.where(np.abs(np.diff(assigned_states)) > 0)[0]

    if assigned_states[0] > 0:
        state_map = {4: 1, 3: 2, 2: 3, 1: 4}
        K = state_map.get(int(assigned_states[0]), 1)
    else:
        state_map = {4: 1, 3: 2, 2: 3, 1: 0}
        K = state_map.get(int(assigned_states[indx[0] + 1]), 0) + 1

    indx2 = indx[K - 1:]
    rem = len(indx2) % 4
    if rem > 0:
        indx2 = indx2[:-rem]

    A = indx2.reshape(-1, 4)  # N×4，N = 完整心跳周期数
    N = len(A)
    print(f"检测到 {N} 个完整心跳周期")

    if N < 2:
        raise ValueError("检测到的心跳周期数不足（<2），无法计算间期")

    # 对每个相邻周期对 i=0..N-2 计算间期
    rows = []
    for i in range(N - 1):
        rr     = int(A[i + 1, 0] - A[i, 0])
        int_s1  = int(A[i, 1] - A[i, 0])
        int_s2  = int(A[i, 3] - A[i, 2])
        int_sys = int(A[i, 2] - A[i, 1])
        int_dia = int(A[i + 1, 0] - A[i, 3])
        rows.append({
            'RR':     rr,
            'IntS1':  int_s1,
            'IntS2':  int_s2,
            'IntSys': int_sys,
            'IntDia': int_dia,
        })

    df_out = pd.DataFrame(rows, columns=['RR', 'IntS1', 'IntS2', 'IntSys', 'IntDia'])
    return df_out


def main():
    print("=" * 60)
    print("PCG 心音间期批量提取")
    print(f"算法目录: {PCG_DIR.relative_to(SCRIPT_DIR)}")
    print(f"输入目录: {INPUT_DIR.relative_to(SCRIPT_DIR)}")
    print(f"输出目录: {OUTPUT_DIR.relative_to(SCRIPT_DIR)}")
    print("=" * 60)

    # 1. 检查输入目录
    if not INPUT_DIR.exists():
        print(f"错误: 输入目录不存在: {INPUT_DIR}")
        print("请在 2_FeaturesExtract_RPM/ 下新建 CSV 文件夹并放入待处理文件")
        return

    # 2. 收集所有 CSV 文件
    csv_files = collect_csv_files(INPUT_DIR)
    if not csv_files:
        print("所选文件夹中未找到任何 CSV 文件，程序退出")
        return
    print(f"共找到 {len(csv_files)} 个 CSV 文件")

    # 3. 确保输出目录存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 4. 预加载 Springer 模型参数（只加载一次）
    print("\n加载 Springer 模型参数...")
    springer_params = load_springer_params()
    print("模型参数加载完成")

    # 5. 遍历处理
    success_count = 0
    fail_count = 0
    for i, csv_path in enumerate(csv_files, 1):
        print(f"\n[{i}/{len(csv_files)}] {csv_path.name}")
        ok = process_csv_file(csv_path, springer_params)
        if ok:
            success_count += 1
        else:
            fail_count += 1

    # 6. 汇总
    print("\n" + "=" * 60)
    print(f"处理完成: 成功 {success_count} 个，跳过/失败 {fail_count} 个")
    print(f"结果已保存至: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
