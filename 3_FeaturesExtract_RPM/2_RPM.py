"""
RPM 批量生成脚本
输入: 2_FeaturesExtract_RPM/Series/ 下所有 .xlsx 文件
输出: 2_FeaturesExtract_RPM/RPM/ 下对应的 PNG 图片
命名格式: <原文件名>_rpm_<列名>.png
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR  = SCRIPT_DIR / "Series"
OUTPUT_DIR = SCRIPT_DIR / "RPM"


def relative_position_matrix(x, k=1):
    # z-score 归一化
    z = (x - np.mean(x)) / np.std(x)
    N = len(z)
    m = int(np.ceil(N / k))
    # PAA 降维
    X = [np.mean(z[i*k : min((i+1)*k, N)]) for i in range(m)]
    X = np.array(X)
    # 构造相对位置矩阵
    M = X.reshape(-1, 1) - X.reshape(1, -1)
    # min-max 归一化到 [0, 255]
    rpm = (M - M.min()) / (M.max() - M.min()) * 255
    return rpm


def process_file(xlsx_path: Path):
    df = pd.read_excel(xlsx_path)
    if df.empty or len(df.columns) < 2:
        print(f"  跳过（列数不足）: {xlsx_path.name}")
        return

    # 取第二列（与原代码一致）
    col_name = df.columns[1]
    x = df.iloc[:, 1].dropna().values.astype(float)
    if len(x) < 2:
        print(f"  跳过（数据不足）: {xlsx_path.name}")
        return

    rpm = relative_position_matrix(x, k=1)
    out_path = OUTPUT_DIR / f"{xlsx_path.stem}_rpm_{col_name}.png"
    plt.imsave(str(out_path), rpm, cmap='jet')
    print(f"  已保存: {out_path.name}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    xlsx_files = sorted(INPUT_DIR.rglob("*.xlsx"))
    if not xlsx_files:
        print(f"在 {INPUT_DIR} 中未找到任何 .xlsx 文件")
        return

    print(f"共找到 {len(xlsx_files)} 个文件，开始处理...\n")
    for i, f in enumerate(xlsx_files, 1):
        print(f"[{i}/{len(xlsx_files)}] {f.name}")
        process_file(f)

    print(f"\n全部完成，图片已保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
