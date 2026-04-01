"""
功能：批量解压文件夹中的压缩包，并将所有文件提取到根目录，重命名以避免冲突
使用说明：
1. 运行脚本后，会弹出一个文件夹选择对话框，选择包含压缩包的文件夹。
2. 脚本会自动处理该文件夹中的所有.zip 压缩包，将其中的文件提取到根目录，并重命名为 "压缩包名_原文件名" 的格式。
3. 处理完成后，脚本会显示处理结果，并等待用户按回车键退出。         
"""

import os
import zipfile
import shutil
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

def select_folder():
    """弹出对话框选择文件夹"""
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="请选择包含压缩包的文件夹")
    return folder_path

def process_and_flatten():
    # 1. 获取用户选择的文件夹
    source_folder = select_folder()
    
    if not source_folder:
        print("未选择文件夹，程序退出。")
        return

    source_path = Path(source_folder)
    print(f"正在处理文件夹: {source_path}\n" + "-"*40)

    # 支持的压缩包后缀
    supported_extensions = ['.zip'] 
    
    # 统计数据
    count_archives = 0
    count_files = 0

    # 遍历文件夹寻找压缩包
    for item in source_path.iterdir():
        if item.is_file() and item.suffix.lower() in supported_extensions:
            count_archives += 1
            archive_name = item.stem  # 不带后缀的文件名 (例如 Data.zip -> Data)
            
            print(f"发现压缩包: {item.name}")
            
            # 创建一个临时的解压目录 (处理完后会删除)
            # 使用 .tmp_ 前缀，一般系统会视为隐藏或临时文件
            temp_extract_dir = source_path / f".tmp_{archive_name}_extract"
            
            try:
                # 2. 解压到临时目录
                if not temp_extract_dir.exists():
                    temp_extract_dir.mkdir()
                
                with zipfile.ZipFile(item, 'r') as zip_ref:
                    zip_ref.extractall(temp_extract_dir)
                
                # 3. 遍历临时目录，重命名并移动文件到根目录
                for root, dirs, files in os.walk(temp_extract_dir):
                    for file in files:
                        # 过滤掉系统隐藏文件 (Mac的__MACOSX 或 .DS_Store)
                        if file.startswith('.') or '__MACOSX' in root:
                            continue
                        
                        source_file_path = Path(root) / file
                        
                        # 构建新文件名：压缩包名_原文件名
                        new_filename = f"{archive_name}_{file}"
                        target_file_path = source_path / new_filename
                        
                        # 处理文件重名冲突 (如果根目录下已经存在同名文件)
                        counter = 1
                        while target_file_path.exists():
                            # 如果冲突，变成 Data_1_report.txt
                            new_filename = f"{archive_name}_{counter}_{file}"
                            target_file_path = source_path / new_filename
                            counter += 1
                        
                        # 移动文件
                        shutil.move(str(source_file_path), str(target_file_path))
                        print(f"   -> 提取: {new_filename}")
                        count_files += 1

                # 4. 删除临时目录
                shutil.rmtree(temp_extract_dir)
                print("   [完成] 清理临时目录。")

            except zipfile.BadZipFile:
                print(f"   [错误] {item.name} 文件损坏。")
                if temp_extract_dir.exists():
                    shutil.rmtree(temp_extract_dir)
            except Exception as e:
                print(f"   [错误] 处理 {item.name} 时出错: {e}")
                if temp_extract_dir.exists():
                    shutil.rmtree(temp_extract_dir)
            
            print("-" * 20)

    if count_archives == 0:
        print("未找到 .zip 压缩包。")
    else:
        print(f"\n全部完成！共处理 {count_archives} 个压缩包，提取了 {count_files} 个文件。")
        print(f"所有文件均已保存在: {source_path}")
        input("按回车键退出...")

if __name__ == "__main__":
    process_and_flatten()