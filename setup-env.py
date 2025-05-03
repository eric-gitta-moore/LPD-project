#!/usr/bin/env python3

# 如果没有libcublas.so和libcudnn.so，则需要创建软链接

import os
import sys

# ln -s libcudnn.so.9 libcudnn.so
try:
    import nvidia.cudnn.lib
    CUDNN_PATH = os.path.dirname(nvidia.cudnn.lib.__file__)
    print(f"CUDNN 路径: {CUDNN_PATH}")
except ImportError:
    print("错误：未找到 nvidia.cudnn.lib，请确保已正确安装 CUDA 和 cuDNN")
    sys.exit(1)


# ln -s libcublas.so.11 libcublas.so
try:
    import nvidia.cublas.lib
    CUBLAS_PATH = os.path.dirname(nvidia.cublas.lib.__file__)
    print(f"CUBLAS 路径: {CUBLAS_PATH}")
except ImportError:
    print("错误：未找到 nvidia.cublas.lib，请确保已正确安装 CUDA 和 cuBLAS")
    sys.exit(1)

def create_symlink(base_path, source_file, target_file):
    """
    创建软链接的辅助函数
    
    参数:
        base_path (str): 基础路径
        source_file (str): 源文件名
        target_file (str): 目标文件名
    """
    source_path = os.path.join(base_path, source_file)
    target_path = os.path.join(base_path, target_file)
    
    try:
        if not os.path.exists(target_path):
            if os.path.exists(source_path):
                os.symlink(source_path, target_path)
                print(f"已创建软链接: {source_path} -> {target_path}")
            else:
                print(f"警告：源文件不存在: {source_path}")
        else:
            print(f"软链接已存在: {target_path}")
    except Exception as e:
        print(f"创建软链接时出错: {e}")

def main():
    """主函数"""
    if os.geteuid() != 0:
        print("错误：此脚本需要root权限运行")
        print("请使用 'sudo python3 setup-env.py' 运行此脚本")
        sys.exit(1)
    
    print("开始设置CUDA库软链接...")
    create_symlink(CUDNN_PATH, "libcudnn.so.8", "libcudnn.so")
    create_symlink(CUBLAS_PATH, "libcublas.so.11", "libcublas.so")
    print("设置完成！")

if __name__ == "__main__":
    main()
