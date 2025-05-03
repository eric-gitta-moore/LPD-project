# 环境设置指南

本项目使用 PaddlePaddle 深度学习框架。请按照以下步骤设置环境：

## 1. 同步项目依赖

在开始环境设置之前，首先需要同步项目依赖：

```bash
uv sync
```

这个命令会：
- 读取项目的依赖配置文件
- 安装所有必需的 Python 包
- 确保所有依赖版本的一致性

## 2. 设置 CUDA 库软链接

首先需要运行 setup-env.py 脚本来设置必要的 CUDA 库软链接：

```bash
sudo python3 setup-env.py
```

这个脚本会：
- 检查 CUDA 和 cuDNN 库是否正确安装
- 创建 libcudnn.so 软链接（指向 libcudnn.so.9）
- 创建 libcublas.so 软链接（指向 libcublas.so.11）

## 3. 设置 CUDA 库路径

运行以下命令设置必要的 CUDA 库路径：

```bash
source setup-env.sh
```

这个脚本会设置以下环境变量：
- 设置基础 WSL 库路径
- 设置 CUDA Runtime 库路径
- 设置 cuDNN 库路径
- 设置 cuBLAS 库路径

## 4. 验证环境设置

最后，运行以下命令检查 PaddlePaddle 环境是否正确安装：

```bash
python check.py
```

这将执行 PaddlePaddle 的环境检查，确保所有组件都正确安装和配置。

## 5. 依赖要求

本项目需要以下主要依赖：
- PaddlePaddle (GPU 版本)
- CUDA Toolkit
- cuDNN
- Python 3.x

## 6. 故障排除

如果遇到 CUDA 相关错误，请确保：
1. CUDA Toolkit 已正确安装
2. cuDNN 已正确安装
3. 已按正确顺序执行环境设置步骤：
   - 先运行 `setup-env.py`
   - 然后运行 `setup-env.sh`
   - 最后运行 `check.py`
4. PaddlePaddle 版本与 CUDA 版本匹配

## 7. 注意事项

- 确保使用 root 权限运行 setup-env.py（使用 sudo）
- 如果使用不同的 Python 环境，可能需要调整脚本中的路径
- WSL 用户需要确保 WSL 已正确配置 CUDA 支持
- 每次开启新的终端会话时，都需要重新运行 `setup-env.sh`
