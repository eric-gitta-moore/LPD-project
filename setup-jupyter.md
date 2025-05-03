# Jupyter Lab 设置和使用指南

## 环境准备

首先，需要设置正确的环境变量：

```bash
source setup-env.sh
```

## 启动 Jupyter Lab

设置好环境变量后，可以直接启动 Jupyter Lab：

```bash
jupyter lab
```

启动后，Jupyter Lab 会在浏览器中自动打开一个新标签页。如果没有自动打开，可以复制终端中显示的URL（通常是 `http://localhost:8888/lab`）到浏览器中访问。

## 在 VS Code 中使用 Jupyter

如果你想在 VS Code 中使用已经运行的 Jupyter 服务器：

1. 在 VS Code 中打开命令面板（Windows/Linux: `Ctrl+Shift+P`, Mac: `Cmd+Shift+P`）
2. 输入并选择 "Jupyter: Specify local or remote Jupyter server for connections"
3. 选择 "Existing Jupyter Server"
4. 输入正在运行的 Jupyter 服务器的 URL（包含 token）
   - URL 可以从终端输出或浏览器地址栏中复制
   - 格式类似：`http://localhost:8888/?token=<your-token>`

连接成功后，你就可以在 VS Code 中创建和使用 Jupyter notebooks 了。

> 参考文档：https://aka.ms/vscodeJuptyerExtKernelPickerExistingServer

## 注意事项

- 确保在启动 Jupyter Lab 之前已经正确设置环境变量
- Jupyter Lab 默认会在端口 8888 上运行，如果该端口被占用，会自动使用下一个可用端口
- 为了安全起见，建议妥善保管 token 信息
