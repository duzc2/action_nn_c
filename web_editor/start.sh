#!/bin/bash

# Action NN-C Web Editor 快速启动脚本

# 进入脚本所在目录
cd "$(dirname "$0")"

# 检查 Python 是否安装
if ! command -v python3 &> /dev/null; then
    echo "错误：未找到 Python3，请先安装 Python3"
    exit 1
fi

# 端口号
PORT=${1:-8080}

# 启动 HTTP 服务器
echo "=========================================="
echo "  Action NN-C Web Editor"
echo "=========================================="
echo ""
echo "正在启动 Web 服务器..."
echo "访问地址：http://localhost:${PORT}"
echo ""
echo "按 Ctrl+C 停止服务器"
echo "=========================================="
python3 -m http.server ${PORT}
