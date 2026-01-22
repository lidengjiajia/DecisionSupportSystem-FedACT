#!/bin/bash
# ================================================================================
# FedACT 对比实验运行脚本 (无攻击场景)
# ================================================================================
# 
# 功能特性:
#   - GPU自动检测与并发执行
#   - 每块GPU 4个实验并发
#   - 断点续跑
#   - 日志输出到 nohup.log
#
# 实验配置:
#   - 数据集: Uci, Xinwang
#   - 异质性: iid, label, feature, quantity  
#   - 对比算法: Centralized, FedAvg, FedProx, Scaffold, Moon, FedACT
#   - 攻击: 无 (纯异质性验证)
#   - 总实验数: 2×4×6×5 = 240次
#
# 使用方法:
#   chmod +x run_comparison.sh
#   nohup ./run_comparison.sh > nohup.log 2>&1 &
#
# ================================================================================

set -e

echo "========================================"
echo "  FedACT 对比实验 (无攻击场景)"
echo "========================================"
echo "开始时间: $(date)"
echo ""

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python"
    exit 1
fi

# 检测GPU
echo "========================================" 
echo "  GPU 检测"
echo "========================================"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "检测到 $GPU_COUNT 块GPU"
else
    echo "未检测到GPU，将使用CPU运行"
    GPU_COUNT=0
fi
echo ""

# 切换到脚本目录
cd "$(dirname "$0")"

# 运行实验
echo "========================================" 
echo "  开始运行对比实验"
echo "========================================"
python run_comparison.py <<< "y"

echo ""
echo "========================================"
echo "  对比实验完成!"
echo "========================================"
echo "结束时间: $(date)"
echo "结果保存: system/results/汇总/FedACT_对比实验_汇总.xlsx"
