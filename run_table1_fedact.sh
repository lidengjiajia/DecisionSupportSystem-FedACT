#!/bin/bash
# ================================================================================
# Table 1 实验启动脚本 - FedACT 攻击检测性能
# ================================================================================
#
# 使用方法:
#   chmod +x run_table1_fedact.sh
#   ./run_table1_fedact.sh [选项]
#
# 选项:
#   --gpu <ids>     指定GPU，如 "0,1,2,3" (默认: 自动检测所有GPU)
#   --parallel <n>  每个GPU并发数 (默认: 3)
#   --resume        断点续跑，跳过已完成实验
#   --dry-run       只打印命令，不执行
#   --help          显示帮助
#
# 实验配置:
#   - 数据集: Uci, Xinwang (2种)
#   - 异质性: iid, label_skew, quantity_skew, feature_skew (4种)
#   - 攻击: 12种 (基础3 + 优化5 + 语义4)
#   - 总实验数: 2×4×12 = 96次
#
# 输出文件:
#   - logs/table1_fedact/*.log      实验日志
#   - results/检测统计/*.json       检测统计JSON
#   - results/检测统计/*.xlsx       检测统计Excel
#   - results/汇总/FedACT_Table1_检测统计.xlsx  汇总文件
#
# ================================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认配置
GPU_IDS=""
PARALLEL_PER_GPU=3
RESUME=false
DRY_RUN=false

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/run_table1_fedact.py"
LOG_DIR="${SCRIPT_DIR}/logs/table1_fedact"
SYSTEM_DIR="${SCRIPT_DIR}/system"

# 帮助信息
show_help() {
    echo "Table 1 实验启动脚本 - FedACT 攻击检测性能"
    echo ""
    echo "使用方法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --gpu <ids>     指定GPU，如 '0,1,2,3' (默认: 自动检测)"
    echo "  --parallel <n>  每个GPU并发数 (默认: 3)"
    echo "  --resume        断点续跑，跳过已完成实验"
    echo "  --dry-run       只打印命令，不执行"
    echo "  --help          显示帮助"
    echo ""
    echo "示例:"
    echo "  $0                          # 使用默认配置运行"
    echo "  $0 --gpu 0,1 --parallel 2   # 使用GPU 0,1，每个并发2"
    echo "  $0 --resume                 # 断点续跑"
    exit 0
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU_IDS="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL_PER_GPU="$2"
            shift 2
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            show_help
            ;;
        *)
            echo -e "${RED}未知参数: $1${NC}"
            show_help
            ;;
    esac
done

# 检查Python脚本
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}错误: 找不到Python脚本 $PYTHON_SCRIPT${NC}"
    exit 1
fi

# 检测GPU
detect_gpus() {
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
        echo -e "${GREEN}检测到 $GPU_COUNT 个GPU${NC}"
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    else
        echo -e "${YELLOW}未检测到NVIDIA GPU，将使用CPU运行${NC}"
        GPU_COUNT=0
    fi
}

# 打印配置
print_config() {
    echo ""
    echo -e "${BLUE}=================================================================================${NC}"
    echo -e "${BLUE}Table 1 实验配置${NC}"
    echo -e "${BLUE}=================================================================================${NC}"
    echo -e "GPU: ${GPU_IDS:-自动检测}"
    echo -e "每GPU并发: $PARALLEL_PER_GPU"
    echo -e "断点续跑: $RESUME"
    echo -e "日志目录: $LOG_DIR"
    echo ""
    echo -e "${YELLOW}实验规模:${NC}"
    echo -e "  - 数据集: Uci, Xinwang (2种)"
    echo -e "  - 异质性: iid, label_skew, quantity_skew, feature_skew (4种)"
    echo -e "  - 攻击类型: 12种"
    echo -e "  - 总实验数: 96次"
    echo -e "${BLUE}=================================================================================${NC}"
}

# 创建目录
setup_dirs() {
    mkdir -p "$LOG_DIR"
    mkdir -p "${SYSTEM_DIR}/results/检测统计"
    mkdir -p "${SYSTEM_DIR}/results/汇总"
}

# 主函数
main() {
    echo -e "${GREEN}"
    echo "╔═══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                    FedACT Table 1 实验 - 攻击检测性能                          ║"
    echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    detect_gpus
    print_config
    setup_dirs
    
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[Dry Run] 将执行以下命令:${NC}"
        echo "python $PYTHON_SCRIPT"
        exit 0
    fi
    
    # 确认运行
    read -p "确认开始实验? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}已取消${NC}"
        exit 0
    fi
    
    # 设置GPU环境变量
    if [ -n "$GPU_IDS" ]; then
        export CUDA_VISIBLE_DEVICES="$GPU_IDS"
        echo -e "${GREEN}使用GPU: $GPU_IDS${NC}"
    fi
    
    # 运行Python脚本
    echo -e "${GREEN}开始运行实验...${NC}"
    START_TIME=$(date +%s)
    
    cd "$SCRIPT_DIR"
    python "$PYTHON_SCRIPT"
    
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    HOURS=$((ELAPSED / 3600))
    MINUTES=$(((ELAPSED % 3600) / 60))
    
    echo ""
    echo -e "${GREEN}=================================================================================${NC}"
    echo -e "${GREEN}实验完成!${NC}"
    echo -e "总耗时: ${HOURS}小时 ${MINUTES}分钟"
    echo -e "日志目录: $LOG_DIR"
    echo -e "结果目录: ${SYSTEM_DIR}/results/"
    echo -e "${GREEN}=================================================================================${NC}"
}

# 运行
main
