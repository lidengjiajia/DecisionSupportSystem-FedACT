#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
对比实验脚本 - FedACT vs 基线联邦学习算法 (无攻击场景)
================================================================================

FedACT: Federated Autoencoder-Committee TLBO
联邦自编码器-委员会-TLBO框架

================================================================================
实验目的
================================================================================
在【无攻击场景】下，对比 FedACT 与其他联邦学习算法在处理异质性数据上的性能。
本实验单纯验证异质性处理能力，不涉及攻击防御。

================================================================================
对比算法 (6种)
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│  算法           │  来源             │  说明                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  Centralized    │  -                │  中心化训练（性能上界）               │
│  FedAvg         │  AISTATS 2017     │  联邦平均                             │
│  FedProx        │  MLSys 2020       │  近端正则化联邦学习                   │
│  Scaffold       │  ICML 2020        │  控制变量联邦学习                     │
│  Moon           │  CVPR 2021        │  对比学习联邦学习                     │
│  FedACT ★       │  本文             │  自编码器+委员会+TLBO                 │
└─────────────────────────────────────────────────────────────────────────────┘

================================================================================
实验配置
================================================================================
- 数据集: Uci, Xinwang
- 异质性类型: iid, label, feature, quantity
- 攻击: 无 (纯异质性验证)
- 客户端数: 10
- 全局轮数: 100
- 重复次数: 5次
- 总实验数: 2×4×6×5 = 240次

================================================================================
功能特性
================================================================================
- ✓ GPU自动检测与并发执行
- ✓ 每块GPU 4个实验并发
- ✓ 断点续跑（检查已完成实验）
- ✓ 进度日志输出到 nohup.log

================================================================================
结果保存
================================================================================
- 汇总结果: system/results/汇总/FedACT_对比实验_汇总.xlsx
- 详细结果: system/results/汇总/FedACT_对比实验_详细.xlsx

作者: FedACT Team
日期: 2026-01-22
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent.absolute()))
sys.path.insert(0, str(Path(__file__).parent.absolute() / "system"))

from utils.experiment_utils import (
    get_gpu_info, print_gpu_info,
    check_completed_experiments, print_progress_info,
    ExperimentRunner, run_single_experiment, save_incremental_results
)

# ================================================================================
# 路径配置
# ================================================================================
BASE_DIR = Path(__file__).parent.absolute()
SYSTEM_DIR = BASE_DIR / "system"
LOGS_DIR = BASE_DIR / "logs" / "comparison"
RESULTS_DIR = SYSTEM_DIR / "results" / "汇总"
NOHUP_LOG = BASE_DIR / "nohup.log"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ================================================================================
# 实验配置
# ================================================================================

DATASETS = ["Uci", "Xinwang"]
HETEROGENEITY_TYPES = ["iid", "label", "feature", "quantity"]

# 对比算法 (6种)
ALGORITHMS = {
    "Centralized": {"name": "中心化训练", "source": "-", "category": "基线"},
    "FedAvg": {"name": "联邦平均", "source": "AISTATS 2017", "category": "经典FL"},
    "FedProx": {"name": "近端优化", "source": "MLSys 2020", "category": "经典FL"},
    "Scaffold": {"name": "控制变量", "source": "ICML 2020", "category": "经典FL"},
    "Moon": {"name": "对比学习", "source": "CVPR 2021", "category": "经典FL"},
    "FedACT": {"name": "FedACT (本文)", "source": "本文", "category": "本文方法"},
}

NUM_RUNS = 5
EXPERIMENTS_PER_GPU = 4

TRAIN_PARAMS = {
    "num_clients": 10,
    "global_rounds": 100,
    "local_epochs": 5,
    "batch_size": 64,
    "learning_rate": 0.01,
    "eval_gap": 10,
}

ALGO_PARAMS = {
    "FedProx": {"mu": 0.01},
    "Moon": {"mu": 0.1, "tau": 0.5},
    "FedACT": {"tlbo_iterations": 10, "committee_size": 5, "use_tlbo": True},
}


def build_command(dataset: str, algo: str, heterogeneity: str, run: int) -> list:
    """构建运行命令 - 无攻击场景"""
    
    # FedACT 使用 FedTLBO 算法
    algo_name = "FedTLBO" if algo == "FedACT" else algo
    
    cmd = [
        sys.executable,
        str(SYSTEM_DIR / "main.py"),
        "-data", dataset,
        "-algo", algo_name,
        "-nc", str(TRAIN_PARAMS["num_clients"]),
        "-gr", str(TRAIN_PARAMS["global_rounds"]),
        "-ls", str(TRAIN_PARAMS["local_epochs"]),
        "-lbs", str(TRAIN_PARAMS["batch_size"]),
        "-lr", str(TRAIN_PARAMS["learning_rate"]),
        "-eg", str(TRAIN_PARAMS["eval_gap"]),
        "-t", str(run),
        "--heterogeneity", heterogeneity,
        # 无攻击
        "--enable_attack", "False",
    ]
    
    # 算法特定参数
    if algo in ALGO_PARAMS:
        for key, value in ALGO_PARAMS[algo].items():
            if key in ["mu", "tau"]:
                cmd.extend([f"-{key}", str(value)])
            else:
                cmd.extend([f"--{key}", str(value)])
    
    return cmd


def generate_all_experiments() -> list:
    """生成所有实验配置"""
    experiments = []
    
    for dataset in DATASETS:
        for heterogeneity in HETEROGENEITY_TYPES:
            for algo in ALGORITHMS.keys():
                for run in range(1, NUM_RUNS + 1):
                    exp_name = f"{dataset}_{heterogeneity}_{algo}_run{run}"
                    cmd = build_command(dataset, algo, heterogeneity, run)
                    
                    experiments.append({
                        "name": exp_name,
                        "cmd": cmd,
                        "info": {
                            "dataset": dataset,
                            "heterogeneity": heterogeneity,
                            "algorithm": algo,
                            "algorithm_name": ALGORITHMS[algo]["name"],
                            "algorithm_category": ALGORITHMS[algo]["category"],
                            "run": run,
                        }
                    })
    
    return experiments


def generate_comparison_summary(results: list):
    """生成对比实验汇总"""
    
    print("\n生成汇总报告...")
    
    # 分组统计
    summary_data = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r.get("success", False):
            key = (r["dataset"], r["heterogeneity"], r["algorithm"])
            summary_data[key]["accuracy"].append(r.get("accuracy", 0))
            summary_data[key]["auc"].append(r.get("auc", 0))
            summary_data[key]["f1"].append(r.get("f1", 0))
    
    # 构建汇总表
    rows = []
    for (dataset, heterogeneity, algo), metrics in summary_data.items():
        rows.append({
            "数据集": dataset,
            "异质性类型": heterogeneity,
            "算法": algo,
            "算法名称": ALGORITHMS.get(algo, {}).get("name", algo),
            "算法类别": ALGORITHMS.get(algo, {}).get("category", ""),
            "来源": ALGORITHMS.get(algo, {}).get("source", "-"),
            "Accuracy均值": round(np.mean(metrics['accuracy']), 4),
            "Accuracy标准差": round(np.std(metrics['accuracy']), 4),
            "Accuracy结果": f"{np.mean(metrics['accuracy']):.4f}±{np.std(metrics['accuracy']):.4f}",
            "AUC均值": round(np.mean(metrics['auc']), 4),
            "AUC标准差": round(np.std(metrics['auc']), 4),
            "AUC结果": f"{np.mean(metrics['auc']):.4f}±{np.std(metrics['auc']):.4f}",
            "F1均值": round(np.mean(metrics['f1']), 4),
            "实验次数": len(metrics['accuracy']),
        })
    
    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values(["数据集", "异质性类型", "算法类别"])
    
    # 保存Excel
    excel_path = RESULTS_DIR / "FedACT_对比实验_汇总.xlsx"
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        summary_df.to_excel(writer, index=False, sheet_name="汇总结果")
        
        for dataset in DATASETS:
            df_d = summary_df[summary_df["数据集"] == dataset]
            df_d.to_excel(writer, index=False, sheet_name=f"{dataset}数据集")
        
        # 算法说明
        algo_df = pd.DataFrame([
            {"算法": k, "名称": v["name"], "类别": v["category"], "来源": v["source"]}
            for k, v in ALGORITHMS.items()
        ])
        algo_df.to_excel(writer, index=False, sheet_name="算法说明")
        
        # 实验设计
        design_df = pd.DataFrame({
            "项目": ["算法名称", "实验类型", "数据集", "异质性类型", "对比算法数", 
                    "重复次数", "总实验数", "攻击设置"],
            "内容": ["FedACT", "对比实验（无攻击）", ", ".join(DATASETS), 
                    ", ".join(HETEROGENEITY_TYPES),
                    f"{len(ALGORITHMS)}种", f"{NUM_RUNS}次",
                    f"{len(DATASETS)*len(HETEROGENEITY_TYPES)*len(ALGORITHMS)*NUM_RUNS}",
                    "无攻击（纯异质性验证）"]
        })
        design_df.to_excel(writer, index=False, sheet_name="实验设计")
    
    print(f"✓ 汇总表已保存: {excel_path}")
    
    # 详细结果
    detail_path = RESULTS_DIR / "FedACT_对比实验_详细.xlsx"
    pd.DataFrame(results).to_excel(detail_path, index=False)
    print(f"✓ 详细结果已保存: {detail_path}")


def run_all_comparison_experiments():
    """运行所有对比实验"""
    
    # 1. 生成所有实验配置
    all_experiments = generate_all_experiments()
    total = len(all_experiments)
    
    # 2. 检测GPU
    gpu_info = get_gpu_info()
    print_gpu_info(gpu_info)
    
    # 3. 检查已完成实验
    completed, remaining = check_completed_experiments(
        RESULTS_DIR, "comparison", all_experiments
    )
    print_progress_info("对比实验", total, len(completed), len(remaining))
    
    # 如果全部完成
    if len(remaining) == 0:
        print("\n✓ 所有实验已完成!")
        return
    
    # 4. 创建实验运行器
    runner = ExperimentRunner(
        experiment_type="comparison",
        results_dir=RESULTS_DIR,
        logs_dir=LOGS_DIR,
        experiments_per_gpu=EXPERIMENTS_PER_GPU,
        nohup_log=NOHUP_LOG,
    )
    
    # 5. 打印启动信息
    runner.print_startup_info(total, len(completed), len(remaining))
    
    # 6. 加载已有结果
    existing_results = []
    detail_file = RESULTS_DIR / "FedACT_对比实验_详细.xlsx"
    if detail_file.exists():
        try:
            df = pd.read_excel(detail_file)
            existing_results = df.to_dict('records')
        except:
            pass
    
    # 7. 并发运行剩余实验
    all_results = runner.run_experiments_concurrent(
        remaining,
        run_single_experiment,
        existing_results,
    )
    
    # 8. 生成汇总报告
    generate_comparison_summary(all_results)
    
    return all_results


def print_experiment_design():
    """打印实验设计"""
    print("""
================================================================================
                       FedACT 对比实验 (无攻击场景)
              Federated Autoencoder-Committee TLBO
================================================================================
  [实验目的]
    验证 FedACT 在异质性场景下的性能，不涉及攻击防御

  [对比算法] 6种
    基线: Centralized
    经典FL: FedAvg, FedProx, Scaffold, Moon
    本文: FedACT (Autoencoder + Committee + TLBO)

  [实验配置]
    数据集: Uci, Xinwang
    异质性: iid, label, feature, quantity
    攻击: 无
    重复: 5次  |  总实验数: 2x4x6x5 = 240次

  [功能特性]
    * GPU自动检测与并发执行
    * 每块GPU 4个实验并发
    * 断点续跑（自动跳过已完成实验）
    * 进度日志输出到 nohup.log

  [结果保存] system/results/汇总/FedACT_对比实验_汇总.xlsx
================================================================================
    """)


if __name__ == "__main__":
    print_experiment_design()
    
    # 检测GPU
    gpu_info = get_gpu_info()
    print_gpu_info(gpu_info)
    
    # 生成所有实验
    all_experiments = generate_all_experiments()
    
    # 检查进度
    completed, remaining = check_completed_experiments(
        RESULTS_DIR, "comparison", all_experiments
    )
    print_progress_info("对比实验", len(all_experiments), len(completed), len(remaining))
    
    if len(remaining) == 0:
        print("\n✓ 所有实验已完成! 是否重新生成汇总报告? (y/n): ", end="")
        if input().strip().lower() == 'y':
            detail_file = RESULTS_DIR / "FedACT_对比实验_详细.xlsx"
            if detail_file.exists():
                df = pd.read_excel(detail_file)
                generate_comparison_summary(df.to_dict('records'))
    else:
        user_input = input("\n是否开始运行实验? (y/n): ").strip().lower()
        if user_input == 'y':
            run_all_comparison_experiments()
        else:
            print("实验已取消")
