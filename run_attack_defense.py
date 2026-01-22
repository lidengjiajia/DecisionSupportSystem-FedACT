#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
攻击防御实验脚本 - FedACT 拜占庭攻击防御能力测试
================================================================================

FedACT: Federated Autoencoder-Committee TLBO
联邦自编码器-委员会-TLBO框架

================================================================================
核心设计：防御检测层与TLBO聚合层解耦
================================================================================

本文提出的 FedACT 框架采用解耦设计:
  - 防御层: 自编码器异常检测 + 委员会投票 (检测并过滤恶意梯度)
  - 聚合层: TLBO优化聚合 (所有方法统一使用)

所有防御方法都使用 TLBO 作为聚合算法，对比的是不同防御检测方法的效果：

┌─────────────────────────────────────────────────────────────────────────────┐
│  防御方法（检测层）                    │ 聚合方法 │  说明                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  None (无防御)                         │   TLBO   │  基线，不过滤任何梯度   │
│  Median (中位数)                       │   TLBO   │  每维取中位数过滤       │
│  TrimmedMean (修剪均值)                │   TLBO   │  去除极值后平均         │
│  Krum (NeurIPS 2017)                   │   TLBO   │  选择最近邻梯度         │
│  MultiKrum (NeurIPS 2017)              │   TLBO   │  Krum多选择版本         │
│  Bulyan (ICML 2018)                    │   TLBO   │  Krum+修剪均值组合      │
│  RFA (几何中位数)                      │   TLBO   │  Weiszfeld算法          │
│  FedACT (自编码器+委员会) ★本文方法    │   TLBO   │  AE检测+委员会投票      │
└─────────────────────────────────────────────────────────────────────────────┘

================================================================================
实验配置
================================================================================
- 数据集: Uci, Xinwang
- 攻击类型: 12种 (基础3种 + 前沿5种 + 其他4种)
- 攻击比例: 0.1, 0.2, 0.3, 0.4
- 防御方法: 8种 (全部使用TLBO聚合)
- 重复次数: 5次
- 总实验数: 2×12×4×8×5 = 3840次

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
- 汇总结果: system/results/汇总/FedACT_攻击防御实验_汇总.xlsx
- 详细结果: system/results/汇总/FedACT_攻击防御实验_详细.xlsx

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
LOGS_DIR = BASE_DIR / "logs" / "attack_defense"
RESULTS_DIR = SYSTEM_DIR / "results" / "汇总"
NOHUP_LOG = BASE_DIR / "nohup.log"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ================================================================================
# 实验配置
# ================================================================================

DATASETS = ["Uci", "Xinwang"]

# 所有攻击类型 (12种)
ATTACK_TYPES = {
    # 基础攻击 (3种)
    "sign_flip": {"name": "符号翻转攻击", "category": "基础攻击", "source": "-"},
    "gaussian": {"name": "高斯噪声攻击", "category": "基础攻击", "source": "-"},
    "scale": {"name": "缩放攻击", "category": "基础攻击", "source": "-"},
    
    # 前沿攻击 (5种，顶会论文)
    "little": {"name": "Little攻击", "category": "前沿攻击", "source": "NeurIPS 2019"},
    "alie": {"name": "ALIE攻击", "category": "前沿攻击", "source": "NeurIPS 2019"},
    "ipm": {"name": "IPM攻击", "category": "前沿攻击", "source": "ICML 2018"},
    "minmax": {"name": "MinMax攻击", "category": "前沿攻击", "source": "IEEE S&P 2020"},
    "trim_attack": {"name": "修剪攻击", "category": "前沿攻击", "source": "-"},
    
    # 其他攻击 (4种)
    "label_flip": {"name": "标签翻转攻击", "category": "其他攻击", "source": "-"},
    "backdoor": {"name": "后门攻击", "category": "其他攻击", "source": "-"},
    "free_rider": {"name": "搭便车攻击", "category": "其他攻击", "source": "-"},
    "collision": {"name": "共谋攻击", "category": "其他攻击", "source": "-"},
}

# 攻击比例
ATTACK_RATIOS = [0.1, 0.2, 0.3, 0.4]

# 防御方法配置 (8种，全部使用TLBO聚合)
DEFENSE_METHODS = {
    "None": {
        "name": "无防御",
        "category": "基线",
        "defense_mode": "none",
        "source": "-",
        "description": "不进行任何恶意梯度检测，直接TLBO聚合",
    },
    "Median": {
        "name": "中位数检测",
        "category": "经典防御",
        "defense_mode": "median",
        "source": "-",
        "description": "对每个维度取中位数，过滤异常值",
    },
    "TrimmedMean": {
        "name": "修剪均值检测",
        "category": "经典防御",
        "defense_mode": "trimmed_mean",
        "source": "ICML 2018",
        "description": "去除最大最小值后平均",
    },
    "Krum": {
        "name": "Krum检测",
        "category": "经典防御",
        "defense_mode": "krum",
        "source": "NeurIPS 2017",
        "description": "选择距离其他梯度最近的梯度",
    },
    "MultiKrum": {
        "name": "Multi-Krum检测",
        "category": "经典防御",
        "defense_mode": "multi_krum",
        "source": "NeurIPS 2017",
        "description": "Krum的多选择版本",
    },
    "Bulyan": {
        "name": "Bulyan检测",
        "category": "经典防御",
        "defense_mode": "bulyan",
        "source": "ICML 2018",
        "description": "Krum选择后再做修剪均值",
    },
    "RFA": {
        "name": "RFA检测",
        "category": "经典防御",
        "defense_mode": "rfa",
        "source": "-",
        "description": "几何中位数（Weiszfeld算法）",
    },
    "FedACT": {
        "name": "FedACT (本文)",
        "category": "本文方法",
        "defense_mode": "fedact",
        "source": "本文",
        "description": "自编码器异常检测 + 委员会投票 + TLBO聚合",
    },
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


def build_command(dataset: str, defense: str, attack: str, ratio: float, run: int) -> list:
    """构建运行命令"""
    defense_info = DEFENSE_METHODS[defense]
    
    cmd = [
        sys.executable,
        str(SYSTEM_DIR / "flcore" / "main.py"),
        "-data", dataset,
        "-algo", "FedTLBO",  # 统一使用TLBO聚合
        "-nc", str(TRAIN_PARAMS["num_clients"]),
        "-gr", str(TRAIN_PARAMS["global_rounds"]),
        "-ls", str(TRAIN_PARAMS["local_epochs"]),
        "-lbs", str(TRAIN_PARAMS["batch_size"]),
        "-lr", str(TRAIN_PARAMS["learning_rate"]),
        "-eg", str(TRAIN_PARAMS["eval_gap"]),
        "-t", str(run),
        # 攻击配置
        "--enable_attack", "True",
        "--attack_mode", attack,
        "--malicious_ratio", str(ratio),
        # 防御配置
        "--defense_mode", defense_info["defense_mode"],
        "--use_tlbo", "True",  # 所有方法都用TLBO聚合
        "--tlbo_iterations", "10",
    ]
    
    # FedACT 特有参数
    if defense == "FedACT":
        cmd.extend([
            "--use_autoencoder", "True",
            "--use_committee", "True",
            "--committee_size", "5",
        ])
    else:
        cmd.extend([
            "--use_autoencoder", "False",
            "--use_committee", "False",
        ])
    
    return cmd


def generate_all_experiments() -> list:
    """生成所有实验配置"""
    experiments = []
    
    for dataset in DATASETS:
        for attack_key, attack_info in ATTACK_TYPES.items():
            for ratio in ATTACK_RATIOS:
                for defense_key, defense_info in DEFENSE_METHODS.items():
                    for run in range(1, NUM_RUNS + 1):
                        exp_name = f"{dataset}_{attack_key}_r{int(ratio*100)}_{defense_key}_run{run}"
                        cmd = build_command(dataset, defense_key, attack_key, ratio, run)
                        
                        experiments.append({
                            "name": exp_name,
                            "cmd": cmd,
                            "info": {
                                "dataset": dataset,
                                "attack": attack_key,
                                "attack_name": attack_info["name"],
                                "attack_category": attack_info["category"],
                                "attack_ratio": ratio,
                                "defense": defense_key,
                                "defense_name": defense_info["name"],
                                "defense_category": defense_info["category"],
                                "run": run,
                            }
                        })
    
    return experiments


def generate_attack_defense_summary(results: list):
    """生成攻击防御实验汇总"""
    
    print("\n生成汇总报告...")
    
    # 分组统计
    summary_data = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r.get("success", False):
            key = (r["dataset"], r["attack"], r["attack_ratio"], r["defense"])
            summary_data[key]["accuracy"].append(r.get("accuracy", 0))
            summary_data[key]["auc"].append(r.get("auc", 0))
            summary_data[key]["f1"].append(r.get("f1", 0))
    
    # 构建汇总表
    rows = []
    for (dataset, attack, ratio, defense), metrics in summary_data.items():
        attack_info = ATTACK_TYPES.get(attack, {})
        defense_info = DEFENSE_METHODS.get(defense, {})
        
        rows.append({
            "数据集": dataset,
            "攻击类型": attack,
            "攻击名称": attack_info.get("name", attack),
            "攻击类别": attack_info.get("category", ""),
            "攻击比例": ratio,
            "防御方法": defense,
            "防御名称": defense_info.get("name", defense),
            "防御类别": defense_info.get("category", ""),
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
    summary_df = summary_df.sort_values(["数据集", "攻击类别", "攻击类型", "攻击比例", "防御类别"])
    
    # 保存Excel
    excel_path = RESULTS_DIR / "FedACT_攻击防御实验_汇总.xlsx"
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        summary_df.to_excel(writer, index=False, sheet_name="汇总结果")
        
        # 按数据集分Sheet
        for dataset in DATASETS:
            df_d = summary_df[summary_df["数据集"] == dataset]
            df_d.to_excel(writer, index=False, sheet_name=f"{dataset}数据集")
        
        # 按攻击类别分Sheet
        for category in ["基础攻击", "前沿攻击", "其他攻击"]:
            df_c = summary_df[summary_df["攻击类别"] == category]
            if not df_c.empty:
                df_c.to_excel(writer, index=False, sheet_name=category)
        
        # 攻击类型说明
        attack_df = pd.DataFrame([
            {"攻击": k, "名称": v["name"], "类别": v["category"], "来源": v["source"]}
            for k, v in ATTACK_TYPES.items()
        ])
        attack_df.to_excel(writer, index=False, sheet_name="攻击类型说明")
        
        # 防御方法说明
        defense_df = pd.DataFrame([
            {"防御": k, "名称": v["name"], "类别": v["category"], 
             "来源": v["source"], "说明": v["description"]}
            for k, v in DEFENSE_METHODS.items()
        ])
        defense_df.to_excel(writer, index=False, sheet_name="防御方法说明")
        
        # 实验设计
        design_df = pd.DataFrame({
            "项目": ["算法名称", "实验类型", "数据集", "攻击类型数", "攻击比例",
                    "防御方法数", "重复次数", "总实验数"],
            "内容": ["FedACT", "攻击防御实验", ", ".join(DATASETS),
                    f"{len(ATTACK_TYPES)}种", ", ".join([str(r) for r in ATTACK_RATIOS]),
                    f"{len(DEFENSE_METHODS)}种", f"{NUM_RUNS}次",
                    f"{len(DATASETS)*len(ATTACK_TYPES)*len(ATTACK_RATIOS)*len(DEFENSE_METHODS)*NUM_RUNS}"]
        })
        design_df.to_excel(writer, index=False, sheet_name="实验设计")
    
    print(f"✓ 汇总表已保存: {excel_path}")
    
    # 详细结果
    detail_path = RESULTS_DIR / "FedACT_攻击防御实验_详细.xlsx"
    pd.DataFrame(results).to_excel(detail_path, index=False)
    print(f"✓ 详细结果已保存: {detail_path}")


def run_all_attack_defense_experiments():
    """运行所有攻击防御实验"""
    
    # 1. 生成所有实验配置
    all_experiments = generate_all_experiments()
    total = len(all_experiments)
    
    # 2. 检测GPU
    gpu_info = get_gpu_info()
    print_gpu_info(gpu_info)
    
    # 3. 检查已完成实验
    completed, remaining = check_completed_experiments(
        RESULTS_DIR, "attack_defense", all_experiments
    )
    print_progress_info("攻击防御实验", total, len(completed), len(remaining))
    
    # 如果全部完成
    if len(remaining) == 0:
        print("\n✓ 所有实验已完成!")
        return
    
    # 4. 创建实验运行器
    runner = ExperimentRunner(
        experiment_type="attack_defense",
        results_dir=RESULTS_DIR,
        logs_dir=LOGS_DIR,
        experiments_per_gpu=EXPERIMENTS_PER_GPU,
        nohup_log=NOHUP_LOG,
    )
    
    # 5. 打印启动信息
    runner.print_startup_info(total, len(completed), len(remaining))
    
    # 6. 加载已有结果
    existing_results = []
    detail_file = RESULTS_DIR / "FedACT_攻击防御实验_详细.xlsx"
    if detail_file.exists():
        try:
            df = pd.read_excel(detail_file)
            existing_results = df.to_dict('records')
        except:
            pass
    
    # 7. 串行运行剩余实验（Windows上更稳定）
    all_results = runner.run_experiments_sequential(
        remaining,
        run_single_experiment,
        existing_results,
    )
    
    # 8. 生成汇总报告
    generate_attack_defense_summary(all_results)
    
    return all_results


def print_experiment_design():
    """打印实验设计"""
    print("""
================================================================================
                       FedACT 攻击防御实验
              Federated Autoencoder-Committee TLBO
================================================================================
  [实验目的]
    测试 FedACT 在各种拜占庭攻击下的防御能力

  [攻击类型] 12种
    基础攻击(3): sign_flip, gaussian, scale
    前沿攻击(5): little, alie, ipm, minmax, trim_attack
    其他攻击(4): label_flip, backdoor, free_rider, collision

  [防御方法] 8种 (全部使用TLBO聚合)
    基线: None
    经典: Median, TrimmedMean, Krum, MultiKrum, Bulyan, RFA
    本文: FedACT (Autoencoder + Committee + TLBO)

  [实验配置]
    数据集: Uci, Xinwang
    攻击比例: 10%, 20%, 30%, 40%
    重复: 5次  |  总实验数: 2x12x4x8x5 = 3840次

  [功能特性]
    * GPU自动检测与并发执行
    * 每块GPU 4个实验并发
    * 断点续跑（自动跳过已完成实验）
    * 进度日志输出到 nohup.log

  [结果保存] system/results/汇总/FedACT_攻击防御实验_汇总.xlsx
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
        RESULTS_DIR, "attack_defense", all_experiments
    )
    print_progress_info("攻击防御实验", len(all_experiments), len(completed), len(remaining))
    
    if len(remaining) == 0:
        print("\n✓ 所有实验已完成! 是否重新生成汇总报告? (y/n): ", end="")
        if input().strip().lower() == 'y':
            detail_file = RESULTS_DIR / "FedACT_攻击防御实验_详细.xlsx"
            if detail_file.exists():
                df = pd.read_excel(detail_file)
                generate_attack_defense_summary(df.to_dict('records'))
    else:
        user_input = input("\n是否开始运行实验? (y/n): ").strip().lower()
        if user_input == 'y':
            run_all_attack_defense_experiments()
        else:
            print("实验已取消")
