#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
消融实验脚本 - FedACT 各组件贡献分析 (全攻击覆盖)
================================================================================

FedACT: Federated Autoencoder-Committee TLBO
联邦自编码器-委员会-TLBO框架

================================================================================
实验目的
================================================================================
1. 分析 FedACT 各核心组件对整体性能的贡献，验证每个组件的必要性
2. 覆盖全部12种攻击，全面验证各组件在不同攻击场景下的贡献
3. **聚合方法对比**: 通过 FedACT_Full vs w/o_TLBO 对比TLBO聚合的贡献
   （此对比替代原来单独的 run_comparison 实验）

================================================================================
FedACT 核心组件
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│                              FedACT 框架                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. 自编码器异常检测 (Autoencoder)                                          │
│     • 训练自编码器学习正常梯度的分布特征                                    │
│     • 通过重构误差识别异常梯度                                              │
│                                                                             │
│  2. 委员会投票 (Committee)                                                  │
│     • 选择多样性委员会成员                                                  │
│     • 对可疑梯度进行投票表决                                                │
│                                                                             │
│  3. TLBO优化聚合 (TLBO)                                                     │
│     • Teacher阶段: 向最优梯度学习                                           │
│     • Learner阶段: 梯度间互相学习                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

================================================================================
消融配置 (Part 1: 3种)
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│  配置名称              │ Autoencoder │ Committee │   TLBO   │  说明         │
├─────────────────────────────────────────────────────────────────────────────┤
│  FedACT_Full           │     ✓       │     ✓     │    ✓     │  完整版本     │
│  w/o_Autoencoder       │     ✗       │     ✓     │    ✓     │  无自编码器   │
│  w/o_Committee         │     ✓       │     ✗     │    ✓     │  无委员会     │
└─────────────────────────────────────────────────────────────────────────────┘

关键对比:
  • FedACT_Full vs w/o_Autoencoder → 自编码器的贡献
  • FedACT_Full vs w/o_Committee   → 委员会投票的贡献
  • FedACT_Full vs Only_FedTLBO    → 防御组件(AE+Committee)的整体贡献

================================================================================
Only_Agg聚合算法对比 (Part 2: 7种)
================================================================================

无防御组件(AE=N, Committee=N)，纯粹对比聚合算法的效果

┌─────────────────────────────────────────────────────────────────────────────┐
│  算法名称              │  来源                  │  说明                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  FedAvg               │  AISTATS 2017          │  标准联邦平均             │
│  FedProx              │  MLSys 2020            │  带近端项的联邦学习       │
│  SCAFFOLD             │  ICML 2020             │  方差修正联邦学习         │
│  MOON                 │  CVPR 2021             │  模型对比联邦学习         │
│  FedPSO               │  ICNN 1995             │  粒子群优化聚合           │
│  FedGWO               │  AES 2014              │  灰狼优化聚合             │
│  FedTLBO (本文)       │  本文                  │  教学优化聚合+FedACT防御  │
└─────────────────────────────────────────────────────────────────────────────┘

================================================================================
测试攻击场景
================================================================================
- 消融实验(Part1): 数据集(2) × 异质性(4) × 攻击(12) × 消融配置(3)
- Only_Agg对比(Part2): 数据集(2) × 异质性(4) × 攻击(3) × 聚合算法(7)
- 攻击比例: 0.3 (30%恶意客户端)
- 数据集: Uci, Xinwang (2种)
- 异质性: iid, label_skew, quantity_skew, feature_skew (4种)
- 重复次数: 1次
- 消融实验数: 2×4×12×3×1 = 288次
- Only_Agg实验数: 2×4×3×7×1 = 168次
- 总实验数: 456次

================================================================================
功能特性
================================================================================
- ✓ GPU自动检测与真正并发执行（支持多GPU）
- ✓ 每块GPU 3个实验并发（5×RTX4090 = 最多15个并发）
- ✓ 通过 CUDA_VISIBLE_DEVICES 环境变量分配GPU
- ✓ 断点续跑（检查已完成实验）
- ✓ 进度日志输出到 nohup.log

================================================================================
结果保存
================================================================================
- 汇总结果: system/results/汇总/FedACT_消融实验_汇总.xlsx
- 详细结果: system/results/汇总/FedACT_消融实验_详细.xlsx

作者: FedACT Team
日期: 2026-01-23
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
    ExperimentRunner, run_single_experiment, save_incremental_results,
    generate_detection_summary_excel
)

# ================================================================================
# 路径配置
# ================================================================================
BASE_DIR = Path(__file__).parent.absolute()
SYSTEM_DIR = BASE_DIR / "system"
LOGS_DIR = BASE_DIR / "logs" / "ablation"
RESULTS_DIR = SYSTEM_DIR / "results" / "汇总"
NOHUP_LOG = BASE_DIR / "nohup.log"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ================================================================================
# 实验配置
# ================================================================================

DATASETS = ["Uci", "Xinwang"]  # 两个数据集

# 异质性类型 (4种)
HETEROGENEITY_TYPES = {
    "iid": {"name": "IID分布", "description": "独立同分布"},
    "label_skew": {"name": "标签倾斜", "description": "客户端标签分布不均"},
    "quantity_skew": {"name": "数量倾斜", "description": "客户端数据量不均"},
    "feature_skew": {"name": "特征倾斜", "description": "客户端特征分布不均"},
}

# ================================================================================
# 消融配置 - 分两部分
# ================================================================================

# Part 1: FedACT组件消融 (3种)
# 注: Only_TLBO 和 w/o_TLBO 已合并到 Part 2 (Only_Agg)
#     Only_TLBO = Only_Agg 中的 FedTLBO
#     w/o_TLBO 的效果可通过 FedACT_Full vs Only_FedTLBO + Only_FedAvg 分析得出
ABLATION_CONFIGS = {
    "FedACT_Full": {
        "name": "FedACT完整版",
        "description": "完整版本（所有组件）",
        "algo": "FedTLBO",
        "use_autoencoder": True,
        "use_committee": True,
        "use_tlbo": True,
    },
    "w/o_Autoencoder": {
        "name": "无自编码器",
        "description": "移除自编码器异常检测",
        "algo": "FedTLBO",
        "use_autoencoder": False,
        "use_committee": True,
        "use_tlbo": True,
    },
    "w/o_Committee": {
        "name": "无委员会",
        "description": "移除委员会投票",
        "algo": "FedTLBO",
        "use_autoencoder": True,
        "use_committee": False,
        "use_tlbo": True,
    },
}

# Part 2: Only_Agg聚合算法对比 (7种)
# 无防御组件(AE=N, Committee=N)，纯粹对比聚合算法的效果
# 注: FedTLBO 就是原来的 Only_TLBO
AGGREGATION_COMPARISON = {
    "FedAvg": {
        "name": "FedAvg",
        "description": "标准联邦平均 (McMahan et al., 2017)",
        "algo": "FedAvg",
        "source": "AISTATS 2017",
    },
    "FedProx": {
        "name": "FedProx",
        "description": "带近端项的联邦学习 (Li et al., 2020)",
        "algo": "FedProx",
        "source": "MLSys 2020",
    },
    "SCAFFOLD": {
        "name": "SCAFFOLD",
        "description": "方差修正联邦学习 (Karimireddy et al., 2020)",
        "algo": "SCAFFOLD",
        "source": "ICML 2020",
    },
    "MOON": {
        "name": "MOON",
        "description": "模型对比联邦学习 (Li et al., 2021)",
        "algo": "MOON",
        "source": "CVPR 2021",
    },
    "FedPSO": {
        "name": "FedPSO",
        "description": "粒子群优化聚合 (Kennedy & Eberhart, 1995)",
        "algo": "FedPSO",
        "source": "ICNN 1995",
    },
    "FedGWO": {
        "name": "FedGWO",
        "description": "灰狼优化聚合 (Mirjalili et al., 2014)",
        "algo": "FedGWO",
        "source": "AES 2014",
    },
    "FedTLBO": {
        "name": "FedTLBO (本文)",
        "description": "教学优化聚合 + FedACT防御",
        "algo": "FedTLBO",
        "source": "本文",
    },
}

# ================================================================================
# 全覆盖攻击场景 (12种攻击，全面验证组件贡献)
# ================================================================================
TEST_ATTACKS = {
    # 基础攻击 (3种)
    "sign_flip": {"name": "符号翻转攻击", "ratio": 0.3, "category": "基础攻击"},
    "gaussian": {"name": "高斯噪声攻击", "ratio": 0.3, "category": "基础攻击"},
    "scale": {"name": "缩放攻击", "ratio": 0.3, "category": "基础攻击"},
    
    # 前沿攻击 (5种)
    "little": {"name": "Little攻击", "ratio": 0.3, "category": "前沿攻击"},
    "alie": {"name": "ALIE攻击", "ratio": 0.3, "category": "前沿攻击"},
    "ipm": {"name": "IPM攻击", "ratio": 0.3, "category": "前沿攻击"},
    "minmax": {"name": "MinMax攻击", "ratio": 0.3, "category": "前沿攻击"},
    "trim_attack": {"name": "修剪攻击", "ratio": 0.3, "category": "前沿攻击"},
    
    # 其他攻击 (4种)
    "label_flip": {"name": "标签翻转攻击", "ratio": 0.3, "category": "其他攻击"},
    "backdoor": {"name": "后门攻击", "ratio": 0.3, "category": "其他攻击"},
    "free_rider": {"name": "搭便车攻击", "ratio": 0.3, "category": "其他攻击"},
    "collision": {"name": "共谋攻击", "ratio": 0.3, "category": "其他攻击"},
}

NUM_RUNS = 1
EXPERIMENTS_PER_GPU = 3  # 每块GPU并发3个实验（RTX 4090显存48GB，每实验约2-4GB）

TRAIN_PARAMS = {
    "num_clients": 10,
    "global_rounds": 100,
    "local_epochs": 5,
    "batch_size": 64,
    "learning_rate": 0.01,
    "eval_gap": 10,
}


def build_ablation_command(dataset: str, heterogeneity: str, config_key: str, attack: str, ratio: float, run: int) -> list:
    """构建消融实验命令"""
    config = ABLATION_CONFIGS[config_key]
    
    # 使用配置中的算法
    algo = config.get("algo", "FedTLBO")
    
    cmd = [
        sys.executable,
        str(SYSTEM_DIR / "flcore" / "main.py"),
        "-data", dataset,
        "-algo", algo,
        "-nc", str(TRAIN_PARAMS["num_clients"]),
        "-gr", str(TRAIN_PARAMS["global_rounds"]),
        "-ls", str(TRAIN_PARAMS["local_epochs"]),
        "-lbs", str(TRAIN_PARAMS["batch_size"]),
        "-lr", str(TRAIN_PARAMS["learning_rate"]),
        "-eg", str(TRAIN_PARAMS["eval_gap"]),
        "-t", str(run),
        "--heterogeneity", heterogeneity,
        "--enable_attack", "True",
        "--attack_mode", attack,
        "--malicious_ratio", str(ratio),
        "--use_autoencoder", str(config["use_autoencoder"]),
        "--use_committee", str(config["use_committee"]),
        "--use_tlbo", str(config["use_tlbo"]),
        "--tlbo_iterations", "10",
        "--committee_size", "5",
    ]
    
    return cmd


def build_comparison_command(dataset: str, heterogeneity: str, algo_key: str, attack: str, ratio: float, run: int) -> list:
    """构建聚合算法对比实验命令 (Only_Agg: 只使用聚合算法，无防御组件)"""
    algo_config = AGGREGATION_COMPARISON[algo_key]
    algo = algo_config["algo"]
    
    cmd = [
        sys.executable,
        str(SYSTEM_DIR / "flcore" / "main.py"),
        "-data", dataset,
        "-algo", algo,
        "-nc", str(TRAIN_PARAMS["num_clients"]),
        "-gr", str(TRAIN_PARAMS["global_rounds"]),
        "-ls", str(TRAIN_PARAMS["local_epochs"]),
        "-lbs", str(TRAIN_PARAMS["batch_size"]),
        "-lr", str(TRAIN_PARAMS["learning_rate"]),
        "-eg", str(TRAIN_PARAMS["eval_gap"]),
        "-t", str(run),
        "--heterogeneity", heterogeneity,
        "--enable_attack", "True",
        "--attack_mode", attack,
        "--malicious_ratio", str(ratio),
    ]
    
    # FedTLBO 使用完整FedACT防御
    if algo == "FedTLBO":
        cmd.extend([
            "--use_autoencoder", "True",
            "--use_committee", "True",
            "--use_tlbo", "True",
            "--tlbo_iterations", "10",
            "--committee_size", "5",
        ])
    else:
        # 其他算法不使用FedACT防御组件
        cmd.extend([
            "--use_autoencoder", "False",
            "--use_committee", "False",
            "--use_tlbo", "False",
        ])
    
    return cmd


def generate_all_experiments() -> list:
    """生成所有实验配置（消融实验 + 聚合算法对比）
    
    Part 1 (消融实验): 分析FedACT各组件贡献
        维度: 数据集(2) × 异质性(4) × 攻击(12) × 消融配置(5) × 重复(3) = 1440次
        
    Part 2 (Only_Agg聚合对比): 只使用各类聚合算法（无防御组件）
        维度: 数据集(2) × 异质性(4) × 攻击(3) × 聚合算法(7) × 重复(3) = 504次
    """
    experiments = []
    
    # Part 1: 消融实验 (数据集 × 异质性 × 攻击 × 配置)
    for dataset in DATASETS:
        for het_key, het_info in HETEROGENEITY_TYPES.items():
            for attack_key, attack_info in TEST_ATTACKS.items():
                for config_key, config in ABLATION_CONFIGS.items():
                    for run in range(1, NUM_RUNS + 1):
                        exp_name = f"ablation_{dataset}_{het_key}_{config_key}_{attack_key}_run{run}"
                        cmd = build_ablation_command(dataset, het_key, config_key, attack_key, attack_info["ratio"], run)
                        
                        experiments.append({
                            "name": exp_name,
                            "cmd": cmd,
                            "exp_type": "ablation",
                            "info": {
                                "exp_type": "ablation",
                                "dataset": dataset,
                                "heterogeneity": het_key,
                                "heterogeneity_name": het_info["name"],
                                "config": config_key,
                                "config_name": config["name"],
                                "config_description": config["description"],
                                "algo": config.get("algo", "FedTLBO"),
                                "use_autoencoder": config["use_autoencoder"],
                                "use_committee": config["use_committee"],
                                "use_tlbo": config["use_tlbo"],
                                "attack": attack_key,
                                "attack_name": attack_info["name"],
                                "attack_category": attack_info["category"],
                                "attack_ratio": attack_info["ratio"],
                                "run": run,
                            }
                        })
    
    # Part 2: Only_Agg聚合算法对比实验
    # 只使用各类聚合算法（无防御组件），用于对比不同聚合方式的效果
    COMPARISON_ATTACKS = ["sign_flip", "alie", "minmax"]  # 典型攻击：基础、ALIE、MinMax
    
    for dataset in DATASETS:
        for het_key, het_info in HETEROGENEITY_TYPES.items():
            for attack_key in COMPARISON_ATTACKS:
                attack_info = TEST_ATTACKS[attack_key]
                for algo_key, algo_config in AGGREGATION_COMPARISON.items():
                    for run in range(1, NUM_RUNS + 1):
                        exp_name = f"only_agg_{dataset}_{het_key}_{algo_key}_{attack_key}_run{run}"
                        cmd = build_comparison_command(dataset, het_key, algo_key, attack_key, attack_info["ratio"], run)
                        
                        experiments.append({
                            "name": exp_name,
                            "cmd": cmd,
                            "exp_type": "only_agg",
                            "info": {
                            "exp_type": "only_agg",
                            "dataset": dataset,
                            "heterogeneity": het_key,
                            "heterogeneity_name": het_info["name"],
                            "algo": algo_key,
                            "algo_name": algo_config["name"],
                            "algo_description": algo_config["description"],
                            "source": algo_config["source"],
                            "attack": attack_key,
                            "attack_name": attack_info["name"],
                            "attack_category": attack_info["category"],
                            "attack_ratio": attack_info["ratio"],
                            "run": run,
                        }
                    })
    
    return experiments


def generate_ablation_summary(results: list):
    """生成消融实验和Only_Agg聚合对比汇总"""
    
    print("\n生成汇总报告...")
    
    # 分离两种实验结果
    ablation_results = [r for r in results if r.get("exp_type") == "ablation" or "config" in r]
    only_agg_results = [r for r in results if r.get("exp_type") == "only_agg"]
    
    # ==================== Part 1: 消融实验汇总 (按异质性维度) ====================
    summary_data = defaultdict(lambda: defaultdict(list))
    for r in ablation_results:
        if r.get("success", False):
            # 按异质性×攻击×配置聚合
            # 按数据集×异质性×攻击×配置聚合
            key = (r.get("dataset", "unknown"), r.get("heterogeneity", "iid"), r["attack"], r.get("config", "unknown"))
            summary_data[key]["accuracy"].append(r.get("accuracy", 0))
            summary_data[key]["auc"].append(r.get("auc", 0))
            summary_data[key]["f1"].append(r.get("f1", 0))
    
    # 构建消融汇总表
    rows = []
    for (dataset, heterogeneity, attack, config), metrics in summary_data.items():
        config_info = ABLATION_CONFIGS.get(config, {})
        attack_info = TEST_ATTACKS.get(attack, {})
        het_info = HETEROGENEITY_TYPES.get(heterogeneity, {})
        
        rows.append({
            "数据集": dataset,
            "异质性": heterogeneity,
            "异质性名称": het_info.get("name", heterogeneity),
            "攻击类型": attack,
            "攻击名称": attack_info.get("name", attack),
            "攻击类别": attack_info.get("category", ""),
            "攻击比例": attack_info.get("ratio", 0.3),
            "消融配置": config,
            "配置名称": config_info.get("name", config),
            "配置说明": config_info.get("description", ""),
            "Autoencoder": "✓" if config_info.get("use_autoencoder", False) else "✗",
            "Committee": "✓" if config_info.get("use_committee", False) else "✗",
            "TLBO": "✓" if config_info.get("use_tlbo", False) else "✗",
            "Accuracy均值": round(np.mean(metrics['accuracy']), 3),
            "Accuracy标准差": round(np.std(metrics['accuracy']), 3),
            "Accuracy结果": f"{np.mean(metrics['accuracy']):.3f}±{np.std(metrics['accuracy']):.3f}",
            "AUC均值": round(np.mean(metrics['auc']), 3),
            "AUC标准差": round(np.std(metrics['auc']), 3),
            "AUC结果": f"{np.mean(metrics['auc']):.3f}±{np.std(metrics['auc']):.3f}",
            "F1均值": round(np.mean(metrics['f1']), 3),
            "实验次数": len(metrics['accuracy']),
        })
    
    summary_df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["数据集", "异质性", "攻击类别", "攻击类型", "消融配置"])
    
    # ==================== Part 2: Only_Agg聚合对比汇总 ====================
    only_agg_data = defaultdict(lambda: defaultdict(list))
    for r in only_agg_results:
        if r.get("success", False):
            # 按数据集×异质性×攻击×算法聚合
            key = (r.get("dataset", "unknown"), r.get("heterogeneity", "iid"), r["attack"], r.get("algo", "unknown"))
            only_agg_data[key]["accuracy"].append(r.get("accuracy", 0))
            only_agg_data[key]["auc"].append(r.get("auc", 0))
            only_agg_data[key]["f1"].append(r.get("f1", 0))
    
    only_agg_rows = []
    for (dataset, heterogeneity, attack, algo), metrics in only_agg_data.items():
        algo_info = AGGREGATION_COMPARISON.get(algo, {})
        attack_info = TEST_ATTACKS.get(attack, {})
        het_info = HETEROGENEITY_TYPES.get(heterogeneity, {})
        
        only_agg_rows.append({
            "数据集": dataset,
            "异质性": heterogeneity,
            "异质性名称": het_info.get("name", heterogeneity),
            "攻击类型": attack,
            "攻击名称": attack_info.get("name", attack),
            "聚合算法": algo,
            "算法名称": algo_info.get("name", algo),
            "算法说明": algo_info.get("description", ""),
            "论文来源": algo_info.get("source", ""),
            "Accuracy均值": round(np.mean(metrics['accuracy']), 3),
            "Accuracy标准差": round(np.std(metrics['accuracy']), 3),
            "Accuracy结果": f"{np.mean(metrics['accuracy']):.3f}±{np.std(metrics['accuracy']):.3f}",
            "AUC均值": round(np.mean(metrics['auc']), 3),
            "AUC结果": f"{np.mean(metrics['auc']):.3f}±{np.std(metrics['auc']):.3f}",
            "实验次数": len(metrics['accuracy']),
        })
    
    only_agg_df = pd.DataFrame(only_agg_rows) if only_agg_rows else pd.DataFrame()
    if not only_agg_df.empty:
        only_agg_df = only_agg_df.sort_values(["数据集", "异质性", "攻击类型", "Accuracy均值"], ascending=[True, True, True, False])
    
    # 保存Excel
    excel_path = RESULTS_DIR / "FedACT_消融实验_汇总.xlsx"
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # ============= 消融实验部分 =============
        if not summary_df.empty:
            summary_df.to_excel(writer, index=False, sheet_name="消融汇总结果")
            
            # 按数据集分Sheet
            for dataset in DATASETS:
                df_d = summary_df[summary_df["数据集"] == dataset]
                if not df_d.empty:
                    df_d.to_excel(writer, index=False, sheet_name=f"消融_{dataset}")
            
            # 按异质性分Sheet
            for het_key in HETEROGENEITY_TYPES.keys():
                df_h = summary_df[summary_df["异质性"] == het_key]
                if not df_h.empty:
                    df_h.to_excel(writer, index=False, sheet_name=f"消融_{het_key}")
            
            # 按攻击类别分Sheet
            for category in ["基础攻击", "前沿攻击", "其他攻击"]:
                df_c = summary_df[summary_df["攻击类别"] == category]
                if not df_c.empty:
                    df_c.to_excel(writer, index=False, sheet_name=f"消融_{category}")
        
        # ============= Only_Agg聚合算法对比部分 =============
        if not only_agg_df.empty:
            only_agg_df.to_excel(writer, index=False, sheet_name="Only_Agg聚合对比")
            
            # 按数据集分Sheet
            for dataset in DATASETS:
                df_d = only_agg_df[only_agg_df["数据集"] == dataset]
                if not df_d.empty:
                    df_d.to_excel(writer, index=False, sheet_name=f"Agg_{dataset}")
        
        # ============= 配置说明 =============
        # 消融配置说明
        config_df = pd.DataFrame([
            {"配置": k, "名称": v["name"], "说明": v["description"],
             "算法": v.get("algo", "FedTLBO"),
             "Autoencoder": "✓" if v["use_autoencoder"] else "✗",
             "Committee": "✓" if v["use_committee"] else "✗",
             "TLBO": "✓" if v["use_tlbo"] else "✗"}
            for k, v in ABLATION_CONFIGS.items()
        ])
        config_df.to_excel(writer, index=False, sheet_name="消融配置说明")
        
        # 聚合算法说明
        algo_df = pd.DataFrame([
            {"算法": k, "名称": v["name"], "说明": v["description"], "来源": v["source"]}
            for k, v in AGGREGATION_COMPARISON.items()
        ])
        algo_df.to_excel(writer, index=False, sheet_name="聚合算法说明")
        
        # 异质性类型说明
        het_df = pd.DataFrame([
            {"类型": k, "名称": v["name"], "说明": v["description"]}
            for k, v in HETEROGENEITY_TYPES.items()
        ])
        het_df.to_excel(writer, index=False, sheet_name="异质性类型说明")
        
        # 攻击场景说明
        attack_df = pd.DataFrame([
            {"攻击": k, "名称": v["name"], "比例": v["ratio"], "类别": v["category"]}
            for k, v in TEST_ATTACKS.items()
        ])
        attack_df.to_excel(writer, index=False, sheet_name="攻击场景说明")
        
        # 实验设计总览
        ablation_count = len(DATASETS) * len(HETEROGENEITY_TYPES) * len(TEST_ATTACKS) * len(ABLATION_CONFIGS) * NUM_RUNS
        comparison_attacks = ["sign_flip", "alie", "minmax"]
        only_agg_count = len(DATASETS) * len(HETEROGENEITY_TYPES) * len(comparison_attacks) * len(AGGREGATION_COMPARISON) * NUM_RUNS
        
        design_df = pd.DataFrame({
            "项目": ["算法名称", "数据集", "异质性类型", "消融配置数", "聚合算法数", "攻击场景", 
                    "消融实验数", "Only_Agg实验数", "总实验数"],
            "内容": ["FedACT", f"{len(DATASETS)}种 (Uci, Xinwang)", f"{len(HETEROGENEITY_TYPES)}种", f"{len(ABLATION_CONFIGS)}种", f"{len(AGGREGATION_COMPARISON)}种",
                    f"消融{len(TEST_ATTACKS)}种/对比{len(comparison_attacks)}种", 
                    str(ablation_count), str(only_agg_count), str(ablation_count + only_agg_count)]
        })
        design_df.to_excel(writer, index=False, sheet_name="实验设计")
        
        # ============= 组件贡献分析 (按数据集×异质性×攻击) =============
        contribution_rows = []
        for dataset in DATASETS:
            for attack_key, attack_info in TEST_ATTACKS.items():
                for het_key, het_info in HETEROGENEITY_TYPES.items():
                    # 获取完整版性能
                    full_key = (dataset, het_key, attack_key, "FedACT_Full")
                    if full_key in summary_data:
                        full_acc = np.mean(summary_data[full_key]["accuracy"])
                        full_auc = np.mean(summary_data[full_key]["auc"])
                        
                        for config in ABLATION_CONFIGS.keys():
                            if config != "FedACT_Full":
                                ablated_key = (dataset, het_key, attack_key, config)
                                if ablated_key in summary_data:
                                    ablated_acc = np.mean(summary_data[ablated_key]["accuracy"])
                                    ablated_auc = np.mean(summary_data[ablated_key]["auc"])
                                    acc_drop = full_acc - ablated_acc
                                    auc_drop = full_auc - ablated_auc
                                    
                                    contribution_rows.append({
                                        "数据集": dataset,
                                        "异质性": het_key,
                                        "异质性名称": het_info["name"],
                                        "攻击类型": attack_key,
                                        "攻击名称": attack_info["name"],
                                        "攻击类别": attack_info["category"],
                                        "移除组件": config,
                                        "移除说明": ABLATION_CONFIGS[config]["description"],
                                        "完整版Accuracy": round(full_acc, 3),
                                        "消融后Accuracy": round(ablated_acc, 3),
                                        "Accuracy下降": round(acc_drop, 3),
                                        "Accuracy下降%": f"{100*acc_drop/full_acc:.2f}%" if full_acc > 0 else "N/A",
                                        "完整版AUC": round(full_auc, 3),
                                        "消融后AUC": round(ablated_auc, 3),
                                        "AUC下降": round(auc_drop, 3),
                                    })
        
        if contribution_rows:
            contribution_df = pd.DataFrame(contribution_rows)
            contribution_df = contribution_df.sort_values(["数据集", "异质性", "攻击类别", "攻击类型"])
            contribution_df.to_excel(writer, index=False, sheet_name="组件贡献分析")
            
            # 按攻击类别汇总组件贡献
            category_contribution = []
            for category in ["基础攻击", "前沿攻击", "其他攻击"]:
                df_cat = contribution_df[contribution_df["攻击类别"] == category]
                for config in ["w/o_Autoencoder", "w/o_Committee"]:
                    df_cfg = df_cat[df_cat["移除组件"] == config]
                    if not df_cfg.empty:
                        category_contribution.append({
                            "攻击类别": category,
                            "移除组件": config,
                            "平均Accuracy下降": round(df_cfg["Accuracy下降"].mean(), 3),
                            "最大Accuracy下降": round(df_cfg["Accuracy下降"].max(), 3),
                            "最小Accuracy下降": round(df_cfg["Accuracy下降"].min(), 3),
                            "平均AUC下降": round(df_cfg["AUC下降"].mean(), 3),
                        })
            
            if category_contribution:
                cat_df = pd.DataFrame(category_contribution)
                cat_df.to_excel(writer, index=False, sheet_name="按类别汇总贡献")
    
    print(f"✓ 汇总表已保存: {excel_path}")
    
    # 详细结果
    detail_path = RESULTS_DIR / "FedACT_消融实验_详细.xlsx"
    pd.DataFrame(results).to_excel(detail_path, index=False)
    print(f"✓ 详细结果已保存: {detail_path}")
    
    # 打印组件贡献摘要
    print(f"\n{'='*80}")
    print("组件贡献分析摘要")
    print(f"{'='*80}")
    
    for config_key, config in ABLATION_CONFIGS.items():
        if config_key != "FedACT_Full":
            drops = []
            for (dataset, attack, cfg), metrics in summary_data.items():
                if cfg == config_key:
                    full_key = (dataset, attack, "FedACT_Full")
                    if full_key in summary_data:
                        full_acc = np.mean(summary_data[full_key]["accuracy"])
                        ablated_acc = np.mean(metrics["accuracy"])
                        drops.append(full_acc - ablated_acc)
            
            if drops:
                avg_drop = np.mean(drops)
                max_drop = np.max(drops)
                print(f"  {config_key}: 平均性能下降 {avg_drop:.4f}, 最大下降 {max_drop:.4f}")
    
    # 生成检测统计汇总（TP/FP/TN/FN等）
    detection_stats_dir = RESULTS_DIR.parent / "检测统计"
    detection_summary_path = RESULTS_DIR / "FedACT_消融实验_检测统计_汇总.xlsx"
    detection_excel = generate_detection_summary_excel(
        stats_dir=str(detection_stats_dir),
        output_file=str(detection_summary_path)
    )
    if detection_excel:
        print(f"✓ 检测统计汇总已保存: {detection_excel}")


def run_all_ablation_experiments():
    """运行所有消融实验"""
    
    # 1. 生成所有实验配置
    all_experiments = generate_all_experiments()
    total = len(all_experiments)
    
    # 2. 检测GPU
    gpu_info = get_gpu_info()
    print_gpu_info(gpu_info)
    
    # 3. 检查已完成实验
    completed, remaining = check_completed_experiments(
        RESULTS_DIR, "ablation", all_experiments
    )
    print_progress_info("消融实验", total, len(completed), len(remaining))
    
    # 如果全部完成
    if len(remaining) == 0:
        print("\n✓ 所有实验已完成!")
        return
    
    # 4. 创建实验运行器
    runner = ExperimentRunner(
        experiment_type="ablation",
        results_dir=RESULTS_DIR,
        logs_dir=LOGS_DIR,
        experiments_per_gpu=EXPERIMENTS_PER_GPU,
        nohup_log=NOHUP_LOG,
    )
    
    # 5. 打印启动信息
    runner.print_startup_info(total, len(completed), len(remaining))
    
    # 6. 加载已有结果
    existing_results = []
    detail_file = RESULTS_DIR / "FedACT_消融实验_详细.xlsx"
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
    generate_ablation_summary(all_results)
    
    return all_results


def print_experiment_design():
    """打印实验设计"""
    print("""
================================================================================
                    FedACT 消融实验 (异质性分析 + 聚合算法对比)
              Federated Autoencoder-Committee TLBO
================================================================================
  [实验目的]
    Part1: 分析防御组件(Autoencoder, Committee)在不同场景下的贡献
    Part2: Only_Agg - 对比只使用各类聚合算法（无防御组件）的效果

  [Part1: 消融配置] 3种 (防御组件消融)
    FedACT_Full      : Autoencoder[Y]  Committee[Y]  TLBO[Y]  (完整版)
    w/o_Autoencoder  : Autoencoder[N]  Committee[Y]  TLBO[Y]  (无自编码器)
    w/o_Committee    : Autoencoder[Y]  Committee[N]  TLBO[Y]  (无委员会)

  [Part2: Only_Agg聚合算法] 7种 (无防御组件，纯聚合对比)
    FedAvg, FedProx, SCAFFOLD, MOON, FedPSO, FedGWO, FedTLBO
    
    分析: FedACT_Full vs FedTLBO(Only_Agg) → 防御组件的整体贡献
          FedTLBO vs 其他算法 → TLBO聚合算法的优势

  [数据集] Uci, Xinwang (2种)

  [异质性类型] 4种
    iid, label_skew, quantity_skew, feature_skew

  [测试攻击] 消融12种, Only_Agg 3种典型(sign_flip, alie, minmax)
    攻击比例: 30%

  [实验配置]
    重复: 1次
    消融实验数: 2×4×12×3×1 = 288次
    Only_Agg实验数: 2×4×3×7×1 = 168次
    总实验数: 456次

  [功能特性]
    * GPU自动检测与并发执行
    * 每块GPU 4个实验并发
    * 断点续跑（自动跳过已完成实验）

  [结果保存] system/results/汇总/FedACT_消融实验_汇总.xlsx
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
        RESULTS_DIR, "ablation", all_experiments
    )
    print_progress_info("消融实验", len(all_experiments), len(completed), len(remaining))
    
    if len(remaining) == 0:
        print("\n✓ 所有实验已完成! 是否重新生成汇总报告? (y/n): ", end="")
        if input().strip().lower() == 'y':
            detail_file = RESULTS_DIR / "FedACT_消融实验_详细.xlsx"
            if detail_file.exists():
                df = pd.read_excel(detail_file)
                generate_ablation_summary(df.to_dict('records'))
    else:
        user_input = input("\n是否开始运行实验? (y/n): ").strip().lower()
        if user_input == 'y':
            run_all_ablation_experiments()
        else:
            print("实验已取消")
