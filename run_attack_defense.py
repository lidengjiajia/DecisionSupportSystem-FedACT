#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
攻击防御实验脚本 - FedACT 拜占庭攻击防御能力测试
================================================================================

FedACT: Federated Autoencoder-Committee TLBO
联邦自编码器-委员会-TLBO框架

================================================================================
核心设计：各防御方法使用原本的聚合方式
================================================================================

公平对比设计：每种防御方法使用其原本的聚合算法，本文FedACT使用TLBO聚合

┌─────────────────────────────────────────────────────────────────────────────┐
│  防御方法                              │ 聚合方法  │  说明                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  None (无防御)                         │  FedAvg   │  基线，不防御直接聚合  │
│  Median (中位数)                       │  FedAvg   │  坐标中值聚合          │
│  TrimmedMean (修剪均值)                │  FedAvg   │  去除极值后平均        │
│  Krum (NeurIPS 2017)                   │  FedAvg   │  选择最可信梯度        │
│  MultiKrum (NeurIPS 2017)              │  FedAvg   │  多个可信梯度聚合      │
│  Bulyan (ICML 2018)                    │  FedAvg   │  Krum+修剪均值组合     │
│  RFA (几何中位数)                      │  FedAvg   │  Weiszfeld算法         │
│  FedACT (本文方法) ★                   │  TLBO     │  AE+委员会+TLBO聚合    │
└─────────────────────────────────────────────────────────────────────────────┘

注：TLBO聚合 vs FedAvg聚合的对比实验在 run_ablation.py 消融实验中进行
    通过 "FedACT_Full" vs "w/o_TLBO" 配置对比TLBO聚合的贡献

================================================================================
实验配置
================================================================================
- 数据集: Uci, Xinwang (2种)
- 异质性类型: iid, label_skew, quantity_skew, feature_skew (4种)
- 攻击类型: 12种 (基础3种 + 前沿5种 + 其他4种)
- 恶意客户端比例: 30% (根据审稿人意见调整)
- 防御方法: 8种 (各自使用原本的聚合方式)
  - 经典防御 (7种): 使用FedAvg聚合
  - FedACT (本文): 使用TLBO聚合
- 重复次数: 1次
- 总实验数: 2×4×12×8×1 = 768次

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
    ExperimentRunner, run_single_experiment, save_incremental_results,
    generate_detection_summary_excel
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

# 数据异质性类型 (4种)
HETEROGENEITY_TYPES = {
    "iid": {"name": "IID分布", "description": "独立同分布"},
    "label_skew": {"name": "标签倾斜", "description": "客户端标签分布不均"},
    "quantity_skew": {"name": "数量倾斜", "description": "客户端数据量不均"},
    "feature_skew": {"name": "特征倾斜", "description": "客户端特征分布不均"},
}

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

# 攻击比例 (固定为0.3，即30%恶意客户端，根据审稿人意见调整)
ATTACK_RATIO = 0.3

# 防御方法配置 (8种)
# 注意：只有FedACT使用TLBO聚合，其他方法使用各自原本的聚合方式
DEFENSE_METHODS = {
    "None": {
        "name": "无防御(FedAvg)",
        "category": "基线",
        "defense_mode": "none",
        "algo": "FedAvg",  # 使用FedAvg聚合
        "use_tlbo": False,
        "source": "-",
        "description": "不进行任何恶意梯度检测，直接FedAvg聚合",
    },
    "Median": {
        "name": "Median",
        "category": "经典防御",
        "defense_mode": "median",
        "algo": "FedAvg",  # 经典方法使用自己的聚合
        "use_tlbo": False,
        "source": "Yin et al., ICML 2018",
        "description": "坐标中值聚合",
    },
    "TrimmedMean": {
        "name": "TrimmedMean",
        "category": "经典防御",
        "defense_mode": "trimmed_mean",
        "algo": "FedAvg",
        "use_tlbo": False,
        "source": "Yin et al., ICML 2018",
        "description": "修剪均值聚合",
    },
    "Krum": {
        "name": "Krum",
        "category": "经典防御",
        "defense_mode": "krum",
        "algo": "FedAvg",
        "use_tlbo": False,
        "source": "Blanchard et al., NeurIPS 2017",
        "description": "选择最可信的单个梯度",
    },
    "MultiKrum": {
        "name": "Multi-Krum",
        "category": "经典防御",
        "defense_mode": "multi_krum",
        "algo": "FedAvg",
        "use_tlbo": False,
        "source": "Blanchard et al., NeurIPS 2017",
        "description": "选择多个可信梯度聚合",
    },
    "Bulyan": {
        "name": "Bulyan",
        "category": "经典防御",
        "defense_mode": "bulyan",
        "algo": "FedAvg",
        "use_tlbo": False,
        "source": "Mhamdi et al., ICML 2018",
        "description": "Krum选择后做修剪均值",
    },
    "RFA": {
        "name": "RFA",
        "category": "经典防御",
        "defense_mode": "rfa",
        "algo": "FedAvg",
        "use_tlbo": False,
        "source": "Pillutla et al., ICML 2022",
        "description": "几何中位数（Weiszfeld算法）",
    },
    "FedACT": {
        "name": "FedACT (本文)",
        "category": "本文方法",
        "defense_mode": "fedact",
        "algo": "FedTLBO",  # 本文方法使用TLBO
        "use_tlbo": True,
        "source": "本文",
        "description": "自编码器异常检测 + 委员会投票 + TLBO聚合",
    },
}

NUM_RUNS = 1
EXPERIMENTS_PER_GPU = 4

TRAIN_PARAMS = {
    "num_clients": 10,
    "global_rounds": 100,
    "local_epochs": 5,
    "batch_size": 64,
    "learning_rate": 0.01,
    "eval_gap": 10,
}


def build_command(dataset: str, defense: str, attack: str, heterogeneity: str, run: int) -> list:
    """构建运行命令
    
    Args:
        dataset: 数据集名称 (Uci/Xinwang)
        defense: 防御方法
        attack: 攻击类型
        heterogeneity: 数据异质性类型 (iid/label_skew/quantity_skew/feature_skew)
        run: 实验编号
    """
    defense_info = DEFENSE_METHODS[defense]
    
    # 使用各方法自己的算法和聚合方式
    algo = defense_info["algo"]
    use_tlbo = defense_info["use_tlbo"]
    
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
        # 数据异质性配置
        "--heterogeneity", heterogeneity,
        # 攻击配置
        "--enable_attack", "True",
        "--attack_mode", attack,
        "--malicious_ratio", str(ATTACK_RATIO),  # 30%恶意客户端
        # 防御配置
        "--defense_mode", defense_info["defense_mode"],
        # 聚合方法配置
        "--use_tlbo", str(use_tlbo),
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
    """生成所有实验配置
    
    实验设计: 2数据集 × 4异质性 × 12攻击 × 8防御 × 5次 = 3840次
    - 数据集: Uci, Xinwang (2种)
    - 异质性: iid, label_skew, quantity_skew, feature_skew (4种)
    - 攻击: 12种
    - 防御: 8种 (各自使用原本的聚合方式，FedACT用TLBO)
    - 重复: 5次
    """
    experiments = []
    
    for dataset in DATASETS:
        for het_key, het_info in HETEROGENEITY_TYPES.items():
            for attack_key, attack_info in ATTACK_TYPES.items():
                for defense_key, defense_info in DEFENSE_METHODS.items():
                    for run in range(1, NUM_RUNS + 1):
                        exp_name = f"{dataset}_{het_key}_{attack_key}_{defense_key}_run{run}"
                        cmd = build_command(dataset, defense_key, attack_key, het_key, run)
                        
                        experiments.append({
                            "name": exp_name,
                            "cmd": cmd,
                            "info": {
                                "dataset": dataset,
                                "heterogeneity": het_key,
                                "heterogeneity_name": het_info["name"],
                                "attack": attack_key,
                                "attack_name": attack_info["name"],
                                "attack_category": attack_info["category"],
                                "defense": defense_key,
                                "defense_name": defense_info["name"],
                                "defense_category": defense_info["category"],
                                "algo": defense_info["algo"],
                                "use_tlbo": defense_info["use_tlbo"],
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
            key = (r["dataset"], r["heterogeneity"], r["attack"], r["defense"])
            summary_data[key]["accuracy"].append(r.get("accuracy", 0))
            summary_data[key]["auc"].append(r.get("auc", 0))
            summary_data[key]["f1"].append(r.get("f1", 0))
            summary_data[key]["precision"].append(r.get("precision", 0))
            summary_data[key]["recall"].append(r.get("recall", 0))
    
    # 构建汇总表
    rows = []
    for (dataset, heterogeneity, attack, defense), metrics in summary_data.items():
        het_info = HETEROGENEITY_TYPES.get(heterogeneity, {})
        attack_info = ATTACK_TYPES.get(attack, {})
        defense_info = DEFENSE_METHODS.get(defense, {})
        
        rows.append({
            "数据集": dataset,
            "异质性类型": heterogeneity,
            "异质性名称": het_info.get("name", heterogeneity),
            "攻击类型": attack,
            "攻击名称": attack_info.get("name", attack),
            "攻击类别": attack_info.get("category", ""),
            "防御方法": defense,
            "防御名称": defense_info.get("name", defense),
            "防御类别": defense_info.get("category", ""),
            "Accuracy均值": round(np.mean(metrics['accuracy']), 3),
            "Accuracy标准差": round(np.std(metrics['accuracy']), 3),
            "Accuracy结果": f"{np.mean(metrics['accuracy']):.3f}±{np.std(metrics['accuracy']):.3f}",
            "AUC均值": round(np.mean(metrics['auc']), 3),
            "AUC标准差": round(np.std(metrics['auc']), 3),
            "AUC结果": f"{np.mean(metrics['auc']):.3f}±{np.std(metrics['auc']):.3f}",
            "Precision均值": round(np.mean(metrics['precision']), 3),
            "Recall均值": round(np.mean(metrics['recall']), 3),
            "F1均值": round(np.mean(metrics['f1']), 3),
            "实验次数": len(metrics['accuracy']),
        })
    
    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values(["数据集", "异质性类型", "攻击类别", "攻击类型", "防御类别"])
    
    # 保存Excel
    excel_path = RESULTS_DIR / "FedACT_攻击防御实验_汇总.xlsx"
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        summary_df.to_excel(writer, index=False, sheet_name="汇总结果")
        
        # 按数据集分Sheet
        for dataset in DATASETS:
            df_d = summary_df[summary_df["数据集"] == dataset]
            df_d.to_excel(writer, index=False, sheet_name=f"{dataset}数据集")
        
        # 按异质性类型分Sheet
        for het_key, het_info in HETEROGENEITY_TYPES.items():
            df_h = summary_df[summary_df["异质性类型"] == het_key]
            if not df_h.empty:
                df_h.to_excel(writer, index=False, sheet_name=het_info["name"])
        
        # 按攻击类别分Sheet
        for category in ["基础攻击", "前沿攻击", "其他攻击"]:
            df_c = summary_df[summary_df["攻击类别"] == category]
            if not df_c.empty:
                df_c.to_excel(writer, index=False, sheet_name=category)
        
        # 异质性类型说明
        het_df = pd.DataFrame([
            {"类型": k, "名称": v["name"], "说明": v["description"]}
            for k, v in HETEROGENEITY_TYPES.items()
        ])
        het_df.to_excel(writer, index=False, sheet_name="异质性类型说明")
        
        # 攻击类型说明
        attack_df = pd.DataFrame([
            {"攻击": k, "名称": v["name"], "类别": v["category"], "来源": v["source"]}
            for k, v in ATTACK_TYPES.items()
        ])
        attack_df.to_excel(writer, index=False, sheet_name="攻击类型说明")
        
        # 防御方法说明
        defense_df = pd.DataFrame([
            {"防御": k, "名称": v["name"], "类别": v["category"], 
             "算法": v["algo"], "来源": v["source"], "说明": v["description"]}
            for k, v in DEFENSE_METHODS.items()
        ])
        defense_df.to_excel(writer, index=False, sheet_name="防御方法说明")
        
        # 实验设计
        total_exp = len(DATASETS) * len(HETEROGENEITY_TYPES) * len(ATTACK_TYPES) * len(DEFENSE_METHODS) * NUM_RUNS
        design_df = pd.DataFrame({
            "项目": ["算法名称", "实验类型", "数据集", "异质性类型", "攻击类型数", 
                    "恶意客户端比例", "防御方法数", "重复次数", "总实验数"],
            "内容": ["FedACT", "攻击防御实验", ", ".join(DATASETS),
                    ", ".join(HETEROGENEITY_TYPES.keys()),
                    f"{len(ATTACK_TYPES)}种", f"{ATTACK_RATIO*100:.0f}%",
                    f"{len(DEFENSE_METHODS)}种", f"{NUM_RUNS}次", str(total_exp)]
        })
        design_df.to_excel(writer, index=False, sheet_name="实验设计")
    
    print(f"✓ 汇总表已保存: {excel_path}")
    
    # 详细结果
    detail_path = RESULTS_DIR / "FedACT_攻击防御实验_详细.xlsx"
    pd.DataFrame(results).to_excel(detail_path, index=False)
    print(f"✓ 详细结果已保存: {detail_path}")
    
    # 生成检测统计汇总（TP/FP/TN/FN等）
    detection_stats_dir = RESULTS_DIR.parent / "检测统计"
    detection_summary_path = RESULTS_DIR / "FedACT_检测统计_汇总.xlsx"
    detection_excel = generate_detection_summary_excel(
        stats_dir=str(detection_stats_dir),
        output_file=str(detection_summary_path)
    )
    if detection_excel:
        print(f"✓ 检测统计汇总已保存: {detection_excel}")


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
    
    # 7. 并发运行剩余实验（Linux上使用多GPU并发）
    all_results = runner.run_experiments_concurrent(
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
    1. 测试 FedACT 在各种拜占庭攻击下的防御能力
    2. 与经典防御方法进行对比
    3. 测试不同数据异质性场景下的表现

  [数据异质性] 4种
    iid, label_skew, quantity_skew, feature_skew

  [攻击类型] 12种
    基础攻击(3): sign_flip, gaussian, scale
    前沿攻击(5): little, alie, ipm, minmax, trim_attack
    其他攻击(4): label_flip, backdoor, free_rider, collision

  [防御方法] 8种
    基线: None (FedAvg)
    经典: Median, TrimmedMean, Krum, MultiKrum, Bulyan, RFA (各自聚合)
    本文: FedACT (Autoencoder + Committee + TLBO聚合)

  [实验配置]
    数据集: Uci, Xinwang (2种)
    恶意客户端比例: 30%
    重复: 5次  |  总实验数: 3840次

  [功能特性]
    * GPU自动检测与并发执行
    * 每块GPU 4个实验并发
    * 断点续跑（自动跳过已完成实验）
    * 进度日志输出到 nohup.log

  [结果保存]
    汇总: system/results/汇总/FedACT_攻击防御实验_汇总.xlsx
    详细: system/results/汇总/FedACT_攻击防御实验_详细.xlsx
    日志: logs/attack_defense/
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
