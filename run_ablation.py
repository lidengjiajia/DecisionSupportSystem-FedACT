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
分析 FedACT 各核心组件对整体性能的贡献，验证每个组件的必要性。
【重要】覆盖全部12种攻击，全面验证各组件在不同攻击场景下的贡献。

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
消融配置
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│  配置名称              │ Autoencoder │ Committee │   TLBO   │  说明         │
├─────────────────────────────────────────────────────────────────────────────┤
│  FedACT_Full           │     ✓       │     ✓     │    ✓     │  完整版本     │
│  w/o_Autoencoder       │     ✗       │     ✓     │    ✓     │  无自编码器   │
│  w/o_Committee         │     ✓       │     ✗     │    ✓     │  无委员会     │
│  w/o_TLBO              │     ✓       │     ✓     │    ✗     │  无TLBO聚合   │
│  Only_TLBO             │     ✗       │     ✗     │    ✓     │  仅TLBO聚合   │
└─────────────────────────────────────────────────────────────────────────────┘

================================================================================
测试攻击场景 (全覆盖12种攻击)
================================================================================
- 基础攻击(3): sign_flip, gaussian, scale
- 前沿攻击(5): little, alie, ipm, minmax, trim_attack
- 其他攻击(4): label_flip, backdoor, free_rider, collision
- 攻击比例: 0.3 (中等强度)
- 数据集: Uci, Xinwang
- 重复次数: 5次
- 总实验数: 2×12×5×5 = 600次

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
- 汇总结果: system/results/汇总/FedACT_消融实验_汇总.xlsx
- 详细结果: system/results/汇总/FedACT_消融实验_详细.xlsx

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
LOGS_DIR = BASE_DIR / "logs" / "ablation"
RESULTS_DIR = SYSTEM_DIR / "results" / "汇总"
NOHUP_LOG = BASE_DIR / "nohup.log"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ================================================================================
# 实验配置
# ================================================================================

DATASETS = ["Uci", "Xinwang"]

# 消融配置 (5种)
ABLATION_CONFIGS = {
    "FedACT_Full": {
        "name": "FedACT完整版",
        "description": "完整版本（所有组件）",
        "use_autoencoder": True,
        "use_committee": True,
        "use_tlbo": True,
    },
    "w/o_Autoencoder": {
        "name": "无自编码器",
        "description": "移除自编码器异常检测",
        "use_autoencoder": False,
        "use_committee": True,
        "use_tlbo": True,
    },
    "w/o_Committee": {
        "name": "无委员会",
        "description": "移除委员会投票",
        "use_autoencoder": True,
        "use_committee": False,
        "use_tlbo": True,
    },
    "w/o_TLBO": {
        "name": "无TLBO聚合",
        "description": "移除TLBO聚合（使用FedAvg）",
        "use_autoencoder": True,
        "use_committee": True,
        "use_tlbo": False,
    },
    "Only_TLBO": {
        "name": "仅TLBO聚合",
        "description": "仅使用TLBO聚合（无防御）",
        "use_autoencoder": False,
        "use_committee": False,
        "use_tlbo": True,
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


def build_command(dataset: str, config_key: str, attack: str, ratio: float, run: int) -> list:
    """构建运行命令"""
    config = ABLATION_CONFIGS[config_key]
    
    # 根据是否使用TLBO选择算法
    algo = "FedTLBO" if config["use_tlbo"] else "FedAvg"
    
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


def generate_all_experiments() -> list:
    """生成所有实验配置"""
    experiments = []
    
    for dataset in DATASETS:
        for attack_key, attack_info in TEST_ATTACKS.items():
            for config_key, config in ABLATION_CONFIGS.items():
                for run in range(1, NUM_RUNS + 1):
                    exp_name = f"{dataset}_{config_key}_{attack_key}_run{run}"
                    cmd = build_command(dataset, config_key, attack_key, attack_info["ratio"], run)
                    
                    experiments.append({
                        "name": exp_name,
                        "cmd": cmd,
                        "info": {
                            "dataset": dataset,
                            "config": config_key,
                            "config_name": config["name"],
                            "config_description": config["description"],
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
    
    return experiments


def generate_ablation_summary(results: list):
    """生成消融实验汇总"""
    
    print("\n生成汇总报告...")
    
    # 分组统计
    summary_data = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r.get("success", False):
            key = (r["dataset"], r["attack"], r["config"])
            summary_data[key]["accuracy"].append(r.get("accuracy", 0))
            summary_data[key]["auc"].append(r.get("auc", 0))
            summary_data[key]["f1"].append(r.get("f1", 0))
    
    # 构建汇总表
    rows = []
    for (dataset, attack, config), metrics in summary_data.items():
        config_info = ABLATION_CONFIGS.get(config, {})
        attack_info = TEST_ATTACKS.get(attack, {})
        
        rows.append({
            "数据集": dataset,
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
    summary_df = summary_df.sort_values(["数据集", "攻击类别", "攻击类型", "消融配置"])
    
    # 保存Excel
    excel_path = RESULTS_DIR / "FedACT_消融实验_汇总.xlsx"
    
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
                df_c.to_excel(writer, index=False, sheet_name=f"消融_{category}")
        
        # 消融配置说明
        config_df = pd.DataFrame([
            {"配置": k, "名称": v["name"], "说明": v["description"],
             "Autoencoder": "✓" if v["use_autoencoder"] else "✗",
             "Committee": "✓" if v["use_committee"] else "✗",
             "TLBO": "✓" if v["use_tlbo"] else "✗"}
            for k, v in ABLATION_CONFIGS.items()
        ])
        config_df.to_excel(writer, index=False, sheet_name="消融配置说明")
        
        # 攻击场景说明
        attack_df = pd.DataFrame([
            {"攻击": k, "名称": v["name"], "比例": v["ratio"], "类别": v["category"]}
            for k, v in TEST_ATTACKS.items()
        ])
        attack_df.to_excel(writer, index=False, sheet_name="攻击场景说明")
        
        # 实验设计
        design_df = pd.DataFrame({
            "项目": ["算法名称", "实验类型", "核心组件", "数据集", "攻击场景",
                    "消融配置数", "重复次数", "总实验数"],
            "内容": ["FedACT", "消融实验（全攻击覆盖）", "Autoencoder + Committee + TLBO", 
                    ", ".join(DATASETS),
                    f"{len(TEST_ATTACKS)}种（全覆盖）", f"{len(ABLATION_CONFIGS)}种", 
                    f"{NUM_RUNS}次",
                    f"{len(DATASETS)*len(TEST_ATTACKS)*len(ABLATION_CONFIGS)*NUM_RUNS}"]
        })
        design_df.to_excel(writer, index=False, sheet_name="实验设计")
        
        # ===============================================================
        # 组件贡献分析（核心！按攻击类型和类别分析）
        # ===============================================================
        contribution_rows = []
        for attack_key, attack_info in TEST_ATTACKS.items():
            for dataset in DATASETS:
                # 获取完整版性能
                full_key = (dataset, attack_key, "FedACT_Full")
                if full_key in summary_data:
                    full_acc = np.mean(summary_data[full_key]["accuracy"])
                    full_auc = np.mean(summary_data[full_key]["auc"])
                    
                    for config in ABLATION_CONFIGS.keys():
                        if config != "FedACT_Full":
                            ablated_key = (dataset, attack_key, config)
                            if ablated_key in summary_data:
                                ablated_acc = np.mean(summary_data[ablated_key]["accuracy"])
                                ablated_auc = np.mean(summary_data[ablated_key]["auc"])
                                acc_drop = full_acc - ablated_acc
                                auc_drop = full_auc - ablated_auc
                                
                                contribution_rows.append({
                                    "数据集": dataset,
                                    "攻击类型": attack_key,
                                    "攻击名称": attack_info["name"],
                                    "攻击类别": attack_info["category"],
                                    "移除组件": config,
                                    "移除说明": ABLATION_CONFIGS[config]["description"],
                                    "完整版Accuracy": round(full_acc, 4),
                                    "消融后Accuracy": round(ablated_acc, 4),
                                    "Accuracy下降": round(acc_drop, 4),
                                    "Accuracy下降%": f"{100*acc_drop/full_acc:.2f}%" if full_acc > 0 else "N/A",
                                    "完整版AUC": round(full_auc, 4),
                                    "消融后AUC": round(ablated_auc, 4),
                                    "AUC下降": round(auc_drop, 4),
                                })
        
        if contribution_rows:
            contribution_df = pd.DataFrame(contribution_rows)
            contribution_df = contribution_df.sort_values(["数据集", "攻击类别", "攻击类型"])
            contribution_df.to_excel(writer, index=False, sheet_name="组件贡献分析")
            
            # 按攻击类别汇总组件贡献
            category_contribution = []
            for category in ["基础攻击", "前沿攻击", "其他攻击"]:
                df_cat = contribution_df[contribution_df["攻击类别"] == category]
                for config in ["w/o_Autoencoder", "w/o_Committee", "w/o_TLBO", "Only_TLBO"]:
                    df_cfg = df_cat[df_cat["移除组件"] == config]
                    if not df_cfg.empty:
                        category_contribution.append({
                            "攻击类别": category,
                            "移除组件": config,
                            "平均Accuracy下降": round(df_cfg["Accuracy下降"].mean(), 4),
                            "最大Accuracy下降": round(df_cfg["Accuracy下降"].max(), 4),
                            "最小Accuracy下降": round(df_cfg["Accuracy下降"].min(), 4),
                            "平均AUC下降": round(df_cfg["AUC下降"].mean(), 4),
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
                    FedACT 消融实验 (全攻击覆盖)
              Federated Autoencoder-Committee TLBO
================================================================================
  [实验目的]
    分析各组件在所有攻击场景下的贡献，全面验证组件必要性

  [消融配置] 5种
    FedACT_Full      : Autoencoder[Y]  Committee[Y]  TLBO[Y]  (完整版)
    w/o_Autoencoder  : Autoencoder[N]  Committee[Y]  TLBO[Y]  (无自编码器)
    w/o_Committee    : Autoencoder[Y]  Committee[N]  TLBO[Y]  (无委员会)
    w/o_TLBO         : Autoencoder[Y]  Committee[Y]  TLBO[N]  (无TLBO聚合)
    Only_TLBO        : Autoencoder[N]  Committee[N]  TLBO[Y]  (仅TLBO聚合)

  [测试攻击] 12种 (全覆盖) 攻击比例: 30%
    基础攻击(3): sign_flip, gaussian, scale
    前沿攻击(5): little, alie, ipm, minmax, trim_attack
    其他攻击(4): label_flip, backdoor, free_rider, collision

  [实验配置]
    数据集: Uci, Xinwang  |  重复: 5次
    总实验数: 2x12x5x5 = 600次

  [功能特性]
    * GPU自动检测与并发执行
    * 每块GPU 4个实验并发
    * 断点续跑（自动跳过已完成实验）
    * 进度日志输出到 nohup.log

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
