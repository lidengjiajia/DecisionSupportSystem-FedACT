#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
Table 1 实验脚本 - FedACT 攻击检测性能
================================================================================

专门运行 FedACT 完整版本的实验，生成论文 Table 1 的数据

目的: 验证 FedACT 在各类攻击下的检测性能 (Precision, Recall, F1)

================================================================================
实验配置
================================================================================
- 数据集: Uci, Xinwang (2种)
- 异质性类型: iid, label_skew, quantity_skew, feature_skew (4种)
- 攻击类型: 12种 (基础3种 + 优化5种 + 语义4种)
- 恶意客户端比例: 30%
- 防御方法: FedACT (完整版)
- 重复次数: 1次
- 总实验数: 2×4×12×1 = 96次

================================================================================
输出文件
================================================================================
每个实验保存:
1. 检测统计: results/检测统计/{dataset}_{het}_{attack}_FedACT_stats.json
2. 检测统计: results/检测统计/{dataset}_{het}_{attack}_FedACT_stats.xlsx
3. 训练曲线: results/FedACT_Table1/{dataset}_{het}_{attack}/*.h5
4. 训练过程: results/FedACT_Table1/{dataset}_{het}_{attack}/*_training_process.csv
5. 模型文件: models/{dataset}/FedTLBO_server.pt

汇总文件:
- results/汇总/FedACT_Table1_检测统计.xlsx

作者: FedACT Team
日期: 2026-01-27
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import subprocess
import threading
import queue

# 路径配置
BASE_DIR = Path(__file__).parent.absolute()
SYSTEM_DIR = BASE_DIR / "system"
LOGS_DIR = BASE_DIR / "logs" / "table1_fedact"
RESULTS_DIR = SYSTEM_DIR / "results" / "汇总"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ================================================================================
# 实验配置
# ================================================================================

DATASETS = ["Uci", "Xinwang"]

HETEROGENEITY_TYPES = {
    "iid": {"name": "IID分布"},
    "label_skew": {"name": "标签倾斜"},
    "quantity_skew": {"name": "数量倾斜"},
    "feature_skew": {"name": "特征倾斜"},
}

# 12种攻击，按论文分类
TEST_ATTACKS = {
    # 基础攻击 (3种)
    "sign_flip": {"name": "Sign-flip", "ratio": 0.3, "category": "Basic"},
    "gaussian": {"name": "Gaussian", "ratio": 0.3, "category": "Basic"},
    "scale": {"name": "Scaling", "ratio": 0.3, "category": "Basic"},
    
    # 优化攻击 (5种)
    "little": {"name": "Little", "ratio": 0.3, "category": "Optimization"},
    "alie": {"name": "ALIE", "ratio": 0.3, "category": "Optimization"},
    "ipm": {"name": "IPM", "ratio": 0.3, "category": "Optimization"},
    "minmax": {"name": "MinMax", "ratio": 0.3, "category": "Optimization"},
    "trim_attack": {"name": "Trim", "ratio": 0.3, "category": "Optimization"},
    
    # 语义攻击 (4种)
    "label_flip": {"name": "Label-flip", "ratio": 0.3, "category": "Semantic"},
    "backdoor": {"name": "Backdoor", "ratio": 0.3, "category": "Semantic"},
    "free_rider": {"name": "Free-rider", "ratio": 0.3, "category": "Semantic"},
    "collision": {"name": "Collision", "ratio": 0.3, "category": "Semantic"},
}

# 训练参数
TRAIN_PARAMS = {
    "num_clients": 10,
    "global_rounds": 100,
    "local_epochs": 5,
    "batch_size": 64,
    "learning_rate": 0.01,
    "eval_gap": 10,
}

# GPU并发配置
EXPERIMENTS_PER_GPU = 3


def get_gpu_info():
    """获取可用GPU信息"""
    try:
        import torch
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
            return list(range(num_gpus)), gpu_names
    except:
        pass
    return [], []


def build_experiment_command(dataset: str, heterogeneity: str, attack: str, ratio: float, run: int) -> list:
    """构建FedACT实验命令"""
    
    # 创建专用结果目录
    result_folder = f"FedACT_Table1/{dataset}_{heterogeneity}_{attack}"
    
    cmd = [
        sys.executable,
        str(SYSTEM_DIR / "flcore" / "main.py"),
        "-data", dataset,
        "-algo", "FedTLBO",  # FedACT使用FedTLBO服务器
        "-nc", str(TRAIN_PARAMS["num_clients"]),
        "-gr", str(TRAIN_PARAMS["global_rounds"]),
        "-ls", str(TRAIN_PARAMS["local_epochs"]),
        "-lbs", str(TRAIN_PARAMS["batch_size"]),
        "-lr", str(TRAIN_PARAMS["learning_rate"]),  # 注意：main.py实际接收的是local_learning_rate
        "-eg", str(TRAIN_PARAMS["eval_gap"]),
        "-t", str(run),
        "-go", f"Table1_{heterogeneity}_{attack}",  # goal参数用于区分不同实验
        "--heterogeneity", heterogeneity,
        # 攻击配置
        "--enable_attack", "True",
        "--attack_mode", attack,
        "--malicious_ratio", str(ratio),
        # FedACT完整版配置
        "--defense_mode", "fedact",
        "--use_autoencoder", "True",
        "--use_committee", "True",
        "--use_tlbo", "True",
        "--tlbo_iterations", "10",
        "--committee_size", "5",
        # 保存结果目录
        "-sfn", result_folder,
    ]
    
    return cmd


def generate_all_experiments() -> list:
    """生成所有Table 1实验"""
    experiments = []
    
    for dataset in DATASETS:
        for het_key, het_info in HETEROGENEITY_TYPES.items():
            for attack_key, attack_info in TEST_ATTACKS.items():
                exp_name = f"{dataset}_{het_key}_{attack_key}_FedACT"
                cmd = build_experiment_command(
                    dataset, het_key, attack_key, 
                    attack_info["ratio"], 1
                )
                
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
                        "attack_ratio": attack_info["ratio"],
                    }
                })
    
    return experiments


def check_completed_experiments(experiments: list, logs_dir: Path) -> tuple:
    """检查已完成的实验"""
    completed = []
    pending = []
    
    for exp in experiments:
        log_file = logs_dir / f"{exp['name']}.log"
        if log_file.exists():
            # 检查日志是否包含完成标记
            try:
                content = log_file.read_text(encoding='utf-8', errors='ignore')
                if "训练完成" in content or "Training completed" in content:
                    completed.append(exp)
                    continue
            except:
                pass
        pending.append(exp)
    
    return completed, pending


def run_single_experiment(exp: dict, gpu_id: int, logs_dir: Path) -> dict:
    """运行单个实验"""
    log_file = logs_dir / f"{exp['name']}.log"
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    start_time = time.time()
    
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"实验: {exp['name']}\n")
            f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"GPU: {gpu_id}\n")
            f.write(f"命令: {' '.join(exp['cmd'])}\n")
            f.write("="*60 + "\n")
            f.flush()
            
            process = subprocess.Popen(
                exp['cmd'],
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(SYSTEM_DIR)
            )
            process.wait()
            
            elapsed = time.time() - start_time
            f.write(f"\n{'='*60}\n")
            f.write(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"耗时: {elapsed/60:.1f}分钟\n")
            f.write(f"返回码: {process.returncode}\n")
        
        return {
            **exp['info'],
            'success': process.returncode == 0,
            'elapsed': elapsed,
            'log_file': str(log_file)
        }
        
    except Exception as e:
        return {
            **exp['info'],
            'success': False,
            'error': str(e),
            'log_file': str(log_file)
        }


def worker(gpu_id: int, task_queue: queue.Queue, results: list, logs_dir: Path):
    """GPU工作线程"""
    while True:
        try:
            exp = task_queue.get_nowait()
        except queue.Empty:
            break
        
        print(f"[GPU {gpu_id}] 开始: {exp['name']}")
        result = run_single_experiment(exp, gpu_id, logs_dir)
        results.append(result)
        
        status = "✓" if result.get('success') else "✗"
        print(f"[GPU {gpu_id}] {status} 完成: {exp['name']}")
        
        task_queue.task_done()


def extract_detection_stats_from_log(log_file: Path) -> dict:
    """从日志文件提取检测统计"""
    try:
        content = log_file.read_text(encoding='utf-8', errors='ignore')
        
        # 查找最终统计
        stats = {}
        
        # 匹配: TP=285, FP=666, TN=34, FN=15
        import re
        match = re.search(r'TP=(\d+),\s*FP=(\d+),\s*TN=(\d+),\s*FN=(\d+)', content)
        if match:
            stats['TP'] = int(match.group(1))
            stats['FP'] = int(match.group(2))
            stats['TN'] = int(match.group(3))
            stats['FN'] = int(match.group(4))
            
            # 计算指标
            tp, fp, tn, fn = stats['TP'], stats['FP'], stats['TN'], stats['FN']
            stats['Precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
            stats['Recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            stats['F1'] = 2 * stats['Precision'] * stats['Recall'] / (stats['Precision'] + stats['Recall']) if (stats['Precision'] + stats['Recall']) > 0 else 0
            stats['Accuracy'] = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
        
        # 查找最终模型准确率
        acc_matches = re.findall(r'Test Accuracy:\s*([\d.]+)', content)
        if acc_matches:
            stats['Model_Accuracy'] = float(acc_matches[-1])
        
        return stats
    except Exception as e:
        print(f"提取日志失败 {log_file}: {e}")
        return {}


def generate_table1_summary(results: list, logs_dir: Path):
    """生成Table 1汇总Excel"""
    
    print("\n" + "="*60)
    print("生成 Table 1 汇总报告")
    print("="*60)
    
    # 收集所有数据
    all_data = []
    
    for r in results:
        if not r.get('success', False):
            continue
        
        # 从日志提取检测统计
        log_file = Path(r.get('log_file', ''))
        if log_file.exists():
            stats = extract_detection_stats_from_log(log_file)
        else:
            stats = {}
        
        row = {
            'Dataset': r['dataset'],
            'Heterogeneity': r['heterogeneity'],
            'Attack': r['attack'],
            'Attack_Name': r['attack_name'],
            'Category': r['attack_category'],
            **stats
        }
        all_data.append(row)
    
    if not all_data:
        print("⚠️ 没有成功的实验结果")
        return
    
    df = pd.DataFrame(all_data)
    
    # 保存详细数据
    detail_file = RESULTS_DIR / "FedACT_Table1_检测统计.xlsx"
    
    with pd.ExcelWriter(detail_file, engine='openpyxl') as writer:
        # Sheet 1: 完整数据
        df.to_excel(writer, sheet_name='详细数据', index=False)
        
        # Sheet 2: 按攻击类型汇总 (论文Table 1格式)
        table1 = df.groupby(['Category', 'Attack_Name']).agg({
            'Precision': 'mean',
            'Recall': 'mean',
            'F1': 'mean',
            'TP': 'sum',
            'FP': 'sum',
            'TN': 'sum',
            'FN': 'sum',
        }).round(3)
        table1.to_excel(writer, sheet_name='Table1_按攻击', index=True)
        
        # Sheet 3: 按异质性汇总
        by_het = df.groupby('Heterogeneity').agg({
            'Precision': 'mean',
            'Recall': 'mean',
            'F1': 'mean',
        }).round(3)
        by_het.to_excel(writer, sheet_name='按异质性', index=True)
        
        # Sheet 4: 总体指标
        overall = pd.DataFrame([{
            'Total_Experiments': len(df),
            'Avg_Precision': df['Precision'].mean(),
            'Avg_Recall': df['Recall'].mean(),
            'Avg_F1': df['F1'].mean(),
            'Total_TP': df['TP'].sum(),
            'Total_FP': df['FP'].sum(),
            'Total_TN': df['TN'].sum(),
            'Total_FN': df['FN'].sum(),
        }])
        overall.to_excel(writer, sheet_name='总体指标', index=False)
    
    print(f"✅ Table 1 汇总已保存: {detail_file}")
    
    # 打印论文Table 1格式
    print("\n" + "="*60)
    print("Table 1: FedACT Detection Performance")
    print("="*60)
    print(f"{'Category':<12} {'Attack':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-"*60)
    
    for category in ['Basic', 'Optimization', 'Semantic']:
        cat_df = df[df['Category'] == category]
        for _, row in cat_df.groupby('Attack_Name').first().iterrows():
            cat_stats = cat_df[cat_df['Attack_Name'] == row.name].mean(numeric_only=True)
            print(f"{category:<12} {row.name:<12} {cat_stats['Precision']:>10.3f} {cat_stats['Recall']:>10.3f} {cat_stats['F1']:>10.3f}")
    
    print("-"*60)
    print(f"{'Overall':<12} {'Average':<12} {df['Precision'].mean():>10.3f} {df['Recall'].mean():>10.3f} {df['F1'].mean():>10.3f}")
    print("="*60)


def main():
    print("="*60)
    print("FedACT Table 1 实验 - 攻击检测性能")
    print("="*60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # GPU检测
    gpu_ids, gpu_names = get_gpu_info()
    if gpu_ids:
        print(f"\n检测到 {len(gpu_ids)} 个GPU:")
        for i, name in enumerate(gpu_names):
            print(f"  GPU {i}: {name}")
    else:
        print("\n未检测到GPU，使用CPU运行")
        gpu_ids = [0]
    
    # 生成实验
    experiments = generate_all_experiments()
    print(f"\n总实验数: {len(experiments)}")
    
    # 检查已完成
    completed, pending = check_completed_experiments(experiments, LOGS_DIR)
    print(f"已完成: {len(completed)}, 待运行: {len(pending)}")
    
    if not pending:
        print("\n所有实验已完成！")
        # 直接生成汇总
        results = []
        for exp in experiments:
            log_file = LOGS_DIR / f"{exp['name']}.log"
            results.append({
                **exp['info'],
                'success': True,
                'log_file': str(log_file)
            })
        generate_table1_summary(results, LOGS_DIR)
        return
    
    # 确认运行
    print(f"\n将运行 {len(pending)} 个实验")
    print(f"每个GPU并发: {EXPERIMENTS_PER_GPU}")
    print(f"预计总并发: {len(gpu_ids) * EXPERIMENTS_PER_GPU}")
    
    response = input("\n确认开始? (y/n): ").strip().lower()
    if response != 'y':
        print("已取消")
        return
    
    # 创建任务队列
    task_queue = queue.Queue()
    for exp in pending:
        task_queue.put(exp)
    
    results = []
    threads = []
    
    # 为每个GPU启动多个工作线程
    for gpu_id in gpu_ids:
        for _ in range(EXPERIMENTS_PER_GPU):
            t = threading.Thread(
                target=worker,
                args=(gpu_id, task_queue, results, LOGS_DIR)
            )
            t.start()
            threads.append(t)
    
    # 等待完成
    for t in threads:
        t.join()
    
    # 合并已完成的实验
    for exp in completed:
        log_file = LOGS_DIR / f"{exp['name']}.log"
        results.append({
            **exp['info'],
            'success': True,
            'log_file': str(log_file)
        })
    
    # 生成汇总
    generate_table1_summary(results, LOGS_DIR)
    
    # 统计
    success_count = sum(1 for r in results if r.get('success', False))
    print(f"\n实验完成: {success_count}/{len(experiments)}")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
