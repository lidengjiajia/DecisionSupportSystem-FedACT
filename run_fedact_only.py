#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
FedACT 专用实验脚本 - 只运行 FedACT 完整版实验
================================================================================

用于调优 FedACT 算法参数后重新运行实验

使用方法:
    python run_fedact_only.py                    # 运行所有实验
    python run_fedact_only.py --dataset Uci      # 只运行 UCI 数据集
    python run_fedact_only.py --attack gaussian  # 只运行特定攻击
    python run_fedact_only.py --het iid          # 只运行特定异质性
    python run_fedact_only.py --resume           # 断点续跑

作者: FedACT Team
日期: 2026-01-27
"""

import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess

# 路径配置
BASE_DIR = Path(__file__).parent.absolute()
SYSTEM_DIR = BASE_DIR / "system"
LOGS_DIR = BASE_DIR / "logs" / "fedact_tuning"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# 实验配置
DATASETS = ["Uci", "Xinwang"]

HETEROGENEITY_TYPES = ["iid", "label_skew", "quantity_skew", "feature_skew"]

# 全部12种攻击
ALL_ATTACKS = [
    "sign_flip", "gaussian", "scale",        # 基础攻击
    "little", "alie", "ipm", "minmax", "trim_attack",  # 前沿攻击
    "label_flip", "backdoor", "free_rider", "collision"  # 其他攻击
]

TRAIN_PARAMS = {
    "num_clients": 10,
    "global_rounds": 100,
    "local_epochs": 5,
    "batch_size": 64,
    "learning_rate": 0.01,
    "eval_gap": 10,
}


def build_fedact_command(dataset: str, heterogeneity: str, attack: str) -> list:
    """构建 FedACT 完整版实验命令"""
    cmd = [
        sys.executable,
        str(SYSTEM_DIR / "flcore" / "main.py"),
        "-data", dataset,
        "-algo", "FedTLBO",
        "-nc", str(TRAIN_PARAMS["num_clients"]),
        "-gr", str(TRAIN_PARAMS["global_rounds"]),
        "-ls", str(TRAIN_PARAMS["local_epochs"]),
        "-lbs", str(TRAIN_PARAMS["batch_size"]),
        "-lr", str(TRAIN_PARAMS["learning_rate"]),
        "-eg", str(TRAIN_PARAMS["eval_gap"]),
        "-t", "1",
        "--heterogeneity", heterogeneity,
        "--enable_attack", "True",
        "--attack_mode", attack,
        "--malicious_ratio", "0.3",
        "--defense_mode", "fedact",
        "--use_autoencoder", "True",
        "--use_committee", "True",
        "--use_tlbo", "True",
        "--tlbo_iterations", "10",
        "--committee_size", "5",
    ]
    return cmd


def get_gpu_info():
    """获取GPU信息"""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free', '--format=csv,noheader'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    gpus.append({
                        'id': int(parts[0]),
                        'name': parts[1],
                        'total_memory': parts[2],
                        'free_memory': parts[3]
                    })
            return gpus
    except Exception:
        pass
    return []


def run_experiment(exp_config, gpu_id):
    """运行单个实验"""
    name = exp_config['name']
    cmd = exp_config['cmd']
    log_file = LOGS_DIR / f"{name}.log"
    
    # 设置环境变量
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # 写入日志头
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"实验: {name}\n")
        f.write(f"GPU ID: {gpu_id}\n")
        f.write(f"CUDA_VISIBLE_DEVICES: {gpu_id}\n")
        f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"命令: {' '.join(cmd)}\n")
        f.write(f"工作目录: {SYSTEM_DIR}\n\n")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(SYSTEM_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(result.stdout)
            f.write(f"\n\n返回码: {result.returncode}\n")
            f.write(f"耗时: {time.time() - start_time:.1f}秒\n")
        
        return {
            'name': name,
            'success': result.returncode == 0,
            'returncode': result.returncode,
            'duration': time.time() - start_time
        }
    except Exception as e:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n错误: {str(e)}\n")
        return {
            'name': name,
            'success': False,
            'error': str(e),
            'duration': time.time() - start_time
        }


def generate_experiments(datasets=None, heterogeneities=None, attacks=None):
    """生成实验配置列表"""
    experiments = []
    
    _datasets = datasets or DATASETS
    _hets = heterogeneities or HETEROGENEITY_TYPES
    _attacks = attacks or ALL_ATTACKS
    
    for dataset in _datasets:
        for het in _hets:
            for attack in _attacks:
                exp_name = f"FedACT_{dataset}_{het}_{attack}"
                cmd = build_fedact_command(dataset, het, attack)
                experiments.append({
                    'name': exp_name,
                    'cmd': cmd,
                    'dataset': dataset,
                    'heterogeneity': het,
                    'attack': attack
                })
    
    return experiments


def check_completed(experiments):
    """检查已完成的实验"""
    completed = set()
    for exp in experiments:
        log_file = LOGS_DIR / f"{exp['name']}.log"
        if log_file.exists():
            content = log_file.read_text(encoding='utf-8', errors='ignore')
            if '返回码: 0' in content or '训练完成' in content:
                completed.add(exp['name'])
    return completed


def main():
    parser = argparse.ArgumentParser(description='FedACT 专用实验脚本')
    parser.add_argument('--dataset', '-d', choices=DATASETS, help='指定数据集')
    parser.add_argument('--attack', '-a', choices=ALL_ATTACKS, help='指定攻击类型')
    parser.add_argument('--het', '-H', choices=HETEROGENEITY_TYPES, help='指定异质性类型')
    parser.add_argument('--resume', '-r', action='store_true', help='断点续跑')
    parser.add_argument('--parallel', '-p', type=int, default=3, help='每块GPU并发数')
    parser.add_argument('--dry-run', action='store_true', help='只打印实验列表')
    args = parser.parse_args()
    
    print("=" * 60)
    print("  FedACT 专用实验脚本")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 获取GPU信息
    gpus = get_gpu_info()
    if gpus:
        print(f"检测到 {len(gpus)} 块 GPU:")
        for gpu in gpus:
            print(f"  GPU {gpu['id']}: {gpu['name']} ({gpu['free_memory']} 可用)")
    else:
        print("未检测到GPU，将使用CPU运行")
        gpus = [{'id': 0}]
    print()
    
    # 生成实验列表
    datasets = [args.dataset] if args.dataset else None
    hets = [args.het] if args.het else None
    attacks = [args.attack] if args.attack else None
    
    experiments = generate_experiments(datasets, hets, attacks)
    
    # 断点续跑
    if args.resume:
        completed = check_completed(experiments)
        experiments = [e for e in experiments if e['name'] not in completed]
        print(f"断点续跑: 跳过 {len(completed)} 个已完成实验")
    
    print(f"待运行实验数: {len(experiments)}")
    print()
    
    if args.dry_run:
        print("实验列表:")
        for i, exp in enumerate(experiments, 1):
            print(f"  {i}. {exp['name']}")
        return
    
    if not experiments:
        print("没有待运行的实验")
        return
    
    # 确认运行
    confirm = input(f"是否开始运行 {len(experiments)} 个实验? (y/n): ")
    if confirm.lower() != 'y':
        print("已取消")
        return
    
    # 并发运行
    total_gpus = len(gpus)
    max_workers = total_gpus * args.parallel
    
    print(f"\n开始运行，GPU数={total_gpus}，每GPU并发={args.parallel}，总并发={max_workers}")
    print("-" * 60)
    
    gpu_index = 0
    completed_count = 0
    failed_count = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for exp in experiments:
            gpu_id = gpus[gpu_index % total_gpus]['id']
            gpu_index += 1
            future = executor.submit(run_experiment, exp, gpu_id)
            futures[future] = exp['name']
        
        for future in as_completed(futures):
            exp_name = futures[future]
            try:
                result = future.result()
                if result['success']:
                    completed_count += 1
                    status = "✓"
                else:
                    failed_count += 1
                    status = "✗"
                print(f"[{completed_count + failed_count}/{len(experiments)}] {status} {exp_name} ({result['duration']:.1f}s)")
            except Exception as e:
                failed_count += 1
                print(f"[{completed_count + failed_count}/{len(experiments)}] ✗ {exp_name} (错误: {e})")
    
    print()
    print("=" * 60)
    print(f"  完成: {completed_count}, 失败: {failed_count}")
    print(f"  结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  日志目录: {LOGS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
