#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验工具模块
提供GPU检测、并发执行、进度管理等实验基础功能
"""

import os
import subprocess
import time
import logging
import multiprocessing
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Callable, Any, Optional
from tqdm import tqdm

# 关键：在Linux上使用spawn方式创建子进程，避免继承父进程的CUDA上下文
# 这样每个子进程都会重新初始化CUDA，CUDA_VISIBLE_DEVICES才能生效
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # 已经设置过了


def get_gpu_info() -> Dict[str, Any]:
    """检测本机GPU信息"""
    gpu_info = {
        "available": False,
        "count": 0,
        "devices": [],
        "total_memory_gb": 0,
    }
    
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", 
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            gpu_info["available"] = True
            gpu_info["count"] = len(lines)
            
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    device = {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_total_mb": int(parts[2]),
                        "memory_free_mb": int(parts[3]),
                    }
                    gpu_info["devices"].append(device)
                    gpu_info["total_memory_gb"] += device["memory_total_mb"] / 1024
    except Exception as e:
        gpu_info["error"] = str(e)
    
    return gpu_info


def print_gpu_info(gpu_info: Dict[str, Any]) -> None:
    """打印GPU信息"""
    print("\n" + "="*80)
    print(" "*30 + "GPU 信息")
    print("="*80)
    
    if not gpu_info["available"]:
        print("  未检测到NVIDIA GPU，将使用CPU运行")
        if "error" in gpu_info:
            print(f"  错误信息: {gpu_info['error']}")
    else:
        print(f"  检测到 {gpu_info['count']} 块GPU")
        print(f"  总显存: {gpu_info['total_memory_gb']:.1f} GB")
        print()
        for device in gpu_info["devices"]:
            print(f"    GPU {device['index']}: {device['name']}")
            print(f"           显存: {device['memory_total_mb']}MB (空闲: {device['memory_free_mb']}MB)")
    
    print("="*80)


def check_completed_experiments(
    results_dir: Path, 
    experiment_type: str,
    all_experiments: List[Dict]
) -> tuple:
    """
    检查已完成的实验，支持断点续跑
    
    Args:
        results_dir: 结果目录
        experiment_type: 实验类型 (comparison/attack_defense/ablation)
        all_experiments: 所有实验列表
    
    Returns:
        tuple: (已完成实验列表, 剩余实验列表)
    """
    completed = []
    remaining = []
    
    detail_files = {
        "comparison": "FedACT_对比实验_详细.xlsx",
        "attack_defense": "FedACT_攻击防御实验_详细.xlsx",
        "ablation": "FedACT_消融实验_详细.xlsx",
    }
    
    detail_file = results_dir / detail_files.get(experiment_type, "")
    completed_names = set()
    
    if detail_file.exists():
        try:
            import pandas as pd
            df = pd.read_excel(detail_file)
            if 'name' in df.columns:
                if 'success' in df.columns:
                    completed_names = set(df[df['success'] == True]['name'].tolist())
                else:
                    completed_names = set(df['name'].tolist())
        except Exception as e:
            print(f"  读取已完成实验记录失败: {e}")
    
    for exp in all_experiments:
        if exp['name'] in completed_names:
            completed.append(exp)
        else:
            remaining.append(exp)
    
    return completed, remaining


def print_progress_info(experiment_type: str, total: int, completed: int, remaining: int) -> None:
    """打印进度信息"""
    print("\n" + "="*80)
    print(" "*25 + f"实验进度检查 [{experiment_type}]")
    print("="*80)
    print(f"  总实验数:     {total}")
    print(f"  已完成:       {completed} ({100*completed/total:.1f}%)")
    print(f"  剩余:         {remaining} ({100*remaining/total:.1f}%)")
    print("="*80)


def calculate_concurrent_workers(gpu_count: int, experiments_per_gpu: int = 4) -> int:
    """计算并发工作进程数"""
    if gpu_count == 0:
        return 4
    return gpu_count * experiments_per_gpu


class ExperimentRunner:
    """实验运行器，支持GPU并发和断点续跑"""
    
    def __init__(
        self,
        experiment_type: str,
        results_dir: Path,
        logs_dir: Path,
        experiments_per_gpu: int = 4,
        nohup_log: Optional[Path] = None,
    ):
        self.experiment_type = experiment_type
        self.results_dir = results_dir
        self.logs_dir = logs_dir
        self.experiments_per_gpu = experiments_per_gpu
        self.nohup_log = nohup_log or Path("nohup.log")
        
        self.gpu_info = get_gpu_info()
        self.num_workers = calculate_concurrent_workers(
            self.gpu_info["count"], experiments_per_gpu
        )
        
        self._setup_logging()
    
    def _setup_logging(self):
        """设置日志"""
        self.logger = logging.getLogger(f"Experiment_{self.experiment_type}")
        self.logger.setLevel(logging.INFO)
        
        fh = logging.FileHandler(self.nohup_log, mode='a', encoding='utf-8')
        fh.setLevel(logging.INFO)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.handlers.clear()
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def log(self, message: str, level: str = "info"):
        """记录日志"""
        if level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
    
    def print_startup_info(self, total: int, completed: int, remaining: int):
        """打印启动信息"""
        self.log("=" * 80)
        self.log(f"{self.experiment_type} 实验启动")
        self.log("=" * 80)
        
        if self.gpu_info["available"]:
            self.log(f"GPU数量: {self.gpu_info['count']}")
            for device in self.gpu_info["devices"]:
                self.log(f"  GPU {device['index']}: {device['name']} ({device['memory_total_mb']}MB)")
        else:
            self.log("GPU: 未检测到，使用CPU运行")
        
        self.log(f"每块GPU并发数: {self.experiments_per_gpu}")
        self.log(f"总并发进程数: {self.num_workers}")
        self.log(f"总实验数: {total}")
        self.log(f"已完成: {completed} ({100*completed/total:.1f}%)")
        self.log(f"剩余: {remaining} ({100*remaining/total:.1f}%)")
        self.log("=" * 80)
    
    def run_experiments_concurrent(
        self,
        experiments: List[Dict],
        run_func: Callable,
        all_results: List[Dict] = None,
    ) -> List[Dict]:
        """
        并发运行实验
        
        Args:
            experiments: 待运行实验列表
            run_func: 运行单个实验的函数
            all_results: 已有结果（用于断点续跑）
        
        Returns:
            list: 所有实验结果
        """
        if all_results is None:
            all_results = []
        
        total = len(experiments)
        completed_count = 0
        start_time = time.time()
        
        self.log(f"开始运行 {total} 个实验，并发数: {self.num_workers}")
        
        gpu_count = max(1, self.gpu_info["count"])
        
        # 正确的GPU分配策略：确保实验均匀分布到每块GPU
        for i, exp in enumerate(experiments):
            exp['gpu_id'] = i % gpu_count
        
        # 打印GPU分配情况
        gpu_allocation = {}
        for exp in experiments:
            gid = exp['gpu_id']
            gpu_allocation[gid] = gpu_allocation.get(gid, 0) + 1
        self.log(f"GPU分配情况: {dict(sorted(gpu_allocation.items()))}")
        
        # 转换为字符串，避免序列化问题
        logs_dir_str = str(self.logs_dir)
        
        # 关键：使用 spawn context 创建进程池，确保子进程不继承父进程的CUDA状态
        # 这样每个子进程会重新初始化CUDA，CUDA_VISIBLE_DEVICES 才能正确生效
        mp_context = multiprocessing.get_context('spawn')
        
        with ProcessPoolExecutor(max_workers=self.num_workers, mp_context=mp_context) as executor:
            future_to_exp = {}
            for exp in experiments:
                # 提交任务时打印启动信息
                self.log(f"[启动] {exp['name']} -> GPU {exp['gpu_id']}")
                future = executor.submit(
                    run_func,
                    exp['cmd'],
                    exp['name'],
                    exp['gpu_id'],
                    logs_dir_str,
                )
                future_to_exp[future] = exp
            
            # 使用 tqdm 显示进度条
            pbar = tqdm(total=total, desc="实验进度", unit="exp",
                       bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
            
            for future in as_completed(future_to_exp):
                exp = future_to_exp[future]
                completed_count += 1
                
                try:
                    result = future.result()
                except Exception as e:
                    result = {"name": exp['name'], "success": False, "error": str(e), "gpu_id": exp['gpu_id']}
                
                result.update(exp.get('info', {}))
                all_results.append(result)
                
                elapsed = time.time() - start_time
                avg_time = elapsed / completed_count
                remaining_time = avg_time * (total - completed_count)
                
                status = '✓' if result.get('success') else '✗'
                acc = result.get('accuracy', 0)
                acc_str = f"{acc:.4f}" if isinstance(acc, float) and acc > 0 else "N/A"
                gpu_id = result.get('gpu_id', exp.get('gpu_id', '?'))
                
                # 更新进度条描述
                pbar.set_postfix_str(f"GPU{gpu_id} {status} Acc:{acc_str}")
                pbar.update(1)
                
                # 详细日志：包含GPU信息
                progress_msg = (
                    f"[{completed_count}/{total}] [GPU {gpu_id}] "
                    f"{exp['name']} - {status} Acc:{acc_str} - "
                    f"剩余: {remaining_time/60:.1f}min"
                )
                self.log(progress_msg)
            
            pbar.close()
        
        total_time = time.time() - start_time
        self.log(f"实验完成! 总耗时: {total_time/60:.1f}分钟")
        
        return all_results
    
    def run_experiments_sequential(
        self,
        experiments: List[Dict],
        run_func: Callable,
        all_results: List[Dict] = None,
    ) -> List[Dict]:
        """
        串行运行实验（更稳定，适合Windows）
        
        Args:
            experiments: 待运行实验列表
            run_func: 运行单个实验的函数
            all_results: 已有结果（用于断点续跑）
        
        Returns:
            list: 所有实验结果
        """
        if all_results is None:
            all_results = []
        
        total = len(experiments)
        start_time = time.time()
        
        self.log(f"开始串行运行 {total} 个实验")
        
        gpu_count = max(1, self.gpu_info["count"])
        
        # 预分配GPU（与并发方法一致）
        for i, exp in enumerate(experiments):
            exp['gpu_id'] = i % gpu_count
        
        # 打印GPU分配情况
        gpu_allocation = {}
        for exp in experiments:
            gid = exp['gpu_id']
            gpu_allocation[gid] = gpu_allocation.get(gid, 0) + 1
        self.log(f"GPU分配情况: {dict(sorted(gpu_allocation.items()))}")
        
        # 转换为字符串
        logs_dir_str = str(self.logs_dir)
        
        # 使用 tqdm 显示进度条
        pbar = tqdm(experiments, desc="实验进度", unit="exp",
                   bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
        
        for i, exp in enumerate(pbar):
            gpu_id = exp['gpu_id']  # 使用预分配的 gpu_id
            
            try:
                result = run_func(
                    exp['cmd'],
                    exp['name'],
                    gpu_id,
                    logs_dir_str,
                )
            except Exception as e:
                result = {"name": exp['name'], "success": False, "error": str(e)}
            
            result.update(exp.get('info', {}))
            all_results.append(result)
            
            elapsed = time.time() - start_time
            completed_count = i + 1
            avg_time = elapsed / completed_count
            remaining_time = avg_time * (total - completed_count)
            
            status = '✓' if result.get('success') else '✗'
            acc = result.get('accuracy', 0)
            acc_str = f"{acc:.4f}" if isinstance(acc, float) and acc > 0 else "N/A"
            
            # 更新进度条描述
            pbar.set_postfix_str(f"{status} Acc:{acc_str}")
            
            # 日志
            progress_msg = (
                f"[{completed_count}/{total}] ({100*completed_count/total:.1f}%) "
                f"{exp['name']} - {'成功' if result.get('success') else '失败'} - "
                f"Acc: {acc_str} - "
                f"剩余时间: {remaining_time/60:.1f}分钟"
            )
            self.log(progress_msg)
        
        total_time = time.time() - start_time
        self.log(f"实验完成! 总耗时: {total_time/60:.1f}分钟")
        
        return all_results


def run_single_experiment(
    cmd: List[str],
    exp_name: str,
    gpu_id: int,
    logs_dir: str,
) -> Dict:
    """
    运行单个实验（在独立的子进程中执行）
    
    Args:
        cmd: 命令列表
        exp_name: 实验名称
        gpu_id: 要使用的GPU编号（物理GPU ID，如0, 1, 2, 3, 4）
        logs_dir: 日志目录（字符串）
    
    Returns:
        dict: 实验结果
    """
    import os
    import subprocess
    from pathlib import Path
    from datetime import datetime
    
    log_file = Path(logs_dir) / f"{exp_name}.log"
    
    start_time = time.time()
    
    # 关键：设置 CUDA_VISIBLE_DEVICES 为指定的物理GPU ID
    # 在 spawn 模式下，这是一个全新的进程，环境变量设置会生效
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 实时打印启动信息
    print(f"[GPU {gpu_id}] 启动: {exp_name}", flush=True)
    
    # main.py 在 system/flcore/main.py
    main_py_path = Path(cmd[1]) if len(cmd) > 1 else Path.cwd()
    
    if main_py_path.is_absolute():
        system_dir = main_py_path.parent.parent.resolve()
        relative_main_py = "flcore/main.py"
    else:
        system_dir = Path.cwd() / main_py_path.parent.parent
        system_dir = system_dir.resolve()
        relative_main_py = "flcore/main.py"
    
    env["PYTHONPATH"] = str(system_dir)
    
    fixed_cmd = cmd.copy()
    fixed_cmd[1] = relative_main_py
    
    try:
        result = subprocess.run(
            fixed_cmd, capture_output=True, text=True, timeout=7200,
            cwd=str(system_dir), env=env
        )
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"实验: {exp_name}\n")
            f.write(f"GPU ID: {gpu_id}\n")
            f.write(f"CUDA_VISIBLE_DEVICES: {gpu_id}\n")
            f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"命令: {' '.join(fixed_cmd)}\n")
            f.write(f"工作目录: {system_dir}\n\n")
            f.write(result.stdout)
            if result.stderr:
                f.write(f"\n===STDERR===\n{result.stderr}")
        
        metrics = parse_experiment_results(result.stdout)
        elapsed = time.time() - start_time
        
        return {
            "name": exp_name,
            "success": result.returncode == 0,
            "elapsed": elapsed,
            "gpu_id": gpu_id,
            **metrics
        }
        
    except subprocess.TimeoutExpired:
        return {"name": exp_name, "success": False, "error": "Timeout", "gpu_id": gpu_id}
    except Exception as e:
        return {"name": exp_name, "success": False, "error": str(e), "gpu_id": gpu_id}


def parse_experiment_results(output: str) -> Dict:
    """解析实验输出，提取指标"""
    metrics = {
        "accuracy": 0.0,
        "auc": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }
    
    # 从后向前遍历，找最后一轮的结果
    for line in reversed(output.split('\n')):
        line_l = line.lower()
        try:
            # 只匹配 "Averaged Test ..." 格式，避免匹配 "Std Test ..."
            if "averaged test accuracy:" in line_l:
                val = line.split(":")[-1].split(",")[0].strip().replace("%", "")
                if metrics["accuracy"] == 0.0:
                    metrics["accuracy"] = float(val)
            if "averaged test auc:" in line_l:
                val = line.split(":")[-1].split(",")[0].strip()
                if metrics["auc"] == 0.0:
                    metrics["auc"] = float(val)
            if "averaged test precision:" in line_l:
                val = line.split(":")[-1].split(",")[0].strip()
                if metrics["precision"] == 0.0:
                    metrics["precision"] = float(val)
            if "averaged test recall:" in line_l:
                val = line.split(":")[-1].split(",")[0].strip()
                if metrics["recall"] == 0.0:
                    metrics["recall"] = float(val)
            if "averaged test f1-score:" in line_l:
                val = line.split(":")[-1].split(",")[0].strip()
                if metrics["f1"] == 0.0:
                    metrics["f1"] = float(val)
        except:
            pass
    
    # 如果仍然是0，检查是否有错误输出表明实验根本没运行成功
    if all(v == 0.0 for v in metrics.values()):
        # 检查是否有错误信息
        if "error" in output.lower() or "exception" in output.lower():
            pass  # 保持为0
    
    return metrics


def save_incremental_results(results: List[Dict], results_dir: Path, detail_filename: str) -> None:
    """增量保存详细结果"""
    import pandas as pd
    
    detail_path = results_dir / detail_filename
    
    if detail_path.exists():
        try:
            existing_df = pd.read_excel(detail_path)
            existing_names = set(existing_df['name'].tolist()) if 'name' in existing_df.columns else set()
            
            new_results = [r for r in results if r.get('name') not in existing_names]
            if new_results:
                new_df = pd.DataFrame(new_results)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df.to_excel(detail_path, index=False)
        except Exception as e:
            print(f"增量保存失败: {e}")
            pd.DataFrame(results).to_excel(detail_path, index=False)
    else:
        pd.DataFrame(results).to_excel(detail_path, index=False)


def generate_detection_summary_excel(stats_dir: str = 'results/检测统计', 
                                      output_file: str = 'results/检测统计/汇总.xlsx'):
    """
    生成检测统计汇总Excel
    
    汇总所有实验的检测结果，按 "数据集_异质性" 分sheet
    
    Args:
        stats_dir: 检测统计JSON文件所在目录
        output_file: 输出的汇总Excel文件路径
        
    Returns:
        生成的Excel文件路径
    """
    import json
    import pandas as pd
    from pathlib import Path
    
    stats_path = Path(stats_dir)
    if not stats_path.exists():
        print(f"❌ 目录不存在: {stats_dir}")
        return None
    
    # 收集所有JSON文件
    json_files = list(stats_path.glob('*_stats.json'))
    if not json_files:
        print(f"❌ 未找到统计文件: {stats_dir}")
        return None
    
    # 按 数据集_异质性 分组
    grouped_data = {}  # key: "数据集_异质性", value: list of records
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            config = data.get('config', {})
            dataset = config.get('dataset', 'unknown')
            heterogeneity = config.get('heterogeneity', 'iid')
            defense_mode = config.get('defense_mode', 'unknown')
            attack_mode = config.get('attack_mode', 'none')
            
            # 生成sheet名
            sheet_name = f"{dataset}_{heterogeneity}"
            
            # 准备记录
            cumulative = data.get('cumulative', {})
            overall = data.get('overall_metrics', {})
            
            record = {
                '防御方法': defense_mode,
                '攻击类型': attack_mode,
                '恶意比例': config.get('malicious_ratio', 0),
                'TP': cumulative.get('tp', 0),
                'FP': cumulative.get('fp', 0),
                'TN': cumulative.get('tn', 0),
                'FN': cumulative.get('fn', 0),
                'Precision': round(overall.get('precision', 0), 4),
                'Recall': round(overall.get('recall', 0), 4),
                'Accuracy': round(overall.get('accuracy', 0), 4),
            }
            
            # 计算F1
            prec = overall.get('precision', 0)
            rec = overall.get('recall', 0)
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            record['F1'] = round(f1, 4)
            
            if sheet_name not in grouped_data:
                grouped_data[sheet_name] = []
            grouped_data[sheet_name].append(record)
            
        except Exception as e:
            print(f"⚠️ 读取失败 {json_file}: {e}")
            continue
    
    if not grouped_data:
        print("❌ 没有有效的统计数据")
        return None
    
    # 写入Excel（多sheet）
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for sheet_name, records in sorted(grouped_data.items()):
                df = pd.DataFrame(records)
                # 按防御方法和攻击类型排序
                if '防御方法' in df.columns and '攻击类型' in df.columns:
                    df = df.sort_values(['防御方法', '攻击类型'])
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"✅ 汇总Excel已生成: {output_path}")
        print(f"   包含 {len(grouped_data)} 个sheet: {', '.join(sorted(grouped_data.keys()))}")
        return str(output_path)
        
    except Exception as e:
        print(f"❌ 生成Excel失败: {e}")
        return None


if __name__ == "__main__":
    gpu_info = get_gpu_info()
    print_gpu_info(gpu_info)
    
    workers = calculate_concurrent_workers(gpu_info["count"])
    print(f"\n计算的并发进程数: {workers}")
