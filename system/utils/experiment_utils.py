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
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Callable, Any, Optional


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
        for i, exp in enumerate(experiments):
            exp['gpu_id'] = i % gpu_count
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_exp = {}
            for exp in experiments:
                future = executor.submit(
                    run_func,
                    exp['cmd'],
                    exp['name'],
                    exp.get('gpu_id', 0),
                    self.logs_dir,
                )
                future_to_exp[future] = exp
            
            for future in as_completed(future_to_exp):
                exp = future_to_exp[future]
                completed_count += 1
                
                try:
                    result = future.result()
                except Exception as e:
                    result = {"name": exp['name'], "success": False, "error": str(e)}
                
                result.update(exp.get('info', {}))
                all_results.append(result)
                
                elapsed = time.time() - start_time
                avg_time = elapsed / completed_count
                remaining_time = avg_time * (total - completed_count)
                
                status = '成功' if result.get('success') else '失败'
                progress_msg = (
                    f"[{completed_count}/{total}] ({100*completed_count/total:.1f}%) "
                    f"{exp['name']} - {status} - "
                    f"Acc: {result.get('accuracy', 'N/A')} - "
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
    logs_dir: Path,
) -> Dict:
    """运行单个实验"""
    log_file = logs_dir / f"{exp_name}.log"
    
    start_time = time.time()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    system_dir = Path(cmd[1]).parent if len(cmd) > 1 else Path.cwd()
    env["PYTHONPATH"] = str(system_dir)
    
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=7200,
            cwd=str(system_dir), env=env
        )
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"实验: {exp_name}\n")
            f.write(f"GPU: {gpu_id}\n")
            f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"命令: {' '.join(cmd)}\n\n")
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
    
    for line in reversed(output.split('\n')):
        line_l = line.lower()
        try:
            if "accuracy:" in line_l:
                val = line.split(":")[-1].split(",")[0].strip().replace("%", "")
                metrics["accuracy"] = float(val)
            if "auc:" in line_l:
                val = line.split("AUC:")[-1].split(",")[0].strip()
                metrics["auc"] = float(val)
            if "precision:" in line_l:
                val = line.split(":")[-1].split(",")[0].strip()
                metrics["precision"] = float(val)
            if "recall:" in line_l:
                val = line.split(":")[-1].split(",")[0].strip()
                metrics["recall"] = float(val)
            if "f1" in line_l and "f1:" in line_l:
                val = line.split(":")[-1].split(",")[0].strip()
                metrics["f1"] = float(val)
        except:
            pass
    
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


if __name__ == "__main__":
    gpu_info = get_gpu_info()
    print_gpu_info(gpu_info)
    
    workers = calculate_concurrent_workers(gpu_info["count"])
    print(f"\n计算的并发进程数: {workers}")
