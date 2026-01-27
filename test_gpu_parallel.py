#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPU 并发测试脚本
用于验证多GPU并发是否正常工作

使用方法:
    python test_gpu_parallel.py

同时在另一个终端运行:
    watch -n 1 nvidia-smi
"""

import os
import sys
import time
import subprocess
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# 关键：使用 spawn 模式，确保子进程不继承 CUDA 上下文
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.absolute()))
sys.path.insert(0, str(Path(__file__).parent.absolute() / "system"))

from system.utils.experiment_utils import get_gpu_info, print_gpu_info


def run_gpu_test(gpu_id: int, test_id: int) -> dict:
    """在指定GPU上运行一个简单测试"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 简单的 Python 命令：检测 GPU 并等待几秒
    test_script = f'''
import torch
import time
import os

gpu_env = os.environ.get("CUDA_VISIBLE_DEVICES", "NOT_SET")
print(f"Test {test_id}: CUDA_VISIBLE_DEVICES = {{gpu_env}}")
print(f"Test {test_id}: torch.cuda.is_available() = {{torch.cuda.is_available()}}")
if torch.cuda.is_available():
    print(f"Test {test_id}: torch.cuda.device_count() = {{torch.cuda.device_count()}}")
    print(f"Test {test_id}: torch.cuda.current_device() = {{torch.cuda.current_device()}}")
    # 在 GPU 上分配一些内存来验证
    x = torch.randn(1000, 1000, device="cuda")
    print(f"Test {test_id}: Allocated tensor on GPU, shape={{x.shape}}")
print(f"Test {test_id}: Sleeping for 10 seconds...")
time.sleep(10)
print(f"Test {test_id}: Done!")
'''
    
    print(f"[主进程] 启动测试 {test_id} -> GPU {gpu_id}", flush=True)
    
    try:
        result = subprocess.run(
            [sys.executable, "-c", test_script],
            capture_output=True,
            text=True,
            timeout=60,
            env=env
        )
        return {
            "test_id": test_id,
            "gpu_id": gpu_id,
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except Exception as e:
        return {
            "test_id": test_id,
            "gpu_id": gpu_id,
            "success": False,
            "error": str(e),
        }


def main():
    print("=" * 60)
    print("GPU 并发测试")
    print("=" * 60)
    
    # 检测GPU
    gpu_info = get_gpu_info()
    print_gpu_info(gpu_info)
    
    gpu_count = gpu_info["count"]
    if gpu_count == 0:
        print("未检测到GPU，退出测试")
        return
    
    # 每个GPU启动2个测试进程
    tests_per_gpu = 2
    total_tests = gpu_count * tests_per_gpu
    
    print(f"\n准备启动 {total_tests} 个并发测试 ({gpu_count} GPUs × {tests_per_gpu} tests/GPU)")
    print("每个测试将在 GPU 上分配内存并等待10秒")
    print("请在另一个终端运行 nvidia-smi 观察GPU使用情况\n")
    
    input("按 Enter 开始测试...")
    
    # 准备测试任务
    tests = []
    for i in range(total_tests):
        gpu_id = i % gpu_count
        tests.append({"test_id": i, "gpu_id": gpu_id})
    
    print(f"\n测试分配: {[(t['test_id'], t['gpu_id']) for t in tests]}")
    
    # 并发执行 - 使用 spawn context
    start_time = time.time()
    results = []
    
    mp_context = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=total_tests, mp_context=mp_context) as executor:
        futures = {
            executor.submit(run_gpu_test, t["gpu_id"], t["test_id"]): t
            for t in tests
        }
        
        for future in as_completed(futures):
            test = futures[future]
            try:
                result = future.result()
                results.append(result)
                status = "✓" if result["success"] else "✗"
                print(f"[完成] 测试 {result['test_id']} (GPU {result['gpu_id']}): {status}", flush=True)
            except Exception as e:
                print(f"[错误] 测试 {test['test_id']}: {e}", flush=True)
    
    elapsed = time.time() - start_time
    
    # 打印结果
    print("\n" + "=" * 60)
    print("测试结果")
    print("=" * 60)
    
    for r in sorted(results, key=lambda x: x["test_id"]):
        print(f"\n--- 测试 {r['test_id']} (GPU {r['gpu_id']}) ---")
        if r["success"]:
            print(r["stdout"])
        else:
            print(f"失败: {r.get('error', r.get('stderr', 'Unknown'))}")
    
    print(f"\n总耗时: {elapsed:.2f} 秒")
    print(f"如果并发正常工作，耗时应该接近 10 秒（而不是 {total_tests * 10} 秒）")
    
    if elapsed < 15:
        print("✓ 并发工作正常！")
    else:
        print("✗ 并发可能有问题，请检查日志")


if __name__ == "__main__":
    main()
