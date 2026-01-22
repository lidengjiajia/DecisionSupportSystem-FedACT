#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一键生成所有数据集的所有异质性模式
运行此脚本将自动生成：
- UCI数据集：Feature, Label, Quantity, IID 四种模式
- Xinwang数据集：Feature, Label, Quantity, IID 四种模式
共计8个数据集

使用方法：
    python dataset/generate_all_datasets_auto.py
"""

import sys
import subprocess
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
from scipy.stats import wasserstein_distance

def run_dataset_generation(dataset_name, choice, dataset_script):
    """
    运行数据集生成
    
    Args:
        dataset_name: 数据集名称（用于显示）
        choice: 异质性类型选项 (1/2/3/4)
        dataset_script: 生成脚本路径
    """
    choice_names = {
        "1": "Feature Heterogeneity (特征异质性)",
        "2": "Label Heterogeneity (标签异质性)",
        "3": "Quantity Heterogeneity (数量异质性)",
        "4": "IID Uniform (独立同分布)"
    }
    
    print("\n" + "="*80)
    print(f"正在生成: {dataset_name} - {choice_names[choice]}")
    print("="*80)
    
    # 使用echo传入选项，自动运行
    cmd = f'echo {choice} | python {dataset_script}'
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"✓ {dataset_name} - {choice_names[choice]} 生成成功！耗时: {elapsed:.1f}秒")
    else:
        print(f"✗ {dataset_name} - {choice_names[choice]} 生成失败！")
        return False
    
    return True

def clear_plot_directory():
    """清空绘图目录"""
    # 使用项目根目录的绝对路径
    project_root = Path(__file__).parent.parent.resolve()
    plot_dir = project_root / "system" / "results" / "汇总" / "heterogeneity_plots"
    if plot_dir.exists():
        shutil.rmtree(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ 已清空并重新创建绘图目录: {plot_dir}")

def load_client_data(dataset_name, heterogeneity_type):
    """加载客户端数据"""
    # 使用脚本所在目录作为基准
    script_dir = Path(__file__).parent.resolve()
    data_dir = script_dir / dataset_name / heterogeneity_type
    client_data = {}
    
    if not data_dir.exists():
        return None
    
    for client_id in range(10):  # 假设10个客户端
        train_file = data_dir / "train" / f"{client_id}.npz"
        test_file = data_dir / "test" / f"{client_id}.npz"
        
        if train_file.exists() and test_file.exists():
            # 使用allow_pickle=True加载数据
            train_data = np.load(train_file, allow_pickle=True)
            test_data = np.load(test_file, allow_pickle=True)
            
            # 数据存储在'data'键中，是一个字典
            train_dict = train_data['data'].item()
            test_dict = test_data['data'].item()
            
            # 合并训练和测试数据
            X = np.concatenate([train_dict['x'], test_dict['x']], axis=0)
            y = np.concatenate([train_dict['y'], test_dict['y']], axis=0)
            
            client_data[client_id] = {'X': X, 'y': y}
    
    return client_data

def setup_nature_style():
    """设置 Nature 期刊风格"""
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 12,
        'axes.linewidth': 1.2,
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#333333',
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'legend.frameon': True,
        'legend.edgecolor': '#CCCCCC',
        'legend.fancybox': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': False,
    })

# Nature 风格配色（柔和优雅）
NATURE_COLORS = {
    'blue': '#4A90A4',      # 柔和蓝
    'orange': '#E07B54',    # 柔和橙
    'green': '#6B9E78',     # 柔和绿
    'purple': '#8B7CB3',    # 柔和紫
    'teal': '#5BA3A8',      # 柔和青
    'red': '#C75B5B',       # 柔和红
}

def plot_label_heterogeneity():
    """绘制标签异质性分布图 - Nature 风格"""
    setup_nature_style()
    colors = [NATURE_COLORS['blue'], NATURE_COLORS['orange']]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 加载UCI数据
    uci_data = load_client_data("Uci", "label")
    # 加载Xinwang数据
    xinwang_data = load_client_data("Xinwang", "label")
    
    if uci_data and xinwang_data:
        clients = list(range(10))
        
        # 计算每个客户端的正样本比例
        uci_pos_ratios = []
        xinwang_pos_ratios = []
        
        for client_id in clients:
            if client_id in uci_data:
                y_uci = uci_data[client_id]['y']
                uci_pos_ratios.append(np.mean(y_uci == 1))
            else:
                uci_pos_ratios.append(0)
                
            if client_id in xinwang_data:
                y_xinwang = xinwang_data[client_id]['y']
                xinwang_pos_ratios.append(np.mean(y_xinwang == 1))
            else:
                xinwang_pos_ratios.append(0)
        
        # 绘制柱状图
        bar_width = 0.35
        x = np.arange(len(clients))
        
        bars1 = ax.bar(x - bar_width/2, uci_pos_ratios, bar_width, 
                      label='UCI', color=colors[0], alpha=0.85, edgecolor='white', linewidth=0.8)
        bars2 = ax.bar(x + bar_width/2, xinwang_pos_ratios, bar_width, 
                      label='Xinwang', color=colors[1], alpha=0.85, edgecolor='white', linewidth=0.8)
        
        # 添加数值标签 - 简洁风格
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.003,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8, color='#555555')
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.003,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8, color='#555555')
        
        ax.set_xlabel('Client ID', fontsize=12)
        ax.set_ylabel('Positive Sample Ratio', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f'C{i}' for i in clients], fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
        
        plt.tight_layout()
        project_root = Path(__file__).parent.parent.resolve()
        save_path = project_root / "system" / "results" / "汇总" / "heterogeneity_plots" / "label_heterogeneity.png"
        plt.savefig(str(save_path), dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(str(save_path).replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        plt.close()

def plot_feature_heterogeneity():
    """绘制特征异质性分布图 - Nature 风格"""
    setup_nature_style()
    colors = [NATURE_COLORS['blue'], NATURE_COLORS['orange']]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 加载UCI数据
    uci_data = load_client_data("Uci", "feature")
    # 加载Xinwang数据
    xinwang_data = load_client_data("Xinwang", "feature")
    
    if uci_data and xinwang_data:
        clients = list(range(10))
        
        # 计算全局特征分布
        def compute_global_distribution(data_dict):
            all_features = []
            for client_data in data_dict.values():
                all_features.append(client_data['X'])
            return np.concatenate(all_features, axis=0)
        
        uci_global = compute_global_distribution(uci_data)
        xinwang_global = compute_global_distribution(xinwang_data)
        
        # 计算每个客户端的Wasserstein距离
        uci_wasserstein = []
        xinwang_wasserstein = []
        
        for client_id in clients:
            if client_id in uci_data:
                client_features = uci_data[client_id]['X']
                # 计算该客户端与全局分布的平均Wasserstein距离（对所有特征）
                distances = []
                for feature_idx in range(client_features.shape[1]):
                    dist = wasserstein_distance(client_features[:, feature_idx], uci_global[:, feature_idx])
                    distances.append(dist)
                uci_wasserstein.append(np.mean(distances))
            else:
                uci_wasserstein.append(0)
                
            if client_id in xinwang_data:
                client_features = xinwang_data[client_id]['X']
                # 计算该客户端与全局分布的平均Wasserstein距离（对所有特征）
                distances = []
                for feature_idx in range(client_features.shape[1]):
                    dist = wasserstein_distance(client_features[:, feature_idx], xinwang_global[:, feature_idx])
                    distances.append(dist)
                xinwang_wasserstein.append(np.mean(distances))
            else:
                xinwang_wasserstein.append(0)
        
        # 绘制柱状图
        bar_width = 0.35
        x = np.arange(len(clients))
        
        bars1 = ax.bar(x - bar_width/2, uci_wasserstein, bar_width, 
                      label='UCI', color=colors[0], alpha=0.85, edgecolor='white', linewidth=0.8)
        bars2 = ax.bar(x + bar_width/2, xinwang_wasserstein, bar_width, 
                      label='Xinwang', color=colors[1], alpha=0.85, edgecolor='white', linewidth=0.8)
        
        # 添加数值标签 - 简洁风格
        max_uci = max(uci_wasserstein) if uci_wasserstein else 0
        max_xinwang = max(xinwang_wasserstein) if xinwang_wasserstein else 0
        max_height = max(max_uci, max_xinwang)
        
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_height * 0.02,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=8, color='#555555')
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_height * 0.02,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=8, color='#555555')
        
        ax.set_xlabel('Client ID', fontsize=12)
        ax.set_ylabel('Average Wasserstein Distance', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f'C{i}' for i in clients], fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
        
        plt.tight_layout()
        project_root = Path(__file__).parent.parent.resolve()
        save_path = project_root / "system" / "results" / "汇总" / "heterogeneity_plots" / "feature_heterogeneity.png"
        plt.savefig(str(save_path), dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(str(save_path).replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        plt.close()

def plot_quantity_heterogeneity():
    """绘制数量异质性分布图 - Nature 风格"""
    setup_nature_style()
    colors = [NATURE_COLORS['blue'], NATURE_COLORS['orange']]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 加载UCI数据
    uci_data = load_client_data("Uci", "quantity")
    # 加载Xinwang数据
    xinwang_data = load_client_data("Xinwang", "quantity")
    
    if uci_data and xinwang_data:
        clients = list(range(10))
        
        # 计算每个客户端的样本数量
        uci_sample_counts = []
        xinwang_sample_counts = []
        
        for client_id in clients:
            if client_id in uci_data:
                uci_sample_counts.append(len(uci_data[client_id]['y']))
            else:
                uci_sample_counts.append(0)
                
            if client_id in xinwang_data:
                xinwang_sample_counts.append(len(xinwang_data[client_id]['y']))
            else:
                xinwang_sample_counts.append(0)
        
        # 绘制柱状图
        bar_width = 0.35
        x = np.arange(len(clients))
        
        bars1 = ax.bar(x - bar_width/2, uci_sample_counts, bar_width, 
                      label='UCI', color=colors[0], alpha=0.85, edgecolor='white', linewidth=0.8)
        bars2 = ax.bar(x + bar_width/2, xinwang_sample_counts, bar_width, 
                      label='Xinwang', color=colors[1], alpha=0.85, edgecolor='white', linewidth=0.8)
        
        # 添加数值标签 - 简洁风格
        max_uci = max(uci_sample_counts) if uci_sample_counts else 0
        max_xinwang = max(xinwang_sample_counts) if xinwang_sample_counts else 0
        max_height = max(max_uci, max_xinwang)
        
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_height * 0.02,
                   f'{int(height)}', ha='center', va='bottom', fontsize=8, color='#555555')
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_height * 0.02,
                   f'{int(height)}', ha='center', va='bottom', fontsize=8, color='#555555')
        
        ax.set_xlabel('Client ID', fontsize=12)
        ax.set_ylabel('Sample Count', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f'C{i}' for i in clients], fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
        
        plt.tight_layout()
        project_root = Path(__file__).parent.parent.resolve()
        save_path = project_root / "system" / "results" / "汇总" / "heterogeneity_plots" / "quantity_heterogeneity.png"
        plt.savefig(str(save_path), dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(str(save_path).replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        plt.close()

def generate_plots():
    """生成所有异质性分布图"""
    print("\n" + "="*80)
    print("                    生成异质性分布图")
    print("="*80)
    
    try:
        plot_label_heterogeneity()
        print("✓ 标签异质性分布图生成完成")
        
        plot_feature_heterogeneity()
        print("✓ 特征异质性分布图生成完成")
        
        plot_quantity_heterogeneity()
        print("✓ 数量异质性分布图生成完成")
        
        print(f"✓ 所有分布图已保存至: results/汇总/heterogeneity_plots/")
        
    except Exception as e:
        print(f"✗ 绘图过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """主函数：依次生成所有数据集"""
    
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == "--plots-only":
        print("\n" + "="*80)
        print("                    生成异质性分布图")
        print("="*80)
        clear_plot_directory()
        generate_plots()
        print("✓ 异质性分布图生成完成！")
        return
    
    print("\n" + "="*80)
    print("              联邦学习数据集批量生成工具")
    print("="*80)
    print("将生成以下数据集：")
    print("  1. UCI - Feature Heterogeneity")
    print("  2. UCI - Label Heterogeneity")
    print("  3. UCI - Quantity Heterogeneity")
    print("  4. UCI - IID Uniform")
    print("  5. Xinwang - Feature Heterogeneity")
    print("  6. Xinwang - Label Heterogeneity")
    print("  7. Xinwang - Quantity Heterogeneity")
    print("  8. Xinwang - IID Uniform")
    print("="*80)
    
    # 清空绘图目录
    clear_plot_directory()
    
    # 确认工作目录（脚本所在目录）
    script_dir = Path(__file__).parent.resolve()
    print(f"\n工作目录: {script_dir}")
    
    # 切换到脚本所在目录
    os.chdir(script_dir)
    
    total_start = time.time()
    success_count = 0
    total_count = 8
    
    # 数据集配置：(数据集名称, 脚本路径, 选项列表)
    # 使用相对于当前目录的路径
    datasets = [
        ("UCI", "generate_Uci.py", ["1", "2", "3", "4"]),
        ("Xinwang", "generate_Xinwang.py", ["1", "2", "3", "4"])
    ]
    
    # 依次生成每个数据集的每种模式
    for dataset_name, script_path, choices in datasets:
        for choice in choices:
            if run_dataset_generation(dataset_name, choice, script_path):
                success_count += 1
            # 短暂延迟，避免资源冲突
            time.sleep(1)
    
    # 总结
    total_elapsed = time.time() - total_start
    print("\n" + "="*80)
    print("                        生成完成统计")
    print("="*80)
    print(f"成功: {success_count}/{total_count}")
    print(f"失败: {total_count - success_count}/{total_count}")
    print(f"总耗时: {total_elapsed:.1f}秒 ({total_elapsed/60:.1f}分钟)")
    print("="*80)
    
    if success_count == total_count:
        print("✓ 所有数据集生成成功！")
        print("\n生成的数据保存在以下目录：")
        print("  - dataset/Uci/feature/")
        print("  - dataset/Uci/label/")
        print("  - dataset/Uci/quantity/")
        print("  - dataset/Uci/iid/")
        print("  - dataset/Xinwang/feature/")
        print("  - dataset/Xinwang/label/")
        print("  - dataset/Xinwang/quantity/")
        print("  - dataset/Xinwang/iid/")
        
        # 生成分布图
        generate_plots()
        
    else:
        print(f"⚠ 有{total_count - success_count}个数据集生成失败，请检查日志")
        sys.exit(1)

if __name__ == "__main__":
    print("\n开始执行数据集批量生成...")
    print("此过程可能需要几分钟时间，请耐心等待...\n")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断执行")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
