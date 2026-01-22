"""
Xinwang信用数据集生成脚本 - 支持特征异质性和标签异质性

特点：
1. 使用toad进行特征选择（100维 → 38维）
2. 两种异质性模式：特征异质性和标签异质性
3. 标签异质性使用Dirichlet(alpha=0.02)实现高异质性
4. 确保每个客户端至少有min_minority_samples个少数类样本

作者：Assistant
日期：2025-12-21
"""

import numpy as np
import pandas as pd
import os
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import dirichlet
from scipy.special import kl_div
import toad

def calculate_kl_divergence(p, q, epsilon=1e-10):
    """计算KL散度 KL(P||Q)"""
    p = np.array(p) + epsilon
    q = np.array(q) + epsilon
    p = p / p.sum()
    q = q / q.sum()
    return np.sum(kl_div(p, q))

def apply_toad_feature_selection(df, target_col='target'):
    """
    使用toad进行特征选择
    
    Args:
        df: 原始数据框（包含target列）
        target_col: 目标列名
    
    Returns:
        selected_df: 特征选择后的数据框
        selected_features: 选择的特征列表
    """
    print("\n" + "="*70)
    print("使用TOAD进行特征选择")
    print("="*70)
    
    y = df[target_col]
    
    print(f"原始特征维度: {df.shape[1] - 1}")
    
    try:
        # 使用toad.selection.select进行特征选择
        # iv=0.05: 删除IV值<0.05的特征
        # corr=0.95: 删除相关性>0.95的特征
        # empty=0.9: 删除缺失率>0.9的特征
        data_selected, dropped = toad.selection.select(
            df, 
            target=y, 
            iv=0.05, 
            corr=0.95, 
            empty=0.9, 
            return_drop=True
        )
        
        selected_features = [col for col in data_selected.columns if col != target_col]
        
        print(f"\n特征选择结果:")
        print(f"  选择特征数: {len(selected_features)}")
        print(f"  删除特征数: {len(dropped)}")
        if len(dropped) > 0:
            print(f"  删除的特征: {list(dropped.keys())[:10]}{'...' if len(dropped) > 10 else ''}")
        
        return data_selected, selected_features
        
    except Exception as e:
        print(f"\n警告：特征选择失败 ({e})")
        print("使用所有原始特征")
        selected_features = [col for col in df.columns if col != target_col]
        return df, selected_features

def dirichlet_partition_high_heterogeneity(labels, num_clients=10, alpha=0.005, min_samples=100, min_minority_samples=20, min_majority_samples=50):
    """
    使用Dirichlet分布实现高异质性标签分配，确保每个客户端正负样本都有
    
    Args:
        labels: 标签数组
        num_clients: 客户端数量
        alpha: Dirichlet参数（越小异质性越高，0.01为极高异质性）
        min_samples: 每个客户端最小样本数
        min_minority_samples: 每个客户端最小少数类（类1）样本数
        min_majority_samples: 每个客户端最小多数类（类0）样本数
    
    Returns:
        client_indices: 列表，每个元素是一个客户端的样本索引
    """
    num_classes = len(np.unique(labels))
    num_samples = len(labels)
    
    # 按类别分组样本索引
    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]
    
    # 为每个客户端生成类别分布（Dirichlet采样）
    client_indices = [[] for _ in range(num_clients)]
    
    for c in range(num_classes):
        # 对当前类别，使用Dirichlet生成客户端分配比例
        proportions = dirichlet.rvs([alpha] * num_clients)[0]
        
        # 限制极端值：任何客户端不能拿超过60%的样本
        max_prop = 0.6
        while proportions.max() > max_prop:
            # 重新采样
            proportions = dirichlet.rvs([alpha] * num_clients)[0]
        
        proportions = (proportions * len(class_indices[c])).astype(int)
        
        # 确保总数匹配
        diff = len(class_indices[c]) - proportions.sum()
        proportions[np.argmax(proportions)] += diff
        
        # 确保每个客户端至少有最小样本数（正负样本都要有）
        if c == 0:  # 多数类（负样本）
            min_count = min_majority_samples
        else:  # 少数类（正样本）
            min_count = min_minority_samples
        
        for i in range(num_clients):
            if proportions[i] < min_count:
                proportions[i] = min_count
        
        # 重新归一化
        total_needed = proportions.sum()
        if total_needed > len(class_indices[c]):
            # 按比例缩减
            scale = len(class_indices[c]) / total_needed
            proportions = (proportions * scale).astype(int)
            # 确保最小值
            for i in range(num_clients):
                if proportions[i] < min_count:
                    proportions[i] = min_count
            diff = len(class_indices[c]) - proportions.sum()
            proportions[np.argmax(proportions)] += diff
        
        # 分配样本到客户端
        shuffled_indices = np.random.permutation(class_indices[c])
        start_idx = 0
        for client_id in range(num_clients):
            end_idx = start_idx + proportions[client_id]
            client_indices[client_id].extend(shuffled_indices[start_idx:end_idx])
            start_idx = end_idx
    
    # 确保每个客户端有最小样本数
    for client_id in range(num_clients):
        if len(client_indices[client_id]) < min_samples:
            # 从其他客户端借样本
            for other_id in range(num_clients):
                if len(client_indices[other_id]) > min_samples + 100:
                    borrow_count = min_samples - len(client_indices[client_id])
                    borrowed = client_indices[other_id][:borrow_count]
                    client_indices[client_id].extend(borrowed)
                    client_indices[other_id] = client_indices[other_id][borrow_count:]
                    break
    
    return client_indices

def generate_iid_uniform(X, y, num_clients=10):
    """
    生成IID均匀分布数据：特征和标签分布都相同
    
    Args:
        X: 特征矩阵
        y: 标签数组
        num_clients: 客户端数量
    
    Returns:
        client_data: 字典，每个客户端的数据
    """
    print(f"\n{'='*70}")
    print("模式3: IID均匀分布 (IID Uniform Distribution)")
    print(f"{'='*70}")
    print("特征分布：相同（无偏移）")
    print("标签分布：相同（均匀随机分配）")
    
    # 随机打乱样本
    num_samples = len(y)
    indices = np.random.permutation(num_samples)
    samples_per_client = num_samples // num_clients
    
    client_data = {}
    global_label_dist = np.bincount(y) / len(y)
    kl_divergences = []
    
    print(f"\n{'='*70}")
    print("客户端数据分布统计")
    print(f"{'='*70}")
    
    for client_id in range(num_clients):
        start_idx = client_id * samples_per_client
        end_idx = start_idx + samples_per_client if client_id < num_clients - 1 else num_samples
        client_indices = indices[start_idx:end_idx]
        
        X_client = X[client_indices]
        y_client = y[client_indices]
        
        client_data[client_id] = {
            'X': X_client,
            'y': y_client
        }
        
        # 计算标签分布（应该与全局分布接近）
        client_label_dist = np.bincount(y_client, minlength=len(global_label_dist)) / len(y_client)
        kl = calculate_kl_divergence(client_label_dist, global_label_dist)
        kl_divergences.append(kl)
        
        print(f"客户端{client_id}: 样本={len(y_client)}, "
              f"类0={np.sum(y_client==0)}, 类1={np.sum(y_client==1)}, "
              f"类1比例={np.mean(y_client):.2%}, KL={kl:.4f}")
    
    avg_kl = np.mean(kl_divergences)
    print(f"\n平均KL散度: {avg_kl:.4f}")
    if avg_kl < 0.01:
        print("✓ IID均匀分布 (KL散度 < 0.01)")
    elif avg_kl < 0.05:
        print("✓ 近似均匀分布 (KL散度 < 0.05)")
    else:
        print("⚠ 分布略有偏差")
    
    return client_data, avg_kl

def auto_select_clusters(X, num_clients, max_iter=10):
    """
    使用轮廓系数自动选择最优簇数
    
    Args:
        X: 特征矩阵
        num_clients: 客户端数量
        max_iter: 最大迭代次数
    
    Returns:
        best_k: 最优簇数
        best_score: 最优轮廓系数
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    num_samples = len(X)
    k_min = max(10, num_clients)
    k_max = min(30, num_samples // 200)  # 降低上限，加快速度
    k_range = range(k_min, k_max + 1, 5)  # 增大步长
    
    best_k = k_min
    best_score = -1
    
    print(f"\n正在选择最优簇数（范围：{k_min}-{k_max}，步长5）...")
    
    for k in k_range:
        print(f"  尝试 k={k}...", end='')
        kmeans = KMeans(n_clusters=k, random_state=42, max_iter=100, n_init=3)
        clusters = kmeans.fit_predict(X)
        score = silhouette_score(X, clusters, sample_size=min(3000, num_samples))
        print(f" 轮廓系数={score:.4f}")
        
        if score > best_score:
            best_score = score
            best_k = k
    
    print(f"✓ 最优簇数: {best_k}, 轮廓系数: {best_score:.4f}")
    return best_k, best_score

def dirichlet_partition_by_clusters(X, y, clusters, num_clients=10, alpha=0.01):
    """
    基于聚类结果将整个簇分配给客户端（而不是分割每个簇）
    
    策略：使用Dirichlet分配将每个簇整体分配给某个客户端
    这样每个客户端只拥有少数几个簇的数据，特征分布会有显著差异
    
    Args:
        X: 特征矩阵
        y: 标签数组
        clusters: 聚类标签
        num_clients: 客户端数量
        alpha: Dirichlet参数（越小簇分配越不均匀，异质性越高）
    
    Returns:
        client_indices: 每个客户端的样本索引列表
    """
    num_clusters = len(np.unique(clusters))
    client_indices = [[] for _ in range(num_clients)]
    
    # 将每个簇整体分配给一个客户端
    for cluster_id in range(num_clusters):
        cluster_samples = np.where(clusters == cluster_id)[0]
        
        if len(cluster_samples) == 0:
            continue
        
        # 用Dirichlet采样决定这个簇分配给哪个客户端
        # alpha越小，簇分配越集中（某个客户端会得到这整个簇）
        proportions = dirichlet.rvs([alpha] * num_clients)[0]
        
        # 将簇分配给概率最高的客户端
        selected_client = np.argmax(proportions)
        client_indices[selected_client].extend(cluster_samples.tolist())
    
    return client_indices

def compute_wasserstein_heterogeneity(X_global, client_data_list):
    """
    计算特征异质性的Wasserstein距离
    
    Args:
        X_global: 全局特征矩阵
        client_data_list: 客户端特征矩阵列表
    
    Returns:
        avg_wasserstein: 平均归一化Wasserstein距离
        feature_wasserstein: 每个特征的详细距离
    """
    from scipy.stats import wasserstein_distance
    
    num_features = X_global.shape[1]
    num_clients = len(client_data_list)
    
    feature_wasserstein = []
    
    for feat_idx in range(num_features):
        global_dist = X_global[:, feat_idx]
        feat_range = global_dist.max() - global_dist.min()
        
        if feat_range == 0:
            feature_wasserstein.append(0)
            continue
        
        client_dists = []
        for X_client in client_data_list:
            if len(X_client) == 0:  # 跳过空客户端
                continue
            client_dist = X_client[:, feat_idx]
            w_dist = wasserstein_distance(global_dist, client_dist)
            w_dist_norm = w_dist / feat_range
            client_dists.append(w_dist_norm)
        
        if len(client_dists) > 0:
            feature_wasserstein.append(np.mean(client_dists))
        else:
            feature_wasserstein.append(0)
    
    avg_wasserstein = np.mean(feature_wasserstein)
    
    return avg_wasserstein, feature_wasserstein

def generate_feature_heterogeneity_manual(X_raw, y, num_clients=10, random_state=42):
    """
    基于特征方差范围手动分区生成特征异质性（Xinwang脱敏数据）
    
    方法：选择方差最大的2个特征，按值域范围划分客户端
    - 自动选择方差最大的2个特征
    - 划分为5×2=10个区域
    
    Args:
        X_raw: 原始特征矩阵（标准化前）
        y: 标签数组
        num_clients: 客户端数量（默认10）
        random_state: 随机种子
    
    Returns:
        client_data: 字典，每个客户端的数据
        avg_wasserstein: 平均Wasserstein距离
    """
    from sklearn.preprocessing import StandardScaler
    
    np.random.seed(random_state)
    
    print(f"\n{'='*70}")
    print("模式1: 特征异质性 (Feature Heterogeneity - Manual Partitioning)")
    print(f"{'='*70}")
    print(f"方法: 基于方差最大特征手动分区 (2D划分)")
    print(f"  自动选择方差最大的2个特征进行划分")
    print(f"  区域数: 5×2 = 10个，对应10个客户端")
    print(f"随机种子: {random_state}")
    
    # Step 1: 选择方差最大的2个特征
    print(f"\nStep 1: 选择方差最大的2个特征...")
    feature_vars = np.var(X_raw, axis=0)
    top2_indices = np.argsort(feature_vars)[-2:][::-1]  # 降序
    
    feat1_idx, feat2_idx = top2_indices[0], top2_indices[1]
    feat1 = X_raw[:, feat1_idx]
    feat2 = X_raw[:, feat2_idx]
    
    print(f"  特征{feat1_idx}: 方差={feature_vars[feat1_idx]:.4f}, 范围=[{feat1.min():.2f}, {feat1.max():.2f}]")
    print(f"  特征{feat2_idx}: 方差={feature_vars[feat2_idx]:.4f}, 范围=[{feat2.min():.2f}, {feat2.max():.2f}]")
    
    # 特征1: 5档划分（0%, 20%, 40%, 60%, 80%, 100%）
    feat1_percentiles = np.percentile(feat1, [0, 20, 40, 60, 80, 100])
    feat1_groups = np.digitize(feat1, feat1_percentiles[1:-1])  # 0-4共5组
    
    # 特征2: 2档划分（0%, 50%, 100%）
    feat2_percentiles = np.percentile(feat2, [0, 50, 100])
    feat2_groups = np.digitize(feat2, feat2_percentiles[1:-1])  # 0-1共2组
    
    print(f"  特征{feat1_idx}分位数: {feat1_percentiles}")
    print(f"  特征{feat2_idx}分位数: {feat2_percentiles}")
    
    # 组合成客户端ID
    client_assignment = feat1_groups * 2 + feat2_groups
    
    # 统计每个客户端的样本数
    client_counts = np.bincount(client_assignment, minlength=num_clients)
    print(f"✓ 划分完成，各客户端初始样本数: {client_counts.tolist()}")
    print(f"  样本数范围: [{client_counts.min()}, {client_counts.max()}]")
    
    # Step 2: 构建客户端索引
    print(f"\nStep 2: 分配样本到客户端...")
    client_indices = [[] for _ in range(num_clients)]
    
    for idx in range(len(y)):
        client_id = client_assignment[idx]
        if client_id < num_clients:
            client_indices[client_id].append(idx)
    
    # Step 2.5: 确保每个客户端都有正负样本
    print(f"\nStep 2.5: 确保每个客户端都有正负样本...")
    global_pos_indices = np.where(y == 1)[0]
    global_neg_indices = np.where(y == 0)[0]
    
    for client_id in range(num_clients):
        if len(client_indices[client_id]) == 0:
            continue
        
        indices = np.array(client_indices[client_id])
        y_client = y[indices]
        n_pos = np.sum(y_client == 1)
        n_neg = np.sum(y_client == 0)
        
        if n_pos < 10:
            need_pos = 10 - n_pos
            available_pos = np.setdiff1d(global_pos_indices, indices)
            if len(available_pos) >= need_pos:
                add_pos = np.random.choice(available_pos, need_pos, replace=False)
                client_indices[client_id].extend(add_pos.tolist())
                print(f"  客户端{client_id}: 添加{need_pos}个正样本")
        
        if n_neg < 10:
            need_neg = 10 - n_neg
            available_neg = np.setdiff1d(global_neg_indices, indices)
            if len(available_neg) >= need_neg:
                add_neg = np.random.choice(available_neg, need_neg, replace=False)
                client_indices[client_id].extend(add_neg.tolist())
                print(f"  客户端{client_id}: 添加{need_neg}个负样本")
    
    # Step 3: 标准化数据
    print(f"\nStep 3: 全局标准化数据...")
    from sklearn.preprocessing import StandardScaler
    global_scaler = StandardScaler()
    X_global_scaled = global_scaler.fit_transform(X_raw)
    
    client_data = {}
    client_X_list = []
    global_label_dist = np.bincount(y) / len(y)
    
    print(f"\n{'='*70}")
    print("客户端数据分布统计")
    print(f"{'='*70}")
    
    for client_id in range(num_clients):
        indices = client_indices[client_id]
        
        if len(indices) == 0:
            print(f"⚠ 警告：客户端{client_id}没有分配到数据")
            continue
        
        X_client = X_global_scaled[indices]
        y_client = y[indices]
        
        client_data[client_id] = {
            'X': X_client,
            'y': y_client
        }
        client_X_list.append(X_client)
        
        # 标签分布统计
        client_label_dist = np.bincount(y_client, minlength=len(global_label_dist)) / len(y_client)
        label_kl = calculate_kl_divergence(client_label_dist, global_label_dist)
        
        # 计算该客户端的特征范围
        feat1_group = client_id // 2
        feat2_group = client_id % 2
        feat1_desc = ["极低", "低", "中", "高", "极高"][feat1_group]
        feat2_desc = ["低", "高"][feat2_group]
        
        print(f"客户端{client_id}: 样本={len(y_client)}, 特征组=[{feat1_desc}F{feat1_idx}+{feat2_desc}F{feat2_idx}], "
              f"类0={np.sum(y_client==0)}, 类1={np.sum(y_client==1)}, "
              f"类1比例={np.mean(y_client):.2%}, 标签KL={label_kl:.4f}")
    
    # Step 4: 计算特征异质性
    print(f"\n{'='*70}")
    print("特征异质性评估 (Wasserstein距离)")
    print(f"{'='*70}")
    
    avg_wasserstein, feature_wasserstein = compute_wasserstein_heterogeneity(X_global_scaled, client_X_list)
    
    print(f"平均Wasserstein距离: {avg_wasserstein:.4f}")
    if avg_wasserstein >= 0.3:
        print("✓ 高特征异质性 (Wasserstein ≥ 0.3)")
    elif avg_wasserstein >= 0.1:
        print("✓ 中等特征异质性 (0.1 ≤ Wasserstein < 0.3)")
    else:
        print("✓ 低特征异质性 (Wasserstein < 0.1)")
    
    print(f"\n前5个特征的Wasserstein距离:")
    for i in range(min(5, len(feature_wasserstein))):
        print(f"  特征{i}: {feature_wasserstein[i]:.4f}")
    
    return client_data, avg_wasserstein

def generate_feature_heterogeneity(X, y, num_clients=10, alpha=0.01, K=None, random_state=42, samples_per_client=1788):
    """
    生成特征异质性数据：每个客户端特征分布不同，标签分布接近IID
    
    方法：K-means聚类(自动K) + Dirichlet分配 + 重采样
    - 使用Elbow方法自动确定最佳聚类数K
    - 为每个客户端使用Dirichlet分布分配簇权重（α=0.01，高异质性）
    - 根据权重从分配的簇中重采样，使每个客户端样本数相同
    - 确保每个客户端都有正负样本
    
    Args:
        X: 特征矩阵
        y: 标签数组
        num_clients: 客户端数量（默认10）
        alpha: Dirichlet参数（默认0.01，控制簇权重分布）
        K: 聚类簇数（None则自动确定）
        random_state: 随机种子（默认42）
        samples_per_client: 每个客户端的样本数（默认1788）
    
    Returns:
        client_data: 字典，每个客户端的数据
        avg_wasserstein: 平均Wasserstein距离
    """
    from sklearn.cluster import KMeans
    from scipy.stats import dirichlet
    
    # 设置随机种子
    np.random.seed(random_state)
    
    print(f"\n{'='*70}")
    print("模式1: 特征异质性 (Feature Heterogeneity)")
    print(f"{'='*70}")
    print(f"方法: K-means聚类(自动K) + Dirichlet分配(α={alpha}) + 重采样")
    print(f"随机种子: {random_state}")
    print(f"目标: 每个客户端{samples_per_client}个样本")
    
    # Step 0: 使用Elbow方法自动确定最佳K值
    if K is None:
        print(f"\nStep 0: 使用Elbow方法自动确定最佳聚类数K...")
        inertias = []
        K_range = range(max(2, num_clients), min(100, len(X) // 100) + 1, 5)  # 测试范围
        print(f"  测试K值范围: {list(K_range)}")
        
        for k in K_range:
            kmeans_temp = KMeans(n_clusters=k, random_state=random_state, max_iter=100, n_init=3)
            kmeans_temp.fit(X)
            inertias.append(kmeans_temp.inertia_)
        
        # 计算拐点（使用二阶差分）
        if len(inertias) >= 3:
            diffs = np.diff(inertias)
            diffs2 = np.diff(diffs)
            elbow_idx = np.argmax(diffs2) + 1  # +1因为二阶差分少了2个元素
            K = list(K_range)[min(elbow_idx, len(K_range)-1)]
        else:
            K = num_clients * 4  # 默认4倍客户端数
        
        print(f"  ✓ 自动确定最佳K值: {K}")
    
    # Step 1: K-means聚类
    print(f"\nStep 1: K-means聚类...")
    print(f"  使用簇数: K={K} (自动确定，大于客户端数{num_clients})")
    
    kmeans = KMeans(n_clusters=K, random_state=random_state, max_iter=300, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    print(f"✓ 聚类完成，共{K}个簇")
    
    # 统计每个簇的样本数
    cluster_counts = np.bincount(clusters)
    print(f"  簇大小范围: [{cluster_counts.min()}, {cluster_counts.max()}]")
    print(f"  平均簇大小: {cluster_counts.mean():.1f}")
    
    # Step 2: Dirichlet分配簇权重给每个客户端
    print(f"\nStep 2: Dirichlet分配簇权重给每个客户端...")
    print(f"  Dirichlet α={alpha}")
    
    client_cluster_weights = []
    for client_id in range(num_clients):
        weights = dirichlet.rvs([alpha] * K, random_state=random_state + client_id)[0]
        client_cluster_weights.append(weights)
        
        # 显示主要簇（权重前5）
        top_clusters = np.argsort(weights)[-5:][::-1]
        top_weights = weights[top_clusters]
        print(f"  客户端{client_id}主要簇(top 5): {top_clusters.tolist()}, 权重和: {top_weights.sum():.3f}")
    
    # Step 3: 根据权重为每个客户端重采样固定数量的样本
    print(f"\nStep 3: 重采样为每个客户端{samples_per_client}个样本...")
    
    client_indices = []
    global_pos_indices = np.where(y == 1)[0]
    global_neg_indices = np.where(y == 0)[0]
    
    for client_id in range(num_clients):
        weights = client_cluster_weights[client_id]
        
        # 收集所有可用样本及其权重
        available_indices = []
        available_weights = []
        
        for cluster_id in range(K):
            cluster_samples = np.where(clusters == cluster_id)[0]
            available_indices.extend(cluster_samples.tolist())
            available_weights.extend([weights[cluster_id]] * len(cluster_samples))
        
        # 归一化权重
        available_weights = np.array(available_weights)
        available_weights = available_weights / available_weights.sum()
        
        # 根据权重采样samples_per_client个样本（允许重复）
        sampled_indices = np.random.choice(
            available_indices, 
            size=samples_per_client, 
            replace=True,  # 允许重复采样
            p=available_weights
        )
        
        # 确保有正负样本
        y_sampled = y[sampled_indices]
        n_pos = np.sum(y_sampled == 1)
        n_neg = np.sum(y_sampled == 0)
        
        min_pos = int(samples_per_client * 0.05)  # 至少5%正样本
        min_neg = int(samples_per_client * 0.05)  # 至少5%负样本
        
        if n_pos < min_pos:
            # 替换一些负样本为正样本
            neg_positions = np.where(y_sampled == 0)[0]
            replace_count = min(min_pos - n_pos, len(neg_positions))
            replace_positions = np.random.choice(neg_positions, replace_count, replace=False)
            
            for pos in replace_positions:
                sampled_indices[pos] = np.random.choice(global_pos_indices)
        
        if n_neg < min_neg:
            # 替换一些正样本为负样本
            pos_positions = np.where(y_sampled == 1)[0]
            replace_count = min(min_neg - n_neg, len(pos_positions))
            replace_positions = np.random.choice(pos_positions, replace_count, replace=False)
            
            for pos in replace_positions:
                sampled_indices[pos] = np.random.choice(global_neg_indices)
        
        client_indices.append(sampled_indices.tolist())
    
    print(f"✓ 重采样完成，每个客户端样本数: {[len(indices) for indices in client_indices]}")
    
    # Step 4: 构建客户端数据
    client_data = {}
    client_X_list = []
    global_label_dist = np.bincount(y) / len(y)
    
    print(f"\n{'='*70}")
    print("客户端数据分布统计")
    print(f"{'='*70}")
    
    for client_id in range(num_clients):
        indices = client_indices[client_id]
        
        X_client = X[indices]
        y_client = y[indices]
        
        client_data[client_id] = {
            'X': X_client,
            'y': y_client
        }
        client_X_list.append(X_client)
        
        # 标签分布统计
        client_label_dist = np.bincount(y_client, minlength=len(global_label_dist)) / len(y_client)
        label_kl = calculate_kl_divergence(client_label_dist, global_label_dist)
        
        print(f"客户端{client_id}: 样本={len(y_client)}, "
              f"类0={np.sum(y_client==0)}, 类1={np.sum(y_client==1)}, "
              f"类1比例={np.mean(y_client):.2%}, 标签KL={label_kl:.4f}")
    
    # Step 5: 计算特征异质性
    print(f"\n{'='*70}")
    print("特征异质性评估 (Wasserstein距离)")
    print(f"{'='*70}")
    
    avg_wasserstein, feature_wasserstein = compute_wasserstein_heterogeneity(X, client_X_list)
    
    print(f"平均Wasserstein距离: {avg_wasserstein:.4f}")
    if avg_wasserstein >= 0.5:
        print("✓ 高特征异质性 (Wasserstein ≥ 0.5)")
    elif avg_wasserstein >= 0.3:
        print("✓ 中高特征异质性 (0.3 ≤ Wasserstein < 0.5)")
    elif avg_wasserstein >= 0.1:
        print("✓ 中等特征异质性 (0.1 ≤ Wasserstein < 0.3)")
    else:
        print("✓ 低特征异质性 (Wasserstein < 0.1)")
    
    # 显示前5个特征的详细距离
    print(f"\n前5个特征的Wasserstein距离:")
    for i in range(min(5, len(feature_wasserstein))):
        print(f"  特征{i}: {feature_wasserstein[i]:.4f}")
    
    return client_data, avg_wasserstein

def generate_label_heterogeneity(X, y, num_clients=10, alpha=0.001, min_samples=800, min_minority_samples=30, min_majority_samples=100, random_state=42):
    """
    生成标签异质性数据：特征分布相同（IID），标签分布不同（Non-IID）
    
    Args:
        X: 特征矩阵
        y: 标签数组
        num_clients: 客户端数量
        alpha: Dirichlet参数（越小异质性越高，0.001为极高异质性，默认0.001）
        min_samples: 每个客户端最小样本数
        min_minority_samples: 每个客户端最小正样本（类1）数
        min_majority_samples: 每个客户端最小负样本（类0）数
        random_state: 随机种子（默认42）
    
    Returns:
        client_data: 字典，每个客户端的数据
    """
    # 设置随机种子
    np.random.seed(random_state)
    print(f"\n{'='*70}")
    print("模式2: 标签异质性 (Label Heterogeneity - 极高异质性)")
    print(f"{'='*70}")
    print(f"Dirichlet Alpha: {alpha} (越小异质性越高)")
    print(f"每个客户端最少正样本: {min_minority_samples}, 最少负样本: {min_majority_samples}")
    print(f"随机种子: {random_state}")
    
    # Non-IID标签分配（使用Dirichlet实现极高异质性）
    client_indices = dirichlet_partition_high_heterogeneity(y, num_clients, alpha, min_samples, min_minority_samples, min_majority_samples)
    
    client_data = {}
    global_label_dist = np.bincount(y) / len(y)
    kl_divergences = []
    
    print(f"\n{'='*70}")
    print("客户端数据分布统计")
    print(f"{'='*70}")
    
    for client_id in range(num_clients):
        indices = client_indices[client_id]
        X_client = X[indices]
        y_client = y[indices]
        
        client_data[client_id] = {
            'X': X_client,
            'y': y_client
        }
        
        # 计算标签异质性（KL散度）
        client_label_dist = np.bincount(y_client, minlength=len(global_label_dist)) / len(y_client)
        kl = calculate_kl_divergence(client_label_dist, global_label_dist)
        kl_divergences.append(kl)
        
        print(f"客户端{client_id}: 样本={len(y_client)}, "
              f"类0={np.sum(y_client==0)}, 类1={np.sum(y_client==1)}, "
              f"类1比例={np.mean(y_client):.2%}, KL={kl:.4f}")
    
    avg_kl = np.mean(kl_divergences)
    print(f"\n平均KL散度: {avg_kl:.4f}")
    if avg_kl >= 0.5:
        print("✓ 高异质性 (KL散度 >= 0.5)")
    elif avg_kl >= 0.3:
        print("✓ 中高异质性 (0.3 <= KL散度 < 0.5)")
    elif avg_kl >= 0.1:
        print("✓ 中等异质性 (0.1 <= KL散度 < 0.3)")
    else:
        print("✗ 低异质性 (KL散度 < 0.1)")
    
    return client_data, avg_kl

def generate_quantity_heterogeneity(X, y, num_clients=10, alpha=0.1, random_state=42, min_samples_per_client=100):
    """
    生成样本数量异质性数据：每个客户端样本数量不同，但特征和标签分布接近IID
    
    方法：使用Dirichlet分布控制样本数量分配，同时保持正负样本比例一致
    
    Args:
        X: 特征矩阵
        y: 标签数组
        num_clients: 客户端数量
        alpha: Dirichlet参数（越小数量异质性越高，推荐0.1，默认0.1）
        random_state: 随机种子（默认42）
        min_samples_per_client: 每个客户端最小样本数（默认100）
    
    Returns:
        client_data: 字典，每个客户端的数据
        sample_counts: 每个客户端的样本数
    """
    # 设置随机种子
    np.random.seed(random_state)
    
    print(f"\n{'='*70}")
    print("模式3: 样本数量异质性 (Quantity Heterogeneity)")
    print(f"{'='*70}")
    print(f"Dirichlet Alpha: {alpha} (越小数量差异越大)")
    print(f"策略: 使用Dirichlet分配样本数量，保持正负样本比例一致")
    print(f"随机种子: {random_state}")
    print(f"每个客户端最小样本数: {min_samples_per_client}")
    
    num_samples = len(y)
    global_positive_ratio = np.mean(y)  # 全局正样本比例
    
    # 分别处理正样本和负样本
    positive_indices = np.where(y == 1)[0]
    negative_indices = np.where(y == 0)[0]
    
    # 打乱顺序
    np.random.shuffle(positive_indices)
    np.random.shuffle(negative_indices)
    
    # 使用Dirichlet采样生成样本数量分配比例
    proportions = dirichlet.rvs([alpha] * num_clients, random_state=random_state)[0]
    sample_counts = (proportions * num_samples).astype(int)
    
    # 确保总数匹配
    diff = num_samples - sample_counts.sum()
    sample_counts[np.argmax(sample_counts)] += diff
    
    # 确保每个客户端至少有指定的最小样本数
    min_samples = min_samples_per_client
    for i in range(num_clients):
        if sample_counts[i] < min_samples:
            sample_counts[i] = min_samples
    
    # 重新归一化
    total_needed = sample_counts.sum()
    if total_needed > num_samples:
        scale = num_samples / total_needed
        sample_counts = (sample_counts * scale).astype(int)
        for i in range(num_clients):
            if sample_counts[i] < min_samples:
                sample_counts[i] = min_samples
        diff = num_samples - sample_counts.sum()
        sample_counts[np.argmax(sample_counts)] += diff
    
    print(f"\nStep 1: 样本数量分配")
    print(f"  总样本数: {num_samples}")
    print(f"  全局正样本比例: {global_positive_ratio:.2%}")
    print(f"  样本数量: {sample_counts}")
    print(f"  样本数量CV: {np.std(sample_counts)/np.mean(sample_counts):.3f}")
    
    # 为每个客户端分配样本（保持正负比例）
    client_data = {}
    global_label_dist = np.bincount(y) / len(y)
    positive_ratios = []
    kl_divergences = []
    
    pos_start = 0
    neg_start = 0
    
    print(f"\n{'='*70}")
    print("客户端数据分布统计")
    print(f"{'='*70}")
    
    for client_id in range(num_clients):
        n_samples = sample_counts[client_id]
        
        # 按全局比例分配正负样本
        n_positive = int(n_samples * global_positive_ratio)
        n_negative = n_samples - n_positive
        
        # 分配正样本
        pos_end = pos_start + n_positive
        if pos_end > len(positive_indices):
            # 循环使用
            pos_indices = np.concatenate([
                positive_indices[pos_start:],
                positive_indices[:pos_end - len(positive_indices)]
            ])
            pos_start = pos_end - len(positive_indices)
        else:
            pos_indices = positive_indices[pos_start:pos_end]
            pos_start = pos_end
        
        # 分配负样本
        neg_end = neg_start + n_negative
        if neg_end > len(negative_indices):
            # 循环使用
            neg_indices = np.concatenate([
                negative_indices[neg_start:],
                negative_indices[:neg_end - len(negative_indices)]
            ])
            neg_start = neg_end - len(negative_indices)
        else:
            neg_indices = negative_indices[neg_start:neg_end]
            neg_start = neg_end
        
        # 合并正负样本
        client_indices = np.concatenate([pos_indices, neg_indices])
        np.random.shuffle(client_indices)
        
        X_client = X[client_indices]
        y_client = y[client_indices]
        
        client_data[client_id] = {
            'X': X_client,
            'y': y_client
        }
        
        # 统计正样本比例
        client_positive_ratio = np.mean(y_client)
        positive_ratios.append(client_positive_ratio)
        
        # 计算标签分布KL散度（应该很小）
        client_label_dist = np.bincount(y_client, minlength=len(global_label_dist)) / len(y_client)
        kl = calculate_kl_divergence(client_label_dist, global_label_dist)
        kl_divergences.append(kl)
        
        print(f"客户端{client_id}: 样本={len(y_client)}, "
              f"类0={np.sum(y_client==0)}, 类1={np.sum(y_client==1)}, "
              f"类1比例={client_positive_ratio:.2%}, KL={kl:.4f}")
    
    # 检查正样本比例的一致性
    ratio_std = np.std(positive_ratios)
    ratio_mean = np.mean(positive_ratios)
    ratio_cv = ratio_std / ratio_mean if ratio_mean > 0 else 0
    avg_kl = np.mean(kl_divergences)
    
    print(f"\n{'='*70}")
    print("异质性评估")
    print(f"{'='*70}")
    print(f"样本数量CV: {np.std(sample_counts)/np.mean(sample_counts):.3f}")
    print(f"正样本比例标准差: {ratio_std:.4f}")
    print(f"正样本比例CV: {ratio_cv:.4f}")
    print(f"平均标签KL散度: {avg_kl:.4f}")
    
    if ratio_std < 0.001:
        print("✓ 正负样本比例高度一致 (σ < 0.001)")
    elif ratio_std < 0.01:
        print("✓ 正负样本比例基本一致 (σ < 0.01)")
    else:
        print("⚠ 正负样本比例有偏差")
    
    if avg_kl < 0.01:
        print("✓ 标签分布接近IID (KL < 0.01)")
    elif avg_kl < 0.05:
        print("✓ 标签分布近似IID (KL < 0.05)")
    else:
        print("⚠ 标签分布有偏差")
    
    return client_data, sample_counts

def save_to_npz(client_data, dataset_name='Xinwang', heterogeneity_type='feature', train_ratio=0.75):
    """保存数据到npz格式，根据异质性类型创建不同的文件夹"""
    base_dir = f'dataset/{dataset_name}/{heterogeneity_type}'
    train_path = os.path.join(base_dir, 'train')
    test_path = os.path.join(base_dir, 'test')
    
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    
    print(f"\n保存数据到 {dataset_name}/{heterogeneity_type}...")
    
    for client_id, data in client_data.items():
        X = data['X']
        y = data['y']
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_ratio, random_state=42
        )
        
        # 保存训练集
        np.savez_compressed(
            os.path.join(train_path, f'{client_id}.npz'),
            data={'x': X_train, 'y': y_train}
        )
        
        # 保存测试集
        np.savez_compressed(
            os.path.join(test_path, f'{client_id}.npz'),
            data={'x': X_test, 'y': y_test}
        )
        
        print(f"  客户端{client_id}: 训练={len(y_train)}, 测试={len(y_test)}")
    
    print(f"✓ {dataset_name}/{heterogeneity_type}数据保存完成")

def save_config(num_clients, num_classes, feature_dim, heterogeneity_type, dataset_name='Xinwang'):
    """保存配置文件"""
    config = {
        'num_clients': num_clients,
        'num_classes': num_classes,
        'non_iid': heterogeneity_type == 'label',
        'balance': heterogeneity_type == 'feature',
        'partition': 'dirichlet' if heterogeneity_type == 'label' else 'feature_shift',
        'feature_dim': feature_dim,
        'batch_size': 10,
        'heterogeneity_type': heterogeneity_type
    }
    
    config_path = f'dataset/{dataset_name}/config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ 配置文件已保存: {config_path}")

def main():
    """主函数"""
    print("\n" + "="*70)
    print("Xinwang信用数据集生成 (含TOAD特征选择)".center(70))
    print("="*70)
    
    # 选择异质性类型
    print("\n请选择异质性类型:")
    print("1. 特征异质性 (Feature Heterogeneity)")
    print("2. 标签异质性 (Label Heterogeneity - 高异质性)")
    print("3. 样本数量异质性 (Quantity Heterogeneity)")
    print("4. IID均匀分布 (IID Uniform Distribution)")
    
    choice = input("\n请输入选项 (1/2/3/4，默认2): ").strip() or "2"
    
    # 加载原始数据（使用绝对路径）
    script_dir = Path(__file__).parent.resolve()
    data_path = script_dir / "Xinwang" / "xinwang.csv"
    df = pd.read_csv(data_path)
    
    # 处理缺失值
    df = df.fillna(df.median(numeric_only=True))
    
    # 使用toad进行特征选择（100维 → ~38维）
    df_selected, selected_features = apply_toad_feature_selection(df, target_col='target')
    
    # 提取特征和标签
    X = df_selected.drop(columns=['target']).values
    y = df_selected['target'].values
    
    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    print(f"\n处理后数据:")
    print(f"  总样本数: {len(y)}")
    print(f"  特征维度: {X.shape[1]} (TOAD选择后)")
    print(f"  类别分布: 类0={np.sum(y==0)}, 类1={np.sum(y==1)}, 比例={np.mean(y):.2%}")
    
    # 根据选择生成数据
    if choice == "1":
        client_data, avg_wasserstein = generate_feature_heterogeneity(X, y, num_clients=10, alpha=0.01, K=None, random_state=42, samples_per_client=1788)
        heterogeneity_type = 'feature'
    elif choice == "3":
        client_data, sample_counts = generate_quantity_heterogeneity(X, y, num_clients=10, alpha=0.1, random_state=42)
        heterogeneity_type = 'quantity'
    elif choice == "4":
        client_data, avg_kl = generate_iid_uniform(X, y, num_clients=10)
        heterogeneity_type = 'iid'
    else:
        client_data, avg_kl = generate_label_heterogeneity(X, y, num_clients=10, alpha=0.001, min_samples=800, min_minority_samples=30, min_majority_samples=50, random_state=42)
        heterogeneity_type = 'label'
    
    # 保存数据
    save_to_npz(client_data, dataset_name='Xinwang', heterogeneity_type=heterogeneity_type)
    save_config(10, 2, X.shape[1], heterogeneity_type, dataset_name='Xinwang')
    
    print("\n" + "="*70)
    print("数据生成完成！")
    print("="*70)

if __name__ == '__main__':
    np.random.seed(42)
    main()
