"""
FedCTI: Federated Committee-based TLBO with Incentive
======================================================

面向金融信用风险场景的联邦学习框架:

核心机制:
1. 委员会投票 (Committee Voting) - 多样性委员会进行异常检测
2. TLBO聚合 (Teaching-Learning Based Optimization) - 应对数据异质性
3. 贡献度激励 (Contribution Incentive) - Merkle树存证 + 信誉机制

应用场景:
- 多机构信用评分联合建模
- 拜占庭攻击防御
- 异质数据分布处理
"""

import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
import json
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
from datetime import datetime

from flcore.clients.clientcti import clientCTI
from flcore.servers.serverbase import Server


# ==============================================================================
# 组件1: 梯度异常检测器
# ==============================================================================

class GradientDetector(nn.Module):
    """
    基于自编码器的梯度异常检测
    
    正常梯度 → 小重构误差
    异常梯度 → 大重构误差
    """
    
    def __init__(self, input_dim: int, latent_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        
        self.input_dim = input_dim
        hidden_dim = min(256, input_dim // 2)
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        
        self.register_buffer('center', torch.zeros(latent_dim))
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z
    
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """计算异常分数"""
        with torch.no_grad():
            recon, z = self.forward(x)
            recon_err = F.mse_loss(recon, x, reduction='none').mean(dim=-1)
            latent_dist = torch.norm(z - self.center, dim=-1)
            return 0.7 * recon_err + 0.3 * latent_dist
    
    def fit(self, gradients: torch.Tensor, epochs: int = 20, lr: float = 1e-3):
        """训练检测器"""
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        for _ in range(epochs):
            recon, z = self.forward(gradients)
            loss = F.mse_loss(recon, gradients)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            _, z = self.forward(gradients)
            self.center = z.mean(dim=0)
        
        self.eval()


# ==============================================================================
# 组件2: 委员会机制
# ==============================================================================

class Committee:
    """
    多样性委员会选择与投票
    
    - 多样性选择: 避免同质化偏见
    - 投票机制: 多数决策
    """
    
    def __init__(self, size: int = 5):
        self.size = size
        self.members: List[int] = []
    
    def select(
        self,
        gradients: List[torch.Tensor],
        client_ids: List[int],
        reputations: Dict[int, float]
    ) -> List[int]:
        """选择多样化委员会"""
        if len(gradients) <= self.size:
            self.members = client_ids.copy()
            return self.members
        
        selected = []
        available = list(range(len(gradients)))
        
        # 第一个: 信誉最高
        first = max(available, key=lambda i: reputations.get(client_ids[i], 1.0))
        selected.append(first)
        available.remove(first)
        
        # 后续: 最大化多样性
        while len(selected) < self.size and available:
            best_idx = None
            min_sim = float('inf')
            
            for idx in available:
                max_sim_to_selected = max([
                    F.cosine_similarity(gradients[idx].unsqueeze(0), gradients[s].unsqueeze(0)).item()
                    for s in selected
                ])
                if max_sim_to_selected < min_sim:
                    min_sim = max_sim_to_selected
                    best_idx = idx
            
            if best_idx is not None:
                selected.append(best_idx)
                available.remove(best_idx)
        
        self.members = [client_ids[i] for i in selected]
        return self.members
    
    def vote(
        self,
        target_grad: torch.Tensor,
        member_grads: List[torch.Tensor],
        threshold: float = 0.3
    ) -> Tuple[bool, float]:
        """委员会投票"""
        if not member_grads:
            return False, 0.0
        
        anomaly_votes = 0
        for mg in member_grads:
            sim = F.cosine_similarity(target_grad.unsqueeze(0), mg.unsqueeze(0)).item()
            if sim < threshold:
                anomaly_votes += 1
        
        ratio = anomaly_votes / len(member_grads)
        return ratio > 0.5, ratio


# ==============================================================================
# 组件3: Merkle存证与激励
# ==============================================================================

class EvidenceChain:
    """
    Merkle树存证链
    
    - 不可篡改记录
    - 贡献度追踪
    - 信誉管理
    """
    
    def __init__(self):
        self.chain: List[Dict] = []
        self.reputations: Dict[int, float] = defaultdict(lambda: 1.0)
        self.contributions: Dict[int, List[float]] = defaultdict(list)
    
    @staticmethod
    def _hash(data: str) -> str:
        return hashlib.sha256(data.encode()).hexdigest()
    
    def _merkle_root(self, items: List[str]) -> str:
        if not items:
            return self._hash("empty")
        
        level = [self._hash(item) for item in items]
        while len(level) > 1:
            if len(level) % 2 == 1:
                level.append(level[-1])
            level = [self._hash(level[i] + level[i+1]) for i in range(0, len(level), 2)]
        
        return level[0]
    
    def record(
        self,
        round_id: int,
        results: Dict[int, Dict]
    ) -> str:
        """记录一轮结果"""
        # 将Tensor转换为Python原生类型
        def convert_value(v):
            if hasattr(v, 'item'):  # Tensor
                return v.item()
            elif isinstance(v, (np.floating, np.integer)):
                return float(v)
            return v
        
        serializable_results = {
            k: {kk: convert_value(vv) for kk, vv in v.items()}
            for k, v in results.items()
        }
        
        items = [json.dumps({**v, 'client': k}, sort_keys=True) for k, v in serializable_results.items()]
        root = self._merkle_root(items)
        
        self.chain.append({
            'round': round_id,
            'root': root,
            'timestamp': datetime.now().isoformat(),
            'count': len(results)
        })
        
        return root
    
    def update_reputation(self, client_id: int, is_normal: bool, contribution: float = 0.0):
        """更新信誉"""
        old = self.reputations[client_id]
        if is_normal:
            self.reputations[client_id] = min(old + 0.05 * contribution, 2.0)
        else:
            self.reputations[client_id] = max(old * 0.7, 0.1)
        
        self.contributions[client_id].append(contribution)
    
    def get_reputation(self, client_id: int) -> float:
        return self.reputations[client_id]
    
    def get_weights(self, client_ids: List[int]) -> Dict[int, float]:
        """基于信誉计算权重"""
        total = sum(self.reputations[c] for c in client_ids)
        return {c: self.reputations[c] / total for c in client_ids}


# ==============================================================================
# 组件4: TLBO聚合
# ==============================================================================

class TLBOAggregator:
    """
    Teaching-Learning Based Optimization聚合
    
    - Teacher阶段: 向最优学习
    - Learner阶段: 学习者互学
    """
    
    def __init__(self, iterations: int = 10):
        self.iterations = iterations
    
    def aggregate(
        self,
        gradients: List[torch.Tensor],
        weights: Dict[int, float],
        client_ids: List[int]
    ) -> torch.Tensor:
        """TLBO聚合"""
        if len(gradients) == 0:
            raise ValueError("No gradients to aggregate")
        if len(gradients) == 1:
            return gradients[0].clone()
        
        # 加权平均作为初始
        w_list = [weights.get(cid, 1.0/len(gradients)) for cid in client_ids]
        w_sum = sum(w_list)
        w_list = [w/w_sum for w in w_list]
        
        result = sum(w * g for w, g in zip(w_list, gradients))
        
        learners = [g.clone() for g in gradients]
        
        for _ in range(self.iterations):
            # Teacher阶段
            sims = [F.cosine_similarity(l.unsqueeze(0), result.unsqueeze(0)).item() for l in learners]
            teacher_idx = np.argmax(sims)
            teacher = learners[teacher_idx]
            mean_l = torch.stack(learners).mean(dim=0)
            TF = np.random.choice([1, 2])
            
            for i, l in enumerate(learners):
                r = np.random.random()
                new_l = l + r * (teacher - TF * mean_l)
                if F.cosine_similarity(new_l.unsqueeze(0), result.unsqueeze(0)).item() > sims[i]:
                    learners[i] = new_l
            
            # Learner阶段
            for i in range(len(learners)):
                j = np.random.choice([k for k in range(len(learners)) if k != i])
                si = F.cosine_similarity(learners[i].unsqueeze(0), result.unsqueeze(0)).item()
                sj = F.cosine_similarity(learners[j].unsqueeze(0), result.unsqueeze(0)).item()
                r = np.random.random()
                diff = learners[j] - learners[i] if sj > si else learners[i] - learners[j]
                new_l = learners[i] + r * diff
                if F.cosine_similarity(new_l.unsqueeze(0), result.unsqueeze(0)).item() > si:
                    learners[i] = new_l
            
            result = torch.stack(learners).mean(dim=0)
        
        return result


# ==============================================================================
# FedCTI服务器
# ==============================================================================

class FedCTI(Server):
    """
    FedCTI: Federated Committee-based TLBO with Incentive
    
    核心流程:
    1. 客户端本地训练
    2. 收集梯度
    3. 委员会+自编码器异常检测
    4. TLBO加权聚合
    5. Merkle存证+信誉更新
    """
    
    def __init__(self, args, times):
        super().__init__(args, times)
        
        # 初始化慢客户端列表
        self.set_slow_clients()
        self.set_clients(clientCTI)
        
        # 梯度维度
        self.grad_dim = sum(p.numel() for p in self.global_model.parameters())
        
        # 核心组件
        self.detector = None
        self.committee = Committee(size=getattr(args, 'committee_size', 5))
        self.evidence = EvidenceChain()
        self.tlbo = TLBOAggregator(iterations=getattr(args, 'tlbo_iterations', 10))
        
        # 攻击配置
        self.enable_attack = getattr(args, 'enable_attack', False)
        self.attack_mode = getattr(args, 'attack_mode', 'none')
        self.malicious_ratio = getattr(args, 'malicious_ratio', 0.2)
        self.malicious_ids: Set[int] = set()
        
        if self.enable_attack and self.attack_mode != 'none':
            n_mal = int(self.num_clients * self.malicious_ratio)
            self.malicious_ids = set(np.random.choice(self.num_clients, n_mal, replace=False))
            for c in self.clients:
                if c.id in self.malicious_ids:
                    c.set_attack(self.attack_mode, getattr(args, 'attack_scale', 3.0))
        
        # 统计
        self.stats = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
        
        self._print_config()
    
    def _print_config(self):
        print(f"\n{'='*60}")
        print(f"FedCTI: Committee-based TLBO with Incentive")
        print(f"{'='*60}")
        print(f"客户端: {self.num_clients}, 轮数: {self.global_rounds}")
        print(f"委员会: {self.committee.size}, TLBO迭代: {self.tlbo.iterations}")
        if self.enable_attack:
            print(f"攻击: {self.attack_mode} ({len(self.malicious_ids)}个恶意)")
        print(f"{'='*60}\n")
    
    def train(self):
        for rnd in range(1, self.global_rounds + 1):
            t0 = time.time()
            
            # 选择客户端
            self.selected_clients = self.select_clients()
            self.send_models()
            
            if rnd % self.eval_gap == 0:
                print(f"\n[Round {rnd}/{self.global_rounds}]")
                self.evaluate()
            
            # 客户端训练并收集梯度
            # 对于高级攻击（alie, ipm, minmax），需要先收集诚实客户端梯度
            gradients, client_ids = [], []
            benign_gradients = []  # 诚实客户端梯度（用于高级攻击）
            
            # 第一步：诚实客户端训练
            for c in self.selected_clients:
                if not c.is_malicious:
                    grad = c.train_and_get_gradient()
                    gradients.append(grad)
                    client_ids.append(c.id)
                    benign_gradients.append(grad)
            
            # 第二步：恶意客户端训练（传入诚实梯度用于高级攻击）
            for c in self.selected_clients:
                if c.is_malicious:
                    # 设置诚实客户端梯度（用于alie, ipm, minmax攻击）
                    if hasattr(c, 'set_benign_gradients') and benign_gradients:
                        c.set_benign_gradients(benign_gradients)
                    grad = c.train_and_get_gradient()
                    gradients.append(grad)
                    client_ids.append(c.id)
            
            # 初始化检测器
            if self.detector is None:
                self.detector = GradientDetector(self.grad_dim).to(self.device)
            
            grad_tensor = torch.stack(gradients).to(self.device)
            self.detector.fit(grad_tensor, epochs=15)
            
            # 异常检测
            scores = self.detector.anomaly_score(grad_tensor)
            threshold = scores.mean() + 2 * scores.std()
            
            # 委员会选择
            reputations = {c: self.evidence.get_reputation(c) for c in client_ids}
            self.committee.select(gradients, client_ids, reputations)
            committee_grads = [gradients[client_ids.index(m)] for m in self.committee.members if m in client_ids]
            
            # 双阈值检测
            results = {}
            normal_ids, anomaly_ids = [], []
            
            for i, cid in enumerate(client_ids):
                score = scores[i].item()
                is_anomaly = score > threshold
                
                # 可疑样本委员会投票
                if score > threshold * 0.7 and score <= threshold * 1.5:
                    is_anomaly, _ = self.committee.vote(gradients[i], committee_grads, threshold=0.3)
                
                results[cid] = {'score': score, 'is_anomaly': is_anomaly}
                
                if is_anomaly:
                    anomaly_ids.append(cid)
                else:
                    normal_ids.append(cid)
            
            # 更新统计
            self._update_stats(normal_ids, anomaly_ids)
            
            # 聚合
            if normal_ids:
                normal_grads = [gradients[client_ids.index(c)] for c in normal_ids]
                weights = self.evidence.get_weights(normal_ids)
                
                # 计算贡献度
                aggregated = self.tlbo.aggregate(normal_grads, weights, normal_ids)
                
                for cid in normal_ids:
                    idx = client_ids.index(cid)
                    contrib = F.cosine_similarity(
                        gradients[idx].unsqueeze(0), aggregated.unsqueeze(0)
                    ).item()
                    contrib = max(0, (contrib + 1) / 2)
                    self.evidence.update_reputation(cid, True, contrib)
                    results[cid]['contribution'] = contrib
                
                for cid in anomaly_ids:
                    self.evidence.update_reputation(cid, False)
                
                # 更新模型
                self._apply_gradient(aggregated)
            
            # 存证
            self.evidence.record(rnd, results)
            
            if rnd % self.eval_gap == 0:
                print(f"  检测: 正常={len(normal_ids)}, 异常={len(anomaly_ids)}")
                print(f"  耗时: {time.time()-t0:.2f}s")
        
        print(f"\n{'='*60}")
        print("训练完成")
        self._print_stats()
        print(f"{'='*60}")
        
        self.save_results()
        self.save_global_model()
    
    def _apply_gradient(self, gradient: torch.Tensor):
        """应用梯度更新"""
        offset = 0
        with torch.no_grad():
            for p in self.global_model.parameters():
                n = p.numel()
                p.data += self.learning_rate * gradient[offset:offset+n].view(p.shape).to(p.device)
                offset += n
    
    def _update_stats(self, normal_ids: List[int], anomaly_ids: List[int]):
        """更新检测统计"""
        if not self.malicious_ids:
            return
        
        for cid in anomaly_ids:
            if cid in self.malicious_ids:
                self.stats['tp'] += 1
            else:
                self.stats['fp'] += 1
        
        for cid in normal_ids:
            if cid in self.malicious_ids:
                self.stats['fn'] += 1
            else:
                self.stats['tn'] += 1
    
    def _print_stats(self):
        """打印检测统计"""
        s = self.stats
        total = s['tp'] + s['fp'] + s['tn'] + s['fn']
        if total == 0:
            return
        
        prec = s['tp'] / max(s['tp'] + s['fp'], 1)
        rec = s['tp'] / max(s['tp'] + s['fn'], 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        
        print(f"\n检测统计:")
        print(f"  TP={s['tp']}, FP={s['fp']}, TN={s['tn']}, FN={s['fn']}")
        print(f"  Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")
