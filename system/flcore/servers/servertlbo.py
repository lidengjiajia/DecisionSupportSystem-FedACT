"""
FedTLBO Server: 联邦TLBO服务器
================================

纯净的服务器实现，只包含训练流程控制。
所有攻击和防御组件都从 flcore/attack 模块导入。

架构设计:
- 攻击实现: flcore/attack/attacks.py
- 防御实现: flcore/attack/defenses.py (包括FedACT组件)
- 客户端: flcore/clients/clienttlbo.py
- 服务器: 本文件 (训练流程控制)

作者: FedACT Team
日期: 2026-01-22
"""

import time
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Set

from flcore.clients.clienttlbo import clientTLBO
from flcore.servers.serverbase import Server

# 从 attack 模块导入所有防御组件
from flcore.attack.defenses import (
    TLBODefense, 
    DefenseStrategy, 
    FedACTDefense,
    GradientDetector,
    Committee,
    EvidenceChain,
    DEFENSE_REGISTRY
)


class FedTLBO(Server):
    """
    FedTLBO/FedACT 服务器
    
    支持的防御模式 (全部从 attack/defenses.py 导入):
    - fedact: 完整FedACT (Autoencoder + Committee + TLBO)
    - none/fedavg: 无防御，简单平均
    - median: 坐标中值
    - trimmed_mean: 修剪均值
    - krum: Krum算法
    - multi_krum: Multi-Krum
    - bulyan: Bulyan算法
    - rfa: Robust Federated Averaging
    - fltrust: FLTrust
    - signsgd: 符号SGD
    - norm_bound: 范数裁剪
    - tlbo: 仅TLBO聚合
    
    核心流程:
    1. 客户端本地训练 (攻击在客户端执行)
    2. 收集梯度
    3. 根据defense_mode选择防御策略 (防御在服务端执行)
    4. 聚合梯度
    """
    
    def __init__(self, args, times):
        super().__init__(args, times)
        
        # 初始化客户端
        self.set_slow_clients()
        self.set_clients(clientTLBO)
        
        # 梯度维度
        self.grad_dim = sum(p.numel() for p in self.global_model.parameters())
        
        # ============ 防御配置 ============
        self.defense_mode = getattr(args, 'defense_mode', 'fedact')
                # 阈值系数配置（用于回应审稿人问题）
        self.anomaly_lower_coef = getattr(args, 'anomaly_lower_coef', 0.7)  # 委员会投票下界系数
        self.anomaly_upper_coef = getattr(args, 'anomaly_upper_coef', 1.5)  # 直接拒绝上界系数
                # 防御组件（从 attack/defenses.py 导入）
        if self.defense_mode == 'fedact':
            # 使用完整的 FedACT 防御
            self.fedact_defense = FedACTDefense(
                committee_size=getattr(args, 'committee_size', 5),
                tlbo_iterations=getattr(args, 'tlbo_iterations', 10),
                autoencoder_epochs=getattr(args, 'autoencoder_epochs', 20),
                vote_threshold=getattr(args, 'vote_threshold', 0.3)
            )
            # 消融实验开关
            self.use_committee = getattr(args, 'use_committee', True)
            self.use_autoencoder = getattr(args, 'use_autoencoder', True)
            self.use_tlbo = getattr(args, 'use_tlbo', True)
            self.use_merkle = getattr(args, 'use_merkle', True)
            self.use_incentive = getattr(args, 'use_incentive', True)
        else:
            # 使用其他防御策略
            self.fedact_defense = None
            self.use_tlbo = getattr(args, 'use_tlbo', True)
            
            if self.defense_mode not in ['none', 'fedavg']:
                self.defense_strategy = DefenseStrategy(
                    defense_mode=self.defense_mode,
                    num_clients=self.num_join_clients,
                    trim_ratio=getattr(args, 'trim_ratio', 0.1),
                    num_selected=getattr(args, 'multi_krum_k', 5)
                )
            else:
                self.defense_strategy = None
        
        # TLBO聚合器
        self.tlbo = TLBODefense(iterations=getattr(args, 'tlbo_iterations', 10))
        
        # ============ 攻击配置 (由客户端执行) ============
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
        
        # 统计（每轮记录）
        self.detection_stats = []  # 每轮的 TP/TN/FP/FN 记录
        self.cumulative_stats = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}  # 累积统计
        
        self._print_config()
    
    def _calculate_detection_metrics(
        self, rnd: int, client_ids: List[int], 
        normal_ids: List[int], anomaly_ids: List[int]
    ) -> dict:
        """计算本轮的 TP/TN/FP/FN"""
        if not self.enable_attack or not self.malicious_ids:
            return None
        
        tp = len([cid for cid in anomaly_ids if cid in self.malicious_ids])  # 正确识别恶意
        fp = len([cid for cid in anomaly_ids if cid not in self.malicious_ids])  # 误判正常为恶意
        tn = len([cid for cid in normal_ids if cid not in self.malicious_ids])  # 正确识别正常
        fn = len([cid for cid in normal_ids if cid in self.malicious_ids])  # 漏判恶意
        
        # 更新累积统计
        self.cumulative_stats['tp'] += tp
        self.cumulative_stats['fp'] += fp
        self.cumulative_stats['tn'] += tn
        self.cumulative_stats['fn'] += fn
        
        # 计算本轮指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(client_ids) if len(client_ids) > 0 else 0.0
        
        return {
            'round': rnd,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'total_clients': len(client_ids),
            'malicious_detected': tp,
            'malicious_total': len([cid for cid in client_ids if cid in self.malicious_ids])
        }
    
    def _save_detection_stats(self):
        """保存检测统计结果到文件"""
        if not self.detection_stats:
            return
        
        import json
        import os
        
        # 创建结果目录
        stats_dir = os.path.join('results', '检测统计')
        os.makedirs(stats_dir, exist_ok=True)
        
        # 生成文件名
        filename = f"{self.defense_mode}_{self.attack_mode}_stats.json"
        filepath = os.path.join(stats_dir, filename)
        
        # 保存详细数据
        output = {
            'config': {
                'defense_mode': self.defense_mode,
                'attack_mode': self.attack_mode,
                'malicious_ratio': self.malicious_ratio,
                'num_clients': self.num_clients,
                'global_rounds': self.global_rounds,
                'anomaly_lower_coef': self.anomaly_lower_coef,
                'anomaly_upper_coef': self.anomaly_upper_coef,
            },
            'malicious_clients': list(self.malicious_ids),
            'round_by_round': self.detection_stats,
            'cumulative': self.cumulative_stats,
            'overall_metrics': {
                'precision': self.cumulative_stats['tp'] / (
                    self.cumulative_stats['tp'] + self.cumulative_stats['fp']
                ) if (self.cumulative_stats['tp'] + self.cumulative_stats['fp']) > 0 else 0.0,
                'recall': self.cumulative_stats['tp'] / (
                    self.cumulative_stats['tp'] + self.cumulative_stats['fn']
                ) if (self.cumulative_stats['tp'] + self.cumulative_stats['fn']) > 0 else 0.0,
                'accuracy': (self.cumulative_stats['tp'] + self.cumulative_stats['tn']) / (
                    sum(self.cumulative_stats.values())
                ) if sum(self.cumulative_stats.values()) > 0 else 0.0,
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 检测统计已保存: {filepath}")
        print(f"累积指标 - Precision: {output['overall_metrics']['precision']:.3f}, "
              f"Recall: {output['overall_metrics']['recall']:.3f}, "
              f"Accuracy: {output['overall_metrics']['accuracy']:.3f}")
    
    def _print_config(self):
        print(f"\n{'='*60}")
        print(f"FedACT Server")
        print(f"{'='*60}")
        print(f"客户端: {self.num_clients}, 轮数: {self.global_rounds}")
        print(f"防御模式: {self.defense_mode}")
        
        if self.defense_mode == 'fedact' and self.fedact_defense:
            print(f"  - 委员会大小: {self.fedact_defense.committee_size}")
            print(f"  - TLBO迭代: {self.fedact_defense.tlbo_iterations}")
            print(f"  - 阈值系数: [{self.anomaly_lower_coef}, {self.anomaly_upper_coef}]")
            print(f"  - 组件开关: AE={self.use_autoencoder}, Committee={self.use_committee}, "
                  f"TLBO={self.use_tlbo}")
        
        if self.enable_attack:
            print(f"攻击模式: {self.attack_mode} (恶意客户端: {len(self.malicious_ids)})")
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
            
            # 收集梯度 (攻击在客户端 train_and_get_gradient 中执行)
            gradients, client_ids = self._collect_gradients()
            
            # 聚合 (防御在这里执行)
            if self.defense_mode == 'fedact':
                aggregated, normal_ids, anomaly_ids = self._fedact_aggregate(
                    gradients, client_ids, rnd
                )
                self._update_stats(normal_ids, anomaly_ids)
            elif self.defense_mode in ['none', 'fedavg']:
                aggregated = sum(gradients) / len(gradients)
                normal_ids, anomaly_ids = client_ids, []
            else:
                aggregated = self._baseline_aggregate(gradients)
                normal_ids, anomaly_ids = client_ids, []
            
            # 更新模型
            if aggregated is not None:
                self._apply_gradient(aggregated)
            
            if rnd % self.eval_gap == 0:
                if self.defense_mode == 'fedact':
                    print(f"  检测: 正常={len(normal_ids)}, 异常={len(anomaly_ids)}")
                print(f"  耗时: {time.time()-t0:.2f}s")
        
        # 训练完成
        print(f"\n{'='*60}")
        print("训练完成")
        if self.defense_mode == 'fedact':
            self._print_stats()
        
        # 保存检测统计
        self._save_detection_stats()
        print(f"{'='*60}")
        
        self.save_results()
        self.save_global_model()
    
    def _collect_gradients(self) -> Tuple[List[torch.Tensor], List[int]]:
        """
        收集所有客户端梯度
        
        攻击在客户端 train_and_get_gradient() 中执行
        """
        gradients, client_ids = [], []
        benign_gradients = []
        
        # 第一步：诚实客户端训练
        for c in self.selected_clients:
            if not c.is_malicious:
                grad = c.train_and_get_gradient()
                gradients.append(grad)
                client_ids.append(c.id)
                benign_gradients.append(grad)
        
        # 第二步：恶意客户端训练（攻击在这里执行）
        for c in self.selected_clients:
            if c.is_malicious:
                if hasattr(c, 'set_benign_gradients') and benign_gradients:
                    c.set_benign_gradients(benign_gradients)
                grad = c.train_and_get_gradient()  # 攻击在这里执行
                gradients.append(grad)
                client_ids.append(c.id)
        
        return gradients, client_ids
    
    def _fedact_aggregate(
        self, 
        gradients: List[torch.Tensor], 
        client_ids: List[int],
        rnd: int
    ) -> Tuple[torch.Tensor, List[int], List[int]]:
        """
        FedACT 聚合
        
        使用 FedACTDefense 类进行聚合，支持消融实验
        """
        if not self.fedact_defense:
            return sum(gradients) / len(gradients), client_ids, []
        
        defense = self.fedact_defense
        
        # 消融: 不使用自编码器
        if not self.use_autoencoder:
            if self.use_tlbo:
                agg = defense.tlbo.aggregate(gradients)
            else:
                agg = sum(gradients) / len(gradients)
            return agg, client_ids, []
        
        # 初始化检测器
        grad_dim = gradients[0].numel()
        device = gradients[0].device
        
        if defense.detector is None:
            defense.detector = GradientDetector(
                input_dim=grad_dim,
                auto_adapt=True
            ).to(device)
            print(f"  自编码器: 输入维度={grad_dim}, 潜在维度={defense.detector.latent_dim}")
        
        grad_tensor = torch.stack(gradients).to(device)
        
        # 训练自编码器（增量策略）
        if rnd <= 3:
            defense.detector.fit(grad_tensor, epochs=defense.autoencoder_epochs)
        elif rnd % 5 == 0:
            defense.detector.fit(grad_tensor, epochs=max(5, defense.autoencoder_epochs // 4))
        
        # 异常检测
        scores = defense.detector.anomaly_score(grad_tensor)
        threshold = scores.mean() + 2 * scores.std()
        
        # 委员会（前5轮不使用，消融时也不使用）
        use_committee = self.use_committee and rnd > 5
        committee_grads = []
        
        if use_committee:
            reputations = {c: defense.evidence.get_reputation(c) for c in client_ids}
            defense.committee.select(gradients, client_ids, reputations)
            committee_grads = [
                gradients[client_ids.index(m)]
                for m in defense.committee.members if m in client_ids
            ]
        
        # 分类
        normal_ids, anomaly_ids = [], []
        results = {}
        
        for i, cid in enumerate(client_ids):
            score = scores[i].item()
            is_anomaly = score > threshold
            
            # 可疑样本委员会投票（使用可配置的阈值系数）
            if use_committee and committee_grads:
                lower_bound = threshold * self.anomaly_lower_coef
                upper_bound = threshold * self.anomaly_upper_coef
                if lower_bound < score <= upper_bound:
                    # 排除自己：如果当前客户端是委员会成员，投票时不包含自己的梯度
                    voting_grads = [
                        committee_grads[j] 
                        for j, m in enumerate(defense.committee.members) 
                        if m != cid
                    ]
                    if voting_grads:
                        is_anomaly, _ = defense.committee.vote(
                            gradients[i], voting_grads, threshold=defense.vote_threshold
                        )
            
            results[cid] = {'score': score, 'is_anomaly': is_anomaly}
            (anomaly_ids if is_anomaly else normal_ids).append(cid)
        
        # 计算 TP/TN/FP/FN（如果启用了攻击）
        round_stats = self._calculate_detection_metrics(
            rnd, client_ids, normal_ids, anomaly_ids
        )
        
        # 聚合正常梯度
        if normal_ids:
            normal_grads = [gradients[client_ids.index(c)] for c in normal_ids]
            
            if self.use_incentive:
                weights = defense.evidence.get_weights(normal_ids)
                weight_list = [weights[c] for c in normal_ids]
            else:
                weight_list = None
            
            if self.use_tlbo:
                aggregated = defense.tlbo.aggregate(normal_grads, weight_list)
            else:
                if weight_list is None:
                    weight_list = [1.0 / len(normal_grads)] * len(normal_grads)
                aggregated = sum(w * g for w, g in zip(weight_list, normal_grads))
            
            # 更新信誉
            if self.use_incentive:
                for cid in normal_ids:
                    idx = client_ids.index(cid)
                    contrib = F.cosine_similarity(
                        gradients[idx].unsqueeze(0), aggregated.unsqueeze(0)
                    ).item()
                    contrib = max(0, (contrib + 1) / 2)
                    defense.evidence.update_reputation(cid, True, contrib)
                
                for cid in anomaly_ids:
                    defense.evidence.update_reputation(cid, False)
            
            # 存证
            if self.use_merkle:
                defense.evidence.record(rnd, results)
            
            # 记录本轮检测统计
            if round_stats:
                self.detection_stats.append(round_stats)
            
            return aggregated, normal_ids, anomaly_ids
        else:
            return defense.tlbo.aggregate(gradients), [], client_ids
    
    def _baseline_aggregate(self, gradients: List[torch.Tensor]) -> torch.Tensor:
        """基线防御方法聚合"""
        if self.defense_strategy:
            defense_result = self.defense_strategy.aggregate(gradients)
            
            if self.use_tlbo:
                # 用防御结果筛选可信梯度
                similarities = [
                    F.cosine_similarity(g.unsqueeze(0), defense_result.unsqueeze(0)).item()
                    for g in gradients
                ]
                threshold = np.median(similarities)
                trusted = [g for g, s in zip(gradients, similarities) if s >= threshold]
                
                if trusted:
                    return self.tlbo.aggregate(trusted)
            
            return defense_result
        
        if self.use_tlbo:
            return self.tlbo.aggregate(gradients)
        return sum(gradients) / len(gradients)
    
    def _apply_gradient(self, gradient: torch.Tensor):
        """应用梯度更新"""
        offset = 0
        with torch.no_grad():
            for p in self.global_model.parameters():
                n = p.numel()
                p.data += self.learning_rate * gradient[offset:offset+n].view(p.shape).to(p.device)
                offset += n
    
    def _update_stats(self, normal_ids: List[int], anomaly_ids: List[int]):
        """更新检测统计（用于评估检测准确率）"""
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
