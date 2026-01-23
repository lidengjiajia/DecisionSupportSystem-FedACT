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
        
        # 获取数据集类型
        self.dataset_type = getattr(args, 'dataset', 'uci').lower()
        
        # 防御组件（从 attack/defenses.py 导入）
        if self.defense_mode == 'fedact':
            # 使用完整的 FedACT 防御（传递数据集类型以启用专用配置）
            self.fedact_defense = FedACTDefense(
                committee_size=getattr(args, 'committee_size', 5),
                tlbo_iterations=getattr(args, 'tlbo_iterations', 10),
                autoencoder_epochs=getattr(args, 'autoencoder_epochs', 20),
                vote_threshold=getattr(args, 'vote_threshold', 0.3),
                dataset_type=self.dataset_type
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
            # 所有防御方法防御过后都采用TLBO聚合
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
        self.stats = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}  # 用于_update_stats和_print_stats
        
        self._print_config()
    
    def _calculate_stage_metrics(
        self, client_ids: List[int], 
        anomaly_ids: List[int]
    ) -> dict:
        """
        计算某个阶段的检测指标
        
        Args:
            client_ids: 所有参与客户端ID列表
            anomaly_ids: 该阶段判定为异常的客户端ID列表
            
        Returns:
            包含 TP/FP/TN/FN 的字典
        """
        if not self.enable_attack or not self.malicious_ids:
            return {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
        
        normal_ids = [cid for cid in client_ids if cid not in anomaly_ids]
        
        tp = len([cid for cid in anomaly_ids if cid in self.malicious_ids])  # 正确识别恶意
        fp = len([cid for cid in anomaly_ids if cid not in self.malicious_ids])  # 误判正常为恶意
        tn = len([cid for cid in normal_ids if cid not in self.malicious_ids])  # 正确识别正常
        fn = len([cid for cid in normal_ids if cid in self.malicious_ids])  # 漏判恶意
        
        return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
    
    def _calculate_detection_metrics(
        self, rnd: int, client_ids: List[int], 
        normal_ids: List[int], anomaly_ids: List[int],
        stage_stats: dict = None
    ) -> dict:
        """
        计算本轮的 TP/TN/FP/FN（支持分阶段统计）
        
        Args:
            rnd: 当前轮次
            client_ids: 所有参与客户端ID列表
            normal_ids: 最终判定为正常的客户端ID列表
            anomaly_ids: 最终判定为异常的客户端ID列表
            stage_stats: 分阶段统计（autoencoder_only, after_committee）
        """
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
        
        result = {
            'round': rnd,
            # 最终结果
            'final_tp': tp, 'final_fp': fp, 'final_tn': tn, 'final_fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'total_clients': len(client_ids),
            'malicious_detected': tp,
            'malicious_total': len([cid for cid in client_ids if cid in self.malicious_ids])
        }
        
        # 添加分阶段统计
        if stage_stats:
            # Autoencoder阶段
            if 'autoencoder_only' in stage_stats:
                ae_stats = stage_stats['autoencoder_only']
                result.update({
                    'ae_tp': ae_stats['tp'],
                    'ae_fp': ae_stats['fp'],
                    'ae_tn': ae_stats['tn'],
                    'ae_fn': ae_stats['fn'],
                })
            # 委员会阶段后
            if 'after_committee' in stage_stats:
                comm_stats = stage_stats['after_committee']
                result.update({
                    'comm_tp': comm_stats['tp'],
                    'comm_fp': comm_stats['fp'],
                    'comm_tn': comm_stats['tn'],
                    'comm_fn': comm_stats['fn'],
                })
        
        return result
    
    def _save_detection_stats(self):
        """保存检测统计结果到文件（JSON + Excel）"""
        if not self.detection_stats:
            return
        
        import json
        import os
        
        # 创建结果目录
        stats_dir = os.path.join('results', '检测统计')
        os.makedirs(stats_dir, exist_ok=True)
        
        # 获取数据集和异质性类型
        dataset_name = getattr(self, 'dataset', 'unknown')
        heterogeneity = getattr(self.args, 'partition', 'iid') if hasattr(self, 'args') else 'iid'
        
        # 生成文件名（包含更多信息）
        base_filename = f"{dataset_name}_{heterogeneity}_{self.defense_mode}_{self.attack_mode}"
        json_filepath = os.path.join(stats_dir, f"{base_filename}_stats.json")
        
        # 计算汇总指标
        overall_metrics = {
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
        
        # 保存JSON详细数据
        output = {
            'config': {
                'dataset': dataset_name,
                'heterogeneity': heterogeneity,
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
            'overall_metrics': overall_metrics
        }
        
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        # 保存Excel文件（单个算法）
        self._save_detection_excel(stats_dir, base_filename, output)
        
        print(f"\n✅ 检测统计已保存: {json_filepath}")
        print(f"累积指标 - Precision: {overall_metrics['precision']:.3f}, "
              f"Recall: {overall_metrics['recall']:.3f}, "
              f"Accuracy: {overall_metrics['accuracy']:.3f}")
    
    def _save_detection_excel(self, stats_dir: str, base_filename: str, data: dict):
        """
        保存检测统计到Excel文件
        
        每个算法一个Excel文件，包含：
        - Sheet1: 逐轮统计
        - Sheet2: 配置信息
        - Sheet3: 汇总指标
        """
        try:
            import os
            import pandas as pd
            
            excel_filepath = os.path.join(stats_dir, f"{base_filename}_stats.xlsx")
            
            # 准备逐轮数据
            rounds_data = []
            for r in data['round_by_round']:
                row = {
                    'Round': r['round'],
                    # 最终结果
                    'Final_TP': r.get('final_tp', r.get('tp', 0)),
                    'Final_FP': r.get('final_fp', r.get('fp', 0)),
                    'Final_TN': r.get('final_tn', r.get('tn', 0)),
                    'Final_FN': r.get('final_fn', r.get('fn', 0)),
                    'Precision': r.get('precision', 0),
                    'Recall': r.get('recall', 0),
                    'F1': r.get('f1', 0),
                    'Accuracy': r.get('accuracy', 0),
                }
                
                # 添加分阶段数据（如果有）
                if 'ae_tp' in r:
                    row.update({
                        'AE_TP': r['ae_tp'],
                        'AE_FP': r['ae_fp'],
                        'AE_TN': r['ae_tn'],
                        'AE_FN': r['ae_fn'],
                    })
                if 'comm_tp' in r:
                    row.update({
                        'Comm_TP': r['comm_tp'],
                        'Comm_FP': r['comm_fp'],
                        'Comm_TN': r['comm_tn'],
                        'Comm_FN': r['comm_fn'],
                    })
                
                rounds_data.append(row)
            
            df_rounds = pd.DataFrame(rounds_data)
            
            # 准备配置数据
            config_data = pd.DataFrame([data['config']])
            
            # 准备汇总数据
            summary_data = pd.DataFrame([{
                **data['cumulative'],
                **data['overall_metrics']
            }])
            
            # 写入Excel
            with pd.ExcelWriter(excel_filepath, engine='openpyxl') as writer:
                df_rounds.to_excel(writer, sheet_name='逐轮统计', index=False)
                config_data.to_excel(writer, sheet_name='配置', index=False)
                summary_data.to_excel(writer, sheet_name='汇总', index=False)
            
            print(f"✅ Excel已保存: {excel_filepath}")
            
        except ImportError:
            print("⚠️ 需要安装 pandas 和 openpyxl 来生成Excel文件")
        except Exception as e:
            print(f"⚠️ 保存Excel失败: {e}")
    
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
        import sys
        for rnd in range(1, self.global_rounds + 1):
            t0 = time.time()
            
            # 实时显示当前轮次（不换行，覆盖显示）
            print(f"\r[Round {rnd}/{self.global_rounds}] 正在训练...", end="", flush=True)
            
            # 选择客户端
            self.selected_clients = self.select_clients()
            self.send_models()
            
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
                # 其他baseline防御方法
                aggregated, normal_ids, anomaly_ids = self._baseline_aggregate_with_detection(
                    gradients, client_ids, rnd
                )
                self._update_stats(normal_ids, anomaly_ids)
            
            # 更新模型
            if aggregated is not None:
                self._apply_gradient(aggregated)
            
            if rnd % self.eval_gap == 0:
                print(f"\r[Round {rnd}/{self.global_rounds}]" + " " * 30)  # 清除行
                self.evaluate()
                if self.defense_mode != 'fedavg' and self.defense_mode != 'none':
                    print(f"  检测: 正常={len(normal_ids)}, 异常={len(anomaly_ids)}")
                print(f"  耗时: {time.time()-t0:.2f}s")
        
        # 训练完成
        print(f"\n{'='*60}")
        print("训练完成")
        # 所有防御方法都打印检测统计（除了无防御模式）
        if self.defense_mode not in ['none', 'fedavg']:
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
        total_clients = len(self.selected_clients)
        
        # 第一步：诚实客户端训练
        client_count = 0
        for c in self.selected_clients:
            if not c.is_malicious:
                client_count += 1
                print(f"\r  客户端训练 {client_count}/{total_clients}...", end="", flush=True)
                grad = c.train_and_get_gradient()
                gradients.append(grad)
                client_ids.append(c.id)
                benign_gradients.append(grad)
        
        # 第二步：恶意客户端训练（攻击在这里执行）
        for c in self.selected_clients:
            if c.is_malicious:
                client_count += 1
                print(f"\r  客户端训练 {client_count}/{total_clients}...", end="", flush=True)
                if hasattr(c, 'set_benign_gradients') and benign_gradients:
                    c.set_benign_gradients(benign_gradients)
                grad = c.train_and_get_gradient()  # 攻击在这里执行
                gradients.append(grad)
                client_ids.append(c.id)
        
        print(f"\r  客户端训练完成 ({total_clients})     ", end="", flush=True)
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
        
        # 初始化检测器（使用数据集专用配置）
        grad_dim = gradients[0].numel()
        device = gradients[0].device
        
        if defense.detector is None:
            # 将数据集名称转换为检测器配置格式 (Uci -> uci, Xinwang -> xinwang)
            dataset_type = self.dataset.lower() if hasattr(self, 'dataset') else None
            defense.detector = GradientDetector(
                input_dim=grad_dim,
                auto_adapt=True,
                dataset_type=dataset_type
            ).to(device)
            print(f"  自编码器[{dataset_type}]: 输入维度={grad_dim:,} → {defense.detector.input_dim:,}, "
                  f"潜在维度={defense.detector.latent_dim}, 隐藏层={defense.detector.hidden_layers}")
        
        grad_tensor = torch.stack(gradients).to(device)
        
        # 训练自编码器（增量策略）
        if rnd <= 3:
            defense.detector.fit(grad_tensor, epochs=defense.autoencoder_epochs)
        elif rnd % 5 == 0:
            defense.detector.fit(grad_tensor, epochs=max(5, defense.autoencoder_epochs // 4))
        
        # 异常检测
        scores = defense.detector.anomaly_score(grad_tensor)
        threshold = scores.mean() + 2 * scores.std()
        
        # =========== 阶段1: 自编码器初步检测 ===========
        # 记录仅使用阈值的初步判断结果
        ae_anomaly_ids = []
        for i, cid in enumerate(client_ids):
            score = scores[i].item()
            if score > threshold:
                ae_anomaly_ids.append(cid)
        
        # 计算自编码器阶段的统计
        stage_stats = {}
        if self.enable_attack and self.malicious_ids:
            stage_stats['autoencoder_only'] = self._calculate_stage_metrics(
                client_ids, ae_anomaly_ids
            )
        
        # =========== 阶段2: 委员会投票 ===========
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
        
        # =========== 最终分类（结合自编码器+委员会） ===========
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
        
        # 计算委员会阶段后的统计（如果使用了委员会）
        if use_committee and self.enable_attack and self.malicious_ids:
            stage_stats['after_committee'] = self._calculate_stage_metrics(
                client_ids, anomaly_ids
            )
        
        # 计算最终 TP/TN/FP/FN（如果启用了攻击）
        round_stats = self._calculate_detection_metrics(
            rnd, client_ids, normal_ids, anomaly_ids,
            stage_stats=stage_stats if stage_stats else None
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
    
    def _baseline_aggregate_with_detection(
        self, 
        gradients: List[torch.Tensor], 
        client_ids: List[int],
        rnd: int
    ) -> Tuple[torch.Tensor, List[int], List[int]]:
        """
        基线防御方法聚合（带检测统计）
        
        Args:
            gradients: 梯度列表
            client_ids: 客户端ID列表
            rnd: 当前轮次
            
        Returns:
            aggregated: 聚合后的梯度
            normal_ids: 选中的客户端ID
            anomaly_ids: 排除的客户端ID
        """
        if not self.defense_strategy:
            if self.use_tlbo:
                return self.tlbo.aggregate(gradients), client_ids, []
            return sum(gradients) / len(gradients), client_ids, []
        
        # 调用防御策略的聚合方法
        defense_result = self.defense_strategy.aggregate(gradients)
        
        # 获取防御方法选中/排除的索引
        defense_instance = self.defense_strategy.defense_instance
        selected_indices = getattr(defense_instance, 'last_selected_indices', list(range(len(gradients))))
        excluded_indices = getattr(defense_instance, 'last_excluded_indices', [])
        
        # 将索引转换为客户端ID
        normal_ids = [client_ids[i] for i in selected_indices if i < len(client_ids)]
        anomaly_ids = [client_ids[i] for i in excluded_indices if i < len(client_ids)]
        
        # 对于不选择特定客户端的方法（如median, trimmed_mean），所有客户端都算正常
        if not selected_indices and not excluded_indices:
            normal_ids = client_ids
            anomaly_ids = []
        
        # 计算检测统计
        round_stats = self._calculate_detection_metrics(
            rnd, client_ids, normal_ids, anomaly_ids
        )
        if round_stats:
            self.detection_stats.append(round_stats)
        
        # 可选：使用TLBO进一步优化
        if self.use_tlbo and normal_ids:
            normal_grads = [gradients[client_ids.index(c)] for c in normal_ids]
            if normal_grads:
                defense_result = self.tlbo.aggregate(normal_grads)
        
        return defense_result, normal_ids, anomaly_ids
    
    def _baseline_aggregate(self, gradients: List[torch.Tensor]) -> torch.Tensor:
        """基线防御方法聚合（向后兼容）"""
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
