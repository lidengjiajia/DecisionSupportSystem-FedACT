#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
中心化训练服务器

作为联邦学习的对比基线，所有数据集中在服务器端进行训练。
用于验证联邦学习在隐私保护条件下的性能损失。

修复: 2026-01-22
- 修正评估逻辑，使用汇聚后的测试数据
- 优化学习率调度
"""

import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

from flcore.servers.serverbase import Server
from flcore.clients.clientavg import clientAVG


class Centralized(Server):
    """
    中心化训练
    
    将所有客户端数据汇聚到服务器进行统一训练，
    作为联邦学习效果的上界参考。
    """
    
    def __init__(self, args, times):
        super().__init__(args, times)
        
        # 初始化慢客户端配置
        self.set_slow_clients()
        self.set_clients(clientAVG)
        
        # 汇聚所有客户端数据
        self.train_loader, self.test_loader = self._aggregate_data()
        
        # 优化器配置 - 使用Adam，更稳定
        self.optimizer = torch.optim.Adam(
            self.global_model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=30, 
            gamma=0.5
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 训练过程记录
        self.train_losses = []
        self.test_accs = []
        self.test_aucs = []
        
        self._print_config()
    
    def _aggregate_data(self):
        """汇聚所有客户端的训练和测试数据"""
        train_datasets = []
        test_datasets = []
        
        for client in self.clients:
            train_data = client.load_train_data()
            test_data = client.load_test_data()
            train_datasets.append(train_data.dataset)
            test_datasets.append(test_data.dataset)
        
        combined_train = ConcatDataset(train_datasets)
        combined_test = ConcatDataset(test_datasets)
        
        train_loader = DataLoader(
            combined_train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )
        
        test_loader = DataLoader(
            combined_test,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )
        
        print(f"[中心化] 汇聚训练数据: {len(combined_train)} 样本")
        print(f"[中心化] 汇聚测试数据: {len(combined_test)} 样本")
        return train_loader, test_loader
    
    def _print_config(self):
        """打印配置信息"""
        print(f"\n{'='*60}")
        print(f"中心化训练 (Centralized Training)")
        print(f"{'='*60}")
        print(f"训练轮数: {self.global_rounds}")
        print(f"批量大小: {self.batch_size}")
        print(f"学习率: {self.learning_rate}")
        print(f"优化器: Adam")
        print(f"{'='*60}\n")
    
    def _evaluate_centralized(self):
        """使用汇聚的测试数据进行评估"""
        self.global_model.eval()
        
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for x, y in self.test_loader:
                if isinstance(x, list):
                    x = [xi.to(self.device) for xi in x]
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                output = self.global_model(x)
                probs = torch.softmax(output, dim=1)
                preds = output.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        # 计算指标
        acc = accuracy_score(all_labels, all_preds)
        
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.5
        
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        return acc, auc, precision, recall, f1
    
    def train(self):
        """执行中心化训练"""
        best_acc = 0.0
        
        for epoch in range(1, self.global_rounds + 1):
            t0 = time.time()
            
            self.global_model.train()
            epoch_loss = 0.0
            batch_count = 0
            
            for x, y in self.train_loader:
                if isinstance(x, list):
                    x = [xi.to(self.device) for xi in x]
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.global_model(x)
                loss = self.criterion(output, y)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), 10.0)
                
                self.optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            avg_loss = epoch_loss / max(batch_count, 1)
            self.train_losses.append(avg_loss)
            
            # 更新学习率
            self.scheduler.step()
            
            # 定期评估
            if epoch % self.eval_gap == 0:
                acc, auc, precision, recall, f1 = self._evaluate_centralized()
                self.test_accs.append(acc)
                self.test_aucs.append(auc)
                
                # 记录到父类的结果中
                self.rs_test_acc.append(acc)
                self.rs_test_auc.append(auc)
                self.rs_train_loss.append(avg_loss)
                
                print(f"\n[轮次 {epoch}/{self.global_rounds}]")
                print(f"  训练损失: {avg_loss:.4f}")
                print(f"  测试 Accuracy: {acc:.4f}, AUC: {auc:.4f}")
                print(f"  测试 Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
                print(f"  学习率: {self.scheduler.get_last_lr()[0]:.6f}")
                print(f"  耗时: {time.time()-t0:.2f}s")
                
                if acc > best_acc:
                    best_acc = acc
                    print(f"  ★ 新最佳准确率!")
        
        print(f"\n{'='*60}")
        print("中心化训练完成")
        print(f"最佳准确率: {best_acc:.4f}")
        print(f"{'='*60}")
        
        self.save_results()
        self.save_global_model()
