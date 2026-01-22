"""
FedCTI Client: 联邦CTI客户端

拜占庭攻击实现参考:
- ICML 2018: "The Hidden Vulnerability of Distributed Learning in Byzantium"
- NeurIPS 2019: "A Little Is Enough: Circumventing Defenses For Distributed Learning"
- IEEE S&P 2020: "Analyzing Federated Learning through an Adversarial Lens"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from typing import Optional, Dict, List

from flcore.clients.clientbase import Client


class clientCTI(Client):
    """
    FedCTI客户端
    
    功能:
    1. 本地训练
    2. 梯度计算
    3. 拜占庭攻击模拟 (实验用)
    
    支持的攻击模式:
    - sign_flip: 符号翻转攻击 (基础)
    - gaussian: 高斯噪声攻击 (基础)
    - scale: 缩放攻击 (基础)
    - little: "A Little Is Enough" 攻击 (NeurIPS 2019)
    - alie: ALIE攻击 - 自适应绕过防御 (NeurIPS 2019)
    - ipm: Inner Product Manipulation 内积操纵攻击 (ICML 2018)
    - minmax: Min-Max攻击 - 最大化损害 (IEEE S&P 2020)
    """
    
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        # 攻击配置
        self.is_malicious = False
        self.attack_mode = 'none'
        self.attack_scale = 1.0
        
        # 协同攻击相关
        self.benign_grads: List[torch.Tensor] = []  # 诚实客户端梯度 (用于高级攻击)
        
        # 状态
        self.gradient = None
    
    def set_attack(self, mode: str, scale: float = 3.0):
        """
        设置为恶意客户端
        
        Args:
            mode: 攻击模式
            scale: 攻击强度系数
        """
        self.is_malicious = True
        self.attack_mode = mode
        self.attack_scale = scale
    
    def set_benign_gradients(self, grads: List[torch.Tensor]):
        """
        设置诚实客户端梯度 (用于高级协同攻击)
        
        Args:
            grads: 诚实客户端梯度列表
        """
        self.benign_grads = grads
    
    def train_and_get_gradient(self) -> torch.Tensor:
        """训练并返回梯度"""
        # 保存训练前参数
        old_params = [p.data.clone() for p in self.model.parameters()]
        
        # 本地训练
        self._local_train()
        
        # 计算梯度
        grads = []
        for old_p, new_p in zip(old_params, self.model.parameters()):
            grads.append((new_p.data - old_p).flatten())
        gradient = torch.cat(grads)
        
        # 攻击处理
        if self.is_malicious:
            gradient = self._attack(gradient)
        
        self.gradient = gradient
        return gradient
    
    def _local_train(self):
        """本地SGD训练"""
        trainloader = self.load_train_data()
        self.model.train()
        
        for epoch in range(self.local_epochs):
            for x, y in trainloader:
                if isinstance(x, list):
                    x = [xi.to(self.device) for xi in x]
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                self.optimizer.zero_grad()
                out = self.model(x)
                loss = self.loss(out, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.optimizer.step()
    
    def _attack(self, gradient: torch.Tensor) -> torch.Tensor:
        """
        执行拜占庭攻击
        
        Args:
            gradient: 原始梯度
            
        Returns:
            恶意梯度
        """
        # === 基础攻击 ===
        if self.attack_mode == 'sign_flip':
            # 符号翻转: 简单取反
            return -gradient
        
        elif self.attack_mode == 'gaussian':
            # 高斯噪声: 添加大方差噪声
            noise = torch.randn_like(gradient) * gradient.std() * self.attack_scale
            return gradient + noise
        
        elif self.attack_mode == 'scale':
            # 缩放攻击: 反向缩放
            return gradient * (-self.attack_scale)
        
        # === 高级攻击 (NeurIPS 2019) ===
        elif self.attack_mode == 'little':
            # "A Little Is Enough" - 利用聚合规则漏洞
            # 恶意梯度 = μ - z_max * σ (刚好绕过异常检测)
            eps = self.attack_scale * gradient.std()
            return gradient + eps * torch.sign(gradient)
        
        elif self.attack_mode == 'alie':
            # ALIE攻击 - 自适应绕过中位数/修剪防御
            # 恶意梯度设计为刚好在修剪边界内
            if len(self.benign_grads) > 0:
                # 计算诚实梯度的均值和标准差
                stacked = torch.stack(self.benign_grads)
                mean = stacked.mean(dim=0)
                std = stacked.std(dim=0)
                
                # z_max 控制偏离程度 (典型值 < 2)
                z_max = min(self.attack_scale, 1.5)
                malicious = mean - z_max * std
                return malicious
            else:
                # 回退到little攻击
                eps = self.attack_scale * gradient.std()
                return gradient + eps * torch.sign(gradient)
        
        # === 高级攻击 (ICML 2018) ===
        elif self.attack_mode == 'ipm':
            # Inner Product Manipulation - 内积操纵攻击
            # 目标: 使聚合梯度与恶意方向对齐
            if len(self.benign_grads) > 0:
                # 计算诚实梯度均值
                benign_mean = torch.stack(self.benign_grads).mean(dim=0)
                
                # 恶意梯度 = -ε * 诚实均值方向
                eps = self.attack_scale
                malicious = -eps * benign_mean / (benign_mean.norm() + 1e-10)
                return malicious * gradient.norm()
            else:
                return -gradient
        
        # === 高级攻击 (IEEE S&P 2020) ===
        elif self.attack_mode == 'minmax':
            # Min-Max攻击 - 最大化与诚实梯度的距离
            if len(self.benign_grads) > 0:
                stacked = torch.stack(self.benign_grads)
                mean = stacked.mean(dim=0)
                
                # 找到距离诚实均值最远的方向
                perturbation = -self.attack_scale * mean
                
                # 确保恶意梯度与诚实均值的内积为负
                if torch.dot(perturbation, mean) > 0:
                    perturbation = -perturbation
                
                return perturbation
            else:
                return -gradient * self.attack_scale
        
        return gradient
