"""
FedTLBO Client: 联邦TLBO客户端

支持多种拜占庭攻击模拟（实验用）

攻击实现参考:
- ICML 2018: "The Hidden Vulnerability of Distributed Learning in Byzantium"
- NeurIPS 2019: "A Little Is Enough: Circumventing Defenses For Distributed Learning"
- IEEE S&P 2020: "Analyzing Federated Learning through an Adversarial Lens"

作者: FedTLBO Team
日期: 2026-01-22
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from typing import Optional, Dict, List

from flcore.clients.clientbase import Client
from flcore.attack.attacks import ByzantineAttacker, ATTACK_REGISTRY


class clientTLBO(Client):
    """
    FedTLBO客户端
    
    功能:
    1. 本地训练
    2. 梯度计算
    3. 拜占庭攻击模拟 (实验用)
    
    支持的攻击模式（见 flcore/attack/attacks.py）:
    - 基础攻击: sign_flip, gaussian, scale
    - 前沿攻击: little, alie, ipm, minmax, trim_attack
    - 其他攻击: label_flip, backdoor, free_rider, collision
    """
    
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        # 攻击配置
        self.is_malicious = False
        self.attack_mode = 'none'
        self.attack_scale = 1.0
        self.attacker: Optional[ByzantineAttacker] = None
        
        # 协同攻击相关
        self.benign_grads: List[torch.Tensor] = []  # 诚实客户端梯度
        
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
        
        # 创建攻击器
        if mode in ATTACK_REGISTRY:
            self.attacker = ByzantineAttacker(mode, scale=scale)
    
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
        
        # 计算梯度 (参数更新量)
        grads = []
        for old_p, new_p in zip(old_params, self.model.parameters()):
            grads.append((new_p.data - old_p).flatten())
        gradient = torch.cat(grads)
        
        # 攻击处理
        if self.is_malicious and self.attacker is not None:
            gradient = self.attacker.attack(gradient, self.benign_grads)
        
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
