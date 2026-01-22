#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
信用评分神经网络模型

针对UCI和Xinwang两个信用评分数据集设计的深度神经网络。
采用残差连接和BatchNorm提升训练稳定性。

参考文献:
- Kolmogorov-Arnold Networks for Credit Default Prediction
- Hybrid Model of KAN and gMLP for Large-Scale Financial Data  
- Monotonic Neural Additive Models for Credit Scoring
"""

import torch
import torch.nn as nn


class CreditNet(nn.Module):
    """
    通用信用评分神经网络
    
    支持的数据集类型:
    - uci: UCI信用卡违约数据集 (23特征, 30000样本)
    - xinwang: 新网银行信用数据集 (38特征, 17886样本)
    
    网络结构特点:
    - 残差连接: 缓解梯度消失，支持更深网络
    - BatchNorm: 加速收敛，提高泛化能力
    - Dropout: 防止过拟合
    """
    
    def __init__(self, input_dim, dataset_type='uci', num_classes=2):
        super(CreditNet, self).__init__()
        
        self.dataset_type = dataset_type
        self.num_classes = num_classes
        
        if dataset_type == 'uci':
            # UCI数据集: 深层网络结构
            self.layers = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.5),
                
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.45),
                
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.4),
                
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.35),
                
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.3),
                
                nn.Linear(32, 16),
                nn.ReLU()
            )
            self.fc = nn.Linear(16, num_classes)
            
        elif dataset_type == 'xinwang':
            # Xinwang数据集: 残差网络结构
            self.input_layer = nn.Linear(input_dim, 512)
            self.bn1 = nn.BatchNorm1d(512)
            
            # 第一个残差块
            self.fc1 = nn.Linear(512, 384)
            self.bn2 = nn.BatchNorm1d(384)
            self.fc2 = nn.Linear(384, 384)
            self.bn3 = nn.BatchNorm1d(384)
            self.shortcut1 = nn.Linear(512, 384)
            
            # 第二个残差块
            self.fc3 = nn.Linear(384, 256)
            self.bn4 = nn.BatchNorm1d(256)
            self.fc4 = nn.Linear(256, 256)
            self.bn5 = nn.BatchNorm1d(256)
            self.shortcut2 = nn.Linear(384, 256)
            
            # 第三个残差块
            self.fc5 = nn.Linear(256, 128)
            self.bn6 = nn.BatchNorm1d(128)
            self.fc6 = nn.Linear(128, 128)
            self.bn7 = nn.BatchNorm1d(128)
            self.shortcut3 = nn.Linear(256, 128)
            
            # 分类层
            self.fc7 = nn.Linear(128, 64)
            self.bn8 = nn.BatchNorm1d(64)
            self.fc8 = nn.Linear(64, 32)
            self.fc = nn.Linear(32, num_classes)
            
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.4)
        
        else:
            raise ValueError(f"不支持的数据集类型: {dataset_type}")
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """Xavier权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        if self.dataset_type == 'uci':
            x = self.layers(x)
            x = self.fc(x)
        else:
            # Xinwang残差网络
            x = self.relu(self.bn1(self.input_layer(x)))
            x = self.dropout(x)
            
            # 残差块1
            identity = self.shortcut1(x)
            x = self.relu(self.bn2(self.fc1(x)))
            x = self.dropout(x)
            x = self.bn3(self.fc2(x))
            x = self.relu(x + identity)
            x = self.dropout(x)
            
            # 残差块2
            identity = self.shortcut2(x)
            x = self.relu(self.bn4(self.fc3(x)))
            x = self.dropout(x)
            x = self.bn5(self.fc4(x))
            x = self.relu(x + identity)
            x = self.dropout(x)
            
            # 残差块3
            identity = self.shortcut3(x)
            x = self.relu(self.bn6(self.fc5(x)))
            x = self.dropout(x)
            x = self.bn7(self.fc6(x))
            x = self.relu(x + identity)
            x = self.dropout(x)
            
            # 分类层
            x = self.relu(self.bn8(self.fc7(x)))
            x = self.dropout(x)
            x = self.relu(self.fc8(x))
            x = self.fc(x)
        
        return x


class UciCreditNet(CreditNet):
    """
    UCI信用卡违约预测网络
    
    数据集信息:
    - 样本数: 30000
    - 特征数: 23
    - 类别: 2 (违约/正常)
    - 不平衡比例: 约22%违约
    """
    
    def __init__(self, input_dim=23, num_classes=2):
        super().__init__(input_dim, dataset_type='uci', num_classes=num_classes)


class XinwangCreditNet(CreditNet):
    """
    新网银行信用评估网络
    
    数据集信息:
    - 样本数: 17886
    - 特征数: 38 (TOAD筛选后)
    - 类别: 2 (好/坏信用)
    - 高维特征需要更深网络
    """
    
    def __init__(self, input_dim=38, num_classes=2):
        super().__init__(input_dim, dataset_type='xinwang', num_classes=num_classes)


class BaseHeadSplit(nn.Module):
    """
    基础-头部分离模块
    
    用于联邦学习中分离特征提取器(base)和分类器(head)，
    支持个性化联邦学习方法。
    """
    
    def __init__(self, base, head):
        super().__init__()
        self.base = base
        self.head = head
    
    def forward(self, x):
        features = self.base(x)
        out = self.head(features)
        return out
