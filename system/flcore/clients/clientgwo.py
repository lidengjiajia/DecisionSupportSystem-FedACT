# FedGWO: Federated Grey Wolf Optimizer
# Based on: Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). 
# Grey wolf optimizer. Advances in engineering software, 69, 46-61.

import torch
import numpy as np
import time
from flcore.clients.clientbase import Client


class clientGWO(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        # 存储从服务器接收的GWO参数
        self.alpha_model = None  # θ_α: Alpha wolf (最优客户端)
        self.beta_model = None   # θ_β: Beta wolf (次优客户端)
        self.delta_model = None  # θ_δ: Delta wolf (第三优客户端)
        
        # GWO控制参数 (原始论文参数)
        self.a = None  # 线性从2递减到0
        self.A1 = None  # 向Alpha学习的系数
        self.A2 = None  # 向Beta学习的系数
        self.A3 = None  # 向Delta学习的系数
        self.C1 = None  # Alpha的权重系数
        self.C2 = None  # Beta的权重系数
        self.C3 = None  # Delta的权重系数
        
        self.current_acc = 0.0  # 当前客户端的准确率

    def train(self):
        trainloader = self.load_train_data()
        
        # Phase 0: 评估当前模型在验证集上的准确率
        self.current_acc = self.get_validation_accuracy()
        
        # Phase 1: Grey Wolf Optimization Update
        # 原始GWO论文公式:
        # D_α = |C1·X_α - X|, X1 = X_α - A1·D_α
        # D_β = |C2·X_β - X|, X2 = X_β - A2·D_β
        # D_δ = |C3·X_δ - X|, X3 = X_δ - A3·D_δ
        # X(t+1) = (X1 + X2 + X3) / 3
        
        if self.alpha_model is not None and self.beta_model is not None and self.delta_model is not None:
            with torch.no_grad():
                for param, alpha_param, beta_param, delta_param in zip(
                    self.model.parameters(),
                    self.alpha_model.parameters(),
                    self.beta_model.parameters(),
                    self.delta_model.parameters()
                ):
                    # 计算距离 D = |C × θ_leader - θ_k|
                    D_alpha = torch.abs(self.C1 * alpha_param.data - param.data)
                    D_beta = torch.abs(self.C2 * beta_param.data - param.data)
                    D_delta = torch.abs(self.C3 * delta_param.data - param.data)
                    
                    # 向三个领导者学习
                    # θ1 = θ_α - A1 × D_α
                    theta_1 = alpha_param.data - self.A1 * D_alpha
                    # θ2 = θ_β - A2 × D_β
                    theta_2 = beta_param.data - self.A2 * D_beta
                    # θ3 = θ_δ - A3 × D_δ
                    theta_3 = delta_param.data - self.A3 * D_delta
                    
                    # 三者平均作为新位置
                    param.data = (theta_1 + theta_2 + theta_3) / 3.0
        
        # Phase 2: Local Training (完整的E轮本地训练)
        self.model.train()
        start_time = time.time()
        
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)
        
        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_gwo_params(self, alpha_model, beta_model, delta_model, a, A1, A2, A3, C1, C2, C3):
        """
        设置GWO算法所需的参数
        
        Args:
            alpha_model: θ_α - Alpha wolf (最优客户端模型)
            beta_model: θ_β - Beta wolf (次优客户端模型)
            delta_model: θ_δ - Delta wolf (第三优客户端模型)
            a: 收敛因子，从2线性递减到0 (原始论文)
            A1, A2, A3: 向三个领导者学习的系数，A = 2a·r - a
            C1, C2, C3: 三个领导者的权重系数，C = 2·r
        """
        self.alpha_model = alpha_model
        self.beta_model = beta_model
        self.delta_model = delta_model
        self.a = a
        self.A1 = A1
        self.A2 = A2
        self.A3 = A3
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3

    def get_validation_accuracy(self):
        """
        在完整验证集/测试集上评估模型准确率
        
        Returns:
            accuracy: 准确率 (0~1之间的浮点数)
        """
        self.model.eval()
        
        test_acc = 0
        test_num = 0
        
        with torch.no_grad():
            for x, y in self.load_test_data():
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]
        
        accuracy = test_acc / test_num if test_num > 0 else 0.0
        
        return accuracy

    def train_metrics(self):
        """
        训练指标统计
        """
        trainloader = self.load_train_data()
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num
