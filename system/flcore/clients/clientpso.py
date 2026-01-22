# FedPSO: Federated Particle Swarm Optimization
# Client implementation
# Based on: Kennedy, J., & Eberhart, R. (1995). 
# Particle swarm optimization. Proceedings of ICNN'95-international conference on neural networks (Vol. 4, pp. 1942-1948). IEEE.
# Citations: 48,000+ (Google Scholar)

import torch
import numpy as np
import time
import copy
from flcore.clients.clientbase import Client


class clientPSO(Client):
    """
    FedPSO客户端：模拟PSO中的一个粒子
    
    每个粒子维护：
    - 位置 (position): 当前模型参数
    - 速度 (velocity): 参数更新方向和幅度
    - 个体最优 (pbest): 该粒子历史最优位置
    - 全局最优 (gbest): 所有粒子中的最优位置（由服务器提供）
    
    PSO速度更新公式：
    v_i(t+1) = w*v_i(t) + c1*r1*(pbest_i - x_i(t)) + c2*r2*(gbest - x_i(t))
    
    位置更新公式：
    x_i(t+1) = x_i(t) + v_i(t+1)
    """
    
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        # PSO粒子状态
        self.velocity = [torch.zeros_like(p.data) for p in self.model.parameters()]  # 速度向量
        self.pbest_model = None  # 个体最优模型参数
        self.gbest_model = None  # 全局最优模型参数
        
        # PSO参数（由服务器设置）
        self.w = 0.9   # 惯性权重
        self.c1 = 2.0  # 个体学习因子
        self.c2 = 2.0  # 社会学习因子
        self.r1 = 0.5  # 随机数1
        self.r2 = 0.5  # 随机数2
        
        # 速度限制（防止速度过大导致发散）
        self.v_max = 0.5  # 最大速度（相对于参数值）
        
        # 适应度（准确率）
        self.current_acc = 0.0
        self.pbest_acc = 0.0

    def train(self):
        """
        PSO训练流程：
        1. PSO速度和位置更新
        2. 梯度下降微调（结合传统训练）
        """
        trainloader = self.load_train_data()
        self.model.train()
        
        start_time = time.time()

        # Step 1: PSO位置更新（在传统训练前）
        if self.pbest_model is not None and self.gbest_model is not None:
            self.pso_update()
        
        # Step 2: 传统梯度下降训练（微调）
        max_local_epochs = self.local_epochs
        
        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def pso_update(self):
        """
        PSO核心更新公式
        
        原始论文公式：
        v_i(t+1) = w*v_i(t) + c1*r1*(pbest_i - x_i(t)) + c2*r2*(gbest - x_i(t))
        x_i(t+1) = x_i(t) + v_i(t+1)
        
        三个成分：
        1. 惯性成分: w*v_i(t) - 保持之前的搜索方向
        2. 认知成分: c1*r1*(pbest_i - x_i(t)) - 向个体历史最优学习
        3. 社会成分: c2*r2*(gbest - x_i(t)) - 向群体全局最优学习
        """
        with torch.no_grad():
            # 遍历模型的每一层参数
            for i, (param, vel, pbest_param, gbest_param) in enumerate(
                zip(self.model.parameters(), self.velocity, self.pbest_model, self.gbest_model)
            ):
                # 当前位置 x_i(t)
                x_current = param.data
                
                # 个体最优位置 pbest_i
                x_pbest = pbest_param
                
                # 全局最优位置 gbest
                x_gbest = gbest_param
                
                # 计算速度更新的三个成分
                # 1. 惯性成分: w * v_i(t)
                inertia = self.w * vel
                
                # 2. 认知成分（个体学习）: c1 * r1 * (pbest_i - x_i(t))
                cognitive = self.c1 * self.r1 * (x_pbest - x_current)
                
                # 3. 社会成分（群体学习）: c2 * r2 * (gbest - x_i(t))
                social = self.c2 * self.r2 * (x_gbest - x_current)
                
                # 速度更新：v_i(t+1) = inertia + cognitive + social
                new_velocity = inertia + cognitive + social
                
                # 速度限制（防止速度过大）
                # 论文中通常限制 V_max = k * X_range，这里使用参数值的比例
                param_range = torch.abs(x_current).mean() + 1e-8  # 避免除零
                v_max = self.v_max * param_range
                new_velocity = torch.clamp(new_velocity, -v_max, v_max)
                
                # 更新速度
                vel.data = new_velocity
                
                # 位置更新：x_i(t+1) = x_i(t) + v_i(t+1)
                param.data = x_current + new_velocity
            
            # 更新速度列表引用
            self.velocity = [v.clone() for v in self.velocity]

    def set_pso_parameters(self, w, c1, c2, r1, r2, pbest_model, gbest_model, velocity):
        """
        设置PSO参数（由服务器传入）
        
        Args:
            w: 惯性权重（inertia weight）
            c1: 个体学习因子（cognitive parameter）
            c2: 社会学习因子（social parameter）
            r1: 随机数1，范围[0, 1]
            r2: 随机数2，范围[0, 1]
            pbest_model: 个体最优模型参数列表
            gbest_model: 全局最优模型参数列表
            velocity: 当前速度
        """
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.r1 = r1
        self.r2 = r2
        self.pbest_model = pbest_model
        self.gbest_model = gbest_model
        self.velocity = velocity

    def get_validation_accuracy(self):
        """
        计算验证集准确率（适应度函数）
        
        在PSO中，适应度函数通常是最小化损失或最大化准确率
        这里使用准确率作为适应度（越大越好）
        
        Returns:
            float: 验证集准确率
        """
        self.model.eval()
        
        test_acc = 0
        test_num = 0
        
        with torch.no_grad():
            testloaderfull = self.load_test_data()
            for x, y in testloaderfull:
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

    def update_pbest(self, new_acc, new_model_params):
        """
        更新个体最优
        
        Args:
            new_acc: 新的准确率
            new_model_params: 新的模型参数列表
        """
        if new_acc > self.pbest_acc:
            self.pbest_acc = new_acc
            self.pbest_model = copy.deepcopy(new_model_params)
            return True
        return False

    def get_fitness(self):
        """
        获取当前适应度值
        
        Returns:
            float: 当前准确率（适应度）
        """
        return self.current_acc

    def reset_velocity(self):
        """
        重置速度为零（在需要时使用）
        """
        self.velocity = [torch.zeros_like(p.data) for p in self.model.parameters()]

    def set_velocity_limit(self, v_max):
        """
        设置速度限制
        
        Args:
            v_max: 最大速度比例
        """
        self.v_max = v_max
