# FedGWO: Federated Grey Wolf Optimizer
# Server implementation
# Based on: Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). 
# Grey wolf optimizer. Advances in engineering software, 69, 46-61.

import time
import numpy as np
import copy
import torch
from flcore.clients.clientgwo import clientGWO
from flcore.servers.serverbase import Server
from threading import Thread


class FedGWO(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # 选择客户端
        self.set_slow_clients()
        self.set_clients(clientGWO)

        print(f"\n加入 {self.num_clients} 个客户端!")
        print("=" * 50)
        print(f"算法: FedGWO (Federated Grey Wolf Optimizer)")
        print(f"基于: Mirjalili et al. (2014) - Advances in Engineering Software")
        print(f"总轮数: {self.global_rounds}")
        print(f"每轮参与客户端数: {self.num_join_clients}")
        print(f"客户端本地训练轮数: {self.local_epochs}")
        print(f"客户端学习率: {self.learning_rate}")
        print("=" * 50)
        print("GWO参数设定 (原始论文):")
        print("  - a: 从2线性递减到0 (收敛因子)")
        print("  - A = 2a·r - a, r∈[0,1] (探索/利用控制)")
        print("  - C = 2·r, r∈[0,1] (权重系数)")
        print("  - 分层结构: α(最优) > β(次优) > δ(第三优)")
        print("=" * 50)
        
        self.Budget = []
        
        # GWO相关记录
        self.client_accuracies = {}  # 记录每个客户端的准确率
        self.alpha_id = None  # Alpha wolf (最优客户端)
        self.beta_id = None   # Beta wolf (次优客户端)
        self.delta_id = None  # Delta wolf (第三优客户端)
        self.alpha_acc = 0.0
        self.beta_acc = 0.0
        self.delta_acc = 0.0

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            
            # Step 1: 发送全局模型给所有参与客户端
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n------------- Round {i} -------------")
                print("评估全局模型...")
                self.evaluate()

            # Step 2: 所有参与客户端评估当前准确率
            print(f"Round {i}: 收集客户端准确率...")
            for client in self.selected_clients:
                client.current_acc = client.get_validation_accuracy()
                self.client_accuracies[client.id] = client.current_acc
            
            # Step 3: 找出α, β, δ三个领导者 (top-3客户端)
            self.find_leaders()
            
            # Step 4: 计算收敛因子 a (原始论文: 从2线性递减到0)
            a = 2.0 - 2.0 * i / self.global_rounds
            
            print(f"Round {i}: a={a:.4f}")
            print(f"  α客户端: ID={self.alpha_id}, acc={self.alpha_acc:.4f}")
            print(f"  β客户端: ID={self.beta_id}, acc={self.beta_acc:.4f}")
            print(f"  δ客户端: ID={self.delta_id}, acc={self.delta_acc:.4f}")
            
            # Step 5: 为每个客户端分配GWO参数并训练
            for client in self.selected_clients:
                # 5.1 获取α, β, δ三个领导者的模型
                alpha_client = self.clients[self.alpha_id]
                beta_client = self.clients[self.beta_id]
                delta_client = self.clients[self.delta_id]
                
                alpha_model = copy.deepcopy(alpha_client.model)
                beta_model = copy.deepcopy(beta_client.model)
                delta_model = copy.deepcopy(delta_client.model)
                
                # 5.2 生成随机系数 (原始论文公式)
                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2.0 * a * r1 - a  # A ∈ [-a, a]
                C1 = 2.0 * r2          # C ∈ [0, 2]
                
                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2.0 * a * r1 - a
                C2 = 2.0 * r2
                
                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2.0 * a * r1 - a
                C3 = 2.0 * r2
                
                # 5.3 设置GWO参数
                client.set_gwo_params(
                    alpha_model=alpha_model,
                    beta_model=beta_model,
                    delta_model=delta_model,
                    a=a,
                    A1=A1, A2=A2, A3=A3,
                    C1=C1, C2=C2, C3=C3
                )
            
            # Step 6: 执行GWO训练
            print(f"Round {i}: 客户端执行GWO训练...")
            for client in self.selected_clients:
                client.train()

            # Step 7: 接收客户端模型
            self.receive_models()
            
            # Step 8: 聚合模型（使用标准FedAvg）
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print(f"Round {i} 完成，耗时: {self.Budget[-1]:.2f}s")
            print('-' * 50)

        print("\n训练完成!")
        print("最终评估...")
        self.evaluate()

        self.save_results()
        self.save_global_model()

    def find_leaders(self):
        """
        找出准确率最高的前三个客户端作为α, β, δ领导者
        原始GWO论文: Alpha (最优), Beta (次优), Delta (第三优)
        """
        # 按准确率排序
        sorted_clients = sorted(self.client_accuracies.items(), 
                               key=lambda x: x[1], reverse=True)
        
        if len(sorted_clients) >= 3:
            self.alpha_id, self.alpha_acc = sorted_clients[0]
            self.beta_id, self.beta_acc = sorted_clients[1]
            self.delta_id, self.delta_acc = sorted_clients[2]
        elif len(sorted_clients) == 2:
            self.alpha_id, self.alpha_acc = sorted_clients[0]
            self.beta_id, self.beta_acc = sorted_clients[1]
            self.delta_id, self.delta_acc = sorted_clients[1]  # Beta和Delta相同
        elif len(sorted_clients) == 1:
            self.alpha_id, self.alpha_acc = sorted_clients[0]
            self.beta_id, self.beta_acc = sorted_clients[0]
            self.delta_id, self.delta_acc = sorted_clients[0]  # 全部相同
        else:
            # 默认使用第一个客户端
            self.alpha_id = self.beta_id = self.delta_id = 0
            self.alpha_acc = self.beta_acc = self.delta_acc = 0.0

    def receive_models(self):
        """
        接收客户端模型
        """
        assert (len(self.selected_clients) > 0)

        active_clients = np.random.choice(self.selected_clients, 
                                         int((1 - self.client_drop_rate) * self.current_num_join_clients), 
                                         replace=False)

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        """
        聚合客户端模型参数（标准FedAvg加权平均）
        """
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)
