# FedPSO: Federated Particle Swarm Optimization
# Server implementation
# Based on: Kennedy, J., & Eberhart, R. (1995). 
# Particle swarm optimization. Proceedings of ICNN'95-international conference on neural networks (Vol. 4, pp. 1942-1948). IEEE.
# Citations: 48,000+ (Google Scholar) - One of the most influential optimization algorithms

import time
import numpy as np
import copy
import torch
from flcore.clients.clientpso import clientPSO
from flcore.servers.serverbase import Server
from threading import Thread


class FedPSO(Server):
    """
    FedPSO: 联邦粒子群优化算法
    
    核心思想：
    - 每个客户端作为一个粒子，在解空间中搜索最优模型参数
    - 每个粒子有位置(position)和速度(velocity)
    - 粒子同时受到个体最优(pbest)和全局最优(gbest)的引导
    - 通过速度更新公式实现探索与利用的平衡
    
    原始PSO公式（Kennedy & Eberhart, 1995）：
    v_i(t+1) = w*v_i(t) + c1*r1*(pbest_i - x_i(t)) + c2*r2*(gbest - x_i(t))
    x_i(t+1) = x_i(t) + v_i(t+1)
    
    其中：
    - w: 惯性权重（inertia weight），控制前一次速度的影响
    - c1: 个体学习因子（cognitive parameter），控制个体历史最优的影响
    - c2: 社会学习因子（social parameter），控制群体全局最优的影响
    - r1, r2: [0,1]之间的随机数，增加随机性
    """
    
    def __init__(self, args, times):
        super().__init__(args, times)

        # 选择客户端
        self.set_slow_clients()
        self.set_clients(clientPSO)

        print(f"\n加入 {self.num_clients} 个客户端!")
        print("=" * 70)
        print(f"算法: FedPSO (Federated Particle Swarm Optimization)")
        print(f"基于: Kennedy & Eberhart (1995) - ICNN")
        print(f"引用次数: 48,000+ (Google Scholar)")
        print(f"总轮数: {self.global_rounds}")
        print(f"每轮参与客户端数: {self.num_join_clients}")
        print(f"客户端本地训练轮数: {self.local_epochs}")
        print(f"客户端学习率: {self.learning_rate}")
        print("=" * 70)
        print("PSO参数设定 (原始论文):")
        print(f"  - 惯性权重 w: 从0.9线性递减到0.4 (Shi & Eberhart, 1998改进)")
        print(f"  - 个体学习因子 c1: 2.0 (认知成分)")
        print(f"  - 社会学习因子 c2: 2.0 (社会成分)")
        print(f"  - 速度限制 Vmax: 控制搜索步长")
        print("=" * 70)
        print("PSO工作流程:")
        print("  1. 初始化粒子位置和速度")
        print("  2. 评估每个粒子的适应度")
        print("  3. 更新个体最优(pbest)和全局最优(gbest)")
        print("  4. 根据PSO公式更新速度和位置")
        print("  5. 重复步骤2-4直到收敛")
        print("=" * 70)
        
        self.Budget = []
        
        # PSO核心参数（基于原始论文和Shi & Eberhart 1998改进）
        self.w_max = 0.9  # 惯性权重最大值
        self.w_min = 0.4  # 惯性权重最小值
        self.c1 = 2.0     # 个体学习因子（认知）
        self.c2 = 2.0     # 社会学习因子（社会）
        
        # PSO状态记录
        self.gbest_model = None  # 全局最优模型参数
        self.gbest_acc = 0.0     # 全局最优准确率
        self.gbest_client_id = None  # 全局最优客户端ID
        
        # 每个客户端的个体最优
        self.pbest_models = {}   # {client_id: pbest_model}
        self.pbest_accs = {}     # {client_id: pbest_acc}
        
        # 速度（每个客户端的速度向量）
        self.velocities = {}     # {client_id: velocity_params}
        
        # 适应度历史
        self.fitness_history = []
        
        print(f"初始化完成！")
        print(f"PSO参数: w∈[{self.w_min}, {self.w_max}], c1={self.c1}, c2={self.c2}")
        print("=" * 70)

    def train(self):
        """
        FedPSO训练流程
        """
        # 初始化：评估所有客户端的初始适应度
        print("\n" + "=" * 70)
        print("阶段0: 初始化PSO粒子群")
        print("=" * 70)
        
        self.send_models()
        for client in self.clients:
            client.current_acc = client.get_validation_accuracy()
            # 初始化个体最优
            self.pbest_models[client.id] = copy.deepcopy([p.data.clone() for p in client.model.parameters()])
            self.pbest_accs[client.id] = client.current_acc
            # 初始化速度为0
            self.velocities[client.id] = [torch.zeros_like(p.data) for p in client.model.parameters()]
        
        # 初始化全局最优
        best_client = max(self.clients, key=lambda c: self.pbest_accs[c.id])
        self.gbest_client_id = best_client.id
        self.gbest_acc = self.pbest_accs[best_client.id]
        self.gbest_model = copy.deepcopy(self.pbest_models[best_client.id])
        
        print(f"✨ 初始全局最优: 客户端 {self.gbest_client_id}, 准确率: {self.gbest_acc:.4f}")
        
        # PSO主循环
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            
            # 计算当前轮的惯性权重 w（线性递减）
            # w(t) = w_max - (w_max - w_min) * t / T
            current_w = self.w_max - (self.w_max - self.w_min) * i / self.global_rounds
            
            print(f"\n{'=' * 70}")
            print(f"Round {i}/{self.global_rounds}")
            print(f"{'=' * 70}")
            print(f"PSO参数: w={current_w:.4f}, c1={self.c1}, c2={self.c2}")
            print(f"当前全局最优: 客户端 {self.gbest_client_id}, 准确率: {self.gbest_acc:.4f}")
            
            # Step 1: 发送模型给选中的客户端
            self.send_models()

            # Step 2: 定期评估全局模型
            if i % self.eval_gap == 0:
                print(f"\n评估全局模型 (Round {i})...")
                self.evaluate()

            # Step 3: 更新每个粒子的速度和位置
            print(f"\nRound {i}: 执行PSO速度和位置更新...")
            for client in self.selected_clients:
                # 生成随机数
                r1 = np.random.rand()
                r2 = np.random.rand()
                
                # 传递PSO参数给客户端
                client.set_pso_parameters(
                    w=current_w,
                    c1=self.c1,
                    c2=self.c2,
                    r1=r1,
                    r2=r2,
                    pbest_model=self.pbest_models[client.id],
                    gbest_model=self.gbest_model,
                    velocity=self.velocities[client.id]
                )
            
            # Step 4: 客户端训练（PSO更新 + 梯度下降微调）
            for client in self.selected_clients:
                client.train()
            
            # Step 5: 评估每个客户端的新适应度
            print(f"\nRound {i}: 评估粒子适应度...")
            fitness_values = []
            for client in self.selected_clients:
                client.current_acc = client.get_validation_accuracy()
                fitness_values.append(client.current_acc)
                
                # 更新个体最优 pbest
                if client.current_acc > self.pbest_accs[client.id]:
                    self.pbest_accs[client.id] = client.current_acc
                    self.pbest_models[client.id] = copy.deepcopy([p.data.clone() for p in client.model.parameters()])
                    print(f"  ✓ 客户端 {client.id} 更新个体最优: {client.current_acc:.4f}")
                
                # 更新全局最优 gbest
                if client.current_acc > self.gbest_acc:
                    self.gbest_acc = client.current_acc
                    self.gbest_client_id = client.id
                    self.gbest_model = copy.deepcopy([p.data.clone() for p in client.model.parameters()])
                    print(f"  🌟 发现新的全局最优! 客户端 {client.id}, 准确率: {self.gbest_acc:.4f}")
                
                # 保存更新后的速度
                self.velocities[client.id] = copy.deepcopy([v.data.clone() for v in client.velocity])
            
            # 记录本轮适应度统计
            if len(fitness_values) > 0:
                self.fitness_history.append({
                    'round': i,
                    'best': max(fitness_values),
                    'worst': min(fitness_values),
                    'mean': np.mean(fitness_values),
                    'std': np.std(fitness_values),
                    'gbest': self.gbest_acc
                })
            
            # Step 6: 接收模型
            self.receive_models()
            
            # Step 7: 聚合模型（使用全局最优作为全局模型）
            # 将gbest模型设置为全局模型
            for global_param, gbest_param in zip(self.global_model.parameters(), self.gbest_model):
                global_param.data = gbest_param.clone()

            self.Budget.append(time.time() - s_t)
            print(f"\nRound {i} 完成，耗时: {self.Budget[-1]:.2f}s")
            print(f"适应度统计 - 最优: {max(fitness_values):.4f}, 最差: {min(fitness_values):.4f}, 平均: {np.mean(fitness_values):.4f}")
            print('-' * 70)

        print("\n" + "=" * 70)
        print("训练完成!")
        print("=" * 70)
        print(f"最终全局最优客户端: {self.gbest_client_id}")
        print(f"最终全局最优准确率: {self.gbest_acc:.4f}")
        print(f"总训练时间: {sum(self.Budget):.2f}s")
        print("=" * 70)
        
        print("\n最终评估...")
        self.evaluate()

        # 保存PSO适应度历史
        self.save_pso_history()
        self.save_results()
        self.save_global_model()

    def receive_models(self):
        """
        接收客户端模型
        """
        assert (len(self.selected_clients) > 0)

        active_clients = np.random.choice(
            self.selected_clients, 
            int((1 - self.client_drop_rate) * self.current_num_join_clients), 
            replace=False
        )

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

    def save_pso_history(self):
        """
        保存PSO适应度历史到文件
        """
        import h5py
        import os
        
        if len(self.fitness_history) == 0:
            return
        
        # 创建results目录（与serverbase保持一致）
        algo_folder = f"{self.dataset}_FedPSO_{self.goal}"
        result_dir = os.path.join("system", "results", algo_folder)
        os.makedirs(result_dir, exist_ok=True)
        
        # 保存文件名
        filename = os.path.join(result_dir, f"{self.dataset}_FedPSO_pso_history_{self.times}.h5")
        
        with h5py.File(filename, 'w') as f:
            # 保存PSO参数
            f.attrs['w_max'] = self.w_max
            f.attrs['w_min'] = self.w_min
            f.attrs['c1'] = self.c1
            f.attrs['c2'] = self.c2
            f.attrs['global_rounds'] = self.global_rounds
            f.attrs['num_clients'] = self.num_clients
            f.attrs['gbest_acc'] = self.gbest_acc
            f.attrs['gbest_client_id'] = self.gbest_client_id
            
            # 保存适应度历史
            rounds = [h['round'] for h in self.fitness_history]
            best_fits = [h['best'] for h in self.fitness_history]
            worst_fits = [h['worst'] for h in self.fitness_history]
            mean_fits = [h['mean'] for h in self.fitness_history]
            std_fits = [h['std'] for h in self.fitness_history]
            gbest_fits = [h['gbest'] for h in self.fitness_history]
            
            f.create_dataset('rounds', data=rounds)
            f.create_dataset('best_fitness', data=best_fits)
            f.create_dataset('worst_fitness', data=worst_fits)
            f.create_dataset('mean_fitness', data=mean_fits)
            f.create_dataset('std_fitness', data=std_fits)
            f.create_dataset('gbest_fitness', data=gbest_fits)
        
        print(f"\n✓ PSO历史已保存到: {filename}")
