"""
防御机制实现模块

包含多种经典和前沿的拜占庭防御方法:

基础防御:
- FedAvg: 简单平均 (无防御)
- Median: 中位数聚合
- TrimmedMean: 修剪均值
- Krum: Krum算法

前沿防御 (顶会论文):
- MultiKrum: Multi-Krum (NeurIPS 2017)
- Bulyan: Bulyan算法 (ICML 2018)
- FLTrust: 基于可信数据的防御 (NDSS 2021)
- SignSGD: 基于符号的聚合
- NormBound: 范数裁剪
- RFA: Robust Federated Averaging

我们的方法:
- TLBO: Teaching-Learning Based Optimization聚合

参考文献:
[1] Blanchard et al., "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent", NeurIPS 2017
[2] El Mhamdi et al., "The Hidden Vulnerability of Distributed Learning in Byzantium", ICML 2018
[3] Cao et al., "FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping", NDSS 2021
[4] Yin et al., "Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates", ICML 2018

作者: FedTLBO Team
日期: 2026-01-22
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Dict, Tuple
from abc import ABC, abstractmethod


# ==============================================================================
# 防御基类
# ==============================================================================

class BaseDefense(ABC):
    """防御基类"""
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    @abstractmethod
    def aggregate(
        self,
        gradients: List[torch.Tensor],
        weights: Optional[List[float]] = None
    ) -> torch.Tensor:
        """聚合梯度"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """防御名称"""
        pass


# ==============================================================================
# 基础防御
# ==============================================================================

class FedAvgDefense(BaseDefense):
    """
    FedAvg - 简单加权平均 (无防御)
    
    作为基线，直接对所有梯度进行加权平均。
    不具备任何拜占庭容错能力。
    """
    
    @property
    def name(self) -> str:
        return "fedavg"
    
    def aggregate(
        self,
        gradients: List[torch.Tensor],
        weights: Optional[List[float]] = None
    ) -> torch.Tensor:
        if not gradients:
            raise ValueError("No gradients to aggregate")
        
        if weights is None:
            weights = [1.0 / len(gradients)] * len(gradients)
        else:
            total = sum(weights)
            weights = [w / total for w in weights]
        
        return sum(w * g for w, g in zip(weights, gradients))


class MedianDefense(BaseDefense):
    """
    中位数聚合
    
    对每个维度取中位数，抵抗异常值。
    理论上可以容忍不超过50%的恶意客户端。
    """
    
    @property
    def name(self) -> str:
        return "median"
    
    def aggregate(
        self,
        gradients: List[torch.Tensor],
        weights: Optional[List[float]] = None
    ) -> torch.Tensor:
        if not gradients:
            raise ValueError("No gradients to aggregate")
        
        stacked = torch.stack(gradients)
        return torch.median(stacked, dim=0).values


class TrimmedMeanDefense(BaseDefense):
    """
    修剪均值聚合
    
    去除最大和最小的若干值后取平均。
    trim_ratio控制修剪比例，默认0.1表示两端各去掉10%。
    """
    
    def __init__(self, trim_ratio: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.trim_ratio = trim_ratio
    
    @property
    def name(self) -> str:
        return "trimmed_mean"
    
    def aggregate(
        self,
        gradients: List[torch.Tensor],
        weights: Optional[List[float]] = None
    ) -> torch.Tensor:
        if not gradients:
            raise ValueError("No gradients to aggregate")
        
        n = len(gradients)
        k = max(1, int(n * self.trim_ratio))
        
        stacked = torch.stack(gradients)
        sorted_grads, _ = torch.sort(stacked, dim=0)
        
        # 去除最大和最小的k个
        trimmed = sorted_grads[k:n-k]
        
        if len(trimmed) == 0:
            return sorted_grads.mean(dim=0)
        
        return trimmed.mean(dim=0)


class KrumDefense(BaseDefense):
    """
    Krum算法 - NeurIPS 2017
    
    论文: "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"
    
    选择与其他梯度距离之和最小的梯度。
    可以容忍 f < n/2 - 1 个恶意客户端。
    """
    
    def __init__(self, num_byzantine: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.num_byzantine = num_byzantine
    
    @property
    def name(self) -> str:
        return "krum"
    
    def aggregate(
        self,
        gradients: List[torch.Tensor],
        weights: Optional[List[float]] = None
    ) -> torch.Tensor:
        if not gradients:
            raise ValueError("No gradients to aggregate")
        
        n = len(gradients)
        f = self.num_byzantine if self.num_byzantine > 0 else n // 4
        
        # 计算两两距离
        distances = torch.zeros(n, n)
        for i in range(n):
            for j in range(i + 1, n):
                dist = torch.norm(gradients[i] - gradients[j]).item()
                distances[i, j] = dist
                distances[j, i] = dist
        
        # 对每个梯度，计算到最近 n-f-1 个梯度的距离之和
        scores = []
        for i in range(n):
            sorted_dists, _ = torch.sort(distances[i])
            score = sorted_dists[1:n-f].sum()  # 排除自己
            scores.append(score)
        
        # 选择得分最小的
        best_idx = np.argmin(scores)
        return gradients[best_idx].clone()


class MultiKrumDefense(BaseDefense):
    """
    Multi-Krum算法 - NeurIPS 2017
    
    Krum的扩展，选择m个最好的梯度进行平均。
    """
    
    def __init__(self, num_byzantine: int = 0, num_select: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.num_byzantine = num_byzantine
        self.num_select = num_select
    
    @property
    def name(self) -> str:
        return "multi_krum"
    
    def aggregate(
        self,
        gradients: List[torch.Tensor],
        weights: Optional[List[float]] = None
    ) -> torch.Tensor:
        if not gradients:
            raise ValueError("No gradients to aggregate")
        
        n = len(gradients)
        f = self.num_byzantine if self.num_byzantine > 0 else n // 4
        m = self.num_select if self.num_select > 0 else n - f
        
        # 计算两两距离
        distances = torch.zeros(n, n)
        for i in range(n):
            for j in range(i + 1, n):
                dist = torch.norm(gradients[i] - gradients[j]).item()
                distances[i, j] = dist
                distances[j, i] = dist
        
        # 计算Krum得分
        scores = []
        for i in range(n):
            sorted_dists, _ = torch.sort(distances[i])
            score = sorted_dists[1:n-f].sum()
            scores.append(score)
        
        # 选择得分最小的m个
        selected_indices = np.argsort(scores)[:m]
        selected_grads = [gradients[i] for i in selected_indices]
        
        return torch.stack(selected_grads).mean(dim=0)


class BulyanDefense(BaseDefense):
    """
    Bulyan算法 - ICML 2018
    
    论文: "The Hidden Vulnerability of Distributed Learning in Byzantium"
    
    结合Krum选择和修剪均值，更强的拜占庭容错。
    先用Multi-Krum选择可信梯度，再用修剪均值聚合。
    """
    
    def __init__(self, num_byzantine: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.num_byzantine = num_byzantine
    
    @property
    def name(self) -> str:
        return "bulyan"
    
    def aggregate(
        self,
        gradients: List[torch.Tensor],
        weights: Optional[List[float]] = None
    ) -> torch.Tensor:
        if not gradients:
            raise ValueError("No gradients to aggregate")
        
        n = len(gradients)
        f = self.num_byzantine if self.num_byzantine > 0 else n // 4
        
        # 需要至少 4f + 3 个客户端
        if n < 4 * f + 3:
            # 回退到修剪均值
            return TrimmedMeanDefense(trim_ratio=0.2).aggregate(gradients, weights)
        
        # 第一步：Multi-Krum选择 n-2f 个
        selection_count = n - 2 * f
        
        # 计算距离
        distances = torch.zeros(n, n)
        for i in range(n):
            for j in range(i + 1, n):
                dist = torch.norm(gradients[i] - gradients[j]).item()
                distances[i, j] = dist
                distances[j, i] = dist
        
        scores = []
        for i in range(n):
            sorted_dists, _ = torch.sort(distances[i])
            score = sorted_dists[1:n-f].sum()
            scores.append(score)
        
        selected_indices = np.argsort(scores)[:selection_count]
        selected_grads = [gradients[i] for i in selected_indices]
        
        # 第二步：修剪均值
        stacked = torch.stack(selected_grads)
        sorted_grads, _ = torch.sort(stacked, dim=0)
        
        trim_k = f
        trimmed = sorted_grads[trim_k:-trim_k] if trim_k > 0 else sorted_grads
        
        return trimmed.mean(dim=0)


# ==============================================================================
# 前沿防御
# ==============================================================================

class FLTrustDefense(BaseDefense):
    """
    FLTrust - NDSS 2021
    
    论文: "FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping"
    
    使用服务器端可信数据计算基准梯度，
    根据客户端梯度与基准梯度的相似度分配权重。
    """
    
    def __init__(self, server_gradient: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(**kwargs)
        self.server_gradient = server_gradient
    
    @property
    def name(self) -> str:
        return "fltrust"
    
    def set_server_gradient(self, gradient: torch.Tensor):
        """设置服务器基准梯度"""
        self.server_gradient = gradient
    
    def aggregate(
        self,
        gradients: List[torch.Tensor],
        weights: Optional[List[float]] = None
    ) -> torch.Tensor:
        if not gradients:
            raise ValueError("No gradients to aggregate")
        
        if self.server_gradient is None:
            # 无基准梯度，使用中位数作为基准
            self.server_gradient = torch.stack(gradients).median(dim=0).values
        
        # 计算相似度并归一化
        trust_scores = []
        for g in gradients:
            # 余弦相似度
            sim = F.cosine_similarity(
                g.unsqueeze(0), 
                self.server_gradient.unsqueeze(0)
            ).item()
            # ReLU确保非负
            trust_scores.append(max(0, sim))
        
        # 归一化权重
        total = sum(trust_scores) + 1e-10
        normalized_weights = [s / total for s in trust_scores]
        
        # 归一化梯度后聚合
        normalized_grads = []
        server_norm = self.server_gradient.norm()
        
        for g, w in zip(gradients, normalized_weights):
            if w > 0:
                # 归一化到与服务器梯度相同范数
                g_norm = g / (g.norm() + 1e-10) * server_norm
                normalized_grads.append(w * g_norm)
        
        if not normalized_grads:
            return self.server_gradient.clone()
        
        return sum(normalized_grads)


class SignSGDDefense(BaseDefense):
    """
    SignSGD聚合
    
    使用多数投票决定每个维度的符号，然后用固定步长更新。
    天然抵抗大幅度异常值。
    """
    
    @property
    def name(self) -> str:
        return "signsgd"
    
    def aggregate(
        self,
        gradients: List[torch.Tensor],
        weights: Optional[List[float]] = None
    ) -> torch.Tensor:
        if not gradients:
            raise ValueError("No gradients to aggregate")
        
        # 符号投票
        signs = torch.stack([torch.sign(g) for g in gradients])
        majority_sign = torch.sign(signs.sum(dim=0))
        
        # 使用平均范数
        avg_norm = sum(g.norm() for g in gradients) / len(gradients)
        
        return majority_sign * avg_norm / np.sqrt(len(gradients[0]))


class NormBoundDefense(BaseDefense):
    """
    范数裁剪防御
    
    将所有梯度裁剪到相同的最大范数，然后平均。
    防止恶意客户端通过大范数梯度主导聚合。
    """
    
    def __init__(self, max_norm: float = 10.0, **kwargs):
        super().__init__(**kwargs)
        self.max_norm = max_norm
    
    @property
    def name(self) -> str:
        return "norm_bound"
    
    def aggregate(
        self,
        gradients: List[torch.Tensor],
        weights: Optional[List[float]] = None
    ) -> torch.Tensor:
        if not gradients:
            raise ValueError("No gradients to aggregate")
        
        if weights is None:
            weights = [1.0 / len(gradients)] * len(gradients)
        else:
            total = sum(weights)
            weights = [w / total for w in weights]
        
        # 范数裁剪
        clipped_grads = []
        for g in gradients:
            norm = g.norm()
            if norm > self.max_norm:
                g = g * self.max_norm / norm
            clipped_grads.append(g)
        
        return sum(w * g for w, g in zip(weights, clipped_grads))


class RFADefense(BaseDefense):
    """
    Robust Federated Averaging (RFA)
    
    使用几何中位数(geometric median)代替算术平均。
    几何中位数是使到所有点距离之和最小的点。
    """
    
    def __init__(self, max_iter: int = 100, tol: float = 1e-5, **kwargs):
        super().__init__(**kwargs)
        self.max_iter = max_iter
        self.tol = tol
    
    @property
    def name(self) -> str:
        return "rfa"
    
    def aggregate(
        self,
        gradients: List[torch.Tensor],
        weights: Optional[List[float]] = None
    ) -> torch.Tensor:
        if not gradients:
            raise ValueError("No gradients to aggregate")
        
        # Weiszfeld算法计算几何中位数
        median = torch.stack(gradients).mean(dim=0)
        
        for _ in range(self.max_iter):
            distances = [torch.norm(g - median) + 1e-10 for g in gradients]
            weights_iter = [1.0 / d for d in distances]
            total_weight = sum(weights_iter)
            
            new_median = sum(w * g for w, g in zip(weights_iter, gradients)) / total_weight
            
            if torch.norm(new_median - median) < self.tol:
                break
            median = new_median
        
        return median


# ==============================================================================
# 我们的方法: TLBO聚合
# ==============================================================================

class TLBODefense(BaseDefense):
    """
    TLBO聚合 - Teaching-Learning Based Optimization
    
    我们的创新方法，将TLBO优化算法应用于梯度聚合。
    
    核心思想:
    1. Teacher阶段: 识别最优梯度作为"教师"，其他梯度向教师学习
    2. Learner阶段: 梯度之间互相学习，好的梯度影响较差的梯度
    
    优势:
    - 自适应权重: 根据梯度质量动态调整
    - 异质性处理: 有效处理Non-IID数据
    - 鲁棒性: 优质梯度自然获得更大影响力
    
    参数:
        iterations: TLBO迭代次数
        alpha: 学习率因子
    """
    
    def __init__(self, iterations: int = 10, alpha: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.iterations = iterations
        self.alpha = alpha
    
    @property
    def name(self) -> str:
        return "tlbo"
    
    def aggregate(
        self,
        gradients: List[torch.Tensor],
        weights: Optional[List[float]] = None
    ) -> torch.Tensor:
        if not gradients:
            raise ValueError("No gradients to aggregate")
        if len(gradients) == 1:
            return gradients[0].clone()
        
        # 初始化权重
        if weights is None:
            weights = [1.0 / len(gradients)] * len(gradients)
        else:
            total = sum(weights)
            weights = [w / total for w in weights]
        
        # 加权平均作为目标
        target = sum(w * g for w, g in zip(weights, gradients))
        
        # 学习者群体
        learners = [g.clone() for g in gradients]
        
        for iteration in range(self.iterations):
            # 评估适应度（与目标的相似度）
            fitness = [
                F.cosine_similarity(l.unsqueeze(0), target.unsqueeze(0)).item() 
                for l in learners
            ]
            
            # ===== Teacher阶段 =====
            teacher_idx = np.argmax(fitness)
            teacher = learners[teacher_idx]
            
            # 计算均值
            mean_learner = torch.stack(learners).mean(dim=0)
            
            # 教学因子 (1或2)
            TF = np.random.choice([1, 2])
            
            # 更新学习者
            for i, l in enumerate(learners):
                r = np.random.random() * self.alpha
                new_learner = l + r * (teacher - TF * mean_learner)
                
                # 如果新位置更好，则接受
                new_fitness = F.cosine_similarity(
                    new_learner.unsqueeze(0), target.unsqueeze(0)
                ).item()
                
                if new_fitness > fitness[i]:
                    learners[i] = new_learner
                    fitness[i] = new_fitness
            
            # ===== Learner阶段 =====
            for i in range(len(learners)):
                # 随机选择另一个学习者
                j = np.random.choice([k for k in range(len(learners)) if k != i])
                
                r = np.random.random() * self.alpha
                
                # 向更好的学习者学习
                if fitness[j] > fitness[i]:
                    new_learner = learners[i] + r * (learners[j] - learners[i])
                else:
                    new_learner = learners[i] + r * (learners[i] - learners[j])
                
                # 如果新位置更好，则接受
                new_fitness = F.cosine_similarity(
                    new_learner.unsqueeze(0), target.unsqueeze(0)
                ).item()
                
                if new_fitness > fitness[i]:
                    learners[i] = new_learner
            
            # 更新目标（使用改进后的学习者）
            target = torch.stack(learners).mean(dim=0)
        
        return target


# ==============================================================================
# FedACT组件1: 梯度异常检测器（自编码器）
# ==============================================================================

class GradientDetector(torch.nn.Module):
    """
    基于自编码器的梯度异常检测
    
    原理:
    - 自编码器学习压缩和重构正常梯度
    - 正常梯度 → 重构误差小
    - 异常梯度 → 重构误差大（无法被正确重构）
    
    自适应特性:
    - 根据梯度维度自动调整网络结构
    - 支持配置潜在空间维度
    """
    
    # 不同数据集/模型规模的推荐配置
    CONFIGS = {
        'small': {'latent_dim': 32, 'hidden_layers': 1, 'dropout': 0.1},   # 梯度维度 < 10000
        'medium': {'latent_dim': 64, 'hidden_layers': 2, 'dropout': 0.2},  # 梯度维度 10000-100000
        'large': {'latent_dim': 128, 'hidden_layers': 3, 'dropout': 0.3},  # 梯度维度 > 100000
    }
    
    @classmethod
    def auto_config(cls, input_dim: int) -> dict:
        """根据输入维度自动选择配置"""
        if input_dim < 10000:
            return cls.CONFIGS['small']
        elif input_dim < 100000:
            return cls.CONFIGS['medium']
        else:
            return cls.CONFIGS['large']
    
    def __init__(self, input_dim: int, latent_dim: int = None, 
                 hidden_layers: int = None, dropout: float = None,
                 auto_adapt: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        
        # 自适应配置
        if auto_adapt and (latent_dim is None or hidden_layers is None):
            config = self.auto_config(input_dim)
            latent_dim = latent_dim or config['latent_dim']
            hidden_layers = hidden_layers or config['hidden_layers']
            dropout = dropout if dropout is not None else config['dropout']
        else:
            latent_dim = latent_dim or 64
            hidden_layers = hidden_layers or 2
            dropout = dropout if dropout is not None else 0.2
        
        self.latent_dim = latent_dim
        
        # 构建编码器
        encoder_layers = []
        current_dim = input_dim
        
        for i in range(hidden_layers):
            next_dim = max(latent_dim, current_dim // 4)
            encoder_layers.extend([
                torch.nn.Linear(current_dim, next_dim),
                torch.nn.LayerNorm(next_dim),
                torch.nn.LeakyReLU(0.2),
                torch.nn.Dropout(dropout),
            ])
            current_dim = next_dim
        
        encoder_layers.append(torch.nn.Linear(current_dim, latent_dim))
        self.encoder = torch.nn.Sequential(*encoder_layers)
        
        # 构建解码器（镜像结构）
        decoder_layers = []
        current_dim = latent_dim
        hidden_dims = [max(latent_dim, input_dim // (4 ** (hidden_layers - i))) 
                       for i in range(hidden_layers)]
        hidden_dims.reverse()
        
        for next_dim in hidden_dims:
            decoder_layers.extend([
                torch.nn.Linear(current_dim, next_dim),
                torch.nn.LayerNorm(next_dim),
                torch.nn.LeakyReLU(0.2),
                torch.nn.Dropout(dropout),
            ])
            current_dim = next_dim
        
        decoder_layers.append(torch.nn.Linear(current_dim, input_dim))
        self.decoder = torch.nn.Sequential(*decoder_layers)
        
        self.register_buffer('center', torch.zeros(latent_dim))
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
    
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z
    
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算异常分数
        
        分数 = 0.7 * 重构误差 + 0.3 * 潜在空间距离
        """
        with torch.no_grad():
            recon, z = self.forward(x)
            recon_err = F.mse_loss(recon, x, reduction='none').mean(dim=-1)
            latent_dist = torch.norm(z - self.center, dim=-1)
            return 0.7 * recon_err + 0.3 * latent_dist
    
    def fit(self, gradients: torch.Tensor, epochs: int = 20, lr: float = 1e-3):
        """训练检测器"""
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        for _ in range(epochs):
            recon, z = self.forward(gradients)
            loss = F.mse_loss(recon, gradients)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            _, z = self.forward(gradients)
            self.center = z.mean(dim=0)
        
        self.eval()


# ==============================================================================
# FedACT组件2: 委员会机制
# ==============================================================================

class Committee:
    """
    多样性委员会选择与投票机制
    
    核心思想:
    1. 选择一组"多样性"的客户端梯度作为评委
    2. 用这组梯度对其他梯度进行投票判定
    
    选择策略:
    - 第一个: 选信誉最高的客户端（最可信）
    - 后续: 选与已选梯度最不相似的（多样性）
    
    投票规则:
    - 计算待检梯度与每个委员会成员的余弦相似度
    - 如果相似度 < 阈值，该成员投"异常"票
    - 超过50%投"异常"票 → 判定为异常
    
    为什么要多样性?
    - 如果委员会成员都很相似，容易被同一种攻击欺骗
    - 多样性委员会能从不同"角度"检测异常
    """
    
    def __init__(self, size: int = 5):
        """
        Args:
            size: 委员会成员数量（建议5-7个）
        """
        self.size = size
        self.members: List[int] = []
    
    def select(
        self,
        gradients: List[torch.Tensor],
        client_ids: List[int],
        reputations: Dict[int, float]
    ) -> List[int]:
        """
        选择多样化委员会
        
        Args:
            gradients: 所有客户端的梯度列表
            client_ids: 对应的客户端ID列表
            reputations: 客户端信誉字典
            
        Returns:
            被选中的委员会成员ID列表
        """
        if len(gradients) <= self.size:
            self.members = client_ids.copy()
            return self.members
        
        selected = []
        available = list(range(len(gradients)))
        
        # 第一个: 信誉最高的客户端
        first = max(available, key=lambda i: reputations.get(client_ids[i], 1.0))
        selected.append(first)
        available.remove(first)
        
        # 后续: 最大化多样性（选择与已选梯度最不相似的）
        while len(selected) < self.size and available:
            best_idx = None
            min_sim = float('inf')
            
            for idx in available:
                # 计算与所有已选梯度的最大相似度
                max_sim_to_selected = max([
                    F.cosine_similarity(
                        gradients[idx].unsqueeze(0), 
                        gradients[s].unsqueeze(0)
                    ).item()
                    for s in selected
                ])
                # 选择最大相似度最小的（即与已选最不相似的）
                if max_sim_to_selected < min_sim:
                    min_sim = max_sim_to_selected
                    best_idx = idx
            
            if best_idx is not None:
                selected.append(best_idx)
                available.remove(best_idx)
        
        self.members = [client_ids[i] for i in selected]
        return self.members
    
    def vote(
        self,
        target_grad: torch.Tensor,
        member_grads: List[torch.Tensor],
        threshold: float = 0.3
    ) -> Tuple[bool, float]:
        """
        委员会投票
        
        Args:
            target_grad: 待检测的梯度
            member_grads: 委员会成员的梯度列表
            threshold: 相似度阈值，低于此值视为"不相似"
            
        Returns:
            (是否异常, 异常票比例)
        """
        if not member_grads:
            return False, 0.0
        
        anomaly_votes = 0
        for mg in member_grads:
            sim = F.cosine_similarity(
                target_grad.unsqueeze(0), 
                mg.unsqueeze(0)
            ).item()
            if sim < threshold:
                anomaly_votes += 1
        
        ratio = anomaly_votes / len(member_grads)
        return ratio > 0.5, ratio


# ==============================================================================
# FedACT组件3: Merkle存证与信誉激励
# ==============================================================================

class EvidenceChain:
    """
    Merkle树存证链 + 信誉管理
    
    存证功能:
    - 使用Merkle树记录每轮检测结果
    - 不可篡改，支持事后审计
    
    信誉机制:
    - 正常行为: 信誉 += 0.05 * 贡献度
    - 异常行为: 信誉 *= 0.7（惩罚）
    - 信誉范围: [0.1, 2.0]
    """
    
    def __init__(self):
        self.chain: List[Dict] = []
        self.reputations: Dict[int, float] = {}
        self.contributions: Dict[int, List[float]] = {}
    
    def _get_default_reputation(self) -> float:
        return 1.0
    
    @staticmethod
    def _hash(data: str) -> str:
        import hashlib
        return hashlib.sha256(data.encode()).hexdigest()
    
    def _merkle_root(self, items: List[str]) -> str:
        if not items:
            return self._hash("empty")
        
        level = [self._hash(item) for item in items]
        while len(level) > 1:
            if len(level) % 2 == 1:
                level.append(level[-1])
            level = [self._hash(level[i] + level[i+1]) for i in range(0, len(level), 2)]
        
        return level[0]
    
    def record(self, round_id: int, results: Dict[int, Dict]) -> str:
        """记录一轮结果到Merkle树"""
        import json
        from datetime import datetime
        
        def convert_value(v):
            if hasattr(v, 'item'):
                return v.item()
            elif isinstance(v, (np.floating, np.integer)):
                return float(v)
            return v
        
        serializable_results = {
            k: {kk: convert_value(vv) for kk, vv in v.items()}
            for k, v in results.items()
        }
        
        items = [json.dumps({**v, 'client': k}, sort_keys=True) 
                 for k, v in serializable_results.items()]
        root = self._merkle_root(items)
        
        self.chain.append({
            'round': round_id,
            'root': root,
            'timestamp': datetime.now().isoformat(),
            'count': len(results)
        })
        
        return root
    
    def update_reputation(self, client_id: int, is_normal: bool, contribution: float = 0.0):
        """更新客户端信誉"""
        if client_id not in self.reputations:
            self.reputations[client_id] = self._get_default_reputation()
        if client_id not in self.contributions:
            self.contributions[client_id] = []
        
        old = self.reputations[client_id]
        if is_normal:
            self.reputations[client_id] = min(old + 0.05 * contribution, 2.0)
        else:
            self.reputations[client_id] = max(old * 0.7, 0.1)
        
        self.contributions[client_id].append(contribution)
    
    def get_reputation(self, client_id: int) -> float:
        return self.reputations.get(client_id, self._get_default_reputation())
    
    def get_weights(self, client_ids: List[int]) -> Dict[int, float]:
        """基于信誉计算权重"""
        reps = {c: self.get_reputation(c) for c in client_ids}
        total = sum(reps.values())
        return {c: reps[c] / total for c in client_ids}


# ==============================================================================
# FedACT完整防御类
# ==============================================================================

class FedACTDefense(BaseDefense):
    """
    FedACT: Federated Autoencoder-Committee TLBO
    
    完整的FedACT防御流程:
    1. 自编码器异常检测 - 初筛异常梯度
    2. 委员会投票 - 对可疑梯度进行二次确认
    3. TLBO优化聚合 - 对正常梯度进行优化聚合
    4. 信誉更新 - 更新客户端信誉值
    
    参数:
        committee_size: 委员会大小（默认5）
        tlbo_iterations: TLBO迭代次数（默认10）
        autoencoder_epochs: 自编码器训练轮数（默认20）
        vote_threshold: 投票相似度阈值（默认0.3）
    """
    
    def __init__(
        self, 
        committee_size: int = 5,
        tlbo_iterations: int = 10,
        autoencoder_epochs: int = 20,
        vote_threshold: float = 0.3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.committee_size = committee_size
        self.tlbo_iterations = tlbo_iterations
        self.autoencoder_epochs = autoencoder_epochs
        self.vote_threshold = vote_threshold
        
        # 组件（延迟初始化）
        self.detector: Optional[GradientDetector] = None
        self.committee = Committee(size=committee_size)
        self.evidence = EvidenceChain()
        self.tlbo = TLBODefense(iterations=tlbo_iterations)
        
        self.round_count = 0
    
    @property
    def name(self) -> str:
        return "fedact"
    
    def aggregate(
        self,
        gradients: List[torch.Tensor],
        weights: Optional[List[float]] = None,
        client_ids: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        FedACT聚合
        
        Args:
            gradients: 梯度列表
            weights: 权重列表（可选，会被信誉系统覆盖）
            client_ids: 客户端ID列表（可选）
            
        Returns:
            聚合后的梯度
        """
        if not gradients:
            raise ValueError("No gradients to aggregate")
        
        self.round_count += 1
        
        # 如果没有提供client_ids，生成默认的
        if client_ids is None:
            client_ids = list(range(len(gradients)))
        
        # 初始化检测器
        grad_dim = gradients[0].numel()
        device = gradients[0].device
        
        if self.detector is None:
            self.detector = GradientDetector(
                input_dim=grad_dim,
                auto_adapt=True
            ).to(device)
        
        grad_tensor = torch.stack(gradients).to(device)
        
        # 训练自编码器（增量策略）
        if self.round_count <= 3:
            self.detector.fit(grad_tensor, epochs=self.autoencoder_epochs)
        elif self.round_count % 5 == 0:
            self.detector.fit(grad_tensor, epochs=max(5, self.autoencoder_epochs // 4))
        
        # 异常检测
        scores = self.detector.anomaly_score(grad_tensor)
        threshold = scores.mean() + 2 * scores.std()
        
        # 委员会投票（前5轮不使用，等信誉稳定）
        use_committee = self.round_count > 5
        committee_grads = []
        
        if use_committee:
            reputations = {c: self.evidence.get_reputation(c) for c in client_ids}
            self.committee.select(gradients, client_ids, reputations)
            committee_grads = [
                gradients[client_ids.index(m)] 
                for m in self.committee.members if m in client_ids
            ]
        
        # 检测异常
        normal_ids, anomaly_ids = [], []
        results = {}
        
        for i, cid in enumerate(client_ids):
            score = scores[i].item()
            is_anomaly = score > threshold
            
            # 可疑样本委员会投票
            if use_committee and committee_grads:
                if score > threshold * 0.7 and score <= threshold * 1.5:
                    is_anomaly, _ = self.committee.vote(
                        gradients[i], committee_grads, threshold=self.vote_threshold
                    )
            
            results[cid] = {'score': score, 'is_anomaly': is_anomaly}
            
            if is_anomaly:
                anomaly_ids.append(cid)
            else:
                normal_ids.append(cid)
        
        # 聚合正常梯度
        if normal_ids:
            normal_grads = [gradients[client_ids.index(c)] for c in normal_ids]
            rep_weights = self.evidence.get_weights(normal_ids)
            weight_list = [rep_weights[c] for c in normal_ids]
            
            aggregated = self.tlbo.aggregate(normal_grads, weight_list)
            
            # 更新信誉
            for cid in normal_ids:
                idx = client_ids.index(cid)
                contrib = F.cosine_similarity(
                    gradients[idx].unsqueeze(0), 
                    aggregated.unsqueeze(0)
                ).item()
                contrib = max(0, (contrib + 1) / 2)
                self.evidence.update_reputation(cid, True, contrib)
            
            for cid in anomaly_ids:
                self.evidence.update_reputation(cid, False)
            
            # 存证
            self.evidence.record(self.round_count, results)
            
            return aggregated
        else:
            # 所有梯度都异常，使用TLBO鲁棒聚合
            return self.tlbo.aggregate(gradients)


# ==============================================================================
# 防御注册表
# ==============================================================================

DEFENSE_REGISTRY: Dict[str, type] = {
    # 基础防御
    'fedavg': FedAvgDefense,
    'median': MedianDefense,
    'trimmed_mean': TrimmedMeanDefense,
    'krum': KrumDefense,
    'multi_krum': MultiKrumDefense,
    'bulyan': BulyanDefense,
    
    # 前沿防御
    'fltrust': FLTrustDefense,
    'signsgd': SignSGDDefense,
    'norm_bound': NormBoundDefense,
    'rfa': RFADefense,
    
    # 我们的方法
    'tlbo': TLBODefense,
    'fedact': FedACTDefense,
}


# ==============================================================================
# 防御策略封装类
# ==============================================================================

class DefenseStrategy:
    """
    防御策略
    
    提供统一的防御接口，支持所有注册的防御方法。
    
    使用示例:
        defense = DefenseStrategy('tlbo', iterations=10)
        aggregated = defense.aggregate(gradients, weights)
    """
    
    def __init__(self, defense_mode: str, **kwargs):
        """
        初始化防御策略
        
        Args:
            defense_mode: 防御模式名称
            **kwargs: 额外参数
        """
        if defense_mode not in DEFENSE_REGISTRY:
            raise ValueError(f"未知防御模式: {defense_mode}. "
                           f"支持的防御: {list(DEFENSE_REGISTRY.keys())}")
        
        self.defense_mode = defense_mode
        self.defense_instance = DEFENSE_REGISTRY[defense_mode](**kwargs)
    
    def aggregate(
        self,
        gradients: List[torch.Tensor],
        weights: Optional[List[float]] = None
    ) -> torch.Tensor:
        """
        聚合梯度
        
        Args:
            gradients: 梯度列表
            weights: 权重列表（可选）
            
        Returns:
            聚合后的梯度
        """
        return self.defense_instance.aggregate(gradients, weights)
    
    @classmethod
    def list_defenses(cls) -> List[str]:
        """列出所有支持的防御"""
        return list(DEFENSE_REGISTRY.keys())
    
    @classmethod
    def get_defense_info(cls, defense_mode: str) -> Dict:
        """获取防御信息"""
        if defense_mode not in DEFENSE_REGISTRY:
            return {}
        
        defense_cls = DEFENSE_REGISTRY[defense_mode]
        instance = defense_cls()
        
        return {
            'name': instance.name,
            'docstring': defense_cls.__doc__
        }


# ==============================================================================
# 辅助函数
# ==============================================================================

def get_defense(defense_mode: str, **kwargs) -> DefenseStrategy:
    """
    获取防御策略的便捷函数
    
    Args:
        defense_mode: 防御模式
        **kwargs: 额外参数
        
    Returns:
        DefenseStrategy实例
    """
    return DefenseStrategy(defense_mode, **kwargs)


def list_all_defenses() -> List[Dict]:
    """列出所有防御及其信息"""
    defenses = []
    for name in DEFENSE_REGISTRY:
        info = DefenseStrategy.get_defense_info(name)
        info['key'] = name
        defenses.append(info)
    return defenses
