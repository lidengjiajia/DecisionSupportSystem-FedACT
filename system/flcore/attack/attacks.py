"""
拜占庭攻击实现模块

包含多种经典和前沿的拜占庭攻击方法:

基础攻击:
- Sign Flip: 符号翻转攻击
- Gaussian: 高斯噪声攻击  
- Scale: 缩放攻击

前沿攻击 (顶会论文):
- Little: "A Little Is Enough" (NeurIPS 2019)
- ALIE: Adaptive Local Model Poisoning (NeurIPS 2019)
- IPM: Inner Product Manipulation (ICML 2018)
- MinMax: Min-Max Attack (IEEE S&P 2020)
- LabelFlip: 标签翻转攻击
- Backdoor: 后门攻击

参考文献:
[1] Baruch et al., "A Little Is Enough: Circumventing Defenses For Distributed Learning", NeurIPS 2019
[2] Xie et al., "Fall of Empires: Breaking Byzantine-tolerant SGD by Inner Product Manipulation", ICML 2018
[3] Fang et al., "Local Model Poisoning Attacks to Byzantine-Robust Federated Learning", USENIX Security 2020
[4] Shejwalkar et al., "Manipulating the Byzantine: Optimizing Model Poisoning Attacks", NDSS 2021

作者: FedTLBO Team
日期: 2026-01-22
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Dict, Callable, Tuple
from abc import ABC, abstractmethod


# ==============================================================================
# 攻击基类
# ==============================================================================

class BaseAttack(ABC):
    """攻击基类"""
    
    def __init__(self, scale: float = 1.0, **kwargs):
        self.scale = scale
        self.kwargs = kwargs
    
    @abstractmethod
    def attack(
        self, 
        gradient: torch.Tensor,
        benign_gradients: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """执行攻击"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """攻击名称"""
        pass
    
    @property
    def requires_benign_grads(self) -> bool:
        """是否需要诚实客户端梯度"""
        return False


# ==============================================================================
# 基础攻击
# ==============================================================================

class SignFlipAttack(BaseAttack):
    """
    符号翻转攻击
    
    最简单的攻击方式，将梯度取反。
    适用于验证基本防御能力。
    """
    
    @property
    def name(self) -> str:
        return "sign_flip"
    
    def attack(
        self, 
        gradient: torch.Tensor,
        benign_gradients: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        return -gradient


class GaussianAttack(BaseAttack):
    """
    高斯噪声攻击
    
    向梯度添加大方差高斯噪声。
    攻击强度由scale参数控制。
    """
    
    @property
    def name(self) -> str:
        return "gaussian"
    
    def attack(
        self, 
        gradient: torch.Tensor,
        benign_gradients: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        noise = torch.randn_like(gradient) * gradient.std() * self.scale
        return gradient + noise


class ScaleAttack(BaseAttack):
    """
    缩放攻击
    
    将梯度反向缩放，破坏聚合结果。
    """
    
    @property
    def name(self) -> str:
        return "scale"
    
    def attack(
        self, 
        gradient: torch.Tensor,
        benign_gradients: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        return gradient * (-self.scale)


# ==============================================================================
# 前沿攻击 (顶会论文)
# ==============================================================================

class LittleAttack(BaseAttack):
    """
    "A Little Is Enough" 攻击 - NeurIPS 2019
    
    论文: "A Little Is Enough: Circumventing Defenses For Distributed Learning"
    作者: Baruch et al.
    
    核心思想: 恶意梯度设计为刚好在异常检测边界内，
    利用聚合规则的漏洞使攻击生效。
    
    恶意梯度 = g + ε * sign(g)
    其中 ε 控制偏离程度，设计为刚好不被检测到。
    """
    
    @property
    def name(self) -> str:
        return "little"
    
    def attack(
        self, 
        gradient: torch.Tensor,
        benign_gradients: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        eps = self.scale * gradient.std()
        return gradient + eps * torch.sign(gradient)


class ALIEAttack(BaseAttack):
    """
    ALIE攻击 - NeurIPS 2019
    
    论文: "A Little Is Enough: Circumventing Defenses For Distributed Learning"
    
    核心思想: 自适应绕过中位数/修剪防御
    恶意梯度设计为刚好在修剪边界内。
    
    恶意梯度 = μ - z_max * σ
    其中 μ, σ 是诚实梯度的均值和标准差，
    z_max 控制偏离程度（典型值 < 2）
    """
    
    @property
    def name(self) -> str:
        return "alie"
    
    @property
    def requires_benign_grads(self) -> bool:
        return True
    
    def attack(
        self, 
        gradient: torch.Tensor,
        benign_gradients: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        if benign_gradients and len(benign_gradients) > 0:
            stacked = torch.stack(benign_gradients)
            mean = stacked.mean(dim=0)
            std = stacked.std(dim=0) + 1e-10
            
            # z_max 控制偏离程度
            z_max = min(self.scale, 1.5)
            malicious = mean - z_max * std
            return malicious
        else:
            # 回退到little攻击
            eps = self.scale * gradient.std()
            return gradient + eps * torch.sign(gradient)


class IPMAttack(BaseAttack):
    """
    Inner Product Manipulation (IPM) 攻击 - ICML 2018
    
    论文: "Fall of Empires: Breaking Byzantine-tolerant SGD by Inner Product Manipulation"
    作者: Xie et al.
    
    核心思想: 操纵内积使聚合梯度偏离正确方向
    恶意梯度与诚实梯度的内积为负，从而破坏聚合。
    
    恶意梯度 = -ε * normalize(mean(benign_grads))
    """
    
    @property
    def name(self) -> str:
        return "ipm"
    
    @property
    def requires_benign_grads(self) -> bool:
        return True
    
    def attack(
        self, 
        gradient: torch.Tensor,
        benign_gradients: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        if benign_gradients and len(benign_gradients) > 0:
            benign_mean = torch.stack(benign_gradients).mean(dim=0)
            direction = benign_mean / (benign_mean.norm() + 1e-10)
            malicious = -self.scale * direction * gradient.norm()
            return malicious
        else:
            return -gradient


class MinMaxAttack(BaseAttack):
    """
    Min-Max攻击 - IEEE S&P 2020
    
    论文: "Local Model Poisoning Attacks to Byzantine-Robust Federated Learning"
    作者: Fang et al.
    
    核心思想: 最大化与诚实梯度的距离，
    同时保持在不被检测的范围内。
    
    优化目标: max distance(malicious, benign_mean)
    约束条件: 不被防御机制检测到
    """
    
    @property
    def name(self) -> str:
        return "minmax"
    
    @property
    def requires_benign_grads(self) -> bool:
        return True
    
    def attack(
        self, 
        gradient: torch.Tensor,
        benign_gradients: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        if benign_gradients and len(benign_gradients) > 0:
            stacked = torch.stack(benign_gradients)
            mean = stacked.mean(dim=0)
            std = stacked.std(dim=0) + 1e-10
            
            # 找到距离诚实均值最远的方向
            perturbation = -self.scale * mean
            
            # 限制在一定范围内
            max_dev = 3 * std
            perturbation = torch.clamp(perturbation, -max_dev, max_dev)
            
            return perturbation
        else:
            return -gradient * self.scale


class TrimmedMeanAttack(BaseAttack):
    """
    针对Trimmed Mean防御的攻击
    
    论文: "Manipulating the Byzantine: Optimizing Model Poisoning Attacks", NDSS 2021
    
    核心思想: 设计恶意梯度刚好在修剪边界内，
    使其不会被修剪掉，从而影响最终结果。
    """
    
    @property
    def name(self) -> str:
        return "trim_attack"
    
    @property
    def requires_benign_grads(self) -> bool:
        return True
    
    def attack(
        self, 
        gradient: torch.Tensor,
        benign_gradients: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        if benign_gradients and len(benign_gradients) > 0:
            stacked = torch.stack(benign_gradients)
            sorted_grads, _ = torch.sort(stacked, dim=0)
            
            # 目标是边界值
            n = len(benign_gradients)
            trim_k = max(1, n // 4)  # 修剪比例
            
            # 选择刚好在边界内的值
            if np.random.random() > 0.5:
                target = sorted_grads[trim_k]  # 下边界
            else:
                target = sorted_grads[-trim_k-1]  # 上边界
            
            return target
        else:
            return -gradient


class LabelFlipAttack(BaseAttack):
    """
    标签翻转攻击 (Data Poisoning)
    
    这是一种数据投毒攻击，通过翻转训练标签来破坏模型。
    在梯度层面模拟标签翻转的效果。
    """
    
    @property
    def name(self) -> str:
        return "label_flip"
    
    def attack(
        self, 
        gradient: torch.Tensor,
        benign_gradients: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        # 标签翻转在梯度层面表现为部分维度翻转
        flip_ratio = min(0.5, self.scale / 10)
        mask = torch.rand_like(gradient) < flip_ratio
        flipped = gradient.clone()
        flipped[mask] = -flipped[mask]
        return flipped


class BackdoorAttack(BaseAttack):
    """
    后门攻击
    
    论文: "How to Backdoor Federated Learning", AISTATS 2020
    
    在梯度中嵌入后门模式，使模型在特定触发下产生恶意行为。
    """
    
    @property
    def name(self) -> str:
        return "backdoor"
    
    def attack(
        self, 
        gradient: torch.Tensor,
        benign_gradients: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        # 在特定位置嵌入后门模式
        backdoor_gradient = gradient.clone()
        
        # 选择一部分维度作为后门
        backdoor_dims = len(gradient) // 10
        backdoor_pattern = torch.randn(backdoor_dims, device=gradient.device) * self.scale
        
        # 嵌入后门
        backdoor_gradient[:backdoor_dims] = backdoor_pattern
        
        return backdoor_gradient


class FreeRiderAttack(BaseAttack):
    """
    搭便车攻击
    
    恶意客户端不进行实际训练，只发送零梯度或极小梯度。
    """
    
    @property
    def name(self) -> str:
        return "free_rider"
    
    def attack(
        self, 
        gradient: torch.Tensor,
        benign_gradients: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        # 发送接近零的梯度
        noise = torch.randn_like(gradient) * 1e-6
        return noise


class CollisionAttack(BaseAttack):
    """
    串通攻击
    
    多个恶意客户端协同发送相同的恶意梯度，
    以绕过某些基于多样性的防御。
    """
    
    def __init__(self, scale: float = 1.0, collision_seed: int = 42, **kwargs):
        super().__init__(scale, **kwargs)
        self.collision_seed = collision_seed
    
    @property
    def name(self) -> str:
        return "collision"
    
    def attack(
        self, 
        gradient: torch.Tensor,
        benign_gradients: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        # 使用固定种子生成相同的恶意梯度
        gen = torch.Generator(device=gradient.device)
        gen.manual_seed(self.collision_seed)
        
        malicious = torch.randn(gradient.shape, generator=gen, device=gradient.device)
        malicious = malicious * gradient.norm() * self.scale
        
        return malicious


# ==============================================================================
# 攻击注册表
# ==============================================================================

ATTACK_REGISTRY: Dict[str, type] = {
    # 基础攻击
    'sign_flip': SignFlipAttack,
    'gaussian': GaussianAttack,
    'scale': ScaleAttack,
    
    # 前沿攻击
    'little': LittleAttack,
    'alie': ALIEAttack,
    'ipm': IPMAttack,
    'minmax': MinMaxAttack,
    'trim_attack': TrimmedMeanAttack,
    
    # 其他攻击
    'label_flip': LabelFlipAttack,
    'backdoor': BackdoorAttack,
    'free_rider': FreeRiderAttack,
    'collision': CollisionAttack,
}


# ==============================================================================
# 攻击器封装类
# ==============================================================================

class ByzantineAttacker:
    """
    拜占庭攻击器
    
    提供统一的攻击接口，支持所有注册的攻击方法。
    
    使用示例:
        attacker = ByzantineAttacker('alie', scale=2.0)
        malicious_grad = attacker.attack(gradient, benign_gradients)
    """
    
    def __init__(self, attack_mode: str, scale: float = 3.0, **kwargs):
        """
        初始化攻击器
        
        Args:
            attack_mode: 攻击模式名称
            scale: 攻击强度
            **kwargs: 额外参数
        """
        if attack_mode not in ATTACK_REGISTRY:
            raise ValueError(f"未知攻击模式: {attack_mode}. "
                           f"支持的攻击: {list(ATTACK_REGISTRY.keys())}")
        
        self.attack_mode = attack_mode
        self.scale = scale
        self.attack_instance = ATTACK_REGISTRY[attack_mode](scale=scale, **kwargs)
    
    @property
    def requires_benign_grads(self) -> bool:
        """是否需要诚实客户端梯度"""
        return self.attack_instance.requires_benign_grads
    
    def attack(
        self, 
        gradient: torch.Tensor,
        benign_gradients: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        执行攻击
        
        Args:
            gradient: 原始梯度
            benign_gradients: 诚实客户端梯度列表（高级攻击需要）
            
        Returns:
            恶意梯度
        """
        return self.attack_instance.attack(gradient, benign_gradients)
    
    @classmethod
    def list_attacks(cls) -> List[str]:
        """列出所有支持的攻击"""
        return list(ATTACK_REGISTRY.keys())
    
    @classmethod
    def get_attack_info(cls, attack_mode: str) -> Dict:
        """获取攻击信息"""
        if attack_mode not in ATTACK_REGISTRY:
            return {}
        
        attack_cls = ATTACK_REGISTRY[attack_mode]
        instance = attack_cls()
        
        return {
            'name': instance.name,
            'requires_benign_grads': instance.requires_benign_grads,
            'docstring': attack_cls.__doc__
        }


# ==============================================================================
# 辅助函数
# ==============================================================================

def get_attack(attack_mode: str, scale: float = 3.0, **kwargs) -> ByzantineAttacker:
    """
    获取攻击器的便捷函数
    
    Args:
        attack_mode: 攻击模式
        scale: 攻击强度
        
    Returns:
        ByzantineAttacker实例
    """
    return ByzantineAttacker(attack_mode, scale, **kwargs)


def list_all_attacks() -> List[Dict]:
    """列出所有攻击及其信息"""
    attacks = []
    for name in ATTACK_REGISTRY:
        info = ByzantineAttacker.get_attack_info(name)
        info['key'] = name
        attacks.append(info)
    return attacks
