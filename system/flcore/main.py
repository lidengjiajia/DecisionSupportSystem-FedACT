#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
联邦学习实验主程序

支持的算法:
- Centralized: 中心化训练 (作为性能上界)
- FedTLBO: 本文提出的方法 (TLBO聚合 + 委员会投票 + Merkle存证 + 激励机制)
- FedAvg: 联邦平均
- FedProx: 联邦近端优化
- SCAFFOLD: 方差缩减方法
- MOON: 模型对比学习
- PerAvg: 个性化联邦平均
- FedRep: 联邦表征学习
- FedGWO: 灰狼优化器聚合
- FedPSO: 粒子群优化聚合

支持的数据集:
- Uci: UCI信用卡违约数据集 (23特征, 30000样本)
- Xinwang: 新网银行信用数据集 (38特征, 17886样本)
"""

import copy
import torch
from torch import nn
import argparse
import os
import time
import warnings
import numpy as np
import logging

# 服务器导入
from flcore.servers.serveravg import FedAvg
from flcore.servers.serverprox import FedProx
from flcore.servers.serverperavg import PerAvg
from flcore.servers.serverscaffold import SCAFFOLD
from flcore.servers.servermoon import MOON
from flcore.servers.serverrep import FedRep
from flcore.servers.servergwo import FedGWO
from flcore.servers.serverpso import FedPSO
from flcore.servers.servercti import FedCTI
from flcore.servers.servertlbo import FedTLBO
from flcore.servers.servercentral import Centralized

# 模型导入
from flcore.trainmodel.creditnet import CreditNet, UciCreditNet, XinwangCreditNet, BaseHeadSplit

# 工具导入
from utils.result_utils import average_data

# 日志配置
logger = logging.getLogger()
logger.setLevel(logging.ERROR)
warnings.simplefilter("ignore")
torch.manual_seed(42)


def run(args):
    """运行训练实验"""
    time_list = []
    model_str = args.model

    for i in range(args.prev, args.times):
        print(f"\n============= 第 {i+1} 次实验 =============")
        print("正在初始化服务器和客户端...")
        start = time.time()

        # 信用评分数据集固定为2分类
        if args.dataset in ['Uci', 'Xinwang']:
            args.num_classes = 2

        # 读取数据集配置
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        config_path = os.path.join(project_root, 'dataset', args.dataset, 'config.json')
        
        feature_dim = 23 if args.dataset == 'Uci' else 38
        if os.path.exists(config_path):
            try:
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if 'num_clients' in config:
                        args.num_clients = config['num_clients']
                    if 'feature_dim' in config:
                        feature_dim = config['feature_dim']
            except Exception as e:
                print(f"[警告] 读取配置文件失败: {e}")

        # 模型初始化
        if model_str == "auto" or model_str == "creditnet":
            if args.dataset == 'Uci':
                args.model = UciCreditNet(input_dim=feature_dim, num_classes=args.num_classes).to(args.device)
            elif args.dataset == 'Xinwang':
                args.model = XinwangCreditNet(input_dim=feature_dim, num_classes=args.num_classes).to(args.device)
            else:
                raise ValueError(f"不支持的数据集: {args.dataset}")
        else:
            raise ValueError(f"不支持的模型: {model_str}")

        print(f"数据集: {args.dataset}, 特征维度: {feature_dim}")
        print(f"模型参数量: {sum(p.numel() for p in args.model.parameters()):,}")

        # 算法选择
        algo = args.algorithm.split('_')[0]
        
        if algo == "Centralized":
            # 中心化训练 (性能上界)
            server = Centralized(args, i)
            
        elif algo == "FedAvg":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedAvg(args, i)
            
        elif algo == "FedProx":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedProx(args, i)
            
        elif algo == "SCAFFOLD":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = SCAFFOLD(args, i)
            
        elif algo == "MOON":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = MOON(args, i)
            
        elif algo == "PerAvg":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = PerAvg(args, i)
            
        elif algo == "FedRep":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedRep(args, i)
            
        elif algo == "FedGWO":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedGWO(args, i)
            
        elif algo == "FedPSO":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedPSO(args, i)
            
        elif algo == "FedCTI":
            # 本文提出的方法 (旧版本，兼容)
            server = FedCTI(args, i)
        
        elif algo == "FedTLBO":
            # 本文提出的方法 (新版本)
            server = FedTLBO(args, i)
            
        else:
            raise NotImplementedError(f"算法 {algo} 未实现")

        # 执行训练
        server.train()
        time_list.append(time.time() - start)

    # 输出统计信息
    print(f"\n平均训练时间: {round(np.average(time_list), 2)}秒")
    
    # 攻击防御实验不需要汇总结果文件，直接从输出解析
    if not args.enable_attack:
        try:
            average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)
        except FileNotFoundError as e:
            print(f"[WARNING] 跳过结果汇总: {e}")


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='联邦学习信用评分实验')
    
    # =========================================================================
    # 基础参数
    # =========================================================================
    parser.add_argument('-go', "--goal", type=str, default="test", help="实验名称标识")
    parser.add_argument('-dev', "--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument('-did', "--device_id", type=str, default="0", help="GPU设备编号")
    parser.add_argument('-data', "--dataset", type=str, default="Uci", 
                        choices=["Uci", "Xinwang"], help="数据集名称")
    parser.add_argument('--heterogeneity', type=str, default="iid",
                        choices=["iid", "label_skew", "quantity_skew", "feature_skew"],
                        help="数据异质性类型: iid, label_skew, quantity_skew, feature_skew")
    parser.add_argument('-m', "--model", type=str, default="auto", help="模型类型")
    parser.add_argument('-nb', "--num_classes", type=int, default=2, help="分类类别数")
    parser.add_argument('-lbs', "--batch_size", type=int, default=32, help="批量大小")
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.01, help="本地学习率")
    parser.add_argument('-gr', "--global_rounds", type=int, default=100, help="全局通信轮数")
    parser.add_argument('-ls', "--local_epochs", type=int, default=5, help="本地训练轮数")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedCTI", help="联邦学习算法")
    parser.add_argument('-nc', "--num_clients", type=int, default=10, help="客户端数量")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0, help="每轮参与比例")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False)
    parser.add_argument('-pv', "--prev", type=int, default=0, help="从第几次实验开始")
    parser.add_argument('-t', "--times", type=int, default=1, help="实验重复次数")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1, help="评估间隔")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='results', help="结果保存目录")
    parser.add_argument('-lrd', "--learning_rate_decay", type=bool, default=False, help="是否启用学习率衰减")
    parser.add_argument('-lrdg', "--learning_rate_decay_gamma", type=float, default=0.99, help="学习率衰减系数")
    
    # =========================================================================
    # 服务器基础参数
    # =========================================================================
    parser.add_argument('-fs', "--few_shot", type=int, default=0)
    parser.add_argument('-ts', "--time_select", type=bool, default=False)
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000)
    parser.add_argument('-tc', "--top_cnt", type=int, default=100)
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0)
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0)
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0)
    
    # =========================================================================
    # 算法特定参数
    # =========================================================================
    # FedProx
    parser.add_argument('-mu', "--mu", type=float, default=0.01, help="FedProx正则化系数")
    parser.add_argument('-bt', "--beta", type=float, default=0.0)
    parser.add_argument('-lam', "--lamda", type=float, default=1.0)
    
    # MOON
    parser.add_argument('-tau', "--tau", type=float, default=1.0, help="MOON温度系数")
    
    # FedRep/PerAvg
    parser.add_argument('-pls', "--plocal_epochs", type=int, default=1, help="个性化训练轮数")
    
    # SCAFFOLD
    parser.add_argument('-slr', "--server_learning_rate", type=float, default=1.0)
    
    # =========================================================================
    # FedCTI 参数
    # =========================================================================
    # 攻击模拟配置
    parser.add_argument('--enable_attack', type=lambda x: str(x).lower() == 'true', 
                        default=False, help='是否启用攻击模拟')
    parser.add_argument('--attack_mode', type=str, default='none',
                        choices=['none', 'sign_flip', 'gaussian', 'scale', 
                                 'little', 'alie', 'ipm', 'minmax', 'trim_attack',
                                 'label_flip', 'backdoor', 'free_rider', 'collision'],
                        help='攻击类型: 基础(sign_flip/gaussian/scale), 前沿(little/alie/ipm/minmax/trim_attack), 其他(label_flip/backdoor/free_rider/collision)')
    parser.add_argument('--malicious_ratio', type=float, default=0.2, help='恶意客户端比例')
    parser.add_argument('--attack_scale', type=float, default=3.0, help='攻击强度')
    
    # =========================================================================
    # 防御配置
    # =========================================================================
    parser.add_argument('--defense_mode', type=str, default='fedact',
                        choices=['none', 'fedavg', 'median', 'trimmed_mean', 'krum', 
                                 'multi_krum', 'bulyan', 'rfa', 'fltrust', 'signsgd', 
                                 'norm_bound', 'tlbo', 'fedact'],
                        help='防御模式: none(无防御), fedavg, median, trimmed_mean, krum, multi_krum, bulyan, rfa, fedact(本文方法)')
    
    # 自编码器检测配置
    parser.add_argument('--use_autoencoder', type=lambda x: str(x).lower() == 'true', 
                        default=True, help='是否使用自编码器异常检测')
    parser.add_argument('--autoencoder_latent_dim', type=int, default=64, help='自编码器潜在维度')
    parser.add_argument('--autoencoder_epochs', type=int, default=20, help='自编码器训练轮数')
    
    # 委员会配置
    parser.add_argument('--committee_size', type=int, default=5, help='委员会成员数量')
    parser.add_argument('--vote_threshold', type=float, default=0.5, help='投票阈值')
    
    # 异常检测配置
    parser.add_argument('--anomaly_threshold', type=float, default=0.5, help='异常判定阈值')
    parser.add_argument('--anomaly_lower_coef', type=float, default=0.7, help='委员会投票下界系数 (threshold * coef)')
    parser.add_argument('--anomaly_upper_coef', type=float, default=1.5, help='直接拒绝上界系数 (threshold * coef)')
    parser.add_argument('--use_committee', type=lambda x: str(x).lower() == 'true', 
                        default=True, help='是否使用委员会投票')
    
    # TLBO配置
    parser.add_argument('--tlbo_iterations', type=int, default=10, help='TLBO迭代次数')
    parser.add_argument('--use_tlbo', type=lambda x: str(x).lower() == 'true', 
                        default=True, help='是否使用TLBO聚合')
    
    # 存证与激励
    parser.add_argument('--use_merkle', type=lambda x: str(x).lower() == 'true', 
                        default=True, help='是否使用Merkle存证')
    parser.add_argument('--use_incentive', type=lambda x: str(x).lower() == 'true', 
                        default=True, help='是否使用激励机制')

    args = parser.parse_args()

    # GPU设置：优先使用已设置的 CUDA_VISIBLE_DEVICES 环境变量
    # 这样并发实验脚本可以通过环境变量分配不同的GPU
    existing_cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if existing_cuda_env is not None and existing_cuda_env != "":
        # 如果环境变量已设置（由并发脚本设置），使用它
        # 注意：此时 torch.cuda.device(0) 对应的是环境变量指定的物理GPU
        print(f"[GPU] 使用环境变量指定的GPU: CUDA_VISIBLE_DEVICES={existing_cuda_env}")
    else:
        # 否则使用命令行参数
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
        print(f"[GPU] 使用命令行参数指定的GPU: device_id={args.device_id}")
    
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[警告] CUDA不可用，切换到CPU")
        args.device = "cpu"

    # 打印配置
    print("=" * 60)
    print("实验配置:")
    print("=" * 60)
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    print("=" * 60)

    return args


if __name__ == "__main__":
    args = get_args()
    run(args)
