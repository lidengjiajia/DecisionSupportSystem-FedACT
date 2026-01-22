# FedACT: 联邦自编码器-委员会-TLBO框架

联邦学习环境下的拜占庭攻击防御系统

## 项目简介

FedACT是一个面向异质性数据的联邦学习安全框架，通过结合自编码器异常检测、委员会投票机制和TLBO优化算法，有效识别和过滤恶意客户端的梯度攻击。

### 核心组件

| 组件 | 功能 | 说明 |
|------|------|------|
| 自编码器检测 | 梯度异常识别 | 学习正常梯度分布，通过重构误差识别异常 |
| 委员会投票 | 二次验证 | 对可疑梯度进行集体表决，降低误判率 |
| TLBO聚合 | 优化聚合 | 教学-学习优化算法，提升模型收敛质量 |

## 目录结构

```
DecisionSupportSystem/
├── system/                     # 核心系统
│   ├── flcore/                 # 联邦学习核心
│   │   ├── attack/             # 攻击与防御模块
│   │   │   ├── attacks.py      # 12种攻击实现
│   │   │   └── defenses.py     # 11种防御方法
│   │   ├── clients/            # 客户端实现
│   │   ├── servers/            # 服务端实现
│   │   │   └── servertlbo.py   # FedACT服务端
│   │   └── main.py             # 主入口
│   ├── utils/                  # 工具模块
│   │   └── experiment_utils.py # 实验工具
│   └── results/                # 结果输出
├── dataset/                    # 数据集
│   ├── Uci/                    # UCI信用评分数据
│   └── Xinwang/                # 新网银行数据
├── run_comparison.py           # 对比实验脚本
├── run_ablation.py             # 消融实验脚本
├── run_attack_defense.py       # 攻击防御实验脚本
└── README_CN.md                # 本文档
```

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.0+ (可选，用于GPU加速)

### 安装依赖

```bash
pip install torch numpy pandas openpyxl scikit-learn
```

### 运行实验

**1. 对比实验（无攻击场景）**
```bash
python run_comparison.py
```
- 对比6种联邦学习算法在4种异质性场景下的性能
- 实验规模：2数据集 × 4异质性 × 6算法 × 5次 = 240次

**2. 消融实验**
```bash
python run_ablation.py
```
- 验证各组件对防御效果的贡献
- 实验规模：2数据集 × 12攻击 × 5配置 × 5次 = 600次

**3. 攻击防御实验**
```bash
python run_attack_defense.py
```
- 对比8种防御方法在12种攻击下的表现
- 实验规模：2数据集 × 12攻击 × 4比例 × 8防御 × 5次 = 3840次

## 支持的攻击类型

| 类别 | 攻击名称 | 说明 |
|------|---------|------|
| 基础攻击 | sign_flip | 梯度符号翻转 |
| 基础攻击 | gaussian | 高斯噪声注入 |
| 基础攻击 | scale | 梯度缩放 |
| 前沿攻击 | little | Little攻击 (NeurIPS 2019) |
| 前沿攻击 | alie | ALIE攻击 (NeurIPS 2019) |
| 前沿攻击 | ipm | IPM攻击 (ICML 2018) |
| 前沿攻击 | minmax | MinMax攻击 (S&P 2020) |
| 前沿攻击 | trim_attack | 修剪攻击 |
| 其他攻击 | label_flip | 标签翻转 |
| 其他攻击 | backdoor | 后门攻击 |
| 其他攻击 | free_rider | 搭便车攻击 |
| 其他攻击 | collision | 共谋攻击 |

## 支持的防御方法

| 类别 | 方法名称 | 来源 |
|------|---------|------|
| 基线 | None | - |
| 经典防御 | Median | - |
| 经典防御 | TrimmedMean | ICML 2018 |
| 经典防御 | Krum | NeurIPS 2017 |
| 经典防御 | MultiKrum | NeurIPS 2017 |
| 经典防御 | Bulyan | ICML 2018 |
| 经典防御 | RFA | - |
| 本文方法 | FedACT | 本文 |

## 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --num_clients | 10 | 客户端数量 |
| --global_rounds | 100 | 全局训练轮数 |
| --malicious_ratio | 0.2 | 恶意客户端比例 |
| --attack_mode | none | 攻击类型 |
| --defense_mode | fedact | 防御模式 |
| --use_autoencoder | True | 是否使用自编码器 |
| --use_committee | True | 是否使用委员会 |
| --use_tlbo | True | 是否使用TLBO |
| --committee_size | 5 | 委员会成员数 |
| --tlbo_iterations | 10 | TLBO迭代次数 |

## 阈值设定说明

FedACT采用自适应阈值策略：

```
基准阈值 = 异常分数均值 + 2 × 标准差

判定区间:
- 直接接受: score < threshold × 0.7
- 委员会投票: threshold × 0.7 ≤ score ≤ threshold × 1.5
- 直接拒绝: score > threshold × 1.5
```

可通过以下参数调整：
- `--anomaly_lower_coef 0.7` 委员会投票下界系数
- `--anomaly_upper_coef 1.5` 直接拒绝上界系数

## 检测指标输出

每轮训练自动记录TP/TN/FP/FN指标，保存至 `results/检测统计/` 目录。

输出格式示例：
```json
{
  "config": {"attack_mode": "sign_flip", "malicious_ratio": 0.3},
  "round_by_round": [
    {"round": 1, "tp": 2, "fp": 0, "tn": 7, "fn": 1, "precision": 1.0, "recall": 0.67}
  ],
  "overall_metrics": {"precision": 0.95, "recall": 0.92, "accuracy": 0.97}
}
```

## 实验结果

结果保存在 `system/results/汇总/` 目录：
- `FedACT_对比实验_汇总.xlsx`
- `FedACT_消融实验_汇总.xlsx`
- `FedACT_攻击防御实验_汇总.xlsx`

## 许可证

MIT License
