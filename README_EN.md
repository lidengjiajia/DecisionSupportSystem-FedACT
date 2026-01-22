# FedACT: Federated Autoencoder-Committee TLBO Framework

A Byzantine attack defense system for federated learning environments.

## Introduction

FedACT is a secure federated learning framework designed for heterogeneous data scenarios. It combines autoencoder-based anomaly detection, committee voting mechanism, and TLBO optimization algorithm to effectively identify and filter malicious gradient attacks.

### Core Components

| Component | Function | Description |
|-----------|----------|-------------|
| Autoencoder Detection | Gradient Anomaly Detection | Learns normal gradient distribution, identifies anomalies via reconstruction error |
| Committee Voting | Secondary Verification | Collective voting on suspicious gradients to reduce false positives |
| TLBO Aggregation | Optimized Aggregation | Teaching-Learning Based Optimization for improved model convergence |

## Project Structure

```
DecisionSupportSystem/
├── system/                     # Core system
│   ├── flcore/                 # Federated learning core
│   │   ├── attack/             # Attack and defense module
│   │   │   ├── attacks.py      # 12 attack implementations
│   │   │   └── defenses.py     # 11 defense methods
│   │   ├── clients/            # Client implementations
│   │   ├── servers/            # Server implementations
│   │   │   └── servertlbo.py   # FedACT server
│   │   └── main.py             # Main entry
│   ├── utils/                  # Utility module
│   │   └── experiment_utils.py # Experiment utilities
│   └── results/                # Output results
├── dataset/                    # Datasets
│   ├── Uci/                    # UCI credit scoring data
│   └── Xinwang/                # Xinwang Bank data
├── run_comparison.py           # Comparison experiment script
├── run_ablation.py             # Ablation experiment script
├── run_attack_defense.py       # Attack-defense experiment script
└── README.md                   # This document
```

## Quick Start

### Requirements

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.0+ (optional, for GPU acceleration)

### Installation

```bash
pip install torch numpy pandas openpyxl scikit-learn
```

### Running Experiments

**1. Comparison Experiment (No Attack Scenario)**
```bash
python run_comparison.py
```
- Compare 6 federated learning algorithms across 4 heterogeneity scenarios
- Scale: 2 datasets × 4 heterogeneity types × 6 algorithms × 5 runs = 240 experiments

**2. Ablation Experiment**
```bash
python run_ablation.py
```
- Validate contribution of each component to defense effectiveness
- Scale: 2 datasets × 12 attacks × 5 configs × 5 runs = 600 experiments

**3. Attack-Defense Experiment**
```bash
python run_attack_defense.py
```
- Compare 8 defense methods against 12 attack types
- Scale: 2 datasets × 12 attacks × 4 ratios × 8 defenses × 5 runs = 3840 experiments

## Supported Attack Types

| Category | Attack Name | Description |
|----------|-------------|-------------|
| Basic | sign_flip | Gradient sign flipping |
| Basic | gaussian | Gaussian noise injection |
| Basic | scale | Gradient scaling |
| Advanced | little | Little Attack (NeurIPS 2019) |
| Advanced | alie | ALIE Attack (NeurIPS 2019) |
| Advanced | ipm | IPM Attack (ICML 2018) |
| Advanced | minmax | MinMax Attack (S&P 2020) |
| Advanced | trim_attack | Trim Attack |
| Other | label_flip | Label flipping |
| Other | backdoor | Backdoor attack |
| Other | free_rider | Free-rider attack |
| Other | collision | Collision attack |

## Supported Defense Methods

| Category | Method | Source |
|----------|--------|--------|
| Baseline | None | - |
| Classic | Median | - |
| Classic | TrimmedMean | ICML 2018 |
| Classic | Krum | NeurIPS 2017 |
| Classic | MultiKrum | NeurIPS 2017 |
| Classic | Bulyan | ICML 2018 |
| Classic | RFA | - |
| Proposed | FedACT | This paper |

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| --num_clients | 10 | Number of clients |
| --global_rounds | 100 | Global training rounds |
| --malicious_ratio | 0.2 | Malicious client ratio |
| --attack_mode | none | Attack type |
| --defense_mode | fedact | Defense mode |
| --use_autoencoder | True | Enable autoencoder detection |
| --use_committee | True | Enable committee voting |
| --use_tlbo | True | Enable TLBO aggregation |
| --committee_size | 5 | Committee member count |
| --tlbo_iterations | 10 | TLBO iteration count |

## Threshold Configuration

FedACT uses an adaptive threshold strategy:

```
Base threshold = Mean(anomaly_scores) + 2 × Std(anomaly_scores)

Decision zones:
- Direct accept: score < threshold × 0.7
- Committee voting: threshold × 0.7 ≤ score ≤ threshold × 1.5
- Direct reject: score > threshold × 1.5
```

Adjustable via parameters:
- `--anomaly_lower_coef 0.7` Committee voting lower bound coefficient
- `--anomaly_upper_coef 1.5` Direct rejection upper bound coefficient

## Detection Metrics Output

TP/TN/FP/FN metrics are automatically recorded each round and saved to `results/detection_stats/`.

Output format example:
```json
{
  "config": {"attack_mode": "sign_flip", "malicious_ratio": 0.3},
  "round_by_round": [
    {"round": 1, "tp": 2, "fp": 0, "tn": 7, "fn": 1, "precision": 1.0, "recall": 0.67}
  ],
  "overall_metrics": {"precision": 0.95, "recall": 0.92, "accuracy": 0.97}
}
```

## Experiment Results

Results are saved in `system/results/summary/` directory:
- `FedACT_comparison_summary.xlsx`
- `FedACT_ablation_summary.xlsx`
- `FedACT_attack_defense_summary.xlsx`

## Citation

If you use this code, please cite our paper:

```bibtex
@article{fedact2026,
  title={FedACT: Federated Autoencoder-Committee TLBO for Byzantine-Resilient Aggregation},
  author={...},
  journal={...},
  year={2026}
}
```

## License

MIT License
