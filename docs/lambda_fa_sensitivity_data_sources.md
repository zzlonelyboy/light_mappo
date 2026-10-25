# \(\lambda_{\mathrm{FA}}\) 敏感性图的数据来源

## 读取与筛选方式

MATLAB 脚本不保存任何手写的实验指标，也不维护人工的“权重—实验目录”
映射。脚本扫描：

```text
experimentsCubic/*/ablation_summary.json
```

从实验记录的 `false_alarm_loss_weight` 读取横坐标，并利用相应运行的
`training/training_config.json` 筛选满足以下条件的实验：

- 单层因果 GRU，hidden dimension 64；
- input encoder dimension 16；
- `inter_uav_consistency` Full 10-D 特征；
- `class_weighting=none`；
- detection loss weight 为 0；
- delay loss weight 为 0；
- 仅改变 `false_alarm_loss_weight`；
- 连续确认步数 K=5。

如果某个要求的权重缺失，或者同一权重匹配到多个实验，脚本会终止，避免
静默使用错误结果。

## 自动发现的六个扫描点

| \(\lambda_{\mathrm{FA}}\) | 实验记录目录 |
|---:|---|
| 0.5 | `experimentsCubic/mlp16_early0_fa0p5` |
| 1 | `experimentsCubic/mlp16_early0_fa1p0` |
| 2 | `experimentsCubic/mlp16_objective_fa_only` |
| 4 | `experimentsCubic/mlp16_early0_fa4p0` |
| 8 | `experimentsCubic/mlp16_early0_fa8p0` |
| 16 | `experimentsCubic/mlp16_early0_fa16p0` |

每个实验读取三个训练种子的测试记录：

```text
runs/inter_uav_consistency_seed20260716/evaluation_test.json
runs/inter_uav_consistency_seed20260717/evaluation_test.json
runs/inter_uav_consistency_seed20260718/evaluation_test.json
```

因此绘图数据来自 18 份逐 seed 的测试 JSON。每个点的中心值和误差棒均由
MATLAB 重新计算，分别为三个种子的均值和样本标准差（分母 N-1）。已有的
`ablation_summary.json` 只用于发现实验及读取权重，不直接提供图中的纵轴值。

## 图中指标及 JSON 字段

| 图中指标 | `evaluation_test.json` 字段 |
|---|---|
| Event precision | `aggregate.event_precision` |
| Event recall | `aggregate.event_recall` |
| Event F1 | `aggregate.event_f1` |
| FAR | `aggregate.false_alarms_per_uav_hour` |
| DR@1s | `aggregate.detection_rate_within_s["1p0"]` |
| 平均检测延迟 | `aggregate.delay_mean_s` |

## 核验后的主要数值

以下数值仅用于人工核验；MATLAB 脚本不会读取本表。

| \(\lambda_{\mathrm{FA}}\) | Event F1 | FAR | DR@1s | 平均延迟 |
|---:|---:|---:|---:|---:|
| 0.5 | 0.9689 ± 0.0097 | 0.837 ± 0.307 | 0.601 ± 0.020 | 1.805 ± 0.080 s |
| 1 | 0.9763 ± 0.0019 | 0.601 ± 0.061 | 0.564 ± 0.009 | 2.026 ± 0.042 s |
| 2 | 0.9819 ± 0.0008 | 0.392 ± 0.034 | 0.541 ± 0.012 | 2.257 ± 0.055 s |
| 4 | 0.9835 ± 0.0016 | 0.349 ± 0.049 | 0.538 ± 0.009 | 2.212 ± 0.025 s |
| 8 | 0.9831 ± 0.0027 | 0.193 ± 0.043 | 0.504 ± 0.016 | 2.548 ± 0.120 s |
| 16 | 0.9832 ± 0.0007 | 0.134 ± 0.034 | 0.476 ± 0.021 | 2.816 ± 0.145 s |

\(\lambda_{\mathrm{FA}}=2\) 是后续外推实验采用的预先指定主设置。更高权重
继续降低 FAR，但早期检测率下降、平均时延增加，Event F1 基本进入平台区。

## 输出文件

运行 `paper/matlab/fig_lambda_fa_sensitivity.m` 后生成：

- `paper/generated_assets/figures/fig_lambda_fa_sensitivity.png`；
- `paper/generated_assets/figures/fig_lambda_fa_sensitivity.pdf`；
- `paper/generated_assets/figures/fig_lambda_fa_sensitivity.fig`；
- `paper/generated_assets/data/lambda_fa_sensitivity_raw.csv`；
- `paper/generated_assets/data/lambda_fa_sensitivity_summary.csv`。
