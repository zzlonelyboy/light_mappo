# ISAC-Assisted Event-Level GPS Spoofing Detection — 实验数据表

> 所有指标均为 test split 上 3 个种子的均值。K=5, τ=0.1, event-likelihood loss (λ_det=0, λ_delay=0.01, λ_FA=0.5, deadline=3.0s)。

---

## Table 1: Feature Ablation (fixed event-likelihood loss)

### (a) RCS = 6 (practical conditions)

| Feature Set | Dim | Event F1 ↑ | Event Prec. ↑ | Event Rec. ↑ | FAR ↓ | Delay (s) ↓ |
|---|---|---|---|---|---|---|
| F3 (no BS) | 3 | 0.873 | 0.925 | 0.826 | 0.99 | 4.86 |
| F8 (ISAC only) | 8 | 0.965 | 0.955 | 0.975 | 0.68 | 2.87 |
| **F9 (ISAC+neighbor)** | **9** | **0.966** | **0.959** | **0.974** | **0.62** | 2.94 |

### (b) RCS = 3 (degraded conditions)

| Feature Set | Dim | Event F1 ↑ | Event Prec. ↑ | Event Rec. ↑ | FAR ↓ | Delay (s) ↓ |
|---|---|---|---|---|---|---|
| F3 (no BS) | 3 | 0.823 | 0.914 | 0.749 | 1.05 | 5.49 |
| F8 (ISAC only) | 8 | 0.943 | 0.944 | 0.942 | 0.83 | 3.32 |
| **F9 (ISAC+neighbor)** | **9** | **0.949** | **0.955** | **0.944** | **0.65** | 3.39 |

> **摘要**: ISAC BS 残差不可替代 (F3→F8, F1 +0.09, FAR −31%)。F9 signed contrast 在 RCS=3 时 FAR 额外降低 22%，RCS=6 时降低 9%。

---

## Table 2: Step→Event Conversion (F9 feature set)

### (a) RCS = 6 (practical conditions)

| Loss | Event F1 ↑ | Event Prec. ↑ | Event Rec. ↑ | FAR ↓ | Delay (s) ↓ | Detect@2s ↑ |
|---|---|---|---|---|---|---|
| BCE (step-level) | 0.944 | 0.912 | 0.979 | 1.40 | 2.87 | 0.576 |
| **Event-likelihood** | **0.966** | **0.959** | 0.974 | **0.62** | 2.94 | 0.574 |

### (b) RCS = 3 (degraded conditions)

| Loss | Event F1 ↑ | Event Prec. ↑ | Event Rec. ↑ | FAR ↓ | Delay (s) ↓ | Detect@2s ↑ |
|---|---|---|---|---|---|---|
| BCE (step-level) | 0.918 | 0.886 | 0.952 | 1.81 | 3.35 | 0.496 |
| **Event-likelihood** | **0.949** | **0.955** | 0.944 | **0.65** | 3.39 | 0.493 |

> **摘要**: Event-likelihood 使 FAR 降低 56% (RCS=6) / 64% (RCS=3)，Event F1 同步提升 (+0.022 / +0.031)，检测延迟几乎不变。

---

## Table 3: Cross-Domain Generalization (event-likelihood loss)

### Panel A: RCS=6 S-curve → RCS=6 Multi-traj (trajectory generalization)

| Train Domain | Test Domain | Feat. | Event F1 ↑ | FAR ↓ | Delay (s) ↓ |
|---|---|---|---|---|---|
| RCS=6 S-curve | RCS=6 S-curve | F8 | 0.965 | 0.68 | 2.87 |
| RCS=6 S-curve | RCS=6 S-curve | F9 | 0.966 | 0.62 | 2.94 |
| RCS=6 S-curve | RCS=6 S-curve | F3 | 0.873 | 0.99 | 4.86 |
| RCS=6 S-curve | RCS=6 Multi-traj | F8 | **0.937** | **1.89** | 2.00 |
| RCS=6 S-curve | RCS=6 Multi-traj | F9 | 0.887 | 3.69 | 1.93 |

> F9 在跨轨迹时 FAR 劣化更严重 (0.62→3.69 vs 0.68→1.89)，signed contrast 对 S-curve 队形存在过拟合。

### Panel B: RCS=6 S-curve → RCS=3 Multi-traj (cross-RCS + cross-trajectory)

| Train Domain | Test Domain | Feat. | Event F1 ↑ | FAR ↓ | Delay (s) ↓ |
|---|---|---|---|---|---|
| RCS=6 S-curve | RCS=3 Multi-traj | F8 | 0.698 | 13.90 | 2.34 |
| RCS=6 S-curve | RCS=3 Multi-traj | F9 | 0.605 | 19.16 | 2.31 |

> ⚠️ RCS 不匹配时性能灾难性下降，FAR 飙升至 14–19 /UAV·h。

### Panel C: RCS=3 S-curve → RCS=6 Multi-traj (reverse cross-RCS)

| Train Domain | Test Domain | Feat. | Event F1 ↑ | FAR ↓ | Delay (s) ↓ |
|---|---|---|---|---|---|
| RCS=3 S-curve | RCS=3 S-curve | F8 | 0.943 | 0.83 | 3.32 |
| RCS=3 S-curve | RCS=3 S-curve | F9 | 0.949 | 0.65 | 3.39 |
| RCS=3 S-curve | RCS=3 S-curve | F3 | 0.823 | 1.05 | 5.49 |
| RCS=3 S-curve | RCS=6 Multi-traj | F8 | **0.968** | **0.81** | 2.09 |
| RCS=3 S-curve | RCS=6 Multi-traj | F9 | **0.967** | **0.80** | 2.09 |

> 🔑 **反直觉发现**: 用 RCS=3 训练反而在 RCS=6 多轨迹测试上泛化更好——劣质感知条件下训练出的模型学到的特征更鲁棒。

---

## 指标定义

| 指标 | 定义 |
|---|---|
| **Event F1** | 事件级 F1 = 2·P·R/(P+R)，其中事件定义为每次攻击的 [t_start, t_end) |
| **Event Precision** | TP / (TP + FP)，K-step 确认告警中真正对应攻击事件的比例 |
| **Event Recall** | TP / (TP + FN)，被检测到的攻击事件比例 |
| **FAR** | 虚警次数 / 总正常飞行小时 (/UAV·h)，正常飞行时间排除所有 UAV 的攻击区间 |
| **Delay** | 从攻击开始到首次 K-step 确认告警的时间 (s) |
| **Detect@2s** | 攻击开始后 2 秒内的累积检出率 |
