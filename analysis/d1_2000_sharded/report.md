# D1-2000 数据集典型分析报告

## 结论

本次读取 2000 个 episode、16000 条无人机序列和 6,400,000 个 step-UAV 观测。完整性审计结果：**未发现结构性错误**。

该数据集可以进入初步训练，但正式结果必须按攻击类型、强度和截断状态分层报告。特别需要注意：GPS 安全噪声当前为 0，且全部数据只有一种 S 型轨迹；因此它验证的是当前数值模型内的检测能力，不能直接支持真实 GPS 噪声或未见轨迹上的泛化结论。

## 1. 完整性与规模

- 分片：20；唯一 episode ID：2000；重复 ID：0。
- 标签不一致单元：0；元数据不一致：0；形状错误：0。
- 4 个 BS 链路总数：25,600,000；无效链路：0。
- ISAC SNR 实际范围：10.24 至 103.84 dB；分析确认没有依赖 clip 后的固定边界。
- clean GPS 与真实位置的最大绝对差为 0 m，与配置中的零 GPS 噪声一致。

## 2. 场景、事件与标签

四类场景在每个 split 内严格等量：训练集每类 350，验证集和测试集每类均为 75。攻击 episode 共 1500 个。

episode 级平均正标签占比为 0.1055 ± 0.1006（episode 间 SD；95% CI 0.1010–0.1099）。这说明逐步标签仍明显偏向安全类，训练损失或采样策略需要处理类别不平衡。

攻击起点十等分计数为 [144, 152, 159, 140, 153, 156, 138, 170, 133, 155]。与均匀分布相比，χ²(9)=7.49，p=0.586，Cohen's w=0.071。
攻击起点与攻击发生时的 S 型轨迹相位未显示明显混杂：χ²(36)=40.69，p=0.271，Cramér's V=0.082。这里的结论仅针对当前 2000 个 episode 和当前相位估计。

实际攻击持续时间中位数为 107.0 steps，均值 101.5 ± 37.8 steps（95% CI 99.6–103.4），范围 1–160 steps。
共有 442/1500 个攻击被 episode 边界截断，比例 29.5%（Wilson 95% CI 27.2%–31.8%）。截断不是文件错误，但会产生极短事件，测试时应将 complete 与 truncated 分开报告。
有 89/1500 个攻击在 24-step 特征预热期内开始，比例 5.9%（95% CI 4.8%–7.2%）。这部分事件的检测延迟下界会受到预热规则影响。

## 3. F15 特征质量

训练集 15 个特征的标准差范围为 0.6472–734.9，没有零方差特征。所有特征均为有限值。
使用训练集均值和标准差后，验证集最大绝对均值漂移为 0.025 SD，测试集为 0.023 SD。

同一攻击 episode 内，以受攻击 UAV 的攻击区间均值与其非攻击区间均值做配对 Wilcoxon 检验，45 个攻击类型×特征比较中有 44 个在 BH-FDR 0.05 后显著。这只是特征响应审计，不等同于模型性能。绝对效应最大的组合为：

| 攻击 | 特征 | 秩二列效应 | 中位配对差 95% CI | q |
|---|---|---:|---:|---:|
| random | mean_range | 1.000 | [4.34, 5.34] | 3.35e-83 |
| replay | mean_range | 1.000 | [6.37, 8.29] | 3.35e-83 |
| replay | gps_isac_position_error_heading_projection | -1.000 | [-4.39, -3.52] | 3.35e-83 |
| random | mean_radial_velocity | 1.000 | [3.91, 5.01] | 3.35e-83 |
| replay | cusum_range | 1.000 | [51.8, 72] | 3.35e-83 |
| replay | max_direction | 1.000 | [7.46, 15.6] | 3.35e-83 |
| replay | max_radial_velocity | 1.000 | [28.7, 41.7] | 3.35e-83 |
| replay | max_range | 1.000 | [17.1, 22.6] | 3.35e-83 |

## 4. 时间记忆与事件结束后的特征尾迹

共有 884 个 episode 同时具备攻击前 24 steps 和攻击后 24 steps，可用于严格的前后配对。在 45 个攻击类型×特征比较中，12 个在 FDR 0.05 后仍有显著前后差异。这意味着标签在攻击结束时立即恢复为 0，但部分 CUSUM/邻机历史特征可能继续携带攻击信息。训练与评价必须保留完整 episode 顺序；不能把 step 随机打散后解释为独立样本。

## 5. 特征相关性

训练集全量 step-UAV 描述性 Pearson 相关中，绝对值最高的特征对为：

| 特征 1 | 特征 2 | r |
|---|---|---:|
| mean_direction | max_direction | 1.000 |
| mean_radial_velocity | max_radial_velocity | 1.000 |
| mean_range | max_range | 0.999 |
| neighbor_geometry_error_mean | neighbor_geometry_error_max | 0.938 |
| cusum_radial_velocity | cusum_direction | 0.931 |

这些相关系数用于检查冗余，不做显著性推断，因为连续 step 具有强时间相关性。

## 6. 是否可以开始训练

可以开始 MLP-F15 与因果 GRU-F15 的初步训练，标准化参数只能由 train split 计算。验证集用于选择模型、K 和训练轮次；测试集保持封存，最终只评估冻结方案。

正式实验至少分层报告：random/replay/stealth、weak/medium/strong、complete/truncated，以及攻击早/中/晚起点。当前数据不应被表述为真实飞行数据，也不能单凭本报告证明检测器有效。

## 输出文件

- `analysis_summary.json`：机器可读审计摘要。
- `dataset_overview.csv`、`attack_event_summary.csv`：规模与事件描述。
- `feature_distribution.csv`：F15 分布及基于训练集的 split 漂移。
- `feature_signal_tests.csv`：攻击区间与安全区间的配对检验。
- `post_event_recovery_tests.csv`：攻击结束后 24-step 特征尾迹。
- `feature_correlation_matrix.csv`：训练集描述性相关矩阵。
- 三张 PNG：事件分布、攻击特征效应、split 特征漂移。

## 统计说明

推断统计以 episode/事件为独立单位；逐 step 数据只用于描述性统计。连续变量使用均值、episode 间 SD、t 型 95% CI 和分位数；比例使用 Wilson 95% CI；分布/独立性使用卡方检验并报告 Cohen's w 或 Cramér's V；配对特征使用 Wilcoxon符号秩检验、秩二列效应和 episode bootstrap 中位差 95% CI，45 重比较采用 BH-FDR。

## 7. 四轮统计复核补充

### 极短与不可观测事件

当 K=5 时，有 23 个攻击 episode（涉及 114 个受攻击 UAV 事件）的实际持续时间不足 5 steps，因而不可能在攻击区间内形成 K 次连续阳性。最终事件指标必须把这些边界截断事件单独列为受审查的删失事件。

另有 3 个仅持续 1 step 的 stealth episode（14 个受攻击 UAV 事件），其 GPS 偏移最大值为 0，但标签为攻击。这来自 stealth 漂移在事件首步从 0 开始的定义，不是存储错误；它们对当步检测在信息论上不可辨识。

### ISAC 基线

episode 级 ISAC 平均定位误差为 1.089 ± 0.133 m（95% CI 1.083–1.095 m）。四类场景间 Kruskal–Wallis H(3)=0.78，p=0.854，epsilon-squared=0.0000；未发现 ISAC 定位误差随 GPS 攻击场景系统变化的证据。

### 无 clip 特征的重尾

没有执行 clip，但这使 9 个特征出现超过训练均值 100 SD 的值，最大达到 661.7 SD。单个特征中 |z|>10 的最高占比为 0.117%。这些点均为有限值，主要出现在高 SNR 链路下的归一化残差，并非 NaN/Inf；但均值–标准差标准化会受其显著影响。

| 特征 | 近似 P99 | 最大值 | 最大标准化值 | |z|>10 占比 |
|---|---:|---:|---:|---:|
| max_range | 70.1 | 5.73e+04 | 661.7 | 0.053% |
| mean_range | 23.8 | 1.43e+04 | 654.9 | 0.053% |
| max_direction | 70.4 | 2.58e+05 | 493.9 | 0.047% |
| mean_direction | 19.2 | 6.46e+04 | 493.8 | 0.047% |
| max_radial_velocity | 115 | 1.2e+05 | 406.1 | 0.067% |
| mean_radial_velocity | 32.3 | 3e+04 | 405.6 | 0.067% |

这不要求重新加入 clip。建议先保持当前无 clip 主数据不变，用训练集 z-score 完成基线，同时记录标准化后的最大输入、梯度范数和损失；若训练不稳定，再把 log1p/稳健标准化作为明确标注的消融，而不是静默改写数据。

### 子组样本量与运行环境

split×攻击类型×强度的 episode 单元格范围为 16–128。这足以做初步分层描述，但测试集最小单元格只有 16 个 episode，子组性能置信区间预计较宽。

本次统计环境为 NumPy 1.26.4、SciPy 1.9.1。当前组合是否触发兼容性警告：True。正式论文结果应在兼容的 NumPy/SciPy 环境中复跑并记录锁定版本。

新增审计文件：`attack_event_counts.csv`、`target_uav_counts.csv`、`feature_tail_audit.csv` 与 `feature_max_contexts.csv`。
