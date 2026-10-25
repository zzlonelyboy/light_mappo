# ISAC辅助无人机GPS欺骗事件检测：当前聊天交接总结

> 更新时间：2026-07-24  
> 项目根目录：`D:\Codes\project_spoofing_detect\light_mappo\isac_spoofing_detection`

## 0. 给新聊天窗口的起始说明

请基于本文继续当前研究，不要重新回到MARL、PX4/Gazebo或必须使用邻域特征的旧方向。当前研究已经收敛为：

1. 使用数值仿真生成无人机飞行、GPS欺骗和ISAC测量数据；
2. 每架无人机独立部署GRU二分类检测器；
3. GRU逐step输出攻击概率；
4. 推理时采用`概率不小于0.5 + 连续K=5步阳性`形成正式报警；
5. 训练时比较纯BCE与面向该报警规则的可微事件对齐损失；
6. 当前最重要的研究问题是：事件对齐损失能否在基本保持Recall的同时，提高事件Precision、F1并降低误报率；
7. 邻域特征目前没有稳定增益，可以降为补充实验，不作为论文主线和必要创新点。

后续讨论必须区分：

- 已经确定的正式方案；
- 已有实验支持但仍需补强的结论；
- 已经讨论过但被放弃的旧方案。

---

## 1. 研究方向的演变与最终定位

### 1.1 已放弃的主线

前期曾考虑：

- MARL控制与检测联合建模；
- PX4 SITL + Gazebo；
- 强化学习环境中的导航动作和攻击检测；
- 群体级统一报警；
- 强行将邻域特征作为核心创新。

这些方向目前不再作为论文主线，原因是：

- 当前问题本质上是时序二分类和攻击事件检测，不要求多智能体联合决策；
- GPS欺骗检测并不天然构成MARL问题；
- 引入PX4/Gazebo会显著扩大实现成本，却不一定提高检测方法本身的学术贡献；
- 每架无人机本地独立报警更符合当前部署假设；
- 当前邻域特征的消融结果没有显示稳定、显著的增益。

### 1.2 当前正式研究对象

当前研究对象是：

> 在预定飞行轨迹和ISAC测量支持下，针对单架无人机的GPS欺骗攻击进行在线事件检测。

无人机群仍可作为数据生成和飞行场景，但论文不必强调“群体协同检测”。每架无人机独立使用本机GPS、ISAC及可选邻域信息进行判断。

### 1.3 当前研究问题

核心研究问题可以表述为：

> 如何利用ISAC提供的独立物理测量构造GPS–ISAC一致性特征，并使GRU的逐step训练目标与实际采用的连续 \(K\) 步事件报警规则对齐，从而提高GPS欺骗事件检测的可靠性？

---

## 2. 系统与数据生成方案

### 2.1 总体仿真方式

当前使用纯数值仿真，不依赖PX4、Gazebo或AirSim。

每个episode中生成：

1. 无人机真实轨迹；
2. 干净GPS或受欺骗GPS位置；
3. 由4个基站产生的ISAC距离、径向速度和方向测量；
4. 基于ISAC测量的无人机位置估计；
5. GPS–ISAC物理一致性特征；
6. 每架无人机、每个step的攻击标签；
7. episode、攻击类型、强度、起止时刻和受攻击无人机编号等元数据。

当前假设4个BS始终有效，不再考虑`valid BS`筛选，也不使用SNR clip等人为裁剪。

### 2.2 飞行轨迹

训练数据当前主要使用S型轨迹。轨迹代表无人机编队质心的运动，编队内各无人机保持给定相对队形。

当前典型设置：

- episode时长：60 s；
- 时间步长：\(\Delta t=0.1\) s；
- 每个episode约600个step；
- 攻击时刻与轨迹相位独立随机；
- 后续已生成直线、圆形、螺旋和S型轨迹的多轨迹测试集。

如果GPS欺骗不改变真实轨迹，这在当前“检测层”研究中是合理的，但论文必须明确：

> 飞行器按照预定真实轨迹运动，欺骗攻击作用于导航观测，而不在本研究中闭环反馈到飞行控制器。

因此当前方案研究的是观测层检测，而不是欺骗后的闭环轨迹偏移。

### 2.3 攻击场景

数据包括四种等比例场景：

1. `clean`：无攻击；
2. `random`：随机位置扰动；
3. `replay`：延迟/重放GPS状态；
4. `stealth`：缓慢累积的欺骗偏移。

当前讨论使用的攻击强度范围为：

```json
{
  "random_sigma_ranges_m": [
    [1.0, 1.5],
    [1.5, 2.5],
    [2.5, 3.5]
  ],
  "replay_delay_ranges_s": [
    [0.2, 0.5],
    [0.5, 1.5],
    [1.5, 3.0]
  ],
  "stealth_rate_ranges_mps": [
    [0.3, 0.5],
    [0.5, 1.0],
    [1.0, 1.5]
  ]
}
```

每个攻击事件：

1. 先确定攻击类型和弱/中/强等级；
2. 在对应范围内随机采样一次事件强度；
3. 整个事件保持该强度参数不变；
4. 下一事件重新独立采样。

当前正式方向要求攻击事件完整落在episode内，避免为了数据生成方便而人为截断攻击事件。历史数据中存在截断事件的版本不应作为最终正式数据。

### 2.4 受攻击无人机数量

部分早期数据允许攻击1–8架无人机，导致攻击目标多的episode贡献更多正样本和攻击事件。

当前用于邻域研究的`half_target`数据将受攻击无人机数量限制为1–4架，以降低群体同步攻击对邻域参考的污染。

训练仍以“每个episode的每架UAV都是一条样本序列”的`all_uavs`方式为主。数据划分必须以episode为单位完成，不能把同一episode中的不同UAV分到不同数据集合。

### 2.5 ISAC参数

当前正式数据倾向使用：

```text
radar_cross_section = 12
```

早期`RCS=24`的数据使ISAC测量过于理想，纯BCE即可得到极高性能，不利于展示方法差异。

需要注意：如果该参数同时吸收天线增益、处理增益和系统损耗，就不应在论文中简单称为无人机的纯物理RCS，而应称为“有效回波强度”或明确给出完整链路预算。

正式论文还应补充RCS或ISAC测量质量敏感性实验，证明方法并非只在单一理想信噪比下有效。

---

## 3. 当前输入特征与模型

### 3.1 当前推荐主输入：无邻域8维特征

在去掉人工时序特征和邻域特征后，基础输入为：

1. `mean_range`
2. `mean_radial_velocity`
3. `mean_direction`
4. `max_range`
5. `max_radial_velocity`
6. `max_direction`
7. `gps_isac_position_error_norm`
8. `gps_isac_position_error_heading_projection`

GRU负责从连续序列中学习时间关系，因此不再将CUSUM、EWMA等人工时间特征作为主输入。

### 3.2 邻域特征的历史尝试

已尝试或讨论过的邻域特征包括：

- 均值/最大值型邻域聚合；
- `signed_self_neighbor_anomaly_contrast`；
- 不确定度加权鲁棒邻域共识；
- `self_consensus_deviation`；
- `neighbor_consensus_bias`；
- `neighbor_consensus_dispersion`；
- 固定3近邻与动态3近邻；
- 纵向和横向编队形变；
- 图注意力或共享邻机编码器。

其中单变量可分性最强的是：

```text
signed_self_neighbor_anomaly_contrast
AUC = 0.6458
AP  = 0.1376
d   = +0.5585
```

但加入GRU后的端到端消融没有显示稳定增益。原因可能包括：

- 本机GPS–ISAC残差已经很强，邻域信息增量有限；
- 邻机也可能受攻击，使邻域参考被污染；
- 简单聚合丢失邻机之间的结构信息；
- 攻击模型不一定具有足够明确的空间相关性；
- 随机种子带来的波动与邻域特征增益处于同一量级。

因此当前建议：

> 论文主方案先不依赖邻域特征；邻域信息作为探索性或补充实验，不再强行作为核心创新。

### 3.3 数据标准化

标准化统计量只能由训练集计算。

计算范围包括：

- 训练集中的clean、random、replay和stealth场景；
- 训练集中的所有UAV；
- 训练集中的所有有效step。

验证集、测试集和外部轨迹测试集只能复用训练集得到的均值和标准差，不得重新估计。

### 3.4 检测网络

当前主模型：

```text
GRU
hidden_dim = 64
num_layers = 1
single logit output
```

在每个step \(t\)，网络输出logit：

\[
z_t\in\mathbb R.
\]

攻击概率为：

\[
p_t=\sigma(z_t).
\]

step级二元判断为：

\[
\hat y_t=\mathbb I(p_t\geq0.5).
\]

---

## 4. 当前报警与事件评价机制

### 4.1 正式报警规则

推理阶段采用：

\[
p_t\geq0.5
\quad\text{连续满足 }K=5\text{ 步}.
\]

也就是说，只有连续5个step都判断为攻击，才形成一次正式报警。该规则用于抑制孤立的逐step误报。

### 4.2 事件匹配逻辑

评价单位是“某架无人机的一次攻击区间”，而不是整个无人机群。

基本规则：

- 一次真实攻击事件只认第一次有效报警；
- 同一事件中的后续持续报警不重复计算TP；
- 一架无人机在后续新攻击事件中可以再次报警；
- 正常区间中形成的新报警事件计为FP；
- 攻击事件结束仍未形成报警计为FN；
- 漏检事件不进入“成功检出事件的平均延迟”，但必须通过Recall、漏检数和漏检率单独体现。

主要事件指标：

- Event Precision；
- Event Recall / Detection Rate；
- Event F1；
- false alarms per UAV hour；
- clean episode alarm probability；
- 首次报警延迟均值、中位数和四分位数；
- 0.5、1、2、5 s内的检出率；
- episode级bootstrap 95%置信区间。

训练日志中直接打印的Accuracy、Precision、Recall和F1默认是step-level；最终评价JSON中的`event_*`才是事件级指标。

---

## 5. 第一阶段：纯BCE训练

### 5.1 损失函数

第一阶段使用：

\[
\mathcal L_{\mathrm{BCE}}
=
-\frac1N
\sum_t
\left[
y_t\log p_t
+(1-y_t)\log(1-p_t)
\right].
\]

当前通常使用`class_weighting=none`，因为正类加权曾明显提高Recall，却导致模型更激进地报警、Precision下降和误报率上升。

### 5.2 BCE的作用

BCE负责建立基本的逐step判别能力。它能够让模型学习：

- 正常GPS–ISAC测量模式；
- 攻击期间的异常物理一致性；
- GRU序列上下文。

纯BCE通常在20–60个epoch左右达到最优，不需要像强化学习一样训练数千轮。这是因为：

- 任务是有监督分类；
- 标签直接提供监督信号；
- 数据量较大；
- 模型规模较小；
- 不是通过稀疏奖励探索策略。

---

## 6. 第二阶段：连续K步报警规则对齐损失

### 6.1 推荐方法名称

可暂称：

> **KAA-GRU: K-step Alarm-Aligned GRU**

中文可称：

> 连续K步报警规则对齐GRU。

名称强调“GRU + 改进损失”，不强调邻域特征。

### 6.2 Soft-Min连续K步确认

对于最近 \(K\) 个logit，定义：

\[
\widetilde z_t
=
-T
\left[
\log
\sum_{i=0}^{K-1}
\exp\left(-\frac{z_{t-i}}{T}\right)
-\log K
\right].
\]

它是最近 \(K\) 个logit最小值的平滑近似。

当 \(T\) 较小时：

\[
\widetilde z_t
\approx
\min(z_{t-K+1},\ldots,z_t).
\]

因此只有最近 \(K\) 个step的logit都为正，\(\widetilde z_t\)才倾向为正。这近似对应硬规则：

\[
p_t\geq0.5
\quad\text{连续 }K\text{ 步}.
\]

定义软确认hazard：

\[
h_t=\sigma(\widetilde z_t).
\]

### 6.3 首次报警概率

定义此前尚未报警的软生存概率，则首次在 \(t\) 报警的概率为：

\[
q_t
=
h_t
\prod_{u<t}(1-h_u).
\]

其中：

- \(h_t\)：当前窗口满足连续 \(K\) 步确认的软概率；
- \(\prod_{u<t}(1-h_u)\)：此前所有窗口都没有报警的概率；
- \(q_t\)：首次报警恰好发生在 \(t\) 的概率。

实现中使用log-space和累计和计算，避免长序列连乘下溢。

### 6.4 事件早检损失

对攻击事件 \(e\)，以首次报警分布计算归一化期望延迟：

\[
\mathcal L_{\mathrm{early}}
=
\frac1{|\mathcal E|}
\sum_{e\in\mathcal E}
\sum_{t\in e}
\widetilde d_{e,t}q_{e,t}.
\]

其中：

\[
\widetilde d_{e,t}
=
\min\left(
\frac{d_{e,t}}{H_d},
1
\right).
\]

\(H_d\)是早检期限，当前常用：

\[
H_d=3\text{ s}.
\]

该项鼓励攻击发生后更早形成第一次正式报警。

### 6.5 正常区间误报损失

对完整正常区间，计算在该区间内始终不形成正式报警的概率，并使用负对数似然惩罚正常期间的软报警风险。

当前实现使用长度归一化的正常区间无报警似然：

\[
\mathcal L_{\mathrm{FA}}
=
-\frac1{|\mathcal N|}
\sum_{n\in\mathcal N}
\frac{\log P_{\mathrm{no},n}}{L_n}.
\]

该设计不是逐step简单累加误报，而是约束一个正常区间内形成连续 \(K\) 步报警的风险。

### 6.6 当前实际使用的总损失

当前代码实际使用：

\[
\mathcal L
=
\mathcal L_{\mathrm{BCE}}
+
\gamma_r
\left(
\lambda_d\mathcal L_{\mathrm{early}}
+
\lambda_f\mathcal L_{\mathrm{FA}}
\right).
\]

当前常见设置：

```text
K = 5
confirmation_temperature = 0.1
delay_loss_weight = 0.01
false_alarm_loss_weight = 0.5
early_detection_deadline_s = 3
```

\(\gamma_r\)是事件损失的引入比例：

- `event_ramp_epochs=0`时，从第一轮起\(\gamma_r=1\)；
- 若设置渐进引入，则在指定epoch内由0逐步上升到1。

当前`detection_loss`虽然仍被计算和打印，但在`training.py`中已经被注释，不参与总损失。因此当前实现严格来说不是完整的`event_likelihood`，更准确地说是：

> BCE + 早检正则项 + 正常区间误报正则项。

正式论文、配置、测试和命名必须与这一事实统一。

### 6.7 checkpoint选择

当前checkpoint统一按照验证集总损失最小保存：

\[
\arg\min_{\text{epoch}}\mathcal L_{\mathrm{validation}}.
\]

验证集参与模型选择和早停，但不参与梯度更新。测试集不能参与模型选择、权重调整或超参数选择。

---

## 7. 当前主要数据集

### 7.1 主要S型轨迹数据

```text
data\d1_5000_complete_seed20260716_new_attack_fast_half_target_f11
data\d1_5000_complete_seed20260716_new_attack_fast_half_target_robust
```

两者均为：

- 5000 episodes；
- 80% train、10% validation、10% test；
- 60 s；
- \(\Delta t=0.1\) s；
- S型轨迹；
- RCS/有效回波强度12；
- clean/random/replay/stealth平衡；
- 攻击1–4架无人机。

两套数据的前8维非邻域特征、标签和episode ID已核对完全一致，最大绝对差为0。因此仅使用`no_neighbor`特征时，两套数据可以比较。

`f11`的3维邻域特征为：

```text
signed_self_neighbor_anomaly_contrast
self_neighbor_common_projection
mean_gps_isac_geometry_error
```

`robust`的3维邻域特征为：

```text
self_consensus_deviation
neighbor_consensus_bias
neighbor_consensus_dispersion
```

### 7.2 多轨迹外部测试集

```text
data\d1_1600_multitrajectory_test_rcs12_seed20260723
```

该数据集为test-only：

- 1600 episodes；
- straight、circle、helix、s_curve各400；
- clean、random、replay、stealth各400；
- 每个轨迹×场景组合100个episode。

目前已有总体多轨迹结果，但仍需按每种轨迹分别汇总，尤其应单独报告未见轨迹：

- straight；
- circle；
- helix。

由于当前测试结果已经多次用于方法讨论，如果这些结果影响了权重或方法选择，则该测试集已经带有开发集性质。最终投稿前应在方案完全锁定后重新生成一个冻结测试集。

---

## 8. 当前关键实验目录

### 8.1 纯BCE与邻域相关实验

```text
experiments\neighbor_loss_ablation_stage1_bce
experiments\neighbor_signed_contrast_only_stage1_bce
experiments\neighbor_signed_vs_none_stage1_bce_combined
experiments\neighbor_topology_ablation_stage1_bce
experiments\formation_deformation_ablation_stage1_bce
```

### 8.2 第二阶段事件损失实验

```text
experiments\neighbor_no_neighbor_stage2_event_likelihood
experiments\neighbor_signed_contrast_stage2_event_likelihood
experiments\neighbor_signed_vs_none_event_likelihood_scratch
experiments\formation_deformation_longitudinal_ablation_event_likelihood
```

### 8.3 多轨迹测试结果

```text
experiments\multitrajectory_test_bce_no_neighbor
experiments\multitrajectory_test_event_likelihood_no_neighbor
```

### 8.4 当前损失说明

```text
docs\current_loss_design.md
```

---

## 9. 已得到的主要结果

### 9.1 S型轨迹、无邻域、3个随机种子平均结果

| 指标 | 纯BCE | 当前事件损失 | 变化 |
|---|---:|---:|---:|
| Event Precision | 0.9387 | 0.9547 | +0.0160 |
| Event Recall | 0.9841 | 0.9826 | -0.0015 |
| Event F1 | 0.9608 | 0.9685 | +0.0077 |
| 误报/UAV·h | 0.938 | 0.675 | 约降低28% |
| 平均检测延迟 | 2.131 s | 2.144 s | 增加0.013 s |

当前事件损失的主要作用是：

- 提高事件Precision；
- 提高事件F1；
- 降低误报事件率；
- 使不同随机种子结果更稳定。

它目前没有证明能够降低平均检测延迟。

### 9.2 多轨迹总体结果

3个随机种子平均：

| 指标 | 纯BCE | 当前事件损失 |
|---|---:|---:|
| Event Precision | 0.9439 | 0.9704 |
| Event Recall | 0.9937 | 0.9933 |
| Event F1 | 0.9679 | 0.9816 |
| 误报/UAV·h | 0.9130 | 0.4674 |
| 平均检测延迟 | 1.4896 s | 1.5162 s |

多轨迹总体上同样表现为：

- Precision、F1和FAR改善；
- Recall基本保持；
- 延迟略微增加。

因此当前最稳妥的论文结论是：

> 报警规则对齐损失能够在基本保持事件检出率的同时，提高事件级Precision和F1，并减少正常飞行期间的误报事件。

目前不能声称该损失显著缩短首次报警延迟。

---

## 10. 当前方案是否构成算法创新

“事件对齐”可以构成算法方法贡献，但不能只因为修改了损失函数就自动被视为充分创新。

当前可组织为两个主要贡献：

### 贡献1：ISAC辅助的GPS物理一致性检测框架

- 利用ISAC独立距离、速度和方向测量；
- 构造GPS–ISAC位置与测量一致性特征；
- 在不依赖IMU或外部高保真平台的情况下形成可控数据生成与检测框架。

这是系统/应用层贡献。

### 贡献2：连续K步报警规则对齐的GRU训练方法

- GRU逐step输出攻击概率；
- Soft-Min将最近 \(K\) 个logit映射为软连续确认；
- hazard和生存概率构造首次报警分布；
- 使用事件早检和正常区间误报损失，使训练目标接近部署时的事件报警指标。

这是算法/方法层贡献。

在不考虑邻域特征时，以上两项对于无人机安全、通信感知或应用型期刊仍可形成清晰论文主线。没有必要为了增加创新点数量而保留无稳定增益的邻域模块。

---

## 11. 当前尚未完全满足的验证要求

### 11.1 Soft-Min与硬K步规则的一致性

数学上：

\[
T\rightarrow0
\quad\Longrightarrow\quad
\widetilde z_t\rightarrow
\min(z_{t-K+1},\ldots,z_t).
\]

但仍需在验证集测试：

```text
T = 1.0, 0.5, 0.2, 0.1, 0.05
```

并报告：

- 软确认与硬K步确认的决策不一致率；
- Soft-Min与硬最小值的误差；
- 梯度范数；
- NaN/Inf比例；
- 最终Precision、Recall、FAR和延迟。

### 11.2 当前单元测试与实现不一致

当前`training.py`注释了：

```python
training_config.detection_loss_weight * detection_loss
```

但部分测试仍按检测损失参与总损失的旧公式验证。这意味着：

- 实现；
- 单元测试；
- 配置字段；
- 日志；
- 论文公式

尚未完全一致。正式实验前必须决定是否彻底移除检测损失，然后同步更新全部相关内容。

### 11.3 ~~第二阶段额外训练的混杂~~（已澄清）

~~第二阶段模型从BCE checkpoint继续训练，因此性能改善可能来自：~~

~~- 新损失；~~
~~- 额外训练轮数；~~
~~- 更小学习率；~~
~~- checkpoint微调。~~

~~必须增加：~~

> ~~从同一个第一阶段checkpoint出发，以相同学习率、epoch和patience继续使用纯BCE训练。~~

~~只有事件损失优于该"continued BCE"对照，才能把增益主要归因于新损失。~~

**2026-07-24 更新**：当前正式 KAA 实验结果来自 `neighbor_signed_vs_none_event_likelihood_scratch`，其 `initial_checkpoint=null`，即 KAA 与 BCE 一样从随机初始化开始训练。两者使用相同的 `max_epochs`、`patience` 和 `lr`，`event_ramp_epochs=0` 意味着事件损失从 epoch 1 起生效。因此 KAA 并非"BCE 之后再训一轮"，而是 BCE 的等训练量替换方案。

但 Continued BCE 对照仍然有价值——用于验证 BCE 是否训练不充分，即给 BCE 额外训练轮数后能否追平 KAA。

### 11.4 损失组成消融缺失

至少需要5组：

| 组别 | 训练目标 |
|---|---|
| A | 纯BCE |
| B | 从A的checkpoint继续纯BCE |
| C | BCE + \(\lambda_d\mathcal L_{\mathrm{early}}\) |
| D | BCE + \(\lambda_f\mathcal L_{\mathrm{FA}}\) |
| E | BCE + \(\lambda_d\mathcal L_{\mathrm{early}}+\lambda_f\mathcal L_{\mathrm{FA}}\) |

这样才能判断：

- 提升是否只是额外训练造成；
- 早检项的独立作用；
- 误报项的独立作用；
- 两项是否互补。

### 11.5 替代损失基线不足

已有旧版单随机种子的onset-weighted BCE实验，但与当前数据和特征不完全一致。

建议在当前正式数据和3个相同随机种子下比较：

- 纯BCE；
- continued BCE；
- onset-weighted BCE；
- Focal Loss；
- 当前KAA损失。

### 11.6 鲁棒性和泛化实验仍需补充

需要正式汇总：

- random/replay/stealth分别的性能；
- weak/medium/strong攻击强度分层；
- straight/circle/helix/s_curve分别的性能；
- seen S-curve与unseen trajectories分别的结果；
- RCS/有效回波强度敏感性；
- 不同 \(K\) 和温度 \(T\)；
- 不同攻击持续时间；
- 长时间、多次攻击场景。

---

## 12. 建议的正式实验矩阵

### 12.1 基线

1. GPS-only GRU + BCE；
2. GPS–ISAC欧氏残差 + 固定阈值；
3. GPS–ISAC残差 + CUSUM；
4. ISAC特征MLP + BCE；
5. ISAC特征GRU + BCE；
6. ISAC特征GRU + Focal Loss；
7. ISAC特征GRU + onset-weighted BCE；
8. KAA-GRU。

其中“只保留一个本机残差特征再输入GRU”不是严格意义上的传统阈值基线；传统物理基线应直接对GPS–ISAC残差使用验证集确定的阈值或CUSUM。

### 12.2 主对比指标

必须同时报告：

- step Precision、Recall、F1；
- event Precision、Recall、F1；
- false alarms/UAV·h；
- clean episode alarm probability；
- 平均和中位检测延迟；
- DR@0.5/1/2/5 s；
- 3个或更多随机种子的均值和标准差；
- episode级bootstrap 95%置信区间。

不能只挑选Precision、Recall或延迟中的单一最优指标。

### 12.3 最终模型与随机种子

多随机种子用于估计训练稳定性，不应根据测试集挑选“最优seed”作为最终部署模型。

合理做法：

1. 用多个seed报告研究结果；
2. 根据验证协议确定结构和超参数；
3. 锁定全部方案；
4. 预先指定一个seed，或在train+validation上重新训练最终模型；
5. 只在新的冻结测试集上做一次最终评价。

---

## 13. 当前最重要的论文叙事

论文不应写成“提出大量特征并使用GRU”，而应形成以下闭环：

1. GPS欺骗使导航观测偏离真实运动；
2. ISAC提供独立于GPS的物理测量；
3. GPS–ISAC一致性为攻击检测提供可解释证据；
4. GRU学习这些物理一致性随时间的变化；
5. 实际报警需要连续 \(K\) 步确认，而普通BCE只优化逐step分类；
6. Soft-Min hazard将连续 \(K\) 步规则变为可微的软事件过程；
7. 事件早检和误报损失使训练目标更接近实际事件评价；
8. 实验检验其是否在维持Recall的同时提高Precision、F1并减少误报；
9. 多轨迹、多攻击类型和强度实验验证鲁棒性。

可概括为：

> 从ISAC物理一致性到报警规则对齐学习的无人机GPS欺骗事件检测框架。

---

## 14. 下一步工作优先级

### P0：在继续正式实验前必须完成

1. 决定`detection_loss`是否彻底移除；
2. 统一训练代码、测试、配置、日志和论文公式；
3. 增加continued BCE公平对照；
4. 完成early-only、FA-only和full loss消融；
5. 不再使用已有test结果调节方法。

### P1：支撑算法创新

1. Soft-Min与硬 \(K\)-step一致性实验；
2. onset-weighted BCE和Focal Loss对比；
3. 多随机种子成对比较；
4. 报告Precision–Recall–FAR–Delay权衡；
5. 对性能差值使用episode级bootstrap置信区间。

### P2：支撑泛化与论文说服力

1. 多轨迹按轨迹分别汇总；
2. 按攻击类型和强度分层；
3. RCS/SNR敏感性；
4. 长时多攻击测试；
5. 生成新的冻结最终测试集。

### P3：可选扩展

- 邻域特征；
- 编队形变特征；
- 图注意力邻机编码；
- IMU；
- 闭环飞行控制或PX4/Gazebo。

这些不应阻塞当前论文主线。

---

## 15. 新聊天继续时建议首先做的事情

新窗口可以直接从下面任务之一继续：

1. **代码一致性修正**：彻底移除未使用的`detection_loss`，同步修改测试、配置和文档；
2. **实验命令设计**：生成A–E五组公平损失消融命令；
3. **Soft-Min验证脚本**：实现软/硬K步一致性、温度和梯度稳定性分析；
4. **结果汇总脚本**：按seed、攻击类型、攻击强度和轨迹生成论文表格；
5. **论文方法章节**：基于当前实际实现撰写System Model、Feature Construction、KAA-GRU和Event Evaluation；
6. **最终数据协议**：锁定配置并生成不再参与调参的最终测试集。

继续工作时，应优先保证“公式—代码—日志—实验—论文结论”完全一致。
