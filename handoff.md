# 当前方案交接文档

> 更新时间：2026-08-05  
> 项目根目录：`D:\Codes\project_spoofing_detect\light_mappo\isac_spoofing_detection`  
> 论文主文件：`D:\BaiduSyncdisk\研究生\研1\论文\ISAC_Assisted_Cooperative_GPS_Spoofing_Detection_for_UAV_Swarms_via_MAPPO (1)\main.tex`  
> 当前状态：方法、三组特征、事件对齐损失和主要实验已经形成闭环；正在重写论文的 Section III 和 Section IV。论文目录名中的 `via_MAPPO` 是历史遗留名称，当前方法不再使用 MAPPO、PPO 或多智能体强化学习。

## 1. 新对话应直接继承的结论

当前研究方案是一个面向无人机集群 GPS 欺骗的 ISAC 辅助时序检测框架。每架无人机利用多基站 ISAC 测量、本机 GPS 信息以及邻近无人机的 ISAC 机间测距，独立构造 10 维特征并运行 GRU 检测器。所有无人机采用相同的已训练参数，但在线推理和 GRU 隐状态相互独立，不进行集中式联合推理。

论文主线不是“利用 MAPPO 协同决策”，而是以下两个互补层次：

1. 在测量侧，以多基站 ISAC 和直接机间 ISAC 测距构造 GPS--ISAC 物理一致性证据，增强弱攻击、局部攻击和长期重复攻击的可观测性。
2. 在决策侧，以事件对齐损失将逐时间步预测映射为可微的事件级报警似然，使训练目标与部署时的连续 $K_{\mathrm c}$ 步确认规则一致。

一句话中心论点：

> Multi-base-station measurements provide local physical consistency evidence, while ISAC-based inter-UAV distance measurements provide complementary neighborhood geometry; an event-aligned GRU converts these measurements into timely and reliable spoofing alarms under a consecutive-confirmation rule.

## 2. 当前系统模型

### 2.1 场景和运动模型

- 无人机集合为 $\mathcal V=\{1,\ldots,N\}$，当前仿真使用 $N=8$。
- 时间步长 $\Delta t=0.1$ s，常规 episode 长度为 60 s。
- 无人机真实位置为 $\mathbf p^n[t]$，运动学模型为

  $$
  \mathbf p^n[t+1]
  =
  \mathbf p^n[t]+\mathbf v^n[t]\Delta t.
  $$

- 计划航向和速度命令来自 flight controller，并假定不受 GPS 欺骗影响。论文应明确这是观测层检测研究：攻击改变 GPS 导航观测，但不在当前模型中闭环反馈到飞行控制和真实轨迹。
- 仿真中的真实位置只用于生成物理测量和标签，不提供给检测器，也不用于动态邻居选择。

### 2.2 GPS 欺骗模型

当前攻击包括三类：

- Random：随机位置扰动；
- Replay：延迟重放历史 GPS 状态；
- Stealth：缓慢累积的位置漂移。

每类攻击分为 Weak、Medium、Strong：

| 攻击 | Weak | Medium | Strong |
| --- | --- | --- | --- |
| Random 标准差 | 0.25--0.75 m | 0.75--1.5 m | 1.5--3.0 m |
| Replay 延迟 | 0.2--0.5 s | 0.5--1.5 s | 1.5--3.0 s |
| Stealth 漂移率 | 0.1--0.3 m/s | 0.3--0.8 m/s | 0.8--1.2 m/s |

一次攻击事件采用固定的采样强度，下一事件重新独立采样。攻击事件完整落在 episode 内。

### 2.3 ISAC 几何测量

固定 ISAC base stations 为每架无人机提供：

- distance；
- line-of-sight direction；
- Doppler-derived radial velocity。

这些测量采用与 sensing SNR 相关的高斯误差模型。由多基站的 distance 和 direction 进一步估计 GPS-independent ISAC position $\hat{\mathbf p}_{\mathrm{isac}}^n[t]$。

邻近无人机之间通过 ISAC sensing 直接测量 pairwise distance。机间测距采用与基站 ISAC 测距相同的 delay-based principle，因此正文不需要重复完整的噪声公式，只需说明将基站 sensing SNR $\gamma^{n,m}[t]$ 替换为 inter-UAV sensing SNR $\gamma_{\mathrm u}^{n,j}[t]$。

建议的机间测量表达为：

$$
\hat d_{\mathrm{u,isac}}^{n,j}[t]
=
\left\|\mathbf p^n[t]-\mathbf p^j[t]\right\|_2
-\epsilon_{\mathrm u,d}^{n,j}[t].
$$

符号正负号必须与 Section III 已定义的基站测距误差模型保持一致。真实位置只在仿真器内部产生该测量。

邻居集合 $\mathcal V_n[t]$ 是根据机间 ISAC 的测量距离选出的 $k$ 个最近邻，当前主设置为 $k=3$。不需要再给出复杂的邻居集合优化公式，一句说明即可：

```latex
Based on the inter-UAV distance measurements defined in
\eqref{eq:inter_uav_distance_measurement}, UAV $n$ selects the $k$
nearest UAVs as its neighbor set $\mathcal{V}_n[t]$.
```

### 2.4 检测和事件定义

系统模型中只需使用抽象检测函数，不要提前展开 GRU 历史状态：

$$
\hat q^n[t]
=
\pi_{\boldsymbol\theta}
\!\left(\mathcal F(\mathcal I^n[t])\right),
\qquad
\hat q^n[t]\in[0,1].
$$

$\pi_{\boldsymbol\theta}$ 是所有无人机采用的同参数检测规则，但每架无人机独立执行。具体时序模型在 Section IV 说明。

正式报警规则为：当 $\hat q^n[t]\ge q_{\mathrm{th}}$ 连续满足 $K_{\mathrm c}=5$ 个时间步时触发报警。实现中的默认阈值为 $q_{\mathrm{th}}=0.5$。报警状态在连续 $K_{\mathrm c}$ 个负预测后关闭并重新 armed。`alarm_cooldown_steps` 是长时间评估中的附加分析参数，不属于基本连续确认规则；正文长期结果建议使用 cooldown=0。

真实攻击事件写作

$$
e=(n_e,t_e^{\mathrm s},t_e^{\mathrm f}),
$$

其中 $n_e$ 是受攻击无人机。第一次有效报警时间必须写成

$$
\tau_e
=
\min\left\{
t\in[t_e^{\mathrm s},t_e^{\mathrm f}]
\mid a^{n_e}[t]=1
\right\},
$$

不能在这里写成含义不明确的 $a^n[t]$。优化目标保持为事件级 missed detection、false alarm 和 delay 的加权组合，优化变量建议写成 $\boldsymbol\theta$：

$$
\min_{\boldsymbol\theta}
\lambda_{\mathrm{MD}}P_{\mathrm{MD}}
+\lambda_{\mathrm{FA}}P_{\mathrm{FA}}
+\lambda_{\mathrm D}\bar L_{\mathrm{delay}}.
$$

具体特征和可微优化方案在 Section IV 中给出。

## 3. 当前三组检测特征

论文正文统一使用

$$
\mathbf o^n[t]
=
\begin{bmatrix}
(\mathbf o_{\mathrm{att}}^n[t])^T &
(\mathbf o_{\mathrm{pos}}^n[t])^T &
(\mathbf o_{\mathrm{nbr}}^n[t])^T
\end{bmatrix}^{T}
\in\mathbb R^{10}.
$$

实验表格可以使用 B、S、U 简称：

| 论文实验名称 | 正文符号 | 维度 | 含义 |
| --- | --- | ---: | --- |
| B | $\mathbf o_{\mathrm{att}}$ | 6 | 多基站 GPS--ISAC 测量残差统计 |
| S | $\mathbf o_{\mathrm{pos}}$ | 2 | GPS--ISAC 位置差 |
| U | $\mathbf o_{\mathrm{nbr}}$ | 2 | GPS 推导距离与 ISAC 机间实测距离的差异 |

### 3.1 $\mathbf o_{\mathrm{att}}$：6 维局部 GPS--ISAC 特征

虽然保留符号 `att`，正文不再称其为 `attitude feature`，因为其中没有 roll、pitch 或 yaw。小节标题建议使用：

```latex
\subsubsection{Local GPS--ISAC Consistency Feature}
```

对 UAV $n$ 和 ISAC base station $m$，构造三个由测量不确定度归一化的残差：

$$
\boldsymbol\delta^{n,m}[t]
=
\begin{bmatrix}
\dfrac{|\hat d_{\mathrm{isac}}^{n,m}[t]-\hat d_{\mathrm{gps}}^{n,m}[t]|}
{\hat\sigma_d^{n,m}[t]}\\[2mm]
\dfrac{|\hat v_{\mathrm{r,isac}}^{n,m}[t]-\hat v_{\mathrm{r,gps}}^{n,m}[t]|}
{\hat\sigma_v^{n,m}[t]}\\[2mm]
\dfrac{|1-(\hat{\boldsymbol u}_{\mathrm{isac}}^{n,m}[t])^T
\hat{\boldsymbol u}_{\mathrm{gps}}^{n,m}[t]|}
{\hat\sigma_u^{n,m}[t]}
\end{bmatrix}.
$$

随后对全部基站分别计算 component-wise mean 和 maximum：

$$
\mathbf o_{\mathrm{att}}^n[t]
=
\begin{bmatrix}
(\bar{\boldsymbol\delta}^{n}[t])^T &
(\boldsymbol\delta_{\max}^{n}[t])^T
\end{bmatrix}^{T}
\in\mathbb R^6.
$$

代码中的准确顺序为：

1. `mean_range`
2. `mean_radial_velocity`
3. `mean_direction`
4. `max_range`
5. `max_radial_velocity`
6. `max_direction`

### 3.2 $\mathbf o_{\mathrm{pos}}$：2 维位置一致性特征

令

$$
\mathbf e_p^n[t]
=
\frac{
\hat{\mathbf p}_{\mathrm{gps}}^n[t]
-\hat{\mathbf p}_{\mathrm{isac}}^n[t]
}{d_p},
$$

其中当前实现的 `position_scale_m` 为 1 m。位置特征为

$$
\mathbf o_{\mathrm{pos}}^n[t]
=
\begin{bmatrix}
\|\mathbf e_p^n[t]\|_2\\
(\mathbf e_p^n[t])^T\mathbf h^n[t]
\end{bmatrix}
\in\mathbb R^2.
$$

第二项是带符号的航向投影，用于保留位置偏差相对计划运动方向的信息。

### 3.3 $\mathbf o_{\mathrm{nbr}}$：2 维机间测距一致性特征

当前邻域特征不是旧方案中的“邻机异常分数”“偏差模式比较”或“邻域共识”。对 $j\in\mathcal V_n[t]$，先由两架无人机上报的 GPS 位置计算

$$
d_{\mathrm{gps}}^{n,j}[t]
=
\left\|
\hat{\mathbf p}_{\mathrm{gps}}^n[t]
-\hat{\mathbf p}_{\mathrm{gps}}^j[t]
\right\|_2,
$$

再与 ISAC sensing 实测的机间距离比较：

$$
r^{n,j}[t]
=
\left|
\hat d_{\mathrm{u,isac}}^{n,j}[t]
-d_{\mathrm{gps}}^{n,j}[t]
\right|.
$$

最终特征为

$$
\mathbf o_{\mathrm{nbr}}^n[t]
=
\begin{bmatrix}
\min_{j\in\mathcal V_n[t]}r^{n,j}[t]\\
\max_{j\in\mathcal V_n[t]}r^{n,j}[t]
\end{bmatrix}
\in\mathbb R^2.
$$

推荐的引入文字：

```latex
The relative geometry among neighboring UAVs provides an additional
GPS-independent reference for spoofing detection. Using the neighbor
set $\mathcal{V}_n[t]$, UAV $n$ compares the pairwise distances inferred
from the GPS-reported positions with the corresponding inter-UAV
distances measured through ISAC sensing. The resulting discrepancies
characterize the consistency between the GPS-reported neighborhood
geometry and the physical geometry measured by ISAC sensing.
```

代码特征名为 `inter_uav_range_min` 和 `inter_uav_range_max`。

### 3.4 两种归一化不要混淆

1. B 特征内部使用 ISAC measurement uncertainty 归一化，用于处理不同物理量、不同 sensing SNR 和不同链路可靠性。
2. 输入 GRU 前再使用仅由 training split 计算的 mean/std 对完整特征做 dataset-level standardization。Validation、test 和外部测试只能复用训练集统计量。

## 4. GRU 检测器和事件对齐损失

### 4.1 模型部署方式

- 模型：单层 `CausalGRUDetector`；
- hidden dimension：64；
- 输出：每个时间步一个 logit；
- batch size：128；
- learning rate：$10^{-3}$；
- weight decay：$10^{-4}$；
- max epochs：500；
- patience：100；
- 训练设备：CUDA。

写作时避免 `a shared GRU jointly processes the swarm` 之类会暗示集中式部署的表达。推荐写法：

> Each UAV independently applies the GRU-based detector to its own feature sequence and maintains a separate recurrent state. The trained detector parameters are the same for all UAVs.

### 4.2 当前实际损失

当前主配置为：

```text
loss_objective = event_likelihood
confirmation_steps = 5
confirmation_temperature = 0.1
detection_loss_weight = 0.0
delay_loss_weight = 0.01
false_alarm_loss_weight = 2.0
early_detection_deadline_s = 3.0
event_ramp_epochs = 0
```

代码中独立 `detection_loss` 虽然会被计算和记录，但已从总损失中注释掉。因此论文只能写：

$$
\mathcal L
=
\mathcal L_{\mathrm{BCE}}
+0.01\mathcal L_{\mathrm{early}}
+2.0\mathcal L_{\mathrm{FA}}.
$$

不要把 `detection_loss_weight=0.1` 写入当前正式实验，也不要额外在总损失中加入 $\mathcal L_{\mathrm{det}}$。

事件对齐过程：

1. 使用 Soft-Min 对最近 $K_{\mathrm c}$ 个 logits 的最小值进行可微近似；
2. 将软确认值转换为 hazard；
3. 由 survival probability 得到首次报警分布；
4. $\mathcal L_{\mathrm{early}}$ 鼓励攻击后尽早首次报警；
5. $\mathcal L_{\mathrm{FA}}$ 抑制正常区间形成完整报警事件。

Section IV 的开头推荐保持以下逻辑：先构造本地与邻域特征，再说明每架无人机独立使用 GRU 建模时间演化，最后引出事件对齐损失如何把 step-level predictions 转换为 differentiable event-level alarm likelihoods。

## 5. 数据资产

### 5.1 主训练数据

`dataNew/d1_5000_multitrajectory_rcs6_seed20260714_interuav`

实际 manifest：

- 5000 episodes；
- episode-level 80/10/10 train/validation/test；
- Clean、Random、Replay、Stealth 各 1250；
- 全部为 S-curve，而不是多轨迹训练；
- 8 UAVs，60 s，$\Delta t=0.1$ s；
- RCS=6；
- 10 维特征；
- `neighbor_feature_mode=inter_uav`；
- `neighbor_selection_mode=dynamic_isac`；
- 训练邻域 $k=3$。

### 5.2 跨 RCS 测试数据

目录格式：

`dataNew/d1_1000_multitrajectory_test_rcs{RCS}_seed20260802_interuav`

其中 RCS 为 3、6、9、12、15、18、21、24。每个数据集 1000 episodes，实际轨迹均为 S-curve。不要被目录名中的 `multitrajectory` 误导。

### 5.3 邻域敏感性视图

RCS=3、6、12 已有 $k=2,4,5,6,7$ 的重构目录；$k=3$ 直接使用源数据。重构脚本只重算邻域特征，保持轨迹、攻击、GPS、基站 ISAC 和原始机间测量不变，因此适合隔离 test-time neighborhood size 的影响。

脚本：`scripts/rebuild_inter_uav_neighbor_views.py`。

### 5.4 轨迹泛化数据

`dataNew/d1_1600_trajectory_generalization_compact_test_rcs6_seed20260803_interuav`

- Straight、Circle、Helix、S-curve 各 400 episodes；
- 四类轨迹使用相同的紧凑刚性编队偏移；
- 只改变编队中心的运动路径；
- 三种特征和三个训练 seed 的评估均已完成。

### 5.5 1 h 长时间多攻击数据

`dataNew/d5_20x1h_multiattack_rcs6_seed20260804_interuav`

- 20 条 1 h episodes；
- 16 条包含多次攻击，4 条 clean；
- 8 UAVs；
- 277 个群组事件，682 个逐 UAV 攻击事件；
- 正文主结果使用 `alarm_cooldown_steps=0`。

### 5.6 当前保留的数据生成脚本

- 常规 training/validation/test 和 test-only：`scripts/generate_d1_multitrajectory.py`；
- 1 h 长时多攻击：`scripts/generate_d5_long_multiattack.py`。

主训练数据的生成命令曾使用：

```powershell
python scripts/generate_d1_multitrajectory.py `
  --dataset-type training `
  --episodes 5000 `
  --trajectory-types s_curve `
  --rcs 6 `
  --num-neighbors 3 `
  --seed 20260714 `
  --workers 16 `
  --episodes-per-shard 100 `
  --output-dir dataNew
```

## 6. 实验组织和主要结果

### 6.1 三种特征消融

| 实验标签 | 输入 |
| --- | --- |
| Only Position S | $\mathbf o_{\mathrm{pos}}$，2 维 |
| Local B+S | $\mathbf o_{\mathrm{att}}+\mathbf o_{\mathrm{pos}}$，8 维 |
| Full B+S+U | 全部三组特征，10 维 |

实验目录：`experimentsNew/direct_inter_uav_rcs6_fa2_ablation`。训练 seeds 为 20260716、20260717、20260718。

内部 test 的三 seed 均值：

| 特征 | Event Precision | Event Recall | Event F1 | FAR (/UAV·h) | Mean Delay |
| --- | ---: | ---: | ---: | ---: | ---: |
| Only Position S | 0.9532 | 0.8187 | 0.8809 | 0.5905 | 4.790 s |
| Local B+S | **0.9818** | 0.9613 | 0.9714 | **0.2630** | 3.011 s |
| Full B+S+U | 0.9621 | **0.9890** | **0.9754** | 0.5744 | **1.699 s** |

可支持的结论：B 是主要判别证据；U 进一步提高 Recall 和早期检出率，并显著缩短时延。内部 test 上 U 存在 Precision/FAR 代价，因此不能声称 Full 在所有指标上均优于 Local。

### 6.2 损失对比

Full 特征下已比较 BCE、Weighted BCE、Focal BCE 和 Event-aligned Loss，各 3 seeds。

| 损失 | Precision | Recall | F1 | FAR (/UAV·h) | Delay |
| --- | ---: | ---: | ---: | ---: | ---: |
| BCE | 0.899 | 0.997 | 0.946 | 1.637 | 1.273 s |
| Weighted BCE | 0.532 | 0.999 | 0.693 | 13.011 | 0.631 s |
| Focal | 0.892 | 0.997 | 0.942 | 1.782 | 1.246 s |
| Event-aligned | **0.962** | 0.989 | **0.975** | **0.574** | 1.699 s |

事件对齐损失的贡献是更合理的 Precision--Recall--FAR--Delay 折中，而不是每个指标都最优。Early-only、FA-only 和 Early+FA 的组成消融已经完成；FA-only 的 F1 略高，但响应更慢，完整损失选择更及时的综合运行点。

### 6.3 1 h 长时间结果，cooldown=0

| 特征 | Precision | Recall | F1 | FAR (/UAV·h) | Mean Delay |
| --- | ---: | ---: | ---: | ---: | ---: |
| Only Position S | 23.47% | 64.71% | 32.40% | 16.164 | 6.647 s |
| Local B+S | **96.36%** | 83.77% | 89.62% | **0.138** | 4.217 s |
| Full B+S+U | 95.93% | **96.29%** | **96.10%** | 0.179 | **2.437 s** |

这是当前最适合突出邻域特征价值的结果。Full 相对 Local 的 Recall 提高 12.51 个百分点、F1 提高 6.48 个百分点，时延降低约 42.2%；Precision 仅下降 0.43 个百分点。原因是机间测距在长期、多次和不同强度攻击中持续提供相对几何证据，弥补单机残差在局部异常不明显时的漏检。

### 6.4 紧凑编队轨迹泛化

| 特征 | Precision | Recall | F1 | FAR (/UAV·h) | Mean Delay |
| --- | ---: | ---: | ---: | ---: | ---: |
| Only Position S | 0.7309 | 0.6837 | 0.6980 | 4.5536 | 5.491 s |
| Local B+S | 0.8828 | 0.9863 | 0.9317 | 1.9950 | 2.023 s |
| Full B+S+U | **0.9805** | **0.9964** | **0.9883** | **0.3028** | **1.328 s** |

控制编队几何后，Full 在四类轨迹上的 F1 为 0.9734--0.9936，说明其高 Recall 并非通过普遍提高轨迹相关误警获得。旧多轨迹数据中 Circle/Helix 的较差结果主要受到扩展编队几何的混杂，不应继续引用。

### 6.5 跨 RCS 和邻域大小

- 跨 RCS：RCS=3 时 Full 相对 Local 的优势最大；RCS 增大后两者逐渐饱和。Full 通常具有更高 Recall 和更短时延，但在部分匹配或中高 RCS 条件下 Precision/FAR 未必优于 Local。
- 攻击类型：U 对 Random 最有效；Replay 和 Stealth 可能部分保留相对几何，因而增益较小。
- 邻域大小：当前模型以 $k=3$ 训练，只改变 test-time $k$。$k=4$--5 的 F1/FAR 较平衡，较小 $k$ 响应更快。不能据此声称 4 或 5 是普遍最优训练邻域。

## 7. 论文写作决策

### 7.1 Section III

建议结构：

1. Scenario and Attack Model；
2. ISAC-based Geometric Measurement Model；
3. Coordinated Detection Problem。

写作边界：

- ISAC measurement subsection 只建模测量，不在那里定义完整邻域信息集合；邻域信息和检测器可用信息放在 Coordinated Detection Problem。
- 基站测量和机间测距原理相同，因此机间部分简要引用同一 delay-based model，不重复完整 SNR 噪声公式。
- 身份绑定过程不需要展开协议设计。若需要，只需假设机间响应经过认证，测距结果能够与 UAV $j$ 对应。不要写过长的 pilot、response timing 和 processing interval 说明。
- 事件级优化问题保留，但对事件、误警和首次报警的定义应尽量简洁。
- 结尾可写 `The specific detection features and optimization method are presented in Section~\ref{sec:coordinated_detector}.`

### 7.2 Section IV

建议结构：

1. Coordinated Detection Framework；
2. Feature Construction；
   - Local GPS--ISAC Consistency Feature；
   - GPS--ISAC Position Consistency Feature；
   - Neighborhood Consistency Feature；
3. GRU-based Online Detector；
4. Event-aligned Training Objective。

必须删除旧 Section IV 中的：

- MAPPO、IPPO、actor、critic、reward、GAE 和 PPO clipping；
- 人工时间差分、滑动窗口统计、EWMA、CUSUM；
- 旧的一维 `signed_self_neighbor_anomaly_contrast`；
- 基于邻机 ISAC 定位结果重构的旧共识/编队偏差特征；
- 暗示所有 UAV 进行联合集中推理的措辞。

推荐的 Section IV 开头逻辑：

```latex
In this section, we first construct a coordinated feature
representation from each UAV's local observations and the information
obtained from neighboring UAVs. Each UAV then applies a GRU-based
detector independently to model the temporal evolution of its feature
sequence. The detector is trained using the proposed event-aligned
loss, which converts step-level predictions into differentiable
event-level alarm likelihoods so that model optimization reflects the
event-level detection objective.
```

### 7.3 用户已经明确的语言偏好

- 不使用 `UAV--BS` 作为正文短语，改写为 `between UAV $n$ and ISAC base station $m$`。
- 描述机间信息时优先使用 `inter-UAV distance measurements` 或 `inter-UAV distances measured through ISAC sensing`，避免反复使用生硬的 `direct inter-UAV ranging`。
- 不使用 `Together, ...` 作为机械总结句。
- 避免重复 `GPS-independent geometric information`。
- 保留 $\mathbf o_{\mathrm{att}}$、$\mathbf o_{\mathrm{pos}}$、$\mathbf o_{\mathrm{nbr}}$ 的符号形式，但正文不得把 $\mathbf o_{\mathrm{att}}$ 解释为无人机姿态角特征。
- 所有英文段落保持自然、简洁、IEEE 风格，尽量沿用当前 `main.tex` 的句式和衔接方式。

## 8. 代码与证据入口

### 8.1 实现

- 特征：`src/isac_spoofing/features.py`
  - `PaperFeatureExtractor.compute`
  - `compute_inter_uav_consistency_from_measurement`
- GRU：`src/isac_spoofing/models/torch_models.py`
  - `CausalGRUDetector`
- Event Loss：`src/isac_spoofing/training.py`
  - `_event_likelihood_losses`
  - 总损失中 `detection_loss` 被注释的位置
- 报警状态机与事件指标：`src/isac_spoofing/metrics.py`
  - `k_step_alarm_events`
  - `apply_alarm_cooldown`

### 8.2 主要结果

- 最新总体审计：`docs/实验总结.md`
- 三特征消融：`experimentsNew/direct_inter_uav_rcs6_fa2_ablation/ablation_summary.csv`
- 跨 RCS：`experimentsNew/direct_inter_uav_rcs6_fa2_ablation/cross_rcs_csv/cross_rcs_overall_summary.csv`
- RCS=6 攻击类型和强度：`experimentsNew/direct_inter_uav_rcs6_fa2_ablation/cross_rcs_csv/rcs6_by_attack_type_severity_summary.csv`
- 邻域敏感性：`experimentsNew/direct_inter_uav_rcs6_fa2_ablation/neighbor_sensitivity_analysis/neighbor_sensitivity_summary.csv`
- 轨迹泛化：`experimentsNew/direct_inter_uav_rcs6_fa2_ablation/trajectory_generalization_compact/trajectory_generalization_summary.csv`
- 主训练配置示例：`experimentsNew/direct_inter_uav_rcs6_fa2_ablation/runs/inter_uav_consistency_seed20260716/training/training_config.json`

### 8.3 旧材料警告

以下材料包含旧邻域特征、RCS=12、旧损失权重或“不以邻域为主线”的历史结论，不应直接用于当前论文：

- `docs/chat_handoff_summary_2026-07-24.md`
- `docs/主线.md`
- `docs/摘要.md`
- `docs/报告2.md`
- 旧的 `current`、`formation_deformation`、`uncertainty_robust` 和 `coordinated` neighbor modes。

当前权威来源优先级：实际代码和训练配置 > 最新数据 manifest/CSV/JSON > `docs/实验总结.md` > 历史文档。

## 9. 后续工作

现在可以继续写论文，但尚未完全投稿就绪。优先级如下：

### P0

1. 完成 Section III 和 Section IV 的正式 LaTeX 重写，保证符号、代码和公式一致。
2. 补充严格使用 validation 选阈值的传统 residual-threshold/CUSUM 基线，避免 test leakage。
3. 在方法、阈值、$K_{\mathrm c}$ 和损失权重完全冻结后生成一次性 final holdout。
4. 建立 master CSV/JSON，记录数据 manifest、checkpoint、seed、评估命令、cooldown、代码版本和结果路径。
5. 对核心比较增加 episode-level 或 hierarchical bootstrap 95% confidence intervals。

### P1

1. 协同欺骗或邻域污染：检查多架邻机受到相关欺骗、相对几何部分保持时 U 的失效边界。
2. 机间链路不完美：测距离群点、丢包、延迟和可用邻居不足。
3. 不同 UAV 数量、编队密度和拓扑。
4. 参数量、单 UAV 推理时延和机间测量/通信开销。

### P2

- 硬件在环或真实飞行；
- 非高斯 ISAC 噪声、多径、遮挡和同步误差；
- GPS 欺骗闭环反馈到飞行控制后的轨迹偏移。

## 10. 建议给新对话的第一条消息

```text
请读取项目根目录下的 handoff.md，并以其中“当前有效方案”为唯一上下文继续工作。
不要恢复 MAPPO/PPO、旧邻域异常对比或人工时序特征。当前任务是继续修改论文
main.tex 的 Section III/Section IV，使系统模型、B/S/U 三组特征、独立部署的 GRU
和事件对齐损失与代码实现完全一致。修改前请优先核对 handoff.md 中列出的代码、
training_config.json 和最新结果 CSV。
```
