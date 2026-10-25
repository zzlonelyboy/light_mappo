## 1. 设计目标

无人机GPS欺骗检测模型在每个采样时刻输出一个攻击概率，但最终检测性能按照攻击事件进行评价。单独使用逐step二分类损失可以建立基本分类能力，却不能直接表达以下事件级目标：

1. 攻击发生后尽早形成首次正式报警；
2. 正常飞行区间内尽量不产生正式报警；
3. 训练阶段的软报警定义应与部署阶段的连续 $K$ 步确认规则保持一致。

因此，当前方案保留Binary Cross-Entropy（BCE）作为基础分类损失，并通过可微分Soft-Min $K$-step hazard构造早检损失和正常区间误报损失。

当前代码支持两种训练目标：

| `loss_objective` | 实际优化目标 |
|---|---|
| `bce` | 纯逐step BCE |
| `event_likelihood` | BCE、早检损失与正常区间误报损失的加权和 |

采用可微分frame-to-event建模：模型仍然输出逐step分数，但训练时增加从逐step输出到事件报警行为的可微映射。

Yoshinaga等人使用可微HSMM将声音事件的逐帧输出转换为包含起止边界的事件后验，并直接进行端到端优化，说明“逐帧预测与事件级目标联合训练”是可行的技术路线[2]。

---

## 2. 模型输出与硬报警规则

对于一架无人机的输入序列，GRU在step $t$ 输出单个logit：

$$
z_t\in\mathbb R.
$$

对应的攻击概率为：

$$
p_t=\sigma(z_t)=\frac{1}{1+\exp(-z_t)}.
$$

逐step硬判决为：

$$
\hat y_t=
\begin{cases}
1, & p_t\geq 0.5,\\
0, & p_t<0.5.
\end{cases}
$$

部署阶段不因为一个阳性step立即报警。只有连续 $K$ 个step均满足 $\hat y_t=1$ 时，才形成正式报警。当前设置为：

$$
K=5.
$$

在采样间隔 $\Delta t=0.1\ \mathrm{s}$ 下，一个确认窗口覆盖5个采样点。若以攻击开始step作为延迟零点，最早确认时刻为：

$$
d_{\min}=(K-1)\Delta t=0.4\ \mathrm{s}.
$$

首次正式报警后，该无人机针对当前攻击事件的检测任务结束；后续持续阳性不会被重复计为该事件的新检出。

---

## 3. 基础逐step BCE

设step标签为 $y_t\in\{0,1\}$，其中 $y_t=1$ 表示该无人机在该step受到GPS欺骗攻击。基础损失为：

$$
\mathcal L_{\mathrm{BCE}}
=
-\frac{1}{N}
\sum_t
\left[
y_t\log p_t
+
(1-y_t)\log(1-p_t)
\right].
$$

纯BCE基线的总损失为：

$$
\boxed{
\mathcal L_{\mathrm{total}}
=
\mathcal L_{\mathrm{BCE}}
}
$$

BCE负责维持逐step概率的基本可分性，并为事件级正则项提供稳定的初始分类目标。

---

## 4. 可微分连续 $K$ 步确认

硬条件“连续 $K$ 个step的概率均不低于0.5”不可直接反向传播。当前方案首先对每个 $K$ 步logit窗口计算归一化Soft-Min：[Max函数、Min函数与平滑的LSE（LogSumExp） - 知乎](https://zhuanlan.zhihu.com/p/611758446)

$$
\widetilde z_t
=
-T
\left[
\log
\sum_{i=0}^{K-1}
\exp\left(-\frac{z_{t-i}}{T}\right)
-\log K
\right],
$$

其中 $T>0$ 是温度参数。

当 $T$ 逐渐减小时：

$$
\widetilde z_t
\longrightarrow
\min_{i=0,\ldots,K-1}z_{t-i}.
$$

随后定义窗口的软条件报警概率（hazard）：

$$
h_t
=
\sigma\left(\frac{\widetilde z_t}{T}\right).
$$

该定义具有以下性质：

- 窗口内所有logit均较高时，$h_t$ 接近1；
- 窗口内只要存在一个明显偏低的logit，Soft-Min就会降低；
- 当窗口内所有logit均为0时，$\widetilde z_t=0$，对应 $p_t=0.5$ 的硬判决边界；
- $T$ 越小，软确认越接近硬连续 $K$ 步确认，但梯度也会更加集中。

当前设置为：

$$
T=0.1.
$$

Soft-Min hazard不是另一个部署状态机，而是硬 $K$ 步报警规则在训练阶段的可微分替代。

### 4.1 为什么连续 $K$ 步条件可以转化为窗口最小值

因为Sigmoid函数单调递增，所以：

$$
p_t\geq0.5
\quad\Longleftrightarrow\quad
z_t\geq0.
$$

一个窗口内连续 $K$ 个step均为阳性，等价于：

$$
z_{t-K+1}\geq0,\ldots,z_t\geq0.
$$

上述逻辑又等价于：

$$
\min_{i=0,\ldots,K-1}z_{t-i}\geq0.
$$

因此，窗口最小logit对应连续 $K$ 步规则中的“最弱一步”。只要其中一个step低于判决边界，硬最小值就会低于0，整个窗口不能形成正式报警。这也是选择Soft-Min而不是窗口均值的原因：均值可能被其余几个很高的概率补偿，无法表达“每一个step都必须为阳性”的逻辑。

Soft-Min用平滑的log-sum-exp替代不可微的硬最小值。Cuturi和Blondel在Soft-DTW中使用soft-min将离散最小化过程转化为可微损失，证明soft-min能够在保留“趋近最小值”含义的同时提供梯度[1]。Soft-DTW处理的是时间序列对齐，并不包含连续 $K$ 步报警；当前方案借鉴的是其soft-min松弛原理。

### 4.2 为什么公式中需要减去 log K

未经归一化的Soft-Min为：

$$
\operatorname{softmin}_T(\mathbf z)
=
-T\log
\sum_{i=1}^{K}
\exp\left(-\frac{z_i}{T}\right).
$$

当窗口中所有logit都恰好位于硬判决边界 $z_i=0$ 时：

$$
\operatorname{softmin}_T(\mathbf 0)
=
-T\log K<0.
$$

这会把软判决边界向负方向偏移，使“所有step都等于 $p=0.5$”的窗口无法保持在零边界。当前实现采用：

$$
\widetilde z_t
=
-T
\left[
\log
\sum_{i=0}^{K-1}
\exp\left(-\frac{z_{t-i}}{T}\right)
-\log K
\right],
$$

等价于给普通Soft-Min增加 Tlog K。于是：

$$
\widetilde z_t=0
\quad\text{when}\quad
z_{t-K+1}=\cdots=z_t=0.
$$

这样可以使软窗口边界与部署时的 $p=0.5$ 边界对齐。这里的归一化是当前实现针对 $K$ 步确认规则作出的设计，不是Soft-DTW原公式的直接复制。

### 4.3 温度参数的直观意义

温度 $T$ 控制软确认与硬确认之间的接近程度：

- $T$ 较大时，hazard变化平缓，接近阈值的step仍能获得梯度；
- $T$ 较小时，hazard更接近“窗口内最弱一步是否超过0”的硬判断；
- $T$ 过大可能使不满足硬规则的窗口仍具有较高软报警概率；
- $T$ 过小则可能使梯度集中在最小logit附近，训练更敏感。

在当前 $K=5$、$T=0.1$ 下，可以得到以下数值：

### 4.4 与硬状态机的区别

硬状态机只维护连续阳性计数：

$$
c_t=
\begin{cases}
c_{t-1}+1, & p_t\geq0.5,\\
0, & p_t<0.5.
\end{cases}
$$

当 $c_t=K$ 时触发报警。该过程包含阈值比较和计数重置，对logit几乎处处不可微，不能直接作为梯度训练目标。

Soft-Min方案不递归维护整数计数，而是并行展开所有长度为 $K$ 的滑动窗口，为每个窗口计算连续满足条件的软程度。部署阶段仍使用硬状态机，训练阶段只使用Soft-Min近似它的窗口判决逻辑。因此，两者的职责分别是：

- 硬状态机：产生最终可解释、确定性的正式报警；
- Soft-Min hazard：为相同的 $K$ 步判决逻辑提供可微训练信号。

---

## 5. 首次报警分布

设 $h_t$ 为时刻 $t$ 的条件窗口报警概率。首次在时刻 $t$ 形成报警的软概率为：

$$
q_t
=
h_t
\prod_{u<t}(1-h_u).
$$

其中：

- h_t 表示当前窗口满足软确认条件；
- $\prod_{u<t}(1-h_u)$ 表示此前所有有效窗口均未形成报警。

整个观察区间内均不报警的概率为：

$$
P_{\mathrm{no}}
=
\prod_t(1-h_t).
$$

为了避免长序列乘法下溢，代码在log空间计算：

$$
\log q_t
=
\log h_t
+
\sum_{u<t}\log(1-h_u),
$$

以及：

$$
\log P_{\mathrm{no}}
=
\sum_t\log(1-h_t).
$$

这种处理不需要逐step递归维护可微分状态，能够并行计算一个batch中的不同事件区间。

从概率结构看，$h_t$ 是离散时间条件hazard，$\prod_{u<t}(1-h_u)$ 是到时刻 $t$ 之前均未发生报警的survival probability，而$q_t$​ 是首次报警时间分布。离散时间神经生存模型同样通过条件hazard的乘积构造生存概率并使用似然训练[3]；DeepHit则直接学习离散时间事件分布，进一步说明首次事件时间可以作为神经网络的概率输出进行优化[4]。这些工作研究的是生存分析，而不是GPS欺骗检测。当前方案借鉴的是“hazard—survival—首次事件分布”的概率分解，再将其中的事件解释为首次正式报警。

![image-20260724090711226](C:\Users\zz\AppData\Roaming\Typora\typora-user-images\image-20260724090711226.png)

---

## 6. 攻击事件与正常区间

训练标签按照每架无人机的完整序列划分为：

- 最大连续攻击区间 $e=[a_e,b_e)$；
- 最大连续正常区间 $n=[c_n,d_n)$。

事件级损失不会使用跨越攻击边界的 $K$ 步窗口，避免将部分攻击、部分正常的窗口错误地用于正常误报约束。

只有满足以下条件的攻击事件才参与事件级损失：

$$
b_e-a_e\geq K.
$$

持续时间少于 $K$ 个step的攻击事件无法形成正式 $K$ 步报警，因此：

- 仍然参与逐step BCE；
- 不参与事件检出损失；
- 不参与早检损失。

当前数据集的 `warmup_steps=0`，所以正常区间从episode开始即可进入误报损失计算。

---

## 7. 事件检出损失

对于攻击事件e，事件期间至少形成一次正式报警的软概率为：

$$
P_{\mathrm{det},e}
=
\sum_{t\in e}q_{e,t}.
$$

事件检出损失定义为：

$$
\mathcal L_{\mathrm{det}}
=
-\frac{1}{|\mathcal E|}
\sum_{e\in\mathcal E}
\log\left(P_{\mathrm{det},e}\right).
$$

该项用于提高每个可观测攻击事件至少被检出一次的概率。

---

## 8. 早检损失

首先将攻击事件内的首次报警概率条件化：

$$
\bar q_{e,t}
=
\frac{q_{e,t}}{P_{\mathrm{det},e}}.
$$

$\bar q_{e,t}$ 表示在该事件最终形成报警的条件下，首次报警发生在时刻 $t$ 的概率。

设攻击开始step为 $a_e$，则首次报警延迟为：

$$
d_{e,t}
=
(t-a_e)\Delta t.
$$

使用早检截止时间 $H_d$ 对延迟归一化：

$$
\widetilde d_{e,t}
=
\min\left(
\frac{d_{e,t}}{H_d},
1
\right).
$$

早检损失定义为：

$$
\boxed{
\mathcal L_{\mathrm{early}}
=
\frac{1}{|\mathcal E|}
\sum_{e\in\mathcal E}
\sum_{t\in e}
\widetilde d_{e,t}\bar q_{e,t}
}
$$

当前设置：

$$
H_d=3\ \mathrm{s}.
$$

因此：

- 延迟越短，代价越接近0；
- 延迟达到或超过3 s，归一化代价为1；
- $H_d$ 是延迟归一化截止时间，不是漏检惩罚；
- 该项是在“事件被软检出”的条件下计算，不会单独惩罚漏检。

当前没有使用 `miss_penalty_s` 或独立的事件漏检损失。事件检出能力主要由BCE维持。

---

## 9. 正常区间误报损失

对于一个最大连续正常区间 $n$，区间内始终不形成报警的概率为：

$$
P_{\mathrm{no},n}
=
\prod_{t\in n}(1-h_t).
$$

设正常区间长度为 $L_n$，当前误报损失定义为：

$$
\boxed{
\mathcal L_{\mathrm{FA}}
=
-\frac{1}{|\mathcal N|}
\sum_{n\in\mathcal N}
\frac{\log P_{\mathrm{no},n}}{L_n}
}
$$

其含义是最大化正常区间的无报警概率，同时按照正常区间长度进行归一化，避免长区间仅因包含更多窗口而产生过大的损失。

该项具有以下边界：

- 惩罚的是正常区间内形成软 $K$ 步报警的风险；
- 连续正常区间分别计算，跨攻击边界的窗口被排除；
- 它不是对每个阳性step逐一计数；
- 它也不直接等于测试指标中的“误报次数/UAV·h”。

因此，$\mathcal L_{\mathrm{FA}}$ 是部署误报率的可微分替代目标，而不是硬误报事件率本身。

---

## 10. 当前实际总损失

当前代码实际参与反向传播的事件正则化损失为：

$$
\boxed{
\mathcal L_{\mathrm{total}}
=
\mathcal L_{\mathrm{BCE}}
+
\gamma_r
\left(
\lambda_d\mathcal L_{\mathrm{early}}
+
\lambda_f\mathcal L_{\mathrm{FA}}
\right)
}
$$

当前推荐实验参数为：

$$
\lambda_d=0.01,
\qquad
\lambda_f=0.5.
$$

因此，在不使用渐进引入时：

$$
\boxed{
\mathcal L_{\mathrm{total}}
=
\mathcal L_{\mathrm{BCE}}
+
0.01\mathcal L_{\mathrm{early}}
+
0.5\mathcal L_{\mathrm{FA}}
}
$$

三个部分的作用分别为：

| 损失项 | 优化目标 |
|---|---|
| $\mathcal L_{\mathrm{BCE}}$ | 保持基本逐step分类能力 |
| $\mathcal L_{\mathrm{early}}$ | 将首次报警概率向攻击早期移动 |
| $\mathcal L_{\mathrm{FA}}$ | 降低正常区间形成连续 $K$ 步报警的风险 |

该设计不是保证Precision、Recall和延迟同时单调改善，而是在BCE分类能力基础上改变早检与误报之间的优化权衡。

| 特征方案    | 损失                 | Step F1           | 事件Precision     | 事件Recall        | 事件F1            | 误报/UAV·h      | 平均延迟/s      |
| ----------- | -------------------- | ----------------- | ----------------- | ----------------- | ----------------- | --------------- | --------------- |
| Full        | BCE                  | 0.9252±0.0013     | 0.9198±0.0244     | **0.9830±0.0017** | 0.9502±0.0139     | 1.249±0.414     | **2.144±0.035** |
| Full        | 改进损失（proposed） | 0.9204±0.0022     | **0.9726±0.0112** | 0.9826±0.0017     | **0.9776±0.0051** | **0.402±0.169** | 2.223±0.046     |
| No neighbor | BCE                  | **0.9276±0.0006** | 0.9410±0.0248     | **0.9834±0.0019** | 0.9616±0.0140     | 0.901±0.404     | **2.138±0.061** |
| No neighbor | 改进损失（proposed） | 0.9203±0.0009     | **0.9793±0.0013** | 0.9819±0.0006     | **0.9806±0.0010** | **0.300±0.019** | 2.220±0.017     |

---

## 11. 事件损失渐进系数

若设置 `event_ramp_epochs=R>0`，训练第 $r$ 个epoch的事件损失系数为：

$$
\gamma_r
=
\min\left(\frac{r}{R},1\right).
$$

这表示事件级正则项在前 $R$ 个epoch中逐渐加入。

当前正式命令使用：

```text
--event-ramp-epochs 0
```

所以：

$$
\gamma_r=1
$$

从第一个epoch开始完整加入早检损失与误报损失。

需要注意，即使训练阶段使用渐进系数，验证阶段也始终按照 $\gamma=1$ 计算完整验证总损失。

---

## 12. Checkpoint选择与早停

当前checkpoint按照验证集总损失最低进行选择：

$$
\theta^\ast
=
\arg\min_\theta
\mathcal L_{\mathrm{validation,total}}(\theta).
$$

对于纯BCE模型，选择依据是验证集BCE：

$$
\mathcal L_{\mathrm{validation,total}}
=
\mathcal L_{\mathrm{BCE}}.
$$

对于事件正则化模型，选择依据是：

$$
\mathcal L_{\mathrm{validation,total}}
=
\mathcal L_{\mathrm{BCE}}
+
\lambda_d\mathcal L_{\mathrm{early}}
+
\lambda_f\mathcal L_{\mathrm{FA}}.
$$

Event Precision、Event Recall、Event F1和硬误报率会在验证阶段打印，但不直接参与checkpoint排序。

若验证总损失连续 `patience` 个epoch没有下降，则触发早停。

---

## 13. 训练与部署的一致性

训练和部署之间的对应关系如下：

| 部署逻辑 | 训练近似 |
|---|---|
| $p_t\geq0.5$ | logit边界 $z_t=0$ |
| 连续 $K=5$ 步阳性 | Soft-Min $K$-step hazard |
| 一个事件只认首次报警 | 首次报警分布 $q_t$ |
| 统计首次报警延迟 | 条件期望早检损失 |
| 正常区间不应形成报警 | 正常区间无报警似然 |

推理阶段不使用Soft-Min，也不使用训练中的温度参数。正式报警仍通过以下硬规则产生：

$$
p_t\geq0.5
\quad+\quad
连续K=5个阳性step.
$$

因此，Soft-Min hazard只用于训练阶段建立可微分的step-to-event映射。

---

## 14. 当前参数汇总

| 参数 | 当前取值 | 含义 |
|---|---:|---|
| `class_weighting` | `none` | BCE不使用正类加权 |
| `confirmation_steps` | 5 | 正式报警所需连续阳性step数 |
| `confirmation_temperature` | 0.1 | Soft-Min和hazard平滑温度 |
| `time_step_s` | 0.1 s | 数据采样间隔 |
| `detection_loss_weight` | 0 | 当前事件检出项不参与总损失 |
| `delay_loss_weight` | 0.01 | 早检损失权重 |
| `false_alarm_loss_weight` | 0.5 | 正常区间误报损失权重 |
| `early_detection_deadline_s` | 3 s | 延迟归一化截止时间 |
| `event_ramp_epochs` | 0 | 从第一轮完整加入事件损失 |
| `warmup_steps` | 0 | episode开始后立即监测 |
| 概率阈值 | 0.5 | 固定逐step二分类阈值 |
| Checkpoint规则 | 最低验证总损失 | 与当前优化目标保持一致 |

---

## 15. 对比实验设计

损失函数实验应保持以下条件完全一致：

- 数据集及train、validation、test划分；
- 输入特征集合；
- GRU结构；
- 随机种子；
- batch size、学习率与weight decay；
- $K=5$ 报警规则；
- checkpoint选择规则。

仅改变 `loss_objective`：

### 纯BCE基线

$$
\mathcal L_{\mathrm{BCE-only}}
=
\mathcal L_{\mathrm{BCE}}.
$$

### 事件正则化BCE

$$
\mathcal L_{\mathrm{event-reg}}
=
\mathcal L_{\mathrm{BCE}}
+
0.01\mathcal L_{\mathrm{early}}
+
0.5\mathcal L_{\mathrm{FA}}.
$$

比较指标至少包括：

1. step-level Precision、Recall和F1；
2. event-level Precision、Recall和F1；
3. Event Detection Rate；
4. 平均与中位首次报警延迟；
5. 0.5 s、1 s、2 s和5 s内检出率；
6. False Alarms per UAV-hour；
7. Clean Episode Alarm Probability；
8. 三个随机种子的均值与标准差。

在验证集完成模型和超参数选择后，test集只能统一打开一次用于最终报告。

## 参考文献

[1] M. Cuturi and M. Blondel, “Soft-DTW: a Differentiable Loss Function for Time-Series,” in *Proceedings of the 34th International Conference on Machine Learning*, PMLR, vol. 70, pp. 894–903, 2017. [Online]. Available: https://proceedings.mlr.press/v70/cuturi17a.html

[2] T. Yoshinaga, K. Tanaka, Y. Bando, K. Imoto, and S. Morishima, “Onset-and-Offset-Aware Sound Event Detection via Differentiable Frame-to-Event Mapping,” *IEEE Signal Processing Letters*, vol. 32, pp. 186–190, 2025, doi: [10.1109/LSP.2024.3509336](https://doi.org/10.1109/LSP.2024.3509336).

[3] M. F. Gensheimer and B. Narasimhan, “A Scalable Discrete-Time Survival Model for Neural Networks,” *PeerJ*, vol. 7, Art. no. e6257, 2019, doi: [10.7717/peerj.6257](https://doi.org/10.7717/peerj.6257).

[4] C. Lee, W. R. Zame, J. Yoon, and M. van der Schaar, “DeepHit: A Deep Learning Approach to Survival Analysis with Competing Risks,” in *Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence*, vol. 32, no. 1, 2018, doi: [10.1609/aaai.v32i1.11842](https://doi.org/10.1609/aaai.v32i1.11842).

### 文献与当前方法的对应边界

| 文献 | 可支撑的思想 | 不能直接声称的内容 |
|---|---|---|
| Cuturi and Blondel [1] | 使用soft-min平滑离散最小化并实现可微训练 | 该文没有提出连续 $K$ 步攻击报警 |
| Yoshinaga et al. [2] | 将逐帧输出可微映射为事件并进行事件级训练 | 该文采用HSMM，不是Soft-Min hazard |
| Gensheimer and Narasimhan [3] | 离散hazard、survival probability与事件似然 | 研究对象是生存分析，不是检测误报 |
| Lee et al. [4] | 直接学习离散首次事件时间分布 | 未使用当前的 $K$ 步确认与误报损失 |

因此，现有文献能够分别支撑Soft-Min松弛、frame-to-event训练和首次事件概率建模，但“归一化Soft-Min $K$-step hazard + 首次报警早检损失 + 正常区间误报损失”的完整组合属于当前研究的具体方法设计，应通过消融实验验证其有效性。









#### ISAC→安全检测的链路缺失

> **现有ISAC研究止步于"感知精度"，未延伸到"安全检测"。** 4个BS产生的距离/速度/方向测量可以用来定位，但没有人系统地将它们构造为"GPS是否被欺骗"的证据。

#### 连续K步报警规则对齐的GRU训练方法

> **几乎所有检测器都用逐step BCE/Focal Loss训练，但部署时采用连续K步确认规则。** 训练目标"每个step对不对"与部署目标"每个攻击事件是否被及时检测"之间存在不可忽视的偏差。
