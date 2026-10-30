# GPS 欺骗检测的闭环运动与 ISAC 感知建模方案

整理日期为 2026 年 10 月 30 日。

[toc]

当前运行暂时关闭报警后的导航缓解，保留 GPS 反馈运动和检测告警。默认 `navigation.enabled=true`、`navigation.mitigation_enabled=false`。文中的 ISAC 导航分支及第 4 节保留为可选能力，仅在显式开启缓解后生效。

本文件记录本轮讨论已经确定的运动、GPS、基站感知、机间测距与检测后导航切换模型。原论文逐项修改位置、特征接口的必要更新和实验迁移建议见 [原稿衔接与改进计划](manuscript_revision_plan.md)。

当前采用由信号模型和 CRB 支撑的测量级仿真。回波表达用于建立误差界，仿真不实际运行 MUSIC、APES 或匹配滤波估计算法。环境根据真实状态和设备配置计算协方差，再生成带噪参数观测。

具体设备与时间参数统一列于第 13 节。模型正文不预先代入设计数值。半波长阵元间距作为阵列结构直接写入公式。

## 1. 时间与状态

考虑 N 架 UAV 和 M 个固定基站。UAV、基站和时间片分别用 n、m、t 表示。所有位置均在统一世界坐标系中表达，时间片长度为 $\Delta t$。

$\mathbf p^n[t]\in\mathbb R^3$ 为 UAV n 在本片开始时的真实位置，$\mathbf v^n[t]\in\mathbb R^3$ 为本片执行的实际速度。

假设时间片足够短，在构造本片感知观测时，将距离和方向近似保持不变，同时保留速度与 Doppler 效应。

每片依次进行机间感知与基站感知。机间阶段，各 UAV 轮流发射，每次有效观测时长为 $T_U$，并接收多个可分离邻机的反射回波。基站阶段，各基站使用互不重叠的频段并行感知，每目标有效观测时长为 $T_B$。两个阶段分时进行，因此可以复用频段。

不另设保护时间，也不在正文列时间预算不等式。具体阶段时长与有效观测时长在参数表中说明。

GPS 在片初更新，导航指令作用于本片。本片等效感知几何统一对应片初位置，感知、定位和检测结果在片末可用。显式开启缓解时，确认报警从下一片开始影响导航。

## 2. 正常 GPS 和攻击

令 $\hat{\mathbf p}_{g,0}^n[t]$ 为未受攻击的 GPS 报告位置。

$$
\hat{\mathbf p}_{g,0}^n[t]
=\mathbf p^n[t]+\boldsymbol\epsilon_g^n[t],
\qquad
\boldsymbol\epsilon_g^n[t]\sim\mathcal N(\mathbf0,\mathbf R_g).
$$

正常 GPS 协方差为

$$
\mathbf R_g
=\operatorname{diag}(\sigma_h^2,\sigma_h^2,\sigma_v^2),
$$

其中 $\sigma_h$ 和 $\sigma_v$ 分别为水平单轴与垂直方向的误差标准差。该模型使用正常 GPS 报告的随机误差，不将 CEP 或水平误差半径直接等同于单轴标准差。

对于加性攻击，最终报告位置为

$$
\hat{\mathbf p}_g^n[t]
=\hat{\mathbf p}_{g,0}^n[t]+\mathbf a^n[t],
$$

其中 $\mathbf a^n[t]$ 为攻击偏置，攻击窗口之外为零。

随机注入采用

$$
\mathbf a^n[t]\sim
\mathcal N(\mathbf0,\sigma_a^2\mathbf I_3),
$$

其中 $\sigma_a$ 为攻击扰动标准差，$\mathbf I_3$ 为三维单位矩阵。

漂移攻击采用

$$
\mathbf a^n[t]
=r_n(t-t_{0,n})\Delta t\,\mathbf q_n,
$$

其中 $r_n$ 为偏置增长速率，$t_{0,n}$ 为攻击开始时隙，$\mathbf q_n$ 为在世界坐标中保持不变的单位方向。该表达在攻击窗口内使用。

重放攻击默认使用提前完成的独立正常飞行记录。令 $\mathbf p_{\mathrm{rec}}^n[k]$ 为录制源 UAV n 在记录索引 k 的真实位置，$\boldsymbol\epsilon_{\mathrm{rec}}^n[k]$ 为独立于受害飞行噪声的正常 GPS 误差，定义已完成的正常 GPS 记录

$$
\mathbf r_g^n[k]
=\mathbf p_{\mathrm{rec}}^n[k]+\boldsymbol\epsilon_{\mathrm{rec}}^n[k],
\qquad
\boldsymbol\epsilon_{\mathrm{rec}}^n[k]
\sim\mathcal N(\mathbf0,\mathbf R_g).
$$

每个使用该模式的重放 episode 单独生成一份录制源。录制源与受害飞行共享外生参考轨迹、初始队形和 UAV 索引，按照第 3 节的位置反馈模型，以自身正常 GPS 报告驱动独立真实运动。录制源没有攻击，不使用 ISAC 导航缓解，也不读取受害飞行攻击后的真实状态。

在攻击窗口内，受攻击 UAV 的报告位置替换为

$$
\hat{\mathbf p}_g^n[t]
=\mathbf r_g^n[t-d_{\mathrm{rep},n}],
$$

其中 $d_{\mathrm{rep},n}$ 为源轨迹的相位偏移步数。源记录在受害飞行递推前已经完成，所以上式不是从受害飞行当前历史中取样，也不把 $d_{\mathrm{rep},n}\Delta t$ 解释为实时链路延迟。重放在源索引有效时启动，不循环或裁剪索引。正常 GPS 噪声已经包含在记录中，复制样本后不再叠加另一份正常 GPS 噪声。

上述模式对应 `recorded_normal_episode`。可选的旧模式 `delayed_clean_stream` 仍采用

$$
\hat{\mathbf p}_g^n[t]
=\hat{\mathbf p}_{g,0}^n[t-d_{\mathrm{rep},n}],
$$

该旧模式读取受害飞行自身历史正常 GPS，$d_{\mathrm{rep},n}$ 才表示观测延迟步数，同样不重复叠加 GPS 噪声。旧保存配置若没有 `attack.replay_source` 字段，加载时保持这一旧语义。新默认配置明确选择独立正常记录模式。

攻击窗口之外和未受攻击 UAV 仍使用自身正常 GPS 报告。本片导航和检测使用同一份最终 GPS 报告，不能分别抽样两份正常 GPS 噪声。两种重放模式都只替换报告位置，受害飞行真值和 ISAC 观测仍由其自身闭环状态产生。

## 3. 参考轨迹、控制和真实运动

预设轨迹提供参考位置 $\mathbf p_{\mathrm{ref}}^n[t]$。参考速度与离散位置保持一致。

$$
\mathbf v_{\mathrm{ref}}^n[t]
=\frac{
\mathbf p_{\mathrm{ref}}^n[t+1]
-\mathbf p_{\mathrm{ref}}^n[t]
}{\Delta t}.
$$

令 $m_{\mathrm{nav}}^n[t]\in\{0,1\}$ 表示导航模式，其中 0 使用 GPS，1 使用 ISAC。当前默认关闭缓解，所有时间片保持模式 0。

令 $\hat{\mathbf p}_I^n[t]$ 为本片末得到的 ISAC 定位结果，它对应本片开始时的位置。令 $\mathbf v_c^n[t]$ 为本片速度指令，则本片可用导航输入为

$$
\hat{\mathbf p}_{\mathrm{nav}}^n[t]
=
\begin{cases}
\hat{\mathbf p}_g^n[t],
&m_{\mathrm{nav}}^n[t]=0,\\[1mm]
\hat{\mathbf p}_I^n[t-1]
+\Delta t\,\mathbf v_c^n[t-1],
&m_{\mathrm{nav}}^n[t]=1.
\end{cases}
$$

ISAC 分支将上一片定位结果传播到当前时刻，使用已知指令速度，不读取仿真真实位置或真实速度。

定义位置反馈增益

$$
\mathbf K_p
=\operatorname{diag}(k_{p,h},k_{p,h},k_{p,v}),
$$

其中 $k_{p,h}$、$k_{p,v}$ 分别为水平与垂直位置反馈增益。

速度指令为

$$
\mathbf v_c^n[t]
=\operatorname{sat}_{v_{\max}}
\left(
\mathbf v_{\mathrm{ref}}^n[t]
+\mathbf K_p
\left[
\mathbf p_{\mathrm{ref}}^n[t]
-\hat{\mathbf p}_{\mathrm{nav}}^n[t]
\right]
\right).
$$

$\operatorname{sat}_{v_{\max}}$ 保持输入向量方向，将模长限制在最大合速度 $v_{\max}$ 以内。

采用理想速度跟踪。

$$
\mathbf v^n[t]=\mathbf v_c^n[t],
\qquad
\mathbf p^n[t+1]
=\mathbf p^n[t]+\Delta t\,\mathbf v^n[t].
$$

初始真实位置为

$$
\mathbf p^n[0]=\mathbf p_{\mathrm{ref}}^n[0].
$$

因此，预设轨迹是跟踪目标。正常 GPS 误差和欺骗偏置都可以通过反馈改变实际运动。该运动模型保留位置外环与理想速度跟踪，没有声明已经模拟完整飞控动力学。

第 2 节的独立正常录制源复用同一运动递推，其导航输入始终为自身的 $\mathbf r_g^n[k]$，参考轨迹与初始位置保持一致，GPS 噪声使用单独的随机流。录制 seed 只确定源 GPS 噪声，完整复现还需对应参考轨迹、初始状态和受害 seed，不能把它等同于重新生成参考轨迹的环境 seed。

## 4. 可选的确认报警后导航切换

本节描述显式开启 `navigation.mitigation_enabled` 后的行为。当前默认只记录检测结果和确认告警，不执行模式切换。

令 $D^n[t]\in\{0,1\}$ 为本片末经过检测器连续确认与报警逻辑得到的报警结果。导航模式更新为

$$
m_{\mathrm{nav}}^n[t+1]
=\max\{m_{\mathrm{nav}}^n[t],D^n[t]\},
\qquad
m_{\mathrm{nav}}^n[0]=0.
$$

确认报警后，本次任务内保持 ISAC 导航。每片使用最近一次可用的 ISAC 定位结果传播，不从首次报警时刻开始无限累计航位推算。

误报同样触发切换，切换不读取真实攻击标签。导航锁定不等于把后续检测标签固定为攻击。原检测器的确认、报警触发与重新布防可以继续运行。

GPS 报告在切换后仍然生成并参与检测，不被 ISAC 位置覆盖。实际位置也不会重置到参考位置或测量位置。

## 5. 基站感知的真实几何

令基站位置为 $\mathbf z_m$，定义相对位置

$$
\boldsymbol\Delta_{nm}[t]
=\mathbf p^n[t]-\mathbf z_m
=
\begin{bmatrix}
\Delta x_{nm}[t]\\
\Delta y_{nm}[t]\\
\Delta z_{nm}[t]
\end{bmatrix}.
$$

真实距离、方位角和仰角为

$$
d_{nm}[t]=\|\boldsymbol\Delta_{nm}[t]\|,
$$

$$
\theta_{nm}[t]
=\operatorname{atan2}
\big(\Delta y_{nm}[t],\Delta x_{nm}[t]\big),
$$

$$
\phi_{nm}[t]
=\operatorname{atan2}
\left(
\Delta z_{nm}[t],
\sqrt{\Delta x_{nm}[t]^2+\Delta y_{nm}[t]^2}
\right).
$$

仰角相对世界水平面定义。定义由基站指向目标的单位方向函数

$$
\boldsymbol\ell(\theta,\phi)
=
\begin{bmatrix}
\cos\phi\cos\theta\\
\cos\phi\sin\theta\\
\sin\phi
\end{bmatrix}.
$$

由 UAV 指向基站的方向为

$$
\mathbf u_{nm}[t]
=-\boldsymbol\ell(\theta_{nm}[t],\phi_{nm}[t]).
$$

规定接近基站的径向速度为正。

$$
v_{r,nm}[t]
=(\mathbf v^n[t])^T\mathbf u_{nm}[t].
$$

令 c 为光速，$f_{c,m}$ 为基站载频，波长为

$$
\lambda_m=\frac{c}{f_{c,m}}.
$$

单基地回波的双程时延与 Doppler 为

$$
\tau_{nm}[t]=\frac{2d_{nm}[t]}{c},
\qquad
\mu_{nm}[t]=\frac{2v_{r,nm}[t]}{\lambda_m}.
$$

这些量属于环境真值，不直接作为处理端几何输入。

## 6. 基站资源、阵列和信号

### 6.1 资源与发射信号

令 $P_m^{\mathrm{tot}}$、$B_m^{\mathrm{tot}}$ 分别为每站总发射功率和总带宽。令 $P_{nm}[t]$、$B_{nm}[t]$ 为目标 n 实际分配的功率和有效带宽。

$$
\sum_{n=1}^{N}P_{nm}[t]=P_m^{\mathrm{tot}},
\qquad
\sum_{n=1}^{N}B_{nm}[t]\le B_m^{\mathrm{tot}}.
$$

参考配置采用均分。

$$
P_{nm}[t]=\frac{P_m^{\mathrm{tot}}}{N},
\qquad
B_{nm}[t]=\frac{B_m^{\mathrm{tot}}}{N}.
$$

基站频段互不重叠，全网带宽为

$$
B_{\mathrm{net}}=\sum_{m=1}^{M}B_m^{\mathrm{tot}}.
$$

令 $s_{nm,t}^B(\xi)$ 为目标 n 对应的已知下行数据流，其平均功率归一化。连续时间 $\xi$ 表示一次有效观测窗口内的信号时间。

令 $\mathbf f_{nm}[t]$ 为已知单位范数发射权重，则

$$
\mathbf x_{nm,t}^B(\xi)
=\sqrt{P_{nm}[t]}\,\mathbf f_{nm}[t]s_{nm,t}^B(\xi).
$$

不另设波形能量归一化积分、子载波网格或采样点数。频分用于分离发射数据流，回波可分离和正确关联作为另外的感知条件。

### 6.2 矩形阵列

基站收发阵列位于世界 xy 平面，两轴分别沿 y、x 方向，阵元间距为半波长。

令 $q\in\{t,r\}$ 分别表示发射与接收，$N_{q,a}^B,N_{q,b}^B$ 为两轴阵元数，总阵元数为

$$
N_q^B=N_{q,a}^BN_{q,b}^B.
$$

单位范数导向矢量按固定顺序排列，其分量为

$$
\left[\mathbf a_{q,m}^B(\theta,\phi)\right]_{i_a,i_b}
=\frac{1}{\sqrt{N_q^B}}
\exp\left[
-\mathrm j\pi
\left(
i_a\cos\phi\sin\theta
+i_b\cos\phi\cos\theta
\right)
\right],
$$

其中

$$
i_a=0,\ldots,N_{q,a}^B-1,
\qquad
i_b=0,\ldots,N_{q,b}^B-1,
$$

$\mathrm j$ 为虚数单位。

对应空间频率为

$$
a_{nm}[t]=\pi\cos\phi_{nm}[t]\sin\theta_{nm}[t],
\qquad
b_{nm}[t]=\pi\cos\phi_{nm}[t]\cos\theta_{nm}[t].
$$

### 6.3 回波与接收功率

令 $\beta_{nm}^B[t]$ 为观测窗口内未知的常复反射系数。分离后的单目标等效阵列回波为

$$
\begin{aligned}
\mathbf y_{nm,t}^B(\xi)
={}&
\sqrt{N_t^BN_r^B}\,\beta_{nm}^B[t]\,
\mathbf a_{r,m}^B(\theta_{nm}[t],\phi_{nm}[t])
\\
&\cdot
\left[
\mathbf a_{t,m}^B(\theta_{nm}[t],\phi_{nm}[t])
\right]^H
\mathbf x_{nm,t}^B(\xi-\tau_{nm}[t])
\\
&\cdot e^{\mathrm j2\pi\mu_{nm}[t]\xi}
+\mathbf w_{nm,t}^B(\xi).
\end{aligned}
$$

H 表示共轭转置。导向矢量的外积是 $N_r^B\times N_t^B$ 的空间响应矩阵。两个阵列均位于基站，UAV 是反射目标。

令 $N_0$ 为热噪声谱密度，$F_B$ 为线性接收噪声系数。各阵元噪声采用独立复白噪声，谱密度为 $N_0F_B$。

发射匹配系数为

$$
\delta_{t,nm}[t]
=
\left[
\mathbf a_{t,m}^B(\theta_{nm}[t],\phi_{nm}[t])
\right]^H\mathbf f_{nm}[t].
$$

令 $\sigma_{\mathrm{RCS},n}$ 为 UAV 的雷达散射截面积，$L_B$ 为线性剩余损耗。采用单位单阵元增益参考，环境设置

$$
|\beta_{nm}^B[t]|^2
=
\frac{\lambda_m^2\sigma_{\mathrm{RCS},n}}
{(4\pi)^3d_{nm}[t]^4L_B}.
$$

全接收阵列的真实信号功率为

$$
P_{r,nm}^B[t]
=N_t^BN_r^BP_{nm}[t]
|\beta_{nm}^B[t]|^2|\delta_{t,nm}[t]|^2.
$$

该功率已包含收发阵列增益。单位范数空间匹配处理对应的等效输出噪声谱密度仍为 $N_0F_B$。后续不再重复乘接收阵元数。

误差界只需要回波功率，所以环境计算 $|\beta_{nm}^B[t]|^2$ 即可。当前不实际生成复回波样本，无须为仿真测量另行生成反射相位。

## 7. 参数 CRB 与几何误差

采用远场点目标、相干阵列处理和白噪声参考模型。复回波幅度作为未知参数处理，不将已知 RCS 与幅度之间的距离关系作为额外测距信息。

### 7.1 空间频率与角度 CRB

矩形接收阵列的空间频率 CRB 为

$$
\mathbf C_{ab,nm}[t]
=
\operatorname{diag}
\left(
\frac{6N_0F_B}
{P_{r,nm}^B[t]T_B((N_{r,a}^B)^2-1)},
\frac{6N_0F_B}
{P_{r,nm}^B[t]T_B((N_{r,b}^B)^2-1)}
\right).
$$

定义

$$
r_{ab,nm}[t]=\sqrt{a_{nm}[t]^2+b_{nm}[t]^2}.
$$

角度恢复关系为

$$
\theta_{nm}[t]=\operatorname{atan2}(a_{nm}[t],b_{nm}[t]),
$$

$$
\phi_{nm}[t]
=
\begin{cases}
+\arccos(r_{ab,nm}[t]/\pi),
&\text{目标位于阵列平面上方},\\
-\arccos(r_{ab,nm}[t]/\pi),
&\text{目标位于阵列平面下方}.
\end{cases}
$$

覆盖半空间由安装和服务区域预先确定，不能由处理端读取仿真真实位置后决定。场景应满足该覆盖条件，并避开角度变换的奇异区域。

在 $0<r_{ab,nm}[t]<\pi$ 的非奇异角域，角度变换 Jacobian 为

$$
\mathbf G_{nm}[t]
=
\begin{bmatrix}
\dfrac{b_{nm}[t]}{r_{ab,nm}[t]^2}
&
-\dfrac{a_{nm}[t]}{r_{ab,nm}[t]^2}
\\[3mm]
-\dfrac{a_{nm}[t]}
{\pi r_{ab,nm}[t]\sin\phi_{nm}[t]}
&
-\dfrac{b_{nm}[t]}
{\pi r_{ab,nm}[t]\sin\phi_{nm}[t]}
\end{bmatrix}.
$$

角度 CRB 为

$$
\mathbf C_{\theta\phi,nm}[t]
=
\mathbf G_{nm}[t]\mathbf C_{ab,nm}[t]\mathbf G_{nm}[t]^T.
$$

### 7.2 时延、Doppler 与距离误差界

采用带内频谱能量近似均匀、窗口内功率近似平稳，并忽略时延与 Doppler 耦合的参考条件。

$$
C_{\tau,nm}^B[t]
\approx
\frac{3N_0F_B}
{2\pi^2P_{r,nm}^B[t]T_BB_{nm}[t]^2},
$$

$$
C_{\mu,nm}^B[t]
\approx
\frac{3N_0F_B}
{2\pi^2P_{r,nm}^B[t]T_B^3}.
$$

忽略耦合是另外声明的条件，不能仅由时间与频率上的均匀能量近似推出。

距离与径向速度 CRB 为

$$
C_{d,nm}^B[t]
=\left(\frac c2\right)^2C_{\tau,nm}^B[t],
\qquad
C_{v_r,nm}^B[t]
=\left(\frac{\lambda_m}{2}\right)^2C_{\mu,nm}^B[t].
$$

本方案沿用参考文献从共享信号和回波模型得到参数 CRB、再构造测量协方差的思路。当前闭式按照自身阵列归一化和接收功率口径表达，不声称逐式照搬参考论文的采样参数公式。

- [An ISAC-Based Beam Tracking Scheme Against Inter-Region Interference for the Multi-RSU V2I Scenario](https://doi.org/10.1109/TVT.2024.3488089)。
- [Cramér–Rao Bounds for Estimating Range, Velocity, and Direction with an Active Array](https://adogandzic.com/CRBpaper.pdf)。

## 8. 由 CRB 生成观测

定义真实测量向量

$$
\boldsymbol\zeta_{nm}[t]
=
\begin{bmatrix}
d_{nm}[t]&\theta_{nm}[t]&\phi_{nm}[t]&v_{r,nm}[t]
\end{bmatrix}^{T}.
$$

令 blkdiag 表示构造分块对角矩阵。生成端协方差为

$$
\mathbf R_{nm}^B[t]
=\operatorname{blkdiag}
\left(
C_{d,nm}^B[t],
\mathbf C_{\theta\phi,nm}[t],
C_{v_r,nm}^B[t]
\right).
$$

生成测量噪声与带噪观测

$$
\boldsymbol\epsilon_{nm}^B[t]
\sim\mathcal N(\mathbf0,\mathbf R_{nm}^B[t]),
\qquad
\hat{\boldsymbol\zeta}_{nm}[t]
=\boldsymbol\zeta_{nm}[t]+\boldsymbol\epsilon_{nm}^B[t].
$$

观测的四个分量为

$$
\hat{\boldsymbol\zeta}_{nm}[t]
=
\begin{bmatrix}
\hat d_{nm}[t]&
\hat\theta_{nm}[t]&
\hat\phi_{nm}[t]&
\hat v_{r,nm}[t]
\end{bmatrix}^{T}.
$$

角度采用弧度，方位角按周期处理。空间频率只用于计算角度误差界，不再独立添加一份空间频率噪声。

测得的 UAV 指向基站的方向为

$$
\hat{\mathbf u}_{nm}^{I}[t]
=-\boldsymbol\ell(\hat\theta_{nm}[t],\hat\phi_{nm}[t]).
$$

取测量协方差等于 CRB，表示理想有效估计的参考情形，不代表实际设备已经达到该精度。没有引入额外误差底限、膨胀系数或非高斯校准参数。

## 9. 功率报告与处理端协方差

真实接收功率 $P_{r,nm}^B[t]$ 是环境量。接收端从回波估计的功率或质量报告记为 $\hat P_{r,nm}^B[t]$。

当前采用准确功率报告假设。

$$
\hat P_{r,nm}^B[t]=P_{r,nm}^B[t].
$$

在测量级仿真中，该报告由环境提供，没有实际运行功率估计算法。等号表示忽略功率估计误差，不表示真实接收机天然知道精确功率。

处理端将报告功率、带噪角度和已知配置代入同一组 CRB 公式，计算 $\widehat{\mathbf R}_{nm}^B[t]$。

| 矩阵 | 计算依据 | 用途 |
|---|---|---|
| $\mathbf R_{nm}^B[t]$ | 真实功率、真实角度和配置 | 环境生成带噪观测 |
| $\widehat{\mathbf R}_{nm}^B[t]$ | 报告功率、带噪角度和配置 | 定位与测量不确定性处理 |

两者可以不同。处理端不读取真实位置、真实角度或真实反射系数。

## 10. 多基站 ISAC 定位

定义距离与角度观测

$$
\hat{\mathbf q}_{nm}[t]
=
\begin{bmatrix}
\hat d_{nm}[t]&\hat\theta_{nm}[t]&\hat\phi_{nm}[t]
\end{bmatrix}^{T}.
$$

其处理端协方差为 $\widehat{\mathbf V}_{nm}[t]$，即 $\widehat{\mathbf R}_{nm}^B[t]$ 的前三维子矩阵。

令 $\mathbf h_m(\mathbf p)$ 为候选位置 $\mathbf p$ 相对基站 m 的距离、方位角和仰角，由第 5 节几何公式将 $\mathbf p^n[t]$ 替换为 $\mathbf p$ 得到。

定义残差

$$
\mathbf e_{nm}(\mathbf p,t)
=\hat{\mathbf q}_{nm}[t]-\mathbf h_m(\mathbf p),
$$

其中方位角差按周期内最短角差计算。

定位采用协方差加权最小二乘。

$$
\hat{\mathbf p}_I^n[t]
=
\arg\min_{\mathbf p}
\sum_{m=1}^{M}
\mathbf e_{nm}(\mathbf p,t)^T
\widehat{\mathbf V}_{nm}[t]^{-1}
\mathbf e_{nm}(\mathbf p,t).
$$

权重先由功率报告和带噪角度计算，在本次定位求解中固定，不随候选位置反复改变。

定位结果来自同一组带噪距离与角度，不融合待检测 GPS，也不另行抽样位置噪声。

## 11. 机间感知与基站测距的对应关系

机间测距采用相同的单基地回波时延模型。UAV n 发射已知 ISAC 数据流并接收邻机 j 的反射。以下只用参数替换压缩重复表述，不改变回波传播、阵列作用、时延 CRB 或测量生成原理。

令 $\mathcal N_n[t]$ 为测量邻机集合。对 $j\in\mathcal N_n[t]$，相对几何为

$$
d_{nj}^U[t]=\|\mathbf p^j[t]-\mathbf p^n[t]\|,
$$

$$
v_{r,nj}^U[t]
=
\frac{
(\mathbf v^n[t]-\mathbf v^j[t])^T
(\mathbf p^j[t]-\mathbf p^n[t])
}{d_{nj}^U[t]}.
$$

复用第 6 至第 8 节中的回波、接收功率、时延 CRB、双程距离转换和带噪距离生成关系，作以下替换。

| 基站模型中的量 | 机间对应量 |
|---|---|
| 距离 $d_{nm}[t]$ | $d_{nj}^U[t]$ |
| 径向速度 $v_{r,nm}[t]$ | $v_{r,nj}^U[t]$ |
| 发射功率 $P_{nm}[t]$ | 机载平均发射功率 $P_U$ |
| 有效带宽 $B_{nm}[t]$ | 机载有效带宽 $B_U$ |
| 有效观测时长 $T_B$ | 每机单次观测时长 $T_U$ |
| 波长 $\lambda_m$ | $\lambda_U=c/f_{c,U}$，其中 $f_{c,U}$ 为机载载频 |
| 噪声系数 $F_B$ | 机载噪声系数 $F_U$ |
| 剩余损耗 $L_B$ | 机载剩余损耗 $L_U$ |
| 目标 RCS $\sigma_{\mathrm{RCS},n}$ | 邻机 RCS $\sigma_{\mathrm{RCS},j}$ |
| 收发增益 $N_t^BN_r^B\lvert \delta_{t,nm}[t]\rvert^2$ | 有效增益 $G_{t,nj}^U[t]G_{r,nj}^U[t]$ |

其中 $G_t^U,G_r^U$ 分别为邻机方向上的有效发射和接收增益。机载姿态、方向匹配与阵列作用包含在有效增益中。接收增益中已经包含阵列增益，不再额外乘阵元数。

上述替换也用于 $\tau=2d/c$ 和 $\mu=2v_r/\lambda$ 的信号参数关系。机间检测只使用距离输出，因此不重复展开空间频率和角度测量部分。

按替换后的接收功率得到距离 CRB $C_{d,nj}^U[t]$，生成

$$
\hat d_{nj}^U[t]
=d_{nj}^U[t]+\epsilon_{nj}^U[t],
\qquad
\epsilon_{nj}^U[t]\sim\mathcal N(0,C_{d,nj}^U[t]).
$$

一次发送服务多个可分离邻机，共同波形的功率和带宽不按邻机数量重复计算或均分。单目标 CRB 与支路误差独立近似，以回波充分分离、正确关联和残余相关较弱为参考条件。

## 12. 检测接口、执行顺序和信息边界

### 12.1 提供给检测器的量

GPS 报告对应的基站距离和方向为

$$
d_{g,nm}[t]
=\|\hat{\mathbf p}_g^n[t]-\mathbf z_m\|,
\qquad
\mathbf u_{g,nm}[t]
=\frac{\mathbf z_m-\hat{\mathbf p}_g^n[t]}{d_{g,nm}[t]}.
$$

定义用于解释的有符号距离与速度残差，以及非负方向残差。

$$
r_{d,nm}[t]=\hat d_{nm}[t]-d_{g,nm}[t],
$$

$$
r_{v,nm}[t]
=\hat v_{r,nm}[t]-(\mathbf v_c^n[t])^T\mathbf u_{g,nm}[t],
$$

$$
r_{u,nm}[t]
=1-(\hat{\mathbf u}_{nm}^{I}[t])^T\mathbf u_{g,nm}[t].
$$

机间有符号残差为

$$
r_{U,nj}[t]
=
\hat d_{nj}^U[t]
-\|\hat{\mathbf p}_g^j[t]-\hat{\mathbf p}_g^n[t]\|.
$$

这些量用于接回原稿的三层特征。原稿式（33）实际使用距离和速度绝对残差，原邻域特征也使用绝对距离差。衔接时继续使用 $|r_d|,|r_v|,r_u,|r_U|$，不把有符号中间量直接替换到原均值和最大值聚合中。

上传稿正文采用因果 GRU 与事件对齐训练。当前模型不将其改为 MAPPO。正常 GPS 误差下的归一化、位置投影方向和报警时间戳需要按衔接计划同步更新，这些接口修订建议不等于已经修改了算法代码。

### 12.2 每片执行顺序

默认重放模式在递推以下受害飞行时间片前，先生成独立正常记录。其他场景不需要该记录。

1. 由已有状态获得 $\mathbf p^n[t]$，生成正常 GPS 并施加攻击。
2. 根据上一片导航模式选择 GPS 或已传播的上一片 ISAC 位置。
3. 计算速度指令，本片按理想速度跟踪执行。
4. 用本片真实位置和实际速度构造机间与基站观测。
5. 处理带噪观测和功率报告，得到 ISAC 位置及检测结果。
6. 片末确认报警，更新下一片导航模式，并按运动方程得到下一片真实位置。

本片末的定位和报警不能反过来决定本片已经执行的速度。

### 12.3 信息边界

| 信息 | 环境是否知道 | 处理端是否直接获得 |
|---|---|---|
| 真实位置、实际速度、真实几何和 RCS | 是 | 不作为真值通道提供 |
| 真实接收功率和生成协方差 | 是 | 不直接作为几何观测提供 |
| 基站位置、设备配置和已知发射数据 | 是 | 是 |
| 已知控制指令 | 是 | 是 |
| 带噪距离、角度、方向、径向速度和机间距离 | 是 | 是 |
| 准确功率报告 | 环境提供 | 在明确的准确报告假设下获得 |
| 独立正常 GPS 录制源、源 ID 和随机种子 | 用于重放构造与离线审计 | 源记录不作为额外检测输入 |
| 真实攻击标签 | 用于数据标注与评价 | 不用于检测决策和导航切换 |

外部噪声的独立生成假设不意味着闭环状态、邻域残差或完整观测序列彼此独立。

独立正常记录单独保存在审计数组 `replay_source_gps_m` 中。没有该录制源的 episode 使用同形零占位数组，由 `metadata.replay_recording.present` 标识是否存在有效记录，不能仅根据数组数值判断。元数据同时记录源 ID、录制与受害 seed、UAV 索引映射和使用的源索引范围。每个重放 episode 私有其录制源，在同次数据集生成中不跨 episode 或数据划分复用。检测器仍仅接收既有 10 维特征，不接收源数组或源标识。

## 13. 参考参数与实现设置

功率均指相应有效观测窗口内的平均功率。公式中的功率、噪声系数和损耗使用线性值。以下是参考设计，不是某一整套硬件已经验证的配置。

| 类别 | 参数 | 当前参考值 |
|---|---|---|
| 系统规模 | UAV 数 N | 8 |
| 系统规模 | 基站数 M | 4 |
| 时间 | 时间片 $\Delta t$ | 20 ms |
| 时间 | GPS、控制、状态与检测更新 | 50 Hz |
| 时间 | 机间阶段 | 前 10 ms |
| 时间 | 基站阶段 | 后 10 ms |
| 时间 | 基站观测时长 $T_B$ | 10 ms |
| 时间 | 每机观测时长 $T_U$ | 1 ms |
| 时间 | 机间轮发净占用 | 8 ms，不另设保护时间 |
| 导航 | $k_{p,h}$ | $0.95\,\mathrm{s^{-1}}$ |
| 导航 | $k_{p,v}$ | $1.0\,\mathrm{s^{-1}}$ |
| 导航 | $v_{\max}$ | 8 m/s |
| 导航 | 模式 | 默认始终使用 GPS 反馈，关闭报警缓解；显式开启后确认报警才在下一片切换并保持 ISAC |
| GPS | $\sigma_h,\sigma_v$ | 1.5 m、3.0 m，已确认的主实验参考，垂直尺度为设计假设 |
| 攻击 | 随机强度、漂移速率、重放源相位偏移和窗口 | 按配置分档，默认 `recorded_normal_episode`，可选 `delayed_clean_stream` |
| 基站频谱 | 全网总带宽 $B_{\mathrm{net}}$ | 160 MHz |
| 基站频谱 | 每站带宽 $B_m^{\mathrm{tot}}$ | 40 MHz |
| 基站频谱 | 每目标有效带宽 $B_{nm}$ | 5 MHz |
| 基站频谱 | 各站中心载频 $f_{c,m}$ | 2.94、2.98、3.02、3.06 GHz |
| 基站功率 | 每站总发射功率 $P_m^{\mathrm{tot}}$ | 27 dBm，约 0.501 W |
| 基站功率 | 每目标发射功率 $P_{nm}$ | 约 17.97 dBm，0.06265 W |
| 基站阵列 | 发射阵列 | $8\times8$，64 阵元 |
| 基站阵列 | 接收阵列 | $8\times8$，64 阵元 |
| 基站阵列 | 阵元间距 | 半波长 |
| 基站匹配 | $\lvert \delta_t\rvert^2$ | 1，理想匹配参考 |
| 基站接收 | 噪声系数 | 8 dB，$F_B\approx6.31$ |
| 基站链路 | 剩余损耗 | 6 dB，$L_B\approx3.98$ |
| 机间频谱 | 载频 $f_{c,U}$ | 3 GHz |
| 机间频谱 | 有效带宽 $B_U$ | 30 MHz，在机间阶段复用上述总频段 |
| 机间发射 | $P_U$ | 0 dBm，1 mW |
| 机间发射 | 结构 | 单个有效宽覆盖发射通道 |
| 机间接收 | 阵列 | $4\times4$，16 阵元，并行相干接收 |
| 机间阵列 | 阵元间距 | 半波长，约 5 cm |
| 机间增益 | $G_t^U$ | 1，即 0 dBi |
| 机间增益 | $G_r^U$ | 理想匹配时为 16，约 12.04 dB |
| 机间接收 | 噪声系数 | 9 dB，$F_U\approx7.94$ |
| 机间链路 | 剩余损耗 | 6 dB，$L_U\approx3.98$ |
| 公共 | 热噪声谱密度 $N_0$ | −174 dBm/Hz |
| 公共 | 固定参考 RCS | $0.05\,\mathrm{m^2}$，约 −13.01 dBsm，取 3GPP 小型 UAV 平均 RCS 的近似值 |
| 观测生成 | 协方差 | 取对应 CRB |
| 功率质量 | 报告假设 | $\hat P_r=P_r$ |
| 目标处理 | 条件 | 回波可分离、正确关联、残余相关较弱 |
| 检测时序 | 连续确认片数 | 默认连续 5 片，20 ms 时间片下对应 100 ms，片数固定而时长随时间片长度变化 |
| 场景 | episode 时长、参考速度、编队和邻居数 | 默认 60 s、4 m/s、8 m 立方编队、3 个邻居，可配置 |
| 场景 | 基站位置 | (25,35,0)、(25,70,0)、(60,35,0)、(85,70,0) m，水平阵列，已知上半空间 |

反馈增益参考 [PX4 v1.14 参数文档](https://docs.px4.io/v1.14/en/advanced_config/parameter_reference#MPC_XY_P)。这里借鉴位置外环参数，不声称复现完整 PX4。

固定标称 RCS 取 [3GPP TR 38.901 V19.5.0 表 7.9.2.1-1](https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/19.05.00_60/tr_138901v190500p.pdf) 中小型 UAV 线性平均单基地 RCS 的近似值，由 −12.81 dBsm 对应的约 0.05236 m² 取整为 0.05 m²。当前不引入该标准的随机起伏模型。

机载矩形阵列维度是参考架构。此前核查的 [3 GHz UAV 阵列研究](https://repository.library.noaa.gov/view/noaa/31715/noaa_31715_DS1.pdf) 可支持该量级的阵列建模，但不能据此声称当前完整多通道机间 ISAC 设备已完成飞行验证。

## 14. 当前方案的边界

已经确定的是带噪 GPS、位置反馈运动、可选的下一片 ISAC 导航切换、资源与阵列显式化、接收功率形式 CRB、带噪观测生成和协方差加权定位。

GPS 标准差已由用户确认。实现采用表中的默认确认时长和地面基站部署，特征尺度同时计入正常 GPS 不确定性，方向残差采用二阶误差尺度。攻击分档可配置，完整训练、泛化和非理想条件实验仍需重新开展。实现接口与验证记录见 [模型实现说明](../model_implementation.md)。

本方案提供了条件明确的物理误差模型。准确功率报告、正确关联和取 CRB 作为协方差仍属于参考假设，不能据此声称已经完成实测校准或覆盖多径、漏检和关联失误。
