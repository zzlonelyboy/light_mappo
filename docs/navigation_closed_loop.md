# GPS 反馈、ISAC 导航切换与观测时序

本说明对应当前 schema 3.0 实现。完整感知公式和参数见 [完整建模方案](D:/Codes/project_spoofing_detect/isac_spoofing_detection_closed_loop/docs/isac_model_improvement/complete_model.md)。该实现是离散运动与测量级仿真，不连接飞控或 GNSS 硬件。

[toc]

## 1. 参考运动与正常 GPS

预定义轨迹是跟踪参考，不直接固定实际轨迹。每片的正常 GPS 和最终报告分别为

$$
\hat{\mathbf p}_{g,0}[t]=\mathbf p[t]+\boldsymbol\epsilon_g[t],
\qquad
\hat{\mathbf p}_g[t]=\hat{\mathbf p}_{g,0}[t]+\mathbf a[t].
$$

正常 GPS 采用单轴标准差 $(1.5,1.5,3.0)$ m。这是已选定的可配置仿真参考值，不是设备实测精度。

随机攻击叠加每报告扰动，漂移在固定世界方向上按物理速率增长。重放读取当前实际运动形成的历史正常 GPS 报告，保留原有噪声，不重复添加正常噪声，也不用首样本代替尚不存在的历史。短 episode 必须提供足够历史或缩短重放延迟。

## 2. 导航输入与动力学

令 $\mathbf p_{\mathrm{ref}}[t]$、$\mathbf v_{\mathrm{ref}}[t]$ 为参考位置和速度，$\mathbf v_c[t]$ 为当前指令速度。模式 $m[t]=0$ 使用 GPS，$m[t]=1$ 使用上一片 ISAC 位置的传播结果。

$$
\hat{\mathbf p}_{\mathrm{nav}}[t]=
\begin{cases}
\hat{\mathbf p}_g[t],&m[t]=0,\\
\hat{\mathbf p}_I[t-1]+\Delta t\,\mathbf v_c[t-1],&m[t]=1.
\end{cases}
$$

控制器只接收参考状态和可用导航位置。

$$
\mathbf v_c[t]=
\operatorname{sat}_{v_{\max}}
\left(
\mathbf v_{\mathrm{ref}}[t]
+\mathbf K_p[\mathbf p_{\mathrm{ref}}[t]-\hat{\mathbf p}_{\mathrm{nav}}[t]]
\right).
$$

当前 $\mathbf K_p=\operatorname{diag}(0.95,0.95,1.0)\,\mathrm{s^{-1}}$，最大合速度为 8 m/s。饱和保持速度方向，只限制三维模长。

$$
\mathbf v[t]=\mathbf v_c[t],
\qquad
\mathbf p[t+1]=\mathbf p[t]+\Delta t\,\mathbf v_c[t].
$$

这是位置外环和理想速度跟踪模型，没有速度滞后、加速度限制或姿态动力学。增益参考 [PX4 v1.14 参数](https://docs.px4.io/v1.14/en/advanced_config/parameter_reference#MPC_XY_P)，不表示复现完整 PX4。

## 3. 报警与导航切换

初始 $m[0]=0$。检测器分数完成连续确认后产生 $D[t]$，模式按以下关系更新。

$$
m[t+1]=\max(m[t],D[t]).
$$

默认阈值为 0.5，确认物理时长为 0.5 s，在 $\Delta t=0.02$ s 时对应 25 步。检测状态机仍可重新布防，导航模式在本 episode 内保持 ISAC，不自动回到 GPS。

误报也触发相同切换。控制不读取真实标签，不重置真实位置，切换后仍保留 GPS 和检测。

环境支持两种调用方式。

- 不提供 detector 时，run_episode 生成共同的 GPS 反馈运动数据，不虚构检测器或报警。
- 提供 detector(features) 回调时，每片返回形状为 (N,) 的概率向量。回调只收到特征，不收到真实位置、攻击标签或跟踪误差。

低层接口按顺序调用 step 和 apply_detection_probabilities。分数必须对应刚完成的时隙，每片只提交一次，并从第零片起连续提交。确认后只影响下一片，不反向改变已经执行的指令。

## 4. 一片的时间顺序

每片为 20 ms。先进行 10 ms 机间阶段，再进行 10 ms 基站阶段。机间每机轮发观测 1 ms，基站每目标观测 10 ms，不另设保护时间。

假设时间片足够短，在构造本片观测时，将距离和方向近似保持不变，同时保留速度与 Doppler 效应。

1. 获得片初真实位置，生成正常 GPS 并注入攻击。
2. 选择 GPS 或上一片 ISAC 的传播结果。
3. 计算并执行本片速度指令。
4. 根据片初真实位置和本片速度生成机间与基站观测。
5. 根据带噪观测和功率报告定位，随后由特征计算概率。
6. 片末确认，下一片使用更新后的导航模式。

time_s 对应片初，available_time_s 对应片末。若攻击从第 $t_s$ 片开始，首报警来自第 $\tau_e$ 片，报警可用延迟为

$$
D_e=(\tau_e-t_s+1)\Delta t.
$$

检测比较同片 GPS 与同片 ISAC。控制只能使用当前可用的定位结果。

## 5. 感知与检测信息边界

当前感知根据接收功率计算 CRB，再生成带噪距离、角度与径向速度。接收端由功率报告和带噪角度计算自己的协方差，WLS 求解时固定权重。$\hat P_r=P_r$ 是准确报告的显式参考假设。

基站采用世界水平面阵列与预先指定的覆盖半空间。默认参考部署将四站高度设为地面，避免旧布局被参考轨迹穿过。自定义场景也需满足覆盖与角域条件，不能用真实位置逐片为接收端决定分支。

机间沿用同一单基地测距原理，只替换相对几何与机载参数。有效接收增益已含阵列作用，不重复乘阵元数。回波可分离、正确关联与残余相关较弱仍是参考条件。

默认输入为 10 维。径向速度预测使用已知指令，位置投影由指令方向得到，零指令时投影置零。没有新增真实速度或真实航向传感器。残差尺度包含正常 GPS 误差，方向项使用二阶误差尺度。

## 6. 数据与复现

schema 3.0 保存片初状态、片末可用时刻、导航输入、模式、确认报警、检测概率和测量质量。未提供 detector 时不生成 detection_probability，confirmed_alarm 和 navigation_mode 均为零。

参考位置、跟踪误差与真值用于审计，不直接送入 detector。已知指令速度用于一致性特征。

    python scripts/validate_navigation_closed_loop.py --duration-s 6 --output tmp/navigation_model_validation.json
    python -m unittest discover -s tests -p test_model_navigation.py -v

验证脚本检查四种参考轨迹和四种场景的动力学、观测时刻与历史重放，并用正常 episode 上的可控分数验证误报后下一片切换。可控分数仅用于接口核验，不代表训练后的检测能力。

当前没有风、气动、姿态、执行误差、避碰或设备实测校准。旧数据和旧模型结果不会自动变为当前结果。
