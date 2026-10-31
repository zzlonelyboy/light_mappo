# ISAC 辅助 GPS 欺骗检测与导航闭环

当前实现对应 [完整建模方案](docs/isac_model_improvement/complete_model.md)。模型使用接收功率形式 CRB、带噪 GPS、位置反馈控制和报警后的 ISAC 导航，检测输入保持 10 维。

## 当前配置

- 时间片为 20 ms，GPS、控制和检测每片更新。前 10 ms 机间轮发，后 10 ms 四站并行感知，结果片末可用。
- S 形参考航线中心从 `(0,0,20)` m 飞至 `(100,100,60)` m，相比此前整体抬高 10 m，编队相对位置保持不变。高度调整通过参考轨迹和初态进入闭环，GPS 和 ISAC 观测仍由递推后的真实状态生成。
- 全网 160 MHz，每站 40 MHz，八目标均分后每目标 5 MHz。每站总功率 27 dBm，收发阵列均为 8×8。
- 机间复用 30 MHz，每机观测 1 ms，功率 0 dBm，有效接收增益 16。同一波形不按邻机数量重复计资源。
- GPS 水平单轴标准差为 1.5 m，垂直为 3.0 m，属于用户选定的主实验参考。水平尺度参考普通 GNSS 模块精度，垂直尺度为设计假设。
- 重放默认采用 `attack.replay_source=recorded_normal_episode`。每个重放 episode 先完成一份独立正常 GPS 反馈飞行记录，与受攻击飞行共享参考计划和初态，但使用独立 GPS 噪声。攻击窗口只复制该记录的 GPS 样本，不重复加噪，实际运动和 ISAC 观测仍由受攻击飞行生成。
- 位置反馈增益为水平 0.95、垂直 1.0 s⁻¹，理想速度跟踪，合速度上限 8 m/s。
- 固定标称 RCS 为 0.05 m²，取 3GPP 小型 UAV 平均 RCS 的近似值。基站/机载噪声系数为 8/9 dB，剩余损耗均为 6 dB。
- 准确功率报告和带噪角度用于处理端协方差。定位为固定协方差权重 WLS，特征归一化计入 GPS 噪声及方向二阶误差。
- 默认连续 5 个时间片达到攻击阈值才确认告警，当前 20 ms 时间片对应 100 ms，配置采用 `navigation.confirmation_steps=5`。当前暂时关闭导航缓解，`navigation.enabled=true`、`navigation.mitigation_enabled=false`，保留 GPS 反馈运动、ISAC 测量和检测告警，告警不改变导航输入。

参考部署保留基站水平坐标，站高统一为 0 m。代码对违反已知上半空间或处于奇异角域的观测报错，不读取真值决定接收端角度分支。

## 配置与数据

新配置使用 configs/default.json 或 configs/multitrajectory_crb.json。旧有效回波系数配置被明确标为历史模型，不把原数值 6 自动当作物理 RCS。

新数据使用 schema 3.0，记录片初观测和片末可用时刻。旧数据和旧权重不会自动变成新模型结果。审计数组含导航模式、导航输入、功率报告、角度和两端协方差，它们不加入检测输入。

采用新高度时，需要在新的空输出目录重新生成数据，例如在原目录名后增加 `_hplus10`。历史数据保留其原配置与几何。

正常重放源保存在审计数组 `replay_source_gps_m`，元数据 `replay_recording` 记录来源、噪声种子和源索引。无独立源时数组为零，以 `present=false` 区分。每个源仅供本数据集中的一个受攻击 episode 使用，生成器核对来源不跨训练、验证和测试复用。重放配置中沿用的 `delay_steps` 在该模式下表示正常记录的轨迹相位偏移。旧的实时正常 GPS 历史延迟模式保留为 `delayed_clean_stream`，缺少 `replay_source` 的历史配置按旧模式加载。命令行 `--replay-source` 可覆盖配置，多进程使用同一模式。

离线采集默认运行共同 GPS 反馈运动，报警不回灌导航。在线运行通过 detector 特征回调或 apply_detection_probabilities 接收每机概率。回调不获得真值或攻击标签，当前默认只检测和记录告警。显式将 `navigation.mitigation_enabled` 设为 `true` 后，确认报警才会在下一片切换并锁定 ISAC 导航，使用上一片定位加已知指令位移。开启缓解时误报也会触发切换。

## 运行入口

在本目录使用 PowerShell 执行。

~~~powershell
python -m pip install -e ".[learning,test]"
python -m isac_spoofing.cli.generate_dataset --config configs/default.json --replay-source recorded_normal_episode --episodes 8 --episodes-per-shard 1 --output data/crb_model --seed 20261029
python experiment.py --data_dir data/crb_model --out_dir runs/gru_crb --model gru --feature_set inter_uav_consistency --fixed_k 5 --k_values 5 --time_step_s 0.02

~~~

未安装项目时先设置当前进程的模块路径。

~~~powershell
$env:PYTHONPATH="$PWD/src"
~~~

`scripts/evaluate_navigation_mitigation.py` 保留为可选缓解评估入口，需要评估时先在专用配置中显式开启 `navigation.mitigation_enabled`。在线评价检查权重的源物理模型，旧权重或不匹配配置需要显式允许域变化，输出标为未验证。

## 实现与验证

见 [模型迁移说明](docs/model_implementation.md)。

当前仍以回波可分离、正确关联、准确功率报告和取 CRB 为协方差作为参考条件。没有模拟实际估计算法或完成硬件校准，本次代码修改也不产生新的论文性能结论。
