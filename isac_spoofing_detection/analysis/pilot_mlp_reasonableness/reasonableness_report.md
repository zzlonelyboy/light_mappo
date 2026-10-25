## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-07-17
- Verification Status: VERIFIED
- Version Label: validation_v1

## Validation Report

- **Source**: `runs/pilot_mlp_seed20260716`
- **Overall Confidence**: **CAUTION**
- **Scope**: D1同生成器训练/验证合理性；未计算测试集性能

### 总体判断

当前97%左右的验证事件F1不是一次随机训练偶然产生的，也没有发现直接标签泄漏、跨split重复或事件匹配放宽。结果在当前数值生成分布内是真实、稳定且可复现的。

但是，当前任务本身高度可分：仅6维瞬时ISAC--GPS残差的线性模型已经达到0.923的step级F1，略高于F15线性模型和当前MLP。因而该结果主要证明当前名义仿真数据容易检测，尚不能证明时序模型、CUSUM、邻机特征或真实环境鲁棒性。

### Statistical Findings

| 检查项 | 方法 | 结果 | 判断 |
|---|---|---:|---|
| 同种子复现 | 完整重跑10轮 | 历史、指标、模型SHA-256完全一致 | SOLID |
| 跨split重复 | episode ID和特征SHA-256 | 三组交集均为0 | SOLID |
| 标签置乱负对照 | 置乱训练标签后线性F15 | F1=0.0953 | SOLID |
| 时序对齐负对照 | 打乱验证标签与固定MLP预测的对齐 | F1=0.1060 | SOLID |
| 三随机种子稳定性 | 20260716/17/18 | step F1=0.9171±0.0016；event F1=0.9752±0.0019 | SOLID（同分布） |
| 事件F1不确定性 | 2000次episode聚类Bootstrap | 0.9744，95% CI [0.9624, 0.9844] | SOLID（条件于验证集K选择） |
| 单特征可分性 | `mean_direction` | AUC=0.9749 | CAUTION：任务偏易 |
| 简单模型 | 单个`max_range`决策树桩 | F1=0.8671 | CAUTION：强单特征信号 |
| 线性F15 | 35万训练step、15万验证step | F1=0.9176 | CAUTION：MLP增益未建立 |
| 线性瞬时残差6维 | 相同抽样 | F1=0.9230 | RED_FLAG：时间特征必要性未建立 |
| 完整事件 | 717个UAV事件 | Recall=1.000 | CAUTION：完整长事件过易 |
| 截断事件 | 255个UAV事件 | Recall=0.9137 | CAUTION |
| 1--4 step事件 | 18个UAV事件，K=5 | 0个检出 | 预期的结构性不可检出 |

注：线性基线和特征消融使用固定随机抽样，作用是合理性诊断，不替代正式的全数据、多种子消融实验。

### 关键证据

#### 1. 未发现直接泄漏

- 训练、验证、测试episode ID无交集。
- 三个split的episode特征哈希无交集。
- 标准化参数只由训练集拟合。
- 特征提取接口不读取标签，且只使用当前与历史信息。
- 事件匹配要求报警触发点严格落入攻击区间。
- 负对照破坏特征--标签关系后，F1降至约0.10。

这些结果支持“不是标签泄漏造成97%”的判断，但不排除同一仿真生成器带来的分布规则泄漏。

#### 2. 当前数据高度可分

| 特征集合/模型 | step级F1 |
|---|---:|
| 当前MLP-F15 | 0.9176 |
| 线性F15 | 0.9176 |
| 线性瞬时残差6维 | 0.9230 |
| 线性残差9维（含CUSUM） | 0.9240 |
| 线性去CUSUM 12维 | 0.9151 |
| 仅CUSUM 3维 | 0.8606 |
| 自身定位差异2维 | 0.5272 |
| 邻机特征4维 | 0.3497 |

当前性能主要由瞬时距离、方向和径向速度残差贡献。现有结果没有证明CUSUM、邻机特征或MLP非线性结构是必要的。

#### 3. 事件级指标受到事件长度和K规则放大

- 平均攻击长度约101.5 step，而K=5只要求连续5个正判决。
- 完整攻击事件的Recall为100%。
- 18个长度小于K的事件全部漏检，符合算法定义。
- 早期/中期攻击Recall均为100%，晚期攻击因截断降至93.0%。

因此，97.7%的事件Recall同时反映模型能力、长攻击带来的多次检出机会以及K规则，不应解释为每个step都有97.7%的识别能力。

#### 4. 分组结果并非完全一致

| 分组 | Event F1 | Median delay |
|---|---:|---:|
| Random | 0.9801 | 0.4 s |
| Replay | 0.9862 | 0.4 s |
| Stealth | 0.9546 | 1.3 s |
| Weak | 0.9672 | 0.7 s |
| Medium | 0.9878 | 0.4 s |
| Strong | 0.9681 | 0.4 s |

Strong没有单调优于Medium，原因之一是较强攻击更容易留下历史特征尾迹并产生后续报警。28次误报中，21次发生在攻击结束后，6次为攻击期间的重复报警，只有1次发生在未受攻击无人机上。

### Warnings

| Type | Detail | Affected |
|---|---|---|
| 同分布验证 | 训练和验证来自相同轨迹、传感器与攻击生成器 | 所有泛化表述 |
| 理想安全GPS | clean GPS噪声为0 | 单特征可分性、误报率 |
| 高SNR归一化 | 数据中存在最高661.7个训练标准差的有限极端特征 | MLP/线性分类边界 |
| 模型必要性 | 线性瞬时残差模型不弱于MLP-F15 | 创新性与模型选择 |
| K选择偏乐观 | K在同一验证集上选择并报告 | 验证Event F1 |
| 事件聚类 | 972个UAV事件来自225个攻击episode，不是972个独立episode | 不确定性估计 |
| 不可检事件 | 长度小于K的事件在定义上无法检出 | 截断/晚起攻击 |
| 误报尾迹 | 75%的误报发生在攻击结束后 | CUSUM及恢复机制 |
| 基率差异 | 数据集四类episode等比例，现实攻击基率通常不同 | Precision与报警负担 |
| 依赖环境 | SciPy 1.9.1对NumPy 1.26.4发出版本兼容警告 | 抽样线性基线 |

### Fallacy Scan

- **Coverage**: 11/11 fallacy types checked

| Fallacy | Severity | Detail | Recommendation |
|---|---|---|---|
| Simpson's Paradox | CAUTION | 汇总Recall=0.977，但完整事件=1.000、截断事件=0.914、1--4 step事件=0 | 保留分层结果 |
| Ecological Fallacy | CAUTION | 972个UAV事件嵌套于225个攻击episode | 以episode聚类估计区间 |
| Berkson's Paradox | CAUTION | 样本来自单一受控生成器和预设参数范围 | 将结论限定于该生成总体 |
| Collider Bias | NOTE | 未发现明显控制变量碰撞；但检测延迟只存在于已检事件 | 延迟与检出率同时解释 |
| Base Rate Neglect | CAUTION | 等比例场景不能代表实际攻击基率 | 同时报每UAV小时误报 |
| Regression to the Mean | NOTE | 当前设计不是按极端性能选组的前后比较 | 非主要风险 |
| Survivorship Bias | CAUTION | 只看成功检出的延迟会排除22个漏检和18个结构性不可检事件 | 同时报漏检和不可检事件 |
| Look-Elsewhere Effect | CAUTION | 搜索4个K并尝试多种特征/模型 | 测试集只评价冻结方案 |
| Garden of Forking Paths | CAUTION | K、特征集、模型、攻击区间和数据处理存在多种选择 | 固定正式实验矩阵 |
| Correlation != Causation | CAUTION | 仿真中的检测相关性不证明真实飞行防护效果 | 结论限于数值仿真 |
| Reverse Causality | NOTE | 攻击先于观测异常，由生成过程保证时间顺序 | 未发现主要风险 |

### Reproducibility

- **Method**: 同种子、同数据、同代码、同GPU环境完整重跑
- **Verdict**: **REPRODUCIBLE**

| Metric | Original | Re-run | Diff | Status |
|---|---:|---:|---:|---|
| Best epoch | 10 | 10 | 0 | MATCH |
| Validation step F1 | 0.9175789520 | 0.9175789520 | 0 | MATCH |
| Selected K | 5 | 5 | 0 | MATCH |
| Event F1 | 0.9743589744 | 0.9743589744 | 0 | MATCH |
| Event Recall | 0.9773662551 | 0.9773662551 | 0 | MATCH |
| False alarms/UAV-hour | 1.2543647663 | 1.2543647663 | 0 | MATCH |
| Checkpoint SHA-256 | `F6A679...CC2A` | `F6A679...CC2A` | identical | MATCH |

### 合理性结论

1. **实现与训练过程合理**：数据划分、标准化、标签对齐、事件匹配和复现性检查通过。
2. **当前名义数据可用于证明检测链路能够工作**：不是随机猜测，也不是明显泄漏。
3. **当前97%结果过于依赖容易分离的同生成器数据**：不能据此证明真实环境鲁棒性。
4. **MLP、CUSUM和邻机特征的必要性尚未建立**：简单瞬时残差线性模型已经达到同等表现。
5. **当前证据等级为CAUTION**：适合作为D1名义基线，尚不足以作为论文最终核心结论。

正式结论所缺少的证据是：带正常GPS误差和ISAC模型失配的独立鲁棒性集、全数据多种子基线/消融，以及冻结方案后的一次性测试集评价。
