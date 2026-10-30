# GPS 参数更新与正式数据生成

正常 GPS 单轴标准差更新为 `(1.5,1.5,3.0)` m，活动配置为 `configs/default.json` 和 `configs/multitrajectory_crb.json`。水平尺度参考普通 GNSS 模块的公开精度量级，垂直尺度作为设计假设，不将该配置称为设备实测值或标准强制值。

## 数据规模与来源

原主集为旧项目 `dataCubic/d1_5000_scurve_cubic_rcs6_seed20260714_interuav`。本轮保留以下实验规模。

| 项目 | 配置 |
|---|---|
| 完整机群 episode | 5000 |
| train / validation / test | 4000 / 500 / 500，按完整 episode 划分 |
| 场景 | clean、random、replay、stealth 各 1250 |
| 轨迹 | s_curve |
| 基础种子 | 20260714 |
| 单 episode | 60 s、8 UAV、20 ms，共 3000 步 |
| 检测输入 | 每 UAV 每片 10 维 |
| 采集方式 | GPS 反馈闭环，离线采集不接入检测报警 |

新输出目录为 `dataCubic/d1_5000_scurve_cubic_crb_gps15_30_seed20260714`，不覆盖旧数据。

## 存储与资源

采用 `--audit-level standard`，仅省略以下三个可选协方差审计矩阵。

- `isac_measurement_covariance`
- `isac_reported_covariance`
- `isac_reported_los_covariance`

测量生成和特征构造仍使用完整协方差。原始基础数组、导航状态、GPS、测量角度、功率报告及最终特征全部保留。处理端协方差可以使用保存的功率报告、带噪角度及配置重新计算。本训练资产不保留逐样本完整生成协方差。

每片保存 2 个 episode，多进程最多缓存 `2 × workers` 个在途任务。该存储方式预计约需 59 GiB，实际大小以完成后的验证报告为准。分片大小属于存储设置，不改变 episode 划分、种子和场景采样。

## 执行与状态

在项目目录运行以下命令。

~~~powershell
python scripts/run_gps_reference_dataset.py --output dataCubic/d1_5000_scurve_cubic_crb_gps15_30_seed20260714 --run-directory tmp/gps_dataset_20261030 --workers 4
~~~

监督入口保存配置快照，启动生成进程，并在生成结束后自动检查所有分片的 episode 唯一性、数据划分、特征形状与有限性、GPS 经验标准差及片末可用时刻。

| 文件 | 用途 |
|---|---|
| `tmp/gps_dataset_20261030/status.json` | 整体运行状态，区分 generating、verifying、completed 和 failed |
| 新数据目录的 `progress.json` | 已生成与已落盘 episode 数 |
| 新数据目录的 `manifest.json` | 生成全部结束后写入的数据清单 |
| `tmp/gps_dataset_20261030/verification.json` | 全部分片核查完成后写入的验证结果 |
| `tmp/gps_dataset_20261030/stdout.log`、`stderr.log` | 生成日志与错误输出 |

只有监督状态为 `completed` 才表示生成与自动核查均完成。生成入口拒绝覆盖非空数据目录。

## 代码验证

GPS 参数变更后，60 项相关回归测试通过。新增存储测试确认 standard 与 full 的保留数组和特征逐值一致，并验证串并行生成的确定性、进度统计与覆盖保护。这些检查不构成正式检测性能结果。
