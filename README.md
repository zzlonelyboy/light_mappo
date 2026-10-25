# ISAC 辅助的无人机 GPS 欺骗检测

本目录是一个独立的数据生成、训练和评估项目。无人机运动由预定义轨迹驱动；检测器只接收 GPS、基站 ISAC、机间 ISAC 及可信运动信息，不接收真实位置。

## 数据生成入口

常规 D1 数据只使用 `scripts/generate_d1_multitrajectory.py`。参数 `--dataset-type` 选择数据用途：

- `training`：按 episode 划分为 `train/validation/test`（80%/10%/10%）。
- `test`：所有 episode 均写入 `test`，用于纯测试或跨 RCS 测试。

生成训练、验证和测试数据：

```powershell
python scripts/generate_d1_multitrajectory.py `
  --dataset-type training `
  --episodes 2000 `
  --trajectory-types straight circle helix s_curve `
  --rcs 6 `
  --seed 20260802 `
  --output-dir dataNew
```

生成一组纯测试数据：

```powershell
python scripts/generate_d1_multitrajectory.py `
  --dataset-type test `
  --episodes 1000 `
  --trajectory-types straight circle helix s_curve `
  --rcs-list 3 6 9 12 15 18 21 24 `
  --seed 20260802 `
  --output-dir dataNew
```

从已经生成的 K=3 原始测试集快速派生不同邻域大小的轻量测试视图：

```powershell
python scripts/rebuild_inter_uav_neighbor_views.py `
  --source-dirs `
    dataNew/d1_1000_multitrajectory_test_rcs3_seed20260802_interuav `
    dataNew/d1_1000_multitrajectory_test_rcs6_seed20260802_interuav `
    dataNew/d1_1000_multitrajectory_test_rcs12_seed20260802_interuav `
  --neighbor-sizes 2 3 4 5 6 7 `
  --workers 4
```

脚本只根据源 shard 中的 GPS 与完整机间 ISAC 测距矩阵重算最后两个
min/max 邻域特征。K=3 直接复用源数据集，其他 K 输出为兼容现有评估器的
轻量 shard，不重新生成轨迹、攻击或测量噪声。

长时 1h 数据单独保留 `scripts/generate_d5_long_multiattack.py`：

```powershell
python scripts/generate_d5_long_multiattack.py `
  --attacked-episodes 16 `
  --clean-episodes 4 `
  --duration-s 3600 `
  --rcs 6 `
  --neighbor-feature-mode inter_uav `
  --num-neighbors 3 `
  --output dataNew/d5_1h_multiattack
```

运行 `python scripts/generate_d1_multitrajectory.py --help` 或 `python scripts/generate_d5_long_multiattack.py --help` 可查看完整参数。

## 机间 ISAC 特征

默认 `neighbor_feature_mode=inter_uav`。每个时刻的处理顺序为：

1. 根据无人机真实位置计算仿真中的真实机间距离。
2. 使用当前配置的载频、带宽、发射功率和 RCS 计算 SNR 与测距方差。
3. 在真实距离上加入独立测距噪声，得到完整的有向机间测量矩阵。
4. 动态模式按测得的机间距离选择 K 个邻居，不使用真实距离选择邻居。
5. 将机间测距与 GPS 位置推导的距离比较，输出 K 个邻居中绝对差异的最小值和最大值。

因此，默认检测特征为 10 维：6 个基站 ISAC/GPS 残差、2 个自身定位残差和 2 个机间一致性特征。NPZ 同时保存以下原始机间测量，便于审计：

- `inter_uav_range_m`
- `inter_uav_valid_mask`
- `inter_uav_snr_db`
- `inter_uav_range_variance`

同一次生成任务中，基站 ISAC 与机间 ISAC 使用相同的配置 RCS；跨 RCS 数据不再经过离线增强，也不再固定使用 RCS=6。

## 安装与测试

```powershell
python -m pip install -e .
python -m unittest discover -s tests -v
```

历史实验设计、结果与论文草稿保存在 `docs/`，新数据生成方式以本页和两个生成脚本的 `--help` 为准。
