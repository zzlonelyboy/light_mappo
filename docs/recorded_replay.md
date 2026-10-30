# 独立正常记录重放

## 当前行为

默认 `attack.replay_source` 为 `recorded_normal_episode`。每个重放 episode 按以下顺序生成。

1. 生成本次任务的参考轨迹和初始队形。
2. 使用独立 GPS 噪声流，先完成一次无攻击、无导航缓解的正常飞行录制。录制源的真实运动同样由正常 GPS 位置反馈产生，并非简单地将参考轨迹加噪。
3. 开始受攻击飞行。攻击窗口内，目标 UAV 的 GPS 报告取自已完成正常记录的对应 UAV 和偏移后索引。源 GPS 已含正常噪声，复制时不再次加噪。
4. 未受攻击 UAV 和攻击窗口外继续使用自身正常 GPS。所有实际运动、ISAC 观测和检测特征均来自本次受攻击飞行。

沿用的配置名 `replay_delay_ranges_s` 和事件字段 `delay_steps` 在新模式下表示源轨迹相位偏移，不表示实时链路延迟。事件元数据另外写入 `phase_offset_steps` 和 `offset_semantics`。源索引必须有效，不循环或截断取样。

保留 `delayed_clean_stream` 作为旧模式。该模式读取受攻击飞行自身的历史正常 GPS 观测。旧保存 JSON 若缺少 `replay_source` 字段，加载时保持旧模式。当前活动配置显式选择新模式，命令行可覆盖两种模式。

## 来源与数据划分

每个使用新模式的重放 episode 有一份独立正常记录，不额外计入 5000 个受攻击/正常评价 episode。生成器保证数据集内部不复用同一来源，并将其归属本 episode 所在的训练、验证或测试划分。

仅供审计的 `replay_source_gps_m` 保存完整源 GPS。没有独立源的 episode 填入同形零数组，必须同时检查 `metadata.replay_recording.present`。元数据还保存源 ID、录制噪声种子、受攻击 episode 种子、参考轨迹类型、UAV 索引映射和实际源索引范围。复现录制源需要保存的参考轨迹、初态、配置及录制噪声种子，录制种子单独不能重建完整任务。

该数组不进入检测输入，检测特征仍为 10 维。源仅记录 GPS，不用源真值替换本次受攻击飞行真值。导航缓解仍默认关闭。

## 正式生成命令

以下 PowerShell 命令生成 5000 个完整机群 episode，按 4000 / 500 / 500 划分，四类场景各 1250。正常重放源自动生成，不需要先单独生成正常数据集。输出目录必须为空或不存在。

```powershell
$projectRoot = 'D:\Codes\project_spoofing_detect\isac_spoofing_detection_closed_loop'
Set-Location -LiteralPath $projectRoot
$env:PYTHONPATH = "$projectRoot\src"
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'

python -m isac_spoofing.cli.generate_dataset `
  --config "$projectRoot\configs\default.json" `
  --replay-source recorded_normal_episode `
  --output "$projectRoot\dataCubic\d1_5000_rcs005_gps15_30_recorded_replay_seed20260714" `
  --episodes 5000 `
  --seed 20260714 `
  --episodes-per-shard 2 `
  --workers 2 `
  --audit-level standard
```

`standard` 只省略原有三个完整协方差审计矩阵，保留正常源 GPS。`progress.json` 表示生成进度，全部完成后写入 `manifest.json`。如需保存配置快照并在结束后自动核查全部分片，可使用 `scripts/run_gps_reference_dataset.py`，该监督入口同样支持 `--replay-source`。

## 本次核查

重放专项检查覆盖正常反馈录制、噪声只复制一次、批量与逐片一致、无效源拒绝、旧配置迁移、双进程命令行覆盖、来源唯一以及审计数组与网络输入隔离。

单进程和双进程各生成 40 个短 episode，两者所有保留数组及 episode 元数据逐值一致，训练 / 验证 / 测试为 32 / 4 / 4，10 份独立重放源不跨划分复用。另检查一次默认配置下的完整 60 s 重放飞行，3000 步的目标 GPS 与源记录对应区间逐值相同，特征形状为 `(3000,8,10)` 且均为有限值。

本次未启动 5000 个 episode 的正式生成，也未训练检测器。回归范围不包含此前已知的圆轨迹误差阈值和软延迟损失两项问题，这两项与本次重放源实现无关。
