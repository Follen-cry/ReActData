# rsi-log

reactdata / RSI 系列实验的审查记录与修改建议，按日期归档。每份记录写明审查对象（实验目录与 run id）、数据快照时间、发现、建议，以及待研究者决定的事项。

| 日期 | 文件 | 对象 | 要点 |
|---|---|---|---|
| 2026-10-27 | [2026-10-27_reactdata-v5_r1_results.md](2026-10-27_reactdata-v5_r1_results.md) | reactdata-v5 / r1 结果 | 16 clip、对比 baseline 重复训练的均值、agent 读取完整逐项结果；6.03 h；"F 缩放中的中间帧作为目标"（c006）在 5 个 seed 中 4 个更好，确认 D +0.030，但 CI [−0.019, +0.078]，选中 baseline；test baseline 均值 .369；check_work 在 NFS 上的崩溃已修复 |
| 2026-10-27 | [2026-10-27_reactdata-v4_r1_results.md](2026-10-27_reactdata-v4_r1_results.md) | reactdata-v4 / r1 结果 | 5.18 h；确认阶段前 3 名 D 为 +0.006 至 +0.022，CI 都覆盖 0，选中 baseline；筛选阶段的提升主要来自偏低的比较基准；视频的 3 次尝试都更差或持平；256 个 clip 并不稀缺 |
| 2026-10-26 | [2026-10-26_v4-target-frame-analysis.md](2026-10-26_v4-target-frame-analysis.md) | reactdata-v4 设计分析 | agent 可以选择目标帧（只允许向前预测、每步至少 8/16 行为末帧、非末帧行禁用评估包装文本），N = 256 个 clip 分层抽取，mp4 逐 clip 颜色校正；风险与决定 |
| 2026-10-26 | [2026-10-26_protocol-v3.md](2026-10-26_protocol-v3.md) | reactdata-v1 协议 v3（已实现，未运行） | 按审查建议实现：baseline 5-seed 噪声估计、筛选 → 确认（4 个配对 seed）→ 选择、10 × 1 串行、谱系起点、只改配置的候选视为重复、鼓励探索的 agent 指令；列出扩大动作空间、限制 clip 数量等待决定的方向 |
| 2026-10-26 | [2026-10-26_reactdata-v1_r1.1_midrun-review.md](2026-10-26_reactdata-v1_r1.1_midrun-review.md) | reactdata-v1 / r1.1（第 7 轮快照；之后已停止） | baseline 换数据顺序的重复实验之间相差约 0.05，δ 和 CI 没有计入训练噪声；agent 改用候选名额做重复实验；父程序选择在追逐噪声；建议先估计训练噪声、多 seed 确认、禁止只改 salt 的候选、改为串行轮次 |

相关实验目录：`../reactdata-v0/`（2d_geometric_transformation，r1 已完成）、`../reactdata-v1/`（scale 训练 / scale+move dev / move test，r1 已完成，r1.1 于第 7 轮后停止）、`../rsi0/`（PhysicTran 上的前身实验）。
