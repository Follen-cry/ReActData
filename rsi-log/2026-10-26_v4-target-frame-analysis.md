# v4 分析：让 agent 选择目标帧，训练池限制为 256 个 clip（2026-10-26）

- **背景**：v0 和 v1 的结果显示，在现有设置下，过程视频没有带来大于约 0.05 的帮助（见 [r1.1 审查](2026-10-26_reactdata-v1_r1.1_midrun-review.md) 和 [协议 v3](2026-10-26_protocol-v3.md)）。原因有三：
  1. 视频起作用的方式被排除了：额外帧只能作为 ViT 输入，目标恒为末帧；
  2. 数据不稀缺：1,600 个 clip 对应 1,600 行，每个 clip 正好用一次；
  3. 跨任务：训练用 scale，测试用 move。
- **研究者的决定**：保留 v3 的 agent 流程，允许 agent 为每一行选择目标帧，N = 256。下面的 5 项细节按 Research Agent 的推荐执行，实现为 `exp/reactdata-v4`。

## 1. 让 agent 选择目标帧需要的修改

| 层 | 文件 | 改动 |
|---|---|---|
| 契约 | `api/recipe.py` | recipe 增加可选字段 `target_frame`，取值 1–59，默认是末帧 59；额外帧必须早于目标帧；每步至少 8 行的目标是末帧 |
| 契约 | `api/text_slots.py` | 新增 `{target_spec}` 槽位（写明目标时刻）。目标不是末帧的行，模板必须包含 `{target_spec}`，并且不能使用 `{wrapped_description}`。原因：评估包装的原文是 “completed final state only … no intermediate steps”，与中间帧标签直接矛盾 |
| 契约 | `api/plugin_api.py` | 每行训练信号增加 `target_frame`；loss probe 可以指定目标帧 |
| Runner | `materialize.py` | 中间帧目标使用校正后的 mp4 帧，末帧目标仍使用 PNG 端点 |
| Runner | `train_entry.py`、`checks.py`、`report.py` | 记录和校验新字段；报告中展示目标帧的分布 |
| Runner | `state_machine.py`、`protocol.yaml` | v4 协议字段：clip 上限、末帧配额、使用哪份 agent 指令 |
| 指令 | `BUILDER_AGENT_v4.md` | 说明目标帧的语义、文本规则和视频时间轴（scale 视频：第 0–8 帧等于 F0；约第 10–28 帧 E 出现并换色；约第 28–48 帧 F 出现并缩放；第 49–59 帧为终态） |
| 测试 | `tests/` | 回归候选：非末帧目标使用评估包装文本、额外帧晚于目标帧、末帧行不足，三种情况都必须被拒 |
| 不变 | | 评分器、dev/test、推理配置、选择统计、沙箱 |

**技术问题：mp4 颜色偏移**
- mp4 帧相对 PNG 有约 4/1/3（BGR）的系统性颜色偏移。中间帧一旦作为目标，模型可能学到与时刻相关的色偏。
- 处理：同一 clip 的 mp4 第 59 帧和 PNG 末帧是同一画面，用两者算出该 clip 的逐通道偏移，统一校正这个 clip 的所有 mp4 帧。

## 2. N = 256 需要的修改

- **数据**：从 1,600 个训练 clip 中按 (scale_from, scale_to) 的 12 种组合分层抽取 256 个。这一步由 Runner 读 metadata 完成，agent 看不到 metadata。dev、test、内层 probe 都不变。
- **baseline**：仍然训练 1,600 行，每个 clip 平均用约 6.25 次。P1 要在 N = 256 下重新估计训练噪声 sd_train。
- **对视频的意义**：每个 clip 最多能提供 59 个不同的目标帧，正好弥补 clip 少、只能反复使用端点的问题，这就是要检验的情形。

## 3. 风险

1. 中间帧目标占比太高，可能让模型在测试时画出不完整的状态。由“末帧配额”和“禁止使用评估包装文本”两条控制。
2. 不同目标帧的 loss 大小本身不同，反馈程序需要按目标帧分组归一化。
3. 确认阶段只能可靠检出约 0.05 以上的效果，N = 256 时噪声可能更大。
4. 原因 3（跨任务）没有解决：仍是 scale 训练、move 测试。

## 4. 决定

| 事项 | 决定（按推荐） |
|---|---|
| 额外帧与目标帧的先后关系 | 只允许额外帧早于目标帧（向前预测） |
| 非末帧目标的配额 | 每步至少 8/16 行的目标是末帧 |
| mp4 颜色校正 | 逐 clip 校正 |
| 训练池 | 仍用 scale |
| 256 个 clip 的抽取方式 | 按 12 种 (scale_from, scale_to) 组合分层抽取 |
| run 形式 | 全新 run，不接续以前的 run，新开 agent 会话；10 轮 × 1 个候选；墙钟 ≤ 8 h |
