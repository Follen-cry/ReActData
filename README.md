# ReActData

由 LLM agent 迭代改写训练数据程序的 RSI 实验 pipeline。本仓库收录最新一轮 **reactdata-v5 / run r1**（2026-10-27）的完整 pipeline 代码、每一轮的结果，以及每一轮的文件修改记录。

**流程**：
- 固定的 Python Runner 在 bwrap 沙箱中启动 Builder Agent（`claude -p --model sonnet`）。
- agent 读完上一轮的全部结果后写出候选程序：`databuilder.py`、`analysis.py`、`text_templates.yaml`、`proposal.md`。
- Runner 用这个候选训练 InternVL-U（100 step，global batch 16），在 dev 上打分。
- 按冻结规则筛选、确认、选择，最后对选中的程序做正式训练和 test。

**研究问题**：只有 16 个 `shape_color_then_scale` 训练视频时，视频的中间过程帧能否让 F0 → final 的生成更好，包括训练池中没有的 move 任务。

## 目录

| 路径 | 内容 |
|---|---|
| [`ROUNDS.md`](ROUNDS.md) | **每一轮一览**：候选、父程序、dev 分数、D、确认结果、各文件改动行数、假设、`analysis.py` 测什么，以及到原始记录的链接 |
| [`pipeline/`](pipeline/) | v5 pipeline 代码：协议 `protocol.yaml`、agent 指令 `BUILDER_AGENT_v5.md`、`runner/`、固定契约 `api/`、打分器 `scorers/`、baseline 程序 `seeds/`、数据清单 `manifests/`、`tests/`。详细说明见 [`pipeline/README.md`](pipeline/README.md) |
| [`runs/r1/`](runs/r1/) | r1 的记录（小文件；checkpoint 和帧图片留在服务器 scratch 上） |
| `runs/r1/report.md` | Runner 生成的最终报告：噪声、筛选、确认、选择、test、成本 |
| `runs/r1/rounds/<nnn>/<cid>/` | 每一轮：`input/report.md` 与 `input/results.json`（agent 当时看到的结果）、`agent_*/`（prompt、命令、输出）、`work/`（agent 交付的文件）、`patch.diff`（相对父程序的修改）、`checks.json` |
| `runs/r1/candidates/<cid>/code/` | 每个候选最终被训练的程序 |
| `runs/r1/jobs/<job>/` | 每个训练 / 评测任务的配置、哈希、首步检查、成本、逐条 dev 分数 |
| `runs/r1/{noise,confirm,selection,summary,state}.json` | 训练噪声、确认统计、选择结果、汇总和完整状态 |
| [`rsi-log/`](rsi-log/) | 全系列（v0、v1、v4、v5）的审查记录、协议分析和结果解读 |

## r1 结果摘要

- **训练噪声**：固定 baseline 用 5 个 seed 训练，dev 为 .391 / .355 / .396 / .420 / .371，sd = δ = .025；Base（未训练）为 .005。
- **筛选**（共 9 轮，每轮 1 个候选）：
  - E 已变色、F 尚未出现的帧，作为目标或上下文都有害（c001–c003，−.04 至 −.09）；
  - F 缩放中的帧作为目标有帮助：c004 → c005 → c006 呈剂量–响应，最高 +.028。
- **确认**（前 3 名 × 4 个配对 seed，Bonferroni 校正）：
  - c006 D +.030，CI [−.019, +.078]，5 个 seed 中 4 个更好，move 上 +.020；
  - c009 D +.025；
  - c008 D −.026。
  - 没有候选通过，选中固定 baseline。
- **test**：baseline 在 3 个正式 seed 上为 .357 / .354 / .395。
- **成本**：GPU 28.5 h，Sonnet $7.93，墙钟 6.03 h。

详见 [`ROUNDS.md`](ROUNDS.md) 和 [`rsi-log/2026-10-27_reactdata-v5_r1_results.md`](rsi-log/2026-10-27_reactdata-v5_r1_results.md)。

## 说明

- **运行中的修复**：r1 在 09:23 因 `check_work` 目录在 NFS 上没能清理而崩溃（`FileExistsError`）。修复后于 09:25 续跑，修复已包含在 `pipeline/runner/checks.py` 中。`report.md` 的 Failures 一节没有记录这次崩溃。
- **运行依赖**：pipeline 依赖不在本仓库中的训练代码 `internvlu-exp/src/internvlu-train`（N0 配置），以及服务器上的数据 `/scratch/network/ssd/junlin/reactdata-v5/data`。因此本仓库是实验记录，不能单独复现运行。
- **保持私有**：仓库包含打分器源码和 test 清单。Builder Agent 的沙箱可以联网，如果仓库公开，后续 run 的隔离就会失效。
