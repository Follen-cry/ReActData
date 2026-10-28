# reactdata-v5

调度按 **ExperimentDesign → 共享阶段操作 → TaskAdapter** 分层，reactdata 和 synthetic task 复用同一套 baseline/confirmation/selection/final 实现。旧 CLI 保留；Agent 回执、授权证据流、训练组件、扩展方式和验证结果见 [MODULARITY.md](MODULARITY.md)。冻结协议和实验定义保持不变；架构入口和依赖方向见该文档的“目录导航与依赖方向”。

**一句话**：一个由 LLM agent（Builder Agent，`claude -p --model sonnet`）提出"训练数据程序"、由固定 Runner 负责训练 / 评测 / 统计选择的自动化实验。
研究问题是：**只有 16 个训练视频时，利用视频的中间过程帧（作为训练目标或 ViT 上下文），能否让 InternVL-U 的单帧 F0 → final 生成变得更好——包括训练池中没有的 move 任务？**

前身是 reactdata-v4（代码从 v4 拷贝，v4 已冻结）。v5 相对 v4 的改动写在 `protocol.yaml` 开头的注释里：

- screening 的 D 改为对比 **5 个 baseline replicate 的逐 item 均值**（v4 只对比 seed 1000 的单次 baseline，它恰好偏低，抬高了所有 D）；
- 训练池缩到 **16 个 clip**（数据稀缺：只训 final 帧时每个 clip 约用 100 次）；
- 训练池仍只有 `shape_color_then_scale`；
- agent 在提议前必须读完全部结果（`results.json`、逐 item dev 结果、自己的 analysis traces），`proposal.md` 必须有固定的 5 个章节并引用 ≥ 3 个 dev item（Runner 机械检查）。

---

## 1. 任务设定

场景是 2×3 的类比网格：上排 A → B → C 是示例，下排 D → ? → ?。

| 数据 | 内容 | 谁能看到 |
|---|---|---|
| 训练池 | 16 个 `shape_color_then_scale` 视频（60 帧，16 fps；12 种尺寸变化分层抽样，seed 20261028） | agent 只能看到像素统计（`pool_view.json`），看不到生成器 metadata |
| inner probe | 32 个 scale clip，从训练侧预留、从不训练；`analysis.py` 可以在训练中请求对它们做 loss / generate probe | 只能通过 probe 结果间接访问 |
| dev | 50 scale + 50 move（move 不在训练池中：跨任务反馈）；dev 分 = 两个任务均值的等权平均 | agent 看结果、失败案例图片 |
| test | 100 个 `shape_color_then_move`（reactdata-v1 r1 已用过一次，不是全新集合） | 从不展示；只在 FROZEN 之后的 P3 使用 |

模型输入是 F0（显示 A、B、C、D），输出 final 帧（补出 E = D 按 A→B 改色，F = E 按 B→C 变换：scale 任务是缩放，move 任务是上下移动）。
视频时间线：0–8 帧 = F0；约 10–28 帧 E 出现并变色；约 28–48 帧 F 出现并缩放；49–59 帧 = 终态（59 = 测试要求的 final 帧）。

**打分**（`scorers/analogy_grid.py`，确定性 CV 规则，agent 只能看 `api/scorer_description.md` 的文字说明）：

```
total = gates × (0.35 E_cell + 0.40 F_cell + 0.25 F_transform)
```

A/B/C/D 不给分，只作为乘法 gate（被破坏则降到 0）；空白 / 重绘背景有上限；拷贝输入得 0 分。

---

## 2. 目录结构

```
reactdata-v5/
├── README.md                 本文件
├── protocol.yaml             冻结的实验协议（v5 / design v3）：数据、训练、噪声、搜索、确认、选择、最终测试、预算、GPU slot
├── BUILDER_AGENT_v5.md       发给 Builder Agent 的完整说明（任务、规则、必需工作流、proposal 章节）
├── .gitignore                允许 json/jsonl/yaml 进 git（父目录默认忽略），忽略 __pycache__
├── configs/
│   └── train.yaml            训练配置：继承 src/internvlu-train/configs/v3/N0.yaml，100 step，512 area，LoRA init seed 42
├── api/                      **固定契约**（只读挂载给 agent 和插件沙箱）
├── runner/                   CLI、训练、沙箱及原基础设施（agent 看不到）
├── experiment/               assembly + workflow / improvement / evidence / runtime
├── tasks/reactdata/           reactdata 的任务能力实现
├── scorers/
│   └── analogy_grid.py       规则打分器（Runner-only）
├── seeds/
│   └── fixed_f0_final/       固定 baseline 程序 c000（也是 agent 第一个 /work 的起点）
├── manifests/                冻结的数据清单与数据检查报告
├── records/                  每次 run 的小文件备份（进 git），大文件留在 scratch
│   ├── smoke1/               冒烟测试 run（4-step 训练、4 条 eval，不是结果）
│   └── r1/                   正式 run
└── tests/                    unit/、integration/、support/、fixtures/；旧脚本保留入口
```

大文件（数据、checkpoint、生成图片、agent HOME）不在本目录，而在：

```
/scratch/network/ssd/junlin/reactdata-v5/
├── data/                     runner/build_data.py 的产物
│   ├── BUILD_DONE.json       存在则拒绝重建
│   ├── frames512/            512×512 帧（硬链接到 v4 数据：F0 / final 用 PNG 端点，其他帧用颜色校正后的 mp4 帧）
│   ├── meta/                 生成器 metadata（Runner-only，用于分层抽样和审计）
│   └── pool_view.json        给 databuilder 的训练池像素统计（frame_change / fg_area）
└── runs/<run_id>/            一次 run 的全部状态（结构见 §5）
    └── <run_id>.log          state_machine 的 stdout 日志（在 runs/ 下）
```

路径常量统一定义在 `runner/common.py`（`SCRATCH`, `DATA`, `RUNS`, `FRAMES`, `TRAIN_SRC`, `PY` 等）。

---

## 3. 每个文件的作用

### 3.1 顶层

| 文件 | 作用 |
|---|---|
| `protocol.yaml` | 实验的唯一配置源。关键字段：`data`（16-clip 训练池，每 run 1,600 行）、`targets`（agent 可选 target 帧 1..59，每 step ≥ 8 行 final）、`training`（100 step × 16 行，4 GPU，screen_seed 1000）、`noise.baseline_seeds`（1000–1004）、`evaluation`（S = [0]，分层 bootstrap B = 10,000）、`search`（最多 10 个候选，串行 10 轮 × 1，`screen_reference: replicate_mean`，proposal 章节与 ≥ 3 item 引用，agent CLI / 工具 / 超时，重试、修复、连续 3 次拒绝停止）、`confirm`（top 3 在 seed 1001–1004 上重训，Bonferroni）、`selection`、`final`（formal seed 2000–2002 在 test 上）、`budget`（8 h 墙钟，8 GPU，4 个 4-GPU slot）、`audit`。run 开始时其 sha256 写入 `run.json`。 |
| `BUILDER_AGENT_v5.md` | agent 的 system 说明：任务、target 帧规则、噪声与"只测一个大效应假设"的建议、必需工作流（先读证据 → 提议 → 决定 analysis.py 测什么）、`proposal.md` 的 5 个章节（Evidence / Hypothesis / Program / What analysis.py measures / Expected result）、可编辑文件、硬规则、输入目录说明。 |
| `configs/train.yaml` | InternVL-U 训练超参：继承 N0 recipe；gen 图 512 area；`max_steps 100`；`seed 42`（所有 arm 相同的 LoRA 初始化）；gen loss 权重恒 0.5；lr 由 `train_entry.py` 构建（6 步线性 warmup + cosine）；关闭内置 eval，checkpoint 每 25 步，只留最后一个。 |

### 3.2 `api/` — 固定契约（agent 可读，不可改）

| 文件 | 作用 |
|---|---|
| `plugin_api.py` | 候选程序必须实现的接口：`databuilder.py` 的 `class Builder`（`next_batch` / `observe` / `state_dict` / `load_state_dict`）和 `analysis.py` 的 `class Analysis`（`update` / `request_probes` / …）；每步调用顺序；`pool` / `rows` / `step_info` / `budget` 的格式；probe 请求格式与上限（`PROBE_MAX_ITEMS 32`，loss probe 间隔 ≥ 10 步最多 20 次，generate probe 间隔 ≥ 50 步最多 4 次）；`ALLOWED_IMPORTS` 白名单；每次调用 10 s 限时；确定性要求。 |
| `recipe.py` | 一行训练数据（recipe）的格式与检查：`{clip_id, context_frames, template_id, target_frame}`；F0 永远是第一张图且是唯一的 clean VAE condition；context 帧 ≤ 2、严格递增、早于 target；每步 ≥ 8 行 target = 59；每步 ViT 图片 ≤ 32。`check_recipe` / `check_batch` / `vit_images`。 |
| `text_slots.py` | 训练文本契约：模板只能是静态文本 + 白名单 slot（`{description}`, `{wrapped_description}`, `{context_list}`, `{n_context}`, `{t_target}`, `{target_spec}`, `{duration}`）；静态文本不能含数字 / 尖括号；非 final target 必须用含 `{target_spec}` 且不含 `{wrapped_description}` 的模板。`check_templates` / `render`。 |
| `scorer_description.md` | 打分规则的文字说明（agent 看不到打分代码，只看这个）。 |

### 3.3 `runner/` — Runner（Runner-only）

| 文件 | 作用 |
|---|---|
| `state_machine.py` | **主程序**。固定状态机，没有 LLM 做决策。按阶段 P0 → P1 → P2 → CONFIRM → SELECT → FROZEN → P3 → REPORT → DONE 运行（§4）；每一步幂等，重跑同一命令即可恢复（已完成的 job 跳过，运行中的 job 重新挂上，基础设施失败从最后的 checkpoint 恢复一次）。还负责：调用 agent、检查、去重、训练候选、审计、磁盘管理（`manage_disk`）、预算（`budget_left_h`）、把小文件备份到 `records/<run_id>/`（`backup`）。参数：`--run-id`、`--smoke`、`--protocol`、`--replay-decisions <src>`（复用源 run 的候选代码，不调用 LLM）、`--continue-from <src>`（继承已完成 run 的候选 / baseline / agent session）。 |
| `agent_call.py` | 在 `bwrap` 沙箱里调用 `claude -p --model sonnet`。沙箱只挂载系统文件、claude 可执行文件、一个持久的 Runner-only agent HOME（存 OAuth 凭据副本和 session）、`/work`（可写）和 `/input`（只读）；工具只有 Read/Edit/Write。通过 `--resume <session_id>` 让整个 run 的 agent 是同一个连续会话。 |
| `report.py` | 为每一轮生成 agent 的 `/input`：`report.md`、`results.json`（所有程序的 dev 分、子分、错误类别、与 baseline 的配对比较和 bootstrap CI、训练 loss 摘要、probe 使用、成本）、`dev_items/<cid>.json`（逐 item 分数 + baseline replicate 均值 / sd / 差值）、`dev_items/baseline_replicates.json`、`analysis_traces/<cid>.json`（每 10 步的 analysis.py 摘要 + probe 结果）、`failures/`（失败案例图：F0 / 生成 / GT）、`candidates/`（历史候选代码）、`train_examples/`（3 个训练 clip 的帧）、`api/`。不包含任何 test、metadata 或打分代码。 |
| `checks.py` | 候选程序的 CPU 检查：文件集合与大小、config key、模板合法性、静态扫描（禁止的 import / 调用 / dunder）、在插件沙箱里做 200 步模拟、确定性（两次运行 trace hash 相同）、`state_dict` 存取往返一致、类别测试（判定 fixed / static / schedule / feedback，与 `candidate.yaml` 声明的类别对比）。可单独运行：`python runner/checks.py <candidate_dir> [--out checks.json]`。 |
| `plugin_host.py` | Runner 端的插件沙箱：用 `bwrap` 启动 `plugin_worker.py`（无网络，只挂载 Python 环境、api/、候选代码、pool view 和一个空 scratch），通过 JSON lines 通信，每次调用限时。 |
| `plugin_worker.py` | 沙箱内部：加载候选的 `databuilder.py` / `analysis.py`，处理 `init` / `next_batch` / `after_step` / `state` / `load` / `ping` 请求。 |
| `materialize.py` | recipe → 渲染文本 → `internvlu_train` Row。唯一生成训练行的地方：F0 和 final target 用 PNG 端点，context 帧和更早的 target 用颜色校正 mp4 帧；`cond_image` 永远是 F0。 |
| `train_entry.py` | 一个训练 job 的入口（`torchrun --nproc_per_node=4 runner/train_entry.py --spec <run_dir>/run.json`）。每步 rank 0 向插件要 16 个 recipe、检查、广播；各 rank 训练自己的行；逐行噪声 / dropout / CFG-drop 由 `(run_seed, k, j)` 决定（与 GPU 数无关）；逐行 loss 汇总到 rank 0 → `Analysis.update` → `Builder.observe` → `request_probes`，probe 无梯度、保存恢复 RNG；每 50 步存 checkpoint（含插件状态）；训练完在 dev 或 test 上生成并打分。`steps = 0` 表示只评测（未训练的 Base）。 |
| `jobs.py` | GPU slot 与 job 启动：一个 job = 一个 4-GPU slot 上的 `train_entry.py` 进程组，nohup 后台启动并把退出码写到 `<run_dir>/exit_code`；启动前确认该 slot 的 GPU 空闲（< 2000 MiB 且无计算进程），不碰其他用户的任务；`status` 把 `DONE.json` 视为完成，每 5 分钟检查进程组存活（防止 wrapper 死掉没写 exit_code 导致卡住）。 |
| `score.py` | 调用 `scorers/analogy_grid.py` 按任务给每张生成图打分，输出 `{score, sub, valid, categories, gates}`；缺失 / 不可读 / 全黑 / 纯色图记为 invalid（0 分）。可单独运行：`python runner/score.py <eval_dir> <eval_jsonl>`。 |
| `stats.py` | 统计与选择规则：逐 item 均值、按任务等权聚合、分层配对 bootstrap、训练噪声 `sd_train`、confirm 比较（`SE² = 2 sd_train² / R + SE_item²`，Bonferroni 校正 over top-k）。 |
| `audit.py` | 训练后文本泄漏审计：检验模板选择是否携带了 description 以外的 target 像素信息（final 帧前景面积四分位，条件于 (scale_from, scale_to)）。统计量为条件互信息 CMI，置换检验 1,000 次；`p < 0.001` 且 `CMI > 0.02 bits` 则拒绝候选。 |
| `build_data.py` | 从冻结的 reactdata-v4 数据构建 v5 数据：分层抽 16 个 clip，硬链接帧，写 manifests 和 `pool_view.json`；`data/BUILD_DONE.json` 存在时拒绝重建。 |
| `final_report.py` | run 结束时生成 `<run_root>/report.md` + `summary.json`（并备份到 `records/<run_id>/`）。`build_v3` 对应当前 design v3。 |
| `common.py` | 共享的路径常量、hash（`sha256_file` / `sha256_tree` / `sha256_json`）、JSON/JSONL 读写、`wrap_prompt`（VBVR target_pred 的评测文本包装）。 |

### 3.4 `scorers/`

| 文件 | 作用 |
|---|---|
| `analogy_grid.py` | 规则打分器 v1.0：检测物体 → 按 y = 256 分行建网格 → 只给 E、F 两个需要生成的格子打分（E_cell / F_cell / F_transform），A–D 作 gate；输出错误类别（`E_missing`, `F_wrong_transform`, `context_damaged` 等）。依赖 `Evaluation/VBVR-CustomEval/scorers/cvlib.py`。在 GT / 拷贝输入 / 不变换等合成案例上的校验结果见 `manifests/scorer_validation.json`。 |

### 3.5 `seeds/fixed_f0_final/` — 固定 baseline（c000）

| 文件 | 作用 |
|---|---|
| `databuilder.py` | 每行从训练池无放回均匀抽一个 clip（每轮一个新的 seeded 排列），无 context 帧，模板 `eval_format`，target = final。无反馈、无 schedule。 |
| `analysis.py` | 只记录训练 loss 的滑动均值，不请求 probe。 |
| `text_templates.yaml` | `eval_format: "{wrapped_description}"`——和 dev/test 完全相同的文本。 |
| `candidate.yaml` | `declared_class: fixed`。 |
| `proposal.md` / `memory.md` | baseline 的说明；空的 agent 笔记（agent 的第一个 /work 从这里复制）。 |

候选程序目录（seed 或 agent 写的）永远是这 6 个文件，不能有其他文件，每个 ≤ 100 kB。

### 3.6 `manifests/` — 冻结数据清单

| 文件 | 作用 |
|---|---|
| `train_pool.jsonl` | 16 个训练 clip（`clip_id`、原始样本目录、`prompt`）。 |
| `inner_probe.jsonl` / `inner_probe_eval.jsonl` | 32 个 inner-probe clip，分别是 clip 格式和评测格式（generate probe 使用后者）。 |
| `dev.jsonl` | 100 条 dev（`dev_scale_*` 50 + `dev_move_*` 50），评测格式（F0、cond_image、target_image、对话文本）。 |
| `test.jsonl` | 100 条 `test_move_*`。 |
| `data_checks.json` | 构建检查：池大小、抽样 seed、12 个尺寸变化 stratum 的计数、选中的 clip 列表等。 |
| `isolation_report.json` | 各 split 之间的隔离检查：md5 重复、生成参数重复、缩略图最近邻距离、生成器 seed 范围。 |
| `scorer_validation.json` | 打分器在合成案例（GT=1.0、拷贝输入=0、不变换≈0.355、错误颜色……）上的校验。 |
| `frame_stats_spec.json` | `pool_view.json` 中 `frame_change` / `fg_area` 两个像素统计的定义和计算代码 hash。 |

### 3.7 `tests/`

| 文件 | 作用 |
|---|---|
| `integration/test_v3_flow.py` | design-v3 状态机的 CPU 端到端测试（无 GPU、无 LLM）：伪造训练结果（真实效应 + 训练噪声 sd 0.02 + item 噪声）和 agent（一个有 +0.08 真实效应的程序，一个只改 config 的副本必须判为 duplicate）；checks / 状态机 / 统计 / 选择 / 报告是真的。断言阶段顺序、10 个串行候选的 lineage、duplicate 规则、top-3 配对 seed 确认、选中 +0.08 程序、P3 多 seed、v3 报告、replicate-mean 参照、agent 输入、proposal 修复。`python tests/test_v3_flow.py` |
| `integration/test_sandbox.py` | 插件沙箱和 agent 沙箱的隔离测试（用 `evil_cand` 尝试读受保护文件、联网；检查 agent 的挂载点）。`python tests/test_sandbox.py` |
| `fixtures/evil_cand/` | 恶意候选：试图在沙箱里读文件 / 联网，必须失败。 |
| `fixtures/probe_cand/` | 正常的测试候选：一半行 target 中间帧（sub-goal），带 context、feedback 和 probe——用来覆盖 checks 和训练的完整功能。 |
| `fixtures/c002_regression_cand/` | 回归用例（reactdata-v0 c002 的失败）：有 context 帧但模板缺 `{context_list}`，checks 必须拒绝。 |
| `fixtures/v4_ctx_after_target/` | 回归用例：context 帧（40）晚于 target 帧（28），checks 必须拒绝。 |
| `fixtures/v4_wrapped_nonfinal/` | 回归用例：非 final target 使用了评测 wrapper 模板，checks 必须拒绝。 |

候选目录可以直接用 `python runner/checks.py tests/fixtures/<cand>` 检查。

### 3.8 `records/<run_id>/` — run 记录备份（进 git）

由 `state_machine.backup()` 从 `runs/<run_id>/` 复制小文件（≤ 2 MB 的 json/jsonl/yaml/md 等），跳过 checkpoint、图片、`input/`、`work_start/`、`agent_home/`、`plugin_scratch/`、逐行大文件。结构与 §5 的 run 目录一致：`run.json`、`state.json`、`noise.json`、`confirm.json`、`selection.json`、`report.md`、`summary.json`、`candidates/<cid>/{candidate.json, code/}`、`rounds/<NNN>/<cid>/`、`jobs/<job>/`、`frozen/<cid>/`。

---

## 4. Pipeline 如何运作

```
build_data (一次性)
   │
   ▼
P0  pilot：baseline 4 步训练 + 8 条 dev 生成 → 测时间、首步验收（失败则阻塞）
   │
   ▼
P1  baseline c000 × 5 个 run seed (1000–1004) + 未训练的 Base，都在 dev 上评测
    → sd_train（训练噪声）、delta = sd_train、逐 item baseline 均值 / sd
   │
   ▼
P2  搜索：10 轮 × 1 个候选（串行，每个候选都看到之前的所有结果）
    每个候选 cNNN：
      ① report.py 写 /input；/work = agent 最近一个有效候选的拷贝（第一个是 c000）+ 上一份 memory.md
      ② agent_call：同一个 Claude 会话 resume 继续，读证据 → 写 proposal.md → 改 /work
      ③ checks.py + proposal 章节 / item 引用检查；失败 → 一次 repair 调用 → 再检查
      ④ 去重：databuilder.py / analysis.py / text_templates.yaml 与已有程序相同 → duplicate（不训练，仍占名额）
      ⑤ train_entry.py 训练 100 步（screen_seed 1000）→ dev 生成 + 打分
      ⑥ audit.py 文本泄漏审计（flag → 拒绝）
      ⑦ screening D = 候选 − baseline replicate 逐 item 均值
    停止条件：10 个候选、墙钟预算（为后续阶段预留 3.2 h）、连续 3 次拒绝
   │
   ▼
CONFIRM  screening D 最大的 top 3，各在 seed 1001–1004 上重训
         与相同 seed 的 baseline replicate 配对：D = 各 seed 差值均值，
         SE² = 2·sd_train²/R + SE_item²，Bonferroni over 3（z = 2.39）
   │
   ▼
SELECT   D ≥ delta 且 CI 下界 > 0 的确认候选中取 D 最大者；否则选固定 baseline c000
   │
   ▼
FROZEN   把选中程序复制到 frozen/<cid>/ 并记录代码 sha256（此后才允许访问 test）
   │
   ▼
P3   选中程序与 c000 各在 formal seed 2000–2002 上重训，加上 Base，在 test 上评测
     → 同样的配对统计（selected vs baseline）
   │
   ▼
REPORT → report.md + summary.json → DONE
```

### 4.1 一个训练 step 内部（`train_entry.py`）

```
recipes = Builder.next_batch(k)          # 沙箱内，16 个 recipe
check_batch(recipes)                     # api/recipe.py：规则不满足 → 候选失败
rows = Materializer.render(recipes)      # 渲染文本 + 读取帧
loss = train_step(rows)                  # 4 GPU，逐行噪声由 (run_seed, k, j) 决定
summary = Analysis.update(k, per_row_stats, step_info, probe_results_of_k-1)
Builder.observe(k, summary)              # 反馈型程序在这里调整数据
requests = Analysis.request_probes(k, budget)   # 在 inner-probe 上做 loss / generate probe
```

job 目录中对应的产物：`rows_realized.jsonl`（实际训练的行）、`signals.jsonl`（analysis 摘要）、`probes.jsonl`、`metrics.jsonl`、`plugin.log`。

### 4.2 隔离与防作弊

- agent 只能读写 `/work` 和只读的 `/input`；看不到 Runner、打分代码、manifests、metadata、test 集、真实的 `~/.claude`。
- 候选代码在无网络、无文件系统的 `bwrap` 沙箱里运行，只能 import 白名单模块，每次调用 10 s 限时，必须确定性且可 `state_dict` 恢复。
- 训练文本只能由静态模板 + Runner 填充的 slot 组成；模板选择是否泄漏 target 信息由 `audit.py` 事后检验。
- test 只能在 FROZEN 之后请求（`state_machine.evals` 里硬性检查）。
- `run.json` 记录 protocol / runner / api / BUILDER_AGENT / 打分器的 sha256，保证结果对应的代码可追溯。

---

## 5. run 目录结构（`/scratch/network/ssd/junlin/reactdata-v5/runs/<run_id>/`）

```
run.json            代码与数据 hash、数据计数、打分器版本
state.json          状态机全部状态（phase、每个候选的 status / dev_mean / comparison / confirm、噪声、agent session id …）
events.jsonl        事件流（phase_start、agent_call、candidate_checked、job_launched、candidate_trained …）
noise.json          P1：5 个 baseline replicate 的 dev 分、mean、sd_train、delta
confirm.json        CONFIRM 结果
selection.json      SELECT 决定
report.md / summary.json   最终报告
agent_home/         持久的 agent HOME（Claude 会话文件；agent 工具读不到）
candidates/<cid>/
  candidate.json    该候选的记录（parent、class、agent 调用、状态、dev 分）
  code/             通过检查的程序（6 个文件）
rounds/<NNN>/<cid>/
  input/            该轮 agent 看到的 /input（report.md、results.json、dev_items/、analysis_traces/、failures/、candidates/、train_examples/、api/、BUILDER_AGENT.md）
  work_start/       /work 的初始内容
  work/             agent 修改后的 /work
  agent_0/, agent_repair/   每次 agent 调用的 prompt.md、cmd.json、stdout.json、stderr.txt、output.json
  check_work/       checks.py 模拟的 scratch 与 plugin.log
  checks.json       检查结果
  patch.diff        相对起点程序的 diff
jobs/<job>/         一个训练 / 评测 job，名为 <cid>-s<seed>-g4、base_dev、p0_pilot、formal_<cid>-s<seed>-g4、base_test
  run.json, config.yaml, train.log, metrics.jsonl, progress.json, first_step.json, hashes.json, cost.json
  rows_realized.jsonl, signals.jsonl, probes.jsonl, probes/, plugin.log, plugin_scratch/
  checkpoints/, eval/<split>/g<seed>/（生成图片与 scores.json）
  audit.json, pid, exit_code, DONE.json 或 FAILED.json
frozen/<cid>/       选中并冻结的程序
```

---

## 6. 常用命令

所有命令在本目录下运行，Python 为 `~/miniconda3/envs/internvlu/bin/python`。

```bash
# 构建数据（只需一次；BUILD_DONE.json 存在时拒绝）
python runner/build_data.py

# CPU 测试
python tests/test_v3_flow.py
python tests/test_sandbox.py

# 冒烟测试（4 步训练、4 条 eval、1 个候选，独立 run id）
python -u runner/state_machine.py --run-id smoke1 --smoke --protocol protocol.yaml

# 正式 run（重跑同一命令即恢复）
nohup setsid python -u runner/state_machine.py --run-id r1 --protocol protocol.yaml \
    >> /scratch/network/ssd/junlin/reactdata-v5/runs/r1.log 2>&1 &

# 单独检查一个候选 / 单独打分一个 eval 目录
python runner/checks.py <candidate_dir> --out checks.json
python runner/score.py <eval_dir> <eval_jsonl>

# 查看进度
tail -f /scratch/network/ssd/junlin/reactdata-v5/runs/r1.log
pgrep -af "[s]tate_machine"          # 用 [s] 避免匹配到 pgrep 自己
```

结果看 `runs/<run_id>/report.md`（结束后也在 `records/<run_id>/report.md`）；运行中各候选的 screening 结果在 `state.json` 的 `candidates` 字段和每轮的 `rounds/<NNN>/<cid>/input/report.md`。

注意：
- `protocol.yaml`、`runner/`、`api/`、`BUILDER_AGENT_v5.md` 的 hash 在 run 开始时写入 `run.json`；run 进行中不要修改它们。
- 杀进程时不要用 `pkill -f <pattern>`：如果 pattern 也出现在你自己 shell 的命令行里，会把自己的 shell 一起杀掉。
