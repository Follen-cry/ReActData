# Curation-bench Linux GPU 启动准备清单

本清单针对当前 `ReActData/curation-bench` 的 Video SSL 实现，默认沿用 `shape_color_then_scale` 任务、InternVL-U 模型和 native 执行模式。下文代码路径均相对于 `ReActData/curation-bench/`；启动命令也在该目录执行。

清单编写日期：2026-10-31。这里的勾选框表示尚待确认，不代表已完成服务器验证。

## 1. 必须提供的信息

下面“实际值”一栏供人工记录；pipeline 不读取本清单。实际生效的位置见第三列。`profiles/server_a.yaml` 和 `campaigns/kickoff_001/campaign.json` 是建议新建的文件名，使用其他名字也可以，但必须更新引用。

| 项目 | 实际值／待填写内容 | 实际填写位置／生效方式 |
|---|---|---|
| Linux 服务器与运行用户 | 待填写 | 服务器登录和运行环境；在这里记录即可，没有对应 task/profile 字段 |
| 服务器仓库绝对路径、代码版本 | 待填写 | 服务器 clone/checkout 的目录及 revision；启动前 `cd` 到该目录；在这里记录路径和版本，没有独立配置字段 |
| Method CLI Python ≥3.12 | 待填写 | 初始化命令 `uv sync --locked --python /ABS/CLI/PYTHON`；如需明确生成命令使用的 CLI Python，在 `configs/runner.yaml` 设置 `cli_python` |
| GPU worker Python 绝对路径 | 待填写 | `profiles/server_a.yaml` → `python` |
| GPU 型号、显存 | 待填写 | 在这里记录并在服务器验证；没有 GPU 型号或显存配置字段 |
| 分配的 GPU 设备 ID | 待填写 | `profiles/server_a.yaml` → `gpu_allocation`，并同步 `campaigns/kickoff_001/campaign.json` → `gpu_allocation` |
| 训练源目录 | 待填写 | `profiles/server_a.yaml` → `dataset_path_map.shape_color_train`；task 的 `dataset_id` 选择该 key |
| 训练 source manifest | 待填写 | `tasks/shape_color_video_ssl.yaml` → `sources_manifest`，是相对于训练源目录的路径；默认 `train_sources.jsonl`；实际视频/帧/文本写进该 JSONL |
| dev manifest | 待填写 | `profiles/server_a.yaml` → `eval_path_map.shape_color_dev`；task 的 `dev_id` 选择该 key |
| test manifest | 待填写 | `profiles/server_a.yaml` → `eval_path_map.shape_color_test`；task 的 `test_id` 选择该 key |
| InternVL-U base snapshot 与版本 | 待填写 | 路径：`profiles/server_a.yaml` → `model_path_map.internvlu_base`；task 的 `model_key` 选择该 key；版本在这里记录，并固定对应资源 |
| 外部 internvlu_train 路径与版本 | 待填写 | 路径：`profiles/server_a.yaml` → `resource_path_map.internvlu_train`；版本通过外部库 checkout 固定并在这里记录 |
| prompt_wrapper | 待填写 | `profiles/server_a.yaml` → `resource_path_map.prompt_wrapper` |
| scorer_cvlib | 待填写 | `profiles/server_a.yaml` → `resource_path_map.scorer_cvlib` |
| training_meta | 待填写 | `profiles/server_a.yaml` → `resource_path_map.training_meta` |
| Agent CLI 可执行文件 | 待填写 | `configs/runner.yaml` → `agent.executable` |
| Agent 模型与推理设置 | 待填写 | `configs/runner.yaml` → `agent.model`、`agent.reasoning_effort`；null 继承服务器 CLI 设置 |
| Agent 认证方式 | 待填写；不要填写密钥 | Native：在实际运行用户环境中完成 CLI 认证；Docker：按需要配置 `configs/docker.yaml` → `credential_files` / `pass_env`；本表只记录方式 |
| campaign ledger 路径 | 待填写 | `run_prompt.md` YAML frontmatter → `campaign`，引用实际 ledger 文件，相对路径按 run_prompt.md 所在目录解析 |
| 截止时间、保留时间、预算 | 待填写 | `campaigns/kickoff_001/campaign.json` → `deadline`、`report_reserve_seconds`、`training_budget`、`max_concurrent_gpus`；已有 `training_attempts` 不可清空 |
| 本次任务、轮数、profile、runner、protocol | 待填写 | `run_prompt.md` YAML frontmatter → `tasks`、`iterations`、`profile`、`runner`、`protocol` |
| 本次 suite 输出目录 | 待填写 | 启动参数 `video-ssl run --prompt run_prompt.md --root /ABS/RUN/DIR`；不是 profile 字段 |

填完后再核对第 3–9 节：这张表指明主要值的落点，后续章节说明数据文件内容、依赖和参数之间必须满足的约束。`workspace/task.json`、`config/backend.json` 和 `workspace/COMMANDS.md` 由框架生成，不是启动前手工填写的位置。

## 2. 机器 profile：替换示例路径

- [ ] 复制 `profiles/example.yaml` 为 `profiles/server_a.yaml`，填写真实服务器配置。
- [ ] 确认相对路径按 profile 文件所在目录解析；建议机器资源使用绝对路径。

| 字段 | 需要指定什么 |
|---|---|
| `python` | 已安装 GPU/model/data 依赖的 Python 可执行文件 |
| `dataset_path_map.shape_color_train` | 训练源目录 |
| `model_path_map.internvlu_base` | 完整 base snapshot 目录 |
| `eval_path_map.shape_color_dev` | dev JSONL 文件 |
| `eval_path_map.shape_color_test` | 独立 test JSONL 文件 |
| `resource_path_map.internvlu_train` | 能支持当前 backend 导入方式的外部训练库路径 |
| `resource_path_map.prompt_wrapper` | 实际文本准备脚本文件；用于准备来源记录与配置检查 |
| `resource_path_map.scorer_cvlib` | 评分时实际加载的 cvlib 文件 |
| `resource_path_map.training_meta` | 外部模型库使用的 metadata 文件 |
| `gpu_allocation` | 本次分配的 GPU ID；与 campaign 完全一致 |
| `env` | 必要的机器环境设置，值必须为字符串 |

`training_meta` 不是本轮训练集。本轮实际训练输入由 accepted submission 转换成 `rows.json`。不要在静态配置里写凭据；不要通过 profile.env 覆盖框架保留的 GPU 和运行控制变量。

## 3. 训练视频与帧库

接口示例：`data/train_sources.example.jsonl`。任务默认寻找训练源目录下的 `train_sources.jsonl`。

- [ ] 准备真实 source JSONL，替换示例中的描述与文本占位符。
- [ ] 每条记录具有唯一 `source_video_id`，不同 ID 不应指向同一个视频路径。
- [ ] source 数量不少于 `rows_per_trial`；当前默认每轮需要 128 个不同视频。
- [ ] 每条记录提供可读的 `video`、`f0`、`final` 文件。
- [ ] `f0`、`final` 和所有可选中间帧为 canonical 512×512 PNG。
- [ ] `frame_count` 与实际视频对应，帧索引为 zero-based。
- [ ] 在 `frames` 中登记希望允许选择的中间帧；索引必须是标准整数字符串，且严格位于首末帧之间。
- [ ] 提供真实、非空的 `description` 和准确的 `eval_text`。
- [ ] 检查 baseline 文本的图像占位符与输入格式匹配。
- [ ] Host 审核视频、帧、文本之间的对应关系，以及内容级重复。

Manifest 内资源路径相对于 manifest 所在目录解析。可选帧是首帧、末帧与 `frames` 中登记的帧，并非固定间隔。Agent 不负责下载数据或创建未登记帧库；训练实际消费构建后的图片监督，而非直接消费 MP4。

## 4. Dev / test 数据

接口示例：`data/eval.example.jsonl`。

- [ ] 分别准备独立的 dev 与 test JSONL，profile 中指向不同文件。
- [ ] 提供唯一字符串 `id`、一张输入 `image`、`target_image` 和准确的 `conversations`。
- [ ] 若指定 `cond_image`，它必须与第一张输入图一致。
- [ ] 确认所有图像文件存在、可读，输入/目标符合固定 F0-to-final 评估语义。
- [ ] 数量与 task contract 一致；当前默认 dev 50 条、test 50 条。
- [ ] Host 检查 train/dev/test 的内容重复、来源关系和划分独立性。
- [ ] 约定最终 test 的使用方式，避免根据 test 反馈继续修改本次实验。

结构校验不能证明 split 的语义独立性。Agent 可以分析公开 dev 反馈，但训练行仍只能来自训练 source 白名单。

## 5. Python、CUDA 与模型依赖

### Method CLI 环境

- [ ] 安装可用的 uv 和 Python ≥3.12。
- [ ] 保留仓库的 `pyproject.toml`、`uv.lock`、`.python-version`。
- [ ] 在项目目录执行 `uv sync --locked --python /ABS/CLI/PYTHON`。

### GPU worker 环境

- [ ] profile.python 指向已有、可运行的 GPU Python。
- [ ] 安装该环境所需的数据工具；参考 `vendor/video_ssl_backend/requirements.txt`。
- [ ] 按外部训练库要求准备完整模型依赖；backend requirements 不包含完整模型环境。
- [ ] 固定并记录外部训练库、base snapshot、PyTorch/CUDA 及关键模型依赖版本。
- [ ] 确认模型权重、配置、tokenizer 和其他模型资源完整且可读。
- [ ] 验证 CUDA、BF16 和 NCCL 分布式执行路径适配实际 GPU/驱动环境。
- [ ] 通过真实 GPU 验收确认模型加载、显存和训练/生成行为。

两个 Python 环境可以不同。运行期间不应让 Agent 安装依赖、替换环境或下载替代模型。静态 check 不会加载模型，也不会证明 GPU 可用。

## 6. Task contract 与训练模板

主配置：`tasks/shape_color_video_ssl.yaml`。低层模板：`configs/training/internvlu.yaml`。

- [ ] 确认本次仍采用当前支持的 `shape_color_then_scale` 任务与评分器。
- [ ] 确认样本数、训练步数和 batch；当前为 128 行、8 步、global batch 16。
- [ ] 保持 `rows_per_trial == max_steps * global_batch`。
- [ ] GPU 数量能够整除 global batch；当前后端配置最多支持 4 张 GPU。
- [ ] 确认初始化、训练、构建与评估种子。
- [ ] 确认 `frame_policy`；当前允许选择帧，最多 4 张输入，目标可选。
- [ ] 确认 `strategy_timeout_seconds`、train/dev/test timeout 和 `estimated_trial_seconds`。
- [ ] 根据服务器测得耗时校准时间估计，默认值不是服务器性能保证。
- [ ] 确认 `preinit`；默认 true。
- [ ] 确认 `finalize`；默认 manual，需要 Host 手动执行最终 test。
- [ ] 核对低层模板的模型结构、adapter、预处理和外部训练库 API 是否匹配。

模板中的 `__MODEL_PATH__`、`__MAX_STEPS__` 等由 resolver 根据 profile/task 自动替换，无需手工逐项替换。显式模板值必须与 task contract 一致。

仅修改 YAML 的任务名称不能实现新任务。若换目标、scorer、模型输入/输出语义，还需修改 `vendor/video_ssl_backend/recipe.py`、`task.py`、评分与模型适配代码，并更新相应协议和验证。

## 7. Campaign：真实截止时间与共享预算

- [ ] 从 `campaigns/example/campaign.json` 创建本次 kickoff 的 ledger，放在独立目录。
- [ ] 将 deadline 占位符替换为带时区的实际 ISO8601 时间。
- [ ] 设置 `report_reserve_seconds`；当前模板为 5400，至少需满足框架要求。
- [ ] `gpu_allocation` 与 profile 完全一致。
- [ ] 指定 `max_concurrent_gpus`、`training_budget`，符合当前 1–4 GPU、1–7 次训练的限制。
- [ ] 全部声明轮次及验收实验都纳入预算，并留出 preinit/dev/test 的时间。
- [ ] 确认运行用户可写 ledger 所在目录，允许锁和原子替换。

只有全新 kickoff 的 `training_attempts` 才应为空。已注册的失败训练也消耗一次预算；不得清空已有 ledger 来重置预算。

## 8. Agent runner 与认证

配置：`configs/runner.yaml`。首次可用默认 `mode: native`。

- [ ] 安装并认证 Agent CLI，使其能非交互运行。
- [ ] `agent.executable` 在运行环境中可找到；当前 runner 按 Codex CLI 构造命令。
- [ ] 验证已安装 CLI 支持 runner 使用的参数和 JSON 输出方式。
- [ ] 明确 `agent.model`、`agent.reasoning_effort`，或记录 null 将继承本机设置。
- [ ] 确保 Agent 所需网络和认证可用，凭据不要写入待冻结配置。
- [ ] 按需要设置整个 Agent 会话的 timeout；它与每轮策略时间、训练时间不同。

`workspace/COMMANDS.md` 由 Host 的 `runner/agent_runner.py` 在启动 Agent 前生成，使用解析后的实际路径。无需手工补写，也不应在已启动实验中编辑它。

### 仅 Docker 模式需要

- [ ] 指定已准备好的 GPU 镜像，替换 `configs/docker.yaml` 的 image 占位符。
- [ ] 验证 Docker GPU 支持、容器内 Agent CLI、CLI Python 和 GPU Python。
- [ ] 核对资源 mounts、模型缓存 symlink 的目标及容器内路径。
- [ ] 配置所需认证文件映射或环境转发，不在配置中填写秘密值。
- [ ] 确认 run 目录和 campaign 目录可写，资源权限符合设计。

Native 模式不读取 docker.yaml，无需为 native 启动补齐 Docker 配置。

## 9. Run declaration 与输出目录

- [ ] 在 `run_prompt.md` 中选择实际 task、profile、runner、protocol 和 campaign。
- [ ] `iterations` 是包含训练 baseline 的总轮数，不包括 preinit 和最终 test。
- [ ] Markdown 正文写明本次研究方向，不用它覆盖 task 的数值约束。
- [ ] 选择新的 suite 输出目录，确认权限和存储空间足够容纳提交图片、checkpoint、预测及日志。
- [ ] 已有实验使用支持的 resume 流程，不通过覆盖目录重置状态。

示例 frontmatter，需根据实际 kickoff 修改：

```yaml
---
tasks: [shape_color_video_ssl]
iterations: 3
profile: server_a
runner: configs/runner.yaml
protocol: method
campaign: campaigns/kickoff_001/campaign.json
---
```

## 10. 按顺序验收

### A. 静态检查，无 GPU 训练

```bash
cd /ABS/REPO/ReActData/curation-bench
uv sync --locked --python /ABS/CLI/PYTHON
uv run --locked video-ssl check --prompt run_prompt.md
```

- [ ] check 成功；确认任务、路径、样本数量、训练配方和预算。
- [ ] 理解 check 不保证模型 API 兼容、GPU/显存充足或 split 独立。

### B. 真实 GPU baseline 验收

可以先通过手动 staged 流程验证后端；它会消耗实际训练预算。命令的完整流程见 README 的 Manual staged operation。

- [ ] init 后检查冻结配置与生成 workspace。
- [ ] preinit（若启用）可正常生成和评分。
- [ ] unchanged baseline 能 build、audit、submit。
- [ ] finetune 生成有效 checkpoint receipt。
- [ ] eval 能重新加载 checkpoint，输出真实 dev metrics 和预测图片。
- [ ] 检查显存、耗时、无效预测数和模型行为，记录版本及资源。

### C. Agent 连续流程

使用为本次实验分配的计划与预算启动：

```bash
uv run --locked video-ssl run --prompt run_prompt.md --root runs/video-ssl-001
```

- [ ] Agent 在 Host 创建的 workspace 中启动，读取 METHOD.md / COMMANDS.md / task.json。
- [ ] Agent 完成声明的 baseline 和改进轮次，最终 next 返回 task_done。
- [ ] 查看 suite README、results.tsv、各轮 REPORT、代码/选择差异与 NOTES.md。
- [ ] 若 finalize 为 manual，由 Host 在搜索完成后执行：

```bash
uv run --frozen --no-sync video-ssl finalize --suite-dir runs/video-ssl-001
```

不要为验收反复重置训练预算或盲目重跑失败/in-flight 操作。发生中断时先检查保存阶段和日志，再按 lifecycle 规则恢复。

## 11. 进一步定位

| 问题 | 首先查看 |
|---|---|
| 配置字段、引用与路径解析 | `docs/configuration.md`、`src/video_ssl/core/layout.py` |
| 初始化、resume、轮次与最佳代码恢复 | `docs/lifecycle.md`、`src/video_ssl/tools/suite.py` |
| Agent prompt、生成命令与启动 | `src/video_ssl/runner/agent_runner.py` |
| 构建数据及帧选择 | `curation/curate.py`、`vendor/video_ssl_backend/frame_policy.py` |
| 数据拒绝原因 | `vendor/video_ssl_backend/dataset.py` |
| 支持的任务与训练契约 | `vendor/video_ssl_backend/recipe.py`、`task.py` |
| 训练、生成与外部库接口 | `vendor/video_ssl_backend/training.py`、`real_backend.py` |
| 评分依赖与实际打分 | `vendor/video_ssl_backend/scorer_loader.py`、`real_score.py` |

## 12. 可指定字段速查：文件、格式、当前值与约束

本节补充前述勾选清单的具体填写位置。下列“当前值”取自仓库模板，不等于服务器实际配置，也不表示必须采用该数值。路径中的 `/ABS/...` 都必须替换；资源 ID 可以自定义，但 task 与 profile 两侧引用必须同步。未知字段通常会被拒绝，不能通过向 YAML 随意加键扩展后端。

### 12.1 run_prompt.md：本次运行声明

以下字段都写在文件顶部 `---` 包围的 YAML frontmatter 中；研究方向写在 frontmatter 之后的 Markdown 正文。

| 字段 | 格式／当前值 | 填写与关联规则 |
|---|---|---|
| `tasks` | 字符串列表；`[shape_color_video_ssl]` | 对应 `tasks/<ID>.yaml`，且文件内 `task_id` 相同；不能填写路径表达式 |
| `iterations` | 正整数；3 | 每个任务的总轮数，含 baseline；不超过 contract.max_trials 和共享剩余预算 |
| `profile` | 名称或 YAML 路径；example | 改为实际 server_a 或实际 profile 路径 |
| `runner` | 名称或 YAML 路径；configs/runner.yaml | 选择 Agent 执行配置 |
| `protocol` | 名称或 Markdown 路径；method | method 为基础协议；instruction/skill 会组合基础协议和对应附录；自定义显式协议路径按其内容加载 |
| `campaign` | ledger 路径 | 相对于本 run_prompt 所在目录；引用已存在的 ledger |
| `tasks_dir` | 可选目录路径 | 相对于本 run_prompt；省略时使用仓库 tasks/ |
| 正文 | Markdown | 研究方向、关注问题；不覆盖冻结数值、资源和权限约束 |

Profile/runner/protocol 引用先尝试声明文件旁的路径，再查仓库对应目录和扩展名。源数据路径相对 manifest，profile 资源路径相对 profile；不要把这些相对路径基准混用。

### 12.2 tasks/shape_color_video_ssl.yaml：资源、策略与预算边界

| 字段 | 格式／当前值 | 填写与关联规则 |
|---|---|---|
| `task_id` | shape_color_video_ssl | 与文件名及 run_prompt.tasks 对应；改名不改变后端任务能力 |
| `goal` | 非空目标文本 | 描述研究目标，传入 Agent prompt 和公开任务上下文 |
| `dataset_id` | shape_color_train | 对应 profile.dataset_path_map 的 key |
| `sources_manifest` | train_sources.jsonl | 相对于该 dataset 根目录；不能是绝对路径或含 `..` |
| `model_key` | internvlu_base | 对应 profile.model_path_map 的 key |
| `training_config` | ../configs/training/internvlu.yaml | 相对于 task YAML 的低层模板路径 |
| `resources.model_library` | internvlu_train | 对应 profile.resource_path_map 的 key |
| `resources.prompt_wrapper` | prompt_wrapper | 同上，对应实际准备脚本 |
| `resources.scorer_cvlib` | scorer_cvlib | 同上，对应实际评分依赖 |
| `dev_id` / `test_id` | shape_color_dev / shape_color_test | 对应 profile.eval_path_map 的 key；必须是不同文件 |
| `strategy_timeout_seconds` | 正有限秒数；1800 | next 激活本轮后开始计时，限制提交；不支持 null/无限 |
| `estimated_trial_seconds` | 正有限秒数；3600 | 保守估计 train+dev 耗时，用于预算余量检查；按服务器验收校准 |
| `preinit` | 布尔；true | 是否先做无 optimizer 更新的 base dev reference |
| `finalize` | manual / auto；manual | manual 由 Host 执行命令；auto 在研究完成后由 Host 框架触发 |
| `frame_policy.mode` | select / endpoints；select | endpoints 只接受单输入 F0 到末帧的兼容组合 |
| `frame_policy.max_input_frames` | 正整数；4 | 最大输入图数；endpoints 模式必须为 1 |
| `frame_policy.target` | select / final；select | final 只允许末帧；endpoints 必须为 final |
| `max_dataset_bytes` | 正整数；500000000 | 提交目录总字节上限，包含未使用文件 |
| `max_text_chars` | 正整数；8000 | source 文本和训练行文本的相应长度限制 |

### 12.3 同一 task YAML 的 contract：训练、评估和选择

下面字段路径均相对于 `contract`。它们是当前后端接受的契约，不是任意 trainer 参数透传。

| 字段 | 当前值 | 可填写范围／需要联动的位置 |
|---|---|---|
| `task` | shape_color_then_scale | 当前唯一支持的真实任务；换任务需实现后端 |
| `rows_per_trial` | 128 | 正整数，等于 max_steps × global_batch，且不大于 source 数量 |
| `max_trials` | 7 | 1–7；限制 run_prompt.iterations，仍需满足 campaign 预算 |
| `model_init_seed` | 42 | 整数；初始化种子 |
| `train_seed` | 1000 | 整数；训练过程随机性 |
| `build_seed` | 1000 | 整数；baseline 选择和顺序 |
| `eval_seed` | 0 | 整数；固定生成种子 |
| `training.max_steps` | 8 | 正整数；联动 rows_per_trial 和 warmup_steps |
| `training.global_batch` | 16 | 正整数；GPU 数必须能整除它 |
| `training.epochs` | 1 | 当前必须为 1 |
| `training.shuffle` | false | 当前必须为 false，按 manifest 顺序训练 |
| `training.warmup_steps` | 3 | 整数，0 ≤ 值 < max_steps |
| `training.scheduler` | linear_warmup_then_cosine | 当前固定支持值 |
| `training.lora_lr` | 0.00002 | 有限非负数；自动写入模板 optim.lr |
| `training.gen_decoder_lr` | 0.0001 | 有限非负数；自动写入模板对应学习率 |
| `training.lm_weight` / `gen_weight` | 0.0 / 0.5 | 有限非负数；自动写入 loss |
| `training.gen_warmup_steps` | 0 | 整数，0–max_steps |
| `training.image_size` | 512 | 当前固定 512；不能只改数值就支持其他图片尺寸 |
| `training.checkpoint` | last_step | 当前固定；不根据 dev 挑中间训练步 |
| `training.restart_from_base_each_trial` | true | 当前必须为 true；每轮不继承上一轮权重 |
| `training.timeout_seconds` | 36000 | 正整数秒；还受 campaign 截止和保留时间限制 |
| `evaluation.samples` | 50 | 正整数；dev/preinit manifest 的精确条数 |
| `evaluation.test_samples` | 50 | 正整数；test manifest 的精确条数 |
| `evaluation.diffusion_steps` | 20 | 正整数；生成步数 |
| `evaluation.all_cfg_scale` / `part_cfg_scale` | 4.5 / 2.0 | 有限非负数；自动写入评估模板 |
| `evaluation.scorer` | analogy_grid_v1_scale | 当前固定评分器；换名称需后端实现 |
| `evaluation.aggregation` | arithmetic_mean_including_invalid | 当前固定；包含无效预测的算术平均 |
| `evaluation.invalid_image_score` | 0.0 | 当前必须为零；进程异常不能当成零分 |
| `evaluation.dev_timeout_seconds` | 28800 | 正整数秒；覆盖 preinit/dev 进程 |
| `evaluation.test_timeout_seconds` | 28800 | 正整数秒；最终 test 进程 |
| `selection.metric` | dev.mean | 当前固定 |
| `selection.tie_break` | baseline_then_earliest | 当前固定；平分优先 baseline，再优先较早轮次 |
| `selection.retrain` | false | 当前必须为 false；最终选择保存的 checkpoint |

### 12.4 configs/training/internvlu.yaml：低层模型设置

这些不是通常需要在换服务器时填写的字段。保留现有值即可开始兼容性验收；若修改，则需要外部 internvlu_train 支持，并验证其对当前 backend 的实际作用。下表覆盖模板字段分组；具体值以该模板为准。

| 位置 | 可指定内容 | 所有权／注意事项 |
|---|---|---|
| `name` | 配置名称 | 实验标识 |
| `model.snapshot` | `__MODEL_PATH__` | 由 profile.model_path_map 解析；不要写成与 profile 不同的权重 |
| `model.conv_template`、`force_image_size`、`vision_select_layer`、`ps_version`、`drop_path_rate`、`grad_checkpoint` | 模型会话模板、视觉输入与模型选项 | 与实际 snapshot 和库版本匹配；force_image_size=448 是视觉编码设置，不等于生成目标图的 512 尺寸 |
| `adapters.vit`、`mlp`、`llm`、`special_tokens`、`gen_decoder`、`vae` | 冻结/全参数/LoRA 策略 | 当前模板冻结 ViT/VAE，LLM 用 LoRA，其他指定模块训练；改变后需验证显存与保存/加载 |
| `adapters.llm_lora_rank`、`llm_lora_alpha`、`llm_lora_dropout` | LoRA 参数 | 当前 rank=32、alpha=null、dropout=0.05；具体 null 语义由外部库决定 |
| `data.train_meta` | `__TRAIN_META__` | 从 profile.resource_path_map.training_meta 自动填入 |
| `data.max_dynamic_patch`、`min_dynamic_patch`、`dynamic_image_size`、`use_thumbnail` | 图像 patch 预处理 | 当前真实 backend 的 item_builder 还显式传 max_dynamic_patch=1；不能假定只改 YAML 就完全改变行为 |
| `data.normalize_type`、`max_seq_length`、`pad_to_max_length`、`system_prompt`、`on_bad_row`、`num_workers` | 数据预处理选项 | 当前 backend 是自定义逐行训练循环；配置字段是否被调用路径消费需检查外部库 |
| `gen_image.size` | 由 task.image_size 解析 | 当前 512 |
| `gen_image.resize_mode`、`cond_min_pixels`、`cond_max_pixels`、`cond_max_length` | 生成条件预处理 | 需保持模型兼容 |
| `gen_image.vae_cond`、`allow_multi_vae_cond` | VAE 条件方式 | 当前 single/false；多输入帧不等于多张 VAE conditioning 图片 |
| `gen_image.cfg_dropout`、`cfg_dropout_all_frac`、`drop_vae_cond_on_all`、`drop_response` | 条件 dropout 和文本处理 | 训练路径会使用相关预处理设置；修改属于配方变更 |
| `loss.gen_weight`、`gen_warmup_steps`、`lm_weight` | task 占位符 | 从 task contract 解析 |
| `loss.vae_latents`、`cond_vae_latents` | latent 处理 | 当前 sample/mode；需外部库支持 |
| `optim.lr`、`gen_decoder_lr`、`warmup_ratio` | task 占位符 | 不要与 task 分别维护两套数值；warmup_ratio=warmup_steps/max_steps |
| `optim.weight_decay`、`betas`、`eps`、`max_grad_norm` | 优化器设置 | 当前 0.05、[0.9,0.999]、1e-8、1.0 |
| `optim.scheduler` | 模板中 cosine | 实际循环在 real_backend.py 中构造 warmup+cosine LambdaLR；调度能力变更需改实现 |
| `train.max_steps`、`global_batch_size`、`seed` | task 占位符 | 从 task 解析，epochs 固定 1 |
| `train.output_dir` | 每次操作的输出目录 | Host/worker 自动设置，不是手工指定 suite 路径的位置 |
| `train.per_device_batch_size`、`log_every` | 模板训练设置 | 实际循环逐行累积且每步记录；改字段不一定改变当前自定义循环 |
| `dist.backend`、`find_unused_parameters`、`static_graph` | 分布式设置 | 当前 nccl/true/false；使用指定 GPU worker 环境验证 |
| `checkpoint.*` | every_steps、keep_last、resume、export_at_end、export_dir | 当前 backend 自己保存 last-step trainable parameters，不按通用 trainer 的周期保存/恢复逻辑执行 |
| `eval.*` | sets、every_steps、intervals、at_start、max_rows、log_images、generation_mode 及 task 生成参数 | 当前 backend 显式执行独立 dev/test，使用固定 manifest，Predictor 指定 image 模式；不是通用训练中 eval 调度接口 |
| `tracking.*` | backends、project、run_name | 模板保留项；当前 backend 自己写 loss/cost/metrics，不能假定启用字段即可接入其他追踪服务 |

### 12.5 configs/runner.yaml 与 configs/docker.yaml

| 文件与字段 | 格式／当前值 | 填写规则 |
|---|---|---|
| runner `mode` | native / docker；native | native 不读取 Docker 文件 |
| runner `agent.executable` | codex 或绝对路径 | 当前 runner 按 Codex CLI 参数构造调用；不是任意 Agent 的通用适配接口 |
| runner `agent.model` | 字符串或 null | null 使用本机 CLI 设置；具体模型可用性由实际 CLI/账号决定 |
| runner `agent.reasoning_effort` | 字符串或 null | 由所选模型和 CLI 支持的值决定 |
| runner `timeout_seconds` | 可选正有限秒数 | 整个会话上限；仍受 campaign 剩余时间限制 |
| runner `cli_python` | 可选解释器选择器/路径 | CLI 环境，不是 profile.python GPU 环境；项目需事先准备 |
| runner `docker` | docker.yaml | 相对于 runner 文件解析，只有 Docker 模式使用 |
| docker `image` | 镜像 tag/digest | 替换占位符，准备含模型环境和 Agent CLI 的 GPU 镜像 |
| docker `cli_python` | /opt/video-ssl-method/.venv/bin/python | 容器内 CLI Python；不要在 runner 与 Docker 中重复声明同一配置项 |
| docker `mount_profile_resources` | 布尔；true | 自动挂载 profile 资源和评估 manifest 的父目录；关闭后自行保证全部容器资源可访问 |
| docker `read_only_mounts` | 路径列表；[] | 相对 Docker YAML 解析；必要时包含模型缓存 blobs 的父目录，不挂整个文件系统根目录 |
| docker `env` | 字符串映射 | 当前含 PYTHONUNBUFFERED 和 UV_PROJECT_ENVIRONMENT；后者指向镜像内预先准备的 CLI venv |
| docker `credential_files` | Host 文件 → 容器绝对文件路径；{} | Host 相对路径按 Docker YAML 解析；例如认证映射到 /tmp/video-ssl-home/.codex/auth.json |
| docker `pass_env` | 环境变量名列表；[] | 传名字，从 Host 取值；不要把秘密值写入 YAML |

Profile env 不能设置 `CUDA_VISIBLE_DEVICES`、`PYTHONPATH`、`VIDEO_SSL_RUN`、`VIDEO_SSL_AGENT`、`VIDEO_SSL_CURRENT_TASK`。Docker env 还不能设置 `HOME`，也不能与 profile.env 重复。GPU 可见性由 campaign launcher 控制，容器内按分配设备重新编号。

### 12.6 Campaign JSON 的全部人工字段

| 字段 | 格式／模板值 | 填写规则 |
|---|---|---|
| `deadline` | 带时区 ISO8601 字符串 | 例如日期后带 `+08:00` 或 `Z`；必须填写启动时仍有效的真实截止时间 |
| `report_reserve_seconds` | 整数；5400 | ≥1800；研究/train/dev 会保留这段时间；最终 test 的进程边界另保留至少 1800 秒 |
| `gpu_allocation` | 非负不重复整数列表；[0,1,2,3] | 与 profile 列表完全一致；长度 ≤ min(4,max_concurrent_gpus)，并整除 global batch |
| `max_concurrent_gpus` | 整数；4 | 1–4，是 GPU 数上限；当前共享 ledger 仍不允许多个 training attempts 同时 active |
| `training_budget` | 整数；7 | 1–7；所有引用该 kickoff 的 runs/tasks 共享 |
| `training_attempts` | 列表；新 kickoff 为 [] | 之后由框架维护；不要修改已登记状态来重置消费 |

`consumed`、`remaining`、各 attempt 的 identity/status/timestamp 等由框架写入，不是需要人工补齐的输入字段。

### 12.7 数据文件中每个字段填在哪里

| 文件 | 字段 | 具体填写要求 |
|---|---|---|
| train_sources.jsonl | `source_video_id` | 每行唯一非空字符串；Agent 在 source_video_id 中引用它 |
| 同上 | `video` | 原始视频文件路径；不同 source 不得别名指向同一规范化路径 |
| 同上 | `f0` / `final` | 首末帧 canonical PNG 文件路径 |
| 同上 | `frame_count` | ≥2 的整数；末帧索引=frame_count−1 |
| 同上 | `frames` | 可选的中间索引字符串→PNG 路径映射；不在这里登记首末帧 |
| 同上 | `description` | 原始任务描述；不得留 REPLACE 占位文本 |
| 同上 | `eval_text` | F0-to-final 冻结输入文本；baseline 原样使用，须含匹配输入图的占位符 |
| dev/test JSONL | `id` | 在各自 manifest 中唯一的非空字符串 |
| 同上 | `image` | 恰好一张输入图的路径列表；注意这里是单数键 image |
| 同上 | `target_image` | 对应末帧目标图路径 |
| 同上 | `cond_image` | 可省略，默认第一张输入；若填写必须与其一致 |
| 同上 | `conversations` | 冻结 human/gpt 对话列表；按模型输入协议填准确任务文本和生成目标标记，示例见 data/eval.example.jsonl |

以上文件内的图像/视频相对路径都按各自 manifest 的父目录解析。新数据的语义准备和 split 审核由 Host 完成，配置解析不会生成正确标签。

### 12.8 运行参数与自动生成文件

| 位置 | 可以指定什么 | 是否手填 |
|---|---|---|
| check/init/run `--prompt` | run declaration 文件；默认 run_prompt.md | 是，命令参数 |
| check/init/run `--profile` | 显式选择机器 profile，优先于 prompt 中的引用 | 可选；应记录本次覆盖 |
| init/run `--root`（别名 `--suite-dir`） | suite 输出目录 | 是，建议显式绝对路径 |
| init/run `--new-suite` | 明确创建新实验，不重置 campaign | 按需使用；不用于绕过既有状态 |
| 已有 suite 的 `--suite-dir`（别名 `--run-dir`） | 要操作的已有 suite | Host 手动操作时显式填写 |
| audit/submit `--path` | 当前构建的 dataset 目录 | Agent 使用 COMMANDS.md 生成的路径 |
| fail `--reason` | 放弃本轮的具体原因 | 按失败规则使用，不能丢弃在途操作 |
| resume `--inspect` | 只检查恢复条件，不启动 Agent | Host 排查时使用 |
| extend `--iterations` | 新的总轮数，含 baseline，不是追加数量 | Host 操作；须满足剩余预算且尚未开始最终 test |
| workspace/task.json | 公开的 sources、配方、帧策略和种子 | 自动生成，不手改 |
| workspace/METHOD.md、AGENTS.md、COMMANDS.md | 当前实验冻结的协议、范围和准确命令 | 自动生成，不手改；改源协议/runner 仅供新实验使用 |
| config/plan.json、backend.input.json、backend.json、training.resolved.yaml | 解析结果和完整性记录 | 自动生成，不另建人工配置副本 |
| suite_state.json、session_state.json、events.jsonl | 生命周期与工具历史 | 框架维护，不通过手改跳过阶段 |
| workspace/curation/、hypothesis.md、NOTES.md | 本轮构建代码、假设和研究笔记 | Agent 按协议维护；代码修改须能复现数据 |

## 13. 填写后的交叉核对

- [ ] run_prompt 的每个引用都能解析到预期文件，实际 profile 已替代 example。
- [ ] task 的 dataset/model/eval/resource ID 全部能在 profile 中找到对应 key。
- [ ] source manifest 路径按 dataset 根目录拼接，行内资源路径按 manifest 父目录解析。
- [ ] task 行数=训练步数×global batch；source 数足够；GPU 数整除 batch。
- [ ] dev/test 行数分别与 evaluation.samples/test_samples 精确匹配。
- [ ] profile 与 campaign 的 GPU 列表完全一致；全部待运行任务和验收训练合计不超预算。
- [ ] deadline 有时区且仍留足运行时间；模型加载和 preinit 时间也纳入安排。
- [ ] task 与训练模板不维护冲突数值；自动替换后的 resolved YAML 符合预期。
- [ ] native 的解释器/库/Agent 都在 Host 可用；Docker 的容器内路径、镜像与 mounts 已核实。
- [ ] `check` 成功后仍做实际 GPU 验收；不将静态通过等同于显存、模型或数据语义验证。

