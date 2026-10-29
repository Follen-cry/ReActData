# pipeline-agent

远程首次实验请先读 [REMOTE_HANDOFF.md](REMOTE_HANDOFF.md)，启动 prompt 见 [REMOTE_KICKOFF.md](REMOTE_KICKOFF.md)。
本次远程任务要求最多4张GPU、从kickoff开始8小时内、合计7次真实训练机会；GPU与总deadline约束当前尚未实现，交接文档要求远程agent在真实训练前落实。短训练诊断/失败也占机会。
远程agent可在首次正式baseline前自主调整并冻结数据/dev/test设计；冻结后搜索遵守统一契约，不能根据反馈更换评测集。

单任务、单 Agent、串行 trials。默认任务为 **shape_color_then_scale**，F0 → final。
Agent 直接运行 Codex，按 Markdown 约束行为；不提供 OS/test 权限隔离。
真实训练/dev/test 后端尚未接入，不会默认成功或自动退回 simulation。

## 结构与调用链

```text
cli.py → experiment.py → task.py (解析固定任务)
                      → dataset.py (校验与来源检查)
                      → training.py (静态 Row 契约；真实后端待接入)
                      → storage.py (状态、快照、锁、hash)
agent.py → Codex CLI → workspace/build_dataset.py → CLI run/status
starter/build_dataset.py → video.py
 tasks/reactdata.yaml       固定实验参数
 protocols/agent.md         Agent 工作规则
 tests/                    CPU 测试与小型 demo
```

参考 v6 的实际 protocol + job overrides，而非直接复制旧 train.yaml：固定 128 rows、8 steps、global batch 16、warmup 3、LoRA lr 2e-5、gen decoder lr 1e-4、生成 loss 权重 0.5、lm loss 0、无生成 warmup。manifest 顺序即训练顺序，不重新 shuffle，每个 trial 从相同基础权重初始化。
初始化 seed 42，训练随机 seed 1000，构建 seed 1000，评测 seed 0。生成使用 20 diffusion steps、CFG 4.5/2.0。tasks/reactdata.yaml 是新流程的独立契约；旧 pipeline 无 import 或运行时依赖。

v6 是三任务实验；本版本只取 scale，不沿用三任务均值、动态 Builder/Analysis、probes、多 seed 或显著性选择。建议单任务 pool 160，dev/test 各固定 50。128 样本预算迁到单任务后需重新验证效果。

## 数据接口与宿主准备

每个 sources 条目包含 video、f0、final、frame_count、description、eval_text；路径相对用户配置解析。
宿主提供不重叠的 train/dev/test 划分、来源帧数、原始描述以及完整 evaluation-format 文本。
eval_text 应与评测输入完全一致，包括需要的 image token，不能用全局通用文本代替。
v6 的 wrapper 在外部 VBVR-DataGeneration 项目中，本仓库没有其实现。external.prompt_wrapper 记录该依赖，**当前不自动执行 wrapper，也不验证其输出语义**；宿主应先生成并核对每个来源的 eval_text。

canonical F0/final 使用 512×512 PNG。v6 优先使用原始端点 PNG，不能用有损 MP4 解码后端点冒充等价输入。
video.prepare_png(source, output) 提供 OpenCV INTER_AREA 的标准化工具；video.endpoints(video) 支持帧提取（GIF CPU 已测，MP4 待实测）。正式训练使用宿主准备的 PNG，source_endpoints 读取标准 PNG。frame_count 由宿主提供，当前不重新解码视频核实。

```json
{"sample_id":"0","source_video_id":"scale_000","input_images":["images/0_f0.png"],"target_image":"images/0_final.png","text":"<image>\n完整 evaluation-format 描述","input_frames":[0],"target_frame":59}
```

正式数据必须恰好 128 行；sample_id 唯一、source_video_id 不重复。init 拒绝多个 source ID 指向同一个视频路径。当前不按视频内容去重，宿主需避免复制视频以不同路径重复列入来源。
校验 schema、相对路径/越界/symlink、图片可读、预算、固定文本、帧号、512 PNG 来源像素一致性和 init 时的 canonical PNG hash。原视频外部引用；trial 复制训练 PNG/manifest，后续 workspace 修改不改变已有训练输入。
Agent 只修改视频筛选与排序；baseline 对排序后的源 ID 用 Random("1000:subset") 无放回抽样，再打乱。
代码与笔记归档上限 5 MB，生成数据放 dataset/，不进入代码归档。hash 用于发现意外变化，不抵抗同账号修改 hash 或校验代码。

## 已实现与待验证

- CPU 可执行：配置解析、独立 seeds、固定单任务、标准数据校验、baseline、快照、预算、幂等、静态 Row JSON、模拟多轮和报告、Codex 调用边界。
- 已实现待 compute node 验证：OpenCV 标准化/MP4 解码、真实 Codex 认证/启动/resume。
- 未实现：模型加载、GPU 训练、checkpoint 保存重载、真实 dev/test 和评分 adapter。v6 scorer 依赖外部 cvlib；external.scorer_cvlib 记录路径，尚未接入。

旧骨架配置不再兼容：旧 seed/max_samples/text/source字符串结构会明确报错。请用新配置创建新实验，不覆盖历史状态或产物。

## 阶段 1：本地 CPU

从 pipeline-agent/ 执行，Python >=3.10：
```sh
python -m pip install -r requirements.txt
python tests/make_demo.py demo
python cli.py --root runs/demo init --config demo/demo.json
python cli.py --root runs/demo task
python runs/demo/workspace/build_dataset.py --config runs/demo/workspace/task.json --tools . --output runs/demo/workspace/dataset
python cli.py --root runs/demo validate --dataset runs/demo/workspace/dataset
python cli.py --root runs/demo run --dataset runs/demo/workspace/dataset --request-id baseline
python cli.py --root runs/demo run --dataset runs/demo/workspace/dataset --request-id second
python cli.py --root runs/demo status
python cli.py --root runs/demo finalize --trial t0001
python -m pytest tests -q
```

demo 明确 simulation_samples=1；正式配置禁止 simulation overrides。synthetic_score 固定 0.5，不表示质量，checkpoint 为 null。检查各 trial 的 code/dataset/config/integrity/result、training_contract.json、rows.json、simulation.json；final.json/report.md 标注 simulated。查看 result.json 排查失败。重建时先移走/删除 workspace/dataset，不能改已有 trials。

## 阶段 2：compute node 单次真实实验（接入前不可成功）

配置独立模型库/权重/训练配置、至少 128 个来源、50 项 dev manifest，并核对数据划分与文本。
```sh
python cli.py --root runs/real-smoke init --config configs/real.yaml
python runs/real-smoke/workspace/build_dataset.py --config runs/real-smoke/workspace/task.json --tools . --output runs/real-smoke/workspace/dataset
python cli.py --root runs/real-smoke run --dataset runs/real-smoke/workspace/dataset --request-id baseline
```

当前会记录 REAL_BACKEND_NOT_IMPLEMENTED，消耗一次机会。不会启动 GPU。
接入 training.py：
1. rows.json 转模型库 Row，接入 ensure_gen_token；images=F0、cond_image=F0、target_image=final。顺序不变。
2. 解析外部训练配置，显式覆盖并核验 training_contract.json 全部参数；不能让旧配置 100 steps/warmup 6 覆盖固定 8/3。记录完整 resolved 配置与模型库/权重版本、hash。
3. 每次从共同基础权重初始化；8 steps 后保存 last checkpoint（即使定期保存间隔大于8），检查可重新加载。不能选择 best-dev checkpoint。
4. 使用固定 dev split/seed/scorer，记录每项分数、子分数、错误类别、生成图像、均值、loss、耗时、显存。无效生成图计0；评测程序失败是 trial failed，不能当低分完成。
5. 在 compute node 实现独立的 backend smoke：少量数据、短训练与少量 dev。**这个诊断命令尚未实现**，不能拿改变128/8/50契约的 smoke 当正式 trial。先完成此诊断，再运行上面的正式 CLI。

验收：checkpoint 重载成功，预测图片可读，指标来自真实评分。排查 training.log/dev 日志、配置差异、显存。当前尚不会生成真实训练日志。

## 阶段 3：真实 baseline

用正式配置新建 runs/baseline，执行相同 init/build/run，首次使用未修改 starter。
默认7次机会含1 baseline + 6改进；失败提交也消耗机会。协议要求 baseline 失败后停下排查。
检查训练恰好128不同视频、一次遍历、8 steps，保存数据顺序、配置、种子、耗时/峰值显存、dev结果和checkpoint。
完成前不进入真实 Agent 搜索。

## 阶段 4：Agent 闭环

Linux 安装并认证 Codex，检查 codex exec --help。继承用户配置/认证/环境。
```sh
python cli.py --root runs/baseline agent
python cli.py --root runs/baseline agent --resume SESSION_ID
# 可选 --executable /opt/bin/codex --model YOUR_MODEL
```

按需启动或继续，不同时启动多个 Agent。直接传入 --dangerously-bypass-approvals-and-sandbox，无 Docker/broker；协议限制不构成权限限制。
每次启动更新 workspace/protocols/agent.md，生成含绝对路径的 COMMANDS.md，前台等待，保存 agent_logs/<id>/launch.json、events.jsonl、stderr.log；返回 session_id 用于显式 resume。
参考 [Codex exec](https://learn.chatgpt.com/docs/developer-commands#codex-exec)。completed 只表示 Codex 正常退出，不代表训练成功。

Agent 自己维护 NOTES.md：证据→假设→筛选/排序变化→预期→实际结果。读取 dev per-item 反馈；禁止改训练配置、文本、图像及历史产物。缺依赖则停止，不切 simulation。
验收两轮真实改进、数据/trial映射、预算、日志和不覆盖历史；本地只验证调用替身。认证失败看 stderr.log/events.jsonl；Ctrl-C 后检查遗留进程，不保证后代清理。

## 阶段 5：最终验收

Agent 在 selection.txt 写 completed trial ID。默认最高 dev.mean；完全相同优先 baseline，否则较早 trial。模拟分数相同优先 baseline，不得据此宣称质量。这个选择规则由协议约束，宿主可显式指定另一个有效 trial。
停止 Agent，宿主执行：
```sh
python cli.py --root runs/baseline finalize --trial t0001
```

当前仅 simulation 可完成。training.test 后续必须重载选定 trial 的现有 last checkpoint，核验 hash，对独立50项test评测一次，不重新训练/换seed。报告记录checkpoint、数据/配置hash、dev/test指标。不得将test反馈用于继续搜索。
test路径和报告与搜索分开只是约定，当前同账号Agent可以访问；不承诺强制隔离。

## 失败语义

run 自行校验；校验失败不占机会，登记后的准备/训练失败占机会。相同 request ID 返回原状态，不重新训练。状态 preparing→running→completed/failed。没有自动训练重试/故障恢复。
独占锁阻止并行run；崩溃可能留锁，宿主确认无活跃进程后人工处理。finalize锁定后无报告则拒绝自动重试，以免重复test。CPU测试覆盖模拟流程，不证明GPU数值、真实恢复或test保密。
