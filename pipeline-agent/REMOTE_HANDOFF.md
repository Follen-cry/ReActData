# Remote handoff: complete the first real pipeline-agent experiment

## Objective and authority

在 Linux compute node 上完成必要接入并运行一次初步、真实、可复查的完整流程：
环境检查 → CPU simulation → 真实训练/checkpoint/dev 接入与验收 → baseline → Agent 数据策略改进 → 锁定 checkpoint → 一次最终 test → 报告。
目标是交付实际代码与结果，不是只写计划，也不要求一定超过 baseline。不得将 simulation、伪造 checkpoint 或占位分数描述为真实验证。

你是本次远程实施负责人，可以编辑 pipeline-agent、配置本次环境路径、实现缺失后端、准备数据、启动真实实验并完成最终评测。先阅读当前仓库适用的 AGENTS.md、README.md、tasks/reactdata.yaml、protocols/agent.md 和本文件。可以参考旧 pipeline-v6 的实现，但新流程不得运行时依赖旧 pipeline 的模块、配置或产物；必要实现迁入新目录并记录来源与许可证。不要修改旧 pipeline、冻结协议或已有实验产物；不提交、不推送。

当前采用最简单的可信 Agent 模式：直接执行 Codex，行为通过 Markdown 约束，不要求 Docker、独立 UID、broker 或强制 test 隔离。实施负责人可以准备 test 清单，但搜索期间不得查看 test 指标、用 test 调参或将 test 信息传给搜索 Agent。

## Hard budgets — implement before the first real training

- 同时最多使用 **4 张 GPU**，整个流程共享额度，包括训练、dev、诊断和 test。优先使用调度器分配的 GPU；不要占用他人 GPU或结束他人进程。只需1–4张，global batch 仍保持16，必要时用梯度累积。
- **总墙钟时间不超过8小时**：从本次 kickoff 开始计时，包含检查、开发、安装、数据准备、排队、运行、评测和报告。记录 UTC started_at/deadline，不因重启、换实验目录或失败而重置。
- **最多7次真实训练机会**，是整个 kickoff 的总预算，不只是某个 experiment 的计数。失败、OOM、人工重试、新目录重跑均不能绕过预算。
- 正常目标为1次 baseline + 最多6次改进。独立的 GPU 短训练/smoke 若执行优化器更新，也消耗一次机会；例如1次smoke + 1次baseline后，只剩5次改进。CPU simulation和纯推理/重载检查不占训练机会，但计入8小时。
- 沿用提交语义：正式run校验失败不计数；通过校验登记后，即使准备或训练失败也消耗一次机会。独立诊断在启动前登记。已登记正式trial不能因失败删除后重建。
- dev及最终test不另计训练机会，因为只加载已保存checkpoint。禁止通过重新训练完成final test。
- 7次是上限，不要求耗尽；必须优先留时间完成最终test与报告。若无有效checkpoint，报告失败原因，不伪造完整结果。

在运行目录保存一个宿主负责的总账，例如 campaign.json：开始时间、绝对截止时间、分配GPU、所有训练尝试及其trial/诊断ID、状态、消耗数和剩余额度。把该总账与CLI预算联通；不能只写在prompt里。使用简单文件锁/单进程调度即可，不建新框架。
总deadline需要覆盖子进程：启动前检查，运行中超时终止本次拥有的进程组（含torchrun子进程），记录中断。所有GPU子进程使用同一批准的设备集合，串行调度。不得只依赖Agent记得停。

建议初始时间分配（根据实测可调整，但不可突破总deadline）：
0–1.5h 环境/接入/CPU验收；1.5–3h smoke及baseline；3–6h 改进；6–7.5h final test；7.5–8h 收尾。
在baseline测得训练/dev耗时后更新预测。新trial启动前必须留出该trial预计时间、完整test时间和报告缓冲；默认至少预留最后90分钟。时间不足就提前锁定已有有效trial。

## Current state — do not mistake interfaces for implemented backends

CPU tests were last verified locally: 19 passed. Re-run on Linux; do not assume this proves GPU behavior.
training.py 的 train_and_dev/test 在real模式明确抛出未实现错误。rows.json与training_contract.json只是输入契约，不是训练结果。模型加载、checkpoint、真实scorer/dev/test需要你接入。
agent.py 已有直接 Codex exec/resume、日志和参数；真实认证与Linux启动仍需验证。不应在 backend 未完成时先启动长时间搜索。

重要参考位置（只读）：
- pipeline-v6/protocol.yaml：实际8 steps、warmup3、学习率等。
- pipeline-v6/configs/train.yaml、runner/train_entry.py：外部模型初始化、训练、Predictor、checkpoint；不要误沿用旧默认100 steps/warmup6。
- pipeline-v6/runner/materialize.py：Row、ensure_gen_token、cond_image=F0。
- pipeline-v6/runner/build_data.py：标准端点PNG、INTER_AREA 512处理。
- pipeline-v6/runner/common.py：外部prompt wrapper入口。
- pipeline-v6/scorers/analogy_grid.py及runner/score.py：评分、invalid处理；cvlib在外部路径。
- pipeline-v6/seeds/fixed_f0_final/databuilder.py：可复现抽样。

## Phase 1 — inspect, design and freeze the experiment

1. 记录仓库版本、工作树修改、Python/CUDA/驱动/模型库版本、GPU分配与可用显存、Codex可执行文件及认证状态；不要输出认证密钥。优先现有环境与权重，不做无关系统升级。
2. 确认可用模型权重和外部依赖。缺失的关键资源若无法在时间内取得，保存具体阻塞与复现命令，不以simulation替代结果。
3. 你可以自主设计本次数据使用、dev和test。默认单任务shape_color_then_scale，训练池160，dev50，test50；每个正式trial从训练池选128个不同视频，只使用一次。可根据资源和时间在首次正式baseline前调整池规模、dev/test样本数，以及必要时的每次训练样本数/steps；必须说明理由、保持global_batch×steps与样本数匹配，并同步代码校验、配置和测试。不要随意扩成多任务或修改scorer来提升分数。
4. 数据划分按原视频/生成实例分组，无重叠，无重复视频别名或内容副本泄漏。不得把同一视频不同帧放入不同split。记录来源、split seed、ID、分组依据、文件hash及检查结果。不要为了填满50项复制样本；不足时明确缩小固定评测集。
5. 优先真实训练分布中未用于调参的test。已有v6 test有历史使用记录，不得声称全新盲测；如复用要披露。最终test不能从看过dev结果的样本中挑选。
6. 准备标准F0/final PNG与逐视频固定evaluation-format文本，记录wrapper/预处理版本。核实frame_count；当前校验器相信宿主提供的帧数和文本，不会替你证明其语义正确。
7. 将决策保存到新run目录的 experiment_design.md、data_audit.json 和解析配置。冻结后所有baseline/改进使用相同训练、dev、scorer与种子，不根据反馈更换dev/test或调训练超参。

如果沿用默认，model_init_seed=42、train_seed=1000、build_seed=1000、eval_seed=0；128 rows、8 steps、global batch16、warmup3、LoRA lr2e-5、gen-decoder lr1e-4、20 diffusion steps、CFG4.5/2.0。数据选择和顺序是搜索变量，不能训练器再次shuffle。预训练权重版本变化须披露，不能声称精确复现v6。

## Phase 2 — CPU flow, backend and small real validation

先在独立simulation目录运行README阶段1与pytest。核验格式错误、重复源、固定文本、快照、预算、状态和幂等。不要把模拟结果混进真实报告。

实现training.py真实adapter及必要的浅层辅助模块：
- 静态manifest → 模型Row/ensure_gen_token；只消费冻结数据，不迁入动态Builder/Analysis。
- 每trial从共同基础权重初始化；固定训练契约和种子；GPU不足用明确的梯度累积保持global batch。
- 保存最后一步checkpoint，即使步数小于周期保存间隔；记录路径/hash并验证可重新加载。不能选best-dev checkpoint。
- 固定dev生成与scorer，保留预测、per_item.jsonl、metrics.json、loss、耗时和峰值显存。
- 成功必须有可用checkpoint和完整评测，不以进程exit=0作为唯一依据。未读/缺失生成图按规则计零；模型/评测程序异常记录failed，不能伪装为低分成功。
- 结果至少包含simulated=false、checkpoint及hash、dev.mean、n、n_invalid、逐项结果路径、配置/数据hash、cost与日志路径。
- 接入finalize的独立test配置与报告，读取选定checkpoint；将checkpoint完整性也纳入核验。

可先做纯模型加载/推理/重载检查（不占训练机会）。若必须做短训练诊断，提前登记并消耗一次机会，不把短训练结果当正式baseline候选。
诊断通过后新建正式实验。当前README中部分诊断命令尚不存在，由你实现并在交接报告写出真实可执行命令。

## Phase 3 — real baseline

用未修改baseline builder构建固定随机子集和顺序，执行真实run与dev。记录日志、checkpoint、数据、训练配置/代码hash和时间。
对照合同检查steps、唯一视频数、批量、种子、无二次shuffle；抽查生成图、scorer子分数、invalid比例。
检验评分器：GT应接近满分、复制F0应低分；若与v6语义不符先排查，不为得到更好结果改变规则。
确认baseline有效后，将其只读意义上的反馈路径提供给搜索Agent。这里只读是协议约定，不是权限机制。

## Phase 4 — autonomous improvement loop

优先启动当前Codex CLI；读取protocols/agent.md、COMMANDS.md与baseline反馈，自主管理上下文和NOTES.md。
每轮：证据→一个假设→修改视频筛选/排序→构建→validate/run→固定训练/dev→读取反馈并记录结论。
不改文本、目标帧、训练超参、scorer或历史产物；不用test；不要引入top-k、多seed、显著性筛选或动态训练反馈机制。
优先争取至少两个真实改进trial；若预算/时间不足，保留已完成结果并解释不足，完整收尾比耗尽7次更重要。
如Codex认证不可用，可由当前实施coding agent按相同协议和CLI完成搜索，不再创建嵌套Agent；明确报告实际执行方式，不冒充独立Codex调用已验证。

## Phase 5 — select, test and report

根据固定dev.mean选择最高的正式completed trial；完全同分优先baseline，再选更早trial。短训练诊断不具备候选资格。
将理由、trial ID、checkpoint路径/hash写入selection，停止搜索；随后由你作为宿主实施负责人运行finalize。
最终test只对已选checkpoint执行一次，不重训，不换seed，不根据test重新选trial。finalize失败先判断是否已运行，禁止盲目重试。

最终交付至少包括：
- REPORT.md：初步结论、实际完成程度、每trial的数据策略/dev均值/invalid/耗时/显存/状态、baseline差值、最终test；单seed不宣称显著性。
- experiment_design.md、data_audit.json、campaign.json、resolved配置/版本/hash。
- baseline与改进的代码/数据快照、真实checkpoint、dev预测/逐项指标；最终选择与test产物。
- 完整可复现命令、环境路径、修改文件列表、CPU/GPU/Agent各自验证结果和剩余限制。
- 实际总墙钟、使用GPU峰值、训练机会消耗（包括诊断/失败），解释任何未完成阶段。

所有新产物放在pipeline-agent/runs/<campaign_id>/或其明确记录的外部大文件目录。已有产物不覆盖。
最终向用户提供报告和关键产物的绝对路径。即使失败也在8h截止前写出失败阶段、已完成验证及下一条可执行行动，不静默挂起或无限重试。
