# 实验设计、任务能力与执行基础设施

当前冻结依据为 **protocol v5 / design v3**。各轮重构均保留 protocol.yaml 的原字节内容、manifests、scorers、seeds、API 和原候选 fixture。训练配置与 recipe/template/probe 约束不变。

## 目录导航与依赖方向

```text
runner/state_machine.py          原 Runner / CLI 兼容入口
experiment/
  assembly.py                   build_experiment、RunPaths；选择实现并注入依赖
  application.py                已组装的应用，连接阶段与搜索 controller
  registry.py                   明确注册可信实现，不做自动发现
  contracts.py                  StagePlan、TaskAdapter、JobRequest 等能力接口
  workflow/                     engine、design、operations、selection、statistics
  improvement/                  search、policy、candidates、submissions、agents、prompts、checks
  evidence/                     service、authorized、access
  runtime/                      execution、slots、resources、store、io、journal、
                                artifacts、continuation、projections、provenance
tasks/reactdata/
  adapter.py                    ReactDataTask：job、测量、审计、pilot、报告
  contract.py                   候选文件、hash、检查器和任务资源
  prompts.py / evidence.py      任务指令与获准来源收集
  probes.py / gpu_probes.py      probe 额度/状态与实际 GPU 操作
runner/                         现有训练、沙箱、CLI 和 reactdata 基础设施
  training_journal.py            训练记录与恢复截断
  plugin_host.py / plugin_worker.py  原沙箱入口，挂载路径不变
tests/
  unit/                         CPU 行为、边界和包依赖回归测试
  integration/                  原流程断言与真实 Linux 沙箱测试
  support/                      numeric adapter、共享 fake、portable flow harness
  fixtures/                     原 candidate fixtures，内容不变
  test_v3_flow.py / test_sandbox.py / portable_flow.py  旧脚本的薄入口
```

推荐阅读顺序：`assembly.py → application.py → workflow/design.py → workflow/operations.py → tasks/reactdata/adapter.py`。需要理解候选修改时读 `improvement/search.py → candidates.py`；需要检查隔离时读 `evidence/service.py → tasks/reactdata/evidence.py → evidence/authorized.py`。

组装层可以依赖具体实现；workflow 只调用 `contracts.TaskAdapter` / `StagePlan` 能力与 runtime 工具，不导入 task 或 runner。improvement 负责候选产生和筛选调度，workflow/selection 负责统计接受规则；两者不再放在同一个 policies 文件。runtime 和 evidence 不导入任务实现。`improvement.policy.stages()` 仅为旧调用保留阶段表视图，实际阶段表由 workflow/design 定义。

公开入口是 `experiment.assembly.RunPaths / build_experiment`、`experiment.evidence` 的证据服务与数据对象、`tasks.reactdata.ReactDataTask`；具体组件从其职责子包导入。不建立旧内部模块的批量别名，也不增加插件框架。

`RunPaths` 集中传递资源路径，`runs` 是多个 run 的父目录，`protocol` 是文件。ReactDataContract 显式接收检查函数和资源路径，通用层不再读取 runner.common；仅 assembly 的默认路径兼容入口仍读取旧 common 常量。Claude 调用函数由 assembly 注入，LegacySelection 的统计能力由 task 注入。reactdata 的训练、审计、报告底层仍使用原 runner 基础设施及其默认路径，尚未把整个训练栈改为路径无关库。

实现哈希由 `runtime/provenance.py` 递归收集 runner/、experiment/、tasks/ 下的 Python 文件，包含子包与相对路径，排除 tests/。assembly 对正在执行的源码树求哈希，不把临时资源根误当作源码根。目录迁移会改变新实现 hash，已有 run.json 原始 hash 保留，后续实现写入 implementations.json。

ExperimentDesign 决定重复训练及 seed 安排；共享 operations 执行同一套统计实验。TaskAdapter 提供任务能力，不提供 baseline/confirmation/final 阶段 handlers。Engine 不理解视频帧、E/F、scorer、Claude 或 GPU。阶段操作是普通函数；有生命周期的会话、slot、writer 使用对象，没有引入工作流数据库、DAG 平台或插件发现。

`tests/support/synthetic_task.py` 的 NumericAdapter 只处理文本整数程序、数值测量、任务资源和格式。它使用 `query:a-17`、`urn:query/beta`，不使用视频 recipe；与 reactdata 共用 NoiseAwareDesign 和 run_baseline/run_confirmation/run_selection/run_final。它的确认现在使用真实配对统计，不再手写 CI。LegacyDesign 也有同一 numeric adapter 的完整 fake 流程测试。

## CLI、配置与兼容

从 pipeline/ 运行：

```bash
python -u runner/state_machine.py --run-id <id> --protocol protocol.yaml
python -u runner/state_machine.py --run-id smoke-new --smoke --protocol protocol.yaml
python -u runner/state_machine.py --run-id replay-new --replay-decisions <source> --protocol protocol.yaml
python -u runner/state_machine.py --run-id next-new --continue-from <finished-source> --protocol protocol.yaml
```

新协议副本可以显式指定：

```yaml
components:
  task: reactdata
  search_policy: default
  selection_policy: noise-aware
  agent_backend: claude
  executor: slots
```

不加这些字段仍选择当前默认实现；design=v3 使用 NoiseAwareDesign，其余支持的旧 design 使用 LegacyDesign。搜索策略不再承担实验阶段定义。未知工厂名报错，不使用 eval 或配置驱动的任意 Python 加载。

新增 adapter 实现 contracts.py 的任务能力，工厂在 registry.TASKS 明确注册，再在新协议里选择。构造参数为 protocol/paths/smoke/hooks。任务不重写共享统计阶段；recipe 和测量文件格式仍可不同。统计核接收显式 group_of，reactdata 保留历史 ID 分组语义，numeric 使用自己的分组规则。

新增 Agent/执行器/搜索与选择策略分别注册到 AGENT_BACKENDS、EXECUTORS、SEARCH_POLICIES、SELECTION_POLICIES。Agent 的调用选项、首次完整指令、短 prompt resume、repair、resume 失败 fallback 均保留；生产 Claude CLI 的沙箱工具、挂载和网络权限未扩大。

Python 入口为 `build_experiment(...)`，可注入 task、store、backend、executor、search_policy。旧 Runner 是兼容代理，保留 run/execute/event、state/root/evidence、常用报告属性及原模块级 call_agent/audit_run/run_checks patch 接口。路径用 RunPaths 传递，不再修改 reactdata 模块全局变量。原 runner/stats.py 是显式兼容 API，转发至同一统计核。

## 状态与恢复边界

RunStore.snapshot() 返回脱离实时状态的递归只读视图。候选、测量、确认、选择、冻结、会话等关键变化通过 commit() 在同一 writer lock 下更新并记录事件。兼容 Runner.state 仍暴露旧 dict，第三方代码不能因此假定任意直接写入都具备事务语义。

这不是跨文件 ACID 事务，也不是完整事件溯源：恢复仍依据 state.json、原 job 结果和具体操作回执。新字段均有缺省值，旧 run/state/job/candidate JSON 不作破坏式迁移。

| 记录 | 用途 |
|---|---|
| state.json / events.jsonl | 原阶段完成标记、候选/测量/协议决策和事件 |
| research_index.json | schema_version=1；proposal/memory/diff、测量/analysis/probes、检查/审计、确认/选择和 job 引用 |
| implementations.json | 新 run 及后续恢复使用的可信实现 hash 历史，保留旧 run.json 的原始 hash |
| rounds/.../evidence_request.json | 输入构建请求与当时披露能力 |
| agent_*/operation.json | 外部调用 started/completed 及完整规范化结果 |
| validation/<digest>/operation.json | 与文件内容和目录清单对应的 CPU 检查回执 |
| jobs/.../executor.json | slot/node/start/attempt、启动及 retry 状态 |

已完成 Agent 调用直接恢复结果和 session；若 CLI 已写 output.json 而协调层尚未记账，也可恢复。没有结果的 started 外部调用抛出 OutcomeUnknown，不自动再次调用付费 Agent，需要检查日志后明确处理。call_keys 避免候选成本重复记账；CPU 检查可以安全重做。检查回执包含目录清单，新增额外文件不能绕过检查；含链接/目录的提交不复用检查缓存。

candidate 发布使用临时目录和 rename，已发布程序在恢复时核对内容。DONE job 不重训，活动 job 按 executor.json 或旧 launch event + pid 挂接。slot 由 lease 独占管理并在异常时释放；基础设施 retry 次数仍为原协议配置。冻结 hash 不会因恢复被重新解释。

Windows 文件替换对短暂 sharing/permission 冲突有五次有界重试，超限仍抛出错误；不改变 Linux job retry 预算。此改动针对本地同步目录中重现的 WinError 5。

## 授权证据流

```text
完整持久化记录 → VisibilityPolicy → 任务收集器 → AuthorizedEvidence → renderer → /input
```

收集器读取获准来源；renderer 只接收 JSON/文本值和明确文件引用，不再接收完整 run_dir/entries。reactdata 仍校验实际 dev manifest，禁止将 test job 改名为 dev。API 和指令使用原白名单资源，scorer 只给文字说明。

默认历史结果、逐 item dev、baseline replicate、analysis/probes、图片选择和摘要截断不变。新协议可用 components.visibility.sections 缩小集合：api、instructions、train_examples、results、dev_items、analysis_traces、failures、candidates。关闭 results 会同时移除 comparison/cost/audit/error 等结果字段，不再只隐藏主分数。

冻结仅授予 test 执行许可，不授予搜索 Agent test 披露许可；slot executor 对恢复的 spec 也核验执行授权。smoke 仍使用 dev stand-in。禁止的 test、生成器 metadata、scorer 源码、凭据和隐藏目录不进入输入包。

使用现存 EvidenceService.rebuild(request_file) 可脱离 Claude session 重建输入。独立重建示例：

```python
from experiment.evidence import EvidenceService, VisibilityPolicy
from tasks.reactdata.evidence import ReactDataEvidence
from runner.report import write_report

service = EvidenceService(ReactDataEvidence(write_report, "/path/manifests/dev.jsonl"), VisibilityPolicy())
service.rebuild("/path/run/rounds/002/c002/evidence_request.json")
```

重建使用保存能力与当前能力的交集。原测量、候选和选定图片必须可用；小文件备份不能恢复已清理的图片。旧 combined renderer 接口保留兼容，新 adapter 应提供 collect_evidence/render_evidence。

研究索引分开 agent_authored、measurements、protocol_decisions，不将 Agent 假设解释为被证明的事实。原始/结构化证据、代码与 lineage、会话和 memory 四种载体均保留，不新增 LLM 总结调用或六文件提交要求。

## 训练侧

DynTrainer 从 284 行缩为约 187 行，保留显式 step 循环和原 optimizer/autocast/分布式同步顺序。tasks/reactdata/probes.py 管理请求校验、额度和 pending；tasks/reactdata/gpu_probes.py 管理 GPU probe 效果、RNG/model-mode 恢复；training_journal.py 管理逐步记录、恢复截断、flush 和 close。

checkpoint 仍在同一个训练 step 保存 plugin、pending、sched、cost、step，JSON 键名不变；组件不会分别独立提交 checkpoint。materializer、recipe/template 检查、插件 worker 内 Analysis.update → Builder.observe → request_probes 顺序不变。

## 扩展接口与资源注入（本轮）

- `contracts.py` 定义 RunPaths、PromptBundle、ReportContext、CheckResult、AgentResult、TaskStatistics，并补齐冻结接口；`assembly.RunPaths` 继续可导入。TypedDict 只描述现有 JSON，不引入新的序列化框架。
- assembly 将默认、自定义 executor 和 execute override 都包装为 AuthorizedExecutor。应用执行入口先调用 task.authorize_job，再进入后端；DONE job 也先检查。SlotExecutor 保留自身检查用于直接调用时的防护。授权边界覆盖应用调用，不试图约束持有底层后端引用的可信 Python 代码。
- selection 工厂显式接收 task.statistics；没有构造后 hasattr 注入。prompt 使用具名 PromptBundle，报告使用 ReportContext。
- RunPaths 传至证据 collector、检查模拟、Materializer、审计和最终报告。新 job spec 带可选 resources，训练入口据此读取配置、素材和 API 挂载来源。旧 spec 没有该字段时沿用旧路径，不改写旧 job。冻结 API 的加载语义、沙箱挂载目标和权限不变。
- 任务模块使用明确的 runner package imports。旧 CLI 常量和替换 hook 留在入口；显式自定义 hook 自行负责所需资源，不对第三方函数强加新参数。scorer_version 也可通过 hook 注入；外部模型、CV 库和 scorer 实现仍沿用固定安装。
- 图片 gallery 与 baseline 展示数据由 ReactDataEvidence.search_options 提供；其他任务可返回自己的证据选项。快照中的只读映射在任务边界转为可序列化值，原展示规则和输入格式不变。

最终验证：49 项 CPU 测试、原流程 15 项断言通过；首次 CPU 测试遇到一次 Windows 目录 rename 的 WinError 5，完整复跑通过。冻结资源和 fixtures 内容未变，统计核 AST 未变。

本轮增加了自定义 executor/override 授权、冻结程序篡改、双任务资源隔离、具名 prompt、API 挂载、旧 spec 保留和证据快照序列化测试。验证命令仍为 `python -m pytest tests/unit -q` 与 `python tests/portable_flow.py`。未运行 GPU 或真实 Agent。

本轮优先落实执行边界、路径和接口；保留策略仍了解 checkpoint 布局，workflow 仍有 v3/legacy 分支，RunStore 仍有兼容可变状态。独立 final 交接包与 final-only CLI 尚未实现，后续应按实际扩展需要单独处理。

## 检查与保留限制

```bash
python -m pip install -r tests/requirements-cpu.txt
python -m pytest tests/unit -q
python tests/portable_flow.py
# 在原 Linux 环境具备数据、bwrap 和外部依赖时：
python tests/test_v3_flow.py
python tests/test_sandbox.py
```

目录重构验证：42 项 CPU 测试（含包依赖、独立导入、递归实现哈希与沙箱挂载资源检查）、reactdata 原流程全部 15 项断言；包含共享设计、numeric 和 legacy fake 流程、Agent/confirmation/freeze 中断恢复、不可变快照、并发提交、披露限制、检查缓存失效、slot lease、probe 状态及日志恢复。统计核除 import 路径外的 AST 与本轮修改前完全一致；此前 aggregate/bootstrap/training_noise/confirmation/null_delta 的逐值对照也已通过。本轮逐字节核对 565 个冻结资源和已有产物文件、22 个迁移后的 candidate fixture 文件，均未改变；沙箱 worker/host、Agent transport 与 jobs 的源码字节也未改变。

portable flow 保留原断言，只在临时目录使用 manifest/图片副本；recipe/template/static 检查及 200-step 模拟是真的，host 是只加载可信 fixture 的进程内实现，训练、Agent、审计、CV provenance 和外部 prompt wrapper 使用 fake。它验证实际输入逐文件重建一致及隐藏数据未被复制，不能证明 Linux 沙箱运行时隔离或 GPU 数值等价。

未运行正式 GPU 实验、真实付费 Agent、集群操作或写入现有实验产物；Linux bwrap、原 scratch 数据和外部训练/CV 环境不可用，因此仍需原环境的沙箱和 GPU 集成验证。

保留的原语义差异：smoke 实际为 2×1 候选；memory.md 在检查器中 optional，原大小检查只遍历 required；protocol 的 50-step checkpoint override 优先；BACKUP_EXT 原本不含 .jsonl。统计设计、名额/停止阈值、repair/infra retry、去重、图片选择和摘要长度均未借重构修改。

公共层仍采用显式阶段、baseline c000、候选 status 和文件式产物。DynTrainer.fit 仍是较长的顺序算法；当前不把它改为通用 callback 框架。跨文件崩溃恢复依赖具体回执，没有提供任意任务程序的自动迁移或完整 ACID 保证。
