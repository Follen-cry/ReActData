import json,pathlib,datetime,sys
p=pathlib.Path(__file__).resolve().parent; e=p/'experiment'
def read(f):return json.loads(f.read_text())
c=read(p/'campaign.json');s=read(e/'state.json');final=read(e/'final.json') if (e/'final.json').exists() else {};freeze=read(p/'freeze.json')
baseline=s['trials']['t0001'].get('result',{}).get('dev',{}).get('mean'); sel=final.get('trial_id'); test=final.get('test',{})
strategies={'t0001':'原始 seeded random baseline','t0002':'优先较大目标图形，选 128 后 seeded shuffle','t0003':'四种目标尺寸各选 32；每 batch 各 4 行'}
lines=[]
for a in c['training_attempts']:
 tid=a['trial_id']; r=s['trials'].get(tid,{});v=r.get('result',{});d=v.get('dev',{});mean=d.get('mean');delta=f'{mean-baseline:+.6f}' if mean is not None and baseline is not None else '未完成';elapsed=(datetime.datetime.fromisoformat(a['ended_at'])-datetime.datetime.fromisoformat(a['started_at'])).total_seconds() if a.get('ended_at') else None
 lines.append(f"| {tid} | {strategies.get(tid,'见 NOTES')} | {a['status']} | {mean if mean is not None else '未完成'} | {delta} | {d.get('n_invalid','未完成')}/{d.get('n','未完成')} | {elapsed:.1f} s | [结果](experiment/trials/{tid}/result.json) · [代码](experiment/trials/{tid}/code/build_dataset.py) · [数据](experiment/trials/{tid}/dataset/manifest.jsonl) |" if elapsed is not None else f'| {tid} | 进行中 | {a["status"]} | 未完成 | 未完成 | 未完成 | 未完成 | — |')
wall=c.get('elapsed_seconds',(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(c['started_at'].replace('Z','+00:00'))).total_seconds())
costs=[read(f) for f in (e/'trials').glob('*/*/cost.json')]
peak=max((max(x.get('peak_memory_per_rank',[0])) for x in costs),default=0)/2**30
gpu_hours=sum(x.get('wall_seconds',0)*x.get('world_size',0) for x in costs)/3600
better=sel is not None and s['trials'][sel]['result']['dev']['mean']>baseline
status='真实完整初步闭环已完成' if final.get('status')=='finalized' and test.get('simulated') is False else '真实流程尚未全部完成'
text=f'''# ReActData-mod pipeline-agent 首次真实端到端实验

## 1. 目标与完成情况

**{status}**。目标是验证单任务 F0→final 的数据策略搜索、固定训练/dev、checkpoint 选择与一次最终 test。baseline dev.mean={baseline}；是否在 dev 上优于 baseline：{'是' if better else '否；两轮改进均低于 baseline'}。所有正式指标来自真实 GPU 生成与规则评分；独立 CPU simulation 仅验证调度，固定 synthetic_score 不作为质量结果。没有多 seed 或显著性结论，也没有用 baseline 的 test 做额外比较。

最初执行曾在准备阶段提前停止，之后用户要求持续执行，本次沿用原始 kickoff 和预算恢复；未重置时钟或次数。原始不完整交付保留为 [历史记录](EXPERIMENT_OVERVIEW_initial_incomplete.md)。

## 2. 实际完整流程

```mermaid
flowchart TD
  RAW[原始视频和端点 PNG] --> PREP[prepare_data.py / audit_data.py: 宿主划分与审计]
  PREP --> FREEZE[freeze.json: 固定数据、配置、实现]
  FREEZE --> BUILDER[Agent / workspace build_dataset.py: 筛选和排序]
  BUILDER --> CLI[cli.py validate/run]
  CLI --> EXP[experiment.py: 校验、快照、状态]
  EXP --> BUDGET[campaign.py: 总账、GPU 锁、deadline]
  BUDGET --> ADAPTER[training.py: 行契约与进程调度]
  ADAPTER --> TRAIN[real_backend.py train / torchrun / internvlu_train]
  TRAIN --> CKPT[最后第 8 步 trainable checkpoint + SHA256]
  CKPT --> DEV[real_backend.py eval: 新进程加载 base 和 checkpoint]
  DEV --> SCORE[real_score.py / analogy_grid.py / cvlib]
  SCORE --> FEEDBACK[50 项 dev per_item 与 mean]
  FEEDBACK --> BUILDER
  FEEDBACK --> SELECT[宿主按 dev 选择并锁定]
  SELECT --> FINAL[cli.py finalize: 同一 checkpoint 一次 test]
  FINAL --> REPORT[结果、日志与本报告]
```

入口代码：[CLI](../../cli.py)、[调度](../../experiment.py)、[预算](../../campaign.py)、[训练适配](../../training.py)、[真实后端](../../real_backend.py)、[评分适配](../../real_score.py)。研究 Agent 不负责训练器和最终 test；宿主完成实现、固定协议与最终验收。

## 3. Agent 修改了什么

实施阶段：从空的真实后端接口接通 InternVL-U 静态 Row、ensure_gen_token、F0 条件图、固定梯度累积与种子；实现最后一步参数保存、哈希，以及独立进程重载 dev/test。每次从共同基础模型初始化，无 best-dev checkpoint、无重训 final。新增单任务 scorer adapter，原样迁入 analogy_grid.py，保留旧评分规则。来源与许可范围见 [provenance](implementation_provenance.md)；无旧 pipeline 的运行时 import。

预算方面新增全 kickoff 账本登记、文件锁、训练次数和时间预留检查，连接 CLI；受控 GPU 进程统一设备集合和串行锁，到期终止拥有的进程组。新增测试验证跨目录次数、失败耗次、预留和超时后代清理。更新 task 配置解析、example 和 README；不修改固定任务协议或旧 pipeline。

开始 HEAD `b649a73ae8ed0628921af75ebe1fce83aa011082`，初始工作树干净，所以本次 Git 变更均为本次实施产生。未提交或推送。见 [实现快照](implementation_snapshot/)、[变更补丁](implementation.patch)、[最终 Git 状态](git_status.txt)。

搜索阶段通过已认证的真实 Codex CLI 执行，协议只允许筛选与排序。完整证据、假设、预期、请求 ID、实际结果在 [NOTES](experiment/workspace/NOTES.md)；每轮代码和数据已快照，不能用当前 workspace 代表旧 trial。已审查 [Agent 的全部 14 条命令](agent_commands_review.txt)，没有打开 test 资源或执行 finalize；任务配置中可见 test 的路径，但未读取其内容。第一轮依据 baseline 最终单元得分 46/50 为零，优先大目标图形，预期更强视觉监督；实际效果见结果表。第二轮依据第一轮 E_cell 下降而 F_cell 改善的反馈，四种尺寸各选 32 个样本，并交错排列使每个 batch 各含 4 项；预期保留小目标监督并提升整体 dev.mean，实际是否达到预期见表和 NOTES。

## 4. 数据与设置

来源：`/scratch/network/ssd/junlin/vbvr_next_frame/raw/shape_color_then_scale` 的真实生成视频与原始 PNG。从 11,000 个实例目录用 split seed 20261029 排序后打乱，选取 train pool 160、dev 50、test 50；每 trial 使用 128 个不同训练实例。原始视频路径/字节 SHA256、端点 SHA256、固定文本和帧数分别在 [train](train_sources.jsonl)、[dev](dev_sources.jsonl)、[test](test_sources.jsonl)。不是重用旧 pipeline 的数据产物，但原始样本历史曝光未知，不能声称全新盲测。

[审计](data_audit.json)：260 个视频完整解码，帧数一致；原视频字节、规范化端点像素无重复，跨 split 生成参数 JSON 无完全重复。按原实例分组，没有帧跨 split。没有通用语义等价视频检测保证；上述是明确的已验证范围。标准 PNG 使用 OpenCV INTER_AREA 到 512，逐视频 eval_text 为外部 wrapper 结果并检查残留视频语言。wrapper 和外部依赖哈希见审计及 [外部源码清单](external_source_inventory.json)。

正式冻结时间 **{freeze['frozen_at']}**，记录在 [freeze.json](freeze.json)；[设计](experiment_design.md)、[解析训练配置](train.yaml)、[正式配置](experiment/config.json) 均可复核。后续 trial 不更换 dev/test、超参或评分器。128 rows / 8 steps / global batch 16 / 4 GPU / 每卡每步累积 4 行，恰好一次遍历且无二次 shuffle。[逐 trial 数据/步数审计](trial_data_audit.json) 与 [baseline 未修改验证](baseline_audit.json) 记录核验结果。初始化/训练/构建/评测 seed=42/1000/1000/0，warmup=3，LoRA lr=2e-5，decoder lr=1e-4，gen loss 权重=.5，lm=0，20 diffusion steps，CFG=4.5/2.0。

评分为原 analogy_grid v1.0 scale，0.35 E_cell + 0.40 F_cell + 0.25 F_transform 并应用原 gates；按 50 项算术平均，无效图计零。模型/评测异常是失败而不是低分成功。[训练集 scorer 对照](scorer_selfcheck.json) 20 项 GT>=.95、复制 F0<=.05，未用 test 调整评分。

## 5. 实验结果

下表包含本 kickoff 的所有真实训练尝试（诊断/失败如有亦纳入）；没有独立短训练诊断。CPU pytest 夹具和 simulation 不含真实优化器更新，不计次数。

| Trial | 数据策略 | 状态 | dev.mean | baseline 差值 | invalid/n | 登记至完成耗时 | 产物 |
|---|---|---|---:|---:|---|---:|---|
{chr(10).join(lines)}

最终选定：**{sel or '未完成'}**。严格按已完成正式 trial 的最高 dev.mean，同分先 baseline 再更早 trial；依据与 checkpoint hash 见 [selection.json](selection.json)。最终 test **{'已完成一次' if test else '未完成'}**，mean={test.get('mean','未完成')}，n={test.get('n','未完成')}，invalid={test.get('n_invalid','未完成')}。[final.json](experiment/final.json) 保存完整结果；[选定 checkpoint](experiment/trials/{sel}/train/last.pt)、[test 指标](experiment/trials/{sel}/test/metrics.json)、[test 逐项结果](experiment/trials/{sel}/test/per_item.jsonl)、[test 图片](experiment/trials/{sel}/test/images/) 可直接核查。未重训、未换 seed、未依据 test 重新选择。单 seed 初步结果不支持统计显著性或广泛泛化结论。

## 6. 资源与预算

kickoff={c['started_at']}，不可重置 deadline={c['deadline']}；本次墙钟 **{wall/60:.2f} 分钟**，包括此前暂停、准备、开发、训练、评测和报告。截至账本真实训练次数 **{c['consumed']}/7**，剩余 {c['remaining']}，峰值并行 GPU=4，单卡 PyTorch allocated 峰值 **{peak:.2f} GiB**；按后端进程耗时乘 GPU 数估计约 **{gpu_hours:.3f} GPU 小时**（不含全部启动/hash 开销，不等同计费时长）。GPU 0–3 为初检空闲 A40（约 46 GB），无调度器分配信息；宿主确认可用后使用，未使用 GPU 4–7、未停止其他任务。每个训练/dev/test 都串行使用同一设备集合。每 trial 的 train/dev/cost.json 记录各 rank 峰值显存和进程耗时，登记耗时还包含启动、hash 和落盘开销。[总账](campaign.json) 为次数/状态的权威记录。默认保留最后 90 分钟给 test/报告，test 自身还保留最后 30 分钟给报告；此次提前收尾，无需耗尽次数。

预算与 Markdown 约束适用于可信同账号 Agent，不是对恶意篡改账本的 OS 安全隔离。watchdog 已通过后代进程终止测试，实际 GPU 后端均经该入口；没有故意使真实 GPU 训练超时来额外耗费训练机会。

## 7. 复现与接手

环境见 [environment.json](environment.json)，Python 为 `/homes/55/junlin/miniconda3/envs/internvlu/bin/python`。模型库及其实际源码 hash 见 [external_source_inventory.json](external_source_inventory.json)，base snapshot f012d760，21 个模型资源文件 hash 见 [model_weights_inventory.json](model_weights_inventory.json)。checkpoint 保存 trainable 参数，必须结合相同 base 和实现重载；不含完整优化器状态，不支持中途训练 resume。baseline 的 [checkpoint 验收](experiment/trials/t0001/train/checkpoint_acceptance.json) 记录 196 个 LoRA B 张量全部非零，日志记录 8 次有限梯度更新；各 dev/reload.json 证明新进程加载。

实际执行过的命令、配置与搜索启动日志见 [commands](commands.md)、[baseline CLI](baseline_cli.log)、[Agent CLI](agent_cli.log)、[Agent 日志目录](experiment/agent_logs/)、[CPU 模拟日志](cpu.log)、[CPU 测试](pytest_real_backend.log)、[watchdog 测试](watchdog_tests.log)。真实 trial 日志位于 `experiment/trials/<id>/train.log`、`dev.log`；最终 test 的日志/图片/逐项结果在选定 trial 的 `test.log`、`test/`。所有 test 均仅加载已有 checkpoint。

剩余限制：单 seed、小训练预算、同分布小评测集，历史原始数据曝光未知；没有跨机器或重新独立全量复现实验。直接 Codex 的 test 限制依赖协议与日志审计，不是文件权限隔离。下一步应在另行授权的新预算下验证更多随机种子和独立未曝光测试集；本 campaign 不追加 test、不更换已选 checkpoint。
'''
(p/'EXPERIMENT_OVERVIEW.md').write_text(text)
(p/'REPORT.md').write_text(text)
print(status)
