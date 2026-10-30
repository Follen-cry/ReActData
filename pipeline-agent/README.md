# pipeline-agent

单任务 `shape_color_then_scale`，静态 manifest，串行 trial，真实 InternVL-U 训练与独立 checkpoint 重载评测。研究 Agent 仅修改训练视频的选择及顺序。Markdown 协议是行为约定，不是 OS 隔离。原始远程要求见 [REMOTE_HANDOFF.md](REMOTE_HANDOFF.md) 与 [protocols/agent.md](protocols/agent.md)。

## 入口与边界

`cli.py → experiment.py → dataset.py / campaign.py → training.py → real_backend.py → internvlu_train`。
训练后保存最后一步 trainable parameters（冻结权重来自同一个 base snapshot），dev/test 用新进程加载 base 与保存参数；不以训练进程退出为成功依据。`real_score.py → analogy_grid.py → 外部 cvlib` 评分，模型/评分异常使 trial 失败；缺失、不可读、黑或平坦预测图按规则计零。

128 个不同视频、8 optimizer steps、global batch 16、每行只使用一次且不 shuffle；不足 16 个 GPU 时按行梯度累积。初始化/训练/构建/评测 seeds 为 42/1000/1000/0。训练 warmup 3、LoRA lr 2e-5、decoder lr 1e-4，20 diffusion steps，CFG 4.5/2.0。最后一步 checkpoint 用于推理，不支持训练中断恢复或隐式重试。

## 总预算与进程

真实配置必须提供 `campaign`（所有实验共享的现有 JSON 总账）及 `estimated_trial_seconds`。登记在数据校验之后，登记后失败也消耗机会；同一 request ID 不重跑。总账最多 7 次真实训练，最多 4 张 GPU，截止时间由宿主在 kickoff 时固定。准备时间、诊断、失败和重试均在总预算内；新目录不能重置总账。

`campaign.run_process` 用共享文件锁串行化 GPU 子进程、设置批准的 `CUDA_VISIBLE_DEVICES`、到期终止其进程组。训练与 dev 截止于总 deadline 前 90 分钟，test 截止于 deadline 前 30 分钟。GPU 必须由宿主确认可用后填写 `gpu_allocation`；此代码不代替集群调度器，也不防御同权限用户篡改账本。宿主不要换账本规避限额。异常中断后检查账本和活动进程，不自动重新提交。

## CPU 验收

```sh
python -m pytest tests -q
python tests/make_demo.py runs/demo-data
python cli.py --root runs/demo init --config runs/demo-data/demo.json
python runs/demo/workspace/build_dataset.py --config runs/demo/workspace/task.json --tools . --output runs/demo/workspace/dataset
python cli.py --root runs/demo run --dataset runs/demo/workspace/dataset --request-id baseline
python cli.py --root runs/demo finalize --trial t0001
```

simulation 明确返回 `simulated=true`、固定 synthetic_score 和 null checkpoint，不是真实结果。不要覆盖既有 run。

## 真实运行

使用具备 PyTorch/CUDA、InternVL-U 模型依赖和本目录 requirements 的 Python。参考 [configs/example.yaml](configs/example.yaml)，配置外部模型库、base snapshot、完整训练配置、冻结 dev/test JSONL、wrapper 与 cvlib。宿主准备并冻结每视频的完整 eval_text、原始视频、标准 512 PNG 端点、帧数和 split 审计。`external.test_manifest` 与 dev 必须分离。模型路径存在不代表加载成功。

```sh
python cli.py --root runs/new-experiment init --config /path/to/frozen-real-config.json
python runs/new-experiment/workspace/build_dataset.py --config runs/new-experiment/workspace/task.json --tools . --output runs/new-experiment/workspace/dataset
python cli.py --root runs/new-experiment validate --dataset runs/new-experiment/workspace/dataset
python cli.py --root runs/new-experiment run --dataset runs/new-experiment/workspace/dataset --request-id baseline
python cli.py --root runs/new-experiment status
# baseline 成功后启动直接 Codex；需要宿主在外层施加搜索时间限制。
python cli.py --root runs/new-experiment agent
# 停止搜索并按固定 dev 规则选定 checkpoint 后，宿主只运行一次：
python cli.py --root runs/new-experiment finalize --trial t0001
```

trial 产物包含 code/dataset 快照、config、rows、training_contract、train.log、train/last.pt 与 hash、loss、dev 图片/per_item/metrics、重载证明及显存/耗时。finalize 校验 trial 数据/代码/配置和 checkpoint hash；中断或失败后不自动重复 test。Agent 不得访问 test 或执行 finalize。

本次实施进展及真实验证范围见 [EXPERIMENT_OVERVIEW.md](runs/kickoff_20261029T163522Z/EXPERIMENT_OVERVIEW.md)。旧交接文件记录的是实施前状态，保留不改；完整验证以该报告和实际日志为准。
