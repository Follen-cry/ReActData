# 实际执行命令与复现入口

以下路径均以仓库根 `/scratch/network/ssd2/junlin/ReActData-mod` 为工作目录（CPU README 流程当时从 pipeline-agent 子目录执行，日志为证据）。真实运行使用现有 internvlu 环境。下列命令在本次已实际执行；不要对既有目录重新 init 或重新运行准备脚本，也不要重置 campaign。

```sh
date -u '+%Y-%m-%dT%H:%M:%SZ'
git rev-parse HEAD
git status --short
nvidia-smi
/homes/55/junlin/miniconda3/envs/internvlu/bin/python -m pytest pipeline-agent/tests -q
/homes/55/junlin/miniconda3/envs/internvlu/bin/python pipeline-agent/runs/kickoff_20261029T163522Z/prepare_data.py
/homes/55/junlin/miniconda3/envs/internvlu/bin/python pipeline-agent/runs/kickoff_20261029T163522Z/audit_data.py
/homes/55/junlin/miniconda3/envs/internvlu/bin/python pipeline-agent/runs/kickoff_20261029T163522Z/configure_real.py
/homes/55/junlin/miniconda3/envs/internvlu/bin/python pipeline-agent/runs/kickoff_20261029T163522Z/freeze.py
/homes/55/junlin/miniconda3/envs/internvlu/bin/python pipeline-agent/cli.py --root pipeline-agent/runs/kickoff_20261029T163522Z/experiment init --config pipeline-agent/runs/kickoff_20261029T163522Z/real_config.json
/homes/55/junlin/miniconda3/envs/internvlu/bin/python pipeline-agent/runs/kickoff_20261029T163522Z/experiment/workspace/build_dataset.py --config pipeline-agent/runs/kickoff_20261029T163522Z/experiment/workspace/task.json --tools pipeline-agent --output pipeline-agent/runs/kickoff_20261029T163522Z/experiment/workspace/dataset
/homes/55/junlin/miniconda3/envs/internvlu/bin/python pipeline-agent/cli.py --root pipeline-agent/runs/kickoff_20261029T163522Z/experiment validate --dataset pipeline-agent/runs/kickoff_20261029T163522Z/experiment/workspace/dataset
/homes/55/junlin/miniconda3/envs/internvlu/bin/python pipeline-agent/cli.py --root pipeline-agent/runs/kickoff_20261029T163522Z/experiment run --dataset pipeline-agent/runs/kickoff_20261029T163522Z/experiment/workspace/dataset --request-id baseline
codex login status
timeout --signal=TERM --kill-after=15s 1800 /homes/55/junlin/miniconda3/envs/internvlu/bin/python pipeline-agent/cli.py --root pipeline-agent/runs/kickoff_20261029T163522Z/experiment agent
```

`configure_real.py` 最初生成继承 N0 的配置，随后通过 `internvlu_train.config.load_config/dump_config` 展开为固定 [train.yaml](train.yaml)。复现应使用展开后的文件并核对 freeze.json，不需要重新展开当前外部 N0。

真实 GPU 命令由 training.py 构建，各 trial 的 `train_train_spec.json`、`eval_dev_spec.json`（最终还包含 `eval_test_spec.json`）保存实际输入：

```text
<internvlu python> -m torch.distributed.run --standalone --nproc_per_node=4 <pipeline-agent>/real_backend.py --spec <trial>/<mode>_<output>_spec.json
```

它们必须由 `campaign.run_process` 运行，不能绕开 CLI 启动训练。子进程设置 CUDA_VISIBLE_DEVICES=0,1,2,3。Agent 的每次真实构建/验证/run 命令在 `experiment/agent_logs/*/events.jsonl` 中，strategy 与 request ID 映射在 workspace/NOTES.md。两个改进的代码均有独立 trial/code 快照。

最终 finalize 的实际选定 trial 与完整命令将在执行后追加，最终 test 不得盲目重试。

## Host finalization command

```sh
/homes/55/junlin/miniconda3/envs/internvlu/bin/python /scratch/network/ssd2/junlin/ReActData-mod/pipeline-agent/cli.py --root /scratch/network/ssd2/junlin/ReActData-mod/pipeline-agent/runs/kickoff_20261029T163522Z/experiment finalize --trial t0001
```
