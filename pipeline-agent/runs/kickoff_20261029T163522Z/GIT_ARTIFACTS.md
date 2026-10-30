# Git 归档范围

本次按用户要求提交实验实现和完整轻量结果。包含报告、配置、数据与哈希审计、baseline 和两轮改进的代码/数据快照、训练日志、dev/test 图片及逐项指标、Agent 日志、CPU 模拟与测试记录。

checkpoint 文件约 3.6 GB/个，未纳入 Git，也未上传 Git LFS 或其他存储；模型基础权重与原始视频属于外部资源，同样未纳入。报告中的 checkpoint 链接在实验主机上有效，普通 clone 后不包含这些文件。加载模型需取回下列参数文件，并使用报告记录的相同基础权重。

| Trial | 本地主机 checkpoint | SHA256 |
|---|---|---|
| t0001 | `/scratch/network/ssd2/junlin/ReActData-mod/pipeline-agent/runs/kickoff_20261029T163522Z/experiment/trials/t0001/train/last.pt` | `782b9dfd432ef3d62f27165b919d63dc73db52a5c8d30aa6c1f40be3b9948334` |
| t0002 | `/scratch/network/ssd2/junlin/ReActData-mod/pipeline-agent/runs/kickoff_20261029T163522Z/experiment/trials/t0002/train/last.pt` | `ff3083c7b0c9bb51fbac7da32d6f6922d008dc180b3a2fd605a115bf501f693a` |
| t0003 | `/scratch/network/ssd2/junlin/ReActData-mod/pipeline-agent/runs/kickoff_20261029T163522Z/experiment/trials/t0003/train/last.pt` | `ee8f9dbcb051154646bb15b02a3ff4cb2eb324e51a93d5932e85324e57cb01de` |

排除 Python 缓存、临时文件、运行锁以及认证状态检查日志。既有产物的内容和哈希保持不变；仅在当前交付报告中补充此归档说明。
