请在当前 ReActData 仓库推进 pipeline-agent 的首次真实端到端实验。

先记录本次开始时间，阅读适用的 AGENTS.md，以及 pipeline-agent/REMOTE_HANDOFF.md、README.md、tasks/reactdata.yaml 和 protocols/agent.md，然后直接实施并运行，不停留在提出计划。

目标：完成环境与CPU检查、真实训练/dev/test后端接入、baseline、数据策略改进、锁定checkpoint、最终一次test和初步结果报告。你可以根据远程机器的数据和资源自主设计训练池、dev和test；在首次正式baseline前记录并冻结设计，确保split无泄漏，搜索不使用test反馈。

硬限制：同时最多4张GPU；从本次kickoff起总墙钟不超过8小时；整个任务最多7次真实训练机会，包含独立短训练smoke、失败和重试，不因新建实验重置。通常1次baseline+最多6次改进；如smoke消耗1次，相应减少改进次数。必须先实现并验证总预算/deadline/GPU约束，再启动真实训练；给test和报告预留时间，必要时提前结束搜索。

允许直接运行Codex并按Markdown约束行为，不要求Docker或强制隔离。仅修改pipeline-agent及本次新运行文件，不修改旧pipeline、冻结协议或已有产物，不提交、不推送。不要把simulation或占位指标当真实结果。资源或认证阻塞时先完成不受影响的工作，记录具体限制；Codex不可用时可由你按相同CLI协议完成搜索并如实说明。

最终请给我：实际跑通的完整调用链、各trial与baseline比较、选定checkpoint与最终test结果、资源/时间/机会消耗、报告及关键产物路径，以及尚未验证的部分。优先交付真实且可复查的初步闭环，不以跑满次数为目标。
