# 候选设计（尚未冻结）

本次候选以原始 `shape_color_then_scale` 样本目录为实例单位，按目录排序后用 Python Random(20261029) 打乱，依次取 160/50/50 个唯一候选。记录分别在 [train_sources.jsonl](train_sources.jsonl)、[dev_sources.jsonl](dev_sources.jsonl)、[test_sources.jsonl](test_sources.jsonl)，逐条含来源、哈希、帧数和包装文本；详情见 [data_audit.json](data_audit.json)。原始来源是 `/scratch/network/ssd/junlin/vbvr_next_frame/raw/shape_color_then_scale`。历史训练或测试使用情况未知，不能称为全新盲测。端点由原始 PNG 使用 OpenCV INTER_AREA 标准化到 512；未从 MP4 冒充 PNG。wrapper SHA256 及 OpenCV 版本见审计文件。文本为逐项 `<image>\n` 加外部 wrapper 输出。

候选方案不是已冻结正式设计：尚未完成不同编码但等价视频的逐帧内容去重、检查包装文本语义、评测器自检以及将 dev/test 转换成真实后端固定格式。审计的所有帧数通过 OpenCV 顺序解码核对；规范化的端点像素和原视频字节散列全局唯一，生成参数 JSON 未发现跨 split 完全重复。未在数据结果上选择样本或运行评测。

拟沿用 [任务契约](../../tasks/reactdata.yaml)：128 条唯一视频，8 steps，global batch 16，warmup 3，LoRA 2e-5，gen decoder 1e-4；model init/train/build/eval seeds 分别为 42/1000/1000/0，20 diffusion steps，CFG 4.5/2.0，固定 dev 均值含无效图 0。正式 baseline 之前必须补全核验、写解析配置并明确冻结时间。最终 test 保留给选择 checkpoint 后一次性评测。

## 正式冻结更新

正式数据与配置于 freeze.json 记录的 UTC 时间冻结，采用上述 160/50/50 候选；完整视频字节、规范化端点像素和生成参数检查均无交叉重复。重编码语义等价的不同视频尚无通用检测保证，历史曝光未知并明确披露。包装文本已检查 residual-video-language 且逐项固定。评分器在 20 个训练样本上 GT >=0.95、复制 F0 <=0.05（scorer_selfcheck.json）；未检查 test 指标。训练和生成参数保持原默认契约。训练配置已展开外部继承为 train.yaml；后端使用单 tile、显式固定数据，不读取其中遗留 train_meta 指向的数据。
