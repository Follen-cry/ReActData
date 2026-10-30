import pathlib,json,datetime,sys,hashlib
p=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(p.parents[1]));from storage import write,digest
files=['train_sources.jsonl','dev_sources.jsonl','test_sources.jsonl','dev.jsonl','test.jsonl','train.yaml','real_config.json','data_audit.json']
hashes={str(p/f):digest(p/f) for f in files}
for f in p.parents[1].glob('*.py'): hashes[str(f)]=digest(f)
for split in ('train','dev','test'):
 for line in (p/f'{split}_sources.jsonl').read_text().splitlines():
  row=json.loads(line)
  for k in ('f0','final'):hashes[row[k]]=digest(row[k])
write(p/'freeze.json',dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),sha256=hashes,changes_after_freeze='Only workspace builder selection/order. Infrastructure fixes must be disclosed and compared before further trials.'))
with open(p/'experiment_design.md','a') as f:f.write('\n## 正式冻结更新\n\n正式数据与配置于 freeze.json 记录的 UTC 时间冻结，采用上述 160/50/50 候选；完整视频字节、规范化端点像素和生成参数检查均无交叉重复。重编码语义等价的不同视频尚无通用检测保证，历史曝光未知并明确披露。包装文本已检查 residual-video-language 且逐项固定。评分器在 20 个训练样本上 GT >=0.95、复制 F0 <=0.05（scorer_selfcheck.json）；未检查 test 指标。训练和生成参数保持原默认契约。训练配置已展开外部继承为 train.yaml；后端使用单 tile、显式固定数据，不读取其中遗留 train_meta 指向的数据。\n')
